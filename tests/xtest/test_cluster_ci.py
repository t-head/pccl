# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import cluster_ci as ci
from execute import run_shard, write_json
from manifest import build_manifest, case_key, digest, write_bundle
from test_single_node_ci import (public_runner_fixture, rewrite_case_runtime, rewrite_hook_runtime,
                                 synthetic_build_record)


ROOT = Path(__file__).resolve().parents[2]


class ClusterCiTests(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.root = Path(temporary.name)
    self.runner = Path(os.environ['PCCL_TESTS_ROOT']).resolve()
    self.spec, self.inputs = ci.expected_inputs('a' * 40, self.runner)
    self.manifest = build_manifest([self.spec], self.inputs)
    write_bundle(self.manifest, self.root / 'manifest')
    self.path = self.root / 'manifest/manifest.json'

  def write_results(self, inherited=None):
    shard = self.manifest['shards'][0]
    build = self.root / 'synthetic-build'
    public_runner_fixture(build, self.runner)
    (build / 'hosts.cfg').write_text('worker-a\nworker-b\n')
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('PCCL_', 'NCCL_', 'HGGC_', 'OMPI_', 'UCX_'))}
    environment.update(inherited or {})
    with mock.patch.dict(os.environ, environment, clear=True):
      self.assertEqual(run_shard(self.path, shard['shard_id'], build, build, self.root / 'results',
                                 ['--hostfile', str(build / 'hosts.cfg')],
                                 build=synthetic_build_record(self.inputs, 'standard')), 0)

  def test_source_export_and_manifest_are_exactly_one_two_node_case(self):
    self.assertEqual(ci.ACTION_REF, 't-head/ci-infra/cluster-run@40165c945cba33ac7b16b41c4485fe53b68f581f')
    self.assertEqual(self.inputs['cluster_action_ref'], ci.ACTION_REF)
    spec, source = ci.load_case()
    self.assertEqual(source['revision'], ci.EXPECTED_SOURCE_REVISION)
    self.assertEqual((spec['nodes'], spec['cards_per_node'], spec['processes_per_node'],
                      spec['parameters']['ranks']), (2, 8, 8, 16))
    self.assertEqual(spec['parameters']['transport'], 'gdr')
    self.assertIn('--mpi-options={mpi_options}', spec['argv'])
    selected = ci.selection(self.path)
    self.assertEqual(selected['case_count'], 1)
    self.assertEqual(selected['shards'][0]['cases'], [{'case_key': case_key(spec), 'spec': spec}])

  def test_manifest_or_catalog_changes_fail_closed(self):
    changed = copy.deepcopy(self.manifest)
    changed['inputs']['cluster_action_ref'] = 'other/action@main'
    specs = [row['spec'] for row in changed['shards'][0]['cases']]
    changed = build_manifest(specs, changed['inputs'])
    path = self.root / 'changed'
    write_bundle(changed, path)
    with self.assertRaises(ValueError):
      ci.selection(path / 'manifest.json')

    catalog = json.loads(ci.CATALOG.read_text())
    catalog['executions'][0]['spec']['parameters']['ranks'] = 8
    catalog['bundle_sha256'] = digest({key: value for key, value in catalog.items()
                                       if key != 'bundle_sha256'})
    changed_catalog = self.root / 'cluster-source-case.json'
    changed_catalog.write_text(json.dumps(catalog))
    with self.assertRaises(ValueError):
      ci.load_case(changed_catalog)

  def test_complete_result_gate_requires_execution_evidence(self):
    self.write_results()
    output = self.root / 'public'
    verify = ci.verify_lifecycle
    with self.assertRaisesRegex(ValueError, 'trusted execution runtime'):
      verify(self.root / 'results', self.manifest['shards'][0])
    with mock.patch.object(ci, 'verify_lifecycle', side_effect=lambda directory, shard, runtime: verify(directory, shard)):
      self.assertEqual(ci.summarize(self.path, self.root / 'results', self.root / 'without-runtime'), 1)
    self.assertEqual(ci.summarize(self.path, self.root / 'results', output), 0)
    summary = json.loads((output / 'summary.json').read_text())
    self.assertEqual((summary['expected'], summary['passed'], summary['complete']), (1, 1, True))
    self.assertEqual(summary['build'], synthetic_build_record(self.inputs, 'standard'))
    process = next((self.root / 'results').rglob('command-process.json'))
    write_json(process, {'active': False, 'state': 'failed', 'return_code': 0})
    self.assertEqual(ci.summarize(self.path, self.root / 'results', self.root / 'failed'), 1)

  def test_integer_only_result_fields_reject_booleans_and_floats(self):
    self.write_results()
    index_path = self.root / 'results/index.json'
    index = ci.read_json(index_path)
    for value in (False, 0.0):
      write_json(index_path, dict(index, failures=value))
      with self.subTest(field='failures', value=value):
        self.assertEqual(ci.summarize(self.path, self.root / 'results', self.root / f'index-{value}'), 1)
    write_json(index_path, index)
    key = self.manifest['shards'][0]['cases'][0]['case_key']
    result_path = self.root / 'results' / key / 'attempt-1/result.json'
    result = ci.read_json(result_path)
    for field, value in (('returncode', False), ('returncode', 0.0), ('attempt', True), ('attempt', 1.0)):
      changed = dict(result, **{field: value})
      write_json(result_path, changed)
      (self.root / 'results/results.jsonl').write_text(json.dumps(changed) + '\n')
      with self.subTest(field=field, value=value):
        self.assertEqual(ci.summarize(self.path, self.root / 'results',
                                      self.root / f'{field}-{value}'), 1)
      write_json(result_path, result)
      (self.root / 'results/results.jsonl').write_text(json.dumps(result) + '\n')

  def test_build_identity_is_required_and_bound_to_manifest(self):
    self.write_results()
    snapshot_path = self.root / 'results/shard-runtime.json'
    index_path = self.root / 'results/index.json'
    snapshot = ci.read_json(snapshot_path)
    for mutation in ('missing', 'revision', 'profile'):
      changed = copy.deepcopy(snapshot)
      if mutation == 'missing':
        changed.pop('build')
      elif mutation == 'revision':
        changed['build']['pccl_revision'] = 'f' * 40
      else:
        changed['build']['build_profile'] = 'notsm'
      write_json(snapshot_path, changed)
      index = ci.read_json(index_path)
      index['runtime_sha256'] = digest(changed)
      write_json(index_path, index)
      with self.subTest(mutation=mutation):
        self.assertEqual(ci.summarize(self.path, self.root / 'results', self.root / f'build-{mutation}'), 1)
      write_json(snapshot_path, snapshot)
      index['runtime_sha256'] = digest(snapshot)
      write_json(index_path, index)

  def test_relocated_summary_needs_no_build_hostfile_or_local_runtime(self):
    self.write_results()
    relocated = self.root / 'downloaded'
    (self.root / 'results').rename(relocated)
    shutil.rmtree(self.root / 'synthetic-build')
    verifier = {**ci.execution_provenance(), 'executor_python_version': '3.12.0'}
    with mock.patch.dict(os.environ, {'PATH': '/summary/bin', 'UCX_TLS': 'summary-only'}, clear=True), \
            mock.patch.object(sys, 'executable', '/summary/python3'), \
            mock.patch.object(ci, 'execution_provenance', return_value=verifier):
      self.assertEqual(ci.summarize(self.path, relocated, self.root / 'public'), 0)

  def test_hook_runtime_tampering_fails_summary_after_rehash(self):
    self.write_results()
    shard = self.manifest['shards'][0]
    for field, value in (('test_root', '/different/root'), ('binary_dir', '/different/bin'),
                         ('result_dir', '/different/results'), ('mpi_options', ['--hostfile', 'different']),
                         ('python', '/different/python3'), ('inherited_environment', {'PATH': '/different/bin'})):
      with self.subTest(field=field):
        directory = self.root / f'hook-{field}'
        shutil.copytree(self.root / 'results', directory)
        rewrite_hook_runtime(directory, shard, field, value)
        self.assertEqual(ci.summarize(self.path, directory, self.root / f'public-{field}'), 1)

  def test_case_runtime_cannot_disagree_with_shard_after_rehash(self):
    self.write_results({'UCX_TLS': 'original'})
    row = self.manifest['shards'][0]['cases'][0]
    shutil.rmtree(self.root / 'synthetic-build')
    for field in ('python', 'inherited_environment', 'binary_dir', 'runner_build_directory',
                  'result_dir', 'benchmark', 'launcher'):
      with self.subTest(field=field):
        directory = self.root / f'case-{field}'
        shutil.copytree(self.root / 'results', directory)
        rewrite_case_runtime(directory / row['case_key'] / 'attempt-1', row['spec'], field)
        self.assertEqual(ci.summarize(self.path, directory, self.root / f'public-{field}'), 1)

  def test_missing_or_unlinked_shard_runtime_fails_summary(self):
    self.write_results()
    directory = self.root / 'results'
    path = directory / 'shard-runtime.json'
    original = path.read_bytes()
    path.unlink()
    self.assertEqual(ci.summarize(self.path, directory, self.root / 'missing'), 1)
    path.write_bytes(original)
    index = ci.read_json(directory / 'index.json')
    index['runtime_sha256'] = '0' * 64
    write_json(directory / 'index.json', index)
    self.assertEqual(ci.summarize(self.path, directory, self.root / 'unlinked'), 1)

  def test_workflow_uses_requested_master_action_and_no_direct_ssh_bootstrap(self):
    workflow = (ROOT / '.github/workflows/source-cluster-debug.yml').read_text()
    # The action is pinned to a reviewed commit; a moving @main cannot back a gate.
    self.assertIn('uses: t-head/ci-infra/cluster-run@40165c945cba33ac7b16b41c4485fe53b68f581f',
                  workflow)
    self.assertNotIn('cluster-run@main', workflow)
    self.assertIn('launch_mode: master', workflow)
    self.assertIn('nnodes: 2', workflow)
    self.assertIn('nproc_per_node: 8', workflow)
    self.assertIn('cluster_ci.py" run', workflow)
    self.assertIn('cluster_ci.py diagnose', workflow)
    self.assertIn('cluster_ci.py summary', workflow)
    self.assertNotIn('github.run_attempt', workflow)
    self.assertIn('build_evidence.py" create', workflow)
    self.assertIn('--build-record "$RUN_ROOT/output_standard.build.json"', workflow)
    self.assertIn('--build-archive "$RUN_ROOT/output_standard.tar.gz"', workflow)
    self.assertNotIn('ppu-scheduler-action@', workflow)
    # The two-node trial must keep the same time budget as the single-node job.
    single = (ROOT / '.github/workflows/source-sanity.yml').read_text()
    self.assertIn('timeout-minutes: 360', workflow)
    self.assertIn('timeout_minutes: 300', workflow)
    self.assertIn('timeout-minutes: 20', workflow)
    self.assertIn('timeout-minutes: 360', single)
    self.assertIn('timeout_minutes: 300', single)
    self.assertIn('timeout-minutes: 20', single)
    self.assertNotIn('ssh-keygen', workflow)
    self.assertNotIn('sshd', workflow)
    self.assertIn('plm_rsh_agent', json.dumps(ci.MPI_OPTIONS))
    self.assertIn('cluster-ssh', ci.MPI_OPTIONS)
    self.assertNotIn('btl_tcp_if_include', ci.MPI_OPTIONS)
    self.assertNotIn('bond1', ci.MPI_OPTIONS)
    self.assertIn('privileged: true', workflow)
    self.assertIn('host_volumes: /dev/infiniband:/dev/infiniband', workflow)
    self.assertNotIn('host_network: true', workflow)
    self.assertNotIn('/sys/class/net/bond1', workflow)
    self.assertIn('filelock-3.20.3-py3-none-any.whl', workflow)
    self.assertIn('--no-index --find-links', workflow)
    self.assertIn('cluster-run-build-deps', workflow)
    self.assertIn('libboost-program-options-dev=1.83.0.1ubuntu2', workflow)
    self.assertIn('sha256sum --check SHA256SUMS', workflow)
    # The standalone workflow is a manual troubleshooting tool now; the PR path lives in
    # source-sanity.yml and reuses its standard build.
    head = workflow.split('\n  cluster:', 1)[0]
    self.assertIn('workflow_dispatch:', head)
    self.assertNotIn('pull_request:', head)
    self.assertNotIn('  push:', head)
    prepare = workflow.split('\n  cluster:', 1)[0]
    self.assertNotIn('cluster-run@', prepare)
    setup = workflow.split('setup_command: |', 1)[1].split('          command: |', 1)[0]
    command = workflow.split('          command: |', 1)[1].split('\n  summary:', 1)[0]
    self.assertNotIn('build.sh', setup)
    self.assertIn('build.sh', command)

  def test_diagnostics_redact_worker_addresses_and_paths(self):
    case = self.root / 'diagnostics/case/attempt-1'
    case.mkdir(parents=True)
    (case / 'command-public.log').write_text(
      'MPI ERROR ppu-t-head-123-worker-0.job.svc.cluster.local 10.1.2.3 '
      '/workspace/source/private token=unsafe\n')
    with mock.patch('builtins.print') as output:
      ci.diagnostics(self.root / 'diagnostics')
    text = '\n'.join(str(call.args[0]) for call in output.call_args_list)
    self.assertIn('<worker>', text)
    self.assertIn('<ip>', text)
    self.assertIn('<workspace>', text)
    self.assertIn('token=<redacted>', text)
    self.assertNotIn('10.1.2.3', text)
    self.assertLessEqual(sum('::notice' in str(call.args[0]) for call in output.call_args_list), 5)

  def test_mpi_prefix_is_derived_from_the_worker_image(self):
    prefix = self.root / 'openmpi'
    (prefix / 'bin').mkdir(parents=True)
    for name in ('mpirun', 'orted'):
      (prefix / 'bin' / name).write_text('')
    with mock.patch.object(ci.shutil, 'which', return_value=str(prefix / 'bin/mpirun')):
      options = ci.mpi_options()
    self.assertEqual(options[:2], ['--prefix', str(prefix.resolve())])
    self.assertEqual(options[2:], ci.MPI_OPTIONS)


if __name__ == '__main__':
  unittest.main()
