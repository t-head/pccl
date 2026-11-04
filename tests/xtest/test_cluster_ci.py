# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cluster_ci as ci
from execute import execution_provenance, write_json
from manifest import build_manifest, case_key, digest, write_bundle


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

  def write_results(self):
    shard = self.manifest['shards'][0]
    row = shard['cases'][0]
    key = row['case_key']
    case = self.root / 'results' / key / 'attempt-1'
    case.mkdir(parents=True)
    log_ref = f'{key}/attempt-1/command.log'
    result = {
      'case_key': key, 'attempt': 1, 'status': 'PASSED', 'returncode': 0, 'error': '',
      'duration': 0.01, 'legacy_source': row['spec']['legacy_source'], 'command_log': log_ref,
    }
    (case / 'command.log').write_text('synthetic two-node command completed\n')
    write_json(case / 'command-process.json', {'active': False, 'state': 'finished', 'return_code': 0})
    write_json(case / 'result.json', result)
    (self.root / 'results/results.jsonl').write_text(json.dumps(result) + '\n')
    write_json(self.root / 'results/index.json', {
      'shard_id': shard['shard_id'], 'expected': [key], 'completed': [key], 'failures': 0,
      'cancelled': False, 'artifact_error': '',
      'selection_sha256': self.manifest['selection_sha256'],
      'execution': execution_provenance(),
    })
    write_json(self.root / 'results/manifest.json', self.manifest)

  def test_source_export_and_manifest_are_exactly_one_two_node_case(self):
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
    catalog['case']['spec']['parameters']['ranks'] = 8
    catalog['bundle_sha256'] = digest({key: value for key, value in catalog.items()
                                       if key != 'bundle_sha256'})
    changed_catalog = self.root / 'cluster-source-case.json'
    changed_catalog.write_text(json.dumps(catalog))
    with self.assertRaises(ValueError):
      ci.load_case(changed_catalog)

  def test_complete_result_gate_requires_execution_evidence(self):
    self.write_results()
    output = self.root / 'public'
    self.assertEqual(ci.summarize(self.path, self.root / 'results', output), 0)
    summary = json.loads((output / 'summary.json').read_text())
    self.assertEqual((summary['expected'], summary['passed'], summary['complete']), (1, 1, True))
    process = next((self.root / 'results').rglob('command-process.json'))
    write_json(process, {'active': False, 'state': 'failed', 'return_code': 0})
    self.assertEqual(ci.summarize(self.path, self.root / 'results', self.root / 'failed'), 1)

  def test_workflow_uses_requested_master_action_and_no_direct_ssh_bootstrap(self):
    workflow = (ROOT / '.github/workflows/xtest-cluster-run-validation.yml').read_text()
    self.assertIn('uses: t-head/ci-infra/cluster-run@feat/master-launch-mode', workflow)
    self.assertIn('launch_mode: master', workflow)
    self.assertIn('nnodes: 2', workflow)
    self.assertIn('nproc_per_node: 8', workflow)
    self.assertIn('cluster_ci.py" run', workflow)
    self.assertIn('cluster_ci.py diagnose', workflow)
    self.assertIn('cluster_ci.py summary', workflow)
    self.assertNotIn('ppu-scheduler-action@', workflow)
    self.assertNotIn('ssh-keygen', workflow)
    self.assertNotIn('sshd', workflow)
    self.assertNotIn('plm_rsh_agent', json.dumps(ci.MPI_OPTIONS))
    prepare = workflow.split('\n  cluster:', 1)[0]
    self.assertNotIn('cluster-run@', prepare)
    setup = workflow.split('setup_command: |', 1)[1].split('          command: |', 1)[0]
    command = workflow.split('          command: |', 1)[1].split('\n  summary:', 1)[0]
    self.assertNotIn('build.sh', setup)
    self.assertIn('build.sh', command)

  def test_diagnostics_redact_worker_addresses_and_paths(self):
    case = self.root / 'diagnostics/case/attempt-1'
    case.mkdir(parents=True)
    (case / 'command.log').write_text(
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
