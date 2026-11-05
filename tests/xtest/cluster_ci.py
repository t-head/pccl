# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare, execute and reconcile the source-derived two-node cluster-run trial."""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from catalog import validate_bundle
from command import check_command
from execute import execution_provenance, read_jsonl, run_shard, verified_runtime, verify_lifecycle, write_json
from manifest import (PCCL_TESTS_REVISION, RUNNER_HASHES, build_manifest, case_key, digest,
                      load_manifest, validate_spec, write_bundle)


CATALOG = Path(__file__).with_name('cluster-source-case.json')
ACTION_REF = 't-head/ci-infra/cluster-run@main'
EXPECTED_SOURCE_REVISION = 'e27acd364858f845eace0abb72466935070feb0e'
EXPECTED_EXPORTER_SHA256 = '61bde8c2acc937c2ef39cc9cee091c8ab63da18a8b17d44f39b6c524436569a4'
EXPECTED_BUNDLE_SHA256 = 'a8e482e8e436eac1cf29382d1e53df6989d06ad2e9b30326a3109216d12b81bc'
HEX40 = re.compile(r'[0-9a-f]{40}\Z')
HEX64 = re.compile(r'[0-9a-f]{64}\Z')
SAFE_NAME = re.compile(r'[A-Za-z0-9_.-]+\Z')
PRIVATE_MARKERS = (
  'comm-internal-testbox', '@alibaba-inc.com', 'reg.docker.alibaba-inc.com',
  'UploadResult', 'work_root', 'setup_script', 'test/cases/', 'test/plans/', 'test/suites/',
)
MPI_OPTIONS = [
  '-mca', 'plm_rsh_agent', 'cluster-ssh',
  '-mca', 'orte_base_help_aggregate', '0',
  '--hostfile', 'hosts.cfg',
]


def mpi_options():
  executable = shutil.which('mpirun')
  if not executable:
    raise ValueError('mpirun is unavailable in the worker image')
  prefix = Path(executable).resolve().parent.parent
  if not (prefix / 'bin/orted').is_file():
    raise ValueError('OpenMPI prefix does not contain the remote orted launcher')
  return ['--prefix', str(prefix), *MPI_OPTIONS]


def diagnostics(results):
  root = Path(results)
  logs = sorted(root.glob('*/attempt-1/command.log'))
  if not logs:
    print('Cluster diagnostic: command log was not created')
    return
  text = logs[0].read_text(encoding='utf-8', errors='replace')
  lines = []
  for line in text.splitlines():
    line = re.sub(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', '<ip>', line)
    line = re.sub(r'[A-Za-z0-9.-]+\.svc\.cluster\.local', '<worker>', line)
    line = re.sub(r'ppu-t-head-[A-Za-z0-9-]+', '<worker>', line)
    line = re.sub(r'/workspace/[^\s:]+', '<workspace>', line)
    line = re.sub(r'/mnt/wl_nas/[^\s:]+', '<nas>', line)
    line = re.sub(r'(?i)\b(token|password|secret)=\S+', r'\1=<redacted>', line)
    lines.append(line[:1000])
  print('Cluster diagnostic (sanitized tail):')
  selected = lines[-120:] if lines else ['<empty command log>']
  print('\n'.join(selected))
  for line in selected[-40:]:
    annotation = line.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')
    print(f'::notice title=Cluster diagnostic::{annotation}')


def read_json(path):
  def unique(pairs):
    result = {}
    for key, value in pairs:
      if key in result:
        raise ValueError(f'Duplicate JSON key: {key}')
      result[key] = value
    return result

  def invalid(value):
    raise ValueError(f'Nonfinite JSON constant: {value}')

  return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
                    parse_constant=invalid)


def _hex(value, pattern):
  return isinstance(value, str) and pattern.fullmatch(value) is not None


def load_case(path=None):
  bundle = validate_bundle(read_json(Path(path or CATALOG)), 'two-node-source-register-cluster-trial')
  source = bundle['source']
  if (bundle['bundle_sha256'] != EXPECTED_BUNDLE_SHA256
      or source['revision'] != EXPECTED_SOURCE_REVISION
      or source['exporter_sha256'] != EXPECTED_EXPORTER_SHA256):
    raise ValueError('Cluster source identity is not the reviewed export')
  if len(bundle['executions']) != 1 or bundle['executions'][0]['profiles'] != ['standard']:
    raise ValueError('Cluster trial requires exactly one standard execution')
  spec = {**copy.deepcopy(bundle['executions'][0]['spec']), 'build_profile': 'standard'}
  validate_spec(spec)
  if (spec['nodes'] != 2 or spec['cards_per_node'] != 8 or spec['processes_per_node'] != 8
      or spec['threads_per_process'] != 1 or spec['devices_per_thread'] != 1
      or spec['parameters'].get('ranks') != 16 or spec['parameters'].get('transport') != 'gdr'
      or '--mpi-options={mpi_options}' not in spec['argv']):
    raise ValueError('Cluster source Case differs from the reviewed execution contract')
  return spec, copy.deepcopy(source)


def _runner_identity(root):
  root = Path(root).resolve()
  revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
  if revision != PCCL_TESTS_REVISION:
    raise ValueError(f'pccl-tests HEAD mismatch: {revision}')
  hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in RUNNER_HASHES}
  if hashes != RUNNER_HASHES:
    raise ValueError('Pinned public runner files differ from their reviewed hashes')
  return revision, hashes


def expected_inputs(pccl_revision, pccl_tests, catalog_path=None):
  if not _hex(pccl_revision, HEX40):
    raise ValueError('PCCL revision must be a lowercase Git SHA')
  spec, source = load_case(catalog_path)
  runner_revision, runner_hashes = _runner_identity(pccl_tests)
  catalog_path = Path(catalog_path or CATALOG)
  inputs = {
    'test_suite': 'cluster-run-trial', 'single_node': False, 'debug_only': False,
    'test_types': ['functional'], 'node_counts': [2], 'shards_per_pool': 1,
    'timings_sha256': digest({}), 'pccl_revision': pccl_revision,
    'pccl_tests_revision': runner_revision, 'runner_sha256': runner_hashes,
    'cluster_action_ref': ACTION_REF,
    'source_catalog_sha256': hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
    'source_catalog_bundle_sha256': EXPECTED_BUNDLE_SHA256,
    'source_catalog_source': source,
  }
  return spec, inputs


def prepare(pccl_revision, pccl_tests, output, catalog_path=None):
  spec, inputs = expected_inputs(pccl_revision, pccl_tests, catalog_path)
  manifest = build_manifest([spec], inputs)
  write_bundle(manifest, output)
  return manifest


def selection(path):
  manifest = load_manifest(path)
  spec, source = load_case()
  inputs = manifest['inputs']
  expected = {
    'test_suite': 'cluster-run-trial', 'single_node': False, 'debug_only': False,
    'test_types': ['functional'], 'node_counts': [2], 'shards_per_pool': 1,
    'timings_sha256': digest({}), 'pccl_revision': inputs.get('pccl_revision'),
    'pccl_tests_revision': PCCL_TESTS_REVISION, 'runner_sha256': RUNNER_HASHES,
    'cluster_action_ref': ACTION_REF,
    'source_catalog_sha256': hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
    'source_catalog_bundle_sha256': EXPECTED_BUNDLE_SHA256,
    'source_catalog_source': source,
  }
  if (not _hex(inputs.get('pccl_revision'), HEX40) or digest(inputs) != digest(expected)
      or manifest['case_count'] != 1 or len(manifest['shards']) != 1
      or manifest['shards'][0]['nodes'] != 2
      or manifest['shards'][0]['cases'] != [{'case_key': case_key(spec), 'spec': spec}]):
    raise ValueError('Cluster manifest differs from the complete reviewed source selection')
  return manifest


def run(manifest_path, test_root, output):
  manifest = selection(manifest_path)
  root = Path(test_root).resolve()
  binary = root / 'pccl-tests' / 'build'
  hosts = (root / 'hosts.cfg').read_text(encoding='utf-8').splitlines()
  if len(hosts) != 2 or len(set(hosts)) != 2 or any(not SAFE_NAME.fullmatch(host) for host in hosts):
    raise ValueError('cluster-run must provide two unique public-safe worker hosts')
  linked = subprocess.run(['ldd', str(binary / 'all_reduce_perf')], capture_output=True, text=True, check=True)
  libraries = re.findall(r'lib(?:pccl|nccl)\.so[^\s]*\s+=>\s+(\S+)', linked.stdout)
  if not libraries or any(Path(path).resolve().parent != (root / 'lib').resolve() for path in libraries):
    raise ValueError('Benchmark is not linked to the freshly built PCCL library')
  shard = manifest['shards'][0]
  return run_shard(manifest_path, shard['shard_id'], root, binary, Path(output), mpi_options())


def summarize(manifest_path, results, output):
  manifest = selection(manifest_path)
  shard = manifest['shards'][0]
  expected = {row['case_key']: row for row in shard['cases']}
  directory = Path(results)
  passed, error = 0, ''
  try:
    index = read_json(directory / 'index.json')
    rows = read_jsonl(directory / 'results.jsonl')
    actual = {row['case_key']: row for row in rows}
    execution = index['execution']
    if (len(actual) != len(rows) or list(actual) != list(expected)
        or index['shard_id'] != shard['shard_id'] or index['expected'] != list(expected)
        or index['completed'] != list(expected) or index['failures'] != 0
        or index['cancelled'] is not False or index['artifact_error'] != ''
        or index['selection_sha256'] != manifest['selection_sha256']
        or digest(load_manifest(directory / 'manifest.json')) != digest(manifest)
        or set(execution) != {'executor_sha256', 'executor_python_version'}
        or execution['executor_sha256'] != execution_provenance()['executor_sha256']
        or not re.fullmatch(r'3\.[0-9]+\.[0-9]+', execution['executor_python_version'])):
      raise ValueError('Cluster shard evidence is incomplete or inconsistent')
    shard_runtime = verified_runtime(directory, index)
    if index.get('lifecycle_error') != '':
      raise ValueError('Cluster Suite lifecycle did not complete cleanly')
    verify_lifecycle(directory, shard, shard_runtime)
    key, row = next(iter(expected.items()))
    result = actual[key]
    case_dir = directory / key / 'attempt-1'
    log_ref = f'{key}/attempt-1/command.log'
    if (result['status'] != 'PASSED' or result['returncode'] != 0 or result['error'] != ''
        or result['attempt'] != 1 or type(result['duration']) not in (int, float)
        or not math.isfinite(result['duration']) or result['duration'] < 0
        or result['command_log'] != log_ref or result['legacy_source'] != row['spec']['legacy_source']
        or digest(read_json(case_dir / 'result.json')) != digest(result)):
      raise ValueError('Cluster Case result is incomplete or inconsistent')
    context = {name: row['spec']['execution'][name]
               for name in ('plan_id', 'suite_id', 'case_id', 'suite_run_id')}
    if result.get('source_context') != context:
      raise ValueError('Cluster result lost its source context')
    runtime = dict(shard_runtime, result_dir=str(case_dir), inputs={**manifest['inputs'], **execution},
                   execution_result_dir=str(Path(shard_runtime['result_dir']) / key / 'attempt-1'),
                   case_key=key, command_log_ref=log_ref)
    check_command(row['spec'], runtime, read_json(case_dir / 'command-process.json'))
    passed = 1
  except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exception:
    error = type(exception).__name__
  if passed != 1:
    diagnostics(directory)
  target = Path(output)
  target.mkdir(parents=True, exist_ok=False)
  complete = passed == 1 and not error
  write_json(target / 'summary.json', {
    'scope': 'two-node-cluster-run-trial', 'complete': complete, 'expected': 1,
    'passed': passed, 'error': error, 'inputs': manifest['inputs'],
  })
  print(f'Cluster-run MPI trial: {passed}/1 passed; complete={complete}')
  return 0 if complete else 1


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('operation', choices=('prepare', 'run', 'summary', 'diagnose'))
  parser.add_argument('--manifest', type=Path)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--pccl-revision')
  parser.add_argument('--pccl-tests', type=Path)
  parser.add_argument('--test-root', type=Path)
  parser.add_argument('--results', type=Path)
  args = parser.parse_args()
  try:
    if args.operation == 'diagnose':
      if args.results is None:
        parser.error('diagnose requires --results')
      diagnostics(args.results)
      return 0
    if args.operation == 'prepare':
      if not args.pccl_revision or args.pccl_tests is None:
        parser.error('prepare requires --pccl-revision and --pccl-tests')
      prepare(args.pccl_revision, args.pccl_tests, args.output)
      return 0
    if args.manifest is None:
      parser.error('run and summary require --manifest')
    if args.operation == 'run':
      if args.test_root is None:
        parser.error('run requires --test-root')
      return run(args.manifest, args.test_root, args.output)
    if args.results is None:
      parser.error('summary requires --results')
    return summarize(args.manifest, args.results, args.output)
  except (OSError, ValueError, subprocess.CalledProcessError) as error:
    parser.error(str(error))


if __name__ == '__main__':
  sys.exit(main())
