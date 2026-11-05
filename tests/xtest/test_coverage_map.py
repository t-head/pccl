# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
import copy
import json
from pathlib import Path
import unittest

from catalog import load_catalog
from command import _runner_index, _runner_options
from manifest import digest


MAP = Path(__file__).with_name('legacy-coverage-map.json')
PRIVATE_MARKERS = (
  '/Users/', 'comm-internal-testbox', '@alibaba-inc.com', 'reg.docker.alibaba-inc.com',
  'UploadResult', 'work_root', 'setup_script', 'test/cases/', 'test/plans/', 'test/suites/',
)


def read_json(path):
  def unique(pairs):
    result = {}
    for key, value in pairs:
      if key in result:
        raise ValueError(f'Duplicate coverage-map JSON key: {key}')
      result[key] = value
    return result

  def invalid(value):
    raise ValueError(f'Nonfinite coverage-map JSON constant: {value}')

  return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique,
                    parse_constant=invalid)


class CoverageMapTests(unittest.TestCase):
  def test_historical_catalog_mapping_is_complete_and_public(self):
    raw = MAP.read_text(encoding='utf-8')
    self.assertFalse([marker for marker in PRIVATE_MARKERS if marker in raw])
    report = read_json(MAP)
    self.assertEqual(set(report), {'schema_version', 'inputs', 'summary', 'cases', 'bundle_sha256'})
    self.assertEqual(report['schema_version'], 1)
    self.assertEqual(report['bundle_sha256'], digest({
      key: value for key, value in report.items() if key != 'bundle_sha256'}))
    self.assertEqual(report['inputs']['source_revision'],
                     'bbbeb207dd7f28980b55336660da0b47866e47aa')
    expected = {
      'exact_m890p_supported': 6,
      'exact_resource_incompatible': 60,
      'missing': 24,
      'near_m890p_supported': 6,
    }
    self.assertEqual(report['summary'], {
      'classifications': expected,
      'legacy_case_count': 96,
      'selected_source_case_count': 9,
      'source_load_errors': 12,
    })
    self.assertEqual(len(report['cases']), 96)
    self.assertEqual(len({row['legacy_case_key'] for row in report['cases']}), 96)
    self.assertEqual(Counter(row['classification'] for row in report['cases']), expected)

  def test_exact_and_near_matches_cover_the_nine_selected_source_cases(self):
    report = read_json(MAP)
    catalog = load_catalog()
    definitions = {row['id']: row['name'] for row in catalog['cases']}
    sources = {definitions[row['case_id']]: row for row in catalog['executions']}
    matched = {}
    for classification in ('exact_m890p_supported', 'near_m890p_supported'):
      ids = set()
      rows = [row for row in report['cases'] if row['classification'] == classification]
      for row in rows:
        selected = [match for match in row['matches'] if match['selected']]
        self.assertEqual(len(selected), 1)
        ids.add(selected[0]['case'])
      matched[classification] = ids

    exact = matched['exact_m890p_supported']
    near = matched['near_m890p_supported']
    self.assertEqual(len(exact), 6)
    self.assertEqual(len(near), 3)
    self.assertFalse(exact & near)
    self.assertTrue(exact | near <= set(sources))
    self.assertTrue(all('devices' not in sources[source_id]['spec']['parameters']
                        for source_id in exact))
    self.assertEqual({sources[source_id]['spec']['parameters']['op'] for source_id in near},
                     {'gather_perf', 'scatter_perf', 'sendrecv_perf'})
    self.assertTrue(all(sources[source_id]['spec']['parameters']['devices'] == '0,1'
                        for source_id in near))


def comparable(spec):
  argv = list(spec['argv'])
  index = _runner_index(argv)
  if index is not None:
    command = _runner_options(argv[index + 1:])
    for field in ('timeout', 'launch-mode', 'process-group', 'mpi-options'):
      command.pop(field, None)
    for field, value in (('nnodes', '1'), ('nprocs', '1'), ('threads', '1'),
                         ('redop', 'sum'), ('register', '0')):
      command.setdefault(field, value)
  else:
    if 'source_context' in spec and argv[0] == 'mpirun':
      argv[argv.index('{mpi_options}'):argv.index('{mpi_options}') + 1] = spec['source_context']['mpi']['options']
    command = [token for token in argv if token != '{mpi_options}']
  return {
    'command': command,
    'layout': [spec[key] for key in ('nodes', 'cards_per_node', 'processes_per_node',
                                    'threads_per_process', 'devices_per_thread')],
    'env': {key: value for key, value in spec['env'].items() if key != 'UMD_PLATFORM_TYPE'},
    'checks': sorted(spec['checks']), 'type': spec['test_type'],
    'profile': spec['build_profile'], 'performance': spec.get('performance'),
  }


class SelectedSanityCoverageTests(unittest.TestCase):
  def test_all_historical_sanity_commands_have_unique_selected_source_contexts(self):
    path = MAP.with_name('sanity-coverage.json')
    raw = path.read_text()
    self.assertFalse([marker for marker in PRIVATE_MARKERS if marker in raw])
    report, catalog = read_json(path), load_catalog()
    self.assertEqual(report['bundle_sha256'], digest({key: value for key, value in report.items()
                                                      if key != 'bundle_sha256'}))
    self.assertEqual(report['inputs']['legacy_catalog_sha256'],
                     '0099b6661c757fdb2e8a0419333912a2400a5c46d5edd0fe47c6f3af587853be')
    self.assertEqual(report['inputs']['source_bundle_sha256'], catalog['bundle_sha256'])
    self.assertEqual(report['inputs']['source'], catalog['source'])
    self.assertEqual(report['summary'], {'expected': 96, 'matched': 96, 'missing': 0,
                                         'ambiguous': 0, 'selected_executions': 108})
    self.assertEqual(len(report['cases']), 96)
    self.assertEqual(len({row['legacy_case_key'] for row in report['cases']}), 96)
    self.assertEqual({row['legacy_case_key'] for row in report['cases']},
                     {row['legacy_case_key'] for row in read_json(MAP)['cases']})
    executions = {row['id']: row for row in catalog['executions']}
    used = set()
    for row in report['cases']:
      legacy = row['legacy_spec']
      self.assertEqual(legacy['tier'], 'sanity')
      self.assertEqual(legacy['nodes'], 1)
      identity = {key: value for key, value in legacy.items() if key not in ('tier', 'legacy_source')}
      self.assertEqual(row['legacy_case_key'], legacy['group'] + '-' + digest(identity)[:20])
      self.assertEqual(len(row['selected']), 1)
      match = row['selected'][0]
      source = executions[match['execution_id']]
      self.assertEqual(source['case_id'], match['case_id'])
      self.assertIn(match['build_profile'], source['profiles'])
      pair = (match['execution_id'], match['build_profile'])
      self.assertNotIn(pair, used)
      used.add(pair)
      spec = {**source['spec'], 'build_profile': match['build_profile']}
      self.assertEqual(comparable(legacy), comparable(spec), row['legacy_case_key'])
    self.assertEqual(len(used), 96)

  def test_coverage_contract_does_not_ignore_environment_or_layout(self):
    report = read_json(MAP.with_name('sanity-coverage.json'))
    spec = next(row['legacy_spec'] for row in report['cases']
                if row['legacy_spec']['group'] == 'copyengine')
    for field, value in (('env', {}), ('threads_per_process', 2), ('checks', [])):
      changed = copy.deepcopy(spec)
      changed[field] = value
      self.assertNotEqual(comparable(spec), comparable(changed))


if __name__ == '__main__':
  unittest.main()
