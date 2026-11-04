# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
import json
from pathlib import Path
import unittest

from catalog import load_catalog
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
    sources = {row['source_id']: row for row in catalog['cases']}
    matched = {}
    for classification in ('exact_m890p_supported', 'near_m890p_supported'):
      ids = set()
      rows = [row for row in report['cases'] if row['classification'] == classification]
      for row in rows:
        selected = [match for match in row['matches'] if match['selected']]
        self.assertEqual(len(selected), 1)
        ids.add(selected[0]['source_id'])
      matched[classification] = ids

    exact = matched['exact_m890p_supported']
    near = matched['near_m890p_supported']
    self.assertEqual(len(exact), 6)
    self.assertEqual(len(near), 3)
    self.assertFalse(exact & near)
    self.assertEqual(exact | near, set(sources))
    self.assertTrue(all('devices' not in sources[source_id]['spec']['parameters']
                        for source_id in exact))
    self.assertEqual({sources[source_id]['spec']['parameters']['op'] for source_id in near},
                     {'gather_perf', 'scatter_perf', 'sendrecv_perf'})
    self.assertTrue(all(sources[source_id]['spec']['parameters']['devices'] == '0,1'
                        for source_id in near))


if __name__ == '__main__':
  unittest.main()
