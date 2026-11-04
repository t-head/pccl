# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standard-library consumer for the reviewed pccl_test_suites export."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re

from manifest import case_key, digest, validate_spec


CATALOG = Path(__file__).with_name('source-catalog.json')
EXPECTED_SOURCE_REVISION = 'bbbeb207dd7f28980b55336660da0b47866e47aa'
EXPECTED_BUNDLE_SHA256 = 'abcb4edbb918deda0428f33c93cce14d6b74ddddc9d185a0ef704dca5a1d0078'
EXPECTED_EXPORTER_SHA256 = 'ea39d086b8f0b68536fb67f42966ffd97fade02307bc6dee155d0648fdf359c7'
HEX40 = re.compile(r'[0-9a-f]{40}\Z')
HEX64 = re.compile(r'[0-9a-f]{64}\Z')
SAFE_NAME = re.compile(r'[A-Za-z0-9_.-]+\Z')
PRIVATE_MARKERS = (
  'comm-internal-testbox', '@alibaba-inc.com', 'reg.docker.alibaba-inc.com',
  'UploadResult', 'work_root', 'setup_script', 'test/cases/', 'test/plans/', 'test/suites/',
)


def _read_json(path):
  def record(pairs):
    result = {}
    for key, value in pairs:
      if key in result:
        raise ValueError(f'Duplicate source catalog JSON key: {key}')
      result[key] = value
    return result

  def invalid(value):
    raise ValueError(f'Nonfinite source catalog JSON constant: {value}')

  return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=record,
                    parse_constant=invalid)


def _hex(value, pattern):
  return isinstance(value, str) and pattern.fullmatch(value) is not None


def _safe_source(source):
  fields = {
    'repository', 'revision', 'declared_framework_version', 'exporter_runtime_version',
    'content_sha256', 'exporter_sha256', 'selection_sha256',
  }
  if (not isinstance(source, dict) or set(source) != fields
      or source['repository'] != 'pccl_test_suites'
      or not _hex(source['revision'], HEX40)
      or any(not _hex(source[name], HEX64)
             for name in ('content_sha256', 'exporter_sha256', 'selection_sha256'))
      or source['declared_framework_version'] != '0.12.0'
      or source['exporter_runtime_version'] not in ('0.12.0', '0.14.0')):
    raise ValueError('Invalid pccl_test_suites source identity')


def _safe_case_source(source):
  fields = {'plan', 'suite', 'case', 'plan_sha256', 'suite_sha256', 'case_sha256'}
  if (not isinstance(source, dict) or set(source) != fields
      or any(not isinstance(source[name], str) or not SAFE_NAME.fullmatch(source[name])
             for name in ('plan', 'suite', 'case'))
      or any(not _hex(source[name], HEX64)
             for name in ('plan_sha256', 'suite_sha256', 'case_sha256'))):
    raise ValueError('Invalid public Case source identity')


def load_catalog(path=None, expected_source=None):
  path = Path(path or os.environ.get('PCCL_SOURCE_CATALOG', CATALOG)).resolve()
  raw = path.read_text(encoding='utf-8')
  if any(marker in raw for marker in PRIVATE_MARKERS):
    raise ValueError('Source catalog contains a forbidden private marker')
  bundle = _read_json(path)
  fields = {'schema_version', 'source', 'cases', 'selections', 'coverage', 'bundle_sha256'}
  if (not isinstance(bundle, dict) or set(bundle) != fields
      or type(bundle['schema_version']) is not int or bundle['schema_version'] != 1
      or not _hex(bundle['bundle_sha256'], HEX64)
      or bundle['bundle_sha256'] != digest({key: value for key, value in bundle.items()
                                          if key != 'bundle_sha256'})):
    raise ValueError('Unsupported or corrupted source catalog')
  _safe_source(bundle['source'])
  if (bundle['source']['revision'] != EXPECTED_SOURCE_REVISION
      or bundle['source']['exporter_sha256'] != EXPECTED_EXPORTER_SHA256
      or bundle['bundle_sha256'] != EXPECTED_BUNDLE_SHA256):
    raise ValueError('Source catalog is not the reviewed pccl_test_suites export')
  if expected_source is not None:
    if not isinstance(expected_source, dict) or digest(bundle['source']) != digest(expected_source):
      raise ValueError('Source catalog differs from the trusted invocation')
  if not isinstance(bundle['cases'], list) or not bundle['cases']:
    raise ValueError('Source catalog contains no Cases')
  records = {}
  sources = set()
  profile_counts = {}
  for row in bundle['cases']:
    if (not isinstance(row, dict) or set(row) != {'source_id', 'profiles', 'source', 'spec'}
        or not _hex(row['source_id'], HEX64) or row['source_id'] in records
        or not isinstance(row['profiles'], list) or not row['profiles']
        or set(row['profiles']) - {'standard', 'notsm'}
        or len(set(row['profiles'])) != len(row['profiles'])):
      raise ValueError('Invalid or duplicate exported Case record')
    _safe_case_source(row['source'])
    if not isinstance(row['spec'], dict) or 'build_profile' in row['spec']:
      raise ValueError('Source Case template must not contain a build profile')
    for profile in row['profiles']:
      spec = {**deepcopy(row['spec']), 'build_profile': profile}
      validate_spec(spec)
      if (spec['tier'] != 'sanity' or spec['test_type'] != 'functional' or spec['nodes'] != 1
          or spec['checks'] != ['exit_code']
          or spec['legacy_source'] != f"pccl_test_suites:{row['source_id'][:24]}"):
        raise ValueError('Exported Case does not match its source/profile identity')
      profile_counts[profile] = profile_counts.get(profile, 0) + 1
    sources.add(row['source_id'])
    records[row['source_id']] = row
  selections = bundle['selections']
  if (not isinstance(selections, dict) or set(selections) != {'sanity'}
      or not isinstance(selections['sanity'], list) or len(selections['sanity']) != len(records)
      or len(set(selections['sanity'])) != len(records) or set(selections['sanity']) != set(records)):
    raise ValueError('Source catalog sanity selection is incomplete or duplicated')
  coverage = bundle['coverage']
  expected_coverage = {
    'scope': 'single-node-source-register-sanity', 'profiles': dict(sorted(profile_counts.items())),
    'execution_count': sum(profile_counts.values()), 'source_case_count': len(sources),
  }
  if not isinstance(coverage, dict) or digest(coverage) != digest(expected_coverage):
    raise ValueError('Source catalog coverage summary is inconsistent')
  return bundle


def source_inputs(path=None):
  bundle = load_catalog(path)
  catalog_path = Path(path or os.environ.get('PCCL_SOURCE_CATALOG', CATALOG))
  return {
    'source_catalog_sha256': hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
    'source_catalog_bundle_sha256': bundle['bundle_sha256'],
    'source_catalog_source': deepcopy(bundle['source']),
  }


def cases(tier='sanity', node_counts=(2, 8, 16), debug=False, *, path=None,
          expected_source=None) -> list[dict]:
  del node_counts
  if tier != 'sanity' or debug:
    raise ValueError('The reviewed source catalog currently supports single-node sanity only')
  bundle = load_catalog(path, expected_source)
  by_id = {row['source_id']: row for row in bundle['cases']}
  result = []
  for source_id in bundle['selections']['sanity']:
    row = by_id[source_id]
    for profile in row['profiles']:
      result.append({**deepcopy(row['spec']), 'build_profile': profile})
  return result


def performance_cases(tier='sanity', *, path=None, expected_source=None):
  if tier != 'sanity':
    raise ValueError('The reviewed source catalog currently supports single-node sanity only')
  load_catalog(path, expected_source)
  return []


def available_test_types():
  return ['functional']
