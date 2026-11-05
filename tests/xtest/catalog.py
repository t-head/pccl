# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standard-library consumer for the reviewed pccl_test_suites export."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re

from manifest import digest, read_json, validate_spec


CATALOG = Path(__file__).with_name('source-catalog.json')
EXPECTED_SOURCE_REVISION = 'e27acd364858f845eace0abb72466935070feb0e'
EXPECTED_BUNDLE_SHA256 = 'f86c21880f872c8ebd3e0ea2985e55f8e03d5b58838a2fed4cd61fb9544d2510'
EXPECTED_EXPORTER_SHA256 = '36d5c7cd61b7aa9c15abd014bfa0f061f0f2a329420852bf0a705129a0c8c2d0'
HEX40 = re.compile(r'[0-9a-f]{40}\Z')
HEX64 = re.compile(r'[0-9a-f]{64}\Z')
SAFE_NAME = re.compile(r'[A-Za-z0-9_.-]+\Z')
PRIVATE_MARKERS = (
    'comm-internal-testbox', '@alibaba-inc.com', 'reg.docker.alibaba-inc.com',
    'UploadResult', 'work_root', 'setup_script', 'test/cases/', 'test/plans/', 'test/suites/',
)


def _hex(value, pattern=HEX64):
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _safe_source(source):
    fields = {'repository', 'revision', 'declared_framework_version', 'exporter_runtime_version',
              'content_sha256', 'exporter_sha256', 'selection_sha256'}
    if (not isinstance(source, dict) or set(source) != fields
            or source['repository'] != 'pccl_test_suites'
            or not _hex(source['revision'], HEX40)
            or any(not _hex(source[name]) for name in ('content_sha256', 'exporter_sha256', 'selection_sha256'))
            or source['declared_framework_version'] != '0.12.0'
            or source['exporter_runtime_version'] not in ('0.12.0', '0.14.0')):
        raise ValueError('Invalid pccl_test_suites source identity')


def validate_bundle(bundle, scope):
    fields = {'schema_version', 'source', 'plans', 'suites', 'cases', 'executions',
              'selections', 'coverage', 'bundle_sha256'}
    if (not isinstance(bundle, dict) or set(bundle) != fields
            or type(bundle['schema_version']) is not int or bundle['schema_version'] != 2
            or not _hex(bundle['bundle_sha256'])
            or bundle['bundle_sha256'] != digest({key: value for key, value in bundle.items()
                                                  if key != 'bundle_sha256'})):
        raise ValueError('Unsupported or corrupted source catalog')
    if any(marker in json.dumps(bundle) for marker in PRIVATE_MARKERS):
        raise ValueError('Source catalog contains a forbidden private marker')
    _safe_source(bundle['source'])
    definitions = {}
    for kind, extra in (('plans', 'suite_ids'), ('suites', 'case_ids'), ('cases', 'command')):
        rows = bundle[kind]
        if not isinstance(rows, list) or not rows:
            raise ValueError('Missing source definitions')
        records = {}
        for row in rows:
            if (not isinstance(row, dict) or set(row) != {'id', 'name', 'sha256', extra}
                    or not _hex(row['id']) or row['id'] in records or not _hex(row['sha256'])
                    or not isinstance(row['name'], str) or not SAFE_NAME.fullmatch(row['name'])):
                raise ValueError('Invalid or duplicate source definition')
            if extra == 'command':
                if not isinstance(row[extra], str) or not row[extra].strip():
                    raise ValueError('Missing original Case command')
            elif (not isinstance(row[extra], list) or not row[extra]
                    or any(not _hex(ref) for ref in row[extra]) or len(set(row[extra])) != len(row[extra])):
                raise ValueError('Invalid source definition references')
            records[row['id']] = row
        definitions[kind] = records
    for parent, refs, child in (('plans', 'suite_ids', 'suites'), ('suites', 'case_ids', 'cases')):
        referenced = {ref for row in definitions[parent].values() for ref in row[refs]}
        if referenced != set(definitions[child]):
            raise ValueError('Dangling or unreferenced source definitions')
    if not isinstance(bundle['executions'], list) or not bundle['executions']:
        raise ValueError('Source catalog contains no executions')
    records, contexts, profile_counts = {}, set(), {}
    used = {kind: set() for kind in definitions}
    for row in bundle['executions']:
        if (not isinstance(row, dict)
                or set(row) != {'id', 'plan_id', 'suite_id', 'case_id', 'profiles', 'spec'}
                or not _hex(row['id']) or row['id'] in records
                or not isinstance(row['profiles'], list) or not row['profiles']
                or any(not isinstance(profile, str) for profile in row['profiles'])
                or set(row['profiles']) - {'standard', 'notsm'}
                or len(set(row['profiles'])) != len(row['profiles'])):
            raise ValueError('Invalid or duplicate exported execution')
        for field, kind in (('plan_id', 'plans'), ('suite_id', 'suites'), ('case_id', 'cases')):
            if not _hex(row[field]) or row[field] not in definitions[kind]:
                raise ValueError('Dangling execution reference')
            used[kind].add(row[field])
        plan = definitions['plans'][row['plan_id']]
        suite = definitions['suites'][row['suite_id']]
        case = definitions['cases'][row['case_id']]
        if row['suite_id'] not in plan['suite_ids'] or row['case_id'] not in suite['case_ids']:
            raise ValueError('Execution does not belong to the selected Plan/Suite')
        identity = tuple(row[field] for field in ('plan_id', 'suite_id', 'case_id'))
        if identity in contexts:
            raise ValueError('Duplicate Plan/Suite/Case execution context')
        contexts.add(identity)
        if not isinstance(row['spec'], dict) or 'build_profile' in row['spec']:
            raise ValueError('Source execution template must not contain a build profile')
        for profile in row['profiles']:
            spec = {**deepcopy(row['spec']), 'build_profile': profile}
            validate_spec(spec)
            execution = spec.get('execution', {})
            if (any(execution.get(field) != row[field] for field in ('plan_id', 'suite_id', 'case_id'))
                    or execution.get('source_command') != case['command']
                    or spec['tier'] != 'sanity'
                    or 'exit_code' not in spec['checks']
                    or not spec['legacy_source'].startswith('pccl_test_suites:')):
                raise ValueError('Exported execution differs from its source definition')
            profile_counts[profile] = profile_counts.get(profile, 0) + 1
        records[row['id']] = row
    expected_contexts = {(plan['id'], suite_id, case_id)
                         for plan in definitions['plans'].values() for suite_id in plan['suite_ids']
                         for case_id in definitions['suites'][suite_id]['case_ids']}
    if contexts != expected_contexts:
        raise ValueError('Source reference graph has missing execution contexts')
    if any(used[kind] != set(definitions[kind]) for kind in definitions):
        raise ValueError('Source definition has no selected execution')
    selections = bundle['selections']
    if (not isinstance(selections, dict) or set(selections) != {'sanity'}
            or not isinstance(selections['sanity'], list)
            or any(not _hex(value) for value in selections['sanity'])
            or len(selections['sanity']) != len(records) or set(selections['sanity']) != set(records)):
        raise ValueError('Source catalog sanity selection is incomplete or duplicated')
    expected = {'scope': scope, 'profiles': dict(sorted(profile_counts.items())),
                'execution_count': sum(profile_counts.values()), 'source_case_count': len(definitions['cases']),
                'plan_count': len(definitions['plans']), 'suite_count': len(definitions['suites']),
                'context_count': len(records)}
    if digest(bundle['coverage']) != digest(expected):
        raise ValueError('Source catalog coverage summary is inconsistent')
    return bundle


def load_catalog(path=None, expected_source=None):
    path = Path(path or os.environ.get('PCCL_SOURCE_CATALOG', CATALOG)).resolve()
    bundle = validate_bundle(read_json(path), 'single-node-source-sanity')
    if (bundle['source']['revision'] != EXPECTED_SOURCE_REVISION
            or bundle['source']['exporter_sha256'] != EXPECTED_EXPORTER_SHA256
            or bundle['bundle_sha256'] != EXPECTED_BUNDLE_SHA256):
        raise ValueError('Source catalog is not the reviewed pccl_test_suites export')
    if expected_source is not None and digest(bundle['source']) != digest(expected_source):
        raise ValueError('Source catalog differs from the trusted invocation')
    if any(row['spec']['nodes'] != 1 for row in bundle['executions']):
        raise ValueError('Single-node catalog cannot request multiple nodes')
    return bundle


def source_inputs(path=None):
    bundle = load_catalog(path)
    catalog_path = Path(path or os.environ.get('PCCL_SOURCE_CATALOG', CATALOG))
    return {'source_catalog_sha256': hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
            'source_catalog_bundle_sha256': bundle['bundle_sha256'],
            'source_catalog_source': deepcopy(bundle['source'])}


def cases(tier='sanity', node_counts=(2, 8, 16), debug=False, *, path=None, expected_source=None):
    if tier != 'sanity' or debug:
        raise ValueError('The reviewed source catalog currently supports single-node sanity only')
    bundle = load_catalog(path, expected_source)
    by_id = {row['id']: row for row in bundle['executions']}
    return [{**deepcopy(by_id[identity]['spec']), 'build_profile': profile}
            for identity in bundle['selections'][tier] for profile in by_id[identity]['profiles']
            if by_id[identity]['spec']['test_type'] == 'functional']


def performance_cases(tier='sanity', *, path=None, expected_source=None):
    if tier != 'sanity':
        raise ValueError('The reviewed source catalog currently supports single-node sanity only')
    bundle = load_catalog(path, expected_source)
    return [{**deepcopy(row['spec']), 'build_profile': profile}
            for row in bundle['executions'] for profile in row['profiles']
            if row['spec']['test_type'] == 'performance']


def available_test_types():
    return ['functional', 'performance']
