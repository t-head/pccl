# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stdlib-only manifest checks; CLI tests require the pinned PCCL_TESTS_ROOT checkout."""

import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from manifest import (build_manifest, case_key, digest, load_manifest, measurement_identity,
                      validate_manifest, validate_selection, write_bundle)


def fixture_spec(index=0):
    return dict(group='fixture', tier='sanity', test_type='functional', build_profile='standard',
                nodes=1, cards_per_node=8, processes_per_node=1, threads_per_process=1,
                devices_per_thread=1, argv=['{python}', '-c', f'print({index})'], env={},
                timeout=5, checks=['exit_code'], legacy_source=f'fixture:{index}', parameters={'index': index})


def source_spec(index=0, plan='plan', suite='suite', reuse=False):
    spec = fixture_spec(index)
    parents = {'plan_id': digest(plan), 'suite_id': digest(suite)}
    spec['execution'] = {
        **parents, 'case_id': digest(index), 'suite_run_id': digest(parents),
        'source_command': f'python3 run_test.py --index {index}',
        'mpi': {'options': [], 'exports': ['PATH', 'LD_LIBRARY_PATH', 'UMD_PLATFORM_TYPE']},
        'resource': {'nodes': 1, 'cards_per_node': 8, 'labels': ['board'], 'topology': 'topo'},
        'environment': {'env_file_sha256': digest('env'), 'setup_sha256': None,
                        'teardown_sha256': None, 'omitted': [],
                        'path_bindings': {'PATH': 'sdk-and-build', 'LD_LIBRARY_PATH': 'sdk-and-build'}},
        'lifecycle': {'setup': [], 'teardown': [], 'reuse': reuse},
        'runtime': {'image': 'public/image:1', 'node_selector': 'board-type=test',
                    'container_options': [], 'mounts': [], 'initialization': 'pccl-source-build-v1'},
    }
    spec['env'] = {'UMD_PLATFORM_TYPE': '1'}
    return spec


def performance_specs(tier='sanity'):
    result = []
    for sample in range(3):
        spec = dict(
            group='synthetic_performance', tier=tier, test_type='performance', build_profile='standard',
            nodes=1, cards_per_node=8, processes_per_node=1, threads_per_process=2,
            devices_per_thread=1,
            argv=['{binary_dir}/all_reduce_perf', '-b', '8388608', '-e', '8388608', '-n', '100',
                  '-w', '5', '-d', 'float', '-o', 'sum', '-t', '2', '-g', '1', '-c', '1',
                  '-k', '2', '-a', '2', '--elapsed_type', '0', '--report_type', '0', '-N', '1'],
            env={'PCCL_ALGO': 'Ring', 'NCCL_ALGO': 'Ring', 'PCCL_PROTO': 'Simple',
                 'NCCL_PROTO': 'Simple', 'PCCL_DEBUG_FILE': '{result_dir}/library.%h.%p.log',
                 'NCCL_DEBUG_FILE': '{result_dir}/library.%h.%p.log'},
            timeout=600, checks=['exit_code', 'validation_summary', 'performance'],
            legacy_source='synthetic:performance',
            parameters={'op': 'all_reduce_perf', 'algo': 'Ring', 'proto': 'Simple', 'dtype': 'float',
                        'redop': 'sum', 'data_sizes': ['8M'], 'ranks': 2, 'iters': 100, 'warmup': 5},
            performance={'expected_sizes': [8388608], 'time_kind': 'cpu_time',
                         'rank_reduction': 'MIN', 'sample_index': sample},
        )
        result.append(spec)
    return result


class ManifestTests(unittest.TestCase):
    def test_suite_is_atomic_even_when_many_shards_are_requested(self):
        specs = [source_spec(index) for index in range(12)]
        result = build_manifest(specs, {}, shards_per_pool=8)
        self.assertEqual(len(result['shards']), 1)
        self.assertEqual(len(result['shards'][0]['suite_runs']), 1)
        self.assertEqual(len(result['shards'][0]['suite_runs'][0]['case_keys']), 12)
        validate_manifest(result)

    def test_compatible_suites_only_share_resources_with_explicit_reuse(self):
        for reuse, count in ((False, 2), (True, 1)):
            specs = [source_spec(0, suite='a', reuse=reuse), source_spec(0, suite='b', reuse=reuse)]
            result = build_manifest(specs, {})
            self.assertEqual(result['case_count'], 2)
            self.assertEqual(len(result['shards']), count)
            self.assertEqual(sum(len(shard['suite_runs']) for shard in result['shards']), 2)
            self.assertNotEqual(case_key(specs[0]), case_key(specs[1]))
            self.assertEqual(result, build_manifest(list(reversed(specs)), {}))
            validate_manifest(result)

    def test_performance_suites_never_automatically_share_an_allocation(self):
        specs = []
        for suite in ('measurement-a', 'measurement-b'):
            for spec in performance_specs():
                spec['execution'] = source_spec(suite=suite, reuse=True)['execution']
                specs.append(spec)
        result = build_manifest(specs, {}, shards_per_pool=8)
        self.assertEqual(len(result['shards']), 2)
        self.assertTrue(all(len(shard['cases']) == 3 for shard in result['shards']))
        self.assertTrue(all(len(shard['suite_runs']) == 1 for shard in result['shards']))
        validate_manifest(result)

    def test_environment_container_topology_and_initialization_isolate_suites(self):
        for path, value in ((('env', 'UMD_PLATFORM_TYPE'), '2'),
                            (('execution', 'runtime', 'image'), 'public/image:2'),
                            (('execution', 'resource', 'topology'), 'other'),
                            (('execution', 'environment', 'setup_sha256'), digest('new-setup')),
                            (('execution', 'lifecycle', 'setup'), [['{python}', '-c', 'print(1)']])):
            specs = [source_spec(0, suite='a', reuse=True), source_spec(0, suite='b', reuse=True)]
            target = specs[1]
            for part in path[:-1]:
                target = target[part]
            target[path[-1]] = value
            with self.subTest(path=path):
                self.assertEqual(len(build_manifest(specs, {})['shards']), 2)
        specs = [source_spec(0), source_spec(1)]
        specs[1]['execution']['resource']['topology'] = 'other'
        with self.assertRaisesRegex(ValueError, 'Suite execution'):
            build_manifest(specs, {})

    def test_shared_definitions_keep_all_plan_suite_case_executions(self):
        from catalog import validate_bundle

        plans, suites, definitions, executions = {}, {}, {}, []
        for plan, suite in (('plan-a', 'suite-a'), ('plan-b', 'suite-a'), ('plan-a', 'suite-b')):
            spec = source_spec(0, plan=plan, suite=suite)
            spec.pop('build_profile')
            context = spec['execution']
            plan_id, suite_id, case_id = (context[name] for name in ('plan_id', 'suite_id', 'case_id'))
            plans.setdefault(plan_id, dict(id=plan_id, name=plan, sha256=digest(plan), suite_ids=[]))
            if suite_id not in plans[plan_id]['suite_ids']:
                plans[plan_id]['suite_ids'].append(suite_id)
            suites[suite_id] = dict(id=suite_id, name=suite, sha256=digest(suite), case_ids=[case_id])
            definitions[case_id] = dict(id=case_id, name='case', sha256=digest('case'),
                                        command=context['source_command'])
            spec['legacy_source'] = 'pccl_test_suites:' + case_id[:24]
            executions.append(dict(id=digest([plan, suite, spec]), plan_id=plan_id, suite_id=suite_id,
                                   case_id=case_id, profiles=['standard'], spec=spec))
        bundle = dict(schema_version=2,
                      source=dict(repository='pccl_test_suites', revision='a' * 40,
                                  declared_framework_version='0.12.0', exporter_runtime_version='0.12.0',
                                  content_sha256='b' * 64, exporter_sha256='c' * 64, selection_sha256='d' * 64),
                      plans=list(plans.values()), suites=list(suites.values()), cases=list(definitions.values()),
                      executions=executions, selections={'sanity': [row['id'] for row in executions]},
                      coverage=dict(scope='fixture', profiles={'standard': 3}, execution_count=3,
                                    source_case_count=1, plan_count=2, suite_count=2, context_count=3))
        bundle['bundle_sha256'] = digest(bundle)
        validate_bundle(bundle, 'fixture')
        expanded = [{**row['spec'], 'build_profile': 'standard'} for row in executions]
        manifest = build_manifest(expanded, {})
        self.assertEqual(manifest['case_count'], 3)
        self.assertEqual(len(manifest['shards']), 3)
        missing = copy.deepcopy(bundle)
        removed = missing['executions'].pop(0)
        missing['selections']['sanity'].remove(removed['id'])
        missing['coverage'].update(profiles={'standard': 2}, execution_count=2, context_count=2)
        missing['bundle_sha256'] = digest({key: value for key, value in missing.items()
                                          if key != 'bundle_sha256'})
        with self.assertRaisesRegex(ValueError, 'missing execution contexts'):
            validate_bundle(missing, 'fixture')

    def test_resolved_source_context_rejects_dangling_and_unsupported_fields(self):
        for patch in ({'suite_run_id': '0' * 64}, {'unexpected': True},
                      {'mpi': {'options': ['{unknown}'], 'exports': []}},
                      {'lifecycle': {'setup': [['sh', '{secret}']], 'teardown': [], 'reuse': False}}):
            spec = source_spec()
            spec['execution'].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                build_manifest([spec], {})

    def test_source_diagnostic_bindings_match_performance_environment(self):
        spec = performance_specs()[0]
        spec['execution'] = source_spec()['execution']
        bindings = spec['execution']['environment']['path_bindings']
        bindings.update({name: spec['env'][name] for name in ('PCCL_DEBUG_FILE', 'NCCL_DEBUG_FILE')})
        validate_manifest(build_manifest([spec], {}))
        for name, value in (('NCCL_DEBUG_FILE', 'library.%h.%p.log'),
                            ('PCCL_DEBUG_FILE', '/tmp/library.%h.%p.log'), ('UNKNOWN_PATH', 'other')):
            changed = copy.deepcopy(spec)
            changed['execution']['environment']['path_bindings'][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                build_manifest([changed], {})
        functional = source_spec()
        functional['execution']['environment']['path_bindings'] = dict(bindings)
        with self.assertRaisesRegex(ValueError, 'Diagnostic path bindings require'):
            build_manifest([functional], {})
        changed = copy.deepcopy(spec)
        changed['env']['PCCL_DEBUG_FILE'] = 'library.%h.%p.log'
        with self.assertRaisesRegex(ValueError, 'Diagnostic path bindings differ'):
            build_manifest([changed], {})

    def test_empty_duplicate_and_unsupported_cases_fail(self):
        for records in ([], [fixture_spec(), fixture_spec()]):
            with self.assertRaises(ValueError):
                build_manifest(records, {})
        for patch in ({'timeout': 0}, {'nodes': True}, {'devices_per_thread': 9},
                      {'checks': []}, {'test_type': 'performance'}, {'argv': []},
                      {'parameters': {'transport': '../escape'}},
                      {'env': {'BAD-NAME': '1'}}, {'env': {'9INVALID': '1'}},
                      {'env': {'UCX_FIXTURE': 1}}, {'env': {'UCX_FIXTURE': True}},
                      {'env': {'UCX_FIXTURE': 'before\x00after'}},
                      {'env': {'UCX_FIXTURE': '{unknown}'}},
                      {'env': {'UCX_FIXTURE': '{result_dir'}},
                      {'argv': ['{python}', '-c', 'print(0)\x00']},
                      {'argv': ['{unknown}/runner.py']}, {'argv': ['{python']},
                      {'argv': ['{python}}']}, {'argv': ['{{python}']}):
            spec = fixture_spec()
            spec.update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                build_manifest([spec], {})

    def test_key_ignores_tier_and_source_line_but_not_build(self):
        spec = fixture_spec()
        expected = case_key(spec)
        spec.update(tier='nightly', legacy_source='moved.py:90')
        self.assertEqual(case_key(spec), expected)
        spec['build_profile'] = 'notsm'
        self.assertNotEqual(case_key(spec), expected)

    def test_deterministic_partition_and_history(self):
        specs = [fixture_spec(i) for i in range(20)]
        timings = {case_key(spec): i + 1 for i, spec in enumerate(specs)}
        result = build_manifest(specs, {'run_id': '5'}, 4, timings)
        self.assertEqual(result, build_manifest(list(reversed(specs)), {'run_id': '5'}, 4, timings))
        self.assertEqual(result['case_count'], 20)
        self.assertEqual(result['sharding'], {'shards_per_pool': 4, 'timings': timings})
        validate_manifest(result)
        keys = [row['case_key'] for shard in result['shards'] for row in shard['cases']]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(set(keys), set(timings))
        loads = [shard['estimated_seconds'] for shard in result['shards']]
        self.assertLessEqual(max(loads) - min(loads), 4)
        for value in (0, -1, float('nan'), float('inf'), '2', True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_manifest(specs, {}, timings={case_key(specs[0]): value})

    def test_resource_pools_and_matrix_limit(self):
        specs = [fixture_spec(i) for i in range(4)]
        specs[1]['build_profile'] = 'notsm'
        specs[2].update(nodes=2, parameters={'transport': 'mnnvl'})
        specs[3].update(nodes=2, parameters={'transport': 'rdma'})
        self.assertEqual(len(build_manifest(specs, {})['shards']), 4)
        many = [fixture_spec(i) for i in range(257)]
        for i, spec in enumerate(many):
            spec['parameters']['transport'] = f'pool-{i}'
        with self.assertRaisesRegex(ValueError, '256'):
            build_manifest(many, {})

    def test_thirty_thousand_cases_stay_out_of_matrix(self):
        result = build_manifest([fixture_spec(i) for i in range(30000)], {}, 14)
        self.assertEqual(result['case_count'], 30000)
        self.assertEqual(len(result['shards']), 14)
        self.assertEqual(sum(len(shard['cases']) for shard in result['shards']), 30000)

    def test_json_only_bundle_round_trip_and_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'bundle'
            manifest = build_manifest([fixture_spec(0), fixture_spec(1)], {})
            self.assertEqual(manifest['schema_version'], 2)
            self.assertEqual(set(manifest), {'schema_version', 'inputs', 'sharding', 'shards',
                                             'case_count', 'selection_sha256'})
            self.assertEqual(manifest['sharding'], {'shards_per_pool': 1, 'timings': {}})
            self.assertTrue(all(set(row) == {'case_key', 'spec'}
                                for shard in manifest['shards'] for row in shard['cases']))
            original = copy.deepcopy(manifest)
            validate_manifest(manifest)
            self.assertEqual(manifest, original)
            write_bundle(manifest, root)
            self.assertEqual(sorted(path.name for path in root.iterdir()), ['manifest.json'])
            self.assertEqual(json.loads((root / 'manifest.json').read_text()), manifest)
            self.assertEqual(load_manifest(root / 'manifest.json'), manifest)
            content = (root / 'manifest.json').read_bytes()
            with self.assertRaises(FileExistsError):
                write_bundle(manifest, root)
            self.assertEqual((root / 'manifest.json').read_bytes(), content)
            empty = Path(temp) / 'empty'
            empty.mkdir()
            with self.assertRaises(FileExistsError):
                write_bundle(manifest, empty)
            self.assertEqual(list(empty.iterdir()), [])
            target = Path(temp) / 'missing-target'
            link = Path(temp) / 'existing-link'
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(FileExistsError):
                write_bundle(manifest, link)
            self.assertTrue(link.is_symlink())
            self.assertFalse(target.exists())

    def test_loading_rejects_duplicate_keys_and_nonfinite_json(self):
        spec = fixture_spec()
        spec['env'] = {'UCX_FIXTURE': 'new'}
        manifest = build_manifest([spec], {'numeric': None})
        content = json.dumps(manifest)
        duplicates = [content.replace('"schema_version": 2', '"schema_version": 2, "schema_version": 2', 1),
                      content.replace('"UCX_FIXTURE": "new"', '"UCX_FIXTURE": "old", "UCX_FIXTURE": "new"', 1)]
        # Last-key-wins decoding would produce the original, correctly hashed manifest.
        for duplicate in duplicates:
            self.assertEqual(json.loads(duplicate), manifest)
        invalid = duplicates + [content.replace('"numeric": null', f'"numeric": {value}', 1)
                                for value in ('NaN', 'Infinity', '-Infinity', '1e999')]
        invalid += ['null', '[]', '{}', content[:-1], content + '{}']
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'manifest.json'
            for index, text in enumerate(invalid):
                with self.subTest(index=index):
                    path.write_text(text)
                    with self.assertRaises(ValueError):
                        load_manifest(path)

    def test_schema_and_canonical_structure_fail_even_with_recomputed_hashes(self):
        original = build_manifest([fixture_spec(i) for i in range(6)], {}, 2)
        mutations = [
            (('schema_version',), 1),
            (('schema_version',), True),
            (('unexpected',), 'extra'),
            (('sharding', 'unexpected'), 'extra'),
            (('sharding', 'shards_per_pool'), 1),
            (('sharding', 'timings'), {case_key(fixture_spec(0)): 100}),
            (('case_count',), 5),
            (('shards',), list(reversed(original['shards']))),
            (('shards', 0, 'cases'), list(reversed(original['shards'][0]['cases']))),
            (('shards', 0, 'estimated_seconds'), 999),
            (('shards', 0, 'unexpected'), 'extra'),
            (('shards', 0, 'cases', 0, 'case_path'), 'cases/test_fixture.py'),
            (('shards', 0, 'cases', 0, 'case_sha256'), '0' * 64),
            (('shards', 0, 'cases', 0, 'case_key'), 'fixture-wrong'),
            (('shards', 0, 'cases', 0, 'spec', 'unexpected'), 'extra'),
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'manifest.json'
            for location, value in mutations:
                with self.subTest(location=location, value=value):
                    manifest = copy.deepcopy(original)
                    target = manifest
                    for key in location[:-1]:
                        target = target[key]
                    target[location[-1]] = value
                    manifest['selection_sha256'] = digest({key: value for key, value in manifest.items()
                                                          if key != 'selection_sha256'})
                    with self.assertRaises(ValueError):
                        validate_manifest(manifest)
                    path.write_text(json.dumps(manifest))
                    with self.assertRaises(ValueError):
                        load_manifest(path)

    def test_data_changes_require_a_new_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'manifest.json'
            original = build_manifest([fixture_spec()], {'run_id': '1'})
            manifest = copy.deepcopy(original)
            manifest['inputs']['run_id'] = '2'
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                validate_manifest(manifest)
            with self.assertRaises(ValueError):
                load_manifest(path)
            manifest['selection_sha256'] = digest({key: value for key, value in manifest.items()
                                                  if key != 'selection_sha256'})
            validate_manifest(manifest)
            path.write_text(json.dumps(manifest))
            self.assertEqual(load_manifest(path), manifest)

    def test_resource_pool_mismatches_fail_with_recomputed_hashes(self):
        original = build_manifest([fixture_spec()], {})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'manifest.json'
            for field, value in (('nodes', 2), ('cards_per_node', 4), ('build_profile', 'notsm'),
                                 ('test_type', 'performance'), ('transport', 'rdma'),
                                 ('shard_id', 'other-shard')):
                with self.subTest(field=field):
                    manifest = copy.deepcopy(original)
                    manifest['shards'][0][field] = value
                    manifest['selection_sha256'] = digest({key: value for key, value in manifest.items()
                                                          if key != 'selection_sha256'})
                    with self.assertRaises(ValueError):
                        validate_manifest(manifest)
                    path.write_text(json.dumps(manifest))
                    with self.assertRaises(ValueError):
                        load_manifest(path)

    def test_failed_write_does_not_publish_manifest(self):
        def fail_during_write(value, stream, **kwargs):
            stream.write('{"schema_version":')
            raise OSError('injected partial write failure')

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'invalid'
            manifest = build_manifest([fixture_spec()], {})
            invalid = copy.deepcopy(manifest)
            invalid['case_count'] += 1
            with self.assertRaises(ValueError):
                write_bundle(invalid, root)
            self.assertFalse((root / 'manifest.json').exists())
            interrupted = Path(temp) / 'interrupted'
            with mock.patch('manifest.json.dump', side_effect=fail_during_write):
                with self.assertRaises(OSError):
                    write_bundle(manifest, interrupted)
            self.assertFalse((interrupted / 'manifest.json').exists())
            blocked = Path(temp) / 'blocked'
            blocked.write_text('not a directory')
            with self.assertRaises(OSError):
                write_bundle(manifest, blocked / 'bundle')
            self.assertFalse((blocked / 'bundle' / 'manifest.json').exists())
            self.assertEqual(blocked.read_text(), 'not a directory')

    def test_full_catalog_selection(self):
        from catalog import available_test_types, cases, load_catalog, performance_cases

        bundle = load_catalog()
        sanity, performance = cases(), performance_cases()
        self.assertEqual(bundle['coverage']['scope'], 'single-node-source-sanity')
        self.assertEqual(bundle['coverage']['source_case_count'], 71)
        self.assertEqual(bundle['coverage']['execution_count'], 108)
        self.assertEqual(bundle['coverage']['profiles'], {'standard': 71, 'notsm': 37})
        self.assertEqual(available_test_types(), ['functional', 'performance'])
        self.assertEqual((len(sanity), len(performance)), (105, 3))
        self.assertEqual(sum(spec['build_profile'] == 'notsm' for spec in sanity), 37)
        self.assertEqual(sum(spec['build_profile'] == 'standard' for spec in sanity), 68)
        self.assertTrue(all(spec['test_type'] == 'functional' for spec in sanity))
        self.assertTrue(all(spec['test_type'] == 'performance' and spec['build_profile'] == 'standard'
                            for spec in performance))
        self.assertTrue(all(spec['nodes'] == 1 and 'exit_code' in spec['checks']
                            for spec in sanity + performance))
        self.assertEqual(len({case_key(spec) for spec in sanity + performance}), 108)
        self.assertEqual(len({spec['execution']['case_id'] for spec in sanity + performance}), 71)
        self.assertEqual({spec['performance']['sample_index'] for spec in performance}, {0, 1, 2})
        self.assertEqual(len({spec['execution']['case_id'] for spec in performance}), 3)
        self.assertEqual(len({measurement_identity(spec) for spec in performance}), 1)
        for select in (cases, performance_cases):
            with self.assertRaises(ValueError):
                select('nightly')
        with self.assertRaises(ValueError):
            cases(debug=True)
        for select, records in ((cases, sanity), (performance_cases, performance)):
            saved = copy.deepcopy(records)
            records[0]['env']['FIXTURE_ONLY'] = '1'
            self.assertEqual(select(), saved)

    def _single_node_cli_command(self, output):
        public_root = os.environ.get('PCCL_TESTS_ROOT')
        self.assertTrue(public_root, 'PCCL_TESTS_ROOT must point to the pinned public runner checkout')
        public_root = Path(public_root).resolve()
        self.assertTrue((public_root / 'run_test.py').is_file())
        repository = Path(__file__).resolve().parents[2]
        revision = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
        return [sys.executable, '-S', '-B', str(Path(__file__).resolve().with_name('manifest.py')),
                '--suite', 'pr', '--single-node', '--pccl-revision', revision,
                '--pccl-tests', str(public_root), '--run-id', '1', '--output', str(output)]

    def test_single_node_cli_preserves_all_single_node_sanity_cases(self):
        from catalog import EXPECTED_SOURCE_REVISION, EXPECTED_BUNDLE_SHA256, cases, performance_cases

        with tempfile.TemporaryDirectory() as temp:
            for types in ([], ['functional'], ['performance'], ['functional', 'performance']):
                with self.subTest(test_types=types):
                    output = Path(temp) / ('-'.join(types) or 'all')
                    command = self._single_node_cli_command(output)
                    if types:
                        command += ['--test-types', *types]
                    run = subprocess.run(command, capture_output=True, text=True, timeout=30)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                    manifest = load_manifest(output / 'manifest.json')
                    inputs = manifest['inputs']
                    self.assertNotIn('xtest_version', inputs)
                    self.assertEqual(inputs['python_version'], platform.python_version())
                    for field, name in (('generator_sha256', 'manifest.py'), ('catalog_sha256', 'catalog.py')):
                        self.assertEqual(inputs[field], hashlib.sha256(
                            Path(__file__).with_name(name).read_bytes()).hexdigest())
                    self.assertEqual(inputs['pccl_revision'], command[command.index('--pccl-revision') + 1])
                    self.assertEqual(inputs['pccl_tests_revision'], '394556512e87259e9ffa25225632c46b0a0bbcd5')
                    self.assertEqual(inputs['runner_sha256'], {
                        'run_test.py': 'fdbdf7623703157b49b9605491fb1cb602fe50ec931ae6763c2645bef7c6b0fc',
                        'run_test_config.py': 'd58194c5c210ebaabe1ed42d7c2ced924c5dde22081285c22516e1ccdf611b1a',
                    })
                    self.assertTrue(inputs['single_node'])
                    self.assertEqual(inputs['node_counts'], [2, 8, 16])
                    self.assertEqual(inputs['shards_per_pool'], 1)
                    self.assertEqual(inputs['timings_sha256'], digest({}))
                    self.assertEqual(inputs['source_catalog_source']['revision'], EXPECTED_SOURCE_REVISION)
                    self.assertEqual(inputs['source_catalog_bundle_sha256'], EXPECTED_BUNDLE_SHA256)
                    selected_types = types or ['functional', 'performance']
                    self.assertEqual(inputs['test_types'], selected_types)
                    self.assertEqual(manifest['case_count'], sum(
                        {'functional': 105, 'performance': 3}[kind] for kind in selected_types))
                    self.assertEqual(sorted(path.name for path in output.iterdir()), ['manifest.json'])
                    matrix = json.loads(run.stdout)
                    self.assertEqual(matrix['include'], [
                        {key: shard[key] for key in ('shard_id', 'nodes', 'build_profile', 'test_type')}
                        for shard in manifest['shards']])
                    rows = [row for shard in manifest['shards'] for row in shard['cases']]
                    expected = [spec for spec in cases() + performance_cases()
                                if spec['test_type'] in selected_types]
                    self.assertEqual({row['case_key']: row['spec'] for row in rows},
                                     {case_key(spec): spec for spec in expected})
                    self.assertTrue(all(row['spec']['nodes'] == 1 for row in rows))
                    validate_selection(manifest, copy.deepcopy(inputs))
                    self.assertEqual(load_manifest(output / 'manifest.json', expected_inputs=inputs), manifest)
            debug_output = Path(temp) / 'debug'
            command = self._single_node_cli_command(debug_output) + ['--debug']
            run = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(run.returncode, 0)
            self.assertFalse((debug_output / 'manifest.json').exists())

    def test_catalog_reconciliation_rejects_rehashed_missing_replaced_and_changed_cases(self):
        from catalog import cases

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'bundle'
            timings = {case_key(spec): index + 1 for index, spec in enumerate(cases()) if spec['nodes'] == 1}
            timing_path = Path(temp) / 'timings.json'
            timing_path.write_text(json.dumps(timings))
            command = self._single_node_cli_command(output) + [
                '--node-counts', '16', '2', '--shards-per-pool', '2',
                '--run-attempt', '3', '--timings', str(timing_path)]
            run = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            repeated = Path(temp) / 'repeated'
            command[command.index('--output') + 1] = str(repeated)
            again = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
            self.assertEqual(run.stdout, again.stdout)
            self.assertEqual((output / 'manifest.json').read_bytes(), (repeated / 'manifest.json').read_bytes())
            original = load_manifest(output / 'manifest.json')
            expected_inputs = copy.deepcopy(original['inputs'])
            self.assertEqual(expected_inputs['node_counts'], [2, 16])
            self.assertEqual(expected_inputs['run_attempt'], 3)
            self.assertEqual(expected_inputs['shards_per_pool'], 2)
            self.assertEqual(expected_inputs['timings_sha256'], digest(timings))
            self.assertEqual(original['sharding'], {'shards_per_pool': 2, 'timings': timings})
            validate_selection(original, expected_inputs)
            specs = [copy.deepcopy(row['spec']) for shard in original['shards'] for row in shard['cases']]
            replacement = copy.deepcopy(specs)
            replacement[0]['timeout'] += 1
            moved = copy.deepcopy(specs)
            moved[0]['legacy_source'] = 'synthetic:moved'
            self.assertNotEqual(case_key(replacement[0]), case_key(specs[0]))
            self.assertEqual(case_key(moved[0]), case_key(specs[0]))
            path = Path(temp) / 'tampered.json'
            for name, records in (('deleted', specs[1:]), ('replaced', replacement),
                                  ('same-key-spec', moved)):
                with self.subTest(change=name):
                    # Rebuild count, identities, shards and hash: only independent catalog selection can reject this.
                    manifest = build_manifest(records, copy.deepcopy(expected_inputs),
                                              original['sharding']['shards_per_pool'],
                                              original['sharding']['timings'])
                    validate_manifest(manifest)
                    path.write_text(json.dumps(manifest))
                    self.assertEqual(load_manifest(path), manifest)
                    with self.assertRaises(ValueError):
                        validate_selection(manifest, expected_inputs)
                    with self.assertRaises(ValueError):
                        load_manifest(path, expected_inputs=expected_inputs)
            for name, inputs in (('changed', {**expected_inputs, 'run_id': '2'}),
                                 ('extra', {**expected_inputs, 'unexpected': True}),
                                 ('missing', {key: value for key, value in expected_inputs.items() if key != 'run_id'})):
                with self.subTest(inputs=name):
                    with self.assertRaises(ValueError):
                        validate_selection(original, inputs)
                    with self.assertRaises(ValueError):
                        load_manifest(output / 'manifest.json', expected_inputs=inputs)
            changed_inputs = {**expected_inputs, 'run_id': '2'}
            manifest = build_manifest(specs, changed_inputs, original['sharding']['shards_per_pool'],
                                      original['sharding']['timings'])
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_manifest(path, expected_inputs=expected_inputs)
            for count, history in ((1, timings), (2, {})):
                with self.subTest(sharding=count, history=bool(history)):
                    manifest = build_manifest(specs, copy.deepcopy(expected_inputs), count, history)
                    validate_manifest(manifest)
                    path.write_text(json.dumps(manifest))
                    with self.assertRaises(ValueError):
                        validate_selection(manifest, expected_inputs)
                    with self.assertRaises(ValueError):
                        load_manifest(path, expected_inputs=expected_inputs)

    def test_cli_rejects_missing_runner_and_incorrect_pccl_revision_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            for mode in ('missing-runner', 'wrong-pccl-revision', 'wrong-public-checkout'):
                with self.subTest(mode=mode):
                    output = Path(temp) / mode
                    command = self._single_node_cli_command(output)
                    if mode == 'missing-runner':
                        index = command.index('--pccl-tests')
                        del command[index:index + 2]
                    elif mode == 'wrong-pccl-revision':
                        index = command.index('--pccl-revision') + 1
                        revision = command[index]
                        command[index] = ('0' if revision[0] != '0' else '1') + revision[1:]
                    else:
                        command[command.index('--pccl-tests') + 1] = str(Path(__file__).resolve().parents[2])
                    run = subprocess.run(command, capture_output=True, text=True, timeout=30)
                    self.assertNotEqual(run.returncode, 0)
                    self.assertFalse((output / 'manifest.json').exists())

    def test_cli_rejects_modified_files_at_the_pinned_public_revision(self):
        with tempfile.TemporaryDirectory() as temp:
            command = self._single_node_cli_command(Path(temp) / 'unused')
            public_root = Path(command[command.index('--pccl-tests') + 1])
            checkout = Path(temp) / 'public-copy'
            clone = subprocess.run(['git', 'clone', '--shared', '--no-checkout', '--quiet',
                                    str(public_root), str(checkout)],
                                   capture_output=True, text=True, timeout=30)
            self.assertEqual(clone.returncode, 0, clone.stdout + clone.stderr)
            revision = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
            self.assertEqual(revision, '394556512e87259e9ffa25225632c46b0a0bbcd5')
            contents = {name: (public_root / name).read_bytes() for name in ('run_test.py', 'run_test_config.py')}
            for name, content in contents.items():
                (checkout / name).write_bytes(content)
            command[command.index('--pccl-tests') + 1] = str(checkout)
            run = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            for name, content in contents.items():
                with self.subTest(file=name):
                    output = Path(temp) / name
                    command = self._single_node_cli_command(output)
                    command[command.index('--pccl-tests') + 1] = str(checkout)
                    (checkout / name).write_bytes(content + b'\n# Modified fixture; HEAD is still pinned.\n')
                    run = subprocess.run(command, capture_output=True, text=True, timeout=30)
                    self.assertNotEqual(run.returncode, 0)
                    self.assertFalse((output / 'manifest.json').exists())
                    (checkout / name).write_bytes(content)

    def test_performance_selection_is_separate_and_bound_to_measurement(self):
        specs = performance_specs()
        self.assertEqual(len(specs), 3)
        self.assertEqual({spec['performance']['sample_index'] for spec in specs}, {0, 1, 2})
        for tier in ('sanity', 'nightly'):
            manifest = build_manifest([fixture_spec()] + performance_specs(tier), {})
            self.assertEqual(manifest['case_count'], 4)
            self.assertTrue(all(len({row['spec']['test_type'] for row in shard['cases']}) == 1
                                for shard in manifest['shards']))
            for shard in manifest['shards']:
                for row in shard['cases']:
                    expected_fields = {'case_key', 'spec'}
                    if row['spec']['test_type'] == 'performance':
                        expected_fields.add('measurement_identity')
                        self.assertEqual(row['measurement_identity'], measurement_identity(row['spec']))
                    self.assertEqual(set(row), expected_fields)
        for field, value in (('argv', specs[0]['argv'] + ['-c', '0']), ('nodes', 2),
                             ('threads_per_process', 1), ('performance', None)):
            spec = copy.deepcopy(specs[0])
            spec[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_manifest([spec], {})
        for field, value in (('rank_reduction', 'AVG'), ('time_kind', 'event_time'),
                             ('sample_index', -1), ('expected_sizes', [16, 32])):
            spec = copy.deepcopy(specs[0])
            spec['performance'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_manifest([spec], {})
        for key in ('PCCL_ALGO', 'NCCL_PROTO', 'PCCL_DEBUG_FILE'):
            spec = copy.deepcopy(specs[0])
            spec['env'][key] = 'mismatched'
            with self.subTest(key=key), self.assertRaises(ValueError):
                build_manifest([spec], {})

    def test_measurement_identity_excludes_only_sample_and_scheduling_metadata(self):
        specs = performance_specs()
        before = copy.deepcopy(specs[0])
        expected = measurement_identity(specs[0])
        self.assertEqual(specs[0], before)
        self.assertEqual(len({case_key(spec) for spec in specs}), 3)
        self.assertEqual({measurement_identity(spec) for spec in specs}, {expected})
        other = copy.deepcopy(specs[0])
        other.update(tier='nightly', legacy_source='synthetic:moved', timeout=900, checks=['exit_code'])
        self.assertEqual(measurement_identity(other), expected)
        other['env'] = dict(reversed(list(other['env'].items())))
        self.assertEqual(measurement_identity(other), expected)
        for field, value in (('argv', specs[0]['argv'] + ['-f', '2']),
                             ('env', {**specs[0]['env'], 'UCX_SYNTHETIC': 'private-value'}),
                             ('parameters', {**specs[0]['parameters'], 'synthetic_knob': 2}),
                             ('nodes', 2), ('cards_per_node', 4), ('processes_per_node', 2),
                             ('threads_per_process', 1), ('devices_per_thread', 2), ('build_profile', 'notsm')):
            other = copy.deepcopy(specs[0])
            other[field] = value
            with self.subTest(field=field):
                self.assertNotEqual(measurement_identity(other), expected)
        self.assertTrue(all(spec['tier'] == 'nightly' for spec in performance_specs('nightly')))

    def test_measurement_identity_excludes_native_case_id_but_retains_source_context(self):
        specs = performance_specs()
        for sample, spec in enumerate(specs):
            spec['execution'] = source_spec()['execution']
            spec['execution']['case_id'] = digest(sample)
        before = copy.deepcopy(specs)
        expected = measurement_identity(specs[0])
        self.assertEqual(len({case_key(spec) for spec in specs}), 3)
        self.assertEqual({measurement_identity(spec) for spec in specs}, {expected})
        self.assertEqual(specs, before)
        for field, value in (('plan_id', digest('other-plan')), ('suite_id', digest('other-suite')),
                             ('suite_run_id', digest('other-run')), ('source_command', 'different command'),
                             ('mpi', {'options': ['--oversubscribe'], 'exports': []})):
            changed = copy.deepcopy(specs[0])
            changed['execution'][field] = value
            with self.subTest(field=field):
                self.assertNotEqual(measurement_identity(changed), expected)

    def test_performance_samples_are_atomic_for_two_or_fourteen_shards(self):
        specs = performance_specs()
        timings = {case_key(spec): index + 1 for index, spec in enumerate(specs)}
        for count in (2, 14):
            with self.subTest(shards_per_pool=count):
                result = build_manifest(specs, {}, count, timings)
                self.assertEqual(len(result['shards']), 1)
                shard, = result['shards']
                self.assertEqual(len(shard['cases']), 3)
                self.assertEqual(shard['estimated_seconds'], 6)
                self.assertEqual({row['spec']['performance']['sample_index'] for row in shard['cases']}, {0, 1, 2})

    def test_different_performance_measurements_still_use_lpt_shards(self):
        specs, timings = [], {}
        for channels, duration in (('1', 9), ('2', 6), ('4', 3)):
            for spec in performance_specs():
                spec['env'].update(PCCL_MAX_NCHANNELS=channels, NCCL_MAX_NCHANNELS=channels)
                specs.append(spec)
                timings[case_key(spec)] = duration
        for count, loads in ((2, [27, 27]), (14, [9, 18, 27])):
            with self.subTest(shards_per_pool=count):
                result = build_manifest(specs, {}, count, timings)
                self.assertEqual(result, build_manifest(list(reversed(specs)), {}, count, timings))
                self.assertEqual(sorted(shard['estimated_seconds'] for shard in result['shards']), loads)
                locations = {}
                for shard in result['shards']:
                    for row in shard['cases']:
                        locations.setdefault(measurement_identity(row['spec']), []).append(shard['shard_id'])
                self.assertEqual(len(locations), 3)
                self.assertTrue(all(len(shards) == 3 and len(set(shards)) == 1 for shards in locations.values()))

    def test_loading_split_performance_json_fails_closed(self):
        specs = performance_specs()
        other = copy.deepcopy(specs)
        for spec in other:
            spec['env'].update(PCCL_MAX_NCHANNELS='2', NCCL_MAX_NCHANNELS='2')
        manifest = build_manifest(specs + other, {}, 2)
        # Model old per-Case LPT splitting, even with the loads and checksum recomputed.
        first, second = manifest['shards']
        second['cases'].append(first['cases'].pop())
        second['cases'].sort(key=lambda row: row['case_key'])
        first['estimated_seconds'] -= 1
        second['estimated_seconds'] += 1
        manifest['selection_sha256'] = digest({key: value for key, value in manifest.items()
                                              if key != 'selection_sha256'})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'manifest.json'
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                validate_manifest(manifest)
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_duplicate_performance_sample_with_distinct_case_key_fails(self):
        specs = performance_specs()
        specs[2]['performance']['sample_index'] = specs[0]['performance']['sample_index']
        specs[2]['timeout'] += 1
        self.assertEqual(len({case_key(spec) for spec in specs}), 3)
        self.assertEqual(len({measurement_identity(spec) for spec in specs}), 1)
        manifest = build_manifest(performance_specs(), {})
        shard, = manifest['shards']
        row = next(row for row in shard['cases'] if row['spec']['performance']['sample_index'] == 2)
        row['spec'] = specs[2]
        row['case_key'] = case_key(specs[2])
        row['measurement_identity'] = measurement_identity(specs[2])
        shard['cases'].sort(key=lambda row: row['case_key'])
        manifest['selection_sha256'] = digest({key: value for key, value in manifest.items()
                                              if key != 'selection_sha256'})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'manifest.json'
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                validate_manifest(manifest)
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_performance_measurement_identity_is_required_and_verified(self):
        original = build_manifest(performance_specs(), {})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'manifest.json'
            for identity in (None, '0' * 64):
                with self.subTest(identity=identity):
                    manifest = copy.deepcopy(original)
                    row = manifest['shards'][0]['cases'][0]
                    if identity is None:
                        del row['measurement_identity']
                    else:
                        row['measurement_identity'] = identity
                    manifest['selection_sha256'] = digest({key: value for key, value in manifest.items()
                                                          if key != 'selection_sha256'})
                    path.write_text(json.dumps(manifest))
                    with self.assertRaises(ValueError):
                        validate_manifest(manifest)
                    with self.assertRaises(ValueError):
                        load_manifest(path)

    def test_source_catalog_commands_are_discovered_and_profile_expanded(self):
        from catalog import cases, load_catalog, performance_cases

        # These nine native definitions predate the coverage expansion. Their identities
        # and source hashes must survive, not merely an equivalent synthetic command.
        original = {
            '0db51deea804adb6a3d019394e718e55c35ec19e6851671706334188c6cfe5ef':
                'c7797d9f989428961a8ecf2fbf4e79a89de3cfcce499111fc5f4640e13da957e',
            '19035a0c32746db0ffc0368bfd2cd3d85c489ac2ced0956f10e8506e11b363ac':
                '05556844e0910812d7a96df28dccb30091617cda05a1b30d7bd1a0a5ba31b785',
            '1b8ad645dedd2f6cc0a39a2da2eae093cf2f7e4867903c4f8724ef2846b58d02':
                '5357322e5a99235593e6615e18492cc9a4d18d1ed3e4202e8f40564da6e243ea',
            '2c0df19e2a2317367521d366068bcd99b079efc12c45b4b7b344686f63d338f1':
                '25b1a5556fb2b14809cd24a99a617e3d0e4de5ba1edab1f0d9320d4dd892e5d6',
            '2db93f48602d163a751256c930b5a7d5e1c2c0f1b0b05181b1c420ab4c73f9fa':
                '4ee479a7c3e6bc3eee3ca065643be72c341dadec043a589948db31fb9f118fdb',
            '7629b3a15935d3e2281c594d01b583c51662d61375256b9c7528541c12d7b2f8':
                'b080f654d5b16df3809a01ea65478b8599ceb21de74a45cf577b999d4a6ed77a',
            '88fc6b809e8e1071a0f6bc941063d0ce1f335513d5ea6f09f79a0b62fd8830e7':
                '2cfb0f696f38f85339ea9ae14de0f40d57652a26cb846f8e858119f6625874eb',
            'c91f5390c82212652d2c1082e6c12698383c814c502cf47a4a6cf62756d47fe6':
                '0e9a7887f3da9050a2e2db04421aabbd79a85cebadc28ed433747296556a537a',
            'f033bd5bafd9d77d66c676655d7bdae3520ce0dcbe39d63ff7babf3af4257f3c':
                '3332464a674c701c77c92508d1d248c654b64660f8b25d0edbc5d8628121f191',
        }
        bundle = load_catalog()
        definitions = {row['id']: row for row in bundle['cases']}
        specs = cases() + performance_cases()
        self.assertEqual(len(definitions), 71)
        self.assertEqual({identity: definitions[identity]['sha256'] for identity in original}, original)
        self.assertEqual(len(specs), 108)
        new_functional = {spec['execution']['case_id'] for spec in cases()} - original.keys()
        self.assertEqual(len(new_functional), 59)
        preserved = [spec for spec in specs if spec['execution']['case_id'] in original]
        self.assertEqual(len(preserved), 18)
        self.assertEqual(sum('devices' in spec['parameters'] for spec in preserved), 6)
        for identity in original:
            self.assertEqual({spec['build_profile'] for spec in preserved
                              if spec['execution']['case_id'] == identity}, {'standard', 'notsm'})
        for spec in preserved:
            self.assertEqual(spec['group'], 'source_register')
            self.assertEqual(spec['parameters']['register'], 1)
            self.assertEqual(spec['timeout'], 7200)
            self.assertEqual(spec['checks'], ['exit_code'])
        for spec in specs:
            context = spec['execution']
            self.assertEqual(context['source_command'], definitions[context['case_id']]['command'])
            self.assertTrue(spec['legacy_source'].startswith('pccl_test_suites:'))
            if '{test_root}/run_test.py' in spec['argv']:
                self.assertEqual(spec['argv'][:2], ['{python}', '{test_root}/run_test.py'])
                self.assertIn('--nprocs', spec['argv'])
                self.assertNotIn('--nproc', spec['argv'])
            else:
                self.assertIn('{binary_dir}/' + spec['parameters']['op'], spec['argv'])

    def test_native_performance_samples_keep_distinct_cases_in_one_atomic_measurement(self):
        from catalog import performance_cases

        specs = performance_cases()
        self.assertEqual(len(specs), 3)
        for count in (2, 14):
            manifest = build_manifest(specs, {}, shards_per_pool=count)
            self.assertEqual(len(manifest['shards']), 1)
            rows = manifest['shards'][0]['cases']
            self.assertEqual(len({row['spec']['execution']['case_id'] for row in rows}), 3)
            self.assertEqual(len({row['case_key'] for row in rows}), 3)
            self.assertEqual(len({row['measurement_identity'] for row in rows}), 1)
            validate_manifest(manifest)

    def test_source_graph_references_and_contexts_are_independently_validated(self):
        from catalog import CATALOG, validate_bundle

        original = json.loads(CATALOG.read_text())
        validate_bundle(original, original['coverage']['scope'])
        for name, mutate in (
            ('dangling-suite', lambda b: b['plans'][0]['suite_ids'].append('f' * 64)),
            ('dangling-case', lambda b: b['suites'][0]['case_ids'].append('e' * 64)),
            ('duplicate-definition', lambda b: b['cases'].append(copy.deepcopy(b['cases'][0]))),
            ('wrong-command', lambda b: b['cases'][0].__setitem__('command', 'python3 other.py')),
            ('wrong-parent', lambda b: b['executions'][0].__setitem__('plan_id', 'd' * 64)),
            ('duplicate-context', lambda b: b['executions'].append({**copy.deepcopy(b['executions'][0]), 'id': 'c' * 64})),
        ):
            changed = copy.deepcopy(original)
            mutate(changed)
            changed['bundle_sha256'] = digest({key: value for key, value in changed.items() if key != 'bundle_sha256'})
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_bundle(changed, original['coverage']['scope'])

    def test_source_catalog_cannot_be_rehashed_or_leak_private_fields(self):
        from catalog import CATALOG, load_catalog

        original = json.loads(CATALOG.read_text())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'source-catalog.json'
            for name, mutate in (
                ('command', lambda value: value['executions'][0]['spec']['argv'].append('--unsupported')),
                ('source', lambda value: value['source'].__setitem__('revision', '0' * 40)),
                ('private', lambda value: value['executions'][0]['spec'].__setitem__(
                    'legacy_source', 'comm-internal-testbox/private')),
            ):
                with self.subTest(change=name):
                    changed = copy.deepcopy(original)
                    mutate(changed)
                    changed['bundle_sha256'] = digest({key: value for key, value in changed.items()
                                                       if key != 'bundle_sha256'})
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        load_catalog(path)


if __name__ == '__main__':
    unittest.main()
