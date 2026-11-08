# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from catalog import cases, performance_cases, source_inputs
from command import check_command
import execute
from execute import execution_provenance, run_shard, write_json
from manifest import RUNNER_HASHES, build_manifest, digest, write_bundle
import single_node_ci as ci
from test_metrics import synthetic_table


def public_runner_fixture(build, runner):
    from test_manifest import performance_specs

    measurement = performance_specs()[0]
    table = synthetic_table('8388608 2097152 float sum -1 100 80 150 0 100 80 150 0', sizes=(8388608,))
    build.mkdir()
    for name, expected in RUNNER_HASHES.items():
        source = runner / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError('Fixture requires the pinned public runner')
        shutil.copyfile(source, build / name)
    (build / 'filelock.py').write_text(
        'from contextlib import nullcontext\n'
        'def FileLock(*args, **kwargs):\n    return nullcontext()\n')
    for name in ('mpirun', 'all_reduce_perf', 'all_gather_perf', 'alltoall_perf',
                 'broadcast_perf', 'reduce_perf', 'reduce_scatter_perf',
                 'gather_perf', 'scatter_perf', 'sendrecv_perf', 'hypercube_perf'):
        launcher = build / name
        # Only the exact CPU/MIN 8 MiB performance invocation gets a table.
        # Direct nonblocking checks need a validation summary, not invented measurements.
        launcher.write_text(
            f'#!{sys.executable}\nimport sys\n'
            'print("# synthetic launcher; no MPI or hardware")\n'
            f'if {name == "all_reduce_perf"!r} and sys.argv[1:] == {measurement["argv"][1:]!r}:\n'
            f'    print({table!r}, end="")\n'
            'else:\n    print("# Out of bounds values : 0 OK")\n')
        launcher.chmod(0o700)


def synthetic_build_record(inputs, profile):
    return {'schema_version': 1, 'pccl_revision': inputs['pccl_revision'],
            'pccl_tests_revision': inputs['pccl_tests_revision'], 'build_profile': profile,
            'artifact_sha256': hashlib.sha256(('synthetic archive: ' + profile).encode()).hexdigest()}


def rewrite_hook_runtime(directory, shard, field, value):
    """Rewrite only hook-owned evidence, including every self-declared hash."""
    suite = next(suite for suite in shard['suite_runs'] if suite['lifecycle']['setup'])
    lifecycle = execute.read_json(directory / 'suite-lifecycle.json')
    record = next(record for record in lifecycle if record['suite_run_id'] == suite['suite_run_id'])
    hook = record['actual_hooks']['setup'][0]
    hook_dir = (directory / hook['command_record']).parent
    bindings = execute.read_json(hook_dir / 'command-invocation.json')['bindings']
    bindings[field] = value
    cases = [row for row in shard['cases'] if row['case_key'] in suite['case_keys']]
    spec = execute._hook_spec(suite['lifecycle']['setup'][0], cases)
    invocation = execute._hook_invocation(spec, suite, 'setup', 1, bindings)
    observed = execute.read_json(hook_dir / 'command-observations.json')
    observed['invocation_sha256'] = digest(invocation)
    observed['launches'] = [dict(executable=invocation['argv'][0],
                               **{key: invocation[key] for key in ('argv', 'cwd', 'effective_environment')})]
    for name, data, checksum in (('command-invocation.json', invocation, 'invocation_sha256'),
                                 ('command-observations.json', observed, 'observations_sha256')):
        write_json(hook_dir / name, data)
        hook[checksum] = hashlib.sha256((hook_dir / name).read_bytes()).hexdigest()
    write_json(hook_dir / 'result.json', hook)
    write_json(directory / 'suite-lifecycle.json', lifecycle)
    # Prove the counterfeit is self-consistent; only the shard anchor rejects it.
    execute._check_hook_invocation(hook_dir, spec, suite, 'setup', 1, bindings, observed['process'])


def rewrite_case_runtime(directory, spec, field):
    """Keep Case-local evidence consistent while disagreeing with its shard."""
    invocation = execute.read_json(directory / 'command-invocation.json')
    observed = execute.read_json(directory / 'command-observations.json')
    process = execute.read_json(directory / 'command-process.json')
    environments = [invocation['effective_environment'], observed['initial_environment'],
                    *(entry['effective_environment'] for entry in observed['records'])]
    if field == 'python':
        for argv in (invocation['original_argv'], invocation['argv'], observed['original_argv']):
            argv[0] = '/different/python3'
    elif field == 'binary_dir':
        invocation['bindings']['binary_dir'] = '/different/bin'
        for environment in environments:
            environment['PATH'] = '/different/bin' + os.pathsep + environment['PATH'].split(os.pathsep, 1)[1]
    elif field == 'runner_build_directory':
        invocation[field] = '/different/bin'
        for entry in observed['records']:
            path = entry['effective_environment']['PATH'].rsplit(os.pathsep, 1)[0]
            entry['effective_environment']['PATH'] = path + os.pathsep + '/different/bin'
    elif field == 'result_dir':
        invocation['bindings'][field] = '/different/results'
        invocation['argv'][invocation['runner_index'] + 1] = '/different/results/command-invocation.json'
    elif field in ('benchmark', 'launcher'):
        for entry in observed['records']:
            position = 0 if field == 'launcher' else entry['argv'].index(spec['parameters']['op'])
            entry['argv'][position] = '/different/bin/' + entry['argv'][position]
            entry['executable'] = entry['argv'][0]
    else:
        invocation['inherited_environment']['UCX_TLS'] = 'different'
        for environment in environments:
            environment['UCX_TLS'] = 'different'
    process['invocation_sha256'] = observed['invocation_sha256'] = digest(invocation)
    process['observations_sha256'] = digest(observed)
    for name, data in (('command-invocation.json', invocation), ('command-observations.json', observed),
                       ('command-process.json', process)):
        write_json(directory / name, data)
    if field not in ('benchmark', 'launcher'):
        check_command(spec, {'result_dir': str(directory)}, process)


class SingleNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.fixture_root = Path(temporary.name)
        cls.fixture_manifest = None

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runner = Path(os.environ['PCCL_TESTS_ROOT']).resolve()
        # Ambient secrets (including short numeric values) must not redact synthetic
        # startup fields or measurements. Keep the Python launcher resolvable offline.
        environment = patch.dict(os.environ, {
            'PATH': os.pathsep.join((str(Path(sys.executable).parent), '/usr/bin', '/bin')),
            'PYTHONDONTWRITEBYTECODE': '1', 'NODE_RANK': '0', 'UCX_TLS': 'original',
        }, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.inputs = dict(test_suite='pr', single_node=True, debug_only=False,
                           test_types=['functional', 'performance'], node_counts=[2, 8, 16], shards_per_pool=1,
                           timings_sha256=digest({}), pccl_revision='a' * 40,
                           pccl_tests_revision='394556512e87259e9ffa25225632c46b0a0bbcd5',
                           **source_inputs())
        specs = [spec for spec in cases() + performance_cases() if spec['nodes'] == 1]
        self.manifest = build_manifest(specs, self.inputs)
        self.assertEqual(self.manifest['case_count'], 108)
        write_bundle(self.manifest, self.root / 'bundle')
        self.path = self.root / 'bundle' / 'manifest.json'
        self.results = self.root / 'results'
        self.output = self.root / 'public'
        self.provenance = execution_provenance()

    def write_results(self):
        # Execute the full, trusted catalog once. Each adversarial test gets an
        # independent artifact copy; neither selection nor evidence verification is mocked.
        cls = type(self)
        golden = self.fixture_root / 'results'
        if cls.fixture_manifest is None:
            build = self.fixture_root / 'synthetic-build'
            public_runner_fixture(build, self.runner)
            for shard in self.manifest['shards']:
                directory = golden / shard['build_profile'] / shard['shard_id']
                environment = dict(os.environ)
                if shard['test_type'] == 'performance':
                    environment.pop('UCX_TLS')
                with patch.dict(os.environ, environment, clear=True):
                    status = run_shard(self.path, shard['shard_id'], build, build, directory, [],
                               build=synthetic_build_record(self.inputs, shard['build_profile']))
                rows = [json.loads(line) for line in (directory / 'results.jsonl').read_text().splitlines()]
                self.assertEqual(status, 0, {'index': execute.read_json(directory / 'index.json'), 'results': rows})
                for row in rows:
                    row['diagnostic'] = 'PRIVATE HOST must not be published'
                    write_json(directory / row['case_key'] / 'attempt-1/result.json', row)
                (directory / 'results.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertEqual(ci.summarize(self.path, golden, self.fixture_root / 'verified'), 0)
            shutil.rmtree(build)
            cls.fixture_manifest = copy.deepcopy(self.manifest)
        self.assertEqual(self.manifest, cls.fixture_manifest)
        shutil.copytree(golden, self.results)

    def runner_row(self, *, mpi=False):
        return next((shard, row) for shard in self.manifest['shards'] for row in shard['cases']
                    if row['spec']['group'] == 'source_register'
                    and row['spec']['argv'][:2] == ['{python}', '{test_root}/run_test.py']
                    and (not mpi or row['spec']['processes_per_node'] > 1))

    def test_complete_synthetic_results_publish_only_safe_fields(self):
        self.write_results()
        shard, row = self.runner_row()
        reproduction = (self.results / shard['build_profile'] / shard['shard_id'] /
                        row['case_key'] / 'attempt-1/command-reproduction.txt')
        reproduction.write_text('token=PRIVATE_HOST\n' + reproduction.read_text())
        self.assertEqual(ci.summarize(self.path, self.results, self.output), 0)
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertTrue(summary['complete'])
        self.assertEqual(len(summary['builds']), 3)
        for shard in self.manifest['shards']:
            self.assertEqual(summary['builds'][shard['shard_id']],
                             synthetic_build_record(self.inputs, shard['build_profile']))
        case_summary = json.loads((self.output / 'case-summary.json').read_text())
        self.assertTrue(case_summary['complete'])
        self.assertEqual(len(case_summary['cases']), 71)
        self.assertEqual(sum(len(case['executions']) for case in case_summary['cases']), 108)
        for case in case_summary['cases']:
            self.assertIsInstance(case['source_command'], str)
            self.assertEqual(set(case['source_context']),
                             {'plan_id', 'suite_id', 'case_id', 'suite_run_id'})
            for execution in case['executions']:
                self.assertIsInstance(execution['argv'], list)
                self.assertIsInstance(execution['resources'], dict)
                relative = Path(execution['reproduction'])
                self.assertTrue((self.results / relative).is_file(), relative)
                self.assertTrue((self.output / relative).is_file(), relative)
                self.assertNotIn('PRIVATE HOST', (self.output / relative).read_text())
                result_path = Path(execution['result'])
                self.assertTrue((self.output / result_path).is_file(), result_path)
                public_result = execute.read_json(self.output / result_path)
                self.assertEqual(set(public_result), {'case_key', 'build_profile', 'test_type',
                                                      'status', 'source_context'})
                self.assertNotIn('diagnostic', public_result)
        self.assertEqual((summary['expected'], summary['passed']), (108, 108))
        rows = execute.read_jsonl(self.output / 'metrics.jsonl')
        self.assertEqual(len(rows), 6)
        self.assertEqual({row['sample_index'] for row in rows}, {0, 1, 2})
        self.assertEqual({row['placement'] for row in rows}, {'out-of-place', 'in-place'})
        self.assertEqual(len({row['measurement_identity'] for row in rows}), 1)
        for row in rows:
            self.assertEqual((row['size_bytes'], row['iters'], row['warmup']), (8388608, 100, 5))
            self.assertEqual((row['time_kind'], row['rank_reduction']), ('cpu_time', 'MIN'))
            self.assertEqual(row['data_status'], 'VALID')
        performance = execute.read_json(self.output / 'performance-summary.json')
        self.assertTrue(performance['complete'])
        self.assertEqual(len(performance['measurements']), 2)
        self.assertTrue(all(row['sample_count'] == 3 and row['sample_status'] == 'STABLE'
                            for row in performance['measurements']))
        self.assertNotIn('PRIVATE HOST', (self.output / 'summary.json').read_text())
        self.assertNotIn('PRIVATE HOST', (self.output / 'metrics.jsonl').read_text())
        self.assertFalse((self.output / 'command.log').exists())

    def test_full_synthetic_execution_roundtrip(self):
        self.write_results()
        hook_shards = 0
        for shard in self.manifest['shards']:
            directory = self.results / shard['build_profile'] / shard['shard_id']
            if any(suite['lifecycle'][phase] for suite in shard['suite_runs'] for phase in ('setup', 'teardown')):
                hook_shards += 1
                with self.assertRaisesRegex(ValueError, 'trusted execution runtime'):
                    execute.verify_lifecycle(directory, shard)
            else:
                execute.verify_lifecycle(directory, shard)
        self.assertGreater(hook_shards, 0)
        # Reproduce the old summary call against successful real catalog setup evidence.
        with patch.object(ci, 'verify_lifecycle', side_effect=lambda directory, shard, runtime:
                          execute.verify_lifecycle(directory, shard)):
            self.assertEqual(ci.summarize(self.path, self.results, self.root / 'without-runtime'), 1)
        self.assertEqual(ci.summarize(self.path, self.results, self.output), 0)
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertEqual(summary['passed'], 108)
        self.assertTrue(summary['complete'])
        self.assertEqual(set(summary['executions']), {shard['shard_id'] for shard in self.manifest['shards']})

    def test_result_gate_accepts_relocated_artifacts_without_execution_paths(self):
        self.write_results()
        relocated = self.root / 'downloaded-results'
        shutil.copytree(self.results, relocated)
        self.assertFalse((self.fixture_root / 'synthetic-build').exists())
        shutil.rmtree(self.results)
        with patch.dict(os.environ, {'PATH': '/summary/bin', 'UCX_TLS': 'summary-only'}, clear=True), \
                patch.object(sys, 'executable', '/summary/python3'):
            self.assertEqual(ci.summarize(self.path, relocated, self.output), 0)
        self.assertEqual(json.loads((self.output / 'summary.json').read_text())['passed'], 108)

    def test_hook_runtime_tampering_fails_summary_even_after_rehash(self):
        self.write_results()
        shard = next(shard for shard in self.manifest['shards']
                     if any(suite['lifecycle']['setup'] for suite in shard['suite_runs']))
        for field, value in (('test_root', '/different/root'), ('binary_dir', '/different/bin'),
                             ('result_dir', '/different/results'), ('mpi_options', ['--hostfile', 'different']),
                             ('python', '/different/python3'), ('inherited_environment', {'PATH': '/different/bin'})):
            with self.subTest(field=field):
                results = self.root / f'hook-{field}'
                shutil.copytree(self.results, results)
                directory = results / shard['build_profile'] / shard['shard_id']
                rewrite_hook_runtime(directory, shard, field, value)
                output = self.root / f'hook-summary-{field}'
                self.assertEqual(ci.summarize(self.path, results, output), 1)
                summary = execute.read_json(output / 'summary.json')
                self.assertEqual(summary['failed_shards'], [shard['shard_id']])

    def test_case_runtime_cannot_disagree_with_shard_after_rehash(self):
        self.write_results()
        # Launcher substitution is meaningful only for a real multi-process runner Case.
        shard, row = self.runner_row(mpi=True)
        self.assertFalse((self.fixture_root / 'synthetic-build').exists())
        for field in ('python', 'inherited_environment', 'binary_dir', 'runner_build_directory',
                      'result_dir', 'benchmark', 'launcher'):
            with self.subTest(field=field):
                results = self.root / f'case-{field}'
                shutil.copytree(self.results, results)
                directory = results / shard['build_profile'] / shard['shard_id']
                rewrite_case_runtime(directory / row['case_key'] / 'attempt-1', row['spec'], field)
                output = self.root / f'case-summary-{field}'
                self.assertEqual(ci.summarize(self.path, results, output), 1)
                summary = execute.read_json(output / 'summary.json')
                self.assertEqual(summary['passed'], 107)
                self.assertEqual(summary['failed_shards'], [])

    def test_missing_or_unlinked_shard_runtime_fails_summary(self):
        self.write_results()
        directory = next(self.results.rglob('shard-runtime.json')).parent
        path = directory / 'shard-runtime.json'
        original = path.read_bytes()
        path.unlink()
        self.assertEqual(ci.summarize(self.path, self.results, self.output), 1)
        path.write_bytes(original)
        index = execute.read_json(directory / 'index.json')
        index['runtime_sha256'] = '0' * 64
        write_json(directory / 'index.json', index)
        self.assertEqual(ci.summarize(self.path, self.results, self.root / 'unlinked'), 1)

    def test_missing_results_fail_with_complete_expected_case_list(self):
        self.assertEqual(ci.summarize(self.path, self.results, self.output), 1)
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertEqual((summary['expected'], summary['passed']), (108, 0))
        self.assertFalse(summary['complete'])

    def test_duplicate_results_and_cleanup_errors_cannot_pass(self):
        self.write_results()
        path = next(self.results.rglob('results.jsonl'))
        path.write_text(path.read_text() + path.read_text().splitlines()[0] + '\n')
        self.assertEqual(ci.summarize(self.path, self.results, self.output), 1)
        index = next(self.results.rglob('index.json'))
        content = json.loads(index.read_text())
        content['artifact_error'] = 'PRIVATE cleanup failure'
        index.write_text(json.dumps(content))
        self.assertEqual(ci.summarize(self.path, self.results, self.root / 'public2'), 1)
        self.assertNotIn('PRIVATE', (self.root / 'public2/summary.json').read_text())

    def test_wrong_run_executor_or_incomplete_index_fails_gate(self):
        self.write_results()
        path = next(self.results.rglob('index.json'))
        original = json.loads(path.read_text())
        mutations = [dict(selection_sha256='0' * 64), dict(cancelled=True), dict(failures=False),
                     dict(expected=original['expected'] * 2), dict(completed=original['completed'][:-1]),
                     dict(execution={**self.provenance, 'executor_sha256': '0' * 64}),
                     dict(execution={**self.provenance, 'executor_python_version': ''}), dict(execution=None)]
        for number, changes in enumerate(mutations):
            write_json(path, {**original, **changes})
            with self.subTest(changes=changes):
                self.assertEqual(ci.summarize(self.path, self.results, self.root / f'invalid-index-{number}'), 1)
        write_json(path, original)
        snapshot = path.with_name('manifest.json')
        changed = copy.deepcopy(self.manifest)
        changed['inputs']['run_id'] = 'different-run'
        specs = [row['spec'] for shard in changed['shards'] for row in shard['cases']]
        write_json(snapshot, build_manifest(specs, changed['inputs']))
        self.assertEqual(ci.summarize(self.path, self.results, self.output), 1)

    def test_finished_result_requires_command_and_check_evidence(self):
        self.write_results()
        path = next(self.results.rglob('command-process.json'))
        original = path.read_text()
        for number, record in enumerate(({}, dict(active=True, state='finished', return_code=0),
                                        dict(active=False, state='cleanup_failed', return_code=0),
                                        dict(active=False, state='finished', return_code=False),
                                        dict(active=False, state='finished', return_code=7))):
            write_json(path, record)
            with self.subTest(record=record):
                self.assertEqual(ci.summarize(self.path, self.results, self.root / f'bad-record-{number}'), 1)
        path.write_text(original)
        path.unlink()
        self.assertEqual(ci.summarize(self.path, self.results, self.output), 1)

    def test_exit_only_source_case_does_not_fabricate_a_log_assertion(self):
        self.write_results()
        shard, row = self.runner_row()
        self.assertEqual(row['spec']['checks'], ['exit_code'])
        path = self.results / shard['build_profile'] / shard['shard_id'] / row['case_key'] / 'attempt-1/command.log'
        path.write_text('source pipeline checks only the successful return code\n')
        self.assertEqual(ci.summarize(self.path, self.results, self.output), 0)
        self.assertEqual(json.loads((self.output / 'summary.json').read_text())['passed'], 108)

    def test_direct_nonblocking_requires_its_validation_summary(self):
        self.write_results()
        shard, row = next((shard, row) for shard in self.manifest['shards'] for row in shard['cases']
                          if row['spec']['test_type'] == 'functional'
                          and row['spec']['checks'] == ['exit_code', 'validation_summary']
                          and '--init_blocking' in row['spec']['argv'])
        path = self.results / shard['build_profile'] / shard['shard_id'] / row['case_key'] / 'attempt-1/command.log'
        self.assertIn('# Out of bounds values : 0 OK', path.read_text())
        self.assertNotIn('# size count', path.read_text())
        path.write_text('synthetic zero exit without validation evidence\n')
        self.assertEqual(ci.summarize(self.path, self.results, self.output), 1)
        self.assertEqual(execute.read_json(self.output / 'summary.json')['passed'], 107)

    def test_performance_measurements_are_verified_not_just_successful_exits(self):
        self.write_results()
        shard = next(shard for shard in self.manifest['shards'] if shard['test_type'] == 'performance')
        row = shard['cases'][0]
        case_dir = self.results / shard['build_profile'] / shard['shard_id'] / row['case_key'] / 'attempt-1'
        log = case_dir / 'command.log'
        self.assertIn('warmup iters: 5 iters: 100', log.read_text())
        log.write_text(log.read_text().replace('warmup iters: 5 iters: 100', 'warmup iters: 5 iters: 10'))
        self.assertEqual(ci.summarize(self.path, self.results, self.output), 1)
        self.assertEqual(execute.read_json(self.output / 'summary.json')['passed'], 107)
        performance = execute.read_json(self.output / 'performance-summary.json')
        self.assertFalse(performance['complete'])
        self.assertEqual(len(execute.read_jsonl(self.output / 'metrics.jsonl')), 4)
        self.assertTrue(all(measurement['sample_count'] == 2
                            and measurement['sample_status'] == 'INSUFFICIENT_SAMPLES'
                            for measurement in performance['measurements']))

    def test_performance_sample_quality_is_observational_even_when_all_cases_pass(self):
        self.write_results()
        summarize_metrics = ci.summarize_metrics
        for status in ('UNSTABLE', 'INSUFFICIENT_SAMPLES'):
            output = self.root / f'quality-{status}'
            with patch.object(ci, 'summarize_metrics',
                              side_effect=lambda rows: [dict(row, sample_status=status)
                                                        for row in summarize_metrics(rows)]):
                self.assertEqual(ci.summarize(self.path, self.results, output), 0)
            summary = execute.read_json(output / 'summary.json')
            self.assertEqual((summary['passed'], summary['expected']), (108, 108))
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['artifact_error'], '')
            performance = execute.read_json(output / 'performance-summary.json')
            self.assertTrue(performance['complete'])
            self.assertTrue(all(row['sample_status'] == status for row in performance['measurements']))
            self.assertTrue(execute.read_json(output / 'case-summary.json')['complete'])

    def test_fixture_emits_measurements_only_for_exact_performance_arguments(self):
        from test_manifest import performance_specs

        build = self.root / 'fixture-build'
        public_runner_fixture(build, self.runner)
        args = performance_specs()[0]['argv'][1:]
        benchmark = str(build / 'all_reduce_perf')
        exact = subprocess.run([benchmark, *args], check=True, capture_output=True, text=True)
        self.assertIn('warmup iters: 5 iters: 100', exact.stdout)
        for option, value in (('-n', '10'), ('-b', '16'), ('--elapsed_type', '1'), ('-a', '0')):
            changed = list(args)
            changed[changed.index(option) + 1] = value
            with self.subTest(option=option):
                result = subprocess.run([benchmark, *changed], check=True, capture_output=True, text=True)
                self.assertIn('# Out of bounds values : 0 OK', result.stdout)
                self.assertNotIn('# size count', result.stdout)
        result = subprocess.run([str(build / 'mpirun'), '-np', '2', benchmark, *args],
                                check=True, capture_output=True, text=True)
        self.assertIn('no MPI or hardware', result.stdout)
        self.assertNotIn('# size count', result.stdout)

    def test_invalid_case_result_or_json_cannot_pass(self):
        self.write_results()
        path = next(self.results.rglob('results.jsonl'))
        original = path.read_text()
        rows = [json.loads(line) for line in original.splitlines()]
        for number, changes in enumerate((dict(status='UNKNOWN'), dict(error='PRIVATE error'),
                                          dict(returncode=False), dict(attempt=2), dict(duration=-1),
                                          dict(duration=10 ** 400), dict(command_log='other/log'),
                                          dict(legacy_source='other'))):
            altered = copy.deepcopy(rows)
            altered[0].update(changes)
            path.write_text(''.join(json.dumps(row) + '\n' for row in altered))
            write_json(path.parent / altered[0]['case_key'] / 'attempt-1/result.json', altered[0])
            with self.subTest(changes=changes):
                self.assertEqual(ci.summarize(self.path, self.results, self.root / f'bad-result-{number}'), 1)
        for number, text in enumerate(('[null]\n', 'null\n', '{"case_key":"x","case_key":"y"}\n',
                                      '{"case_key":"x","duration":NaN}\n')):
            path.write_text(text)
            with self.subTest(text=text):
                self.assertEqual(ci.summarize(self.path, self.results, self.root / f'bad-json-{number}'), 1)

    def test_multinode_or_incomplete_selection_is_rejected(self):
        _, row = self.runner_row()
        single = [copy.deepcopy(row['spec'])]
        multinode = copy.deepcopy(single)
        multinode[0].update(nodes=2, parameters={**multinode[0]['parameters'], 'ranks': 4})
        multinode[0]['source_context']['resource']['nodes'] = 2
        for name, specs in (('partial', single), ('multinode', multinode)):
            directory = self.root / name
            write_bundle(build_manifest(specs, self.inputs), directory)
            with self.assertRaises(ValueError):
                ci.selection(directory / 'manifest.json')
        specs = [row['spec'] for shard in self.manifest['shards'] for row in shard['cases']]
        for field in ('legacy_source', 'tier'):
            altered = copy.deepcopy(specs)
            altered[0][field] = 'other' if field == 'legacy_source' else 'nightly'
            directory = self.root / field
            write_bundle(build_manifest(altered, self.inputs), directory)
            with self.assertRaises(ValueError):
                ci.selection(directory / 'manifest.json')

    def test_matrix_uses_suite_allocation_not_case_rank_count(self):
        matrix = ci.matrix(self.path)
        self.assertEqual(len(matrix['include']), len(self.manifest['shards']))
        for batch, shard in zip(matrix['include'], self.manifest['shards']):
            self.assertEqual(batch['nproc_per_node'], shard['resource']['cards_per_node'])
            self.assertEqual(batch['image'], shard['runtime']['image'])
            self.assertEqual(batch['node_selector'], shard['runtime']['node_selector'])
            self.assertEqual(batch['shard_id'], shard['shard_id'])
            self.assertEqual(batch['nnodes'], 0)
            self.assertEqual(batch['nproc_per_node'], 8)
            for row in shard['cases']:
                spec = row['spec']
                ranks = spec['parameters']['ranks']
                self.assertEqual(ranks, spec['processes_per_node'] * spec['threads_per_process']
                                 * spec['devices_per_thread'])
                self.assertLessEqual(ranks, batch['nproc_per_node'])
        self.assertEqual({row['spec']['parameters']['ranks'] for shard in self.manifest['shards']
                          for row in shard['cases']}, {2, 4})

    def test_build_identity_is_required_and_bound_to_each_shard(self):
        self.write_results()
        for shard in self.manifest['shards']:
            directory = self.results / shard['build_profile'] / shard['shard_id']
            snapshot_path = directory / 'shard-runtime.json'
            index_path = directory / 'index.json'
            snapshot = execute.read_json(snapshot_path)
            for mutation in ('missing', 'revision', 'profile'):
                changed = copy.deepcopy(snapshot)
                if mutation == 'missing':
                    changed.pop('build')
                elif mutation == 'revision':
                    changed['build']['pccl_revision'] = 'f' * 40
                else:
                    changed['build']['build_profile'] = 'notsm' if shard['build_profile'] == 'standard' else 'standard'
                write_json(snapshot_path, changed)
                index = execute.read_json(index_path)
                index['runtime_sha256'] = digest(changed)
                write_json(index_path, index)
                with self.subTest(shard=shard['shard_id'], mutation=mutation):
                    self.assertEqual(ci.summarize(self.path, self.results,
                                                  self.root / f"build-{shard['shard_id']}-{mutation}"), 1)
                write_json(snapshot_path, snapshot)
                index['runtime_sha256'] = digest(snapshot)
                write_json(index_path, index)

    def test_run_rejects_sdk_library_and_stops_after_shard_failure(self):
        build = self.root / 'build'
        archive = self.root / 'output.tar.gz'
        archive.write_bytes(b'synthetic archive')
        record = synthetic_build_record(self.inputs, 'standard')
        record['artifact_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
        record_path = self.root / 'build.json'
        write_json(record_path, record)
        shard_id = next(shard['shard_id'] for shard in self.manifest['shards']
                        if shard['build_profile'] == 'standard')
        with patch.object(ci.subprocess, 'run') as linked, patch.object(ci, 'run_shard') as run:
            linked.return_value.stdout = 'libpccl.so.2 => /usr/local/PPU_SDK/lib/libpccl.so.2 (0x123)'
            with self.assertRaisesRegex(ValueError, 'freshly built'):
                ci.run(self.path, 'standard', build, self.results, shard_id, record_path, archive)
            run.assert_not_called()
            linked.return_value.stdout = f'libpccl.so.2 => {build}/lib/libpccl.so.2 (0x123)'
            run.return_value = 1
            self.assertEqual(ci.run(self.path, 'standard', build, self.results, shard_id, record_path, archive), 1)
            self.assertEqual(run.call_count, 1)
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                ci.run(self.path, 'standard', build, self.results, 'unknown-batch', record_path, archive)


if __name__ == '__main__':
    unittest.main()
