# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct JSON execution with synthetic commands/logs only; no MPI or hardware evidence."""

import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import execute
from execute import junit
from manifest import build_manifest, load_manifest, write_bundle
from test_manifest import fixture_spec


ROOT = Path(__file__).resolve().parent


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / 'output'
        self.bundle = self.root / 'bundle'
        environment = patch.dict(os.environ, {key: value for key, value in os.environ.items()
                                 if not key.startswith(('PCCL_', 'NCCL_', 'HGGC_', 'OMPI_', 'UCX_'))}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def prepare(self, scripts):
        specs = []
        for index, code in enumerate(scripts):
            spec = fixture_spec(index)
            spec['argv'] = ['{python}', '-c', code]
            specs.append(spec)
        manifest = build_manifest(specs, {})
        write_bundle(manifest, self.bundle)
        return manifest

    def command(self, manifest):
        return [sys.executable, '-S', '-B', str(ROOT / 'execute.py'), '--manifest', str(self.bundle / 'manifest.json'),
                '--shard', manifest['shards'][0]['shard_id'], '--test-root', str(self.root),
                '--binary-dir', str(self.root), '--output', str(self.output)]

    def invoke(self, manifest, env=None):
        return subprocess.run(self.command(manifest), capture_output=True, text=True, timeout=40,
                              env={**os.environ, 'NODE_RANK': '0', **(env or {})})

    def test_direct_shard_emits_complete_results(self):
        manifest = self.prepare(["print('one')", "print('two')"])
        result = self.invoke(manifest)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = [json.loads(line) for line in (self.output / 'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['status'] == 'PASSED' and row['returncode'] == 0 for row in rows))
        index = json.loads((self.output / 'index.json').read_text())
        self.assertEqual(index['execution'], execute.execution_provenance())
        self.assertEqual(index['selection_sha256'], manifest['selection_sha256'])
        runtime = execute.verified_runtime(self.output, index)
        self.assertEqual(runtime['test_root'], str(self.root.resolve()))
        self.assertEqual(runtime['binary_dir'], str(self.root.resolve()))
        self.assertEqual(runtime['result_dir'], str(self.output))
        self.assertEqual(runtime['runner_build_directory'], str((self.root / 'build').resolve()))
        self.assertEqual(Path(runtime['python']).resolve(), Path(sys.executable).resolve())
        self.assertEqual(runtime['mpi_options'], [])
        self.assertEqual(runtime['inherited_environment'], execute.safe_environment(os.environ))
        self.assertFalse(list(self.output.rglob('framework*')))
        self.assertFalse(list(self.output.rglob('xtest-result.json')))
        self.assertNotIn('framework_log', rows[0])
        self.assertNotIn('source_context', rows[0])
        self.assertNotIn('lifecycle_error', index)
        self.assertFalse((self.output / 'suite-lifecycle.json').exists())
        report = ET.parse(self.output / 'junit.xml').getroot()
        self.assertEqual(report.get('tests'), '2')
        self.assertEqual(report.get('failures'), '0')
        self.assertEqual(len(json.loads((self.output / 'timings.json').read_text())), 2)
        again = self.invoke(manifest)
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(len((self.output / 'results.jsonl').read_text().splitlines()), 2)

    def test_runtime_snapshot_precedes_commands_and_excludes_credentials(self):
        manifest = self.prepare(["print('case')"])
        original = execute.run_command
        calls = []

        def run(spec, runtime, interrupted):
            snapshot = execute.read_json(self.output / 'shard-runtime.json')
            self.assertFalse((self.output / 'index.json').exists())
            for key in ('test_root', 'binary_dir', 'mpi_options', 'python', 'inherited_environment'):
                self.assertEqual(runtime[key], snapshot['runtime'][key])
            calls.append(snapshot)
            return original(spec, runtime, interrupted)

        with patch.dict(os.environ, {'CI_ACCESS_TOKEN': 'credential-not-for-publication'}), \
                patch.object(execute, 'run_command', side_effect=run):
            self.assertEqual(execute.run_shard(self.bundle / 'manifest.json', manifest['shards'][0]['shard_id'],
                                              self.root, self.root, self.output, []), 0)
        self.assertEqual(len(calls), 1)
        self.assertNotIn('CI_ACCESS_TOKEN', json.dumps(calls[0]))
        self.assertNotIn('credential-not-for-publication', json.dumps(calls[0]))
        index = execute.read_json(self.output / 'index.json')
        self.assertEqual(index['runtime_sha256'], execute.digest(calls[0]))
        self.assertEqual(execute.verified_runtime(self.output, index), calls[0]['runtime'])

    def test_runtime_snapshot_rejects_invalid_identity_and_unsafe_fields(self):
        manifest = self.prepare(["print('case')"])
        self.assertEqual(self.invoke(manifest).returncode, 0)
        index = execute.read_json(self.output / 'index.json')
        path = self.output / 'shard-runtime.json'
        original = execute.read_json(path)
        for key, value in (('shard_id', 'other'), ('selection_sha256', '0' * 64),
                           ('execution', {}), ('schema_version', True), ('extra', 'unexpected')):
            snapshot = dict(original, **{key: value})
            execute.write_json(path, snapshot)
            with self.subTest(identity=key), self.assertRaises(ValueError):
                execute.verified_runtime(self.output, dict(index, runtime_sha256=execute.digest(snapshot)))
        for key, value in (('python', 'python3'), ('test_root', 'relative'), ('binary_dir', '/private-project'),
                           ('result_dir', None), ('mpi_options', ['--hostfile', 'token=unsafe']),
                           ('inherited_environment', {'CI_ACCESS_TOKEN': 'hidden'}),
                           ('inherited_environment', {'PATH': 'https://private.invalid'})):
            snapshot = dict(original, runtime=dict(original['runtime'], **{key: value}))
            execute.write_json(path, snapshot)
            with self.subTest(runtime=key), self.assertRaises(ValueError):
                execute.verified_runtime(self.output, dict(index, runtime_sha256=execute.digest(snapshot)))
        execute.write_json(path, original)
        with self.assertRaises(ValueError):
            execute.verified_runtime(self.output, dict(index, runtime_sha256='0' * 64))
        path.unlink()
        with self.assertRaises((OSError, ValueError)):
            execute.verified_runtime(self.output, index)

    def test_provenance_requires_runner_capture_and_hashes_its_contents(self):
        original_read = Path.read_bytes
        original = execute.execution_provenance()

        def changed(path):
            data = original_read(path)
            return data + b'\n# synthetic change\n' if path.name == 'runner_capture.py' else data

        def missing(path):
            if path.name == 'runner_capture.py':
                raise FileNotFoundError('runner_capture.py')
            return original_read(path)

        with patch.object(Path, 'read_bytes', changed):
            self.assertNotEqual(execute.execution_provenance()['executor_sha256'], original['executor_sha256'])
        with patch.object(Path, 'read_bytes', missing), self.assertRaises(FileNotFoundError):
            execute.execution_provenance()

    def test_failure_is_not_hidden_and_other_case_still_runs(self):
        manifest = self.prepare(["raise SystemExit(7)", "print('valid other case')"])
        result = self.invoke(manifest)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = [json.loads(line) for line in (self.output / 'results.jsonl').read_text().splitlines()]
        self.assertCountEqual([row['status'] for row in rows], ['FAILED', 'PASSED'])
        self.assertEqual(ET.parse(self.output / 'junit.xml').getroot().get('failures'), '1')

    def test_single_process_assertion_failure_allows_following_case(self):
        marker = self.root / 'synthetic-first-case'
        code = (f'from pathlib import Path; p = Path({str(marker)!r}); '
                'print("# Out of bounds values : 0 OK" if p.exists() else "synthetic missing summary"); '
                'p.touch()')
        specs = [fixture_spec(index) for index in range(2)]
        for spec in specs:
            spec.update(argv=['{python}', '-c', code], checks=['exit_code', 'validation_summary'])
        manifest = build_manifest(specs, {})
        write_bundle(manifest, self.bundle)
        result = self.invoke(manifest)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        rows = [json.loads(line) for line in (self.output / 'results.jsonl').read_text().splitlines()]
        self.assertEqual([row['status'] for row in rows], ['FAILED', 'PASSED'])
        self.assertIn('Missing successful data-validation summary', rows[0]['error'])
        self.assertEqual(ET.parse(self.output / 'junit.xml').getroot().get('failures'), '1')

    def test_synthetic_mpi_failure_stops_shard_for_each_launch_shape(self):
        # These are local Python stubs, not MPI or hardware runs. Check each
        # independent classification path, even when local cleanup succeeded.
        hosts = self.root / 'synthetic-hosts'
        hosts.write_text('synthetic-host.invalid\n')
        for shape in ('nodes', 'processes', 'mpirun'):
            self.bundle, self.output = self.root / f'bundle-{shape}', self.root / f'output-{shape}'
            marker = self.root / f'invocations-{shape}'
            code = (f'from pathlib import Path; p = Path({str(marker)!r}); '
                    'p.write_text((p.read_text() if p.exists() else "") + "synthetic-called\\n"); '
                    'raise SystemExit(7)')
            specs = [fixture_spec(index) for index in range(2)]
            if shape == 'nodes':
                runner = self.root / 'run_test.py'
                runner.write_text('# Synthetic local runner; never launches MPI.\n' + code + '\n')
                for spec in specs:
                    spec.update(nodes=2, argv=['{python}', str(runner), '--process-group', 'normal',
                                               '--launch-mode', 'normal', '--mpi-options={mpi_options}'])
            elif shape == 'processes':
                for spec in specs:
                    spec.update(processes_per_node=2, argv=['{python}', '-c', code])
            else:
                launcher = self.root / 'mpirun'
                launcher.write_text(f'#!{sys.executable}\n# Synthetic stub, no MPI.\n' + code + '\n')
                launcher.chmod(0o700)
                for spec in specs:
                    spec['argv'] = [str(launcher), '{mpi_options}']
            manifest = build_manifest(specs, {})
            write_bundle(manifest, self.bundle)
            with self.subTest(synthetic_shape=shape):
                result = subprocess.run(self.command(manifest) + ['--mpi-options', json.dumps(['--hostfile', str(hosts)])],
                                        capture_output=True, text=True, timeout=40,
                                        env={**os.environ, 'NODE_RANK': '0'})
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                rows = [json.loads(line) for line in (self.output / 'results.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]['status'], 'FAILED')
                self.assertEqual(marker.read_text(), 'synthetic-called\n')
                record = json.loads(next(self.output.glob('*/attempt-1/command-process.json')).read_text())
                self.assertFalse(record['active'])
                self.assertEqual(record['state'], 'failed')
                report = ET.parse(self.output / 'junit.xml').getroot()
                self.assertEqual(report.get('tests'), '2')
                self.assertEqual(report.get('failures'), '2')
                self.assertTrue(any(case.find('failure').get('message') == 'Missing Case result'
                                    for case in report.findall('testcase')))

    def test_corrupted_manifest_fails_before_execution(self):
        manifest = self.prepare(["print('must not run')"])
        manifest['shards'][0]['cases'][0]['spec']['timeout'] = 999
        (self.bundle / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            load_manifest(self.bundle / 'manifest.json')
        self.assertNotEqual(self.invoke(manifest).returncode, 0)
        self.assertFalse(self.output.exists())

    def test_worker_cannot_start_the_same_shard(self):
        manifest = self.prepare(["print('must not run')"])
        self.assertNotEqual(self.invoke(manifest, {'NODE_RANK': '1'}).returncode, 0)
        self.assertFalse(self.output.exists())

    def test_missing_executable_is_error_and_stops_shard(self):
        specs = [fixture_spec(index) for index in range(2)]
        for spec in specs:
            spec['argv'] = [str(self.root / 'missing-executable')]
        manifest = build_manifest(specs, {})
        write_bundle(manifest, self.bundle)
        result = self.invoke(manifest)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        rows = execute.read_jsonl(self.output / 'results.jsonl')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['status'], 'ERROR')
        self.assertIsNone(rows[0]['returncode'])
        self.assertEqual(ET.parse(self.output / 'junit.xml').getroot().get('failures'), '2')

    def test_unknown_shard_and_existing_output_symlink_refuse_launch(self):
        manifest = self.prepare(["print('must not run')"])
        with self.assertRaisesRegex(ValueError, 'Unknown shard'):
            execute.run_shard(self.bundle / 'manifest.json', 'unknown', self.root, self.root, self.output, [])
        target = self.root / 'missing-output-target'
        self.output.symlink_to(target, target_is_directory=True)
        self.assertNotEqual(self.invoke(manifest).returncode, 0)
        self.assertTrue(self.output.is_symlink())
        self.assertFalse(target.exists())

    def test_jsonl_rejects_duplicate_keys_and_nonfinite_values(self):
        path = self.root / 'results.jsonl'
        for text in ('{"case_key":"one","case_key":"two"}\n', '{"duration":NaN}\n',
                     '{"duration":1e999}\n'):
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                execute.read_jsonl(path)

    def test_missing_and_duplicate_results_fail_junit(self):
        path = self.root / 'report.xml'
        self.assertEqual(junit([], ['expected'], path), 1)
        self.assertEqual(ET.parse(path).getroot().find('testcase/failure').get('message'), 'Missing Case result')
        row = dict(case_key='expected', duration=1, status='PASSED', error='')
        with self.assertRaises(ValueError):
            junit([row, row], ['expected'], path)
        with self.assertRaises(ValueError):
            junit([row], ['other'], path)

    def test_cleanup_failure_is_saved_and_stops_shard(self):
        manifest = self.prepare(["print('one')", "print('two')"])
        import command

        with patch.dict(os.environ, {'NODE_RANK': '0'}), \
                patch.object(command, '_members', side_effect=OSError('cleanup evidence unavailable')):
            result = execute.run_shard(self.bundle / 'manifest.json', manifest['shards'][0]['shard_id'],
                                       self.root, self.root, self.output, [])
        self.assertEqual(result, 1)
        rows = [json.loads(line) for line in (self.output / 'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['status'], 'ERROR')
        self.assertIn('cleanup evidence unavailable', rows[0]['error'])
        self.assertEqual(ET.parse(self.output / 'junit.xml').getroot().get('failures'), '2')

    def test_empty_cleanup_error_cannot_pass(self):
        manifest = self.prepare(["print('one')", "print('two')"])
        with patch.object(execute, 'read_json', side_effect=OSError()):
            result = execute.run_shard(self.bundle / 'manifest.json', manifest['shards'][0]['shard_id'],
                                       self.root, self.root, self.output, [])
        self.assertEqual(result, 1)
        rows = execute.read_jsonl(self.output / 'results.jsonl')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['status'], 'ERROR')
        self.assertIn('OSError', rows[0]['error'])
        self.assertEqual(ET.parse(self.output / 'junit.xml').getroot().get('failures'), '2')

    def test_unfinished_record_without_returncode_still_attempts_safe_cleanup(self):
        import command

        manifest = self.prepare(["print('one')", "print('two')"])
        original = command._write_record

        def fail_final_record(path, record):
            if not record['active']:
                raise OSError('final record unavailable')
            original(path, record)

        with patch.object(command, '_write_record', fail_final_record), \
                patch.object(execute, 'stop_recorded_process', wraps=command.stop_recorded_process) as stop:
            result = execute.run_shard(self.bundle / 'manifest.json', manifest['shards'][0]['shard_id'],
                                       self.root, self.root, self.output, [])
        self.assertEqual(result, 1)
        stop.assert_called_once()
        rows = execute.read_jsonl(self.output / 'results.jsonl')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['status'], 'ERROR')
        self.assertIn('final record unavailable', rows[0]['error'])
        self.assertIn('cleanup', rows[0]['error'])

    def test_timeout_stops_shard_and_keeps_failure_reason(self):
        manifest = self.prepare(['import time; time.sleep(60)'])
        result = self.invoke(manifest)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        row, = [json.loads(line) for line in (self.output / 'results.jsonl').read_text().splitlines()]
        self.assertEqual(row['status'], 'ERROR')
        self.assertIn('timed out', row['error'])
        record = json.loads(next(self.output.glob('*/attempt-1/command-process.json')).read_text())
        self.assertEqual(record['state'], 'timeout')
        self.assertFalse(record['active'])

    def prepare_performance(self, text, exit_code=0, samples=3, shards_per_pool=1):
        from test_manifest import performance_specs

        specs = performance_specs()[:samples]
        manifest = build_manifest(specs, {'pccl_revision': '1' * 40,
                                          'pccl_tests_revision': '2' * 40}, shards_per_pool)
        write_bundle(manifest, self.bundle)
        binary = self.root / 'all_reduce_perf'
        binary.write_text(f'#!{sys.executable}\nimport os\n'
                          'assert os.environ["PCCL_DEBUG_FILE"].endswith("/library.%h.%p.log")\n'
                          f'print({text!r}, end="")\nraise SystemExit({exit_code})\n')
        binary.chmod(0o700)
        return manifest

    def test_insufficient_performance_samples_fail_shard_for_resampling(self):
        from test_metrics import synthetic_table

        text = synthetic_table('8388608 2097152 float sum -1 100 80 80 0 110 72 72 0', sizes=(8388608,))
        manifest = self.prepare_performance(text, samples=2)
        with patch.dict(os.environ, {'PATH': os.environ.get('PATH', os.defpath)}, clear=True):
            result = self.invoke(manifest)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        rows = [json.loads(line) for line in (self.output / 'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['status'] == 'PASSED' and row['data_status'] == 'VALID' for row in rows))
        summary = json.loads((self.output / 'performance-summary.json').read_text())
        self.assertFalse(summary['complete'])
        self.assertTrue(all(row['sample_count'] == 2 and row['sample_status'] == 'INSUFFICIENT_SAMPLES'
                            for row in summary['measurements']))
        index = json.loads((self.output / 'index.json').read_text())
        self.assertEqual(index['failures'], 1)
        self.assertEqual(index['artifact_error'], 'PERFORMANCE_SAMPLE_QUALITY')

    def test_synthetic_performance_through_real_cli(self):
        from test_metrics import synthetic_table

        text = synthetic_table('8388608 2097152 float sum -1 100 80 80 0 110 72 72 0', sizes=(8388608,))
        manifest = self.prepare_performance(text, shards_per_pool=2)
        # Host credentials must not contaminate this synthetic measurement fixture.
        with patch.dict(os.environ, {'PATH': os.environ.get('PATH', os.defpath)}, clear=True):
            result = self.invoke(manifest)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = [json.loads(line) for line in (self.output / 'metrics.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 6)
        self.assertEqual({row['sample_index'] for row in rows}, {0, 1, 2})
        for row in rows:
            self.assertEqual(row['pccl_revision'], '1' * 40)
            self.assertEqual((self.output / row['log_path']).read_text(), text)
            self.assertEqual(row['data_status'], 'VALID')
        summary = json.loads((self.output / 'performance-summary.json').read_text())
        self.assertTrue(summary['complete'])
        self.assertEqual(len(summary['measurements']), 2)
        self.assertTrue(all(row['comparison_status'] == 'NO_BASELINE' and row['sample_count'] == 3
                            and row['sample_status'] == 'STABLE' and row['topology_status'] == 'UNVERIFIED'
                            for row in summary['measurements']))
        result_rows = [json.loads(line) for line in (self.output / 'results.jsonl').read_text().splitlines()]
        self.assertTrue(all(row['data_status'] == 'VALID' for row in result_rows))
        first = manifest['shards'][0]['cases'][0]
        directory = self.output / first['case_key'] / 'attempt-1'
        (directory / 'metrics.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'do not match'):
            execute.verified_metrics(directory, first, {**manifest['inputs'], **execute.execution_provenance()},
                                     str((directory / 'command.log').relative_to(self.output)))

    def test_redacted_performance_evidence_is_not_accepted(self):
        from test_metrics import synthetic_table

        text = synthetic_table('8388608 2097152 float sum -1 100 80 80 0 110 72 72 0', sizes=(8388608,))
        manifest = self.prepare_performance(text, samples=1)
        with patch.dict(os.environ, {'PATH': os.environ.get('PATH', os.defpath)}, clear=True):
            result = self.invoke(manifest, {'CI_ACCESS_TOKEN': '8388608'})
        self.assertEqual(result.returncode, 1)
        row = execute.read_json(next(self.output.glob('*/attempt-1/result.json')))
        self.assertEqual(row['data_status'], 'DATA_ERROR')
        self.assertFalse(list(self.output.rglob('metrics.jsonl')))
        log = next(self.output.rglob('command.log')).read_text()
        self.assertNotIn('8388608', log)
        self.assertIn('[REDACTED]', log)

    def test_performance_rejects_missing_data_even_with_exit_zero(self):
        manifest = self.prepare_performance('# Out of bounds values : 0 OK\n', samples=1)
        result = self.invoke(manifest)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        row = json.loads((self.output / 'results.jsonl').read_text())
        self.assertEqual(row['data_status'], 'DATA_ERROR')
        self.assertFalse(list(self.output.rglob('metrics.jsonl')))
        self.assertEqual(ET.parse(self.output / 'junit.xml').getroot().get('failures'), '1')

    def test_performance_nonzero_never_publishes_metrics(self):
        from test_metrics import synthetic_table

        manifest = self.prepare_performance(
            synthetic_table('8388608 2097152 float sum -1 100 80 80 0 110 72 72 0', sizes=(8388608,)),
            exit_code=7, samples=1)
        self.assertEqual(self.invoke(manifest).returncode, 1)
        self.assertFalse(list(self.output.rglob('metrics.jsonl')))

    def test_performance_cli_rejects_table_only_and_contradictory_launch(self):
        from test_metrics import synthetic_table, SYNTHETIC_RANKS

        complete = synthetic_table('8388608 2097152 float sum -1 100 80 80 0 110 72 72 0', sizes=(8388608,))
        texts = [synthetic_table('8388608 2097152 float sum -1 100 80 80 0 110 72 72 0', startup=False),
                 complete.replace('nThread 2', 'nThread 1').replace('iters: 100', 'iters: 1')
                 .replace('warmup iters: 5', 'warmup iters: 0').replace('average: MIN', 'average: AVG'),
                 complete.replace(SYNTHETIC_RANKS[1], ''),
                 complete.replace(SYNTHETIC_RANKS[1], SYNTHETIC_RANKS[0])]
        for index, text in enumerate(texts):
            self.bundle, self.output = self.root / f'bundle-{index}', self.root / f'output-{index}'
            with self.subTest(synthetic=index):
                manifest = self.prepare_performance(text, samples=1)
                result = self.invoke(manifest)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                row = json.loads((self.output / 'results.jsonl').read_text())
                self.assertEqual(row['data_status'], 'DATA_ERROR')
                self.assertFalse(list(self.output.rglob('metrics.jsonl')))

    def test_verified_metrics_revalidates_startup_instead_of_trusting_artifacts(self):
        from metrics import parse_metrics, write_metrics
        from test_metrics import synthetic_table
        from test_manifest import performance_specs
        from manifest import case_key

        spec = performance_specs()[0]
        row = {'spec': spec, 'case_key': case_key(spec)}
        inputs = {**execute.execution_provenance(), 'pccl_revision': 'synthetic', 'pccl_tests_revision': 'synthetic'}
        text = synthetic_table('8388608 2097152 float sum -1 100 80 80 0 110 72 72 0', sizes=(8388608,))
        log = self.root / 'command.log'
        log.write_text(text)
        write_metrics(parse_metrics(text, spec, row['case_key'], inputs, 'synthetic/command.log'), self.root)
        self.assertEqual(len(execute.verified_metrics(self.root, row, inputs, 'synthetic/command.log')), 2)
        for bad in (text.replace('average: MIN', 'average: AVG'),
                    synthetic_table('8388608 2097152 float sum -1 100 80 80 0 110 72 72 0', startup=False)):
            log.write_text(bad)
            with self.subTest(synthetic=bad), self.assertRaisesRegex(ValueError, 'startup'):
                execute.verified_metrics(self.root, row, inputs, 'synthetic/command.log')

    def test_performance_rejects_undeclared_inherited_environment(self):
        from command import _prepare
        from manifest import measurement_identity
        from test_manifest import performance_specs

        runtime = dict(test_root=str(self.root), binary_dir=str(self.root), result_dir=str(self.output), mpi_options=[])
        for key in ('PCCL_MAX_NCHANNELS', 'NCCL_MAX_NCHANNELS', 'HGGC_VISIBLE_DEVICES',
                    'OMPI_MCA_btl', 'UCX_TLS'):
            spec = performance_specs()[0]
            with self.subTest(key=key), patch.dict(os.environ, {key: 'private-synthetic-value'}, clear=True):
                with self.assertRaisesRegex(ValueError, 'explicit env') as raised:
                    _prepare(spec, runtime)
                self.assertNotIn('private-synthetic-value', str(raised.exception))
                original = measurement_identity(spec)
                spec['env'][key] = None
                _, env = _prepare(spec, runtime)
                self.assertNotIn(key, env)
                self.assertNotEqual(measurement_identity(spec), original)
                removed = measurement_identity(spec)
                spec['env'][key] = 'declared-synthetic-value'
                _, env = _prepare(spec, runtime)
                self.assertEqual(env[key], 'declared-synthetic-value')
                self.assertNotEqual(measurement_identity(spec), removed)
                spec['test_type'] = 'functional'
                del spec['env'][key]
                self.assertEqual(_prepare(spec, runtime)[1][key], 'private-synthetic-value')

    def test_performance_cli_rejects_hidden_environment_before_command(self):
        from test_metrics import synthetic_table

        text = synthetic_table('8388608 2097152 float sum -1 100 80 80 0 110 72 72 0', sizes=(8388608,))
        manifest = self.prepare_performance(text, samples=1)
        result = self.invoke(manifest, {'PCCL_MAX_NCHANNELS': 'synthetic-undeclared-value'})
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertFalse(list(self.output.rglob('command.log')))
        self.assertFalse(list(self.output.rglob('metrics.jsonl')))
        row = json.loads((self.output / 'results.jsonl').read_text())
        self.assertEqual(row['data_status'], 'DATA_ERROR')
        self.assertIn('explicit env', row['error'])
        self.assertNotIn('synthetic-undeclared-value', row['error'])

    def test_unsafe_runtime_environment_fails_before_publication_or_execution(self):
        manifest = self.prepare(['raise AssertionError("must not run")'])
        result = self.invoke(manifest, {'UCX_TLS': 'private-synthetic-value'})
        self.assertEqual(result.returncode, 1)
        self.assertFalse(self.output.exists())
        self.assertNotIn('private-synthetic-value', result.stdout + result.stderr)

    def test_mpi_options_file_is_consumed_and_invalid_file_fails_before_execution(self):
        options_file = self.root / 'options.json'
        options = ['--hostfile', str(self.root / 'hosts with spaces')]
        options_file.write_text(json.dumps(options))
        manifest = self.prepare([f'import sys; assert sys.argv[1:] == {options!r}'])
        row = manifest['shards'][0]['cases'][0]
        spec = row['spec']
        spec['argv'].append('{mpi_options}')
        second = self.root / 'options-bundle'
        manifest = build_manifest([spec], {})
        write_bundle(manifest, second)
        self.bundle = second
        options_file.write_text('not JSON')
        self.assertNotEqual(self.invoke(manifest, {'PPU_MPI_OPTIONS_FILE': str(options_file)}).returncode, 0)
        self.assertFalse(self.output.exists())
        options_file.write_text(json.dumps(options))
        result = self.invoke(manifest, {'PPU_MPI_OPTIONS_FILE': str(options_file)})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cancellation_terminates_the_command(self):
        manifest = self.prepare(["import time; print('ready', flush=True); time.sleep(60)"])
        log = (self.root / 'executor.log').open('w')
        self.addCleanup(log.close)
        process = subprocess.Popen(self.command(manifest), stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, 'NODE_RANK': '0'}, start_new_session=True)

        def finish():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)

        self.addCleanup(finish)
        deadline = time.monotonic() + 15
        while True:
            records = list(self.output.glob('*/attempt-1/command-process.json'))
            if records and execute.read_json(records[0]).get('active') is True:
                break
            self.assertIsNone(process.poll(), (self.root / 'executor.log').read_text())
            self.assertLess(time.monotonic(), deadline, 'Command did not start')
            try:
                process.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                pass
        process.send_signal(signal.SIGTERM)
        self.assertNotEqual(process.wait(timeout=20), 0)
        index = json.loads((self.output / 'index.json').read_text())
        self.assertTrue(index['cancelled'])
        self.assertEqual(index['failures'], 1)
        record = json.loads(next(self.output.glob('*/attempt-1/command-process.json')).read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(record['pid'], 0)


class LifecycleTests(unittest.TestCase):
    """Exercise real local Python processes, not runner capture, MPI or hardware.

    A loader stub supplies already-partitioned Suites; grouping belongs to the
    manifest tests. The command adapter removes source-runner metadata only for
    these synthetic Python Cases. Production command validation is not changed.
    """

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / 'output'
        self.shard = {'shard_id': 'synthetic-suites', 'cases': [], 'suite_runs': []}
        self.manifest = {'inputs': {}, 'shards': [self.shard], 'selection_sha256': 'synthetic'}

    @staticmethod
    def argv(code):
        return ['{python}', '-c', code]

    def suite(self, name, codes, setup=(), teardown=(), timeout=5, mpi=False):
        suite = dict(suite_run_id=name, plan_id='synthetic-plan', suite_id=name,
                     case_keys=[], lifecycle=dict(setup=[self.argv(code) for code in setup],
                                                  teardown=[self.argv(code) for code in teardown], reuse=True))
        for number, code in enumerate(codes):
            key = f'{name}-case-{number}'
            spec = fixture_spec(len(self.shard['cases']))
            spec.update(argv=self.argv(code), timeout=timeout, processes_per_node=2 if mpi else 1,
                        source_context=dict(plan_id=suite['plan_id'], suite_id=name, case_id=key, suite_run_id=name))
            self.shard['cases'].append(dict(case_key=key, spec=spec))
            suite['case_keys'].append(key)
        self.shard['suite_runs'].append(suite)
        return suite

    def run_suites(self, mpi_options=()):
        original_run, original_check = execute.run_command, execute.check_command
        self.runtime = dict(test_root=str(self.root.resolve()), binary_dir=str(self.root.resolve()),
                            result_dir=str(self.output.absolute()), mpi_options=list(mpi_options),
                            python=sys.executable, inherited_environment=execute.safe_environment(os.environ))

        def synthetic_run(spec, runtime, interrupted):
            return original_run({key: value for key, value in spec.items() if key != 'source_context'},
                                runtime, interrupted)

        def synthetic_check(spec, runtime, record):
            return original_check({key: value for key, value in spec.items() if key != 'source_context'}, runtime, record)

        with patch.dict(os.environ, {'NODE_RANK': '0'}), \
                patch.object(execute, 'load_manifest', return_value=self.manifest), \
                patch.object(execute, 'run_command', side_effect=synthetic_run), \
                patch.object(execute, 'check_command', side_effect=synthetic_check):
            return execute.run_shard(self.root / 'manifest.json', self.shard['shard_id'],
                                     self.root, self.root, self.output, list(mpi_options))

    def evidence(self):
        return json.loads((self.output / 'suite-lifecycle.json').read_text())

    def index(self):
        return json.loads((self.output / 'index.json').read_text())

    def rewrite_hook_evidence(self, invocation, observed, process):
        """Recompute all artifact links so negative tests exercise the contract."""
        records = self.evidence()
        hook = records[0]['actual_hooks']['setup'][0]
        directory = (self.output / hook['command_record']).parent
        observed['invocation_sha256'] = execute.digest(invocation)
        observed['process'] = process
        for filename, value, checksum in (
                ('command-invocation.json', invocation, 'invocation_sha256'),
                ('command-observations.json', observed, 'observations_sha256'),
                ('command-process.json', process, 'record_sha256')):
            execute.write_json(directory / filename, value)
            hook[checksum] = execute._file_sha256(directory / filename)
        execute.write_json(directory / 'result.json', hook)
        execute.write_json(self.output / 'suite-lifecycle.json', records)

    def test_hook_logs_redact_inherited_credentials(self):
        value = 'offline-fixture-value-7319'
        code = 'import os; print(os.environ["CI_ACCESS_TOKEN"]); print("hook-safe-output")'
        self.suite('first', [], setup=[code], teardown=[code])
        with patch.dict(os.environ, {'CI_ACCESS_TOKEN': value}):
            self.assertEqual(self.run_suites(), 0)
        for path in self.output.rglob('*'):
            if path.is_file():
                self.assertNotIn(value, path.read_text(), str(path))
        logs = list(self.output.rglob('command.log'))
        self.assertEqual(len(logs), 2)
        self.assertTrue(all('hook-safe-output' in path.read_text() for path in logs))
        execute.verify_lifecycle(self.output, self.shard, self.runtime)

    def test_real_generic_hooks_publish_observed_invocations_without_source_execution(self):
        # No command/check/prepare stubs: hooks really run through command's
        # normal direct-command path, without a fabricated run_test contract.
        self.suite('first', [], setup=[
            'import json, os, sys; from pathlib import Path; '
            'Path("actual-hook.json").write_text(json.dumps(dict(pid=os.getpid(), argv=sys.argv[1:], '
            'cwd=os.getcwd(), path=os.environ["PATH"], setting=os.environ["UCX_TLS"]))); print("setup")'
        ], teardown=['from pathlib import Path; assert Path("actual-hook.json").is_file(); print("teardown")'])
        self.shard['suite_runs'][0]['lifecycle']['setup'][0].extend(
            ['{test_root}', '{binary_dir}', '{result_dir}', '{mpi_options}'])
        with patch.dict(os.environ, {'NODE_RANK': '0', 'UCX_TLS': 'synthetic',
                                    'CI_ACCESS_TOKEN': 'not-published-in-hook-evidence'}):
            self.runtime = dict(test_root=str(self.root.resolve()), binary_dir=str(self.root.resolve()),
                                result_dir=str(self.output), mpi_options=['--hostfile', 'synthetic hosts'],
                                python=sys.executable, inherited_environment=execute.safe_environment(os.environ))
            with patch.object(execute, 'load_manifest', return_value=self.manifest):
                self.assertEqual(execute.run_shard(self.root / 'manifest.json', self.shard['shard_id'],
                                                  self.root, self.root, self.output,
                                                  self.runtime['mpi_options']), 0)
        actual = json.loads((self.root / 'actual-hook.json').read_text())
        hook_dir = self.output / 'suite-lifecycle' / 'first' / 'setup-1'
        invocation = execute.read_json(hook_dir / 'command-invocation.json')
        observed = execute.read_json(hook_dir / 'command-observations.json')
        process = execute.read_json(hook_dir / 'command-process.json')
        self.assertEqual(invocation['source_ids'], dict(suite_run_id='first', plan_id='synthetic-plan',
                                                      suite_id='first', phase='setup', number=1))
        self.assertEqual(invocation['argv'][3:], actual['argv'])
        self.assertEqual(invocation['argv'][0], sys.executable)
        self.assertEqual(invocation['cwd'], actual['cwd'])
        self.assertEqual(invocation['effective_environment']['PATH'], actual['path'])
        self.assertEqual(invocation['effective_environment']['UCX_TLS'], actual['setting'])
        self.assertEqual(observed['process'], process)
        self.assertEqual(observed['process']['pid'], actual['pid'])
        self.assertEqual(observed['launches'][0]['argv'], invocation['argv'])
        self.assertEqual(observed['launches'][0]['effective_environment'], invocation['effective_environment'])
        self.assertNotIn('CI_ACCESS_TOKEN', json.dumps(invocation))
        self.assertNotIn('not-published-in-hook-evidence', json.dumps(observed))
        self.assertEqual(execute.verify_lifecycle(self.output, self.shard, self.runtime), self.evidence())
        report = ET.parse(self.output / 'junit.xml').getroot()
        self.assertEqual(report.get('failures'), '0')

    def test_hook_ownership_and_expansion_tampering_rejected_after_rehash(self):
        self.suite('first', ['print("case")'], setup=['print("setup")'])
        self.assertEqual(self.run_suites(), 0)
        directory = self.output / 'suite-lifecycle' / 'first' / 'setup-1'
        original = execute.read_json(directory / 'command-invocation.json')
        observed = execute.read_json(directory / 'command-observations.json')
        process = execute.read_json(directory / 'command-process.json')
        for field, value in (('suite_run_id', 'other-run'), ('plan_id', 'other-plan'),
                             ('suite_id', 'other-suite'), ('phase', 'teardown'), ('number', 2), ('number', True)):
            invocation = copy.deepcopy(original)
            invocation['source_ids'][field] = value
            self.rewrite_hook_evidence(invocation, copy.deepcopy(observed), process)
            with self.subTest(identity=field), self.assertRaisesRegex(ValueError, 'trusted manifest/runtime'):
                execute.verify_lifecycle(self.output, self.shard, self.runtime)
        for field, value in (('argv', [sys.executable, '-c', 'print("unrelated")']),
                             ('effective_environment', {'PATH': '/wrong/bin'})):
            invocation, changed = copy.deepcopy(original), copy.deepcopy(observed)
            invocation[field] = value
            changed['launches'][0][field] = value
            self.rewrite_hook_evidence(invocation, changed, process)
            with self.subTest(expansion=field), self.assertRaisesRegex(ValueError, 'trusted manifest/runtime'):
                execute.verify_lifecycle(self.output, self.shard, self.runtime)

    def test_hook_bindings_cannot_supply_their_own_trust_anchor(self):
        suite = self.suite('first', ['print("case")'], setup=['print("setup")'])
        self.assertEqual(self.run_suites(), 0)
        directory = self.output / 'suite-lifecycle' / 'first' / 'setup-1'
        original = execute.read_json(directory / 'command-invocation.json')
        observed = execute.read_json(directory / 'command-observations.json')
        process = execute.read_json(directory / 'command-process.json')
        spec = execute._hook_spec(suite['lifecycle']['setup'][0], self.shard['cases'])
        for field, value in (('test_root', '/wrong/root'), ('binary_dir', '/wrong/bin'),
                             ('result_dir', '/wrong/results'), ('mpi_options', ['--hostfile', 'other']),
                             ('python', '/wrong/python'), ('inherited_environment', {'PATH': '/wrong/inherited'})):
            bindings = copy.deepcopy(original['bindings'])
            bindings[field] = value
            invocation = execute._hook_invocation(spec, suite, 'setup', 1, bindings)
            changed = copy.deepcopy(observed)
            changed['launches'][0] = {key: invocation[key] for key in ('argv', 'cwd', 'effective_environment')}
            changed['launches'][0]['executable'] = invocation['argv'][0]
            self.rewrite_hook_evidence(invocation, changed, process)
            with self.subTest(binding=field), self.assertRaisesRegex(ValueError, 'trusted manifest/runtime'):
                execute.verify_lifecycle(self.output, self.shard, self.runtime)
        self.rewrite_hook_evidence(original, observed, process)
        with self.assertRaisesRegex(ValueError, 'trusted execution runtime'):
            execute.verify_lifecycle(self.output, self.shard)
        # Offline verification uses execution-time paths and environment, not
        # the summary host's current environment or artifact download directory.
        relocated = self.root / 'downloaded'
        self.output.rename(relocated)
        with patch.dict(os.environ, {'PATH': '/summary/bin', 'UCX_TLS': 'summary-only'}, clear=True):
            self.assertEqual(len(execute.verify_lifecycle(relocated, self.shard, self.runtime)), 1)

    def test_observed_launch_and_process_identity_are_required_after_rehash(self):
        self.suite('first', ['print("case")'], setup=['print("setup")'])
        self.assertEqual(self.run_suites(), 0)
        directory = self.output / 'suite-lifecycle' / 'first' / 'setup-1'
        invocation = execute.read_json(directory / 'command-invocation.json')
        original = execute.read_json(directory / 'command-observations.json')
        process = execute.read_json(directory / 'command-process.json')
        for launches in ([], original['launches'] * 2,
                         [dict(original['launches'][0], argv=[sys.executable, '-c', 'print("wrong hook")'])],
                         [dict(original['launches'][0], effective_environment={'PATH': '/wrong'})]):
            observed = dict(original, launches=launches)
            self.rewrite_hook_evidence(invocation, observed, process)
            with self.subTest(launches=launches), self.assertRaisesRegex(ValueError, 'observed launch'):
                execute.verify_lifecycle(self.output, self.shard, self.runtime)
        for change in ({'pid': None}, {'pid': True}, {'pgid': process['pgid'] + 1}, {'identity': None},
                       {'identity': dict(process['identity'], pid=process['pid'] + 1)},
                       {'identity': dict(process['identity'], starttime='')}):
            self.rewrite_hook_evidence(invocation, copy.deepcopy(original), dict(process, **change))
            with self.subTest(process=change), self.assertRaisesRegex(ValueError, 'process identity'):
                execute.verify_lifecycle(self.output, self.shard, self.runtime)

    def test_two_suites_keep_setup_cases_teardown_order_and_isolation(self):
        for name in ('first', 'second'):
            self.suite(name, [
                f'from pathlib import Path; assert Path("marker").read_text() == {name!r}; '
                f'Path("order").open("a").write({name!r} + "-case-{number}\\n")' for number in range(2)
            ], setup=[
                'from pathlib import Path; assert not Path("marker").exists(); '
                f'Path("marker").write_text({name!r}); Path("order").open("a").write({name!r} + "-setup\\n")',
                'from pathlib import Path; import sys; assert Path(sys.argv[1]).is_dir(); '
                'assert Path(sys.argv[2]).is_dir(); assert Path(sys.argv[3]).is_dir()'
            ], teardown=[
                'from pathlib import Path; Path("marker").unlink(); '
                f'Path("order").open("a").write({name!r} + "-teardown\\n")'
            ])
            self.shard['suite_runs'][-1]['lifecycle']['setup'][1].extend(
                ['{test_root}', '{binary_dir}', '{result_dir}'])
        self.assertEqual(self.run_suites(), 0)
        self.assertEqual((self.root / 'order').read_text().splitlines(),
                         [f'{name}-{step}' for name in ('first', 'second')
                          for step in ('setup', 'case-0', 'case-1', 'teardown')])
        self.assertFalse((self.root / 'marker').exists())
        self.assertEqual(execute.verify_lifecycle(self.output, self.shard, self.runtime), self.evidence())
        self.assertEqual(self.index()['lifecycle_error'], '')
        rows = execute.read_jsonl(self.output / 'results.jsonl')
        for row, definition in zip(rows, self.shard['cases']):
            self.assertEqual(row['source_context'], definition['spec']['source_context'])
        report = ET.parse(self.output / 'junit.xml').getroot()
        self.assertEqual([case.get('classname') for case in report.findall('testcase')],
                         ['synthetic-plan.first'] * 2 + ['synthetic-plan.second'] * 2)
        self.assertEqual(len(list((self.output / 'suite-lifecycle').glob('*/*/result.json'))), 6)

    def test_empty_hooks_and_empty_suite_still_have_records(self):
        self.suite('empty-suite', [], setup=['from pathlib import Path; Path("empty").touch()'],
                   teardown=['from pathlib import Path; Path("empty").unlink()'])
        self.suite('empty-hooks', ['print("case")'])
        self.assertEqual(self.run_suites(), 0)
        records = execute.verify_lifecycle(self.output, self.shard, self.runtime)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]['case_results'], [])
        self.assertEqual(records[1]['expected_hooks'], {'setup': [], 'teardown': []})
        self.assertEqual(records[1]['actual_hooks'], {'setup': [], 'teardown': []})
        self.assertFalse((self.root / 'empty').exists())

    def test_hook_mpi_placeholder_and_environment_use_case_bindings(self):
        suite = self.suite('first', ['print("case")'], setup=[
            'import os, sys; assert sys.argv[1:] == ["--hostfile", "synthetic hosts"]; '
            'assert os.environ["PCCL_HOOK_PATH"].endswith("/suite-lifecycle/first/setup-1")'
        ])
        suite['lifecycle']['setup'][0].append('{mpi_options}')
        self.shard['cases'][0]['spec']['env'] = {'PCCL_HOOK_PATH': '{result_dir}'}
        self.assertEqual(self.run_suites(['--hostfile', 'synthetic hosts']), 0)
        execute.verify_lifecycle(self.output, self.shard, self.runtime)

    def test_setup_failure_attempts_all_teardowns_and_never_next_suite(self):
        self.suite('first', ['raise AssertionError("must not run")'],
                   setup=['from pathlib import Path; Path("marker").touch(); raise SystemExit(7)',
                          'raise AssertionError("must not run")'],
                   teardown=['raise SystemExit(8)', 'from pathlib import Path; Path("marker").unlink()'])
        self.suite('second', ['print("must not run")'], setup=['raise AssertionError("must not run")'])
        self.assertEqual(self.run_suites(), 1)
        records = self.evidence()
        self.assertEqual([record['status'] for record in records], ['FAILED', 'NOT_RUN'])
        self.assertEqual([hook['returncode'] for hook in records[0]['actual_hooks']['setup']], [7])
        self.assertEqual([hook['returncode'] for hook in records[0]['actual_hooks']['teardown']], [8, 0])
        self.assertFalse((self.root / 'marker').exists())
        self.assertEqual(execute.read_jsonl(self.output / 'results.jsonl'), [])
        self.assertTrue(self.index()['lifecycle_error'])
        self.assertGreater(self.index()['failures'], 0)
        with self.assertRaises(ValueError):
            execute.verify_lifecycle(self.output, self.shard, self.runtime)

    def test_setup_residual_child_is_cleaned_and_teardown_still_runs(self):
        self.suite('first', ['print("must not run")'], setup=[
            'import subprocess, sys; from pathlib import Path; '
            'p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]); '
            'Path("child-pid").write_text(str(p.pid))'
        ], teardown=['from pathlib import Path; Path("cleaned").touch()'])
        self.assertEqual(self.run_suites(), 1)
        self.assertTrue((self.root / 'cleaned').exists())
        self.assertEqual(execute.read_jsonl(self.output / 'results.jsonl'), [])
        hook = self.evidence()[0]['actual_hooks']['setup'][0]
        self.assertEqual(hook['status'], 'ERROR')
        self.assertIn('residual children', hook['error'])
        import command

        process = json.loads((self.output / hook['command_record']).read_text())
        self.assertFalse(command._members(process['pgid']))
        self.assertTrue(self.index()['lifecycle_error'])

    def test_teardown_failure_counts_even_when_every_case_passes(self):
        self.suite('first', ['print("pass")'], teardown=['raise SystemExit(9)'])
        self.assertEqual(self.run_suites(), 1)
        self.assertEqual(execute.read_jsonl(self.output / 'results.jsonl')[0]['status'], 'PASSED')
        self.assertEqual(self.index()['failures'], 1)
        self.assertTrue(self.index()['lifecycle_error'])
        self.assertEqual(self.evidence()[0]['actual_hooks']['teardown'][0]['status'], 'FAILED')
        report = ET.parse(self.output / 'junit.xml').getroot()
        self.assertEqual(report.get('tests'), '2')
        self.assertEqual(report.get('failures'), '1')
        ordinary, lifecycle = report.findall('testcase')
        self.assertIsNone(ordinary.find('failure'))
        self.assertEqual(lifecycle.get('classname'), 'pccl.lifecycle')
        self.assertEqual(lifecycle.find('failure').get('message'), self.index()['lifecycle_error'])

    def test_successful_unrelated_process_cannot_satisfy_hook(self):
        import command

        self.suite('first', ['print("case")'], setup=[
            'from pathlib import Path; Path("expected-hook").touch()'
        ])
        prepare = command._prepare

        def wrong_hook(spec, runtime):
            argv, environment = prepare(spec, runtime)
            if 'suite-lifecycle' in Path(runtime['result_dir']).parts:
                argv = [sys.executable, '-c', 'print("unrelated successful process")']
            return argv, environment

        with patch.object(command, '_prepare', side_effect=wrong_hook):
            self.assertEqual(self.run_suites(), 1)
        self.assertFalse((self.root / 'expected-hook').exists())
        self.assertTrue(self.index()['lifecycle_error'])
        with self.assertRaises(ValueError):
            execute.verify_lifecycle(self.output, self.shard, self.runtime)

    def test_successful_hook_with_wrong_actual_environment_is_rejected(self):
        import command

        self.suite('first', ['print("case")'], setup=['print("setup")'], teardown=['print("cleanup")'])
        self.shard['cases'][0]['spec']['env'] = {'UCX_TLS': 'declared'}
        prepare = command._prepare

        def wrong_environment(spec, runtime):
            argv, environment = prepare(spec, runtime)
            if Path(runtime['result_dir']).name == 'setup-1':
                environment['UCX_TLS'] = 'unexpected'
            return argv, environment

        with patch.object(command, '_prepare', side_effect=wrong_environment):
            self.assertEqual(self.run_suites(), 1)
        hooks = self.evidence()[0]['actual_hooks']
        self.assertEqual(hooks['setup'][0]['returncode'], 0)
        self.assertEqual(hooks['setup'][0]['status'], 'ERROR')
        self.assertIn('observed launch', hooks['setup'][0]['error'])
        self.assertEqual(hooks['teardown'][0]['status'], 'PASSED')
        self.assertEqual(execute.read_jsonl(self.output / 'results.jsonl'), [])

    def test_teardown_failure_stops_next_suite(self):
        self.suite('first', ['print("pass")'], teardown=['raise SystemExit(9)'])
        self.suite('second', [], setup=['from pathlib import Path; Path("unexpected").touch()'])
        self.assertEqual(self.run_suites(), 1)
        self.assertFalse((self.root / 'unexpected').exists())
        self.assertEqual(self.evidence()[1]['status'], 'NOT_RUN')

    def test_ordinary_case_failure_continues_and_cleans_each_suite(self):
        self.suite('first', ['raise SystemExit(7)', 'print("remaining")'],
                   teardown=['from pathlib import Path; Path("cleaned").touch()'])
        self.suite('second', ['from pathlib import Path; assert Path("cleaned").exists()'])
        self.assertEqual(self.run_suites(), 1)
        self.assertEqual([row['status'] for row in execute.read_jsonl(self.output / 'results.jsonl')],
                         ['FAILED', 'PASSED', 'PASSED'])
        self.assertEqual(self.index()['failures'], 1)
        self.assertEqual(self.index()['lifecycle_error'], '')
        self.assertEqual(execute.verify_lifecycle(self.output, self.shard, self.runtime)[0]['status'], 'FAILED')

    def test_mpi_case_failure_stops_batch_but_runs_current_teardown(self):
        self.suite('first', ['raise SystemExit(7)', 'print("must not run")'], mpi=True,
                   teardown=['from pathlib import Path; Path("cleaned").touch()'])
        self.suite('second', ['print("must not run")'])
        self.assertEqual(self.run_suites(), 1)
        self.assertTrue((self.root / 'cleaned').exists())
        self.assertEqual(len(execute.read_jsonl(self.output / 'results.jsonl')), 1)
        self.assertEqual(self.evidence()[1]['status'], 'NOT_RUN')

    def test_case_error_stops_batch_but_runs_current_teardown(self):
        self.suite('first', ['print("unused")'], teardown=['from pathlib import Path; Path("cleaned").touch()'])
        self.shard['cases'][0]['spec']['argv'] = [str(self.root / 'missing-program')]
        self.suite('second', ['print("must not run")'])
        self.assertEqual(self.run_suites(), 1)
        self.assertTrue((self.root / 'cleaned').exists())
        self.assertEqual(execute.read_jsonl(self.output / 'results.jsonl')[0]['status'], 'ERROR')
        self.assertEqual(self.evidence()[1]['status'], 'NOT_RUN')

    def test_setup_and_case_timeout_cleanup_before_teardown(self):
        for phase in ('setup', 'case'):
            with self.subTest(phase=phase):
                self.output = self.root / f'output-{phase}'
                self.shard['cases'], self.shard['suite_runs'] = [], []
                self.suite('first', ['import time; time.sleep(60)' if phase == 'case' else 'print("unused")'],
                           setup=['import time; time.sleep(60)'] if phase == 'setup' else [], timeout=1,
                           teardown=['from pathlib import Path; Path("cleaned-' + phase + '").touch()'])
                self.suite('second', ['print("must not run")'])
                self.assertEqual(self.run_suites(), 1)
                self.assertTrue((self.root / f'cleaned-{phase}').exists())
                self.assertEqual(self.evidence()[1]['status'], 'NOT_RUN')
                path = (self.output / 'suite-lifecycle' / 'first' / 'setup-1' if phase == 'setup'
                        else self.output / 'first-case-0' / 'attempt-1')
                process = json.loads((path / 'command-process.json').read_text())
                self.assertEqual(process['state'], 'timeout')
                self.assertFalse(process['active'])
                with self.assertRaises(ProcessLookupError):
                    os.kill(process['pid'], 0)

    def test_setup_and_case_cancellation_still_run_teardown(self):
        cancel = 'import os, signal, time; os.kill(os.getppid(), signal.SIGTERM); time.sleep(60)'
        for phase in ('setup', 'case'):
            with self.subTest(phase=phase):
                self.output = self.root / f'output-{phase}'
                self.shard['cases'], self.shard['suite_runs'] = [], []
                self.suite('first', [cancel if phase == 'case' else 'print("unused")'],
                           setup=[cancel] if phase == 'setup' else [],
                           teardown=['from pathlib import Path; Path("cleaned-' + phase + '").touch()'])
                self.suite('second', ['print("must not run")'])
                self.assertEqual(self.run_suites(), 1)
                self.assertTrue(self.index()['cancelled'])
                self.assertTrue((self.root / f'cleaned-{phase}').exists())
                self.assertEqual(self.evidence()[0]['actual_hooks']['teardown'][0]['status'], 'PASSED')
                self.assertEqual(self.evidence()[1]['status'], 'NOT_RUN')
                path = (self.output / 'suite-lifecycle' / 'first' / 'setup-1' if phase == 'setup'
                        else self.output / 'first-case-0' / 'attempt-1')
                process = json.loads((path / 'command-process.json').read_text())
                self.assertFalse(process['active'])
                with self.assertRaises(ProcessLookupError):
                    os.kill(process['pid'], 0)

    def test_cancelled_teardown_remains_bounded_and_other_cleanup_runs(self):
        self.suite('first', ['import os, signal; os.kill(os.getppid(), signal.SIGTERM)'], timeout=1,
                   teardown=['import time; time.sleep(60)', 'from pathlib import Path; Path("cleaned").touch()'])
        started = time.monotonic()
        self.assertEqual(self.run_suites(), 1)
        self.assertLess(time.monotonic() - started, 10)
        self.assertTrue((self.root / 'cleaned').exists())
        hooks = self.evidence()[0]['actual_hooks']['teardown']
        self.assertEqual([hook['status'] for hook in hooks], ['ERROR', 'PASSED'])
        process = json.loads((self.output / hooks[0]['command_record']).read_text())
        self.assertEqual(process['state'], 'timeout')
        self.assertFalse(process['active'])

    def test_missing_and_altered_lifecycle_artifacts_are_rejected(self):
        self.suite('first', ['print("case")'], setup=['print("setup")'], teardown=['print("cleanup")'])
        self.suite('second', ['print("case")'])
        self.assertEqual(self.run_suites(), 0)
        original = self.evidence()
        path = self.output / 'suite-lifecycle.json'
        for corrupt in (None, [], list(reversed(original)), original + [original[0]]):
            with self.subTest(lifecycle=corrupt):
                if corrupt is None:
                    path.unlink()
                else:
                    execute.write_json(path, corrupt)
                with self.assertRaises(ValueError):
                    execute.verify_lifecycle(self.output, self.shard, self.runtime)
                execute.write_json(path, original)
        for key, value in (('expected_hooks', {'setup': [], 'teardown': []}),
                           ('actual_hooks', {'setup': [], 'teardown': []}), ('case_results', []),
                           ('case_keys', []), ('suite_id', 'other'), ('status', 'NOT_RUN')):
            altered = copy.deepcopy(original)
            altered[0][key] = value
            with self.subTest(field=key):
                execute.write_json(path, altered)
                with self.assertRaises(ValueError):
                    execute.verify_lifecycle(self.output, self.shard, self.runtime)
        execute.write_json(path, original)
        hook_dir = self.output / 'suite-lifecycle' / 'first' / 'setup-1'
        for filename in ('command.log', 'command-process.json', 'command-invocation.json',
                         'command-observations.json', 'result.json'):
            artifact = hook_dir / filename
            content = artifact.read_bytes()
            with self.subTest(artifact=filename):
                artifact.unlink()
                with self.assertRaises(ValueError):
                    execute.verify_lifecycle(self.output, self.shard, self.runtime)
                artifact.write_bytes(b'corrupt')
                with self.assertRaises(ValueError):
                    execute.verify_lifecycle(self.output, self.shard, self.runtime)
                artifact.write_bytes(content)
        self.assertEqual(execute.verify_lifecycle(self.output, self.shard, self.runtime), original)

    def test_failed_process_cannot_be_forged_as_successful_hook(self):
        self.suite('first', ['print("case")'], teardown=['raise SystemExit(9)'])
        self.assertEqual(self.run_suites(), 1)
        records = self.evidence()
        records[0]['status'] = 'PASSED'
        hook = records[0]['actual_hooks']['teardown'][0]
        hook.update(status='PASSED', error='', returncode=0)
        execute.write_json(self.output / 'suite-lifecycle' / 'first' / 'teardown-1' / 'result.json', hook)
        execute.write_json(self.output / 'suite-lifecycle.json', records)
        with self.assertRaisesRegex(ValueError, 'successful completed command'):
            execute.verify_lifecycle(self.output, self.shard, self.runtime)

    def test_process_cleanup_state_is_rechecked_even_with_updated_checksums(self):
        self.suite('first', ['print("case")'], teardown=['print("cleanup")'])
        self.assertEqual(self.run_suites(), 0)
        original = self.evidence()
        for patch_record in ({'active': True}, {'state': 'cleanup_failed'}, {'return_code': 1},
                             {'state': 'timeout'}, {'state': 'cancelled'}, {'return_code': False}):
            with self.subTest(record=patch_record):
                records = copy.deepcopy(original)
                hook = records[0]['actual_hooks']['teardown'][0]
                path = self.output / hook['command_record']
                process = json.loads(path.read_text())
                process.update(active=False, state='finished', return_code=0)
                process.update(patch_record)
                execute.write_json(path, process)
                hook['record_sha256'] = execute._file_sha256(path)
                execute.write_json(path.parent / 'result.json', hook)
                execute.write_json(self.output / 'suite-lifecycle.json', records)
                with self.assertRaises(ValueError):
                    execute.verify_lifecycle(self.output, self.shard, self.runtime)

    def test_case_context_and_completion_must_match_manifest_exactly(self):
        self.suite('first', ['print("case")'])
        self.suite('second', ['print("case")'])
        self.assertEqual(self.run_suites(), 0)
        path = self.output / 'results.jsonl'
        original = path.read_text()
        rows = execute.read_jsonl(path)
        for altered in (rows[:1], rows + rows[:1], list(reversed(rows))):
            with self.subTest(results=altered):
                path.write_text(''.join(json.dumps(row) + '\n' for row in altered))
                with self.assertRaises(ValueError):
                    execute.verify_lifecycle(self.output, self.shard, self.runtime)
        for field in ('plan_id', 'suite_id', 'case_id', 'suite_run_id'):
            altered = copy.deepcopy(rows)
            altered[0]['source_context'][field] = 'wrong'
            path.write_text(''.join(json.dumps(row) + '\n' for row in altered))
            execute.write_json(self.output / rows[0]['case_key'] / 'attempt-1' / 'result.json', altered[0])
            with self.subTest(context=field), self.assertRaises(ValueError):
                execute.verify_lifecycle(self.output, self.shard, self.runtime)
        path.write_text(original)
        execute.write_json(self.output / rows[0]['case_key'] / 'attempt-1' / 'result.json', rows[0])
        self.assertEqual(execute.verify_lifecycle(self.output, self.shard, self.runtime), self.evidence())

    def test_legacy_shards_require_no_lifecycle_file(self):
        self.assertEqual(execute.verify_lifecycle(self.root, {'cases': []}), [])


if __name__ == '__main__':
    unittest.main()
