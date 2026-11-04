# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct JSON execution with synthetic commands/logs only; no MPI or hardware evidence."""

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
        self.assertFalse(list(self.output.rglob('framework*')))
        self.assertFalse(list(self.output.rglob('xtest-result.json')))
        self.assertNotIn('framework_log', rows[0])
        report = ET.parse(self.output / 'junit.xml').getroot()
        self.assertEqual(report.get('tests'), '2')
        self.assertEqual(report.get('failures'), '0')
        self.assertEqual(len(json.loads((self.output / 'timings.json').read_text())), 2)
        again = self.invoke(manifest)
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(len((self.output / 'results.jsonl').read_text().splitlines()), 2)

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

    def test_synthetic_performance_through_real_cli(self):
        from test_metrics import synthetic_table

        text = synthetic_table('8388608 2097152 float sum -1 100 80 80 0 110 72 72 0', sizes=(8388608,))
        manifest = self.prepare_performance(text, shards_per_pool=2)
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
        result = self.invoke(manifest, {'PCCL_MAX_NCHANNELS': 'private-synthetic-value'})
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertFalse(list(self.output.rglob('command.log')))
        self.assertFalse(list(self.output.rglob('metrics.jsonl')))
        row = json.loads((self.output / 'results.jsonl').read_text())
        self.assertEqual(row['data_status'], 'DATA_ERROR')
        self.assertIn('explicit env', row['error'])
        self.assertNotIn('private-synthetic-value', row['error'])

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
            logs = list(self.output.glob('*/attempt-1/command.log'))
            if logs and 'ready' in logs[0].read_text():
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


if __name__ == '__main__':
    unittest.main()
