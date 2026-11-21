# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import run_cases


SOURCE = 'a' * 40
RUNNER = 'b' * 40
HEADER = '# size count type redop root cpu_time algbw busbw #wrong cpu_time algbw busbw #wrong'
UNITS = '#   (B)   (elements)   us   GB/s   us   GB/s'
SUMMARY = '# Out of bounds values : 0 OK'


class CsvRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input = self.root / 'input'
        (self.input / 'envs').mkdir(parents=True)
        self.test_root = self.root / 'build-output'
        self.runner = self.test_root / 'pccl-tests'
        self.binary = self.runner / 'build'
        self.binary.mkdir(parents=True)
        (self.test_root / 'lib').mkdir()
        helpers = self.test_root / 'pccl' / 'tests' / 'xtest'
        (helpers / 'cluster-bin').mkdir(parents=True)
        for name in ('cluster-ssh', 'cluster-exec', 'cluster_case_ssh.sh'):
            (helpers / 'cluster-bin' / name).write_text('# test helper\n')
        (helpers / 'remote_recovery.py').write_text(
            'import os, sys\n'
            "launch = next(arg for arg in sys.argv if arg.startswith('--launch='))\n"
            "os.execv('/bin/sh', ['/bin/sh', '-c', launch.partition('=')[2]])\n")
        (self.runner / 'run_test.py').write_text(
            'import os, sys\n'
            'print("arguments=" + repr(sys.argv[1:]))\n'
            'print("executable=" + sys.executable)\n'
            'print("version=" + ".".join(str(part) for part in sys.version_info[:2]))\n'
            'print("marker=" + os.environ.get("TEST_MARKER", ""))\n'
            'print("devices=" + os.environ.get("HGGC_VISIBLE_DEVICES", ""))\n', encoding='utf-8')
        (self.runner / 'run_test_config.py').write_text('# test fixture\n', encoding='utf-8')
        (self.input / 'export.json').write_text(json.dumps({
            'format_version': 1, 'source_revision': SOURCE, 'runner_revision': RUNNER,
            'xtest_version': '0.14.0'}) + '\n')
        (self.input / 'envs' / 'suite.env').write_text(
            'TEST_MARKER=from shared env\n'
            'PATH=/sdk/bin:$PATH\n'
            'LD_LIBRARY_PATH=/sdk/lib:$LD_LIBRARY_PATH\n')
        self.output_number = 0

    def tearDown(self):
        self.temporary.cleanup()

    def write_csv(self, name, fields, rows):
        with (self.input / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)

    def inputs(self, cases, nodes=1, timeout=10, profile='standard'):
        suite = {'plan': 'plan', 'suite': 'suite', 'nodes': str(nodes),
                 'cards_per_node': '8', 'timeout': str(timeout), 'env': 'suite.env'}
        self.write_csv('suites.csv', run_cases.SUITE_FIELDS, [suite])
        rows = []
        for index, case in enumerate(cases):
            rows.append({'plan': 'plan', 'suite': 'suite', 'name': case.get('name', f'case_{index}'),
                         'check': case.get('check', 'exit_code'), 'cmd': case['cmd']})
        self.write_csv(f'{profile}.csv', run_cases.CASE_FIELDS, rows)
        return run_cases.load_inputs(self.input, profile)

    def execute(self, inputs, hostfile=None, mpi_interface=None, build_identity=None,
                map_nvidia_devices=False, nvidia_devices=None):
        self.output_number += 1
        if build_identity is None:
            build_identity = {'schema_version': 1, 'pccl_revision': 'c' * 40,
                              'pccl_tests_revision': inputs['metadata']['runner_revision'],
                              'build_profile': inputs['profile'], 'artifact_sha256': 'd' * 64}
        with mock.patch.object(run_cases, '_capture_dmesg') as self.capture_dmesg:
            environment = {'PATH': '/usr/bin:/bin', 'LD_LIBRARY_PATH': '', 'PPU_HOME': '/sdk'}
            if nvidia_devices is not None:
                environment['NVIDIA_VISIBLE_DEVICES'] = nvidia_devices
            return run_cases.run_all(
                inputs, self.test_root, self.binary, self.root / f'output-{self.output_number}', hostfile,
                environment, mpi_interface=mpi_interface, build_identity=build_identity,
                map_nvidia_devices=map_nvidia_devices)

    def test_810e_device_allocation_is_mapped_and_forwarded(self):
        inputs = self.inputs([
            {'cmd': '/usr/bin/printenv HGGC_VISIBLE_DEVICES', 'name': 'default'},
            {'cmd': 'HGGC_VISIBLE_DEVICES=0,3 /usr/bin/printenv HGGC_VISIBLE_DEVICES', 'name': 'inline'},
            {'cmd': 'python run_test.py --devices 0,3', 'name': 'runner'},
        ])
        rows = self.execute(inputs, map_nvidia_devices=True, nvidia_devices='9,11,14,17')
        self.assertEqual([row['status'] for row in rows], ['PASSED'] * 3)
        self.assertEqual((self.root / 'output-1/plan/suite/default/raw.log').read_text().strip(), '9,11,14,17')
        self.assertEqual((self.root / 'output-1/plan/suite/inline/raw.log').read_text().strip(), '9,17')
        runner_log = (self.root / 'output-1/plan/suite/runner/raw.log').read_text()
        self.assertIn("'--devices', '9,17'", runner_log)
        self.assertIn('devices=9,11,14,17', runner_log)
        mpi = run_cases._with_device_environment('mpirun -np 2 true', '9,17')
        self.assertIn('mpirun -x HGGC_VISIBLE_DEVICES=9,17 -np 2', mpi)
        quoted = run_cases._with_device_environment("python run_test.py --mpi-options='-np 2'", '9,17')
        self.assertIn("--mpi-options='-x HGGC_VISIBLE_DEVICES=9,17 -np 2'", quoted)

        sparse = self.execute(self.inputs([{'cmd': 'HGGC_VISIBLE_DEVICES=0,3 /usr/bin/printenv HGGC_VISIBLE_DEVICES'}]),
                              map_nvidia_devices=True, nvidia_devices='8,13')
        self.assertEqual(sparse[0]['status'], 'PASSED')
        self.assertEqual((self.root / 'output-2/plan/suite/case_0/raw.log').read_text().strip(), '8,13')

    def test_810e_device_allocation_rejects_invalid_or_out_of_range_selection(self):
        with self.assertRaisesRegex(ValueError, 'finite'):
            run_cases._allocated_devices({'NVIDIA_VISIBLE_DEVICES': 'all'})
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            run_cases._allocated_devices({'NVIDIA_VISIBLE_DEVICES': '9,9'})
        rows = self.execute(self.inputs([{'cmd': 'HGGC_VISIBLE_DEVICES=0,1,4 true'}]),
                            map_nvidia_devices=True, nvidia_devices='9,11')
        self.assertEqual(rows[0]['status'], 'FAILED')
        self.assertIn('requires 3 PPUs but only 2 were allocated', rows[0]['error'])

    def test_device_mapping_is_opt_in(self):
        rows = self.execute(self.inputs([{'cmd': 'HGGC_VISIBLE_DEVICES=0,3 /usr/bin/printenv HGGC_VISIBLE_DEVICES'}]),
                            nvidia_devices='9,11,14,17')
        self.assertEqual(rows[0]['status'], 'PASSED')
        self.assertEqual((self.root / 'output-1/plan/suite/case_0/raw.log').read_text().strip(), '0,3')

    def test_device_environment_preserves_terminal_mpi_options_path_quotes(self):
        for quote in ("'", '"'):
            with self.subTest(quote=quote):
                command = (f"python run_test.py --mpi-options={quote}--hostfile /hosts "
                           f"-x PATH=/usr/bin{quote} --cross-node gdr")
                rewritten = run_cases._with_device_environment(
                    command, "0,1", logical=True, path="/wrappers:/usr/bin")
                arguments = shlex.split(rewritten)
                mpi_options = next(arg.split("=", 1)[1] for arg in arguments
                                   if arg.startswith("--mpi-options="))
                self.assertEqual(shlex.split(mpi_options),
                                 ["-x", "HGGC_VISIBLE_DEVICES=0,1",
                                  "-x", "PCCL_LOGICAL_VISIBLE_DEVICES=0,1",
                                  "--hostfile", "/hosts", "-x", "PATH=/wrappers:/usr/bin"])
                self.assertEqual(arguments[-2:], ["--cross-node", "gdr"])

    def test_device_environment_replaces_direct_mpi_path(self):
        command = "mpirun -x PATH=/usr/bin -np 2 all_reduce_perf"
        rewritten = run_cases._with_device_environment(
            command, "0,1", logical=True, path="/wrappers:/usr/bin")
        self.assertEqual(shlex.split(rewritten),
                         ["mpirun", "-x", "HGGC_VISIBLE_DEVICES=0,1",
                          "-x", "PCCL_LOGICAL_VISIBLE_DEVICES=0,1",
                          "-x", "PATH=/wrappers:/usr/bin", "-np", "2", "all_reduce_perf"])

    def test_multi_node_device_selection_is_mapped_per_worker(self):
        hostfile = self.root / 'hosts.cfg'
        hostfile.write_text('worker-a\nworker-b\n')
        rows = self.execute(self.inputs([{'cmd': 'mpirun -x PATH={path} --hostfile {hostfile} -np 2 true'}], nodes=2), hostfile,
                            map_nvidia_devices=True, nvidia_devices='9,11')
        command = (self.root / 'output-1/plan/suite/case_0/command.txt').read_text()
        self.assertIn('PCCL_LOGICAL_VISIBLE_DEVICES=0,1', command)
        self.assertIn('device-map-wrappers', command)
        self.assertNotIn('-x HGGC_VISIBLE_DEVICES=9,11', command)

        real = self.root / 'real-bin'
        real.mkdir()
        benchmark = real / 'all_reduce_perf'
        benchmark.write_text('#!/bin/sh\nprintf "%s" "$HGGC_VISIBLE_DEVICES"\n')
        benchmark.chmod(0o755)
        wrappers = run_cases._stage_device_wrappers(self.test_root)
        for allocated, expected in (('9,11', '9,11'), ('2,4', '2,4')):
            environment = dict(os.environ, NVIDIA_VISIBLE_DEVICES=allocated,
                               PCCL_LOGICAL_VISIBLE_DEVICES='0,1',
                               PATH=str(wrappers) + os.pathsep + str(real) + os.pathsep + '/usr/bin:/bin')
            result = subprocess.run(['all_reduce_perf'],
                                    env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, expected)
        environment = dict(os.environ, NVIDIA_VISIBLE_DEVICES='9',
                           PCCL_LOGICAL_VISIBLE_DEVICES='0,1',
                           PATH=str(wrappers) + os.pathsep + str(real) + os.pathsep + '/usr/bin:/bin')
        result = subprocess.run(['all_reduce_perf'], env=environment,
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('exceeds this worker', result.stderr)

    def test_multi_node_fallback_preserves_logical_indices(self):
        command, effective, _ = run_cases._map_case_devices(
            'HGGC_VISIBLE_DEVICES=0,3 true', [9, 11], physical=False)
        self.assertEqual(command, 'HGGC_VISIBLE_DEVICES=0,1 true')
        self.assertEqual(effective, '0,1')

    def test_dmesg_is_captured_only_for_failed_cases(self):
        rows = self.execute(self.inputs([{'cmd': 'exit 1'}, {'cmd': 'true'}]))
        self.assertEqual([row['status'] for row in rows], ['FAILED', 'PASSED'])
        self.capture_dmesg.assert_called_once()
        case, runtime = self.capture_dmesg.call_args.args
        self.assertEqual(case['name'], 'case_0')
        self.assertEqual(runtime['output'], (self.root / 'output-1').resolve())

    def test_dmesg_capture_writes_local_and_worker_logs(self):
        case = {'plan': 'plan', 'suite': 'suite', 'name': 'case', 'suite_spec': {'nodes': 1}}
        runtime = {'output': self.root, 'hostfile': None}
        directory = self.root / 'plan/suite/case'
        directory.mkdir(parents=True)
        completed = mock.Mock(stdout=b'kernel diagnostic\n', returncode=0)
        with mock.patch.object(run_cases.subprocess, 'run', return_value=completed) as invoke:
            run_cases._capture_dmesg(case, runtime)
        self.assertEqual((directory / 'dmesg.log').read_bytes(), b'kernel diagnostic\n')
        self.assertEqual(invoke.call_args.args[0], ['dmesg'])

        hostfile = self.root / 'hosts.cfg'
        hostfile.write_text('worker-a slots=2\nworker-b slots=2\n')
        case['suite_spec']['nodes'] = 2
        runtime['hostfile'] = hostfile
        with mock.patch.object(run_cases.subprocess, 'run', return_value=completed) as invoke:
            run_cases._capture_dmesg(case, runtime)
        self.assertEqual((directory / 'dmesg-worker-0.log').read_bytes(), b'kernel diagnostic\n')
        self.assertEqual((directory / 'dmesg-worker-1.log').read_bytes(), b'kernel diagnostic\n')
        self.assertEqual([call.args[0] for call in invoke.call_args_list],
                         [['cluster-ssh', 'worker-a', 'dmesg'], ['cluster-ssh', 'worker-b', 'dmesg']])

    def test_dmesg_capture_is_concurrent_and_best_effort(self):
        case = {'plan': 'plan', 'suite': 'suite', 'name': 'case', 'suite_spec': {'nodes': 2}}
        hostfile = self.root / 'hosts.cfg'
        hostfile.write_text('worker-a\nworker-b\n')
        directory = self.root / 'plan/suite/case'
        directory.mkdir(parents=True)
        runtime = {'output': self.root, 'hostfile': hostfile}
        barrier = threading.Barrier(2)

        def capture(*args, **kwargs):
            barrier.wait(timeout=2)
            return mock.Mock(stdout=b'kernel diagnostic\n', returncode=0)

        with mock.patch.object(run_cases.subprocess, 'run', side_effect=capture):
            run_cases._capture_dmesg(case, runtime)
        self.assertTrue((directory / 'dmesg-worker-0.log').is_file())
        self.assertTrue((directory / 'dmesg-worker-1.log').is_file())

        runtime['hostfile'] = self.root / 'missing-hosts'
        run_cases._capture_dmesg(case, runtime)
        self.assertIn('Unable to capture dmesg', (directory / 'dmesg.log').read_text())
        hostfile.write_text('# no workers\n')
        runtime['hostfile'] = hostfile
        run_cases._capture_dmesg(case, runtime)
        self.assertIn('No workers are available', (directory / 'dmesg.log').read_text())

    def test_verified_build_identity_is_retained_with_results(self):
        inputs = self.inputs([{'cmd': 'printf "build identity\\n"'}])
        record = {'schema_version': 1, 'pccl_revision': 'c' * 40, 'pccl_tests_revision': RUNNER,
                  'build_profile': 'standard', 'artifact_sha256': 'd' * 64}
        path = self.root / 'output_standard.build.json'
        path.write_text(json.dumps(record))
        identity = run_cases._verified_build_identity(inputs, path, 'c' * 40, 'd' * 64)
        rows = self.execute(inputs, build_identity=identity)
        self.assertEqual(rows[0]['status'], 'PASSED')
        retained = json.loads((self.root / 'output-1/build.json').read_text())
        self.assertEqual(retained, record)
        with self.assertRaisesRegex(ValueError, 'Build record differs'):
            run_cases._verified_build_identity(inputs, path, 'e' * 40, 'd' * 64)

    def test_csv_shell_text_and_raw_log_are_preserved(self):
        command = "printf '%s\\n' 'USER=root NCCL WARN https://example.invalid/a,b \"quoted\" @tag'"
        rows = self.execute(self.inputs([{'cmd': command}]))
        self.assertEqual(rows[0]['status'], 'PASSED')
        raw = self.root / 'output-1/plan/suite/case_0/raw.log'
        self.assertEqual(raw.read_text(), 'USER=root NCCL WARN https://example.invalid/a,b "quoted" @tag\n')
        selected = self.inputs([{'cmd': command}])['cases'][0]['cmd']
        self.assertEqual(selected, command)
        self.assertNotIn('xtest', sys.modules)
        self.assertNotIn('mmengine', sys.modules)

    def test_single_node_failure_does_not_stop_following_case(self):
        rows = self.execute(self.inputs([{'cmd': 'printf "first failed\\n"; exit 7'},
                                     {'cmd': 'printf "second ran\\n"'}]))
        self.assertEqual([row['status'] for row in rows], ['FAILED', 'PASSED'])
        self.assertEqual(rows[0]['return_code'], 7)
        self.assertTrue((self.root / 'output-1/plan/suite/case_1/raw.log').is_file())

    def test_runner_staging_and_shared_environment(self):
        rows = self.execute(self.inputs([{'cmd': "python3 run_test.py --value 'a,b \"quoted\"'"}]))
        self.assertEqual(rows[0]['status'], 'PASSED')
        text = (self.root / 'output-1/plan/suite/case_0/raw.log').read_text()
        self.assertIn('arguments=' + repr(['--value', 'a,b "quoted"']), text)
        self.assertIn('marker=from shared env', text)
        self.assertIn('version=' + '.'.join(str(part) for part in sys.version_info[:2]), text)
        work = self.root / 'output-1/plan/suite/case_0/work'
        self.assertEqual((work / 'run_test.py').resolve(), (self.runner / 'run_test.py').resolve())
        self.assertEqual((work / 'run_test_config.py').resolve(), (self.runner / 'run_test_config.py').resolve())
        self.assertEqual((work / 'build').resolve(), self.binary.resolve())

    def test_validation_summary_is_required(self):
        passing = self.execute(self.inputs([{'cmd': f"printf '%s\\n' '{SUMMARY}'", 'check': 'validation_summary'}]))
        self.assertEqual(passing[0]['status'], 'PASSED')
        failing = self.execute(self.inputs([{'cmd': 'printf "no summary\\n"', 'check': 'validation_summary'}]))
        self.assertEqual(failing[0]['status'], 'FAILED')
        self.assertIn('validation summary', failing[0]['error'])

    def performance_command(self, rows='16 4 float sum -1 12.5 1.25 0.75 0 10.0 1.5 0.9 0'):
        log = self.test_root / 'performance.log'
        log.write_text('\n'.join([
            'NCCL WARN this warning must stay verbatim', '# out-of-place in-place', HEADER, UNITS,
            rows, SUMMARY, '# Avg bus bandwidth : 0.9', 'NO_BASELINE sample_spread=200%']) + '\n')
        return f'cat {log}'

    def test_performance_is_recorded_without_baseline_or_stability_gate(self):
        rows = self.execute(self.inputs([{'cmd': self.performance_command(), 'check': 'performance'}]))
        self.assertEqual(rows[0]['status'], 'PASSED')
        metrics = self.root / 'output-1/plan/suite/case_0/metrics.csv'
        with metrics.open(newline='') as stream:
            values = list(csv.DictReader(stream))
        self.assertEqual(len(values), 2)
        self.assertEqual({row['placement'] for row in values}, {'out-of-place', 'in-place'})
        self.assertEqual(values[0]['time_kind'], 'cpu_time')
        raw = (self.root / 'output-1/plan/suite/case_0/raw.log').read_text()
        self.assertIn('NCCL WARN this warning must stay verbatim', raw)
        self.assertIn('NO_BASELINE sample_spread=200%', raw)

    def test_performance_requires_measurements_and_valid_data(self):
        rows = self.execute(self.inputs([{'cmd': f"printf '%s\\n' '{SUMMARY}'", 'check': 'performance'}]))
        self.assertEqual(rows[0]['status'], 'FAILED')
        self.assertIn('Missing performance table', rows[0]['error'])
        command = self.performance_command('16 4 float sum -1 12.5 1.25 0.75 1 10.0 1.5 0.9 0')
        rows = self.execute(self.inputs([{'cmd': command, 'check': 'performance'}]))
        self.assertEqual(rows[0]['status'], 'FAILED')
        self.assertIn('wrong values', rows[0]['error'])

    def test_timeout_terminates_case_and_preserves_log(self):
        rows = self.execute(self.inputs([{'cmd': 'printf "before timeout\\n"; sleep 30'}], timeout=1))
        self.assertEqual(rows[0]['status'], 'TIMEOUT')
        self.assertLess(rows[0]['duration_seconds'], 5)
        self.assertIn('before timeout', (self.root / 'output-1/plan/suite/case_0/raw.log').read_text())

    def test_multinode_binds_real_hostfile_and_runs_on_controller(self):
        hostfile = self.root / 'allocated-hosts.cfg'
        hostfile.write_text('worker-a slots=8\nworker-b slots=8\n')
        command = "test -f {hostfile} && printf '%s\\n' '{path}' '{ld_library_path}' '{mpi_interface}'"
        rows = self.execute(self.inputs([{'cmd': command}], nodes=2), hostfile, mpi_interface='eth0')
        self.assertEqual(rows[0]['status'], 'PASSED')
        raw = (self.root / 'output-1/plan/suite/case_0/raw.log').read_text()
        self.assertIn('/sdk/bin:', raw)
        self.assertIn('eth0', raw)
        self.assertTrue((self.root / 'output-1/plan/suite/case_0/work/hosts.cfg').is_symlink())
        with mock.patch.dict(os.environ, {'NODE_RANK': '1'}):
            with self.assertRaisesRegex(ValueError, 'master/controller'):
                self.execute(self.inputs([{'cmd': 'true'}], nodes=2), hostfile)

    def test_multinode_failure_recovers_and_continues(self):
        hostfile = self.root / 'hosts.cfg'
        hostfile.write_text('worker-a\nworker-b\n')
        marker = self.root / 'must-not-exist'
        with mock.patch.object(run_cases, '_recover_multinode', return_value=True) as recovery:
            rows = self.execute(self.inputs([{'cmd': 'exit 1'}, {'cmd': f'touch {marker}'}], nodes=2), hostfile)
        self.assertEqual([row['status'] for row in rows], ['FAILED', 'PASSED'])
        self.assertTrue(marker.exists())
        recovery.assert_called_once()

    def test_multinode_recovery_failure_stops_remaining_cases(self):
        hostfile = self.root / 'hosts.cfg'
        hostfile.write_text('worker-a\nworker-b\n')
        marker = self.root / 'must-not-exist'
        with mock.patch.object(run_cases, '_recover_multinode', return_value=False):
            rows = self.execute(self.inputs([{'cmd': 'exit 1'}, {'cmd': f'touch {marker}'}], nodes=2), hostfile)
        self.assertEqual([row['status'] for row in rows], ['FAILED', 'NOT_RUN'])
        self.assertFalse(marker.exists())

    def test_remote_recovery_faults_are_unsafe_and_controller_pid_is_forwarded(self):
        script = self.root / 'remote_recovery.py'
        completed = mock.Mock(stdout='{"safe": true}\n', stderr='', returncode=0)
        with mock.patch.object(run_cases.subprocess, 'run', return_value=completed) as invoke:
            result = run_cases._remote_recovery('worker-a', script, 'a' * 32)
        self.assertTrue(result['safe'])
        command = invoke.call_args.args[0]
        self.assertEqual(command[-2:], ['--case-id', 'a' * 32])
        for result_value in (
                mock.Mock(stdout='not-json\n', stderr='', returncode=0),
                mock.Mock(stdout='{"safe": false, "errors": ["worker failed"]}\n',
                          stderr='worker failed', returncode=1)):
            with self.subTest(result=result_value):
                with mock.patch.object(run_cases.subprocess, 'run', return_value=result_value):
                    self.assertFalse(run_cases._remote_recovery('worker-a', script, 5)['safe'])
        with mock.patch.object(run_cases.subprocess, 'run',
                               side_effect=subprocess.TimeoutExpired('cluster-exec', 15)):
            self.assertFalse(run_cases._remote_recovery('worker-a', script, 5)['safe'])

    def test_recover_multinode_uses_one_group_case_id(self):
        hostfile = self.root / 'hosts.cfg'
        hostfile.write_text('worker-a\nworker-b\n')
        script = self.test_root / 'pccl' / 'tests' / 'xtest' / 'remote_recovery.py'
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text('# helper\n')
        case = self.inputs([{'cmd': 'exit 1'}], nodes=2)['cases'][0]
        output = self.root / 'recovery-output'
        directory = output / case['plan'] / case['suite'] / case['name']
        directory.mkdir(parents=True)
        runtime = {'hostfile': hostfile, 'test_root': self.test_root, 'output': output,
                   'recovery_id': 'a' * 32}
        row = {'duration_seconds': 1.0}
        with mock.patch.object(run_cases, '_remote_recovery', return_value={'safe': True}) as recover:
            safe = run_cases._recover_multinode(case, runtime, row)
        self.assertTrue(safe)
        self.assertEqual(recover.call_args_list[0].args[0], 'worker-a')
        self.assertEqual(recover.call_args_list[0].args[-1], 'a' * 32)
        self.assertEqual(recover.call_args_list[1].args[0], 'worker-b')
        self.assertEqual(recover.call_args_list[1].args[-1], 'a' * 32)
        self.assertTrue((directory / 'recovery.json').is_file())
        with mock.patch.object(run_cases, '_remote_recovery', side_effect=[{'safe': True}, {'safe': False}]):
            self.assertFalse(run_cases._recover_multinode(case, runtime, row))

    def test_environment_is_data_not_a_shell_script(self):
        marker = self.root / 'env-was-executed'
        with self.assertRaisesRegex(ValueError, 'Shell evaluation'):
            run_cases._expand_environment([('BAD', f'$(touch {marker})')], {}, {})
        self.assertFalse(marker.exists())

    def test_standard_and_notsm_are_independent_profiles(self):
        standard = self.inputs([{'cmd': 'true'}])
        self.write_csv('notsm.csv', run_cases.CASE_FIELDS, [
            {'plan': 'plan', 'suite': 'suite', 'name': 'notsm_only', 'check': 'exit_code', 'cmd': 'true'},
            {'plan': 'plan', 'suite': 'suite', 'name': 'notsm_other', 'check': 'exit_code', 'cmd': 'true'}])
        notsm = run_cases.load_inputs(self.input, 'notsm')
        self.assertEqual(len(standard['cases']), 1)
        self.assertEqual([row['name'] for row in notsm['cases']], ['notsm_only', 'notsm_other'])

    def test_node_filter_separates_mixed_profile_without_rewriting_csv(self):
        hostfile = self.root / 'hosts.cfg'
        hostfile.write_text('worker-a\nworker-b\n')
        self.inputs([{'cmd': 'true', 'name': 'single'}])
        self.write_csv('suites.csv', run_cases.SUITE_FIELDS, [
            {'plan': 'plan', 'suite': 'suite', 'nodes': '1', 'cards_per_node': '8',
             'timeout': '10', 'env': 'suite.env'},
            {'plan': 'plan', 'suite': 'cluster', 'nodes': '2', 'cards_per_node': '8',
             'timeout': '10', 'env': 'suite.env'}])
        self.write_csv('standard.csv', run_cases.CASE_FIELDS, [
            {'plan': 'plan', 'suite': 'suite', 'name': 'single', 'check': 'exit_code', 'cmd': 'true'},
            {'plan': 'plan', 'suite': 'cluster', 'name': 'cluster', 'check': 'exit_code', 'cmd': 'true'}])
        single = run_cases.load_inputs(self.input, 'standard', nodes=1)
        cluster = run_cases.load_inputs(self.input, 'standard', nodes=2)
        self.assertEqual([case['name'] for case in single['cases']], ['single'])
        self.assertEqual([case['name'] for case in cluster['cases']], ['cluster'])
        selected = run_cases.load_inputs(self.input, 'standard', nodes=1, suite='suite')
        self.assertEqual([case['name'] for case in selected['cases']], ['single'])
        with self.assertRaisesRegex(ValueError, 'no Cases'):
            run_cases.load_inputs(self.input, 'standard', nodes=1, suite='missing')
        with self.assertRaisesRegex(ValueError, 'portable'):
            run_cases.load_inputs(self.input, 'standard', suite='bad suite')
        self.assertEqual(self.execute(single)[0]['status'], 'PASSED')
        self.assertEqual(self.execute(cluster, hostfile)[0]['status'], 'PASSED')
        missing = self.inputs([{'cmd': 'printf {mpi_interface}', 'name': 'missing_interface'}], nodes=2)
        self.assertEqual(self.execute(missing, hostfile)[0]['status'], 'FAILED')

    def test_daily_python_launcher_and_unit_test_staging(self):
        (self.runner / 'unit_test.py').write_text('# compatibility fixture\n')
        command = "{python_env}/bin/python3.11 -c 'import sys; print(sys.version_info[:2])'"
        rows = self.execute(self.inputs([{'cmd': command}]))
        self.assertEqual(rows[0]['status'], 'PASSED')
        raw = self.root / 'output-1/plan/suite/case_0/raw.log'
        self.assertIn(str(sys.version_info[:2]), raw.read_text())
        work = self.root / 'output-1/plan/suite/case_0/work'
        self.assertEqual((work / 'unit_test.py').resolve(), (self.runner / 'unit_test.py').resolve())

    def test_structural_validation_fails_closed(self):
        scenarios = {
            'missing_suite': {'plan': 'other', 'suite': 'suite', 'name': 'case', 'check': 'exit_code', 'cmd': 'true'},
            'unknown_check': {'plan': 'plan', 'suite': 'suite', 'name': 'case', 'check': 'guess', 'cmd': 'true'},
            'empty_command': {'plan': 'plan', 'suite': 'suite', 'name': 'case', 'check': 'exit_code', 'cmd': ' '},
            'newline': {'plan': 'plan', 'suite': 'suite', 'name': 'case', 'check': 'exit_code', 'cmd': 'true\nfalse'},
        }
        for name, row in scenarios.items():
            with self.subTest(name=name):
                self.write_csv('suites.csv', run_cases.SUITE_FIELDS, [
                    {'plan': 'plan', 'suite': 'suite', 'nodes': '1', 'cards_per_node': '8',
                     'timeout': '10', 'env': 'suite.env'}])
                self.write_csv('standard.csv', run_cases.CASE_FIELDS, [row])
                with self.assertRaises(ValueError):
                    run_cases.load_inputs(self.input, 'standard')
        self.write_csv('standard.csv', run_cases.CASE_FIELDS, [
            {'plan': 'plan', 'suite': 'suite', 'name': 'duplicate', 'check': 'exit_code', 'cmd': 'true'},
            {'plan': 'plan', 'suite': 'suite', 'name': 'duplicate', 'check': 'exit_code', 'cmd': 'true'}])
        with self.assertRaisesRegex(ValueError, 'Duplicate Case'):
            run_cases.load_inputs(self.input, 'standard')

    def test_unknown_placeholder_and_unsafe_binding_are_rejected(self):
        rows = self.execute(self.inputs([{'cmd': 'printf {unknown_placeholder}'}]))
        self.assertEqual(rows[0]['status'], 'FAILED')
        self.assertIn('Unsupported placeholder', rows[0]['error'])
        with self.assertRaisesRegex(ValueError, 'shell-unsafe'):
            run_cases._command_binding('/tmp/path with spaces')


if __name__ == '__main__':
    unittest.main()
