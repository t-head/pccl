# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import os
from pathlib import Path
import sys
import tempfile
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
        (self.runner / 'run_test.py').write_text(
            'import os, sys\n'
            'print("arguments=" + repr(sys.argv[1:]))\n'
            'print("marker=" + os.environ.get("TEST_MARKER", ""))\n', encoding='utf-8')
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

    def execute(self, inputs, hostfile=None):
        self.output_number += 1
        return run_cases.run_all(inputs, self.test_root, self.binary,
                                 self.root / f'output-{self.output_number}', hostfile,
                                 {'PATH': '/usr/bin:/bin', 'LD_LIBRARY_PATH': '', 'PPU_HOME': '/sdk'})

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
        command = "test -f {hostfile} && printf '%s\\n' '{path}' '{ld_library_path}'"
        rows = self.execute(self.inputs([{'cmd': command}], nodes=2), hostfile)
        self.assertEqual(rows[0]['status'], 'PASSED')
        raw = (self.root / 'output-1/plan/suite/case_0/raw.log').read_text()
        self.assertIn('/sdk/bin:', raw)
        self.assertTrue((self.root / 'output-1/plan/suite/case_0/work/hosts.cfg').is_symlink())
        with mock.patch.dict(os.environ, {'NODE_RANK': '1'}):
            with self.assertRaisesRegex(ValueError, 'master/controller'):
                self.execute(self.inputs([{'cmd': 'true'}], nodes=2), hostfile)

    def test_multinode_failure_stops_remaining_cases(self):
        hostfile = self.root / 'hosts.cfg'
        hostfile.write_text('worker-a\nworker-b\n')
        marker = self.root / 'must-not-exist'
        rows = self.execute(self.inputs([{'cmd': 'exit 1'}, {'cmd': f'touch {marker}'}], nodes=2), hostfile)
        self.assertEqual([row['status'] for row in rows], ['FAILED', 'NOT_RUN'])
        self.assertFalse(marker.exists())

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
        self.assertEqual(self.execute(single)[0]['status'], 'PASSED')
        self.assertEqual(self.execute(cluster, hostfile)[0]['status'], 'PASSED')

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
