# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
from pathlib import Path
import tempfile
import unittest

import reconcile_results
import run_cases


SOURCE = 'a' * 40
RUNNER = 'b' * 40
PCCL = 'c' * 40
STANDARD_SHA = '1' * 64
NOTSM_SHA = '2' * 64
HEADER = '# size count type redop root cpu_time algbw busbw #wrong cpu_time algbw busbw #wrong'
MEASUREMENT = '16 4 float sum -1 12.5 1.25 0.75 0 10.0 1.5 0.9 0'
SUMMARY = '# Out of bounds values : 0 OK'


class ReconcileResultsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input = self.root / 'input'
        (self.input / 'envs').mkdir(parents=True)
        (self.input / 'export.json').write_text(json.dumps({
            'format_version': 1, 'source_revision': SOURCE, 'runner_revision': RUNNER,
            'xtest_version': '0.12.0'}))
        (self.input / 'envs/suite.env').write_text('PATH=$PATH\nLD_LIBRARY_PATH=$LD_LIBRARY_PATH\n')
        self.write_csv('suites.csv', run_cases.SUITE_FIELDS, [
            {'plan': 'plan', 'suite': 'suite', 'nodes': '1', 'cards_per_node': '8',
             'timeout': '10', 'env': 'suite.env'},
            {'plan': 'plan', 'suite': 'cluster', 'nodes': '2', 'cards_per_node': '8',
             'timeout': '10', 'env': 'suite.env'}])
        self.write_csv('standard.csv', run_cases.CASE_FIELDS, [
            {'plan': 'plan', 'suite': 'suite', 'name': 'exit', 'check': 'exit_code', 'cmd': 'true'},
            {'plan': 'plan', 'suite': 'suite', 'name': 'performance', 'check': 'performance', 'cmd': 'true'},
            {'plan': 'plan', 'suite': 'cluster', 'name': 'two_node', 'check': 'exit_code', 'cmd': 'true'}])
        self.write_csv('notsm.csv', run_cases.CASE_FIELDS, [
            {'plan': 'plan', 'suite': 'suite', 'name': 'notsm', 'check': 'exit_code', 'cmd': 'true'}])
        self.results = self.root / 'results'
        self.write_group('standard', run_cases.load_inputs(self.input, 'standard', 1), STANDARD_SHA)
        self.write_group('notsm', run_cases.load_inputs(self.input, 'notsm', 1), NOTSM_SHA)
        self.write_group('cluster', run_cases.load_inputs(self.input, 'standard', 2), STANDARD_SHA)
        self.public_number = 0

    def tearDown(self):
        self.temporary.cleanup()

    def write_csv(self, name, fields, rows):
        with (self.input / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)

    def write_group(self, name, inputs, archive_sha):
        output = self.results / name
        output.mkdir(parents=True)
        identity = {'schema_version': 1, 'pccl_revision': PCCL, 'pccl_tests_revision': RUNNER,
                    'build_profile': inputs['profile'], 'artifact_sha256': archive_sha}
        run_cases._write_json(output / 'build.json', identity)
        rows = []
        for case in inputs['cases']:
            directory = output / case['plan'] / case['suite'] / case['name']
            directory.mkdir(parents=True)
            text = '\n'.join([HEADER, MEASUREMENT, SUMMARY]) + '\n' if case['check'] == 'performance' else 'passed\n'
            (directory / 'raw.log').write_text(text)
            if case['check'] == 'performance':
                with (directory / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=run_cases.METRIC_FIELDS, lineterminator='\n')
                    writer.writeheader()
                    writer.writerows(run_cases.parse_performance(text))
            row = {'plan': case['plan'], 'suite': case['suite'], 'name': case['name'],
                   'check': case['check'], 'status': 'PASSED', 'return_code': 0,
                   'duration_seconds': 0.01, 'error': '', 'raw_log': 'raw.log'}
            run_cases._write_json(directory / 'result.json', row)
            rows.append(row)
        run_cases._write_summary(output, rows)

    def reconcile(self):
        self.public_number += 1
        return reconcile_results.reconcile(
            self.input, self.results, self.root / f'public-{self.public_number}',
            PCCL, STANDARD_SHA, NOTSM_SHA)

    def case_directory(self, group, suite, name):
        return self.results / group / 'plan' / suite / name

    def test_exact_coverage_records_metrics_and_build_identities_are_accepted(self):
        self.assertEqual(self.reconcile(), {'standard': 2, 'notsm': 1, 'cluster': 1})
        public = self.root / 'public-1'
        self.assertTrue((public / 'standard/build.json').is_file())
        metrics = list(public.glob('standard/*/*/*/metrics.csv'))
        self.assertEqual(len(metrics), 1)
        self.assertGreater(metrics[0].stat().st_size, 0)

    def test_scheduled_selections_cover_both_complete_profile_exports(self):
        selections = reconcile_results.load_selections(self.input)

        def key(case):
            return case["plan"], case["suite"], case["name"]

        for profile in ("standard", "notsm"):
            complete = run_cases.load_inputs(self.input, profile)["cases"]
            scheduled = [case for inputs in selections.values() if inputs["profile"] == profile
                         for case in inputs["cases"]]
            self.assertEqual(sorted(map(key, complete)), sorted(map(key, scheduled)))

    def test_unsupported_node_counts_are_rejected_before_results_are_accepted(self):
        scenarios = (("standard", 4), ("standard", 8), ("standard", 16), ("notsm", 2))
        for number, (profile, nodes) in enumerate(scenarios):
            with self.subTest(profile=profile, nodes=nodes):
                if number:
                    self.tearDown()
                    self.setUp()
                suites = run_cases._read_csv(self.input / 'suites.csv', run_cases.SUITE_FIELDS)
                suites.append({'plan': 'plan', 'suite': 'unsupported', 'nodes': str(nodes),
                               'cards_per_node': '8', 'timeout': '10', 'env': 'suite.env'})
                self.write_csv('suites.csv', run_cases.SUITE_FIELDS, suites)
                rows = run_cases._read_csv(self.input / f'{profile}.csv', run_cases.CASE_FIELDS)
                rows.append({'plan': 'plan', 'suite': 'unsupported', 'name': 'never_run',
                             'check': 'exit_code', 'cmd': 'false'})
                self.write_csv(f'{profile}.csv', run_cases.CASE_FIELDS, rows)
                with self.assertRaisesRegex(ValueError, 'Unsupported CI resources'):
                    reconcile_results.load_selections(self.input)
                with self.assertRaisesRegex(ValueError, 'Unsupported CI resources'):
                    self.reconcile()
                self.assertFalse((self.root / 'public-1').exists())

    def test_unsupported_card_counts_are_rejected(self):
        suites = run_cases._read_csv(self.input / 'suites.csv', run_cases.SUITE_FIELDS)
        suites[0]['cards_per_node'] = '16'
        self.write_csv('suites.csv', run_cases.SUITE_FIELDS, suites)
        with self.assertRaisesRegex(ValueError, 'cards_per_node=16'):
            self.reconcile()

    def test_missing_scheduled_group_is_rejected(self):
        cases = run_cases._read_csv(self.input / 'standard.csv', run_cases.CASE_FIELDS)
        self.write_csv('standard.csv', run_cases.CASE_FIELDS,
                       [case for case in cases if case['suite'] != 'cluster'])
        with self.assertRaisesRegex(ValueError, 'cluster: no Cases'):
            reconcile_results.load_selections(self.input)

    def test_duplicate_and_unexpected_case_keys_are_rejected_even_with_matching_totals(self):
        path = self.results / 'standard/results.csv'
        rows = run_cases._read_csv(path, run_cases.RESULT_FIELDS)
        rows[1] = rows[0]
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=run_cases.RESULT_FIELDS, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, 'duplicate Case result keys'):
            self.reconcile()

    def test_unexpected_case_keys_are_rejected_even_with_matching_totals(self):
        path = self.results / 'notsm/results.csv'
        rows = run_cases._read_csv(path, run_cases.RESULT_FIELDS)
        rows[0]['name'] = 'unrelated'
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=run_cases.RESULT_FIELDS, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, 'Case coverage differs'):
            self.reconcile()

    def test_empty_logs_missing_records_and_empty_metrics_are_rejected(self):
        scenarios = ('empty_log', 'missing_record', 'empty_metrics')
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                if scenario != scenarios[0]:
                    self.tearDown()
                    self.setUp()
                if scenario == 'empty_log':
                    (self.case_directory('standard', 'suite', 'exit') / 'raw.log').write_text('')
                elif scenario == 'missing_record':
                    (self.case_directory('notsm', 'suite', 'notsm') / 'result.json').unlink()
                else:
                    (self.case_directory('standard', 'suite', 'performance') / 'metrics.csv').write_text('')
                with self.assertRaises(ValueError):
                    self.reconcile()

    def test_result_record_and_performance_contents_must_match(self):
        record = self.case_directory('standard', 'suite', 'exit') / 'result.json'
        data = json.loads(record.read_text())
        data['return_code'] = 1
        record.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'return_code'):
            self.reconcile()

    def test_standard_single_node_and_cluster_must_use_the_same_verified_archive(self):
        path = self.results / 'cluster/build.json'
        identity = json.loads(path.read_text())
        identity['artifact_sha256'] = '3' * 64
        path.write_text(json.dumps(identity))
        with self.assertRaisesRegex(ValueError, 'build identity'):
            self.reconcile()

    def test_metrics_must_match_the_raw_performance_table(self):
        path = self.case_directory('standard', 'suite', 'performance') / 'metrics.csv'
        rows = run_cases._read_csv(path, run_cases.METRIC_FIELDS)
        rows[0]['busbw'] = '999'
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=run_cases.METRIC_FIELDS, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, 'do not match the raw log'):
            self.reconcile()


if __name__ == '__main__':
    unittest.main()
