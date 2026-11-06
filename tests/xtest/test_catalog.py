# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import subprocess
import sys
import unittest

from catalog import PRIVATE_MARKERS, case_views, load_catalog


CATALOG = Path(__file__).with_name('catalog.py')


class CaseReadingTests(unittest.TestCase):
    def test_case_projection_preserves_execution_model_and_single_source_command(self):
        bundle = load_catalog()
        views = case_views(bundle)
        self.assertEqual(len(views), 71)
        self.assertEqual(sum(len(view['executions']) for view in views), 71)
        self.assertEqual(sum(len(execution['profiles']) for view in views
                             for execution in view['executions']), 108)
        for case in bundle['cases']:
            self.assertNotIn('command', case)
        for view in views:
            self.assertIsInstance(view['source_command'], str)
            self.assertTrue(view['source_command'])
            self.assertNotEqual(view['source_command'], view['executions'][0]['argv'])
            self.assertEqual({execution['id'] for execution in view['executions']},
                             set(next(case for case in bundle['cases'] if case['id'] == view['id'])
                                 ['execution_ids']))

    def test_describe_json_groups_everything_under_the_case(self):
        view = case_views()[0]
        result = subprocess.run([sys.executable, str(CATALOG), 'describe', '--case', view['id'], '--json'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = json.loads(result.stdout)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['id'], view['id'])
        self.assertEqual(rows[0]['source_command'], view['source_command'])
        self.assertEqual(rows[0]['executions'], view['executions'])

    def test_human_reader_distinguishes_source_template_and_actual_reproduction(self):
        view = case_views()[0]
        result = subprocess.run([sys.executable, str(CATALOG), 'describe', '--case', view['name']],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Source command (audit only; never an execution entry):', result.stdout)
        self.assertIn('Execution templates (placeholders require runtime expansion):', result.stdout)
        self.assertIn('command-reproduction.txt', result.stdout)
        self.assertLess(result.stdout.index('Source command'), result.stdout.index('Execution templates'))
        self.assertFalse([marker for marker in PRIVATE_MARKERS if marker in result.stdout])

    def test_case_selection_requires_one_exact_match(self):
        result = subprocess.run([sys.executable, str(CATALOG), 'describe', '--case', 'not-a-case'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('exactly one Case', result.stderr)


if __name__ == '__main__':
    unittest.main()
