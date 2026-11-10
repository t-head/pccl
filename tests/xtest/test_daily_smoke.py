# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
from pathlib import Path
import re
import unittest

import run_cases


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'tests/xtest/daily-minimal-csv'
WORKFLOW = ROOT / '.github/workflows/m890p-daily-smoke.yml'


class DailySmokeTests(unittest.TestCase):
    def test_seven_supported_single_node_suites_are_selected(self):
        selected = run_cases.load_inputs(INPUT, 'standard', nodes=1)
        self.assertEqual(len(selected['cases']), 7)
        self.assertEqual(len({(row['plan'], row['suite']) for row in selected['cases']}), 7)
        cluster = run_cases.load_inputs(INPUT, 'standard', nodes=2)
        self.assertEqual(len(cluster['cases']), 6)
        self.assertEqual(len({(row['plan'], row['suite']) for row in cluster['cases']}), 6)
        self.assertTrue(all(row['suite_spec']['nodes'] == 2 for row in cluster['cases']))
        register = next(row for row in cluster['cases']
                        if row['suite'] == 'register_test_shm_multi_nodes2_suite')
        self.assertIn('--ranks 16', register['cmd'])
        self.assertIn('--nprocs 8', register['cmd'])
        self.assertIn('--nnodes 2', register['cmd'])
        self.assertIn('--hostfile {hostfile}', register['cmd'])
        copyengine = next(row for row in cluster['cases']
                          if row['suite'] == 'copyengine_perf_shm_multi_nodes2_suite')
        self.assertIn('-np 16', copyengine['cmd'])
        self.assertIn('-npernode 8', copyengine['cmd'])
        self.assertIn('--hostfile {hostfile}', copyengine['cmd'])
        direct_multi = [row for row in cluster['cases'] if row['cmd'].startswith('mpirun ')]
        self.assertEqual(len(direct_multi), 5)
        for row in direct_multi:
            self.assertIn('-np 16', row['cmd'])
            self.assertIn('-npernode 8', row['cmd'])
            self.assertIn('--hostfile {hostfile}', row['cmd'])
        sanity = next(row for row in selected['cases'] if row['suite'] == 'sanity_m890p_register_suite')
        self.assertIn('--register 1', sanity['cmd'])
        self.assertEqual(sum(row['check'] == 'performance' for row in selected['cases']), 1)
        self.assertEqual(sum(row['check'] == 'exit_code' for row in selected['cases']), 6)
        self.assertTrue(all(row['suite_spec']['nodes'] == 1 for row in selected['cases']))
        self.assertEqual({row['suite_spec']['timeout'] for row in selected['cases']}, {300, 3600, 7200})

    def test_daily_commands_use_only_documented_runtime_bindings(self):
        with (INPUT / 'standard.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            self.assertNotIn('bond1', row['cmd'])
            self.assertNotIn('$PPU_HOME', row['cmd'])
            self.assertNotIn('/home/image', row['cmd'])
            self.assertNotIn('unit_test.py', row['cmd'])
        self.assertEqual(sum('{python_env}' in row['cmd'] for row in rows), 1)
        self.assertEqual(sum(row['cmd'].startswith('python3 run_test.py') for row in rows), 3)
        single_node_rows = run_cases.load_inputs(INPUT, 'standard', nodes=1)['cases']
        self.assertEqual(sum(row['cmd'].startswith('mpirun ') for row in single_node_rows), 4)
        self.assertEqual(sum(row['cmd'].startswith('mpirun ') for row in rows), 9)

    def test_daily_metadata_and_empty_notsm_are_explicit(self):
        metadata = json.loads((INPUT / 'export.json').read_text())
        self.assertEqual(metadata['xtest_version'], '0.12.0')
        self.assertTrue(re.fullmatch(r'[0-9a-f]{40}', metadata['source_revision']))
        with (INPUT / 'notsm.csv').open(newline='') as stream:
            self.assertEqual(list(csv.DictReader(stream)), [])
        with self.assertRaisesRegex(ValueError, 'no Cases'):
            run_cases.load_inputs(INPUT, 'notsm', nodes=1)

    def test_daily_workflow_has_matching_gate_and_cleanup(self):
        text = WORKFLOW.read_text()
        self.assertIn('branches: [ci/m890p-daily-csv]', text)
        self.assertIn('tests/xtest/daily-minimal-csv', text)
        self.assertIn("expected = {'standard': 7, 'cluster': 6}", text)
        self.assertIn('len(metrics) != 1', text)
        self.assertIn('artifact_cache.py cleanup', text)
        self.assertIn('NEEDS: ${{ toJSON(needs) }}', text)
        self.assertIn('retention-days: 3', text)
        self.assertIn('name: m890p-daily-build-standard', text)
        self.assertIn('name: m890p-daily-build-${{ matrix.profile }}', text)
        self.assertIn('Two-node master-only daily smoke', text)
        self.assertIn('/sys/class/infiniband:/sys/class/infiniband', text)
        self.assertIn('gid_attrs/types', text)
        self.assertNotIn('human-review', text)
        self.assertNotIn('build-notsm', text)
        self.assertNotIn("expected = {('standard', 1): 71", text)


if __name__ == '__main__':
    unittest.main()
