# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
from pathlib import Path
import re
import tempfile
import unittest

import group_cases
import run_cases


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'tests/xtest/daily-minimal-csv'
FULL_INPUT = ROOT / 'tests/xtest/daily-full-csv'
FULL_WORKFLOW = ROOT / '.github/workflows/m890p-daily-full.yml'
BUILD_SCRIPT = ROOT / '.github/workflows/scripts/build.sh'
RETRY_ACTION = ROOT / '.github/actions/download-artifact-retry/action.yml'


class DailySmokeTests(unittest.TestCase):
    def test_eleven_supported_single_node_suites_are_selected(self):
        selected = run_cases.load_inputs(INPUT, 'standard', nodes=1)
        self.assertEqual(len(selected['cases']), 11)
        self.assertEqual(len({(row['plan'], row['suite']) for row in selected['cases']}), 11)
        cluster = run_cases.load_inputs(INPUT, 'standard', nodes=2)
        self.assertEqual(len(cluster['cases']), 8)
        self.assertEqual(len({(row['plan'], row['suite']) for row in cluster['cases']}), 8)
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
        self.assertEqual(len(direct_multi), 7)
        for row in direct_multi:
            if row['suite'] == 'commabort_test_shm_multi_nodes2_suite':
                self.assertIn('-np 4', row['cmd'])
                self.assertIn('-npernode 2', row['cmd'])
            else:
                self.assertIn('-np 16', row['cmd'])
                self.assertIn('-npernode 8', row['cmd'])
            self.assertIn('--hostfile {hostfile}', row['cmd'])
        rdma = next(row for row in cluster['cases']
                    if row['suite'] == 'op_perf_rdma_shm_multi_nodes2_suite')
        self.assertIn('NCCL_MNNVL_ENABLE=0', rdma['cmd'])
        self.assertNotIn('orte_keep_fqdn_hostnames', rdma['cmd'])
        self.assertNotIn('NCCL_SOCKET_IFNAME', rdma['cmd'])
        for suite in ('device_api_shm_single_node_single_process_suite',
                      'f8_test_shm_single_node_suite', 'sym_test_shm_single_node_suite'):
            self.assertIn(suite, {row['suite'] for row in selected['cases']})
        commabort = next(row for row in cluster['cases']
                         if row['suite'] == 'commabort_test_shm_multi_nodes2_suite')
        self.assertIn('-np 4 -npernode 2', commabort['cmd'])
        self.assertIn('pcclCommAbortTest -b 1g -a 0', commabort['cmd'])
        for suite in ('local_register_perf_shm_multi_nodes2_suite',
                      'symetric_perf_shm_multi_nodes2_suite'):
            case = next(row for row in cluster['cases'] if row['suite'] == suite)
            self.assertIn('NCCL_IB_DISABLE=1', case['cmd'])
            self.assertIn('NCCL_MNNVL_ENABLE=1', case['cmd'])
            self.assertIn('-mca btl ^openib', case['cmd'])
        sanity = next(row for row in selected['cases'] if row['suite'] == 'sanity_m890p_register_suite')
        self.assertIn('--register 1', sanity['cmd'])
        self.assertEqual(sum(row['check'] == 'performance' for row in selected['cases']), 1)
        self.assertEqual(sum(row['check'] == 'exit_code' for row in selected['cases']), 10)
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
        self.assertEqual(sum(row['cmd'].startswith('python3 run_test.py') for row in rows), 5)
        single_node_rows = run_cases.load_inputs(INPUT, 'standard', nodes=1)['cases']
        self.assertEqual(sum(row['cmd'].startswith('mpirun ') for row in single_node_rows), 5)
        self.assertEqual(sum(row['cmd'].startswith('mpirun ') for row in rows), 12)

    def test_daily_metadata_and_empty_notsm_are_explicit(self):
        metadata = json.loads((INPUT / 'export.json').read_text())
        self.assertEqual(metadata['xtest_version'], '0.12.0')
        self.assertTrue(re.fullmatch(r'[0-9a-f]{40}', metadata['source_revision']))
        with (INPUT / 'notsm.csv').open(newline='') as stream:
            self.assertEqual(list(csv.DictReader(stream)), [])
        with self.assertRaisesRegex(ValueError, 'no Cases'):
            run_cases.load_inputs(INPUT, 'notsm', nodes=1)

    def test_full_export_is_runner_scoped_and_available_as_an_explicit_dataset(self):
        coverage = json.loads((FULL_INPUT / 'coverage.json').read_text())
        self.assertEqual(coverage['selected_cases'], 8899)
        self.assertEqual(coverage['skipped_cases'], 3391)
        self.assertEqual(len(coverage['selected_suites']), 34)
        self.assertEqual(len(coverage['skipped_suites']), 10)
        skipped = {record['suite']: record['reasons'] for record in coverage['skipped_suites']}
        self.assertIn('outside PCCL scope', skipped['sailucx_test_shm_single_node_suite'][0])
        self.assertIn('outside PCCL scope', skipped['sailbandwidth_perf_shm_multi_nodes2_suite'][0])
        self.assertIn('PCCL composite_test binary', skipped['composite_test_shm_single_node_suite'][0])
        expected = {1: 4565, 2: 1924, 4: 1691, 8: 710, 16: 9}
        for nodes, count in expected.items():
            selected = run_cases.load_inputs(FULL_INPUT, 'standard', nodes=nodes)
            self.assertEqual(len(selected['cases']), count)
        with (FULL_INPUT / 'standard.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 8899)
        self.assertEqual(sum('pcclCommAbortTest' in row['cmd'] for row in rows), 16)
        self.assertIn('!tests/xtest/daily-full-csv/*.csv', (ROOT / '.gitignore').read_text())
        for row in rows:
            self.assertNotIn('no_tsm', row['plan'])
            self.assertNotIn('no_tsm', row['suite'])
            for token in ('unit_test.py', 'composite_test', 'sailbandwidth',
                          'ucx_perftest', 'p2pBandwidthLatencyTest'):
                self.assertNotIn(token, row['cmd'])

    def test_daily_resource_policy_excludes_8_and_16_nodes_exactly(self):
        plan = group_cases.build_plan(FULL_INPUT, 500, allowed_nodes=(1, 2, 4))
        self.assertEqual(sum(len(group['cases']) for group in plan['groups']), 8180)
        self.assertEqual(len(plan['groups']), 20)
        self.assertEqual({nodes: sum(len(group['cases']) for group in plan['groups']
                                     if group['nodes'] == nodes)
                          for nodes in (1, 2, 4)}, {1: 4565, 2: 1924, 4: 1691})
        self.assertEqual([(record['nodes'], record['case_count']) for record in plan['exclusions']],
                         [(8, 710), (16, 9)])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'groups'
            group_cases.write_plan(plan, output, FULL_INPUT)
            coverage = json.loads((output / 'execution-coverage.json').read_text())
        selected = {(case['plan'], case['suite'], case['name'])
                    for group in plan['groups'] for case in group['cases']}
        excluded = {tuple(key) for key in sum((record['case_keys'] for record in plan['exclusions']), [])}
        self.assertEqual(coverage['total_cases'], 8899)
        self.assertEqual(coverage['excluded_cases'], 719)
        self.assertEqual(len(selected | excluded), 8899)
        self.assertTrue(selected.isdisjoint(excluded))
        with self.assertRaisesRegex(ValueError, 'disabled by daily resource policy'):
            group_cases.build_plan(FULL_INPUT, 500, nodes=8, allowed_nodes=(1, 2, 4))
        with self.assertRaisesRegex(ValueError, 'disabled by daily resource policy'):
            group_cases.build_plan(FULL_INPUT, 500, nodes=16, allowed_nodes=(1, 2, 4))
        oversized = group_cases.build_plan(FULL_INPUT, 25, allowed_nodes=(1, 2, 4))
        self.assertGreater(len(oversized['groups']), 256)

    def test_commabort_binary_is_built_into_the_existing_artifact(self):
        text = BUILD_SCRIPT.read_text()
        self.assertIn('${COMM_ABORT_BUILD}/pcclCommAbortTest', text)
        self.assertIn('PPU_ARCHS=ppu_10,ppu_15 MPI=1', text)
        self.assertIn('cp ${COMM_ABORT_BUILD}/pcclCommAbortTest ${source_home}/pccl-tests/build/', text)
        self.assertGreater(text.index('Build PCCL CommAbort test'), text.index('Build pccl-tests'))

    def test_full_workflow_is_grouped_daily_execution_not_a_pr_gate(self):
        text = FULL_WORKFLOW.read_text()
        self.assertIn('workflow_dispatch:', text)
        self.assertIn('paths: [.github/workflows/m890p-daily-full.yml]', text)
        self.assertIn('options: [smoke, full]', text)
        self.assertIn('default: smoke', text)
        self.assertIn('default: 500', text)
        self.assertIn("DAILY_ALLOWED_NODES: '1,2,4'", text)
        self.assertIn('execution-coverage.json', text)
        self.assertIn('Unexpected or out-of-policy result groups', text)
        self.assertIn('cancel-in-progress: false', text)
        self.assertIn('group_cases.py', text)
        self.assertIn('name: m890p-daily-case-groups', text)
        self.assertIn('groups_sha256: ${{ steps.selection.outputs.groups_sha256 }}', text)
        self.assertIn('test_case_group_cache.py', text)
        self.assertIn('matrix: ${{ fromJSON(needs.prepare.outputs.groups) }}', text)
        self.assertIn('max-parallel: 2', text)
        self.assertIn('fail-fast: false', text)
        self.assertIn('nnodes: ${{ matrix.nodes }}', text)
        self.assertIn('nproc_per_node: ${{ matrix.cards_per_node }}', text)
        self.assertEqual(text.count('--input "$CASE_INPUT/$GROUP_ID"'), 2)
        self.assertEqual(text.count('case_group_cache.py stage'), 2)
        self.assertIn('case_group_cache.py publish', text)
        self.assertIn('cache/case-groups/${{ env.EXPECTED_GROUPS_SHA }}', text)
        self.assertIn('Restore the NAS build cache', text)
        self.assertEqual(text.count('--build-record "$CACHE_DIR/output_standard.build.json"'), 2)
        self.assertIn('--profile "$PROFILE"', text)
        self.assertIn('results/full/$GROUP_ID', text)
        self.assertIn('shared-standard/$GROUP_ID', text)
        self.assertNotIn('shared-standard/${{ env.EXPECTED_SHA }}/$GROUP_ID', text)
        self.assertIn('rm -rf "$SHARED_ROOT"', text)
        self.assertNotIn('SHARED_PARENT=', text)
        self.assertNotIn('SHARED_BASE=', text)
        self.assertNotIn('rmdir "$SHARED_PARENT"', text)
        self.assertIn('artifact_cache.py cleanup', text)
        self.assertIn("'groups.json').read_text())", text)
        self.assertIn('daily_reconcile.py', text)
        self.assertIn('Daily grouped result summary and NAS cleanup', text)
        self.assertLess(text.index('SHARED_ROOT=/mnt/wl_nas'),
                        text.index('CASE_INPUT="$SHARED_ROOT/case-input"'))
        self.assertIn('GitHub Actions matrix limit of 256', text)
        self.assertIn('timeout-minutes: 750', text)
        self.assertNotIn('retention-days: 3', text)
        self.assertIn('if results_root.is_dir() else set()', text)
        single = text.split('name: Run the automatic group on one node', 1)[1].split(
            'name: Run the automatic multi-node group on master', 1)[0]
        self.assertIn('host_network: true', single)
        self.assertNotIn('setup_command:', single)
        self.assertIn("raise SystemExit('No non-zero RoCE GID is visible inside the worker')", single)
        self.assertIn('alixpu_peermem is not loaded; DMA-BUF support is still possible', single)
        selected = run_cases.load_inputs(FULL_INPUT, 'standard', nodes=1,
                                        suite='device_api_shm_single_node_single_process_suite')
        self.assertEqual(len(selected['cases']), 26)
        self.assertNotIn('pull_request:', text)
        self.assertNotIn('pull-requests: write', text)
        self.assertNotIn('statuses: write', text)
        self.assertNotIn('PCCL Tests Gate', text)
        self.assertNotIn('build-notsm', text)
        self.assertNotIn('default: device_api_shm_single_node_single_process_suite', text)
        self.assertNotIn('--profile standard --nodes "$NODES" --suite "$SUITE"', text)

    def test_schedule_runs_full_daily_at_2200_beijing_time(self):
        text = FULL_WORKFLOW.read_text()
        self.assertIn("schedule:\n    # GitHub cron is UTC: 14:00 UTC is 22:00 Asia/Shanghai (UTC+8).\n"
                        "    - cron: '0 14 * * *'", text)
        self.assertIn(
            "DATASET: ${{ inputs.dataset || (github.event_name == 'schedule' && 'full' || 'smoke') }}",
            text)
        self.assertIn('default: smoke', text)
        self.assertIn('default: 500', text)

    def test_all_daily_downloads_use_the_bounded_retry_action(self):
        action = RETRY_ACTION.read_text()
        self.assertEqual(action.count('uses: actions/download-artifact@v4'), 3)
        self.assertEqual(action.count('continue-on-error: true'), 2)
        self.assertIn('sleep 15', action)
        self.assertIn('sleep 30', action)
        self.assertIn('shutil.rmtree(target)', action)
        self.assertIn("parent.relative_to(workspace)", action)
        self.assertIn("target = parent / relative.name", action)
        text = FULL_WORKFLOW.read_text()
        self.assertNotIn('uses: actions/download-artifact@v4', text)
        self.assertEqual(text.count('uses: t-head/pccl/.github/actions/download-artifact-retry@'
                                    'c7eb80eadb2ca48976a02d8f0fcc1c020534ce73'), 6)


if __name__ == '__main__':
    unittest.main()
