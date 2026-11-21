# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
from pathlib import Path
import re
import tempfile
import unittest

import generate_cases
import group_cases
import run_cases


ROOT = Path(__file__).resolve().parents[2]
DEFINITIONS = ROOT / 'tests/xtest/test_lists'
ALL_FAMILIES = (
    "op_perf_shm", "postsil_daily_shm", "pccl_fun_rdma_shm", "register_test_shm",
    "sanity_m890p_register", "sym_test_shm", "f8_test_shm", "device_api_shm",
    "local_register_perf_shm", "symetric_perf_shm", "copyengine_perf_shm",
    "op_perf_shm_2_rack", "op_perf_rdma_shm", "commabort_test_shm", "postsil_unit_test_shm",
)
FULL_INPUT = None
SMOKE_INPUT = None
SOURCE_COVERAGE = DEFINITIONS / 'source-coverage.json'
FULL_WORKFLOW = ROOT / '.github/workflows/pccl-daily-full.yml'
BUILD_SCRIPT = ROOT / '.github/workflows/scripts/build.sh'
RETRY_ACTION = ROOT / '.github/actions/download-artifact-retry/action.yml'


def generate_input(output, profile, families):
    bundles = [generate_cases.generate_bundle(
        DEFINITIONS / f'{family}.json', DEFINITIONS / 'suites.json',
        '1' * 40, '2' * 40, profile) for family in families]
    bundle = generate_cases.combine_bundles(bundles) if len(bundles) > 1 else bundles[0]
    generate_cases.write_bundle(bundle, output, DEFINITIONS / 'envs')


class DailySmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        global FULL_INPUT, SMOKE_INPUT
        FULL_INPUT = root / 'full'
        SMOKE_INPUT = root / 'smoke'
        generate_input(FULL_INPUT, 'full', ALL_FAMILIES)
        generate_input(SMOKE_INPUT, 'smoke', ('op_perf_shm', 'postsil_daily_shm', 'postsil_unit_test_shm'))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_full_daily_commands_cover_expected_suites_and_resources(self):
        selected = run_cases.load_inputs(FULL_INPUT, 'standard', nodes=1)
        self.assertEqual(len(selected['cases']), 5859)
        self.assertEqual(len({(row['plan'], row['suite']) for row in selected['cases']}), 12)
        cluster = run_cases.load_inputs(FULL_INPUT, 'standard', nodes=2)
        self.assertEqual(len(cluster['cases']), 1924)
        self.assertEqual(len({(row['plan'], row['suite']) for row in cluster['cases']}), 10)
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
        self.assertEqual(len(direct_multi), 92)
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
        for suite in ('device_api_perf_shm_single_node_single_process_suite',
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
        self.assertEqual(sum(row['check'] == 'performance' for row in selected['cases']), 45)
        self.assertEqual(sum(row['check'] == 'exit_code' for row in selected['cases']), 5814)
        self.assertTrue(all(row['suite_spec']['nodes'] == 1 for row in selected['cases']))
        self.assertEqual({row['suite_spec']['timeout'] for row in selected['cases']}, {300, 3600, 7200})

    def test_daily_commands_use_only_documented_runtime_bindings(self):
        with (FULL_INPUT / 'standard.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            self.assertNotIn('bond1', row['cmd'])
            self.assertNotIn('$PPU_HOME', row['cmd'])
            self.assertNotIn('/home/image', row['cmd'])
            self.assertNotIn('unit_test.py', row['cmd'])
        self.assertEqual(sum('{python_env}' in row['cmd'] for row in rows), 8028)
        self.assertEqual(sum(row['cmd'].startswith('python3 run_test.py') for row in rows), 510)
        single_node_rows = run_cases.load_inputs(FULL_INPUT, 'standard', nodes=1)['cases']
        self.assertEqual(sum(row['cmd'].startswith('mpirun ') for row in single_node_rows), 148)
        self.assertEqual(sum(row['cmd'].startswith('mpirun ') for row in rows), 335)

    def test_daily_metadata_and_empty_notsm_are_explicit(self):
        metadata = json.loads((SMOKE_INPUT / 'export.json').read_text())
        self.assertEqual(metadata['xtest_version'], 'not-used')
        self.assertTrue(re.fullmatch(r'[0-9a-f]{40}', metadata['source_revision']))
        with (SMOKE_INPUT / 'notsm.csv').open(newline='') as stream:
            self.assertEqual(list(csv.DictReader(stream)), [])
        with self.assertRaisesRegex(ValueError, 'no Cases'):
            run_cases.load_inputs(SMOKE_INPUT, 'notsm', nodes=1)

    def test_full_export_is_runner_scoped_and_available_as_an_explicit_dataset(self):
        coverage = json.loads(SOURCE_COVERAGE.read_text())
        self.assertEqual(coverage['selected_cases'], 10193)
        self.assertEqual(coverage['skipped_cases'], 2062)
        self.assertEqual(len(coverage['selected_suites']), 35)
        self.assertEqual(len(coverage['skipped_suites']), 10)
        skipped = {record['suite']: record['reasons'] for record in coverage['skipped_suites']}
        self.assertIn('outside PCCL scope', skipped['sailucx_test_shm_single_node_suite'][0])
        self.assertIn('outside PCCL scope', skipped['sailbandwidth_perf_shm_multi_nodes2_suite'][0])
        self.assertIn('PCCL composite_test binary', skipped['composite_test_shm_single_node_suite'][0])
        expected = {1: 5859, 2: 1924, 4: 1691, 8: 710, 16: 9}
        for nodes, count in expected.items():
            selected = run_cases.load_inputs(FULL_INPUT, 'standard', nodes=nodes)
            self.assertEqual(len(selected['cases']), count)
        with (FULL_INPUT / 'standard.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 10193)
        self.assertEqual(sum('pcclCommAbortTest' in row['cmd'] for row in rows), 16)
        self.assertFalse((ROOT / 'tests/xtest/daily-full-csv').exists())
        self.assertFalse((ROOT / 'tests/xtest/daily-minimal-csv').exists())
        for row in rows:
            self.assertNotIn('no_tsm', row['plan'])
            self.assertNotIn('no_tsm', row['suite'])
            for token in ('unit_test.py', 'composite_test', 'sailbandwidth',
                          'ucx_perftest', 'p2pBandwidthLatencyTest'):
                self.assertNotIn(token, row['cmd'])

    def test_daily_resource_policy_excludes_only_16_nodes_exactly(self):
        plan = group_cases.build_plan(FULL_INPUT, 500, allowed_nodes=(1, 2, 4, 8))
        self.assertEqual(sum(len(group['cases']) for group in plan['groups']), 10184)
        self.assertEqual(len(plan['groups']), 28)
        self.assertEqual({nodes: sum(len(group['cases']) for group in plan['groups']
                                     if group['nodes'] == nodes)
                          for nodes in (1, 2, 4, 8)},
                         {1: 5859, 2: 1924, 4: 1691, 8: 710})
        self.assertEqual([(record['nodes'], record['case_count']) for record in plan['exclusions']],
                         [(16, 9)])
        friday = group_cases.build_plan(FULL_INPUT, 500, allowed_nodes=(1, 2, 4, 8, 16))
        self.assertEqual(sum(len(group['cases']) for group in friday['groups']), 10193)
        self.assertEqual(len(friday['groups']), 29)
        self.assertEqual(friday['exclusions'], [])
        self.assertEqual(next(group for group in friday['groups'] if group['nodes'] == 16)['rack_policy'],
                         'spread')
        cross_rack = {group['nodes']: (group['rack_policy'], group['rack_count'])
                      for group in plan['groups'] if group['cross_rack']}
        self.assertEqual(cross_rack, {2: ('spread', 2), 4: ('spread', 2), 8: ('spread', 2)})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'groups'
            group_cases.write_plan(plan, output, FULL_INPUT)
            coverage = json.loads((output / 'execution-coverage.json').read_text())
        selected = {(case['plan'], case['suite'], case['name'])
                    for group in plan['groups'] for case in group['cases']}
        excluded = {tuple(key) for key in sum((record['case_keys'] for record in plan['exclusions']), [])}
        self.assertEqual(coverage['total_cases'], 10193)
        self.assertEqual(coverage['excluded_cases'], 9)
        self.assertEqual(len(selected | excluded), 10193)
        self.assertTrue(selected.isdisjoint(excluded))
        eight_nodes = group_cases.build_plan(FULL_INPUT, 500, nodes=8, allowed_nodes=(1, 2, 4, 8))
        self.assertEqual(sum(len(group['cases']) for group in eight_nodes['groups']), 710)
        self.assertEqual(len(eight_nodes['groups']), 3)
        with self.assertRaisesRegex(ValueError, 'disabled by daily resource policy'):
            group_cases.build_plan(FULL_INPUT, 500, nodes=16, allowed_nodes=(1, 2, 4, 8))
        with self.assertRaisesRegex(ValueError, r'allowed_nodes.*61'):
            group_cases.build_plan(FULL_INPUT, 500, allowed_nodes=(1, 2, 4, 8, 61))
        oversized = group_cases.build_plan(FULL_INPUT, 25, allowed_nodes=(1, 2, 4, 8))
        self.assertGreater(len(oversized['groups']), 256)

    def test_native_unit_binaries_are_built_into_the_existing_artifact(self):
        text = BUILD_SCRIPT.read_text()
        make_command = text.split('make -C ${source_home}/pccl/tests/unit_test/src', 1)[1].split(
            'cp ${NATIVE_UNIT_BUILD}', 1)[0]
        copy_command = 'cp ${NATIVE_UNIT_BUILD}' + text.split(
            'cp ${NATIVE_UNIT_BUILD}', 1)[1].split('\n\necho', 1)[0]
        for binary in ('collPcclUnitTest', 'hggcGraphUnitTest', 'pcclCommAbortTest'):
            with self.subTest(binary=binary):
                self.assertIn(f'${{NATIVE_UNIT_BUILD}}/{binary}', make_command)
                self.assertIn(f'${{NATIVE_UNIT_BUILD}}/{binary}', copy_command)
        self.assertIn('PPU_ARCHS=ppu_10,ppu_15 MPI=1', text)
        self.assertIn('${source_home}/pccl-tests/build/', copy_command)
        self.assertGreater(text.index('Build PCCL native unit tests'), text.index('Build pccl-tests'))
        workflow = FULL_WORKFLOW.read_text()
        for family in ('postsil_unit_test_810e', 'all_gather_unit_test_810e',
                       'postsil_unit_test_810e_smoke'):
            self.assertIn(f'--definition tests/xtest/test_lists/{family}.json', workflow)

    def test_full_workflow_is_grouped_daily_execution_not_a_pr_gate(self):
        text = FULL_WORKFLOW.read_text()
        self.assertIn('workflow_dispatch:', text)
        self.assertIn('paths: [.github/workflows/pccl-daily-full.yml]', text)
        self.assertIn('options: [smoke, full]', text)
        self.assertIn('description: Optional node-count debug filter', text)
        self.assertIn('options: [all, m890p, oam810e_eic]', text)
        self.assertIn("PLATFORM: ${{ inputs.platform || 'all' }}", text)
        self.assertIn("list(bundles.values()) if platform == 'all' else [bundles[platform]]", text)
        self.assertIn('description: Optional debug allocation override; Case rank/layout definitions are unchanged', text)
        self.assertNotIn('generated-smoke', text)
        self.assertNotIn('generated-full', text)
        self.assertNotIn('daily-full-csv', text)
        self.assertNotIn('daily-minimal-csv', text)
        self.assertIn('default: smoke', text)
        self.assertIn('default: 500', text)
        self.assertIn("ALLOWED_NODES: ${{ inputs.allowed_nodes || (github.event.schedule == '0 14 * * 5' "
                      "&& '1,2,4,8,16' || '1,2,4,8') }}", text)
        self.assertIn('VALIDATION_NODE_SELECTOR: board-type=ZW-M890P,nic-type=cx8', text)
        self.assertIn('execution-coverage.json', text)
        self.assertIn('Unexpected or out-of-policy result groups', text)
        self.assertIn('cancel-in-progress: false', text)
        self.assertIn('group_cases.py', text)
        self.assertIn('name: m890p-daily-case-groups', text)
        self.assertIn('groups_sha256: ${{ steps.selection.outputs.groups_sha256 }}', text)
        self.assertIn('test_case_group_cache.py', text)
        self.assertIn('matrix: ${{ fromJSON(needs.prepare.outputs.groups) }}', text)
        self.assertIn("'cross_rack', 'rack_policy', 'rack_count', 'case_count'", text)
        self.assertIn("assert {group_cases._rack_policy(case) for case in selected['cases']}", text)
        self.assertNotIn('max-parallel:', text)
        self.assertIn('fail-fast: false', text)
        self.assertIn('nnodes: ${{ matrix.nodes }}', text)
        self.assertIn('nproc_per_node: ${{ inputs.cards_per_node || matrix.cards_per_node }}', text)
        self.assertEqual(text.count('--input "$CASE_INPUT/$GROUP_ID"'), 3)
        self.assertEqual(text.count('case_group_cache.py stage'), 3)
        self.assertIn('case_group_cache.py publish', text)
        self.assertIn('cache/case-groups/${{ env.EXPECTED_GROUPS_SHA }}', text)
        self.assertIn('Restore the NAS build cache', text)
        self.assertEqual(text.count('--build-record "$CACHE_DIR/output_standard.build.json"'), 3)
        self.assertIn('--profile "$PROFILE"', text)
        self.assertIn('results/full/$GROUP_ID', text)
        self.assertIn('shared-standard/$GROUP_ID', text)
        self.assertNotIn('shared-standard/${{ env.EXPECTED_SHA }}/$GROUP_ID', text)
        self.assertIn('rm -rf "$SHARED_ROOT"', text)
        self.assertNotIn('SHARED_PARENT=', text)
        self.assertNotIn('SHARED_BASE=', text)
        self.assertNotIn('rmdir "$SHARED_PARENT"', text)
        self.assertIn('artifact_cache.py cleanup', text)
        self.assertIn('name: Publish the standard NAS cache', text)
        self.assertLess(text.index('name: Publish the standard NAS cache'),
                        text.index('name: m890p-daily-full-build-standard'))
        self.assertIn('cache/standard/${{ steps.identity.outputs.artifact_sha256 }}', text)
        self.assertEqual(text.count('artifact_cache.py probe'), 4)
        self.assertNotIn('chmod -R a+rwX "$CACHE_DIR"', text)
        self.assertNotIn('chmod -R a+rwX "$BUILD_CACHE"', text)
        self.assertIn('--preserve-cache', text)
        self.assertIn('id: results-upload', text)
        self.assertIn('/results/**/raw.log', text)
        self.assertIn('/results/**/dmesg*.log', text)
        self.assertIn('if-no-files-found: warn', text)
        self.assertIn('id: performance-upload', text)
        self.assertIn('steps.results-upload.outcome', text)
        self.assertIn('steps.performance-upload.outcome', text)
        self.assertIn('[ "$PRESERVE" = 1 ]', text)
        self.assertNotIn('[[ "$PRESERVE" == 1 ]]', text)
        self.assertIn("'groups.json').read_text())", text)
        self.assertIn('daily_reconcile.py', text)
        self.assertIn('Daily grouped result summary and NAS cleanup', text)
        self.assertLess(text.index('SHARED_ROOT=/mnt/wl_nas'),
                        text.index('CASE_INPUT="$SHARED_ROOT/case-input"'))
        self.assertIn('GitHub Actions matrix limit of 256', text)
        self.assertIn(
            'timeout-minutes: ${{ fromJSON(needs.prepare.outputs.execution_job_timeout_minutes) }}',
            text)
        self.assertIn(
            'execution_timeout_minutes: ${{ steps.selection.outputs.execution_timeout_minutes }}',
            text)
        self.assertIn(
            'execution_job_timeout_minutes: ${{ steps.selection.outputs.execution_job_timeout_minutes }}',
            text)
        self.assertIn('execution_timeout_minutes={execution_timeout_minutes}', text)
        self.assertIn('execution_job_timeout_minutes={execution_timeout_minutes + 30}', text)
        self.assertEqual(text.count('timeout_minutes: ${{ env.EXECUTION_TIMEOUT_MINUTES }}'), 3)
        self.assertNotIn('timeout_minutes: 720', text)
        self.assertIn('Execution timeout must be a positive integer', text)
        self.assertIn('retention-days: 90', text)
        self.assertIn('if results_root.is_dir() else set()', text)
        single = text.split('name: Run the automatic group on one node', 1)[1].split(
            'name: Run the automatic multi-node group on master', 1)[0]
        self.assertIn('host_network: true', single)
        self.assertNotIn('setup_command:', single)
        self.assertIn("raise SystemExit('No non-zero RoCE GID is visible inside the worker')", single)
        self.assertIn('alixpu_peermem is not loaded; DMA-BUF support is still possible', single)
        selected = run_cases.load_inputs(FULL_INPUT, 'standard', nodes=1,
                                        suite='device_api_perf_shm_single_node_single_process_suite')
        self.assertEqual(len(selected['cases']), 26)
        self.assertNotIn('pull_request:', text)
        self.assertNotIn('pull-requests: write', text)
        self.assertNotIn('statuses: write', text)
        self.assertNotIn('PCCL Tests Gate', text)
        self.assertNotIn('build-notsm', text)
        self.assertNotIn('default: device_api_perf_shm_single_node_single_process_suite', text)
        self.assertNotIn('--profile standard --nodes "$NODES" --suite "$SUITE"', text)

    def test_cross_rack_groups_use_the_validation_action_without_duplicate_execution(self):
        text = FULL_WORKFLOW.read_text()
        ordinary = text.split('name: Run the automatic multi-node group on master', 1)[1].split(
            'name: Run the automatic cross-rack group on master', 1)[0]
        cross = text.split('name: Run the automatic cross-rack group on master', 1)[1].split(
            '\n  summary:', 1)[0]
        action = 'uses: t-head/ci-infra/cluster-run@main'
        self.assertIn(action, ordinary)
        self.assertIn(action, cross)
        self.assertIn("if: env.NODES != '1' && env.CROSS_RACK != 'true'", ordinary)
        self.assertIn("if: env.NODES != '1' && env.CROSS_RACK == 'true'", cross)
        self.assertIn('rack_policy: ${{ matrix.rack_policy }}', ordinary)
        self.assertIn('rack_policy: ${{ matrix.rack_policy }}', cross)
        self.assertIn('rack_count: ${{ matrix.rack_count }}', cross)
        self.assertIn('node_selector: ${{ matrix.node_selector }}', cross)
        for section in (ordinary, cross):
            self.assertNotIn("cluster-ssh \"$host\" 'hostname'", section)
            self.assertNotIn('mpirun --allow-run-as-root', section)
            self.assertIn('test "$(wc -l < "$CLUSTER_HOSTS_FILE")" -eq "$NODES"', section)
        self.assertNotIn('icn-rack=', text)
        self.assertEqual(text.count(action), 2)
        self.assertNotIn('t-head/ci-infra/cluster-run@feat/rack-node-allocation', text)

    def test_schedule_runs_full_daily_at_2200_beijing_time(self):
        text = FULL_WORKFLOW.read_text()
        self.assertIn("schedule:\n    # GitHub cron is UTC: 14:00 UTC is 22:00 Asia/Shanghai (UTC+8).\n"
                        "    - cron: '0 14 * * 1-4'\n    - cron: '0 14 * * 5'", text)
        self.assertNotIn("- cron: '0 14 * * *'", text)
        self.assertIn("github.event.schedule == '0 14 * * 5'", text)
        self.assertIn("'1,2,4,8,16'", text)
        self.assertIn('allowed_nodes:', text)
        self.assertIn('Manual runs default to 1,2,4,8; enter 1,2,4,8,16 to include 16-node cases.', text)
        self.assertIn(
            "DATASET: ${{ inputs.dataset || (github.event_name == 'schedule' && 'full' || 'smoke') }}",
            text)
        self.assertIn('default: smoke', text)
        self.assertIn('default: 500', text)
        self.assertIn('execution_timeout_minutes:', text)
        self.assertIn('Per-group queue and execution timeout in minutes for manual runs', text)
        self.assertIn(
            "REQUESTED_EXECUTION_TIMEOUT_MINUTES: ${{ github.event_name == 'workflow_dispatch' "
            "&& format('{0}', inputs.execution_timeout_minutes) || "
            "(github.event.schedule == '0 14 * * 5' && 2160 || 720) }}", text)
        self.assertIn(
            'EXECUTION_TIMEOUT_MINUTES: ${{ needs.prepare.outputs.execution_timeout_minutes }}', text)

    def test_all_daily_downloads_use_the_bounded_retry_action(self):
        action = RETRY_ACTION.read_text()
        self.assertEqual(action.count('uses: actions/download-artifact@v4'), 3)
        self.assertEqual(action.count('continue-on-error: true'), 4)
        self.assertIn('PREFLIGHT_ONLY', action)
        self.assertIn('VALIDATOR_PATH', action)
        self.assertNotIn('github.action_path', action)
        self.assertEqual(action.count('validation.outcome'), 4)
        self.assertIn('sleep 15', action)
        self.assertIn('sleep 30', action)
        self.assertIn('shutil.rmtree(target)', action)
        self.assertIn("parent.relative_to(workspace)", action)
        self.assertIn("target = parent / relative.name", action)
        text = FULL_WORKFLOW.read_text()
        self.assertNotIn('uses: actions/download-artifact@v4', text)
        self.assertEqual(text.count('uses: t-head/pccl/.github/actions/download-artifact-retry@'
                                    '42c06876282e700ae0434b4577fedd757b6f2db1'), 7)
        self.assertEqual(text.count('expected-sha256: ${{ env.EXPECTED_SHA }}'), 2)
        self.assertEqual(text.count('sha256-file: output_standard.tar.gz'), 2)


if __name__ == '__main__':
    unittest.main()
