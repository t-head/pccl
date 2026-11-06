# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap
import unittest

import artifact_cache
import reconcile_results
import run_cases


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / '.github/workflows/source-sanity.yml'
PR_JOBS = ('precheck', 'commit-checks', 'ai-code-review', 'human-review')
REQUIRED_JOBS = ('prepare', 'build-standard', 'build-notsm', 'stage-cache',
                 'single-node', 'two-node', 'results')


class WorkflowContractTests(unittest.TestCase):
    def test_all_artifact_downloads_use_bounded_retries(self):
        action = (ROOT / '.github/actions/download-artifact-retry/action.yml').read_text()
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
        workflow = (ROOT / '.github/workflows/source-sanity.yml').read_text()
        self.assertNotIn('uses: actions/download-artifact@v4', workflow)
        self.assertEqual(workflow.count('uses: t-head/pccl/.github/actions/download-artifact-retry@'
                                        '42c06876282e700ae0434b4577fedd757b6f2db1'), 5)
        self.assertEqual(workflow.count('expected-sha256: ${{ env.EXPECTED_SHA }}'), 3)
        self.assertEqual(workflow.count('sha256-file: output_'), 3)

    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding='utf-8')

    def job(self, name):
        return re.search(rf'^  {re.escape(name)}:\n.*?(?=^  [\w-]+:|\Z)',
                         self.text, re.M | re.S).group()

    def needs(self, event='pull_request'):
        values = {name: {'result': 'success'} for name in PR_JOBS + REQUIRED_JOBS}
        if event != 'pull_request':
            for name in PR_JOBS:
                values[name]['result'] = 'skipped'
        return values

    def test_810e_pr_allocation_uses_two_cards_per_node(self):
        self.assertIn("Path('tests/xtest/minimal-csv-oam810e-eic'), 2)", self.text)
        self.assertEqual(self.text.count('platform: oam810e_eic, cards_per_node: 2'), 3)
        self.assertNotIn('platform: oam810e_eic, cards_per_node: 4', self.text)
        self.assertNotIn('platform: oam810e_eic, cards_per_node: 16', self.text)
        self.assertIn("--cards-per-node '2'", self.text)

    def test_all_platforms_map_allocated_devices_in_daily_and_sanity(self):
        daily = (ROOT / '.github/workflows/pccl-daily-full.yml').read_text(encoding='utf-8')
        self.assertEqual(self.text.count('--map-nvidia-devices'), 2)
        self.assertEqual(daily.count('--map-nvidia-devices'), 3)
        for workflow in (self.text, daily):
            self.assertNotIn("matrix.platform == 'oam810e_eic' && '--map-nvidia-devices'", workflow)

    def results_eligible(self, needs, event='pull_request', repository='t-head/pccl',
                         head_repository='t-head/pccl', job='results'):
        condition = re.search(r'^    if: >-\n((?:      .+\n)+)', self.job(job), re.M).group(1)
        expression = ' '.join(line.strip() for line in condition.splitlines())
        for key, value in (('github.event.pull_request.head.repo.full_name', head_repository),
                           ('github.event_name', event), ('github.repository', repository)):
            expression = expression.replace(key, repr(value))
        expression = re.sub(r'needs\.([\w-]+)\.result',
                            lambda match: repr(needs[match.group(1)]['result']), expression)
        expression = (expression.replace('always()', 'True').replace('!cancelled()', 'True')
                      .replace('&&', ' and ').replace('||', ' or '))

        def evaluate(node):
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.BoolOp):
                operation = all if isinstance(node.op, ast.And) else any
                return operation(evaluate(value) for value in node.values)
            if isinstance(node, ast.Compare) and len(node.ops) == 1:
                left, right = evaluate(node.left), evaluate(node.comparators[0])
                if isinstance(node.ops[0], ast.Eq):
                    return left == right
                if isinstance(node.ops[0], ast.NotEq):
                    return left != right
            self.fail(f'Unexpected workflow condition syntax: {ast.dump(node)}')

        return evaluate(ast.parse(expression, mode='eval').body)

    def run_gate(self, needs, event='pull_request'):
        section = self.job('summary')
        script = textwrap.dedent(section.split("python3 - <<'PY'\n", 1)[1].rsplit('          PY', 1)[0])
        return subprocess.run([sys.executable, '-S', '-B', '-c', script],
                              env={**os.environ, 'NEEDS': json.dumps(needs), 'EVENT_NAME': event},
                              capture_output=True, text=True)

    def test_supported_events_and_review_gates_are_preserved(self):
        for expected in ('pull_request:', "branches: [main, 'release/**']", 'branches: [main]',
                         'pull-requests: write', 'statuses: read',
                         'workflow_dispatch:', 'cancel-in-progress: true',
                         't-head/.github/.github/workflows/user-precheck.yml@main',
                         't-head/.github/.github/workflows/ai-code-review.yml@main',
                         't-head/.github/.github/workflows/human-review.yml@main',
                         'Reject unsupported repository or event',
                         'Require successful results, cleanup and all prerequisite jobs'):
            self.assertIn(expected, self.text)
        self.assertNotIn('github.event.pull_request.head.repo.full_name', self.text)

    def test_artifact_upload_download_retention_and_cleanup_are_connected(self):
        for expected in ('name: pccl-build-standard', 'name: pccl-build-notsm',
                         'name: Stage ${{ matrix.profile }} from artifact or valid NAS cache',
                         't-head/pccl/.github/actions/download-artifact-retry@', 'retention-days: 90',
                         'artifact_cache.py publish', 'artifact_cache.py validate',
                         'artifact_cache.py cleanup', 'trap cleanup_shared_build EXIT'):
            self.assertIn(expected, self.text)
        self.assertLess(self.text.index('name: Stage ${{ matrix.profile }} from artifact or valid NAS cache'),
                        self.text.index('name: Single-node CSV'))
        self.assertLess(self.text.index('name: Reconcile all selected hardware results'),
                        self.text.index('name: Remove run-owned NAS build artifacts'))
        self.assertIn("needs: [precheck, commit-checks, prepare, ai-code-review, human-review", self.text)

    def test_build_publishes_nas_before_backup_and_consumers_probe_without_chmod(self):
        for profile in ('standard', 'notsm'):
            publish = self.text.index(f'name: Publish the {profile} NAS cache')
            upload = self.text.index(f'name: pccl-build-{profile}', publish)
            self.assertLess(publish, upload)
            self.assertIn(f'cache/{profile}/${{{{ steps.identity.outputs.artifact_sha256 }}}}', self.text[publish:upload])
        self.assertEqual(self.text.count('artifact_cache.py probe'), 4)
        self.assertNotIn('chmod -R a+rwX "$CACHE_DIR"', self.text)
        self.assertIn('--preserve-cache', self.text)
        self.assertIn('id: results-upload', self.text)
        upload = self.text[self.text.index('id: results-upload'):]
        upload = upload[:upload.index('- id: cleanup')]
        self.assertIn('if: always()', upload)
        self.assertIn('/results/**/raw.log', upload)
        self.assertIn('/results/**/dmesg*.log', upload)
        self.assertIn('if-no-files-found: warn', upload)
        self.assertIn('RESULTS_UPLOAD_OUTCOME', self.text)
        self.assertIn('[ "$PRESERVE" = 1 ]', self.text)
        self.assertNotIn('[[ "$PRESERVE" == 1 ]]', self.text)
        self.assertIn("names = ('prepare', 'build-standard', 'build-notsm', 'stage-cache',", self.text)

    def test_required_gate_is_preserved_and_support_check_always_runs(self):
        self.assertIn('  summary:\n    name: PCCL Tests Gate\n', self.text)
        summary = self.job('summary')
        support = summary[summary.index('      - id: support'):summary.index('      - name: Require successful')]
        self.assertNotIn('        if:', support)
        self.assertIn('shell: bash', support)
        self.assertIn('if [ "$SUPPORTED" != \'true\' ]', support)
        script = textwrap.dedent(support.split('        run: |\n', 1)[1])
        shells = ['/bin/sh', '/bin/bash']
        if Path('/bin/dash').is_file():
            shells.append('/bin/dash')
        for shell in shells:
            for value, status in (('true', 0), ('false', 1), ('', 1)):
                with self.subTest(shell=shell, supported=value):
                    result = subprocess.run([shell, '-e', '-c', script], env={'SUPPORTED': value},
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, status, result.stderr)
        self.assertNotIn('42478314103', self.text)
        self.assertNotIn('42480089838', self.text)

    def test_final_gate_is_hosted_and_never_executes_repository_code(self):
        summary = self.job('summary')
        self.assertIn('    if: always()', summary)
        self.assertIn('    runs-on: ubuntu-24.04', summary)
        self.assertIn('    permissions: {}', summary)
        for unsafe in ('actions/checkout', 'tests/xtest/', '/wl_nas/', 'k8s-runner', 'container:'):
            self.assertNotIn(unsafe, summary)
        dependencies = re.search(r'    needs: \[(.+)\]', summary).group(1).split(', ')
        self.assertEqual(set(dependencies), set(PR_JOBS + REQUIRED_JOBS))

    def test_internal_results_require_pr_approval_and_supported_events(self):
        self.assertTrue(self.results_eligible(self.needs()))
        for name in PR_JOBS + ('prepare',):
            for status in ('failure', 'cancelled', 'skipped'):
                with self.subTest(job=name, result=status):
                    needs = self.needs()
                    needs[name]['result'] = status
                    self.assertFalse(self.results_eligible(needs))
        self.assertTrue(self.results_eligible(self.needs(), head_repository='someone/fork'))
        self.assertFalse(self.results_eligible(self.needs(), repository='someone/pccl'))
        self.assertFalse(self.results_eligible(self.needs(), event='pull_request_target'))
        for event in ('push', 'workflow_dispatch'):
            self.assertTrue(self.results_eligible(self.needs(event), event))
        for name in ('build-standard', 'single-node', 'two-node'):
            needs = self.needs()
            needs[name]['result'] = 'failure'
            self.assertTrue(self.results_eligible(needs), 'Approved failed runs still need cleanup')
        results = self.job('results')
        self.assertIn("if: always() && steps.checkout.outcome == 'success'", results)

    def test_fork_prs_run_hosted_prechecks_and_require_approval(self):
        for job in ('precheck', 'commit-checks'):
            self.assertTrue(self.results_eligible(self.needs(), job=job))
            self.assertTrue(self.results_eligible(self.needs(), head_repository='someone/fork', job=job))
            self.assertFalse(self.results_eligible(self.needs(), event='workflow_dispatch', job=job))
        for job in ('build-standard', 'build-notsm'):
            self.assertTrue(self.results_eligible(self.needs(), head_repository='someone/fork', job=job))
            needs = self.needs()
            needs['human-review']['result'] = 'failure'
            self.assertFalse(self.results_eligible(needs, head_repository='someone/fork', job=job))

    def test_final_gate_rejects_failed_cancelled_or_skipped_required_jobs(self):
        self.assertEqual(self.run_gate(self.needs()).returncode, 0)
        for name in PR_JOBS + REQUIRED_JOBS:
            for status in ('failure', 'cancelled', 'skipped'):
                with self.subTest(job=name, result=status):
                    needs = self.needs()
                    needs[name]['result'] = status
                    result = self.run_gate(needs)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(f'{name}: {status}', result.stdout)

    def test_final_gate_allows_only_expected_non_pr_review_skips(self):
        for event in ('push', 'workflow_dispatch'):
            with self.subTest(event=event):
                self.assertEqual(self.run_gate(self.needs(event), event).returncode, 0)
                needs = self.needs(event)
                needs['results']['result'] = 'skipped'
                self.assertNotEqual(self.run_gate(needs, event).returncode, 0)

    def test_non_pr_review_skips_do_not_skip_hardware_jobs(self):
        expected = "(github.event_name != 'pull_request' && needs.human-review.result == 'skipped')"
        self.assertEqual(self.text.count(expected), 4)
        single = self.text[self.text.index('  single-node:'):self.text.index('  two-node:')]
        cluster = self.job('two-node')
        for section in (single, cluster):
            self.assertIn(expected, section)
            self.assertIn("needs.stage-cache.result == 'success'", section)

    def test_retried_consumers_restore_a_verified_cache_and_record_build_identity(self):
        single = self.text[self.text.index('  single-node:'):self.text.index('  two-node:')]
        cluster = self.job('two-node')
        for section in (single, cluster):
            self.assertIn('name: Check the NAS cache before hardware allocation', section)
            self.assertIn('name: Restore a cleaned NAS cache from the build artifact', section)
            self.assertIn('artifact_cache.py publish', section)
            self.assertIn('path: build-download', section)
            self.assertIn('--build-record "$CACHE_DIR/output_', section)
            self.assertIn("--pccl-revision '${{ github.sha }}'", section)
            self.assertIn("--expected-sha256 '${{ env.EXPECTED_SHA }}'", section)

    def test_final_reconciliation_uses_exact_case_and_build_identity_checks(self):
        results = self.job('results')
        for expected in ('tests/xtest/reconcile_results.py',
                         '--pccl-revision',
                         '--standard-sha256',
                         '--notsm-sha256'):
            self.assertIn(expected, results)
        prepare = self.text[self.text.index('  prepare:'):self.text.index('  ai-code-review:')]
        for module in ('test_generate_cases.py', 'test_group_cases.py', 'test_daily_smoke.py'):
            self.assertIn(f"-p '{module}'", prepare)
        self.assertIn("test_reconcile_results.py", prepare)
        self.assertIn("test_workflow_contract.py", prepare)
        self.assertIn('reconcile_results.load_selections(', prepare)
        self.assertIn('--cards-per-node', results)

    def test_prepare_runs_every_xtest_regression_module(self):
        prepare = self.text[self.text.index('  prepare:'):self.text.index('  ai-code-review:')]
        modules = sorted(path.name for path in (ROOT / 'tests/xtest').glob('test_*.py'))
        self.assertTrue(modules)
        for module in modules:
            with self.subTest(module=module):
                self.assertIn(f"-p '{module}'", prepare)

    def test_hpn_sdk_defaults_to_installation_and_reaches_every_hardware_step(self):
        daily = (ROOT / '.github/workflows/pccl-daily-full.yml').read_text(encoding='utf-8')
        url = ('https://zone-ci.oss-cn-hangzhou.aliyuncs.com/hpn-sdk/2.1.3/'
               'hpnsdk2.1.3-ubuntu2404-202611121836-x86_64.bin')
        for workflow in (self.text, daily):
            self.assertIn('hpn_sdk_url:', workflow)
            self.assertIn(f'default: {url}', workflow)
            self.assertIn('  HPN_SDK_URL: ' + url, workflow)
            self.assertNotIn('inputs.hpn_sdk_url', workflow)
            self.assertIn('context.payload.inputs?.hpn_sdk_url || process.env.HPN_SDK_URL', workflow)
            self.assertIn('hpn_sdk_url: ${{ steps.sdk-selection.outputs.hpn_sdk_url }}', workflow)
            self.assertEqual(workflow.count('scripts/install_hpn_sdk.sh'), 4)
            self.assertEqual(workflow.count('HPN_SDK_URL:?Selected HPN SDK URL'), 4)
        self.assertIn('--image "$VALIDATION_IMAGE"', daily)
        self.assertIn('--hpn-sdk-url "$HPN_SDK_URL"', daily)

    def test_only_validated_public_sdk_selection_reaches_consumer_jobs(self):
        workflows = ((self.text, ('build-standard', 'build-notsm', 'stage-cache',
                                 'single-node', 'two-node', 'results')),
                     ((ROOT / '.github/workflows/pccl-daily-full.yml').read_text(encoding='utf-8'),
                      ('build-standard', 'stage-cache', 'execute', 'summary')))
        scripts = []
        for text, jobs in workflows:
            script = text.split('        name: Select the public HPN SDK URL\n', 1)[1]
            script = script.split('          script: |\n', 1)[1].split('      - ', 1)[0]
            scripts.append(textwrap.dedent(script).strip())
            self.assertIn('/[\\s?#@]/.test(selected)', script)
            self.assertIn("core.setOutput('hpn_sdk_url', selected)", script)
            self.assertIn("core.setFailed('HPN SDK URL must be a public", script)
            self.assertNotIn('GITHUB_EVENT_PATH', script)
            self.assertNotIn('secrets.', script)
            for job in jobs:
                block = text.split('  ' + job + ':\n', 1)[1]
                block = re.split(r'^  [a-zA-Z0-9_-]+:', block, maxsplit=1, flags=re.MULTILINE)[0]
                self.assertIn('      HPN_SDK_URL: ${{ needs.prepare.outputs.hpn_sdk_url }}', block)
        self.assertEqual(scripts[0], scripts[1])

    def test_shared_build_is_cleaned_only_by_master_and_results_are_separate(self):
        self.assertIn('SHARED_ROOT=/mnt/wl_nas/devops/pccl-artifact-cache-${{ github.run_id }}/shared-standard/', self.text)
        self.assertIn('RESULT_ROOT=/mnt/wl_nas/devops/pccl-artifact-cache-${{ github.run_id }}/results/', self.text)
        start = self.text.index('cleanup_shared_build()')
        end = self.text.index('trap cleanup_shared_build EXIT', start)
        cleanup = self.text[start:end]
        self.assertNotIn('rm -rf "$SHARED_ROOT"', cleanup)
        self.assertIn('rm -rf "$(dirname "$SHARED_ROOT")"', cleanup)
        self.assertNotIn('rm -rf "$RESULT_ROOT"', cleanup)

    def test_sanity_hardware_uses_the_validated_host_network_rdma_runtime(self):
        single = self.text[self.text.index('  single-node:'):self.text.index('  two-node:')]
        cluster = self.job('two-node')
        for section in (single, cluster):
            self.assertIn('timeout -k 30s 300s apt-get update', section)
            self.assertIn('timeout -k 30s 600s apt-get install', section)
            self.assertIn('host_network: true', section)
            self.assertIn('dns_policy: ClusterFirstWithHostNet', section)
            self.assertIn('host_volumes: /dev/infiniband:/dev/infiniband', section)
            self.assertIn('export NCCL_SOCKET_IFNAME="$MPI_INTERFACE"', section)
            self.assertIn('export OMPI_MCA_btl_tcp_if_include="$MPI_INTERFACE"', section)
            self.assertIn('export OMPI_MCA_oob_tcp_if_include="$MPI_INTERFACE"', section)
            self.assertIn('alixpu_peermem is loaded', section)
            self.assertIn('alixpu_peermem is not loaded; DMA-BUF support is still possible', section)
            self.assertIn('ibv_devinfo', section)
            self.assertIn('rdma link show', section)
            self.assertIn("raise SystemExit('No non-zero RoCE GID is visible inside the worker')", section)
            self.assertIn('--mpi-interface "$MPI_INTERFACE"', section)
        self.assertIn("timeout -k 5s 30s cluster-ssh", cluster)
        self.assertIn('timeout -k 10s 60s mpirun --allow-run-as-root', cluster)
        self.assertIn("rack_policy: ${{ matrix.platform == 'm890p' && 'same' || '' }}", cluster)
        self.assertNotIn('icn-rack=', self.text)

    def test_legacy_executor_and_experimental_triggers_are_absent(self):
        for removed in ('source-catalog.json', 'cluster-source-case.json', 'manifest.py',
                        'cluster_ci.py', 'single_node_ci.py', 'runner_capture.py',
                        'test_migration_contract.py', 'experiment/minimal-csv',
                        't-head/ppu-scheduler-action@'):
            self.assertNotIn(removed, self.text)
        for path in ('catalog.py', 'command.py', 'execute.py', 'manifest.py', 'metrics.py',
                     'runner_capture.py', 'single_node_ci.py', 'cluster_ci.py', 'build_evidence.py',
                     'source-catalog.json', 'cluster-source-case.json', 'sanity-coverage.json'):
            self.assertFalse((ROOT / 'tests/xtest' / path).exists(), path)
        self.assertFalse((ROOT / '.github/workflows/source-cluster-debug.yml').exists())

    def test_symk_correctness_runs_in_the_required_standard_job(self):
        section = self.job('single-node')
        script = re.search(r"            if \[ '\$\{\{ matrix.profile \}\}' = standard \]; then\n"
                           r".*?            fi", section, re.S).group()
        self.assertIn('run_symk_validation.sh', script)
        self.assertIn('tests/unit_test/build/symkMultiTaskTest', script)
        self.assertIn('set -euo pipefail', section)
        build = (ROOT / '.github/workflows/scripts/build.sh').read_text()
        self.assertIn('${NATIVE_UNIT_BUILD}/symkMultiTaskTest', build)
        script = textwrap.dedent(script).replace('"$RESULT_ROOT/symk-validation.log"', '/dev/null')
        for profile, status, expected in (('standard', 0, 0), ('standard', 23, 23), ('notsm', 23, 0)):
            with self.subTest(profile=profile, status=status):
                command = ('set -euo pipefail\nbash() { return ' + str(status) + '; }\n' +
                           script.replace('${{ matrix.profile }}', profile))
                result = subprocess.run(['bash', '-c', command],
                                        env={**os.environ, 'build_output': '/unused'},
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, expected, result.stderr)

    def test_symk_runner_derives_capacity_and_checks_tag_fallback(self):
        script = (ROOT / 'tests/unit_test/run_symk_validation.sh').read_text()
        for obsolete in ('SYM_KERNEL_ID', 'SYM_WARPS', 'USE_MULTI_TASK', 'SYM_OP_ACC_PRECISION_INC'):
            self.assertNotIn(obsolete, script)
        self.assertIn('PCCL_CONF_FILE=/dev/null', script)
        self.assertIn('exports+=(-x NCCL_SYM_KERNEL)', script)
        mock = r'''timeout() {
  if [ "${3:-}" = param-contract ]; then printf 'PARAM_CONTRACT mocked=1\n'; return; fi
  return 91
}
run() {
  if [ "$1" = layout ]; then printf 'SYMK_LOADED rank=0 library=test\n560 64 55\n'; return; fi
  if [ "$1" = p2p-empty ]; then printf 'P2P_EMPTY mocked=1 graph=%s\n' "$2"; return; fi
  if [ "$1" = p2p-budget ]; then printf 'P2P_BUDGET mocked=1 budget=%s graph=%s\n' "$2" "$3"; return; fi
  if [ "$1" = budget-contract ]; then printf 'BUDGET_CONTRACT mocked=1 graph=%s\n' "$2"; return; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${PCCL_SYM_KERNEL:-auto}" "$1" "$3" "${12:-0}" "${14:-0}" "${PCCL_WORK_ARGS_BYTES:-4096}"
  if [[ -n ${NCCL_SYM_KERNEL+x} ]]; then
    printf 'ENV_CONTRACT pccl=%s nccl=%s path=%s\n' "${PCCL_SYM_KERNEL:-auto}" "$NCCL_SYM_KERNEL" "${12:-0}"
  fi
  if [ "${16:-0}" != 0 ]; then
    printf 'ORACLE kernel=%s coll=%s dtype=%s count=%s pattern=%s\n' "${PCCL_SYM_KERNEL:-auto}" "$1" "$2" "$4" "${16}"
  fi
}'''
        for header, capacity in ((560, 55), (640, 54)):
            with self.subTest(header=header, capacity=capacity):
                mocked = re.sub(r'run\(\) \{.*?\n\}',
                                lambda _: mock.replace('560 64 55', f'{header} 64 {capacity}'),
                                script, count=1, flags=re.S)
                result = subprocess.run(['bash', '-c', mocked, 'symk-validation', sys.executable, str(ROOT)],
                                        env={**os.environ, 'RANKS': '2'}, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('SYMK_VALIDATION_PASS', result.stdout)
                self.assertEqual(result.stdout.count('PARAM_CONTRACT mocked=1'), 1)
                self.assertEqual(result.stdout.count('P2P_EMPTY mocked=1'), 2)
                self.assertEqual(result.stdout.count('BUDGET_CONTRACT mocked=1'), 2)
                self.assertEqual(result.stdout.count('P2P_BUDGET mocked=1 budget=1024'), 2)
                self.assertIn('ENV_CONTRACT pccl=auto nccl=unknown_symk_kernel path=1', result.stdout)
                self.assertIn('ENV_CONTRACT pccl=AllReduce_RSxLD_AGxST nccl=unknown_symk_kernel path=1', result.stdout)
                self.assertIn('ENV_CONTRACT pccl=unknown_symk_kernel nccl=AllReduce_RSxLD_AGxST path=2', result.stdout)
                calls = {tuple(line.split('\t')) for line in result.stdout.splitlines() if '\t' in line}
                self.assertIn(('auto', 'ar', '0', '1', '0', '4096'), calls)
                for coll, ld, ll in (('ar', 'AllReduce_RSxLD_AGxST', 'AllReduce_AGxLL_R'),
                                     ('ag', 'AllGather_ST', 'AllGather_LL'),
                                     ('rs', 'ReduceScatter_LD', 'ReduceScatter_LL'),
                                     ('rs-avg', 'ReduceScatter_LD', 'ReduceScatter_LL')):
                    self.assertIn((ld, coll, str(capacity), '1', '1', '4096'), calls)
                    self.assertIn((ld, coll, str(capacity + 1), '1', '2', '4096'), calls)
                    self.assertIn((ld, coll, str(2*capacity + 1), '1', '3', '4096'), calls)
                    self.assertIn((ll, coll, '1', '1', '1', '4096'), calls)
                    self.assertIn((ll, coll, '17', '2', '0', '4096'), calls)
                self.assertIn(('AllReduce_RSxLD_AGxST', 'ar', '17', '1', '3', str(header + 7*64)), calls)
                self.assertIn(('unknown_symk_kernel', 'ar', '17', '2', '0', '4096'), calls)
                for kernel in ('AllReduce_AGxLLMC_R', 'AllReduce_RSxTmaLD_AGxTmaST',
                               'AllReduce_RSxLDMC_AGxSTMC'):
                    self.assertIn((kernel, 'ar', '17', '2', '0', '4096'), calls)
                self.assertIn(('ReduceScatter_RailA2A_LsaLD', 'rs', '17', '2', '0', '4096'), calls)
                self.assertIn(('^', 'ar', '17', '1', '1', '4096'), calls)
                for kernel in ('ReduceScatter_LL', 'ReduceScatter_LD'):
                    for dtype in ('f8e4m3', 'f8e5m2'):
                        self.assertIn(f'ORACLE kernel={kernel} coll=rs-avg dtype={dtype} count=16387 pattern=223',
                                      result.stdout)

    def test_symk_cta_model_preserves_ppu_tuning_and_limits(self):
        source = (ROOT / 'src/sym_kernels.cc').read_text()
        self.assertIn('int nMaxBlocks = k == ncclSymkKernelId_AllReduce_RSxLD_AGxST ? 32', source)
        self.assertIn(': k == ncclSymkKernelId_ReduceScatter_LD ? 48 : 24;', source)
        self.assertIn('std::min(ncclSymkMaxBlocks, std::max(nMaxBlocks, comm->config.minCTAs))', source)
        self.assertIn('nMaxBlocks = std::min<int>(nMaxBlocks, comm->config.maxCTAs);', source)
        self.assertIn('if (nUserCTAs > 0) nMinBlocks = nMaxBlocks = nUserCTAs;', source)
        self.assertIn('nMinBlocks = pow2Up(nMinBlocks);', source)
        self.assertIn('for (int bn = nMinBlocks; bn < nMaxBlocks; bn *= 2)', source)
        self.assertNotIn('DEFAULT_SYM_MAX_CTAS', source)
        self.assertIn('if (nWorks > 1) kmask &= ~kernelMask_LL;', source)
        self.assertIn('if (winRegType != ncclSymSendRegRecvReg && comm->nNodes > 1) '
                      'kmask &= ~kernelMask_Gin;', source)
        self.assertNotIn('ncclCeInit', source)
        args = (ROOT / 'src/include/sym_kernels.h').read_text()
        self.assertIn('char buf4K[4096];', args)
        self.assertNotIn('incPrecision', args)
        self.assertNotIn('symkWin', args)
        self.assertNotIn('KernelList_fast', args)
        makefile = (ROOT / 'tests/unit_test/src/Makefile').read_text()
        self.assertIn('-DSYMK_TEST_PUBLIC_ONLY', makefile)
        self.assertIn('$(if $(filter 1,$(NO_TSM)),-DNO_TSM)', makefile)

    def test_ce_initialization_uses_only_current_group_jobs(self):
        group = (ROOT / 'src/group.cc').read_text()
        ce = (ROOT / 'src/ce_coll.cc').read_text()
        self.assertIn('job->base.func = ncclCeInitFunc;', group)
        self.assertIn('PTHREADCHECKGOTO(pthread_create(&job->thread, nullptr, ncclAsyncJobMain, job)', group)
        self.assertNotIn('SYSCHECKGOTO(pthread_create(&job->thread, nullptr, ncclAsyncJobMain, job)', group)
        self.assertIn('comm->ceColl.initialized || ncclIntruQueueEmpty(&comm->planner.collCeTaskQueue)', group)
        initialization = group.index('job->base.func = ncclCeInitFunc;')
        self.assertLess(group.index('ncclTasksRegAndEnqueue(comm)'), initialization)
        self.assertLess(initialization, group.index('doLaunches(groupCommHeadMain'))
        self.assertNotIn('participants.size()', ce)
        self.assertNotIn('peer->planner.collCeTaskQueue', ce)
        self.assertIn('return ncclCeInitOnDevice(comm);', ce)
        self.assertIn('hggcStreamCreateWithFlags(&initStream, hggcStreamNonBlocking)', ce)
        self.assertIn('hggcMemsetAsync(cec->ceSyncBase, 0, ceDevBaseSize, initStream)', ce)
        self.assertNotIn('hggcMemset(cec->ceSyncBase', ce)
        self.assertIn('hggcStreamSynchronize(initStream)', ce)

    def test_alignment_fallback_and_dma_dependency(self):
        scheduler = (ROOT / 'src/scheduler/symmetric_sched.cc').read_text()
        batch = scheduler.index('for (ncclTaskColl* batchTask = headTask;')
        check = scheduler.index('ncclDevrWindowHasSysmemSegment(batchTask->sendWin)')
        query = scheduler.index('NOWARN(ncclGetAlgoInfo(comm, headTask, 0, 0, 1), NCCL_COLL)')
        self.assertLess(scheduler.index('ncclSymkPickKernel(comm'), query)
        self.assertLess(query, batch)
        self.assertLess(batch, check)
        self.assertIn('batchTask = batchTask->next', scheduler)
        self.assertIn('ncclDevrWindowHasSysmemSegment(batchTask->recvWin)', scheduler)
        self.assertIn('if (needFallback || batchTask->isSymLast) break;', scheduler)
        self.assertLess(check, scheduler.index('if (kernelId == ncclSymkKernelId_Count || needFallback)'))
        self.assertNotIn('ncclDevrWindowHasSysmemSegment(headTask->sendWin)', scheduler)
        self.assertNotIn('ncclDevrWindowHasSysmemSegment(task->sendWin)', scheduler)
        self.assertNotIn('NCCLCHECKGOTO(ncclGetAlgoInfo(comm, headTask', scheduler)
        group = (ROOT / 'src/group.cc').read_text()
        start = group.index('if (hasSymmetricTasks && pcclP2pDmaEnabled(comm))')
        end = group.index('while (!ncclIntruQueueEmpty(&comm->devrState.commCreateTaskQueue))', start)
        branch = group[start:end]
        self.assertLess(branch.index('ncclCeInit(comm)'), branch.index('pcclP2pDmaInitOnce(comm)'))
        legacy = (ROOT / 'src/include/symmetric.h').read_text()
        self.assertNotIn('blockOffset', legacy)
        self.assertNotIn('nAllElts', legacy)
        self.assertIn('sizeof(pcclSymDevArgs) == 80', legacy)
        suite = (ROOT / 'tests/unit_test/src/symkMultiTaskTest.cu').read_text()
        for name in ('alignment-policies', 'p2p-dma-first', 'p2p-dma-init-fault'):
            self.assertIn(name, suite)
        self.assertIn('PCCL_CTA_POLICY_ZERO', suite)
        self.assertIn('comm->p2pDmaState.peerSeq[source]', suite)
        self.assertIn('optional_query_fault=1', suite)

    def test_symk_fp8_bulk_tile_coverage(self):
        body = (ROOT / 'src/collectives/device/symmetric/all_reduce.cuh').read_text()
        self.assertIn('constexpr int BytePerPack = 16, UnrollPacks = 4;', body)
        self.assertIn('UnrollPeers = sizeof(T) == 1 && sizeof(typename Red::EltType) > sizeof(T) ? 2 : 4;', body)
        for ranks in (2, 4, 8):
            for blocks in (1, 3, 11, 64):
                warps = ranks * blocks * 16
                for offset in (0, 1, 3, 127):
                    for count in (0, 1, 7, 511, 512, 513, 2048, 2049, 9967, 17886, 25805, 1048576):
                        prefix = min((128-offset) % 128, count)
                        tiles = (count-prefix) // 2048
                        assigned = [tile for warp in range(warps)
                                    for tile in range(warp, tiles, warps)]
                        self.assertEqual(sorted(assigned), list(range(tiles)))
                        self.assertEqual(len(assigned), len(set(assigned)))
                        self.assertLessEqual(prefix + tiles*2048, count)
                        self.assertLess(count-prefix-tiles*2048, 2048)

    def test_symk_bodies_resolve_windows_via_device_api(self):
        directory = ROOT / 'src/collectives/device/symmetric'
        for name in ('all_reduce.cuh', 'all_gather.cuh', 'reduce_scatter.cuh'):
            with self.subTest(source=name):
                source = (directory / name).read_text()
                self.assertIn('.lsaPtr(0)', source)
                self.assertIn('ncclSymPtr<', source)
                self.assertNotIn('prim.peerPtr(', source)
                self.assertNotIn('ncclSymDevArgs', source)
        primitives = (directory / 'primitives.cuh').read_text()
        self.assertNotIn('T* peerPtr(int peer, T* selfPtr)', primitives)

    def test_real_csv_selection_and_source_revision(self):
        directory = ROOT / 'tests/xtest/minimal-csv'
        metadata = json.loads((directory / 'export.json').read_text())
        self.assertEqual(metadata['format_version'], 1)
        self.assertEqual(metadata['xtest_version'], '0.12.0')
        self.assertEqual(metadata['runner_revision'], artifact_cache.PCCL_TESTS_REVISION)
        self.assertTrue(re.fullmatch(r'[0-9a-f]{40}', metadata['source_revision']))
        expected = {('standard', 1): 71, ('standard', 2): 1, ('notsm', 1): 37}
        for (profile, nodes), count in expected.items():
            selected = run_cases.load_inputs(directory, profile, nodes)
            self.assertEqual(len(selected['cases']), count)
        selections = reconcile_results.load_selections(directory)
        self.assertEqual({name: len(inputs['cases']) for name, inputs in selections.items()},
                         {'standard': 71, 'notsm': 37, 'cluster': 1})

    def test_public_runtime_dependencies_remain_minimal(self):
        self.assertEqual((ROOT / 'tests/xtest/requirements.txt').read_text().strip(),
                         '# Required by the public run_test.py entry point, not the CSV executor.\n'
                         'filelock==4.2.10')
        action = 'uses: t-head/ci-infra/cluster-run@main'
        self.assertIn('VALIDATION_NODE_SELECTOR: board-type=ZW-M890P,nic-type=cx8', self.text)
        for name in ('build-standard', 'build-notsm', 'single-node', 'two-node'):
            self.assertIn(action, self.job(name))
        self.assertEqual(self.text.count(action), 4)
        self.assertNotIn('t-head/ci-infra/cluster-run@feat/rack-node-allocation', self.text)



if __name__ == '__main__':
    unittest.main()
