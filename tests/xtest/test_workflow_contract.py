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
                         'actions/download-artifact@v4', 'retention-days: 3',
                         'artifact_cache.py publish', 'artifact_cache.py validate',
                         'artifact_cache.py cleanup', 'trap cleanup_shared_build EXIT'):
            self.assertIn(expected, self.text)
        self.assertLess(self.text.index('name: Stage ${{ matrix.profile }} from artifact or valid NAS cache'),
                        self.text.index('name: Single-node CSV'))
        self.assertLess(self.text.index('name: Reconcile all selected hardware results'),
                        self.text.index('name: Remove run-owned NAS build artifacts'))
        self.assertIn("needs: [precheck, commit-checks, prepare, ai-code-review, human-review", self.text)

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

    def test_two_node_rdma_runtime_preserves_the_existing_case_and_identity(self):
        cluster = self.job('two-node')
        setup = cluster.split('setup_command: |', 1)[1].split('\n          command: |', 1)[0]
        self.assertIn('host_network: true', cluster)
        self.assertIn('dns_policy: ClusterFirstWithHostNet', cluster)
        self.assertIn('export HWLOC_COMPONENTS=-gl', setup)
        self.assertIn('export OMPI_MCA_orte_keep_fqdn_hostnames=1', setup)
        self.assertIn('export OMPI_MCA_oob_tcp_if_include="$MPI_INTERFACE"', setup)
        self.assertIn('export OMPI_MCA_btl_tcp_if_include="$MPI_INTERFACE"', setup)
        self.assertIn('export NCCL_SOCKET_IFNAME="$MPI_INTERFACE"', setup)
        self.assertIn("raise SystemExit('No non-zero RoCE GID is visible inside the worker')", setup)
        self.assertNotIn('extra_env: NCCL_NET=IB', cluster)
        self.assertIn('test "$checked_hosts" -eq 2', cluster)
        probe = cluster.split('timeout -k 10s 60s mpirun', 1)[1].split('/workspace/minimal-csv-cluster-env', 1)[0]
        self.assertNotIn('--mca', probe)
        self.assertIn('--build-record "$CACHE_DIR/output_standard.build.json"', cluster)
        self.assertIn("--expected-sha256 '${{ env.EXPECTED_SHA }}'", cluster)

    def test_final_reconciliation_uses_exact_case_and_build_identity_checks(self):
        results = self.job('results')
        for expected in ('tests/xtest/reconcile_results.py',
                         '--pccl-revision',
                         '--standard-sha256',
                         '--notsm-sha256'):
            self.assertIn(expected, results)
        prepare = self.text[self.text.index('  prepare:'):self.text.index('  ai-code-review:')]
        self.assertIn("test_reconcile_results.py", prepare)
        self.assertIn("test_workflow_contract.py", prepare)
        self.assertIn('reconcile_results.load_selections(', prepare)
        self.assertIn('--cards-per-node', results)

    def test_shared_build_is_cleaned_only_by_master_and_results_are_separate(self):
        self.assertIn('SHARED_ROOT=/mnt/wl_nas/devops/pccl-artifact-cache-${{ github.run_id }}/shared-standard/', self.text)
        self.assertIn('RESULT_ROOT=/mnt/wl_nas/devops/pccl-artifact-cache-${{ github.run_id }}/results/', self.text)
        start = self.text.index('cleanup_shared_build()')
        end = self.text.index('trap cleanup_shared_build EXIT', start)
        cleanup = self.text[start:end]
        self.assertIn('rm -rf "$SHARED_ROOT"', cleanup)
        self.assertNotIn('rm -rf "$RESULT_ROOT"', cleanup)

    def test_legacy_executor_and_experimental_triggers_are_absent(self):
        for removed in ('source-catalog.json', 'cluster-source-case.json', 'manifest.py',
                        'cluster_ci.py', 'single_node_ci.py', 'runner_capture.py',
                        'test_migration_contract.py', 'experiment/minimal-csv',
                        't-head/ci-infra/cluster-run@main'):
            self.assertNotIn(removed, self.text)
        for path in ('catalog.py', 'command.py', 'execute.py', 'manifest.py', 'metrics.py',
                     'runner_capture.py', 'single_node_ci.py', 'cluster_ci.py', 'build_evidence.py',
                     'source-catalog.json', 'cluster-source-case.json', 'sanity-coverage.json'):
            self.assertFalse((ROOT / 'tests/xtest' / path).exists(), path)
        self.assertFalse((ROOT / '.github/workflows/source-cluster-debug.yml').exists())

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
                         'filelock==3.20.3')
        self.assertIn('t-head/ci-infra/cluster-run@40165c945cba33ac7b16b41c4485fe53b68f581f', self.text)
        self.assertIn('t-head/ppu-scheduler-action@aac83e464f2a3fcbddddc60c8016021c996092a6', self.text)


if __name__ == '__main__':
    unittest.main()
