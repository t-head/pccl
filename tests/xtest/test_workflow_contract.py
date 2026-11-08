# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import re
import unittest

import artifact_cache
import run_cases


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / '.github/workflows/source-sanity.yml'


class WorkflowContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding='utf-8')

    def test_supported_events_and_review_gates_are_preserved(self):
        for expected in ('pull_request:', "branches: [main, 'release/**']", 'branches: [main]',
                         'workflow_dispatch:', 'cancel-in-progress: true',
                         't-head/.github/.github/workflows/user-precheck.yml@main',
                         't-head/.github/.github/workflows/ai-code-review.yml@main',
                         't-head/.github/.github/workflows/human-review.yml@main',
                         'github.event.pull_request.head.repo.full_name == github.repository',
                         'Reject unsupported repository or event',
                         'Require successful results, cleanup and all prerequisite jobs'):
            self.assertIn(expected, self.text)

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

    def test_public_runtime_dependencies_remain_minimal(self):
        self.assertEqual((ROOT / 'tests/xtest/requirements.txt').read_text().strip(),
                         '# Required by the public run_test.py entry point, not the CSV executor.\n'
                         'filelock==3.20.3')
        self.assertIn('t-head/ci-infra/cluster-run@40165c945cba33ac7b16b41c4485fe53b68f581f', self.text)
        self.assertIn('t-head/ppu-scheduler-action@aac83e464f2a3fcbddddc60c8016021c996092a6', self.text)


if __name__ == '__main__':
    unittest.main()
