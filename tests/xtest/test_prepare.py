# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
import venv

from catalog import cases, performance_cases, source_inputs
from manifest import build_manifest, digest, write_bundle


ROOT = Path(__file__).resolve().parents[2]


class PreparationTests(unittest.TestCase):
    def test_clean_interpreter_cli_without_framework_imports(self):
        runner = Path(os.environ['PCCL_TESTS_ROOT']).resolve()
        revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            venv.EnvBuilder(with_pip=False).create(root / 'env')
            python = root / 'env/bin/python'
            code = '''
import importlib.util
import pathlib
import runpy
import sys

source = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(source.parent))
for name in ('xtest', 'mmengine', 'filelock', 'pip'):
    assert importlib.util.find_spec(name) is None, name

def audit(event, args):
    if event == 'import' and args[0].split('.')[0] in ('xtest', 'mmengine', 'filelock'):
        raise AssertionError('Unexpected framework import: ' + args[0])
    if event in ('socket.connect', 'socket.getaddrinfo'):
        raise AssertionError('Unexpected network access')

sys.addaudithook(audit)
sys.argv = sys.argv[1:]
runpy.run_path(str(source), run_name='__main__')
'''
            command = [str(python), '-I', '-S', '-B', '-c', code, str(ROOT / 'tests/xtest/manifest.py'),
                       '--suite', 'pr', '--single-node', '--pccl-revision', revision,
                       '--pccl-tests', str(runner), '--run-id', '1', '--output', str(root / 'bundle')]
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual([path.name for path in (root / 'bundle').iterdir()], ['manifest.json'])
            manifest = json.loads((root / 'bundle/manifest.json').read_text())
            self.assertEqual((manifest['schema_version'], manifest['case_count']), (3, 108))
            expected = build_manifest(cases() + performance_cases(), manifest['inputs'])
            self.assertEqual(manifest, expected)
            self.assertEqual(manifest['inputs']['test_types'], ['functional', 'performance'])
            self.assertNotIn('xtest_version', manifest['inputs'])
            self.assertNotIn('case_path', json.dumps(manifest))
            self.assertEqual(json.loads(result.stdout)['include'], [
                {key: shard[key] for key in ('shard_id', 'nodes', 'build_profile', 'test_type')}
                for shard in expected['shards']])

    def test_execution_and_gate_without_framework_imports(self):
        from test_manifest import fixture_spec

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            venv.EnvBuilder(with_pip=False).create(root / 'env')
            spec = fixture_spec(0)
            spec['argv'] = ['{python}', '-S', '-B', '-c', 'print("synthetic command, no hardware")']
            manifest = build_manifest([spec], {})
            write_bundle(manifest, root / 'command')
            specs = [spec for spec in cases() + performance_cases() if spec['nodes'] == 1]
            inputs = dict(test_suite='pr', single_node=True, debug_only=False,
                          test_types=['functional', 'performance'], node_counts=[2, 8, 16], shards_per_pool=1,
                          timings_sha256=digest({}), **source_inputs())
            write_bundle(build_manifest(specs, inputs), root / 'sanity')
            code = '''
import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])
root = Path(sys.argv[2])
for name in ('xtest', 'mmengine', 'filelock', 'pip'):
    assert importlib.util.find_spec(name) is None, name

def audit(event, args):
    if event == 'import' and args[0].split('.')[0] in ('xtest', 'mmengine', 'filelock'):
        raise AssertionError('Unexpected framework import: ' + args[0])
    if event in ('socket.connect', 'socket.getaddrinfo'):
        raise AssertionError('Unexpected network access')

sys.addaudithook(audit)
from execute import run_shard
from manifest import load_manifest
from single_node_ci import selection, summarize
manifest = load_manifest(root / 'command/manifest.json')
assert run_shard(root / 'command/manifest.json', manifest['shards'][0]['shard_id'],
                 root, root, root / 'executed', []) == 0
assert selection(root / 'sanity/manifest.json')['case_count'] == 108
assert summarize(root / 'sanity/manifest.json', root / 'missing-results', root / 'summary') == 1
assert not ({'xtest', 'mmengine', 'filelock'} & sys.modules.keys())
'''
            result = subprocess.run([str(root / 'env/bin/python'), '-I', '-S', '-B', '-c', code,
                                     str(ROOT / 'tests/xtest'), str(root)], capture_output=True, text=True,
                                    timeout=30, env={**os.environ, 'NODE_RANK': '0'})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            index = json.loads((root / 'executed/index.json').read_text())
            self.assertEqual(index['failures'], 0)
            self.assertNotIn('xtest', json.dumps(index))
            self.assertFalse(json.loads((root / 'summary/summary.json').read_text())['complete'])

    def test_workflow_separates_offline_preparation_from_hardware_execution(self):
        workflow = (ROOT / '.github/workflows/source-sanity.yml').read_text()
        self.assertEqual(re.findall(r'^  ([a-z_-]+):$', workflow, re.M),
                         ['pull_request', 'workflow_dispatch', 'prepare',
                          'build-standard', 'build-notsm', 'sanity', 'cluster', 'summary'])
        prepare = workflow.split('\n  build-standard:', 1)[0]
        build_standard = workflow.split('\n  build-standard:', 1)[1].split('\n  build-notsm:', 1)[0]
        build_notsm = workflow.split('\n  build-notsm:', 1)[1].split('\n  sanity:', 1)[0]
        sanity = workflow.split('\n  sanity:', 1)[1].split('\n  cluster:', 1)[0]
        cluster = workflow.split('\n  cluster:', 1)[1].split('\n  summary:', 1)[0]
        self.assertIn('runs-on: ubuntu-24.04', prepare)
        self.assertIn('venv --without-pip', prepare)
        self.assertIn('unshare --net', prepare)
        for forbidden in ('pip install', 'requirements.txt', 'ppu-scheduler-action@',
                          'single_node_ci.py" run', 'single_node_ci.py summary'):
            self.assertNotIn(forbidden, prepare)
        for suite in ('test_catalog.py', 'test_manifest.py', 'test_metrics.py', 'test_command.py', 'test_execute.py',
                      'test_single_node_ci.py', 'test_coverage_map.py', 'test_prepare.py',
                      'test_migration_contract.py'):
            self.assertIn(suite, prepare)
        self.assertIn('ppu-scheduler-action@aac83e464f2a3fcbddddc60c8016021c996092a6', workflow)
        self.assertIn('fromJSON(needs.prepare.outputs.matrix)', workflow)
        self.assertIn('single_node_ci.py matrix', prepare)
        self.assertIn('image: ${{ needs.prepare.outputs.image }}', workflow)
        self.assertIn('node_selector: ${{ needs.prepare.outputs.node_selector }}', workflow)
        self.assertIn('nnodes: 0', workflow)
        self.assertIn('cards_per_node', workflow)
        self.assertIn('nproc_per_node: ${{ needs.prepare.outputs.cards_per_node }}', workflow)
        self.assertIn('nproc_per_node: ${{ matrix.nproc_per_node }}', workflow)
        self.assertIn('matrix.profile', workflow)
        self.assertIn('matrix.shards', workflow)
        self.assertIn('--shard "$shard"', workflow)
        self.assertIn('max-parallel: 2', sanity)
        self.assertIn('single_node_ci.py" run', workflow)
        self.assertIn('single_node_ci.py summary', workflow)
        # Re-running failed jobs keeps the run id but bumps the attempt, so the build and
        # the result roots must not embed the attempt number.
        self.assertEqual(workflow.count('${{ github.run_attempt }}'), 0)
        self.assertIn('/devops/pccl-source-sanity/${{ github.run_id }}/build', workflow)
        self.assertIn('/devops/pccl-source-sanity/${{ github.run_id }}/results', workflow)
        self.assertIn('rm -rf "$RESULT_ROOT/${{ matrix.profile }}/$shard"', sanity)
        self.assertIn('rm -rf "$RESULT_ROOT/standard"', cluster)
        self.assertIn('build.sh', build_standard)
        self.assertIn('notsm_build.sh', build_notsm)
        self.assertIn('tar -czf', build_standard)
        self.assertIn('tar -czf', build_notsm)
        self.assertIn('sha256sum', build_standard)
        self.assertIn('sha256sum', build_notsm)
        self.assertIn('tar -xzf', sanity)
        self.assertIn('sha256sum --check', sanity)
        # Performance identity must not inherit the internal build-system publish path.
        self.assertIn('unset PCCL_PATH', sanity)
        self.assertNotIn('build.sh', sanity)
        # The two-node job must reuse the standard build instead of compiling again.
        self.assertIn('needs: [prepare, build-standard]', cluster)
        self.assertIn('output_standard.tar.gz', cluster)
        # Both pods must see the same binaries: stage the build on the shared NAS, never
        # in the master-only workspace.
        self.assertIn('export build_output="$RUN_ROOT/output_standard"', cluster)
        self.assertIn('sha256sum --check "$BUILD_ROOT/output_standard.tar.gz.sha256"', cluster)
        self.assertIn('tar -xzf "$BUILD_ROOT/output_standard.tar.gz" -C "$RUN_ROOT"', cluster)
        self.assertNotIn('build_output="$SOURCE_DIR/output_standard"', cluster)
        self.assertIn('cluster_ci.py" run', cluster)
        self.assertNotIn('build.sh', cluster)
        self.assertIn('privileged: true', cluster)
        self.assertIn('host_volumes: /dev/infiniband:/dev/infiniband', cluster)
        self.assertIn('cluster-run-source-manifest', prepare)
        self.assertIn('cluster-run-python-deps', prepare)
        self.assertIn('cluster-run-build-deps', prepare)
        self.assertIn('cluster_ci.py prepare', prepare)
        self.assertIn('cluster_ci.py summary', workflow)
        summary = workflow.split('\n  summary:', 1)[1]
        for job in (prepare, build_standard, build_notsm, sanity, cluster, summary):
            self.assertIn('github.event.pull_request.head.repo.full_name == github.repository', job)
        self.assertNotIn("if: always() && needs.prepare.result == 'success'", summary)
        self.assertNotIn('notsm_build.sh', sanity)
        self.assertNotIn('-m xtest', workflow)
        self.assertNotIn('art-pub.eng.t-head.cn', workflow)

    def test_result_gate_requires_prepare_single_node_and_two_node_success(self):
        workflow = (ROOT / '.github/workflows/source-sanity.yml').read_text()
        gate = workflow.split("          python3 - <<'PY'\n", 1)[1].split('\n          PY', 1)[0]
        gate = textwrap.dedent(gate)
        names = ('prepare', 'build-standard', 'build-notsm', 'sanity', 'cluster')
        base = {name: {'result': 'success'} for name in names}

        def evaluate(changes):
            needs = {name: dict(value) for name, value in base.items()}
            for name, result in changes.items():
                needs[name]['result'] = result
            return subprocess.run([sys.executable, '-c', gate], capture_output=True, text=True,
                                  env={'NEEDS': json.dumps(needs)}).returncode

        self.assertEqual(evaluate({}), 0)
        self.assertEqual(evaluate({'build-notsm': 'skipped'}), 0)
        self.assertEqual(evaluate({'prepare': 'failure', 'sanity': 'skipped', 'cluster': 'skipped'}), 1)
        self.assertEqual(evaluate({'cluster': 'skipped'}), 1)
        self.assertEqual(evaluate({'sanity': 'skipped'}), 1)


if __name__ == '__main__':
    unittest.main()
