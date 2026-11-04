# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
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
            self.assertEqual((manifest['schema_version'], manifest['case_count'], len(manifest['shards'])), (2, 18, 2))
            self.assertNotIn('xtest_version', manifest['inputs'])
            self.assertNotIn('case_path', json.dumps(manifest))
            self.assertEqual(len(json.loads(result.stdout)['include']), 2)

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
                          test_types=['functional'], node_counts=[2, 8, 16], shards_per_pool=1,
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
assert selection(root / 'sanity/manifest.json')['case_count'] == 18
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
        workflow = (ROOT / '.github/workflows/xtest-single-node-validation.yml').read_text()
        self.assertEqual(re.findall(r'^  ([a-z_]+):$', workflow, re.M),
                         ['push', 'workflow_dispatch', 'prepare', 'sanity', 'summary'])
        prepare = workflow.split('\n  sanity:', 1)[0]
        self.assertIn('runs-on: ubuntu-24.04', prepare)
        self.assertIn('venv --without-pip', prepare)
        self.assertIn('unshare --net', prepare)
        for forbidden in ('pip install', 'requirements.txt', 'ppu-scheduler-action@',
                          'single_node_ci.py" run', 'single_node_ci.py summary'):
            self.assertNotIn(forbidden, prepare)
        for suite in ('test_manifest.py', 'test_metrics.py', 'test_command.py', 'test_execute.py',
                      'test_single_node_ci.py', 'test_coverage_map.py', 'test_prepare.py',
                      'test_migration_contract.py'):
            self.assertIn(suite, prepare)
        self.assertIn('ppu-scheduler-action@aac83e464f2a3fcbddddc60c8016021c996092a6', workflow)
        self.assertIn('profile: standard', workflow)
        self.assertIn('profile: notsm', workflow)
        self.assertIn('single_node_ci.py" run', workflow)
        self.assertIn('single_node_ci.py summary', workflow)
        self.assertNotIn('-m xtest', workflow)
        self.assertNotIn('art-pub.eng.t-head.cn', workflow)


if __name__ == '__main__':
    unittest.main()
