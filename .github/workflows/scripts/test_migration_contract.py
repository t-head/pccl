#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline contract between the source-derived catalog and pinned public runner."""

import argparse
from collections import Counter
from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parent
REVISION = '394556512e87259e9ffa25225632c46b0a0bbcd5'
HASHES = {
    'run_test.py': 'fdbdf7623703157b49b9605491fb1cb602fe50ec931ae6763c2645bef7c6b0fc',
    'run_test_config.py': 'd58194c5c210ebaabe1ed42d7c2ced924c5dde22081285c22516e1ccdf611b1a',
}
ENV = {'PATH': '/usr/bin:/bin'}


def load_public(root):
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if head != REVISION:
        raise ValueError(f'pccl-tests HEAD mismatch: {head}')
    for name, expected in HASHES.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'{name} sha256 mismatch: {actual}')
    with mock.patch.object(sys, 'dont_write_bytecode', True), mock.patch.dict(sys.modules):
        for name in ('run_test_config', 'run_test'):
            spec = importlib.util.spec_from_file_location(name, root / f'{name}.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    return module


def load_catalog():
    path = SCRIPTS.parents[2] / 'tests/xtest/catalog.py'
    spec = importlib.util.spec_from_file_location('migration_catalog', path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def shell_calls(path, first, last):
    calls = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if first <= number <= last:
            argv = shlex.split(line)
            if argv[:2] != ['python3', '${TEST_ROOT}/run_test.py']:
                raise ValueError(f'Unexpected register command at line {number}')
            calls.append(argv[2:])
    if len(calls) != last - first + 1:
        raise ValueError('Missing reviewed register commands')
    return calls


class RunnerContract(unittest.TestCase):
    runner = None
    catalog = None

    def setUp(self):
        self.assertIsNotNone(self.runner, 'Run this script with --pccl-tests PATH')
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.dict(os.environ, ENV, clear=True))
        stack.enter_context(mock.patch.object(self.runner, 'CaseList', []))
        stack.enter_context(redirect_stdout(io.StringIO()))
        for target in ('subprocess.Popen', 'os.system', 'socket.socket'):
            stack.enter_context(mock.patch(target, side_effect=AssertionError('Offline contract only')))

    def parse(self, argv, allow_empty=False):
        self.assertFalse(any(re.search(r'[$`;&|<>()\n]', token) for token in argv), argv)
        with mock.patch.object(sys, 'argv', ['run_test.py', *argv]):
            args = self.runner.obtain_test_params()
        self.runner.fill_case_list_from_args(args)
        cases = list(self.runner.CaseList)
        if not allow_empty:
            self.assertTrue(cases, f'Empty CaseList is not success: {argv}')
        return args, cases

    def case_identity(self, argv):
        args, cases = self.parse(argv)
        self.assertEqual(len(cases), 1)
        case = cases[0]
        fields = vars(case).copy()
        self.runner.apply_env_args_rule(args)
        self.runner.apply_env_case_rule(args, case)
        with redirect_stdout(io.StringIO()) as output:
            key, command = self.runner.cmd_rule(args, case)
        self.assertEqual(output.getvalue().splitlines(), [key, command])
        self.assertEqual(shlex.split(command).count(case.op), 1)
        return json.dumps(fields, sort_keys=True), args

    def measurement_identity(self, identity):
        fields = json.loads(identity)
        for name in ('nprocs', 'threads'):
            fields.pop(name)
        return json.dumps(fields, sort_keys=True)

    def test_source_cases_match_reviewed_register_commands(self):
        exact_expected = {self.case_identity(argv)[0]
                          for argv in shell_calls(SCRIPTS / 'run_register_test.sh', 14, 19)}
        near_expected = {self.measurement_identity(self.case_identity(argv)[0])
                         for argv in shell_calls(SCRIPTS / 'run_register_test.sh', 22, 27)}
        exact_seen, near_seen, profiles = set(), set(), Counter()
        for source in self.catalog.cases():
            argv = [arg.replace('{mpi_options}', '') for arg in source['argv'][2:]]
            identity, args = self.case_identity(argv)
            if 'devices' in source['parameters']:
                self.assertNotIn(identity, exact_expected)
                measurement = self.measurement_identity(identity)
                self.assertIn(measurement, near_expected)
                pair = (measurement, source['build_profile'])
                self.assertNotIn(pair, near_seen)
                near_seen.add(pair)
                self.assertEqual(args.devices, '0,1')
            else:
                self.assertIn(identity, exact_expected)
                pair = (identity, source['build_profile'])
                self.assertNotIn(pair, exact_seen)
                exact_seen.add(pair)
                self.assertEqual(args.devices, '')
            profiles[source['build_profile']] += 1
            self.assertEqual(args.launch_mode, 'normal')
            self.assertEqual(args.process_group, 'normal')
            self.assertEqual(args.register, 1)
        self.assertEqual(len(exact_expected), 6)
        self.assertEqual(len(near_expected), 3)
        self.assertEqual(len({identity for identity, _ in exact_seen}), 6)
        self.assertEqual(len({identity for identity, _ in near_seen}), 3)
        self.assertEqual(profiles, Counter(standard=9, notsm=9))

    def test_default_and_invalid_layout_contracts(self):
        args, _ = self.parse([])
        self.assertEqual((args.elapsed_type, args.warmup, args.average, args.register), (0, 0, 2, 0))
        for layout in ('--ranks 1 --threads 8', '--ranks 3 --nnodes 2', '--ranks 17'):
            with self.subTest(layout=layout):
                argv = ['--op', 'all_reduce_perf', *shlex.split(layout)]
                _, cases = self.parse(argv, allow_empty=True)
                self.assertEqual(cases, [])

    def test_catalog_has_no_framework_or_private_runner_dependency(self):
        self.assertEqual(len(self.catalog.cases()), 18)
        for spec in self.catalog.cases():
            self.assertEqual(spec['argv'][:2], ['{python}', '{test_root}/run_test.py'])
            self.assertNotIn('--nproc', spec['argv'])
            self.assertIn('--nprocs', spec['argv'])
            self.assertTrue(spec['legacy_source'].startswith('pccl_test_suites:'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pccl-tests', type=Path, required=True)
    args = parser.parse_args()
    try:
        RunnerContract.runner = load_public(args.pccl_tests.resolve())
        RunnerContract.catalog = load_catalog()
    except (OSError, ValueError, subprocess.CalledProcessError, ImportError) as error:
        parser.error(str(error))
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(RunnerContract))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
