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
LEGACY_CATALOG_SHA256 = '0099b6661c757fdb2e8a0419333912a2400a5c46d5edd0fe47c6f3af587853be'
# Canonical {legacy_case_key: legacy_spec} for the original 96 single-node sanity
# executions. This pins the diagnostic's expected data without importing old code.
LEGACY_SPECS_SHA256 = '1d042858c61069a0db6cbd2dcb45c8eb6989a77e38e851673477a6065fbcee20'


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

    def runner_contract(self, argv):
        # The runner mutates both Case objects and the environment. Isolate each
        # parse/cmd_rule call so a previous Case cannot make the next one pass.
        with mock.patch.dict(os.environ, ENV, clear=True):
            args, cases = self.parse(argv)
            self.assertEqual(len(cases), 1)
            case = cases[0]
            fields = json.dumps(vars(case), sort_keys=True)
            self.runner.apply_env_args_rule(args)
            self.runner.apply_env_case_rule(args, case)
            with redirect_stdout(io.StringIO()) as output:
                key, command = self.runner.cmd_rule(args, case)
            self.assertEqual(output.getvalue().splitlines(), [key, command])
            tokens = shlex.split(command)
            self.assertEqual(tokens.count(case.op), 1)
        return fields, args, tokens

    def runner_argv(self, spec):
        self.assertEqual(spec['argv'][:2], ['{python}', '{test_root}/run_test.py'])
        return [arg.replace('{mpi_options}', '') for arg in spec['argv'][2:]]

    def benchmark_argv(self, argv, op):
        positions = [index for index, token in enumerate(argv) if Path(token).name == op]
        self.assertEqual(len(positions), 1, argv)
        tail = argv[positions[0] + 1:]
        self.assertFalse(any(re.search(r'[$`;&|<>()\n]', token) for token in tail), argv)
        return [op, *tail]

    def test_source_cases_match_all_reviewed_register_layouts_exactly(self):
        expected = {self.runner_contract(argv)[0]
                    for argv in shell_calls(SCRIPTS / 'run_register_test.sh', 14, 27)}
        self.assertEqual(len(expected), 14)
        standard, notsm, devices = set(), set(), set()
        for source in self.catalog.cases():
            if (source['parameters'].get('register') != 1
                    or '{test_root}/run_test.py' not in source['argv']):
                continue
            identity, args, _ = self.runner_contract(self.runner_argv(source))
            self.assertEqual(args.launch_mode, 'normal')
            self.assertEqual(args.process_group, 'normal')
            self.assertEqual(args.register, 1)
            if args.devices:
                # Retain the three device-list Cases as extras. They must NOT
                # substitute for the six missing process/thread layouts.
                self.assertEqual(args.devices, '0,1')
                self.assertNotIn(identity, expected)
                pair = (identity, source['build_profile'])
                self.assertNotIn(pair, devices)
                devices.add(pair)
            else:
                self.assertIn(identity, expected)
                seen = standard if source['build_profile'] == 'standard' else notsm
                self.assertNotIn(identity, seen)
                seen.add(identity)
        self.assertEqual(standard, expected)
        original_six = {self.runner_contract(argv)[0]
                        for argv in shell_calls(SCRIPTS / 'run_register_test.sh', 14, 19)}
        self.assertEqual(notsm, original_six)
        self.assertEqual(len(devices), 6)
        self.assertEqual(Counter(profile for _, profile in devices), Counter(standard=3, notsm=3))
        self.assertEqual({identity for identity, profile in devices if profile == 'standard'},
                         {identity for identity, profile in devices if profile == 'notsm'})

    def test_all_historical_sanity_contracts_match_selected_native_cases(self):
        report_path = SCRIPTS.parents[2] / 'tests/xtest/sanity-coverage.json'
        report = json.loads(report_path.read_text())
        bundle = self.catalog.load_catalog()
        self.assertEqual(report['schema_version'], 1)
        self.assertEqual(report['bundle_sha256'], self.catalog.digest(
            {key: value for key, value in report.items() if key != 'bundle_sha256'}))
        self.assertEqual(report['inputs']['legacy_catalog_sha256'], LEGACY_CATALOG_SHA256)
        self.assertEqual(report['inputs']['source_bundle_sha256'], bundle['bundle_sha256'])
        self.assertEqual(report['inputs']['source'], bundle['source'])
        self.assertEqual(report['summary'], dict(expected=96, matched=96, missing=0, ambiguous=0,
                                                 selected_executions=108))
        rows = report['cases']
        self.assertEqual(len(rows), 96)
        historical = {row['legacy_case_key']: row['legacy_spec'] for row in rows}
        self.assertEqual(len(historical), 96)
        self.assertEqual(self.catalog.digest(historical), LEGACY_SPECS_SHA256)
        selected = {(row['id'], profile): {**row['spec'], 'build_profile': profile}
                    for row in bundle['executions'] for profile in row['profiles']}
        matched, counts = set(), Counter()
        for row in rows:
            legacy = row['legacy_spec']
            with self.subTest(legacy_case_key=row['legacy_case_key']):
                self.assertEqual(len(row['selected']), 1)
                target, = row['selected']
                reference = (target['execution_id'], target['build_profile'])
                self.assertNotIn(reference, matched)
                matched.add(reference)
                native = selected[reference]
                self.assertEqual(native['source_context']['case_id'], target['case_id'])
                self.assertEqual(native['build_profile'], legacy['build_profile'])
                self.assertEqual(native['test_type'], legacy['test_type'])
                if '{test_root}/run_test.py' in legacy['argv']:
                    old_fields, old_args, old_command = self.runner_contract(self.runner_argv(legacy))
                    fields, args, command = self.runner_contract(self.runner_argv(native))
                    self.assertEqual(fields, old_fields)
                    self.assertEqual(vars(args), vars(old_args))
                    self.assertEqual(command, old_command)
                    counts['runner'] += 1
                else:
                    op = legacy['parameters']['op']
                    self.assertEqual(self.benchmark_argv(native['argv'], op),
                                     self.benchmark_argv(legacy['argv'], op))
                    counts['direct'] += 1
        self.assertEqual(counts, Counter(runner=77, direct=19))
        self.assertEqual(len(matched), 96)
        self.assertEqual(len(set(selected) - matched), 12)

    def test_default_and_invalid_layout_contracts(self):
        args, _ = self.parse([])
        self.assertEqual((args.elapsed_type, args.warmup, args.average, args.register), (0, 0, 2, 0))
        for layout in ('--ranks 1 --threads 8', '--ranks 3 --nnodes 2', '--ranks 17'):
            with self.subTest(layout=layout):
                argv = ['--op', 'all_reduce_perf', *shlex.split(layout)]
                _, cases = self.parse(argv, allow_empty=True)
                self.assertEqual(cases, [])

    def test_every_selected_runner_is_nonempty_and_direct_source_argv_is_preserved(self):
        specs = self.catalog.cases() + self.catalog.performance_cases()
        self.assertEqual(len(specs), 108)
        counts, profiles = Counter(), Counter()
        for spec in specs:
            with self.subTest(case_id=spec['source_context']['case_id'], profile=spec['build_profile']):
                self.assertTrue(spec['legacy_source'].startswith('pccl_test_suites:'))
                profiles[spec['build_profile']] += 1
                if '{test_root}/run_test.py' in spec['argv']:
                    self.assertNotIn('--nproc', spec['argv'])
                    self.assertIn('--nprocs', spec['argv'])
                    identity, args, _ = self.runner_contract(self.runner_argv(spec))
                    fields = json.loads(identity)
                    self.assertEqual(fields['op'], spec['parameters']['op'])
                    self.assertEqual(fields['ranks'], spec['parameters']['ranks'])
                    self.assertEqual(fields['nprocs'], spec['processes_per_node'])
                    self.assertEqual(fields['threads'], spec['threads_per_process'])
                    self.assertEqual(fields['nnodes'], spec['nodes'])
                    self.assertEqual(fields['ranks'], fields['nnodes'] * fields['nprocs']
                                     * fields['threads'] * spec['devices_per_thread'])
                    self.assertEqual((args.launch_mode, args.process_group), ('normal', 'normal'))
                    counts['runner'] += 1
                else:
                    op = spec['parameters']['op']
                    source = shlex.split(spec['source_context']['source_command'])
                    self.assertEqual(self.benchmark_argv(spec['argv'], op), self.benchmark_argv(source, op))
                    counts['direct'] += 1
        self.assertEqual(counts, Counter(runner=89, direct=19))
        self.assertEqual(profiles, Counter(standard=71, notsm=37))


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
