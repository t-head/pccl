#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local CI regression tests requiring PyYAML but no PPU or cluster access."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import yaml


SCRIPTS = Path(__file__).resolve().parent
WORKFLOWS = SCRIPTS.parent
WORKFLOW = yaml.load((WORKFLOWS / 'ppu-build-test.yml').read_text(), Loader=yaml.BaseLoader)


def inline_python(job, step):
    script = WORKFLOW['jobs'][job]['steps'][step]['run']
    return script.split('\n', 1)[1].rsplit('\nPYEOF', 1)[0]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scripts = self.root / 'pccl/.github/workflows/scripts'
        shutil.copytree(SCRIPTS, self.scripts, ignore=shutil.ignore_patterns('__pycache__'))

    def prepare(self, suite='pr', nodes='2', timeout='300'):
        result = subprocess.run(
            [sys.executable, '-c', inline_python('prepare', 1)], cwd=self.root,
            env={**os.environ, 'TEST_SUITE': suite, 'TEST_NNODES': nodes, 'TEST_TIMEOUT': timeout},
            text=True, capture_output=True, check=False)
        outputs = dict(line.split('=', 1) for line in result.stdout.splitlines())
        return result, outputs

    def matrix(self, suite):
        result, outputs = self.prepare(suite)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(outputs['matrix'])

    def test_pr_keeps_original_cases_and_limits(self):
        result, outputs = self.prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        entries = json.loads(outputs['matrix'])['include']
        original = [line.split()[1] for line in (SCRIPTS / 'casecmd.list').read_text().splitlines()
                    if line.startswith('bash ')]
        single = [e for e in entries if e['nnodes'] == 0]
        self.assertEqual([e['test_script'] for e in single], original)
        self.assertTrue(all(e['test_args'] == '' for e in single))
        self.assertEqual([e for e in entries if e['nnodes'] > 0], [
            {'test_script': 'run_multinode_smoke_test.sh', 'test_args': '',
             'test_id': 'run_multinode_smoke_test.sh', 'nnodes': 2}])
        self.assertEqual(outputs['job_timeout'], '360')
        inputs = WORKFLOW['on']['workflow_call']['inputs']
        self.assertEqual(inputs['test_suite']['default'], 'pr')
        self.assertEqual(inputs['timeout_minutes']['default'], '300')
        pr = yaml.load((WORKFLOWS / 'pr-test-ppu.yml').read_text(), Loader=yaml.BaseLoader)
        self.assertEqual(pr['jobs']['build-test']['with']['test_suite'], 'pr')

    def test_nightly_expands_coverage(self):
        entries = self.matrix('nightly')['include']
        single = {e['test_script'] for e in entries if e['nnodes'] == 0}
        pr = {e['test_script'] for e in self.matrix('pr')['include'] if e['nnodes'] == 0}
        self.assertEqual(single - pr, {'run_acu_test.sh', 'run_nightly_test.sh'})
        self.assertTrue(pr <= single)
        multi = [e for e in entries if e['nnodes'] > 0]
        self.assertEqual({e['test_args'] for e in multi}, {'socket', 'gdr', 'non_gdr'})
        self.assertTrue(all(e['nnodes'] == 2 for e in multi))
        self.assertEqual(len({e['test_id'] for e in entries}), len(entries))
        nightly = yaml.load((WORKFLOWS / 'nightly-test-ppu.yml').read_text(), Loader=yaml.BaseLoader)
        inputs = nightly['jobs']['build-test']['with']
        result, outputs = self.prepare(inputs['test_suite'], inputs['nnodes'], inputs['timeout_minutes'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(outputs['job_timeout'], '660')

    def test_invalid_inputs_fail_before_allocation(self):
        for suite, nodes, timeout in [('unknown', '2', '300'), ('nightly', '1', '300'),
                                      ('nightly', '2.5', '300'), ('pr', '2', '0'),
                                      ('nightly', '9', '300'), ('nightly', '2', '1201')]:
            with self.subTest(suite=suite, nodes=nodes, timeout=timeout):
                result, _ = self.prepare(suite, nodes, timeout)
                self.assertNotEqual(result.returncode, 0)

    def test_invalid_lists_fail_closed(self):
        for text in ['# Empty list\n', 'bash run_missing_test.sh\n',
                     'bash run_common_test.sh unexpected\n',
                     'bash run_common_test.sh\nbash run_common_test.sh\n',
                     'bash ../run_common_test.sh\n']:
            with self.subTest(text=text):
                (self.scripts / 'casecmd.list').write_text(text)
                result, _ = self.prepare()
                self.assertNotEqual(result.returncode, 0)
        (self.scripts / 'casecmd.nightly.multinode.list').write_text('bash run_multinode_test.sh invalid\n')
        result, _ = self.prepare('nightly')
        self.assertNotEqual(result.returncode, 0)

    def test_every_node_has_a_distinct_required_report(self):
        matrix = self.matrix('nightly')
        results = self.root / 'results'
        paths = [results / e['test_id'] / f'node-{rank}' / 'test-results.xml'
                 for e in matrix['include'] for rank in range(max(1, e['nnodes']))]
        paths.append(results / 'notsm_run_common_test/test-results.xml')
        self.assertEqual(len(paths), 16)
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('<testsuites tests="1"><testsuite><testcase name="fixture"/></testsuite></testsuites>')
        env = {**os.environ, 'TEST_MATRIX': json.dumps(matrix), 'NAS_RESULT_DIR': str(results)}
        env.pop('GITHUB_STEP_SUMMARY', None)
        command = [sys.executable, '-c', inline_python('summary', 1)]
        result = subprocess.run(command, cwd=self.root, env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        # A successful primary node must not hide a missing worker report.
        missing = next(path for path in paths if path.parent.name == 'node-1')
        missing.rename(missing.with_suffix('.missing'))
        result = subprocess.run(command, cwd=self.root, env=env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(str(missing), result.stdout)

    def test_pr_smoke_is_fixed_at_two_nodes_and_required(self):
        result, outputs = self.prepare(nodes='4')
        self.assertEqual(result.returncode, 0, result.stderr)
        entries = json.loads(outputs['matrix'])['include']
        self.assertEqual([e['nnodes'] for e in entries if e['nnodes'] > 0], [2])
        (self.scripts / 'run_multinode_smoke_test.sh').unlink()
        result, _ = self.prepare()
        self.assertNotEqual(result.returncode, 0)

    def test_pr_reports_reject_empty_malformed_failed_and_skipped_results(self):
        matrix = self.matrix('pr')
        results = self.root / 'results'
        paths = [results / e['test_id'] / f'node-{rank}' / 'test-results.xml'
                 for e in matrix['include'] for rank in range(max(1, e['nnodes']))]
        paths.append(results / 'notsm_run_common_test/test-results.xml')
        self.assertEqual(len(paths), 10)
        passing = '<testsuites tests="1"><testsuite><testcase name="fixture"/></testsuite></testsuites>'
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(passing)
        env = {**os.environ, 'TEST_MATRIX': json.dumps(matrix), 'NAS_RESULT_DIR': str(results)}
        env.pop('GITHUB_STEP_SUMMARY', None)
        command = [sys.executable, '-c', inline_python('summary', 1)]
        result = subprocess.run(command, cwd=self.root, env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        worker = results / 'run_multinode_smoke_test.sh/node-1/test-results.xml'
        fixtures = ['<invalid', '<testsuites tests="0"/>', passing.replace('tests="1"', 'tests="0"'),
                    passing.replace('tests="1"', 'tests="1" failures="1"')]
        fixtures += [f'<testsuites tests="1"><testsuite><testcase><{tag}/></testcase></testsuite></testsuites>'
                     for tag in ('failure', 'error', 'skipped')]
        for fixture in fixtures:
            with self.subTest(fixture=fixture):
                worker.write_text(fixture)
                result = subprocess.run(command, cwd=self.root, env=env, text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(str(worker), result.stdout)
        worker.unlink()
        result = subprocess.run(command, cwd=self.root, env=env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(str(worker), result.stdout)

    def test_gate_requires_matrix_and_reports(self):
        gate = WORKFLOW['jobs']['tests-gate']
        self.assertTrue({'test', 'summary'} <= set(gate['needs']))
        self.assertEqual(WORKFLOW['jobs']['test']['strategy']['fail-fast'], 'false')
        script = gate['steps'][0]['run'].split('\n', 1)[1].rsplit('\nPY', 1)[0]
        results = {name: {'result': 'success'} for name in gate['needs']}
        for name, status in [(None, 'success')] + [(name, status) for name in results
                                                 for status in ('failure', 'cancelled', 'skipped')]:
            with self.subTest(name=name, status=status):
                candidate = {**results, **({name: {'result': status}} if name else {})}
                result = subprocess.run([sys.executable, '-c', script], text=True, capture_output=True,
                                        env={**os.environ, 'JOB_RESULTS': json.dumps(candidate)})
                self.assertEqual(result.returncode == 0, status == 'success')
        pr = yaml.load((WORKFLOWS / 'pr-test-ppu.yml').read_text(), Loader=yaml.BaseLoader)
        self.assertIn('build-test', pr['jobs']['tests-gate']['needs'])
        self.assertIn('human-review', pr['jobs']['build-test']['needs'])
        nightly = yaml.load((WORKFLOWS / 'nightly-test-ppu.yml').read_text(), Loader=yaml.BaseLoader)
        self.assertEqual(nightly['on']['schedule'][0]['cron'], '0 14 * * *')

    def test_node_logs_do_not_mask_test_exit_status(self):
        command = WORKFLOW['jobs']['test']['steps'][0]['with']['command']
        self.assertIn('set -eo pipefail', command)
        start = command.index('set +e\n')
        end = command.index('\nset -e', start)
        capture = re.sub(r'\$\{\{.*?\}\}', 'fixture', command[start:end])
        for exit_code in (0, 7):
            with self.subTest(exit_code=exit_code):
                harness = f'python3() {{ return {exit_code}; }}; set -o pipefail\n' + capture
                result = subprocess.run(['bash', '-c', harness + '\nexit "$TEST_EXIT_CODE"'],
                                        cwd=self.root, text=True, capture_output=True)
                self.assertEqual(result.returncode, exit_code, result.stderr)
        self.assertIn('cp test-results.xml test-output.log "$NAS_RESULT_DIR/"', command)

    def test_log_and_report_persistence_failures_fail_the_node(self):
        command = WORKFLOW['jobs']['test']['steps'][0]['with']['command']
        script = re.sub(r'\$\{\{.*?\}\}', 'fixture', command[command.index('set +e\n'):])
        script = re.sub(r'^NAS_RESULT_DIR=.*$', 'NAS_RESULT_DIR="$RESULT_DIR"', script, flags=re.M)
        for failure in ('tee', 'missing-xml', 'copy'):
            with self.subTest(failure=failure):
                directory = self.root / failure
                directory.mkdir()
                harness = 'set -o pipefail\npython3() { printf "test output\\n"; '
                if failure != 'missing-xml':
                    harness += 'printf "<testsuites/>" > test-results.xml; '
                harness += 'return 0; };\n'
                if failure == 'tee':
                    harness += 'tee() { command tee "$@"; return 9; };\n'
                if failure == 'copy':
                    harness += 'cp() { return 10; };\n'
                result = subprocess.run(['bash', '-c', harness + script], cwd=directory,
                                        env={**os.environ, 'RESULT_DIR': str(directory / 'results')},
                                        text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)

    def test_non_mpi_build_is_isolated_and_enabled_for_both_suites(self):
        build = WORKFLOW['jobs']['build']['steps'][2]['with']['command']
        self.assertIn('export PCCL_BUILD_K8S=1', build)
        text = (SCRIPTS / 'build.sh').read_text()
        block = re.search(r'if \[ "\$\{PCCL_BUILD_K8S.*?\nfi', text, re.S).group()
        command = 'bash() { printf "%s %s %s\\n" "$MPI" "$BUILDDIR" "$*"; };\n' + block
        for enabled in ('0', '1'):
            result = subprocess.run(['bash', '-c', command], text=True, capture_output=True,
                                    env={**os.environ, 'PCCL_BUILD_K8S': enabled,
                                         'PCCL_TESTS_DIR': '/fixture/pccl-tests'})
            self.assertEqual(result.returncode, 0, result.stderr)
            expected = '0 /fixture/pccl-tests/build-k8s build.sh\n' if enabled == '1' else ''
            self.assertEqual(result.stdout, expected)

    def run_script(self, script, transport='socket', overrides=None, binary=True):
        binary_path = self.root / 'pccl-tests/build-k8s/all_reduce_perf'
        if binary:
            binary_path.parent.mkdir(parents=True, exist_ok=True)
            binary_path.touch()
            binary_path.chmod(0o755)
        env = {**os.environ, 'TEST_ROOT': str(self.root), 'NNODES': '2', 'NPROC_PER_NODE': '8',
               'NODE_RANK': '0', 'RANK': '0', 'WORLD_SIZE': '2',
               'MASTER_ADDR': 'worker-0.service', 'MASTER_PORT': '29500', **(overrides or {})}
        # Keep real Bash validation and argument handling; mock setup, DNS and test execution.
        harness = r'''
source() { :; }
python3() {
  if [[ "$1" == - ]]; then
    printf 'DNS\t%s\n' "$MASTER_ADDR" >&2
    printf '10.0.0.1\n'
    return "${MOCK_DNS_EXIT_CODE:-0}"
  fi
  printf 'CMD'; printf '\t%s' "$@"; printf '\n'
  printf 'ENV\t%s\t%s\t%s\t%s\n' "$PCCL_COMM_ID" "$NCCL_COMM_ID" "$PCCL_NET" "$PATH"
  printf 'BOOTSTRAP\t%s\t%s\n' "$MASTER_ADDR" "$NCCL_BARRIER_PORT"
  return "${MOCK_EXIT_CODE:-0}"
}
export -f source python3
bash "$@"
'''
        return subprocess.run(['bash', '-c', harness, 'harness', str(SCRIPTS / script), transport],
                              env=env, text=True, capture_output=True, check=False)

    def test_multinode_commands_use_all_nodes_and_correct_transport(self):
        hostname = 'worker-0.pccl-nightly-1234567890-multinode-gdr.ppu-sched.svc.cluster.local'
        self.assertGreater(len(hostname), 64)
        self.assertTrue(all(len(label) <= 63 for label in hostname.split('.')))
        for transport in ('socket', 'gdr', 'non_gdr'):
            for rank in ('0', '1'):
                with self.subTest(transport=transport, rank=rank):
                    result = self.run_script('run_multinode_test.sh', transport,
                                             {'NODE_RANK': rank, 'RANK': rank, 'MASTER_ADDR': hostname})
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stderr.count(f'DNS\t{hostname}\n'), 1)
                    bootstraps = [line for line in result.stdout.splitlines() if line.startswith('BOOTSTRAP\t')]
                    self.assertEqual(bootstraps, [f'BOOTSTRAP\t{hostname}\t29501'] * 3)
                    commands = [line.split('\t')[1:] for line in result.stdout.splitlines()
                                if line.startswith('CMD\t')]
                    self.assertEqual(len(commands), 3)
                    for args in commands:
                        for option, value in [('--launch-mode', 'k8s'), ('--nnodes', '2'), ('--ranks', '16'),
                                              ('--threads', '8'), ('--nprocs', '1'), ('--cross-node', transport)]:
                            self.assertEqual(args[args.index(option) + 1], value)
                        self.assertNotIn('--no-data-check', args)
                        self.assertNotIn('LL128', args)
                        self.assertNotIn('Signal', args)
                    network = 'Socket' if transport == 'socket' else 'IB'
                    prefix = f'ENV\t10.0.0.1:29500\t10.0.0.1:29500\t{network}\t'
                    environments = [line for line in result.stdout.splitlines() if line.startswith('ENV\t')]
                    self.assertEqual(len(environments), 3)
                    self.assertTrue(all(line.startswith(prefix + str(self.root / 'pccl-tests/build-k8s') + ':')
                                        for line in environments))

    def test_multinode_dns_failure_stops_before_collectives(self):
        for transport in ('socket', 'gdr', 'non_gdr'):
            for rank in ('0', '1'):
                with self.subTest(transport=transport, rank=rank):
                    result = self.run_script('run_multinode_test.sh', transport,
                                             {'MOCK_DNS_EXIT_CODE': '7', 'NODE_RANK': rank, 'RANK': rank,
                                              'PCCL_COMM_ID': 'stale:29500', 'NCCL_COMM_ID': 'stale:29500'})
                    self.assertEqual(result.returncode, 7, result.stderr)
                    self.assertNotIn('CMD\t', result.stdout)
                    self.assertNotIn('ENV\t', result.stdout)

    def test_multinode_resolver_retries_and_enforces_deadline(self):
        script = (SCRIPTS / 'run_multinode_test.sh').read_text()
        code = script.split("master_ipv4=$(python3 - <<'PY'\n", 1)[1].split('\nPY\n)', 1)[0]
        hostname = 'worker-0.pccl-nightly-1234567890-multinode-gdr.ppu-sched.svc.cluster.local'
        output = io.StringIO()
        with mock.patch.dict(os.environ, {'MASTER_ADDR': hostname}), \
                mock.patch('subprocess.run', return_value=
                           subprocess.CompletedProcess([], 0, stdout='10.0.0.1\n')) as run, \
                redirect_stdout(output):
            exec(compile(code, 'nightly-dns', 'exec'), {})
        self.assertEqual(output.getvalue(), '10.0.0.1\n')
        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[0]
        self.assertEqual(command[:2], [sys.executable, '-c'])
        self.assertEqual(command[-1], hostname)
        self.assertEqual(run.call_args.kwargs,
                         {'capture_output': True, 'text': True, 'timeout': 60, 'check': True})
        output = io.StringIO()
        with mock.patch.object(sys, 'argv', ['-c', hostname]), \
                mock.patch('socket.gethostbyname', side_effect=[socket.gaierror('not ready'), '10.0.0.1']) as dns, \
                mock.patch('time.sleep') as sleep, redirect_stdout(output):
            exec(compile(command[2], 'dns-child', 'exec'), {})
        self.assertEqual(output.getvalue(), '10.0.0.1\n')
        self.assertEqual(dns.call_args_list, [mock.call(hostname), mock.call(hostname)])
        sleep.assert_called_once_with(1)
        for error in (subprocess.TimeoutExpired(command, 60), subprocess.CalledProcessError(1, command)):
            with self.subTest(error=type(error).__name__), mock.patch.dict(os.environ, {'MASTER_ADDR': hostname}), \
                    mock.patch('subprocess.run', side_effect=error), redirect_stdout(io.StringIO()) as output:
                with self.assertRaisesRegex(SystemExit, 'resolve worker-0 IPv4 address within 60 seconds'):
                    exec(compile(code, 'nightly-dns', 'exec'), {})
                self.assertEqual(output.getvalue(), '')

    def test_multinode_resolver_accepts_numeric_ipv4(self):
        script = (SCRIPTS / 'run_multinode_test.sh').read_text()
        code = script.split("master_ipv4=$(python3 - <<'PY'\n", 1)[1].split('\nPY\n)', 1)[0]
        result = subprocess.run([sys.executable, '-c', code],
                                env={**os.environ, 'MASTER_ADDR': '127.0.0.1'},
                                text=True, capture_output=True, timeout=10, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '127.0.0.1\n')

    def test_multinode_dns_timeout_kills_and_reaps_blocked_resolver(self):
        script = (SCRIPTS / 'run_multinode_test.sh').read_text()
        code = script.split("master_ipv4=$(python3 - <<'PY'\n", 1)[1].split('\nPY\n)', 1)[0]
        pid_file = self.root / 'dns-child.pid'
        original_run = subprocess.run

        def blocked_dns_run(command, **kwargs):
            self.assertEqual(kwargs['timeout'], 60)
            prefix = ('import os, socket, time\nfrom pathlib import Path\n'
                      f'Path({str(pid_file)!r}).write_text(str(os.getpid()))\n'
                      'socket.gethostbyname = lambda hostname: time.sleep(60)\n')
            return original_run([*command[:2], prefix + command[2], *command[3:]],
                                **{**kwargs, 'timeout': 0.5})

        with mock.patch.dict(os.environ, {'MASTER_ADDR': 'worker-0.service'}), \
                mock.patch('subprocess.run', side_effect=blocked_dns_run), redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(SystemExit, 'resolve worker-0 IPv4 address within 60 seconds'):
                exec(compile(code, 'nightly-dns', 'exec'), {})
        self.assertEqual(output.getvalue(), '')
        with self.assertRaises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)

    def test_multinode_rejects_missing_or_inconsistent_environment(self):
        for overrides in [{'NNODES': ''}, {'NNODES': '1', 'WORLD_SIZE': '1'}, {'RANK': '1'},
                          {'NODE_RANK': '2'}, {'MASTER_ADDR': ''}, {'MASTER_PORT': '65535'},
                          {'NPROC_PER_NODE': '0'}, {'WORLD_SIZE': '1'}]:
            with self.subTest(overrides=overrides):
                result = self.run_script('run_multinode_test.sh', overrides=overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('CMD\t', result.stdout)

    def test_multinode_requires_non_mpi_artifacts(self):
        result = self.run_script('run_multinode_test.sh', binary=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Missing K8s test artifacts', result.stderr)

    def test_test_failures_propagate(self):
        for script in ('run_multinode_test.sh', 'run_nightly_test.sh'):
            with self.subTest(script=script):
                result = self.run_script(script, overrides={'MOCK_EXIT_CODE': '7'})
                self.assertEqual(result.returncode, 7)
                self.assertEqual(result.stdout.count('CMD\t'), 1)

    def run_smoke(self, output, overrides=None, binary=True):
        binary_path = self.root / 'pccl-tests/build-k8s/all_reduce_perf'
        if binary:
            binary_path.parent.mkdir(parents=True, exist_ok=True)
            binary_path.write_text('#!/bin/bash\nprintf "%s" "$SMOKE_OUTPUT"\n'
                                   'printf "ENV %s %s %s %s %s\\n" "$PCCL_NET" "$PCCL_IB_DISABLE" '
                                   '"$PCCL_NET_GDR_LEVEL" "$PCCL_ALGO" "$PCCL_PROTO"\n'
                                   'printf "ADDR %s %s\\n" "$PCCL_COMM_ID" "$NCCL_COMM_ID"\n'
                                   'exit "${SMOKE_EXIT_CODE:-0}"\n')
            binary_path.chmod(0o755)
        env = {**os.environ, 'TEST_ROOT': str(self.root), 'NNODES': '2', 'NPROC_PER_NODE': '8',
               'NODE_RANK': '0', 'RANK': '0', 'WORLD_SIZE': '2', 'MASTER_ADDR': '127.0.0.1',
               'MASTER_PORT': '29500', 'SMOKE_OUTPUT': output, **(overrides or {})}
        return subprocess.run(['bash', '-c', 'source() { :; }; export -f source; bash "$1"',
                               'harness', str(SCRIPTS / 'run_multinode_smoke_test.sh')],
                              env=env, text=True, capture_output=True, check=False)

    def smoke_output(self):
        return ('8388608 2097152 float sum -1 100 80 150 0 100 80 150 0\n'
                '# Out of bounds values : 0 OK\n')

    def test_smoke_runs_one_checked_allreduce_on_both_nodes(self):
        for rank in ('0', '1'):
            with self.subTest(rank=rank):
                result = self.run_smoke(self.smoke_output() if rank == '0' else '',
                                        {'NODE_RANK': rank, 'RANK': rank})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('ENV IB 0 SYS Ring Simple', result.stdout)
                self.assertIn('ADDR 127.0.0.1:29500 127.0.0.1:29500', result.stdout)
                self.assertIn('-t 8 -g 1 -b 8M -e 8M -d float -o sum -n 20 -w 5 -c 1 -k 2', result.stdout)
                self.assertIn('-N 1 --init_only 0 --report_type 0', result.stdout)
                self.assertEqual(result.stdout.count('/build-k8s/all_reduce_perf'), 1)
                self.assertNotIn('mpirun', result.stdout)

    def test_smoke_rejects_noop_and_wrong_data_on_either_layout(self):
        for output in ('', 'Usage: all_reduce_perf\n',
                       self.smoke_output().replace('# Out of bounds values : 0 OK', ''),
                       self.smoke_output().replace(': 0 OK', ': 1 FAILED'),
                       self.smoke_output().replace('150 0 100', '150 1 100'),
                       self.smoke_output().replace('150 0\n', '150 1\n'),
                       self.smoke_output().replace('150 0', '150 N/A'),
                       self.smoke_output() * 2,
                       self.smoke_output().replace('8388608 2097152 float sum', '32 8 float sum')):
            with self.subTest(output=output):
                self.assertNotEqual(self.run_smoke(output).returncode, 0)

    def test_smoke_propagates_failure_on_either_node(self):
        for rank in ('0', '1'):
            result = self.run_smoke(self.smoke_output(),
                                    {'NODE_RANK': rank, 'RANK': rank, 'SMOKE_EXIT_CODE': '7'})
            self.assertEqual(result.returncode, 7)

    def test_smoke_requires_scheduler_contract_and_non_mpi_binary(self):
        for overrides in ({'NNODES': '1'}, {'NPROC_PER_NODE': '4'}, {'WORLD_SIZE': '1'},
                          {'NODE_RANK': '2'}, {'RANK': '1'}, {'MASTER_PORT': '65535'},
                          {'MASTER_PORT': ''}, {'MASTER_ADDR': ''}):
            with self.subTest(overrides=overrides):
                result = self.run_smoke(self.smoke_output(), overrides, binary=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('AllReduce binary', result.stderr)
        result = self.run_smoke(self.smoke_output(), binary=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Missing non-MPI AllReduce binary', result.stderr)

    def test_smoke_timeout_is_bounded_even_without_output(self):
        script = (SCRIPTS / 'run_multinode_smoke_test.sh').read_text()
        code = script.split("python3 - <<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
        with mock.patch.dict(os.environ, {'TEST_ROOT': str(self.root), 'NODE_RANK': '0',
                                          'MASTER_ADDR': 'worker-0.service', 'MASTER_PORT': '29500'}), \
                mock.patch('os.access', return_value=True), \
                mock.patch('subprocess.run', side_effect=[
                    subprocess.CompletedProcess([], 0, stdout='127.0.0.1\n'),
                    subprocess.TimeoutExpired('all_reduce_perf', 600)]) as run, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(SystemExit, '600-second deadline'):
                exec(compile(code, 'smoke', 'exec'), {})
        self.assertEqual(run.call_args.kwargs['timeout'], 600)

    def test_smoke_resolves_long_names_and_rejects_dns_timeout(self):
        script = (SCRIPTS / 'run_multinode_smoke_test.sh').read_text()
        code = script.split("python3 - <<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
        hostname = 'worker-0.' + 'job-' * 30 + '.ppu-sched.svc.cluster.local'
        env = {'TEST_ROOT': str(self.root), 'NODE_RANK': '0', 'MASTER_ADDR': hostname, 'MASTER_PORT': '29500'}
        with mock.patch.dict(os.environ, env), mock.patch('os.access', return_value=True), \
                mock.patch('subprocess.run', side_effect=[
                    subprocess.CompletedProcess([], 0, stdout='10.0.0.1\n'),
                    subprocess.CompletedProcess([], 0, stdout=self.smoke_output())]) as run, \
                redirect_stdout(io.StringIO()):
            exec(compile(code, 'smoke', 'exec'), {})
            self.assertEqual(os.environ['PCCL_COMM_ID'], '10.0.0.1:29500')
            self.assertEqual(os.environ['NCCL_COMM_ID'], os.environ['PCCL_COMM_ID'])
        self.assertEqual(run.call_args_list[0].args[0][-1], hostname)
        self.assertEqual(run.call_args_list[0].kwargs['timeout'], 60)
        with mock.patch.dict(os.environ, env), mock.patch('os.access', return_value=True), \
                mock.patch('subprocess.run', side_effect=subprocess.TimeoutExpired('DNS', 60)) as run:
            with self.assertRaisesRegex(SystemExit, 'resolve worker-0 IPv4'):
                exec(compile(code, 'smoke', 'exec'), {})
        self.assertEqual(run.call_count, 1)

    def test_smoke_timeout_kills_and_reaps_the_real_process(self):
        script = (SCRIPTS / 'run_multinode_smoke_test.sh').read_text()
        code = script.split("python3 - <<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
        binary = self.root / 'pccl-tests/build-k8s/all_reduce_perf'
        binary.parent.mkdir(parents=True)
        pid_file = self.root / 'child.pid'
        original_run = subprocess.run

        def short_run(*args, **kwargs):
            if kwargs['timeout'] == 600:
                kwargs['timeout'] = 0.5
            return original_run(*args, **kwargs)

        for partial in (False, True):
            with self.subTest(partial=partial):
                binary.write_text(f'#!{sys.executable}\nimport os, time\nfrom pathlib import Path\n'
                                  f'Path({str(pid_file)!r}).write_text(str(os.getpid()))\n'
                                  + ('print("partial output", flush=True)\n' if partial else '')
                                  + 'time.sleep(60)\n')
                binary.chmod(0o755)
                output = io.StringIO()
                with mock.patch.dict(os.environ, {'TEST_ROOT': str(self.root), 'NODE_RANK': '0',
                                                  'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500'}), \
                        mock.patch('subprocess.run', side_effect=short_run), redirect_stdout(output):
                    with self.assertRaisesRegex(SystemExit, '600-second deadline'):
                        exec(compile(code, 'smoke', 'exec'), {})
                with self.assertRaises(ProcessLookupError):
                    os.kill(int(pid_file.read_text()), 0)
                if partial:
                    self.assertIn('partial output', output.getvalue())

    def test_nightly_scans_eight_cards_and_large_messages(self):
        result = self.run_script('run_nightly_test.sh')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = next(line.split('\t')[1:] for line in result.stdout.splitlines() if line.startswith('CMD\t'))
        self.assertEqual(args[args.index('--ranks') + 1], '8')
        self.assertEqual(args[args.index('--nprocs') + 1], '8')
        self.assertIn('1GB', args)
        self.assertIn('hypercube_perf', args)
        self.assertNotIn('--no-data-check', args)


if __name__ == '__main__':
    unittest.main()
