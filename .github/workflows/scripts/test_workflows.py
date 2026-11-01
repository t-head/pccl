#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local CI regression tests requiring PyYAML but no PPU or cluster access."""

from contextlib import contextmanager, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tarfile
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

    def prepare(self, suite='pr', nodes='[2]', timeout='300'):
        result = subprocess.run(
            [sys.executable, '-c', inline_python('prepare', 1)], cwd=self.root,
            env={**os.environ, 'TEST_SUITE': suite, 'TEST_NNODES': nodes, 'TEST_TIMEOUT': timeout,
                 'DEBUG_BUILD_RUN_ID': '', 'DEBUG_BUILD_ATTEMPT': '1',
                 'GITHUB_RUN_ID': '50000000000', 'GITHUB_RUN_ATTEMPT': '7'},
            text=True, capture_output=True, check=False)
        outputs = dict(line.split('=', 1) for line in result.stdout.splitlines())
        return result, outputs

    def matrix(self, suite):
        nodes = WORKFLOW['on']['workflow_call']['inputs']['nnodes']['default']
        if suite == 'nightly':
            nightly = yaml.load((WORKFLOWS / 'nightly-test-ppu.yml').read_text(), Loader=yaml.BaseLoader)
            nodes = nightly['jobs']['build-test']['with']['nnodes']
        result, outputs = self.prepare(suite, nodes)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(outputs['matrix'])

    @contextmanager
    def debug_build(self, *, env=None, source=None, jobs=None, info=None, archive=True, diffs=(0, 0)):
        environment = {
            'TEST_SUITE': 'pr', 'TEST_NNODES': '[2]', 'TEST_TIMEOUT': '300',
            'DEBUG_BUILD_RUN_ID': '41293442849', 'DEBUG_BUILD_ATTEMPT': '1',
            'GITHUB_EVENT_NAME': 'workflow_dispatch', 'GITHUB_REPOSITORY': 't-head/pccl',
            'GITHUB_API_URL': 'https://api.github.com', 'GH_TOKEN': 'fixture-token',
            'GITHUB_RUN_ID': '50000000000', 'GITHUB_RUN_ATTEMPT': '7', 'GITHUB_SHA': 'b' * 40,
            'GITHUB_STEP_SUMMARY': str(self.root / 'debug-summary'), **(env or {}),
        }
        run = {
            'id': 41293442849, 'run_attempt': 1, 'head_repository': {'full_name': 't-head/pccl'},
            'event': 'push', 'path': '.github/workflows/pr-test-ppu.yml', 'head_sha': 'a' * 40,
            **(source or {}),
        }
        if jobs is None:
            jobs = [{'name': 'PPU Build and Test / Build (ubuntu24_hggcrt13)', 'conclusion': 'success'}]
        directory = self.root / 'debug-build'
        directory.mkdir(exist_ok=True)
        info_path = directory / 'env_info.json'
        info_path.write_text(json.dumps({
            'config_name': 'ubuntu24_hggcrt13',
            'docker_image': 'download.t-head.cn/docker_release/ppu:sdk2.2.0-hggcrt3-ubuntu24.04-py312-20261005',
            **(info or {}),
        }))
        archive_path = directory / 'output_hggcrt13.tar.gz'
        archive_path.unlink(missing_ok=True)
        if archive:
            with tarfile.open(archive_path, 'w:gz') as bundle:
                payload = b'fixture build output\n'
                member = tarfile.TarInfo('output_hggcrt13/fixture')
                member.size = len(payload)
                bundle.addfile(member, io.BytesIO(payload))
        summary = Path(environment['GITHUB_STEP_SUMMARY'])
        summary.unlink(missing_ok=True)
        nas = Path('/wl_nas/devops/pccl-ci/41293442849/1/build_output')
        real_read_text, real_open = Path.read_text, Path.open

        def read_text(path, *args, **kwargs):
            if Path('/wl_nas') in path.parents:
                self.assertEqual(path, nas / 'env_info.json')
                return real_read_text(info_path, *args, **kwargs)
            return real_read_text(path, *args, **kwargs)

        def open_path(path, *args, **kwargs):
            if Path('/wl_nas') in path.parents:
                self.assertEqual(path, nas / 'output_hggcrt13.tar.gz')
                self.assertEqual(args, ('rb',))
                return real_open(archive_path, *args, **kwargs)
            return real_open(path, *args, **kwargs)

        responses = [io.BytesIO(json.dumps(value).encode()) for value in (run, {'jobs': jobs})]
        output = io.StringIO()
        with mock.patch.dict(os.environ, environment, clear=True), \
                mock.patch('urllib.request.urlopen', side_effect=responses) as api, \
                mock.patch('subprocess.run', side_effect=[
                    subprocess.CompletedProcess([], code) for code in (0, *diffs)]) as git, \
                mock.patch.object(Path, 'read_text', read_text), mock.patch.object(Path, 'open', open_path), \
                redirect_stdout(output):
            previous = Path.cwd()
            try:
                os.chdir(self.root)
                yield {'stdout': output, 'summary': summary, 'archive': archive_path, 'api': api, 'git': git}
            finally:
                os.chdir(previous)

    def execute_debug_prepare(self, fixture):
        exec(compile(inline_python('prepare', 1), 'prepare', 'exec'), {})
        return dict(line.split('=', 1) for line in fixture['stdout'].getvalue().splitlines())

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
            {'test_script': 'run_multinode_smoke_test.sh', 'test_args': transport,
             'test_id': f'run_multinode_smoke_test.sh-{transport}', 'nnodes': 2}
            for transport in ('mnnvl', 'rdma')])
        self.assertEqual(len({e['test_id'] for e in entries}), len(entries))
        self.assertEqual(WORKFLOW['jobs']['test']['steps'][-1]['with']['node_selector'], 'board-type=ZW-M890P')
        self.assertEqual(outputs['job_timeout'], '360')
        inputs = WORKFLOW['on']['workflow_call']['inputs']
        self.assertEqual(inputs['test_suite']['default'], 'pr')
        self.assertEqual(inputs['timeout_minutes']['default'], '300')
        pr = yaml.load((WORKFLOWS / 'pr-test-ppu.yml').read_text(), Loader=yaml.BaseLoader)
        self.assertEqual(pr['jobs']['build-test']['with']['test_suite'], 'pr')

    def test_only_smokes_and_nightly_multinode_enable_host_access(self):
        options = WORKFLOW['jobs']['test']['steps'][-1]['with']['container_options']
        keys = ('privileged', 'host_ipc', 'host_network')
        expressions = re.findall(r'\$\{\{\s*(.*?)\s*\}\}', options, re.S)
        self.assertEqual(len(expressions), 3)
        self.assertEqual(len(set(expressions)), 1)

        def render(entry):
            def evaluate(match):
                results = []
                for clause in match.group(1).split('||'):
                    condition = re.fullmatch(
                        r"matrix\.nnodes (==|>=) ([0-9]+) && matrix\.test_script == '([^']+)'",
                        clause.strip().strip('()'))
                    self.assertIsNotNone(condition, clause)
                    nodes = int(condition[2])
                    matches_nodes = (entry['nnodes'] == nodes if condition[1] == '=='
                                     else entry['nnodes'] >= nodes)
                    results.append(matches_nodes and entry['test_script'] == condition[3])
                return str(any(results)).lower()

            rendered = re.sub(r'\$\{\{\s*(.*?)\s*\}\}', evaluate, options, flags=re.S)
            enabled = ((entry['nnodes'] == 2 and entry['test_script'] == 'run_multinode_smoke_test.sh')
                       or (entry['nnodes'] in range(2, 17) and entry['test_script'] == 'run_multinode_test.sh'))
            self.assertEqual(rendered.splitlines(), [f'{key}: {str(enabled).lower()}' for key in keys])
            # The scheduler matches ^key: and takes whitespace field 2; it does not parse JSON/YAML.
            parsed = {}
            for key in keys:
                lines = re.findall(rf'^{key}:.*$', rendered, re.M)
                self.assertEqual(len(lines), 1)
                parsed[key] = lines[0].split()[1]
                self.assertEqual(parsed[key], str(enabled).lower())
            return all(value == 'true' for value in parsed.values())

        for suite, nodes in [('pr', 2)] + [('nightly', n) for n in range(2, 17)]:
            with self.subTest(suite=suite, nodes=nodes):
                result, outputs = self.prepare(suite, json.dumps([nodes]))
                self.assertEqual(result.returncode, 0, result.stderr)
                entries = json.loads(outputs['matrix'])['include']
                selected = [entry['test_id'] for entry in entries if render(entry)]
                expected = ([f'run_multinode_smoke_test.sh-{t}' for t in ('mnnvl', 'rdma')]
                            if suite == 'pr' else
                            [f'run_multinode_test.sh-{t}-{nodes}nodes' for t in ('mnnvl', 'socket', 'gdr', 'non_gdr')])
                self.assertEqual(selected, expected)
        excluded = [(n, 'run_multinode_smoke_test.sh') for n in (0, 1, 3, 4, 8, 16)]
        excluded += [(n, 'run_multinode_test.sh') for n in (0, 1)]
        excluded += [(n, script) for n in (0, 1, 2, 4, 8, 16)
                     for script in ('run_common_test.sh', 'run_nightly_test.sh')]
        for nodes, script in excluded:
            with self.subTest(nodes=nodes, script=script):
                self.assertFalse(render({'nnodes': nodes, 'test_script': script}))
        for name in ('build', 'build-test-notsm'):
            for step in WORKFLOW['jobs'][name]['steps']:
                if step.get('uses') == 't-head/ppu-scheduler-action@main':
                    self.assertNotIn('container_options', step['with'])

    def test_nightly_expands_coverage(self):
        entries = self.matrix('nightly')['include']
        single = {e['test_script'] for e in entries if e['nnodes'] == 0}
        pr = {e['test_script'] for e in self.matrix('pr')['include'] if e['nnodes'] == 0}
        self.assertEqual(single - pr, {'run_acu_test.sh', 'run_nightly_test.sh'})
        self.assertTrue(pr <= single)
        multi = [e for e in entries if e['nnodes'] > 0]
        self.assertEqual(multi, [
            {'test_script': 'run_multinode_test.sh', 'test_args': transport,
             'test_id': f'run_multinode_test.sh-{transport}-{nodes}nodes', 'nnodes': nodes}
            for nodes in (2, 8, 16) for transport in ('mnnvl', 'socket', 'gdr', 'non_gdr')])
        self.assertEqual(len(entries), 21)
        self.assertEqual(len({e['test_id'] for e in entries}), len(entries))
        nightly = yaml.load((WORKFLOWS / 'nightly-test-ppu.yml').read_text(), Loader=yaml.BaseLoader)
        self.assertEqual(nightly['on'], {
            'push': {'branches': ['ci/nightly-shared-tests']},
            'schedule': [{'cron': '0 14 * * *'}], 'workflow_dispatch': ''})
        self.assertNotIn('strategy', nightly['jobs']['build-test'])
        inputs = nightly['jobs']['build-test']['with']
        result, outputs = self.prepare(inputs['test_suite'], inputs['nnodes'], inputs['timeout_minutes'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(outputs['job_timeout'], '660')

    def test_invalid_inputs_fail_before_allocation(self):
        invalid_nodes = ('', '2', '[]', '[1]', '[17]', '[2.5]', '[true]', '["8"]',
                         '[2, 2]', '[2, 8, 16, 16]', 'null', '{}', '[[2]]', '[2,')
        for nodes in invalid_nodes:
            with self.subTest(nodes=nodes):
                result, outputs = self.prepare('nightly', nodes)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('matrix', outputs)
        for suite, nodes, timeout in [('unknown', '[2]', '300'), ('pr', '[2]', '0'),
                                      ('nightly', '[2]', '1201')]:
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

    def test_formal_prepare_uses_current_build_without_debug_reuse(self):
        for suite in ('pr', 'nightly'):
            with self.subTest(suite=suite):
                result, outputs = self.prepare(suite)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(outputs['debug_reuse'], 'false')
                self.assertEqual(outputs['build_key'], '50000000000/7')
                self.assertEqual(outputs['build_sha256'], '')

    def test_debug_prepare_reuses_trusted_attempt_and_records_provenance(self):
        for attempt, event, workflow in [('7', 'push', 'pr-test-ppu.yml'),
                                         ('9', 'schedule', 'nightly-test-ppu.yml'),
                                         ('11', 'workflow_dispatch', 'pr-test-ppu.yml')]:
            with self.subTest(event=event, attempt=attempt), self.debug_build(
                    env={'GITHUB_RUN_ATTEMPT': attempt},
                    source={'event': event, 'path': f'.github/workflows/{workflow}'}) as fixture:
                outputs = self.execute_debug_prepare(fixture)
                self.assertEqual(json.loads(outputs['matrix'])['include'], [
                    {'test_script': 'run_multinode_smoke_test.sh', 'test_args': transport,
                     'test_id': f'run_multinode_smoke_test.sh-{transport}', 'nnodes': 2}
                    for transport in ('mnnvl', 'rdma')])
                self.assertEqual(outputs['debug_reuse'], 'true')
                self.assertEqual(outputs['build_key'], '41293442849/1')
                self.assertEqual(outputs['job_timeout'], '360')
                checksum = hashlib.sha256(fixture['archive'].read_bytes()).hexdigest()
                self.assertEqual(outputs['build_sha256'], checksum)
                summary = fixture['summary'].read_text()
                for text in ('Debug only: reused build 41293442849/1', f'Build source: `{"a" * 40}`',
                             f'scripts: `{"b" * 40}`', f'Archive SHA-256: `{checksum}`',
                             'Only two-node MNNVL/RDMA smokes run', 'not full CI validation'):
                    self.assertIn(text, summary)
                api = 'https://api.github.com/repos/t-head/pccl/actions/runs/41293442849/attempts/1'
                calls = fixture['api'].call_args_list
                self.assertEqual([call.args[0].full_url for call in calls],
                                 [api, api + '/jobs?per_page=100&page=1'])
                for call in calls:
                    self.assertEqual(call.kwargs, {'timeout': 30})
                    self.assertEqual(call.args[0].get_header('Authorization'), 'Bearer fixture-token')
                    self.assertEqual(call.args[0].get_header('Accept'), 'application/vnd.github+json')
                self.assertEqual(fixture['git'].call_args_list, [
                    mock.call(['git', '-C', 'pccl', 'fetch', '--no-tags', 'origin', 'a' * 40], check=True),
                    mock.call(['git', '-C', 'pccl', 'diff', '--quiet', 'a' * 40, 'HEAD', '--',
                               '.', ':(exclude).github/workflows'], check=False),
                    mock.call(['git', '-C', 'pccl', 'diff', '--quiet', 'a' * 40, 'HEAD', '--',
                               '.github/workflows/scripts/build.sh',
                               '.github/workflows/scripts/pccl_build.sh'], check=False),
                ])

    def test_debug_prepare_rejects_invalid_ids_before_api_access(self):
        for name, values in [('DEBUG_BUILD_RUN_ID', ('0', '../41293442849', '1' * 21, 'abc')),
                             ('DEBUG_BUILD_ATTEMPT', ('', '-1', '01', '1.5'))]:
            for value in values:
                with self.subTest(name=name, value=value), self.debug_build(env={name: value}) as fixture:
                    with self.assertRaisesRegex(SystemExit, 'positive integer IDs'):
                        self.execute_debug_prepare(fixture)
                    fixture['api'].assert_not_called()
                    fixture['git'].assert_not_called()
                    self.assertEqual(fixture['stdout'].getvalue(), '')

    def test_debug_prepare_requires_manual_pr_suite(self):
        for env in ({'GITHUB_EVENT_NAME': 'pull_request'}, {'GITHUB_EVENT_NAME': 'push'},
                    {'GITHUB_EVENT_NAME': 'schedule'}, {'TEST_SUITE': 'nightly'}):
            with self.subTest(env=env), self.debug_build(env=env) as fixture:
                with self.assertRaisesRegex(SystemExit, 'manually dispatched PR-suite'):
                    self.execute_debug_prepare(fixture)
                fixture['api'].assert_not_called()
                fixture['git'].assert_not_called()

    def test_debug_prepare_rejects_untrusted_or_mismatched_source_runs(self):
        for source in ({'id': 41293442850}, {'run_attempt': 2},
                       {'head_repository': {'full_name': 'fork/pccl'}}, {'event': 'pull_request'},
                       {'event': 'pull_request_target'}, {'event': 'workflow_run'},
                       {'path': '.github/workflows/ppu-build-test.yml'},
                       {'path': '.github/workflows/unrelated.yml'}):
            with self.subTest(source=source), self.debug_build(source=source) as fixture:
                with self.assertRaisesRegex(SystemExit, 'trusted build from this repository'):
                    self.execute_debug_prepare(fixture)
                self.assertEqual(fixture['api'].call_count, 1)
                fixture['git'].assert_not_called()
                self.assertEqual(fixture['stdout'].getvalue(), '')

    def test_debug_prepare_requires_successful_standard_build_and_valid_sha(self):
        standard = 'PPU Build and Test / Build (ubuntu24_hggcrt13)'
        batches = [[], [{'name': 'PPU Build and Test / Build & Test (ubuntu24_notsm, ZW-890P)',
                        'conclusion': 'success'}]]
        batches += [[{'name': standard, 'conclusion': status}]
                    for status in ('failure', 'skipped', 'cancelled', None)]
        for jobs in batches:
            with self.subTest(jobs=jobs), self.debug_build(jobs=jobs) as fixture:
                with self.assertRaisesRegex(SystemExit, 'no successful standard build'):
                    self.execute_debug_prepare(fixture)
                self.assertEqual(fixture['api'].call_count, 2)
                fixture['git'].assert_not_called()
        for sha in ('a' * 39, 'g' * 40, 'a' * 41):
            with self.subTest(sha=sha), self.debug_build(source={'head_sha': sha}) as fixture:
                with self.assertRaisesRegex(SystemExit, 'Invalid build source SHA'):
                    self.execute_debug_prepare(fixture)
                fixture['git'].assert_not_called()

    def test_debug_prepare_rejects_changed_source_or_build_scripts(self):
        for diffs in ((1, 0), (0, 1)):
            with self.subTest(diffs=diffs), self.debug_build(diffs=diffs) as fixture:
                with self.assertRaisesRegex(SystemExit, 'PCCL source or build scripts changed'):
                    self.execute_debug_prepare(fixture)
                self.assertEqual(fixture['git'].call_count, 3)
                self.assertFalse(fixture['summary'].exists())
                self.assertEqual(fixture['stdout'].getvalue(), '')

    def test_debug_prepare_rejects_environment_mismatch_and_missing_archive(self):
        for info in ({'config_name': 'ubuntu24_notsm'}, {'docker_image': 'untrusted/image:latest'}):
            with self.subTest(info=info), self.debug_build(info=info) as fixture:
                with self.assertRaisesRegex(SystemExit, 'Build environment does not match'):
                    self.execute_debug_prepare(fixture)
                self.assertFalse(fixture['summary'].exists())
                self.assertEqual(fixture['stdout'].getvalue(), '')
        with self.debug_build(archive=False) as fixture:
            with self.assertRaisesRegex(FileNotFoundError, 'output_hggcrt13.tar.gz'):
                self.execute_debug_prepare(fixture)
            self.assertFalse(fixture['summary'].exists())
            self.assertEqual(fixture['stdout'].getvalue(), '')

    def test_every_node_has_a_distinct_required_report(self):
        matrix = self.matrix('nightly')
        results = self.root / 'results'
        paths = [results / e['test_id'] / f'node-{rank}' / 'test-results.xml'
                 for e in matrix['include'] for rank in range(max(1, e['nnodes']))]
        paths.append(results / 'notsm_run_common_test/test-results.xml')
        self.assertEqual(len(paths), 114)
        self.assertEqual(len(set(paths)), 114)
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('<testsuites tests="1"><testsuite><testcase name="fixture"/></testsuite></testsuites>')
        env = {**os.environ, 'TEST_MATRIX': json.dumps(matrix), 'NAS_RESULT_DIR': str(results),
               'DEBUG_REUSE': 'false'}
        env.pop('GITHUB_STEP_SUMMARY', None)
        command = [sys.executable, '-c', inline_python('summary', 1)]
        result = subprocess.run(command, cwd=self.root, env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for nodes in (2, 8, 16):
            for transport in ('mnnvl', 'socket', 'gdr', 'non_gdr'):
                worker = results / f'run_multinode_test.sh-{transport}-{nodes}nodes' / f'node-{nodes - 1}'
                report = worker / 'test-results.xml'
                passing = report.read_text()
                for tag in ('failure', 'error', 'skipped'):
                    with self.subTest(nodes=nodes, transport=transport, tag=tag):
                        report.write_text('<testsuites tests="1"><testsuite><testcase>'
                                          f'<{tag}/></testcase></testsuite></testsuites>')
                        result = subprocess.run(command, cwd=self.root, env=env, text=True, capture_output=True)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn(str(report), result.stdout)
                report.unlink()
                result = subprocess.run(command, cwd=self.root, env=env, text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(str(report), result.stdout)
                report.write_text(passing)

    def test_pr_smoke_is_fixed_at_two_nodes_and_required(self):
        result, outputs = self.prepare(nodes='[2, 8, 16]')
        self.assertEqual(result.returncode, 0, result.stderr)
        entries = json.loads(outputs['matrix'])['include']
        self.assertEqual([e['nnodes'] for e in entries if e['nnodes'] > 0], [2, 2])
        (self.scripts / 'run_multinode_smoke_test.sh').unlink()
        result, _ = self.prepare()
        self.assertNotEqual(result.returncode, 0)

    def test_pr_reports_reject_empty_malformed_failed_and_skipped_results(self):
        matrix = self.matrix('pr')
        results = self.root / 'results'
        paths = [results / e['test_id'] / f'node-{rank}' / 'test-results.xml'
                 for e in matrix['include'] for rank in range(max(1, e['nnodes']))]
        paths.append(results / 'notsm_run_common_test/test-results.xml')
        self.assertEqual(len(paths), 12)
        self.assertEqual(len(set(paths)), 12)
        passing = '<testsuites tests="1"><testsuite><testcase name="fixture"/></testsuite></testsuites>'
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(passing)
        env = {**os.environ, 'TEST_MATRIX': json.dumps(matrix), 'NAS_RESULT_DIR': str(results),
               'DEBUG_REUSE': 'false'}
        env.pop('GITHUB_STEP_SUMMARY', None)
        command = [sys.executable, '-c', inline_python('summary', 1)]
        result = subprocess.run(command, cwd=self.root, env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        fixtures = ['<invalid', '<testsuites tests="0"/>', passing.replace('tests="1"', 'tests="0"'),
                    passing.replace('tests="1"', 'tests="1" failures="1"')]
        fixtures += [f'<testsuites tests="1"><testsuite><testcase><{tag}/></testcase></testsuite></testsuites>'
                     for tag in ('failure', 'error', 'skipped')]
        for transport in ('mnnvl', 'rdma'):
            for rank in (0, 1):
                worker = results / f'run_multinode_smoke_test.sh-{transport}/node-{rank}/test-results.xml'
                for fixture in fixtures:
                    with self.subTest(transport=transport, rank=rank, fixture=fixture):
                        worker.write_text(fixture)
                        result = subprocess.run(command, cwd=self.root, env=env, text=True, capture_output=True)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn(str(worker), result.stdout)
                worker.unlink()
                result = subprocess.run(command, cwd=self.root, env=env, text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(str(worker), result.stdout)
                worker.write_text(passing)

    def test_debug_summary_requires_all_four_smoke_reports_without_notsm(self):
        with self.debug_build() as fixture:
            matrix = self.execute_debug_prepare(fixture)['matrix']
        results = self.root / 'results'
        paths = [results / entry['test_id'] / f'node-{rank}' / 'test-results.xml'
                 for entry in json.loads(matrix)['include'] for rank in range(entry['nnodes'])]
        self.assertEqual(len(paths), 4)
        self.assertEqual(len(set(paths)), 4)
        passing = '<testsuites tests="1"><testsuite><testcase name="fixture"/></testsuite></testsuites>'
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(passing)
        self.assertFalse((results / 'notsm_run_common_test').exists())
        env = {**os.environ, 'TEST_MATRIX': matrix, 'NAS_RESULT_DIR': str(results), 'DEBUG_REUSE': 'true'}
        env.pop('GITHUB_STEP_SUMMARY', None)

        def summarize():
            return subprocess.run([sys.executable, '-c', inline_python('summary', 1)], cwd=self.root,
                                  env=env, text=True, capture_output=True)

        result = summarize()
        self.assertEqual(result.returncode, 0, result.stderr)
        for path in paths:
            with self.subTest(missing=str(path)):
                path.unlink()
                result = summarize()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f'Missing test report: {path}', result.stdout)
                path.write_text(passing)
        invalid = ['', '<invalid', '<testsuites tests="0"/>',
                   passing.replace('tests="1"', 'tests="1" failures="1"')]
        invalid += [f'<testsuites tests="1"><testsuite><testcase><{tag}/></testcase></testsuite></testsuites>'
                    for tag in ('failure', 'error', 'skipped')]
        for worker in (path for path in paths if path.parent.name == 'node-1'):
            for report in invalid:
                with self.subTest(worker=str(worker), report=report):
                    worker.write_text(report)
                    result = summarize()
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(f'Invalid test report {worker}', result.stdout)
            worker.write_text(passing)
        result = summarize()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_debug_gate_only_allows_build_and_notsm_to_be_skipped(self):
        gate = WORKFLOW['jobs']['tests-gate']
        self.assertEqual(set(gate['needs']), {'prepare', 'build', 'test', 'build-test-notsm', 'summary'})
        self.assertEqual(gate['steps'][0]['env']['DEBUG_REUSE'], '${{ needs.prepare.outputs.debug_reuse }}')
        script = gate['steps'][0]['run'].split('\n', 1)[1].rsplit('\nPY', 1)[0]
        results = {name: {'result': 'skipped' if name in ('build', 'build-test-notsm') else 'success'}
                   for name in gate['needs']}
        candidates = [(None, None)] + [(name, status) for name in results
                                      for status in ('success', 'failure', 'cancelled', 'skipped')
                                      if status != results[name]['result']]
        for name, status in candidates:
            with self.subTest(name=name, status=status):
                candidate = {**results, **({name: {'result': status}} if name else {})}
                result = subprocess.run([sys.executable, '-c', script], text=True, capture_output=True,
                                        env={**os.environ, 'DEBUG_REUSE': 'true',
                                             'JOB_RESULTS': json.dumps(candidate)})
                self.assertEqual(result.returncode == 0, name is None, result.stdout + result.stderr)
                self.assertIn('Debug-only gate: reused build; full CI was not run', result.stdout)
                if name:
                    self.assertIn(f'::error::{name}: {status}', result.stdout)

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
                                        env={**os.environ, 'DEBUG_REUSE': 'false',
                                             'JOB_RESULTS': json.dumps(candidate)})
                self.assertEqual(result.returncode == 0, status == 'success')
        pr = yaml.load((WORKFLOWS / 'pr-test-ppu.yml').read_text(), Loader=yaml.BaseLoader)
        self.assertIn('build-test', pr['jobs']['tests-gate']['needs'])
        self.assertIn('human-review', pr['jobs']['build-test']['needs'])
        nightly = yaml.load((WORKFLOWS / 'nightly-test-ppu.yml').read_text(), Loader=yaml.BaseLoader)
        self.assertEqual(nightly['on']['schedule'][0]['cron'], '0 14 * * *')

    def test_debug_inputs_and_current_checkout_are_manual_only(self):
        pr = yaml.load((WORKFLOWS / 'pr-test-ppu.yml').read_text(), Loader=yaml.BaseLoader)
        prepare = WORKFLOW['jobs']['prepare']
        for name, default in (('debug_build_run_id', ''), ('debug_build_attempt', '1')):
            for inputs in (pr['on']['workflow_dispatch']['inputs'], WORKFLOW['on']['workflow_call']['inputs']):
                self.assertEqual(inputs[name]['type'], 'string')
                self.assertEqual(inputs[name]['default'], default)
            self.assertEqual(pr['jobs']['build-test']['with'][name],
                             "${{ github.event_name == 'workflow_dispatch' && inputs." + name
                             + " || '" + default + "' }}")
            self.assertEqual(prepare['steps'][1]['env'][name.upper()], '${{ inputs.' + name + ' }}')
        for name in ('debug_reuse', 'build_key', 'build_sha256'):
            self.assertEqual(prepare['outputs'][name], '${{ steps.set-matrix.outputs.' + name + ' }}')
        nightly = yaml.load((WORKFLOWS / 'nightly-test-ppu.yml').read_text(), Loader=yaml.BaseLoader)
        self.assertNotIn('debug_build_run_id', nightly['jobs']['build-test']['with'])
        for name in ('build', 'build-test-notsm'):
            self.assertEqual(WORKFLOW['jobs'][name]['if'], "needs.prepare.outputs.debug_reuse != 'true'")
        test = WORKFLOW['jobs']['test']
        checkout, scheduler = test['steps']
        self.assertEqual(checkout['uses'], 'actions/checkout@v4')
        self.assertEqual(checkout['if'], "needs.prepare.outputs.debug_reuse == 'true'")
        self.assertEqual(checkout['with'], {'path': 'pccl', 'persist-credentials': 'false'})
        self.assertEqual(scheduler['uses'], 't-head/ppu-scheduler-action@main')
        self.assertEqual(' '.join(test['if'].split()),
                         "!cancelled() && needs.prepare.result == 'success' && "
                         "(needs.build.result == 'success' || "
                         "(needs.prepare.outputs.debug_reuse == 'true' && needs.build.result == 'skipped'))")

    def test_debug_restore_verifies_before_extracting_and_overlays_current_scripts_only(self):
        command = WORKFLOW['jobs']['test']['steps'][-1]['with']['command']
        self.assertIn('set -eo pipefail', command)
        self.assertIn('NAS_BUILD_DIR="/mnt/wl_nas/devops/pccl-ci/${{ needs.prepare.outputs.build_key }}/build_output"',
                      command)
        self.assertIn('NAS_RESULT_DIR="/mnt/wl_nas/devops/pccl-ci/${{ github.run_id }}/${{ github.run_attempt }}'
                      '/results/${{ matrix.test_id }}/node-${NODE_RANK:-0}"', command)
        summary_env = WORKFLOW['jobs']['summary']['steps'][1]['env']
        self.assertEqual(summary_env['NAS_RESULT_DIR'],
                         '/wl_nas/devops/pccl-ci/${{ github.run_id }}/${{ github.run_attempt }}/results')
        self.assertEqual(summary_env['DEBUG_REUSE'], '${{ needs.prepare.outputs.debug_reuse }}')
        start = command.index('NAS_BUILD_DIR=')
        restore = command[start:command.index('\necho ', start)]
        self.assertLess(restore.index('sha256sum --check'), restore.index('tar -xzf'))
        self.assertLess(restore.index('tar -xzf'), restore.index('cp pccl/'))
        harness = '''set -eo pipefail
mkdir() { :; }
sha256sum() {
  IFS= read -r checksum
  printf 'checksum:%s:%s\\n' "$*" "$checksum"
  return "$CHECKSUM_EXIT"
}
tar() { printf 'extract:%s\\n' "$*"; }
cp() { printf 'overlay:%s\\n' "$*"; }
'''
        archive = '/mnt/wl_nas/devops/pccl-ci/41293442849/1/build_output/output_hggcrt13.tar.gz'
        for debug, exit_code in (('true', 0), ('true', 7), ('false', 7)):
            with self.subTest(debug=debug, checksum_exit=exit_code):
                values = {'needs.prepare.outputs.debug_reuse': debug,
                          'needs.prepare.outputs.build_key': '41293442849/1',
                          'needs.prepare.outputs.build_sha256': 'c' * 64}
                rendered = re.sub(r'\$\{\{\s*(.*?)\s*\}\}', lambda match: values[match[1]], restore)
                result = subprocess.run(['bash', '-c', harness + rendered], text=True, capture_output=True,
                                        env={**os.environ, 'CHECKSUM_EXIT': str(exit_code)})
                self.assertEqual(result.returncode, exit_code if debug == 'true' else 0, result.stderr)
                events = [line for line in result.stdout.splitlines()
                          if line.startswith(('checksum:', 'extract:', 'overlay:'))]
                expected = [f'checksum:--check:{"c" * 64}  {archive}'] if debug == 'true' else []
                if debug == 'false' or exit_code == 0:
                    expected.append(f'extract:-xzf {archive} -C /workspace/source')
                    if debug == 'true':
                        expected.append('overlay:' + ' '.join('pccl/.github/workflows/scripts/' + name for name in
                                        ('run_multinode_smoke_test.sh', 'setup_test_env.sh', 'pr-ut-runner.py'))
                                        + ' output_hggcrt13/')
                self.assertEqual(events, expected)

    def test_node_logs_do_not_mask_test_exit_status(self):
        command = WORKFLOW['jobs']['test']['steps'][-1]['with']['command']
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
        command = WORKFLOW['jobs']['test']['steps'][-1]['with']['command']
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
  for key in PCCL_NET NCCL_NET PCCL_IB_DISABLE NCCL_IB_DISABLE \
    PCCL_MNNVL_ENABLE NCCL_MNNVL_ENABLE PCCL_MNNVL_UUID NCCL_MNNVL_UUID \
    PCCL_MNNVL_CLIQUE_ID NCCL_MNNVL_CLIQUE_ID \
    PCCL_MNNVL_FAKE_ADVANCED_FEATURE NCCL_MNNVL_FAKE_ADVANCED_FEATURE; do
    printf 'SETTING\t%s\t%s\n' "$key" "${!key-UNSET}"
  done
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
        for nodes in (2, 8, 16):
            for transport in ('mnnvl', 'socket', 'gdr', 'non_gdr'):
                for rank in ('0', str(nodes - 1)):
                    with self.subTest(nodes=nodes, transport=transport, rank=rank):
                        result = self.run_script('run_multinode_test.sh', transport,
                                                 {'NNODES': str(nodes), 'WORLD_SIZE': str(nodes),
                                                  'NODE_RANK': rank, 'RANK': rank, 'MASTER_ADDR': hostname})
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn(f'Nightly {transport}: nodes={nodes} ranks={nodes * 8}', result.stdout)
                        self.assertEqual(result.stderr.count(f'DNS\t{hostname}\n'), 1)
                        bootstraps = [line for line in result.stdout.splitlines() if line.startswith('BOOTSTRAP\t')]
                        self.assertEqual(bootstraps, [f'BOOTSTRAP\t{hostname}\t29501'] * 3)
                        commands = [line.split('\t')[1:] for line in result.stdout.splitlines()
                                    if line.startswith('CMD\t')]
                        self.assertEqual(len(commands), 3)
                        runner_transport = 'gdr' if transport == 'mnnvl' else transport
                        for args in commands:
                            for option, value in [('--launch-mode', 'k8s'), ('--nnodes', str(nodes)),
                                                  ('--ranks', str(nodes * 8)), ('--threads', '8'), ('--nprocs', '1'),
                                                  ('--cross-node', runner_transport)]:
                                self.assertEqual(args[args.index(option) + 1], value)
                            self.assertNotIn('--no-data-check', args)
                            self.assertNotIn('LL128', args)
                            self.assertNotIn('Signal', args)
                        network = '' if transport == 'mnnvl' else 'Socket' if transport == 'socket' else 'IB'
                        prefix = f'ENV\t10.0.0.1:29500\t10.0.0.1:29500\t{network}\t'
                        environments = [line for line in result.stdout.splitlines() if line.startswith('ENV\t')]
                        self.assertEqual(len(environments), 3)
                        self.assertTrue(all(line.startswith(prefix + str(self.root / 'pccl-tests/build-k8s') + ':')
                                            for line in environments))

    def test_multinode_transport_overrides_inherited_mnnvl_settings(self):
        for transport in ('mnnvl', 'socket', 'gdr', 'non_gdr'):
            with self.subTest(transport=transport):
                inherited = {prefix + key: 'stale' for prefix in ('PCCL_', 'NCCL_')
                             for key in ('NET', 'IB_DISABLE', 'MNNVL_ENABLE', 'MNNVL_FAKE_ADVANCED_FEATURE',
                                         'MNNVL_UUID', 'MNNVL_CLIQUE_ID')}
                result = self.run_script('run_multinode_test.sh', transport, inherited)
                self.assertEqual(result.returncode, 0, result.stderr)
                settings = [line.split('\t')[1:] for line in result.stdout.splitlines()
                            if line.startswith('SETTING\t')]
                if transport == 'mnnvl':
                    expected = {'NET': 'UNSET', 'IB_DISABLE': '0', 'MNNVL_ENABLE': '1',
                                'MNNVL_FAKE_ADVANCED_FEATURE': '1', 'MNNVL_UUID': '666', 'MNNVL_CLIQUE_ID': '0'}
                else:
                    expected = {'NET': 'Socket' if transport == 'socket' else 'IB',
                                'IB_DISABLE': '1' if transport == 'socket' else '0',
                                'MNNVL_ENABLE': '0', 'MNNVL_FAKE_ADVANCED_FEATURE': '0'}
                for prefix in ('PCCL_', 'NCCL_'):
                    for key, value in expected.items():
                        self.assertEqual([v for k, v in settings if k == prefix + key], [value] * 3)

    def test_multinode_dns_failure_stops_before_collectives(self):
        for transport in ('mnnvl', 'socket', 'gdr', 'non_gdr'):
            for rank in ('0', '15'):
                with self.subTest(transport=transport, rank=rank):
                    result = self.run_script('run_multinode_test.sh', transport,
                                             {'NNODES': '16', 'WORLD_SIZE': '16', 'MOCK_DNS_EXIT_CODE': '7',
                                              'NODE_RANK': rank, 'RANK': rank,
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
        cases = [('run_multinode_test.sh', t) for t in ('mnnvl', 'socket', 'gdr', 'non_gdr')]
        cases.append(('run_nightly_test.sh', 'socket'))
        for script, transport in cases:
            with self.subTest(script=script, transport=transport):
                result = self.run_script(script, transport, overrides={'MOCK_EXIT_CODE': '7',
                                                                      'NNODES': '16', 'WORLD_SIZE': '16'})
                self.assertEqual(result.returncode, 7)
                self.assertEqual(result.stdout.count('CMD\t'), 1)

    def run_nonblocking(self, outputs=None, exit_codes=None, runner_failure=0):
        binaries = self.root / 'mock-bin'
        binaries.mkdir(exist_ok=True)
        for name in ('all_reduce_perf', 'alltoall_perf'):
            binary = binaries / name
            output = (outputs or {}).get(name, '# Out of bounds values : 0 OK\n')
            exit_code = (exit_codes or {}).get(name, 0)
            binary.write_text(
                f'#!{sys.executable}\nimport json, os, signal, sys\n'
                f'print("BENCHMARK\\t" + json.dumps([{name!r}, sys.argv[1:], '
                'os.environ.get("NCCL_COMM_BLOCKING")]), flush=True)\n'
                'if "-B" in sys.argv:\n'
                '    print("invalid option -- B\\nUSAGE: benchmark")\n'
                '    sys.exit(0)\n'
                f'print({output!r}, end="", flush=True)\n'
                f'if {exit_code} < 0:\n'
                '    os.kill(os.getpid(), signal.SIGTERM)\n'
                f'sys.exit({exit_code})\n')
            binary.chmod(0o755)
        harness = r'''
source() { :; }
python3() {
  if [[ "$1" == - ]]; then
    "$REAL_PYTHON" "$@"
    return $?
  fi
  RUNNER_COUNT=$((${RUNNER_COUNT:-0} + 1))
  printf 'RUNNER'; printf '\t%s' "$@"; printf '\n'
  if [[ "$RUNNER_COUNT" == "$RUNNER_FAILURE" ]]; then return 7; fi
}
export -f source python3
bash "$@"
'''
        return subprocess.run(
            ['bash', '-c', harness, 'harness', str(SCRIPTS / 'run_nonblocking_test.sh')],
            cwd=self.root, text=True, capture_output=True, check=False,
            env={**os.environ, 'TEST_ROOT': str(self.root), 'PATH': f'{binaries}:{os.environ["PATH"]}',
                 'REAL_PYTHON': sys.executable, 'RUNNER_FAILURE': str(runner_failure)})

    def test_nonblocking_uses_supported_options_and_preserves_coverage(self):
        result = self.run_nonblocking(outputs={
            'all_reduce_perf': 'PCCL WARN diagnostic warning\n# Out of bounds values : 0 OK\n'})
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [json.loads(line.split('\t', 1)[1]) for line in result.stdout.splitlines()
                    if line.startswith('BENCHMARK\t')]
        self.assertEqual([command[0] for command in commands], ['all_reduce_perf', 'alltoall_perf'])
        for _, args, blocking in commands:
            self.assertEqual(blocking, '0')
            for option, value in [('-b', '4'), ('-e', '1G'), ('-f', '2'), ('-g', '2'), ('-R', '0'),
                                  ('--init_blocking', '0'), ('-c', '1')]:
                self.assertIn(option, args)
                self.assertEqual(args[args.index(option) + 1], value)
            self.assertNotIn('-B', args)
        runners = [line.split('\t')[1:] for line in result.stdout.splitlines() if line.startswith('RUNNER\t')]
        self.assertEqual(len(runners), 3)
        self.assertEqual([(args[args.index('--op') + 1], args[args.index('--data-type') + 1])
                          for args in runners],
                         [('alltoall_perf', 'int8'), ('all_reduce_perf', 'half'), ('all_reduce_perf', 'float')])
        self.assertTrue(all('--non-blocking' in args for args in runners))

    def test_nonblocking_rejects_zero_exit_without_successful_validation(self):
        for index, name in enumerate(('all_reduce_perf', 'alltoall_perf'), 1):
            for output in ('', 'invalid option -- B\nUSAGE: benchmark\n',
                           '# Out of bounds values : 1 FAILED\n', '# Out of bounds values : -1 SKIPPED\n',
                           '# Out of bounds values : 0\n'):
                with self.subTest(binary=name, output=output):
                    result = self.run_nonblocking(outputs={name: output})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('Missing successful data-validation summary', result.stderr)
                    self.assertEqual(result.stdout.count('BENCHMARK\t'), index)
                    self.assertNotIn('RUNNER\t', result.stdout)

    def test_nonblocking_propagates_benchmark_failures_and_signals(self):
        for index, name in enumerate(('all_reduce_perf', 'alltoall_perf'), 1):
            for exit_code in (7, -1):
                with self.subTest(binary=name, exit_code=exit_code):
                    result = self.run_nonblocking(exit_codes={name: exit_code})
                    self.assertEqual(result.returncode, 7 if exit_code == 7 else 1)
                    self.assertEqual(result.stdout.count('BENCHMARK\t'), index)
                    self.assertNotIn('RUNNER\t', result.stdout)

    def test_nonblocking_propagates_runner_failures(self):
        for index in (1, 2, 3):
            with self.subTest(runner=index):
                result = self.run_nonblocking(runner_failure=index)
                self.assertEqual(result.returncode, 7)
                self.assertEqual(result.stdout.count('RUNNER\t'), index)

    def run_smoke(self, output, overrides=None, binary=True, transport='rdma'):
        binary_path = self.root / 'pccl-tests/build-k8s/all_reduce_perf'
        if binary:
            binary_path.parent.mkdir(parents=True, exist_ok=True)
            names = ('NET', 'IB_DISABLE', 'NET_GDR_LEVEL', 'IB_QPS_PER_CONNECTION', 'MNNVL_ENABLE',
                     'MNNVL_UUID', 'MNNVL_CLIQUE_ID', 'MNNVL_FAKE_ADVANCED_FEATURE',
                     'ALGO', 'PROTO', 'DEBUG', 'DEBUG_SUBSYS', 'DEBUG_DEV', 'DEBUG_FILE')
            binary_path.write_text(f'#!{sys.executable}\nimport os, sys\n'
                                   'print(os.environ["SMOKE_OUTPUT"], end="")\n'
                                   f'for name in {names!r}:\n'
                                   '    print("ENV", name, os.environ.get("PCCL_" + name, "unset"), '
                                   'os.environ.get("NCCL_" + name, "unset"))\n'
                                   'print("ADDR", os.environ["PCCL_COMM_ID"], os.environ["NCCL_COMM_ID"])\n'
                                   'sys.exit(int(os.environ.get("SMOKE_EXIT_CODE", "0")))\n')
            binary_path.chmod(0o755)
        env = {**os.environ, 'TEST_ROOT': str(self.root), 'NNODES': '2', 'NPROC_PER_NODE': '8',
               'NODE_RANK': '0', 'RANK': '0', 'WORLD_SIZE': '2', 'MASTER_ADDR': '127.0.0.1',
               'MASTER_PORT': '29500', 'SMOKE_OUTPUT': output, **(overrides or {})}
        args = [] if transport is None else [transport]
        return subprocess.run(['bash', '-c', 'source() { :; }; export -f source; bash "$@"',
                               'harness', str(SCRIPTS / 'run_multinode_smoke_test.sh'), *args],
                              env=env, text=True, capture_output=True, check=False)

    def smoke_output(self, rank='0'):
        if rank == '0':
            return ('8388608 2097152 float sum -1 100 80 150 0 100 80 150 0\n'
                    '# Out of bounds values : 0 OK\n')
        return ''

    def test_smoke_runs_one_checked_allreduce_on_both_nodes(self):
        for transport in ('mnnvl', 'rdma'):
            for rank in ('0', '1'):
                with self.subTest(transport=transport, rank=rank):
                    result = self.run_smoke(self.smoke_output(rank),
                                            {'NODE_RANK': rank, 'RANK': rank}, transport=transport)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('ADDR 127.0.0.1:29500 127.0.0.1:29500', result.stdout)
                    self.assertIn('-t 8 -g 1 -b 8M -e 8M -d float -o sum -n 20 -w 5 -c 1 -k 2', result.stdout)
                    self.assertIn('-N 1 --init_only 0 --report_type 0', result.stdout)
                    self.assertEqual(result.stdout.count('/build-k8s/all_reduce_perf'), 1)
                    self.assertNotIn('mpirun', result.stdout)
                    self.assertIn(f'AllReduce {transport} smoke passed on node {rank}', result.stdout)

    def test_smoke_transport_environments_override_inherited_settings(self):
        common = {'ALGO': 'Ring', 'PROTO': 'Simple', 'DEBUG': 'INFO', 'DEBUG_SUBSYS': 'INIT,P2P,NET',
                  'DEBUG_DEV': '-1', 'DEBUG_FILE': 'unset'}
        expected = {
            'mnnvl': {'NET': 'unset', 'IB_DISABLE': '1', 'MNNVL_ENABLE': '1', 'MNNVL_UUID': '666',
                      'MNNVL_CLIQUE_ID': '0', 'MNNVL_FAKE_ADVANCED_FEATURE': '1'},
            'rdma': {'NET': 'IB', 'IB_DISABLE': '0', 'MNNVL_ENABLE': '0', 'MNNVL_FAKE_ADVANCED_FEATURE': '0',
                     'NET_GDR_LEVEL': 'SYS', 'IB_QPS_PER_CONNECTION': '8'},
        }
        for transport, values in expected.items():
            with self.subTest(transport=transport):
                values = {**common, **values}
                inherited = {prefix + key: 'stale' for key in values for prefix in ('PCCL_', 'NCCL_')}
                inherited.update(PCCL_DEBUG_DEV='0', NCCL_DEBUG_DEV='0')
                result = self.run_smoke(self.smoke_output(), inherited, transport=transport)
                self.assertEqual(result.returncode, 0, result.stderr)
                for name, value in values.items():
                    self.assertIn(f'ENV {name} {value} {value}\n', result.stdout)

    def test_smoke_requires_an_explicit_supported_transport(self):
        for transport in (None, '', 'socket', 'gdr', 'unknown'):
            with self.subTest(transport=transport):
                result = self.run_smoke('', binary=False, transport=transport)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Specify a two-node smoke transport', result.stderr)
                self.assertNotIn('AllReduce binary', result.stderr)

    def test_smoke_rejects_noop_and_wrong_data_on_either_layout(self):
        for transport in ('mnnvl', 'rdma'):
            passing = self.smoke_output()
            for output in ('', 'Usage: all_reduce_perf\n',
                           passing.replace('# Out of bounds values : 0 OK', ''),
                           passing.replace(': 0 OK', ': 1 FAILED'),
                           passing.replace('150 0 100', '150 1 100'),
                           passing.replace('150 0\n', '150 1\n'),
                           passing.replace('150 0', '150 N/A'), passing * 2,
                           passing.replace('8388608 2097152 float sum', '32 8 float sum')):
                with self.subTest(transport=transport, output=output):
                    self.assertNotEqual(self.run_smoke(output, transport=transport).returncode, 0)

    def test_smoke_transport_logs_are_diagnostic_only(self):
        logs = ('',
                'PCCL INFO MNNVL 0 cliqueId 0 cliqueSize 8 cliqueRank 0 nvlDomainSize 8\n',
                'PCCL INFO Channel 00/0 : 0[1000] -> 8[1000] [send] via NET/IB/0\n',
                'PCCL INFO Channel 00/0 : 0[1000]  -> 8[1000] [send] via NET/Socket/0\n')
        for transport in ('mnnvl', 'rdma'):
            for rank in ('0', '1'):
                for log in logs:
                    with self.subTest(transport=transport, rank=rank, log=log):
                        result = self.run_smoke(log + self.smoke_output(rank),
                                                {'NODE_RANK': rank, 'RANK': rank}, transport=transport)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn(log, result.stdout)

    def test_smoke_propagates_failure_on_either_node(self):
        for transport in ('mnnvl', 'rdma'):
            for rank in ('0', '1'):
                result = self.run_smoke(self.smoke_output(rank),
                                        {'NODE_RANK': rank, 'RANK': rank, 'SMOKE_EXIT_CODE': '7'},
                                        transport=transport)
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
        env = {'TEST_ROOT': str(self.root), 'NODE_RANK': '0', 'MASTER_ADDR': hostname, 'MASTER_PORT': '29500',
               'PCCL_SMOKE_TRANSPORT': 'rdma'}
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
                kwargs['timeout'] = 2
            return original_run(*args, **kwargs)

        for partial in (False, True):
            with self.subTest(partial=partial):
                pid_file.unlink(missing_ok=True)
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
