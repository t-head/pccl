# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from catalog import cases, performance_cases, source_inputs
from execute import execution_provenance, run_shard, write_json
from manifest import build_manifest, digest, write_bundle
from metrics import parse_metrics, write_metrics
import single_node_ci as ci
from test_metrics import synthetic_table


class SingleNodeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(os.environ, {key: value for key, value in os.environ.items()
                                 if not key.startswith(('PCCL_', 'NCCL_', 'HGGC_', 'OMPI_', 'UCX_'))}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.inputs = dict(test_suite='pr', single_node=True, debug_only=False,
                           test_types=['functional'], node_counts=[2, 8, 16], shards_per_pool=1,
                           timings_sha256=digest({}), pccl_revision='a' * 40,
                           pccl_tests_revision='394556512e87259e9ffa25225632c46b0a0bbcd5',
                           **source_inputs())
        specs = [spec for spec in cases() + performance_cases() if spec['nodes'] == 1]
        self.manifest = build_manifest(specs, self.inputs)
        write_bundle(self.manifest, self.root / 'bundle')
        self.path = self.root / 'bundle' / 'manifest.json'
        self.results = self.root / 'results'
        self.output = self.root / 'public'
        self.provenance = execution_provenance()
        self.metric_inputs = {**self.inputs, **self.provenance}
        self.text = synthetic_table('8388608 2097152 float sum -1 100 80 150 0 100 80 150 0', sizes=(8388608,))

    def write_results(self):
        for shard in self.manifest['shards']:
            directory = self.results / shard['build_profile'] / shard['shard_id']
            directory.mkdir(parents=True)
            rows = []
            for row in shard['cases']:
                key = row['case_key']
                case = directory / key / 'attempt-1'
                case.mkdir(parents=True)
                log_ref = f'{key}/attempt-1/command.log'
                result = dict(case_key=key, attempt=1, status='PASSED', returncode=0, error='', duration=0.01,
                              legacy_source=row['spec']['legacy_source'], command_log=log_ref,
                              diagnostic='PRIVATE HOST must not be published')
                (case / 'command.log').write_text(self.text)
                write_json(case / 'command-process.json', dict(active=False, state='finished', return_code=0))
                if row['spec']['test_type'] == 'performance':
                    result.update(data_status='VALID', metrics_log=f'{key}/attempt-1/metrics.jsonl')
                    metrics = parse_metrics(self.text, row['spec'], key, self.metric_inputs, log_ref)
                    write_metrics(metrics, case)
                write_json(case / 'result.json', result)
                rows.append(result)
            (directory / 'results.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            keys = [row['case_key'] for row in rows]
            write_json(directory / 'index.json', dict(
                shard_id=shard['shard_id'], expected=keys, completed=keys, failures=0,
                cancelled=False, artifact_error='', selection_sha256=self.manifest['selection_sha256'],
                execution=self.provenance))
            write_json(directory / 'manifest.json', self.manifest)

    def test_complete_synthetic_results_publish_only_safe_fields(self):
        self.write_results()
        self.assertEqual(ci.summarize(self.path, self.results, self.output), 0)
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertTrue(summary['complete'])
        self.assertEqual((summary['expected'], summary['passed']), (18, 18))
        self.assertFalse((self.output / 'metrics.jsonl').exists())
        self.assertNotIn('PRIVATE HOST', (self.output / 'summary.json').read_text())
        self.assertFalse((self.output / 'command.log').exists())

    def test_full_synthetic_execution_roundtrip(self):
        build = self.root / 'synthetic-build'
        build.mkdir()
        binary = build / 'bin'
        binary.mkdir()
        source = f'#!{sys.executable}\nprint({self.text!r}, end="")\n'
        (build / 'run_test.py').write_text(source)
        names = {'mpirun'} | {Path(row['spec']['argv'][0]).name for shard in self.manifest['shards']
                             for row in shard['cases'] if row['spec']['argv'][0].startswith('{binary_dir}')}
        for name in names:
            path = binary / name
            path.write_text(source)
            path.chmod(0o700)
        for shard in self.manifest['shards']:
            directory = self.results / shard['build_profile'] / shard['shard_id']
            self.assertEqual(run_shard(self.path, shard['shard_id'], build, binary, directory, []), 0)
        self.assertEqual(ci.summarize(self.path, self.results, self.output), 0)
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertEqual(summary['passed'], 18)
        self.assertTrue(summary['complete'])
        self.assertEqual(len(summary['executions']), 2)

    def test_missing_results_fail_with_complete_expected_case_list(self):
        self.assertEqual(ci.summarize(self.path, self.results, self.output), 1)
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertEqual((summary['expected'], summary['passed']), (18, 0))
        self.assertFalse(summary['complete'])

    def test_duplicate_results_and_cleanup_errors_cannot_pass(self):
        self.write_results()
        path = next(self.results.rglob('results.jsonl'))
        path.write_text(path.read_text() + path.read_text().splitlines()[0] + '\n')
        self.assertEqual(ci.summarize(self.path, self.results, self.output), 1)
        index = next(self.results.rglob('index.json'))
        content = json.loads(index.read_text())
        content['artifact_error'] = 'PRIVATE cleanup failure'
        index.write_text(json.dumps(content))
        self.assertEqual(ci.summarize(self.path, self.results, self.root / 'public2'), 1)
        self.assertNotIn('PRIVATE', (self.root / 'public2/summary.json').read_text())

    def test_wrong_run_executor_or_incomplete_index_fails_gate(self):
        self.write_results()
        path = next(self.results.rglob('index.json'))
        original = json.loads(path.read_text())
        mutations = [dict(selection_sha256='0' * 64), dict(cancelled=True), dict(failures=False),
                     dict(expected=original['expected'] * 2), dict(completed=original['completed'][:-1]),
                     dict(execution={**self.provenance, 'executor_sha256': '0' * 64}),
                     dict(execution={**self.provenance, 'executor_python_version': ''}), dict(execution=None)]
        for number, changes in enumerate(mutations):
            write_json(path, {**original, **changes})
            with self.subTest(changes=changes):
                self.assertEqual(ci.summarize(self.path, self.results, self.root / f'invalid-index-{number}'), 1)
        write_json(path, original)
        snapshot = path.with_name('manifest.json')
        changed = copy.deepcopy(self.manifest)
        changed['inputs']['run_id'] = 'different-run'
        specs = [row['spec'] for shard in changed['shards'] for row in shard['cases']]
        write_json(snapshot, build_manifest(specs, changed['inputs']))
        self.assertEqual(ci.summarize(self.path, self.results, self.output), 1)

    def test_finished_result_requires_command_and_check_evidence(self):
        self.write_results()
        path = next(self.results.rglob('command-process.json'))
        original = path.read_text()
        for number, record in enumerate(({}, dict(active=True, state='finished', return_code=0),
                                        dict(active=False, state='cleanup_failed', return_code=0),
                                        dict(active=False, state='finished', return_code=False),
                                        dict(active=False, state='finished', return_code=7))):
            write_json(path, record)
            with self.subTest(record=record):
                self.assertEqual(ci.summarize(self.path, self.results, self.root / f'bad-record-{number}'), 1)
        path.write_text(original)
        path.unlink()
        self.assertEqual(ci.summarize(self.path, self.results, self.output), 1)

    def test_exit_only_source_case_does_not_fabricate_a_log_assertion(self):
        self.write_results()
        shard = next(shard for shard in self.manifest['shards'] if shard['test_type'] == 'functional'
                     and shard['build_profile'] == 'standard')
        row = shard['cases'][0]
        path = self.results / shard['build_profile'] / shard['shard_id'] / row['case_key'] / 'attempt-1/command.log'
        path.write_text('source pipeline checks only the successful return code\n')
        self.assertEqual(ci.summarize(self.path, self.results, self.output), 0)
        self.assertEqual(json.loads((self.output / 'summary.json').read_text())['passed'], 18)

    def test_invalid_case_result_or_json_cannot_pass(self):
        self.write_results()
        path = next(self.results.rglob('results.jsonl'))
        original = path.read_text()
        rows = [json.loads(line) for line in original.splitlines()]
        for number, changes in enumerate((dict(status='UNKNOWN'), dict(error='PRIVATE error'),
                                          dict(returncode=False), dict(attempt=2), dict(duration=-1),
                                          dict(duration=10 ** 400), dict(command_log='other/log'),
                                          dict(legacy_source='other'))):
            altered = copy.deepcopy(rows)
            altered[0].update(changes)
            path.write_text(''.join(json.dumps(row) + '\n' for row in altered))
            write_json(path.parent / altered[0]['case_key'] / 'attempt-1/result.json', altered[0])
            with self.subTest(changes=changes):
                self.assertEqual(ci.summarize(self.path, self.results, self.root / f'bad-result-{number}'), 1)
        for number, text in enumerate(('[null]\n', 'null\n', '{"case_key":"x","case_key":"y"}\n',
                                      '{"case_key":"x","duration":NaN}\n')):
            path.write_text(text)
            with self.subTest(text=text):
                self.assertEqual(ci.summarize(self.path, self.results, self.root / f'bad-json-{number}'), 1)

    def test_multinode_or_incomplete_selection_is_rejected(self):
        multinode = copy.deepcopy(cases())
        multinode[0].update(nodes=2, parameters={**multinode[0]['parameters'], 'ranks': 4})
        for name, specs in (('partial', cases()[:1]), ('multinode', multinode)):
            directory = self.root / name
            write_bundle(build_manifest(specs, self.inputs), directory)
            with self.assertRaises(ValueError):
                ci.selection(directory / 'manifest.json')
        specs = [row['spec'] for shard in self.manifest['shards'] for row in shard['cases']]
        for field in ('legacy_source', 'tier'):
            altered = copy.deepcopy(specs)
            altered[0][field] = 'other' if field == 'legacy_source' else 'nightly'
            directory = self.root / field
            write_bundle(build_manifest(altered, self.inputs), directory)
            with self.assertRaises(ValueError):
                ci.selection(directory / 'manifest.json')

    def test_run_rejects_sdk_library_and_stops_after_shard_failure(self):
        build = self.root / 'build'
        with patch.object(ci.subprocess, 'run') as linked, patch.object(ci, 'run_shard') as run:
            linked.return_value.stdout = 'libpccl.so.2 => /usr/local/PPU_SDK/lib/libpccl.so.2 (0x123)'
            with self.assertRaisesRegex(ValueError, 'freshly built'):
                ci.run(self.path, 'standard', build, self.results)
            run.assert_not_called()
            linked.return_value.stdout = f'libpccl.so.2 => {build}/lib/libpccl.so.2 (0x123)'
            run.return_value = 1
            self.assertEqual(ci.run(self.path, 'standard', build, self.results), 1)
            self.assertEqual(run.call_count, 1)


if __name__ == '__main__':
    unittest.main()
