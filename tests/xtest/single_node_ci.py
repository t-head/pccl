# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
from pathlib import Path
import re
import subprocess
import sys

from catalog import cases, performance_cases
from command import check_command
from execute import execution_provenance, read_jsonl, run_shard, verified_metrics, write_json
from manifest import case_key, digest, load_manifest, read_json, validate_selection
from metrics import summarize_metrics, write_metrics


def selection(path):
    manifest = load_manifest(path)
    inputs = manifest['inputs']
    if (inputs.get('test_suite') != 'pr' or inputs.get('single_node') is not True
            or inputs.get('debug_only') is not False):
        raise ValueError('Requires the explicit single-node sanity selection')
    validate_selection(manifest, inputs)
    expected = {case_key(spec): spec for spec in cases() + performance_cases() if spec['nodes'] == 1}
    rows = [row for shard in manifest['shards'] for row in shard['cases']]
    if digest({row['case_key']: row['spec'] for row in rows}) != digest(expected) or len(rows) != len(expected):
        raise ValueError('Single-node sanity coverage differs from the catalog')
    if any(row['spec']['nodes'] != 1 for row in rows):
        raise ValueError('Multinode cases are not authorized in this workflow')
    return manifest


def run(manifest_path, profile, test_root, output):
    manifest = selection(manifest_path)
    root = Path(test_root).resolve()
    binary = root / 'pccl-tests' / 'build'
    linked = subprocess.run(['ldd', str(binary / 'all_reduce_perf')], capture_output=True, text=True, check=True)
    libraries = re.findall(r'lib(?:pccl|nccl)\.so[^\s]*\s+=>\s+(\S+)', linked.stdout)
    if not libraries or any(Path(path).resolve().parent != (root / 'lib').resolve() for path in libraries):
        raise ValueError('Benchmark is not linked to the freshly built PCCL library')
    selected = [shard for shard in manifest['shards'] if shard['build_profile'] == profile]
    if not selected:
        raise ValueError('No shards for the requested build profile')
    Path(output).mkdir(parents=True, exist_ok=False)
    for shard in selected:
        if run_shard(manifest_path, shard['shard_id'], root, binary, Path(output) / shard['shard_id'], []):
            return 1
    return 0


def summarize(manifest_path, results, output):
    manifest = selection(manifest_path)
    published, measurements, errors, executions = [], [], [], {}
    source_hash = execution_provenance()['executor_sha256']
    for shard in manifest['shards']:
        directory = Path(results) / shard['build_profile'] / shard['shard_id']
        expected = {row['case_key']: row for row in shard['cases']}
        actual, inputs = {}, {}
        valid_shard = False
        try:
            index = read_json(directory / 'index.json')
            rows = read_jsonl(directory / 'results.jsonl')
            actual = {row['case_key']: row for row in rows}
            execution = index['execution']
            if (len(actual) != len(rows) or list(actual) != list(expected)
                    or index['shard_id'] != shard['shard_id'] or index['expected'] != list(expected)
                    or index['completed'] != list(expected) or type(index['failures']) is not int
                    or index['failures'] != 0 or index['cancelled'] is not False or index['artifact_error'] != ''
                    or index['selection_sha256'] != manifest['selection_sha256']
                    or digest(load_manifest(directory / 'manifest.json')) != digest(manifest)
                    or set(execution) != {'executor_sha256', 'executor_python_version'}
                    or execution['executor_sha256'] != source_hash
                    or not re.fullmatch(r'3\.[0-9]+\.[0-9]+', execution['executor_python_version'])):
                raise ValueError('Shard evidence does not match this manifest and executor')
            inputs = {**manifest['inputs'], **execution}
            executions[shard['shard_id']] = execution
            valid_shard = True
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            errors.append(shard['shard_id'])
        for key, row in expected.items():
            passed = False
            if valid_shard:
                try:
                    result = actual[key]
                    case_dir = directory / key / 'attempt-1'
                    log_ref = f'{key}/attempt-1/command.log'
                    if (result['status'] != 'PASSED' or type(result['returncode']) is not int
                            or result['returncode'] != 0 or result['error'] != ''
                            or type(result['attempt']) is not int or result['attempt'] != 1
                            or type(result['duration']) not in (int, float)
                            or not math.isfinite(result['duration']) or result['duration'] < 0
                            or result['command_log'] != log_ref
                            or result['legacy_source'] != row['spec']['legacy_source']
                            or digest(read_json(case_dir / 'result.json')) != digest(result)):
                        raise ValueError('Case result is incomplete or inconsistent')
                    runtime = dict(result_dir=str(case_dir), inputs=inputs, case_key=key, command_log_ref=log_ref)
                    check_command(row['spec'], runtime, read_json(case_dir / 'command-process.json'))
                    if row['spec']['test_type'] == 'performance':
                        if (result['data_status'] != 'VALID'
                                or result['metrics_log'] != f'{key}/attempt-1/metrics.jsonl'):
                            raise ValueError('Missing valid performance result')
                        measurements.extend(verified_metrics(case_dir, row, inputs, log_ref))
                    passed = True
                except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError):
                    pass
            published.append(dict(case_key=key, build_profile=shard['build_profile'],
                                  test_type=row['spec']['test_type'], status='PASSED' if passed else 'FAILED'))
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    passed = sum(row['status'] == 'PASSED' for row in published)
    complete = passed == len(published) and not errors
    artifact_error = ''
    try:
        if measurements:
            summaries = summarize_metrics(measurements)
            write_metrics(measurements, target)
            write_json(target / 'performance-summary.json', dict(complete=complete, measurements=summaries))
    except (OSError, ValueError) as exception:
        complete, artifact_error = False, type(exception).__name__
    write_json(target / 'summary.json', dict(scope='sanity-single-node', complete=complete,
                                           expected=len(published), passed=passed,
                                           failed_shards=errors, artifact_error=artifact_error,
                                           cases=published, inputs=manifest['inputs'], executions=executions))
    print(f'Single-node sanity: {passed}/{len(published)} passed; complete={complete}')
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('run', 'summary'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=('standard', 'notsm'))
    parser.add_argument('--test-root', type=Path)
    parser.add_argument('--results', type=Path)
    args = parser.parse_args()
    if args.operation == 'run':
        if args.profile is None or args.test_root is None:
            parser.error('run requires --profile and --test-root')
        return run(args.manifest, args.profile, args.test_root, args.output)
    if args.results is None:
        parser.error('summary requires --results')
    return summarize(args.manifest, args.results, args.output)


if __name__ == '__main__':
    sys.exit(main())
