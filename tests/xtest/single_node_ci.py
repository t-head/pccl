# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import subprocess
import sys

from catalog import cases, load_catalog, performance_cases
from command import check_command
from execute import (execution_provenance, read_jsonl, run_shard, verified_metrics, verified_runtime,
                     verify_lifecycle, write_json)
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


def matrix(manifest_path):
    manifest = selection(manifest_path)
    include = []
    for shard in manifest['shards']:
        runtime = shard.get('runtime')
        resource = shard.get('resource')
        if not runtime or not resource or resource['nodes'] != 1:
            raise ValueError('Matrix requires explicit single-node Suite resources')
        include.append(dict(shard_id=shard['shard_id'], profile=shard['build_profile'],
                            build_script='build.sh' if shard['build_profile'] == 'standard' else 'notsm_build.sh',
                            image=runtime['image'], node_selector=runtime['node_selector'],
                            nnodes=0, nproc_per_node=resource['cards_per_node']))
    return {'include': include}


def run(manifest_path, profile, test_root, output, shard_id):
    manifest = selection(manifest_path)
    selected = [shard for shard in manifest['shards']
                if shard['build_profile'] == profile and shard['shard_id'] == shard_id]
    if len(selected) != 1:
        raise ValueError('Select exactly one resource batch for this allocation')
    root = Path(test_root).resolve()
    binary = root / 'pccl-tests' / 'build'
    linked = subprocess.run(['ldd', str(binary / 'all_reduce_perf')], capture_output=True, text=True, check=True)
    libraries = re.findall(r'lib(?:pccl|nccl)\.so[^\s]*\s+=>\s+(\S+)', linked.stdout)
    if not libraries or any(Path(path).resolve().parent != (root / 'lib').resolve() for path in libraries):
        raise ValueError('Benchmark is not linked to the freshly built PCCL library')
    Path(output).mkdir(parents=True, exist_ok=True)
    return run_shard(manifest_path, shard_id, root, binary, Path(output) / shard_id, [])


def case_summaries(manifest, published, results):
    """Project execution evidence back onto its source Case for reviewers."""
    definitions = {row['id']: row for row in load_catalog()['cases']}
    templates = {}
    for shard in manifest['shards']:
        for row in shard['cases']:
            templates[(row['case_key'], shard['build_profile'])] = (shard, row['spec'])
    grouped = defaultdict(list)
    case_contexts = {}
    for result in published:
        case_id = result['source_context']['case_id']
        case_contexts[case_id] = result['source_context']
        shard, spec = templates[(result['case_key'], result['build_profile'])]
        relative = Path(shard['build_profile']) / shard['shard_id'] / result['case_key'] / 'attempt-1'
        reproduction = relative / 'command-reproduction.txt'
        grouped[case_id].append({
            'case_key': result['case_key'], 'build_profile': result['build_profile'],
            'test_type': result['test_type'], 'status': result['status'],
            'argv': spec['argv'], 'env': spec['env'],
            'resources': {name: spec[name] for name in
                          ('nodes', 'cards_per_node', 'processes_per_node',
                           'threads_per_process', 'devices_per_thread')},
            'result': str(relative / 'result.json'),
            'reproduction': str(reproduction) if (Path(results) / reproduction).is_file() else None,
        })
    cases = []
    for case_id, executions in sorted(grouped.items()):
        definition = definitions[case_id]
        contexts = set()
        for row in published:
            if row['source_context']['case_id'] == case_id:
                _, spec = templates[(row['case_key'], row['build_profile'])]
                contexts.add(spec['source_context']['source_command'])
        if len(contexts) != 1:
            raise ValueError('Case result contains inconsistent source commands')
        cases.append({'id': case_id, 'name': definition['name'], 'sha256': definition['sha256'],
                      'source_command': next(iter(contexts)),
                      'source_context': case_contexts[case_id],
                      'executions': sorted(executions, key=lambda row: (row['build_profile'], row['case_key']))})
    return {'schema_version': 1, 'scope': 'source-case-results', 'cases': cases}


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
            shard_runtime = verified_runtime(directory, index)
            if shard.get('suite_runs'):
                if index.get('lifecycle_error') != '':
                    raise ValueError('Suite lifecycle did not complete cleanly')
                verify_lifecycle(directory, shard, shard_runtime)
            inputs = {**manifest['inputs'], **execution}
            executions[shard['shard_id']] = execution
            valid_shard = True
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exception:
            print(f'Rejected shard {shard["shard_id"]}: {type(exception).__name__}: {exception}',
                  flush=True)
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
                    if 'source_context' in row['spec']:
                        context = {name: row['spec']['source_context'][name]
                                   for name in ('plan_id', 'suite_id', 'case_id', 'suite_run_id')}
                        if result.get('source_context') != context:
                            raise ValueError('Case result lost its Plan/Suite context')
                    runtime = dict(shard_runtime, result_dir=str(case_dir), inputs=inputs,
                                   execution_result_dir=str(Path(shard_runtime['result_dir']) / key / 'attempt-1'),
                                   case_key=key, command_log_ref=log_ref)
                    check_command(row['spec'], runtime, read_json(case_dir / 'command-process.json'))
                    if row['spec']['test_type'] == 'performance':
                        if (result['data_status'] != 'VALID'
                                or result['metrics_log'] != f'{key}/attempt-1/metrics.jsonl'):
                            raise ValueError('Missing valid performance result')
                        measurements.extend(verified_metrics(case_dir, row, inputs, log_ref))
                    passed = True
                except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exception:
                    print(f'Rejected case {key}: {type(exception).__name__}: {exception}', flush=True)
            context = row['spec'].get('source_context', {})
            published.append(dict(case_key=key, build_profile=shard['build_profile'],
                                  test_type=row['spec']['test_type'], status='PASSED' if passed else 'FAILED',
                                  source_context={name: context[name] for name in
                                                  ('plan_id', 'suite_id', 'case_id', 'suite_run_id') if name in context}))
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
    case_summary = case_summaries(manifest, published, results)
    case_summary['complete'] = complete
    write_json(target / 'case-summary.json', case_summary)
    write_json(target / 'summary.json', dict(scope='sanity-single-node', complete=complete,
                                           expected=len(published), passed=passed,
                                           failed_shards=errors, artifact_error=artifact_error,
                                           cases=published, inputs=manifest['inputs'], executions=executions))
    print(f'Single-node sanity: {passed}/{len(published)} passed; complete={complete}')
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('run', 'summary', 'matrix'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--profile', choices=('standard', 'notsm'))
    parser.add_argument('--shard')
    parser.add_argument('--test-root', type=Path)
    parser.add_argument('--results', type=Path)
    args = parser.parse_args()
    if args.operation == 'matrix':
        print(json.dumps(matrix(args.manifest)))
        return 0
    if args.output is None:
        parser.error('run and summary require --output')
    if args.operation == 'run':
        if args.profile is None or args.test_root is None or args.shard is None:
            parser.error('run requires --profile, --shard and --test-root')
        return run(args.manifest, args.profile, args.test_root, args.output, args.shard)
    if args.results is None:
        parser.error('summary requires --results')
    return summarize(args.manifest, args.results, args.output)


if __name__ == '__main__':
    sys.exit(main())
