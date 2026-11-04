# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile


PCCL_TESTS_REVISION = '394556512e87259e9ffa25225632c46b0a0bbcd5'
RUNNER_HASHES = {
    'run_test.py': 'fdbdf7623703157b49b9605491fb1cb602fe50ec931ae6763c2645bef7c6b0fc',
    'run_test_config.py': 'd58194c5c210ebaabe1ed42d7c2ced924c5dde22081285c22516e1ccdf611b1a',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def case_key(spec):
    identity = {key: value for key, value in spec.items() if key not in ('legacy_source', 'tier')}
    return f"{spec['group']}-{digest(identity)[:20]}"


def measurement_identity(spec):
    identity = {key: value for key, value in spec.items()
                if key not in ('tier', 'legacy_source', 'group', 'timeout', 'checks')}
    identity['performance'] = {key: value for key, value in spec.get('performance', {}).items()
                               if key != 'sample_index'}
    return digest(identity)


def validate_spec(spec):
    fields = {'group', 'tier', 'test_type', 'build_profile', 'nodes', 'cards_per_node',
              'processes_per_node', 'threads_per_process', 'devices_per_thread',
              'argv', 'env', 'timeout', 'checks', 'legacy_source', 'parameters'}
    if (not isinstance(spec, dict) or not fields <= spec.keys()
            or spec.keys() - fields - {'performance'}):
        raise ValueError('Invalid Case fields')
    if spec['tier'] not in ('sanity', 'nightly') or spec['test_type'] not in ('functional', 'performance'):
        raise ValueError('Unsupported tier or test type')
    if not isinstance(spec['parameters'], dict):
        raise ValueError('Parameters must be a record')
    if spec['test_type'] == 'performance':
        performance = spec.get('performance', {})
        if (not isinstance(performance, dict)
                or set(performance) != {'expected_sizes', 'time_kind', 'rank_reduction', 'sample_index'}):
            raise ValueError('Performance configuration must be a complete record')
        sizes = performance['expected_sizes']
        if (spec['parameters'].get('op') != 'all_reduce_perf' or spec['parameters'].get('dtype') != 'float'
            or spec['parameters'].get('redop') != 'sum' or not isinstance(sizes, list) or len(sizes) != 1
            or any(type(size) is not int or size <= 0 or size % 4 for size in sizes)
            or performance['time_kind'] != 'cpu_time' or performance['rank_reduction'] != 'MIN'
            or type(performance['sample_index']) is not int or performance['sample_index'] < 0):
            raise ValueError('Unsupported or incomplete performance measurement')
    elif 'performance' in spec:
        raise ValueError('Performance configuration requires a performance Case')
    if (spec['build_profile'] not in ('standard', 'notsm') or not isinstance(spec['group'], str)
            or not re.fullmatch(r'[a-zA-Z0-9_-]+', spec['group'])):
        raise ValueError('Invalid build profile or group')
    transport = spec['parameters'].get('transport', 'local')
    if not isinstance(transport, str) or not re.fullmatch(r'[a-zA-Z0-9_-]+', transport):
        raise ValueError('Invalid transport')
    for name in ('nodes', 'cards_per_node', 'processes_per_node', 'threads_per_process',
                 'devices_per_thread', 'timeout'):
        if type(spec[name]) is not int or spec[name] <= 0:
            raise ValueError(f'Invalid {name}')
    devices = spec['processes_per_node'] * spec['threads_per_process'] * spec['devices_per_thread']
    if devices > spec['cards_per_node']:
        raise ValueError('Execution layout exceeds the allocation')
    ranks = spec['parameters'].get('ranks')
    if ranks is not None and (type(ranks) is not int or ranks != spec['nodes'] * devices):
        raise ValueError('Rank count differs from the execution layout')
    if (not isinstance(spec['argv'], list) or not spec['argv']
            or any(not isinstance(arg, str) or not arg or '\x00' in arg for arg in spec['argv'])):
        raise ValueError('Expected a nonempty argv without NUL')
    if (not isinstance(spec['checks'], list) or not spec['checks']
            or any(not isinstance(check, str) for check in spec['checks'])
            or len(set(spec['checks'])) != len(spec['checks'])
            or set(spec['checks']) - {'exit_code', 'validation_summary', 'smoke_8mib', 'performance'}):
        raise ValueError('Unsupported result checks')
    if ('performance' in spec['checks']) != (spec['test_type'] == 'performance'):
        raise ValueError('Performance checks require a performance Case')
    if not isinstance(spec['env'], dict):
        raise ValueError('Environment must be a record')
    for key, value in spec['env'].items():
        if (not isinstance(key, str) or not re.fullmatch(r'(?:PCCL|NCCL|HGGC|OMPI|UCX)_[A-Za-z0-9_]+', key)
                or not (value is None or isinstance(value, str) and '\x00' not in value)):
            raise ValueError('Invalid environment override')
    for token in spec['argv'] + [value for value in spec['env'].values() if value is not None]:
        literal = re.sub(r'\{(?:python|test_root|binary_dir|result_dir|mpi_options)\}', '', token)
        if '{' in literal or '}' in literal:
            raise ValueError('Unsupported command placeholder')
    if not isinstance(spec['legacy_source'], str) or not spec['legacy_source']:
        raise ValueError('Missing coverage reference')
    if spec['test_type'] == 'performance':
        parameters = spec['parameters']
        if (spec['nodes'] != 1 or spec['processes_per_node'] != 1
            or parameters.get('ranks') != devices or devices < 2
            or type(parameters.get('iters')) is not int or parameters['iters'] <= 0
            or type(parameters.get('warmup')) is not int or parameters['warmup'] < 0):
            raise ValueError('Unsupported performance layout or iteration count')
        expected = ['{binary_dir}/all_reduce_perf', '-b', str(sizes[0]), '-e', str(sizes[0]),
                    '-n', str(parameters['iters']), '-w', str(parameters['warmup']), '-d', 'float',
                    '-o', 'sum', '-t', str(spec['threads_per_process']), '-g', str(spec['devices_per_thread']),
                    '-c', '1', '-k', '2', '-a', '2', '--elapsed_type', '0', '--report_type', '0', '-N', '1']
        if spec['argv'] != expected:
            raise ValueError('Performance argv does not match measurement metadata')
        for prefix in ('PCCL', 'NCCL'):
            for name in ('ALGO', 'PROTO'):
                if not parameters.get(name.lower()) or spec['env'].get(prefix + '_' + name) != parameters[name.lower()]:
                    raise ValueError('Performance algorithm/protocol must match explicit environment')
            if spec['env'].get(prefix + '_DEBUG_FILE') != '{result_dir}/library.%h.%p.log':
                raise ValueError('Performance diagnostics must use a separate log')


def build_manifest(specs, inputs, shards_per_pool=1, timings=None):
    if type(shards_per_pool) is not int or not 1 <= shards_per_pool <= 256:
        raise ValueError('shards_per_pool must be between 1 and 256')
    if not isinstance(inputs, dict):
        raise ValueError('Inputs must be a record')
    timings = {} if timings is None else timings
    if (not isinstance(timings, dict) or any(not isinstance(key, str) for key in timings)
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                   for value in timings.values())):
        raise ValueError('Timings must be finite positive seconds')
    if not specs:
        raise ValueError('Empty test selection')
    pools, seen = defaultdict(list), set()
    for spec in specs:
        validate_spec(spec)
        key = case_key(spec)
        if key in seen:
            raise ValueError(f'Duplicate Case: {key}')
        seen.add(key)
        pool = (spec['build_profile'], spec['nodes'], spec['cards_per_node'], spec['test_type'],
                spec['parameters'].get('transport', 'local'))
        row = {'case_key': key, 'spec': spec}
        if spec['test_type'] == 'performance':
            row['measurement_identity'] = measurement_identity(spec)
        pools[pool].append(row)
    shards = []
    for pool, records in sorted(pools.items()):
        units = defaultdict(list)
        for record in records:
            key = record.get('measurement_identity', record['case_key'])
            units[key].append(record)
        weights = {key: sum(timings.get(row['case_key'], 1)
                            for row in sorted(unit, key=lambda row: row['case_key']))
                   for key, unit in units.items()}
        bins = [[] for _ in range(min(shards_per_pool, len(units)))]
        loads = [0.0] * len(bins)
        for key in sorted(units, key=lambda key: (-weights[key], key)):
            index = min(range(len(bins)), key=lambda i: (loads[i], i))
            bins[index].extend(units[key])
            loads[index] += weights[key]
        for index, records in enumerate(bins):
            shard_id = f'{pool[0]}-{pool[1]}n-{pool[2]}c-{pool[3]}-{pool[4]}-{index:03d}'
            shards.append({'shard_id': shard_id, 'build_profile': pool[0], 'nodes': pool[1],
                           'cards_per_node': pool[2], 'test_type': pool[3], 'transport': pool[4],
                           'estimated_seconds': loads[index], 'cases': sorted(records, key=lambda row: row['case_key'])})
    if len(shards) > 256:
        raise ValueError('Matrix exceeds 256 jobs')
    manifest = {'schema_version': 2, 'inputs': inputs, 'shards': shards, 'case_count': len(seen),
                'sharding': {'shards_per_pool': shards_per_pool, 'timings': timings}}
    manifest['selection_sha256'] = digest(manifest)
    return manifest


def validate_manifest(manifest):
    fields = {'schema_version', 'inputs', 'shards', 'case_count', 'sharding', 'selection_sha256'}
    if (not isinstance(manifest, dict) or set(manifest) != fields
            or type(manifest['schema_version']) is not int or manifest['schema_version'] != 2):
        raise ValueError('Unsupported manifest schema or fields')
    if (not isinstance(manifest['shards'], list) or not manifest['shards']
            or not isinstance(manifest['sharding'], dict)
            or set(manifest['sharding']) != {'shards_per_pool', 'timings'}):
        raise ValueError('Invalid shards or sharding parameters')
    specs, owners, samples = [], {}, defaultdict(set)
    for shard in manifest['shards']:
        if (not isinstance(shard, dict) or not isinstance(shard.get('shard_id'), str)
                or not isinstance(shard.get('cases'), list) or not shard['cases']):
            raise ValueError('Invalid or empty shard')
        for row in shard['cases']:
            if not isinstance(row, dict) or not isinstance(row.get('spec'), dict):
                raise ValueError('Invalid Case record')
            spec = row['spec']
            validate_spec(spec)
            specs.append(spec)
            if spec['test_type'] == 'performance':
                identity = measurement_identity(spec)
                if owners.setdefault(identity, shard['shard_id']) != shard['shard_id']:
                    raise ValueError('Performance measurement samples split across shards')
                sample = spec['performance']['sample_index']
                if sample in samples[identity]:
                    raise ValueError('Duplicate performance sample')
                samples[identity].add(sample)
    expected = build_manifest(specs, manifest['inputs'], **manifest['sharding'])
    if digest(manifest) != digest(expected):
        raise ValueError('Manifest integrity or canonical partition check failed')


def selected_specs(inputs, source_catalog=None):
    from catalog import cases, performance_cases

    if inputs.get('test_suite') != 'pr':
        raise ValueError('The reviewed source catalog currently supports PR sanity only')
    for flag in ('single_node', 'debug_only'):
        if type(inputs.get(flag)) is not bool:
            raise ValueError('Selection flags must be booleans')
    types = inputs.get('test_types')
    nodes = inputs.get('node_counts')
    if (not isinstance(types, list) or types != ['functional']
            or len(set(types)) != len(types)
            or not isinstance(nodes, list) or not nodes
            or any(type(node) is not int or node not in (2, 8, 16) for node in nodes)
            or len(set(nodes)) != len(nodes)):
        raise ValueError('Invalid test types or node counts')
    if inputs['debug_only'] or inputs['single_node'] is not True:
        raise ValueError('The reviewed source catalog requires explicit single-node non-debug selection')
    source = inputs.get('source_catalog_source')
    if not isinstance(source, dict):
        raise ValueError('Missing trusted source catalog identity')
    specs = cases('sanity', tuple(nodes), False, path=source_catalog, expected_source=source)
    performance_cases('sanity', path=source_catalog, expected_source=source)
    return [spec for spec in specs if not inputs['single_node'] or spec['nodes'] == 1]


def validate_selection(manifest, expected_inputs):
    if digest(manifest['inputs']) != digest(expected_inputs):
        raise ValueError('Manifest inputs differ from the trusted invocation')
    expected = {case_key(spec): spec for spec in selected_specs(expected_inputs)}
    actual = {row['case_key']: row['spec'] for shard in manifest['shards'] for row in shard['cases']}
    if digest(actual) != digest(expected):
        raise ValueError('Manifest selection differs from the complete catalog')
    if (manifest['sharding']['shards_per_pool'] != expected_inputs['shards_per_pool']
            or digest(manifest['sharding']['timings']) != expected_inputs['timings_sha256']):
        raise ValueError('Sharding differs from the trusted invocation')


def read_json(path):
    def record(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')

    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=record,
                      parse_constant=invalid_constant)


def load_manifest(path, *, expected_inputs=None):
    manifest = read_json(path)
    validate_manifest(manifest)
    if expected_inputs is not None:
        validate_selection(manifest, expected_inputs)
    return manifest


def write_bundle(manifest, output):
    validate_manifest(manifest)
    root = Path(output).absolute()
    root.mkdir(parents=True, exist_ok=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=root, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        load_manifest(temporary)
        os.link(temporary, root / 'manifest.json')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=('pr', 'nightly'), required=True)
    parser.add_argument('--node-counts', type=int, nargs='+', default=[2, 8, 16])
    parser.add_argument('--shards-per-pool', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--single-node', action='store_true')
    parser.add_argument('--test-types', nargs='+', choices=('functional', 'performance'))
    parser.add_argument('--pccl-revision', required=True)
    parser.add_argument('--pccl-tests', type=Path, required=True)
    parser.add_argument('--source-catalog', type=Path,
                        default=Path(__file__).with_name('source-catalog.json'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--run-attempt', type=int, default=1)
    parser.add_argument('--timings', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if (not re.fullmatch(r'[0-9a-f]{40}', args.pccl_revision) or args.run_attempt < 1
                or not args.run_id.isdigit()):
            raise ValueError('Invalid revision, run ID or run attempt')
        source = Path(__file__).resolve()
        head = subprocess.check_output(['git', '-C', str(source.parents[2]), 'rev-parse', 'HEAD'], text=True).strip()
        if head != args.pccl_revision:
            raise ValueError('PCCL checkout differs from the requested revision')
        runner_head = subprocess.check_output(['git', '-C', str(args.pccl_tests), 'rev-parse', 'HEAD'], text=True).strip()
        if runner_head != PCCL_TESTS_REVISION:
            raise ValueError('pccl-tests checkout differs from the pinned revision')
        for name, checksum in RUNNER_HASHES.items():
            if hashlib.sha256((args.pccl_tests / name).read_bytes()).hexdigest() != checksum:
                raise ValueError(f'Public runner content changed: {name}')
        timings = read_json(args.timings) if args.timings else {}
        from catalog import available_test_types, source_inputs

        inputs = {'pccl_revision': args.pccl_revision, 'pccl_tests_revision': PCCL_TESTS_REVISION,
                  'python_version': platform.python_version(), 'generator_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'runner_sha256': RUNNER_HASHES, 'run_id': args.run_id, 'run_attempt': args.run_attempt,
                  'catalog_sha256': hashlib.sha256(source.with_name('catalog.py').read_bytes()).hexdigest(),
                  'test_suite': args.suite, 'debug_only': args.debug, 'single_node': args.single_node,
                  'test_types': sorted(args.test_types or available_test_types()),
                  'node_counts': sorted(args.node_counts), 'shards_per_pool': args.shards_per_pool,
                  'timings_sha256': digest(timings), **source_inputs(args.source_catalog)}
        manifest = build_manifest(selected_specs(inputs, args.source_catalog), inputs,
                                  args.shards_per_pool, timings)
        validate_selection(manifest, inputs)
        root = write_bundle(manifest, args.output)
        load_manifest(root / 'manifest.json', expected_inputs=inputs)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(json.dumps({'include': [{key: shard[key] for key in ('shard_id', 'nodes', 'build_profile', 'test_type')}
                                  for shard in manifest['shards']]}))


if __name__ == '__main__':
    main()
