# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from contextlib import closing, contextmanager
from contextvars import ContextVar
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from command import _prepare as prepare_command
from command import cancellation_signals, check_command, run_command, safe_environment, safe_text, stop_recorded_process
from manifest import digest, load_manifest, read_json
from metrics import parse_metrics, summarize_metrics, write_metrics


def execution_provenance():
    root = Path(__file__).resolve().parent
    files = ('execute.py', 'command.py', 'manifest.py', 'catalog.py', 'source-catalog.json',
             'metrics.py', 'single_node_ci.py', 'cluster_ci.py', 'runner_capture.py')
    return {'executor_sha256': digest({name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                       for name in files}),
            'executor_python_version': platform.python_version()}


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_jsonl(path):
    def record(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result

    rows = [json.loads(line, object_pairs_hook=record) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    digest(rows)
    return rows


def junit(rows, expected, path, lifecycle_error=''):
    by_key = {row['case_key']: row for row in rows}
    if len(by_key) != len(rows) or set(by_key) - set(expected):
        raise ValueError('Duplicate or unexpected results')
    suite = ET.Element('testsuite', name='PCCL cases', tests=str(len(expected) + bool(lifecycle_error)))
    failures = 0
    for key in expected:
        row = by_key.get(key)
        context = row.get('source_context') if row else None
        classname = f"{context['plan_id']}.{context['suite_id']}" if context else 'pccl.cases'
        case = ET.SubElement(suite, 'testcase', name=key, classname=classname,
                             time=str(row['duration'] if row else 0))
        if row is None or row['status'] != 'PASSED':
            failures += 1
            message = row['error'] if row else 'Missing Case result'
            ET.SubElement(case, 'failure', message=message or 'Case did not pass')
    if lifecycle_error:
        failures += 1
        case = ET.SubElement(suite, 'testcase', name='Suite lifecycle acceptance', classname='pccl.lifecycle', time='0')
        ET.SubElement(case, 'failure', message=lifecycle_error)
    suite.set('failures', str(failures))
    temporary = path.with_suffix('.xml.tmp')
    ET.ElementTree(suite).write(temporary, encoding='utf-8', xml_declaration=True)
    temporary.replace(path)
    return failures


def verified_metrics(directory, row, inputs, log_ref):
    expected = parse_metrics((directory / 'command.log').read_bytes().decode('utf-8'), row['spec'],
                             row['case_key'], inputs, log_ref)
    actual = read_jsonl(directory / 'metrics.jsonl')
    with (directory / 'metrics.csv').open(newline='', encoding='utf-8') as stream:
        csv_rows = list(csv.DictReader(stream))
    if (digest(actual) != digest(expected)
            or csv_rows != [{key: str(value) for key, value in metric.items()} for metric in expected]):
        raise ValueError('Performance artifacts do not match command log and manifest')
    return expected


HOOK_TIMEOUT = 300
SOURCE_CONTEXT_FIELDS = ('plan_id', 'suite_id', 'case_id', 'suite_run_id')


def _validate_runtime(runtime):
    if (not isinstance(runtime, dict) or set(runtime) !=
            {'test_root', 'binary_dir', 'result_dir', 'runner_build_directory',
             'mpi_options', 'python', 'inherited_environment'}):
        raise ValueError('Invalid shard runtime fields')
    for key in ('test_root', 'binary_dir', 'result_dir', 'runner_build_directory', 'python'):
        value = runtime[key]
        # Homebrew's versioned interpreter directory is not an email address.
        if key == 'python' and isinstance(value, str):
            value = re.sub(r'(?<=/)python@([0-9]+\.[0-9]+)(?=/)', r'python-\1', value)
        if not Path(safe_text(value)).is_absolute():
            raise ValueError('Shard runtime requires absolute execution paths')
    if not isinstance(runtime['mpi_options'], list):
        raise ValueError('Invalid shard runtime MPI options')
    for token in runtime['mpi_options']:
        safe_text(token)
    environment = runtime['inherited_environment']
    if not isinstance(environment, dict) or safe_environment(environment) != environment:
        raise ValueError('Invalid shard runtime environment')
    return runtime


def verified_runtime(directory, index):
    """Load the shared runtime after the caller validates manifest/executor identity.

    This independently recorded shard contract anchors both hooks and Cases.
    Hashes detect inconsistent artifacts, not a coordinated rewrite or forgery.
    Original paths are data only: offline acceptance must not resolve them.
    """
    snapshot = read_json(Path(directory) / 'shard-runtime.json')
    if (not isinstance(snapshot, dict)
            or set(snapshot) != {'schema_version', 'shard_id', 'selection_sha256', 'execution', 'runtime'}
            or type(snapshot['schema_version']) is not int or snapshot['schema_version'] != 1
            or any(snapshot[key] != index[key] for key in ('shard_id', 'selection_sha256', 'execution'))
            or digest(snapshot) != index.get('runtime_sha256')):
        raise ValueError('Shard runtime differs from manifest/executor index')
    return _validate_runtime(snapshot['runtime'])


def _lifecycle_record(suite):
    return {**{key: suite[key] for key in ('suite_run_id', 'plan_id', 'suite_id', 'case_keys')},
            'reuse': suite['lifecycle']['reuse'],
            'expected_hooks': {phase: suite['lifecycle'][phase] for phase in ('setup', 'teardown')},
            'actual_hooks': {'setup': [], 'teardown': []}, 'case_results': [], 'status': 'NOT_RUN'}


def _hook_spec(argv, cases):
    return {'argv': argv, 'test_type': 'functional', 'checks': ['exit_code'],
            'nodes': 1, 'processes_per_node': 1, 'threads_per_process': 1, 'devices_per_thread': 1,
            'env': cases[0]['spec']['env'] if cases else {},
            'timeout': min(HOOK_TIMEOUT, max((case['spec']['timeout'] for case in cases),
                                           default=HOOK_TIMEOUT))}


def _file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


_HOOK_LAUNCHES = ContextVar('hook_launches', default=None)


def _observe_hook_launch(event, arguments):
    launches = _HOOK_LAUNCHES.get()
    if event != 'subprocess.Popen' or launches is None:
        return
    executable, argv, cwd, environment = arguments
    # command's process-group inspection uses ps with neither cwd nor env.
    if cwd is None and environment is None:
        return
    launches.append(dict(executable=executable, argv=list(argv), cwd=cwd,
                         effective_environment=safe_environment(os.environ if environment is None else environment)))


# Install once, and enable only during a synchronous hook call. Do not replace
# Popen: command must retain ownership of spawn, PID identity, wait and cleanup.
sys.addaudithook(_observe_hook_launch)


@contextmanager
def _capture_hook_launches():
    launches = []
    token = _HOOK_LAUNCHES.set(launches)
    try:
        yield launches
    finally:
        _HOOK_LAUNCHES.reset(token)


def _hook_invocation(spec, suite, phase, number, runtime):
    # The caller supplies a trusted execution-root/environment contract, not
    # bindings or inherited settings taken from the hook's own evidence.
    bindings = {key: runtime[key] for key in
                ('test_root', 'binary_dir', 'result_dir', 'mpi_options', 'python', 'inherited_environment')}
    if (not Path(bindings['python']).is_absolute()
            or safe_environment(bindings['inherited_environment']) != bindings['inherited_environment']):
        raise ValueError('Invalid trusted hook runtime')
    argv, environment = prepare_command(spec, bindings, inherited=bindings['inherited_environment'],
                                        validate_paths=False, python=bindings['python'])
    return dict(schema_version=1, spec_sha256=digest(spec),
                source_ids={**{key: suite[key] for key in ('suite_run_id', 'plan_id', 'suite_id')},
                            'phase': phase, 'number': number},
                argv=argv, cwd=bindings['test_root'], effective_environment=safe_environment(environment),
                bindings=bindings)


def _check_hook_invocation(directory, spec, suite, phase, number, runtime, process):
    check_command(spec, {'result_dir': str(directory)}, process)
    invocation = read_json(directory / 'command-invocation.json')
    expected = _hook_invocation(spec, suite, phase, number, runtime)
    if digest(invocation) != digest(expected):
        raise ValueError('Hook invocation differs from trusted manifest/runtime')
    observed = read_json(directory / 'command-observations.json')
    launch = {key: expected[key] for key in ('argv', 'cwd', 'effective_environment')}
    launch['executable'] = expected['argv'][0]
    evidence = dict(schema_version=1, source='executor-audit', event='subprocess.Popen',
                    invocation_sha256=digest(expected), launches=[launch], process=process)
    if digest(observed) != digest(evidence):
        raise ValueError('Hook process invocation differs from observed launch')
    # An audit event precedes spawn. Only the matching final command record
    # supplies successful completion; declarations or an event alone cannot.
    pid, identity = process.get('pid'), process.get('identity')
    if (type(pid) is not int or pid <= 1 or type(process.get('pgid')) is not int or process['pgid'] != pid
            or not isinstance(identity, dict) or type(identity.get('pid')) is not int or identity['pid'] != pid
            or not isinstance(identity.get('starttime'), str) or not identity['starttime']
            or identity.get('identity_source') not in ('proc', 'proc_pidinfo')
            or (identity['identity_source'] == 'proc' and not identity.get('boot_id'))):
        raise ValueError('Missing hook process identity evidence')


def _run_hook(output, suite, phase, number, argv, cases, runtime, interrupted):
    directory = output / 'suite-lifecycle' / suite['suite_run_id'] / f'{phase}-{number}'
    record_path, log_path = directory / 'command-process.json', directory / 'command.log'
    result = {'phase': phase, 'number': number, 'argv': argv, 'status': 'ERROR', 'error': '',
              'returncode': None, 'command_log': str(log_path.relative_to(output)),
              'command_record': str(record_path.relative_to(output)),
              'log_sha256': None, 'record_sha256': None,
              'invocation_sha256': None, 'observations_sha256': None}
    invocation, launches = None, []
    spec = _hook_spec(argv, cases)
    hook_runtime = {**runtime, 'result_dir': str(directory), 'command_log_ref': result['command_log']}
    try:
        directory.mkdir(parents=True)
        invocation = _hook_invocation(spec, suite, phase, number, hook_runtime)
        write_json(directory / 'command-invocation.json', invocation)
        # Generic hooks have no run_test.py execution contract. Observe the
        # direct launch instead of manufacturing source-runner metadata.
        with _capture_hook_launches() as launches:
            process = run_command(spec, hook_runtime, interrupted)
        check_command(spec, hook_runtime, process)
        result['status'] = 'PASSED'
    except subprocess.CalledProcessError as exception:
        result.update(status='FAILED', error='; '.join([str(exception),
                      *getattr(exception, '__notes__', [])]))
    except Exception as exception:
        result['error'] = '; '.join([str(exception) or type(exception).__name__,
                                    *getattr(exception, '__notes__', [])])
    finally:
        try:
            process = read_json(record_path)
            result['returncode'] = process.get('return_code')
            if process.get('active'):
                stopped = stop_recorded_process(record_path)
                raise RuntimeError('Hook cleanup was not confirmed'
                                   + ('' if stopped else '; process identity unavailable'))
            if process.get('active') is not False or process.get('state') == 'cleanup_failed':
                raise RuntimeError('Hook process cleanup failed')
            if invocation is not None:
                observed = dict(schema_version=1, source='executor-audit', event='subprocess.Popen',
                                invocation_sha256=digest(invocation), launches=launches, process=process)
                write_json(directory / 'command-observations.json', observed)
                result['invocation_sha256'] = _file_sha256(directory / 'command-invocation.json')
                result['observations_sha256'] = _file_sha256(directory / 'command-observations.json')
            if result['status'] == 'PASSED':
                _check_hook_invocation(directory, spec, suite, phase, number, hook_runtime, process)
            result['record_sha256'] = _file_sha256(record_path)
            result['log_sha256'] = _file_sha256(log_path)
        except Exception as exception:
            result['status'] = 'ERROR'
            result['error'] = '; '.join(filter(None, (result['error'],
                                      f'Hook evidence error: {type(exception).__name__}: {exception}')))
        write_json(directory / 'result.json', result)
    return result


def _suite_cases(shard, output, runtime, interrupted, results, records):
    if 'suite_runs' not in shard:
        yield from shard['cases']
        return
    by_key = {row['case_key']: row for row in shard['cases']}
    for suite, record in zip(shard['suite_runs'], records):
        if interrupted:
            return
        cases = [by_key[key] for key in suite['case_keys']]
        record['status'] = 'RUNNING'
        try:
            write_json(output / 'suite-lifecycle.json', records)
            for number, argv in enumerate(suite['lifecycle']['setup'], 1):
                hook = _run_hook(output, suite, 'setup', number, argv, cases, runtime, interrupted)
                record['actual_hooks']['setup'].append(hook)
                write_json(output / 'suite-lifecycle.json', records)
                if hook['status'] != 'PASSED' or interrupted:
                    return
            for row in cases:
                if interrupted:
                    return
                yield row
        finally:
            # Latched cancellation must not skip bounded teardown.
            try:
                for number, argv in enumerate(suite['lifecycle']['teardown'], 1):
                    try:
                        hook = _run_hook(output, suite, 'teardown', number, argv, cases, runtime, [])
                        record['actual_hooks']['teardown'].append(hook)
                    except Exception as exception:
                        record['error'] = f'Teardown evidence error: {type(exception).__name__}: {exception}'
            finally:
                record['case_results'] = [{'case_key': row['case_key'], 'status': row['status']}
                                          for row in results if row['case_key'] in suite['case_keys']]
                hooks_complete = all(len(record['actual_hooks'][phase]) == len(record['expected_hooks'][phase])
                                     and all(hook['status'] == 'PASSED' for hook in record['actual_hooks'][phase])
                                     for phase in ('setup', 'teardown'))
                complete = [row['case_key'] for row in record['case_results']] == suite['case_keys']
                record['status'] = ('PASSED' if hooks_complete and complete and not record.get('error')
                                    and all(row['status'] == 'PASSED' for row in record['case_results']) else 'FAILED')
                write_json(output / 'suite-lifecycle.json', records)
        if not hooks_complete or record.get('error'):
            return


def verify_lifecycle(directory, shard, runtime=None):
    """Accept hook evidence only with caller-trusted execution paths/environment.

    runtime contains test_root, binary_dir, result_dir (the execution output
    root, possibly different from directory on an offline verifier), mpi_options,
    python and inherited_environment (command.safe_environment at execution).
    Never recover this trust anchor from command-invocation.json.
    """
    if 'suite_runs' not in shard:
        return []
    directory = Path(directory)
    try:
        if any(suite['lifecycle'][phase] for suite in shard['suite_runs'] for phase in ('setup', 'teardown')):
            if runtime is None:
                raise ValueError('Hook verification requires a trusted execution runtime')
        records = read_json(directory / 'suite-lifecycle.json')
        if not isinstance(records, list) or len(records) != len(shard['suite_runs']):
            raise ValueError('Incomplete Suite lifecycle list')
        rows = read_jsonl(directory / 'results.jsonl')
        expected_keys = [row['case_key'] for row in shard['cases']]
        if ([key for suite in shard['suite_runs'] for key in suite['case_keys']] != expected_keys
                or len(set(expected_keys)) != len(expected_keys)
                or [row['case_key'] for row in rows] != expected_keys):
            raise ValueError('Suite Case completion or order differs from manifest')
        by_key = {row['case_key']: row for row in shard['cases']}
        results = {row['case_key']: row for row in rows}
        seen = set()
        for suite, record in zip(shard['suite_runs'], records):
            identifier = suite['suite_run_id']
            if (not isinstance(identifier, str) or not identifier or identifier in ('.', '..')
                    or Path(identifier).name != identifier or identifier in seen):
                raise ValueError('Invalid Suite run identity')
            seen.add(identifier)
            expected = _lifecycle_record(suite)
            if (set(record) != set(expected)
                    or any(digest(record[key]) != digest(expected[key]) for key in
                           ('suite_run_id', 'plan_id', 'suite_id', 'case_keys', 'reuse', 'expected_hooks'))
                    or set(record['actual_hooks']) != {'setup', 'teardown'}):
                raise ValueError('Suite lifecycle ownership or definition differs from manifest')
            case_results = []
            for key in suite['case_keys']:
                spec, result = by_key[key]['spec'], results[key]
                context = {field: spec['execution'][field] for field in SOURCE_CONTEXT_FIELDS}
                if (any(context[field] != suite[field] for field in ('plan_id', 'suite_id', 'suite_run_id'))
                        or result.get('source_context') != context
                        or result['status'] not in ('PASSED', 'FAILED', 'ERROR')
                        or digest(read_json(directory / key / 'attempt-1' / 'result.json')) != digest(result)):
                    raise ValueError('Suite Case ownership or result differs from manifest')
                case_results.append({'case_key': key, 'status': result['status']})
            if record['case_results'] != case_results:
                raise ValueError('Suite Case completion differs from results')
            for phase in ('setup', 'teardown'):
                hooks = record['actual_hooks'][phase]
                if not isinstance(hooks, list) or len(hooks) != len(expected['expected_hooks'][phase]):
                    raise ValueError('Incomplete Suite hooks')
                for number, (argv, hook) in enumerate(zip(expected['expected_hooks'][phase], hooks), 1):
                    relative = Path('suite-lifecycle') / identifier / f'{phase}-{number}'
                    hook_dir = directory / relative
                    process = read_json(hook_dir / 'command-process.json')
                    if (hook['phase'] != phase or type(hook['number']) is not int or hook['number'] != number
                            or hook['argv'] != argv or hook['status'] != 'PASSED' or hook['error'] != ''
                            or type(hook['returncode']) is not int or hook['returncode'] != 0
                            or hook['command_log'] != str(relative / 'command.log')
                            or hook['command_record'] != str(relative / 'command-process.json')
                            or hook['log_sha256'] != _file_sha256(hook_dir / 'command.log')
                            or hook['record_sha256'] != _file_sha256(hook_dir / 'command-process.json')
                            or hook['invocation_sha256'] != _file_sha256(hook_dir / 'command-invocation.json')
                            or hook['observations_sha256'] != _file_sha256(hook_dir / 'command-observations.json')
                            or digest(read_json(hook_dir / 'result.json')) != digest(hook)):
                        raise ValueError('Suite hook failed or evidence differs from lifecycle record')
                    hook_runtime = {**runtime, 'result_dir': str(Path(runtime['result_dir']) / relative)}
                    spec = _hook_spec(argv, [by_key[key] for key in suite['case_keys']])
                    _check_hook_invocation(hook_dir, spec, suite, phase, number, hook_runtime, process)
            status = 'PASSED' if all(row['status'] == 'PASSED' for row in case_results) else 'FAILED'
            if record['status'] != status:
                raise ValueError('Suite lifecycle status differs from evidence')
        return records
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exception:
        raise ValueError(f'Invalid Suite lifecycle evidence: {exception}') from exception


def run_shard(manifest_path, shard_id, test_root, binary_dir, output, mpi_options):
    manifest = load_manifest(Path(manifest_path).resolve())
    matches = [shard for shard in manifest['shards'] if shard['shard_id'] == shard_id]
    if len(matches) != 1:
        raise ValueError('Unknown shard')
    shard = matches[0]
    if os.environ.get('NODE_RANK', '0') != '0':
        raise ValueError('Only the controller executes a shard')
    test_root, binary_dir = (Path(value).resolve() for value in (test_root, binary_dir))
    output = Path(output).absolute()
    if not test_root.is_dir() or not binary_dir.is_dir():
        raise ValueError('Missing build or binary directory')
    provenance = execution_provenance()
    inputs = {**manifest['inputs'], **provenance}
    hook_runtime = _validate_runtime(dict(test_root=str(test_root), binary_dir=str(binary_dir),
                                         result_dir=str(output), mpi_options=list(mpi_options), python=sys.executable,
                                         runner_build_directory=str((test_root / 'build').resolve()),
                                         inherited_environment=safe_environment(os.environ)))
    snapshot = dict(schema_version=1, shard_id=shard_id, selection_sha256=manifest['selection_sha256'],
                    execution=provenance, runtime=hook_runtime)
    runtime_sha256 = digest(snapshot)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'manifest.json', manifest)
    # Publish before any hook or Case; never reconstruct this anchor from their declarations.
    write_json(output / 'shard-runtime.json', snapshot)
    expected = [row['case_key'] for row in shard['cases']]
    results, measurements = [], []
    lifecycle = [_lifecycle_record(suite) for suite in shard.get('suite_runs', [])]
    if 'suite_runs' in shard:
        write_json(output / 'suite-lifecycle.json', lifecycle)
    with cancellation_signals() as interrupted:
        try:
            with closing(_suite_cases(shard, output, hook_runtime, interrupted, results, lifecycle)) as cases, \
                    (output / 'results.jsonl').open('x', encoding='utf-8') as stream:
                for number, row in enumerate(cases, 1):
                    if interrupted:
                        break
                    key = row['case_key']
                    spec = row['spec']
                    performance = spec['test_type'] == 'performance'
                    directory = output / key / 'attempt-1'
                    directory.mkdir(parents=True)
                    log_ref = str((directory / 'command.log').relative_to(output))
                    runtime = {**hook_runtime, 'result_dir': str(directory),
                               'execution_result_dir': str(directory),
                               'case_key': key, 'inputs': inputs, 'command_log_ref': log_ref}
                    print(f'[{number}/{len(expected)}] {key}', flush=True)
                    started = time.monotonic()
                    status, error, returncode, cleanup_error = 'ERROR', '', None, None
                    try:
                        record = run_command(spec, runtime, interrupted)
                    except subprocess.CalledProcessError as exception:
                        status = 'FAILED'
                        error = '; '.join([f'Command exited with return code {exception.returncode}',
                                           *getattr(exception, '__notes__', [])])
                    except Exception as exception:
                        error = '; '.join([str(exception) or type(exception).__name__,
                                           *getattr(exception, '__notes__', [])])
                    else:
                        try:
                            metrics = check_command(spec, runtime, record)
                            status = 'PASSED'
                        except ValueError as exception:
                            status = 'ERROR' if performance else 'FAILED'
                            error = str(exception)
                        except Exception as exception:
                            error = str(exception)
                    finally:
                        try:
                            record_path = directory / 'command-process.json'
                            if record_path.exists():
                                record = read_json(record_path)
                                returncode = record.get('return_code')
                                if record.get('active'):
                                    stopped = stop_recorded_process(record_path)
                                    raise RuntimeError('Command cleanup was not confirmed'
                                                       + ('' if stopped else '; process identity unavailable'))
                                if record.get('state') == 'cleanup_failed':
                                    raise RuntimeError('Command process cleanup failed')
                                if record.get('state') in ('timeout', 'cancelled'):
                                    status = 'ERROR'
                                if status == 'PASSED' and (record.get('active') is not False
                                        or record.get('state') != 'finished' or type(returncode) is not int
                                        or returncode != 0):
                                    raise RuntimeError('Missing successful final command record')
                            elif status == 'PASSED':
                                raise RuntimeError('Missing final command record')
                        except Exception as exception:
                            cleanup_error = exception
                    if cleanup_error is not None:
                        status = 'ERROR'
                        error = '; '.join(filter(None, (error,
                                          f'cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}')))
                    if performance and status == 'PASSED':
                        try:
                            write_metrics(metrics, directory)
                            measurements.extend(verified_metrics(directory, row, inputs, log_ref))
                        except Exception as exception:
                            status, error = 'ERROR', f'Performance data error: {exception}'
                    result = {'case_key': key, 'attempt': 1, 'status': status, 'error': error,
                              'returncode': returncode, 'duration': time.monotonic() - started,
                              'legacy_source': spec['legacy_source'], 'command_log': log_ref}
                    if 'execution' in spec:
                        result['source_context'] = {field: spec['execution'][field] for field in SOURCE_CONTEXT_FIELDS}
                    if performance:
                        result.update(data_status='VALID' if status == 'PASSED' else 'DATA_ERROR',
                                      metrics_log=str((directory / 'metrics.jsonl').relative_to(output))
                                      if status == 'PASSED' else None)
                    write_json(directory / 'result.json', result)
                    stream.write(json.dumps(result, allow_nan=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    results.append(result)
                    print(f'{key}: {status}', flush=True)
                    mpi_case = (spec['nodes'] > 1 or spec['processes_per_node'] > 1
                                or Path(spec['argv'][0]).name == 'mpirun')
                    # Local cleanup cannot certify remote MPI cleanup after a failure.
                    if interrupted or status == 'ERROR' or (mpi_case and status == 'FAILED'):
                        break
        finally:
            lifecycle_error = ''
            try:
                verify_lifecycle(output, shard, hook_runtime)
            except ValueError as exception:
                lifecycle_error = str(exception)
            failures = junit(results, expected, output / 'junit.xml', lifecycle_error)
            artifact_error = ''
            try:
                if measurements:
                    summaries = summarize_metrics(measurements)
                    write_metrics(measurements, output)
                    write_json(output / 'performance-summary.json',
                               {'complete': not failures and not interrupted, 'measurements': summaries})
            except (OSError, ValueError) as exception:
                artifact_error = str(exception)
                failures += 1
            write_json(output / 'timings.json', {row['case_key']: row['duration'] for row in results})
            write_json(output / 'index.json', {'shard_id': shard_id, 'expected': expected,
                                             'completed': [row['case_key'] for row in results],
                                             'cancelled': bool(interrupted), 'failures': failures,
                                             'artifact_error': artifact_error,
                                             **({'lifecycle_error': lifecycle_error} if 'suite_runs' in shard else {}),
                                             'selection_sha256': manifest['selection_sha256'],
                                             'runtime_sha256': runtime_sha256,
                                             'execution': provenance})
    return 1 if failures or interrupted else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--shard', required=True)
    parser.add_argument('--test-root', type=Path, required=True)
    parser.add_argument('--binary-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mpi-options', help='JSON array; otherwise read PPU_MPI_OPTIONS_FILE when set')
    args = parser.parse_args()
    try:
        raw = args.mpi_options
        if raw is None:
            source = os.environ.get('PPU_MPI_OPTIONS_FILE')
            raw = Path(source).read_text() if source else '[]'
        options = json.loads(raw)
    except (OSError, ValueError) as error:
        parser.error(f'Cannot load MPI options: {error}')
    if not isinstance(options, list) or any(not isinstance(value, str) or '\x00' in value for value in options):
        parser.error('--mpi-options must be a JSON string array without NUL')
    return run_shard(args.manifest, args.shard, args.test_root, args.binary_dir, args.output, options)


if __name__ == '__main__':
    sys.exit(main())
