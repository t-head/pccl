# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from command import cancellation_signals, check_command, run_command, stop_recorded_process
from manifest import digest, load_manifest, read_json
from metrics import parse_metrics, summarize_metrics, write_metrics


def execution_provenance():
    root = Path(__file__).resolve().parent
    files = ('execute.py', 'command.py', 'manifest.py', 'catalog.py', 'source-catalog.json',
             'metrics.py', 'single_node_ci.py')
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


def junit(rows, expected, path):
    by_key = {row['case_key']: row for row in rows}
    if len(by_key) != len(rows) or set(by_key) - set(expected):
        raise ValueError('Duplicate or unexpected results')
    suite = ET.Element('testsuite', name='PCCL cases', tests=str(len(expected)))
    failures = 0
    for key in expected:
        row = by_key.get(key)
        case = ET.SubElement(suite, 'testcase', name=key, classname='pccl.cases',
                             time=str(row['duration'] if row else 0))
        if row is None or row['status'] != 'PASSED':
            failures += 1
            message = row['error'] if row else 'Missing Case result'
            ET.SubElement(case, 'failure', message=message or 'Case did not pass')
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
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'manifest.json', manifest)
    expected = [row['case_key'] for row in shard['cases']]
    results, measurements = [], []
    with cancellation_signals() as interrupted:
        try:
            with (output / 'results.jsonl').open('x', encoding='utf-8') as stream:
                for number, row in enumerate(shard['cases'], 1):
                    if interrupted:
                        break
                    key = row['case_key']
                    spec = row['spec']
                    performance = spec['test_type'] == 'performance'
                    directory = output / key / 'attempt-1'
                    directory.mkdir(parents=True)
                    log_ref = str((directory / 'command.log').relative_to(output))
                    runtime = {'test_root': str(test_root), 'binary_dir': str(binary_dir),
                               'result_dir': str(directory), 'mpi_options': mpi_options,
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
            failures = junit(results, expected, output / 'junit.xml')
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
                                             'selection_sha256': manifest['selection_sha256'],
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
