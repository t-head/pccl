# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict metrics for the public schema13 AllReduce output, not runner duration."""

from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import median
import tempfile

from manifest import measurement_identity


_TIME_KINDS = ('cpu_time', 'event_time', 'hgpti_time')
_STARTUP = re.compile(
  r'# nThread (?P<threads>[0-9]+) nGpus (?P<devices>[0-9]+) '
  r'minBytes (?P<minimum>[0-9]+) maxBytes (?P<maximum>[0-9]+) '
  r'offset <(?P<send_offset>[0-9]+)>/<(?P<recv_offset>[0-9]+)> '
  r'step: (?P<step>[0-9]+)\((?P<step_kind>bytes|factor|random)\) '
  r'warmup iters: (?P<warmup>[0-9]+) iters: (?P<iters>[0-9]+) '
  r'redop: (?P<redop>\S+) validation: (?P<validation>[0-9]+) '
  r'single_test: (?P<single_test>[0-9]+) test_buffer_kind: (?P<buffer_kind>\S+) '
  r'elapsed_type: (?P<elapsed_type>\S+) average: (?P<average>\S+) register: (?P<register>[0-9]+)\Z')
_RANK = re.compile(
  r'#\s+Rank\s+(?P<rank>[0-9]+) Group\s+(?P<group>[0-9]+) Pid\s+(?P<pid>[0-9]+) '
  r'on\s+(?P<host>\S+) device\s+(?P<device>[0-9]+) \[0x[0-9a-fA-F]+\] .+\Z')
_NUMBER = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z')
_LAYOUT = re.compile(r'#[-\s]*out-of-place[-\s]+in-place[-\s]*\Z')
_SUMMARY = re.compile(r'#\s*Out of bounds values\s*:\s*([0-9]+)\s+OK\s*\Z')
_UNITS = '# (B) (elements) (us) (GB/s) (GB/s) (us) (GB/s) (GB/s)'.split()
_MEASUREMENTS = ('time', 'algbw', 'busbw')
_NOT_COMPARABLE = frozenset((
  'case_key', 'sample_index', 'log_path', 'log_sha256', 'source_line',
  'time', 'algbw', 'busbw', 'data_status', 'comparison_status',
))


def _integer(value, name, minimum=1):
  if type(value) is not int or value < minimum:
    raise ValueError('{} must be an integer >= {}'.format(name, minimum))
  return value


def _string(value, name):
  if not isinstance(value, str) or not value.strip():
    raise ValueError('{} must be a nonempty string'.format(name))
  return value


def _decimal(token, name):
  if not _NUMBER.fullmatch(token):
    raise ValueError('{} must be a finite decimal number'.format(name))
  try:
    return Decimal(token)
  except InvalidOperation as error:
    raise ValueError('{} has an unsupported decimal exponent'.format(name)) from error


def _measurement(token, name):
  exact = _decimal(token, name)
  value = float(exact)
  if not math.isfinite(value):
    raise ValueError('{} must be finite'.format(name))
  if exact < 0 or (name == 'time' and (exact <= 0 or value <= 0)):
    raise ValueError('{} must be {}'.format(name, '> 0' if name == 'time' else '>= 0'))
  return value


def _metadata(spec, case_key, inputs, log_path):
  if not isinstance(spec, dict) or not isinstance(inputs, dict):
    raise ValueError('spec and inputs must be records')
  parameters = spec.get('parameters')
  performance = spec.get('performance')
  if not isinstance(parameters, dict) or not isinstance(performance, dict):
    raise ValueError('spec must contain parameters and performance records')
  if (parameters.get('op') != 'all_reduce_perf' or parameters.get('dtype') != 'float'
      or parameters.get('redop') != 'sum'):
    raise ValueError('only float32 sum all_reduce_perf is supported (4-byte alignment)')
  sizes = performance.get('expected_sizes')
  if not isinstance(sizes, list) or not sizes:
    raise ValueError('expected_sizes must be a nonempty list of positive byte sizes')
  for size in sizes:
    _integer(size, 'expected_sizes entry')
    if size % 4:
      raise ValueError('expected_sizes must be float32 aligned (multiples of 4 bytes)')
  if len(set(sizes)) != len(sizes):
    raise ValueError('expected_sizes contains duplicate sizes')
  kind = performance.get('time_kind', 'cpu_time')
  reduction = performance.get('rank_reduction', 'MIN')
  if kind not in _TIME_KINDS:
    raise ValueError('unsupported time_kind: {!r}'.format(kind))
  if reduction not in ('MIN', 'MAX', 'AVG'):
    raise ValueError('rank_reduction must be MIN, MAX or AVG')
  metadata = {
    'schema_version': 2,
    'case_key': _string(case_key, 'case_key'),
    'measurement_identity': measurement_identity(spec),
    'sample_index': _integer(performance.get('sample_index', 0), 'sample_index', 0),
    'op': parameters['op'],
    'dtype': parameters['dtype'],
    'redop': _string(parameters.get('redop'), 'redop'),
    'algo': _string(parameters.get('algo', 'Auto'), 'algo'),
    'proto': _string(parameters.get('proto', 'Auto'), 'proto'),
    'transport': _string(parameters.get('transport', 'local'), 'transport'),
    'time_kind': kind,
    'time_unit': 'us',
    'rank_reduction': reduction,
    'bandwidth_unit': 'GB/s',
  }
  for name in ('ranks', 'iters', 'warmup'):
    metadata[name] = _integer(parameters.get(name), name, 0 if name == 'warmup' else 1)
  for name in ('nodes', 'cards_per_node', 'processes_per_node', 'threads_per_process', 'devices_per_thread'):
    metadata[name] = _integer(spec.get(name), name)
  local_ranks = metadata['processes_per_node'] * metadata['threads_per_process'] * metadata['devices_per_thread']
  if metadata['ranks'] != metadata['nodes'] * local_ranks or local_ranks > metadata['cards_per_node']:
    raise ValueError('rank layout does not match the allocation')
  if metadata['nodes'] != 1 or metadata['processes_per_node'] != 1:
    raise ValueError('startup verification currently supports only one node and one process')
  if 'build_profile' in spec:
    metadata['build_profile'] = _string(spec['build_profile'], 'build_profile')
  for name in ('pccl_revision', 'pccl_tests_revision', 'executor_sha256', 'executor_python_version'):
    metadata[name] = _string(inputs.get(name), name)
  if not re.fullmatch(r'[0-9a-f]{64}', metadata['executor_sha256']):
    raise ValueError('executor_sha256 must be a 64-character lowercase hex SHA256')
  try:
    metadata['log_path'] = _string(os.fspath(log_path), 'log_path')
  except TypeError as error:
    raise ValueError('log_path must be a string or path') from error
  metadata['data_status'] = 'VALID'
  metadata['comparison_status'] = 'NO_BASELINE'
  return metadata, set(sizes)


def _validate_startup(match, metadata, expected, parameters):
  if match is None:
    raise ValueError('malformed or prefixed startup evidence')
  # Undeclared controls must match the pinned common.cu defaults.
  factor = parameters.get('step_factor', 1)
  values = {
    'threads': metadata['threads_per_process'], 'devices': metadata['devices_per_thread'],
    'minimum': min(expected), 'maximum': max(expected),
    'warmup': metadata['warmup'], 'iters': metadata['iters'], 'redop': metadata['redop'],
    'validation': 1, 'buffer_kind': 'both', 'average': metadata['rank_reduction'],
    'elapsed_type': 'kernel_time' if metadata['time_kind'] == 'hgpti_time' else metadata['time_kind'],
    'send_offset': parameters.get('sendbuff_offset', 0), 'recv_offset': parameters.get('recvbuff_offset', 0),
    'single_test': parameters.get('single_test', 0), 'register': parameters.get('register', 0),
    'step': factor if factor > 1 else parameters.get('step_bytes', 1048576),
    'step_kind': 'factor' if factor > 1 else 'bytes',
  }
  for name, value in values.items():
    if match[name] != str(value):
      raise ValueError('startup {} does not match measurement declaration'.format(name))


def _validate_rank(match, ranks, metadata):
  if match is None:
    raise ValueError('malformed or prefixed Rank evidence')
  rank = {name: int(match[name]) for name in ('rank', 'group', 'pid', 'device')}
  rank['host'] = match['host']
  if rank['rank'] != len(ranks) or rank['rank'] >= metadata['ranks']:
    raise ValueError('duplicate, unordered or unexpected Rank evidence')
  if rank['group'] != 0 or rank['pid'] <= 0:
    raise ValueError('Rank Group/Pid does not match the single-process layout')
  if ranks and (rank['host'] != ranks[0]['host'] or rank['pid'] != ranks[0]['pid']):
    raise ValueError('Rank host/Pid contradicts the single-process layout')
  if any(rank['device'] == previous['device'] for previous in ranks):
    raise ValueError('Rank evidence repeats a device')
  # Host/Pid consistency is not topology proof and is never exported into metrics.
  ranks.append(rank)


def parse_metrics(text, spec, case_key, inputs, log_path) -> list[dict]:
  """Require one verified startup/Rank block; concatenated launches cannot share evidence."""
  if not isinstance(text, str):
    raise ValueError('text must be the raw UTF-8 log as a string')
  metadata, expected = _metadata(spec, case_key, inputs, log_path)
  metadata['log_sha256'] = hashlib.sha256(text.encode('utf-8')).hexdigest()
  kind = metadata['time_kind']
  header = ('# size count type redop root {0} algbw busbw #wrong '
            '{0} algbw busbw #wrong'.format(kind)).split()
  rows, seen = [], set()
  startup, ranks = False, []
  state, table_rows, summaries = 'startup', 0, 0
  for line_number, raw in enumerate(text.splitlines(), 1):
    line = raw.strip()
    if not line:
      continue
    try:
      if re.search(r'\bfailed\b', line, re.I):
        raise ValueError('failed marker in log')
      if re.search(r'\bnThread\b|\bnGpus\b', line):
        if startup or state != 'startup':
          raise ValueError('duplicate or misplaced startup evidence')
        _validate_startup(_STARTUP.fullmatch(line), metadata, expected, spec['parameters'])
        startup = True
        continue
      if re.search(r'\bRank\b', line):
        if not startup or state != 'startup':
          raise ValueError('Rank evidence must follow startup and precede all tables')
        _validate_rank(_RANK.fullmatch(line), ranks, metadata)
        continue
      if re.search(r'out\s+of\s+bounds', line, re.I):
        match = _SUMMARY.fullmatch(line)
        if not match or int(match.group(1)) != 0:
          raise ValueError('unsuccessful or malformed validation summary')
        if state not in ('data', 'ended') or not table_rows:
          raise ValueError('validation summary without a complete nonempty table')
        state = 'ended'
        summaries += 1
        continue
      if 'out-of-place' in line or 'in-place' in line:
        if not _LAYOUT.fullmatch(line):
          raise ValueError('expected unprefixed dual layout: out-of-place then in-place')
        if state not in ('startup', 'ended'):
          raise ValueError('new layout before preceding table and summary were complete')
        if not startup or len(ranks) != metadata['ranks']:
          raise ValueError('missing complete startup/Rank evidence before table')
        state, table_rows = 'header', 0
        continue
      tokens = line.split()
      if re.search(r'\bsize\s+count\b', line) or line.startswith('# size'):
        if state != 'header' or tokens != header:
          if any(token in _TIME_KINDS and token != kind for token in tokens):
            raise ValueError('header time_kind does not match spec: {}'.format(kind))
          raise ValueError('expected fixed schema13 column header after dual layout')
        state = 'units'
        continue
      if '(B)' in line or '(elements)' in line or '(GB/s)' in line or '(us)' in line:
        if state != 'units' or tokens != _UNITS:
          raise ValueError('expected exact schema13 units: bytes, elements, us and decimal GB/s')
        state = 'data'
        continue
      if line.startswith('#'):
        continue
      if state == 'startup':
        if _NUMBER.fullmatch(tokens[0]):
          raise ValueError('numeric row before a complete table header')
        continue
      if state != 'data':
        raise ValueError('unexpected text while awaiting {}; incomplete table'.format(state))
      if len(tokens) != 13:
        raise ValueError('numeric row must have exactly 13 columns; truncated or contaminated row')
      if not re.fullmatch(r'[0-9]+', tokens[0]) or not re.fullmatch(r'[0-9]+', tokens[1]):
        raise ValueError('size and count must be unprefixed decimal integers')
      size, count = int(tokens[0]), int(tokens[1])
      if size not in expected:
        raise ValueError('unexpected size: {}'.format(size))
      if size in seen:
        raise ValueError('duplicate size: {}'.format(size))
      if count != size // 4:
        raise ValueError('count must equal size / 4 for float32 AllReduce')
      if tokens[2] != metadata['dtype'] or tokens[3] != metadata['redop']:
        raise ValueError('row dtype/redop does not match spec')
      if tokens[4] != '-1':
        raise ValueError('AllReduce root must be -1')
      for placement, offset in (('out-of-place', 5), ('in-place', 9)):
        values = {name: _measurement(tokens[offset + index], name)
                  for index, name in enumerate(_MEASUREMENTS)}
        if _decimal(tokens[offset + 3], '#wrong') != 0:
          raise ValueError('#wrong must be numeric zero in both layouts')
        rows.append({**metadata, 'size_bytes': size, 'count': count, 'placement': placement,
                     **values, 'source_line': line_number})
      seen.add(size)
      table_rows += 1
    except ValueError as error:
      raise ValueError('line {}: {}'.format(line_number, error)) from error
  if state != 'ended' or not summaries:
    raise ValueError('incomplete log: missing table, header, units, numeric row or successful summary')
  if seen != expected:
    raise ValueError('missing expected sizes: {}'.format(sorted(expected - seen)))
  return rows


def _validated_rows(rows):
  try:
    result = [dict(row) for row in rows]
  except (TypeError, ValueError) as error:
    raise ValueError('rows must be an iterable of records') from error
  if not result or not result[0]:
    raise ValueError('rows must be nonempty records')
  fields = set(result[0])
  if any(not isinstance(field, str) or not field for field in fields):
    raise ValueError('field names must be nonempty strings')
  for row in result:
    if set(row) != fields:
      raise ValueError('all rows must have identical fields')
    for field, value in row.items():
      if type(value) not in (str, int, float):
        raise ValueError('{} must be a string or number, not nested data or null'.format(field))
      if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('{} must be finite; NaN/Inf cannot be written'.format(field))
      if field in _MEASUREMENTS:
        if type(value) not in (int, float):
          raise ValueError('{} must be a numeric measurement'.format(field))
        _measurement(str(value), field)
  return result


def write_metrics(rows, directory):
  """Atomically publish each file without replacement; roll back on exceptions."""
  rows = _validated_rows(rows)
  directory = Path(directory)
  if not directory.is_dir():
    raise ValueError('metrics directory must already exist')
  targets = [directory / 'metrics.jsonl', directory / 'metrics.csv']
  for target in targets:
    if os.path.lexists(target):
      raise FileExistsError('refusing to overwrite {}'.format(target))
  staged, published = [], []
  complete = False
  try:
    for target in targets:
      fd, name = tempfile.mkstemp(prefix='.metrics-', suffix='.tmp', dir=str(directory))
      temporary = Path(name)
      staged.append(temporary)
      # Published evidence must stay readable by the summary job, which runs as a
      # different user than the executor container.
      os.fchmod(fd, 0o644)
      with os.fdopen(fd, 'w', encoding='utf-8', newline='') as stream:
        if target.suffix == '.jsonl':
          for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        else:
          writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
          writer.writeheader()
          writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    for temporary, target in zip(staged, targets):
      identity = temporary.stat()
      os.link(temporary, target)
      published.append((target, identity))
    for temporary in staged:
      temporary.unlink()
    complete = True
  finally:
    if not complete:
      for target, identity in reversed(published):
        try:
          current = target.lstat()
          if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
            target.unlink()
        except FileNotFoundError:
          pass
    for temporary in staged:
      try:
        temporary.unlink()
      except FileNotFoundError:
        pass


def _groups(rows):
  groups = {}
  for row in _validated_rows(rows):
    if row.get('schema_version') != 2 or row.get('data_status') != 'VALID':
      raise ValueError('summary requires VALID schema_version=2 metrics')
    _string(row.get('case_key'), 'case_key')
    if not re.fullmatch(r'[0-9a-f]{64}', _string(row.get('measurement_identity'), 'measurement_identity')):
      raise ValueError('summary requires a measurement_identity SHA256')
    _integer(row.get('sample_index'), 'sample_index', 0)
    for name in _MEASUREMENTS:
      if type(row.get(name)) not in (int, float):
        raise ValueError('summary requires numeric {}'.format(name))
      _measurement(str(row[name]), name)
    key = tuple(sorted((field, value) for field, value in row.items() if field not in _NOT_COMPARABLE))
    group = groups.setdefault(key, [])
    identity = (row['case_key'], row['sample_index'])
    if any((sample['case_key'], sample['sample_index']) == identity for sample in group):
      raise ValueError('duplicate case_key/sample_index is not an independent sample')
    group.append(row)
  return groups


def _sample_statistics(samples):
  result, unstable = {}, False
  for name in _MEASUREMENTS:
    values = [sample[name] for sample in samples]
    center = median(values)
    if not math.isfinite(center):
      raise ValueError('non-finite median for {}'.format(name))
    spread = max(values) - min(values)
    relative = spread / center if center else (0.0 if spread == 0 else None)
    if relative is not None and not math.isfinite(relative):
      raise ValueError('non-finite relative spread for {}'.format(name))
    result[name + '_median'] = center
    result[name + '_relative_spread'] = relative
    unstable |= relative is None or relative > 0.1
  result['sample_count'] = len(samples)
  result['sample_status'] = ('INSUFFICIENT_SAMPLES' if len(samples) < 3 else
                             'UNSTABLE' if unstable else 'STABLE')
  return result


def summarize_metrics(rows, baselines=None) -> list[dict]:
  """Observe comparable raw samples; topology is unverified and never yields PASS."""
  groups = _groups(rows)
  references = _groups(baselines) if baselines is not None and len(baselines) else {}
  summaries = []
  for key, samples in groups.items():
    result = {**dict(key), **_sample_statistics(samples), 'topology_status': 'UNVERIFIED',
              'comparison_status': 'NO_BASELINE'}
    reference = references.get(key)
    if reference:
      baseline = _sample_statistics(reference)
      result['baseline_sample_count'] = baseline['sample_count']
      statuses = (result['sample_status'], baseline['sample_status'])
      if 'INSUFFICIENT_SAMPLES' in statuses:
        result['comparison_status'] = 'INSUFFICIENT_SAMPLES'
      elif 'UNSTABLE' in statuses:
        result['comparison_status'] = 'UNSTABLE'
      else:
        result['comparison_status'] = 'TOPOLOGY_UNVERIFIED'
      if 'INSUFFICIENT_SAMPLES' not in statuses:
        for name in _MEASUREMENTS:
          value, base = result[name + '_median'], baseline[name + '_median']
          delta = (value / base - 1) * 100 if base else None
          if delta is not None and not math.isfinite(delta):
            raise ValueError('non-finite percentage delta for {}'.format(name))
          result[name + '_delta_percent'] = delta
    summaries.append(result)
  return summaries
