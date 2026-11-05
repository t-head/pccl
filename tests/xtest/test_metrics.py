# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic schema13 examples only; no successful hardware log is claimed."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import metrics


SYNTHETIC_LAYOUT = '#                       out-of-place                       in-place'
SYNTHETIC_HEADER = '# size count type redop root cpu_time algbw busbw #wrong cpu_time algbw busbw #wrong'
SYNTHETIC_UNITS = '# (B) (elements) (us) (GB/s) (GB/s) (us) (GB/s) (GB/s)'
SYNTHETIC_ROW = '16 4 float sum -1 12.5 1.25 0.75 0 10 1.5 0.9 0'
SYNTHETIC_ROW_32 = '32 8 float sum -1 14 2.25 1.75 0 11 2.5 1.9 0'
SYNTHETIC_SUMMARY = '# Out of bounds values : 0 OK'
SYNTHETIC_RANKS = (
  '#   Rank  0 Group  0 Pid   12345 on synthetic-host.invalid device  0 [0x10] synthetic PPU',
  '#   Rank  1 Group  0 Pid   12345 on synthetic-host.invalid device  1 [0x11] synthetic PPU',
)


def synthetic_startup(*, kind='cpu_time', sizes=(16,)):
  elapsed = 'kernel_time' if kind == 'hgpti_time' else kind
  return '\n'.join((
    '# SYNTHETIC fixture, not hardware evidence',
    f'# nThread 2 nGpus 1 minBytes {min(sizes)} maxBytes {max(sizes)} offset <0>/<0> '
    'step: 1048576(bytes) warmup iters: 5 iters: 100 redop: sum validation: 1 single_test: 0 '
    f'test_buffer_kind: both elapsed_type: {elapsed} average: MIN register: 0',
    *SYNTHETIC_RANKS,
  )) + '\n'


def synthetic_table(*rows, kind='cpu_time', sizes=(16,), startup=True):
  # All valid parser/CLI fixtures carry the public startup format; no production
  # bypass exists. Set startup=False only for additional tables or negative tests.
  return ((synthetic_startup(kind=kind, sizes=sizes) if startup else '')
          + '\n'.join((SYNTHETIC_LAYOUT, SYNTHETIC_HEADER.replace('cpu_time', kind),
                       SYNTHETIC_UNITS, *rows, SYNTHETIC_SUMMARY)) + '\n')


def synthetic_spec():
  return {
    'group': 'synthetic', 'tier': 'sanity', 'test_type': 'performance', 'build_profile': 'standard',
    'nodes': 1, 'cards_per_node': 8, 'processes_per_node': 1, 'threads_per_process': 2,
    'devices_per_thread': 1, 'argv': ['synthetic-placeholder'], 'env': {'SECRET': 'never-output'},
    'timeout': 60, 'checks': ['exit_code', 'validation_summary'], 'legacy_source': 'synthetic',
    'parameters': {'op': 'all_reduce_perf', 'dtype': 'float', 'redop': 'sum', 'ranks': 2,
                   'iters': 100, 'warmup': 5},
    'performance': {'expected_sizes': [16], 'time_kind': 'cpu_time', 'rank_reduction': 'MIN',
                    'sample_index': 0},
  }


class ParserTests(unittest.TestCase):
  def setUp(self):
    self.spec = synthetic_spec()
    self.inputs = {'pccl_revision': 'synthetic-pccl', 'pccl_tests_revision': '3945565',
                   'executor_sha256': 'a' * 64, 'executor_python_version': '3.11.0-synthetic',
                   'env': {'PRIVATE': 'never-output'}}
    self.text = 'INFO synthetic startup only\n' + synthetic_table(SYNTHETIC_ROW)

  def parse(self, text=None):
    return metrics.parse_metrics(self.text if text is None else text, self.spec, 'synthetic-case',
                                 self.inputs, '/synthetic/command.log')

  def test_dual_layout_fields_units_and_provenance(self):
    before = copy.deepcopy((self.spec, self.inputs))
    out, inside = self.parse()
    common = {
      'schema_version': 2, 'case_key': 'synthetic-case', 'sample_index': 0,
      'measurement_identity': metrics.measurement_identity(self.spec),
      'op': 'all_reduce_perf', 'dtype': 'float', 'redop': 'sum', 'algo': 'Auto', 'proto': 'Auto',
      'transport': 'local', 'time_kind': 'cpu_time', 'time_unit': 'us', 'rank_reduction': 'MIN',
      'bandwidth_unit': 'GB/s', 'size_bytes': 16, 'count': 4, 'ranks': 2, 'nodes': 1,
      'cards_per_node': 8, 'processes_per_node': 1, 'threads_per_process': 2, 'devices_per_thread': 1,
      'iters': 100, 'warmup': 5, 'build_profile': 'standard', 'pccl_revision': 'synthetic-pccl',
      'pccl_tests_revision': '3945565', 'executor_sha256': 'a' * 64,
      'executor_python_version': '3.11.0-synthetic', 'source_line': 9,
      'log_path': '/synthetic/command.log', 'log_sha256': hashlib.sha256(self.text.encode('utf-8')).hexdigest(),
      'data_status': 'VALID', 'comparison_status': 'NO_BASELINE',
    }
    self.assertEqual(out, {**common, 'placement': 'out-of-place', 'time': 12.5, 'algbw': 1.25, 'busbw': 0.75})
    self.assertEqual(inside, {**common, 'placement': 'in-place', 'time': 10.0, 'algbw': 1.5, 'busbw': 0.9})
    self.assertEqual((self.spec, self.inputs), before)
    self.assertNotIn('never-output', json.dumps([out, inside]))
    self.assertNotIn('synthetic-host.invalid', json.dumps([out, inside]))

  def test_metadata_requires_matching_startup_not_runner_duration_or_nested_resources(self):
    self.spec['parameters'].update(algo='Ring', proto='LL', transport='socket', nodes=99, duration=999)
    self.spec['performance'].update(sample_index=7, rank_reduction='AVG')
    with self.assertRaisesRegex(ValueError, 'startup average'):
      self.parse()
    row = self.parse(self.text.replace('average: MIN', 'average: AVG'))[0]
    self.assertEqual([row[key] for key in ('algo', 'proto', 'transport', 'sample_index', 'rank_reduction')],
                     ['Ring', 'LL', 'socket', 7, 'AVG'])
    self.assertEqual(row['nodes'], 1)
    self.assertEqual(row['time'], 12.5)
    self.assertNotIn('duration', row)
    del self.spec['performance']['rank_reduction']
    self.assertEqual(self.parse()[0]['rank_reduction'], 'MIN')

  def test_all_time_kinds_require_both_header_positions(self):
    for kind in ('cpu_time', 'event_time', 'hgpti_time'):
      self.spec['performance']['time_kind'] = kind
      text = synthetic_table(SYNTHETIC_ROW, kind=kind)
      with self.subTest(kind=kind):
        self.assertEqual(self.parse(text)[0]['time_kind'], kind)
      other = 'event_time' if kind == 'cpu_time' else 'cpu_time'
      for position in (6, 10):
        header = SYNTHETIC_HEADER.replace('cpu_time', kind).split()
        header[position] = other
        bad = text.replace(SYNTHETIC_HEADER.replace('cpu_time', kind), ' '.join(header))
        with self.subTest(kind=kind, position=position), self.assertRaisesRegex(ValueError, 'time_kind'):
          self.parse(bad)

  def test_multiple_complete_tables_and_multiple_rows(self):
    self.spec['performance']['expected_sizes'] = [32, 16]
    multiple = (synthetic_table(SYNTHETIC_ROW, sizes=(16, 32))
                + synthetic_table(SYNTHETIC_ROW_32, startup=False))
    for text in (synthetic_table(SYNTHETIC_ROW, SYNTHETIC_ROW_32, sizes=(16, 32)), multiple):
      with self.subTest(text=text):
        rows = self.parse(text)
        self.assertEqual([row['size_bytes'] for row in rows], [16, 16, 32, 32])
    self.assertEqual(self.parse(multiple)[2]['source_line'], 13)

  def test_empty_truncated_missing_units_or_summary(self):
    lines = synthetic_table(SYNTHETIC_ROW).splitlines()
    for text in ('', 'INFO synthetic startup\n', synthetic_table(),
                 *('\n'.join(lines[:end]) for end in range(1, len(lines))),
                 self.text.replace(SYNTHETIC_UNITS, ''),
                 self.text + SYNTHETIC_LAYOUT, self.text + SYNTHETIC_HEADER,
                 self.text + SYNTHETIC_LAYOUT + '\n' + SYNTHETIC_HEADER + '\n' + SYNTHETIC_UNITS):
      with self.subTest(text=text), self.assertRaises(ValueError):
        self.parse(text)

  def test_each_table_needs_rows_and_its_own_summary(self):
    self.spec['performance']['expected_sizes'] = [16, 32]
    texts = (synthetic_table(sizes=(16, 32)) + synthetic_table(SYNTHETIC_ROW, SYNTHETIC_ROW_32, startup=False),
             synthetic_table(SYNTHETIC_ROW, sizes=(16, 32)).replace(SYNTHETIC_SUMMARY, '')
             + synthetic_table(SYNTHETIC_ROW_32, startup=False),
             synthetic_table(SYNTHETIC_ROW, SYNTHETIC_ROW_32, sizes=(16, 32)) + synthetic_table(startup=False))
    for text in texts:
      with self.subTest(text=text), self.assertRaises(ValueError):
        self.parse(text)

  def test_duplicate_and_missing_sizes(self):
    for text in (synthetic_table(SYNTHETIC_ROW, SYNTHETIC_ROW),
                 self.text + synthetic_table(SYNTHETIC_ROW, startup=False)):
      with self.subTest(text=text), self.assertRaisesRegex(ValueError, 'duplicate size'):
        self.parse(text)
    self.spec['performance']['expected_sizes'] = [16, 32]
    with self.assertRaisesRegex(ValueError, 'missing expected sizes'):
      self.parse(synthetic_table(SYNTHETIC_ROW, sizes=(16, 32)))

  def test_reject_wrong_layout_column_counts_and_reordered_units(self):
    for layout in ('# in-place', '# out-of-place', '# in-place out-of-place',
                   '[rank0] ' + SYNTHETIC_LAYOUT):
      with self.subTest(layout=layout), self.assertRaisesRegex(ValueError, 'dual layout'):
        self.parse(self.text.replace(SYNTHETIC_LAYOUT, layout))
    for columns in (9, 15, 23):
      header = SYNTHETIC_HEADER.split()[:columns + 1]
      row = SYNTHETIC_ROW.split()[:columns]
      if columns > 13:
        header += ['api_time'] * (columns - 13)
        row += ['0'] * (columns - 13)
      with self.subTest(header_columns=columns), self.assertRaisesRegex(ValueError, 'schema13'):
        self.parse(self.text.replace(SYNTHETIC_HEADER, ' '.join(header)))
      with self.subTest(row_columns=columns), self.assertRaisesRegex(ValueError, '13 columns'):
        self.parse(self.text.replace(SYNTHETIC_ROW, ' '.join(row)))
    for units in (SYNTHETIC_UNITS.replace('(us)', '(ms)'),
                  SYNTHETIC_UNITS.replace('(GB/s)', '(GiB/s)'),
                  SYNTHETIC_UNITS.replace('(B) (elements)', '(elements) (B)'),
                  '# (B) (elements) (us) (GB/s) (GB/s)'):
      with self.subTest(units=units), self.assertRaisesRegex(ValueError, 'units'):
        self.parse(self.text.replace(SYNTHETIC_UNITS, units))

  def test_wrong_must_be_numeric_exact_zero(self):
    for position in (8, 12):
      for value in ('1', '-1', 'N/A', 'NaN', 'Inf', '1e-999', '-1e-999'):
        row = SYNTHETIC_ROW.split()
        row[position] = value
        with self.subTest(position=position, value=value), self.assertRaisesRegex(ValueError, '#wrong'):
          self.parse(synthetic_table(' '.join(row)))
      for value in ('0.0', '-0', '0e20'):
        row = SYNTHETIC_ROW.split()
        row[position] = value
        self.assertEqual(len(self.parse(synthetic_table(' '.join(row)))), 2)

  def test_all_measurements_finite_and_nonnegative_and_time_positive(self):
    for position in (5, 6, 7, 9, 10, 11):
      for value in ('NaN', 'Inf', '-Inf', '1e999', '-1', '-1e-999', '1_000', 'N/A'):
        row = SYNTHETIC_ROW.split()
        row[position] = value
        with self.subTest(position=position, value=value), self.assertRaises(ValueError):
          self.parse(synthetic_table(' '.join(row)))
      row = SYNTHETIC_ROW.split()
      row[position] = '0'
      if position in (5, 9):
        with self.assertRaisesRegex(ValueError, 'time must be > 0'):
          self.parse(synthetic_table(' '.join(row)))
      else:
        self.assertEqual(len(self.parse(synthetic_table(' '.join(row)))), 2)
    row = SYNTHETIC_ROW.replace('12.5', '1e-999')
    with self.assertRaisesRegex(ValueError, 'time must be > 0'):
      self.parse(synthetic_table(row))

  def test_failure_footer_cannot_be_hidden_by_a_success_footer(self):
    for footer in ('# Out of bounds values : 1 OK', '# Out of bounds values : 0 FAILED',
                   '# Out of bounds values : 7 FAILED', '# Out of bounds values : N/A OK',
                   '# Out of bounds values : 0 OK trailing', '# Out of bounds values :',
                   '[rank0] ' + SYNTHETIC_SUMMARY, '# test FAILED'):
      for text in (self.text.replace(SYNTHETIC_SUMMARY, footer), self.text + footer,
                   footer + '\n' + self.text):
        with self.subTest(footer=footer, text=text), self.assertRaises(ValueError):
          self.parse(text)
    self.assertEqual(len(self.parse(self.text + SYNTHETIC_SUMMARY + '\n# Avg bus bandwidth : 0.9\n')), 2)

  def test_no_cross_line_repair_or_prefix_stripping(self):
    parts = SYNTHETIC_ROW.split()
    for row in ('16 4 INFO synthetic inserted\n' + ' '.join(parts[2:]),
                '16 4\nINFO synthetic inserted\n' + ' '.join(parts[2:]),
                '16 4\n' + ' '.join(parts[2:]), '[rank0] ' + SYNTHETIC_ROW,
                SYNTHETIC_ROW + ' INFO inserted', SYNTHETIC_ROW + '\nINFO inserted'):
      with self.subTest(row=row), self.assertRaises(ValueError):
        self.parse(synthetic_table(row))
    for text in (self.text + '\nINFO after summary',
                 self.text.replace(SYNTHETIC_HEADER, 'INFO inserted\n' + SYNTHETIC_HEADER),
                 self.text.replace(SYNTHETIC_UNITS, '[rank0] ' + SYNTHETIC_UNITS)):
      with self.subTest(text=text), self.assertRaises(ValueError):
        self.parse(text)

  def test_size_count_dtype_redop_root_matching(self):
    for position, value, reason in ((0, '20', 'unexpected size'), (1, '5', 'count'),
                                    (1, '4.0', 'integers'), (2, 'half', 'dtype/redop'),
                                    (3, 'max', 'dtype/redop'), (4, '0', 'root')):
      row = SYNTHETIC_ROW.split()
      row[position] = value
      with self.subTest(position=position), self.assertRaisesRegex(ValueError, reason):
        self.parse(synthetic_table(' '.join(row)))

  def test_spec_validation(self):
    original = synthetic_spec()
    for field, values in (('expected_sizes', ([], [0], [-4], [6], [True], [16.0], [16, 16], '16')),
                          ('time_kind', ('runner_duration',)), ('rank_reduction', ('AUTO',)),
                          ('sample_index', (-1, True))):
      for value in values:
        self.spec = copy.deepcopy(original)
        self.spec['performance'][field] = value
        with self.subTest(field=field, value=value), self.assertRaises(ValueError):
          self.parse()
    for container, field, value in (('parameters', 'op', 'reduce_scatter_perf'),
                                    ('parameters', 'dtype', 'half'), ('parameters', 'iters', 0),
                                    ('parameters', 'redop', None), ('parameters', 'redop', 'max'),
                                    ('parameters', 'ranks', 4), ('parameters', 'warmup', -1)):
      self.spec = copy.deepcopy(original)
      self.spec[container][field] = value
      with self.subTest(field=field), self.assertRaises(ValueError):
        self.parse()
    self.spec = original
    self.spec['nodes'] = 0
    with self.assertRaisesRegex(ValueError, 'nodes'):
      self.parse()

  def test_missing_provenance_is_not_fabricated(self):
    for name in ('pccl_revision', 'pccl_tests_revision', 'executor_sha256', 'executor_python_version'):
      original = self.inputs.pop(name)
      with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
        self.parse()
      self.inputs[name] = original

  def test_executor_hash_requires_exact_lowercase_sha256(self):
    for value in (None, 123, True, b'a' * 64, [], {}, '', ' ', 'a' * 63, 'a' * 65,
                  'A' * 64, 'g' * 64, '0x' + 'a' * 64, ' ' + 'a' * 64, 'a' * 64 + '\n'):
      self.inputs['executor_sha256'] = value
      with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'executor_sha256'):
        self.parse()
    for value in ('a' * 64, '0123456789abcdef' * 4):
      self.inputs['executor_sha256'] = value
      with self.subTest(value=value):
        self.assertEqual(self.parse()[0]['executor_sha256'], value)

  def test_executor_python_version_requires_nonempty_string(self):
    for value in (None, 311, 3.11, True, b'3.11.0', [], {}, '', ' ', '\t\n'):
      self.inputs['executor_python_version'] = value
      with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'executor_python_version'):
        self.parse()
    version = '3.12.0 (synthetic build, synthetic date) [synthetic compiler]'
    self.inputs['executor_python_version'] = version
    self.assertEqual(self.parse()[0]['executor_python_version'], version)

  def test_xtest_version_is_not_exported_or_a_provenance_fallback(self):
    expected = self.parse()
    self.inputs['xtest_version'] = 'synthetic-legacy-xtest'
    self.assertEqual(self.parse(), expected)
    self.assertTrue(all('xtest_version' not in row for row in expected))
    for name in ('executor_sha256', 'executor_python_version'):
      original = self.inputs.pop(name)
      with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
        self.parse()
      self.inputs[name] = original

  def test_hash_uses_entire_raw_utf8_log(self):
    text = ('INFO synthetic 启动\n' + synthetic_table(SYNTHETIC_ROW)).replace('\n', '\r\n')
    row = self.parse(text)[0]
    self.assertEqual(row['log_sha256'], hashlib.sha256(text.encode('utf-8')).hexdigest())
    self.assertEqual(row['source_line'], 9)

  def test_timestamped_startup_allowed_but_stray_numeric_rows_rejected(self):
    text = '2026-11-03 00:00:00 INFO synthetic startup\n[rank0] INFO synthetic startup\n' + self.text
    self.assertEqual(len(self.parse(text)), 2)
    with self.assertRaisesRegex(ValueError, 'numeric row before'):
      self.parse(SYNTHETIC_ROW + '\n' + self.text)

  def test_extreme_decimal_exponents_raise_value_error(self):
    for position in (5, 8):
      row = SYNTHETIC_ROW.split()
      row[position] = '1e999999999999999999999999999999'
      with self.subTest(position=position), self.assertRaisesRegex(ValueError, 'line 8:'):
        self.parse(synthetic_table(' '.join(row)))

  def test_reproduced_false_metadata_is_rejected(self):
    text = (self.text.replace('nThread 2', 'nThread 1').replace('iters: 100', 'iters: 1')
            .replace('warmup iters: 5', 'warmup iters: 0').replace('average: MIN', 'average: AVG'))
    with self.assertRaisesRegex(ValueError, 'startup'):
      self.parse(text)

  def test_each_startup_declaration_is_bound_to_evidence(self):
    for old, new in (('nThread 2', 'nThread 1'), ('nGpus 1', 'nGpus 2'),
                     ('minBytes 16', 'minBytes 4'), ('maxBytes 16', 'maxBytes 32'),
                     ('warmup iters: 5', 'warmup iters: 0'), ('iters: 100', 'iters: 1'),
                     ('redop: sum', 'redop: max'), ('validation: 1', 'validation: 0'),
                     ('test_buffer_kind: both', 'test_buffer_kind: inplace'),
                     ('test_buffer_kind: both', 'test_buffer_kind: outplace'),
                     ('elapsed_type: cpu_time', 'elapsed_type: event_time'), ('average: MIN', 'average: AVG'),
                     ('offset <0>/<0>', 'offset <4>/<0>'), ('step: 1048576(bytes)', 'step: 2(factor)'),
                     ('single_test: 0', 'single_test: 1'), ('register: 0', 'register: 1')):
      with self.subTest(field=old), self.assertRaisesRegex(ValueError, 'startup'):
        self.parse(self.text.replace(old, new))
    self.spec['performance']['time_kind'] = 'hgpti_time'
    with self.assertRaisesRegex(ValueError, 'startup elapsed_type'):
      self.parse(synthetic_table(SYNTHETIC_ROW, kind='hgpti_time')
                 .replace('elapsed_type: kernel_time', 'elapsed_type: hgpti_time'))

  def test_missing_duplicate_or_misplaced_startup_and_rank_evidence(self):
    startup_line = synthetic_startup().splitlines()[1]
    for text in (synthetic_table(SYNTHETIC_ROW, startup=False),
                 self.text.replace(startup_line, ''), self.text.replace(SYNTHETIC_RANKS[1], ''),
                 self.text.replace(startup_line, startup_line + '\n' + startup_line),
                 self.text.replace(SYNTHETIC_RANKS[1], SYNTHETIC_RANKS[0]),
                 self.text.replace(startup_line, '[rank0] ' + startup_line),
                 self.text.replace(SYNTHETIC_RANKS[0], '[rank0] ' + SYNTHETIC_RANKS[0]),
                 self.text + startup_line, self.text + '\n' + SYNTHETIC_RANKS[1],
                 SYNTHETIC_RANKS[0] + '\n' + self.text):
      with self.subTest(text=text), self.assertRaisesRegex(ValueError, 'startup|Rank'):
        self.parse(text)

  def test_contradictory_rank_records_are_rejected_without_leaking_hosts(self):
    rank = SYNTHETIC_RANKS[1]
    for altered in (rank.replace('Rank  1', 'Rank  2'), rank.replace('Group  0', 'Group  1'),
                    rank.replace('12345', '12346'), rank.replace('12345', '0'),
                    rank.replace('device  1', 'device  0'),
                    rank.replace('synthetic-host.invalid', '192.0.2.1'), '#   Rank broken'):
      with self.subTest(rank=altered), self.assertRaisesRegex(ValueError, 'Rank') as raised:
        self.parse(self.text.replace(rank, altered))
      self.assertNotIn('192.0.2.1', str(raised.exception))
      self.assertNotIn('synthetic-host.invalid', str(raised.exception))
    reversed_ranks = self.text.replace('\n'.join(SYNTHETIC_RANKS), '\n'.join(reversed(SYNTHETIC_RANKS)))
    with self.assertRaisesRegex(ValueError, 'Rank'):
      self.parse(reversed_ranks)

  def test_startup_cannot_be_borrowed_across_concatenated_launches(self):
    self.spec['performance']['expected_sizes'] = [16, 32]
    first = synthetic_table(SYNTHETIC_ROW, sizes=(16, 32))
    for second in (synthetic_table(SYNTHETIC_ROW_32, sizes=(16, 32)),
                   synthetic_table(SYNTHETIC_ROW_32, sizes=(16, 32)).replace('iters: 100', 'iters: 1')):
      with self.subTest(second=second), self.assertRaisesRegex(ValueError, 'duplicate.*startup'):
        self.parse(first + second)


class WriterTests(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.directory = Path(temporary.name)
    self.rows = metrics.parse_metrics(synthetic_table(SYNTHETIC_ROW), synthetic_spec(), 'synthetic-case',
                                     {'pccl_revision': 'synthetic-pccl', 'pccl_tests_revision': '3945565',
                                      'executor_sha256': 'a' * 64, 'executor_python_version': '3.11.0-synthetic'},
                                     '/synthetic/log,with-逗号.log')

  def test_csv_json_fields_values_and_numbers_match(self):
    before = copy.deepcopy(self.rows)
    self.rows[1] = dict(reversed(list(self.rows[1].items())))
    metrics.write_metrics(self.rows, self.directory)
    json_rows = [json.loads(line) for line in (self.directory / 'metrics.jsonl').read_text().splitlines()]
    with (self.directory / 'metrics.csv').open(newline='', encoding='utf-8') as stream:
      csv_rows = list(csv.DictReader(stream))
    self.assertEqual(json_rows, before)
    self.assertEqual(self.rows, before)
    for csv_row, json_row in zip(csv_rows, json_rows):
      self.assertEqual(set(csv_row), set(json_row))
      restored = {field: type(json_row[field])(value) for field, value in csv_row.items()}
      self.assertEqual(restored, json_row)
    self.assertEqual(sorted(path.name for path in self.directory.iterdir()), ['metrics.csv', 'metrics.jsonl'])
    for name in ('metrics.csv', 'metrics.jsonl'):
      self.assertEqual((self.directory / name).stat().st_mode & 0o777, 0o644, name)

  def test_existing_either_artifact_refuses_all_writes(self):
    for name in ('metrics.csv', 'metrics.jsonl'):
      target = self.directory / name
      target.write_bytes(b'original bytes')
      with self.subTest(name=name), self.assertRaises(FileExistsError):
        metrics.write_metrics(self.rows, self.directory)
      self.assertEqual(target.read_bytes(), b'original bytes')
      self.assertEqual(list(self.directory.iterdir()), [target])
      target.unlink()

  def test_dangling_symlink_is_not_overwritten(self):
    target = self.directory / 'metrics.csv'
    target.symlink_to(self.directory / 'absent')
    with self.assertRaises(FileExistsError):
      metrics.write_metrics(self.rows, self.directory)
    self.assertTrue(target.is_symlink())
    self.assertFalse((self.directory / 'metrics.jsonl').exists())

  def test_reject_empty_invalid_nan_inf_or_different_fields_before_publishing(self):
    invalid = [[], [{}], [{**self.rows[0], 'extra': {}}],
               [self.rows[0], {**self.rows[1], 'extra': 1}]]
    for value in (float('nan'), float('inf'), -float('inf')):
      invalid.append([{**self.rows[0], 'time': value}])
      invalid.append([{**self.rows[0], 'extra': value}])
    for value in ('NaN', 'Inf', '12.5', 0, -1):
      invalid.append([{**self.rows[0], 'time': value}])
    for rows in invalid:
      with self.subTest(rows=rows), self.assertRaises(ValueError):
        metrics.write_metrics(rows, self.directory)
      self.assertEqual(list(self.directory.iterdir()), [])

  def test_does_not_create_missing_output_directory(self):
    missing = self.directory / 'missing'
    with self.assertRaisesRegex(ValueError, 'already exist'):
      metrics.write_metrics(self.rows, missing)
    self.assertFalse(missing.exists())

  def test_second_publish_failure_rolls_back_first_and_cleans_temporaries(self):
    real_link = metrics.os.link

    def link(source, destination):
      if Path(destination).suffix == '.csv':
        raise OSError('synthetic publication failure')
      real_link(source, destination)

    with patch.object(metrics.os, 'link', side_effect=link), self.assertRaisesRegex(OSError, 'synthetic'):
      metrics.write_metrics(self.rows, self.directory)
    self.assertEqual(list(self.directory.iterdir()), [])

  def test_write_or_sync_failure_leaves_no_success_files(self):
    with patch.object(metrics.os, 'fsync', side_effect=OSError('synthetic sync failure')):
      with self.assertRaisesRegex(OSError, 'synthetic'):
        metrics.write_metrics(self.rows, self.directory)
    self.assertEqual(list(self.directory.iterdir()), [])
    with patch.object(metrics.csv.DictWriter, 'writerows', side_effect=OSError('synthetic csv failure')):
      with self.assertRaisesRegex(OSError, 'synthetic'):
        metrics.write_metrics(self.rows, self.directory)
    self.assertEqual(list(self.directory.iterdir()), [])

  def test_publication_race_does_not_clobber_another_writer(self):
    real_link = metrics.os.link

    def link(source, destination):
      if Path(destination).suffix == '.csv':
        Path(destination).write_bytes(b'other writer')
      real_link(source, destination)

    with patch.object(metrics.os, 'link', side_effect=link), self.assertRaises(FileExistsError):
      metrics.write_metrics(self.rows, self.directory)
    self.assertEqual([path.name for path in self.directory.iterdir()], ['metrics.csv'])
    self.assertEqual((self.directory / 'metrics.csv').read_bytes(), b'other writer')


class SummaryTests(unittest.TestCase):
  def samples(self, times=(10, 10, 10), case_key='synthetic-case'):
    result = []
    for index, value in enumerate(times):
      spec = synthetic_spec()
      spec['performance']['sample_index'] = index
      tokens = SYNTHETIC_ROW.split()
      tokens[5] = tokens[9] = str(value)
      result.extend(metrics.parse_metrics(synthetic_table(' '.join(tokens)), spec, case_key,
                                          {'pccl_revision': 'synthetic-pccl', 'pccl_tests_revision': '3945565',
                                           'executor_sha256': 'a' * 64,
                                           'executor_python_version': '3.11.0-synthetic'},
                                          '/synthetic/{}.log'.format(index)))
    return result

  def test_no_baseline_and_sample_quality_are_separate(self):
    for times, quality in (((10, 10, 10), 'STABLE'), ((10, 10), 'INSUFFICIENT_SAMPLES'),
                           ((10, 10, 12), 'UNSTABLE')):
      rows = metrics.summarize_metrics(self.samples(times))
      self.assertEqual(len(rows), 2)
      for row in rows:
        self.assertEqual(row['comparison_status'], 'NO_BASELINE')
        self.assertEqual(row['sample_status'], quality)
        self.assertEqual(row['topology_status'], 'UNVERIFIED')

  def test_valid_reference_gives_median_deltas_never_pass_and_is_not_mutated(self):
    baseline = self.samples(case_key='synthetic-baseline')
    current = self.samples((12, 12, 12))
    for row in current:
      row['algbw'] *= 1.5
      row['busbw'] *= 0.8
    before = copy.deepcopy((baseline, current))
    summaries = metrics.summarize_metrics(current, baseline)
    for row in summaries:
      self.assertEqual(row['comparison_status'], 'TOPOLOGY_UNVERIFIED')
      self.assertEqual(row['sample_count'], 3)
      self.assertEqual(row['baseline_sample_count'], 3)
      self.assertAlmostEqual(row['time_delta_percent'], 20)
      self.assertAlmostEqual(row['algbw_delta_percent'], 50)
      self.assertAlmostEqual(row['busbw_delta_percent'], -20)
      self.assertNotIn('log_path', row)
      self.assertNotIn('case_key', row)
    self.assertEqual((baseline, current), before)

  def test_insufficient_samples_skip_deltas_but_instability_is_observation_only(self):
    stable = self.samples()
    for other, status in ((self.samples((10, 10)), 'INSUFFICIENT_SAMPLES'),
                          (self.samples((10, 10, 12)), 'UNSTABLE')):
      for current, reference in ((other, stable), (stable, other)):
        for row in metrics.summarize_metrics(current, reference):
          self.assertEqual(row['comparison_status'], status)
          self.assertEqual('time_delta_percent' in row, status == 'UNSTABLE')
    boundary = metrics.summarize_metrics(self.samples((10, 10, 11)), stable)
    self.assertEqual(boundary[0]['sample_status'], 'STABLE')

  def test_bandwidth_variation_and_zero_reference(self):
    current = self.samples()
    current[-1]['busbw'] *= 2
    self.assertEqual(metrics.summarize_metrics(current, self.samples())[1]['comparison_status'], 'UNSTABLE')
    baseline = self.samples()
    for row in baseline:
      row['busbw'] = 0.0
    for row in metrics.summarize_metrics(self.samples(), baseline):
      self.assertIsNone(row['busbw_delta_percent'])
      self.assertNotEqual(row['comparison_status'], 'PASS')

  def test_grouping_keeps_configuration_units_revisions_and_topology_distinct(self):
    for field, value in (('size_bytes', 32), ('algo', 'Tree'), ('proto', 'LL'),
                         ('transport', 'rdma'), ('ranks', 4), ('nodes', 2), ('cards_per_node', 4),
                         ('time_kind', 'event_time'), ('rank_reduction', 'AVG'), ('time_unit', 'ms'),
                         ('bandwidth_unit', 'GiB/s'), ('build_profile', 'notsm'), ('iters', 200),
                         ('warmup', 10), ('pccl_revision', 'other'), ('pccl_tests_revision', 'other'),
                         ('executor_sha256', 'b' * 64), ('executor_python_version', '3.12.0-synthetic')):
      baseline = self.samples()
      current = self.samples()
      for row in current:
        row[field] = value
      with self.subTest(field=field):
        self.assertTrue(all(row['comparison_status'] == 'NO_BASELINE'
                            for row in metrics.summarize_metrics(current, baseline)))
        combined = metrics.summarize_metrics(current + baseline)
        self.assertEqual(len(combined), 4)
        self.assertTrue(all(row['sample_count'] == 3 for row in combined))

  def test_same_sample_cannot_be_counted_twice_and_invalid_reference_rejected(self):
    rows = self.samples()
    with self.assertRaisesRegex(ValueError, 'independent sample'):
      metrics.summarize_metrics(rows + [rows[0]])
    baseline = self.samples()
    baseline[0]['data_status'] = 'INVALID'
    with self.assertRaisesRegex(ValueError, 'VALID'):
      metrics.summarize_metrics(rows, baseline)

  def test_old_schema_is_rejected_for_samples_and_baselines_without_fallback(self):
    current = self.samples()
    for legacy_provenance in (False, True):
      old = self.samples(case_key='synthetic-old-schema')
      for row in old:
        row['schema_version'] = 1
        if legacy_provenance:
          del row['executor_sha256']
          del row['executor_python_version']
          row['xtest_version'] = 'synthetic-legacy-xtest'
      for samples, baseline in ((old, None), (old, current), (current, old)):
        with self.subTest(legacy_provenance=legacy_provenance, old_samples=samples is old):
          with self.assertRaisesRegex(ValueError, 'VALID schema_version=2'):
            metrics.summarize_metrics(samples, baseline)

  def test_different_channel_environments_are_not_three_stable_samples(self):
    from manifest import case_key
    from test_manifest import performance_specs

    rows = []
    for spec, channels in zip(performance_specs(), ('1', '2', '4')):
      spec['env'].update(PCCL_MAX_NCHANNELS=channels, NCCL_MAX_NCHANNELS=channels)
      rows.extend(metrics.parse_metrics(
        synthetic_table('8388608 2097152 float sum -1 100 80 80 0 110 72 72 0', sizes=(8388608,)),
        spec, case_key(spec), {'pccl_revision': 'synthetic', 'pccl_tests_revision': 'synthetic',
                              'executor_sha256': 'a' * 64, 'executor_python_version': '3.11.0-synthetic'},
        '/synthetic/log'))
    self.assertEqual(len({row['case_key'] for row in rows}), 3)
    self.assertEqual(len({row['measurement_identity'] for row in rows}), 3)
    summary = metrics.summarize_metrics(rows)
    self.assertEqual(len(summary), 6)
    self.assertTrue(all(row['sample_count'] == 1 and row['sample_status'] == 'INSUFFICIENT_SAMPLES'
                        for row in summary))
    self.assertNotIn('MAX_NCHANNELS', json.dumps(rows + summary))

  def test_measurement_identity_is_required_and_other_settings_split_groups(self):
    rows = self.samples()
    for row in rows:
      del row['measurement_identity']
    with self.assertRaisesRegex(ValueError, 'measurement_identity'):
      metrics.summarize_metrics(rows)
    for field, value in (('argv', ['different-synthetic-command']),
                         ('env', {'HGGC_SYNTHETIC': 'private-synthetic-value'}),
                         ('parameters', {**synthetic_spec()['parameters'], 'extra_measurement_knob': 2})):
      spec = synthetic_spec()
      spec[field] = value
      changed = metrics.parse_metrics(synthetic_table(SYNTHETIC_ROW), spec, 'synthetic-other',
                                      {'pccl_revision': 'synthetic-pccl', 'pccl_tests_revision': '3945565',
                                       'executor_sha256': 'a' * 64, 'executor_python_version': '3.11.0-synthetic'},
                                      '/synthetic/other.log')
      summary = metrics.summarize_metrics(self.samples() + changed)
      with self.subTest(field=field):
        self.assertEqual(len(summary), 4)
        self.assertEqual(sorted(row['sample_count'] for row in summary), [1, 1, 3, 3])
        self.assertNotIn('private-synthetic-value', json.dumps(summary))


if __name__ == '__main__':
  unittest.main()
