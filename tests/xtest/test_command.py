# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic local subprocess tests using only the standard library, without site packages."""

import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

import command

FOOTER = '# Out of bounds values : 0 OK\n'
ROW = '8388608 2097152 float sum -1 100 80 150 0 100 80 150 0\n'


class CommandTests(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory(prefix='pccl command ')
    self.addCleanup(temporary.cleanup)
    self.root = Path(temporary.name).resolve()
    self.binary = self.root / 'bin space'
    self.binary.mkdir()
    self.serial = 0
    self.runtime = dict(test_root=str(self.root), binary_dir=str(self.binary),
                        result_dir=str(self.root / 'result'), mpi_options=[])
    self.spec = dict(argv=['{python}', '-S', '-B', '{binary_dir}/child.py'], env={}, timeout=3,
                     nodes=1, processes_per_node=1, threads_per_process=1, devices_per_thread=1,
                     checks=['exit_code', 'validation_summary'])
    self.env = {key: value for key, value in os.environ.items()
                if not key.startswith(('PCCL_', 'NCCL_', 'HGGC_', 'OMPI_', 'UCX_', 'XTEST_', 'HOLMES_'))
                and key not in ('NODE_RANK', 'PYTHONPATH', 'RERUN_FAILED_CASE')}

  def prepare_case(self, source, spec=None, runtime=None, env=None, name='child.py',
                   before_run='', before_check=''):
    self.serial += 1
    directory = self.root / f'case{self.serial}'
    directory.mkdir()
    case_spec = dict(self.spec, **(spec or {}))
    config = dict(self.runtime, result_dir=str(directory / 'results'))
    config.update(runtime or {})
    (self.binary / name).write_text(source)
    (self.binary / name).chmod(0o700)
    inputs = directory / 'inputs.json'
    inputs.write_text(json.dumps(dict(spec=case_spec, runtime=config)))
    result = directory / 'result.json'
    driver = directory / 'driver.py'
    # Fault injection stays inside this real subprocess and targets only local I/O
    # boundaries. No framework or runtime package is installed, imported, or mocked.
    driver.write_text(
      'import json, os, signal, sys\nfrom pathlib import Path\n'
      'sys.path.insert(0, sys.argv[1])\nimport command\n'
      'inputs = json.loads(Path(sys.argv[2]).read_text())\n'
      'before = json.dumps(inputs, sort_keys=True)\n'
      'spec, runtime = inputs["spec"], inputs["runtime"]\n'
      'handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}\n'
      'data = dict(status="FAILED", run_returned=False, check_called=False)\n'
      'try:\n'
      '  with command.cancellation_signals() as interrupted:\n'
      + textwrap.indent(before_run, '    ') + '\n'
      '    record = command.run_command(spec, runtime, interrupted)\n'
      '    data.update(run_returned=True, record=dict(record))\n'
      + textwrap.indent(before_check, '    ') + '\n'
      '    data["check_called"] = True\n'
      '    data["metrics"] = command.check_command(spec, runtime, record)\n'
      '    data["status"] = "PASSED"\n'
      'except BaseException as error:\n'
      '  data.update(error=str(error), error_type=type(error).__name__,\n'
      '              notes=getattr(error, "__notes__", []),\n'
      '              returncode=getattr(error, "returncode", None),\n'
      '              timeout=getattr(error, "timeout", None))\n'
      '  path = Path(runtime["result_dir"]) / "command-process.json"\n'
      '  if path.is_file():\n'
      '    try:\n'
      '      command.check_command(spec, runtime, json.loads(path.read_text()))\n'
      '    except ValueError:\n'
      '      data["record_rejected"] = True\n'
      'finally:\n'
      '  data["handlers_restored"] = all(signal.getsignal(sig) == handler for sig, handler in handlers.items())\n'
      '  data["inputs_unchanged"] = before == json.dumps(inputs, sort_keys=True)\n'
      '  data["framework_modules"] = [name for name in sys.modules\n'
      '                               if name.split(".")[0] in ("xtest", "mmengine")]\n'
      '  data["no_site"] = sys.flags.no_site\n'
      '  data["no_bytecode"] = sys.dont_write_bytecode\n'
      '  data["python"] = sys.executable\n'
      '  Path(sys.argv[3]).write_text(json.dumps(data))\n'
      'sys.exit(0 if data["status"] == "PASSED" else 1)\n')
    invocation = [sys.executable, '-S', '-B', str(driver), str(Path(__file__).resolve().parent),
                  str(inputs), str(result)]
    return invocation, dict(self.env, **(env or {})), result, Path(config['result_dir'])

  def case(self, source, passed=True, **kwargs):
    invocation, env, result, directory = self.prepare_case(source, **kwargs)
    run = subprocess.run(invocation, cwd=self.root, env=env, capture_output=True, text=True, timeout=20)
    diagnostic = run.stdout + run.stderr
    self.assertTrue(result.is_file(), diagnostic)
    data = json.loads(result.read_text())
    diagnostic += json.dumps(data)
    self.assertEqual(run.returncode, 0 if passed else 1, diagnostic)
    self.assertEqual(data['status'], 'PASSED' if passed else 'FAILED', diagnostic)
    self.assertTrue(data['handlers_restored'], diagnostic)
    self.assertTrue(data['inputs_unchanged'], diagnostic)
    self.assertEqual(data['framework_modules'], [])
    self.assertEqual(data['no_site'], 1)
    self.assertTrue(data['no_bytecode'])
    self.assertEqual(data['python'], sys.executable)
    return directory, data, diagnostic

  def test_direct_lifecycle_and_no_framework_artifacts(self):
    output = 'error is a diagnostic word\n' + FOOTER
    directory, data, _ = self.case(f'print({output!r})')
    record = json.loads((directory / 'command-process.json').read_text())
    self.assertEqual((directory / 'command-process.json').stat().st_mode & 0o777, 0o644)
    self.assertEqual(record, data['record'])
    self.assertEqual(set(record), {'active', 'state', 'pid', 'pgid', 'identity', 'return_code'})
    self.assertEqual((record['active'], record['state'], record['return_code']), (False, 'finished', 0))
    self.assertEqual(record['pid'], record['pgid'])
    self.assertTrue(record['identity']['starttime'])
    self.assertEqual(data['metrics'], [])
    self.assertFalse(command.stop_recorded_process(directory / 'command-process.json'))
    self.assertEqual(sorted(path.name for path in directory.iterdir()), ['command-process.json', 'command.log'])
    self.assertEqual(sorted(path.name for path in directory.parent.iterdir()),
                     ['driver.py', 'inputs.json', 'result.json', 'results'])
    self.assertEqual(sorted(path.name for path in self.root.iterdir()), ['bin space', 'case1'])

  def test_nonzero_cannot_be_reset_by_checker(self):
    directory, data, _ = self.case(f'print({FOOTER!r}); raise SystemExit(7)', passed=False)
    self.assertEqual(data['error_type'], 'CalledProcessError')
    self.assertEqual(data['returncode'], 7)
    self.assertFalse(data['run_returned'])
    self.assertFalse(data['check_called'])
    self.assertTrue(data['record_rejected'])
    record = json.loads((directory / 'command-process.json').read_text())
    self.assertEqual((record['state'], record['return_code']), ('failed', 7))

  def test_exit_zero_requires_footer(self):
    for output in ('', 'Usage: all_reduce_perf\n', FOOTER.replace('0 OK', '1 FAILED'),
                   FOOTER.replace('0 OK', '-1 SKIPPED'), FOOTER.replace('0 OK', '0')):
      with self.subTest(output=output):
        _, data, _ = self.case(f'print({output!r})', passed=False)
        self.assertEqual(data['error_type'], 'ValueError')
        self.assertTrue(data['run_returned'])
        self.assertTrue(data['check_called'])

  def test_smoke_preserves_all_old_assertions(self):
    checks = dict(checks=['exit_code', 'smoke_8mib'])
    self.case(f'print({(ROW + FOOTER)!r})', spec=checks)
    bad = ['', 'Usage: all_reduce_perf\n', ROW, ROW + FOOTER.replace('0 OK', '1 FAILED'), FOOTER,
           ROW * 2 + FOOTER, ROW.replace('8388608 2097152', '32 8') + FOOTER,
           ' '.join(ROW.split()[:-1]) + '\n' + FOOTER]
    for index in (8, 12):
      for value in ('1', '-1', 'N/A', '0.0'):
        fields = ROW.split()
        fields[index] = value
        bad.append(' '.join(fields) + '\n' + FOOTER)
    for output in bad:
      with self.subTest(output=output):
        self.case(f'print({output!r})', spec=checks, passed=False)

  def test_literal_argv_and_environment(self):
    hostfile = self.root / 'hosts with spaces'
    hostfile.write_text('localhost\n')
    options = ['--hostfile', str(hostfile), '-x', 'PCCL_DEBUG=a b;$(touch forbidden)']
    extra = ['{mpi_options}', '--mpi-options={mpi_options}', '--path={test_root}/a b',
             '$(touch forbidden)', '`touch forbidden`', ';touch forbidden', '{python}']
    source = ('import json, os, sys\nprint(json.dumps(dict(argv=sys.argv[1:], cwd=os.getcwd(), '
              'python=sys.executable, path=os.environ["PATH"], ld=os.environ["LD_LIBRARY_PATH"], '
              'debug=os.environ.get("PCCL_DEBUG"), net=os.environ.get("NCCL_NET"))))\n')
    directory, _, _ = self.case(source, spec=dict(argv=self.spec['argv'] + extra, checks=['exit_code'],
                                                env={'PCCL_DEBUG': 'a b;$(touch forbidden)', 'NCCL_NET': None}),
                                runtime=dict(mpi_options=options),
                                env={'NCCL_NET': 'IB', 'LD_LIBRARY_PATH': '/base lib'})
    actual = json.loads((directory / 'command.log').read_text())
    self.assertEqual(actual['argv'], options + ['--mpi-options=' + shlex.join(options),
                     f'--path={self.root}/a b', '$(touch forbidden)', '`touch forbidden`',
                     ';touch forbidden', sys.executable])
    self.assertEqual((actual['cwd'], actual['python']), (str(self.root), sys.executable))
    self.assertTrue(actual['path'].startswith(str(self.binary) + os.pathsep))
    self.assertEqual(actual['ld'], str(self.root / 'lib') + ':/base lib')
    self.assertEqual(actual['debug'], 'a b;$(touch forbidden)')
    self.assertIsNone(actual['net'])
    self.assertFalse((self.root / 'forbidden').exists())

  def test_multinode_and_worker_guards(self):
    for options in ([], ['--hostfile'], ['--hostfile', str(self.root / 'missing')]):
      with self.subTest(options=options):
        directory, data, _ = self.case('raise Exception("must not execute")', passed=False,
                                      spec=dict(nodes=2), runtime=dict(mpi_options=options))
        self.assertIn('hostfile', data['error'])
        self.assertFalse((directory / 'command.log').exists())
    self.case(f'print({FOOTER!r})', passed=False, env={'NODE_RANK': '1'})
    self.case(f'print({FOOTER!r})', env={'NODE_RANK': '0'})
    hostfile = self.root / 'hosts'
    hostfile.write_text('localhost slots=2\n')
    options = dict(mpi_options=['--hostfile', str(hostfile)])
    self.case(f'print({FOOTER!r})', passed=False, spec=dict(nodes=2), runtime=options)
    self.case(f'print({FOOTER!r})', passed=False, runtime=options,
              spec=dict(nodes=2, argv=self.spec['argv'] + ['{mpi_options}']))
    # Synthetic launchers verify forwarding only, not MPI or hardware.
    source = (f'#!{sys.executable}\nimport sys\nassert sys.argv[-2:] == {options["mpi_options"]!r}\n'
              f'assert "PATH" in sys.argv and "LD_LIBRARY_PATH" in sys.argv\nprint({FOOTER!r})')
    self.case(source, name='mpirun', spec=dict(nodes=2, argv=['mpirun', '{mpi_options}']), runtime=options)
    argv = ['{python}', '{binary_dir}/run_test.py', '--process-group', 'normal', '--mpi-options={mpi_options}']
    self.case(f'print({FOOTER!r})', name='run_test.py', spec=dict(nodes=2, argv=argv), runtime=options)
    self.case(f'print({FOOTER!r})', name='run_test.py', spec=dict(argv=argv[:2]), passed=False)
    self.case('', name='run_test.py', spec=dict(argv=argv + ['--launch-mode=k8s']), passed=False)

  def test_mpi_environment_forwarding_and_runner_literal_guard(self):
    argv = ['{python}', '{binary_dir}/run_test.py', '--process-group', 'normal',
            '--launch-mode', 'normal', '--mpi-options={mpi_options}']
    spec = dict(argv=argv, processes_per_node=2, checks=['exit_code'])
    for value in ('/tmp/log dir/output', 'a"b', 'a\\b', 'line\nbreak'):
      with self.subTest(value=value):
        directory, data, _ = self.case('', name='run_test.py',
                                      spec=dict(spec, env={'PCCL_DEBUG_FILE': value}), passed=False)
        self.assertEqual(data['error_type'], 'ValueError')
        self.assertFalse((directory / 'command.log').exists())
    self.case('', name='run_test.py', spec=spec, env={'NCCL_DEBUG_FILE': '/tmp/inherited log'}, passed=False)
    source = 'import json, sys\nprint(json.dumps(sys.argv[1:]))'
    directory, _, _ = self.case(source, name='run_test.py', spec=dict(spec, env={'PCCL_DEBUG_FILE': '/tmp/log'}))
    actual = json.loads((directory / 'command.log').read_text())
    self.assertEqual(shlex.split(actual[-1].split('=', 1)[1]), ['-x', 'LD_LIBRARY_PATH', '-x', 'PATH'])
    source = (f'#!{sys.executable}\nimport json, os, sys\n'
              'print(json.dumps(dict(argv=sys.argv[1:], debug=os.environ["PCCL_DEBUG_FILE"])))')
    directory, _, _ = self.case(source, name='mpirun',
                                spec=dict(argv=['mpirun', '{mpi_options}', 'all_reduce_perf'], checks=['exit_code'],
                                          env={'PCCL_DEBUG_FILE': '/tmp/log dir/output', 'PCCL_NET': 'IB'}))
    actual = json.loads((directory / 'command.log').read_text())
    self.assertEqual(actual['argv'], ['-x', 'LD_LIBRARY_PATH', '-x', 'PATH',
                                     '-x', 'PCCL_DEBUG_FILE', '-x', 'PCCL_NET', 'all_reduce_perf'])
    self.assertEqual(actual['debug'], '/tmp/log dir/output')

  def test_invalid_inputs_and_missing_execution_evidence(self):
    for override in (dict(timeout=0), dict(timeout=float('nan')), dict(timeout=True), dict(nodes=0),
                     dict(env={'PATH': '/bad'}), dict(env={'RERUN_FAILED_CASE': '1'}),
                     dict(env={'PCCL_DEBUG': 1}), dict(argv=['{python.__class__}']),
                     dict(argv=['x{mpi_options}']), dict(argv='shell string')):
      with self.subTest(override=override):
        directory, data, _ = self.case('', spec=override, passed=False)
        self.assertEqual(data['error_type'], 'ValueError')
        self.assertFalse((directory / 'command.log').exists())
    self.case('', runtime=dict(test_root='relative'), passed=False)
    final = dict(active=False, state='finished', return_code=0)
    records = [None, {}, {'return_code': 0}, {'active': False, 'return_code': 0},
               {'state': 'finished', 'return_code': 0}]
    records += [dict(final, active=value) for value in (True, None, 0)]
    records += [dict(final, return_code=value) for value in (None, False, 0.0, '0', 1, -9)]
    records += [dict(final, state=value) for value in ('running', 'failed', 'timeout', 'cleanup_failed')]
    for record in records:
      with self.subTest(record=record):
        _, data, _ = self.case(f'print({FOOTER!r})', passed=False, before_check=f'record = {record!r}\n')
        self.assertEqual(data['error_type'], 'ValueError')
        self.assertIn('No successful completed command', data['error'])
    _, data, _ = self.case(f'print({FOOTER!r})', passed=False,
                          before_check='(Path(runtime["result_dir"]) / "command.log").unlink()\n')
    self.assertIn('Missing command log', data['error'])
    for checks in ([], None, 'exit_code', ['unknown'], [{}]):
      with self.subTest(checks=checks):
        _, data, _ = self.case('', spec=dict(checks=checks), passed=False)
        self.assertEqual(data['error_type'], 'ValueError')
        self.assertIn('checks', data['error'])

  def test_log_reuse_and_launch_error(self):
    directory, _, _ = self.case(f'print({FOOTER!r})')
    log_path, record_path = directory / 'command.log', directory / 'command-process.json'
    original, original_record = log_path.read_bytes(), record_path.read_bytes()
    self.case('print("overwrite")', passed=False, runtime=dict(result_dir=str(directory)))
    self.assertEqual(record_path.read_bytes(), original_record)
    self.assertEqual(log_path.read_bytes(), original)
    record_path.unlink()
    self.case('print("overwrite")', passed=False, runtime=dict(result_dir=str(directory)))
    self.assertEqual(log_path.read_bytes(), original)
    self.assertFalse(record_path.exists())
    log_path.unlink()
    record_path.symlink_to(directory / 'missing-record')
    self.case('print("overwrite")', passed=False, runtime=dict(result_dir=str(directory)))
    self.assertTrue(record_path.is_symlink())
    self.assertFalse(log_path.exists())
    directory, data, _ = self.case('', passed=False, spec=dict(argv=[str(self.root / 'no-program')]))
    record = json.loads((directory / 'command-process.json').read_text())
    self.assertFalse(record['active'])
    self.assertIsNone(record['return_code'])
    self.assertEqual(data['error_type'], 'FileNotFoundError')
    self.assertTrue(data['record_rejected'])

  def test_inspection_failure_still_reaps_owned_child(self):
    hook = 'def fail_inspection(pgid):\n  raise OSError("ps failed")\ncommand._members = fail_inspection\n'
    for source, timeout, error_type in (('import time; time.sleep(60)', 0.1, 'TimeoutExpired'),
                                        ('raise SystemExit(7)', 3, 'CalledProcessError'),
                                        ('', 3, 'OSError')):
      with self.subTest(error_type=error_type):
        directory, data, _ = self.case(source, spec=dict(timeout=timeout), passed=False, before_run=hook)
        record = json.loads((directory / 'command-process.json').read_text())
        self.assertEqual(record['state'], 'cleanup_failed')
        self.assertFalse(record['active'])
        self.assertEqual(command._members(record['pgid']), [])
        self.assertEqual(data['error_type'], error_type)
        self.assertFalse(data['run_returned'])
        self.assertTrue(data['record_rejected'])
        if error_type != 'OSError':
          self.assertTrue(any(note.startswith('Cleanup failed:') for note in data['notes']))
        if error_type == 'TimeoutExpired':
          self.assertEqual(data['timeout'], 0.1)
          self.assertLess(record['return_code'], 0)
        elif error_type == 'CalledProcessError':
          self.assertEqual(data['returncode'], 7)

  def test_signal_during_process_record_creation(self):
    for sig, state in ((signal.SIGTERM, 'running'), (signal.SIGINT, 'running'),
                       (signal.SIGTERM, 'finished'), (signal.SIGINT, 'finished')):
      hook = ('original = command._write_record\n'
              'def cancel_before_record(path, record):\n'
              f'  if record["state"] == {state!r}:\n'
              f'    os.kill(os.getpid(), {int(sig)})\n'
              '  original(path, record)\n'
              'command._write_record = cancel_before_record\n')
      with self.subTest(signal=sig, state=state):
        source = 'import time; time.sleep(60)' if state == 'running' else f'print({FOOTER!r})'
        directory, data, _ = self.case(source, passed=False, before_run=hook)
        record = json.loads((directory / 'command-process.json').read_text())
        self.assertEqual(data['error_type'], 'RuntimeError')
        self.assertIn('cancelled by signal', data['error'])
        self.assertFalse(data['run_returned'])
        self.assertTrue(data['record_rejected'])
        self.assertFalse(record['active'])
        self.assertEqual(record['state'], 'failed')
        self.assertEqual(command._members(record['pgid']), [])
        if state == 'running':
          self.assertLess(record['return_code'], 0)
        else:
          self.assertEqual(record['return_code'], 0)

  def child_source(self, residual=False):
    child = ('import os, signal, time; from pathlib import Path; '
             'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
             'Path("child.pid").write_text(str(os.getpid())); time.sleep(60)')
    return ('import subprocess, sys, time, signal\nfrom pathlib import Path\n'
            f'subprocess.Popen([sys.executable, "-S", "-B", "-c", {child!r}])\n'
            'deadline = time.monotonic() + 3\n'
            'while not Path("child.pid").exists() and time.monotonic() < deadline: time.sleep(0.01)\n'
            + (f'print({FOOTER!r})\n' if residual else
               'signal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(60)\n'))

  def assert_group_gone(self, directory):
    record = json.loads((directory / 'command-process.json').read_text())
    self.assertFalse(record['active'])
    self.assertEqual(command._members(record['pgid']), [])
    child = int((self.root / 'child.pid').read_text())
    self.assertNotIn(child, command._members(record['pgid']))
    return record

  def test_silent_timeout_and_residual_children(self):
    for residual in (False, True):
      with self.subTest(residual=residual):
        (self.root / 'child.pid').unlink(missing_ok=True)
        directory, data, _ = self.case(self.child_source(residual), passed=False, spec=dict(timeout=1))
        record = self.assert_group_gone(directory)
        self.assertFalse(data['run_returned'])
        self.assertFalse(data['check_called'])
        self.assertTrue(data['record_rejected'])
        if residual:
          self.assertEqual(data['error_type'], 'RuntimeError')
          self.assertIn('residual children', data['error'])
          self.assertEqual(record['state'], 'cleanup_failed')
          self.assertEqual(record['return_code'], 0)
        else:
          self.assertEqual(data['error_type'], 'TimeoutExpired')
          self.assertEqual(data['timeout'], 1)
          self.assertEqual(record['state'], 'timeout')
          self.assertEqual(record['return_code'], -signal.SIGKILL)
          self.assertEqual((directory / 'command.log').read_text(), '')
    (self.root / 'child.pid').unlink()
    directory, data, _ = self.case(self.child_source(True) + 'raise SystemExit(7)\n', passed=False)
    record = self.assert_group_gone(directory)
    self.assertEqual(data['error_type'], 'CalledProcessError')
    self.assertEqual(data['returncode'], 7)
    self.assertIn('Cleanup failed: Command left residual children', data['notes'])
    self.assertEqual(record['state'], 'cleanup_failed')
    self.assertTrue(data['record_rejected'])

  def test_recorded_cancellation_and_stale_identity(self):
    sibling = subprocess.Popen([sys.executable, '-S', '-B', '-c', 'import time; time.sleep(60)'],
                               start_new_session=True)
    self.addCleanup(sibling.wait, timeout=3)
    self.addCleanup(sibling.kill)
    invocation, env, result, directory = self.prepare_case(self.child_source(), spec=dict(timeout=15))
    process = subprocess.Popen(invocation, cwd=self.root, env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    path = directory / 'command-process.json'
    try:
      deadline = time.monotonic() + 10
      while not (path.exists() and (self.root / 'child.pid').exists()):
        self.assertIsNone(process.poll())
        self.assertLess(time.monotonic(), deadline)
        time.sleep(0.02)
      record = json.loads(path.read_text())
      stale = self.root / 'stale.json'
      for change in (dict(identity=dict(starttime='expired')), dict(pid=sibling.pid, pgid=sibling.pid)):
        stale.write_text(json.dumps(dict(record, **change)))
        self.assertFalse(command.stop_recorded_process(stale))
        self.assertIsNone(process.poll())
        self.assertIsNone(sibling.poll())
      self.assertTrue(command.stop_recorded_process(path))
      self.assertEqual(process.wait(timeout=10), 1)
      data = json.loads(result.read_text())
      self.assertEqual(data['status'], 'FAILED')
      self.assertEqual(data['error_type'], 'CalledProcessError')
      self.assertFalse(data['run_returned'])
      self.assertTrue(data['record_rejected'])
      self.assert_group_gone(directory)
      self.assertIsNone(sibling.poll())
    finally:
      if path.exists():
        command.stop_recorded_process(path)
      if process.poll() is None:
        process.kill()
      process.wait(timeout=5)

  def test_cancellation_before_launch_refuses_execution(self):
    source = 'from pathlib import Path\nPath("must-not-launch").touch()'
    for hook in ('interrupted.append(signal.SIGTERM)\n', 'os.kill(os.getpid(), signal.SIGINT)\n',
                 'original = command._prepare\n'
                 'def cancel_during_prepare(spec, runtime):\n'
                 '  prepared = original(spec, runtime)\n'
                 '  os.kill(os.getpid(), signal.SIGTERM)\n'
                 '  return prepared\n'
                 'command._prepare = cancel_during_prepare\n'):
      with self.subTest(hook=hook):
        directory, data, _ = self.case(source, passed=False, before_run=hook)
        self.assertEqual(data['error_type'], 'RuntimeError')
        self.assertIn('cancelled by signal', data['error'])
        self.assertFalse(data['run_returned'])
        self.assertFalse((self.root / 'must-not-launch').exists())
        path = directory / 'command-process.json'
        if path.exists():
          record = json.loads(path.read_text())
          self.assertIsNone(record['pid'])
          self.assertFalse(record['active'])

  def test_performance_returns_metrics_without_writing_them(self):
    spec = dict(test_type='performance', cards_per_node=2, threads_per_process=2,
                checks=['exit_code', 'validation_summary', 'performance'],
                parameters=dict(op='all_reduce_perf', dtype='float', redop='sum', ranks=2, iters=100, warmup=5),
                performance=dict(expected_sizes=[8388608], time_kind='cpu_time', rank_reduction='MIN', sample_index=0))
    inputs = dict(pccl_revision='synthetic-pccl', pccl_tests_revision='synthetic-tests',
                  executor_sha256='a' * 64, executor_python_version='synthetic-python',
                  private='not-exported')
    runtime = dict(case_key='synthetic-case', inputs=inputs, command_log_ref='cases/synthetic-case/command.log')
    text = ('# SYNTHETIC fixture, not hardware evidence\n'
            '# nThread 2 nGpus 1 minBytes 8388608 maxBytes 8388608 offset <0>/<0> '
            'step: 1048576(bytes) warmup iters: 5 iters: 100 redop: sum validation: 1 single_test: 0 '
            'test_buffer_kind: both elapsed_type: cpu_time average: MIN register: 0\n'
            '# Rank 0 Group 0 Pid 12345 on synthetic-host.invalid device 0 [0x10] synthetic device\n'
            '# Rank 1 Group 0 Pid 12345 on synthetic-host.invalid device 1 [0x11] synthetic device\n'
            '# out-of-place in-place\n'
            '# size count type redop root cpu_time algbw busbw #wrong cpu_time algbw busbw #wrong\n'
            '# (B) (elements) (us) (GB/s) (GB/s) (us) (GB/s) (GB/s)\n' + ROW + FOOTER)
    directory, data, _ = self.case(f'print({text!r}, end="")', spec=spec, runtime=runtime)
    self.assertEqual(len(data['metrics']), 2)
    self.assertEqual([row['placement'] for row in data['metrics']], ['out-of-place', 'in-place'])
    for row in data['metrics']:
      self.assertEqual(row['case_key'], runtime['case_key'])
      self.assertEqual(row['log_path'], runtime['command_log_ref'])
      self.assertEqual(row['pccl_revision'], inputs['pccl_revision'])
      self.assertEqual(row['schema_version'], 2)
      self.assertEqual(row['executor_sha256'], inputs['executor_sha256'])
      self.assertEqual(row['executor_python_version'], inputs['executor_python_version'])
      self.assertEqual(row['time'], 100)
    self.assertNotIn('not-exported', json.dumps(data['metrics']))
    self.assertEqual(sorted(path.name for path in directory.iterdir()), ['command-process.json', 'command.log'])
    for output in (ROW + FOOTER, text.replace('validation: 1', 'validation: 0'),
                   text.replace('0 OK', '1 FAILED')):
      with self.subTest(output=output):
        directory, data, _ = self.case(f'print({output!r})', spec=spec, runtime=runtime, passed=False)
        self.assertEqual(data['error_type'], 'ValueError')
        self.assertTrue(data['run_returned'])
        self.assertEqual(sorted(path.name for path in directory.iterdir()), ['command-process.json', 'command.log'])
    _, data, _ = self.case('import sys; sys.stdout.buffer.write(b"\\xff")', spec=spec,
                          runtime=runtime, passed=False)
    self.assertEqual(data['error_type'], 'UnicodeDecodeError')
    self.case(f'print({text!r})', spec=spec, runtime=runtime, env={'PCCL_DEBUG': 'INFO'}, passed=False)


if __name__ == '__main__':
  unittest.main(verbosity=2)
