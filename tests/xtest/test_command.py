# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local subprocess tests, including optional pinned public runner checks with no MPI or hardware."""

import json
import os
from pathlib import Path
import shlex
import shutil
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
  def test_evidence_accepts_generated_paths_but_not_opaque_credentials(self):
    from catalog import cases
    from manifest import build_manifest, case_key

    specs = cases()
    manifest = build_manifest(specs, {})
    for shard in manifest['shards']:
      for row in shard['cases']:
        path = f"/workspace/results/{shard['shard_id']}/{row['case_key']}/attempt-1"
        with self.subTest(path=path):
          self.assertEqual(command.safe_text(path), path)
    for group in ('source_register_cluster', '2026_register_sanity', '_source_register_cluster',
                  '-source_register_cluster'):
      key = case_key({**specs[0], 'group': group})
      path = f'/workspace/results/{key}/attempt-1'
      with self.subTest(group=group):
        self.assertEqual(command.safe_text(path), path)
    hook = '/workspace/suite-lifecycle/' + 'a' * 64 + '/setup-1/library.%h.%p.log'
    self.assertEqual(command.safe_text(hook), hook)
    for value in (key, 'a' * 48, '/workspace/' + 'a' * 48, '/workspace/secret/' + key,
                  '/workspace/suite-lifecycle/' + 'a' * 63, '/workspace/other/' + 'a' * 64,
                  '/workspace/private/' + key, '/workspace/ghp_' + 'a' * 48,
                  '/workspace/' + key + '/token'):
      with self.subTest(value=value), self.assertRaises(ValueError):
        command.safe_text(value)

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
    self.env = {'PATH': os.defpath, 'PYTHONDONTWRITEBYTECODE': '1'}

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
    self.assertEqual(sorted(path.name for path in directory.iterdir()), ['command-process.json', 'command-reproduction.txt', 'command.log'])
    reproduction = (directory / 'command-reproduction.txt').read_text()
    self.assertIn(f'cd {shlex.quote(str(self.root))} && \\\n', reproduction)
    path = shlex.quote(f'PATH={self.binary}{os.pathsep}{os.defpath}')
    self.assertIn(f'  {path} \\\n', reproduction)
    argv = [sys.executable, '-S', '-B', str(self.binary / 'child.py')]
    self.assertIn(f'  {shlex.join(argv)}\n', reproduction)
    self.assertNotIn('PYTHONDONTWRITEBYTECODE', reproduction)
    self.assertEqual((directory / 'command-reproduction.txt').stat().st_mode & 0o777, 0o644)
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

  def test_internal_pccl_control_variables_are_not_test_environment(self):
    from unittest.mock import patch
    with patch.dict(os.environ, {'PCCL_REVISION': 'a' * 40,
                                 'PCCL_TESTS_ROOT': '/workspace/pccl-tests'}, clear=False):
      _, env = command._prepare(self.spec, self.runtime)
    self.assertNotIn('PCCL_REVISION', env)
    self.assertNotIn('PCCL_TESTS_ROOT', env)

  def test_runner_timeout_adds_outer_grace_and_unbuffers_python(self):
    self.assertEqual(command.execution_timeout(
      {'timeout': 300}, ['python3', '/workspace/run_test.py', '--timeout', '300']), 330)
    self.assertEqual(command.execution_timeout({'timeout': 300}, ['/bin/echo', 'test']), 300)
    _, env = command._prepare(self.spec, self.runtime)
    self.assertEqual(env['PYTHONUNBUFFERED'], '1')

  def test_generic_logs_redact_environment_on_success_failure_and_timeout(self):
    values = {'CI_ACCESS_TOKEN': 'offline-fixture-value-7319', 'UNLISTED_SETTING': 'unlisted-fixture-value-8241',
              'PCCL_ACCESS_TOKEN': 'whitelisted-prefix-value-9273'}
    output = 'safe output\n' + ROW + FOOTER
    source = ('import os, time\n'
              f'for key in {list(values)!r}: print(os.environ[key], flush=True)\n'
              'print("SECRET diagnostic details", flush=True)\n'
              'print("https://example.invalid/diagnostic", flush=True)\n'
              f'print({output!r}, flush=True)\n')
    for extra, timeout, error in (('', 3, None), ('raise SystemExit(7)', 3, 'CalledProcessError'),
                                ('time.sleep(60)', 0.3, 'TimeoutExpired')):
      with self.subTest(error=error):
        directory, data, _ = self.case(source + extra, env=values, passed=error is None,
                                      spec=dict(timeout=timeout, checks=['exit_code', 'smoke_8mib']))
        if error:
          self.assertEqual(data['error_type'], error)
          self.assertTrue(data['record_rejected'])
        artifacts = '\n'.join(path.read_text() for path in directory.iterdir())
        for value in values.values():
          self.assertNotIn(value, artifacts)
        log = (directory / 'command.log').read_text()
        self.assertNotIn('SECRET diagnostic details', log)
        self.assertNotIn('https://example.invalid/diagnostic', log)
        self.assertIn('safe output\n' + ROW + FOOTER, log)
        self.assertIn('[REDACTED]', log)

  def test_generic_log_is_not_published_while_command_runs(self):
    source = ('import sys\nfrom pathlib import Path\n'
              'assert not (Path(sys.argv[1]) / "command.log").exists()\n'
              f'print({FOOTER!r})')
    self.case(source, spec=dict(argv=self.spec['argv'] + ['{result_dir}']))

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

  def direct_spec(self, *, op='all_reduce_perf', mpi=False, prefix=None, arguments=None, environment=None):
    if arguments is None:
      arguments = ['-b', '5111004', '-e', '1023M', '-f', '2', '-x', '2', '-R', '2']
    if prefix is None:
      prefix = ['--allow-run-as-root', '-np', '2', '-x', 'PCCL_CE_ENABLE_AR=1', '-x', 'LD_LIBRARY_PATH']
    mpi_options = prefix if mpi else []
    environment = environment or {}
    source = (['env'] + [key + '=' + value for key, value in environment.items()] if environment else [])
    source += (['mpirun'] + mpi_options if mpi else []) + [op] + arguments
    return dict(self.spec, argv=(['mpirun', '{mpi_options}'] if mpi else []) + ['{binary_dir}/' + op] + arguments,
                env={'UCX_TLS': 'rc', **environment}, processes_per_node=2 if mpi else 1,
                source_context=dict(plan_id='direct-plan', suite_id='direct-suite', case_id='direct-case',
                               suite_run_id='direct-suite-run', source_command=shlex.join(source),
                               mpi=dict(options=mpi_options, exports=['PATH', 'LD_LIBRARY_PATH'])))

  def direct_case(self, *, spec=None, launcher_extra='', **kwargs):
    spec = self.direct_spec() if spec is None else spec
    launcher = 'mpirun' if spec['argv'][0] == 'mpirun' else Path(spec['argv'][0]).name
    source = (f'#!{sys.executable}\nimport json, os, sys, time\nfrom pathlib import Path\n'
              'Path("actual-direct.json").write_text(json.dumps(dict(argv=sys.argv, env=dict(os.environ))))\n'
              + launcher_extra + '\n' + f'print({FOOTER!r}, flush=True)\n')
    return self.case(source, spec=spec, name=launcher, **kwargs)

  def test_direct_source_argv_and_real_audit_capture(self):
    for op in sorted(command.DIRECT_COLLECTIVES):
      for mpi in (False, True):
        with self.subTest(op=op, mpi=mpi):
          spec = self.direct_spec(op=op, mpi=mpi, environment={'PCCL_CE_ENABLE_AR': '1'})
          directory, _, _ = self.direct_case(spec=spec)
          invocation = command.read_json(directory / 'command-invocation.json')
          observed = command.read_json(directory / 'command-observations.json')
          actual = command.read_json(self.root / 'actual-direct.json')
          entry, = observed['records']
          self.assertIsNone(invocation['runner_index'])
          self.assertEqual(invocation['argv'][:3], [sys.executable, '-S', '-B'])
          self.assertEqual(invocation['source_command'], spec['source_context']['source_command'])
          self.assertEqual(invocation['mpi_export_rule'], command.DIRECT_EXPORT_RULE)
          self.assertEqual(entry['argv'], invocation['original_argv'])
          self.assertEqual(entry['argv'][1:], actual['argv'][1:])
          self.assertEqual(entry['executable'], 'mpirun' if mpi else str(self.binary / op))
          self.assertEqual(entry['effective_environment'], command.safe_environment(actual['env']))
          self.assertEqual(entry['effective_environment'], invocation['effective_environment'])
          self.assertEqual(entry['environment_source'], 'inherited')
          self.assertFalse(entry['private_environment_changed'])
          self.assertNotIn('env', entry['argv'])
          position = entry['argv'].index(str(self.binary / op))
          self.assertEqual(entry['argv'][position + 1:], spec['argv'][3 if mpi else 1:])
          if mpi:
            self.assertEqual(invocation['mpi_options'][:len(spec['source_context']['mpi']['options'])],
                             spec['source_context']['mpi']['options'])
            self.assertIn('UCX_TLS', invocation['mpi_exports'])
          else:
            self.assertEqual(invocation['mpi_options'], [])

  def test_direct_source_custom_mpi_prefix_and_environment_exports(self):
    prefix = ['--allow-run-as-root', '-mca', 'orte_base_help_aggregate', '0', '--mca', 'btl', '^openib',
              '-np', '2', '-npernode', '2', '-x', 'PCCL_CE_ENABLE_AR=1', '-x', 'LD_LIBRARY_PATH']
    environment = {'PCCL_CE_ENABLE_AR': '1', 'PCCL_DEBUG': 'INFO', 'NCCL_DEBUG': 'WARN', 'UMD_PLATFORM_TYPE': '1'}
    spec = self.direct_spec(mpi=True, prefix=prefix, environment=environment)
    directory, _, _ = self.direct_case(spec=spec, runtime=dict(mpi_options=['--prefix', '/opt/openmpi']))
    invocation = command.read_json(directory / 'command-invocation.json')
    # Launcher/network MCA options are infrastructure-owned and may be replaced or omitted.
    expected_prefix = ['--allow-run-as-root', '--mca', 'btl', '^openib', '-np', '2', '-npernode', '2',
                       '-x', 'PCCL_CE_ENABLE_AR=1', '-x', 'LD_LIBRARY_PATH']
    self.assertEqual(invocation['mpi_options'][:len(expected_prefix)], expected_prefix)
    options = command._unique_mpi(command._mpi_entries(invocation['mpi_options']))
    self.assertEqual(options['prefix'], '/opt/openmpi')
    self.assertEqual(options['env:PCCL_CE_ENABLE_AR'], '1')
    self.assertNotIn('mca:orte_base_help_aggregate', options)
    for key in environment:
      self.assertIn(key, invocation['mpi_exports'])
    self.assertEqual(invocation['mpi_exports'].count('LD_LIBRARY_PATH'), 1)

  def test_direct_source_fixed_options_and_layout_defaults(self):
    args = ['-b', '8M', '-e', '1G', '-f', '2', '-R', '2', '-d', 'float', '-g', '2', '-t', '4',
            '-x', '2', '-o', 'sum', '-n', '10', '-w', '5', '-c', '1', '-k', '2', '-a', '2',
            '-N', '1', '-D', '1', '-V', '16', '--init_blocking', '0', '--elapsed_type', '0', '--report_type', '0']
    spec = dict(self.direct_spec(arguments=args), threads_per_process=4, devices_per_thread=2)
    self.direct_case(spec=spec)
    self.direct_case(spec=self.direct_spec(arguments=[]))
    spec = self.direct_spec(mpi=True, prefix=['-n', '1'])
    spec['processes_per_node'] = 1
    self.direct_case(spec=spec)  # An explicit single-process MPI source remains MPI.

  def test_direct_source_rejects_unknown_unsafe_or_changed_contracts(self):
    spec = self.direct_spec()
    changes = [
      ('argv', ['/other/bin/all_reduce_perf'] + spec['argv'][1:]),
      ('argv', ['all_reduce_perf'] + spec['argv'][1:]),
      ('argv', ['{binary_dir}/other/all_reduce_perf'] + spec['argv'][1:]),
      ('argv', ['{python}', '{binary_dir}/run_test.py']),
      ('argv', spec['argv'] + ['-c', '0']),
      ('nodes', 2), ('processes_per_node', 2), ('threads_per_process', 2), ('devices_per_thread', 2),
    ]
    for key, value in changes:
      with self.subTest(key=key, value=value), self.assertRaises(ValueError):
        command._prepare(dict(spec, **{key: value}), self.runtime, inherited=self.env)
    for source in ('./all_reduce_perf -b 32', '/opt/bin/all_reduce_perf -b 32', 'sh -c all_reduce_perf',
                   'all_gather_perf -b 32', 'mpirun all_reduce_perf -b 32',
                   'all_reduce_perf -b 32 ; true', 'all_reduce_perf -b $(id)', 'all_reduce_perf -b `id`',
                   'all_reduce_perf -b 32 > log', 'all_reduce_perf -b *', 'all_reduce_perf -b {result_dir}',
                   'all_reduce_perf -n 10 -n 10', 'all_reduce_perf -n', 'all_reduce_perf --unknown 0',
                   'all_reduce_perf --report_check_filename output', 'all_reduce_perf -n /bin/true',
                   'env all_reduce_perf', 'env PATH=/other all_reduce_perf', 'env -i all_reduce_perf'):
      changed = json.loads(json.dumps(spec))
      changed['source_context']['source_command'] = source
      with self.subTest(source=source), self.assertRaises(ValueError):
        command._prepare(changed, self.runtime, inherited=self.env)
    # Self-consistent argv/source changes must still satisfy the option grammar.
    for arguments in (['--unknown', '0'], ['-n', '/tmp/file'], ['-b', '{result_dir}'], ['-t', '0'],
                      ['-n', '10', '-n', '10'], ['-n'], ['--report_type', 'cpu_time']):
      with self.subTest(arguments=arguments), self.assertRaises(ValueError):
        command._prepare(self.direct_spec(arguments=arguments), self.runtime, inherited=self.env)

  def test_direct_source_mpi_ownership_and_environment_conflicts(self):
    for prefix in (['--allow-run-as-root'], ['-np', '3'], ['-np', '2', '-npernode', '1'],
                   ['-np', '2', '-n', '2'], ['-np', '2', '--prefix', '/opt/openmpi'],
                   ['-np', '2', '--hostfile', 'hosts'], ['-np', '2', '--mca', 'plm_rsh_agent', 'custom-ssh'],
                   ['-np', '2', '--mca'], ['-np', '2', '-x', 'CI_TOKEN=abc'],
                   ['-np', '2', '-x', 'PCCL_DEBUG=INFO', '-x', 'PCCL_DEBUG=INFO'],
                   ['-np', '2', '--unknown', 'value']):
      with self.subTest(prefix=prefix), self.assertRaises(ValueError):
        command._prepare(self.direct_spec(mpi=True, prefix=prefix), self.runtime, inherited=self.env)
    spec = self.direct_spec(mpi=True)
    for options in (['-np', '2'], ['-x', 'PCCL_DEBUG'], ['--mca', 'btl', '^openib'],
                    ['--prefix', '/opt/mpi', '--prefix', '/opt/mpi']):
      with self.subTest(options=options), self.assertRaises(ValueError):
        command._prepare(spec, dict(self.runtime, mpi_options=options), inherited=self.env)
    with self.assertRaisesRegex(ValueError, 'environment'):
      command._prepare(spec, self.runtime, inherited={**self.env, 'PCCL_CE_ENABLE_AR': '0'})
    changed = json.loads(json.dumps(spec))
    changed['source_context']['mpi']['options'] += ['-x', 'PCCL_DEBUG']
    with self.assertRaisesRegex(ValueError, 'prefix'):
      command._prepare(changed, self.runtime, inherited=self.env)
    custom = self.direct_spec(mpi=True, prefix=['-np', '2', '--mca', 'orte_base_help_aggregate', '0'])
    argv, _ = command._prepare(custom, dict(self.runtime, mpi_options=['-mca', 'orte_base_help_aggregate', '0']),
                               inherited=self.env)
    self.assertEqual(argv.count('orte_base_help_aggregate'), 1)
    with self.assertRaisesRegex(ValueError, 'Non-MPI'):
      command._prepare(self.direct_spec(), dict(self.runtime, mpi_options=['--prefix', '/opt/mpi']), inherited=self.env)

  def test_direct_source_environment_prefix_and_literal_diagnostics(self):
    for key in ('DEBUG_FILE', 'PCCL_DEBUG_FILE', 'NCCL_DEBUG_FILE'):
      for mpi in (False, True):
        with self.subTest(key=key, mpi=mpi):
          spec = self.direct_spec(mpi=mpi, environment={key: 'library.%h.%p.log'})
          directory, _, _ = self.direct_case(spec=spec)
          observed = command.read_json(directory / 'command-observations.json')
          self.assertEqual(observed['records'][0]['effective_environment'][key], 'library.%h.%p.log')
          for value in (None, '{result_dir}/library.%h.%p.log', '/tmp/library.%h.%p.log', 'different.log'):
            with self.subTest(value=value), self.assertRaises(ValueError):
              command._prepare(dict(spec, env={key: value}), self.runtime, inherited=self.env)
      for value in ('/tmp/library.log', '../library.log', '{result_dir}/library.%h.%p.log'):
        with self.subTest(key=key, value=value), self.assertRaises(ValueError):
          command._prepare(self.direct_spec(environment={key: value}), self.runtime, inherited=self.env)
    spec = self.direct_spec(environment={'PCCL_DEBUG': 'INFO'})
    for source in (spec['source_context']['source_command'].replace('PCCL_DEBUG=INFO', 'PCCL_DEBUG=INFO PCCL_DEBUG=INFO'),
                   spec['source_context']['source_command'].replace('PCCL_DEBUG=INFO', 'PCCL_DEBUG=WARN')):
      changed = json.loads(json.dumps(spec))
      changed['source_context']['source_command'] = source
      with self.assertRaises(ValueError):
        command._prepare(changed, self.runtime, inherited=self.env)
    with self.assertRaises(ValueError):
      command._prepare(dict(self.direct_spec(), env={'DEBUG_FILE': 'library.%h.%p.log'}),
                       self.runtime, inherited=self.env)

  def test_source_performance_diagnostics_require_explicit_path_bindings(self):
    names = ('PCCL_DEBUG_FILE', 'NCCL_DEBUG_FILE')
    spec = self.direct_spec(environment={name: 'library.%h.%p.log' for name in names})
    spec['test_type'] = 'performance'
    bindings = {name: '{result_dir}/library.%h.%p.log' for name in names}
    spec['env'].update(bindings)
    spec['source_context']['environment'] = {'path_bindings': bindings}
    directory, _, _ = self.direct_case(spec=spec)
    observed = command.read_json(directory / 'command-observations.json')
    invocation = command.read_json(directory / 'command-invocation.json')
    self.assertEqual(invocation['source_command'], spec['source_context']['source_command'])
    for name in names:
      self.assertEqual(observed['records'][0]['effective_environment'][name],
                       str(directory / 'library.%h.%p.log'))
    for mutation in ('missing', 'different', 'functional'):
      changed = json.loads(json.dumps(spec))
      if mutation == 'missing':
        changed['source_context']['environment']['path_bindings'].pop(names[0])
      elif mutation == 'different':
        changed['source_context']['environment']['path_bindings'][names[0]] = '/tmp/library.%h.%p.log'
      else:
        changed['test_type'] = 'functional'
      with self.subTest(mutation=mutation), self.assertRaises(ValueError):
        command._prepare(changed, self.runtime, inherited=self.env)

  def test_direct_source_missing_and_rehashed_tampered_observations(self):
    for mpi in (False, True):
      directory, data, _ = self.direct_case(spec=self.direct_spec(mpi=mpi))
      inputs = command.read_json(directory.parent / 'inputs.json')
      invocation = command.read_json(directory / 'command-invocation.json')
      observed = command.read_json(directory / 'command-observations.json')
      original = observed['records'][0]['argv']
      position = next(i for i, value in enumerate(original) if value == str(self.binary / 'all_reduce_perf'))
      edits = [
        (('records', 0, 'executable'), '/other/launcher'),
        (('records', 0, 'argv', position), '/other/bin/all_reduce_perf'),
        (('records', 0, 'argv', position), 'all_reduce_perf'),
        (('records', 0, 'argv', -1), '1'),
        (('records', 0, 'effective_environment', 'UCX_TLS'), 'tcp'),
        (('records', 0, 'environment_source'), 'explicit'),
        (('records', 0, 'private_environment_changed'), True),
        (('records',), []), (('records',), observed['records'] * 2), (('state',), 'running'),
      ]
      if mpi:
        edits += [(('records', 0, 'argv', 0), '/other/mpirun'), (('records', 0, 'argv', 3), '3')]
      for keys, value in edits:
        obs = json.loads(json.dumps(observed))
        item = obs
        for key in keys[:-1]:
          item = item[key]
        item[keys[-1]] = value
        if keys[:3] == ('records', 0, 'argv'):
          obs['records'][0]['executable'] = obs['records'][0]['argv'][0]
        (directory / 'command-observations.json').write_text(json.dumps(obs))
        record = dict(data['record'], observations_sha256=command.digest(obs))
        with self.subTest(mpi=mpi, keys=keys), self.assertRaises(ValueError):
          command.check_command(inputs['spec'], inputs['runtime'], record)
      (directory / 'command-observations.json').unlink()
      with self.assertRaises(ValueError):
        command.check_command(inputs['spec'], inputs['runtime'], data['record'])
      (directory / 'command-observations.json').write_text(json.dumps(observed))
      for field, value in (('original_argv', ['/other/bin/all_reduce_perf']),
                           ('argv', ['/other/python3'] + invocation['argv'][1:]), ('runner_index', 0)):
        inv = dict(invocation, **{field: value})
        obs = dict(observed, invocation_sha256=command.digest(inv))
        record = dict(data['record'], invocation_sha256=command.digest(inv), observations_sha256=command.digest(obs))
        (directory / 'command-invocation.json').write_text(json.dumps(inv))
        (directory / 'command-observations.json').write_text(json.dumps(obs))
        with self.subTest(field=field), self.assertRaises(ValueError):
          command.check_command(inputs['spec'], dict(inputs['runtime'], python=sys.executable), record)

  def test_direct_source_requires_a_real_observation_before_success(self):
    # Substitute a no-audit child at the launch boundary, not fabricated evidence.
    program = f'print({FOOTER!r})'
    hook = ('original = command._invocation\n'
            'def without_capture(spec, runtime, argv, env):\n'
            '  invocation = original(spec, runtime, argv, env)\n'
            f'  invocation["argv"] = [sys.executable, "-S", "-B", "-c", {program!r}]\n'
            '  return invocation\n'
            'command._invocation = without_capture\n')
    directory, data, _ = self.direct_case(before_run=hook, passed=False)
    self.assertFalse(data['run_returned'])
    self.assertTrue(data['record_rejected'])
    self.assertFalse((directory / 'command-observations.json').exists())
    self.assertIn('REDACTED', (directory / 'command.log').read_text())
    record = command.read_json(directory / 'command-process.json')
    self.assertFalse(record['active'])
    self.assertEqual(record['return_code'], 0)
    self.assertEqual(command._members(record['pgid']), [])

  def test_direct_source_failure_cleanup_and_log_redaction(self):
    value = 'offline-direct-fixture-7319'
    for extra, timeout, error in (('', 3, None), ('raise SystemExit(7)', 3, 'CalledProcessError'),
                                ('time.sleep(60)', 0.3, 'TimeoutExpired')):
      with self.subTest(error=error):
        directory, data, _ = self.direct_case(
          spec=dict(self.direct_spec(), timeout=timeout), env={'CI_ACCESS_TOKEN': value}, passed=error is None,
          launcher_extra='print(os.environ["CI_ACCESS_TOKEN"], flush=True)\n' + extra)
        self.assertNotIn(value, '\n'.join(path.read_text() for path in directory.iterdir()))
        record = command.read_json(directory / 'command-process.json')
        self.assertFalse(record['active'])
        self.assertEqual(command._members(record['pgid']), [])
        if error:
          self.assertEqual(data['error_type'], error)
          self.assertTrue(data['record_rejected'])
    directory, data, _ = self.direct_case(launcher_extra=self.child_source(True) + 'raise SystemExit(7)', passed=False)
    record = self.assert_group_gone(directory)
    self.assertEqual(data['error_type'], 'CalledProcessError')
    self.assertEqual(record['state'], 'cleanup_failed')
    self.assertTrue(any('residual children' in note for note in data['notes']))

  def test_direct_source_relocated_trusted_runtime(self):
    runner_root = self.root / 'direct-root'
    runner_root.mkdir()
    directory, data, _ = self.direct_case(
      spec=self.direct_spec(mpi=True, environment={'PCCL_DEBUG_FILE': 'library.%h.%p.log'}),
      runtime=dict(test_root=str(runner_root)))
    inputs = command.read_json(directory.parent / 'inputs.json')
    invocation = command.read_json(directory / 'command-invocation.json')
    observed = command.read_json(directory / 'command-observations.json')
    target = self.root / 'collected-direct'
    shutil.copytree(directory, target)
    shutil.rmtree(directory)
    shutil.rmtree(runner_root)
    runtime = dict(inputs['runtime'], result_dir=str(target), execution_result_dir=str(directory),
                   runner_build_directory=invocation['runner_build_directory'], python=sys.executable,
                   inherited_environment=invocation['inherited_environment'])
    self.assertEqual(command.check_command(inputs['spec'], runtime, data['record']), [])
    snapshot = {path.name: path.read_bytes() for path in target.iterdir()}
    for key, value in (('test_root', '/other/root'), ('binary_dir', '/other/bin'),
                       ('execution_result_dir', '/other/results'), ('runner_build_directory', '/other/build'),
                       ('python', '/other/python3'), ('inherited_environment', {'PATH': '/other/bin'}),
                       ('mpi_options', ['--prefix', '/other/mpi'])):
      with self.subTest(binding=key), self.assertRaises(ValueError):
        command.check_command(inputs['spec'], dict(runtime, **{key: value}), data['record'])
    self.assertEqual({path.name: path.read_bytes() for path in target.iterdir()}, snapshot)
    # Coordinated, rehashed relocation cannot replace independently trusted bindings.
    inv = json.loads(json.dumps(invocation))
    inv['bindings']['result_dir'] = '/other/results'
    inv['argv'][4] = '/other/results/command-invocation.json'
    obs = dict(observed, invocation_sha256=command.digest(inv))
    record = dict(data['record'], invocation_sha256=command.digest(inv), observations_sha256=command.digest(obs))
    (target / 'command-invocation.json').write_text(json.dumps(inv))
    (target / 'command-observations.json').write_text(json.dumps(obs))
    with self.assertRaisesRegex(ValueError, 'shard runtime'):
      command.check_command(inputs['spec'], runtime, record)

  def test_direct_source_performance_argv_stays_unwrapped(self):
    args = ['-b', '8388608', '-e', '8388608', '-n', '100', '-w', '5', '-d', 'float', '-o', 'sum',
            '-t', '2', '-g', '1', '-c', '1', '-k', '2', '-a', '2', '--elapsed_type', '0',
            '--report_type', '0', '-N', '1']
    spec = dict(self.direct_spec(arguments=args), test_type='performance', threads_per_process=2)
    argv, env = command._prepare(spec, self.runtime, inherited=self.env)
    self.assertEqual(spec['argv'], ['{binary_dir}/all_reduce_perf'] + args)
    self.assertEqual(argv, [str(self.binary / 'all_reduce_perf')] + args)
    invocation = command._invocation(spec, self.runtime, argv, env)
    self.assertEqual(invocation['original_argv'], argv)
    self.assertEqual(invocation['argv'][6:], argv)
    self.direct_case(spec=spec)

  def source_case(self, *, mpi=False, repeats=1, runner_extra='', launcher_extra='', after_launch='',
                  launch='subprocess.run(shlex.split(shlex.join(argv)), check=True)', spec_changes=None, **kwargs):
    """Pinned runner rules with real local children; never hardware/MPI evidence."""
    source_options = (['-mca', 'btl_tcp_if_include', 'bond1', '-x', 'NCCL_NET_GDR_LEVEL=4',
                       '--hostfile', '{hostfile}'] if mpi else [])
    source_args = ['--op', 'all_reduce_perf', '--algo', 'Auto', '--proto', 'Auto', '--data-type', 'float',
                   '--data-size', '32MB', '--iters', '10', '--ranks', '2' if mpi else '1',
                   '--nnodes', '2' if mpi else '1', '--nprocs', '1', '--threads', '1', '--register', '1']
    original_options = [x if x != '{hostfile}' else './hosts.cfg' for x in source_options]
    original_args = source_args + (['--mpi-options=' + shlex.join(original_options)] if mpi else [])
    source_command = shlex.join(['python3', 'run_test.py'] + original_args)
    spec = dict(argv=['{python}', '-S', '-B', '{binary_dir}/run_test.py'] + source_args +
                     ['--mpi-options={mpi_options}', '--process-group', 'normal', '--launch-mode', 'normal'],
                env={'UMD_PLATFORM_TYPE': '1', 'PCCL_ALGO': 'Tree', 'PCCL_PROTO': 'LL',
                     'PCCL_BUFFSIZE': '128', 'UCX_TLS': 'rc'}, nodes=2 if mpi else 1, checks=['exit_code'],
                source_context=dict(plan_id='plan', suite_id='suite', case_id='case', suite_run_id='suite-run',
                               source_command=source_command,
                               mpi=dict(options=source_options,
                                        exports=['PATH', 'LD_LIBRARY_PATH', 'UMD_PLATFORM_TYPE'])))
    spec.update(spec_changes or {})
    runtime = kwargs.pop('runtime', {})
    if mpi:
      (self.root / 'hosts.cfg').write_text('localhost slots=2\n')
      runtime.setdefault('mpi_options', ['-mca', 'plm_rsh_agent', 'cluster-ssh', '-mca',
                                         'orte_base_help_aggregate', '0', '--hostfile', 'hosts.cfg'])
    launcher = self.binary / ('mpirun' if mpi else 'all_reduce_perf')
    launcher.write_text(f'#!{sys.executable}\nimport json, os, sys, time\nfrom pathlib import Path\n'
                       'with Path("actual.jsonl").open("a") as stream:\n'
                       '  stream.write(json.dumps(dict(argv=sys.argv, env=dict(os.environ))) + "\\n")\n'
                       + launcher_extra + '\n')
    launcher.chmod(0o700)
    runner = (
      'import json, os, shlex, subprocess, sys\n'
      'print("mpirun this line is deliberately NOT the actual invocation")\n'
      'os.environ["PATH"] += os.pathsep + os.path.realpath("build")\n'
      'os.environ["PCCL_SOCKET_IFNAME"] = "^flannel,docker,lo"\n'
      'for key in ("PCCL_ALGO", "PCCL_PROTO", "PCCL_BUFFSIZE"):\n'
      '  os.environ.pop(key, None)\n'
      + runner_extra + '\n'
      'argv = ["all_reduce_perf", "-t", "1", "-g", "1", "-d", "float", "-b", "33554432",\n'
      '        "-e", "33554432", "-f", "1", "-n", "10", "-w", "0", "--core_dump", "0",\n'
      '        "--elapsed_type", "0", "-o", "sum", "-a", "2", "-R", "1", "-x", "0", "-p", "0"]\n')
    if mpi:
      runner += ('mpi = ["mpirun", "--allow-run-as-root", "-np", "2", "-npernode", "1"]\n'
                 'for key, value in os.environ.items():\n'
                 '  if key.startswith(("PCCL_", "NCCL_")): mpi += ["-x", key + "=" + value]\n'
                 'mpi += shlex.split(next(x.split("=", 1)[1] for x in sys.argv if x.startswith("--mpi-options=")))\n'
                 'argv = mpi + argv\n')
    runner += f'for _ in range({repeats}):\n  {launch}\n' + after_launch + '\n'
    env = dict(PATH='/usr/bin:/bin:/workspace/public/bin', LD_LIBRARY_PATH='/opt/public/lib')
    env.update(kwargs.pop('env', {}))
    return self.case(runner, name='run_test.py', spec=spec, runtime=runtime, env=env, **kwargs)

  def test_runner_decimal_size_semantics_and_source_env_prefix(self):
    for size, expected in (('100.5MB', '105381888'), ('1.2GB', '1288490188'),
                           ('9007199254740993B', '9007199254740992'), ('0.5KB', '512')):
      with self.subTest(size=size):
        self.assertEqual(command._runner_size(size), expected)
    for size in ('NaN', 'infMB', '-1MB', '0B', '1e3MB', '1.MB', '.5MB', '0.00001B', '9' * 400 + 'GB'):
      with self.subTest(size=size), self.assertRaises(ValueError):
        command._runner_size(size)
    hook = ('spec = json.loads(json.dumps(spec))\n'
            'spec["source_context"]["source_command"] = "env UCX_TLS=rc " + spec["source_context"]["source_command"]\n')
    self.source_case(before_run=hook)
    self.source_case(before_run=hook + 'spec["env"]["UCX_TLS"] = "tcp"\n', passed=False)

  def test_source_invocation_and_observed_environment(self):
    for mpi in (False, True):
      with self.subTest(mpi=mpi):
        (self.root / 'actual.jsonl').unlink(missing_ok=True)
        directory, data, _ = self.source_case(mpi=mpi, repeats=2,
                                             env={'CI_ACCESS_TOKEN': 'credential-value-not-for-artifacts'})
        invocation = json.loads((directory / 'command-invocation.json').read_text())
        observed = json.loads((directory / 'command-observations.json').read_text())
        actual = [json.loads(line) for line in (self.root / 'actual.jsonl').read_text().splitlines()]
        self.assertEqual(invocation['schema_version'], 1)
        self.assertEqual(invocation['effective_environment']['PCCL_ALGO'], 'Tree')
        self.assertEqual(invocation['effective_environment']['UMD_PLATFORM_TYPE'], '1')
        self.assertEqual(invocation['mpi_export_rule'], command.EXPORT_RULE)
        self.assertIn('UCX_TLS', invocation['mpi_exports'])
        self.assertIn('UMD_PLATFORM_TYPE', invocation['mpi_exports'])
        self.assertEqual(observed['source'], 'observed-subprocess')
        self.assertEqual(observed['state'], 'complete')
        self.assertEqual(observed['original_argv'], invocation['original_argv'])
        self.assertEqual(len(observed['records']), 2)
        for entry, child in zip(observed['records'], actual):
          self.assertEqual(entry['source'], 'observed-subprocess')
          self.assertEqual(entry['argv'][1:], child['argv'][1:])
          self.assertEqual(Path(child['argv'][0]).name, entry['argv'][0])
          self.assertEqual(entry['effective_environment'], command.safe_environment(child['env']))
          self.assertNotIn('PCCL_ALGO', entry['effective_environment'])
          self.assertNotIn('PCCL_PROTO', entry['effective_environment'])
          self.assertNotIn('PCCL_BUFFSIZE', entry['effective_environment'])
          self.assertEqual(entry['effective_environment']['PCCL_SOCKET_IFNAME'], '^flannel,docker,lo')
          self.assertTrue(entry['effective_environment']['PATH'].endswith(str(self.root / 'build')))
        if mpi:
          self.assertEqual(invocation['mpi_options'].count('--hostfile'), 1)
          self.assertIn('NCCL_NET_GDR_LEVEL=4', invocation['mpi_options'])
          self.assertNotIn('bond1', invocation['mpi_options'])
          self.assertIn('cluster-ssh', invocation['mpi_options'])
        else:
          self.assertEqual(observed['records'][0]['argv'][0], 'all_reduce_perf')
        artifact_text = '\n'.join(path.read_text() for path in directory.iterdir())
        self.assertNotIn('CI_ACCESS_TOKEN', artifact_text)
        self.assertNotIn('credential-value-not-for-artifacts', artifact_text)
        self.assertTrue(data['check_called'])

  def test_source_mpi_conflicts_fail_before_launch(self):
    options = [
      ['--hostfile', 'hosts.cfg', '--hostfile', 'hosts.cfg'],
      ['--hostfile', 'hosts.cfg', '--hostfile=hosts.cfg'],
      ['--hostfile', 'hosts.cfg', '-x', 'NCCL_NET_GDR_LEVEL=3'],
      ['--hostfile', 'hosts.cfg', '-mca'], [],
    ]
    for value in options:
      with self.subTest(value=value):
        directory, data, _ = self.source_case(mpi=True, runtime=dict(mpi_options=value), passed=False)
        self.assertEqual(data['error_type'], 'ValueError')
        self.assertFalse((directory / 'command-invocation.json').exists())
    self.source_case(mpi=True, env={'NCCL_NET_GDR_LEVEL': '3'}, passed=False)
    self.source_case(mpi=True, env={'NCCL_NET_GDR_LEVEL': '4'})
    directory, _, _ = self.source_case(mpi=True, runtime=dict(
      mpi_options=['--hostfile', 'hosts.cfg', '-mca', 'btl_tcp_if_include', 'bond0']))
    invocation = json.loads((directory / 'command-invocation.json').read_text())
    self.assertIn('bond0', invocation['mpi_options'])
    self.assertNotIn('bond1', invocation['mpi_options'])
    (self.root / 'hosts with spaces').write_text('localhost slots=2\n')
    directory, _, _ = self.source_case(mpi=True, runtime=dict(mpi_options=['--hostfile=hosts with spaces']))
    invocation = json.loads((directory / 'command-invocation.json').read_text())
    self.assertEqual(invocation['mpi_options'].count('--hostfile'), 1)
    self.assertIn('hosts with spaces', invocation['mpi_options'])

  def test_source_mpi_prefix_is_infrastructure_only(self):
    prefix = str(self.root / 'openmpi installation')
    directory, _, _ = self.source_case(mpi=True, runtime=dict(
      mpi_options=['--prefix', prefix, '-mca', 'orte_base_help_aggregate', '0', '--hostfile', 'hosts.cfg']))
    invocation = command.read_json(directory / 'command-invocation.json')
    observed = command.read_json(directory / 'command-observations.json')
    self.assertEqual(command._unique_mpi(command._mpi_entries(invocation['mpi_options']))['prefix'], prefix)
    self.assertEqual(observed['records'][0]['argv'].count('--prefix'), 1)
    self.assertIn(prefix, observed['records'][0]['argv'])
    for options in (['--prefix'], ['--prefix', ''], ['--prefix', 'relative/openmpi'],
                    ['--prefix', '--allow-run-as-root'], ['--prefix', '/opt/{unknown}'],
                    ['--prefix', prefix, '--prefix', prefix]):
      with self.subTest(options=options):
        directory, data, _ = self.source_case(mpi=True, passed=False,
                                              runtime=dict(mpi_options=['--hostfile', 'hosts.cfg'] + options))
        self.assertEqual(data['error_type'], 'ValueError')
        self.assertFalse((directory / 'command-invocation.json').exists())
    # Even a self-consistent source command/contract cannot own installation paths.
    hook = ('import shlex\nspec = json.loads(json.dumps(spec))\n'
            'spec["source_context"]["mpi"]["options"] += ["--prefix", "/opt/openmpi"]\n'
            'argv = shlex.split(spec["source_context"]["source_command"])\n'
            'argv[-1] += " --prefix /opt/openmpi"\n'
            'spec["source_context"]["source_command"] = shlex.join(argv)\n')
    directory, data, _ = self.source_case(mpi=True, passed=False, before_run=hook)
    self.assertIn('ownership', data['error'])
    self.assertFalse((directory / 'command-invocation.json').exists())

  def test_source_unknown_sensitive_and_inconsistent_spec(self):
    changes = [
      'spec["source_context"]["source_command"] = "sh -c whoami"',
      'spec["source_context"]["source_command"] += " --hw-perf"',
      'spec["source_context"]["source_command"] += " ; printenv"',
      'spec["source_context"]["source_command"] = spec["source_context"]["source_command"].replace("32MB", "8MB")',
      'spec["source_context"]["mpi"]["exports"].append("API_TOKEN")',
      'spec["env"]["PCCL_DEBUG"] = "https://private.invalid/bearer"',
      'spec["env"]["PCCL_ACCESS_TOKEN"] = "not-public"',
      'spec["env"]["UMD_PLATFORM_TYPE"] = "ghp_abcdefghijklmnop"',
      'spec["env"]["PCCL_DEBUG_FILE"] = "/workspace/private/artifact"',
      'spec["devices_per_thread"] = 2',
      'spec["source_context"]["source_command"] = "python3 run_test.py"\n'
      'spec["argv"] = ["{python}", "{binary_dir}/run_test.py", "--mpi-options={mpi_options}", '
      '"--process-group", "normal"]',
    ]
    for hook in changes:
      with self.subTest(hook=hook):
        directory, data, _ = self.source_case(before_run='spec = json.loads(json.dumps(spec))\n' + hook + '\n',
                                             passed=False)
        self.assertFalse((directory / 'command.log').exists())
        self.assertEqual(data['error_type'], 'ValueError')

  def test_source_tampered_or_missing_evidence(self):
    for name in ('command-invocation.json', 'command-observations.json'):
      for action in ('unlink()', 'write_text("{}")'):
        with self.subTest(name=name, action=action):
          _, data, _ = self.source_case(passed=False,
                                        before_check=f'(Path(runtime["result_dir"]) / {name!r}).{action}\n')
          self.assertEqual(data['error_type'], 'ValueError')
          self.assertTrue(data['run_returned'])
    # Update the hashes too: semantic checking must not trust self-consistent hashes.
    edits = [
      'inv["source_command"] += " --dbg"',
      'inv["effective_environment"]["UMD_PLATFORM_TYPE"] = "2"',
      'inv["mpi_options"] += ["-x", "UCX_NET_DEVICES=eth0"]',
      'obs["records"][0]["argv"][2] = "99"',
      'obs["records"][0]["effective_environment"]["UMD_PLATFORM_TYPE"] = "2"',
      'obs["records"][0]["effective_environment"]["PCCL_ALGO"] = "Tree"',
      'obs["records"] = []',
      'obs["initial_environment"]["UCX_TLS"] = "tcp"',
      'obs["pid"] += 1',
    ]
    for edit in edits:
      hook = ('p = Path(runtime["result_dir"])\n'
              'inv = json.loads((p / "command-invocation.json").read_text())\n'
              'obs = json.loads((p / "command-observations.json").read_text())\n'
              + edit + '\n'
              'record["invocation_sha256"] = command.digest(inv)\n'
              'obs["invocation_sha256"] = command.digest(inv)\n'
              'record["observations_sha256"] = command.digest(obs)\n'
              '(p / "command-invocation.json").write_text(json.dumps(inv))\n'
              '(p / "command-observations.json").write_text(json.dumps(obs))\n')
      with self.subTest(edit=edit):
        _, data, _ = self.source_case(before_check=hook, passed=False)
        self.assertEqual(data['error_type'], 'ValueError')
        self.assertTrue(data['run_returned'])

  def test_source_rejects_same_basename_executable_tokens(self):
    for mpi in (False, True):
      with self.subTest(mpi=mpi):
        directory, data, _ = self.source_case(mpi=mpi)
        inputs = command.read_json(directory.parent / 'inputs.json')
        observed = command.read_json(directory / 'command-observations.json')
        benchmark = observed['records'][0]['argv'].index('all_reduce_perf')
        for position in sorted({0, benchmark}):
          for prefix in ('/different/bin/', './'):
            obs = json.loads(json.dumps(observed))
            entry = obs['records'][0]
            entry['argv'][position] = prefix + entry['argv'][position]
            entry['executable'] = entry['argv'][0]
            record = dict(data['record'], observations_sha256=command.digest(obs))
            (directory / 'command-observations.json').write_text(json.dumps(obs))
            with self.subTest(position=position, prefix=prefix), self.assertRaises(ValueError):
              command.check_command(inputs['spec'], inputs['runtime'], record)
        _, data, _ = self.source_case(mpi=mpi, passed=False, launch=(
          'argv[0] = os.path.join(os.path.dirname(__file__), argv[0]); subprocess.run(argv, check=True)'))
        self.assertEqual(data['error_type'], 'ValueError')
        self.assertFalse(data['run_returned'])

  def test_source_capture_failure_timeout_and_publication(self):
    for extra, timeout, error in (('raise SystemExit(7)', 3, 'CalledProcessError'),
                                   ('time.sleep(60)', 0.3, 'TimeoutExpired')):
      with self.subTest(error=error):
        directory, data, _ = self.source_case(launcher_extra=extra, spec_changes=dict(timeout=timeout), passed=False)
        self.assertEqual(data['error_type'], error)
        self.assertFalse(data['run_returned'])
        self.assertTrue((directory / 'command-invocation.json').is_file())
        observed = json.loads((directory / 'command-observations.json').read_text())
        self.assertEqual(len(observed['records']), 1)
        record = json.loads((directory / 'command-process.json').read_text())
        self.assertFalse(record['active'])
        self.assertEqual(command._members(record['pgid']), [])
    hooks = [
      'original = command._write_record\n'
      'def fail(path, record):\n'
      '  if path.name == "command-invocation.json": raise OSError("publish failure")\n'
      '  original(path, record)\ncommand._write_record = fail\n',
      'original = command._write_record\n'
      'def fail(path, record):\n'
      '  original(path, record)\n'
      '  if path.name == "command-invocation.json": path.write_text("{}")\n'
      'command._write_record = fail\n',
    ]
    for hook in hooks:
      (self.root / 'actual.jsonl').unlink(missing_ok=True)
      self.source_case(before_run=hook, passed=False)
      self.assertFalse((self.root / 'actual.jsonl').exists())
    directory, _, _ = self.source_case(repeats=0, passed=False)
    self.assertEqual(json.loads((directory / 'command-observations.json').read_text())['records'], [])

  def test_source_explicit_env_residuals_and_capture_write_failure(self):
    directory, _, _ = self.source_case(launch='subprocess.run(argv, env=dict(os.environ), check=True)')
    observed = json.loads((directory / 'command-observations.json').read_text())
    self.assertEqual(observed['records'][0]['environment_source'], 'explicit')
    (self.root / 'actual.jsonl').unlink()
    directory, data, _ = self.source_case(
      launcher_extra='time.sleep(60)', launch='subprocess.Popen(argv)', passed=False,
      after_launch='import time\nfrom pathlib import Path\n'
                   'while not Path("actual.jsonl").exists(): time.sleep(0.01)')
    record = json.loads((directory / 'command-process.json').read_text())
    self.assertEqual(record['state'], 'cleanup_failed')
    self.assertIn('residual children', data['error'])
    self.assertEqual(command._members(record['pgid']), [])
    directory, data, _ = self.source_case(passed=False, runner_extra=(
      'from pathlib import Path\n'
      'path = Path(next(x for x in sys.orig_argv if x.endswith("command-invocation.json")))\n'
      'path = path.with_name("command-observations.json")\npath.unlink()\npath.mkdir()'))
    self.assertFalse(data['run_returned'])
    self.assertIn('REDACTED', (directory / 'command.log').read_text())
    record = json.loads((directory / 'command-process.json').read_text())
    self.assertEqual(command._members(record['pgid']), [])

  def test_source_explicit_environment_removal(self):
    directory, _, _ = self.source_case(mpi=True, env={'UMD_PLATFORM_TYPE': '1', 'UCX_TLS': 'tcp'},
                                       spec_changes=dict(env={'UMD_PLATFORM_TYPE': None, 'UCX_TLS': None}))
    for entry in json.loads((directory / 'command-observations.json').read_text())['records']:
      self.assertNotIn('UMD_PLATFORM_TYPE', entry['effective_environment'])
      self.assertNotIn('UCX_TLS', entry['effective_environment'])

  def test_falsey_original_error_survives_cleanup_failure(self):
    hook = ('import time\nclass FalseError(RuntimeError):\n  def __bool__(self): return False\n'
            'original = command._write_record\n'
            'def fail_record(path, record):\n'
            '  original(path, record)\n'
            '  if record["state"] == "running":\n'
            '    deadline = time.monotonic() + 3\n'
            '    while not Path("ready").exists() and time.monotonic() < deadline: time.sleep(0.01)\n'
            '    raise FalseError("original failure")\n'
            'def fail_members(pid): raise OSError("cleanup failure")\n'
            'command._write_record = fail_record\ncommand._members = fail_members\n')
    source = 'import time\nfrom pathlib import Path\nPath("ready").touch()\ntime.sleep(60)'
    directory, data, _ = self.case(source, before_run=hook, passed=False)
    self.assertEqual(data['error_type'], 'FalseError')
    self.assertEqual(data['error'], 'original failure')
    # The fallback kill can itself fail after the child exits on macOS; either
    # cleanup exception must be secondary to the falsey original exception.
    self.assertTrue(any(note.startswith('Cleanup failed:') for note in data['notes']))
    record = json.loads((directory / 'command-process.json').read_text())
    self.assertEqual(record['state'], 'cleanup_failed')
    self.assertEqual(command._members(record['pgid']), [])

  def test_source_summary_does_not_reinterpret_checker_environment(self):
    hook = ('os.environ["UMD_PLATFORM_TYPE"] = "999"\n'
            'os.environ["PATH"] = "/different/checker/path"\n'
            'runtime = {"result_dir": runtime["result_dir"]}\n')
    self.source_case(before_check=hook)
    self.case('', passed=False, spec=dict(test_type='performance'), env={'UMD_PLATFORM_TYPE': '1'})

  def test_source_summary_relocates_artifacts_not_execution_bindings(self):
    directory, data, _ = self.source_case(spec_changes=dict(env={'PCCL_DEBUG_FILE': '{result_dir}/library.log'}))
    inputs = command.read_json(directory.parent / 'inputs.json')
    target = self.root / 'collected-artifacts'
    shutil.copytree(directory, target)
    shutil.rmtree(directory)
    snapshot = {path.name: path.read_bytes() for path in target.iterdir()}
    runtime = dict(inputs['runtime'], result_dir=str(target))
    self.assertEqual(command.check_command(inputs['spec'], runtime, data['record']), [])
    self.assertEqual(command.check_command(inputs['spec'], {'result_dir': str(target)}, data['record']), [])
    # Execution still compares every binding, including result_dir. Summary may
    # omit execution paths, but cannot override any supplied non-artifact binding.
    with self.assertRaisesRegex(ValueError, 'runtime binding mismatch'):
      command._check_invocation(inputs['spec'], runtime, data['record'])
    for key, value in (('test_root', '/other/runner'), ('binary_dir', '/other/bin'),
                       ('mpi_options', ['--prefix', '/other/openmpi'])):
      with self.subTest(binding=key), self.assertRaisesRegex(ValueError, 'runtime binding mismatch'):
        command.check_command(inputs['spec'], dict(runtime, **{key: value}), data['record'])
    invocation = command.read_json(target / 'command-invocation.json')
    self.assertEqual(invocation['bindings']['result_dir'], str(directory))
    self.assertEqual(invocation['effective_environment']['PCCL_DEBUG_FILE'], str(directory / 'library.log'))
    self.assertEqual({path.name: path.read_bytes() for path in target.iterdir()}, snapshot)

  def test_relocated_source_evidence_rejects_rehashed_tampering(self):
    directory, data, _ = self.source_case()
    inputs = command.read_json(directory.parent / 'inputs.json')
    invocation = command.read_json(directory / 'command-invocation.json')
    observed = command.read_json(directory / 'command-observations.json')
    target = self.root / 'collected-artifacts'
    shutil.copytree(directory, target)
    shutil.rmtree(directory)
    runtime = {'result_dir': str(target)}
    self.assertEqual(command.check_command(inputs['spec'], runtime, data['record']), [])
    edits = [
      ('inv', ('bindings', 'result_dir'), str(target)),
      ('inv', ('bindings', 'test_root'), '/other/runner'),
      ('inv', ('bindings', 'binary_dir'), '/other/bin'),
      ('inv', ('bindings', 'mpi_options'), ['--prefix', '/other/openmpi']),
      ('inv', ('source_command',), invocation['source_command'].replace('32MB', '8MB')),
      ('inv', ('spec_sha256',), '0' * 64),
      ('inv', ('original_argv', 5), 'reduce_perf'),
      ('inv', ('argv', invocation['runner_index'] + 1), str(target / 'command-invocation.json')),
      ('obs', ('records', 0, 'argv', 2), '99'),
      ('obs', ('records', 0, 'effective_environment', 'UMD_PLATFORM_TYPE'), '2'),
    ]
    edits += [('inv', ('source_ids', key), 'other') for key in invocation['source_ids']]
    for evidence, path, value in edits:
      inv, obs = json.loads(json.dumps(invocation)), json.loads(json.dumps(observed))
      item = inv if evidence == 'inv' else obs
      for key in path[:-1]:
        item = item[key]
      item[path[-1]] = value
      record = dict(data['record'], invocation_sha256=command.digest(inv))
      obs['invocation_sha256'] = record['invocation_sha256']
      record['observations_sha256'] = command.digest(obs)
      (target / 'command-invocation.json').write_text(json.dumps(inv))
      (target / 'command-observations.json').write_text(json.dumps(obs))
      with self.subTest(evidence=evidence, path=path), self.assertRaises(ValueError):
        command.check_command(inputs['spec'], runtime, record)

  def test_source_runtime_anchors_are_enforced_during_execution(self):
    for field in ('runner_build_directory', 'execution_result_dir'):
      with self.subTest(field=field):
        _, data, _ = self.source_case(runtime={field: '/different/path'}, passed=False)
        self.assertEqual(data['error_type'], 'ValueError')
        self.assertFalse(data['run_returned'])
    self.source_case(before_run=(
      'runtime = dict(runtime, execution_result_dir=runtime["result_dir"],\n'
      '               runner_build_directory=str((Path(runtime["test_root"]) / "build").resolve()))\n'))

  def test_relocated_evidence_uses_independent_runtime_anchors(self):
    runner_root = self.root / 'runner-root'
    runner_root.mkdir()
    build = self.root / 'resolved-build'
    (runner_root / 'build').symlink_to(build, target_is_directory=True)
    directory, data, _ = self.source_case(
      runtime=dict(test_root=str(runner_root), runner_build_directory=str(build)),
      spec_changes=dict(env={'PCCL_DEBUG_FILE': '{result_dir}/library.log'}))
    inputs = command.read_json(directory.parent / 'inputs.json')
    invocation = command.read_json(directory / 'command-invocation.json')
    observed = command.read_json(directory / 'command-observations.json')
    target = self.root / 'collected-artifacts'
    shutil.copytree(directory, target)
    shutil.rmtree(directory)
    shutil.rmtree(runner_root)
    runtime = dict(inputs['runtime'], result_dir=str(target), execution_result_dir=str(directory))
    self.assertFalse(runner_root.exists())
    self.assertFalse(directory.exists())
    self.assertFalse(build.exists())
    self.assertEqual(command.check_command(inputs['spec'], runtime, data['record']), [])
    for field, message in (('runner_build_directory', 'Runner build directory'),
                           ('execution_result_dir', 'result directory')):
      with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
        command.check_command(inputs['spec'], dict(runtime, **{field: '/different/path'}), data['record'])
      inv, obs = json.loads(json.dumps(invocation)), json.loads(json.dumps(observed))
      if field == 'runner_build_directory':
        inv['runner_build_directory'] = '/different/build'
        obs['records'][0]['effective_environment']['PATH'] = (
          inv['effective_environment']['PATH'] + os.pathsep + inv['runner_build_directory'])
      else:
        inv['bindings']['result_dir'] = '/different/results/attempt-1'
        inv['argv'][inv['runner_index'] + 1] = '/different/results/attempt-1/command-invocation.json'
        for environment in (inv['effective_environment'], obs['initial_environment'],
                            obs['records'][0]['effective_environment']):
          environment['PCCL_DEBUG_FILE'] = '/different/results/attempt-1/library.log'
      record = dict(data['record'], invocation_sha256=command.digest(inv))
      obs['invocation_sha256'] = record['invocation_sha256']
      record['observations_sha256'] = command.digest(obs)
      (target / 'command-invocation.json').write_text(json.dumps(inv))
      (target / 'command-observations.json').write_text(json.dumps(obs))
      with self.subTest(rehashed=field), self.assertRaisesRegex(ValueError, message):
        command.check_command(inputs['spec'], runtime, record)
      (target / 'command-invocation.json').write_text(json.dumps(invocation))
      (target / 'command-observations.json').write_text(json.dumps(observed))

  def test_source_logs_redact_environment_on_success_failure_and_timeout(self):
    value = 'offline-fixture-value-7319'
    for extra, timeout, error in (('', 3, None), ('raise SystemExit(7)', 3, 'CalledProcessError'),
                                ('time.sleep(60)', 0.3, 'TimeoutExpired')):
      with self.subTest(error=error):
        directory, data, _ = self.source_case(
          env={'CI_ACCESS_TOKEN': value}, passed=error is None, spec_changes=dict(timeout=timeout),
          runner_extra='print(os.environ["CI_ACCESS_TOKEN"], flush=True)',
          launcher_extra='print("safe child output", flush=True)\n' + extra)
        self.assertNotIn(value, '\n'.join(path.read_text() for path in directory.iterdir()))
        log = (directory / 'command.log').read_text()
        self.assertIn('REDACTED', log)
        if error != 'TimeoutExpired':
          self.assertIn('safe child output', log)
        if error:
          self.assertEqual(data['error_type'], error)
          self.assertTrue(data['record_rejected'])

  def test_source_private_log_and_child_environment(self):
    directory, _, _ = self.source_case(runner_extra='os.environ["NEW_PRIVATE"] = "opaque-value-xyz"\n'
                                                   'print(os.environ["NEW_PRIVATE"])')
    self.assertNotIn('opaque-value-xyz', '\n'.join(path.read_text() for path in directory.iterdir()))
    value = 'never-publish-this-value'
    directory, _, _ = self.source_case(env={'CREDENTIAL': value},
                                       runner_extra='print(os.environ["CREDENTIAL"])')
    self.assertNotIn(value, (directory / 'command.log').read_text())
    for value in ('https://private.invalid/access', 'ghp_abcdefghijklmnop', 'secret-value'):
      with self.subTest(value=value):
        directory, _, _ = self.source_case(runner_extra=f'os.environ["PCCL_DEBUG"] = {value!r}\n'
                                                       'print(os.environ["PCCL_DEBUG"])', passed=False)
        self.assertNotIn(value, '\n'.join(path.read_text() for path in directory.iterdir()))
        self.assertEqual(json.loads((directory / 'command-observations.json').read_text())['state'], 'invalid')

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
    value = 'offline-fixture-value-7319'
    directory, data, _ = self.case('import os\nprint(os.environ["CI_ACCESS_TOKEN"])\n'
                                   f'print({text!r}, end="")', spec=spec, runtime=runtime,
                                   env={'CI_ACCESS_TOKEN': value})
    self.assertNotIn(value, (directory / 'command.log').read_text())
    self.assertIn(text, (directory / 'command.log').read_text())
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
    self.assertEqual(sorted(path.name for path in directory.iterdir()), ['command-process.json', 'command-reproduction.txt', 'command.log'])
    for output in (ROW + FOOTER, text.replace('validation: 1', 'validation: 0'),
                   text.replace('0 OK', '1 FAILED')):
      with self.subTest(output=output):
        directory, data, _ = self.case(f'print({output!r})', spec=spec, runtime=runtime, passed=False)
        self.assertEqual(data['error_type'], 'ValueError')
        self.assertTrue(data['run_returned'])
        self.assertEqual(sorted(path.name for path in directory.iterdir()), ['command-process.json', 'command-reproduction.txt', 'command.log'])
    _, data, _ = self.case('import sys; sys.stdout.buffer.write(b"\\xff")', spec=spec,
                          runtime=runtime, passed=False)
    self.assertEqual(data['error_type'], 'UnicodeDecodeError')
    self.case(f'print({text!r})', spec=spec, runtime=runtime, env={'PCCL_DEBUG': 'INFO'}, passed=False)


@unittest.skipUnless(os.environ.get('PCCL_TESTS_ROOT'), 'Set PCCL_TESTS_ROOT to the pinned public checkout')
class PublicRunnerTests(unittest.TestCase):
  """Read-only public runner with a local fake launcher; no MPI or hardware."""

  @staticmethod
  def filelock_fixture(root):
    (root / 'filelock.py').write_text(
      'from contextlib import nullcontext\n'
      'def FileLock(*args, **kwargs):\n    return nullcontext()\n')

  def test_decimal_sizes_nonblocking_and_protocols_with_public_runner(self):
    from unittest.mock import patch

    temporary = tempfile.TemporaryDirectory(prefix='pccl-public-decimal-')
    self.addCleanup(temporary.cleanup)
    root = Path(temporary.name).resolve()
    runner = Path(os.environ['PCCL_TESTS_ROOT']).resolve()
    (root / 'run_test.py').symlink_to(runner / 'run_test.py')
    (root / 'run_test_config.py').symlink_to(runner / 'run_test_config.py')
    self.filelock_fixture(root)
    binary = root / 'bin'
    binary.mkdir()
    source = (f'#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n'
              'Path("decimal-actual.json").write_text(json.dumps(dict(argv=sys.argv, env=dict(os.environ))))\n'
              f'print({FOOTER!r})\n')
    for name in ('mpirun', 'all_reduce_perf'):
      (binary / name).write_text(source)
      (binary / name).chmod(0o700)
    serial = 0
    for size, expected in (('100.5MB', '105381888'), ('1.2GB', '1288490188')):
      for proto in ('SimpleBulk', 'Signal', 'SignalBulk'):
        for processes in (1, 2):
          with self.subTest(size=size, proto=proto, processes=processes):
            serial += 1
            args = ['--op', 'all_reduce_perf', '--algo', 'Auto', '--proto', proto, '--data-type', 'float',
                    '--data-size', size, '--iters', '10', '--ranks', '2', '--nnodes', '1',
                    '--nprocs', str(processes), '--threads', '1', '--non-blocking']
            spec = dict(argv=['{python}', '-B', '{test_root}/run_test.py'] + args +
                             ['--mpi-options={mpi_options}', '--process-group', 'normal', '--launch-mode', 'normal'],
                        env={'PCCL_SIGNAL_BUFFSIZE': '128', 'PCCL_SIMPLE_BULK_BUFFSIZE': '128',
                             'PCCL_SIGNAL_BULK_BUFFSIZE': '128'}, timeout=5,
                        nodes=1, processes_per_node=processes, threads_per_process=1, devices_per_thread=2 // processes,
                        checks=['exit_code', 'validation_summary'],
                        source_context=dict(source_command=shlex.join(['python3', 'run_test.py'] + args),
                                       mpi=dict(options=[], exports=['PATH', 'LD_LIBRARY_PATH'])))
            runtime = dict(test_root=str(root), binary_dir=str(binary), result_dir=str(root / f'case{serial}'),
                           mpi_options=[])
            with patch.dict(os.environ, {'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1'}, clear=True):
              try:
                record = command.run_command(spec, runtime, [])
              except subprocess.CalledProcessError:
                self.fail((Path(runtime['result_dir']) / 'command.log').read_text())
              self.assertEqual(command.check_command(spec, runtime, record), [])
            observed = command.read_json(Path(runtime['result_dir']) / 'command-observations.json')
            actual = command.read_json(root / 'decimal-actual.json')
            entry, = observed['records']
            self.assertEqual(entry['argv'][1:], actual['argv'][1:])
            self.assertEqual(entry['effective_environment'], command.safe_environment(actual['env']))
            self.assertEqual(entry['effective_environment']['PCCL_PROTO'], proto)
            for name in spec['env']:
              self.assertNotIn(name, entry['effective_environment'])
            benchmark = entry['argv'][entry['argv'].index('all_reduce_perf'):]
            values = command._benchmark_options(benchmark)
            self.assertEqual((values['-b'], values['-e']), (expected, expected))
            self.assertEqual(values['--init_blocking'], '0')
            # An int-rounded (rather than truncated) decimal is not equivalent.
            if size == '1.2GB':
              entry['argv'][entry['argv'].index('-b') + 1] = str(int(expected) + 1)
              (Path(runtime['result_dir']) / 'command-observations.json').write_text(json.dumps(observed))
              record['observations_sha256'] = command.digest(observed)
              with self.assertRaisesRegex(ValueError, 'benchmark arguments'):
                command.check_command(spec, runtime, record)

  def test_cluster_prefix_and_relocated_capture_with_public_runner(self):
    from unittest.mock import patch

    import cluster_ci

    runner = Path(os.environ['PCCL_TESTS_ROOT']).resolve()
    spec, inputs = cluster_ci.expected_inputs('a' * 40, runner)
    temporary = tempfile.TemporaryDirectory(prefix='pccl-public-runner-')
    self.addCleanup(temporary.cleanup)
    root = Path(temporary.name).resolve()
    binary = root / 'openmpi installation' / 'bin'
    binary.mkdir(parents=True)
    # The script and its sibling imports remain in the read-only checkout.
    (root / 'run_test.py').symlink_to(runner / 'run_test.py')
    (root / 'run_test_config.py').symlink_to(runner / 'run_test_config.py')
    self.filelock_fixture(root)
    (root / 'hosts.cfg').write_text('localhost slots=8\nlocalhost slots=8\n')
    (binary / 'orted').touch()  # mpi_options only checks its existence; never executed.
    launcher = binary / 'mpirun'
    launcher.write_text(
      f'#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n'
      'env = {key: value for key, value in os.environ.items()\n'
      '       if key in ("PATH", "LD_LIBRARY_PATH", "UMD_PLATFORM_TYPE")\n'
      '       or key.startswith(("PCCL_", "NCCL_", "HGGC_", "OMPI_", "UCX_"))}\n'
      'Path("actual.json").write_text(json.dumps(dict(argv=sys.argv, env=env)))\n'
      'print("synthetic launcher; no MPI or hardware")\n')
    launcher.chmod(0o700)
    environment = {'PATH': str(binary) + os.pathsep + '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1'}
    with patch.dict(os.environ, environment, clear=True):
      options = cluster_ci.mpi_options()
      self.assertEqual(options, ['--prefix', str(binary.parent), '-mca', 'plm_rsh_agent', 'cluster-ssh',
                                 '-mca', 'orte_base_help_aggregate', '0', '--hostfile', 'hosts.cfg'])
      runtime = dict(test_root=str(root), binary_dir=str(binary), mpi_options=options,
                     runner_build_directory=str((root / 'build').resolve()),
                     execution_result_dir=str(root / 'execution' / 'case' / 'attempt-1'),
                     result_dir=str(root / 'execution' / 'case' / 'attempt-1'))
      with command.cancellation_signals() as interrupted:
        record = command.run_command(spec, runtime, interrupted)
      self.assertEqual(command.check_command(spec, runtime, record), [])
    directory = Path(runtime['result_dir'])
    invocation = command.read_json(directory / 'command-invocation.json')
    observed = command.read_json(directory / 'command-observations.json')
    actual = command.read_json(root / 'actual.json')
    entry, = observed['records']
    self.assertEqual(entry['argv'][1:], actual['argv'][1:])
    self.assertEqual(entry['effective_environment'], actual['env'])
    self.assertEqual(entry['executable'], 'mpirun')
    self.assertEqual(Path(actual['argv'][0]), launcher)
    position = entry['argv'].index('all_reduce_perf')
    merged = command._unique_mpi(command._mpi_entries(entry['argv'][1:position]))
    self.assertEqual(merged['prefix'], str(binary.parent))
    self.assertEqual((merged['np'], merged['npernode']), ('16', '8'))
    self.assertEqual(merged['env:NCCL_NET_GDR_LEVEL'], '4')
    self.assertEqual(merged['mca:plm_rsh_agent'], 'cluster-ssh')
    self.assertNotIn('mca:btl_tcp_if_include', merged)
    self.assertEqual(merged['hostfile'], 'hosts.cfg')
    self.assertEqual(actual['env']['UMD_PLATFORM_TYPE'], '1')
    self.assertEqual(invocation['spec_sha256'], command.digest(spec))
    self.assertEqual(invocation['source_ids'], {key: spec['source_context'][key]
                                              for key in ('plan_id', 'suite_id', 'case_id', 'suite_run_id')})
    collected = root / 'summary' / 'case' / 'attempt-1'
    shutil.copytree(directory, collected)
    shutil.rmtree(root / 'execution')
    summary_runtime = dict(result_dir=str(collected), inputs=inputs,
                           runner_build_directory=runtime['runner_build_directory'],
                           execution_result_dir=runtime['execution_result_dir'],
                           command_log_ref='case/attempt-1/command.log')
    self.assertEqual(command.check_command(spec, summary_runtime, record), [])
    self.assertEqual(command.read_json(collected / 'command-invocation.json'), invocation)


if __name__ == '__main__':
  unittest.main(verbosity=2)
