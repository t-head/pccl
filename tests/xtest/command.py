# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct command execution and checking for trusted schema2 case inputs."""

from contextlib import contextmanager
import ctypes
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time

ENV_KEY = re.compile(r'(?:PCCL|NCCL|HGGC|OMPI|UCX)_[A-Za-z0-9_]+')
SUMMARY = re.compile(r'^#\s*Out of bounds values\s*:\s*0\s+OK\b', re.M)
GRACE = 0.5


def _prepare(spec, runtime):
  if os.environ.get('NODE_RANK', '0') != '0':
    raise ValueError('Only controller NODE_RANK=0 may run a case')
  for key in ('test_root', 'binary_dir', 'result_dir'):
    if not isinstance(runtime[key], str) or not Path(runtime[key]).is_absolute():
      raise ValueError(f'{key} must be an absolute path')
  for key in ('test_root', 'binary_dir'):
    if not Path(runtime[key]).is_dir():
      raise ValueError(f'{key} must be an existing directory')
  timeout = spec['timeout']
  if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
    raise ValueError('timeout must be positive finite seconds')
  for key in ('nodes', 'processes_per_node', 'threads_per_process', 'devices_per_thread'):
    if type(spec[key]) is not int or spec[key] <= 0:
      raise ValueError(f'{key} must be a positive integer')
  options, tokens = runtime['mpi_options'], spec['argv']
  if not isinstance(options, list) or any(not isinstance(x, str) or '\0' in x for x in options):
    raise ValueError('mpi_options must be list[str]')
  if not isinstance(tokens, list) or not tokens or any(not isinstance(x, str) or '\0' in x for x in tokens):
    raise ValueError('argv must be a nonempty list[str]')
  if spec['nodes'] > 1:
    hostfiles = [options[i + 1] for i, x in enumerate(options[:-1]) if x == '--hostfile']
    hostfiles += [x.split('=', 1)[1] for x in options if x.startswith('--hostfile=')]
    if not hostfiles or any(not x or not (Path(runtime['test_root']) / x).is_file() for x in hostfiles):
      raise ValueError('Multinode execution requires mpi_options with an existing --hostfile')
    if not any(x in tokens for x in ('{mpi_options}', '--mpi-options={mpi_options}')):
      raise ValueError('Multinode argv must consume mpi_options; non-MPI bootstrap requires P2')
  values = dict(python=sys.executable, test_root=runtime['test_root'], binary_dir=runtime['binary_dir'],
                result_dir=runtime['result_dir'])

  def replace(match):
    if match[1] not in values:
      raise ValueError(f'Unsupported placeholder: {match[0]}')
    return values[match[1]]

  argv = []
  for token in tokens:
    if token == '{mpi_options}':
      argv.extend(options)
    elif token == '--mpi-options={mpi_options}':
      argv.append('--mpi-options=' + shlex.join(options))
    else:
      argv.append(re.sub(r'\{([^{}]*)\}', replace, token))
  if not argv or not argv[0]:
    raise ValueError('Expanded argv must contain an executable')
  is_runner = any(Path(x).name == 'run_test.py' for x in argv[:2])
  if spec['nodes'] > 1 and not ((Path(argv[0]).name == 'mpirun' and '{mpi_options}' in tokens)
                              or (is_runner and '--mpi-options={mpi_options}' in tokens)):
    raise ValueError('Multinode requires mpirun or MPI-aware run_test.py; non-MPI bootstrap requires P2')
  if is_runner:
    groups = [argv[i + 1] for i, x in enumerate(argv[:-1]) if x == '--process-group']
    groups += [x.split('=', 1)[1] for x in argv if x.startswith('--process-group=')]
    modes = [argv[i + 1] for i, x in enumerate(argv[:-1]) if x == '--launch-mode']
    modes += [x.split('=', 1)[1] for x in argv if x.startswith('--launch-mode=')]
    if groups != ['normal'] or any(mode != 'normal' for mode in modes):
      raise ValueError('run_test.py requires --process-group normal and normal launch mode')
  env = os.environ.copy()
  if not isinstance(spec['env'], dict):
    raise ValueError('env must be a dictionary')
  if spec.get('test_type') == 'performance':
    # Undeclared inherited settings would invalidate the measurement identity.
    prefixes = ('PCCL_', 'NCCL_', 'HGGC_', 'OMPI_', 'UCX_')
    undeclared = sorted(key for key in env if key.startswith(prefixes) and key not in spec['env'])
    if undeclared:
      raise ValueError('Performance requires explicit env overrides/removals for inherited settings: '
                       + ', '.join(undeclared))
  for key, value in spec['env'].items():
    if not isinstance(key, str) or not ENV_KEY.fullmatch(key):
      raise ValueError('env override is outside the PCCL/NCCL/HGGC/OMPI/UCX whitelist')
    if value is None:
      env.pop(key, None)
    elif isinstance(value, str) and '\0' not in value:
      env[key] = re.sub(r'\{([^{}]*)\}', replace, value)
    else:
      raise ValueError('env values must be strings or None')
  for key, prefix in (('PATH', runtime['binary_dir']), ('LD_LIBRARY_PATH', str(Path(runtime['test_root']) / 'lib'))):
    env[key] = prefix + (os.pathsep + env[key] if env.get(key) else '')
  if is_runner and spec['nodes'] * spec['processes_per_node'] > 1:
    # The pinned runner interpolates these values before calling shlex.split again.
    for key, value in env.items():
      if key.startswith(('PCCL_', 'NCCL_')) and shlex.split(f'{key}={value}') != [f'{key}={value}']:
        raise ValueError(f'Public runner cannot preserve environment value: {key}')
  if is_runner or Path(argv[0]).name == 'mpirun':
    names = sorted({'PATH', 'LD_LIBRARY_PATH'} | {key for key in env if ENV_KEY.fullmatch(key)
                   and (not is_runner or not key.startswith(('PCCL_', 'NCCL_')))})
    exports = [token for name in names for token in ('-x', name)]
    if is_runner:
      index = next(i for i, token in enumerate(tokens) if token == '--mpi-options={mpi_options}')
      argv[index] = '--mpi-options=' + shlex.join(options + exports)
    else:
      index = tokens.index('{mpi_options}')
      argv[index:index] = exports
  return argv, env


class _MacBsdInfo(ctypes.Structure):
  # macOS sys/proc_info.h: proc_bsdinfo (PROC_PIDTBSDINFO = 3).
  _fields_ = [('ids', ctypes.c_uint32 * 12), ('names', ctypes.c_char * 48),
              ('group', ctypes.c_uint32 * 6), ('seconds', ctypes.c_uint64), ('microseconds', ctypes.c_uint64)]


def _identity(pid):
  try:
    if sys.platform.startswith('linux'):
      fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
      return dict(pid=pid, starttime=fields[19], identity_source='proc',
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    if sys.platform == 'darwin':
      info = _MacBsdInfo()
      query = ctypes.CDLL('/usr/lib/libproc.dylib').proc_pidinfo
      query.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
      query.restype = ctypes.c_int
      if query(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info)) == ctypes.sizeof(info) and info.ids[3] == pid:
        return dict(pid=pid, starttime=f'{info.seconds}.{info.microseconds:06d}', identity_source='proc_pidinfo')
  except (OSError, IndexError):
    pass
  return None


def _matches(pid, pgid, identity):
  try:
    return bool(identity) and _identity(pid) == identity and os.getpgid(pid) == pgid and os.getsid(pid) == pgid
  except ProcessLookupError:
    return False


def _members(pgid):
  output = subprocess.run(['/bin/ps', '-axo', 'pid=,pgid=,stat='], check=True,
                          capture_output=True, text=True, timeout=2).stdout
  return [int(p) for p, g, state in (line.split() for line in output.splitlines())
          if int(g) == pgid and not state.startswith('Z')]


def _stop_group(pgid, witnesses):
  # Keep original live members as witnesses after the leader exits. Never chase a reused PGID.
  for sig in (signal.SIGTERM, signal.SIGKILL):
    members = _members(pgid)
    if not members:
      return
    if not any(_matches(pid, pgid, identity) for pid, identity in witnesses.items()):
      raise RuntimeError('Cannot safely identify remaining process group')
    witnesses.update({pid: _identity(pid) for pid in members})
    try:
      os.killpg(pgid, sig)
    except ProcessLookupError:
      return
    deadline = time.monotonic() + GRACE
    while time.monotonic() < deadline:
      if not _members(pgid):
        return
      time.sleep(0.02)
  if _members(pgid):
    raise RuntimeError('Process group survived SIGKILL')


def _write_record(path, record):
  with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
    temporary = Path(stream.name)
    try:
      # Hardware execution and summary run as different users against shared
      # storage. This record contains process identity/state only, not secrets.
      os.fchmod(stream.fileno(), 0o644)
      json.dump(record, stream)
      stream.flush()
      os.fsync(stream.fileno())
      os.replace(temporary, path)
    finally:
      temporary.unlink(missing_ok=True)


def stop_recorded_process(path):
  """Cancel a trusted live leader; stale identities and detached sessions are not killed."""
  path = Path(path)
  record = json.loads(path.read_text())
  pid = record.get('pid')
  if (not record.get('active') or type(pid) is not int or pid <= 1 or record.get('pgid') != pid
      or not _matches(pid, pid, record.get('identity'))):
    return False
  witnesses = {member: _identity(member) for member in _members(pid)}
  if not _matches(pid, pid, record['identity']):
    return False
  _stop_group(pid, witnesses)
  # The owner reaps and publishes the definitive return code in its finally block.
  return True


@contextmanager
def cancellation_signals():
  """Let the caller defer cancellation until the owned command has been cleaned up."""
  interrupted = []
  previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
  try:
    for sig in previous:
      signal.signal(sig, lambda signum, frame: interrupted.append(signum))
    yield interrupted
  finally:
    for sig, handler in previous.items():
      signal.signal(sig, handler)


def run_command(spec, runtime, interrupted):
  """Return a final record only after successful execution and complete cleanup."""
  if interrupted:
    raise RuntimeError(f'Command cancelled by signal {interrupted[0]}')
  argv, env = _prepare(spec, runtime)
  directory = Path(runtime['result_dir'])
  directory.mkdir(parents=True, exist_ok=True)
  record_path, log_path = directory / 'command-process.json', directory / 'command.log'
  if os.path.lexists(record_path):
    raise FileExistsError('Case process record already exists')
  process, record = None, dict(active=False, state='launch_failed', pid=None, pgid=None,
                              identity=None, return_code=None)
  with log_path.open('x') as output:
    try:
      if interrupted:
        raise RuntimeError(f'Command cancelled by signal {interrupted[0]}')
      process = subprocess.Popen(argv, shell=False, cwd=runtime['test_root'], env=env,
                                 stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
      record.update(pid=process.pid, pgid=process.pid, identity=_identity(process.pid), active=True, state='running')
      _write_record(record_path, record)
      if not record['identity']:
        raise RuntimeError('Cannot record command process identity')
      deadline = time.monotonic() + spec['timeout']
      while True:
        if interrupted:
          raise RuntimeError(f'Command cancelled by signal {interrupted[0]}')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
          raise subprocess.TimeoutExpired(argv, spec['timeout'])
        try:
          process.wait(timeout=min(remaining, 0.1))
          break
        except subprocess.TimeoutExpired:
          pass
      if interrupted:
        raise RuntimeError(f'Command cancelled by signal {interrupted[0]}')
      if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, argv)
      record['state'] = 'finished'
    except BaseException as error:
      record['state'] = 'timeout' if isinstance(error, subprocess.TimeoutExpired) else 'failed'
      raise
    finally:
      original_error = sys.exc_info()[1]
      try:
        if process is not None:
          exited = process.returncode is not None
          cleaned = False
          try:
            members = _members(process.pid)
            if members:
              _stop_group(process.pid, {pid: _identity(pid) for pid in members})
            cleaned = True
          finally:
            try:
              if not cleaned and process.returncode is None:
                # This direct child is still unreaped, so its PID cannot be reused.
                for sig in (signal.SIGTERM, signal.SIGKILL):
                  try:
                    os.killpg(process.pid, sig)
                  except ProcessLookupError:
                    break
                  if sig == signal.SIGTERM:
                    time.sleep(GRACE)
            finally:
              process.wait(timeout=2)
          if members and exited:
            raise RuntimeError('Command left residual children')
      except BaseException as cleanup_error:
        record['state'] = 'cleanup_failed'
        if original_error is None:
          raise
        original_error.add_note(f'Cleanup failed: {cleanup_error}')
      finally:
        pending_error = sys.exc_info()[1]
        record.update(active=False, return_code=process.returncode if process else None)
        if interrupted and record['state'] == 'finished':
          record['state'] = 'failed'
        try:
          _write_record(record_path, record)
        except BaseException as record_error:
          if pending_error is None:
            raise
          pending_error.add_note(f'Final record publication failed: {record_error}')
  if interrupted:
    if record['state'] == 'finished':
      record['state'] = 'failed'
      _write_record(record_path, record)
    raise RuntimeError(f'Command cancelled by signal {interrupted[0]}')
  return record


def check_command(spec, runtime, record):
  """Validate final execution evidence and return metrics without publishing files."""
  if (not isinstance(record, dict) or record.get('active') is not False
      or record.get('state') != 'finished' or type(record.get('return_code')) is not int
      or record['return_code'] != 0):
    raise ValueError('No successful completed command')
  log = Path(runtime['result_dir']) / 'command.log'
  if not log.is_file():
    raise ValueError('Missing command log')
  checks = spec.get('checks')
  if (not isinstance(checks, list) or not checks or any(not isinstance(check, str) for check in checks)
      or set(checks) - {'exit_code', 'validation_summary', 'smoke_8mib', 'performance'}):
    raise ValueError('Unsupported or empty checks')
  text = log.read_bytes().decode('utf-8', errors='strict' if 'performance' in checks else 'replace')
  if any(x in checks for x in ('validation_summary', 'smoke_8mib')) and not SUMMARY.search(text):
    raise ValueError('Missing successful data-validation summary')
  if 'smoke_8mib' in checks:
    rows = [line.split() for line in text.splitlines() if re.match(r'^\s*8388608\s+2097152\s+float\s+sum\s', line)]
    if len(rows) != 1 or len(rows[0]) != 13:
      raise ValueError('Missing or unexpected 8 MiB AllReduce result')
    if rows[0][8] != '0' or rows[0][12] != '0':
      raise ValueError('AllReduce validation failed or was disabled for a buffer layout')
  if 'performance' in checks:
    from metrics import parse_metrics

    return parse_metrics(text, spec, runtime['case_key'], runtime['inputs'], runtime['command_log_ref'])
  return []
