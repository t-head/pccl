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

from runner_capture import ENV_KEY, PRIVATE, SECRET, digest, env_name, read_json, safe_environment, safe_text

SUMMARY = re.compile(r'^#\s*Out of bounds values\s*:\s*0\s+OK\b', re.M)
GRACE = 0.5
EXPORT_RULE = 'source names plus present allowed non-PCCL/NCCL names; runner exports present PCCL/NCCL values'
RUNNER_VALUES = set('op algo proto redop data-type data-size min-data-size max-data-size random-size-num '
                    'iters ranks threads nprocs nnodes register mpi-options cross-node devices timeout '
                    'launch-mode process-group warmup average cta-policy init_type elapsed_type '
                    'test-buffer-kind'.split())
RUNNER_FLAGS = {'auto-detect', 'group-launch', 'non-blocking', 'enable-single-test', 'no-data-check', 'use_4nics'}
COLLECTIVES = {'all_reduce_perf', 'all_gather_perf', 'broadcast_perf', 'reduce_perf', 'reduce_scatter_perf',
               'alltoall_perf', 'alltoallv_perf', 'gather_perf', 'scatter_perf', 'sendrecv_perf', 'hypercube_perf'}
DIRECT_COLLECTIVES = {'all_reduce_perf', 'alltoall_perf', 'reduce_scatter_perf', 'gather_perf', 'scatter_perf'}
DIRECT_VALUES = set('-b -e -f -R -d -g -t -x -o -n -w -c -k -a -N -D -V '
                    '--init_blocking --elapsed_type --report_type'.split())
DIRECT_EXPORT_RULE = 'source names plus present allowed names including PCCL/NCCL; no runner-added exports'
INFRA_MCA = {'plm_rsh_agent', 'orte_base_help_aggregate', 'btl_tcp_if_include'}
RUNNER_TIMEOUT_GRACE = 30
CONTROL_ENV = frozenset({
  'PCCL_REVISION', 'PCCL_TESTS_ROOT', 'PCCL_SOURCE_CATALOG', 'PCCL_MPI_OPTIONS_FILE',
})


def _runner_size(size):
  match = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)(B|KB|MB|GB)?', size)
  if not match:
    raise ValueError('Unsupported source benchmark size contract')
  # Match the fixed public runner's obtain_data_size, including binary-float
  # rounding followed by truncation. Decimal arithmetic would change its argv.
  amount = float(match[1]) * {'B': 1, 'KB': 1024, 'MB': 1024 ** 2, 'GB': 1024 ** 3, None: 1}[match[2]]
  if not math.isfinite(amount) or amount < 1:
    raise ValueError('Unsupported source benchmark size contract')
  return str(int(amount))


def _runner_index(argv):
  if not argv or not (argv[0] == '{python}' or re.fullmatch(r'python(?:3(?:\.\d+)?)?', Path(argv[0]).name)):
    return None
  index = 1
  while index < len(argv) and argv[index] in ('-u', '-S', '-B'):
    index += 1
  return index if index < len(argv) and Path(argv[index]).name == 'run_test.py' else None


def _runner_options(tokens):
  result, index = {}, 0
  while index < len(tokens):
    token = tokens[index]
    name, equal, value = token.removeprefix('--').partition('=')
    name = 'nprocs' if name == 'nproc' else name
    if not token.startswith('--') or name in result or name not in RUNNER_VALUES | RUNNER_FLAGS:
      raise ValueError('Unknown or duplicate source runner option')
    index += 1
    if name in RUNNER_FLAGS:
      if equal:
        raise ValueError('Runner flag must not have a value')
      result[name] = True
    else:
      if not equal:
        if index >= len(tokens) or tokens[index].startswith('--'):
          raise ValueError('Missing source runner option value')
        value = tokens[index]
        index += 1
      safe_text(value)
      if not value:
        raise ValueError('Empty runner option value')
      result[name] = value
  return result


def _mpi_entries(tokens):
  if not isinstance(tokens, list) or any(not isinstance(x, str) or not x for x in tokens):
    raise ValueError('MPI options must be nonempty string tokens')
  entries, index = [], 0
  while index < len(tokens):
    token = tokens[index]
    count = 3 if token in ('-mca', '--mca') else 2
    if token == '--allow-run-as-root':
      count = 1
    elif token.startswith('--hostfile='):
      entries.append(('hostfile', token.split('=', 1)[1], [token]))
      index += 1
      continue
    elif token not in ('-mca', '--mca', '-x', '--hostfile', '--prefix', '-np', '-n', '-npernode'):
      raise ValueError('Unsupported MPI option')
    if index + count > len(tokens):
      raise ValueError('Missing MPI option value')
    group = tokens[index:index + count]
    for value in group:
      safe_text(value)
    if token in ('-mca', '--mca'):
      key, value = 'mca:' + group[1], group[2]
    elif token == '-x':
      name, equal, value = group[1].partition('=')
      if not env_name(name) or SECRET.search(name):
        raise ValueError('MPI export is outside the safe environment whitelist')
      key, value = 'env:' + name, value if equal else None
    else:
      key = {'--hostfile': 'hostfile', '--prefix': 'prefix', '-np': 'np', '-n': 'np', '-npernode': 'npernode',
             '--allow-run-as-root': 'allow-root'}[token]
      value = group[1] if count == 2 else True
      # OpenMPI's installation prefix belongs to the worker image, not the
      # benchmark contract. Summary must not require that installation locally.
      if key == 'prefix' and (not Path(value).is_absolute() or '{' in value or '}' in value):
        raise ValueError('MPI --prefix must be an absolute installation path')
    entries.append((key, value, group))
    index += count
  return entries


def _unique_mpi(entries):
  result = {}
  for key, value, _ in entries:
    if key in result:
      raise ValueError('Duplicate MPI option')
    result[key] = value
  return result


def _source_environment(argv, spec):
  if not argv or argv[0] != 'env':
    return argv, set()
  index, names = 1, set()
  while index < len(argv) and '=' in argv[index]:
    name, _, value = argv[index].partition('=')
    bindings = spec['execution'].get('environment', {}).get('path_bindings', {})
    diagnostic = (spec.get('test_type') == 'performance'
                  and name in ('PCCL_DEBUG_FILE', 'NCCL_DEBUG_FILE') and value == 'library.%h.%p.log'
                  and bindings.get(name) == '{result_dir}/' + value)
    expected = bindings[name] if diagnostic else value
    if (not (ENV_KEY.fullmatch(name) or name == 'DEBUG_FILE') or SECRET.search(name)
        or name in names or not isinstance(spec['env'], dict) or spec['env'].get(name) != expected
        or '{' in value or '}' in value):
      raise ValueError('Source environment prefix differs from spec or is unsafe')
    if (name == 'DEBUG_FILE' or name.endswith('_DEBUG_FILE')) and value != 'library.%h.%p.log':
      raise ValueError('Unsupported source-relative diagnostic file')
    names.add(name)
    index += 1
  if not names or index == len(argv):
    raise ValueError('Incomplete source environment command')
  return argv[index:], names


def _direct_contract(spec, source, mpi):
  is_mpi = bool(source) and source[0] == 'mpirun'
  position = next((i for i, token in enumerate(source) if token in DIRECT_COLLECTIVES), None)
  if position is None or (not is_mpi and position != 0):
    raise ValueError('Unknown direct source executable')
  if any(not re.fullmatch(r'[A-Za-z0-9_./,:=+^@%\-]+', token) for token in source):
    raise ValueError('Direct source requires fixed shell-safe tokens')
  prefix, benchmark = source[1:position] if is_mpi else [], source[position:]
  if prefix != mpi['options']:
    raise ValueError('Direct source and execution MPI prefix disagree')
  entries = _mpi_entries(prefix)
  layout = _unique_mpi(entries)
  if any(key in ('hostfile', 'prefix', 'mca:plm_rsh_agent') for key in layout):
    raise ValueError('Direct source MPI options conflict with runtime ownership')
  if (spec['nodes'] != 1 or (is_mpi and layout.get('np') != str(spec['processes_per_node']))
      or (not is_mpi and spec['processes_per_node'] != 1)
      or ('npernode' in layout and layout['npernode'] != str(spec['processes_per_node']))):
    raise ValueError('Direct source MPI layout differs from spec')
  if len(benchmark) % 2 != 1:
    raise ValueError('Direct benchmark options require fixed values')
  values = {}
  for key, value in zip(benchmark[1::2], benchmark[2::2]):
    if key not in DIRECT_VALUES or key in values:
      raise ValueError('Unknown or duplicate direct benchmark option')
    pattern = (r'[0-9]+(?:\.[0-9]+)?(?:[KMG]B?|B)?' if key in ('-b', '-e') else
               r'[A-Za-z][A-Za-z0-9_]*' if key in ('-d', '-o') else r'-?[0-9]+')
    if not re.fullmatch(pattern, value):
      raise ValueError('Direct benchmark option requires a fixed literal value')
    values[key] = value
  if (values.get('-t', '1') != str(spec['threads_per_process'])
      or values.get('-g', '1') != str(spec['devices_per_thread'])):
    raise ValueError('Direct source benchmark layout differs from spec')
  expected = (['mpirun', '{mpi_options}'] if is_mpi else []) + ['{binary_dir}/' + benchmark[0]] + benchmark[1:]
  if spec['argv'] != expected:
    raise ValueError('Direct source and spec argv disagree')
  return dict(direct=True, mpi=is_mpi, op=benchmark[0]), entries


def _source_contract(spec):
  execution = spec['execution']
  if not isinstance(execution, dict):
    raise ValueError('execution must be an object')
  ids = {key: execution[key] for key in ('plan_id', 'suite_id', 'case_id', 'suite_run_id') if key in execution}
  for value in ids.values():
    if (not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]+', value)
        or SECRET.search(value) or PRIVATE.search(value)):
      raise ValueError('Invalid source identity')
  source = execution.get('source_command')
  if not isinstance(source, str) or not source or re.search(r'[\x00-\x1f$`;&|<>]', source):
    raise ValueError('Unknown or unsafe source command')
  source_argv = shlex.split(source)
  for token in source_argv:
    safe_text(token)
  source_argv, _ = _source_environment(source_argv, spec)
  mpi = execution.get('mpi')
  if not isinstance(mpi, dict) or set(mpi) != {'options', 'exports'}:
    raise ValueError('Invalid source MPI contract')
  names = mpi['exports']
  if (not isinstance(names, list) or any(not env_name(x) or SECRET.search(x) for x in names)
      or len(names) != len(set(names))):
    raise ValueError('Invalid source MPI exports')
  source_index, index = _runner_index(source_argv), _runner_index(spec['argv'])
  if source_index is None:
    original, entries = _direct_contract(spec, source_argv, mpi)
    return ids, original, entries
  if index is None:
    raise ValueError('Source runner differs from spec executable')
  if ('-u' in source_argv[1:source_index]) != ('-u' in spec['argv'][1:index]):
    raise ValueError('Source interpreter buffering option differs from spec')
  original = _runner_options(source_argv[source_index + 1:])
  actual = _runner_options(spec['argv'][index + 1:])
  if actual.get('mpi-options') != '{mpi_options}':
    raise ValueError('Source argv must consume exactly one MPI options placeholder')
  source_entries = _mpi_entries(shlex.split(original.get('mpi-options', '')))
  contract_entries = _mpi_entries(mpi['options'])
  source_all, contract_all = _unique_mpi(source_entries), _unique_mpi(contract_entries)
  layout_keys = ('np', 'npernode', 'allow-root', 'prefix')
  if any(key in layout_keys for key in (*source_all, *contract_all)):
    raise ValueError('Source MPI options conflict with runner layout ownership')

  def portable(entries):
    # Network/launcher MCA choices are supplied by the execution environment.
    # The source declaration is retained for provenance but cannot pin them.
    return {key: value for key, value in entries.items()
            if not (key.startswith('mca:') and key[4:] in INFRA_MCA)}

  source_map, contract_map = portable(source_all), portable(contract_all)
  if 'hostfile' in source_map:
    # The original command keeps its safe source path; only the contract is portable.
    safe_text(source_map['hostfile'])
    source_map['hostfile'] = '{hostfile}'
  if source_map != contract_map:
    raise ValueError('Source command and execution MPI options disagree')
  defaults = dict(nnodes='1', nprocs='1', threads='1', redop='sum', timeout=str(spec['timeout']),
                  **{'launch-mode': 'normal', 'process-group': 'normal'})
  for options in (original, actual):
    options.pop('mpi-options', None)
    for key, value in defaults.items():
      options.setdefault(key, value)
  if original != actual:
    raise ValueError('Source command and spec argv disagree')
  if original['launch-mode'] != 'normal' or original['process-group'] != 'normal':
    raise ValueError('Only normal runner launch/process groups support capture')
  for name, field in (('nnodes', 'nodes'), ('nprocs', 'processes_per_node'), ('threads', 'threads_per_process')):
    if original[name] != str(spec[field]):
      raise ValueError('Source runner layout differs from spec')
  if (not {'op', 'data-type', 'data-size', 'iters', 'ranks'} <= original.keys()
      or original['op'] not in COLLECTIVES
      or not re.fullmatch(r'[1-9][0-9]*', original['iters'])
      or original.get('random-size-num', '0') != '0'):
    raise ValueError('Unsupported or incomplete source benchmark contract')
  _runner_size(original['data-size'])
  ranks = spec['nodes'] * spec['processes_per_node'] * spec['threads_per_process'] * spec['devices_per_thread']
  if original['ranks'] != str(ranks):
    raise ValueError('Source rank count differs from spec')
  return ids, original, contract_entries


def _source_mpi(spec, runtime, env):
  _, options, source = _source_contract(spec)
  infrastructure = _mpi_entries(runtime['mpi_options'])
  infra_map = _unique_mpi(infrastructure)
  if any(key not in ('hostfile', 'prefix') and key not in {'mca:' + name for name in INFRA_MCA} for key in infra_map):
    raise ValueError('Runtime MPI options conflict with source ownership')
  if options.get('direct') and not options['mpi']:
    if infrastructure:
      raise ValueError('Non-MPI direct source cannot consume runtime MPI options')
    return []
  merged, used = [], set()
  for key, value, tokens in source:
    if key.startswith('mca:') and key[4:] in INFRA_MCA:
      continue
    if key == 'hostfile':
      if value != '{hostfile}' or not infra_map.get(key):
        raise ValueError('Source hostfile requires a runtime --hostfile binding')
      tokens = ['--hostfile', safe_text(infra_map[key])]
      used.add(key)
    elif key in infra_map:
      raise ValueError('Source and runtime MPI options conflict')
    if key.startswith('env:') and value is not None and key[4:] in env and env[key[4:]] != value:
      raise ValueError('Source MPI export conflicts with the command environment')
    merged.extend(tokens)
  merged.extend(token for key, _, tokens in infrastructure if key not in used for token in tokens)
  existing = {key[4:] for key, _, _ in source if key.startswith('env:')}
  names = set(spec['execution']['mpi']['exports']) | {'PATH', 'LD_LIBRARY_PATH'}
  names |= {key for key in env if env_name(key) and (options.get('direct') or not key.startswith(('PCCL_', 'NCCL_')))}
  for name in sorted(names - existing):
    merged.extend(['-x', name])
  return merged


def _prepare(spec, runtime, *, inherited=None, validate_paths=True, python=None):
  if validate_paths and os.environ.get('NODE_RANK', '0') != '0':
    raise ValueError('Only controller NODE_RANK=0 may run a case')
  for key in ('test_root', 'binary_dir', 'result_dir'):
    if not isinstance(runtime[key], str) or not Path(runtime[key]).is_absolute():
      raise ValueError(f'{key} must be an absolute path')
  for key in ('test_root', 'binary_dir'):
    if validate_paths and not Path(runtime[key]).is_dir():
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
  source_names = set()
  if 'execution' in spec:
    options = _source_mpi(spec, runtime, {})
    _, source_names = _source_environment(shlex.split(spec['execution']['source_command']), spec)
  if spec['nodes'] > 1:
    hostfiles = [options[i + 1] for i, x in enumerate(options[:-1]) if x == '--hostfile']
    hostfiles += [x.split('=', 1)[1] for x in options if x.startswith('--hostfile=')]
    if (not hostfiles or any(not x for x in hostfiles)
        or (validate_paths and any(not (Path(runtime['test_root']) / x).is_file() for x in hostfiles))):
      raise ValueError('Multinode execution requires mpi_options with an existing --hostfile')
    if not any(x in tokens for x in ('{mpi_options}', '--mpi-options={mpi_options}')):
      raise ValueError('Multinode argv must consume mpi_options; non-MPI bootstrap requires P2')
  values = dict(python=python or sys.executable, test_root=runtime['test_root'], binary_dir=runtime['binary_dir'],
                result_dir=runtime['result_dir'])

  def replace(match):
    if match[1] not in values:
      raise ValueError(f'Unsupported placeholder: {match[0]}')
    return values[match[1]]

  def expand(options):
    argv = []
    for token in tokens:
      if token == '{mpi_options}':
        argv.extend(options)
      elif token == '--mpi-options={mpi_options}':
        argv.append('--mpi-options=' + shlex.join(options))
      else:
        argv.append(re.sub(r'\{([^{}]*)\}', replace, token))
    return argv

  argv = expand(options)
  if not argv or not argv[0]:
    raise ValueError('Expanded argv must contain an executable')
  is_runner = _runner_index(argv) is not None
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
  env = {key: value for key, value in (os.environ if inherited is None else inherited).items()
         if key not in CONTROL_ENV}
  if not isinstance(spec['env'], dict):
    raise ValueError('env must be a dictionary')
  if spec.get('test_type') == 'performance':
    # Undeclared inherited settings would invalidate the measurement identity.
    undeclared = sorted(key for key in env if ENV_KEY.fullmatch(key) and key not in spec['env'])
    if undeclared:
      raise ValueError('Performance requires explicit env overrides/removals for inherited settings: '
                       + ', '.join(undeclared))
  for key, value in spec['env'].items():
    source_diagnostic = key == 'DEBUG_FILE' and key in source_names and value == 'library.%h.%p.log'
    if not isinstance(key, str) or not (ENV_KEY.fullmatch(key) or source_diagnostic):
      raise ValueError('env override is outside the PCCL/NCCL/HGGC/OMPI/UCX/UMD_PLATFORM_TYPE whitelist')
    if value is None:
      env.pop(key, None)
    elif isinstance(value, str) and '\0' not in value:
      env[key] = re.sub(r'\{([^{}]*)\}', replace, value)
    else:
      raise ValueError('env values must be strings or None')
  for key, prefix in (('PATH', runtime['binary_dir']), ('LD_LIBRARY_PATH', str(Path(runtime['test_root']) / 'lib'))):
    env[key] = prefix + (os.pathsep + env[key] if env.get(key) else '')
  # Keep runner diagnostics in the command log even when the process is
  # killed by a timeout; buffered Python output otherwise disappears.
  env.setdefault('PYTHONUNBUFFERED', '1')
  if is_runner and spec['nodes'] * spec['processes_per_node'] > 1:
    # The pinned runner interpolates these values before calling shlex.split again.
    for key, value in env.items():
      if key.startswith(('PCCL_', 'NCCL_')) and shlex.split(f'{key}={value}') != [f'{key}={value}']:
        raise ValueError(f'Public runner cannot preserve environment value: {key}')
  if 'execution' in spec:
    safe_environment(env)
    argv = expand(_source_mpi(spec, runtime, env))
  elif is_runner or Path(argv[0]).name == 'mpirun':
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
      # storage. Callers must validate evidence before publication; no raw env.
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


def _capture_argv(argv, index, capture, directory, python):
  prefix = argv[:index] if index is not None else [python, '-S', '-B']
  original = argv[index:] if index is not None else argv
  return prefix + [capture, str(Path(directory) / 'command-invocation.json'), '--'] + original


def _invocation(spec, runtime, argv, env):
  ids, source, _ = _source_contract(spec)
  index = _runner_index(argv)
  for key in ('test_root', 'binary_dir', 'result_dir'):
    safe_text(runtime[key])
  capture = safe_text(str(Path(__file__).with_name('runner_capture.py').resolve()))
  launched = _capture_argv(argv, index, capture, runtime['result_dir'], sys.executable)
  options = _source_mpi(spec, runtime, env)
  return dict(schema_version=1, spec_sha256=digest(spec), source_command=spec['execution']['source_command'],
              source_ids=ids, argv=launched, original_argv=argv, runner_index=index, cwd=runtime['test_root'],
              runner_build_directory=safe_text(str((Path(runtime['test_root']) / 'build').resolve())),
              effective_environment=safe_environment(env), inherited_environment=safe_environment(os.environ),
              mpi_options=options, mpi_exports=[key[4:] for key, _, _ in _mpi_entries(options)
                                                if key.startswith('env:')],
              mpi_export_rule=DIRECT_EXPORT_RULE if source.get('direct') else EXPORT_RULE,
              bindings={key: runtime[key] for key in ('test_root', 'binary_dir', 'result_dir', 'mpi_options')})


def _benchmark_options(argv):
  if not argv or argv[0] not in COLLECTIVES or len(argv) % 2 != 1:
    raise ValueError('Unknown observed benchmark argv')
  allowed = {'-t', '-g', '-d', '-b', '-e', '-f', '-n', '-w', '-o', '-a', '-R', '-x', '-p', '-c', '-s', '-k',
             '--core_dump', '--elapsed_type', '--random_size_num', '--sizelist', '--init_blocking'}
  result = {}
  for key, value in zip(argv[1::2], argv[2::2]):
    safe_text(value)
    if key not in allowed:
      raise ValueError('Unknown observed benchmark option')
    # The public runner appends -R 1 for reduce and may append --init_blocking 0 twice.
    if key in result and (key != '-R' or value != '1') and (key != '--init_blocking' or value != '0'):
      raise ValueError('Duplicate observed benchmark option')
    result[key] = value
  return result


def _observed_contract(invocation, options, entry, spec):
  if (not isinstance(entry, dict) or set(entry) != {'sequence', 'source', 'executable', 'argv', 'cwd',
                                                  'effective_environment', 'private_environment_changed',
                                                  'environment_source'}
      or entry.get('source') != 'observed-subprocess'
      or entry.get('environment_source') not in ('explicit', 'inherited')
      or type(entry.get('private_environment_changed')) is not bool
      or entry.get('cwd') != invocation['cwd']):
    raise ValueError('Invalid observed subprocess evidence')
  argv = entry.get('argv')
  if not isinstance(argv, list) or not argv or entry.get('executable') != argv[0]:
    raise ValueError('Observed executable/argv mismatch')
  for token in argv:
    safe_text(token)
  environment = entry.get('effective_environment')
  if not isinstance(environment, dict) or safe_environment(environment) != environment:
    raise ValueError('Invalid observed environment')
  if options.get('direct'):
    # Unlike run_test.py, the direct capture child has no argv/env transformations.
    # Match every token (including absolute benchmark path and MPI option order).
    if (argv != invocation['original_argv'] or environment != invocation['effective_environment']
        or entry['environment_source'] != 'inherited' or entry['private_environment_changed']):
      raise ValueError('Observed direct command differs from source expansion')
    return
  mpi = argv[0] == 'mpirun'
  ranks = spec['nodes'] * spec['processes_per_node']
  if mpi != (ranks > 1):
    raise ValueError('Observed MPI launch differs from source layout')
  if mpi:
    # Locate the benchmark structurally, never by parsing stdout.
    position = next((i for i, token in enumerate(argv[1:], 1) if token == options['op']), None)
    if position is None:
      raise ValueError('Missing observed MPI benchmark')
    mpi_entries = _mpi_entries(argv[1:position])
    expected = [('allow-root', True, []), ('np', str(ranks), []),
                ('npernode', str(spec['processes_per_node']), [])]
    expected += [('env:' + key, value, []) for key, value in environment.items()
                 if key.startswith(('PCCL_', 'NCCL_'))]
    expected += _mpi_entries(invocation['mpi_options'])

    def resolved(entries):
      values = {}
      for key, value, _ in entries:
        if key.startswith('env:'):
          value = environment.get(key[4:]) if value is None else value
        if key in values and (not key.startswith('env:') or values[key] != value):
          raise ValueError('Conflicting or duplicate observed MPI option')
        values[key] = value
      return values

    if resolved(mpi_entries) != resolved(expected):
      raise ValueError('Observed MPI options/exports differ from source and runtime')
    argv = argv[position:]
  if argv[0] != options['op']:
    raise ValueError('Observed operation differs from source')
  benchmark = _benchmark_options(argv)
  expected_benchmark = {
    '-t': options['threads'], '-g': str(int(options['ranks']) // (ranks * int(options['threads']))),
    '-d': options['data-type'], '-n': options['iters'], '-w': options.get('warmup', '0'),
    '-o': options['redop'], '-a': options.get('average', '2'), '-R': options.get('register', '0'),
    '-x': options.get('cta-policy', '0'), '-p': options.get('init_type', '0'),
    '--core_dump': '0', '--elapsed_type': options.get('elapsed_type', '0'),
  }
  if options.get('random-size-num', '0') != '0':
    raise ValueError('Unsupported source benchmark size contract')
  amount = _runner_size(options['data-size'])
  expected_benchmark.update({'-b': amount, '-e': amount, '-f': '1'})
  if int(options['ranks']) // spec['nodes'] == 8 and spec['nodes'] <= 2:
    if options['op'] == 'reduce_perf':
      expected_benchmark['-R'] = '1'
    elif options['op'] == 'broadcast_perf':
      expected_benchmark['--init_blocking'] = '0'
  for flag, key, value in (('no-data-check', '-c', '0'), ('enable-single-test', '-s', '1'),
                           ('non-blocking', '--init_blocking', '0')):
    if options.get(flag):
      expected_benchmark[key] = value
  if options.get('test-buffer-kind', '2') != '2':
    expected_benchmark['-k'] = options['test-buffer-kind']
  if benchmark != expected_benchmark:
    raise ValueError('Observed benchmark arguments differ from source')
  expected_env = dict(invocation['effective_environment'])
  expected_env['PATH'] += os.pathsep + invocation['runner_build_directory']
  expected_env['PCCL_SOCKET_IFNAME'] = '^flannel,docker,lo'
  if options.get('devices'):
    expected_env['HGGC_VISIBLE_DEVICES'] = options['devices']
  if options.get('group-launch'):
    expected_env['PCCL_LAUNCH_MODE'] = 'GROUP'
  if options.get('use_4nics'):
    expected_env['PCCL_IB_HCA'] = 'mlx5_bond_1,mlx5_bond_3,mlx5_bond_5,mlx5_bond_7'
  for suffix in ('ALGO', 'DISABLE_FC_GRAPH', 'PROTO', 'LL_BUFFSIZE', 'LL128_BUFFSIZE', 'BUFFSIZE',
                 'SIGNAL_BUFFSIZE', 'SIMPLE_BULK_BUFFSIZE', 'SIGNAL_BULK_BUFFSIZE'):
    expected_env.pop('PCCL_' + suffix, None)
  for name, default in (('algo', 'Ring'), ('proto', 'LL')):
    value = options.get(name, default)
    if value != 'Auto':
      expected_env['PCCL_' + name.upper()] = value
  if options.get('algo', 'Ring') == 'FC':
    expected_env['PCCL_DISABLE_FC_GRAPH'] = '0'
  if environment != expected_env:
    raise ValueError('Observed environment differs from public runner rules')


def _check_invocation(spec, runtime, record, *, relocate_artifacts=False):
  directory = Path(runtime['result_dir'])
  invocation = read_json(directory / 'command-invocation.json')
  if (not isinstance(invocation, dict)
      or set(invocation) != {'schema_version', 'spec_sha256', 'source_command', 'source_ids', 'argv',
                             'original_argv', 'runner_index', 'cwd', 'effective_environment', 'inherited_environment',
                             'runner_build_directory', 'mpi_options', 'mpi_exports', 'mpi_export_rule', 'bindings'}):
    raise ValueError('Invalid command invocation')
  try:
    ids, options, _ = _source_contract(spec)
    if (type(invocation['schema_version']) is not int or invocation['schema_version'] != 1
        or invocation['spec_sha256'] != digest(spec) or invocation['source_ids'] != ids
        or invocation['source_command'] != spec['execution']['source_command']
        or record.get('invocation_sha256') != digest(invocation)):
      raise ValueError('Invocation does not match source spec')
    bindings = invocation['bindings']
    if (set(bindings) != {'test_root', 'binary_dir', 'result_dir', 'mpi_options'}
        or invocation['cwd'] != bindings['test_root']):
      raise ValueError('Invalid invocation runtime bindings')
    # Execution binds every path. Summary may only relocate the artifact lookup;
    # replay argv/environment against the original bindings below, never the
    # summary host's paths. The caller still verifies manifest and command_log_ref.
    for key in bindings:
      if relocate_artifacts and key == 'result_dir':
        continue
      if key in runtime and runtime[key] != bindings[key]:
        raise ValueError('Invocation runtime binding mismatch')
    if 'execution_result_dir' in runtime and bindings['result_dir'] != runtime['execution_result_dir']:
      raise ValueError('Invocation execution result directory differs from shard runtime')
    build = safe_text(invocation['runner_build_directory'])
    if not Path(build).is_absolute():
      raise ValueError('Invalid runner build directory')
    if 'runner_build_directory' in runtime:
      if build != runtime['runner_build_directory']:
        raise ValueError('Runner build directory differs from shard runtime')
    elif 'test_root' in runtime and Path(runtime['test_root']).is_dir():
      if build != str((Path(runtime['test_root']) / 'build').resolve()):
        raise ValueError('Runner build directory resolution mismatch')
    inherited = invocation['inherited_environment']
    if safe_environment(inherited) != inherited:
      raise ValueError('Invalid recorded inherited environment')
    if 'inherited_environment' in runtime and runtime['inherited_environment'] != inherited:
      raise ValueError('Invocation inherited environment differs from shard runtime')
    original, launched = invocation['original_argv'], invocation['argv']
    python = launched[0] if options.get('direct') else original[0]
    if 'python' in runtime and runtime['python'] != python:
      raise ValueError('Invocation Python differs from shard runtime')
    expected_argv, expected_env = _prepare(spec, bindings, inherited=inherited,
                                          validate_paths=False, python=runtime.get('python', python))
    index = _runner_index(expected_argv)
    capture = Path(launched[3 if index is None else index])
    if (original != expected_argv or invocation['runner_index'] != index
        or (index is not None and type(invocation['runner_index']) is not int)
        or (index is None and (not Path(python).is_absolute()
                              or not re.fullmatch(r'python(?:3(?:\.\d+)?)?', Path(python).name)))
        or invocation['effective_environment'] != safe_environment(expected_env)
        or not capture.is_absolute() or capture.name != 'runner_capture.py'
        or launched != _capture_argv(original, index, str(capture), bindings['result_dir'], python)
        or invocation['mpi_export_rule'] != (DIRECT_EXPORT_RULE if options.get('direct') else EXPORT_RULE)):
      raise ValueError('Invocation argv/environment differs from spec expansion')
    merged = _source_mpi(spec, bindings, expected_env)
    if (invocation['mpi_options'] != merged or invocation['mpi_exports'] !=
        [key[4:] for key, _, _ in _mpi_entries(merged) if key.startswith('env:')]):
      raise ValueError('Invocation MPI merge/export mismatch')
    observed = read_json(directory / 'command-observations.json')
    if (set(observed) != {'schema_version', 'source', 'event', 'invocation_sha256', 'pid', 'original_argv',
                          'cwd', 'initial_environment', 'state', 'records'}
        or type(observed['schema_version']) is not int or observed['schema_version'] != 1
        or observed['source'] != 'observed-subprocess'
        or observed['event'] != 'subprocess.Popen' or observed['state'] != 'complete'
        or observed['pid'] != record['pid'] or observed['original_argv'] != original
        or observed['cwd'] != invocation['cwd']
        or observed['initial_environment'] != invocation['effective_environment']
        or observed['invocation_sha256'] != digest(invocation)
        or record.get('observations_sha256') != digest(observed)
        or not isinstance(observed['records'], list) or not observed['records']):
      raise ValueError('Missing, incomplete or inconsistent observed subprocess evidence')
    if options.get('direct') and len(observed['records']) != 1:
      raise ValueError('Direct capture requires exactly one observed subprocess')
    for index, entry in enumerate(observed['records']):
      if type(entry['sequence']) is not int or entry['sequence'] != index:
        raise ValueError('Invalid observed subprocess sequence')
      _observed_contract(invocation, options, entry, spec)
  except (KeyError, TypeError, IndexError, AttributeError, ZeroDivisionError) as error:
    raise ValueError('Malformed command evidence') from error


def _publish_log(output, path, env, directory, *, requires_observation):
  output.flush()
  output.seek(0)
  text = output.read()
  if requires_observation:
    try:
      observed = read_json(directory / 'command-observations.json')
      if (observed.get('state') != 'complete' or not observed['records']
          or any(entry.get('private_environment_changed') for entry in observed['records'])):
        text = '[REDACTED: incomplete or unrecorded subprocess environment]\n'
    except (ValueError, KeyError, TypeError, AttributeError):
      text = '[REDACTED: missing subprocess evidence]\n'
  # The spool is anonymous/private. Never publish inherited credentials, even
  # when a failed runner printed them before its Popen audit event.
  excluded = {value for key, value in env.items() if (not env_name(key) or SECRET.search(key)) and value
              and (len(value) >= 4 or SECRET.search(key))}
  for value in sorted(excluded, key=len, reverse=True):
    text = text.replace(value, '[REDACTED]')
  with path.open('x', encoding='utf-8', errors='surrogateescape') as stream:
    for line in text.splitlines(keepends=True):
      checked = line.replace('/private/var/', '/var/').replace('/private/tmp/', '/tmp/')
      stream.write('[REDACTED: private diagnostic]\n' if SECRET.search(line) or PRIVATE.search(checked) else line)


def execution_timeout(spec, argv):
  """Allow the runner's own timeout to fire and report before the outer kill."""
  return spec['timeout'] + (RUNNER_TIMEOUT_GRACE if _runner_index(argv) is not None else 0)


def run_command(spec, runtime, interrupted):
  """Return a final record only after successful execution and complete cleanup."""
  if interrupted:
    raise RuntimeError(f'Command cancelled by signal {interrupted[0]}')
  argv, env = _prepare(spec, runtime)
  directory = Path(runtime['result_dir'])
  directory.mkdir(parents=True, exist_ok=True)
  record_path, log_path = directory / 'command-process.json', directory / 'command.log'
  capture = 'execution' in spec
  invocation_path = directory / 'command-invocation.json'
  reserved = [record_path, log_path]
  if capture:
    reserved += [invocation_path, directory / 'command-observations.json']
  if any(os.path.lexists(path) for path in reserved):
    raise FileExistsError('Case command evidence already exists')
  process, record = None, dict(active=False, state='launch_failed', pid=None, pgid=None,
                              identity=None, return_code=None)
  if capture:
    invocation = _invocation(spec, runtime, argv, env)
    _write_record(invocation_path, invocation)
    if read_json(invocation_path) != invocation:
      raise ValueError('Invocation publication verification failed')
    record['invocation_sha256'] = digest(invocation)
    argv = invocation['argv']
  with tempfile.TemporaryFile(mode='w+', encoding='utf-8', errors='surrogateescape') as output:
    try:
      if interrupted:
        raise RuntimeError(f'Command cancelled by signal {interrupted[0]}')
      process = subprocess.Popen(argv, shell=False, cwd=runtime['test_root'], env=env,
                                 stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
      record.update(pid=process.pid, pgid=process.pid, identity=_identity(process.pid), active=True, state='running')
      _write_record(record_path, record)
      if not record['identity']:
        raise RuntimeError('Cannot record command process identity')
      timeout = execution_timeout(spec, argv)
      deadline = time.monotonic() + timeout
      while True:
        if interrupted:
          raise RuntimeError(f'Command cancelled by signal {interrupted[0]}')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
          raise subprocess.TimeoutExpired(argv, timeout)
        try:
          process.wait(timeout=min(remaining, 0.1))
          break
        except subprocess.TimeoutExpired:
          pass
      if interrupted:
        raise RuntimeError(f'Command cancelled by signal {interrupted[0]}')
      if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, argv)
      if capture:
        record['observations_sha256'] = digest(read_json(directory / 'command-observations.json'))
        _check_invocation(spec, runtime, record)
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
          try:
            _publish_log(output, log_path, env, directory, requires_observation=capture)
          except BaseException:
            if record['state'] == 'finished':
              record['state'] = 'failed'
            raise
          finally:
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
  if 'execution' in spec:
    _check_invocation(spec, runtime, record, relocate_artifacts=True)
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
