# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Child-local CPython audit evidence, not syscall tracing or remote-rank evidence.

No subprocess function is replaced. Events precede the OS spawn and do not prove
that exec succeeded. A trusted runner's exit status supplies the success signal.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tempfile

ENV_KEY = re.compile(r'(?:(?:PCCL|NCCL|HGGC|OMPI|UCX)_[A-Za-z0-9_]+|UMD_PLATFORM_TYPE)')
SECRET = re.compile(r'secret|password|passwd|credential|token|private[_-]?key|api[_-]?key|authorization', re.I)
PRIVATE = re.compile(
  r'://|[\w.+-]+@[\w.-]+|alibaba-inc|comm-internal|\binternal\b|'
  r'-----BEGIN|\bBearer\b|\bAKIA[A-Z0-9]{16}\b|\b(?:gh[pousr]_|github_pat_|sk-)[A-Za-z0-9_]+|'
  r'\beyJ[A-Za-z0-9_-]+\.|(?:^|/)(?:\.ssh|\.aws|\.config/gcloud)(?:/|$)|'
  r'(?:^|[/_.-])private(?:[/_.-]|$)|work_root|setup_script|UploadResult|test/(?:cases|plans|suites)/', re.I)
SAFE = re.compile(r'[A-Za-z0-9_./,:=+^@%{}\- ]*')
GENERATED_DIRECTORY = re.compile(
  r'(?<=/)(?:[A-Za-z0-9_-]+-[0-9a-f]{20}|suite-lifecycle/[0-9a-f]{64}|'
  r'(?:standard|notsm)-[1-9][0-9]*n-[1-9][0-9]*c-(?:functional|performance)-'
  r'[A-Za-z0-9_-]+-[0-9a-f]{12}-[0-9]{3,})(?=/|$)')


def digest(value):
  return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def safe_text(value):
  # Do not include rejected input in exception messages or public records.
  if not isinstance(value, str):
    raise ValueError('Unsafe evidence type')
  # macOS resolves ordinary temporary directories through /private; it is not
  # a private-project marker. Public /workspace paths are also allowed.
  checked = value.replace('/private/var/', '/var/').replace('/private/tmp/', '/tmp/')
  # Structured Case/batch directory names are not opaque credential strings.
  opaque = GENERATED_DIRECTORY.sub('generated', value) if value.startswith('/') else value
  if (not SAFE.fullmatch(value) or PRIVATE.search(checked) or SECRET.search(value)
      or re.search(r'(?<![A-Za-z0-9])[A-Za-z0-9_+-]{40,}(?![A-Za-z0-9])', opaque)):
    raise ValueError('Unsafe or private evidence value')
  return value


def env_name(name):
  # DEBUG_FILE is observable, but only a matching literal source prefix may
  # override it (command.py); it is not a general public env override.
  return isinstance(name, str) and (name in ('PATH', 'LD_LIBRARY_PATH', 'DEBUG_FILE') or ENV_KEY.fullmatch(name))


def safe_environment(env):
  result = {}
  for key, value in sorted(env.items()):
    if env_name(key):
      if SECRET.search(key):
        raise ValueError('Sensitive test environment key')
      result[key] = safe_text(value)
  return result


def read_json(path):
  def pairs(items):
    result = {}
    for key, value in items:
      if key in result:
        raise ValueError('Duplicate evidence key')
      result[key] = value
    return result

  def invalid(value):
    raise ValueError('Invalid evidence constant')

  try:
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)
  except (OSError, UnicodeError, TypeError) as error:
    raise ValueError('Missing or unreadable command evidence') from error


def publish(path, value):
  with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
    temporary = Path(stream.name)
    try:
      json.dump(value, stream, sort_keys=True)
      stream.flush()
      os.fsync(stream.fileno())
      os.fchmod(stream.fileno(), 0o644)
      os.replace(temporary, path)
    finally:
      temporary.unlink(missing_ok=True)


def main():
  invocation_path = Path(sys.argv[1])
  invocation = read_json(invocation_path)
  original = invocation['original_argv']
  index = invocation['runner_index']
  if sys.argv[2:] != ['--'] + original[index:]:
    raise ValueError('Capture launcher argv mismatch')
  path = invocation_path.with_name('command-observations.json')
  if os.path.lexists(path):
    raise ValueError('Capture evidence already exists')
  owner = os.getpid()
  private_environment = {key: value for key, value in os.environ.items() if not env_name(key)}
  evidence = dict(schema_version=1, source='observed-subprocess', event='subprocess.Popen',
                  invocation_sha256=digest(invocation), pid=owner, original_argv=original,
                  cwd=os.getcwd(), initial_environment=safe_environment(os.environ),
                  state='running', records=[])
  publish(path, evidence)

  def audit(event, args):
    if event != 'subprocess.Popen' or os.getpid() != owner:
      return
    try:
      executable, argv, cwd, env = args
      if not isinstance(argv, (list, tuple)) or not argv:
        raise ValueError('Non-vector subprocess invocation')
      effective = os.environ if env is None else env
      private_changed = any(not env_name(key) and private_environment.get(key) != value
                            for key, value in effective.items())
      entry = dict(sequence=len(evidence['records']), source='observed-subprocess',
                   executable=safe_text(os.fsdecode(executable)), argv=[safe_text(x) for x in argv],
                   cwd=safe_text(os.path.abspath(cwd) if cwd is not None else os.getcwd()),
                   effective_environment=safe_environment(effective), private_environment_changed=private_changed,
                   environment_source='inherited' if env is None else 'explicit')
      evidence['records'].append(entry)
      publish(path, evidence)
    except BaseException:
      evidence['state'] = 'invalid'
      # Never serialize exception text, argv or rejected environment values.
      publish(path, evidence)
      raise

  sys.addaudithook(audit)
  try:
    if index is None:
      # This real child launch emits the same CPython Popen audit event as a
      # runner launch. Never synthesize observation records from the spec.
      subprocess.run(original, check=True)
    else:
      # Match ordinary script startup, including sibling imports and sys.argv[0].
      sys.argv = original[index:]
      # CPython uses the invoked script path's directory, not the symlink
      # target's directory. Preserve that behavior so sibling test fixtures
      # and dependencies are resolved exactly as an ordinary invocation.
      sys.path[0] = str(Path(original[index]).absolute().parent)
      runpy.run_path(original[index], run_name='__main__')
  finally:
    if evidence['state'] != 'invalid':
      evidence['state'] = 'complete'
    publish(path, evidence)


if __name__ == '__main__':
  main()
