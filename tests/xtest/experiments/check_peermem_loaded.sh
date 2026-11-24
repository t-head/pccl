#!/bin/bash
# Read-only loaded-module check. No host mounts and no module loading.
set -euo pipefail
root=${1:?Output directory required}
mkdir -p "$root"
set +e
lsmod 2> "$root/lsmod.stderr" | grep alixpu_peermem > "$root/lsmod-grep.stdout" 2> "$root/grep.stderr"
statuses=("${PIPESTATUS[@]}")
set -e
python3 - "$root" "${statuses[0]}" "${statuses[1]}" <<'PY'
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
root=Path(sys.argv[1])
lsmod_code, grep_code = map(int, sys.argv[2:])
output=(root/'lsmod-grep.stdout').read_text()
actual=os.environ.get('NODE_NAME', os.uname().nodename)
expected=os.environ.get('TARGET_HOSTNAME')
complete=lsmod_code == 0 and grep_code in (0, 1) and (not expected or expected == actual)
record={'checked_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'node': actual, 'expected_node': expected, 'kernel': os.uname().release,
        'command': 'lsmod | grep alixpu_peermem', 'lsmod_exit_code': lsmod_code,
        'grep_exit_code': grep_code, 'stdout': output,
        'inspection_completed': complete,
        'module_loaded': any(line.split()[0] == 'alixpu_peermem' for line in output.splitlines() if line.split()) if complete else None}
(root/'check.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record))
raise SystemExit(0 if complete else 1)
PY
