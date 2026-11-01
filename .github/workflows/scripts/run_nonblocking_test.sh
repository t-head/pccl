#!/bin/bash
set -ex

source "$(dirname "$0")/setup_test_env.sh"

export NCCL_COMM_BLOCKING=0

echo "######################### $(date +%Y%m%d%H%M%S) nonblocking test ##########################"
echo $PWD
echo "pwd has:"
ls

python3 - <<'PY'
import re
import shlex
import subprocess

for binary in ('all_reduce_perf', 'alltoall_perf'):
    command = [binary, '-b', '4', '-e', '1G', '-f', '2', '-g', '2', '-R', '0',
               '--init_blocking', '0', '-c', '1']
    print(shlex.join(command), flush=True)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, check=False)
    print(result.stdout, end='', flush=True)
    if result.returncode:
        raise SystemExit(result.returncode if result.returncode > 0 else 1)
    # Unsupported options can print usage and return zero without running a test.
    if not re.search(r'^#\s*Out of bounds values\s*:\s*0\s+OK\b', result.stdout, re.M):
        raise SystemExit(f'{binary}: Missing successful data-validation summary')
PY

python3 ${TEST_ROOT}/run_test.py --op alltoall_perf --algo Ring --proto Auto --ranks 2 --data-type int8 --data-size 32MB --iters 10 --nproc 1 --threads 2 --nnodes 1 --auto-detect --non-blocking
python3 ${TEST_ROOT}/run_test.py --op all_reduce_perf --algo Ring --proto LL --ranks 2 --data-type half --data-size 32MB --iters 10 --nproc 1 --threads 2 --nnodes 1 --auto-detect --non-blocking
python3 ${TEST_ROOT}/run_test.py --op all_reduce_perf --algo Ring --proto LL --ranks 2 --data-type float --data-size 32MB --iters 10 --nproc 1 --threads 2 --nnodes 1 --auto-detect --non-blocking

echo "######################### $(date +%Y%m%d%H%M%S) nonblocking test finish ##########################"
