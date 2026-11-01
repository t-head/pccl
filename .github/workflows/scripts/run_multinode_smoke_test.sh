#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail

if [[ $# != 1 || ( $1 != mnnvl && $1 != rdma ) ]]; then
  echo "Specify a two-node smoke transport: mnnvl or rdma" >&2
  exit 1
fi
export PCCL_SMOKE_TRANSPORT="$1"

if [[ ${NNODES:-} != 2 || ${NPROC_PER_NODE:-} != 8 || ${WORLD_SIZE:-} != 2 ||
      ! ${NODE_RANK:-} =~ ^[01]$ || ${RANK:-} != "$NODE_RANK" ||
      -z ${MASTER_ADDR:-} || ! ${MASTER_PORT:-} =~ ^[1-9][0-9]*$ ]]; then
  echo "Two-node smoke requires two nodes, eight PPUs per node and consistent scheduler ranks" >&2
  exit 1
fi
if (( MASTER_PORT >= 65535 )); then
  echo "Invalid rendezvous port" >&2
  exit 1
fi

source "$(dirname "$0")/setup_test_env.sh"
export NCCL_BARRIER_PORT=$((MASTER_PORT + 1))
export PCCL_BLOCKING_BOOTSTRAP=1
export PCCL_SOCKET_IFNAME='^flannel,docker,lo'
export PCCL_ALGO=Ring NCCL_ALGO=Ring
export PCCL_PROTO=Simple NCCL_PROTO=Simple
export PCCL_DEBUG=INFO NCCL_DEBUG=INFO
export PCCL_DEBUG_SUBSYS=INIT,P2P,NET NCCL_DEBUG_SUBSYS=INIT,P2P,NET
export PCCL_DEBUG_DEV=-1 NCCL_DEBUG_DEV=-1
unset PCCL_DEBUG_FILE NCCL_DEBUG_FILE

if [[ "$PCCL_SMOKE_TRANSPORT" == mnnvl ]]; then
  unset PCCL_NET NCCL_NET
  export PCCL_IB_DISABLE=1 NCCL_IB_DISABLE=1
  export PCCL_MNNVL_ENABLE=1 NCCL_MNNVL_ENABLE=1
  export PCCL_MNNVL_UUID=666 NCCL_MNNVL_UUID=666
  export PCCL_MNNVL_CLIQUE_ID=0 NCCL_MNNVL_CLIQUE_ID=0
  export PCCL_MNNVL_FAKE_ADVANCED_FEATURE=1 NCCL_MNNVL_FAKE_ADVANCED_FEATURE=1
else
  export PCCL_NET=IB NCCL_NET=IB
  export PCCL_IB_DISABLE=0 NCCL_IB_DISABLE=0
  export PCCL_MNNVL_ENABLE=0 NCCL_MNNVL_ENABLE=0
  export PCCL_MNNVL_FAKE_ADVANCED_FEATURE=0 NCCL_MNNVL_FAKE_ADVANCED_FEATURE=0
  export PCCL_NET_GDR_LEVEL=SYS NCCL_NET_GDR_LEVEL=SYS
  export PCCL_IB_QPS_PER_CONNECTION=8 NCCL_IB_QPS_PER_CONNECTION=8
fi

python3 - <<'PY'
import os
from pathlib import Path
import re
import subprocess
import sys

binary = Path(os.environ['TEST_ROOT']) / 'pccl-tests/build-k8s/all_reduce_perf'
if not os.access(binary, os.X_OK):
    raise SystemExit(f'Missing non-MPI AllReduce binary: {binary}')
# The benchmark's hostname buffer cannot hold the scheduler's long service FQDN.
resolve = '''
import socket, sys, time
while True:
    try:
        print(socket.gethostbyname(sys.argv[1]))
        break
    except socket.gaierror:
        time.sleep(1)
'''
try:
    address = subprocess.run([sys.executable, '-c', resolve, os.environ['MASTER_ADDR']],
                             capture_output=True, text=True, timeout=60, check=True).stdout.strip()
except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
    raise SystemExit('Could not resolve worker-0 IPv4 address within 60 seconds')
os.environ['PCCL_COMM_ID'] = os.environ['NCCL_COMM_ID'] = f"{address}:{os.environ['MASTER_PORT']}"
command = [str(binary), '-t', '8', '-g', '1', '-b', '8M', '-e', '8M',
           '-d', 'float', '-o', 'sum', '-n', '20', '-w', '5', '-c', '1', '-k', '2',
           '-N', '1', '--init_only', '0', '--report_type', '0']
print(f"Node {os.environ['NODE_RANK']}/2: {' '.join(command)}", flush=True)
try:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=600, check=False)
except subprocess.TimeoutExpired as error:
    output = error.stdout or b''
    print(output.decode(errors='replace') if isinstance(output, bytes) else output, flush=True)
    raise SystemExit('AllReduce exceeded the 600-second deadline')
print(result.stdout, end='', flush=True)
if result.returncode:
    raise SystemExit(result.returncode if result.returncode > 0 else 1)

transport = os.environ['PCCL_SMOKE_TRANSPORT']
if os.environ['NODE_RANK'] == '0':
    if not re.search(r'^#\s*Out of bounds values\s*:\s*0\s+OK\b', result.stdout, re.M):
        raise SystemExit('Missing successful data-validation summary')
    rows = [line.split() for line in result.stdout.splitlines()
            if re.match(r'^\s*8388608\s+2097152\s+float\s+sum\s', line)]
    if len(rows) != 1 or len(rows[0]) != 13:
        raise SystemExit('Missing or unexpected 8 MiB AllReduce result')
    # Table error counts aggregate all processes; the footer is local in non-MPI mode.
    if rows[0][8] != '0' or rows[0][12] != '0':
        raise SystemExit('AllReduce validation failed or was disabled for a buffer layout')
print(f"AllReduce {transport} smoke passed on node {os.environ['NODE_RANK']}", flush=True)
PY
