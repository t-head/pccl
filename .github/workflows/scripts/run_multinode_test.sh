#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail

transport=${1:-}
case "$transport" in
  mnnvl|socket|gdr|non_gdr) ;;
  *) echo "Specify a multinode transport: mnnvl, socket, gdr or non_gdr" >&2; exit 1 ;;
esac

# A missing scheduler environment must not silently fall back to single-node execution.
if [[ ! ${NNODES:-} =~ ^[1-9][0-9]*$ || ! ${NPROC_PER_NODE:-} =~ ^[1-9][0-9]*$ ||
      ! ${NODE_RANK:-} =~ ^(0|[1-9][0-9]*)$ || ! ${MASTER_PORT:-} =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid or missing PPU multinode scheduler environment" >&2
  exit 1
fi
if (( NNODES < 2 || NODE_RANK >= NNODES || MASTER_PORT >= 65535 )) ||
   [[ -z ${MASTER_ADDR:-} || ${WORLD_SIZE:-} != "$NNODES" || ${RANK:-} != "$NODE_RANK" ]]; then
  echo "Inconsistent node count, rank or rendezvous configuration" >&2
  exit 1
fi

source "$(dirname "$0")/setup_test_env.sh"

# MPI binaries ignore WORLD_SIZE/RANK, so select the isolated non-MPI build.
k8s_bin="${TEST_ROOT}/pccl-tests/build-k8s"
if [[ ! -x "$k8s_bin/all_reduce_perf" ]]; then
  echo "Missing K8s test artifacts: $k8s_bin" >&2
  exit 1
fi
export PATH="$k8s_bin:$PATH"
# The benchmark's hostname buffer cannot hold the scheduler's long service FQDN.
master_ipv4=$(python3 - <<'PY'
import os
import subprocess
import sys

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
print(address)
PY
)
export PCCL_COMM_ID="${master_ipv4}:${MASTER_PORT}"
export NCCL_COMM_ID="$PCCL_COMM_ID"
export NCCL_BARRIER_PORT=$((MASTER_PORT + 1))
runner_transport="$transport"
if [[ "$transport" == mnnvl ]]; then
  unset PCCL_NET NCCL_NET
  export PCCL_IB_DISABLE=0 NCCL_IB_DISABLE=0
  export PCCL_MNNVL_ENABLE=1 NCCL_MNNVL_ENABLE=1
  export PCCL_MNNVL_UUID=666 NCCL_MNNVL_UUID=666
  export PCCL_MNNVL_CLIQUE_ID=0 NCCL_MNNVL_CLIQUE_ID=0
  export PCCL_MNNVL_FAKE_ADVANCED_FEATURE=1 NCCL_MNNVL_FAKE_ADVANCED_FEATURE=1
  # The runner has no MNNVL option; gdr mode keeps IB enabled and preserves MNNVL settings.
  runner_transport=gdr
else
  export PCCL_MNNVL_ENABLE=0 NCCL_MNNVL_ENABLE=0
  export PCCL_MNNVL_FAKE_ADVANCED_FEATURE=0 NCCL_MNNVL_FAKE_ADVANCED_FEATURE=0
  if [[ "$transport" == socket ]]; then
    export PCCL_NET=Socket NCCL_NET=Socket
    export PCCL_IB_DISABLE=1 NCCL_IB_DISABLE=1
  else
    export PCCL_NET=IB NCCL_NET=IB
    export PCCL_IB_DISABLE=0 NCCL_IB_DISABLE=0
  fi
fi

echo "Nightly $transport: nodes=$NNODES ranks=$((NNODES * NPROC_PER_NODE))"

# Use one process per node and one thread per PPU; avoid unsupported RDMA protocols.
common_args=(
  --launch-mode k8s --cross-node "$runner_transport"
  --nnodes "$NNODES" --ranks "$((NNODES * NPROC_PER_NODE))"
  --nprocs 1 --threads "$NPROC_PER_NODE"
  --data-type int8 half float double --data-size 32B 32KB 8MB 256MB
  --iters 100 --warmup 5 --timeout 1200 --stop-once-failed --auto-detect
)
python3 "${TEST_ROOT}/run_test.py" "${common_args[@]}" \
  --op all_reduce_perf all_gather_perf reduce_scatter_perf broadcast_perf reduce_perf \
       alltoall_perf alltoallv_perf gather_perf scatter_perf sendrecv_perf hypercube_perf \
  --algo Ring --proto Simple

# Point-to-point operators use Simple; limit LL coverage to supported collectives.
python3 "${TEST_ROOT}/run_test.py" "${common_args[@]}" \
  --op all_reduce_perf all_gather_perf reduce_scatter_perf broadcast_perf reduce_perf \
  --algo Ring --proto LL

python3 "${TEST_ROOT}/run_test.py" "${common_args[@]}" \
  --op all_reduce_perf --algo Tree --proto Simple LL
