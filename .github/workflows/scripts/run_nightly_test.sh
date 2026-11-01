#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail

source "$(dirname "$0")/setup_test_env.sh"

# Avoid --all because upstream includes node counts beyond this allocation.
python3 "${TEST_ROOT}/run_test.py" \
  --op all_reduce_perf all_gather_perf reduce_scatter_perf broadcast_perf reduce_perf \
       alltoall_perf alltoallv_perf gather_perf scatter_perf sendrecv_perf hypercube_perf \
  --algo Auto --proto Auto \
  --ranks 8 --nprocs 8 --threads 1 --nnodes 1 \
  --data-type int8 half float double \
  --data-size 32B 32KB 8MB 256MB 1GB \
  --iters 100 --warmup 5 --timeout 1200 --stop-once-failed --auto-detect
