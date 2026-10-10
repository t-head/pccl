#!/bin/bash
set -ex

source "$(dirname "$0")/setup_test_env.sh"

export NCCL_P2P_USE_CUDA_MEMCPY=1

echo "######################### $(date +%Y%m%d%H%M%S) memcpy test ##########################"
echo $PWD
echo "pwd has:"
ls

python3 ${TEST_ROOT}/run_test.py --op alltoall_perf --algo Auto --proto Auto --data-type float --data-size 32MB --iters 10 --ranks 2 --nproc 1 --threads 2 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op alltoall_perf --algo Auto --proto Auto --data-type float --data-size 32MB --iters 10 --ranks 2 --nproc 2 --threads 1 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op all_gather_perf --algo Auto --proto Auto --data-type float --data-size 32MB --iters 10 --ranks 2 --nproc 1 --threads 2 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op all_gather_perf --algo Auto --proto Auto --data-type float --data-size 32MB --iters 10 --ranks 2 --nproc 2 --threads 1 --auto-detect

echo "######################### $(date +%Y%m%d%H%M%S) memcpy test finish ##########################"
