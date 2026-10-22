#!/bin/bash
set -ex

source "$(dirname "$0")/setup_test_env.sh"

export NCCL_COMM_BLOCKING=0

echo "######################### $(date +%Y%m%d%H%M%S) nonblocking test ##########################"
echo $PWD
echo "pwd has:"
ls

NCCL_COMM_BLOCKING=0 all_reduce_perf -b 4 -e 1G -f 2 -g 8 -R 0 -B 0
NCCL_COMM_BLOCKING=0 alltoall_perf -b 4 -e 1G -f 2 -g 8 -R 0 -B 0

python3 ${TEST_ROOT}/run_test.py --op alltoall_perf --algo Ring --proto Auto --ranks 8 --data-type int8 --data-size 32MB --iters 10 --nproc 1 --threads 8 --nnodes 1 --auto-detect --non-blocking
python3 ${TEST_ROOT}/run_test.py --op all_reduce_perf --algo Ring --proto LL --ranks 8 --data-type half --data-size 32MB --iters 10 --nproc 1 --threads 8 --nnodes 1 --auto-detect --non-blocking
python3 ${TEST_ROOT}/run_test.py --op all_reduce_perf --algo Ring --proto LL --ranks 8 --data-type float --data-size 32MB --iters 10 --nproc 1 --threads 8 --nnodes 1 --auto-detect --non-blocking

echo "######################### $(date +%Y%m%d%H%M%S) nonblocking test finish ##########################"
