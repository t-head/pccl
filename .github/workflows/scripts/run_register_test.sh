#!/bin/bash
set -ex

source "$(dirname "$0")/setup_test_env.sh"

echo "######################### $(date +%Y%m%d%H%M%S) register test ##########################"
echo $PWD
echo "pwd has:"
ls

mpirun --allow-run-as-root -np 2 scatter_perf -b 32 -e 1G -f 2 -R 1 -d half
mpirun --allow-run-as-root -np 2 gather_perf -b 32 -e 1G -f 2 -R 1 -d half

python3 ${TEST_ROOT}/run_test.py --op alltoall_perf --algo Auto --proto Auto --data-type float --data-size 32MB --iters 10 --ranks 2 --nproc 1 --threads 2 --auto-detect --register 1
python3 ${TEST_ROOT}/run_test.py --op alltoall_perf --algo Auto --proto Auto --data-type float --data-size 32MB --iters 10 --ranks 2 --nproc 2 --threads 1 --auto-detect --register 1
python3 ${TEST_ROOT}/run_test.py --op all_gather_perf --algo Auto --proto Auto --data-type float --data-size 32MB --iters 10 --ranks 2 --nproc 1 --threads 2 --auto-detect --register 1
python3 ${TEST_ROOT}/run_test.py --op all_gather_perf --algo Auto --proto Auto --data-type float --data-size 32MB --iters 10 --ranks 2 --nproc 2 --threads 1 --auto-detect --register 1
python3 ${TEST_ROOT}/run_test.py --op all_reduce_perf --algo Auto --proto Auto --data-type float --data-size 32MB --iters 10 --ranks 2 --nproc 1 --threads 2 --auto-detect --register 1
python3 ${TEST_ROOT}/run_test.py --op all_reduce_perf --algo Auto --proto Auto --data-type float --data-size 32MB --iters 10 --ranks 2 --nproc 2 --threads 1 --auto-detect --register 1
python3 ${TEST_ROOT}/run_test.py --op hypercube_perf --algo Ring --proto Simple --data-type int8 --data-size 32MB --iters 10 --ranks 2 --nproc 1 --threads 2 --auto-detect --register 1
python3 ${TEST_ROOT}/run_test.py --op hypercube_perf --algo Ring --proto Simple --data-type int8 --data-size 32MB --iters 10 --ranks 2 --nproc 2 --threads 1 --auto-detect --register 1
python3 ${TEST_ROOT}/run_test.py --op gather_perf --algo Ring --proto Simple --data-type int8 --data-size 32MB --iters 10 --ranks 2 --nproc 1 --threads 2 --auto-detect --register 1
python3 ${TEST_ROOT}/run_test.py --op gather_perf --algo Ring --proto Simple --data-type int8 --data-size 32MB --iters 10 --ranks 2 --nproc 2 --threads 1 --auto-detect --register 1
python3 ${TEST_ROOT}/run_test.py --op scatter_perf --algo Ring --proto Simple --data-type int8 --data-size 32MB --iters 10 --ranks 2 --nproc 1 --threads 2 --auto-detect --register 1
python3 ${TEST_ROOT}/run_test.py --op scatter_perf --algo Ring --proto Simple --data-type int8 --data-size 32MB --iters 10 --ranks 2 --nproc 2 --threads 1 --auto-detect --register 1
python3 ${TEST_ROOT}/run_test.py --op sendrecv_perf --algo Ring --proto Simple --data-type int8 --data-size 32MB --iters 10 --ranks 2 --nproc 1 --threads 2 --auto-detect --register 1
python3 ${TEST_ROOT}/run_test.py --op sendrecv_perf --algo Ring --proto Simple --data-type int8 --data-size 32MB --iters 10 --ranks 2 --nproc 2 --threads 1 --auto-detect --register 1

echo "######################### $(date +%Y%m%d%H%M%S) register test finish ##########################"
