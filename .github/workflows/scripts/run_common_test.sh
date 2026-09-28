#!/bin/bash
set -ex

source "$(dirname "$0")/setup_test_env.sh"

echo "######################### $(date +%Y%m%d%H%M%S) common test ##########################"
echo $PWD
echo "pwd has:"
ls

python3 ${TEST_ROOT}/run_test.py --op alltoall_perf --algo Ring --proto Simple --data-type int8 --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op all_gather_perf --algo Ring --proto LL --data-type int8 --data-size 32MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op all_gather_perf --algo Ring --proto LL128 --data-type int8 --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op all_gather_perf --algo Ring --proto Simple --data-type int8 --data-size 512MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op all_reduce_perf --algo Ring --proto LL --data-type float --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op all_reduce_perf --algo Ring --proto LL128 --data-type half --data-size 1.2GB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op all_reduce_perf --algo Ring --proto Simple --data-type float --data-size 512MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op all_reduce_perf --algo Ring --proto SimpleBulk --data-type float --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op all_reduce_perf --algo Ring --proto SimpleBulk --data-type float --data-size 1.2GB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op all_reduce_perf --algo Ring --proto Signal --data-type float --data-size 512MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op all_reduce_perf --algo Ring --proto SignalBulk --data-type half --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op all_reduce_perf --algo Tree --proto LL --data-type half --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op all_reduce_perf --algo Tree --proto LL128 --data-type half --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op all_reduce_perf --algo Tree --proto Simple --data-type half --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op all_reduce_perf --algo Tree --proto SimpleBulk --data-type half --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op all_reduce_perf --algo Tree --proto SimpleBulk --data-type float --data-size 1.2GB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op all_reduce_perf --algo Tree --proto Signal --data-type float --data-size 512MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op all_reduce_perf --algo Tree --proto SignalBulk --data-type half --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op broadcast_perf --algo Ring --proto LL --data-type int8 --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op broadcast_perf --algo Ring --proto LL128 --data-type int8 --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op broadcast_perf --algo Ring --proto Simple --data-type int8 --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op broadcast_perf --algo Ring --proto Simple --data-type int8 --data-size 1.2GB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op reduce_perf --algo Ring --proto LL --data-type half --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op reduce_perf --algo Ring --proto LL128 --data-type half --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op reduce_perf --algo Ring --proto Simple --data-type half --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op reduce_scatter_perf --algo Ring --proto LL --data-type half --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op reduce_scatter_perf --algo Ring --proto LL128 --data-type float --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python3 ${TEST_ROOT}/run_test.py --op reduce_scatter_perf --algo Ring --proto Simple --data-type half --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect

echo "######################### $(date +%Y%m%d%H%M%S) common test finish ##########################"
