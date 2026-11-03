#!/bin/bash
set -ex

source "$(dirname "$0")/setup_test_env.sh"
export PCCL_NET_PLUGIN_ENABLE=1

echo "######################### $(date +%Y%m%d%H%M%S) acu test ##########################"
echo $PWD
echo "pwd has:"
ls

echo "running acu test 1"
python3 ${TEST_ROOT}/run_test.py --op alltoall_perf --algo Ring --proto Simple --ranks 8 --data-type double --data-size 32KB --iters 1 --threads 1 --nprocs 1 --profile acu

echo "######################### $(date +%Y%m%d%H%M%S) acu test finish ##########################"
