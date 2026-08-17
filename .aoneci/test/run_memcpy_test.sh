#!/bin/bash
set -ex
pip install filelock==3.20.0
export NCCL_DEBUG=INFO
export NCCL_P2P_USE_CUDA_MEMCPY=1
source ./PPU_SDK/envsetup.sh
export LD_LIBRARY_PATH=$TEST_ROOT/lib:$LD_LIBRARY_PATH
echo $TEST_ROOT
echo $LD_LIBRARY_PATH
echo $PATH

cd $TEST_ROOT/comm-internal-testbox/scripts/pccl-tests
echo "######################### $(date +%Y%m%d%H%M%S) memcpy test ##########################"
echo $PWD
echo "pwd has:"
ls

python run_test.py --op alltoall_perf --algo Auto --proto Auto --data-type float --data-size 32MB --iters 10 --ranks 2 --nproc 1 --threads 2 --auto-detect
python run_test.py --op alltoall_perf --algo Auto --proto Auto --data-type float --data-size 32MB --iters 10 --ranks 2 --nproc 2 --threads 1 --auto-detect
python run_test.py --op all_gather_perf --algo Auto --proto Auto --data-type float --data-size 32MB --iters 10 --ranks 2 --nproc 1 --threads 2 --auto-detect
python run_test.py --op all_gather_perf --algo Auto --proto Auto --data-type float --data-size 32MB --iters 10 --ranks 2 --nproc 2 --threads 1 --auto-detect

echo "######################### $(date +%Y%m%d%H%M%S) memcpy test finish ##########################"
