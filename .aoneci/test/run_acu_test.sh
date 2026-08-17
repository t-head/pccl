#!/bin/bash
set -ex
pip install filelock==3.20.0
export NCCL_DEBUG=INFO
source ./PPU_SDK/envsetup.sh
export LD_LIBRARY_PATH=$TEST_ROOT/lib:$LD_LIBRARY_PATH
echo $TEST_ROOT
echo $LD_LIBRARY_PATH
echo $PATH

cd $TEST_ROOT/comm-internal-testbox/scripts/pccl-tests/
echo "######################### $(date +%Y%m%d%H%M%S) acu test ##########################"
echo $PWD
echo "pwd has:"
ls
echo "running acu test 1"
python run_test.py --op alltoall_perf --algo Ring --proto Simple --ranks 2 --data-type double --data-size 32KB --iters 1 --threads 1 --nprocs 1 --profile acu

echo "######################### $(date +%Y%m%d%H%M%S) acu test finish ##########################"
