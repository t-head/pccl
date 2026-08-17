#!/bin/bash
set -ex
pip install filelock==3.20.0
export NCCL_DEBUG=INFO
source ./PPU_SDK/envsetup.sh
export LD_LIBRARY_PATH=$TEST_ROOT/lib:$LD_LIBRARY_PATH
echo $TEST_ROOT
echo $LD_LIBRARY_PATH
echo $PATH

cd $TEST_ROOT/comm-internal-testbox
echo "######################### $(date +%Y%m%d%H%M%S) copyengine test ##########################"
echo $PWD
echo "pwd has:"
ls

mpirun --allow-run-as-root -np 2 -x PCCL_CE_ENABLE_AR=1 ./tests/pccl-tests/build/all_reduce_perf -b 5111004 -e 1023M -f 2 -x 2 -R 2
mpirun --allow-run-as-root -np 2 -x PCCL_OKI_ENABLE=1 -x PCCL_CE_ENABLE_AR=1 -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./tests/pccl-tests/build/all_reduce_perf -b 5111004 -e 1023M -f 2 -x 2 -R 2

echo "######################### $(date +%Y%m%d%H%M%S) copyengine test finish ##########################"
