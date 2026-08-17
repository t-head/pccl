#!/bin/bash
set -ex
pip install filelock==3.20.0
export NCCL_DEBUG=INFO
source ./PPU_SDK/envsetup.sh
export LD_LIBRARY_PATH=$TEST_ROOT/lib:$LD_LIBRARY_PATH
echo $TEST_ROOT
echo $LD_LIBRARY_PATH
echo $PATH

echo "######################### $(date +%Y%m%d%H%M%S) commshrink test ##########################"
echo $PWD
echo "pwd has:"
ls
cd $TEST_ROOT/comm-internal-testbox
echo "running commshrink test 1"
mpirun --allow-run-as-root -np 4 ./tests/unit_test/build/pcclCommShrinkTest -r 1 -r 2 -b 0 -s 1 -f 0
echo "running commshrink test 2"
mpirun --allow-run-as-root -np 4 ./tests/unit_test/build/pcclCommShrinkTest -r 1 -b 1 -s 0 -f 0
echo "running commshrink test 3"
mpirun --allow-run-as-root -np 4 ./tests/unit_test/build/pcclCommShrinkTest -r 1 -b 1 -s 0 -f 1

echo "######################### $(date +%Y%m%d%H%M%S) commshrink test finish ##########################"
