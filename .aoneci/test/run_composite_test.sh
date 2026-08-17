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
echo "######################### $(date +%Y%m%d%H%M%S) composite test ##########################"
echo $PWD
echo "pwd has:"
ls
echo "running composite test 1"
./tests/composite_test/build/composite_test ndevs=2 ops='allgather@32F,33554432' measure --delay=1 --niters=20 --register=2
echo "running composite test 2"
./tests/composite_test/build/composite_test ndevs=2 ops='allreduce@32F,33554432' measure --delay=1 --niters=20 --register=2
echo "running composite test 3"
./tests/composite_test/build/composite_test ndevs=2 ops='reduce_scatter@32F,33554432' measure --delay=1 --niters=20 --register=2
echo "running composite test 4"
NCCL_CTA_POLICY=0 ./tests/composite_test/build/composite_test ndevs=2 ops='allgather@32F,33554432' measure --delay=1 --niters=20 --register=2

echo "######################### $(date +%Y%m%d%H%M%S) composite test finish ##########################"
