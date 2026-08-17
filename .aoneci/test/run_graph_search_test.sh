#!/bin/bash
set -ex
pip install filelock==3.20.0
export NCCL_DEBUG=INFO
source ./PPU_SDK/envsetup.sh
export LD_LIBRARY_PATH=$TEST_ROOT/lib:$LD_LIBRARY_PATH
echo $TEST_ROOT
echo $LD_LIBRARY_PATH
echo $PATH

cd $TEST_ROOT/pccl/tests/build
echo "######################### $(date +%Y%m%d%H%M%S) acu test ##########################"
echo $PWD
echo "pwd has:"
ls

./graphSearchTest -r ../topo_and_reference_graph/ --gtest_filter='graphSearchTestWithoutPXN*'
./graphSearchTest -r ../topo_and_reference_graph/ --gtest_filter='graphSearchTestWithoutPXN*'

echo "######################### $(date +%Y%m%d%H%M%S) acu test finish ##########################"
