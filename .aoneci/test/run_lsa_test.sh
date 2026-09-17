#!/bin/bash
set -ex
export NCCL_DEBUG=INFO
source ./PPU_SDK/envsetup.sh
export LD_LIBRARY_PATH=$TEST_ROOT/lib:$LD_LIBRARY_PATH
echo $TEST_ROOT
echo $LD_LIBRARY_PATH
echo $PATH

cd $TEST_ROOT/comm-internal-testbox/tests/pccl-tests/build/
echo "######################### $(date +%Y%m%d%H%M%S) lsa test ##########################"
echo $PWD
echo "pwd has:"
ls

./all_reduce_perf -b 8 -e 2G -f 2 --elapsed_type 1 -R 2 -t 2 -g 1 -D 0
./alltoall_perf -b 8 -e 2G -f 2 --elapsed_type 1 -R 2 -t 2 -g 1 -D 0
./all_reduce_perf -b 8 -e 2G -f 2 --elapsed_type 1 -R 2 -t 2 -g 1 -D 1 -V 16
./all_reduce_perf -b 8 -e 2G -f 2 --elapsed_type 1 -R 2 -t 2 -g 1 -D 2 -V 16
./alltoall_perf -b 8 -e 2G -f 2 --elapsed_type 1 -R 2 -t 2 -g 1 -D 1 -V 16
./alltoall_perf -b 8 -e 2G -f 2 --elapsed_type 1 -R 2 -t 2 -g 1 -D 2 -V 16