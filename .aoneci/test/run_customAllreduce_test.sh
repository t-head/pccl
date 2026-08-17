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
echo "######################### $(date +%Y%m%d%H%M%S) customAllreduce test ##########################"
echo $PWD
echo "pwd has:"
ls

mpirun --allow-run-as-root -mca orte_base_help_aggregate 0 --mca btl ^openib -np 2 ./tests/pccl-tests/build/all_reduce_perf -b 33K -e 33K -f 2 -d half
mpirun --allow-run-as-root -mca orte_base_help_aggregate 0 --mca btl ^openib -np 2 ./tests/pccl-tests/build/all_reduce_perf -b 301K -e 301K -f 2 -d half
echo "######################### $(date +%Y%m%d%H%M%S) customAllreduce test finish ##########################"
