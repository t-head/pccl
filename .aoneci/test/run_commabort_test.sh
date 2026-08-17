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
echo "######################### $(date +%Y%m%d%H%M%S) commabort test ##########################"
echo $PWD
echo "pwd has:"
ls

mpirun -np 2 -npernode 2 -mca btl_tcp_if_include bond0 -mca orte_base_help_aggregate 0 --mca btl ^openib --allow-run-as-root ./tests/unit_test//build/pcclCommAbortTest -b 1g -a 0 -n 2 -c 0 -t 1 -g 1
mpirun -np 2 -npernode 2 -mca btl_tcp_if_include bond0 -mca orte_base_help_aggregate 0 --mca btl ^openib --allow-run-as-root ./tests/unit_test//build/pcclCommAbortTest -b 1g -a 0 -n 2 -c 1 -t 1 -g 1
mpirun -np 2 -npernode 2 -mca btl_tcp_if_include bond0 -mca orte_base_help_aggregate 0 --mca btl ^openib --allow-run-as-root ./tests/unit_test//build/pcclCommAbortTest -b 1g -a 1 -n 200 -c 0 -t 1 -g 1
mpirun -np 2 -npernode 2 -mca btl_tcp_if_include bond0 -mca orte_base_help_aggregate 0 --mca btl ^openib --allow-run-as-root ./tests/unit_test//build/pcclCommAbortTest -b 1g -a 1 -n 200 -c 1 -t 1 -g 1
mpirun -np 2 -npernode 2 -mca btl_tcp_if_include bond0 -mca orte_base_help_aggregate 0 --mca btl ^openib --allow-run-as-root ./tests/unit_test//build/pcclCommAbortTest -b 1g -a 2 -n 2 -c 0 -t 1 -g 1
mpirun -np 2 -npernode 2 -mca btl_tcp_if_include bond0 -mca orte_base_help_aggregate 0 --mca btl ^openib --allow-run-as-root ./tests/unit_test//build/pcclCommAbortTest -b 1g -a 2 -n 2 -c 1 -t 1 -g 1
mpirun -np 1 -npernode 1 -mca btl_tcp_if_include bond0 -mca orte_base_help_aggregate 0 --mca btl ^openib --allow-run-as-root ./tests/unit_test//build/pcclCommAbortTest -b 1g -a 0 -n 2 -c 0 -t 1 -g 2
mpirun -np 1 -npernode 1 -mca btl_tcp_if_include bond0 -mca orte_base_help_aggregate 0 --mca btl ^openib --allow-run-as-root ./tests/unit_test//build/pcclCommAbortTest -b 1g -a 0 -n 2 -c 1 -t 1 -g 2
mpirun -np 1 -npernode 1 -mca btl_tcp_if_include bond0 -mca orte_base_help_aggregate 0 --mca btl ^openib --allow-run-as-root ./tests/unit_test//build/pcclCommAbortTest -b 1g -a 1 -n 200 -c 0 -t 1 -g 2
mpirun -np 1 -npernode 1 -mca btl_tcp_if_include bond0 -mca orte_base_help_aggregate 0 --mca btl ^openib --allow-run-as-root ./tests/unit_test//build/pcclCommAbortTest -b 1g -a 1 -n 200 -c 1 -t 1 -g 2
mpirun -np 1 -npernode 1 -mca btl_tcp_if_include bond0 -mca orte_base_help_aggregate 0 --mca btl ^openib --allow-run-as-root ./tests/unit_test//build/pcclCommAbortTest -b 1g -a 2 -n 2 -c 0 -t 1 -g 2
mpirun -np 1 -npernode 1 -mca btl_tcp_if_include bond0 -mca orte_base_help_aggregate 0 --mca btl ^openib --allow-run-as-root ./tests/unit_test//build/pcclCommAbortTest -b 1g -a 2 -n 2 -c 1 -t 1 -g 2

echo "######################### $(date +%Y%m%d%H%M%S) commabort test finish ##########################"
