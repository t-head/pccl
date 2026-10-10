#!/bin/bash
set -ex

source "$(dirname "$0")/setup_test_env.sh"

echo "######################### $(date +%Y%m%d%H%M%S) copyengine test ##########################"
echo $PWD
echo "pwd has:"
ls

mpirun --allow-run-as-root -np 2 -x PCCL_CE_ENABLE_AR=1 -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH all_reduce_perf -b 5111004 -e 1023M -f 2 -x 2 -R 2
mpirun --allow-run-as-root -np 2 -x PCCL_OKI_ENABLE=1 -x PCCL_CE_ENABLE_AR=1 -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH all_reduce_perf -b 5111004 -e 1023M -f 2 -x 2 -R 2

# 5110984 rather than a 16B-aligned base: the per-rank slice lands at 4 mod 16, which the CE
# ReduceScatter gate used to reject.
mpirun --allow-run-as-root -np 2 -x PCCL_CE_ENABLE_RS=1 -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH reduce_scatter_perf -b 5110984 -e 1023M -f 2 -x 2 -R 2
mpirun --allow-run-as-root -np 2 -x PCCL_CE_ENABLE_RS=1 -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH reduce_scatter_perf -b 5110984 -e 1023M -f 2 -x 2 -R 2 -o avg

echo "######################### $(date +%Y%m%d%H%M%S) copyengine test finish ##########################"
