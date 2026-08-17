#!/bin/bash
set -ex
pip install filelock==3.20.0
export NCCL_DEBUG=INFO
export NCCL_COMM_BLOCKING=0
source ./PPU_SDK/envsetup.sh
export LD_LIBRARY_PATH=$TEST_ROOT/lib:$LD_LIBRARY_PATH
echo $TEST_ROOT
echo $LD_LIBRARY_PATH
echo $PATH

cd $TEST_ROOT/comm-internal-testbox
echo "######################### $(date +%Y%m%d%H%M%S) nonblocking test ##########################"
echo $PWD
echo "pwd has:"
ls

NCCL_COMM_BLOCKING=0 ./tests/pccl-tests/build/all_reduce_perf -b 4 -e 1G -f 2 -g 2 -R 0 -B 0
NCCL_COMM_BLOCKING=0 ./tests/pccl-tests/build/alltoall_perf -b 4 -e 1G -f 2 -g 2 -R 0 -B 0


cd $TEST_ROOT/comm-internal-testbox/scripts/pccl-tests
python run_test.py --op alltoall_perf --algo Ring --proto Auto --ranks 2 --data-type int8 --data-size 32MB --iters 10 --nproc 1 --threads 2 --nnodes 1 --mpi-options='-mca btl_tcp_if_include bond1 -x NCCL_DEBUG=WARN' --auto-detect --non-blocking
python run_test.py --op all_reduce_perf --algo Ring --proto LL --ranks 2 --data-type half --data-size 32MB --iters 10 --nproc 1 --threads 2 --nnodes 1 --mpi-options='-mca btl_tcp_if_include bond1 -x NCCL_DEBUG=WARN' --auto-detect --non-blocking
python run_test.py --op all_reduce_perf --algo Ring --proto LL --ranks 2 --data-type float --data-size 32MB --iters 10 --nproc 1 --threads 2 --nnodes 1 --mpi-options='-mca btl_tcp_if_include bond1 -x NCCL_DEBUG=WARN' --auto-detect --non-blocking

echo "######################### $(date +%Y%m%d%H%M%S) nonblocking test finish ##########################"
