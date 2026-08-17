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
echo "######################### $(date +%Y%m%d%H%M%S) register test ##########################"
echo $PWD
echo "pwd has:"
ls

mpirun --allow-run-as-root -np 2 ./tests/pccl-tests/build/scatter_perf -b 32 -e 1G -f 2 -R 1 -d half
mpirun --allow-run-as-root -np 2 ./tests/pccl-tests/build/gather_perf -b 32 -e 1G -f 2 -R 1 -d half

cd $TEST_ROOT/comm-internal-testbox/scripts/pccl-tests
python run_test.py --op alltoall_perf --algo Auto --proto Auto --data-type float --data-size 32MB --iters 10 --ranks 2 --nproc 1 --threads 2 --auto-detect --register 1
python run_test.py --op alltoall_perf --algo Auto --proto Auto --data-type float --data-size 32MB --iters 10 --ranks 2 --nproc 2 --threads 1 --auto-detect --register 1
python run_test.py --op all_gather_perf --algo Auto --proto Auto --data-type float --data-size 32MB --iters 10 --ranks 2 --nproc 1 --threads 2 --auto-detect --register 1
python run_test.py --op all_gather_perf --algo Auto --proto Auto --data-type float --data-size 32MB --iters 10 --ranks 2 --nproc 2 --threads 1 --auto-detect --register 1
python run_test.py --op all_reduce_perf --algo Auto --proto Auto --data-type float --data-size 32MB --iters 10 --ranks 2 --nproc 1 --threads 2 --auto-detect --register 1
python run_test.py --op all_reduce_perf --algo Auto --proto Auto --data-type float --data-size 32MB --iters 10 --ranks 2 --nproc 2 --threads 1 --auto-detect --register 1
python run_test.py --op hypercube_perf --algo Ring --proto Simple --data-type int8 --data-size 32MB --iters 10 --ranks 2 --nproc 1 --threads 2 --auto-detect --register 1
python run_test.py --op hypercube_perf --algo Ring --proto Simple --data-type int8 --data-size 32MB --iters 10 --ranks 2 --nproc 2 --threads 1 --auto-detect --register 1
python run_test.py --op gather_perf --algo Ring --proto Simple --data-type int8 --data-size 32MB --iters 10 --ranks 2 --nproc 1 --threads 2 --auto-detect --register 1
python run_test.py --op gather_perf --algo Ring --proto Simple --data-type int8 --data-size 32MB --iters 10 --ranks 2 --nproc 2 --threads 1 --auto-detect --register 1
python run_test.py --op scatter_perf --algo Ring --proto Simple --data-type int8 --data-size 32MB --iters 10 --ranks 2 --nproc 1 --threads 2 --auto-detect --register 1
python run_test.py --op scatter_perf --algo Ring --proto Simple --data-type int8 --data-size 32MB --iters 10 --ranks 2 --nproc 2 --threads 1 --auto-detect --register 1
python run_test.py --op sendrecv_perf --algo Ring --proto Simple --data-type int8 --data-size 32MB --iters 10 --ranks 2 --nproc 1 --threads 2 --auto-detect --register 1
python run_test.py --op sendrecv_perf --algo Ring --proto Simple --data-type int8 --data-size 32MB --iters 10 --ranks 2 --nproc 2 --threads 1 --auto-detect --register 1

echo "######################### $(date +%Y%m%d%H%M%S) register test finish ##########################"
