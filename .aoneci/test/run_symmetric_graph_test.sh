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
echo "######################### $(date +%Y%m%d%H%M%S) symmetric graph test ##########################"
echo $PWD
echo "pwd has:"
ls

PERF=./tests/pccl-tests/build

# Symmetric (register=2) collectives, run both with a cuda graph (-G 1) and without
# one (-G 0). The graph variant is the regression for the persistentDestructor
# NULL-deref crash: at -b 4 the per-rank count (count/nranks with count=1 element)
# floors to zero for nranks>=2, so allgather/reducescatter enqueue no plan yet used
# to still register a graph destructor with a null plan list and segfault on graph
# teardown. The -G 0 variant keeps the plain symmetric LSA path covered.
for G in 1 0; do
  mpirun --allow-run-as-root -np 2 -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH $PERF/all_reduce_perf     -b 4 -e 128M -f 2 -R 2 -G $G
  mpirun --allow-run-as-root -np 2 -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH $PERF/all_gather_perf      -b 4 -e 128M -f 2 -R 2 -G $G
  mpirun --allow-run-as-root -np 2 -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH $PERF/reduce_scatter_perf  -b 4 -e 128M -f 2 -R 2 -G $G
done

# Symmetric p2p over the OKI copy-engine DMA path. It needs PCCL_P2P_DMA_ENABLE=1
# and CTA policy ZERO (-x 2, which ncclCeSupported requires), and is only chosen
# outside graph capture (under a graph the planner uses the kernel p2p path). The
# second line exercises graph-captured sendrecv on the kernel p2p path.
mpirun --allow-run-as-root -np 2 -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -x PCCL_P2P_DMA_ENABLE=1 $PERF/sendrecv_perf -b 4 -e 128M -f 2 -R 2 -x 2 -G 0
mpirun --allow-run-as-root -np 2 -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH $PERF/sendrecv_perf -b 4 -e 128M -f 2 -R 2 -G 1

echo "######################### $(date +%Y%m%d%H%M%S) symmetric graph test finish ##########################"
