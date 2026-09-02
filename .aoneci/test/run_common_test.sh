#!/bin/bash
set -ex
pip install filelock==3.20.0
export NCCL_DEBUG=INFO
source ./PPU_SDK/envsetup.sh
export LD_LIBRARY_PATH=$TEST_ROOT/lib:$LD_LIBRARY_PATH
echo $TEST_ROOT
echo $LD_LIBRARY_PATH
echo $PATH


echo "######################### $(date +%Y%m%d%H%M%S) common test ##########################"
echo $PWD
echo "pwd has:"
ls
cd $TEST_ROOT/comm-internal-testbox/scripts/unit_test
python unit_test.py -u -a=1.0 -t=alltoallvTest --num_gpus=4 --num_iterations=10 --send_counts=262144,0,109051904,32,32,1024,21495808,1048576,1179648,4096,32,1,256,128,512,512 --algo=Ring --auto_detect=1 --proto=SimpleBulk
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=AllGather --dtype=float --max_num_elems=268435456 --min_num_elems=67108864 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=LL --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=AllGather --dtype=float --max_num_elems=268435456 --min_num_elems=67108864 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=LL128 --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=AllGather --dtype=float --max_num_elems=268435456 --min_num_elems=67108864 --num_gpus=2 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=Signal --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=AllGather --dtype=float --max_num_elems=268435456 --min_num_elems=67108864 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=SignalBulk --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=AllGather --dtype=float --max_num_elems=268435456 --min_num_elems=67108864 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=Simple --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=AllGather --dtype=float --max_num_elems=268435456 --min_num_elems=67108864 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=SimpleBulk --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=AllReduce --dtype=float --max_num_elems=268435456 --min_num_elems=67108864 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=LL --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=AllReduce --dtype=float --max_num_elems=268435456 --min_num_elems=67108864 --num_gpus=4 --num_iterations=10 --algo=Tree --auto_detect=1 --data_size=256MB-1GB --proto=LL --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=AllReduce --dtype=float --max_num_elems=268435456 --min_num_elems=67108864 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=LL128 --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=AllReduce --dtype=float --max_num_elems=268435456 --min_num_elems=67108864 --num_gpus=4 --num_iterations=10 --algo=Tree --auto_detect=1 --data_size=256MB-1GB --proto=LL128 --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=AllReduce --dtype=float --max_num_elems=268435456 --min_num_elems=67108864 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=Signal --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=AllReduce --dtype=float --max_num_elems=268435456 --min_num_elems=67108864 --num_gpus=4 --num_iterations=10 --algo=Tree --auto_detect=1 --data_size=256MB-1GB --proto=Signal --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=AllReduce --dtype=float --max_num_elems=268435456 --min_num_elems=67108864 --num_gpus=4 --num_iterations=10 --algo=Tree --auto_detect=1 --data_size=256MB-1GB --proto=SignalBulk --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=AllReduce --dtype=float --max_num_elems=268435456 --min_num_elems=67108864 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=Simple --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=AllReduce --dtype=float --max_num_elems=268435456 --min_num_elems=67108864 --num_gpus=4 --num_iterations=10 --algo=Tree --auto_detect=1 --data_size=256MB-1GB --proto=Simple --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=AllReduce --dtype=float --max_num_elems=268435456 --min_num_elems=67108864 --num_gpus=4 --num_iterations=10 --algo=Tree --auto_detect=1 --data_size=256MB-1GB --proto=SimpleBulk --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=AllReduce --dtype=float --max_num_elems=268435456 --min_num_elems=67108864 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=SimpleBulk --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=AlltoAll --dtype=half --max_num_elems=536870912 --min_num_elems=134217728 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=SignalBulk --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=AlltoAll --dtype=half --max_num_elems=536870912 --min_num_elems=134217728 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=Simple --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=AlltoAll --dtype=half --max_num_elems=536870912 --min_num_elems=134217728 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=SimpleBulk --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=BroadCast --dtype=float --max_num_elems=268435456 --min_num_elems=67108864 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=LL --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=BroadCast --dtype=float --max_num_elems=268435456 --min_num_elems=67108864 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=LL128 --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=BroadCast --dtype=float --max_num_elems=268435456 --min_num_elems=67108864 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=Signal --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=BroadCast --dtype=float --max_num_elems=268435456 --min_num_elems=67108864 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=SignalBulk --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=BroadCast --dtype=float --max_num_elems=268435456 --min_num_elems=67108864 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=Simple --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=BroadCast --dtype=float --max_num_elems=268435456 --min_num_elems=67108864 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=SimpleBulk --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=Reduce --dtype=half --max_num_elems=536870912 --min_num_elems=134217728 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=LL --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=Reduce --dtype=half --max_num_elems=536870912 --min_num_elems=134217728 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=LL128 --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=Reduce --dtype=half --max_num_elems=536870912 --min_num_elems=134217728 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=Signal --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=Reduce --dtype=half --max_num_elems=536870912 --min_num_elems=134217728 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=SignalBulk --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=ReduceScatter --dtype=half --max_num_elems=536870912 --min_num_elems=134217728 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=LL --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=ReduceScatter --dtype=half --max_num_elems=536870912 --min_num_elems=134217728 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=LL128 --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=ReduceScatter --dtype=half --max_num_elems=536870912 --min_num_elems=134217728 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=Signal --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=ReduceScatter --dtype=half --max_num_elems=536870912 --min_num_elems=134217728 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=SignalBulk --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=ReduceScatter --dtype=half --max_num_elems=536870912 --min_num_elems=134217728 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=Simple --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=ReduceScatter --dtype=half --max_num_elems=536870912 --min_num_elems=134217728 --num_gpus=4 --num_iterations=10 --algo=Ring --auto_detect=1 --data_size=256MB-1GB --proto=SimpleBulk --run_type=ppu_umd
python unit_test.py -u -a=1.0 -t=collPcclUnitTest --coll_type=AllReduce --dtype=double --max_num_elems=134217728 --min_num_elems=33554432 --num_gpus=2 --num_iterations=10 --algo=Tree --auto_detect=1 --data_size=256MB-1GB --proto=SignalBulk --run_type=ppu_umd

cd $TEST_ROOT/comm-internal-testbox/scripts/pccl-tests
python run_test.py --op alltoall_perf --algo Ring --proto Simple --data-type int8 --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op all_gather_perf --algo Ring --proto LL --data-type int8 --data-size 32MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op all_gather_perf --algo Ring --proto LL128 --data-type int8 --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op all_gather_perf --algo Ring --proto Simple --data-type int8 --data-size 512MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op all_reduce_perf --algo Ring --proto LL --data-type float --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op all_reduce_perf --algo Ring --proto LL128 --data-type half --data-size 1.2GB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op all_reduce_perf --algo Ring --proto Simple --data-type float --data-size 512MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op all_reduce_perf --algo Ring --proto SimpleBulk --data-type float --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op all_reduce_perf --algo Ring --proto SimpleBulk --data-type float --data-size 1.2GB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op all_reduce_perf --algo Ring --proto Signal --data-type float --data-size 512MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op all_reduce_perf --algo Ring --proto SignalBulk --data-type half --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op all_reduce_perf --algo Tree --proto LL --data-type half --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op all_reduce_perf --algo Tree --proto LL128 --data-type half --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op all_reduce_perf --algo Tree --proto Simple --data-type half --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op all_reduce_perf --algo Tree --proto SimpleBulk --data-type half --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op all_reduce_perf --algo Tree --proto SimpleBulk --data-type float --data-size 1.2GB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op all_reduce_perf --algo Tree --proto Signal --data-type float --data-size 512MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op all_reduce_perf --algo Tree --proto SignalBulk --data-type half --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op broadcast_perf --algo Ring --proto LL --data-type int8 --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op broadcast_perf --algo Ring --proto LL128 --data-type int8 --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op broadcast_perf --algo Ring --proto Simple --data-type int8 --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op broadcast_perf --algo Ring --proto Simple --data-type int8 --data-size 1.2GB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op reduce_perf --algo Ring --proto LL --data-type half --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op reduce_perf --algo Ring --proto LL128 --data-type half --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op reduce_perf --algo Ring --proto Simple --data-type half --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op reduce_scatter_perf --algo Ring --proto LL --data-type half --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op reduce_scatter_perf --algo Ring --proto LL128 --data-type float --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect
python run_test.py --op reduce_scatter_perf --algo Ring --proto Simple --data-type half --data-size 100.5MB --iters 10 --ranks 4 --devices 0,1,2,3 --auto-detect

echo "######################### $(date +%Y%m%d%H%M%S) common test finish ##########################"
