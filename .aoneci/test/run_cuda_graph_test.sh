#!/bin/bash
set -ex
pip install filelock==3.20.0
export NCCL_DEBUG=INFO
source ./PPU_SDK/envsetup.sh
export LD_LIBRARY_PATH=$TEST_ROOT/lib:$LD_LIBRARY_PATH
echo $TEST_ROOT
echo $LD_LIBRARY_PATH
echo $PATH

cd $TEST_ROOT/comm-internal-testbox/scripts/unit_test
echo "######################### $(date +%Y%m%d%H%M%S) cuda graph test ##########################"
echo $PWD
echo "pwd has:"
ls

python unit_test.py -u -a=1.0 -t=cudaGraphUnitTest --num_threads=4 --test_config_file=$TEST_ROOT/comm-internal-testbox/tests/unit_test/cuda_graph_test_configs/mixing_group_test.json --auto_detect=1
python unit_test.py -u -a=1.0 -t=cudaGraphUnitTest --num_threads=4 --test_config_file=$TEST_ROOT/comm-internal-testbox/tests/unit_test/cuda_graph_test_configs/simple_test.json --auto_detect=1
python unit_test.py -u -a=1.0 -t=cudaGraphUnitTest --num_threads=4 --test_config_file=$TEST_ROOT/comm-internal-testbox/tests/unit_test/cuda_graph_test_configs/simple_alltoall_test.json --auto_detect=1
python unit_test.py -u -a=1.0 -t=cudaGraphUnitTest --num_threads=4 --test_config_file=$TEST_ROOT/comm-internal-testbox/tests/unit_test/feature_test_configs/group_aggregation_test.json --auto_detect=1
python unit_test.py -u -a=1.0 -t=cudaGraphUnitTest --num_threads=2 --test_config_file=$TEST_ROOT/comm-internal-testbox/tests/unit_test/cuda_graph_test_configs/simple_test.json --auto_detect=1

echo "######################### $(date +%Y%m%d%H%M%S) cuda graph test finish ##########################"
