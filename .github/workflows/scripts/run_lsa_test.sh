#!/bin/bash
set -ex

source "$(dirname "$0")/setup_test_env.sh"

echo "######################### $(date +%Y%m%d%H%M%S) lsa test ##########################"
echo $PWD
echo "pwd has:"
ls

cd ${TEST_ROOT}/pccl-tests/build

./all_reduce_perf -b 8 -e 2G -f 2 --elapsed_type 1 -R 2 -t 2 -g 1 -D 0
./alltoall_perf -b 8 -e 2G -f 2 --elapsed_type 1 -R 2 -t 2 -g 1 -D 0
./all_reduce_perf -b 8 -e 2G -f 2 --elapsed_type 1 -R 2 -t 2 -g 1 -D 1 -V 16
./all_reduce_perf -b 8 -e 2G -f 2 --elapsed_type 1 -R 2 -t 2 -g 1 -D 2 -V 16
./alltoall_perf -b 8 -e 2G -f 2 --elapsed_type 1 -R 2 -t 2 -g 1 -D 1 -V 16
./alltoall_perf -b 8 -e 2G -f 2 --elapsed_type 1 -R 2 -t 2 -g 1 -D 2 -V 16

echo "######################### $(date +%Y%m%d%H%M%S) lsa test finish ##########################"
