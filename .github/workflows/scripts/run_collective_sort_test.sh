#!/bin/bash
set -eo pipefail
source "$(dirname "$0")/setup_test_env.sh"
hgcc -std=c++14 -I"${TEST_ROOT}/include" \
  "${TEST_ROOT}/pcclCollectiveSortTest.cu" \
  -L"${TEST_ROOT}/lib" -lpccl -lhggc_wrapper -lhg_wrapper \
  -o "${TEST_ROOT}/pcclCollectiveSortTest"
export PCCL_ENABLE_EXT_KERNEL=0
for ranks in 2 4; do
  timeout 180s "${TEST_ROOT}/pcclCollectiveSortTest" "$ranks" 0
  PCCL_WORK_ARGS_BYTES=512 timeout 180s "${TEST_ROOT}/pcclCollectiveSortTest" "$ranks" 1
done
