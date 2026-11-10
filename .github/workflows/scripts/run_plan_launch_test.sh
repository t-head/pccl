#!/bin/bash
set -eo pipefail

source "$(dirname "$0")/setup_test_env.sh"

export PCCL_DEBUG=INFO
export PCCL_DEBUG_SUBSYS=INIT,COLL
export PCCL_ALGO=Ring
export PCCL_PROTO=Simple
export PCCL_ENABLE_EXT_KERNEL=0
export PCCL_SYM_NOWIN_ENABLE=0
export PCCL_WORK_FIFO_BYTES=4096
export PCCL_WORK_ARGS_BYTES=1024

binary="${TEST_ROOT}/pccl/tests/unit_test/build/pcclPlanLaunchTest"
log_dir="${TEST_ROOT}/plan-launch-logs"
mkdir -p "$log_dir"

# A fresh process is required when changing cached PCCL parameters. Do not
# retry a failure: leave the first failure and its device state for diagnosis.
for mode in PARALLEL GROUP; do
  export PCCL_LAUNCH_MODE="$mode"
  for ops in 1 4 8 16 1024; do
    "$binary" "$ops" 2>&1 | tee "$log_dir/${mode}-${ops}.log"
  done
  # Check that the intended storage paths were reached, not just correct output.
  grep -q 'workStorageType 0, persistent 0' "$log_dir/${mode}-1.log"
  grep -q 'workStorageType 1, persistent 0' "$log_dir/${mode}-1024.log"
  "$binary" 1024 threaded 2>&1 | tee "$log_dir/${mode}-threaded.log"
  "$binary" 1024 graph 2>&1 | tee "$log_dir/${mode}-graph.log"
  grep -q 'workStorageType 2, persistent 1' "$log_dir/${mode}-graph.log"
done
