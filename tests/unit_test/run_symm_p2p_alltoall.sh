#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# The first run checks planner selection. Keep its tuning logs outside the timed run.
if (( $# > 4 )) || [[ ${1:-} == --help ]]; then
  echo "Usage: $0 [num_gpus [num_elems_per_peer [num_iterations [start_dev_id]]]]"
  if (( $# > 4 )); then exit 1; fi
  exit 0
fi
ranks=${1:-2}
elements=${2:-262144}
iterations=${3:-20}
first=${4:-0}
binary="$(dirname "$0")/build/sendRecvPcclUnitTest"
args=(--mode=symm_alltoall --num_gpus="$ranks" --num_elems="$elements" --start_dev_id="$first")

log=$(mktemp)
trap 'rm -f "$log"' EXIT
if ! output=$(PCCL_DEBUG=INFO PCCL_DEBUG_SUBSYS=TUNING PCCL_DEBUG_FILE="$log" \
    "$binary" "${args[@]}" --num_iterations=1); then
  echo "$output" >&2
  exit 1
fi
if ! grep -Eq "SendRecv \[Symmetric DMA( [^]]+)?\]: batch nSend $ranks nRecv $ranks nSelfSend 1 " "$log"; then
  echo "PCCL did not select symmetric P2P DMA for all $ranks peers." >&2
  exit 1
fi
PCCL_DEBUG=WARN "$binary" "${args[@]}" --num_iterations="$iterations"
