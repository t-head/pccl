#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

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

exec "$binary" "${args[@]}" --num_iterations="$iterations"
