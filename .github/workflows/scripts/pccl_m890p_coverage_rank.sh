#!/bin/bash
# An MPI rank enters here after xTest maps the worker's allocated PPU indices.
set -euo pipefail
runtime=${PCCL_COVERAGE_RUNTIME:?Missing coverage runtime}
rank=${OMPI_COMM_WORLD_RANK:?Missing MPI rank}
local_rank=${OMPI_COMM_WORLD_LOCAL_RANK:?Missing local MPI rank}
[[ "$rank" =~ ^([0-9]|1[0-5])$ && "$local_rank" =~ ^[0-7]$ ]]
[[ "${OMPI_COMM_WORLD_SIZE:-}" == 16 && "${OMPI_COMM_WORLD_LOCAL_SIZE:-}" == 8 ]]
profile="$runtime/../ranks/rank-$rank"
mkdir "$profile" # Fail on duplicate or stale rank data.
binary="$runtime/benchmark/all_reduce_perf"
export LD_LIBRARY_PATH="$runtime/lib:${LD_LIBRARY_PATH:-}"
export GCOV_PREFIX="$profile"
# Build path is /workspace/source/pccl/build; preserve obj/... on NAS.
export GCOV_PREFIX_STRIP=4
export GCOV_ERROR_FILE="$profile/gcov-errors.log"
export GCOV_EXIT_AT_ERROR=1
export PCCL_DEBUG_FILE="$profile/pccl.log"
ldd "$binary" > "$profile/ldd.txt"
resolved=$(awk '$1 ~ /^libpccl\.so\./ && $2 == "=>" {print $3}' "$profile/ldd.txt")
test -n "$resolved"
test "$(readlink -f "$resolved")" = "$(readlink -f "$runtime/lib/libpccl.so")"
printf '%s\t%s\t%s\n' "$rank" "$local_rank" "${NODE_NAME:?Missing allocated physical node name}" > "$profile/rank.tsv"
exec "$binary" "$@"
