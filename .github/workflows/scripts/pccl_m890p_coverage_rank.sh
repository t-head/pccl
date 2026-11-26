#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# The entry point is installed in the group's test-root/pccl-tests/build.
set -euo pipefail
bindir=$(cd -- "$(dirname -- "$0")" && pwd -P)
group=$(cd -- "$bindir/../../.." && pwd -P)
runtime=$(cd -- "$bindir/../../coverage-runtime" && pwd -P)
name=${0##*/}
node=${NODE_NAME:-$(hostname)}
slot=${OMPI_COMM_WORLD_LOCAL_RANK:-local}
[[ "$node" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ && "$slot" =~ ^(local|[0-9]+)$ ]]
binary="$runtime/benchmark/$name"
profile="$group/export/profiles/$node/$slot"
mkdir -p "$profile"

# Sequential Cases can reuse one profile for each process slot. Serialize any
# accidental overlap on the worker's local filesystem, not across NAS clients.
attempt=${group%/groups/*}
lock_root="${TMPDIR:-/tmp}/pccl-gcov-${attempt##*/}/${group##*/}/$node"
mkdir -p "$lock_root"
exec 9>"$lock_root/$slot.lock"
flock --exclusive --timeout 30 9

export LD_LIBRARY_PATH="$runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GCOV_PREFIX="$profile"
# The compile directory is /workspace/source/pccl/build; keep obj/... below each slot.
export GCOV_PREFIX_STRIP=4
export GCOV_ERROR_FILE="$profile/gcov-errors.log"
export GCOV_EXIT_AT_ERROR=1

# Check once per benchmark/worker/slot, including on the remote MPI workers.
if [[ ! -f "$profile/.loaded-$name" ]]; then
    resolved=$(ldd "$binary" | awk '$1 ~ /^libpccl\.so\./ && $2 == "=>" {print $3}')
    if [[ -z "$resolved" || "$(readlink -f "$resolved")" != "$(readlink -f "$runtime/lib/libpccl.so")" ]]; then
        echo "Wrong PCCL runtime for $name on $node" >&2
        exit 1
    fi
    touch "$profile/.loaded-$name"
fi
exec "$binary" "$@"
