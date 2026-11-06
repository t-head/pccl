#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
# Run correctness and launch-path checks against one explicitly selected PCCL build.
set -euo pipefail
binary=${1:?symkMultiTaskTest binary required}
library_dir=${2:?PCCL library directory required}
ranks=${RANKS:-8}
test -x "$binary"
test -d "$library_dir"
export LD_LIBRARY_PATH="$library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PCCL_ENABLE_EXT_KERNEL=0 PCCL_DEBUG=WARN
unset PCCL_SYM_KERNEL PCCL_SYM_KERNEL_ID
unset PCCL_SYM_CTAS PCCL_SYM_WARPS
mpi=(mpirun --allow-run-as-root --bind-to none --mca pml ob1 --mca btl self,vader,tcp
     -np "$ranks" -x LD_LIBRARY_PATH -x PCCL_ENABLE_EXT_KERNEL -x PCCL_DEBUG)
run() {
  timeout 300 "${mpi[@]}" "$binary" "$@"
}
for graph in 0 1; do
  unset PCCL_SYM_KERNEL PCCL_SYM_CTAS
  # Four count boundaries, five data types, three collectives and three batch sizes.
  run functional-sweep float 1 1 3 "$graph" 3 0 0 0 0 1
  for ctas in 2 3 11 24 64; do
    export PCCL_SYM_CTAS=$ctas
    for coll in ar ag rs; do
      case "$coll" in
        ar) export PCCL_SYM_KERNEL=AllReduce_AGxLL_R ;;
        ag) export PCCL_SYM_KERNEL=AllGather_LL ;;
        rs) export PCCL_SYM_KERNEL=ReduceScatter_LL ;;
      esac
      # Full and partial/fused slices, consecutive batches, and exact launch-count assertions.
      run "$coll" float 17 1027 3 "$graph" 30 1 0 0 0 1 0 3
      run "$coll" float 17 1027 3 "$graph" 3 0 0 0 0 1 8 2
      run "$coll" float 4 0 3 "$graph" 3 0 0 0 0 1
    done
  done
  # Registered LD/ST bodies with odd lengths and non-power-of-two CTA overrides.
  for ctas in 11 24; do
    export PCCL_SYM_CTAS=$ctas
    export PCCL_SYM_KERNEL=AllReduce_RSxLD_AGxST
    run ar float 17 16387 3 "$graph" 3 1 0 0 0 1 0 3
    export PCCL_SYM_KERNEL=ReduceScatter_LD
    run rs float 17 16387 3 "$graph" 3 1 0 0 0 1 0 3
    export PCCL_SYM_KERNEL=AllGather_ST
    launches=4
    if [ "$ctas" = 11 ]; then launches=17; fi
    run ag float 17 16387 3 "$graph" 3 1 0 0 1 1 0 "$launches"
  done
  export PCCL_SYM_CTAS=2
  for coll in ar ag rs; do
    case "$coll" in
      ar) export PCCL_SYM_KERNEL=AllReduce_AGxLL_R ;;
      ag) export PCCL_SYM_KERNEL=AllGather_LL ;;
      rs) export PCCL_SYM_KERNEL=ReduceScatter_LL ;;
    esac
    # Seed the real communication epochs near wrap, then validate eager/graph execution and reuse.
    run "$coll" float 17 1027 3 "$graph" 30 1 0 0 0 1 0 3 1
  done
  export PCCL_SYM_KERNEL=AllReduce_RSxLD_AGxST
  run ar float 17 1027 3 "$graph" 30 1 0 0 0 1 0 3 1
  export PCCL_SYM_KERNEL=AllReduce_AGxLL_R PCCL_SYM_CTAS=1
  for dtype in f8e4m3 f8e5m2; do
    run ar "$dtype" 1 1 3 "$graph" 30 1 0 0 1 1 0 1 1
    for count in 1 2 4 7 8 9 257; do
      for pattern in 1 17 83 147 223; do
        # Verify sub-pack and packed FP8 paths against a per-rank rounding oracle.
        run ar "$dtype" 1 "$count" 3 "$graph" 3 1 0 0 1 1 0 1 0 "$pattern"
      done
    done
  done
done
printf 'SYMK_VALIDATION_PASS\n'
