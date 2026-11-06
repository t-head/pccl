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
export PCCL_ENABLE_EXT_KERNEL=0 PCCL_DEBUG=WARN PCCL_CONF_FILE=/dev/null
unset PCCL_SYM_KERNEL PCCL_SYM_CTAS PCCL_SYM_NOWIN_ENABLE PCCL_WORK_ARGS_BYTES
mpi=(mpirun --allow-run-as-root --bind-to none --mca pml ob1 --mca btl self,vader,tcp
     -np "$ranks" -x LD_LIBRARY_PATH -x PCCL_ENABLE_EXT_KERNEL -x PCCL_DEBUG)
run() {
  local name
  local exports=()
  while IFS= read -r name; do exports+=(-x "$name"); done < <(compgen -e | grep '^PCCL_')
  if [[ -n ${NCCL_SYM_KERNEL+x} ]]; then exports+=(-x NCCL_SYM_KERNEL); fi
  timeout "${RUN_TIMEOUT:-900}" "${mpi[@]}" "${exports[@]}" "$binary" "$@"
}
layout=$(run layout | awk '/^[0-9]+ [0-9]+ [0-9]+$/ {print; found++} END {if (found != 1) exit 1}')
read -r header work_bytes capacity <<< "$layout"
[[ $header =~ ^[0-9]+$ && $work_bytes =~ ^[0-9]+$ && $capacity =~ ^[0-9]+$ ]]
(( work_bytes == 64 && capacity > 17 && header + capacity*work_bytes <= 4096 ))
timeout "${RUN_TIMEOUT:-900}" "$binary" param-contract
for graph in 0 1; do
  unset PCCL_SYM_KERNEL PCCL_SYM_CTAS PCCL_WORK_ARGS_BYTES
  run p2p-empty "$graph"
  run p2p-budget 1024 "$graph"
  run budget-contract "$graph"
  run kernel-cache-contract float 1 1027 3 "$graph" 3 1 0 0 1 1
  run ar float 0 0 3 "$graph" 3 0 0 0 0 1
  run functional-sweep float 1 1 3 "$graph" 3 0 0 0 0 1
  for ctas in 2 3 11 24 64; do
    export PCCL_SYM_CTAS=$ctas
    for coll in ar ag rs rs-avg; do
      case "$coll" in
        ar) export PCCL_SYM_KERNEL=AllReduce_RSxLD_AGxST ;;
        ag) export PCCL_SYM_KERNEL=AllGather_ST ;;
        rs|rs-avg) export PCCL_SYM_KERNEL=ReduceScatter_LD ;;
      esac
      for works in 1 7 17 "$capacity" "$((capacity + 1))" "$((2*capacity + 1))"; do
        launches=$(((works + capacity - 1)/capacity))
        run "$coll" float "$works" 16387 3 "$graph" 3 1 0 0 1 1 0 "$launches"
      done
      run "$coll" float 17 1027 3 "$graph" 3 1 0 0 0 1 8 1
      run "$coll" float 4 0 3 "$graph" 3 0 0 0 0 1
      run "$coll" float 1 1027 3 "$graph" 30 1 0 0 1 1 0 1 1
      case "$coll" in
        ar) export PCCL_SYM_KERNEL=AllReduce_AGxLL_R ;;
        ag) export PCCL_SYM_KERNEL=AllGather_LL ;;
        rs|rs-avg) export PCCL_SYM_KERNEL=ReduceScatter_LL ;;
      esac
      run "$coll" float 1 1027 3 "$graph" 30 1 0 0 1 1 0 1 1
      run "$coll" float 17 1027 3 "$graph" 3 1 0 0 0 2
    done
  done
  export PCCL_SYM_CTAS=2
  for coll in ar ag rs rs-avg; do
    for reg in 0 1 2 3; do
      case "$coll" in
        ar) export PCCL_SYM_KERNEL=AllReduce_AGxLL_R ;;
        ag) export PCCL_SYM_KERNEL=AllGather_LL ;;
        rs|rs-avg) export PCCL_SYM_KERNEL=ReduceScatter_LL ;;
      esac
      run "$coll" float 1 1027 "$reg" "$graph" 3 1 0 0 1 1 0 1
      case "$coll" in
        ar) export PCCL_SYM_KERNEL=AllReduce_RSxLD_AGxST; eligible=$((reg == 3)) ;;
        ag) export PCCL_SYM_KERNEL=AllGather_ST; eligible=$(((reg & 2) != 0)) ;;
        rs|rs-avg) export PCCL_SYM_KERNEL=ReduceScatter_LD; eligible=$(((reg & 1) != 0)) ;;
      esac
      path=$((eligible ? 1 : 2))
      run "$coll" float 17 1027 "$reg" "$graph" 3 1 0 0 1 "$path"
    done
  done
  export PCCL_SYM_KERNEL=AllReduce_RSxLD_AGxST
  export PCCL_WORK_ARGS_BYTES=$((header + 7*work_bytes))
  limited_capacity=$(((PCCL_WORK_ARGS_BYTES - header)/work_bytes))
  run ar float 17 1027 3 "$graph" 3 1 0 0 1 1 0 "$(((17 + limited_capacity - 1)/limited_capacity))"
  unset PCCL_WORK_ARGS_BYTES
  export PCCL_SYM_KERNEL=unknown_symk_kernel
  run ar float 17 1027 3 "$graph" 3 1 0 0 1 2
  export PCCL_SYM_TMA_ENABLE=1
  for kernel in AllReduce_AGxLLMC_R AllReduce_RSxTmaLD_AGxTmaST AllReduce_RSxLDMC_AGxSTMC; do
    export PCCL_SYM_KERNEL=$kernel
    run ar float 17 1027 3 "$graph" 3 1 0 0 1 2
  done
  export PCCL_SYM_KERNEL=ReduceScatter_RailA2A_LsaLD
  run rs float 17 1027 3 "$graph" 3 1 0 0 1 2
  unset PCCL_SYM_TMA_ENABLE
  export PCCL_SYM_KERNEL='^'
  run ar float 17 1027 3 "$graph" 3 1 0 0 1 1 0 1
  unset PCCL_SYM_KERNEL
  export NCCL_SYM_KERNEL=unknown_symk_kernel
  run ar float 17 1027 3 "$graph" 3 1 0 0 1 1 0 1
  export PCCL_SYM_KERNEL=AllReduce_RSxLD_AGxST
  run ar float 17 1027 3 "$graph" 3 1 0 0 1 1 0 1
  export PCCL_SYM_KERNEL=unknown_symk_kernel NCCL_SYM_KERNEL=AllReduce_RSxLD_AGxST
  run ar float 17 1027 3 "$graph" 3 1 0 0 1 2
  unset NCCL_SYM_KERNEL
  export PCCL_SYM_KERNEL=AllReduce_AGxLL_R PCCL_SYM_CTAS=1
  if [ "$ranks" = 8 ]; then
    run ar half 1 257 3 "$graph" 3 1 0 1 1 1 0 1
    export PCCL_SYM_KERNEL=AllReduce_RSxLD_AGxST
    run ar half 1 16387 3 "$graph" 3 1 0 1 1 1 0 1
    export PCCL_SYM_KERNEL=AllReduce_AGxLL_R
  fi
  for dtype in f8e4m3 f8e5m2; do
    run ar "$dtype" 1 1 3 "$graph" 30 1 0 0 1 1 0 1 1
    for count in 1 2 4 7 8 9 257; do
      for pattern in 1 17 83 147 223; do
        run ar "$dtype" 1 "$count" 3 "$graph" 3 1 0 0 1 1 0 1 0 "$pattern"
      done
    done
    for kernel in ReduceScatter_LL ReduceScatter_LD; do
      export PCCL_SYM_KERNEL=$kernel
      for count in 1 7 257 16387; do
        for pattern in 1 17 83 147 223; do
          run rs-avg "$dtype" 1 "$count" 3 "$graph" 3 1 0 0 1 1 0 1 0 "$pattern"
        done
      done
    done
    export PCCL_SYM_KERNEL=AllReduce_AGxLL_R
  done
done
printf 'SYMK_VALIDATION_PASS\n'
