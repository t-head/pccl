#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE.txt for more license information

data_types_full="i8 u8 i32 u32 i64 u64 f16 f32 f64 bf16 f8e4m3 f8e5m2"
data_types=$data_types_full
coll_ops_full="sendrecv all_reduce all_gather broadcast reduce reduce_scatter"
coll_ops=$coll_ops_full
reduct_ops_full="sum prod max min premulsum sumpostdiv"
reduct_ops=$reduct_ops_full
keep=0
SAVE_TEMPS_OPTION=""
targets="GENOBJS := \\\\\n"

function usage
{
    script_name=$0
    echo "Usage:"
    echo "  $script_name --obj_dir obj_dir [--coll_ops sendrecv/all_reduce/all_gather/broadcast/reduce/reduce_scatter]"
    echo "               [--reduct_ops sum/prod/max/min/premulsum/sumpostdiv] [--data_types i8/u8/i32/u32/i64/u64/f16/f32/f64/bf16/f8e4m3/f8e5m2] "
    echo ""
    echo "  Parameters:"
    echo "    obj_dir:         Specify obj directory to store built device object files."
    echo "    coll_ops:        Specify coll ops to build."
    echo "    reduct_ops:      Specify reduction ops to build."
    echo "    data_types:      Specify data types to build."
}

function contains
{
    #[[ $1 =~ (^|[[:space:]])$2($|[[:space:]]) ]] && exit(0) || exit(1)
    coll_op=$1
    reduct_op=$2
    dt=$3

    echo $coll_ops | grep -w -q $coll_op
    if [ $? == 1 ]; then
        return 1
    fi

    echo $reduct_ops | grep -w -q $reduct_op
    if [ $? == 1 ]; then
        return 1
    fi

    echo $data_types | grep -w -q $dt
    if [ $? == 1 ]; then
        return 1
    fi
}

function check_and_correct_params
{
    if [[ $data_types == "none" ]]; then
        data_types=$data_types_full
    fi
    if [[ $coll_ops == "none" ]]; then
        coll_ops=$coll_ops_full
    fi
    if [[ ${keep} -eq 1 ]]; then
        SAVE_TEMPS_OPTION="--keep-dir "
    fi
}

while [[ $# -ge 1 ]]
do
    key="$1"
    case $key in
        --obj_dir)
            obj_dir="$2"
            shift
            ;;
        --coll_ops)
            coll_ops="$2"
            shift
            ;;
        --reduct_ops)
            reduct_ops="$2"
            shift
            ;;
        --data_types)
            data_types="$2"
            shift
            ;;
        --keep)
            keep=$2
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $key"
            usage
            exit 1
            ;;
    esac
    shift
done

check_and_correct_params

fake=0
for base in $coll_ops_full; do
  opn=0
  for op in $reduct_ops_full; do
    if [[ "$base" == "sendrecv" || "$base" == "all_gather" || "$base" == "broadcast" ]]; then
      if [[ "$op" != "sum" ]]; then
        continue
      fi
    fi
    dtn=0
    for dt in $data_types_full; do
      contains $base $op $dt
      fake=$?
      SAVE_TEMPS_PATH=""
      echo "${obj_dir}/${base}_${op}_${dt}.o : ${base}.cu ${obj_dir}/${base}.dep"
      echo "	@printf \"Compiling  %-35s > %s\\\\n\" ${base}.cu ${obj_dir}/${base}_${op}_${dt}.o"
      echo "	mkdir -p ${obj_dir}"
      if [[ ${keep} -eq 1 ]]; then
        SAVE_TEMPS_PATH="${obj_dir}/${base}_${op}_${dt}"
        echo "	mkdir -p ${SAVE_TEMPS_PATH}"
      fi
      echo "	\${HGCC} -DNCCL_OP=${opn} -DNCCL_TYPE=${dtn} -DFAKE_KERNEL=${fake} \${HGFLAGS} ${SAVE_TEMPS_OPTION}${SAVE_TEMPS_PATH} ${base}.cu -o ${obj_dir}/${base}_${op}_${dt}.o"
      echo ""
      targets="$targets\t${obj_dir}/${base}_${op}_${dt}.o \\\\\n"
      dtn=$(($dtn + 1))
    done
    opn=$(($opn + 1))
  done
done
echo -e "$targets"
