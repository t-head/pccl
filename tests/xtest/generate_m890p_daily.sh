#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

profile=${1:?Expected smoke or full}
source_revision=${2:?Missing PCCL revision}
runner_revision=${3:?Missing pccl-tests revision}
output=${4:?Missing output directory}
root=$(cd -- "$(dirname -- "$0")" && pwd)
lists="$root/test_lists"

definitions=(op_perf_shm postsil_daily_shm postsil_unit_test_shm)
case "$profile" in
    smoke) ;;
    full)
        definitions+=(pccl_fun_rdma_shm register_test_shm sanity_m890p_register sym_test_shm
            f8_test_shm device_api_shm local_register_perf_shm symetric_perf_shm
            copyengine_perf_shm op_perf_shm_2_rack op_perf_rdma_shm commabort_test_shm)
        ;;
    *) echo "Unsupported M890P daily dataset: $profile" >&2; exit 2 ;;
esac
args=()
for definition in "${definitions[@]}"; do
    args+=(--definition "$lists/$definition.json")
done
exec python3 -S -B "$root/generate_cases.py" "${args[@]}" \
    --suites "$lists/suites.json" --env-root "$lists/envs" --profile "$profile" \
    --platform m890p --source-revision "$source_revision" --runner-revision "$runner_revision" \
    --output "$output"
