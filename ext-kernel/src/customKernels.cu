//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include "customKernels.h"

// Unified ext-kernel func list — one flat table covering every collective,
// mirroring PCCL core's ncclKerns[]. Element order MUST match the per-op base
// offsets in customKernels.h: [ AR block | AG block | RS block ]. The AR block
// is ONE_SHOT, TWO_SHOT, TWO_SHOT_REG, ONESHOT_LL in that order (matches
// extKernelGetFuncIndex). Kernels are defined and explicitly instantiated in
// their respective op .cu files; here we only take their addresses.
void* customKernels[] = {
  // AllReduce block  (EXT_FUNC_AR_BASE)
  ONE_SHOT_FUNCS()
  TWO_SHOT_FUNCS()
  TWO_SHOT_REG_FUNCS()
  ONESHOT_LL_FUNCS()
  // AllGather block  (EXT_FUNC_AG_BASE)
  AG_ONE_SHOT_FUNCS()
  // ReduceScatter block  (EXT_FUNC_RS_BASE)
  RS_ONE_SHOT_FUNCS()
};

static_assert(sizeof(customKernels) / sizeof(customKernels[0]) == EXT_TOTAL_FUNCS_NUM,
              "customKernels[] size must equal EXT_TOTAL_FUNCS_NUM; array order must match per-op base offsets");
