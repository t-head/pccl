/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef CUSTOM_KERNELS_H_
#define CUSTOM_KERNELS_H_
#pragma once

#include "customAllReduceKernels.h"
#include "customAllGatherKernels.h"
#include "customReduceScatterKernels.h"

// ============================================================================
// Unified ext-kernel func list — mirrors PCCL core's ncclKerns[].
//
// All collective kernels live in a single flat pointer array (customKernels)
// and share a single dispatch-strategy array (funcStrategy). The collective op
// is no longer a separate list; it is just one dimension of a global func-index
// space, encoded as a per-op base offset. This matches the core FUNC_INDEX
// scheme where {func, devredop, type, algo, proto} index one array.
//
// The array element order MUST match these bases exactly:
//   [ AR block | AG block | RS block ]
// AR block is itself ONE_SHOT, TWO_SHOT, TWO_SHOT_REG, ONESHOT_LL in order.
// ============================================================================
#define EXT_FUNC_AR_BASE 0
#define EXT_FUNC_AG_BASE (AR_EXT_FUNCS_NUM)
#define EXT_FUNC_RS_BASE (AR_EXT_FUNCS_NUM + AG_ONESHOT_FUNCS_NUM)
#define EXT_TOTAL_FUNCS_NUM (AR_EXT_FUNCS_NUM + AG_ONESHOT_FUNCS_NUM + RS_ONESHOT_FUNCS_NUM)

extern void* customKernels[];

#endif  // CUSTOM_KERNELS_H_
