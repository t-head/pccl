/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef CUSTOM_ALL_GATHER_H_
#define CUSTOM_ALL_GATHER_H_
#pragma once

#include "customDeviceUtils.h"

// AllGather only instantiates int8 kernels: PCCL core's argcheck.cc converts
// every AllGather to ncclInt8 + count-as-bytes before dispatch, and the kernel
// body is a byte-level memcpy (BulkLoad128/Store128) — fully type-agnostic.
// NOTE: AG_DTYPES_NUM MUST remain 1 (only int8_t is supported by design).
// AllGather kernels do not use RELAX_MODE (Phase 2 reads only from the local
// buffer, so no trailing barrier is needed).
// 1 dtype × 4 nranks × 4 align = 16 kernel variants per strategy.
#define AG_DTYPES_NUM 1

// Forward declaration of the AllGather kernel template. Defined and explicitly
// instantiated in customAllGatherKernels.cu; declared here so the unified
// customKernels[] array (customKernels.cu) can reference it.
template <typename T, int NRANKS, int INPUT_ALIGN, int OUTPUT_ALIGN>
__global__ void pcclKernel_oneShotAllGatherKernel(struct oneShotDevParams params);

// ---- AllGather kernel function pointer array macros ----

#define AG_ONE_SHOT_FUNCS_NAME(input_align, output_align, nRanks, dtype) \
  (void*)pcclKernel_oneShotAllGatherKernel<dtype, nRanks, input_align, output_align>,

#define AG_ONE_SHOT_FUNCS_ALIGN(input_align, output_align, nRanks, dtype) \
  AG_ONE_SHOT_FUNCS_NAME(input_align, output_align, nRanks, dtype)

#define AG_ONE_SHOT_FUNCS_RANK(nRanks, dtype) \
  AG_ONE_SHOT_FUNCS_ALIGN(0, 0, nRanks, dtype) \
  AG_ONE_SHOT_FUNCS_ALIGN(0, 1, nRanks, dtype) \
  AG_ONE_SHOT_FUNCS_ALIGN(1, 0, nRanks, dtype) \
  AG_ONE_SHOT_FUNCS_ALIGN(1, 1, nRanks, dtype)

#define AG_ONE_SHOT_FUNCS_DTYPE(dtype) \
  AG_ONE_SHOT_FUNCS_RANK(2, dtype) \
  AG_ONE_SHOT_FUNCS_RANK(4, dtype) \
  AG_ONE_SHOT_FUNCS_RANK(8, dtype) \
  AG_ONE_SHOT_FUNCS_RANK(16, dtype)

#define AG_ONE_SHOT_FUNCS() \
  AG_ONE_SHOT_FUNCS_DTYPE(int8_t)

#define AG_ONESHOT_FUNCS_NUM (AG_DTYPES_NUM * CUSTOM_NRANKS_NUM * CUSTOM_ALIGN_NUM)

// ---- AllGather template instantiation macros ----

#define AG_ONE_SHOT_FUNCS_IMPL_NAME(input_align, output_align, nRanks, dtype) \
  template __global__ void pcclKernel_oneShotAllGatherKernel<dtype, nRanks, input_align, output_align>(struct oneShotDevParams params);

#define AG_ONE_SHOT_FUNCS_IMPL_ALIGN(input_align, output_align, nRanks, dtype) \
  AG_ONE_SHOT_FUNCS_IMPL_NAME(input_align, output_align, nRanks, dtype)

#define AG_ONE_SHOT_FUNCS_IMPL_RANK(nRanks, dtype) \
  AG_ONE_SHOT_FUNCS_IMPL_ALIGN(0, 0, nRanks, dtype) \
  AG_ONE_SHOT_FUNCS_IMPL_ALIGN(0, 1, nRanks, dtype) \
  AG_ONE_SHOT_FUNCS_IMPL_ALIGN(1, 0, nRanks, dtype) \
  AG_ONE_SHOT_FUNCS_IMPL_ALIGN(1, 1, nRanks, dtype)

#define AG_ONE_SHOT_FUNCS_IMPL_DTYPE(dtype) \
  AG_ONE_SHOT_FUNCS_IMPL_RANK(2, dtype) \
  AG_ONE_SHOT_FUNCS_IMPL_RANK(4, dtype) \
  AG_ONE_SHOT_FUNCS_IMPL_RANK(8, dtype) \
  AG_ONE_SHOT_FUNCS_IMPL_RANK(16, dtype)

#define AG_ONE_SHOT_FUNCS_IMPL() \
  AG_ONE_SHOT_FUNCS_IMPL_DTYPE(int8_t)

// Index calculation: dtype dimension removed (int8-only), relax dimension removed
// (AG reads only from local buffer in Phase 2, no trailing barrier needed).
// Layout matches macro expansion order: rank → align.
static int getAllGatherKernelIndex(int nRanks, int inputAlign, int outputAlign) {
  const int alignNum = CUSTOM_ALIGN_NUM;
  int nRanksIdx = ranksToIndex(nRanks);
  return nRanksIdx * alignNum +
         (inputAlign * 2 + outputAlign);
}

pcclResult_t customAllGatherImpl(struct extKernelDevParams& devParams, struct extKernelHostParams& hostParams, hggcStream_t stream);

#endif  // CUSTOM_ALL_GATHER_H_
