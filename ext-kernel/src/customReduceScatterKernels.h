/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef CUSTOM_REDUCE_SCATTER_H_
#define CUSTOM_REDUCE_SCATTER_H_
#pragma once

#include "customDeviceUtils.h"

// 3 dtypes × 4 nranks × 4 align × 2 elt_wise_tail = 96 kernel variants.
#define RS_DTYPES_NUM 3

// Forward declaration of the ReduceScatter kernel template. Defined and
// explicitly instantiated in customReduceScatterKernels.cu; declared here so the
// unified customKernels[] array (customKernels.cu) can reference it.
template <typename T, int NRANKS, int INPUT_ALIGN, int OUTPUT_ALIGN, int ELT_WISE_TAIL>
__global__ void pcclKernel_oneShotReduceScatterKernel(struct oneShotDevParams params);

// ---- ReduceScatter kernel function pointer array macros ----

#define RS_ONE_SHOT_FUNCS_NAME(input_align, output_align, elt_wise_tail, nRanks, dtype) \
  (void*)pcclKernel_oneShotReduceScatterKernel<dtype, nRanks, input_align, output_align, elt_wise_tail>,

#define RS_ONE_SHOT_FUNCS_ALIGN(input_align, output_align, nRanks, dtype) \
  RS_ONE_SHOT_FUNCS_NAME(input_align, output_align, 0, nRanks, dtype) \
  RS_ONE_SHOT_FUNCS_NAME(input_align, output_align, 1, nRanks, dtype)

#define RS_ONE_SHOT_FUNCS_RANK(nRanks, dtype) \
  RS_ONE_SHOT_FUNCS_ALIGN(0, 0, nRanks, dtype) \
  RS_ONE_SHOT_FUNCS_ALIGN(0, 1, nRanks, dtype) \
  RS_ONE_SHOT_FUNCS_ALIGN(1, 0, nRanks, dtype) \
  RS_ONE_SHOT_FUNCS_ALIGN(1, 1, nRanks, dtype)

#define RS_ONE_SHOT_FUNCS_DTYPE(dtype) \
  RS_ONE_SHOT_FUNCS_RANK(2, dtype) \
  RS_ONE_SHOT_FUNCS_RANK(4, dtype) \
  RS_ONE_SHOT_FUNCS_RANK(8, dtype) \
  RS_ONE_SHOT_FUNCS_RANK(16, dtype)

#define RS_ONE_SHOT_FUNCS() \
  RS_ONE_SHOT_FUNCS_DTYPE(half) \
  RS_ONE_SHOT_FUNCS_DTYPE(__ppu_bfloat16) \
  RS_ONE_SHOT_FUNCS_DTYPE(float)

#define RS_TAIL_NUM 2
#define RS_ONESHOT_FUNCS_NUM (RS_DTYPES_NUM * CUSTOM_NRANKS_NUM * CUSTOM_ALIGN_NUM * RS_TAIL_NUM)

// ---- ReduceScatter template instantiation macros ----

#define RS_ONE_SHOT_FUNCS_IMPL_NAME(input_align, output_align, elt_wise_tail, nRanks, dtype) \
  template __global__ void pcclKernel_oneShotReduceScatterKernel<dtype, nRanks, input_align, output_align, elt_wise_tail>(struct oneShotDevParams params);

#define RS_ONE_SHOT_FUNCS_IMPL_ALIGN(input_align, output_align, nRanks, dtype) \
  RS_ONE_SHOT_FUNCS_IMPL_NAME(input_align, output_align, 0, nRanks, dtype) \
  RS_ONE_SHOT_FUNCS_IMPL_NAME(input_align, output_align, 1, nRanks, dtype)

#define RS_ONE_SHOT_FUNCS_IMPL_RANK(nRanks, dtype) \
  RS_ONE_SHOT_FUNCS_IMPL_ALIGN(0, 0, nRanks, dtype) \
  RS_ONE_SHOT_FUNCS_IMPL_ALIGN(0, 1, nRanks, dtype) \
  RS_ONE_SHOT_FUNCS_IMPL_ALIGN(1, 0, nRanks, dtype) \
  RS_ONE_SHOT_FUNCS_IMPL_ALIGN(1, 1, nRanks, dtype)

#define RS_ONE_SHOT_FUNCS_IMPL_DTYPE(dtype) \
  RS_ONE_SHOT_FUNCS_IMPL_RANK(2, dtype) \
  RS_ONE_SHOT_FUNCS_IMPL_RANK(4, dtype) \
  RS_ONE_SHOT_FUNCS_IMPL_RANK(8, dtype) \
  RS_ONE_SHOT_FUNCS_IMPL_RANK(16, dtype)

#define RS_ONE_SHOT_FUNCS_IMPL() \
  RS_ONE_SHOT_FUNCS_IMPL_DTYPE(half) \
  RS_ONE_SHOT_FUNCS_IMPL_DTYPE(__ppu_bfloat16) \
  RS_ONE_SHOT_FUNCS_IMPL_DTYPE(float)

// RS dtype index: half=0, bfloat16=1, float=2 (same mapping as AR).
static inline int rsDtypeToIndex(pcclDataType_t dtype) {
  if (dtype == pcclHalf) return 0;
  else if (dtype == pcclBfloat16) return 1;
  else return 2;
}

// Index calculation: layout matches macro expansion order: dtype → rank → align → elt_wise_tail.
static int getReduceScatterKernelIndex(pcclDataType_t dataType, int nRanks, int inputAlign, int outputAlign, int eltWiseTail) {
  const int alignNum = CUSTOM_ALIGN_NUM * RS_TAIL_NUM;
  const int rankNum = CUSTOM_NRANKS_NUM * alignNum;

  int nRanksIdx = ranksToIndex(nRanks);
  int dtypeIndex = rsDtypeToIndex(dataType);
  return dtypeIndex * rankNum +
         nRanksIdx * alignNum +
         (inputAlign * 2 + outputAlign) * RS_TAIL_NUM +
         eltWiseTail;
}

pcclResult_t customReduceScatterImpl(struct extKernelDevParams& devParams, struct extKernelHostParams& hostParams, hggcStream_t stream);

#endif  // CUSTOM_REDUCE_SCATTER_H_
