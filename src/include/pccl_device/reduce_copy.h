/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_REDUCE_COPY_H_
#define _PCCL_DEVICE_REDUCE_COPY_H_
#include "core.h"
#include "impl/reduce_copy__types.h"

// Forward declarations for public API functions
// Implementations are in impl/reduce_copy__funcs.h

#if PCCL_CHECK_HGGCCC
// SERIES 1.x - Generic ReduceCopy with RedOp (LSA sources only)
template<typename T, typename Coop, typename SrcLambda, typename DstLambda,
         typename RedOp, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLsaReduceLsaCopy(Coop, SrcLambda, int, DstLambda, int, RedOp const&, IntCount);

template<typename T, typename Coop, typename SrcLambda, typename DstLambda,
         typename RedOp, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLsaReduceMultimemCopy(Coop, SrcLambda, int, DstLambda, int, RedOp const&, IntCount);

// SERIES 2.x - Sum-Specific ReduceCopy (lambda-based foundation)
template<typename T, typename Coop, typename SrcLambda, typename DstLambda,
         typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLsaReduceSumLsaCopy(Coop, SrcLambda, int, DstLambda, int, IntCount);

template<typename T, typename Coop, typename SrcLambda, typename DstLambda,
         typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLsaReduceSumMultimemCopy(Coop, SrcLambda, int, DstLambda, int, IntCount);

template<typename T, typename Coop, typename SrcLambda, typename DstLambda,
         typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclMultimemReduceSumLsaCopy(Coop, SrcLambda, int, DstLambda, int, IntCount);

template<typename T, typename Coop, typename SrcLambda, typename DstLambda,
         typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclMultimemReduceSumMultimemCopy(Coop, SrcLambda, int, DstLambda, int, IntCount);

// SERIES 3.x - ReduceSum (N->1)
template<typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLsaReduceSum(Coop, SrcLambda, int, T*, IntCount);

template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLsaReduceSum(Coop, pcclSymPtr<T>, T*, IntCount, pcclTeam);

template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLsaReduceSum(Coop, pcclSymPtr<T>, T*, IntCount, pcclDevComm_t);

template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLsaReduceSum(Coop, pcclWindow_t, size_t, T*, IntCount, pcclTeam);

template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLsaReduceSum(Coop, pcclWindow_t, size_t, T*, IntCount, pcclDevComm_t);

template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclMultimemReduceSum(Coop, pcclSymPtr<T>, T*, IntCount, pcclMultimemHandle);

// 3.3b] Multimem ReduceSum (with raw pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclMultimemReduceSum(Coop, T*, T*, IntCount);

template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclMultimemReduceSum(Coop, pcclWindow_t, size_t, T*, IntCount, pcclMultimemHandle);

// 3.4] Local ReduceSum (lambda-based)
template<typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLocalReduceSum(Coop, SrcLambda, int, T*, IntCount);

// 3.5] Local ReduceSum (strided)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLocalReduceSum(Coop, int, T*, size_t, T*, IntCount);

// SERIES 4.x - Copy/Broadcast (1->N)

// 4.1] LSA Copy (lambda-based)
template<typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLsaCopy(Coop, T*, DstLambda, int, IntCount);

// 4.2a] LSA Copy (with pcclSymPtr + team)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLsaCopy(Coop, T*, pcclSymPtr<T>, IntCount, pcclTeam);

// 4.2b] LSA Copy (with pcclSymPtr + devComm)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLsaCopy(Coop, T*, pcclSymPtr<T>, IntCount, pcclDevComm_t);

// 4.2c] LSA Copy (with window + offset + team)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLsaCopy(Coop, T*, pcclWindow_t, size_t, IntCount, pcclTeam);

// 4.2d] LSA Copy (with window + offset + devComm)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLsaCopy(Coop, T*, pcclWindow_t, size_t, IntCount, pcclDevComm_t);

// 4.3a] Multimem Copy (with pcclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclMultimemCopy(Coop, T*, pcclSymPtr<T>, IntCount, pcclMultimemHandle);

// 4.3b] Multimem Copy (with raw pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclMultimemCopy(Coop, T*, T*, IntCount);

// 4.3c] Multimem Copy (with window + offset)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclMultimemCopy(Coop, T*, pcclWindow_t, size_t, IntCount, pcclMultimemHandle);

// 4.4] Local Copy (lambda-based)
template<typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLocalCopy(Coop, T*, DstLambda, int, IntCount);

// 4.5] Local Copy (strided)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLocalCopy(Coop, T*, int, T*, size_t, IntCount);

// SERIES 5.x - ReduceSumCopy (N->M)

// 5.1a] LSA ReduceSumCopy (same team)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLsaReduceSumCopy(Coop, pcclSymPtr<T>, pcclSymPtr<T>, IntCount, pcclTeam);

// 5.1b] LSA ReduceSumCopy (with devComm)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLsaReduceSumCopy(Coop, pcclSymPtr<T>, pcclSymPtr<T>, IntCount, pcclDevComm_t);

// 5.1c] LSA ReduceSumCopy (with windows + team)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLsaReduceSumCopy(Coop, pcclWindow_t, size_t, pcclWindow_t, size_t, IntCount, pcclTeam);

// 5.1d] LSA ReduceSumCopy (with windows + devComm)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLsaReduceSumCopy(Coop, pcclWindow_t, size_t, pcclWindow_t, size_t, IntCount, pcclDevComm_t);

// 5.1e] LSA ReduceSumCopy (different teams)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLsaReduceSumCopy(Coop, pcclSymPtr<T>, pcclTeam, pcclSymPtr<T>, pcclTeam, IntCount);

// 5.2a] Multimem ReduceSumCopy (with pcclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclMultimemReduceSumCopy(Coop, pcclSymPtr<T>, pcclMultimemHandle, pcclSymPtr<T>, pcclMultimemHandle, IntCount);

// 5.2b] Multimem ReduceSumCopy (with raw pointers)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclMultimemReduceSumCopy(Coop, T*, T*, IntCount);

// 5.2c] Multimem ReduceSumCopy (with windows)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclMultimemReduceSumCopy(Coop, pcclWindow_t, size_t, pcclMultimemHandle, pcclWindow_t, size_t, pcclMultimemHandle, IntCount);

// 5.3a] LSA -> Multimem ReduceSumCopy (with pcclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLsaReduceSumMultimemCopy(Coop, pcclSymPtr<T>, pcclTeam, pcclSymPtr<T>, pcclMultimemHandle, IntCount);

// 5.3b] LSA -> Multimem ReduceSumCopy (with raw dst pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLsaReduceSumMultimemCopy(Coop, pcclSymPtr<T>, pcclTeam, T*, IntCount);

// 5.3c] Multimem -> LSA ReduceSumCopy (with pcclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclMultimemReduceSumLsaCopy(Coop, pcclSymPtr<T>, pcclMultimemHandle, pcclSymPtr<T>, pcclTeam, IntCount);

// 5.3d] Multimem -> LSA ReduceSumCopy (with raw src pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclMultimemReduceSumLsaCopy(Coop, T*, pcclSymPtr<T>, pcclTeam, IntCount);

// 5.4] Local ReduceSumCopy (strided)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
PCCL_DEVICE_INLINE void pcclLocalReduceSumCopy(Coop, int, T*, size_t, int, T*, size_t, IntCount);

#endif // PCCL_CHECK_HGGCCC

#endif // _PCCL_DEVICE_REDUCE_COPY_H_
