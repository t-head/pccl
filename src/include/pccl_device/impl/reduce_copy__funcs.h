/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_REDUCE_COPY__FUNCS_H_
#define _PCCL_DEVICE_REDUCE_COPY__FUNCS_H_

#include "../reduce_copy.h"

#if PCCL_CHECK_HGGCCC
#if defined(__HGGCCC_EXTENDED_LAMBDA__)

#include "reduce_copy__impl.h"
#include <cassert>

// ============================================================================
// UNROLL Parameter Documentation
// ============================================================================
//
// All public API functions in this file accept an UNROLL template parameter
// that specifies the number of ELEMENTS to unroll (not packs).
//
// Default: UNROLL = 4*16/sizeof(T) (matches all_reduce.cuh UnrollPacks=4)
//   - For float/int32 (4 bytes): 16 elements
//   - For half/int16 (2 bytes): 32 elements
//   - For int8 (1 byte): 64 elements
// Note: UNROLL_PACKS is capped at 4 to match all_reduce.cuh behavior
//
// Note: Internally, this is converted to UNROLL_PACKS based on the vectorization
// strategy (Pack type). The distinction is:
//   - UNROLL (user-facing): Number of elements to process per iteration
//   - UNROLL_PACKS (internal): Number of vector packs to process per iteration
//
// ============================================================================

// SERIES 1.x - Generic ReduceCopy with RedOp (LSA sources only)

template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename RedOp, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceLsaCopy(
    Coop coop,
    SrcLambda srcLambda, int nSrc,
    DstLambda dstLambda, int nDst,
    RedOp const& redOp,
    IntCount count) {
  // Compute alignment for opaque lambdas with fallback to smaller pack sizes
  auto alignment = pccl::utility::computeLambdaAlignmentOffsetWithFallback<T>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);

  pccl::utility::reduceCopy<T, RedOp, Coop, false, false, SrcLambda, DstLambda, IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, redOp, count, alignment.alignOffset, alignment.maxPackBytes);
}

template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename RedOp, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceMultimemCopy(
    Coop coop,
    SrcLambda srcLambda, int nSrc,
    DstLambda dstLambda, int nDst,
    RedOp const& redOp,
    IntCount count) {
  assert(false && "multimem is not supported on PPU");
  // Compute alignment for opaque lambdas with fallback to smaller pack sizes
  auto alignment = pccl::utility::computeLambdaAlignmentOffsetWithFallback<T>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);

  pccl::utility::reduceCopy<T, RedOp, Coop, false, true, SrcLambda, DstLambda, IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, redOp, count, alignment.alignOffset, alignment.maxPackBytes);
}

// SERIES 2.x - Sum-Specific ReduceCopy

template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSumLsaCopy(
    Coop coop,
    SrcLambda srcLambda, int nSrc,
    DstLambda dstLambda, int nDst,
    IntCount count) {
  pcclLsaReduceLsaCopy<T, Coop, SrcLambda, DstLambda, pccl::utility::OpSum<T>, IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, pccl::utility::OpSum<T>{}, count);
}

// [ID 2.2] LSA <-> Multimem ReduceSum
template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSumMultimemCopy(
    Coop coop,
    SrcLambda srcLambda, int nSrc,
    DstLambda dstLambda, int nDst,
    IntCount count) {
  assert(false && "multimem is not supported on PPU");
  pcclLsaReduceMultimemCopy<T, Coop, SrcLambda, DstLambda, pccl::utility::OpSum<T>, IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, pccl::utility::OpSum<T>{}, count);
}

// [ID 2.3] Multimem <-> LSA ReduceSum
template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemReduceSumLsaCopy(
    Coop coop,
    SrcLambda srcLambda, int nSrc,
    DstLambda dstLambda, int nDst,
    IntCount count) {
  assert(false && "multimem is not supported on PPU");
  // Compute alignment for opaque lambdas with fallback to smaller pack sizes
  auto alignment = pccl::utility::computeLambdaAlignmentOffsetWithFallback<T>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);

  // Multimem source only supports Sum
  pccl::utility::reduceCopy<T, pccl::utility::OpSum<T>, Coop, true, false, SrcLambda, DstLambda, IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, pccl::utility::OpSum<T>{}, count, alignment.alignOffset, alignment.maxPackBytes);
}

// [ID 2.4] Multimem <-> Multimem ReduceSum
template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemReduceSumMultimemCopy(
    Coop coop,
    SrcLambda srcLambda, int nSrc,
    DstLambda dstLambda, int nDst,
    IntCount count) {
  assert(false && "multimem is not supported on PPU");
  // Compute alignment for opaque lambdas with fallback to smaller pack sizes
  auto alignment = pccl::utility::computeLambdaAlignmentOffsetWithFallback<T>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);

  // Both multimem - only supports Sum
  pccl::utility::reduceCopy<T, pccl::utility::OpSum<T>, Coop, true, true, SrcLambda, DstLambda, IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, pccl::utility::OpSum<T>{}, count, alignment.alignOffset, alignment.maxPackBytes);
}

// ============================================================================
// SERIES 3.x - ReduceSum (N->1)
// ============================================================================

// [ID 3.1] LSA ReduceSum: N sources -> 1 local destination (lambda-based)
template<typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSum(Coop coop,
                                            SrcLambda srcLambda, int nSrc,
                                            T* dstPtr,
                                            IntCount count) {
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dstPtr;
  };
  constexpr int nDst = 1;  // Reduce has single destination
  pcclLsaReduceSumLsaCopy<T, Coop, SrcLambda, decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 3.2a] LSA ReduceSum: N sources from team -> 1 local destination (with pcclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSum(Coop coop,
                                            pcclSymPtr<T> src,
                                            T* dstPtr,
                                            IntCount count,
                                            pcclTeam team) {
  // Create lambda: N sources from team
  auto srcLambda = [=] __device__ (int i) -> T* {
    return src.peerPtr(team, i);
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dstPtr;
  };

  // LSA translation ensures all peers have the same alignment
  // Only need to check the first pointer (much cheaper than checking all N)
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && team.nRanks > 0) {
    pccl::utility::computePointerPairAlignmentWithFallback<T>(
        srcLambda(0), dstLambda(0), count, alignOffset, maxPackBytes);
  }

  constexpr int nDst = 1;
  pccl::utility::reduceCopy<T, pccl::utility::OpSum<T>, Coop, false, false, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, team.nRanks, dstLambda, nDst, pccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

// [ID 3.2b] LSA ReduceSum: N sources from devComm -> 1 local destination (with pcclDevComm_t)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSum(Coop coop,
                                            pcclSymPtr<T> src,
                                            T* dstPtr,
                                            IntCount count,
                                            pcclDevComm_t devComm) {
  // Extract team from devComm
  pcclTeam team = pcclTeamLsa(devComm);

  pcclLsaReduceSum<T, Coop, IntCount, UNROLL>(coop, src, dstPtr, count, team);
}

// [ID 3.2c] LSA ReduceSum: N sources from window+offset -> 1 local destination (with pcclWindow_t + pcclTeam)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSum(Coop coop,
                                            pcclWindow_t window,
                                            size_t offset,
                                            T* dstPtr,
                                            IntCount count,
                                            pcclTeam team) {
  // Construct pcclSymPtr from window and offset using direct initialization
  pcclSymPtr<T> src{window, offset};

  pcclLsaReduceSum<T, Coop, IntCount, UNROLL>(coop, src, dstPtr, count, team);
}

// [ID 3.2d] LSA ReduceSum: N sources from window+offset -> 1 local destination (with pcclWindow_t + pcclDevComm_t)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSum(Coop coop,
                                            pcclWindow_t window,
                                            size_t offset,
                                            T* dstPtr,
                                            IntCount count,
                                            pcclDevComm_t devComm) {
  // Construct pcclSymPtr from window and offset using direct initialization
  pcclSymPtr<T> src{window, offset};

  pcclLsaReduceSum<T, Coop, IntCount, UNROLL>(coop, src, dstPtr, count, devComm);
}

// [ID 3.3a] Multimem ReduceSum: 1 multimem source -> 1 local destination (with pcclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemReduceSum(Coop coop,
                                                 pcclSymPtr<T> src,
                                                 T* dstPtr,
                                                 IntCount count,
                                                 pcclMultimemHandle multimemHandle) {
  assert(false && "multimem is not supported on PPU");
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return src.multimemPtr(multimemHandle);
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dstPtr;
  };
  constexpr int nSrc = 1;
  constexpr int nDst = 1;
  pcclMultimemReduceSumLsaCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 3.3b] Multimem ReduceSum: 1 multimem source -> 1 local destination (with raw pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemReduceSum(Coop coop,
                                                 T* mcSrcPtr,
                                                 T* dstPtr,
                                                 IntCount count) {
  assert(false && "multimem is not supported on PPU");
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return mcSrcPtr;
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dstPtr;
  };
  constexpr int nSrc = 1;
  constexpr int nDst = 1;
  pcclMultimemReduceSumLsaCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 3.3c] Multimem ReduceSum: 1 multimem source -> 1 local destination (with pcclWindow_t)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemReduceSum(Coop coop,
                                                 pcclWindow_t window,
                                                 size_t offset,
                                                 T* dstPtr,
                                                 IntCount count,
                                                 pcclMultimemHandle multimemHandle) {
  assert(false && "multimem is not supported on PPU");
  pcclSymPtr<T> src{window, offset};
  pcclMultimemReduceSum<T, Coop, IntCount, UNROLL>(coop, src, dstPtr, count, multimemHandle);
}

// [ID 3.4] Local ReduceSum: N local chunks -> 1 local destination (lambda-based)
template<typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLocalReduceSum(Coop coop,
                                              SrcLambda srcLambda, int nSrc,
                                              T* dstPtr,
                                              IntCount count) {
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dstPtr;
  };
  constexpr int nDst = 1;  // Reduce has single destination
  pcclLsaReduceSumLsaCopy<T, Coop, SrcLambda, decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 3.5] Local ReduceSum: reduce n chunks separated by displacement
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLocalReduceSum(Coop coop,
                                              int nSrc,
                                              T* basePtr,
                                              size_t displ,
                                              T* dstPtr,
                                              IntCount count) {
  // Fast alignment computation for strided addressing with fallback
  IntCount alignOffset;
  int maxPackBytes;
  pccl::utility::computeStridedAlignmentWithFallback<T>(
      basePtr, displ * sizeof(T), count, alignOffset, maxPackBytes);

  // Create lambda: n local sources separated by displ
  auto srcLambda = [=] __device__ (int i) -> T* {
    return basePtr + i * displ;
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dstPtr;
  };

  constexpr int nDst = 1;
  pccl::utility::reduceCopy<T, pccl::utility::OpSum<T>, Coop, false, false, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, pccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

// ============================================================================
// SERIES 4.x - Copy (Broadcast) (1->N)
// ============================================================================

// [ID 4.1] Lambda-based version
template<typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaCopy(Coop coop,
                                       T* srcPtr,
                                       DstLambda dstLambda, int nDst,
                                       IntCount count) {
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return srcPtr;
  };
  constexpr int nSrc = 1;  // Copy has single source
  pcclLsaReduceSumLsaCopy<T, Coop, decltype(srcLambda), DstLambda, IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 4.2a] LSA Copy: 1 local source -> N destinations (with pcclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaCopy(Coop coop,
                                       T* srcPtr,
                                       pcclSymPtr<T> dst,
                                       IntCount count,
                                       pcclTeam team) {
  // Create lambda: N destinations to team
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return srcPtr;
  };
  auto dstLambda = [=] __device__ (int i) -> T* {
    return dst.peerPtr(team, i);
  };

  // LSA translation ensures all peers have the same alignment
  // Only need to check the first pointer (much cheaper than checking all N)
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && team.nRanks > 0) {
    pccl::utility::computePointerPairAlignmentWithFallback<T>(
        srcLambda(0), dstLambda(0), count, alignOffset, maxPackBytes);
  }

  constexpr int nSrc = 1;
  pccl::utility::reduceCopy<T, pccl::utility::OpSum<T>, Coop, false, false, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, team.nRanks, pccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

// [ID 4.2b] LSA Copy: 1 local source -> N destinations (with pcclDevComm_t)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaCopy(Coop coop,
                                       T* srcPtr,
                                       pcclSymPtr<T> dst,
                                       IntCount count,
                                       pcclDevComm_t devComm) {
  // Extract team from devComm
  pcclTeam team = pcclTeamLsa(devComm);

  pcclLsaCopy<T, Coop, IntCount, UNROLL>(coop, srcPtr, dst, count, team);
}

// [ID 4.2c] LSA Copy: 1 local source -> N destinations (with pcclWindow_t + pcclTeam)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaCopy(Coop coop,
                                       T* srcPtr,
                                       pcclWindow_t window,
                                       size_t offset,
                                       IntCount count,
                                       pcclTeam team) {
  // Construct pcclSymPtr from window and offset
  pcclSymPtr<T> dst{window, offset};

  pcclLsaCopy<T, Coop, IntCount, UNROLL>(coop, srcPtr, dst, count, team);
}

// [ID 4.2d] LSA Copy: 1 local source -> N destinations (with pcclWindow_t + pcclDevComm_t)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaCopy(Coop coop,
                                       T* srcPtr,
                                       pcclWindow_t window,
                                       size_t offset,
                                       IntCount count,
                                       pcclDevComm_t devComm) {
  // Construct pcclSymPtr from window and offset
  pcclSymPtr<T> dst{window, offset};

  pcclLsaCopy<T, Coop, IntCount, UNROLL>(coop, srcPtr, dst, count, devComm);
}

// [ID 4.3a] Multimem Copy: 1 local source -> 1 multimem destination (with pcclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemCopy(Coop coop,
                                            T* srcPtr,
                                            pcclSymPtr<T> dst,
                                            IntCount count,
                                            pcclMultimemHandle multimemHandle) {
  assert(false && "multimem is not supported on PPU");
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return srcPtr;
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dst.multimemPtr(multimemHandle);
  };
  constexpr int nSrc = 1;
  constexpr int nDst = 1;
  pcclLsaReduceSumMultimemCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 4.3b] Multimem Copy: 1 local source -> 1 multimem destination (with raw pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemCopy(Coop coop,
                                            T* srcPtr,
                                            T* mcDstPtr,
                                            IntCount count) {
  assert(false && "multimem is not supported on PPU");
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return srcPtr;
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return mcDstPtr;
  };
  constexpr int nSrc = 1;
  constexpr int nDst = 1;
  pcclLsaReduceSumMultimemCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 4.3c] Multimem Copy: 1 local source -> 1 multimem destination (with pcclWindow_t)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemCopy(Coop coop,
                                            T* srcPtr,
                                            pcclWindow_t window,
                                            size_t offset,
                                            IntCount count,
                                            pcclMultimemHandle multimemHandle) {
  assert(false && "multimem is not supported on PPU");
  pcclSymPtr<T> dst{window, offset};
  pcclMultimemCopy<T, Coop, IntCount, UNROLL>(coop, srcPtr, dst, count, multimemHandle);
}

// [ID 4.4] Local Copy: 1 source -> N local destinations (lambda-based)
template<typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLocalCopy(Coop coop,
                                         T* srcPtr,
                                         DstLambda dstLambda, int nDst,
                                         IntCount count) {
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return srcPtr;
  };
  constexpr int nSrc = 1;  // Copy has single source
  pcclLsaReduceSumLsaCopy<T, Coop, decltype(srcLambda), DstLambda, IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 4.5] Local Copy: copy to n chunks separated by displacement
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLocalCopy(Coop coop,
                                         T* srcPtr,
                                         int nDst,
                                         T* basePtr,
                                         size_t displ,
                                         IntCount count) {
  // Create lambda: n local destinations separated by displ
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return srcPtr;
  };
  auto dstLambda = [=] __device__ (int i) -> T* {
    return basePtr + i * displ;
  };

  constexpr int nSrc = 1;
  // Compute alignment across src and all strided destinations
  auto alignment = pccl::utility::computeLambdaAlignmentOffsetWithFallback<T>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);
  pccl::utility::reduceCopy<T, pccl::utility::OpSum<T>, Coop, false, false, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, pccl::utility::OpSum<T>{},
      count, alignment.alignOffset, alignment.maxPackBytes);
}

// ============================================================================
// SERIES 5.x - ReduceSumCopy (N->M)
// ============================================================================

// [ID 5.1a] LSA ReduceSumCopy: same team for src and dst (most common case)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSumCopy(Coop coop,
                                                pcclSymPtr<T> src,
                                                pcclSymPtr<T> dst,
                                                IntCount count,
                                                pcclTeam team) {
  auto srcLambda = [=] __device__ (int i) -> T* {
    return src.peerPtr(team, i);
  };
  auto dstLambda = [=] __device__ (int i) -> T* {
    return dst.peerPtr(team, i);
  };

  // LSA translation ensures all peers have the same alignment
  // Only need to check the first src and dst pointers (much cheaper than checking all N)
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && team.nRanks > 0) {
    pccl::utility::computePointerPairAlignmentWithFallback<T>(
        srcLambda(0), dstLambda(0), count, alignOffset, maxPackBytes);
  }

  pccl::utility::reduceCopy<T, pccl::utility::OpSum<T>, Coop, false, false, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, team.nRanks, dstLambda, team.nRanks, pccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

// [ID 5.1b] LSA ReduceSumCopy: with pcclDevComm_t (extract team)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSumCopy(Coop coop,
                                                pcclSymPtr<T> src,
                                                pcclSymPtr<T> dst,
                                                IntCount count,
                                                pcclDevComm_t devComm) {
  pcclTeam team = pcclTeamLsa(devComm);
  pcclLsaReduceSumCopy<T, Coop, IntCount, UNROLL>(coop, src, dst, count, team);
}

// [ID 5.1c] LSA ReduceSumCopy: with windows + pcclTeam
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSumCopy(Coop coop,
                                                pcclWindow_t srcWindow, size_t srcOffset,
                                                pcclWindow_t dstWindow, size_t dstOffset,
                                                IntCount count,
                                                pcclTeam team) {
  // Construct pcclSymPtr from window and offset using direct initialization
  pcclSymPtr<T> src{srcWindow, srcOffset};
  pcclSymPtr<T> dst{dstWindow, dstOffset};
  pcclLsaReduceSumCopy<T, Coop, IntCount, UNROLL>(coop, src, dst, count, team);
}

// [ID 5.1d] LSA ReduceSumCopy: with windows + pcclDevComm_t
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSumCopy(Coop coop,
                                                pcclWindow_t srcWindow, size_t srcOffset,
                                                pcclWindow_t dstWindow, size_t dstOffset,
                                                IntCount count,
                                                pcclDevComm_t devComm) {
  // Construct pcclSymPtr from window and offset using direct initialization
  pcclSymPtr<T> src{srcWindow, srcOffset};
  pcclSymPtr<T> dst{dstWindow, dstOffset};
  pcclLsaReduceSumCopy<T, Coop, IntCount, UNROLL>(coop, src, dst, count, devComm);
}

// [ID 5.1e] LSA ReduceSumCopy: different teams for src and dst (advanced use case)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSumCopy(Coop coop,
                                                pcclSymPtr<T> src, pcclTeam srcTeam,
                                                pcclSymPtr<T> dst, pcclTeam dstTeam,
                                                IntCount count) {
  auto srcLambda = [=] __device__ (int i) -> T* {
    return src.peerPtr(srcTeam, i);
  };
  auto dstLambda = [=] __device__ (int i) -> T* {
    return dst.peerPtr(dstTeam, i);
  };

  // LSA translation ensures all peers have the same alignment within each team
  // Always check both src and dst pointers together for relative alignment when both are available
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && srcTeam.nRanks > 0 && dstTeam.nRanks > 0) {
    void* srcPtr = srcLambda(0);
    void* dstPtr = dstLambda(0);
    pccl::utility::computePointerPairAlignmentWithFallback<T>(
        srcPtr, dstPtr, count, alignOffset, maxPackBytes);
  }

  pccl::utility::reduceCopy<T, pccl::utility::OpSum<T>, Coop, false, false, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, srcTeam.nRanks, dstLambda, dstTeam.nRanks, pccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemReduceSumCopy(Coop coop,
                                                     pcclSymPtr<T> src, pcclMultimemHandle srcHandle,
                                                     pcclSymPtr<T> dst, pcclMultimemHandle dstHandle,
                                                     IntCount count) {
  assert(false && "multimem is not supported on PPU");
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return src.multimemPtr(srcHandle);
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dst.multimemPtr(dstHandle);
  };
  constexpr int nSrc = 1;
  constexpr int nDst = 1;
  pcclMultimemReduceSumMultimemCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 5.2b] Multimem ReduceSumCopy (with raw pointers)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemReduceSumCopy(Coop coop,
                                                     T* mcSrcPtr,
                                                     T* mcDstPtr,
                                                     IntCount count) {
  assert(false && "multimem is not supported on PPU");
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return mcSrcPtr;
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return mcDstPtr;
  };
  constexpr int nSrc = 1;
  constexpr int nDst = 1;
  pcclMultimemReduceSumMultimemCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 5.2c] Multimem ReduceSumCopy (with pcclWindow_t)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemReduceSumCopy(Coop coop,
                                                     pcclWindow_t srcWindow, size_t srcOffset, pcclMultimemHandle srcHandle,
                                                     pcclWindow_t dstWindow, size_t dstOffset, pcclMultimemHandle dstHandle,
                                                     IntCount count) {
  assert(false && "multimem is not supported on PPU");
  pcclSymPtr<T> src{srcWindow, srcOffset};
  pcclSymPtr<T> dst{dstWindow, dstOffset};
  pcclMultimemReduceSumCopy<T, Coop, IntCount, UNROLL>(coop, src, srcHandle, dst, dstHandle, count);
}

// [ID 5.3a] LSA source -> Multimem destination
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSumMultimemCopy(Coop coop,
                                                        pcclSymPtr<T> src, pcclTeam srcTeam,
                                                        pcclSymPtr<T> dst, pcclMultimemHandle dstHandle,
                                                        IntCount count) {
  assert(false && "multimem is not supported on PPU");
  auto srcLambda = [=] __device__ (int i) -> T* {
    return src.peerPtr(srcTeam, i);
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dst.multimemPtr(dstHandle);
  };
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && srcTeam.nRanks > 0) {
    void* srcPtr = srcLambda(0);
    void* dstPtr = dstLambda(0);
    pccl::utility::computePointerPairAlignmentWithFallback<T>(
        srcPtr, dstPtr, count, alignOffset, maxPackBytes);
  }
  constexpr int nDst = 1;
  pccl::utility::reduceCopy<T, pccl::utility::OpSum<T>, Coop, false, true, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, srcTeam.nRanks, dstLambda, nDst, pccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

// [ID 5.3b] LSA source -> Multimem destination (with raw dst pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSumMultimemCopy(Coop coop,
                                                        pcclSymPtr<T> src, pcclTeam srcTeam,
                                                        T* mcDstPtr,
                                                        IntCount count) {
  assert(false && "multimem is not supported on PPU");
  auto srcLambda = [=] __device__ (int i) -> T* {
    return src.peerPtr(srcTeam, i);
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return mcDstPtr;
  };
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && srcTeam.nRanks > 0) {
    void* srcPtr = srcLambda(0);
    void* dstPtr = dstLambda(0);
    pccl::utility::computePointerPairAlignmentWithFallback<T>(
        srcPtr, dstPtr, count, alignOffset, maxPackBytes);
  }
  constexpr int nDst = 1;
  pccl::utility::reduceCopy<T, pccl::utility::OpSum<T>, Coop, false, true, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, srcTeam.nRanks, dstLambda, nDst, pccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

// [ID 5.3c] Multimem source -> LSA destination
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemReduceSumLsaCopy(Coop coop,
                                                        pcclSymPtr<T> src, pcclMultimemHandle srcHandle,
                                                        pcclSymPtr<T> dst, pcclTeam dstTeam,
                                                        IntCount count) {
  assert(false && "multimem is not supported on PPU");
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return src.multimemPtr(srcHandle);
  };
  auto dstLambda = [=] __device__ (int i) -> T* {
    return dst.peerPtr(dstTeam, i);
  };
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && dstTeam.nRanks > 0) {
    void* srcPtr = srcLambda(0);
    void* dstPtr = dstLambda(0);
    pccl::utility::computePointerPairAlignmentWithFallback<T>(
        srcPtr, dstPtr, count, alignOffset, maxPackBytes);
  }
  constexpr int nSrc = 1;
  pccl::utility::reduceCopy<T, pccl::utility::OpSum<T>, Coop, true, false, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, dstTeam.nRanks, pccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

// [ID 5.3d] Multimem source -> LSA destination (with raw src pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemReduceSumLsaCopy(Coop coop,
                                                        T* mcSrcPtr,
                                                        pcclSymPtr<T> dst, pcclTeam dstTeam,
                                                        IntCount count) {
  assert(false && "multimem is not supported on PPU");
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return mcSrcPtr;
  };
  auto dstLambda = [=] __device__ (int i) -> T* {
    return dst.peerPtr(dstTeam, i);
  };
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && dstTeam.nRanks > 0) {
    void* srcPtr = srcLambda(0);
    void* dstPtr = dstLambda(0);
    pccl::utility::computePointerPairAlignmentWithFallback<T>(
        srcPtr, dstPtr, count, alignOffset, maxPackBytes);
  }
  constexpr int nSrc = 1;
  pccl::utility::reduceCopy<T, pccl::utility::OpSum<T>, Coop, true, false, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, dstTeam.nRanks, pccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
} 

// [ID 5.4] Local ReduceSumCopy: N local sources -> M local destinations
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLocalReduceSumCopy(Coop coop,
                                                  int nSrc, T* srcBasePtr, size_t srcDispl,
                                                  int nDst, T* dstBasePtr, size_t dstDispl,
                                                  IntCount count) {
  // Fast alignment computation for strided addressing with fallback
  IntCount alignOffset = count;
  int maxPackBytes = static_cast<int>(sizeof(T));  // Default to scalar if nothing works

  // Try each pack size from largest to smallest using explicit template instantiations
  if (pccl::utility::tryComplexStridedAlignmentForPackSize<T, 16>(
          srcBasePtr, srcDispl * sizeof(T), dstBasePtr, dstDispl * sizeof(T), alignOffset, maxPackBytes)) {
    // Found working pack size
  } else if (pccl::utility::tryComplexStridedAlignmentForPackSize<T, 4>(
          srcBasePtr, srcDispl * sizeof(T), dstBasePtr, dstDispl * sizeof(T), alignOffset, maxPackBytes)) {
    // Found working pack size
  }

  auto srcLambda = [=] __device__ (int i) -> T* {
    return srcBasePtr + i * srcDispl;
  };
  auto dstLambda = [=] __device__ (int i) -> T* {
    return dstBasePtr + i * dstDispl;
  };

  pccl::utility::reduceCopy<T, pccl::utility::OpSum<T>, Coop, false, false, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, pccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}
#else // __HGGCCC_EXTENDED_LAMBDA__

// SERIES 1.x - Generic ReduceCopy with RedOp (LSA sources only)
template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename RedOp, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceLsaCopy(
    Coop, SrcLambda, int, DstLambda, int, RedOp const&, IntCount) {
  // C++11 - C++17 considers this a template invalid cannot have a valid specialation.
  // By making this a dependent static_assert, there may exists an overload of always_false that is true, it is valid.
  // "The validity of a template checked prior to any instantiation."
  // C++20+ may make an exception for static_assert for this exact situation. https://eel.is/c++draft/temp.res#general-6
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename RedOp, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceMultimemCopy(
    Coop, SrcLambda, int, DstLambda, int, RedOp const&, IntCount) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// SERIES 2.x - Sum-Specific ReduceCopy (lambda-based foundation)
template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSumLsaCopy(
    Coop, SrcLambda, int, DstLambda, int, IntCount) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSumMultimemCopy(
    Coop, SrcLambda, int, DstLambda, int, IntCount) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemReduceSumLsaCopy(
    Coop, SrcLambda, int, DstLambda, int, IntCount) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemReduceSumMultimemCopy(
    Coop, SrcLambda, int, DstLambda, int, IntCount) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// SERIES 3.x - ReduceSum (N->1)
template<typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSum(
    Coop, SrcLambda, int, T*, IntCount) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSum(
    Coop, pcclSymPtr<T>, T*, IntCount, pcclTeam) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSum(
    Coop, pcclSymPtr<T>, T*, IntCount, pcclDevComm_t) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSum(
    Coop, pcclWindow_t, size_t, T*, IntCount, pcclTeam) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSum(
    Coop, pcclWindow_t, size_t, T*, IntCount, pcclDevComm_t) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemReduceSum(
    Coop, pcclSymPtr<T>, T*, IntCount, pcclMultimemHandle) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemReduceSum(
    Coop, T*, T*, IntCount) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemReduceSum(
    Coop, pcclWindow_t, size_t, T*, IntCount, pcclMultimemHandle) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 3.4] Local ReduceSum (lambda-based)
template<typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLocalReduceSum(
    Coop, SrcLambda, int, T*, IntCount) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 3.5] Local ReduceSum (strided)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLocalReduceSum(
    Coop, int, T*, size_t, T*, IntCount) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// SERIES 4.x - Copy/Broadcast (1->N)

// 4.1] LSA Copy (lambda-based)
template<typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaCopy(
    Coop, T*, DstLambda, int, IntCount) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 4.2a] LSA Copy (with pcclSymPtr + team)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaCopy(
    Coop, T*, pcclSymPtr<T>, IntCount, pcclTeam) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 4.2b] LSA Copy (with pcclSymPtr + devComm)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaCopy(
    Coop, T*, pcclSymPtr<T>, IntCount, pcclDevComm_t) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 4.2c] LSA Copy (with window + offset + team)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaCopy(
    Coop, T*, pcclWindow_t, size_t, IntCount, pcclTeam) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 4.2d] LSA Copy (with window + offset + devComm)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaCopy(
    Coop, T*, pcclWindow_t, size_t, IntCount, pcclDevComm_t) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 4.3a] Multimem Copy (with pcclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemCopy(
    Coop, T*, pcclSymPtr<T>, IntCount, pcclMultimemHandle) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 4.3b] Multimem Copy (with raw pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemCopy(
    Coop, T*, T*, IntCount) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 4.3c] Multimem Copy (with window + offset)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemCopy(
    Coop, T*, pcclWindow_t, size_t, IntCount, pcclMultimemHandle) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 4.4] Local Copy (lambda-based)
template<typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLocalCopy(
    Coop, T*, DstLambda, int, IntCount) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 4.5] Local Copy (strided)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLocalCopy(
    Coop, T*, int, T*, size_t, IntCount) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// SERIES 5.x - ReduceSumCopy (N->M)

// 5.1a] LSA ReduceSumCopy (same team)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSumCopy(
    Coop, pcclSymPtr<T>, pcclSymPtr<T>, IntCount, pcclTeam) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.1b] LSA ReduceSumCopy (with devComm)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSumCopy(
    Coop, pcclSymPtr<T>, pcclSymPtr<T>, IntCount, pcclDevComm_t) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.1c] LSA ReduceSumCopy (with windows + team)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSumCopy(
    Coop, pcclWindow_t, size_t, pcclWindow_t, size_t, IntCount, pcclTeam) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.1d] LSA ReduceSumCopy (with windows + devComm)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSumCopy(
    Coop, pcclWindow_t, size_t, pcclWindow_t, size_t, IntCount, pcclDevComm_t) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.1e] LSA ReduceSumCopy (different teams)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSumCopy(
    Coop, pcclSymPtr<T>, pcclTeam, pcclSymPtr<T>, pcclTeam, IntCount) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.2a] Multimem ReduceSumCopy (with pcclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemReduceSumCopy(
    Coop, pcclSymPtr<T>, pcclMultimemHandle, pcclSymPtr<T>, pcclMultimemHandle, IntCount) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.2b] Multimem ReduceSumCopy (with raw pointers)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemReduceSumCopy(
    Coop, T*, T*, IntCount) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.2c] Multimem ReduceSumCopy (with windows)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemReduceSumCopy(
    Coop, pcclWindow_t, size_t, pcclMultimemHandle, pcclWindow_t, size_t, pcclMultimemHandle, IntCount) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.3a] LSA -> Multimem ReduceSumCopy (with pcclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSumMultimemCopy(
    Coop, pcclSymPtr<T>, pcclTeam, pcclSymPtr<T>, pcclMultimemHandle, IntCount) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.3b] LSA -> Multimem ReduceSumCopy (with raw dst pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLsaReduceSumMultimemCopy(
    Coop, pcclSymPtr<T>, pcclTeam, T*, IntCount) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.3c] Multimem -> LSA ReduceSumCopy (with pcclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemReduceSumLsaCopy(
    Coop, pcclSymPtr<T>, pcclMultimemHandle, pcclSymPtr<T>, pcclTeam, IntCount) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.3d] Multimem -> LSA ReduceSumCopy (with raw src pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclMultimemReduceSumLsaCopy(
    Coop, T*, pcclSymPtr<T>, pcclTeam, IntCount) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.4] Local ReduceSumCopy (strided)
template<typename T, typename Coop, typename IntCount, int UNROLL>
PCCL_DEVICE_INLINE void pcclLocalReduceSumCopy(
    Coop, int, T*, size_t, int, T*, size_t, IntCount) {
  static_assert(pccl::utility::always_false<T>::value,
     "PCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

#endif // __HGGCCC_EXTENDED_LAMBDA__
#endif // PCCL_CHECK_HGGCCC

#endif // _PCCL_DEVICE_REDUCE_COPY__FUNCS_H_
