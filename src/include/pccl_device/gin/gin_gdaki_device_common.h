/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_GIN_GDAKI_DEVICE_COMMON_H_
#define PCCL_GIN_GDAKI_DEVICE_COMMON_H_

#include "gin_device_common.h"

#if PCCL_CHECK_HGGCCC
template <pcclGinGdakiProvider provider>
struct pcclGinGdakiApi_Wait {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx, pcclGinRequest_t& outRequest, bool hasDescriptor,
                                      pcclGinDescriptorSmem* descriptor, hggc::memory_order ord, uint32_t* abortFlag);
};

template <pcclGinGdakiProvider provider>
struct pcclGinGdakiApi_FlushAsync {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx, uint32_t peer, pcclGinRequest_t* outRequest, uint32_t optFlags);
};

template <pcclGinGdakiProvider provider>
struct pcclGinGdakiApi_Get {
  template <typename Coop>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx, Coop coop, int peer, pcclGinWindow_t remoteWin, size_t remoteOff,
                                      pcclGinWindow_t localWin, size_t localOff, size_t bytes,
                                      bool hasDescriptor, pcclGinDescriptorSmem* descriptor,
                                      uint32_t optFlags = pcclGinOptFlagsDefault);
};

template <pcclGinGdakiProvider provider>
struct pcclGinGdakiApi_Put {
  template <typename Coop>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx, Coop coop, int peer, bool hasWins,
                                      pcclGinWindow_t dstWin, size_t dstOff, pcclGinWindow_t srcWin,
                                      size_t srcOff, size_t bytes,
                                      pcclGinSignalDescriptor signal, pcclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasCounter,
                                      pcclGinCounter_t counterId, bool hasDescriptor,
                                      pcclGinDescriptorSmem* descriptor,
                                      hggc::thread_scope required, hggc::thread_scope given,
                                      uint32_t optFlags = pcclGinOptFlagsDefault);
};

template <pcclGinGdakiProvider provider>
struct pcclGinGdakiApi_PutValue {
  template <typename Coop, typename T>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx, Coop coop, int peer, pcclGinWindow_t dstWin,
                                      size_t dstOff, T srcData,
                                      pcclGinSignalDescriptor signal, pcclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor,
                                      pcclGinDescriptorSmem* descriptor,
                                      hggc::thread_scope required, hggc::thread_scope given,
                                      uint32_t optFlags = pcclGinOptFlagsDefault);
};

template <pcclGinGdakiProvider provider>
struct pcclGinGdakiApi_GetSignalPtr {
  PCCL_DEVICE_INLINE static uint64_t* call(pcclGinCtx, pcclGinSignal_t signalId);
};

template <pcclGinGdakiProvider provider>
struct pcclGinGdakiApi_GetCounterPtr {
  PCCL_DEVICE_INLINE static uint64_t* call(pcclGinCtx, pcclGinCounter_t counterId);
};

// Counter observation is backend-specific: some NICs update the counter
// autonomously, while others must first consume their send CQ on the GPU.
template <pcclGinGdakiProvider provider>
struct pcclGinGdakiApi_ReadCounter {
  PCCL_DEVICE_INLINE static uint64_t call(pcclGinCtx ctx, pcclGinCounter_t counter,
                                          int bits, hggc::memory_order ord) {
    uint64_t* ptr = pcclGinGdakiApi_GetCounterPtr<provider>::call(ctx, counter);
    uint64_t mask = uint64_t(-1) >> (64 - bits);
    return mask & hggc::atomic_ref<uint64_t>{*ptr}.load(ord);
  }
};

template <pcclGinGdakiProvider provider>
struct pcclGinGdakiApi_WaitCounter {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, pcclGinCounter_t counter,
                                      uint64_t least, int bits, hggc::memory_order ord,
                                      uint32_t* abortFlag) {
    uint32_t steps = 0;
    uint64_t* ptr = pcclGinGdakiApi_GetCounterPtr<provider>::call(ctx, counter);
    uint64_t got;
    #pragma unroll 1
    do got = hggc::atomic_ref<uint64_t>{*ptr}.load(ord);
    while (!pccl::utility::rollingLessEq(least, got, bits) &&
           !pccl::utility::testAbort(abortFlag, steps));
  }
};

template <pcclGinGdakiProvider provider>
struct pcclGinGdakiApi_ResetSignal {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx, pcclGinSignalDescriptor signal);
};

template <pcclGinGdakiProvider provider>
struct pcclGinGdakiApi_ResetCounter {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx, pcclGinCounter_t counterId);
};

template <pcclGinGdakiProvider provider>
struct pcclGinGdakiApi_Flush {
  template <typename Coop>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx, Coop, hggc::memory_order ord, uint32_t* abortFlag);
};

#endif
#endif
