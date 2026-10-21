/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_GIN_PROXY_H_
#define _PCCL_DEVICE_GIN_PROXY_H_

//#include <config.h>

#include <cstdint>
#include <hggc_runtime.h>
#include <cooperative_groups.h>
#include "pccl.h"
#include "pccl_device/gin/gin_device_common.h"
#include "pccl_device/utility.h"
#include "../gin_device_host_common.h"
#include "gin_proxy_device_host_common.h"

struct pcclGinCpuProxyRequest {
  int peer;
  uint32_t nextGfdIdx;
};
static_assert(sizeof(pcclGinCpuProxyRequest) <= sizeof(pcclGinRequest_t),
              "pcclGinCpuProxyRequest must fit in pcclGinRequest_t");

namespace pccl {
namespace gin {
namespace proxy {

// Chunk size for Gin Proxy GFD operations
static constexpr size_t DataChunkSize = 1ULL << 30;  // 1 GB

PCCL_DEVICE_INLINE void waitForGfdComplete(pcclGinProxyGpuCtx_t* proxyCtx, uint32_t pe, uint32_t nextGfdIdx, hggc::memory_order ord, uint32_t* abortFlag) {
  using pccl::utility::loadConst;
  using pccl::utility::rollingLessEq;
  using pccl::utility::testAbort;
  hggc::atomic_ref<uint32_t, hggc::thread_scope_system> ci(loadConst(&proxyCtx->cis)[pe]);
  uint32_t steps = 0;
  // The PI and CI can keep moving because of concurrent threads posting GFDs to this queue, and the CPU consuming them.
  // Therefore, to prevent overflow issues in the while statement, we need to use a special comparison function.
#pragma unroll 1
  while (!rollingLessEq<uint32_t>(nextGfdIdx, ci.load(ord)) && !testAbort(abortFlag, steps)) continue;
}

PCCL_DEVICE_INLINE void flush(pcclGinProxyGpuCtx_t* proxyCtx, uint32_t pe, hggc::memory_order ord, uint32_t* abortFlag) {
  using pccl::utility::loadConst;
  hggc::atomic_ref<uint32_t, hggc::thread_scope_system> pi(loadConst(&proxyCtx->pis)[pe]);
  hggc::atomic_ref<uint32_t, hggc::thread_scope_system> ci(loadConst(&proxyCtx->cis)[pe]);
  uint32_t p = pi.load(hggc::memory_order_relaxed);
  pccl::gin::proxy::waitForGfdComplete(proxyCtx, pe, p, ord, abortFlag);
}

template <typename Coop>
PCCL_DEVICE_INLINE void postGfd(Coop coop, pcclGinProxyGpuCtx_t* proxyCtx, pcclGinProxyGfd_t* gfd,
                                uint32_t pe, uint32_t* gfdIdx = nullptr) {
  using pccl::utility::loadConst;
  hggc::atomic_ref<uint32_t, hggc::thread_scope_system> pi(loadConst(&proxyCtx->pis)[pe]);
  hggc::atomic_ref<uint32_t, hggc::thread_scope_system> ci(loadConst(&proxyCtx->cis)[pe]);
  pcclGinProxyGfd_t* q = &loadConst(&proxyCtx->queues)[pe * proxyCtx->queueSize];
  uint32_t queueSize = loadConst(&proxyCtx->queueSize);
  if (coop.thread_rank() == 0) {
    // claim a slot in the gfd queue
    uint32_t idx = pi.fetch_add(1, hggc::memory_order_relaxed);
    if (gfdIdx != nullptr) {
      *gfdIdx = idx;
    }
    // wait for credits
    while (queueSize <= idx - ci.load(hggc::memory_order_relaxed)) {
    }
    idx &= queueSize - 1;
// 16 byte stores with the write-through cache hint
#pragma unroll
    for (uint8_t i = 0; i < sizeof(pcclGinProxyGfd_t) / sizeof(uint4); i++) {
      __stwt((uint4*)&q[idx] + i, ((uint4*)gfd)[i]);
    }
  }
}

template <typename T>
// Descriptor must be at least GWQ_GFD_SIZE bytes and it should be aligned
// Assumes little-endian, which is okay.
__device__ __forceinline__ void buildGfd(pcclGinProxyGfd_t* gfd, pcclGinProxyOp_t op, T srcVal,
                                         bool hasInline, size_t srcOff, pcclGinWindow_t srcHandle,
                                         size_t dstOff, pcclGinWindow_t dstHandle, size_t size,
                                         pcclGinCounter_t counterId, pcclGinSignal_t signalId,
                                         uint64_t signalVal, pcclGinWindow_t signalWindow,
                                         size_t signalOff) {

  for (int i = 0; i < pcclGinProxyGfdQwords; i++) {
    gfd->qword[i].flag.v = 1;
  }

  gfd->qword[pcclGinProxyGfdHeader].header.version = (uint64_t)PCCL_GIN_PROXY_GFD_VERSION;
  gfd->qword[pcclGinProxyGfdHeader].header.size = (uint64_t)size;
  gfd->qword[pcclGinProxyGfdHeaderExt].headerExt.op = (uint16_t)op;

  if (op & pcclGinProxyOpFlush) {
    return;
  }

  if (hasInline) {
    uint64_t srcValBits = 0;
    memcpy(&srcValBits, &srcVal, sizeof(T));
    gfd->qword[pcclGinProxyGfdInlineLow].inlineLow.inlineValLow = (uint32_t)srcValBits;
    if (sizeof(T) > 4)
      gfd->qword[pcclGinProxyGfdInlineLow].inlineLow.inlineValLow2 = (uint64_t)srcValBits >> 32;
    if (sizeof(T) > 6)
      gfd->qword[pcclGinProxyGfdInlineHigh].inlineHigh.inlineValHigh = (uint64_t)srcValBits >> 48;
  } else if (op & pcclGinProxyOpVASignal) {
    gfd->qword[pcclGinProxyGfdVASignalOff].vaSignalOff.vaSignalOff = (uint64_t)signalOff;
    gfd->qword[pcclGinProxyGfdVASignalHandle].vaSignalHandle.vaSignalHandle = (uint64_t)signalWindow;
  } else {
    gfd->qword[pcclGinProxyGfdSrcOff].srcOff.srcOff = (uint64_t)srcOff;
    gfd->qword[pcclGinProxyGfdSrcHandle].srcHandle.srcHandle = (uint64_t)srcHandle;
  }

  gfd->qword[pcclGinProxyGfdDstOff].dstOff.dstOff = (uint64_t)dstOff;
  gfd->qword[pcclGinProxyGfdDstHandle].dstHandle.dstHandle = (uint64_t)dstHandle;

  gfd->qword[pcclGinProxyGfdCompletion].completion.counterId = counterId;
  gfd->qword[pcclGinProxyGfdCompletion].completion.signalId = signalId;

  // The signal value is split between two qwords, as the signal value is a full 64 bits
  gfd->qword[pcclGinProxyGfdCompletion].completion.signalValLow = (uint16_t)signalVal;
  gfd->qword[pcclGinProxyGfdSignalVal].signalVal.signalValLow2 = (uint16_t)(signalVal >> 16);
  gfd->qword[pcclGinProxyGfdSignalVal].signalVal.signalValHigh = (uint32_t)(signalVal >> 32);
}

__device__ __forceinline__ void constructProxyOp(pcclGinProxyOp_t& op, bool isGet, bool isFlush, bool hasInline,
                                                 pcclGinSignalType signalType, pcclGinSignalOp_t signalOp,
                                                 bool hasCounter) {
  op = (pcclGinProxyOp_t)(0);
  if (isGet) {
    op = static_cast<pcclGinProxyOp_t>(static_cast<uint16_t>(op) | static_cast<uint16_t>(pcclGinProxyOpGet));
    return;
  }

  if (isFlush) {
    op = static_cast<pcclGinProxyOp_t>(static_cast<uint16_t>(op) | static_cast<uint16_t>(pcclGinProxyOpFlush));
    return;
  }

  if (signalType != PCCL_GIN_SIGNAL_TYPE_NONE) {
    switch (signalOp) {
      case pcclGinSignalInc:
        op = static_cast<pcclGinProxyOp_t>(static_cast<uint16_t>(op) | static_cast<uint16_t>(pcclGinProxyOpWithSignalInc));
        break;
      case pcclGinSignalAdd:
        op = static_cast<pcclGinProxyOp_t>(static_cast<uint16_t>(op) | static_cast<uint16_t>(pcclGinProxyOpWithSignalAdd));
        break;
      default:
        __builtin_unreachable();
    }
  }
  if (signalType == PCCL_GIN_SIGNAL_TYPE_VA) {
    op = static_cast<pcclGinProxyOp_t>(static_cast<uint16_t>(op) | static_cast<uint16_t>(pcclGinProxyOpVASignal));
    return;
  }
  op = static_cast<pcclGinProxyOp_t>(static_cast<uint16_t>(op) | static_cast<uint16_t>(pcclGinProxyOpPut));
  if (hasInline)
    op = static_cast<pcclGinProxyOp_t>(static_cast<uint16_t>(op) |
                                       static_cast<uint16_t>(pcclGinProxyOpWithInline));
  if (hasCounter)
    op = static_cast<pcclGinProxyOp_t>(static_cast<uint16_t>(op) |
                                       static_cast<uint16_t>(pcclGinProxyOpWithCounter));
}

template <typename Coop>
PCCL_DEVICE_INLINE void get(Coop coop, pcclGinProxyGpuCtx_t* proxyCtx,
                            int peer, pcclGinWindow_t remoteWnd, size_t remoteOff,
                            pcclGinWindow_t localWnd, size_t localOff, size_t bytes,
                            pcclGinProxyGfd_t* desc) {
  using pccl::gin::proxy::DataChunkSize;
  while (bytes > 0) {
    size_t sendSize = min(bytes, DataChunkSize);
    pcclGinProxyOp_t op;
    constructProxyOp(op, /*isGet*/true, /*isFlush*/false, /*hasInline*/false, PCCL_GIN_SIGNAL_TYPE_NONE, pcclGinSignalInc, /*hasCounter*/false);
    pccl::gin::proxy::buildGfd(desc, op, /*srcVal*/0, /*hasInline*/false, remoteOff, remoteWnd,
                               localOff, localWnd, sendSize, /*counterId*/0, /*signalId*/0,
                               /*signalVal*/0, nullptr, 0);
    pccl::gin::proxy::postGfd<Coop>(coop, proxyCtx, desc, peer);
    bytes -= sendSize;
    remoteOff += sendSize;
    localOff += sendSize;
  }
}

template <typename Coop, typename T>
PCCL_DEVICE_INLINE void put(Coop coop, pcclGinProxyGfd_t* gfd, pcclGinProxyGpuCtx_t* proxyCtx,
                            int peer, pcclGinWindow_t dstWnd, size_t dstOff, T srcVal,
                            bool hasInline, pcclGinWindow_t srcWnd, size_t srcOff, size_t bytes,
                            pcclGinSignalDescriptor signal, pcclGinSignalOp_t signalOp,
                            uint64_t signalVal, bool hasCounter, pcclGinCounter_t counterId,
                            hggc::thread_scope required, hggc::thread_scope given) {
  if ((int)given > (int)hggc::thread_scope_system) {
    hggc::atomic_thread_fence(hggc::memory_order_release, hggc::thread_scope_system);
  }

  using pccl::gin::proxy::DataChunkSize;
  while (bytes > DataChunkSize) {
    pcclGinProxyOp_t op;
    constructProxyOp(op, /*isGet*/false, /*isFlush*/false, /*hasInline*/false, PCCL_GIN_SIGNAL_TYPE_NONE, signalOp, /*hasCounter*/false);
    pccl::gin::proxy::buildGfd(gfd, op, /*srcVal*/0, /*hasInline*/false, srcOff, srcWnd,
                               dstOff, dstWnd, DataChunkSize, /*counterId*/0, /*signalId*/0,
                               /*signalVal*/0, nullptr, 0);
    pccl::gin::proxy::postGfd<Coop>(coop, proxyCtx, gfd, peer);
    bytes -= DataChunkSize;
    srcOff += DataChunkSize;
    dstOff += DataChunkSize;
  }

  pcclGinSignalType putSignalType;
  uint64_t putSignalVal;
  pcclGinSignal_t putSignalId;
  switch (signal.type) {
    case PCCL_GIN_SIGNAL_TYPE_INDEXED:
      putSignalType = PCCL_GIN_SIGNAL_TYPE_INDEXED;
      putSignalVal = signalVal;
      putSignalId = signal.indexedSignal.signalId;
      break;
    case PCCL_GIN_SIGNAL_TYPE_VA: // VA signals must be in a separate GFD. Use no signal during first put.
    case PCCL_GIN_SIGNAL_TYPE_NONE:
      putSignalType = PCCL_GIN_SIGNAL_TYPE_NONE;
      putSignalVal = 0;
      putSignalId = 0;
      break;
    default:
      __builtin_unreachable();
  }
  if (hasInline || hasCounter || srcWnd != nullptr || putSignalType != PCCL_GIN_SIGNAL_TYPE_NONE) {
    pcclGinProxyOp_t op;
    constructProxyOp(op, /*isGet*/false, /*isFlush*/false, hasInline, putSignalType, signalOp, hasCounter);
    pccl::gin::proxy::buildGfd(gfd, op, srcVal, hasInline, srcOff, srcWnd, dstOff, dstWnd, bytes,
                              hasCounter ? counterId : 0, putSignalId, putSignalVal, nullptr, 0);
    pccl::gin::proxy::postGfd<Coop>(coop, proxyCtx, gfd, peer);
  }

  // Handle additional GFD for VA signals.
  if (signal.type == PCCL_GIN_SIGNAL_TYPE_VA) {
    pcclGinProxyOp_t op;
    constructProxyOp(op, /*isGet*/false, /*isFlush*/false, /*hasInline*/false, PCCL_GIN_SIGNAL_TYPE_VA, signalOp, /*hasCounter*/false);
    pccl::gin::proxy::buildGfd(gfd, op, /*srcVal*/0, /*hasInline*/false, 0, nullptr,
                               0, nullptr, 0, 0, 0, signalVal, signal.vaSignal.signalWindow, signal.vaSignal.signalOffset);
    pccl::gin::proxy::postGfd<Coop>(coop, proxyCtx, gfd, peer);
  }
}
}  // namespace proxy
}  // namespace gin
}  // namespace pccl

template <>
struct pcclGinApi_Get<PCCL_NET_DEVICE_GIN_PROXY> {
  template <typename Coop>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, Coop coop, int peer, pcclGinWindow_t remoteWin, size_t remoteOff,
                                      pcclGinWindow_t localWin, size_t localOff, size_t bytes,
                                      bool hasDescriptor, pcclGinDescriptorSmem* descriptor,
                                      uint32_t optFlags) {
    pcclGinProxyGfd_t tmpDesc;
    pcclGinProxyGfd_t* desc = hasDescriptor ? (pcclGinProxyGfd_t*)descriptor : &tmpDesc;
    pcclGinProxyGpuCtx_t* proxyCtx = &((pcclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    pccl::gin::proxy::get<Coop>(coop, proxyCtx, peer, remoteWin, remoteOff, localWin, localOff, bytes,desc);
  }
};

template<>
struct pcclGinApi_FlushAsync<PCCL_NET_DEVICE_GIN_PROXY> {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, int peer, pcclGinRequest_t* outRequest, uint32_t optFlags) {
    using pccl::utility::loadConst;
    pcclGinCpuProxyRequest* req = reinterpret_cast<pcclGinCpuProxyRequest*>(outRequest);
    req->peer = peer;
    pcclGinProxyGpuCtx_t* proxyCtx = &((pcclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    hggc::atomic_ref<uint32_t, hggc::thread_scope_system> pi(loadConst(&proxyCtx->pis)[peer]);
    req->nextGfdIdx = pi.load(hggc::memory_order_relaxed);
  }
};
template <>
struct pcclGinApi_Wait<PCCL_NET_DEVICE_GIN_PROXY> {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, pcclGinRequest_t& request, bool hasDescriptor,
                                      pcclGinDescriptorSmem* descriptor, hggc::memory_order ord, uint32_t* abortFlag) {
    pcclGinCpuProxyRequest& req = reinterpret_cast<pcclGinCpuProxyRequest&>(request);
    pcclGinProxyGpuCtx_t* proxyCtx = &((pcclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    pccl::gin::proxy::waitForGfdComplete(proxyCtx, req.peer, req.nextGfdIdx, hggc::memory_order_relaxed, abortFlag);

    // Ensure gets are visible by issuing a local flush
    pcclGinProxyGfd_t tmpGfd;
    pcclGinProxyGfd_t* desc = hasDescriptor ? (pcclGinProxyGfd_t*)descriptor : &tmpGfd;
    pcclGinProxyOp_t op;
    uint32_t flushGfdIdx;
    pccl::gin::proxy::constructProxyOp(op, /*isGet*/false, /*isFlush*/true, /*hasInline*/false, PCCL_GIN_SIGNAL_TYPE_NONE, pcclGinSignalInc, /*hasCounter*/false);
    pccl::gin::proxy::buildGfd(desc, op, /*srcVal*/0, /*hasInline*/false, 0, nullptr,
                               0, nullptr, 0, 0, 0, 0, nullptr, 0);
    pccl::gin::proxy::postGfd(pcclCoopThread(), proxyCtx, desc, ctx.rank, &flushGfdIdx);
    pccl::gin::proxy::waitForGfdComplete(proxyCtx, ctx.rank, flushGfdIdx + 1, ord, abortFlag);
  }
};

template <>
struct pcclGinApi_GetCounterPtr<PCCL_NET_DEVICE_GIN_PROXY> {
  PCCL_DEVICE_INLINE static uint64_t* call(pcclGinCtx ctx, int peer, pcclGinCounter_t counterId) {
    (void)peer;
    pcclGinProxyGpuCtx_t* proxyCtx = &((pcclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    uint64_t* counter = pccl::utility::loadConst(&proxyCtx->counters) + counterId;
    return counter;
  }
};

template <>
struct pcclGinApi_ResetCounter<PCCL_NET_DEVICE_GIN_PROXY> {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, pcclGinCounter_t counterId) {
    pcclGinProxyGpuCtx_t* proxyCtx = &((pcclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    uint64_t* counter = pccl::utility::loadConst(&proxyCtx->counters) + counterId;
    *counter = 0;
  }
};

template <>
struct pcclGinApi_GetSignalPtr<PCCL_NET_DEVICE_GIN_PROXY> {
  PCCL_DEVICE_INLINE static uint64_t* call(pcclGinCtx ctx, int peer, pcclGinSignal_t signalId) {
    (void)peer;
    pcclGinProxyGpuCtx_t* proxyCtx = &((pcclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    uint64_t* signal = pccl::utility::loadConst(&proxyCtx->signals) + signalId;
    return signal;
  }
};

template <>
struct pcclGinApi_ResetSignal<PCCL_NET_DEVICE_GIN_PROXY> {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, pcclGinSignalDescriptor signal) {
    pcclGinProxyGpuCtx_t* proxyCtx = &((pcclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    if (signal.type == PCCL_GIN_SIGNAL_TYPE_VA) {
      uint64_t* signalPtr = (uint64_t*)pcclGetLocalPointer(signal.vaSignal.pcclWindow, signal.vaSignal.signalOffset);
      *signalPtr = 0;
    } else {
      pccl::utility::loadConst(&proxyCtx->signals)[signal.indexedSignal.signalId] = 0;
    }
  }
};

template <>
struct pcclGinApi_Flush<PCCL_NET_DEVICE_GIN_PROXY> {
  template <typename Coop>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, Coop coop, hggc::memory_order ord, uint32_t* abortFlag) {
    pcclGinProxyGpuCtx_t* proxyCtx = &((pcclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    // wait for all GFDs to be completed
#pragma unroll 1
    for (int pe = coop.thread_rank(); pe < ctx.nRanks; pe += coop.size()) {
      pccl::gin::proxy::flush(proxyCtx, pe, ord, abortFlag);
    }
  }
};

template <>
struct pcclGinApi_Put<PCCL_NET_DEVICE_GIN_PROXY> {
  template <typename Coop>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                      pcclGinWindow_t dstWin, size_t dstOff, pcclGinWindow_t srcWin,
                                      size_t srcOff, size_t bytes,
                                      pcclGinSignalDescriptor signal, pcclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasCounter,
                                      pcclGinCounter_t counterId, bool hasDescriptor,
                                      pcclGinDescriptorSmem* descriptor,
                                      hggc::thread_scope required, hggc::thread_scope given,
                                      uint32_t optFlags = pcclGinOptFlagsDefault) {
    pcclGinProxyGfd_t tmpDesc;
    pcclGinProxyGfd_t* desc = hasDescriptor ? (pcclGinProxyGfd_t*)descriptor : &tmpDesc;
    pcclGinProxyGpuCtx_t* proxyCtx = &((pcclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    pccl::gin::proxy::put<Coop, uint64_t>(coop, desc, proxyCtx, peer, dstWin, dstOff, 0, false,
                                          srcWin, srcOff, bytes, signal, signalOp, signalOpArg,
                                          hasCounter, counterId, required, given);
  }
};

template <>
struct pcclGinApi_PutValue<PCCL_NET_DEVICE_GIN_PROXY> {
  template <typename Coop, typename T>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, Coop coop, int peer, pcclGinWindow_t dstWin,
                                      size_t dstOff, T srcVal,
                                      pcclGinSignalDescriptor signal, pcclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor,
                                      pcclGinDescriptorSmem* descriptor,
                                      hggc::thread_scope required, hggc::thread_scope given,
                                      uint32_t optFlags = pcclGinOptFlagsDefault) {
    pcclGinProxyGfd_t tmpDesc;
    pcclGinProxyGfd_t* desc = hasDescriptor ? (pcclGinProxyGfd_t*)descriptor : &tmpDesc;
    pcclGinProxyGpuCtx_t* proxyCtx = &((pcclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    pccl::gin::proxy::put<Coop, T>(coop, desc, proxyCtx, peer, dstWin,
                                   dstOff, srcVal, true, nullptr, 0, sizeof(T), signal,
                                   signalOp, signalOpArg, false, 0, required, given);
  }
};

#endif
