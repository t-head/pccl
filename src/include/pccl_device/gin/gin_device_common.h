/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_GIN_DEVICE_COMMON_H_
#define _PCCL_GIN_DEVICE_COMMON_H_

#include <stdint.h>
#include "../net_device.h"
#include "../utility.h"
#include "gin_device_host_common.h"

#ifndef PCCL_GIN_PROXY_ENABLE
#define PCCL_GIN_PROXY_ENABLE 1
#endif

#ifndef PCCL_GIN_GDAKI_ENABLE
#define PCCL_GIN_GDAKI_ENABLE 1
#endif

enum pcclGinOptFlags {
  pcclGinOptFlagsDefault = 0,
  pcclGinOptFlagsMaySkipCreditCheck = (1 << 0),
  pcclGinOptFlagsAggregateRequests = (1 << 1),
};

#define PCCL_GIN_BACKEND_MASK_ALL                                               \
  (((PCCL_GIN_PROXY_ENABLE) ? 1u : 0u) << (unsigned)PCCL_NET_DEVICE_GIN_PROXY | \
   ((PCCL_GIN_GDAKI_ENABLE) ? 1u : 0u) << (unsigned)PCCL_NET_DEVICE_GIN_GDAKI)

// Resource sharing mode for a given pcclGin/pcclGin_C *instance*.
// This mode is selected at construction time and is carried by the pcclGin
// object, then copied into pcclGinCtx for each call. It is not stored as
// persistent per-context state in the communicator (i.e., different pcclGin
// instantiations that target the same contextIndex may use different modes).
enum pcclGinResourceSharingMode : uint8_t {
  PCCL_GIN_RESOURCE_SHARING_GPU = 0,
  PCCL_GIN_RESOURCE_SHARING_CTA = 1,
};

struct pcclGinCtx {
  unsigned backendMask;
  pcclNetDeviceType backend;
  int rank;
  int nRanks;
  void* handle;
  int contextId;
  uint8_t resourceSharingMode;
};

template <unsigned backendMask>
struct pcclGinCtx_M : pcclGinCtx {};

struct pcclGinDescriptorSmem {
  alignas(16) char space[64];
};

enum pcclGinSignalType {
  PCCL_GIN_SIGNAL_TYPE_NONE,
  PCCL_GIN_SIGNAL_TYPE_VA,
  PCCL_GIN_SIGNAL_TYPE_INDEXED,
};

struct pcclGinSignalDescriptor {
  pcclGinSignalType type;
  union {
    struct {
      pcclGinWindow_t signalWindow;
      size_t signalOffset;
      pcclWindow_t pcclWindow;
    } vaSignal;
    struct {
      pcclGinSignal_t signalId;
    } indexedSignal;
  };
};


#if PCCL_CHECK_HGGCCC
template <pcclNetDeviceType backend>
struct pcclGinApi_Wait {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx, pcclGinRequest_t& outRequest, bool hasDescriptor,
                                      pcclGinDescriptorSmem* descriptor, hggc::memory_order ord, uint32_t* abortFlag);
};

template <pcclNetDeviceType backend>
struct pcclGinApi_FlushAsync {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx, uint32_t peer, pcclGinRequest_t* outRequest, uint32_t optFlags);
};

template <pcclNetDeviceType backend>
struct pcclGinApi_Get {
  template <typename Coop>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx, Coop coop, int peer, pcclGinWindow_t remoteWin, size_t remoteOff,
                                      pcclGinWindow_t localWin, size_t localOff, size_t bytes,
                                      bool hasDescriptor, pcclGinDescriptorSmem* descriptor,
                                      uint32_t optFlags = pcclGinOptFlagsDefault);
};

template <pcclNetDeviceType backend>
struct pcclGinApi_Put {
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

template <pcclNetDeviceType backend>
struct pcclGinApi_PutValue {
  template <typename Coop, typename T>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx, Coop coop, int peer, pcclGinWindow_t dstWin,
                                      size_t dstOff, T srcData,
                                      pcclGinSignalDescriptor signal, pcclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor,
                                      pcclGinDescriptorSmem* descriptor,
                                      hggc::thread_scope required, hggc::thread_scope given,
                                      uint32_t optFlags = pcclGinOptFlagsDefault);
};

template <pcclNetDeviceType backend>
struct pcclGinApi_GetSignalPtr {
  PCCL_DEVICE_INLINE static uint64_t* call(pcclGinCtx, int peer, pcclGinSignal_t signalId);
};

template <pcclNetDeviceType backend>
struct pcclGinApi_GetCounterPtr {
  PCCL_DEVICE_INLINE static uint64_t* call(pcclGinCtx, int peer, pcclGinCounter_t counterId);
};

template <pcclNetDeviceType backend>
struct pcclGinApi_ResetSignal {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx, pcclGinSignalDescriptor signal);
};

template <pcclNetDeviceType backend>
struct pcclGinApi_ResetCounter {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx, pcclGinCounter_t counterId);
};

template <pcclNetDeviceType backend>
struct pcclGinApi_Flush {
  template <typename Coop>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx, Coop, hggc::memory_order ord, uint32_t* abortFlag);
};

template <template <pcclNetDeviceType> class ApiFn, typename... Arg>
PCCL_DEVICE_INLINE static decltype(auto) pcclGinCallImpl(unsigned beMask, pcclGinCtx ctx, Arg&&... arg) {
  bool singleton = (beMask & (beMask - 1)) == 0;  // Only one bit set
  switch (singleton ? __popc(beMask - 1) : (int)ctx.backend) {
#if PCCL_GIN_PROXY_ENABLE
    case (int)PCCL_NET_DEVICE_GIN_PROXY:
      if (!(1 & (beMask >> (int)PCCL_NET_DEVICE_GIN_PROXY))) __builtin_unreachable();
      return ApiFn<PCCL_NET_DEVICE_GIN_PROXY>::call(ctx, static_cast<Arg&&>(arg)...);
#endif
#if PCCL_GIN_GDAKI_ENABLE
    case (int)PCCL_NET_DEVICE_GIN_GDAKI:
      if (!(1 & (beMask >> (int)PCCL_NET_DEVICE_GIN_GDAKI))) __builtin_unreachable();
      return ApiFn<PCCL_NET_DEVICE_GIN_GDAKI>::call(ctx, static_cast<Arg&&>(arg)...);
#endif
    default:
      __builtin_unreachable();
  }
}

template <template <pcclNetDeviceType> class ApiFn, typename... Arg>
PCCL_DEVICE_INLINE static decltype(auto) pcclGinCall(pcclGinCtx ctx, Arg&&... arg) {
  return pcclGinCallImpl<ApiFn>(ctx.backendMask, ctx, static_cast<Arg&&>(arg)...);
}

template <template <pcclNetDeviceType> class ApiFn, unsigned beMask, typename... Arg>
PCCL_DEVICE_INLINE static decltype(auto) pcclGinCall(pcclGinCtx_M<beMask> ctx, Arg&&... arg) {
  return pcclGinCallImpl<ApiFn>(beMask, ctx, static_cast<Arg&&>(arg)...);
}
#endif // PCCL_CHECK_HGGCCC

#endif
