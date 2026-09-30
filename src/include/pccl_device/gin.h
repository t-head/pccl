/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_GIN_SESSION_H_
#define _PCCL_DEVICE_GIN_SESSION_H_
#include "core.h"
#include "gin/gin_device_common.h"

#if PCCL_CHECK_HGGCCC
struct pcclGinCtx; // Definition in pccl_device/gin/gin_device_host_common.h
template<unsigned> struct pcclGinCtx_M; // ...

struct pcclGinDescriptorSmem; // A type user allocates in __shared__ memory

// Used as completion actions for pcclGinSession::put
struct pcclGin_None {};

struct pcclGin_VASignalInc { pcclWindow_t signalWindow; size_t signalOffset; };
struct pcclGin_VASignalAdd { pcclWindow_t signalWindow; size_t signalOffset; uint64_t value; };

struct pcclGin_SignalAdd { pcclGinSignal_t signal; uint64_t value; };
// SignalInc: equivalent to SignalAdd{+1} except it may not be mixed with any
// other signal operator without intervening signal reset(). Formally: for a
// given signal, all operations between successive reset()'s of that signal must
// either all be SignalInc or all not SignalInc.
struct pcclGin_SignalInc { pcclGinSignal_t signal; };
// Support deferred:
// struct pcclGin_SignalSet { pcclGinSignal_t signal; uint64_t value; };
struct pcclGin_CounterInc { pcclGinCounter_t counter; };

struct pcclGin_DescriptorSmem { pcclGinDescriptorSmem* descriptor; };

// Segment type tags describe the composition of a buffer's physical cuMem segments.
struct pcclGin_SegmentDevice {};       // all segments are device-backed
struct pcclGin_SegmentMixed {}; // mix of HOST_NUMA and device-backed segments
struct pcclGin_SegmentHostNuma {};     // all segments are HOST_NUMA (CPU-backed)

template<unsigned backendMask>
struct pcclGin_BackendMask;

template<pcclNetDeviceType backend>
using pcclGin_BackendOne = pcclGin_BackendMask<(1u<<(int)backend)>;

using pcclGin = pcclGin_BackendMask<PCCL_GIN_BACKEND_MASK_ALL>;


struct pcclGin_C {
  pcclDevComm const& comm;
  uint32_t nConnections:8, connectionId:8, _ginBackend:8;
  uint32_t contextId;
  pcclGinResourceSharingMode resourceSharingMode;

  //////////////////////////////////////////////////////////////////////////////
  // internal:
  void* _ginHandle;
  uint64_t* _signalShadows;
  unsigned backendMask;

  PCCL_DEVICE_INLINE pcclGin_C(pcclDevComm const& comm_, unsigned backendMask_, int contextIndex,
                               pcclGinResourceSharingMode resourceSharingMode_ = PCCL_GIN_RESOURCE_SHARING_GPU);
};

// Helper init function that wraps placement new
PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE void pcclGin_C_init(
  pcclGin_C* net, unsigned backendMask, pcclDevComm const& comm, int contextIndex);

// Helper init function with explicit resource sharing mode.
PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE void pcclGin_C_initWithResourceSharingMode(
  pcclGin_C* net, unsigned backendMask, pcclDevComm const& comm, int contextIndex,
  pcclGinResourceSharingMode resourceSharingMode);

PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE void pcclGinPut(
  pcclGin_C* net,
  pcclTeam team, int peer,
  pcclWindow_t dstWin, size_t dstOffset,
  pcclWindow_t srcWin, size_t srcOffset, size_t bytes,
  bool isSignal, pcclGinSignal_t signalId, pcclGinSignalOp_t signalOp, uint64_t signalOpArg,
  bool isCounter, pcclGinCounter_t counterId,
  pcclCoopAny coop,
  bool isDescriptor, pcclGinDescriptorSmem* descriptor,
  hggc::thread_scope givenRelease, hggc::thread_scope requiredRelease);

PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE void pcclGinSignal(
  pcclGin_C* net,
  pcclTeam team, int peer,
  bool isSignal, pcclGinSignal_t signalId, pcclGinSignalOp_t signalOp, uint64_t signalOpArg,
  pcclCoopAny coop,
  bool isDescriptor, pcclGinDescriptorSmem* descriptor,
  hggc::thread_scope givenRelease, hggc::thread_scope requiredRelease);

PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE void pcclGinPut_v2(
  pcclGin_C* net,
  pcclTeam team, int peer,
  pcclWindow_t dstWin, size_t dstOffset,
  pcclWindow_t srcWin, size_t srcOffset, size_t bytes,
  bool isSignal, pcclGinSignal_t signalId, pcclGinSignalOp_t signalOp, uint64_t signalOpArg,
  bool isCounter, pcclGinCounter_t counterId,
  pcclCoopAny coop,
  bool isDescriptor, pcclGinDescriptorSmem* descriptor,
  hggc::thread_scope givenRelease, hggc::thread_scope requiredRelease,
  uint32_t optFlags);

PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE void pcclGinSignal_v2(
  pcclGin_C* net,
  pcclTeam team, int peer,
  bool isSignal, pcclGinSignal_t signalId, pcclGinSignalOp_t signalOp, uint64_t signalOpArg,
  pcclCoopAny coop,
  bool isDescriptor, pcclGinDescriptorSmem* descriptor,
  hggc::thread_scope givenRelease, hggc::thread_scope requiredRelease,
  uint32_t optFlags);

PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE void pcclGinFlush(
  pcclGin_C* net,
  pcclCoopAny coop,
  hggc::memory_order ord);

PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE uint64_t pcclGinReadCounter(
  pcclGin_C* net,
  pcclGinCounter_t counter,
  int bits,
  hggc::memory_order ord);

PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE void pcclGinWaitCounter(
  pcclGin_C* net,
  pcclCoopAny coop,
  pcclGinCounter_t counter,
  uint64_t least,
  int bits,
  hggc::memory_order ord);

PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE uint64_t pcclGinReadSignal(
  pcclGin_C* net,
  pcclGinSignal_t signal,
  int bits,
  hggc::memory_order ord);

PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE void pcclGinWaitSignal(
  pcclGin_C* net,
  pcclCoopAny coop,
  pcclGinSignal_t signal,
  uint64_t least,
  int bits,
  hggc::memory_order ord);

PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE void pcclGinResetCounter(
  pcclGin_C* net,
  pcclGinCounter_t counter);

PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE void pcclGinResetSignal(
  pcclGin_C* net,
  pcclGinSignal_t signal);

PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE void pcclGinPutValue(
  pcclGin_C* net,
  pcclTeam team, int peer,
  pcclWindow_t dstWin, size_t dstOffset,
  uint64_t value, size_t size,
  bool isSignal, pcclGinSignal_t signalId, pcclGinSignalOp_t signalOp, uint64_t signalOpArg,
  pcclCoopAny coop,
  bool isDescriptor, pcclGinDescriptorSmem* descriptor,
  hggc::thread_scope givenRelease, hggc::thread_scope requiredRelease);

PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE void pcclGinPutValue_v2(
  pcclGin_C* net,
  pcclTeam team, int peer,
  pcclWindow_t dstWin, size_t dstOffset,
  uint64_t value, size_t size,
  bool isSignal, pcclGinSignal_t signalId, pcclGinSignalOp_t signalOp, uint64_t signalOpArg,
  pcclCoopAny coop,
  bool isDescriptor, pcclGinDescriptorSmem* descriptor,
  hggc::thread_scope givenRelease, hggc::thread_scope requiredRelease,
  uint32_t optFlags);

PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE uint64_t* pcclGinGetSignalShadowPtr(
  pcclGin_C* net,
  pcclGinSignal_t signal);

template<unsigned backendMask>
struct pcclGin_BackendMask {
  pcclDevComm const& comm;
  uint32_t nConnections:8, connectionId:8, _ginBackend:8;
  uint32_t contextId;
  // Runtime-selected resource sharing mode for this context.
  pcclGinResourceSharingMode resourceSharingMode;

  // Loads GIN context into registers. Each context has one QP per peer.
  PCCL_DEVICE_INLINE pcclGin_BackendMask(
    pcclDevComm const&, int contextIndex,
    pcclGinResourceSharingMode resourceSharingMode_ = PCCL_GIN_RESOURCE_SHARING_GPU);

  template <typename Coop = pcclCoopThread>
  PCCL_DEVICE_INLINE void flushAsync(pcclTeam team, uint32_t peer, pcclGinRequest_t* outRequest,
                                     Coop coop = pcclCoopThread{}, uint32_t optFlags = pcclGinOptFlagsDefault) const;

  template <typename Coop = pcclCoopThread, typename DescriptorSmem = pcclGin_None>
  PCCL_DEVICE_INLINE void wait(pcclGinRequest_t& outRequest,
                               Coop coop = pcclCoopThread{}, DescriptorSmem descriptor = pcclGin_None{},
                               hggc::memory_order ord = hggc::memory_order_acquire) const;

  template<typename Coop = pcclCoopThread,
           typename DescriptorSmem = pcclGin_None,
           typename SegmentType = pcclGin_SegmentDevice>
  PCCL_DEVICE_INLINE void get(
    pcclTeam, int peer,
    pcclWindow_t remoteWnd, size_t remoteOffset,
    pcclWindow_t localWnd, size_t localOffset,
    size_t bytes, Coop coop = pcclCoopThread{},
    DescriptorSmem descriptor = pcclGin_None{},
    uint32_t optFlags = pcclGinOptFlagsDefault,
    SegmentType bufType = pcclGin_SegmentDevice{}) const;

  template<
    // Action to take on peer when put completes. If a signalling action is used
    // then that signal will be visible only after the payload of this put as well as
    // the payloads of preceding puts on this netContext to the same peer are settled.
    typename RemoteAction = pcclGin_None, // one of pcclGin_{None|SignalInc|SignalAdd|SignalSet}
    // Action to take locally when source has been consumed.
    typename LocalAction = pcclGin_None, // one of pcclGin_{None|CounterInc}
    // Set of threads participating in this put. Must be a subset of Coop.
    typename Coop = pcclCoopThread,
    // Optional smem descriptor space to use. Either pcclGin_{None|DescriptorSmem}
    typename DescriptorSmem = pcclGin_None,
    // Use pcclGin_SegmentMixed or pcclGin_SegmentHostNuma when the VA contains
    // CPU-backed (HOST_NUMA) segments
    typename SegmentType = pcclGin_SegmentDevice
  >
  PCCL_DEVICE_INLINE void put(
    pcclTeam, int peer,
    pcclWindow_t dstWnd, size_t dstOffset,
    pcclWindow_t srcWnd, size_t srcOffset, size_t bytes,
    RemoteAction remoteAction = pcclGin_None{},
    LocalAction localAction = pcclGin_None{},
    Coop coop = pcclCoopThread{},
    DescriptorSmem descriptor = pcclGin_None{},
    hggc::thread_scope givenRelease = hggc::thread_scope_thread,
    hggc::thread_scope requiredRelease = hggc::thread_scope_device,
    uint32_t optFlags = pcclGinOptFlagsDefault,
    SegmentType bufType = pcclGin_SegmentDevice{}
  ) const;

  template<
    typename T,
    // Action to take on peer when put completes. If a signalling action is used
    // then that signal will be visible only after the payload of this put as well as
    // the payloads of preceding puts on this context to the same peer are settled.
    typename RemoteAction = pcclGin_None, // one of pcclGin_{None|SignalInc|SignalAdd|SignalSet}
    // Action to take locally when source has been consumed.
    typename LocalAction = pcclGin_None, // one of pcclGin_{None|CounterInc}
    // Set of threads participating in this put. Must be a subset of Coop.
    typename Coop = pcclCoopThread,
    // Optional smem descriptor space to use. Either pcclGin_{None|DescriptorSmem}
    typename DescriptorSmem = pcclGin_None,
    // One of pcclGin_{SegmentDevice|SegmentMixed|SegmentHostNuma}; use a non-Device tag when the VA contains CPU-backed (HOST_NUMA) segments
    typename SegmentType = pcclGin_SegmentDevice
  >
  PCCL_DEVICE_INLINE void put(
    pcclTeam, int peer,
    pcclSymPtr<T> dstElts, pcclSymPtr<T> srcElts, size_t nElts,
    RemoteAction remoteAction = pcclGin_None{},
    LocalAction localAction = pcclGin_None{},
    Coop coop = pcclCoopThread{},
    DescriptorSmem descriptor = pcclGin_None{},
    hggc::thread_scope givenRelease = hggc::thread_scope_thread,
    hggc::thread_scope requiredRelease = hggc::thread_scope_device,
    uint32_t optFlags = pcclGinOptFlagsDefault,
    SegmentType bufType = pcclGin_SegmentDevice{}
  ) const;

  template<
    typename T, // requires sizeof(T) <= 8
    // See put() for all template arguments.
    typename RemoteAction = pcclGin_None,
    typename Coop = pcclCoopThread,
    typename DescriptorSmem = pcclGin_None
  >
  PCCL_DEVICE_INLINE void putValue(
    pcclTeam, int peer,
    pcclWindow_t dstWnd, size_t dstOffset, T value,
    RemoteAction remoteAction = pcclGin_None{},
    Coop coop = pcclCoopThread{},
    DescriptorSmem descriptor = pcclGin_None{},
    hggc::thread_scope givenRelease = hggc::thread_scope_thread,
    hggc::thread_scope requiredRelease = hggc::thread_scope_device,
    uint32_t optFlags = pcclGinOptFlagsDefault
  ) const;

  template<
    typename T, // requires sizeof(T) <= 8
    // See put() for all template arguments.
    typename RemoteAction = pcclGin_None,
    typename Coop = pcclCoopThread,
    typename DescriptorSmem = pcclGin_None
  >
  PCCL_DEVICE_INLINE void putValue(
    pcclTeam, int peer,
    pcclSymPtr<T> dst, T value,
    RemoteAction remoteAction = pcclGin_None{},
    Coop coop = pcclCoopThread{},
    DescriptorSmem descriptor = pcclGin_None{},
    hggc::thread_scope givenRelease = hggc::thread_scope_thread,
    hggc::thread_scope requiredRelease = hggc::thread_scope_device,
    uint32_t optFlags = pcclGinOptFlagsDefault
  ) const;

  template<typename RemoteAction,
           typename Coop = pcclCoopThread,
           typename DescriptorSmem = pcclGin_None>
  PCCL_DEVICE_INLINE void signal(
    pcclTeam, int peer, RemoteAction remoteAction,
    Coop coop = pcclCoopThread(),
    DescriptorSmem descriptor = pcclGin_None{},
    hggc::thread_scope givenRelease = hggc::thread_scope_thread,
    hggc::thread_scope requiredRelease = hggc::thread_scope_device,
    uint32_t optFlags = pcclGinOptFlagsDefault
  ) const;

  // All source buffers from put's from any thread in this coop will be safe to reuse.
  // Flush does not guarantee that data has settled in remote memory.
  template<typename Coop>
  PCCL_DEVICE_INLINE void flush(Coop, hggc::memory_order ord = hggc::memory_order_acquire) const;

  // Counter and signal wait use "rolling" comparison logic of a given bit-width
  // such that unsigned overflow does not disturb the property that: x < x+1.
  //
  // bool rolling_less_equal(uint64_t a, uint64_t b, int bits) {
  //   uint64_t m = uint64_t(-1)>>(64-bits);
  //   return ((b-a) & m) <= (m>>1);
  // }
  //
  // The condition waited for is that the supplied value is rolling_less_equal
  // to the internal value.
  //
  // Counters are restricted to using a maximum of 56 bits despite that being fewer
  // than a uint64_t can carry.

  PCCL_DEVICE_INLINE uint64_t readCounter(pcclGinCounter_t counter, int bits=56, hggc::memory_order ord = hggc::memory_order_acquire) const;

  template<typename Coop>
  PCCL_DEVICE_INLINE void waitCounter(Coop, pcclGinCounter_t counter, uint64_t least, int bits=56, hggc::memory_order ord = hggc::memory_order_acquire) const;

  // Each signal has a dedicated "shadow" which the user is free to manipulate for
  // any reason. The only calls which manipulate the shadow are `increaseSignalShadow`
  // and `resetSignal`.
  PCCL_DEVICE_INLINE uint64_t* getSignalShadowPtr(pcclGinSignal_t signal) const;
  PCCL_DEVICE_INLINE void increaseSignalShadow(pcclGinSignal_t signal, uint64_t delta) const;

  // Returns current value of signal with all but bottom bits set to zero.
  PCCL_DEVICE_INLINE uint64_t readSignal(pcclGinSignal_t signal, int bits=64, hggc::memory_order ord = hggc::memory_order_acquire) const;

  // Returns current value of VA signal at given window and offset with all but bottom bits set to zero.
  PCCL_DEVICE_INLINE uint64_t readSignal(pcclWindow_t signalWindow, size_t signalOffset, int bits=64, hggc::memory_order ord = hggc::memory_order_acquire) const;

  // Wait for signal to meet or exceed value.
  template<typename Coop>
  PCCL_DEVICE_INLINE void waitSignal(Coop, pcclGinSignal_t signal, uint64_t least, int bits=64, hggc::memory_order ord = hggc::memory_order_acquire) const;

  // Wait for VA signal at given window and offset to meet or exceed value.
  template<typename Coop>
  PCCL_DEVICE_INLINE void waitSignal(Coop, pcclWindow_t signalWindow, size_t signalOffset, uint64_t least, int bits=64, hggc::memory_order ord = hggc::memory_order_acquire) const;

  // Wait for signal to meet or exceed shadow value.
  template<typename Coop>
  PCCL_DEVICE_INLINE void waitSignalMeetShadow(Coop, pcclGinSignal_t signal, int bits=64, hggc::memory_order ord = hggc::memory_order_acquire) const;

  // Wait until signal exceeds shadow by `leastDelta` (typically 1), updates shadow
  // with latest value, and returns with `before` equal to previous shadow value
  // and `delta` equal to difference.
  template<typename Coop, typename Uint>
  PCCL_DEVICE_INLINE void waitSignalFollowShadow(Coop, pcclGinSignal_t signal, Uint leastDelta, Uint* before, Uint* delta, int bits=64, hggc::memory_order ord = hggc::memory_order_acquire) const;

  // Sets to zero. May not race with concurrent modifications to counter.
  PCCL_DEVICE_INLINE void resetCounter(pcclGinCounter_t counter) const;
  // Sets signal and shadow to zero. May not race with concurrent modifcations to signal.
  PCCL_DEVICE_INLINE void resetSignal(pcclGinSignal_t signal) const;
  // Resets a VA signal at the given window and offset.
  PCCL_DEVICE_INLINE void resetSignal(pcclWindow_t signalWindow, size_t signalOffset) const;

  //////////////////////////////////////////////////////////////////////////////
  // internal:

  void* _ginHandle;
  uint64_t* _signalShadows;

  PCCL_DEVICE_INLINE pcclGinCtx_M<backendMask> _makeCtx() const;
};
#endif // PCCL_CHECK_HGGCCC

#endif // _PCCL_DEVICE_GIN_SESSION_H_
