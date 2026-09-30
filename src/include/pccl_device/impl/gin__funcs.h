/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_GIN_SESSION__FUNCS_H_
#define _PCCL_DEVICE_GIN_SESSION__FUNCS_H_
#include "gin__types.h"
#include "ptr__types.h"
#include <new>

#if PCCL_CHECK_HGGCCC
#include "pccl_device/gin/gin_device_api.h"
namespace pccl {
namespace gin {
namespace internal {
PCCL_DEVICE_INLINE size_t windowOffsetToGinOffset(pcclWindow_t window, size_t offset) {
  using pccl::utility::loadConst;
  return 4096*size_t(loadConst(&window->ginOffset4K)) + offset;
}

PCCL_DEVICE_INLINE pcclGinWindow_t getGinWindow(pcclWindow_t window, int connectionId) {
  using pccl::utility::loadConst;
  return loadConst(&window->ginWins[connectionId]);
}

PCCL_DEVICE_INLINE int teamRankToGinRank(pcclDevComm const& comm, pcclTeam team, int teamRank) {
  int worldRank = pcclTeamRankToWorld(comm, team, teamRank);
  if (comm.ginIsRailed) {
    return utility::idivFast32(worldRank, comm.lsaSize, comm.lsaSize_rcp32);
  } else {
    return worldRank;
  }
}

// Multi-segment put/get helpers
PCCL_DEVICE_INLINE void findSegmentFromWindow(pcclWindow_t win, size_t offset, int* outSeg, size_t* outSegOffset) {
  int seg = 0;
  size_t segOffset = offset;
  size_t cumulative = 0;
  for (int i = 0; i < win->numSegments; i++) {
    struct pcclSegmentWindow const& w = win->ginMultiSegmentWins[i];
    if (cumulative + w.segmentSize > offset) {
      seg = i;
      segOffset = offset - cumulative;
      break;
    }
    cumulative += w.segmentSize;
  }
  *outSeg = seg;
  *outSegOffset = segOffset;
}

PCCL_DEVICE_INLINE size_t getSegmentChunkSize(size_t srcRemaining, size_t dstRemaining, size_t remaining) {
  size_t chunk = srcRemaining;
  if (dstRemaining < chunk) chunk = dstRemaining;
  if (remaining < chunk) chunk = remaining;
  return chunk;
}

PCCL_DEVICE_INLINE void advanceSegmentCursor(
    int* seg, size_t* segOffset, size_t chunkSize, size_t segmentSize) {
  *segOffset += chunkSize;
  if (*segOffset >= segmentSize) {
    (*seg)++;
    *segOffset = 0;
  }
}

} // namespace internal
} // namespace gin
} // namespace pccl


// Common initialization helper for GIN backend
template<typename GinType>
PCCL_DEVICE_INLINE void pcclGinInitCommon(GinType* gin, pcclDevComm const& comm, int contextIndex) {
  gin->nConnections = comm.ginConnectionCount;

  static_assert(PCCL_GIN_MAX_CONNECTIONS == 4, "Required for following modulo hack to work.");
  // this->connectionId = contextIndex % comm.ginConnectionCount;
  gin->connectionId = comm.ginConnectionCount == 3
    ? uint32_t(contextIndex)%3 // 3 is only non power of 2
    : contextIndex & (comm.ginConnectionCount-1); // powers of 2
  // gin->contextId = contextIndex / comm.ginConnectionCount;
  gin->contextId = comm.ginConnectionCount == 3
    ? uint32_t(contextIndex)/3 // 3 is only non power of 2
    : contextIndex >> (comm.ginConnectionCount==4 ? 2 : comm.ginConnectionCount-1); // powers of 2

  gin->_ginBackend = comm.ginNetDeviceTypes[gin->connectionId];
  gin->_ginHandle = comm.ginHandles[gin->connectionId];
  gin->_signalShadows = comm.ginSignalShadows + contextIndex * comm.ginSignalCount;
}

template<unsigned beMask>
PCCL_DEVICE_INLINE pcclGin_BackendMask<beMask>::pcclGin_BackendMask(
    pcclDevComm const& comm, int contextIndex, pcclGinResourceSharingMode resourceSharingMode_):
  comm(comm), resourceSharingMode(resourceSharingMode_) {
  pcclGinInitCommon(this, comm, contextIndex);
}

PCCL_DEVICE_INLINE pcclGin_C::pcclGin_C(
    pcclDevComm const& comm, unsigned backendMask, int contextIndex,
    pcclGinResourceSharingMode resourceSharingMode)
  : comm(comm), resourceSharingMode(resourceSharingMode), backendMask(backendMask) {
  pcclGinInitCommon(this, comm, contextIndex);
}

PCCL_DEVICE_INLINE void pcclGin_C_init(
    pcclGin_C* net, unsigned backendMask, pcclDevComm const& comm, int contextIndex) {
  ::new (net) pcclGin_C(comm, backendMask, contextIndex, PCCL_GIN_RESOURCE_SHARING_GPU);
}

PCCL_DEVICE_INLINE void pcclGin_C_initWithResourceSharingMode(
    pcclGin_C* net, unsigned backendMask, pcclDevComm const& comm, int contextIndex,
    pcclGinResourceSharingMode resourceSharingMode) {
  ::new (net) pcclGin_C(comm, backendMask, contextIndex, resourceSharingMode);
}

template<unsigned beMask>
PCCL_DEVICE_INLINE pcclGinCtx_M<beMask> pcclGin_BackendMask<beMask>::_makeCtx() const {
  pcclGinCtx_M<beMask> ans;
  ans.backend = (pcclNetDeviceType)_ginBackend;
  if (comm.ginIsRailed) {
    pcclTeam teamRail = pcclTeamRail(comm);
    ans.rank = teamRail.rank;
    ans.nRanks = teamRail.nRanks;
  } else {
    ans.rank = comm.rank;
    ans.nRanks = comm.nRanks;
  }
  ans.handle = _ginHandle;
  ans.contextId = contextId;
  ans.resourceSharingMode = (uint8_t)this->resourceSharingMode;
  return ans;
}

PCCL_DEVICE_INLINE pcclGinCtx pcclGin_C_makeCtx(pcclGin_C* net) {
  pcclGinCtx ans;
  ans.backendMask = net->backendMask;
  ans.backend = (pcclNetDeviceType)net->_ginBackend;
  if (net->comm.ginIsRailed) {
    pcclTeam teamRail = pcclTeamRail(net->comm);
    ans.rank = teamRail.rank;
    ans.nRanks = teamRail.nRanks;
  } else {
    ans.rank = net->comm.rank;
    ans.nRanks = net->comm.nRanks;
  }
  ans.handle = net->_ginHandle;
  ans.contextId = net->contextId;
  ans.resourceSharingMode = (uint8_t)net->resourceSharingMode;
  return ans;
}

////////////////////////////////////////////////////////////////////////////////
// pcclGin descriptor helpers:

template<typename Descriptor>
PCCL_DEVICE_INLINE constexpr bool pcclGin_isDescriptor(Descriptor) { return false; }
template<typename Descriptor>
PCCL_DEVICE_INLINE constexpr pcclGinDescriptorSmem* pcclGin_getDescriptor(Descriptor) { return nullptr; }

PCCL_DEVICE_INLINE constexpr bool pcclGin_isDescriptor(pcclGin_DescriptorSmem) { return true; }
PCCL_DEVICE_INLINE constexpr pcclGinDescriptorSmem* pcclGin_getDescriptor(pcclGin_DescriptorSmem arg) { return arg.descriptor; }

////////////////////////////////////////////////////////////////////////////////
// pcclGin signal helpers:

template<typename RemoteAction>
PCCL_DEVICE_INLINE constexpr pcclGinSignalOp_t pcclGin_getSignalOp(RemoteAction) { return (pcclGinSignalOp_t)0; }
template<typename RemoteAction>
PCCL_DEVICE_INLINE constexpr uint64_t pcclGin_getSignalOpArg(RemoteAction) { return 0; }
template<typename RemoteAction>
PCCL_DEVICE_INLINE constexpr pcclGinSignalDescriptor pcclGin_getSignalDescriptor(pcclGin const&, RemoteAction) {
  pcclGinSignalDescriptor desc{};
  desc.type = PCCL_GIN_SIGNAL_TYPE_NONE;
  return desc;
}

PCCL_DEVICE_INLINE constexpr pcclGinSignalDescriptor pcclGin_getSignalDescriptor(pcclGin const& net, pcclGin_SignalInc arg) {
  pcclGinSignalDescriptor desc{};
  desc.type = PCCL_GIN_SIGNAL_TYPE_INDEXED;
  desc.indexedSignal.signalId = arg.signal;
  return desc;
}
PCCL_DEVICE_INLINE constexpr pcclGinSignalOp_t pcclGin_getSignalOp(pcclGin_SignalInc) { return pcclGinSignalInc; }
PCCL_DEVICE_INLINE constexpr uint64_t pcclGin_getSignalOpArg(pcclGin_SignalInc) { return 1; }

PCCL_DEVICE_INLINE constexpr pcclGinSignalDescriptor pcclGin_getSignalDescriptor(pcclGin const& net, pcclGin_SignalAdd arg) {
  pcclGinSignalDescriptor desc{};
  desc.type = PCCL_GIN_SIGNAL_TYPE_INDEXED;
  desc.indexedSignal.signalId = arg.signal;
  return desc;
}
PCCL_DEVICE_INLINE constexpr pcclGinSignalOp_t pcclGin_getSignalOp(pcclGin_SignalAdd) {
  return pcclGinSignalAdd;
}
PCCL_DEVICE_INLINE constexpr uint64_t pcclGin_getSignalOpArg(pcclGin_SignalAdd arg) { return arg.value; }

PCCL_DEVICE_INLINE pcclGinSignalDescriptor pcclGin_getSignalDescriptor(pcclGin const& net, pcclGin_VASignalInc arg) {
  pcclGinSignalDescriptor desc{};
  desc.type = PCCL_GIN_SIGNAL_TYPE_VA;
  desc.vaSignal.signalWindow = pccl::gin::internal::getGinWindow(arg.signalWindow, net.connectionId);
  desc.vaSignal.signalOffset = pccl::gin::internal::windowOffsetToGinOffset(arg.signalWindow, arg.signalOffset);
  desc.vaSignal.pcclWindow = arg.signalWindow;
  return desc;
}
PCCL_DEVICE_INLINE constexpr pcclGinSignalOp_t pcclGin_getSignalOp(pcclGin_VASignalInc arg) {
  return pcclGinSignalInc;
}
PCCL_DEVICE_INLINE constexpr uint64_t pcclGin_getSignalOpArg(pcclGin_VASignalInc arg) { return 1; }

PCCL_DEVICE_INLINE pcclGinSignalDescriptor pcclGin_getSignalDescriptor(pcclGin const& net, pcclGin_VASignalAdd arg) {
  pcclGinSignalDescriptor desc{};
  desc.type = PCCL_GIN_SIGNAL_TYPE_VA;
  desc.vaSignal.signalWindow = pccl::gin::internal::getGinWindow(arg.signalWindow, net.connectionId);
  desc.vaSignal.signalOffset = pccl::gin::internal::windowOffsetToGinOffset(arg.signalWindow, arg.signalOffset);
  desc.vaSignal.pcclWindow = arg.signalWindow;
  return desc;
}
PCCL_DEVICE_INLINE constexpr pcclGinSignalOp_t pcclGin_getSignalOp(pcclGin_VASignalAdd arg) {
  return pcclGinSignalAdd;
}
PCCL_DEVICE_INLINE constexpr uint64_t pcclGin_getSignalOpArg(pcclGin_VASignalAdd arg) { return arg.value; }


////////////////////////////////////////////////////////////////////////////////
// pcclGin counter helpers:

template<typename LocalAction>
PCCL_DEVICE_INLINE constexpr bool pcclGin_isCounter(LocalAction) { return false; }
template<typename LocalAction>
PCCL_DEVICE_INLINE constexpr pcclGinCounter_t pcclGin_getCounterId(pcclGin const&, LocalAction) { return -1u; }

PCCL_DEVICE_INLINE constexpr bool pcclGin_isCounter(pcclGin_CounterInc) { return true; }
PCCL_DEVICE_INLINE constexpr pcclGinCounter_t pcclGin_getCounterId(pcclGin const& net, pcclGin_CounterInc arg) { return arg.counter; }

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// pcclGin bufType helpers:

PCCL_DEVICE_INLINE constexpr bool pcclGin_isDeviceOnly(pcclGin_SegmentDevice) { return true; }
PCCL_DEVICE_INLINE constexpr bool pcclGin_isDeviceOnly(pcclGin_SegmentMixed) { return false; }
PCCL_DEVICE_INLINE constexpr bool pcclGin_isDeviceOnly(pcclGin_SegmentHostNuma) { return false; }
////////////////////////////////////////////////////////////////////////////////

PCCL_DEVICE_INLINE void pcclGinPut_v2(
    pcclGin_C* net,
    pcclTeam team, int peer,
    pcclWindow_t dstWin, size_t dstOffset,
    pcclWindow_t srcWin, size_t srcOffset, size_t bytes,
    bool isSignal, pcclGinSignal_t signalId, pcclGinSignalOp_t signalOp, uint64_t signalOpArg,
    bool isCounter, pcclGinCounter_t counterId,
    pcclCoopAny coop,
    bool isDescriptor, pcclGinDescriptorSmem* descriptor,
    hggc::thread_scope givenRelease, hggc::thread_scope requiredRelease,
    uint32_t optFlags
  ) {
  using pccl::utility::loadConst;
  using pccl::gin::internal::teamRankToGinRank;
  pcclGinCtx ctx = pcclGin_C_makeCtx(net);
  coop.sync();
  pcclGinSignalDescriptor signal{};
  signal.type = PCCL_GIN_SIGNAL_TYPE_NONE;
  if (isSignal) {
    signal.type = PCCL_GIN_SIGNAL_TYPE_INDEXED;
    signal.indexedSignal.signalId = signalId;
  }
  if (coop.thread_rank() == 0) {
    pcclGinCall<pcclGinApi_Put>(ctx,
      pcclCoopThread(), teamRankToGinRank(net->comm, team, peer), /*hasWins=*/true,
      loadConst(&dstWin->ginWins[net->connectionId]),
      4096*size_t(loadConst(&dstWin->ginOffset4K)) + dstOffset,
      loadConst(&srcWin->ginWins[net->connectionId]),
      4096*size_t(loadConst(&srcWin->ginOffset4K)) + srcOffset, bytes,
      signal,
      signalOp,
      signalOpArg,
      isCounter,
      counterId,
      isDescriptor,
      descriptor,
      requiredRelease,
      givenRelease,
      optFlags
    );
  }
  coop.sync();
}

PCCL_DEVICE_INLINE void pcclGinPut(
    pcclGin_C* net,
    pcclTeam team, int peer,
    pcclWindow_t dstWin, size_t dstOffset,
    pcclWindow_t srcWin, size_t srcOffset, size_t bytes,
    bool isSignal, pcclGinSignal_t signalId, pcclGinSignalOp_t signalOp, uint64_t signalOpArg,
    bool isCounter, pcclGinCounter_t counterId,
    pcclCoopAny coop,
    bool isDescriptor, pcclGinDescriptorSmem* descriptor,
    hggc::thread_scope givenRelease, hggc::thread_scope requiredRelease
  ) {
  pcclGinPut_v2(net, team, peer, dstWin, dstOffset, srcWin, srcOffset, bytes, isSignal, signalId,
               signalOp, signalOpArg, isCounter, counterId, coop, isDescriptor, descriptor,
               givenRelease, requiredRelease, pcclGinOptFlagsDefault);
}

template<unsigned beMask>
template<
  typename RemoteAction, // one of pcclGin_{None|SignalInc}
  typename LocalAction, // one of pcclGin_{None|CounterInc}
  typename Coop,
  typename DescriptorSmem,
  typename SegmentType // one of pcclGin_{SegmentDevice|SegmentMixed|SegmentHostNuma}
>
PCCL_DEVICE_INLINE void pcclGin_BackendMask<beMask>::put(
    pcclTeam team, int peer,
    pcclWindow_t dstWin, size_t dstOffset,
    pcclWindow_t srcWin, size_t srcOffset, size_t bytes,
    RemoteAction remoteAction, LocalAction localAction,
    Coop coop,
    DescriptorSmem descriptor,
    hggc::thread_scope givenRelease, hggc::thread_scope requiredRelease,
    uint32_t optFlags, SegmentType bufType
  ) const {
  using pccl::utility::loadConst;
  using pccl::gin::internal::teamRankToGinRank;
  pcclGinCtx_M<beMask> ctx = this->_makeCtx();
  coop.sync();
  if (coop.thread_rank() == 0) {
    if (pcclGin_isDeviceOnly(bufType)) {
      pcclGinCall<pcclGinApi_Put>(ctx,
          pcclCoopThread(), teamRankToGinRank(this->comm, team, peer), /*hasWins=*/true,
          loadConst(&dstWin->ginWins[this->connectionId]),
          4096*size_t(loadConst(&dstWin->ginOffset4K)) + dstOffset,
          loadConst(&srcWin->ginWins[this->connectionId]),
          4096*size_t(loadConst(&srcWin->ginOffset4K)) + srcOffset, bytes,
          pcclGin_getSignalDescriptor(*this, remoteAction),
          pcclGin_getSignalOp(remoteAction),
          pcclGin_getSignalOpArg(remoteAction),
          pcclGin_isCounter(localAction),
          pcclGin_getCounterId(*this, localAction),
          pcclGin_isDescriptor(descriptor),
          pcclGin_getDescriptor(descriptor),
          requiredRelease,
          givenRelease,
          optFlags
          );
    } else {
      if (srcWin->numSegments == 1 && dstWin->numSegments == 1) {
        pcclGinCall<pcclGinApi_Put>(ctx,
            pcclCoopThread(), teamRankToGinRank(this->comm, team, peer), /*hasWins=*/true,
            loadConst(&dstWin->ginWins[this->connectionId]),
            4096*size_t(loadConst(&dstWin->ginOffset4K)) + dstOffset,
            loadConst(&srcWin->ginWins[this->connectionId]),
            4096*size_t(loadConst(&srcWin->ginOffset4K)) + srcOffset, bytes,
            pcclGin_getSignalDescriptor(*this, remoteAction),
            pcclGin_getSignalOp(remoteAction),
            pcclGin_getSignalOpArg(remoteAction),
            pcclGin_isCounter(localAction),
            pcclGin_getCounterId(*this, localAction),
            pcclGin_isDescriptor(descriptor),
            pcclGin_getDescriptor(descriptor),
            hggc::thread_scope_system, // for safety, escalate to system regardless of what the user requested
            givenRelease,
            optFlags
            );
      } else {
        // Multi-segment case. The puts are chunked to handle multiple registration entries and src/dst windows that potentially have a different number of segments
        int srcSeg;
        size_t srcSegOffset;
        pccl::gin::internal::findSegmentFromWindow(srcWin, srcOffset, &srcSeg, &srcSegOffset);
        int dstSeg;
        size_t dstSegOffset;
        pccl::gin::internal::findSegmentFromWindow(dstWin, dstOffset, &dstSeg, &dstSegOffset);
        bool doneSysmemFence = false;
        size_t remaining = bytes;
        hggc::thread_scope localRequiredRelease = requiredRelease;

        while (remaining > 0) {
          struct pcclSegmentWindow const& dstSegmentWindow = dstWin->ginMultiSegmentWins[dstSeg];
          struct pcclSegmentWindow const& srcSegmentWindow = srcWin->ginMultiSegmentWins[srcSeg];
          if (!doneSysmemFence && srcSegmentWindow.memType == HG_MEM_LOCATION_TYPE_HOST_NUMA) {
            localRequiredRelease = hggc::thread_scope_system; // for safety, escalate to system regardless of what the user requested
            doneSysmemFence = true;
          }
          const size_t srcRemaining = srcSegmentWindow.segmentSize - srcSegOffset;
          const size_t dstRemaining = dstSegmentWindow.segmentSize - dstSegOffset;
          const size_t putSize = pccl::gin::internal::getSegmentChunkSize(srcRemaining, dstRemaining, remaining);
          const bool isLastPut = (remaining == putSize);
          pcclGinCall<pcclGinApi_Put>(ctx,
              pcclCoopThread(), teamRankToGinRank(this->comm, team, peer), /*hasWins=*/true,
              loadConst(&dstWin->ginMultiSegmentWins[dstSeg].ginWins[this->connectionId]),
              dstSegOffset,
              loadConst(&srcWin->ginMultiSegmentWins[srcSeg].ginWins[this->connectionId]),
              srcSegOffset, putSize,
              isLastPut ? pcclGin_getSignalDescriptor(*this, remoteAction) : pcclGin_getSignalDescriptor(*this, pcclGin_None{}),
              pcclGin_getSignalOp(remoteAction),
              pcclGin_getSignalOpArg(remoteAction),
              isLastPut ? pcclGin_isCounter(localAction) : false,
              pcclGin_getCounterId(*this, localAction),
              pcclGin_isDescriptor(descriptor),
              pcclGin_getDescriptor(descriptor),
              localRequiredRelease,
              givenRelease,
              optFlags
              );
          remaining -= putSize;
          pccl::gin::internal::advanceSegmentCursor(&srcSeg, &srcSegOffset, putSize, srcSegmentWindow.segmentSize);
          pccl::gin::internal::advanceSegmentCursor(&dstSeg, &dstSegOffset, putSize, dstSegmentWindow.segmentSize);
          localRequiredRelease = hggc::thread_scope_thread;
        }
      }
    }
  }
  coop.sync();
}

template<unsigned beMask>
template<
  typename T,
  typename RemoteAction, // one of pcclGin_{None|SignalInc}
  typename LocalAction, // one of pcclGin_{None|CounterInc}
  typename Coop,
  typename DescriptorSmem,
  typename SegmentType
>
PCCL_DEVICE_INLINE void pcclGin_BackendMask<beMask>::put(
    pcclTeam team, int peer,
    pcclSymPtr<T> dstElts, pcclSymPtr<T> srcElts, size_t nElts,
    RemoteAction remoteAction, LocalAction localAction,
    Coop coop,
    DescriptorSmem descriptor,
    hggc::thread_scope givenRelease,
    hggc::thread_scope requiredRelease,
    uint32_t optFlags, SegmentType bufType
  ) const {
  this->put(
    team, peer, dstElts.window, dstElts.offset, srcElts.window, srcElts.offset, nElts*sizeof(T),
    remoteAction, localAction, coop, descriptor, givenRelease, requiredRelease, optFlags, bufType
  );
}

template<unsigned beMask>
template<
  typename T,
  typename RemoteAction, // one of pcclGin_{None|SignalInc}
  typename Coop,
  typename DescriptorSmem
>
PCCL_DEVICE_INLINE void pcclGin_BackendMask<beMask>::putValue(
    pcclTeam team, int peer,
    pcclWindow_t dstWin, size_t dstOffset, T value,
    RemoteAction remoteAction,
    Coop coop,
    DescriptorSmem descriptor,
    hggc::thread_scope givenRelease,
    hggc::thread_scope requiredRelease,
    uint32_t optFlags
  ) const {
  static_assert(sizeof(T) <= 8, "Required: sizeof(T) <= 8");
  using pccl::utility::loadConst;
  using pccl::gin::internal::teamRankToGinRank;
  coop.sync();
  if (coop.thread_rank() == 0) {
    pcclGinCall<pcclGinApi_PutValue>(this->_makeCtx(),
      pcclCoopThread(), teamRankToGinRank(this->comm, team, peer),
      loadConst(&dstWin->ginWins[this->connectionId]),
      4096*size_t(loadConst(&dstWin->ginOffset4K)) + dstOffset,
      value,
      pcclGin_getSignalDescriptor(*this, remoteAction),
      pcclGin_getSignalOp(remoteAction),
      pcclGin_getSignalOpArg(remoteAction),
      pcclGin_isDescriptor(descriptor),
      pcclGin_getDescriptor(descriptor),
      requiredRelease, givenRelease, optFlags
    );
  }
  coop.sync();
}

PCCL_DEVICE_INLINE void pcclGinPutValue_v2(
    pcclGin_C* net,
    pcclTeam team, int peer,
    pcclWindow_t dstWin, size_t dstOffset,
    uint64_t value, size_t size,
    bool isSignal, pcclGinSignal_t signalId, pcclGinSignalOp_t signalOp, uint64_t signalOpArg,
    pcclCoopAny coop,
    bool isDescriptor, pcclGinDescriptorSmem* descriptor,
    hggc::thread_scope givenRelease, hggc::thread_scope requiredRelease,
    uint32_t optFlags
  ) {
  using pccl::utility::loadConst;
  using pccl::gin::internal::teamRankToGinRank;
  coop.sync();
  if (coop.thread_rank() == 0) {
    pcclGinSignalDescriptor signal{};
    signal.type = PCCL_GIN_SIGNAL_TYPE_NONE;
    if (isSignal) {
      signal.type = PCCL_GIN_SIGNAL_TYPE_INDEXED;
      signal.indexedSignal.signalId = signalId;
    }
    pcclGinCtx ctx = pcclGin_C_makeCtx(net);
    // Dispatch based on size to call with appropriate type
    if (size == 1) {
      pcclGinCall<pcclGinApi_PutValue>(ctx,
        pcclCoopThread(), teamRankToGinRank(net->comm, team, peer),
        loadConst(&dstWin->ginWins[net->connectionId]),
        4096*size_t(loadConst(&dstWin->ginOffset4K)) + dstOffset,
        (uint8_t)value,
        signal, signalOp, signalOpArg,
        isDescriptor, descriptor, requiredRelease, givenRelease, optFlags);
    } else if (size == 2) {
      pcclGinCall<pcclGinApi_PutValue>(ctx,
        pcclCoopThread(), teamRankToGinRank(net->comm, team, peer),
        loadConst(&dstWin->ginWins[net->connectionId]),
        4096*size_t(loadConst(&dstWin->ginOffset4K)) + dstOffset,
        (uint16_t)value,
        signal, signalOp, signalOpArg,
        isDescriptor, descriptor, requiredRelease, givenRelease, optFlags);
    } else if (size == 4) {
      pcclGinCall<pcclGinApi_PutValue>(ctx,
        pcclCoopThread(), teamRankToGinRank(net->comm, team, peer),
        loadConst(&dstWin->ginWins[net->connectionId]),
        4096*size_t(loadConst(&dstWin->ginOffset4K)) + dstOffset,
        (uint32_t)value,
        signal, signalOp, signalOpArg,
        isDescriptor, descriptor, requiredRelease, givenRelease, optFlags);
    } else {
      pcclGinCall<pcclGinApi_PutValue>(ctx,
        pcclCoopThread(), teamRankToGinRank(net->comm, team, peer),
        loadConst(&dstWin->ginWins[net->connectionId]),
        4096*size_t(loadConst(&dstWin->ginOffset4K)) + dstOffset,
        value,
        signal, signalOp, signalOpArg,
        isDescriptor, descriptor, requiredRelease, givenRelease, optFlags);
    }
  }
  coop.sync();
}

PCCL_DEVICE_INLINE void pcclGinPutValue(
    pcclGin_C* net,
    pcclTeam team, int peer,
    pcclWindow_t dstWin, size_t dstOffset,
    uint64_t value, size_t size,
    bool isSignal, pcclGinSignal_t signalId, pcclGinSignalOp_t signalOp, uint64_t signalOpArg,
    pcclCoopAny coop,
    bool isDescriptor, pcclGinDescriptorSmem* descriptor,
    hggc::thread_scope givenRelease, hggc::thread_scope requiredRelease
  ) {
  pcclGinPutValue_v2(net, team, peer, dstWin, dstOffset, value, size, isSignal, signalId, signalOp,
                    signalOpArg, coop, isDescriptor, descriptor, givenRelease, requiredRelease,
                    pcclGinOptFlagsDefault);
}

template<unsigned beMask>
template<
  typename T,
  typename RemoteAction, // one of pcclGin_{None|SignalInc}
  typename Coop,
  typename DescriptorSmem
>
PCCL_DEVICE_INLINE void pcclGin_BackendMask<beMask>::putValue(
    pcclTeam team, int peer,
    pcclSymPtr<T> dst, T value,
    RemoteAction remoteAction,
    Coop coop,
    DescriptorSmem descriptor,
    hggc::thread_scope givenRelease,
    hggc::thread_scope requiredRelease,
    uint32_t optFlags
  ) const {
  static_assert(sizeof(T) <= sizeof(uint64_t), "Required: T must fit into 64 bits");
  this->putValue(
    team, peer, dst.window, dst.offset, value, remoteAction, coop, descriptor, givenRelease, requiredRelease, optFlags
  );
}

template<unsigned beMask>
template<typename RemoteAction, typename Coop, typename DescriptorSmem>
PCCL_DEVICE_INLINE void pcclGin_BackendMask<beMask>::signal(
    pcclTeam team, int peer, RemoteAction action, Coop coop, DescriptorSmem descriptor,
    hggc::thread_scope givenRelease,
    hggc::thread_scope requiredRelease,
    uint32_t optFlags
  ) const {
  using pccl::gin::internal::teamRankToGinRank;
  coop.sync();
  if (coop.thread_rank() == 0) {
    pcclGinCall<pcclGinApi_Put>(this->_makeCtx(),
      pcclCoopThread(), teamRankToGinRank(this->comm, team, peer),
      /*hasWins=*/false, nullptr, 0, nullptr, 0, 0,
      pcclGin_getSignalDescriptor(*this, action),
      pcclGin_getSignalOp(action),
      pcclGin_getSignalOpArg(action),
      /*hasCounter=*/false, 0,
      pcclGin_isDescriptor(descriptor),
      pcclGin_getDescriptor(descriptor),
      requiredRelease, givenRelease, optFlags
    );
  }
  coop.sync();
}

PCCL_DEVICE_INLINE void pcclGinSignal_v2(
    pcclGin_C* net,
    pcclTeam team, int peer,
    bool isSignal, pcclGinSignal_t signalId, pcclGinSignalOp_t signalOp, uint64_t signalOpArg,
    pcclCoopAny coop,
    bool isDescriptor, pcclGinDescriptorSmem* descriptor,
    hggc::thread_scope givenRelease, hggc::thread_scope requiredRelease,
    uint32_t optFlags
  ) {
  using pccl::gin::internal::teamRankToGinRank;
  coop.sync();
  pcclGinSignalDescriptor signal{};
  signal.type = PCCL_GIN_SIGNAL_TYPE_NONE;
  if (isSignal) {
    signal.type = PCCL_GIN_SIGNAL_TYPE_INDEXED;
    signal.indexedSignal.signalId = signalId;
  }
  if (coop.thread_rank() == 0) {
    pcclGinCtx ctx = pcclGin_C_makeCtx(net);
    pcclGinCall<pcclGinApi_Put>(ctx,
      pcclCoopThread(), teamRankToGinRank(net->comm, team, peer),
      /*hasWins=*/false, nullptr, 0, nullptr, 0, 0,
      signal,
      signalOp,
      signalOpArg,
      /*hasCounter=*/false, 0,
      isDescriptor,
      descriptor,
      requiredRelease, givenRelease,
      optFlags
    );
  }
  coop.sync();
}

PCCL_DEVICE_INLINE void pcclGinSignal(
    pcclGin_C* net,
    pcclTeam team, int peer,
    bool isSignal, pcclGinSignal_t signalId, pcclGinSignalOp_t signalOp, uint64_t signalOpArg,
    pcclCoopAny coop,
    bool isDescriptor, pcclGinDescriptorSmem* descriptor,
    hggc::thread_scope givenRelease, hggc::thread_scope requiredRelease
  ) {
  pcclGinSignal_v2(net, team, peer, isSignal, signalId, signalOp, signalOpArg, coop, isDescriptor,
                  descriptor, givenRelease, requiredRelease, pcclGinOptFlagsDefault);
}

template<unsigned beMask>
template<typename Coop>
PCCL_DEVICE_INLINE void pcclGin_BackendMask<beMask>::flush(Coop coop, hggc::memory_order ord) const {
  coop.sync();
  pcclGinCall<pcclGinApi_Flush>(this->_makeCtx(), coop, ord, this->comm.abortFlag);
  coop.sync();
}

PCCL_DEVICE_INLINE void pcclGinFlush(
    pcclGin_C* net,
    pcclCoopAny coop,
    hggc::memory_order ord
  ) {
  coop.sync();
  pcclGinCtx ctx = pcclGin_C_makeCtx(net);
  pcclGinCall<pcclGinApi_Flush>(ctx, coop, ord, net->comm.abortFlag);
  coop.sync();
}

template<unsigned beMask>
template<typename Coop>
PCCL_DEVICE_INLINE void pcclGin_BackendMask<beMask>::waitCounter(
    Coop coop, pcclGinCounter_t counter, uint64_t least, int bits, hggc::memory_order ord
  ) const {
  using pccl::utility::testAbort;
  uint32_t steps = 0;
  coop.sync();
  if (coop.thread_rank() == 0) {
    uint64_t* ptr = pcclGinCall<pcclGinApi_GetCounterPtr>(this->_makeCtx(), counter);
    uint64_t got;
    #pragma unroll 1
    do got = hggc::atomic_ref<uint64_t>{*ptr}.load(ord);
    while (!pccl::utility::rollingLessEq(least, got, bits) && !testAbort(this->comm.abortFlag, steps));
  }
  coop.sync();
}

PCCL_DEVICE_INLINE void pcclGinWaitCounter(
    pcclGin_C* net,
    pcclCoopAny coop,
    pcclGinCounter_t counter,
    uint64_t least,
    int bits,
    hggc::memory_order ord
  ) {
  using pccl::utility::testAbort;
  uint32_t steps = 0;
  coop.sync();
  if (coop.thread_rank() == 0) {
    pcclGinCtx ctx = pcclGin_C_makeCtx(net);
    uint64_t* ptr = pcclGinCall<pcclGinApi_GetCounterPtr>(ctx, counter);
    uint64_t got;
    #pragma unroll 1
    do got = hggc::atomic_ref<uint64_t>{*ptr}.load(ord);
    while (!pccl::utility::rollingLessEq(least, got, bits) && !testAbort(net->comm.abortFlag, steps));
  }
  coop.sync();
}

template<unsigned beMask>
PCCL_DEVICE_INLINE uint64_t pcclGin_BackendMask<beMask>::readCounter(pcclGinCounter_t counter, int bits, hggc::memory_order ord) const {
  uint64_t* ptr = pcclGinCall<pcclGinApi_GetCounterPtr>(this->_makeCtx(), counter);
  uint64_t mask = uint64_t(-1)>>(64-bits);
  return mask & hggc::atomic_ref<uint64_t>{*ptr}.load(ord);
}

PCCL_DEVICE_INLINE uint64_t pcclGinReadCounter(
    pcclGin_C* net,
    pcclGinCounter_t counter,
    int bits,
    hggc::memory_order ord
  ) {
  pcclGinCtx ctx = pcclGin_C_makeCtx(net);
  uint64_t* ptr = pcclGinCall<pcclGinApi_GetCounterPtr>(ctx, counter);
  uint64_t mask = uint64_t(-1)>>(64-bits);
  return mask & hggc::atomic_ref<uint64_t>{*ptr}.load(ord);
}

template<unsigned beMask>
PCCL_DEVICE_INLINE uint64_t* pcclGin_BackendMask<beMask>::getSignalShadowPtr(pcclGinSignal_t signal) const {
  return &this->_signalShadows[signal];
}

PCCL_DEVICE_INLINE uint64_t* pcclGinGetSignalShadowPtr(
    pcclGin_C* net,
    pcclGinSignal_t signal
  ) {
  return &net->_signalShadows[signal];
}

template<unsigned beMask>
PCCL_DEVICE_INLINE void pcclGin_BackendMask<beMask>::increaseSignalShadow(pcclGinSignal_t signal, uint64_t delta) const {
  asm volatile("red.relaxed.cta.add.u64 [%0],%1;" :: "l"(this->_signalShadows + signal), "l"(delta) : "memory");
}

template<unsigned beMask>
PCCL_DEVICE_INLINE uint64_t pcclGin_BackendMask<beMask>::readSignal(pcclGinSignal_t signal, int bits, hggc::memory_order ord) const {
  uint64_t* ptr = pcclGinCall<pcclGinApi_GetSignalPtr>(this->_makeCtx(), signal);
  uint64_t mask = uint64_t(-1)>>(64-bits);
  return mask & hggc::atomic_ref<uint64_t>{*ptr}.load(ord);
}

template<unsigned beMask>
PCCL_DEVICE_INLINE uint64_t pcclGin_BackendMask<beMask>::readSignal(pcclWindow_t signalWindow, size_t signalOffset, int bits, hggc::memory_order ord) const {
  uint64_t* ptr = (uint64_t*)pcclGetLocalPointer(signalWindow, signalOffset);
  uint64_t mask = uint64_t(-1)>>(64-bits);
  return mask & hggc::atomic_ref<uint64_t>{*ptr}.load(ord);
}

PCCL_DEVICE_INLINE uint64_t pcclGinReadSignal(
    pcclGin_C* net,
    pcclGinSignal_t signal,
    int bits,
    hggc::memory_order ord
  ) {
  pcclGinCtx ctx = pcclGin_C_makeCtx(net);
  uint64_t* ptr = pcclGinCall<pcclGinApi_GetSignalPtr>(ctx, signal);
  uint64_t mask = uint64_t(-1)>>(64-bits);
  return mask & hggc::atomic_ref<uint64_t>{*ptr}.load(ord);
}


template<unsigned beMask>
template<typename Coop, typename DescriptorSmem>
PCCL_DEVICE_INLINE void pcclGin_BackendMask<beMask>::wait(pcclGinRequest_t& request, Coop coop,
                                                          DescriptorSmem descriptor, hggc::memory_order ord) const {
  coop.sync();
  if (coop.thread_rank() == 0) {
    pcclGinCall<pcclGinApi_Wait>(this->_makeCtx(), request,
        pcclGin_isDescriptor(descriptor), pcclGin_getDescriptor(descriptor), ord, this->comm.abortFlag);
  }
  coop.sync();
}

template<unsigned beMask>
template<typename Coop>
PCCL_DEVICE_INLINE void pcclGin_BackendMask<beMask>::flushAsync(pcclTeam team, uint32_t peer, pcclGinRequest_t* outRequest,
                                                                Coop coop, uint32_t optFlags) const {
  using pccl::gin::internal::teamRankToGinRank;
  coop.sync();
  if (coop.thread_rank() == 0) {
    pcclGinCall<pcclGinApi_FlushAsync>(this->_makeCtx(), teamRankToGinRank(this->comm, team, peer), outRequest,
                                       optFlags);
  }
  coop.sync();
}

template<unsigned beMask>
template<typename Coop, typename DescriptorSmem, typename SegmentType>
PCCL_DEVICE_INLINE void pcclGin_BackendMask<beMask>::get(
    pcclTeam team, int peer,
    pcclWindow_t remoteWnd, size_t remoteOffset,
    pcclWindow_t localWnd, size_t localOffset,
    size_t bytes, Coop coop,
    DescriptorSmem descriptor,
    uint32_t optFlags, SegmentType bufType) const {
  using pccl::utility::loadConst;
  using pccl::gin::internal::teamRankToGinRank;
  pcclGinCtx_M<beMask> ctx = this->_makeCtx();
  coop.sync();
  if (pcclGin_isDeviceOnly(bufType)) {
    pcclGinCall<pcclGinApi_Get>(ctx, coop,
        teamRankToGinRank(this->comm, team, peer),
        loadConst(&remoteWnd->ginWins[this->connectionId]),
        4096*size_t(loadConst(&remoteWnd->ginOffset4K)) + remoteOffset,
        loadConst(&localWnd->ginWins[this->connectionId]),
        4096*size_t(loadConst(&localWnd->ginOffset4K)) + localOffset, bytes,
        pcclGin_isDescriptor(descriptor),
        pcclGin_getDescriptor(descriptor),
        optFlags);
  } else {
    if (remoteWnd->numSegments == 1 && localWnd->numSegments == 1) {
      pcclGinCall<pcclGinApi_Get>(ctx, coop,
          teamRankToGinRank(this->comm, team, peer),
          loadConst(&remoteWnd->ginWins[this->connectionId]),
          4096*size_t(loadConst(&remoteWnd->ginOffset4K)) + remoteOffset,
          loadConst(&localWnd->ginWins[this->connectionId]),
          4096*size_t(loadConst(&localWnd->ginOffset4K)) + localOffset, bytes,
          pcclGin_isDescriptor(descriptor),
          pcclGin_getDescriptor(descriptor),
          optFlags);
    } else {
      // Multi-segment case: chunk the get across separate per-segment registrations
      int remoteSeg, localSeg;
      size_t remoteSegOffset, localSegOffset;
      pccl::gin::internal::findSegmentFromWindow(remoteWnd, remoteOffset, &remoteSeg, &remoteSegOffset);
      pccl::gin::internal::findSegmentFromWindow(localWnd, localOffset, &localSeg, &localSegOffset);
      size_t remaining = bytes;
      while (remaining > 0) {
        struct pcclSegmentWindow const& remoteSegmentWindow = remoteWnd->ginMultiSegmentWins[remoteSeg];
        struct pcclSegmentWindow const& localSegmentWindow = localWnd->ginMultiSegmentWins[localSeg];
        const size_t remoteRemaining = remoteSegmentWindow.segmentSize - remoteSegOffset;
        const size_t localRemaining = localSegmentWindow.segmentSize - localSegOffset;
        const size_t getSize = pccl::gin::internal::getSegmentChunkSize(remoteRemaining, localRemaining, remaining);
        pcclGinCall<pcclGinApi_Get>(ctx, coop,
            teamRankToGinRank(this->comm, team, peer),
            loadConst(&remoteWnd->ginMultiSegmentWins[remoteSeg].ginWins[this->connectionId]),
            remoteSegOffset,
            loadConst(&localWnd->ginMultiSegmentWins[localSeg].ginWins[this->connectionId]),
            localSegOffset, getSize,
            pcclGin_isDescriptor(descriptor),
            pcclGin_getDescriptor(descriptor),
            optFlags);
        remaining -= getSize;
        pccl::gin::internal::advanceSegmentCursor(&remoteSeg, &remoteSegOffset, getSize, remoteSegmentWindow.segmentSize);
        pccl::gin::internal::advanceSegmentCursor(&localSeg, &localSegOffset, getSize, localSegmentWindow.segmentSize);
      }
    }
  }
  coop.sync();
}

PCCL_DEVICE_INLINE void pcclGinGet(
  pcclGin_C* net,
  pcclTeam team, int peer,
  pcclWindow_t remoteWnd, size_t remoteOffset,
  pcclWindow_t localWnd, size_t localOffset, size_t bytes,
  pcclCoopAny coop,
  bool isDescriptor,
  pcclGinDescriptorSmem* descriptor,
  uint32_t optFlags
) {
  using pccl::utility::loadConst;
  using pccl::gin::internal::teamRankToGinRank;
  coop.sync();
  pcclGinCtx ctx = pcclGin_C_makeCtx(net);
  pcclGinCall<pcclGinApi_Get>(ctx, coop,
      teamRankToGinRank(net->comm, team, peer),
      loadConst(&remoteWnd->ginWins[net->connectionId]),
      4096*size_t(loadConst(&remoteWnd->ginOffset4K)) + remoteOffset,
      loadConst(&localWnd->ginWins[net->connectionId]),
      4096*size_t(loadConst(&localWnd->ginOffset4K)) + localOffset, bytes,
      pcclGin_isDescriptor(descriptor),
      pcclGin_getDescriptor(descriptor),
      optFlags);
  coop.sync();
}

template<unsigned beMask>
template<typename Coop>
PCCL_DEVICE_INLINE void pcclGin_BackendMask<beMask>::waitSignal(Coop coop, pcclGinSignal_t signal, uint64_t least, int bits, hggc::memory_order ord) const {
  using pccl::utility::testAbort;
  uint32_t steps = 0;
  coop.sync();
  if (coop.thread_rank() == 0) {
    uint64_t* ptr = pcclGinCall<pcclGinApi_GetSignalPtr>(this->_makeCtx(), signal);
    uint64_t got;
    #pragma unroll 1
    do got = hggc::atomic_ref<uint64_t>{*ptr}.load(ord);
    while (!pccl::utility::rollingLessEq(least, got, bits) && !testAbort(this->comm.abortFlag, steps));
  }
  coop.sync();
}

template<unsigned beMask>
template<typename Coop>
PCCL_DEVICE_INLINE void pcclGin_BackendMask<beMask>::waitSignal(Coop coop, pcclWindow_t signalWindow, size_t signalOffset, uint64_t least, int bits, hggc::memory_order ord) const {
  using pccl::utility::loadConst;
  using pccl::utility::testAbort;
  uint32_t steps = 0;
  coop.sync();
  if (coop.thread_rank() == 0) {
    uint64_t* ptr = (uint64_t*)pcclGetLocalPointer(signalWindow, signalOffset);
    uint64_t got;
    #pragma unroll 1
    do {
      got = hggc::atomic_ref<uint64_t>{*ptr}.load(ord);
    } while (!pccl::utility::rollingLessEq(least, got, bits) && !testAbort(this->comm.abortFlag, steps));
  }
  coop.sync();
}

PCCL_DEVICE_INLINE void pcclGinWaitSignal(
    pcclGin_C* net,
    pcclCoopAny coop,
    pcclGinSignal_t signal,
    uint64_t least,
    int bits,
    hggc::memory_order ord
  ) {
  using pccl::utility::testAbort;
  uint32_t steps = 0;
  coop.sync();
  if (coop.thread_rank() == 0) {
    pcclGinCtx ctx = pcclGin_C_makeCtx(net);
    uint64_t* ptr = pcclGinCall<pcclGinApi_GetSignalPtr>(ctx, signal);
    uint64_t got;
    #pragma unroll 1
    do got = hggc::atomic_ref<uint64_t>{*ptr}.load(ord);
    while (!pccl::utility::rollingLessEq(least, got, bits) && !testAbort(net->comm.abortFlag, steps));
  }
  coop.sync();
}

template<unsigned beMask>
template<typename Coop>
PCCL_DEVICE_INLINE void pcclGin_BackendMask<beMask>::waitSignalMeetShadow(Coop coop, pcclGinSignal_t signal, int bits, hggc::memory_order ord) const {
  using pccl::utility::testAbort;
  uint32_t steps = 0;
  coop.sync();
  if (coop.thread_rank() == 0) {
    uint64_t* ptr = pcclGinCall<pcclGinApi_GetSignalPtr>(this->_makeCtx(), signal);
    uint64_t least = this->_signalShadows[signal];
    uint64_t got;
    #pragma unroll 1
    do got = hggc::atomic_ref<uint64_t>{*ptr}.load(ord);
    while (!pccl::utility::rollingLessEq(least, got, bits) && !testAbort(this->comm.abortFlag, steps));
  }
  coop.sync();
}

template<unsigned beMask>
template<typename Coop, typename Uint>
PCCL_DEVICE_INLINE void pcclGin_BackendMask<beMask>::waitSignalFollowShadow(Coop coop, pcclGinSignal_t signal, Uint leastDelta, Uint* before, Uint* delta, int bits, hggc::memory_order ord) const {
  using pccl::utility::testAbort;
  uint32_t steps = 0;
  coop.sync();
  uint64_t before64 = this->_signalShadows[signal];
  uint64_t after64;
  if (coop.thread_rank() == 0) {
    uint64_t* ptr = pcclGinCall<pcclGinApi_GetSignalPtr>(this->_makeCtx(), signal);
    #pragma unroll 1
    do after64 = hggc::atomic_ref<uint64_t>{*ptr}.load(ord);
    while (!pccl::utility::rollingLessEq(before64 + leastDelta, after64, bits) && !testAbort(this->comm.abortFlag, steps));
    this->_signalShadows[signal] = after64;
  }
  if (pcclCoopWithinWarp(coop) && bits <= 32) { // do a single __shfl_sync instead of 2
    uint32_t mask = uint32_t(-1)>>(32-bits);
    after64 = pcclCoopBcast(coop, (uint32_t)after64, 0, /*entrySync=*/false);
    *before = (Uint)(mask & before64);
    *delta = (Uint)(mask & (after64 - before64));
  } else {
    uint64_t mask = uint64_t(-1)>>(64-bits);
    after64 = pcclCoopBcast(coop, after64, 0, /*entrySync=*/false);
    *before = (Uint)(mask & before64);
    *delta = (Uint)(mask & (after64 - before64));
  }
}

template<unsigned beMask>
PCCL_DEVICE_INLINE void pcclGin_BackendMask<beMask>::resetCounter(pcclGinCounter_t counter) const {
  pcclGinCall<pcclGinApi_ResetCounter>(this->_makeCtx(), counter);
}

PCCL_DEVICE_INLINE void pcclGinResetCounter(
    pcclGin_C* net,
    pcclGinCounter_t counter
  ) {
  pcclGinCtx ctx = pcclGin_C_makeCtx(net);
  pcclGinCall<pcclGinApi_ResetCounter>(ctx, counter);
}

template<unsigned beMask>
PCCL_DEVICE_INLINE void pcclGin_BackendMask<beMask>::resetSignal(pcclGinSignal_t signal) const {
  pcclGinSignalDescriptor signalDesc;
  signalDesc.type = PCCL_GIN_SIGNAL_TYPE_INDEXED;
  signalDesc.indexedSignal.signalId = signal;
  pcclGinCall<pcclGinApi_ResetSignal>(this->_makeCtx(), signalDesc);
  this->_signalShadows[signal] = 0;
}

template<unsigned beMask>
PCCL_DEVICE_INLINE void pcclGin_BackendMask<beMask>::resetSignal(pcclWindow_t signalWindow, size_t signalOffset) const {
  pcclGinSignalDescriptor signal;
  signal.type = PCCL_GIN_SIGNAL_TYPE_VA;
  signal.vaSignal.signalWindow = pccl::gin::internal::getGinWindow(signalWindow, this->connectionId);
  signal.vaSignal.signalOffset = pccl::gin::internal::windowOffsetToGinOffset(signalWindow, signalOffset);
  signal.vaSignal.pcclWindow = signalWindow;
  pcclGinCall<pcclGinApi_ResetSignal>(this->_makeCtx(), signal);
}

PCCL_DEVICE_INLINE void pcclGinResetSignal(
    pcclGin_C* net,
    pcclGinSignal_t signal
  ) {
  pcclGinCtx ctx = pcclGin_C_makeCtx(net);
  pcclGinSignalDescriptor signalDesc{};
  signalDesc.type = PCCL_GIN_SIGNAL_TYPE_INDEXED;
  signalDesc.indexedSignal.signalId = signal;
  pcclGinCall<pcclGinApi_ResetSignal>(ctx, signalDesc);
  net->_signalShadows[signal] = 0;
}
#endif // PCCL_CHECK_HGGCCC

#endif // _PCCL_DEVICE_GIN_SESSION__FUNCS_H_
