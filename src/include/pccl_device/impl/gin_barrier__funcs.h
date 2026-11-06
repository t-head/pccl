/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_GIN_BARRIER__FUNCS_H_
#define _PCCL_DEVICE_GIN_BARRIER__FUNCS_H_
#include "gin_barrier__types.h"
#include "gin__funcs.h"

#if PCCL_CHECK_HGGCCC
template <typename Coop>
PCCL_DEVICE_INLINE pcclGinBarrierSession<Coop>::pcclGinBarrierSession(
  Coop coop, pcclGin net, pcclTeam team, pcclGinBarrierHandle handle, uint32_t barrierIndex)
  : pcclGinBarrierSession_internal<Coop>{coop, net, team, handle, (int)barrierIndex} {
  this->signal = handle.signal0 + barrierIndex * team.nRanks;
  this->fenceAllContexts = false;
}
#endif

#if PCCL_CHECK_HGGCCC
template <typename Coop>
PCCL_DEVICE_INLINE pcclGinBarrierSession<Coop>::pcclGinBarrierSession(Coop coop, pcclGin net, pcclTeamTagRail,
                                                                      uint32_t barrierIndex)
  : pcclGinBarrierSession(coop, net, pcclTeamRail(net.comm), net.comm.railGinBarrier, barrierIndex) {}
#endif

#if PCCL_CHECK_HGGCCC
template <typename Coop>
PCCL_DEVICE_INLINE pcclGinBarrierSession<Coop>::pcclGinBarrierSession(Coop coop, pcclGin net, pcclTeamTagWorld,
                                                                      uint32_t barrierIndex)
  : pcclGinBarrierSession(coop, net, pcclTeamWorld(net.comm), net.comm.worldGinBarrier, barrierIndex) {}
#endif

// All-contexts constructors: build a single-context gin (context 0) for the signal/wait
// path, then flip the `fenceAllContexts` flag so the fence iterates every GIN context on
// the comm.
#if PCCL_CHECK_HGGCCC
template <typename Coop>
PCCL_DEVICE_INLINE pcclGinBarrierSession<Coop>::pcclGinBarrierSession(
  Coop coop, pcclGinAllContexts allCtx, pcclTeam team, pcclGinBarrierHandle handle, uint32_t barrierIndex)
  : pcclGinBarrierSession_internal<Coop>{coop, pcclGin(allCtx.comm, 0), team, handle, (int)barrierIndex} {
  this->signal = handle.signal0 + barrierIndex * team.nRanks;
  this->fenceAllContexts = true;
}
#endif

#if PCCL_CHECK_HGGCCC
template <typename Coop>
PCCL_DEVICE_INLINE pcclGinBarrierSession<Coop>::pcclGinBarrierSession(Coop coop, pcclGinAllContexts allCtx,
                                                                      pcclTeamTagRail, uint32_t barrierIndex)
  : pcclGinBarrierSession(coop, allCtx, pcclTeamRail(allCtx.comm), allCtx.comm.railGinBarrier, barrierIndex) {}
#endif

#if PCCL_CHECK_HGGCCC
template <typename Coop>
PCCL_DEVICE_INLINE pcclGinBarrierSession<Coop>::pcclGinBarrierSession(Coop coop, pcclGinAllContexts allCtx,
                                                                      pcclTeamTagWorld, uint32_t barrierIndex)
  : pcclGinBarrierSession(coop, allCtx, pcclTeamWorld(allCtx.comm), allCtx.comm.worldGinBarrier, barrierIndex) {}
#endif

#if PCCL_CHECK_HGGCCC
template <typename Coop>
PCCL_DEVICE_INLINE pcclGinBarrierSession<Coop>::~pcclGinBarrierSession() {}
#endif

#if PCCL_CHECK_HGGCCC
template <typename Coop>
template <bool EnableTimeout>
PCCL_DEVICE_INLINE pcclResult_t pcclGinBarrierSession_internal<Coop>::syncInternal(
  Coop, hggc::memory_order ord, pcclGinFenceLevel fence, uint64_t timeoutCycles) {
  uint64_t startCycle;
  pcclResult_t ret = pcclSuccess;
  this->coop.sync();

  // Drain outgoing puts/gets across either the bound context or every GIN context on the
  // comm. The multi-context branch flattens (ctx, peer) to 1D and assigns one (ctx, peer)
  // pair per thread so the flush is parallelised on both axes.
  auto fenceFlush = [&](hggc::memory_order order) {
    if (this->fenceAllContexts) {
      int nCtx = (int)this->net.comm.ginContextCount;
      int nPeers = this->team.nRanks;
      int total = nCtx * nPeers;
      #pragma unroll 1
      for (int i = this->coop.thread_rank(); i < total; i += this->coop.size()) {
        int ctx = i / nPeers;
        int peer = i - ctx * nPeers;
        pcclGin scratch(this->net.comm, ctx, this->net.resourceSharingMode);
        pcclGinRequest_t req;
        scratch.flushAsync(this->team, (uint32_t)peer, &req);
        scratch.wait(req, pcclCoopThread{}, pcclGin_None{}, order);
      }
    } else {
      this->net.flush(this->coop, order);
    }
  };
  auto signalPeer = [&](pcclGin& net, int peer) {
    net.signal(this->team, peer, pcclGin_SignalInc{this->signal + this->team.rank}, pcclCoopThread(), pcclGin_None(),
               pccl::utility::releaseOrderOf(ord) != hggc::memory_order_relaxed ? hggc::thread_scope_thread :
                                                                                  hggc::thread_scope_system);
  };

  auto waitForPeer = [&](pcclGin& net, int peer) -> pcclResult_t {
    uint32_t* shadowPtr = (uint32_t*)net.getSignalShadowPtr(this->signal + peer);
    int waitVal = ++*shadowPtr;
    if PCCL_IF_CONSTEXPR (EnableTimeout) {
      uint32_t abortSteps = 0;
      while (true) {
        uint64_t got = net.readSignal(this->signal + peer, 32, pccl::utility::acquireOrderOf(ord));
        if (pccl::utility::rollingLessEq(static_cast<uint64_t>(waitVal), got, 32)) break;
        if (clock64() - startCycle >= timeoutCycles) return pcclTimeout;
        if (pccl::utility::testAbort(net.comm.abortFlag, abortSteps)) return pcclSuccess;
      }
    } else {
      net.waitSignal(pcclCoopThread(), this->signal + peer, waitVal, 32, pccl::utility::acquireOrderOf(ord));
    }
    return pcclSuccess;
  };

  if PCCL_IF_CONSTEXPR (EnableTimeout) {
    startCycle = clock64();
  }

  // Drain conservatively: the legacy plugin ABI has no strong-signal capability
  // query, so do not assume that a signal flushes other connections' traffic.
  if (static_cast<uint32_t>(fence) & static_cast<uint32_t>(pcclGinFenceLevel::Put)) {
    fenceFlush(pccl::utility::acquireOrderOf(ord));
  }

  // Signal/wait with the calling rank's own slot included on Put so self-puts get the same
  // visibility guarantee as puts to other peers. Peer rotation `peer = (rank+1+i) % nRanks`
  // spreads the load and visits self last (only when Put is requested).
  int nPeerSigs = (static_cast<uint32_t>(fence) & static_cast<uint32_t>(pcclGinFenceLevel::Put)) ? this->team.nRanks : this->team.nRanks - 1;
  if (this->fenceAllContexts) {
    // Signal on each context, not just context 0: signals and puts on different QPs are
    // not ordered at the receiving NIC, so a ctx-0 signal could overtake an in-flight
    // ctx-X put. Each context has its own signal memory and shadow slot for the same
    // signal id, so no extra slot allocation is needed.
    int nCtx = (int)this->net.comm.ginContextCount;
    int total = nCtx * nPeerSigs;
    #pragma unroll 1
    for (int i = this->coop.thread_rank(); i < total; i += this->coop.size()) {
      // Unflatten i back into (ctx, peerStep): ctx picks the GIN context to signal on,
      // peerStep is the index into the peer rotation (peer is computed just below).
      int ctx = i / nPeerSigs;
      int peerStep = i - ctx * nPeerSigs;
      int peer = 1 + this->team.rank + peerStep;
      if (this->team.nRanks <= peer) peer -= this->team.nRanks;
      pcclGin scratch(this->net.comm, ctx, this->net.resourceSharingMode);
      signalPeer(scratch, peer);
    }
    #pragma unroll 1
    for (int i = this->coop.thread_rank(); i < total; i += this->coop.size()) {
      int ctx = i / nPeerSigs;
      int peerStep = i - ctx * nPeerSigs;
      int peer = 1 + this->team.rank + peerStep;
      if (this->team.nRanks <= peer) peer -= this->team.nRanks;
      pcclGin scratch(this->net.comm, ctx, this->net.resourceSharingMode);
      if ((ret = waitForPeer(scratch, peer)) != pcclSuccess) goto exit;
    }
  } else {
    #pragma unroll 1
    for (int i = this->coop.thread_rank(); i < nPeerSigs; i += this->coop.size()) {
      int peer = 1 + this->team.rank + i;
      if (this->team.nRanks <= peer) peer -= this->team.nRanks;
      signalPeer(this->net, peer);
    }
    #pragma unroll 1
    for (int i = this->coop.thread_rank(); i < nPeerSigs; i += this->coop.size()) {
      int peer = 1 + this->team.rank + i;
      if (this->team.nRanks <= peer) peer -= this->team.nRanks;
      if ((ret = waitForPeer(this->net, peer)) != pcclSuccess) goto exit;
    }
  }

  // Post-signal flush ensures our prior gets have completed before the barrier returns.
  // Placed after signal/wait so peers don't have to wait for our gets to complete.
  if (static_cast<uint32_t>(fence) & static_cast<uint32_t>(pcclGinFenceLevel::Get)) {
    fenceFlush(pccl::utility::acquireOrderOf(ord));
  }
  goto exit; // Silence a compiler warning.
exit:
  this->coop.sync();
  return ret;
}
#endif

#if PCCL_CHECK_HGGCCC
template <typename Coop>
PCCL_DEVICE_INLINE void pcclGinBarrierSession<Coop>::sync(Coop coop, hggc::memory_order ord, pcclGinFenceLevel fence) {
  (void)(this->template syncInternal</*EnableTimeout=*/false>(coop, ord, fence, 0ULL));
}
#endif

#if PCCL_CHECK_HGGCCC
template <typename Coop>
PCCL_DEVICE_INLINE pcclResult_t pcclGinBarrierSession<Coop>::sync(Coop coop, hggc::memory_order ord,
                                                                  pcclGinFenceLevel fence, uint64_t timeoutCycles) {
  return this->template syncInternal</*EnableTimeout=*/true>(coop, ord, fence, timeoutCycles);
}
#endif

// Free-function GIN barrier: thin wrappers around session construct + sync + destruct.
#if PCCL_CHECK_HGGCCC
template <typename Coop>
PCCL_DEVICE_INLINE void pcclGinBarrier(Coop coop, pcclGin gin, pcclTeam team, pcclGinBarrierHandle handle,
                                       uint32_t index, hggc::memory_order ord, pcclGinFenceLevel fence) {
  pcclGinBarrierSession<Coop> session(coop, gin, team, handle, index);
  session.sync(coop, ord, fence);
}

template <typename Coop>
PCCL_DEVICE_INLINE void pcclGinBarrier(Coop coop, pcclGin gin, pcclTeamTagRail tag, uint32_t index,
                                       hggc::memory_order ord, pcclGinFenceLevel fence) {
  pcclGinBarrierSession<Coop> session(coop, gin, tag, index);
  session.sync(coop, ord, fence);
}

template <typename Coop>
PCCL_DEVICE_INLINE void pcclGinBarrier(Coop coop, pcclGin gin, pcclTeamTagWorld tag, uint32_t index,
                                       hggc::memory_order ord, pcclGinFenceLevel fence) {
  pcclGinBarrierSession<Coop> session(coop, gin, tag, index);
  session.sync(coop, ord, fence);
}

template <typename Coop>
PCCL_DEVICE_INLINE void pcclGinBarrier(Coop coop, pcclGinAllContexts allCtx, pcclTeam team, pcclGinBarrierHandle handle,
                                       uint32_t index, hggc::memory_order ord, pcclGinFenceLevel fence) {
  pcclGinBarrierSession<Coop> session(coop, allCtx, team, handle, index);
  session.sync(coop, ord, fence);
}

template <typename Coop>
PCCL_DEVICE_INLINE void pcclGinBarrier(Coop coop, pcclGinAllContexts allCtx, pcclTeamTagRail tag, uint32_t index,
                                       hggc::memory_order ord, pcclGinFenceLevel fence) {
  pcclGinBarrierSession<Coop> session(coop, allCtx, tag, index);
  session.sync(coop, ord, fence);
}

template <typename Coop>
PCCL_DEVICE_INLINE void pcclGinBarrier(Coop coop, pcclGinAllContexts allCtx, pcclTeamTagWorld tag, uint32_t index,
                                       hggc::memory_order ord, pcclGinFenceLevel fence) {
  pcclGinBarrierSession<Coop> session(coop, allCtx, tag, index);
  session.sync(coop, ord, fence);
}
#endif

#endif // _PCCL_DEVICE_GIN_BARRIER__FUNCS_H_
