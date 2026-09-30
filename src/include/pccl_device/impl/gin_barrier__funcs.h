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

#if PCCL_CHECK_HGGCCC
template<typename Coop>
PCCL_DEVICE_INLINE pcclGinBarrierSession<Coop>::pcclGinBarrierSession(
    Coop coop, pcclGin net, pcclTeam team, pcclGinBarrierHandle handle, uint32_t barrierIndex
  ):
  pcclGinBarrierSession_internal<Coop>{coop, net, team, handle, (int)barrierIndex} {
  this->signal = handle.signal0 + barrierIndex * team.nRanks;
}

template<typename Coop>
PCCL_DEVICE_INLINE pcclGinBarrierSession<Coop>::pcclGinBarrierSession(
    Coop coop, pcclGin net, pcclTeamTagRail, uint32_t barrierIndex
  ):
  pcclGinBarrierSession(coop, net, pcclTeamRail(net.comm), net.comm.railGinBarrier, barrierIndex) {
}

template<typename Coop>
PCCL_DEVICE_INLINE pcclGinBarrierSession<Coop>::pcclGinBarrierSession(
    Coop coop, pcclGin net, pcclTeamTagWorld, uint32_t barrierIndex
  ):
  pcclGinBarrierSession(coop, net, pcclTeamWorld(net.comm), net.comm.worldGinBarrier, barrierIndex) {
}

template<typename Coop>
PCCL_DEVICE_INLINE pcclGinBarrierSession<Coop>::~pcclGinBarrierSession() {
}

template<typename Coop>
template<bool EnableTimeout>
PCCL_DEVICE_INLINE pcclResult_t pcclGinBarrierSession_internal<Coop>::syncInternal(Coop, hggc::memory_order ord,
                                                                      pcclGinFenceLevel fence, uint64_t timeoutCycles) {
  uint64_t startCycle;
  pcclResult_t ret = pcclSuccess;
  this->coop.sync();
  if PCCL_IF_CONSTEXPR (EnableTimeout) {
    startCycle = clock64();
  }
  #pragma unroll 1
  for (int i=this->coop.thread_rank(); i < this->team.nRanks-1; i += this->coop.size()) {
    // Use a rotating pattern to avoid hot spots
    int peer = 1 + this->team.rank + i;
    if (this->team.nRanks <= peer) peer -= this->team.nRanks;

    // Initiate signal
    this->net.signal(
      this->team, peer, pcclGin_SignalInc{this->signal + this->team.rank}, pcclCoopThread(), pcclGin_None(),
      pccl::utility::releaseOrderOf(ord) != hggc::memory_order_relaxed
        ? hggc::thread_scope_thread
        : hggc::thread_scope_system
    );

    // Load and update barrier state in memory. The load/store should be covered by the GIN signal latency.
    uint32_t* shadowPtr = (uint32_t*)this->net.getSignalShadowPtr(this->signal + peer);
    int waitVal = ++*shadowPtr;

    if PCCL_IF_CONSTEXPR (EnableTimeout) {
      while (true) {
        uint64_t got = this->net.readSignal(this->signal + peer, 32, pccl::utility::acquireOrderOf(ord));
        if (pccl::utility::rollingLessEq(static_cast<uint64_t>(waitVal), got, 32)) break;
        if (clock64() - startCycle >= timeoutCycles) {
          ret = pcclTimeout;
          goto exit;
        }
      }
    } else {
#ifdef PCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG
      // Debug mode: poll with periodic print to diagnose signal hang
      uint64_t pollStart = clock64();
      uint64_t pollCount = 0;
      while (true) {
        uint64_t got = this->net.readSignal(this->signal + peer, 32, pccl::utility::acquireOrderOf(ord));
        if (pccl::utility::rollingLessEq(static_cast<uint64_t>(waitVal), got, 32)) break;
        pollCount++;
        if (pollCount == 1 || (pollCount % 100000000ULL == 0)) {
          uint64_t elapsed = clock64() - pollStart;
          printf("[GIN_BAR_DBG] rank=%d tid=%d peer=%d signal_idx=%d waitVal=%d got=%llu pollCount=%llu elapsed_cycles=%llu\n",
                 this->team.rank, this->coop.thread_rank(), peer,
                 (int)(this->signal + peer), waitVal, (unsigned long long)got,
                 (unsigned long long)pollCount, (unsigned long long)elapsed);
        }
      }
#else
      this->net.waitSignal(pcclCoopThread(), this->signal + peer, waitVal, 32, pccl::utility::acquireOrderOf(ord));
#endif
    }
  }
  goto exit; // Silence a compiler warning.
exit:
  this->coop.sync();
  return ret;
}


template<typename Coop>
PCCL_DEVICE_INLINE void pcclGinBarrierSession<Coop>::sync(Coop coop, hggc::memory_order ord, pcclGinFenceLevel fence) {
  (void)(this->template syncInternal</*EnableTimeout=*/false>(coop, ord, fence, 0ULL));
}

template<typename Coop>
PCCL_DEVICE_INLINE pcclResult_t pcclGinBarrierSession<Coop>::sync(
    Coop coop, hggc::memory_order ord, pcclGinFenceLevel fence, uint64_t timeoutCycles) {
  return this->template syncInternal</*EnableTimeout=*/true>(coop, ord, fence, timeoutCycles);
}
#endif // PCCL_CHECK_HGGCCC

#endif // _PCCL_DEVICE_GIN_BARRIER__FUNCS_H_
