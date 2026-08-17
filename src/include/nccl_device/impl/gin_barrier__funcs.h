/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
#define _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
#include "gin_barrier__types.h"

#if NCCL_CHECK_HGGCCC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
    Coop coop, ncclGin net, ncclTeam team, ncclGinBarrierHandle handle, uint32_t barrierIndex
  ):
  ncclGinBarrierSession_internal<Coop>{coop, net, team, handle, (int)barrierIndex} {
  this->signal = handle.signal0 + barrierIndex * team.nRanks;
}

template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
    Coop coop, ncclGin net, ncclTeamTagRail, uint32_t barrierIndex
  ):
  ncclGinBarrierSession(coop, net, ncclTeamRail(net.comm), net.comm.railGinBarrier, barrierIndex) {
}

template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
    Coop coop, ncclGin net, ncclTeamTagWorld, uint32_t barrierIndex
  ):
  ncclGinBarrierSession(coop, net, ncclTeamWorld(net.comm), net.comm.worldGinBarrier, barrierIndex) {
}

template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::~ncclGinBarrierSession() {
}

template<typename Coop>
template<bool EnableTimeout>
NCCL_DEVICE_INLINE pcclResult_t ncclGinBarrierSession_internal<Coop>::syncInternal(Coop, hggc::memory_order ord,
                                                                      ncclGinFenceLevel fence, uint64_t timeoutCycles) {
  uint64_t startCycle;
  pcclResult_t ret = pcclSuccess;
  this->coop.sync();
  if NCCL_IF_CONSTEXPR (EnableTimeout) {
    startCycle = clock64();
  }
  #pragma unroll 1
  for (int i=this->coop.thread_rank(); i < this->team.nRanks-1; i += this->coop.size()) {
    // Use a rotating pattern to avoid hot spots
    int peer = 1 + this->team.rank + i;
    if (this->team.nRanks <= peer) peer -= this->team.nRanks;

    // Initiate signal
    this->net.signal(
      this->team, peer, ncclGin_SignalInc{this->signal + this->team.rank}, ncclCoopThread(), ncclGin_None(),
      nccl::utility::releaseOrderOf(ord) != hggc::memory_order_relaxed
        ? hggc::thread_scope_thread
        : hggc::thread_scope_system
    );

    // Load and update barrier state in memory. The load/store should be covered by the GIN signal latency.
    uint32_t* shadowPtr = (uint32_t*)this->net.getSignalShadowPtr(this->signal + peer);
    int waitVal = ++*shadowPtr;

    if NCCL_IF_CONSTEXPR (EnableTimeout) {
      while (true) {
        uint64_t got = this->net.readSignal(this->signal + peer, 32, nccl::utility::acquireOrderOf(ord));
        if (nccl::utility::rollingLessEq(static_cast<uint64_t>(waitVal), got, 32)) break;
        if (clock64() - startCycle >= timeoutCycles) {
          ret = pcclTimeout;
          goto exit;
        }
      }
    } else {
#ifdef NCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG
      // Debug mode: poll with periodic print to diagnose signal hang
      uint64_t pollStart = clock64();
      uint64_t pollCount = 0;
      while (true) {
        uint64_t got = this->net.readSignal(this->signal + peer, 32, nccl::utility::acquireOrderOf(ord));
        if (nccl::utility::rollingLessEq(static_cast<uint64_t>(waitVal), got, 32)) break;
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
      this->net.waitSignal(ncclCoopThread(), this->signal + peer, waitVal, 32, nccl::utility::acquireOrderOf(ord));
#endif
    }
  }
  goto exit; // Silence a compiler warning.
exit:
  this->coop.sync();
  return ret;
}


template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrierSession<Coop>::sync(Coop coop, hggc::memory_order ord, ncclGinFenceLevel fence) {
  (void)(this->template syncInternal</*EnableTimeout=*/false>(coop, ord, fence, 0ULL));
}

template<typename Coop>
NCCL_DEVICE_INLINE pcclResult_t ncclGinBarrierSession<Coop>::sync(
    Coop coop, hggc::memory_order ord, ncclGinFenceLevel fence, uint64_t timeoutCycles) {
  return this->template syncInternal</*EnableTimeout=*/true>(coop, ord, fence, timeoutCycles);
}
#endif // NCCL_CHECK_HGGCCC

#endif // _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
