/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_MEM_BARRIER__FUNCS_H_
#define _PCCL_DEVICE_MEM_BARRIER__FUNCS_H_
#include "lsa_barrier__types.h"
#include "comm__types.h"
#include <cassert>

#if PCCL_CHECK_HGGCCC
template<typename Coop>
PCCL_DEVICE_INLINE pcclLsaBarrierSession<Coop>::pcclLsaBarrierSession(
    Coop coop, pcclDevComm const& comm, pcclTeam team,
    pcclLsaBarrierHandle handle, uint32_t index,
    bool multimem, pcclMultimemHandle mmHandle
  ):
  pcclLsaBarrierSession_internal<Coop>{
    coop, comm, team, handle, (int)index,
    multimem, mmHandle, /*epoch=*/0
  } {
  assert(!multimem && "multimem is not supported on PPU");
  uint32_t* state = (uint32_t*)pcclGetResourceBufferLocalPointer(comm, handle.bufHandle);
  this->epoch = state[(this->multimem ? 0 : 1)*this->handle.nBarriers + this->index];
}

template<typename Coop>
PCCL_DEVICE_INLINE pcclLsaBarrierSession<Coop>::pcclLsaBarrierSession(
    Coop coop, pcclDevComm const& comm, pcclTeamTagLsa, uint32_t index, bool multimem
  ): pcclLsaBarrierSession(
    coop, comm, pcclTeamLsa(comm), comm.lsaBarrier, index, multimem, comm.lsaMultimem
  ) {
}

template<typename Coop>
PCCL_DEVICE_INLINE pcclLsaBarrierSession<Coop>::~pcclLsaBarrierSession() {
  uint32_t* state = (uint32_t*)pcclGetResourceBufferLocalPointer(this->comm, this->handle.bufHandle);
  if (this->coop.thread_rank() == 0) {
    state[(this->multimem ? 0 : 1)*this->handle.nBarriers + this->index] = this->epoch;
  }
  this->coop.sync();
}

template<typename Coop>
PCCL_DEVICE_INLINE void pcclLsaBarrierSession<Coop>::arrive(Coop, hggc::memory_order order) {
  this->coop.sync();
  if (this->multimem) {
    assert(false && "multimem is not supported on PPU");
  } else {
    if (this->team.nRanks > 1) {
      hggc::atomic_thread_fence(pccl::utility::releaseOrderOf(order));
    }
    #pragma unroll 1
    for (int i = this->coop.thread_rank(); i < this->team.nRanks-1; i += this->coop.size()) {
      int peer = i + (this->team.rank <= i ? 1 : 0);
      hggc::atomic_ref<uint32_t> inbox(*this->ucInbox(peer, this->team.rank));
      inbox.store(this->epoch+1, hggc::memory_order_relaxed);
    }
  }
}

template<typename Coop>
template<bool EnableTimeout>
PCCL_DEVICE_INLINE pcclResult_t pcclLsaBarrierSession_internal<Coop>::waitInternal(Coop, hggc::memory_order order,
                                                                                   uint64_t timeoutCycles) {
  using pccl::utility::testAbort;
  uint32_t steps;
  uint64_t startCycle;
  pcclResult_t ret = pcclSuccess;
  if PCCL_IF_CONSTEXPR (EnableTimeout) {
    startCycle = clock64();
  } else {
    steps = 0;
  }
  if (this->multimem) {
    assert(false && "multimem is not supported on PPU");
  } else {
    #pragma unroll 1
    for (int i = this->coop.thread_rank(); i < this->team.nRanks-1; i += this->coop.size()) {
      int peer = i + (this->team.rank <= i ? 1 : 0);
      hggc::atomic_ref<uint32_t> inbox(*this->ucInbox(this->team.rank, peer));
      #pragma unroll 1
      while (true) {
        uint32_t got = inbox.load(pccl::utility::acquireOrderOf(order));
        if (got - (this->epoch + 1) <= uint32_t(-1)>>1) break;

        if PCCL_IF_CONSTEXPR (EnableTimeout) {
          if (clock64() - startCycle >= timeoutCycles) {
            ret = pcclTimeout;
            goto exit;
          }
        } else {
          if (testAbort(this->comm.abortFlag, steps)) break;
        }
      }
    }
    this->epoch += 1;
  }
  goto exit; // Silence a compiler warning.
exit:
  this->coop.sync();
  return ret;
}

template<typename Coop>
PCCL_DEVICE_INLINE void pcclLsaBarrierSession<Coop>::wait(Coop coop, hggc::memory_order order) {
  (void)(this->template waitInternal</*EnableTimeout=*/false>(coop, order, 0ULL));
}

template<typename Coop>
PCCL_DEVICE_INLINE pcclResult_t pcclLsaBarrierSession<Coop>::wait(
    Coop coop, hggc::memory_order order, uint64_t timeoutCycles) {
  this->coop.sync();
  return this->template waitInternal</*EnableTimeout=*/true>(coop, order, timeoutCycles);
}

template<typename Coop>
PCCL_DEVICE_INLINE void pcclLsaBarrierSession<Coop>::sync(Coop coop, hggc::memory_order order) {
  this->arrive(coop, order);
  this->wait(coop, order);
}

template<typename Coop>
PCCL_DEVICE_INLINE pcclResult_t pcclLsaBarrierSession<Coop>::sync(
    Coop coop, hggc::memory_order order, uint64_t timeoutCycles) {
  this->arrive(coop, order);
  return this->wait(coop, order, timeoutCycles);
}
#endif // PCCL_CHECK_HGGCCC

#endif // _PCCL_DEVICE_MEM_BARRIER__FUNCS_H_
