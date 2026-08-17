/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_MEM_BARRIER__FUNCS_H_
#define _NCCL_DEVICE_MEM_BARRIER__FUNCS_H_
#include "lsa_barrier__types.h"
#include "comm__types.h"
#include <cassert>

#if NCCL_CHECK_HGGCCC
template<typename Coop>
NCCL_DEVICE_INLINE ncclLsaBarrierSession<Coop>::ncclLsaBarrierSession(
    Coop coop, ncclDevComm const& comm, ncclTeam team,
    ncclLsaBarrierHandle handle, uint32_t index,
    bool multimem, ncclMultimemHandle mmHandle
  ):
  ncclLsaBarrierSession_internal<Coop>{
    coop, comm, team, handle, (int)index,
    multimem, mmHandle, /*epoch=*/0
  } {
  assert(!multimem && "multimem is not supported on PPU");
  uint32_t* state = (uint32_t*)ncclGetResourceBufferLocalPointer(comm, handle.bufHandle);
  this->epoch = state[(this->multimem ? 0 : 1)*this->handle.nBarriers + this->index];
}

template<typename Coop>
NCCL_DEVICE_INLINE ncclLsaBarrierSession<Coop>::ncclLsaBarrierSession(
    Coop coop, ncclDevComm const& comm, ncclTeamTagLsa, uint32_t index, bool multimem
  ): ncclLsaBarrierSession(
    coop, comm, ncclTeamLsa(comm), comm.lsaBarrier, index, multimem, comm.lsaMultimem
  ) {
}

template<typename Coop>
NCCL_DEVICE_INLINE ncclLsaBarrierSession<Coop>::~ncclLsaBarrierSession() {
  uint32_t* state = (uint32_t*)ncclGetResourceBufferLocalPointer(this->comm, this->handle.bufHandle);
  if (this->coop.thread_rank() == 0) {
    state[(this->multimem ? 0 : 1)*this->handle.nBarriers + this->index] = this->epoch;
  }
  this->coop.sync();
}

template<typename Coop>
NCCL_DEVICE_INLINE void ncclLsaBarrierSession<Coop>::arrive(Coop, hggc::memory_order order) {
  this->coop.sync();
  if (this->multimem) {
    assert(false && "multimem is not supported on PPU");
  #if __HGGC_ARCH__ >= 900
    if (this->coop.thread_rank() == 0) {
      uint32_t* inbox = this->mcInbox(/*multimem=*/true);
      if (nccl::utility::releaseOrderOf(order) != hggc::memory_order_relaxed) {
        asm volatile("multimem.red.release.sys.add.u32 [%0],1;" :: "l"(inbox) : "memory");
      } else {
        asm volatile("multimem.red.relaxed.sys.add.u32 [%0],1;" :: "l"(inbox) : "memory");
      }
    }
  #endif
  } else {
    if (this->team.nRanks > 1) {
      hggc::atomic_thread_fence(nccl::utility::releaseOrderOf(order));
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
NCCL_DEVICE_INLINE pcclResult_t ncclLsaBarrierSession_internal<Coop>::waitInternal(Coop, hggc::memory_order order,
                                                                                   uint64_t timeoutCycles) {
  using nccl::utility::testAbort;
  uint32_t steps;
  uint64_t startCycle;
  pcclResult_t ret = pcclSuccess;
  if NCCL_IF_CONSTEXPR (EnableTimeout) {
    startCycle = clock64();
  } else {
    steps = 0;
  }
  if (this->multimem) {
    assert(false && "multimem is not supported on PPU");
  #if __HGGC_ARCH__ >= 900
    if (this->coop.thread_rank() == 0) {
      hggc::atomic_ref<uint32_t> inbox(*this->mcInbox(/*multimem=*/false));
      #pragma unroll 1
      while (true) {
        uint32_t got = inbox.load(nccl::utility::acquireOrderOf(order));
        if (got - (this->epoch + this->team.nRanks) <= uint32_t(-1)>>1) break;

        if NCCL_IF_CONSTEXPR (EnableTimeout) {
          if (clock64() - startCycle >= timeoutCycles) {
            ret = pcclTimeout;
            goto exit;
          }
        } else {
          if (testAbort(this->comm.abortFlag, steps)) break;
        }
      }
      this->epoch += this->team.nRanks;
    }
  #endif
  } else {
    #pragma unroll 1
    for (int i = this->coop.thread_rank(); i < this->team.nRanks-1; i += this->coop.size()) {
      int peer = i + (this->team.rank <= i ? 1 : 0);
      hggc::atomic_ref<uint32_t> inbox(*this->ucInbox(this->team.rank, peer));
      #pragma unroll 1
      while (true) {
        uint32_t got = inbox.load(nccl::utility::acquireOrderOf(order));
        if (got - (this->epoch + 1) <= uint32_t(-1)>>1) break;

        if NCCL_IF_CONSTEXPR (EnableTimeout) {
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
NCCL_DEVICE_INLINE void ncclLsaBarrierSession<Coop>::wait(Coop coop, hggc::memory_order order) {
  (void)(this->template waitInternal</*EnableTimeout=*/false>(coop, order, 0ULL));
}

template<typename Coop>
NCCL_DEVICE_INLINE pcclResult_t ncclLsaBarrierSession<Coop>::wait(
    Coop coop, hggc::memory_order order, uint64_t timeoutCycles) {
  this->coop.sync();
  return this->template waitInternal</*EnableTimeout=*/true>(coop, order, timeoutCycles);
}

template<typename Coop>
NCCL_DEVICE_INLINE void ncclLsaBarrierSession<Coop>::sync(Coop coop, hggc::memory_order order) {
  this->arrive(coop, order);
  this->wait(coop, order);
}

template<typename Coop>
NCCL_DEVICE_INLINE pcclResult_t ncclLsaBarrierSession<Coop>::sync(
    Coop coop, hggc::memory_order order, uint64_t timeoutCycles) {
  this->arrive(coop, order);
  return this->wait(coop, order, timeoutCycles);
}
#endif // NCCL_CHECK_HGGCCC

#endif // _NCCL_DEVICE_MEM_BARRIER__FUNCS_H_
