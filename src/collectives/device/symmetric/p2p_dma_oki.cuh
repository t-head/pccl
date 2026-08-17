/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
#ifndef PCCL_P2P_DMA_OKI_CUH_
#define PCCL_P2P_DMA_OKI_CUH_

#include "p2p_dma.h"

#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
// Shared, protocol-free OKI primitives (segment split/count, QP index, copy WQE)
// live in oki_utils.cuh; this file holds only the p2p path's own OKI logic.
#include "symmetric/oki_utils.cuh"

// This path calls okiWriteCopyWqe with withIcnFence=false: a self send rides
// slot 0 with local=true, a remote send uses the ICN path, and completion comes
// from the tail CQ fence plus the explicit done write rather than a per-transfer
// ICN fence.


// Small sends run on die 0 only: every QP operation (reserve/write/post/ring)
// costs a host-link round trip through the OKI device library, and at small
// sizes that fixed cost outweighs the second die's bandwidth.
template<int DIE_NUM>
static __device__ __inline__ int p2pDmaActiveDies(size_t bytes, uint32_t oki1DieMax) {
  return (DIE_NUM > 1 && bytes != 0 && bytes < oki1DieMax) ? 1 : DIE_NUM;
}

// The shared split/copy helpers take the die count as a template argument, but on
// this path it is a per-op runtime value (p2pDmaActiveDies above returns 1 for a
// small send, DIE_NUM otherwise). These two wrappers own that runtime->compile-time
// dispatch so it stays out of the shared layer and out of the collectives' calls
// (which pass a literal DIE_NUM and stay branch-free). The extra branch is
// negligible next to each queue's host-link submission cost.
template<int DIE_NUM>
static __device__ __inline__ ssize_t p2pDmaLaneBytes(size_t size, int activeDies, int laneId) {
  return activeDies == 1 ? okiLaneBytes<1>(size, laneId) : okiLaneBytes<DIE_NUM>(size, laneId);
}

template<int DIE_NUM>
static __device__ __inline__ uint32_t p2pDmaWriteCopy(int activeDies, int laneId, ltOkiQP_t qp,
                                                     size_t size, uint8_t* src, uint8_t* dst, bool local) {
  return activeDies == 1
      ? okiWriteCopyWqe<1>(laneId, qp, size, src, dst, local, /*withIcnFence=*/false)
      : okiWriteCopyWqe<DIE_NUM>(laneId, qp, size, src, dst, local, /*withIcnFence=*/false);
}

// Which QP slot a peer rides. rankToQid is the peer's distance from this rank,
// so a self send falls on slot 0 -- the local D2D queues, which never touch the
// ICN -- and remote peers fold onto slots 1..qpSlots. The modulo folds cliques
// wider than the array onto the remote slots it has; when the clique fits, the
// fold is the identity.
static __device__ __inline__ int p2pDmaSlotOf(int peer, int rank, int nRanks, int qpSlots) {
  int rankToQid = (peer + nRanks - rank) % nRanks; // 0 for self, 1..nRanks-1 otherwise
  if (rankToQid == 0) return 0;                    // guards the modulo when qpSlots == 0
  return 1 + (rankToQid - 1) % qpSlots;
}

// Total slots a batch can address: the local D2D set plus every remote slot.
static __device__ __inline__ int p2pDmaSlotCount(int qpSlots) {
  return qpSlots + 1;
}

// The QP index of one (slot, die) is okiQpGroupToQid from oki_utils.cuh: the
// first DIE_NUM entries of comm->okiQps are the local D2D queues and the rest are
// one queue per (remote slot, die), so slot 0 *is* the local set. This p2p path
// and the CE collectives therefore index the same okiQps array identically.

// True when some non-zero-byte send of this batch drives the (slot, lane) queue.
// Recomputed rather than cached so every thread agrees without shared state.
// Must stay the exact complement of the phase-1 sizing filter: a queue reported
// used here is one whose fence was written, and only those may be polled.
template<int DIE_NUM>
static __device__ __inline__ bool p2pDmaSlotLaneUsed(struct pcclP2pDmaWork* sends, int nSend, int slot, int lane,
                                                     int rank, int nRanks, int qpSlots, uint32_t oki1DieMax) {
  for (int i = 0; i < nSend; i++) {
    size_t bytes = (size_t)sends[i].bytes;
    if (bytes == 0) continue;
    if (lane >= p2pDmaActiveDies<DIE_NUM>(bytes, oki1DieMax)) continue;
    if (p2pDmaSlotOf(sends[i].peer, rank, nRanks, qpSlots) != slot) continue;
    return true;
  }
  return false;
}

#endif // PCCL_HGGC_VERSION >= 2.2.0

#endif // PCCL_P2P_DMA_OKI_CUH_
