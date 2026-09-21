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
#include "symmetric/oki_utils.cuh"

// P2P completion uses a CQ fence and done signals; no per-copy ICN fence is needed.

// Use one die for small sends to reduce QP submission overhead.
template<int DIE_NUM>
static __device__ __inline__ int p2pDmaActiveDies(size_t bytes, uint32_t oki1DieMax) {
  return (DIE_NUM > 1 && bytes != 0 && bytes < oki1DieMax) ? 1 : DIE_NUM;
}

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

// Slot 0 is local D2D; remote peers fold onto slots 1..qpSlots.
static __device__ __inline__ int p2pDmaSlotOf(int peer, int rank, int nRanks, int qpSlots) {
  int rankToQid = (peer + nRanks - rank) % nRanks; // 0 for self, 1..nRanks-1 otherwise
  if (rankToQid == 0) return 0;                    // guards the modulo when qpSlots == 0
  return 1 + (rankToQid - 1) % qpSlots;
}

static __device__ __inline__ int p2pDmaSlotCount(int qpSlots) {
  return qpSlots + 1;
}

// Match the submission filter: only queues with a posted fence may be polled.
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
