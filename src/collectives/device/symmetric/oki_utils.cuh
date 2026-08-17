/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Primitives shared by both OKI transports -- the collectives in
// ce_kernel_oki.cuh and the symmetric p2p path in p2p_dma_oki.cuh. These carry
// no protocol: they split a transfer across dies, count 1 GB segments, index the
// QP array, and write the copy WQEs. Each transport keeps its own protocol logic
// (flow control, mailbox handshake, phase structure) in its own file.

#ifndef PCCL_OKI_UTILS_CUH_
#define PCCL_OKI_UTILS_CUH_

#include <algorithm>
#include "lt_oki.h"
#include "lt_oki_device.cuh"
#include "ce_coll_oki.h"

// Calculates how many DMA segments are needed to transfer `size` bytes.
// OKI DMA has a maximum transfer size limit (PCCL_LT_OKI_MAX_DMA_BYTE_CNT).
// For large transfers, we must split into multiple segments.
// Example: if size=3GB and maxDMA=1GB, returns 3 segments.
static __device__ __inline__ uint32_t calcDmaSegmentCount(ssize_t size) {
  uint32_t segCount = 0;
  ssize_t copied = 0;
  while (copied < size) {
    ssize_t copySize = std::min(size - copied, (ssize_t)PCCL_LT_OKI_MAX_DMA_BYTE_CNT);
    copied += copySize;
    segCount += 1;
  }
  return segCount;
}

// QP index of one (group, die). A QP group is DIE_NUM queues (one per die);
// group 0 is the local D2D pair (isRemote=false), groups 1.. are remote ICN
// queues. Matches the okiQps allocation order in init.cc, so every kernel that
// touches the array -- collective submit/sync and the p2p path -- indexes it
// identically.
static __device__ __inline__ int okiQpGroupToQid(int group, int laneId, int dieNum) {
  if (group == 0) return laneId;
  return dieNum + (group - 1) * dieNum + laneId;
}

// This lane's slice of a `DIES`-way split of `size`, computed in one place so the
// sizing pass and the WQE writer never disagree: `offset` is where the lane's
// bytes start (laneId * size/DIES) and `bytes` is how many it copies (size/DIES,
// the last lane taking the remainder). DIES is a template argument so the divide
// folds to a shift (or vanishes for DIES == 1) instead of a runtime 64-bit
// division.
struct okiLaneSlice {
  ssize_t offset;
  ssize_t bytes;
};

template<int DIES>
static __device__ __inline__ okiLaneSlice okiLaneSliceOf(size_t size, int laneId) {
  ssize_t splitSize = size / DIES;
  ssize_t remainSize = (ssize_t)size - laneId * splitSize;
  if (laneId != DIES - 1) remainSize = (remainSize > splitSize) ? splitSize : remainSize;
  okiLaneSlice s;
  s.offset = laneId * splitSize;
  s.bytes = remainSize > 0 ? remainSize : 0;
  return s;
}

// Byte-only accessor for the reservation sizing pass, which needs the count but
// not the pointers.
template<int DIES>
static __device__ __inline__ ssize_t okiLaneBytes(size_t size, int laneId) {
  return okiLaneSliceOf<DIES>(size, laneId).bytes;
}

// Write one transfer's DMA copy WQEs for this lane.
//
// The payload is split across `DIES` lanes -- pass DIE_NUM to use every die, or 1
// to keep a small transfer on die 0 -- and each lane's share is then segmented at
// PCCL_LT_OKI_MAX_DMA_BYTE_CNT. `local` selects the intra-PPU D2D engine instead
// of the ICN path. When `withIcnFence` is set, one ICN fence WQE follows the last
// segment of a remote transfer (not one per segment), which is how the
// collectives signal remote DMA completion.
//
// Returns the number of WQEs written, the fence included. Callers that post
// exactly what they wrote need this: ltOkiReserveWqeSlots rounds the reservation
// up, so posting the reserved count would submit uninitialised WQEs.
template<int DIES>
static __device__ __inline__ uint32_t okiWriteCopyWqe(
  int laneId,
  ltOkiQP_t qp,
  size_t size,
  uint8_t* srcPtr,
  uint8_t* dstPtr,
  bool local,
  bool withIcnFence = false,
  const uint64_t* remoteFenceAddr = nullptr,
  uint64_t remoteFenceValue = 0
) {
  if (size == 0) return 0;
  okiLaneSlice slice = okiLaneSliceOf<DIES>(size, laneId);
  ssize_t remainSize = slice.bytes;
  if (remainSize <= 0) return 0;
  uint64_t curSrc = (uint64_t)(srcPtr + slice.offset);
  uint64_t curDst = (uint64_t)(dstPtr + slice.offset);
  uint32_t segCount = calcDmaSegmentCount(remainSize);
  uint32_t written = 0;
  // Original CE structure: segCount-1 full segments, then the remainder segment.
  for (uint32_t ii = 0; ii + 1 < segCount; ++ii) {
    uint32_t segSize = PCCL_LT_OKI_MAX_DMA_BYTE_CNT;
    if (local) {
      ltOkiQPWriteDmaCopyWqe(qp, curSrc, curDst, segSize, laneId, false);
    } else {
      ltOkiQPWriteIcnDmaCopyWqe(qp, curSrc, curDst, segSize, true);  // Local2Remote
    }
    curSrc += segSize;
    curDst += segSize;
    remainSize -= segSize;
    written++;
  }
  if (local) {
    ltOkiQPWriteDmaCopyWqe(qp, curSrc, curDst, remainSize, laneId, false);
  } else {
    ltOkiQPWriteIcnDmaCopyWqe(qp, curSrc, curDst, remainSize, true);  // Local2Remote
    if (withIcnFence) {
      ltOkiQPWriteIcnFenceWqe(qp, (uint64_t)remoteFenceAddr, remoteFenceValue, /*relax*/false);
      written++;
    }
  }
  written++; // remainder copy
  return written;
}

#endif // PCCL_OKI_UTILS_CUH_
