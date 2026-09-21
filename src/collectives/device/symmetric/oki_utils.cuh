/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Shared OKI queue and copy primitives; protocols remain in their callers.

#ifndef PCCL_OKI_UTILS_CUH_
#define PCCL_OKI_UTILS_CUH_

#include <algorithm>
#include "lt_oki.h"
#include "lt_oki_device.cuh"
#include "ce_coll_oki.h"

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

// Match init.cc: local group first, then remote groups, with one QP per die.
static __device__ __inline__ int okiQpGroupToQid(int group, int laneId, int dieNum) {
  if (group == 0) return laneId;
  return dieNum + (group - 1) * dieNum + laneId;
}

// The last die takes the remainder; sizing and submission must use the same split.
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

template<int DIES>
static __device__ __inline__ ssize_t okiLaneBytes(size_t size, int laneId) {
  return okiLaneSliceOf<DIES>(size, laneId).bytes;
}

// Returns emitted WQEs, including the optional ICN fence after the final remote segment.
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
