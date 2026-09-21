/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef __CE_KERNEL_OKI_CUH_
#define __CE_KERNEL_OKI_CUH_

#include "symmetric.h"
#include "devcomm.h"
#include "symmetric/kernel.cuh"
#include "ce_coll.h"
#include "utils.h"
#include "lt_oki.h"
#include "lt_oki_device.cuh"
#include "ce_coll_oki.h"
#include "symmetric/oki_utils.cuh"

// Group 0 is local D2D; remote ops share ICN groups, with peers encoded in destination addresses.
static __device__ __inline__ int okiOpToQpGroup(int opId, int numQpGroups) {
  if (opId == 0) return 0;
  int numRemoteGroups = numQpGroups - 1;
  return 1 + (opId - 1) % numRemoteGroups;
}

// Clamp slices to nElts, matching the host partition without overrunning partial ranks.
struct okiReduceSlice {
  int64_t offsetElts;
  int64_t elts;
};

static __device__ __inline__ okiReduceSlice okiReduceSliceOf(uint64_t nElts, int64_t chunkElts, int rank) {
  int64_t total = (int64_t)nElts;
  int64_t off = (int64_t)rank * chunkElts;
  if (off > total) off = total;
  int64_t elts = total - off;
  if (elts > chunkElts) elts = chunkElts;
  if (elts < 0) elts = 0;
  okiReduceSlice s;
  s.offsetElts = off;
  s.elts = elts;
  return s;
}

// Kernel-side fence wait, used by the AR cmd path instead of a wait-fence WQE: a WQE that
// polls occupies the DMA ring for as long as it waits, so nothing else queued behind it can
// run. Every address waited on lives in this rank's own memory -- the peers' reduce blocks
// remote-write the tail slots -- so this is a local load loop.
static __device__ __inline__ void okiWaitFence(volatile uint64_t* addr, uint64_t value) {
  const uint32_t pauseClocks = 100;  // ~125ns, keeps the poll off the memory pipe
  // Use a signed delta for wraparound-safe fence comparison.
  while ((int64_t)(volatileLoadU64(addr) - value) < 0) {
    asm volatile("ppu.nanosleep.u32 %0;" :: "r"(pauseClocks));
  }
}

// Match the copy writer's per-die payload and optional ICN fence count.
template<int DIE_NUM>
static __device__ __inline__ uint32_t okiChunkCopyWqeCount(size_t size, int die, bool local, bool withIcnFence) {
  if (size == 0) return 0;
  ssize_t bytes = okiLaneBytes<DIE_NUM>(size, die);
  if (bytes <= 0) return 0;
  return calcDmaSegmentCount(bytes) + ((!local && withIcnFence) ? 1u : 0u);
}

// Only driver threads enter this path; CTA barriers belong to the caller.
template<int DIE_NUM>
static __device__ __inline__ void okiReduceDmaCmd(const struct pcclCeOkiReduceArgs& args, int nRanks) {
  ltOkiQP_t* qps = args.qps;
  void* sendBuff = args.sendBuff;
  uint8_t* myRecvBuff = args.myRecvBuff;
  uint8_t* rank0RecvBuff = args.rank0RecvBuff;
  uint8_t* rank0StaBuff = args.rank0StaBuff;
  int myRank = (int)args.myRank;
  uint64_t nElts = args.nElts;
  int eltSize = args.eltSize;
  int loopCnt = (int)args.loopCnt;
  uint64_t* rank0RemoteFenceAddrs = args.rank0RemoteFenceAddrs;
  uint64_t okiBaseStep = args.okiBaseStep;
  uint64_t* tailPtr = args.tailPtr;
  uint64_t* headPtr = args.headPtr;
  uint64_t ceReduceStepBase = (uint64_t)args.reduceStepBase;
  int stride4G = (int)args.stride4G;
  bool broadcast = args.broadcast != 0;

  int tid = threadIdx.x;
  int wid = tid / WARP_SIZE;
  int laneId = tid % WARP_SIZE;

  // One warp per (rank, die); only lane 0 may ring its doorbell.
  if (wid >= nRanks * DIE_NUM || laneId != 0) return;

  uint32_t targetRank = wid / DIE_NUM;
  int die = wid % DIE_NUM;
  // shift myRank to 0 as only queue[0:DIE_NUM-1] support local D2D
  uint32_t rankToQid = (targetRank + nRanks - myRank) % nRanks;
  bool local = (targetRank == myRank);

  int64_t chunkElts = args.chunkElts;
  okiReduceSlice target = okiReduceSliceOf(nElts, chunkElts, targetRank);
  int64_t targetRankElts = target.elts;
  int64_t rankOffset = target.offsetElts * eltSize;

  ltOkiQP_t qp = qps[rankToQid * DIE_NUM + die];
  volatile uint64_t* cqAddr = ltOkiQPGetCQ(qp);

  uint64_t waitValue = ceReduceStepBase + 1 - pcclCeReduceMaxSect;
  uint64_t postValue = ceReduceStepBase + 1;
  int64_t splitElts = pcclCeReduceChunkSize / eltSize;
  bool waitAllTails = loopCnt > pcclCeOkiWaitAllTailsInterval;

  // Advance DMA completion, slot consumption and slot publication epochs together.

  for (int iter=0; iter < loopCnt; iter++) {
    uint32_t sect = iter % 2;
    int64_t perChunkSize = 0;

    if (targetRankElts > splitElts) perChunkSize = pcclCeReduceChunkSize;
    else perChunkSize = targetRankElts * eltSize;

    uint8_t* mySendBuff = (uint8_t*)sendBuff + rankOffset + iter * splitElts * eltSize;

    uint8_t* peerStaBuff = (uint8_t*)add4G((uint8_t*)rank0StaBuff, targetRank * stride4G);
    uint8_t* peerStaDst = peerStaBuff + myRank * pcclSymMidBufferPerRank + sect * splitElts * eltSize;

    size_t size = perChunkSize > 0 ? perChunkSize : 0;

    // Publish the head fence even when this die has no payload.
    uint64_t* headSlot = headPtr + targetRank * PCCL_CE_MAX_DIE_NUM + die;
    uint64_t* remoteFenceAddr = (uint64_t*)add4G((uint8_t*)rank0RemoteFenceAddrs, targetRank * stride4G) + myRank * PCCL_CE_MAX_DIE_NUM + die;

    uint32_t wqeNeeded = okiChunkCopyWqeCount<DIE_NUM>(size, die, local, /*withIcnFence=*/true) + 1;
    uint32_t skip;
    uint32_t wqeBase = ltOkiReserveWqeSlots(qp, wqeNeeded, skip);
    okiWriteCopyWqe<DIE_NUM>(die, qp, size, mySendBuff, peerStaDst, local, /*withIcnFence=*/true, remoteFenceAddr, okiBaseStep);
    ltOkiQPWriteFenceWqe(qp, (uint64_t)headSlot, postValue, die, true);

    // Periodically wait for every peer to limit pipeline drift.
    bool fullSync = waitAllTails && (iter % pcclCeOkiWaitAllTailsInterval) == 0;

    if (fullSync) {
      for (int peer = 0; peer < nRanks; peer++) {
        okiWaitFence((volatile uint64_t*)(tailPtr + peer * PCCL_CE_MAX_DIE_NUM + die), waitValue);
      }
    } else {
      okiWaitFence((volatile uint64_t*)(tailPtr + targetRank * PCCL_CE_MAX_DIE_NUM + die), waitValue);
    }

    ltOkiPostSend(qp, wqeBase, wqeNeeded);
    ltOkiRingDB(qp);

    okiBaseStep += 1;
    waitValue++;
    postValue++;
  }

  {
    // AllReduce broadcasts the reduced slice; both collectives close the CQ at base + loopCnt + 1.
    uint64_t expectedTail = ceReduceStepBase + loopCnt;
    uint32_t preCount = 1;
    int64_t preElts = (broadcast && loopCnt > 1) ? preCount * pcclCeReduceChunkSize / eltSize : 0;
    int64_t myRankElts = okiReduceSliceOf(nElts, chunkElts, myRank).elts;
    int64_t restElts = myRankElts - preElts;
    int64_t preBytes = preElts * eltSize;

    if (broadcast && loopCnt > 1) {
      // Pre-launch one segment before waiting on the final tail, so it overlaps the last
      // reduce chunk.
      okiWaitFence((volatile uint64_t*)(tailPtr + myRank * PCCL_CE_MAX_DIE_NUM + die), expectedTail - 1);

      size_t size = (!local && preElts > 0) ? preElts * eltSize : 0;
      uint32_t wqeNeeded = okiChunkCopyWqeCount<DIE_NUM>(size, die, local, /*withIcnFence=*/false);
      if (wqeNeeded != 0) {
        uint8_t* peerDst = (uint8_t*)add4G((uint8_t*)rank0RecvBuff, targetRank * stride4G);
        uint32_t skip;
        uint32_t wqeBase = ltOkiReserveWqeSlots(qp, wqeNeeded, skip);
        okiWriteCopyWqe<DIE_NUM>(die, qp, size, myRecvBuff, peerDst, local, /*withIcnFence=*/false);
        ltOkiPostSend(qp, wqeBase, wqeNeeded);
        ltOkiRingDB(qp);
      }
    }

    {
      okiWaitFence((volatile uint64_t*)(tailPtr + myRank * PCCL_CE_MAX_DIE_NUM + die), expectedTail);

      uint64_t* remoteFenceAddr = (uint64_t*)add4G((uint8_t*)rank0RemoteFenceAddrs, targetRank * stride4G) + myRank * PCCL_CE_MAX_DIE_NUM + die;
      uint8_t* peerDst = (uint8_t*)add4G((uint8_t*)rank0RecvBuff, targetRank * stride4G) + preBytes;
      uint8_t* src = myRecvBuff + preBytes;

      size_t size = (broadcast && !local && restElts > 0) ? restElts * eltSize : 0;
      uint32_t wqeNeeded = okiChunkCopyWqeCount<DIE_NUM>(size, die, local, /*withIcnFence=*/true) + 1;  // + closing CQ fence
      uint32_t skip;
      uint32_t wqeBase = ltOkiReserveWqeSlots(qp, wqeNeeded, skip);
      okiWriteCopyWqe<DIE_NUM>(die, qp, size, src, peerDst, local, /*withIcnFence=*/true, remoteFenceAddr, okiBaseStep);
      ltOkiQPWriteFenceWqe(qp, (uint64_t)cqAddr, okiBaseStep, die, true);

      ltOkiPostSend(qp, wqeBase, wqeNeeded);
      ltOkiRingDB(qp);
    }

    // The closing CQ fence covers every earlier WQE.
    ltOkiPollCQ((uint64_t*)cqAddr, okiBaseStep);
  }
}

// okiMemcpyBatchAsync: performs batched async DMA copy with dependency management.
//
// Parallel decomposition: one warp per QP group, one lane per die within the group.
// A warp owns its queues exclusively and walks every op assigned to its group, so
// ltOkiReserveWqeSlots is never contended and the trailing CQ fence is genuinely the
// last WQE on the ring. Sharing a ring between warps would break both properties.
//
// Flow per queue:
//   1. One WaitFenceWqe on the dependency fence (queue head, once).
//   2. For each op in this group: DMA copy WQEs (+ that op's remote ICN fence).
//   3. One FenceWqe to the local CQ (queue tail, once). Since WQEs on a ring retire in
//      order, the CQ reaching this value means every op in the group has landed, which is
//      what okiBatchSynchronize polls for.
template<int DIE_NUM>
__global__ void okiMemcpyBatchAsync(
    struct ncclCeCollKernelLaunchArgs args, // use struct to prevent compiler opt array to pointer.
    ltOkiQP_t* qps,
    uint64_t* dependencyFenceAddr,
    uint64_t step,
    int myRank,
    int stride4G,
    int numQpGroups
) {
  assert(gridDim.x == 1);
  int tid = threadIdx.x;
  int group = tid / WARP_SIZE;     // warp index = QP group
  int laneId = tid % WARP_SIZE;    // lane index within warp = die index (0 to DIE_NUM-1)
  bool worker = (group < numQpGroups) && (laneId < DIE_NUM);

  if (worker) {
    int qid = okiQpGroupToQid(group, laneId, DIE_NUM);
    ltOkiQP_t qp = qps[qid];
    volatile uint64_t* cqAddr = ltOkiQPGetCQ(qp);  // completion queue address
    uint64_t finalStep = step + 1;

    uint32_t wqeNeeded = 0;
    int myOpCount = 0;
    for (int opId = 0; opId < args.nOps; opId++) {
      if (okiOpToQpGroup(opId, numQpGroups) != group) continue;
      if (args.sizes[opId] == 0) continue;
      wqeNeeded += 2 * calcDmaSegmentCount(args.sizes[opId] / DIE_NUM);
      myOpCount++;
    }

    if (myOpCount > 0) {
      wqeNeeded += 2;  // head dependency wait fence + tail CQ fence
      uint32_t skip;
      uint32_t wqeBase = ltOkiReserveWqeSlots(qp, wqeNeeded, skip);

      // Step 1: wait on the dependency fence so no DMA starts before all ranks arrived.
      ltOkiQPWriteWaitFenceWqe(qp, (uint64_t)dependencyFenceAddr, finalStep, 0, 0);

      // Step 2: issue every op assigned to this group, in a deterministic order.
      for (int opId = 0; opId < args.nOps; opId++) {
        if (okiOpToQpGroup(opId, numQpGroups) != group) continue;
        size_t size = args.sizes[opId];
        if (size == 0) continue;
        int targetRank = args.targets[opId];
        // Fence slots are per (target rank, source rank, die), independent of QP folding.
        uint64_t* remoteFenceAddr = (uint64_t*)add4G((uint8_t*)args.rank0RemoteFenceAddrs, targetRank * stride4G) + myRank * PCCL_CE_MAX_DIE_NUM + laneId;
        okiWriteCopyWqe<DIE_NUM>(laneId, qp, size, (uint8_t*)args.srcs[opId], (uint8_t*)args.dsts[opId], /*local=*/targetRank == myRank, /*withIcnFence=*/true, remoteFenceAddr, finalStep);
      }

      ltOkiQPWriteFenceWqe(qp, (uint64_t)cqAddr, finalStep, laneId, true);

      ltOkiPostSend(qp, wqeBase, wqeNeeded);
    }
  }

  __syncthreads();
  // Ring the doorbell once per queue in this group (one thread per group).
  if (group < numQpGroups && laneId == 0) {
    bool hasWork = false;
    for (int opId = 0; opId < args.nOps; opId++) {
      if (okiOpToQpGroup(opId, numQpGroups) == group && args.sizes[opId] > 0) { hasWork = true; break; }
    }
    if (hasWork) {
      #pragma unroll
      for (int ii = 0; ii < DIE_NUM; ii++) {
        ltOkiQP_t qp = qps[okiQpGroupToQid(group, ii, DIE_NUM)];
        ltOkiRingDB(qp);  // ring doorbell to kick off the DMA engine
      }
    }
  }
  // Ring all doorbells before any CQ wait; idle threads must participate.
  __syncthreads();
}

// okiBatchSynchronize: polls completion queue to ensure all DMA operations have finished.
// Called after okiMemcpyBatchAsync to guarantee all prior transfers are complete.
//
// Mirrors the submit kernel's decomposition: one warp per QP group, one lane per die.
// Because the submit kernel writes exactly one CQ fence at the tail of each queue, a
// single poll per queue covers every op multiplexed onto it.
template<int DIE_NUM>
__global__ void okiBatchSynchronize(
    struct ncclCeCollKernelLaunchArgs args, // use struct to prevent compiler opt array to pointer.
    ltOkiQP_t* qps,
    int myRank,
    uint64_t pollValue,
    int numQpGroups
) {
  int tid = threadIdx.x;
  int group = tid / WARP_SIZE;     // warp index = QP group
  int laneId = tid % WARP_SIZE;    // lane index = die index
  if (group >= numQpGroups || laneId >= DIE_NUM) return;

  // A queue only carries a fence if the submit kernel found work for this group.
  bool hasWork = false;
  for (int opId = 0; opId < args.nOps; opId++) {
    if (okiOpToQpGroup(opId, numQpGroups) != group) continue;
    if (args.sizes[opId] == 0) continue;
    // The submit kernel splits each op across dies; a die with no share writes no WQE.
    ssize_t size = args.sizes[opId];
    ssize_t splitSize = size / DIE_NUM;
    ssize_t remainSize = size - laneId * splitSize;
    if (laneId != DIE_NUM - 1) remainSize = (remainSize > splitSize) ? splitSize : remainSize;
    if (remainSize > 0) { hasWork = true; break; }
  }

  if (hasWork) {
    ltOkiQP_t qp = qps[okiQpGroupToQid(group, laneId, DIE_NUM)];
    uint64_t* cq = ltOkiQPGetCQ(qp);
    ltOkiPollCQ(cq, pollValue);  // poll until completion value is observed
  }
}

template
__global__ void okiMemcpyBatchAsync<1>(
    struct ncclCeCollKernelLaunchArgs args,
    ltOkiQP_t* qps,
    uint64_t* dependencyFenceAddr,
    uint64_t step,
    int myRank,
    int stride4G,
    int numQpGroups
);
template
__global__ void okiMemcpyBatchAsync<2>(
    struct ncclCeCollKernelLaunchArgs args,
    ltOkiQP_t* qps,
    uint64_t* dependencyFenceAddr,
    uint64_t step,
    int myRank,
    int stride4G,
    int numQpGroups
);

template
__global__ void okiBatchSynchronize<1>(
    struct ncclCeCollKernelLaunchArgs args, // use struct to prevent compiler opt array to pointer.
    ltOkiQP_t* qps,
    int myRank,
    uint64_t pollValue,
    int numQpGroups
);

template
__global__ void okiBatchSynchronize<2>(
    struct ncclCeCollKernelLaunchArgs args, // use struct to prevent compiler opt array to pointer.
    ltOkiQP_t* qps,
    int myRank,
    uint64_t pollValue,
    int numQpGroups
);

#endif
