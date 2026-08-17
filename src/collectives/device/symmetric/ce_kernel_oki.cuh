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

// QP multiplexing map. There are only LT_OKI_MAX_QUEUE_NUM hardware DMA rings, so once
// nRanks exceeds LT_OKI_MAX_QUEUE_NUM/DIE_NUM the ops must share queues instead of owning
// one each. A QP group is DIE_NUM queues (one per die); group 0 is the local D2D pair
// created with isRemote=false, groups 1.. are the remote ICN queues. okiQpGroupToQid
// (oki_utils.cuh) maps a group to its qid; this file owns only the op->group folding,
// which is specific to the collectives' by-value op array. Both the submit and the
// synchronize kernel must agree on it.
//
// ICN DMA is not bound to a peer -- ltOkiQPWriteIcnDmaCopyWqe takes no target rank, the
// destination is encoded entirely in the LSA flat address -- so any remote queue can carry
// traffic to any peer.
static __device__ __inline__ int okiOpToQpGroup(int opId, int numQpGroups) {
  // op 0 is always this rank's own copy (or the zero-size in-place placeholder).
  if (opId == 0) return 0;
  int numRemoteGroups = numQpGroups - 1;
  return 1 + (opId - 1) % numRemoteGroups;
}

template<int DIE_NUM>
__global__ void okiMemcpyBatchAsyncForAR(
  ltOkiQP_t* qps,
  uint64_t* dependencyFenceAddr,
  void* sendBuff,
  uint8_t* myRecvBuff,
  uint8_t* rank0RecvBuff,
  uint8_t* rank0StaBuff,
  int myRank,
  int nRanks,
  uint64_t nElts,
  int eltSize,
  int loopCnt,
  uint64_t* rank0RemoteFenceAddrs,
  uint64_t okiBaseStep,
  uint64_t* tailPtr,
  uint64_t* headPtr,
  uint64_t ceReduceStepBase,
  int stride4G
) {
  int tid = threadIdx.x;
  int wid = tid / WARP_SIZE;
  int laneId = tid % WARP_SIZE;
  uint32_t targetRank = 0xFFFF;
  uint32_t rankToQid = 0xFFFF;
  uint32_t qid = 0xFFFF;
  if (wid < nRanks && laneId < DIE_NUM) {
    targetRank = wid;
    rankToQid = (targetRank + nRanks -  myRank) % nRanks; // shift myRank to 0 as only queue[0:DIE_NUM-1] support local D2D
    qid = rankToQid * DIE_NUM + laneId;
  }

  int64_t avgElts = DIVUP(nElts, nRanks);
  avgElts = ROUNDUP(avgElts, 512/eltSize);

  if (targetRank < nRanks) {// only valid worker thread can enter.
    int64_t targetRankElts;
    targetRankElts = nElts > avgElts ? avgElts : nElts;   // the extra elts handle by rank0
    int64_t rankOffset = 0;
    for (int i = 0; i < targetRank; i++) {
      int64_t remain = ((int64_t)nElts - ((int64_t)i+1) * avgElts) ;
      rankOffset = rankOffset + targetRankElts * eltSize;  // pre rank offset  + pre rank elts
      targetRankElts = (remain >= 0) ? avgElts : (nElts - (i+1) * avgElts);
    }

    ltOkiQP_t qp = qps[qid];
    volatile uint64_t* localFenceAddr = ltOkiQPGetCQ(qp);

    uint64_t waitValue = ceReduceStepBase + 1 - pcclCeReduceMaxSect;
    uint64_t postValue = ceReduceStepBase + 1;
    uint64_t* peerTailPtr = tailPtr + targetRank * PCCL_CE_MAX_DIE_NUM + laneId; // each die use different tailPtr
    headPtr = headPtr + targetRank * PCCL_CE_MAX_DIE_NUM + laneId;

    int64_t splitElts = pcclCeReduceChunkSize / eltSize;

    // WQE (Work Queue Entry) slot allocation for okiMemcpyBatchAsyncForAR.
    //
    // Sized for the ICN path, which is the worst case: the local D2D warp
    // (targetRank == myRank) skips both allgather copies and writes fewer slots.
    //
    // Breakdown of wqeNeeded = 1 + 6*loopCnt + (loopCnt > 1 ? 3 : 0) + 2 + segCount + 1:
    //
    //   1             : WaitFenceWqe on dependencyFenceAddr — the host/device handoff
    //                   trigger. Kernel waits here until all ranks have arrived.
    //
    //   6 * loopCnt   : per reduce-loop iteration, in issue order:
    //                     1 x WaitFenceWqe on peer's tail  (slot free to overwrite)
    //                     1 x WaitFenceWqe on local CQ     (previous step visible)
    //                     1 x Dma/IcnDmaCopyWqe            (this chunk to the peer)
    //                     1 x IcnFenceWqe                  (chunk landed on the peer)
    //                     1 x FenceWqe to local CQ         (publish to reduce kernel)
    //                     1 x FenceWqe to head pointer     (peer may now consume)
    //                   One segment per iteration: perChunkSize is capped at
    //                   pcclCeReduceChunkSize, far below PCCL_LT_OKI_MAX_DMA_BYTE_CNT.
    //
    //   3             : pre-launched allgather DMA overlapping the last reduce
    //   (loopCnt > 1)   iteration — 2 x WaitFenceWqe plus 1 x IcnDmaCopyWqe. Single
    //                   segment because preCount is 1.
    //
    //   1             : WaitFenceWqe on my own tail — my whole slice is reduced.
    //
    //   segCount + 1  : final allgather copy. writeCopyWqe emits one IcnFenceWqe
    //                   after the last segment, not one per segment, so the fence
    //                   costs a single slot regardless of segCount.
    //
    //   1             : closing FenceWqe to local CQ, polled by ceLocalReduce.
    //
    //   Long pipelines (loopCnt > pcclCeOkiWaitAllTailsInterval) insert one full sync every
    //   pcclCeOkiWaitAllTailsInterval iterations, each costing (nRanks-1) extra WaitFenceWqe:
    //   that chunk waits on every peer's tail, which re-aligns all ranks' chunk pipelines.
    //
    // The QP depth (LT_OKI_QP_DEPTH) must be >= this wqeNeeded, otherwise the
    // ltOkiReserveWqeSlots call will fail or wrap, causing DMA to be lost.
    uint32_t segCount = calcDmaSegmentCount(avgElts * eltSize / DIE_NUM);
    bool waitAllTails = loopCnt > pcclCeOkiWaitAllTailsInterval;
    uint32_t fullSyncIters = waitAllTails ? (uint32_t)DIVUP(loopCnt, pcclCeOkiWaitAllTailsInterval) : 0u;
    uint32_t wqeNeeded = 1 + 6*loopCnt + (uint32_t)(nRanks - 1) * fullSyncIters + (loopCnt > 1 ? 3 : 0) + 2 + segCount + 1;
    uint32_t skip;
    uint32_t wqeBase = ltOkiReserveWqeSlots(qp, wqeNeeded, skip);

    ltOkiQPWriteWaitFenceWqe(qp, (uint64_t)dependencyFenceAddr, okiBaseStep, 4, 0);

    // Fence step synchronization — three counters that must stay in lockstep:
    //
    // okiBaseStep: the OKI fabric DMA-completion epoch, published to remote ranks
    //   via IcnFenceWqe to signal "my DMA for step N is complete and visible in
    //   your station buffer." Also waited on locally by the reduce kernel via
    //   localCQ fence at (okiBaseStep - 1), ensuring the reduce kernel does not
    //   read data until the previous iteration's DMA is fully visible.
    //   All ranks agree on this value — it is the "global fabric epoch" for
    //   cross-rank DMA ordering, distinct from buffer-slot synchronization.
    //
    // waitValue: the peer tail-pointer value this rank waits on before sending
    //   each chunk. Initialized to (ceReduceStepBase + 1 - pcclCeReduceMaxSect),
    //   then incremented each iteration. Represents "peer has consumed up to
    //   this many reduce chunks, so a slot is free for me to send."
    //
    // postValue: the head-pointer value this rank publishes after each chunk,
    //   signaling to peers that their data is ready. Incremented each iteration.
    //   Represents "I have produced this many chunks, peer can now read."
    //
    // All three advance by exactly +1 per loop iteration. They differ only in
    // their synchronization object and starting offset:
    //   okiBaseStep → fabric ICN fence (OKI global ordering)
    //   waitValue   → peer tail pointer (buffer slot availability)
    //   postValue   → peer head pointer (data availability)
    // A mismatch between them would cause a rank to wait on a stale or
    // prematurely updated fence, resulting in deadlock or data read-before-written.

    for (int iter=0; iter < loopCnt; iter++) {
      uint32_t sect = iter % 2;
      int64_t perChunkSize = 0;

      if (targetRankElts > splitElts) perChunkSize = pcclCeReduceChunkSize;
      else perChunkSize = targetRankElts * eltSize;

      uint8_t* mySendBuff = (uint8_t*)sendBuff + rankOffset + iter * splitElts * eltSize;

      uint8_t* peerStaBuff = (uint8_t*)add4G((uint8_t*)rank0StaBuff, targetRank * stride4G);
      uint8_t* peerStaDst = peerStaBuff + myRank * pcclSymMidBufferPerRank + sect * splitElts * eltSize;

      size_t size = perChunkSize > 0 ? perChunkSize : 0;

      if (waitAllTails && (iter % pcclCeOkiWaitAllTailsInterval) == 0) {
        // Periodic full sync: gate this chunk on every peer's consumption, not just the
        // target's, so the ranks re-align instead of drifting apart. Tail slots are local.
        for (int peer = 0; peer < nRanks; peer++) {
          ltOkiQPWriteWaitFenceWqe(qp, (uint64_t)(tailPtr + peer * PCCL_CE_MAX_DIE_NUM + laneId), waitValue, 4, 0);
        }
      } else {
        ltOkiQPWriteWaitFenceWqe(qp, (uint64_t)peerTailPtr, waitValue, 4, 0);
      }
      ltOkiQPWriteWaitFenceWqe(qp, (uint64_t)localFenceAddr, okiBaseStep - 1, 4, 0);
      uint64_t* remoteFenceAddr = (uint64_t*)add4G((uint8_t*)rank0RemoteFenceAddrs, targetRank * stride4G) + myRank * PCCL_CE_MAX_DIE_NUM + laneId;
      okiWriteCopyWqe<DIE_NUM>(laneId, qp, size, mySendBuff, peerStaDst, /*local=*/targetRank == myRank, /*withIcnFence=*/true, remoteFenceAddr, okiBaseStep);

      ltOkiQPWriteFenceWqe(qp, (uint64_t)localFenceAddr, okiBaseStep, laneId, true);
      ltOkiQPWriteFenceWqe(qp, (uint64_t)headPtr, postValue, laneId, true);
      okiBaseStep += 1;

      // TODO: we can do a fast startup here.
      waitValue++;
      postValue++;
    }

    {
      // wait my self reduce done. then do the allgather
      // for multi segments, pre launch a segment dma to overlap with the last reduce segment.
      uint64_t* myTailPtr = tailPtr + myRank * PCCL_CE_MAX_DIE_NUM + laneId;
      uint64_t expectedTail = ceReduceStepBase + loopCnt;
      uint32_t preCount = 1;

      if (loopCnt > 1) {
        ltOkiQPWriteWaitFenceWqe(qp, (uint64_t)localFenceAddr, okiBaseStep - 1, 4, 0);
        ltOkiQPWriteWaitFenceWqe(qp, (uint64_t)myTailPtr, expectedTail - 1, 4, 0);

        int64_t myRankElts;
        myRankElts = preCount * pcclCeReduceChunkSize / eltSize;
        uint8_t* tmpRecvBuff = nullptr;
        uint8_t* tmpPeerDst = nullptr;

        if (targetRank != myRank && myRankElts > 0) {
          size_t size = myRankElts * eltSize;
          uint8_t* peerDst = (uint8_t*)add4G((uint8_t*)rank0RecvBuff, targetRank * stride4G);

          okiWriteCopyWqe<DIE_NUM>(laneId, qp, size, myRecvBuff, peerDst, /*local=*/targetRank == myRank, /*withIcnFence=*/false);
        }
      }

      ltOkiQPWriteWaitFenceWqe(qp, (uint64_t)myTailPtr, expectedTail, 4, 0);

      int64_t myRankElts;
      myRankElts = nElts > avgElts ? avgElts : nElts;   // the extra elts handle by rank0
      for (int i = 0; i < myRank; i++) {
        int64_t remain = ((int64_t)nElts - ((int64_t)i+1) * avgElts) ;
        myRankElts = (remain >= 0) ? avgElts : (nElts - (i+1) * avgElts);
      }

      if (loopCnt > 1) myRankElts -= preCount * pcclCeReduceChunkSize / eltSize;

      if (targetRank != myRank && myRankElts > 0) {
        ssize_t size = myRankElts * eltSize;
        uint64_t* remoteFenceAddr = (uint64_t*)add4G((uint8_t*)rank0RemoteFenceAddrs, targetRank * stride4G) + myRank * PCCL_CE_MAX_DIE_NUM + laneId;
        uint8_t* peerDst = (uint8_t*)add4G((uint8_t*)rank0RecvBuff, targetRank * stride4G);

        if (loopCnt > 1) {
          myRecvBuff = (uint8_t*)((char*)myRecvBuff + preCount * pcclCeReduceChunkSize);
          peerDst = peerDst + preCount * pcclCeReduceChunkSize;
        }

        okiWriteCopyWqe<DIE_NUM>(laneId, qp, size, myRecvBuff, peerDst, /*local=*/targetRank == myRank, /*withIcnFence=*/true, remoteFenceAddr, okiBaseStep);
      }
      ltOkiQPWriteFenceWqe(qp, (uint64_t)localFenceAddr, okiBaseStep, laneId, true);
      okiBaseStep += 1;
    }

    ltOkiPostSend(qp, wqeBase, wqeNeeded);
  }

  __syncthreads();
  // one wake up thread per warp.
  if (targetRank < nRanks && laneId == 0) {    // only worker thread can enter.
    #pragma unroll
    for (int ii=0; ii<DIE_NUM; ii++) {
      uint32_t qid = 0xFFFF;
      rankToQid = (targetRank + nRanks -  myRank) % nRanks; // shift myRank to 0 as only queue[0] support local D2D
      qid = rankToQid * DIE_NUM + ii;

      ltOkiQP_t qp = qps[qid];
      ltOkiRingDB(qp);
    }
  }
  __syncthreads();
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

    // Total WQE budget for this queue: the head wait fence, then every op's DMA
    // (segmented by PCCL_LT_OKI_MAX_DMA_BYTE_CNT) plus its ICN fence, then the tail CQ
    // fence. Under-counting here silently drops DMA, so it must cover all ops.
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
        // Remote fence slot is per (target rank, source rank, die), so ops sharing this
        // queue never collide on it.
        uint64_t* remoteFenceAddr = (uint64_t*)add4G((uint8_t*)args.rank0RemoteFenceAddrs, targetRank * stride4G) + myRank * PCCL_CE_MAX_DIE_NUM + laneId;
        okiWriteCopyWqe<DIE_NUM>(laneId, qp, size, (uint8_t*)args.srcs[opId], (uint8_t*)args.dsts[opId], /*local=*/targetRank == myRank, /*withIcnFence=*/true, remoteFenceAddr, finalStep);
      }

      // Step 3: single completion fence covering every op on this queue.
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
__global__ void okiMemcpyBatchAsyncForAR<1>(
  ltOkiQP_t* qps,
  uint64_t* dependencyFenceAddr,
  void* sendBuff,
  uint8_t* myRecvBuff,
  uint8_t* rank0RecvBuff,
  uint8_t* rank0StaBuff,
  int myRank,
  int nRanks,
  uint64_t nElts,
  int eltSize,
  int loopCnt,
  uint64_t* rank0RemoteFenceAddrs,
  uint64_t okiBaseStep,
  uint64_t* tailPtr,
  uint64_t* headPtr,
  uint64_t ceReduceStepBase,
  int stride4G
);

template
__global__ void okiMemcpyBatchAsyncForAR<2>(
  ltOkiQP_t* qps,
  uint64_t* dependencyFenceAddr,
  void* sendBuff,
  uint8_t* myRecvBuff,
  uint8_t* rank0RecvBuff,
  uint8_t* rank0StaBuff,
  int myRank,
  int nRanks,
  uint64_t nElts,
  int eltSize,
  int loopCnt,
  uint64_t* rank0RemoteFenceAddrs,
  uint64_t okiBaseStep,
  uint64_t* tailPtr,
  uint64_t* headPtr,
  uint64_t ceReduceStepBase,
  int stride4G
);

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
