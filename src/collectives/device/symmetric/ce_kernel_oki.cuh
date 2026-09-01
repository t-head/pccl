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

// The slice of an nElts buffer owned by `rank` when it is cut into chunkElts-sized pieces.
// Must agree with the host partition in pcclCeOkiReduceSetup, which clamps the last rank to the
// remainder. The open-coded loops this replaces derived rank i's size from rank i+1's bound, so a
// trailing partial rank was handed a full slice -- the DMA then read past the end of sendBuff and
// the broadcast wrote past the end of recvBuff.
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
  // Signed difference, not a direct compare: the first chunk of a fresh communicator waits on
  // (ceReduceStepBase + 1 - pcclCeReduceMaxSect), which is negative and wraps as a uint64_t.
  while ((int64_t)(volatileLoadU64(addr) - value) < 0) {
    asm volatile("ppu.nanosleep.u32 %0;" :: "r"(pauseClocks));
  }
}

// WQEs okiWriteCopyWqe will write for one die's share of `size`: one per DMA segment, plus
// the ICN fence that closes a remote transfer when the caller asks for one. Zero when this
// die has no bytes to move.
//
// `withIcnFence` must match what the caller passes to okiWriteCopyWqe. ltOkiPostSend only
// accepts the slot count ltOkiReserveWqeSlots handed back, so an over-count publishes a slot
// nobody wrote and the engine executes whatever stale bytes it holds.
template<int DIE_NUM>
static __device__ __inline__ uint32_t okiChunkCopyWqeCount(size_t size, int die, bool local, bool withIcnFence) {
  if (size == 0) return 0;
  ssize_t bytes = okiLaneBytes<DIE_NUM>(size, die);
  if (bytes <= 0) return 0;
  return calcDmaSegmentCount(bytes) + ((!local && withIcnFence) ? 1u : 0u);
}

// The reduce-family cmd path: post the reduce-scatter chunks, then -- for AllReduce -- the
// allgather, then wait for this queue to drain. Runs in the last block of the fused kernel, which
// barriers with the peers immediately before and after the call -- the barrier is what used to be
// the dependency fence handoff from gpuBarrierWithTrigger to a separate cmd kernel.
//
// args.broadcast == 0 is ReduceScatter: each rank's reduced slice already lives in its own
// recvBuff, so the allgather phase is skipped. The closing CQ fence is posted either way.
//
// Contains no __syncthreads(): non-driver threads return right away and wait at the caller's
// post-barrier instead.
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

  // One driver thread per (target rank, die), so each warp owns exactly one queue.
  //
  // All flow control is a kernel spin now (okiWaitFence), so a chunk's WQEs can only be
  // written once that chunk is allowed to run: the kernel walks the chunks one at a time,
  // reserving, writing, posting and ringing the doorbell for each. Driving both dies from one
  // thread serialized those spins -- die 1 could not reserve until die 0's tail fence passed --
  // so the dies get a warp each and advance independently.
  //
  // The driver is still lane 0: ltOkiRingDB hangs the engine when called from any other lane,
  // which is why the dies are spread over warps rather than over lanes of one warp.
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

  // Fence step synchronization — three counters that must stay in lockstep:
  //
  // okiBaseStep: the OKI fabric DMA-completion epoch, published to remote ranks
  //   via IcnFenceWqe to signal "my DMA for step N is complete and visible in
  //   your station buffer." Also waited on locally before each chunk via the
  //   queue's CQ at (okiBaseStep - 1), ensuring this queue's previous step --
  //   the previous collective's closing fence on the first iteration -- has
  //   completed and is visible before the next chunk is issued.
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

    // Nothing touches the queue until the fences pass: reserve, build, post and ring all
    // happen after the spin. The head fence is written even when this die has no bytes,
    // otherwise the peer would wait forever.
    uint64_t* headSlot = headPtr + targetRank * PCCL_CE_MAX_DIE_NUM + die;
    uint64_t* remoteFenceAddr = (uint64_t*)add4G((uint8_t*)rank0RemoteFenceAddrs, targetRank * stride4G) + myRank * PCCL_CE_MAX_DIE_NUM + die;

    // okiChunkCopyWqeCount must predict okiWriteCopyWqe exactly: ltOkiReserveWqeSlots
    // adjusts wqeNeeded in place and ltOkiPostSend only accepts that adjusted count, so
    // a reservation that does not match what is written publishes stale ring slots.
    uint32_t wqeNeeded = okiChunkCopyWqeCount<DIE_NUM>(size, die, local, /*withIcnFence=*/true) + 1;
    uint32_t skip;
    uint32_t wqeBase = ltOkiReserveWqeSlots(qp, wqeNeeded, skip);
    okiWriteCopyWqe<DIE_NUM>(die, qp, size, mySendBuff, peerStaDst, local, /*withIcnFence=*/true, remoteFenceAddr, okiBaseStep);
    ltOkiQPWriteFenceWqe(qp, (uint64_t)headSlot, postValue, die, true);

    // Periodic full sync: gate this chunk on every peer's consumption, not just the
    // target's, so the ranks re-align instead of drifting apart. Tail slots are local.
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
    // AllReduce: wait for my own reduce to finish, then allgather my slice to every peer.
    // ReduceScatter: the reduced slice already sits in this rank's recvBuff, so only the closing
    // fence is posted -- the host advances ceOkiStep by loopCnt + 1 for both collectives, so the
    // extra step has to be consumed either way, and it is also what drains this queue.
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

    // Wait for this queue to retire everything posted above. WQEs retire in order, so the
    // closing fence reaching the CQ means every chunk of this collective has landed.
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
