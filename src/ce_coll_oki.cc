//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include "pccl_internal.h"
#include "param.h"
#include "comm.h"
#include "register.h"
#include "cudawrap.h"
#include "ce_coll.h"
#include "dev_runtime.h"
#include "alloc.h"
#include "devcomm.h"
#include "symmetric.h"
#include "bitops.h"
#include "utils.h"
#include "lt_oki.h"
#include "ce_coll_oki.h"
extern int ppuDiesNum;
extern void* pcclCeOkiArFuncs[];

// pcclCeLaunchBatchOpsMultiStreamWithOki — OKI batch DMA two-kernel pipeline
//
// Why two kernels on different streams?
//
// Kernel 1 (okiMemcpyBatchAsync): Launched on comm->ceStreams[0].
//   - All threads cooperatively fill WQE (Work Queue Entry) slots for their
//     assigned DMA ops. Threads in the same warp share the same QP.
//   - Returns immediately after enqueuing WQEs to the hardware queue.
//     DMA is queued, NOT complete — the GPU has submitted work to the DMA engine.
//
// Kernel 2 (okiBatchSynchronize): Launched on the caller's 'stream'.
//   - Polls each QP's Completion Queue (CQ) until the posted fence value is
//     observed (ltOkiPollCQ). Blocks GPU threads until DMA is actually done.
//
// The dependencyFenceAddr cross-kernel handoff:
//   - A prior GPU kernel (gpuBarrierWithTrigger, launched by ncclMemOpSyncWithTrigger
//     on the caller's stream) writes dependencyFenceAddr <- (baseStep+1) on the device
//     via __ppu_global_store_volatile_b64, after all ranks have arrived at the barrier.
//   - Kernel 1 waits on this fence before issuing DMA WQEs (ltOkiQPWriteWaitFenceWqe).
//     This ensures no rank starts DMA until all ranks have reached the trigger.
//   - Kernel 2 polls the QP's CQ, which was written by Kernel 1's fence write,
//     confirming DMA completion per-QP.
//
// Stream ordering: hggcEventRecord on ceStreams[0] + hggcStreamWaitEvent on the
//   caller's stream provides stream-level ordering on the host side, but does NOT
//   replace the fence handoff above — the fence is required for GPU-side queue
//   ordering. Without it, a later collective on a different stream could start
//   before DMA from this collective is visible to remote ranks.
static ncclResult_t pcclCeLaunchBatchOpsMultiStreamWithOki(ncclComm_t comm, const struct ncclCeBatchOpsParams* params, ncclCeCollArgs* ceCollArgs, uint64_t baseStep, uint64_t syncStep, hggcStream_t stream) {
  // Check if there are any operations to perform
  if (params->numOps == 0) {
    return ncclSuccess;
  }
  struct ncclCeCollKernelLaunchArgs args;
  for (int i = 0; i < params->numOps; i++) {
    args.dsts[i] = (void*)((uintptr_t)params->dsts[i]);
    args.srcs[i] = (void*)((uintptr_t)params->srcs[i]);
    args.sizes[i] = ((uintptr_t)params->sizes[i]);
    args.targets[i] = ((uintptr_t)params->targets[i]);
  }
  args.nOps = params->numOps;
  char* rankRemoteFenceAddrs = (char*)ncclSymDevBase_getKuiRemoteFenceAddr((struct ncclSymDevBase*)comm->ceColl.ceSyncBase, comm->nRanks);
  uint64_t offset = (uint8_t*)rankRemoteFenceAddrs - (uint8_t*)comm->ceColl.ceSyncBase;
  NCCLCHECK(ncclDevrGetLsaRankPtr(comm, comm->ceColl.ceSyncWin, offset, 0, (void**)&args.rank0RemoteFenceAddrs));

  uint64_t* dependencyFenceAddr = comm->okiDependencyAddr;

  uint32_t stride4G = comm->devrState.bigSize >> 32;
  ltOkiQP_t* qps = comm->okiQps;
  // One warp per QP group (a group is one queue per die), one lane per die. Sizing the block
  // by group count keeps it independent of nRanks; the count is validated at init.
  int numQpGroups = comm->okiQpCount / ppuDiesNum;
  void* pargs[] = {&args, &qps, &dependencyFenceAddr, &baseStep, &comm->rank, &stride4G, &numQpGroups};
  struct dispatchStrategyExAD launchConfig = {1, 1, 1, -2, /*cooperative=*/0};
  dim3 grid = {1, 1, 1};
  dim3 block = {(uint32_t)(WARP_SIZE * numQpGroups), 1, 1};

  void* func;
  if (ppuDiesNum == 1) func = (void**) okiMemcpyBatchAsync<1>;
  else if (ppuDiesNum == 2) func = (void**) okiMemcpyBatchAsync<2>;
  else return ncclInternalError;
  // okiMemcpyBatchAsync kernel will wait dependencyFenceAddr to meet request then launch DMA.
  LAUNCH_KERNEL(func, grid, block, pargs, (ssize_t)0, comm->ceStreams[0], 1, launchConfig);

  // cannot use wait Event to implement the dependency as we use queue directly, there is no stream concept.
  if (ppuDiesNum == 1) func = (void*)okiBatchSynchronize<1>;
  else if (ppuDiesNum == 2) func = (void*)okiBatchSynchronize<2>;
  else return ncclInternalError;
  void* pargs1[] = {&args, &qps, &comm->rank, &syncStep, &numQpGroups};
  // okiBatchSynchronize kernel will wait DMA done launched by okiMemcpyBatchAsync
  LAUNCH_KERNEL(func, grid, block, pargs1, (ssize_t)0, stream, 1, launchConfig);
  return ncclSuccess;
}

struct pcclCeOkiAllReduceState {
  ssize_t rankElts[8];
  ssize_t rankOffset[8];
  uint8_t* myRecvBuff;
  uint8_t* myStaBuff;
  uint64_t* tailBasePtr;
  uint64_t* headBasePtr;
  ssize_t currentRankElts;
  uint32_t rank0LoopCnt;
  ssize_t splitElts;
};

static ncclResult_t pcclCeOkiAllReduceSetup(
  ncclComm_t comm,
  struct ncclCeCollArgs* args,
  struct pcclCeOkiAllReduceState* state
);

static ncclResult_t pcclCeOkiAllReduceLaunch(
  ncclComm_t comm,
  struct ncclCeCollArgs* args,
  struct pcclCeOkiAllReduceState* state,
  hggcStream_t stream
);

// One cooperative kernel does the whole collective on the caller's stream: blocks
// [0, pcclCeReduceMaxBlocks) reduce the chunks as the peers' DMA delivers them, and the last
// block barriers with the peers, drives the OKI queues for both the reduce-scatter and the
// allgather, waits for its own completion queues, and barriers again.
//
// The barriers used to be two kernels (gpuBarrierWithTrigger / gpuBarrier) with a device fence
// carrying the handoff to the cmd kernel on comm->ceStreams[0]. Fused, the handoff is program
// order, and because the kernel does not retire until the cmd block has drained its queues,
// stream ordering alone provides the back-pressure that the host-side sliding window used to.
ncclResult_t pcclCeOkiAllReduce(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream) {
  struct pcclCeOkiAllReduceState state = {};

  NCCLCHECK(pcclCeOkiAllReduceSetup(comm, args, &state));
  NCCLCHECK(pcclCeOkiAllReduceLaunch(comm, args, &state, stream));

  comm->ceOkiStep += (state.rank0LoopCnt + 1);
  comm->ceReduceStep += state.rank0LoopCnt;
  return ncclSuccess;
}

static ncclResult_t pcclCeOkiAllReduceSetup(
  ncclComm_t comm,
  struct ncclCeCollArgs* args,
  struct pcclCeOkiAllReduceState* state
) {
  if (comm->nRanks > 8) {
    ERROR("pcclCeOkiAllReduce: OKI path only supports nRanks <= 8, got nRanks=%d", comm->nRanks);
    return ncclInternalError;
  }

  ssize_t avgElts = DIVUP(args->nElts, comm->nRanks);
  constexpr int bulkInstSize = 512;
  avgElts = ROUNDUP(avgElts, bulkInstSize/args->eltSize);

  state->rankElts[0] = args->nElts > avgElts ? avgElts : args->nElts;
  state->rankOffset[0] = 0;
  for (int i = 1; i < comm->nRanks; i++) {
    int64_t remain = ((int64_t)args->nElts - ((int64_t)i+1) * avgElts);
    state->rankElts[i] = (remain >= 0) ? avgElts : std::max<ssize_t>(0, args->nElts - i * avgElts);
    state->rankOffset[i] = state->rankOffset[i-1] + state->rankElts[i-1] * args->eltSize;
  }

  state->myRecvBuff = (uint8_t*)args->recvBuff + state->rankOffset[comm->rank];
  state->myStaBuff = (uint8_t*)ncclSymDevBase_getStaBuf((struct ncclSymDevBase*)comm->ceColl.ceSyncBase, comm->nRanks);
  state->currentRankElts = state->rankElts[comm->rank];
  state->splitElts = pcclCeReduceChunkSize / args->eltSize;
  state->rank0LoopCnt = DIVUP(state->rankElts[0], state->splitElts);

  state->tailBasePtr = (uint64_t*)ncclSymDevBase_getCeTailBase((struct ncclSymDevBase*)comm->ceColl.ceSyncBase, comm->nRanks);
  state->headBasePtr = (uint64_t*)ncclSymDevBase_getCeHeadBase((struct ncclSymDevBase*)comm->ceColl.ceSyncBase, comm->nRanks);

  return ncclSuccess;
}

static ncclResult_t pcclCeOkiAllReduceLaunch(
  ncclComm_t comm,
  struct ncclCeCollArgs* args,
  struct pcclCeOkiAllReduceState* state,
  hggcStream_t stream
) {
  int funcIndex = localReduceFuncIndex(args->datatype, comm->nRanks);
  if (funcIndex == -1) {
    ERROR("not implemented");
    return ncclInternalError;
  }

  struct pcclCeOkiArArgs kargs = {};
  kargs.comm = args->comm;
  kargs.qps = comm->okiQps;
  kargs.sendBuff = args->sendBuff;
  kargs.myRecvBuff = state->myRecvBuff;
  kargs.myStaBuff = state->myStaBuff;
  kargs.tailPtr = state->tailBasePtr;
  kargs.headPtr = state->headBasePtr;
  kargs.gridSyncPtr = ncclSymDevBase_getGridSyncBase((struct ncclSymDevBase*)comm->ceColl.ceSyncBase, comm->nRanks);
  kargs.myElts = state->currentRankElts;
  kargs.nElts = args->nElts;
  kargs.okiBaseStep = comm->ceOkiStep + 1;
  kargs.reduceStepBase = comm->ceReduceStep;
  kargs.eltSize = args->eltSize;
  kargs.loopCnt = state->rank0LoopCnt;
  kargs.nDies = ppuDiesNum;
  kargs.stride4G = comm->devrState.bigSize >> 32;
  kargs.myRank = comm->rank;

  // The cmd block reaches a peer by adding targetRank * stride4G to rank 0's lsa-flat address;
  // the reduce blocks reach their own slot on a peer the same way.
  char* rankRemoteFenceAddrs = (char*)ncclSymDevBase_getKuiRemoteFenceAddr((struct ncclSymDevBase*)comm->ceColl.ceSyncBase, comm->nRanks);
  uint64_t offset = (uint8_t*)rankRemoteFenceAddrs - (uint8_t*)comm->ceColl.ceSyncBase;
  NCCLCHECK(ncclDevrGetLsaRankPtr(comm, comm->ceColl.ceSyncWin, offset, 0, (void**)&kargs.rank0RemoteFenceAddrs));

  offset = state->myStaBuff - (uint8_t*)comm->ceColl.ceSyncBase;
  NCCLCHECK(ncclDevrGetLsaRankPtr(comm, comm->ceColl.ceSyncWin, offset, 0, (void**)&kargs.rank0StaBuff));

  offset = (uint8_t*)args->recvBuff - (uint8_t*)args->recvWin->userPtr;
  NCCLCHECK(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, 0, (void**)&kargs.rank0RecvBuff));
  kargs.rank0RecvBuff += state->myRecvBuff - (uint8_t*)args->recvBuff;

  offset = (uint8_t*)(state->tailBasePtr + comm->rank * PCCL_CE_MAX_DIE_NUM) - (uint8_t*)comm->ceColl.ceSyncBase;
  NCCLCHECK(ncclDevrGetLsaRankPtr(comm, comm->ceColl.ceSyncWin, offset, 0, (void**)&kargs.tailOnRank0));

  offset = (uint8_t*)(state->headBasePtr + comm->rank * PCCL_CE_MAX_DIE_NUM) - (uint8_t*)comm->ceColl.ceSyncBase;
  NCCLCHECK(ncclDevrGetLsaRankPtr(comm, comm->ceColl.ceSyncWin, offset, 0, (void**)&kargs.headOnRank0));

  // Cooperative: the reduce blocks grid_sync among themselves and the cmd block spins on the
  // fences they publish, so every block must be resident.
  struct dispatchStrategyExAD launchConfig = {/*blocksPerMultiprocessor=*/1, /*gridStepX=*/1, /*gridStepY=*/1, /*subDispatch=*/-2, /*cooperative=*/1};
  dim3 grid = {pcclCeReduceMaxBlocks + 1, 1, 1};
  dim3 block = {pcclCeReduceMaxThreads, 1, 1};
  void* pargs[] = {&kargs};

  LAUNCH_KERNEL(pcclCeOkiArFuncs[funcIndex], grid, block, pargs, (ssize_t)0, stream, 1, launchConfig);

  return ncclSuccess;
}

// only support one stage algo for now.
ncclResult_t pcclCeOkiAllGather(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream) {
  ncclResult_t res = ncclSuccess;

  // Calculate the size of each rank's data chunk
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff + comm->rank * chunkBytes;
  void* peerRecvBuff;
  size_t offset;
  struct ncclCeAllGatherStrategy* strategy;

  // oki only support one stage all gather for now.
  strategy = comm->ceCollConfigs.agOneStageStrategy;
  int *icnConnectedPeers = strategy->icnConnectedPeers;
  int peerNumStage1 = strategy->peerNumStage1;

  struct ncclCeBatchOpsParams batchOpsParams = {};
  uint64_t* dependencyAddr = nullptr;
  uint64_t baseStep = comm->ceOkiStep;
  uint64_t syncStep = baseStep + 1;
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, comm->nRanks), res, fail);

  // 3-phase synchronization protocol for OKI batch operations:
  //
  // Phase 1 (device barrier): ncclMemOpSyncWithTrigger launches gpuBarrierWithTrigger,
  //   which performs a barrier arrive/wait across all ranks on the caller's stream, then
  //   stores dependencyAddr <- (baseStep+1) via __ppu_global_store_volatile_b64.
  //   All ranks must arrive at the barrier before any rank proceeds — this is the
  //   hardware-enforced ordering point that ensures no rank starts DMA while another
  //   is still filling its WQEs.
  //
  // Phase 2 (device, okiMemcpyBatchAsync): each thread-block waits on
  //   dependencyAddr == (baseStep+1) before issuing DMA WQEs. This provides
  //   GPU-side ordering: all blocks see the trigger before any DMA fires.
  //
  // Phase 3 (device, okiBatchSynchronize): each thread polls its QP's CQ until
  //   the posted value == (syncStep == baseStep+1), confirming DMA completion.
  //
  // NOTE: on error return from any phase, other ranks may be left waiting at
  //   Phase 1 (barrier) or Phase 2 (dependency wait) indefinitely — the collective
  //   has no timeout or abort mechanism. This is a known correctness limitation.

  // Ensure all ranks are ready before starting transfers
  dependencyAddr = comm->okiDependencyAddr;
  NCCLCHECKGOTO(ncclMemOpSyncWithTrigger(args, stream, dependencyAddr, baseStep+1), res, fail);

  // Copy own data to receive buffer if operation is out-of-place
  if (myRecvBuff != mySendBuff) {
    batchOpsParams.targets[batchOpsParams.numOps] = comm->rank;
    batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
    batchOpsParams.dsts[batchOpsParams.numOps] = (void*)myRecvBuff;
    batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
    batchOpsParams.numOps++;
  } else {
    // Inplace case (sendBuff == recvBuff): we still need to submit a dummy op
    // for every rank so that the hardware queue sees a matched pair of
    // submit (okiMemcpyBatchAsync) and poll (okiBatchSynchronize) on all QPs.
    // The zero-size dummy op is used as a no-op placeholder — okiMemcpyBatchAsync skips
    // any op where size == 0(without actual DMA, with writeFence).
    // This relies on the kernel not treating size==0 as an error or performing
    // a zero-length DMA that could corrupt state.
    batchOpsParams.sizes[batchOpsParams.numOps] = 0;
    batchOpsParams.numOps++;
  }

  offset = myRecvBuff - (uint8_t*)args->recvWin->userPtr;

  for (int r = 0; r < peerNumStage1; r++) {
    int targetRank = icnConnectedPeers[r];
    NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, targetRank, &peerRecvBuff), res, fail);
    batchOpsParams.targets[batchOpsParams.numOps] = targetRank;
    batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
    batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerRecvBuff;
    batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
    batchOpsParams.numOps++;
  }

  NCCLCHECKGOTO(pcclCeLaunchBatchOpsMultiStreamWithOki(comm, &batchOpsParams, args, baseStep, syncStep, stream), res, fail);
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), res, fail);
  comm->ceOkiStep += 1;

exit:
  NCCLCHECK(ncclCeFreeBatchOpsParams(&batchOpsParams));
  return res;
fail:
  goto exit;
}

