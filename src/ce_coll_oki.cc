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
extern void* localReduceFuncs[];

// Sliding-window depth for the AllReduce cmd kernel: the cmd kernel of collective N is not
// launched until collective N-depth's DMA has drained. This keeps a backlog of queued cmd
// work from monopolizing the queue and blocking other cmd tasks from being submitted.
PCCL_PARAM(CeOkiPipeDepth, "CE_OKI_PIPE_DEPTH", 2);

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
  uint32_t blockNum;
};

static ncclResult_t pcclCeOkiAllReduceSetup(
  ncclComm_t comm,
  struct ncclCeCollArgs* args,
  struct pcclCeOkiAllReduceState* state
);

static ncclResult_t pcclCeOkiAllReduceLocalReduce(
  ncclComm_t comm,
  struct ncclCeCollArgs* args,
  struct pcclCeOkiAllReduceState* state,
  hggcStream_t stream
);

static ncclResult_t pcclCeOkiAllReduceDma(
  ncclComm_t comm,
  struct ncclCeCollArgs* args,
  struct pcclCeOkiAllReduceState* state,
  hggcStream_t stream
);

static ncclResult_t pcclCeOkiAllReduceFinalize(
  ncclComm_t comm,
  struct ncclCeCollArgs* args,
  struct pcclCeOkiAllReduceState* state,
  hggcStream_t stream
);

ncclResult_t pcclCeOkiAllReduce(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream) {
  ncclResult_t res = ncclSuccess;
  struct ncclCeBatchOpsParams batchOpsParams = {};
  struct pcclCeOkiAllReduceState state = {};
  uint64_t* dependencyFenceAddr = comm->okiDependencyAddr;

  int64_t depthParam = pcclParamCeOkiPipeDepth();
  if (depthParam < 1) depthParam = 1;
  if (depthParam > PCCL_CE_OKI_MAX_PIPE_DEPTH) depthParam = PCCL_CE_OKI_MAX_PIPE_DEPTH;
  uint32_t depth = (uint32_t)depthParam;
  uint32_t slot = (uint32_t)(comm->okiCollCount % depth);

  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, comm->nRanks), res, fail);
  NCCLCHECKGOTO(ncclMemOpSyncWithTrigger(args, stream, dependencyFenceAddr, comm->ceOkiStep + 1), res, fail);
  NCCLCHECKGOTO(pcclCeOkiAllReduceSetup(comm, args, &state), res, fail);
  // Sliding-window flow control: hold the cmd kernel until collective (N-depth) has drained,
  // so queued cmd work cannot pile up and block other cmd tasks from being submitted.
  if (comm->okiCollCount >= depth) {
    HGGCCHECKGOTO(hggcStreamWaitEvent(comm->ceStreams[0], comm->okiPipeEvents[slot]), res, fail);
  }
  NCCLCHECKGOTO(pcclCeOkiAllReduceDma(comm, args, &state, stream), res, fail);
  NCCLCHECKGOTO(pcclCeOkiAllReduceLocalReduce(comm, args, &state, stream), res, fail);
  // Local reduce runs concurrently with the cmd kernel; join before finalize.
  HGGCCHECKGOTO(hggcStreamWaitEvent(stream, comm->ceEvents[0]), res, fail);
  // Mark this collective's drain point: local reduce polls the CQ until DMA completes, so its
  // completion on `stream` means this collective's queued cmd work is fully consumed.
  HGGCCHECKGOTO(hggcEventRecord(comm->okiPipeEvents[slot], stream), res, fail);
  comm->okiCollCount += 1;
  NCCLCHECKGOTO(pcclCeOkiAllReduceFinalize(comm, args, &state, stream), res, fail);

exit:
  NCCLCHECK(ncclCeFreeBatchOpsParams(&batchOpsParams));
  return res;
fail:
  goto exit;
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

  uint32_t maxBlockNum = pcclCeReduceMaxBlocks;
  constexpr uint32_t maxThreadNum = pcclCeReduceMaxThreads;
  state->blockNum = DIVUP(state->rankElts[0], (maxThreadNum * bulkBytesPerThread / args->eltSize));
  state->blockNum = nextPow2(state->blockNum);
  state->blockNum = std::min(state->blockNum, maxBlockNum);
  state->blockNum = std::max(state->blockNum, 1U);

  return ncclSuccess;
}

static ncclResult_t pcclCeOkiAllReduceLocalReduce(
  ncclComm_t comm,
  struct ncclCeCollArgs* args,
  struct pcclCeOkiAllReduceState* state,
  hggcStream_t stream
) {
  ncclResult_t res = ncclSuccess;
  uint32_t myRank = comm->rank;
  uint32_t threadNum = pcclCeReduceMaxThreads;

  void* myTailOnRank0Ptr = nullptr;
  uint64_t* myTailPtr = state->tailBasePtr + comm->rank * PCCL_CE_MAX_DIE_NUM;
  ssize_t offset = (uint8_t*)myTailPtr - (uint8_t*)comm->ceColl.ceSyncBase;
  NCCLCHECK(ncclDevrGetLsaRankPtr(comm, comm->ceColl.ceSyncWin, offset, 0, &myTailOnRank0Ptr));

  void* myHeadOnRank0Ptr = nullptr;
  uint64_t* myHeadPtr = state->headBasePtr + comm->rank * PCCL_CE_MAX_DIE_NUM;
  offset = (uint8_t*)myHeadPtr - (uint8_t*)comm->ceColl.ceSyncBase;
  NCCLCHECK(ncclDevrGetLsaRankPtr(comm, comm->ceColl.ceSyncWin, offset, 0, &myHeadOnRank0Ptr));

  struct ncclCeGridSyncMem* gridSyncPtr =
    (struct ncclCeGridSyncMem*)ncclSymDevBase_getGridSyncBase((struct ncclSymDevBase*)comm->ceColl.ceSyncBase, comm->nRanks);

  int funcIndex = localReduceFuncIndex(args->datatype, comm->nRanks);
  if (funcIndex == -1) {
    ERROR("not implemented");
    return ncclInternalError;
  }
  const void* func = localReduceFuncs[funcIndex];
  uint32_t stride4G = comm->devrState.bigSize >> 32;
  int64_t baseStep = comm->ceReduceStep;
  uint64_t okiStep = comm->ceOkiStep + 1;
  ltOkiQP_t* qps = comm->okiQps;

  struct dispatchStrategyExAD launchConfig = {/*blocksPerMultiprocessor=*/1, /*gridStepX=*/1, /*gridStepY=*/1, /*subDispatch=*/-2, /*cooperative=*/1};
  dim3 grid = {state->blockNum, 1, 1};
  dim3 block = {threadNum, 1, 1};
  void* pargs[] = {
    &state->myStaBuff, &state->myRecvBuff, &state->currentRankElts, &myRank,
    &state->blockNum, &ppuDiesNum, &myTailOnRank0Ptr, &myHeadOnRank0Ptr,
    &state->rank0LoopCnt, &stride4G, &gridSyncPtr, &baseStep, &qps, &okiStep
  };
  LAUNCH_KERNEL(func, grid, block, pargs, (ssize_t)0, stream, 1, launchConfig);

  return res;
}

static ncclResult_t pcclCeOkiAllReduceDma(
  ncclComm_t comm,
  struct ncclCeCollArgs* args,
  struct pcclCeOkiAllReduceState* state,
  hggcStream_t stream
) {
  ncclResult_t res = ncclSuccess;
  uint64_t* dependencyFenceAddr = comm->okiDependencyAddr;
  ltOkiQP_t* qps = comm->okiQps;

  uint64_t* rank0RemoteFenceAddrs = nullptr;
  uint64_t okiBaseStep = comm->ceOkiStep + 1;
  uint8_t* rank0StaBuff = nullptr;
  uint8_t* rank0RecvBuff = nullptr;
  uint32_t stride4G = comm->devrState.bigSize >> 32;
  struct dispatchStrategyExAD launchConfig = {1, 1, 1, -2, /*cooperative=*/0};
  dim3 grid = {1, 1, 1};
  dim3 block = {(uint32_t)WARP_SIZE * comm->nRanks, 1, 1};
  void* func = nullptr;

  if (ppuDiesNum == 1) func = (void**)okiMemcpyBatchAsyncForAR<1>;
  else if (ppuDiesNum == 2) func = (void**)okiMemcpyBatchAsyncForAR<2>;
  else return ncclInternalError;

  char* rankRemoteFenceAddrs = (char*)ncclSymDevBase_getKuiRemoteFenceAddr((struct ncclSymDevBase*)comm->ceColl.ceSyncBase, comm->nRanks);
  uint64_t offset = (uint8_t*)rankRemoteFenceAddrs - (uint8_t*)comm->ceColl.ceSyncBase;
  NCCLCHECK(ncclDevrGetLsaRankPtr(comm, comm->ceColl.ceSyncWin, offset, 0, (void**)&rank0RemoteFenceAddrs));

  offset = state->myStaBuff - (uint8_t*)comm->ceColl.ceSyncBase;
  NCCLCHECK(ncclDevrGetLsaRankPtr(comm, comm->ceColl.ceSyncWin, offset, 0, (void**)&rank0StaBuff));

  offset = (uint8_t*)args->recvBuff - (uint8_t*)args->recvWin->userPtr;
  NCCLCHECK(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, 0, (void**)&rank0RecvBuff));
  rank0RecvBuff += state->myRecvBuff - (uint8_t*)args->recvBuff;

  void* pargs[] = {
    &qps,
    &dependencyFenceAddr,
    &args->sendBuff,
    &state->myRecvBuff,
    &rank0RecvBuff,
    &rank0StaBuff,
    &comm->rank,
    &comm->nRanks,
    &args->nElts,
    &args->eltSize,
    &state->rank0LoopCnt,
    &rank0RemoteFenceAddrs,
    &okiBaseStep,
    &state->tailBasePtr,
    &state->headBasePtr,
    &comm->ceReduceStep,
    &stride4G
  };

  LAUNCH_KERNEL(func, grid, block, pargs, (ssize_t)0, comm->ceStreams[0], 1, launchConfig);
  HGGCCHECK(hggcEventRecord(comm->ceEvents[0], comm->ceStreams[0]));

  return res;
}

static ncclResult_t pcclCeOkiAllReduceFinalize(
  ncclComm_t comm,
  struct ncclCeCollArgs* args,
  struct pcclCeOkiAllReduceState* state,
  hggcStream_t stream
) {
  NCCLCHECK(ncclMemOpSync(comm, args, stream));
  comm->ceOkiStep += (state->rank0LoopCnt + 1);
  comm->ceReduceStep += state->rank0LoopCnt;
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

