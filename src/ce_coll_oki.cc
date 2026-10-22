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
extern void* pcclCeOkiReduceFuncs[];

// Keep submission and polling on the caller's stream so both read the epoch after
// preceding graph/eager work has finished using these QPs. Stream ordering covers
// submission only; the CQ poll is still needed to wait for the asynchronous DMA.
static ncclResult_t pcclCeLaunchBatchOpsWithOki(
    ncclComm_t comm, const struct ncclCeBatchOpsParams* params, hggcStream_t stream) {
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

  auto* sync = (struct ncclCeGridSyncMem*)ncclSymDevBase_getGridSyncBase(
      (struct ncclSymDevBase*)comm->ceColl.ceSyncBase, comm->nRanks);
  uint64_t* stepDev = &sync->okiStep;

  uint32_t stride4G = comm->devrState.bigSize >> 32;
  ltOkiQP_t* qps = comm->okiQps;
  // One warp per QP group, one lane per die.
  int numQpGroups = comm->okiQpCount / ppuDiesNum;
  void* pargs[] = {&args, &qps, &stepDev, &comm->rank, &stride4G, &numQpGroups};
  struct dispatchStrategyExAD launchConfig = {1, 1, 1, -2, /*cooperative=*/0};
  dim3 grid = {1, 1, 1};
  dim3 block = {(uint32_t)(WARP_SIZE * numQpGroups), 1, 1};

  void* func;
  if (ppuDiesNum == 1) func = (void**) okiMemcpyBatchAsync<1>;
  else if (ppuDiesNum == 2) func = (void**) okiMemcpyBatchAsync<2>;
  else return ncclInternalError;
  LAUNCH_KERNEL(func, grid, block, pargs, (ssize_t)0, stream, 0, launchConfig);

  if (ppuDiesNum == 1) func = (void*)okiBatchSynchronize<1>;
  else if (ppuDiesNum == 2) func = (void*)okiBatchSynchronize<2>;
  else return ncclInternalError;
  void* pargs1[] = {&args, &qps, &comm->rank, &stepDev, &numQpGroups};
  // okiBatchSynchronize kernel will wait DMA done launched by okiMemcpyBatchAsync
  LAUNCH_KERNEL(func, grid, block, pargs1, (ssize_t)0, stream, 0, launchConfig);
  return ncclSuccess;
}

struct pcclCeOkiReduceState {
  ncclFunc_t func;
  ssize_t rankElts[8];
  ssize_t rankOffset[8];
  uint8_t* myRecvBuff;
  uint8_t* myStaBuff;
  uint64_t* tailBasePtr;
  uint64_t* headBasePtr;
  ssize_t currentRankElts;
  uint32_t rank0LoopCnt;
  ssize_t splitElts;
  int64_t chunkElts;
  uint64_t totalElts;
};

static ncclResult_t pcclCeOkiReduceSetup(
  ncclComm_t comm,
  struct ncclCeCollArgs* args,
  struct pcclCeOkiReduceState* state
);

static ncclResult_t pcclCeOkiReduceLaunch(
  ncclComm_t comm,
  struct ncclCeCollArgs* args,
  struct pcclCeOkiReduceState* state,
  hggcStream_t stream
);

// One cooperative kernel does the whole collective on the caller's stream: blocks
// [0, pcclCeReduceMaxBlocks) reduce the chunks as the peers' DMA delivers them, and the last
// block barriers with the peers, drives the OKI queues for the reduce-scatter and -- for
// AllReduce -- the allgather, waits for its own completion queues, and barriers again.
//
// Because the kernel does not retire until the cmd block drains its queues,
// stream ordering also serializes access to the shared execution-time epochs.
static ncclResult_t pcclCeOkiReduce(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream, ncclFunc_t func) {
  struct pcclCeOkiReduceState state = {};
  state.func = func;

  NCCLCHECK(pcclCeOkiReduceSetup(comm, args, &state));
  NCCLCHECK(pcclCeOkiReduceLaunch(comm, args, &state, stream));

  return ncclSuccess;
}

ncclResult_t pcclCeOkiAllReduce(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream) {
  return pcclCeOkiReduce(comm, args, stream, ncclFuncAllReduce);
}

ncclResult_t pcclCeOkiReduceScatter(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream) {
  return pcclCeOkiReduce(comm, args, stream, ncclFuncReduceScatter);
}

static ncclResult_t pcclCeOkiReduceSetup(
  ncclComm_t comm,
  struct ncclCeCollArgs* args,
  struct pcclCeOkiReduceState* state
) {
  if (comm->nRanks > 8) {
    ERROR("pcclCeOkiReduce: OKI path only supports nRanks <= 8, got nRanks=%d", comm->nRanks);
    return ncclInternalError;
  }

  bool isReduceScatter = (state->func == ncclFuncReduceScatter);
  // Keep ReduceScatter slices exact; only AllReduce slices are rounded up.
  ssize_t totalElts = isReduceScatter ? (ssize_t)args->nElts * comm->nRanks : (ssize_t)args->nElts;
  int64_t chunkElts;
  if (isReduceScatter) {
    chunkElts = (int64_t)args->nElts;
  } else {
    constexpr int bulkInstSize = 512;
    chunkElts = DIVUP(totalElts, comm->nRanks);
    chunkElts = ROUNDUP(chunkElts, bulkInstSize/args->eltSize);
  }
  state->chunkElts = chunkElts;
  state->totalElts = (uint64_t)totalElts;

  state->rankElts[0] = totalElts > chunkElts ? chunkElts : totalElts;
  state->rankOffset[0] = 0;
  for (int i = 1; i < comm->nRanks; i++) {
    int64_t remain = ((int64_t)totalElts - ((int64_t)i+1) * chunkElts);
    state->rankElts[i] = (remain >= 0) ? chunkElts : std::max<ssize_t>(0, totalElts - i * chunkElts);
    state->rankOffset[i] = state->rankOffset[i-1] + state->rankElts[i-1] * args->eltSize;
  }

  // ReduceScatter's recvBuff is this rank's slice; AllReduce's holds every rank's.
  state->myRecvBuff = (uint8_t*)args->recvBuff + (isReduceScatter ? 0 : state->rankOffset[comm->rank]);
  state->myStaBuff = (uint8_t*)ncclSymDevBase_getStaBuf((struct ncclSymDevBase*)comm->ceColl.ceSyncBase, comm->nRanks);
  state->currentRankElts = state->rankElts[comm->rank];
  state->splitElts = pcclCeReduceChunkSize / args->eltSize;
  state->rank0LoopCnt = DIVUP(state->rankElts[0], state->splitElts);

  state->tailBasePtr = (uint64_t*)ncclSymDevBase_getCeTailBase((struct ncclSymDevBase*)comm->ceColl.ceSyncBase, comm->nRanks);
  state->headBasePtr = (uint64_t*)ncclSymDevBase_getCeHeadBase((struct ncclSymDevBase*)comm->ceColl.ceSyncBase, comm->nRanks);

  return ncclSuccess;
}

static ncclResult_t pcclCeOkiReduceLaunch(
  ncclComm_t comm,
  struct ncclCeCollArgs* args,
  struct pcclCeOkiReduceState* state,
  hggcStream_t stream
) {
  int funcIndex = localReduceFuncIndex(args->datatype, comm->nRanks, args->redOp);
  if (funcIndex == -1) {
    ERROR("not implemented: datatype %d, nRanks %d, op %d", args->datatype, comm->nRanks, args->redOp);
    return ncclInternalError;
  }

  struct pcclCeOkiReduceArgs kargs = {};
  kargs.comm = args->comm;
  kargs.qps = comm->okiQps;
  kargs.sendBuff = args->sendBuff;
  kargs.myRecvBuff = state->myRecvBuff;
  kargs.myStaBuff = state->myStaBuff;
  kargs.tailPtr = state->tailBasePtr;
  kargs.headPtr = state->headBasePtr;
  kargs.gridSyncPtr = ncclSymDevBase_getGridSyncBase((struct ncclSymDevBase*)comm->ceColl.ceSyncBase, comm->nRanks);
  kargs.myElts = state->currentRankElts;
  kargs.nElts = state->totalElts;
  kargs.chunkElts = state->chunkElts;
  kargs.eltSize = args->eltSize;
  kargs.loopCnt = state->rank0LoopCnt;
  kargs.nDies = ppuDiesNum;
  kargs.stride4G = comm->devrState.bigSize >> 32;
  kargs.myRank = comm->rank;
  kargs.broadcast = (state->func != ncclFuncReduceScatter) ? 1 : 0;

  // Peer addresses are rank-0 LSA bases plus targetRank * stride4G.
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

  LAUNCH_KERNEL(pcclCeOkiReduceFuncs[funcIndex], grid, block, pargs, (ssize_t)0, stream, 0, launchConfig);

  return ncclSuccess;
}

// only support one stage algo for now.
ncclResult_t pcclCeOkiAllGather(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream) {
  ncclResult_t res = ncclSuccess;

  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff + comm->rank * chunkBytes;
  void* peerRecvBuff;
  size_t offset;
  struct pcclCeAllGatherStrategy* strategy;

  strategy = comm->ceCollConfigs.agOneStageStrategy;
  int *icnConnectedPeers = strategy->icnConnectedPeers;
  int peerNumStage1 = strategy->peerNumStage1;

  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, comm->nRanks), res, fail);

  // Both DMA kernels follow this barrier on the same stream. The poll kernel
  // advances the shared epoch only after every submitted CQ fence completes.
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), res, fail);

  // Copy own data to receive buffer if operation is out-of-place
  if (myRecvBuff != mySendBuff) {
    batchOpsParams.targets[batchOpsParams.numOps] = comm->rank;
    batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
    batchOpsParams.dsts[batchOpsParams.numOps] = (void*)myRecvBuff;
    batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
    batchOpsParams.numOps++;
  } else {
    // Keep op 0 reserved for the local QP group even when no self copy is needed.
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

  NCCLCHECKGOTO(pcclCeLaunchBatchOpsWithOki(comm, &batchOpsParams, stream), res, fail);
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), res, fail);

exit:
  NCCLCHECK(ncclCeFreeBatchOpsParams(&batchOpsParams));
  return res;
fail:
  goto exit;
}

