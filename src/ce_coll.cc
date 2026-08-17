/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "pccl_internal.h"
#include "param.h"
#include "comm.h"
#include "register.h"
#include "cudawrap.h"
#include "graph/topo.h"
#include "ce_coll.h"
#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
#include "ce_coll_oki.h"
#endif
#include "alloc.h"
#include "symmetric.h"
#include "bitops.h"
#include "utils.h"
#include "dev_runtime.h"
#include "hgtx.h"

#define CE_COLL_DMA_MIN_SIZE 128*1024
PCCL_PARAM(CeAg2StageMinSize, "CE_AG_2STAGE_MIN_SIZE", 0);
PCCL_PARAM(CeCollOneStream, "CE_COLL_ONE_STREAM", 0);
PCCL_PARAM(CeAg2StageOptMinSize, "CE_AG_2STAGE_OPT_MIN_SIZE", 0);
PCCL_PARAM(CeEnableAr, "CE_ENABLE_AR", 0);
PCCL_PARAM(CeEnableRs, "CE_ENABLE_RS", 0);
PCCL_PARAM(CeDieByRank, "CE_DIE_BY_RANK", 1);

// Translate a local symmetric pointer (inside devr window `win`) to the
// corresponding address on targetRank. `win->userPtr` is this rank's base of
// the window, so the intra-window offset is localPtr - userPtr.
static inline ncclResult_t ceGetSymPeerPtr(ncclComm_t comm, struct ncclDevrWindow* win, void* localPtr, int targetRank, void** outPtr) {
  ssize_t offset = (uint8_t*)localPtr - (uint8_t*)win->userPtr;
  return ncclDevrGetLsaRankPtr(comm, win, offset, targetRank, outPtr);
}

// Machine types that support two-stage topology optimization for CE allgather
static const PPU_TYPE ceAllGatherTwoStageTypes[] = {
  _PPU_ZW810,
  _PPU_ZW810E,
  _ZW_M890P,
  _ZW_M890L,
  _ZW_M530
};
static const int ceAllGatherTwoStageTypesCount = sizeof(ceAllGatherTwoStageTypes) / sizeof(ceAllGatherTwoStageTypes[0]);

static inline bool isCeAllGatherTwoStageSupported(PPU_TYPE ppuType) {
  for (int i = 0; i < ceAllGatherTwoStageTypesCount; i++) {
    if (ppuType == ceAllGatherTwoStageTypes[i]) return true;
  }
  return false;
}

extern int ppuDiesNum;

ncclResult_t ncclInitCeAllGatherStrategy(struct ncclCeAllGatherStrategy* strategy, ncclComm_t comm, int twoStage) {
  int* peerNumStage1 = &(strategy->peerNumStage1);
  int* peerNumStage2 = &(strategy->peerNumStage2);
  int* icnConnectedPeers = strategy->icnConnectedPeers;

  // Initialize counts: by default, all peers participate in stage 1
  *peerNumStage1 = comm->nRanks - 1;
  *peerNumStage2 = 0;
  for (int i = 0; i < comm->nRanks - 1; i++) {
    icnConnectedPeers[i] = (i + comm->rank + 1) % comm->nRanks;
  }

  if (twoStage) {
    // Apply topology optimization for specific PPU hardware types when running with 8 ranks
    // See ceAllGatherTwoStageTypes for supported machine types
    if (comm->nRanks == 8 && comm->topo->nodes[NVS].count == 0) {
      if (isCeAllGatherTwoStageSupported(comm->ppuType)) {
        int* topoArray = NULL;
        NCCLCHECK(ncclCalloc(&topoArray, comm->nRanks * comm->nRanks));
        // OPTIMIZATION OPPORTUNITY: Cache topology information in the comm structure
        // Rationale: Topology data is computed here but could be reused elsewhere in the system.
        // Storing it in comm would avoid redundant computations in other parts of the codebase.

        // FUTURE IMPROVEMENT: Implement topology validation for all machine types
        // Action: Compare the runtime-discovered topology with expected/reference topologies
        // Benefit: Early detection of unexpected network configurations or hardware issues
        NCCLCHECK(pcclFillPPUTopoArray(topoArray, comm));
        int directPeers = 0;
        for (int peer = 0; peer < comm->nRanks; peer++) {
          if (peer != comm->rank) {
            if (topoArray[comm->rank*comm->nRanks + peer]) {
              icnConnectedPeers[directPeers++] = peer;
            }
          }
        }

        // Topology requirement: Each peer must have exactly 4 direct connections via ICN
        // This constraint ensures optimal two-stage communication patterns for this specific hardware
        if (directPeers == 4) {
          int isolatedPeerIndex = 0;
          for (int i = 0; i < directPeers; i++) {
            int peerI = icnConnectedPeers[i];
            bool isolatePeerFlag = true;
            for (int j = 0; j < directPeers; j++) {
              if (i == j) continue;
              int peerJ = icnConnectedPeers[j];
              if (topoArray[peerI * comm->nRanks + peerJ]){
                isolatePeerFlag = false;
                break;
              }
            }
            if (isolatePeerFlag) {
              isolatedPeerIndex = i;
              break;
            }
          }

          // Reorder so that the isolated peer (not in the 4FC region) comes first
          std::swap(icnConnectedPeers[0], icnConnectedPeers[isolatedPeerIndex]);

          // Update stage counts for optimized two-stage operation
          *peerNumStage1 = 4;
          *peerNumStage2 = 3;
        }
        NCCLCHECK(ncclFree(topoArray));
      }
    }
  }

  return ncclSuccess;
}

// Datatype/op gating shared by reduce-family CE collectives (AllReduce, ReduceScatter).
// Runtime gating that needs comm/buffers (nRanks, dies, recvbuff alignment) lives in
// ncclCeAvailable; here we only have coll/red/ty.
static inline bool ceReduceTypeSupported(int/*ncclRedOp_t*/ red, ncclDataType_t ty) {
  if (ty != pcclHalf && ty != pcclBfloat16 && ty != pcclFloat) return false;
  if (red != ncclSum) return false;
  return true;
}

bool ncclCeImplemented(ncclFunc_t coll, int/*ncclRedOp_t*/ red, ncclDataType_t ty) {
  switch (coll) {
  case ncclFuncAllGather:
  case ncclFuncAlltoAll:
  case ncclFuncScatter:
  case ncclFuncGather:
    return true;
  case ncclFuncAllReduce:
    if (pcclParamCeEnableAr() != 1) return false;
    if (ppuDiesNum > PCCL_CE_MAX_DIE_NUM) {
      WARN("CE: config %d dies, but max %d dies supported for ce AllReduce", ppuDiesNum, PCCL_CE_MAX_DIE_NUM);
      return false;
    }
    return ceReduceTypeSupported(red, ty);
  case ncclFuncReduceScatter:
    if (pcclParamCeEnableRs() != 1) return false;
    if (ppuDiesNum > PCCL_CE_MAX_DIE_NUM) {
      WARN("CE: config %d dies, but max %d dies supported for ce ReduceScatter", ppuDiesNum, PCCL_CE_MAX_DIE_NUM);
      return false;
    }
    return ceReduceTypeSupported(red, ty);
  default:
    return false;
  }
  return false;
}

// Same graph state that comm->planner.capturingGraph ends up holding: ncclPlannerSetCapturingGraph
// runs this very query, only later, during task append — so the planner is not yet populated when
// CE admission is decided here. A failed query counts as capturing, i.e. CE is not used.
static inline bool ceStreamCapturing(hggcStream_t stream) {
  struct ncclCudaGraph graph;
  if (ncclCudaGetCapturingGraph(&graph, stream) != ncclSuccess) return true;
  return ncclCudaGraphValid(graph);
}

bool ncclCeAvailable(ncclComm_t comm, ncclFunc_t coll, int/*ncclRedOp_t*/ red, ncclDataType_t ty, ncclSymRegType_t winRegType, const void* recvbuff, hggcStream_t stream) {
  if (!ncclCeImplemented(coll, red, ty)) {
    TRACE(NCCL_TUNING, "Skipping CE collective: not implemented");
    return false;
  }
  bool isReduceFamily = (coll == ncclFuncAllReduce || coll == ncclFuncReduceScatter);
  // The reduce-family tail/head handshake is driven by host-side step counters
  // (comm->ceReduceStep, comm->ceOkiStep) that advance once per launch. A captured graph
  // replays the values baked in at capture time, so the waits would never be satisfied.
  if (isReduceFamily && ceStreamCapturing(stream)) {
    TRACE(NCCL_TUNING, "Skipping CE reduce: graph capture is not supported");
    return false;
  }
  if (coll == ncclFuncAllReduce && comm->nRanks > 8) {
    WARN("CE: AllReduce only support nRanks <=8");
    return false;
  }
  // Reduce-family CE kernels only exist for nRanks 2/4/8 (see localReduceFuncIndex).
  if (isReduceFamily && comm->nRanks != 2 && comm->nRanks != 4 && comm->nRanks != 8) {
    TRACE(NCCL_TUNING, "Skipping CE reduce: nRanks %d not in {2,4,8}", comm->nRanks);
    return false;
  }
  // TODO: the initial version of ce reduce always stores with bulk first,
  // without the pre/aft handling of the symmetric kernel, so recvbuff must be 4B aligned.
  if (isReduceFamily && (uint64_t)recvbuff % 4 != 0) {
    TRACE(NCCL_TUNING, "Skipping CE reduce: recvbuff %p is not 4-byte aligned", recvbuff);
    return false;
  }
  if (ncclTeamLsa(comm).nRanks < comm->nRanks) {
    TRACE(NCCL_TUNING, "Skipping CE collective: not all ranks have NVLink connectivity");
    return false;
  }
  if (!comm->symmetricSupport) {
    TRACE(NCCL_TUNING, "Skipping CE collective: symmetric support is not enabled");
    return false;
  }
  if (winRegType != ncclSymSendRegRecvReg && winRegType != ncclSymSendNonregRecvReg) {
    TRACE(NCCL_TUNING, "Skipping CE collective: window registration type %d is not supported", winRegType);
    return false;
  }
  return true;
}

// Allocate CE-private symmetric storage as a devr-backed window. Layout is
// the existing ncclSymDevBase so ncclSymDevBase_get* accessors index from
// comm->ceColl.ceSyncBase. Called lazily from ncclSymkInitOnce.
ncclResult_t ncclCeInit(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclCeColl* cec = &comm->ceColl;
  if (cec->initialized) return ncclSuccess;

  uint8_t* ceDevBase = nullptr;
  // CE accessors index with comm->nRanks; allocate with the same so peer
  // access never overflows the buffer. CE is single-node, so this matches
  // ncclTeamLsa(comm).nRanks in practice.
  size_t ceDevBaseSize = ncclSymDevBase::size(comm->nRanks);
  ncclWindow_vidmem* ceWinDev = nullptr;
  ncclWindow_vidmem* ceWinDevHost = nullptr;
  uint64_t* ceArStepInitMem = nullptr;

  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);
  NCCLCHECKGOTO(pcclMemAlloc((void**)&ceDevBase, ceDevBaseSize), ret, fail);
  NCCLCHECKGOTO(ncclDevrWindowRegisterInGroup(comm, ceDevBase, ceDevBaseSize, NCCL_WIN_COLL_SYMMETRIC, &ceWinDev), ret, fail);
  NCCLCHECKGOTO(ncclShadowPoolToHost(&comm->devrState.shadows, ceWinDev, &ceWinDevHost), ret, fail);
  cec->ceSyncWin = (struct ncclDevrWindow*)ceWinDevHost->winHost;
  cec->ceSyncBase = (uint8_t*)cec->ceSyncWin->userPtr;
  cec->ceSyncSize = ceDevBaseSize;
  // gpuBarrier kernel reads peers via add4G(base, (peer-rank)*stride4G); base
  // must be the lsa-flat self slot (NOT userPtr, which is the local cumem
  // VA -- mapped to the same physical memory but a separate VA range).
  cec->symDevComm.base = (struct ncclSymDevBase*)((uint8_t*)ceWinDevHost->lsaFlatBase
                          + (size_t)ceWinDevHost->lsaRank * comm->devrState.bigSize);
  cec->symDevComm.stride4G = comm->devrState.bigSize >> 32;
  cec->symDevComm.nRanks = comm->localRanks;
  cec->symDevComm.nRanks_rcp32 = idivRcp32(comm->localRanks);
  cec->symDevComm.rank = comm->localRank;

  HGGCCHECKGOTO(hggcMemset(cec->ceSyncBase, 0, ceDevBaseSize), ret, fail);
  // AllReduce and ReduceScatter share the same CE tail/head sync arrays and the
  // same waitValue semantics (waitValue = 1 + iter - pcclCeReduceMaxSect, with the
  // first iter == comm->ceReduceStep). The arrays must be pre-seeded to ceReduceStep
  // so the first GEQ wait passes; otherwise a reduce-family collective deadlocks.
  // Seed whenever either reduce collective is CE-enabled (not just AllReduce).
  if (ncclCeImplemented(ncclFuncAllReduce, ncclSum, pcclFloat32) ||
      ncclCeImplemented(ncclFuncReduceScatter, ncclSum, pcclFloat32)) {
    int count = comm->nRanks * PCCL_CE_MAX_DIE_NUM;
    NCCLCHECKGOTO(ncclCalloc(&ceArStepInitMem, count), ret, fail);
    for (int i = 0; i < count; i++) ceArStepInitMem[i] = comm->ceReduceStep;
    uint64_t* tailBase = (uint64_t*)ncclSymDevBase_getCeTailBase((struct ncclSymDevBase*)cec->ceSyncBase, comm->nRanks);
    uint64_t* headBase = (uint64_t*)ncclSymDevBase_getCeHeadBase((struct ncclSymDevBase*)cec->ceSyncBase, comm->nRanks);
    HGGCCHECKGOTO(hggcMemcpy(tailBase, ceArStepInitMem, count * sizeof(uint64_t), hggcMemcpyHostToDevice), ret, fail);
    HGGCCHECKGOTO(hggcMemcpy(headBase, ceArStepInitMem, count * sizeof(uint64_t), hggcMemcpyHostToDevice), ret, fail);
  }
  cec->initialized = true;

exit:
  if (ceArStepInitMem) ncclFree(ceArStepInitMem);
  return ret;
fail:
  if (ceWinDev) (void)pcclCommWindowDeregister(comm, ceWinDev);
  if (ceDevBase) (void)pcclMemFree(ceDevBase);
  cec->ceSyncWin = nullptr;
  cec->ceSyncBase = nullptr;
  goto exit;
}

ncclResult_t ncclCeFinalize(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclCeColl* cec = &comm->ceColl;
  if (!cec->initialized) return ncclSuccess;
  if (cec->ceSyncWin) {
    void* userPtr = cec->ceSyncWin->userPtr;
    NCCLCHECKIGNORE(pcclCommWindowDeregister(comm, cec->ceSyncWin->vidmem), ret);
    NCCLCHECKIGNORE(pcclMemFree(userPtr), ret);
    cec->ceSyncWin = nullptr;
    cec->ceSyncBase = nullptr;
  }
  cec->initialized = false;
  return ret;
}

//INFO: implement with PPU compact sync
ncclResult_t ncclMemOpSync(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream) {
  void* ceSyncHandle = NULL;
  // Start CE sync profiling
  NCCLCHECK(ncclProfilerStartCeSyncEvent(comm, args, stream, &ceSyncHandle));
  void* pargs[1];
  pargs[0] = &args->comm;
  int nThreads = ROUNDUP(args->comm.nRanks, WARP_SIZE);
  HGGCCHECK(hggcLaunchKernel((const void*)gpuBarrier, 1, nThreads, pargs, 0, stream));
  // Stop CE sync profiling - always attempt if started, even on error
  ncclProfilerStopCeSyncEvent(comm, ceSyncHandle, stream);
  return ncclSuccess;
}

#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
ncclResult_t ncclMemOpSyncWithTrigger(struct ncclCeCollArgs* args, hggcStream_t stream, uint64_t* triggerPtr, uint64_t trigger) {
  void* pargs[] = {&args->comm, &triggerPtr, &trigger};
  int nThreads = ROUNDUP(args->comm.nRanks, WARP_SIZE);
  HGGCCHECK(hggcLaunchKernel((const void*)gpuBarrierWithTrigger, 1, nThreads, pargs, 0, stream));
  return ncclSuccess;
}
#endif

ncclResult_t ncclCeInitBatchOpsParams(struct ncclCeBatchOpsParams* params, int nRanks) {
  ncclResult_t ret = ncclSuccess;

  params->srcs = nullptr;
  params->dsts = nullptr;
  params->sizes = nullptr;
  params->targets = nullptr;
  params->numOps = 0;

#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
  params->attrs = nullptr;
  params->attrIdxs = nullptr;
  params->numAttrs = 0;
#endif

  NCCLCHECKGOTO(ncclCalloc(&params->srcs, nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&params->dsts, nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&params->sizes, nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&params->targets, nRanks), ret, fail);

#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
  NCCLCHECKGOTO(ncclCalloc(&params->attrs, nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&params->attrIdxs, nRanks), ret, fail);
#endif

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCeFreeBatchOpsParams(struct ncclCeBatchOpsParams* params) {
  NCCLCHECK(ncclFree(params->srcs));
  NCCLCHECK(ncclFree(params->dsts));
  NCCLCHECK(ncclFree(params->sizes));
  NCCLCHECK(ncclFree(params->targets));
#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
  NCCLCHECK(ncclFree(params->attrs));
  NCCLCHECK(ncclFree(params->attrIdxs));
#endif
  return ncclSuccess;
}

void splitSize(ssize_t totalSize, int round, int numRounds, ssize_t* size) {
  ssize_t perRoundSize = DIVUP(totalSize, numRounds);
  if (perRoundSize < CE_COLL_DMA_MIN_SIZE) perRoundSize = CE_COLL_DMA_MIN_SIZE;

  *size = totalSize - perRoundSize * round;
  if (*size > perRoundSize) *size = perRoundSize;
}

ncclResult_t ncclCeLaunchBatchOpsAsync(ncclComm_t comm, const struct ncclCeBatchOpsParams* params, hggcStream_t stream, int64_t iter, uint64_t* tailBasePtr, uint64_t* headBasePtr) {
  ncclResult_t ret = ncclSuccess;

  // Check if there are any operations to perform
  if (params->numOps == 0) {
    return ncclSuccess;
  }
  uint64_t waitValue = 1L + iter - pcclCeReduceMaxSect;
  uint64_t postValue = 1L + iter;

#ifdef SYNC_WITH_KERNEL
  void* pargs[] = {&tailBasePtr, &waitValue, &ppuDiesNum};
  HGGCCHECK(hggcLaunchKernel((const void*)gpuPreDma, 1, comm->nRanks, pargs, 0, comm->waitReduceStream));
  HGGCCHECKGOTO(hggcEventRecord(comm->dmaWaitEvent, comm->waitReduceStream), ret, fail);
#endif


  for (int i = 0; i < params->numOps; i++) {
    ssize_t offset = 0;
    int targetRank = params->targets[i];
    for (int dieIdx = 0; dieIdx < ppuDiesNum; dieIdx++) {
      ssize_t size = 0;
      if (targetRank == comm->rank) {
        size = dieIdx == 0 ? params->sizes[i] : 0;
      } else {
        splitSize(params->sizes[i], dieIdx, ppuDiesNum, &size);
      }

      // fix streamId by rank and die, to guarantee head/tail store consequently.
      int streamId = (targetRank % HG_DMA_RING_PER_DIE) * ppuDiesNum + dieIdx;
#ifdef SYNC_WITH_WAIT_VALUE
      uint64_t* tailPtrPerDie = tailBasePtr + targetRank * PCCL_CE_MAX_DIE_NUM + dieIdx;
      HGCHECKGOTO(hgStreamWaitValue64(comm->ceStreams[streamId], (HGdeviceptr)tailPtrPerDie, waitValue, HG_STREAM_WAIT_VALUE_GEQ), ret, fail);
#endif


      if (size >= 0) {
#ifdef SYNC_WITH_KERNEL
        HGGCCHECKGOTO(hggcStreamWaitEvent(comm->ceStreams[streamId], comm->dmaWaitEvent), ret, fail);
#endif
        HGGCCHECKGOTO(hggcMemcpyAsync(
          (void*)((uintptr_t)params->dsts[i] + offset),
          (void*)((uintptr_t)params->srcs[i] + offset),
          size,
          hggcMemcpyDeviceToDevice,
          comm->ceStreams[streamId]), ret, fail);
        offset += size;
      }

#ifdef SYNC_WITH_WAIT_VALUE
      uint64_t* headPtrPerDie = headBasePtr + targetRank * PCCL_CE_MAX_DIE_NUM + dieIdx;
      HGCHECKGOTO(hgStreamWriteValue64(comm->ceStreams[streamId], (HGdeviceptr)headPtrPerDie, postValue, HG_STREAM_WRITE_VALUE_DEFAULT), ret, fail);
#endif
    }
  }

#ifdef SYNC_WITH_KERNEL
  {
    for (int i = 0; i < params->numOps; i++) {
      int targetRank = params->targets[i];
      for (int dieIdx = 0; dieIdx < ppuDiesNum; dieIdx++) {
        int id = (targetRank % HG_DMA_RING_PER_DIE) * ppuDiesNum + dieIdx;
        HGGCCHECKGOTO(hggcEventRecord(comm->ceEvents[id], comm->ceStreams[id]), ret, fail);
        HGGCCHECKGOTO(hggcStreamWaitEvent(comm->notifyReduceStream, comm->ceEvents[id]), ret, fail);
      }
    }
    void* pargs1[] = {&headBasePtr, &postValue};
    HGGCCHECK(hggcLaunchKernel((const void*)gpuPostDma, 1, comm->nRanks, pargs1, 0, comm->notifyReduceStream));
  }
#endif


exit:
  return ret;
fail:
  goto exit;
}

// Collectives that build one op per target rank with a symmetric target set, so
// the distance-keyed die assignment below stays balanced on both the sending and
// the receiving side. Scatter and Gather have a single sender resp. receiver, so
// only that side has anything to balance, and that side is the bottleneck.
//
// The reduce family is left out: its scatter phase goes through
// ncclCeLaunchBatchOpsAsync, which pairs the even split with per-die head/tail
// counters, so only its broadcast phase would change here and that combination is
// not measured yet.
static inline bool ceDieByRankApplies(ncclFunc_t coll) {
  switch (coll) {
  case ncclFuncAllGather:
  case ncclFuncAlltoAll:
  case ncclFuncScatter:
  case ncclFuncGather:
    return true;
  default:
    return false;
  }
}

// Die and DMA queue assignment for a batch op targeting `targetRank`.
//
// Default: the transfer is split evenly across all dies, queue picked by target rank.
//
// dieByRank: the transfer stays whole on a single die, which halves the number of
// DMA ops. The die is keyed on the rank distance (targetRank - myRank) mod nRanks,
// not on targetRank: a receiver sees exactly the same set of distances that a
// sender does, so egress and ingress balance across dies at the same time. Keying
// on the target instead makes every sender reach a given peer on the same die,
// which starves that peer's other die.
//
// nRanks-1 need not divide by the die count. The leftover distances, and the local
// copy, keep the even split, which is what makes each rank's ingress land on both
// dies in exactly equal parts.
static inline void ceAssignDie(bool dieByRank, int myRank, int targetRank, int nRanks,
                               int* dieBegin, int* dieEnd, int* queue) {
  if (dieByRank) {
    int splitDist = (nRanks - 1) % ppuDiesNum;
    int dist = (targetRank - myRank + nRanks) % nRanks;
    if (dist > splitDist) {
      int idx = dist - splitDist - 1;
      *dieBegin = idx % ppuDiesNum;
      *dieEnd = *dieBegin + 1;
      *queue = (idx / ppuDiesNum) % HG_DMA_RING_PER_DIE;
      return;
    }
    // Queues of the split ops start above the ones the whole ops occupy.
    int wholePerDie = (nRanks - 1 - splitDist) / ppuDiesNum;
    *dieBegin = 0;
    *dieEnd = ppuDiesNum;
    *queue = (wholePerDie + dist) % HG_DMA_RING_PER_DIE;
    return;
  }

  *dieBegin = 0;
  *dieEnd = ppuDiesNum;
  *queue = targetRank % HG_DMA_RING_PER_DIE;
}

static ncclResult_t ncclCeLaunchBatchOpsMultiStream(ncclComm_t comm, struct ncclCeCollArgs* args,
                                             const struct ncclCeBatchOpsParams* params, hggcStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  void* ceBatchHandle = NULL;
  bool dieByRank;
  int dieBegin, dieEnd, queue;

  // Start CE batch profiling
  NCCLCHECK(ncclProfilerStartCeBatchEvent(comm, args, params, stream, &ceBatchHandle));

  // Check if there are any operations to perform
  if (params->numOps == 0) {
    return ncclSuccess;
  }

  dieByRank = ppuDiesNum > 1 && comm->icnRmtType == PCCL_ICNLINK_SWITCH &&
              ceDieByRankApplies(args->func) && pcclParamCeDieByRank() != 0;
  // Tailored to 64 ranks for a smoother curve.
  if (comm->nRanks == 64) {
    if (args->func == ncclFuncAllGather || args->func == ncclFuncAlltoAll) {
      ssize_t size = args->nElts * args->eltSize;
      if (size < 512 * 1024 && size > 128 * 1024) {
        dieByRank = 0;
      }
    }
  }

  // Check if we are in a HGGC graph capture
  HGGCCHECKGOTO(hggcEventRecord(comm->ceSyncEvent, stream), ret, fail);
  {
    for (int i = 0; i < params->numOps; i++) {
      ssize_t offset = 0;
      int32_t targetRank = params->targets[i];
      ceAssignDie(dieByRank, comm->rank, targetRank, comm->nRanks, &dieBegin, &dieEnd, &queue);
      // A single die in the range means this op is carried whole; otherwise it
      // still takes one contiguous slab per die.
      bool whole = (dieEnd - dieBegin) == 1;
      for (int dieId = dieBegin; dieId < dieEnd; dieId++) {
        ssize_t size = 0;
        int streamId = queue * ppuDiesNum + dieId;
        if (whole) {
          size = params->sizes[i];
        } else {
          splitSize(params->sizes[i], dieId, ppuDiesNum, &size);
        }
        HGGCCHECKGOTO(hggcStreamWaitEvent(comm->ceStreams[streamId], comm->ceSyncEvent), ret, fail);
        if (size >= 0) {
          HGGCCHECKGOTO(hggcMemcpyAsync(
            (void*)((uintptr_t)params->dsts[i] + offset),
            (void*)((uintptr_t)params->srcs[i] + offset),
            size,
            hggcMemcpyDeviceToDevice,
            comm->ceStreams[streamId]), ret, fail);
        }
        HGGCCHECKGOTO(hggcEventRecord(comm->ceEvents[streamId], comm->ceStreams[streamId]), ret, fail);
        offset += size;
      }
    }
    for (int i = 0; i < params->numOps; i++) {
      int32_t targetRank = params->targets[i];
      ceAssignDie(dieByRank, comm->rank, targetRank, comm->nRanks, &dieBegin, &dieEnd, &queue);
      for (int dieId = dieBegin; dieId < dieEnd; dieId++) {
        int streamId = queue * ppuDiesNum + dieId;
        HGGCCHECKGOTO(hggcStreamWaitEvent(stream, comm->ceEvents[streamId]), ret, fail);
      }
    }
  }

exit:
  // Stop CE batch profiling - always attempt if started, even on error
  ncclProfilerStopCeBatchEvent(comm, ceBatchHandle, stream);
  return ret;
fail:
  goto exit;
}

// single P2p stream per die.
static ncclResult_t launchBatchOpsWithSingleP2pStream(ncclComm_t comm, const struct ncclCeBatchOpsParams* params, hggcStream_t stream) {
  HGGCCHECK(hggcEventRecord(comm->ceSyncEvent, stream));
  for (int local = 0; local < 2; local++) {
    for (int dieId = 0; dieId < ppuDiesNum; dieId++) {
      int streamId = (local % HG_DMA_RING_PER_DIE) * ppuDiesNum + dieId;
      HGGCCHECK(hggcStreamWaitEvent(comm->ceStreams[streamId], comm->ceSyncEvent));
    }
  }

  for (int i = 0; i < params->numOps; i++) {
    ssize_t offset = 0;
    int local = (params->targets[i] == comm->rank) ? 1 : 0;
    for (int dieId = 0; dieId < ppuDiesNum; dieId++) {
      ssize_t size = 0;
      int streamId = (local % HG_DMA_RING_PER_DIE) * ppuDiesNum + dieId;
      splitSize(params->sizes[i], dieId, ppuDiesNum, &size);
      if (size >= 0) {
        HGGCCHECK(hggcMemcpyAsync(
          (void*)((uintptr_t)params->dsts[i] + offset),
          (void*)((uintptr_t)params->srcs[i] + offset),
          size,
          hggcMemcpyDeviceToDevice,
          comm->ceStreams[streamId]));
      }
      offset += size;
    }
  }
  for (int local = 0; local < 2; local++) {
    for (int dieId = 0; dieId < ppuDiesNum; dieId++) {
      int streamId = (local % HG_DMA_RING_PER_DIE) * ppuDiesNum + dieId;
      HGGCCHECK(hggcEventRecord(comm->ceEvents[streamId], comm->ceStreams[streamId]));
      HGGCCHECK(hggcStreamWaitEvent(stream, comm->ceEvents[streamId]));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclCeLaunchBatchOps(ncclComm_t comm, struct ncclCeCollArgs* args,
                                  struct ncclCeBatchOpsParams* params, hggcStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  uint32_t driverVersion;
  void* ceBatchHandle = NULL;

  // Start CE batch profiling
  NCCLCHECK(ncclProfilerStartCeBatchEvent(comm, args, params, stream, &ceBatchHandle));

  // Check if there are any operations to perform
  if (params->numOps == 0) {
    return ncclSuccess;
  }

  // Check if we are in a HGGC graph capture
  bool capturing = ncclCudaGraphValid(comm->planner.capturingGraph);

  NCCLCHECKGOTO(ncclHggcDriverVersion(&driverVersion), ret, fail);

  //--------------Graph capture--------------
  // hggcMemcpyBatchAsync is not supported during HGGC graph capture
  if (capturing) {
    NCCLCHECKGOTO(launchBatchOpsWithSingleP2pStream(comm, params, stream), ret, fail);
  }
  //--------------No graph capture--------------
  else {
    if (PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0) && driverVersion >= PCCL_SYSTEM_DRIVER_VERSION(6, 0, 0)) {
#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
      params->attrs[0] = {};
      params->attrs[0].srcAccessOrder = hggcMemcpySrcAccessOrderStream;
      params->attrs[0].flags = hggcMemcpyFlagPreferOverlapWithCompute;
      params->attrIdxs[0] = 0;
      params->numAttrs = 1;

#if HGGCRT_VERSION >= 13000
      HGGCCHECKGOTO(hggcMemcpyBatchAsync(params->dsts, params->srcs, params->sizes, params->numOps, params->attrs, params->attrIdxs, params->numAttrs, stream),
                    ret, fail);
#else
      HGGCCHECKGOTO(hggcMemcpyBatchAsync(params->dsts, params->srcs, params->sizes, params->numOps, params->attrs, params->attrIdxs, params->numAttrs, nullptr, stream),
                    ret, fail);
#endif
  #endif
    } else {
      NCCLCHECKGOTO(launchBatchOpsWithSingleP2pStream(comm, params, stream), ret, fail);
    }
  }

exit:
  // Stop CE batch profiling - always attempt if started, even on error
  ncclProfilerStopCeBatchEvent(comm, ceBatchHandle, stream);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCeLaunchBatchOpsImpl(ncclComm_t comm, struct ncclCeCollArgs* args,
                                      struct ncclCeBatchOpsParams* params, hggcStream_t stream) {
  if (pcclParamCeCollOneStream() > 0) {
    return ncclCeLaunchBatchOps(comm, args, params, stream);
  } else {
    return ncclCeLaunchBatchOpsMultiStream(comm, args, params, stream);
  }
}

ncclResult_t ncclCeAllGather(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // Calculate the size of each rank's data chunk
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff + comm->rank * chunkBytes;
  void* peerRecvBuff;
  size_t offset;
  struct ncclCeAllGatherStrategy* strategy;
  // for unmatched topo OneStageStrategy and TwoStageStrategy are exactly the same.
  if (comm->nRanks != 8 || chunkBytes <= pcclParamCeAg2StageMinSize()) {
    strategy = comm->ceCollConfigs.agOneStageStrategy;
  } else {
    strategy = comm->ceCollConfigs.agTwoStageStrategy;
  }
  int *icnConnectedPeers = strategy->icnConnectedPeers;
  int peerNumStage1 = strategy->peerNumStage1;
  int peerNumStage2 = strategy->peerNumStage2;

  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, comm->nRanks), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

  // Copy own data to receive buffer if operation is out-of-place
  if (myRecvBuff != mySendBuff) {
    batchOpsParams.targets[batchOpsParams.numOps] = comm->rank;
    batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
    batchOpsParams.dsts[batchOpsParams.numOps] = (void*)myRecvBuff;
    batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
    batchOpsParams.numOps++;
  }

  // Copy data to other ranks
  offset = myRecvBuff - (uint8_t*)args->recvWin->userPtr;

  for (int r = 0; r < peerNumStage1; r++) {
    int targetRank = icnConnectedPeers[r];
    NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, targetRank, &peerRecvBuff), ret, fail);
    batchOpsParams.targets[batchOpsParams.numOps] = targetRank;
    batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
    batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerRecvBuff;
    batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
    batchOpsParams.numOps++;
  }

  NCCLCHECKGOTO(ncclCeLaunchBatchOpsImpl(comm, args, &batchOpsParams, stream), ret, fail);
  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

  if (peerNumStage2 > 0) {
    batchOpsParams.numOps = 0;
    int stage2Source = icnConnectedPeers[0];
    size_t firstPartBytes = chunkBytes;
    if (chunkBytes > pcclParamCeAg2StageOptMinSize()) firstPartBytes = (firstPartBytes * 3 / 4);
    offset = (uint8_t*)args->recvBuff + stage2Source * chunkBytes - (uint8_t*)args->recvWin->userPtr;
    for (int r = 0; r < peerNumStage2; r++) {
      int targetRank = icnConnectedPeers[r+1];
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, targetRank, &peerRecvBuff), ret, fail);
      batchOpsParams.targets[batchOpsParams.numOps] = targetRank;
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)((uintptr_t)args->recvBuff + chunkBytes * stage2Source);
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerRecvBuff;
      batchOpsParams.sizes[batchOpsParams.numOps] = firstPartBytes;
      batchOpsParams.numOps++;
    }

    size_t secondPartBytes = chunkBytes - firstPartBytes;
    if (secondPartBytes > 0) {
      int targetRank = icnConnectedPeers[0];
      for (int r = 0; r < peerNumStage2; r++) {
        int srcRank = icnConnectedPeers[r+1];
        offset = (uint8_t*)args->recvBuff + srcRank * chunkBytes - (uint8_t*)args->recvWin->userPtr;
        NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, targetRank, &peerRecvBuff), ret, fail);
        batchOpsParams.targets[batchOpsParams.numOps] = targetRank;
        batchOpsParams.srcs[batchOpsParams.numOps] = (void*)((uintptr_t)args->recvBuff + chunkBytes * srcRank + firstPartBytes);
        batchOpsParams.dsts[batchOpsParams.numOps] = (void*)((uintptr_t)peerRecvBuff + firstPartBytes);
        batchOpsParams.sizes[batchOpsParams.numOps] = secondPartBytes;
        batchOpsParams.numOps++;
      }
    }

    NCCLCHECKGOTO(ncclCeLaunchBatchOpsImpl(comm, args, &batchOpsParams, stream), ret, fail);
    NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);
  }

exit:
  NCCLCHECK(ncclCeFreeBatchOpsParams(&batchOpsParams));
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCeAlltoAll(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // Calculate the size of data each rank sends to every other rank
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  void* peerRecvBuff;
  size_t offset;

  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, comm->nRanks * comm->nRanks), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

  // Copy data to other ranks: send data chunk for each destination rank
  for (int r = 0; r < comm->nRanks; r++) {
    int dstRank = (comm->rank + r) % comm->nRanks;
    uint8_t* srcPtr = mySendBuff + dstRank * chunkBytes;
    uint8_t* dstPtr = myRecvBuff + comm->rank * chunkBytes;

    if (dstRank == comm->rank) {
      // Local copy for own data
      batchOpsParams.targets[batchOpsParams.numOps] = dstRank;
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)dstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    } else {
      // Remote copy to other ranks: send to rank dstRank's receive buffer at position comm->rank
      offset = dstPtr - (uint8_t*)args->recvWin->userPtr;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, dstRank, &peerRecvBuff), ret, fail);
      batchOpsParams.targets[batchOpsParams.numOps] = dstRank;
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerRecvBuff;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }
  }

  // Launch the batch operations
  NCCLCHECKGOTO(ncclCeLaunchBatchOpsImpl(comm, args, &batchOpsParams, stream), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCeScatter(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // Calculate the size of data root sends to each rank
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  int rootRank = args->rootRank;
  void* peerDstPtr;
  size_t offset;

  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, comm->nRanks), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

  if (comm->rank == rootRank) {
    // Check if this is an in-place scatter operation
    bool isInPlace = (myRecvBuff == mySendBuff + comm->rank * chunkBytes);

    // Copy root's own data first if not in-place
    if (!isInPlace) {
      uint8_t* srcPtr = mySendBuff + comm->rank * chunkBytes;
      uint8_t* dstPtr = myRecvBuff;
      batchOpsParams.targets[batchOpsParams.numOps] = comm->rank;
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)dstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }

    // Root rank distributes data to other ranks
    for (int r = 1; r < comm->nRanks; r++) {
      int dstRank = (comm->rank + r) % comm->nRanks;
      uint8_t* srcPtr = mySendBuff + dstRank * chunkBytes;
      uint8_t* dstPtr = isInPlace ? srcPtr : myRecvBuff;

      offset = dstPtr - (uint8_t*)args->recvWin->userPtr;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, dstRank, &peerDstPtr), ret, fail);
      batchOpsParams.targets[batchOpsParams.numOps] = dstRank;
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerDstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }
  }
  // Non-root ranks don't need to perform any copy operations

  // Launch the batch operations
  NCCLCHECKGOTO(ncclCeLaunchBatchOpsImpl(comm, args, &batchOpsParams, stream), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCeGather(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // Calculate the size of data each rank sends to root
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  int rootRank = args->rootRank;
  void* peerRecvBuff;
  size_t offset;

  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, 1), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

  if (comm->rank == rootRank) {
    // Root rank copies its own data to the correct position in receive buffer
    uint8_t* dstPtr = myRecvBuff + comm->rank * chunkBytes;
    if (mySendBuff != dstPtr) {
      batchOpsParams.targets[batchOpsParams.numOps] = comm->rank;
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)dstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }
  } else {
    // Non-root ranks send their data to root's receive buffer
    uint8_t* rootRecvPtr = (uint8_t*)args->recvBuff + comm->rank * chunkBytes;
    offset = rootRecvPtr - (uint8_t*)args->recvWin->userPtr;
    NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, rootRank, &peerRecvBuff), ret, fail);
    batchOpsParams.targets[batchOpsParams.numOps] = rootRank;
    batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
    batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerRecvBuff;
    batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
    batchOpsParams.numOps++;
  }

  // Launch the batch operations
  NCCLCHECKGOTO(ncclCeLaunchBatchOpsImpl(comm, args, &batchOpsParams, stream), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

extern void* localReduceFuncs[];

// Shared scatter-reduce pipeline: DMA scatter chunks to staging buffers,
// then GPU ceLocalReduce reduces them into myRecvBuff.
// Caller must fully populate ctx (partition, myRecvBuff, etc.) before calling.
// On return, ctx->batchOpsParams is initialized for optional use by ceReduceBroadcast.
static ncclResult_t ceReduceScatterImpl(ncclComm_t comm, struct ncclCeCollArgs* args,
                                        hggcStream_t stream, struct pcclCeReduceCtx* ctx) {
  ncclResult_t ret = ncclSuccess;
  hggcEvent_t event = nullptr;

  ssize_t splitElts = pcclCeReduceChunkSize / args->eltSize;
  uint8_t* myStaBuff = (uint8_t*)ncclSymDevBase_getStaBuf((struct ncclSymDevBase*)comm->ceColl.ceSyncBase, comm->nRanks);

  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&ctx->batchOpsParams, comm->nRanks), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

  HGGCCHECKGOTO(hggcEventCreateWithFlags(&event, hggcEventDisableTiming), ret, fail);
  HGGCCHECKGOTO(hggcEventRecord(event, stream), ret, fail);
  for (int i = 0; i < comm->nRanks * ppuDiesNum; i++) {
    HGGCCHECKGOTO(hggcStreamWaitEvent(comm->ceStreams[i], event), ret, fail);
  }

  {
    uint32_t maxBlockNum = pcclCeReduceMaxBlocks;
    constexpr uint32_t maxThreadNum = pcclCeReduceMaxThreads;
    uint32_t myRank = comm->rank;
    // every rank should have the same blockNum, even they may process different elts.
    uint32_t blockNum = DIVUP(ctx->rankElts[0], (maxThreadNum * bulkBytesPerThread / args->eltSize));
    blockNum = nextPow2(blockNum);
    blockNum = std::min(blockNum, maxBlockNum);
    blockNum = std::max(blockNum, 1U);
    uint32_t threadNum = maxThreadNum;

    uint64_t* myTailPtr = ctx->tailBasePtr + comm->rank * PCCL_CE_MAX_DIE_NUM;
    void* myTailOnRank0Ptr = nullptr;
    NCCLCHECKGOTO(ceGetSymPeerPtr(comm, comm->ceColl.ceSyncWin, myTailPtr, 0, &myTailOnRank0Ptr), ret, fail);

    uint64_t* myHeadPtr = ctx->headBasePtr + comm->rank * PCCL_CE_MAX_DIE_NUM;
    void* myHeadOnRank0Ptr = nullptr;
    NCCLCHECKGOTO(ceGetSymPeerPtr(comm, comm->ceColl.ceSyncWin, myHeadPtr, 0, &myHeadOnRank0Ptr), ret, fail);

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

    struct dispatchStrategyExAD launchConfig = {1, 1, 1, -2, /*cooperative=*/1};
    dim3 grid = {blockNum, 1, 1};
    dim3 block = {threadNum, 1, 1};
    void* dummyPtr = nullptr;
    uint64_t dummy = 0;
    void* pargs[] = {&myStaBuff, &ctx->myRecvBuff, &ctx->currentRankElts, &myRank,
                     &blockNum, &ppuDiesNum, &myTailOnRank0Ptr, &myHeadOnRank0Ptr,
                     &ctx->rank0LoopCnt, &stride4G, &gridSyncPtr, &baseStep, &dummyPtr, &dummy};
    LAUNCH_KERNEL(func, grid, block, pargs, (ssize_t)0, stream, 1, launchConfig);
  }

  // DMA scatter: send each rank's chunk to the corresponding peer's staging buffer
  // Use local copy to avoid mutating ctx->rankElts (which may be needed by caller)
  ssize_t rankEltsRemain[8];
  for (int i = 0; i < comm->nRanks; i++) rankEltsRemain[i] = ctx->rankElts[i];

  for (uint32_t iter = 0; iter < ctx->rank0LoopCnt; ++iter) {
    ctx->batchOpsParams.numOps = 0;
    int32_t sect = iter % 2;

    for (int r = 0; r < comm->nRanks; r++) {
      int targetRank = (comm->rank + r + 1) % comm->nRanks;
      uint8_t* peerStaBuff = nullptr;

      ssize_t perChunkSize = 0;
      if (rankEltsRemain[targetRank] > splitElts) perChunkSize = splitElts * args->eltSize;
      else perChunkSize = rankEltsRemain[targetRank] * args->eltSize;
      rankEltsRemain[targetRank] -= splitElts;

      uint8_t* mySendBuff = (uint8_t*)args->sendBuff + ctx->rankOffset[targetRank] +
                             iter * splitElts * args->eltSize;

      NCCLCHECKGOTO(ceGetSymPeerPtr(comm, comm->ceColl.ceSyncWin, myStaBuff, targetRank, (void**)&peerStaBuff), ret, fail);
      uint8_t* peerStaDst = peerStaBuff + comm->rank * pcclSymMidBufferPerRank +
                             sect * splitElts * args->eltSize;

      ctx->batchOpsParams.targets[ctx->batchOpsParams.numOps] = targetRank;
      ctx->batchOpsParams.srcs[ctx->batchOpsParams.numOps] = (void*)mySendBuff;
      ctx->batchOpsParams.dsts[ctx->batchOpsParams.numOps] = (void*)peerStaDst;
      ctx->batchOpsParams.sizes[ctx->batchOpsParams.numOps] = perChunkSize > 0 ? perChunkSize : 0;
      ctx->batchOpsParams.numOps++;
    }

    NCCLCHECKGOTO(ncclCeLaunchBatchOpsAsync(comm, &ctx->batchOpsParams, stream,
                  comm->ceReduceStep + iter, ctx->tailBasePtr, ctx->headBasePtr), ret, fail);
  }

exit:
  if (event) hggcEventDestroy(event);
  return ret;
fail:
  goto exit;
}

// Broadcast phase: copy this rank's reduced slice to all peers' recvBuff.
// Only called by AllReduce after ceReduceScatterImpl.
// When sync=false, caller is responsible for the final barrier.
static ncclResult_t ceReduceBroadcast(ncclComm_t comm, struct ncclCeCollArgs* args,
                                      hggcStream_t stream, struct pcclCeReduceCtx* ctx,
                                      bool sync = true) {
  ncclResult_t ret = ncclSuccess;
  ssize_t sliceOffset = 0;

  ctx->batchOpsParams.numOps = 0;
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

  sliceOffset = ctx->myRecvBuff - (uint8_t*)args->recvBuff;

  for (int r = 1; r < comm->nRanks; r++) {
    int targetRank = (comm->rank + r) % comm->nRanks;
    void* peerRecvBase = nullptr;
    NCCLCHECKGOTO(ceGetSymPeerPtr(comm, args->recvWin, args->recvBuff, targetRank, &peerRecvBase), ret, fail);
    uint8_t* peerRecvBuff = (uint8_t*)peerRecvBase + sliceOffset;
    ctx->batchOpsParams.targets[ctx->batchOpsParams.numOps] = targetRank;
    ctx->batchOpsParams.srcs[ctx->batchOpsParams.numOps] = (void*)ctx->myRecvBuff;
    ctx->batchOpsParams.dsts[ctx->batchOpsParams.numOps] = (void*)peerRecvBuff;
    ctx->batchOpsParams.sizes[ctx->batchOpsParams.numOps] = ctx->currentRankElts * args->eltSize;
    ctx->batchOpsParams.numOps++;
  }

  NCCLCHECKGOTO(ncclCeLaunchBatchOpsImpl(comm, args, &ctx->batchOpsParams, stream), ret, fail);
  if (sync) {
    NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);
  }

exit:
  return ret;
fail:
  goto exit;
}

// Initialize partition context: compute per-rank element counts, byte offsets,
// loop count, and sync pointers. Caller must set ctx->myRecvBuff after this call.
static void ceReduceCtxInit(struct pcclCeReduceCtx* ctx, ncclComm_t comm,
                            ssize_t totalElts, size_t eltSize,
                            size_t recvCount = 0) {
  ssize_t eltsPerRank = DIVUP(totalElts, comm->nRanks);
  eltsPerRank = ROUNDUP(eltsPerRank, 512 / eltSize);

  ctx->totalElts = totalElts;

  // For ReduceScatter: each rank's chunk is exactly recvcount elements (no rounding).
  // For AllReduce: use rounded-up eltsPerRank for kernel alignment.
  ssize_t chunkSize = (recvCount > 0) ? (ssize_t)recvCount : eltsPerRank;

  ctx->rankElts[0] = totalElts > chunkSize ? chunkSize : totalElts;
  ctx->rankOffset[0] = 0;
  for (int i = 1; i < comm->nRanks; i++) {
    int64_t remain = ((int64_t)totalElts - ((int64_t)i + 1) * chunkSize);
    ctx->rankElts[i] = (remain >= 0) ? chunkSize : std::max<ssize_t>(0, totalElts - i * chunkSize);
    ctx->rankOffset[i] = ctx->rankOffset[i - 1] + ctx->rankElts[i - 1] * eltSize;
  }
  ctx->currentRankElts = ctx->rankElts[comm->rank];

  ssize_t splitElts = pcclCeReduceChunkSize / eltSize;
  ctx->rank0LoopCnt = DIVUP(ctx->rankElts[0], splitElts);

  ctx->tailBasePtr = (uint64_t*)ncclSymDevBase_getCeTailBase((struct ncclSymDevBase*)comm->ceColl.ceSyncBase, comm->nRanks);
  ctx->headBasePtr = (uint64_t*)ncclSymDevBase_getCeHeadBase((struct ncclSymDevBase*)comm->ceColl.ceSyncBase, comm->nRanks);
}

// INFO: not all oop can be processed without buffer easily without staBuff;
//       although it can be done. but need careful handle for scenario as below:
//       totalElts 14 elts for 4 ranks. we can use rank 0 to handle all 14 elts but
//       rank0's recvbuff can only hold 14 elts from one rank.
ncclResult_t pcclCeAllReduce(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  struct pcclCeReduceCtx ctx = {};
  ceReduceCtxInit(&ctx, comm, (ssize_t)args->nElts, args->eltSize);
  // AllReduce: recvBuff has nElts elements, this rank's slice starts at rankOffset[rank]
  ctx.myRecvBuff = (uint8_t*)args->recvBuff + ctx.rankOffset[comm->rank];

  NCCLCHECKGOTO(ceReduceScatterImpl(comm, args, stream, &ctx), ret, fail);
  NCCLCHECKGOTO(ceReduceBroadcast(comm, args, stream, &ctx, /*sync=*/false), ret, fail);
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);
  comm->ceReduceStep += ctx.rank0LoopCnt;

exit:
  NCCLCHECK(ncclCeFreeBatchOpsParams(&ctx.batchOpsParams));
  return ret;
fail:
  goto exit;
}

ncclResult_t pcclCeReduceScatter(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  struct pcclCeReduceCtx ctx = {};
  // For ReduceScatter: args->nElts = recvcount (per-rank count).
  // totalElts = recvcount * nRanks (full sendBuff size in elements).
  ssize_t totalElts = (ssize_t)args->nElts * comm->nRanks;
  ceReduceCtxInit(&ctx, comm, totalElts, args->eltSize, args->nElts);
  // ReduceScatter: recvBuff IS this rank's reduced slice (no offset)
  ctx.myRecvBuff = (uint8_t*)args->recvBuff;

  NCCLCHECKGOTO(ceReduceScatterImpl(comm, args, stream, &ctx), ret, fail);
  // No broadcast phase — each rank already has its reduced slice in recvBuff.
  // Final sync to ensure all ranks complete before returning.
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);
  comm->ceReduceStep += ctx.rank0LoopCnt;

exit:
  NCCLCHECK(ncclCeFreeBatchOpsParams(&ctx.batchOpsParams));
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclLaunchCeColl(ncclComm_t comm, struct ncclKernelPlan* plan) {
  ncclResult_t ret = ncclSuccess;
  hggcStream_t stream = comm->planner.streams->stream;
  struct ncclCeCollArgs* args = &plan->ceCollArgs;
  // Same aggregation granularity as ncclLaunchKernel: one hgtx range per group.
  struct pcclHgtxInfo hgtxCeCollInfo = pcclHgtxInfo(comm->hgtxMessage);

  // Start CE collective profiling
  NCCLCHECKGOTO(ncclProfilerStartCeCollEvent(comm, args, stream), ret, fail);

  switch (args->func) {
    case ncclFuncAllGather:
#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
      if (comm->okiSupport && !plan->persistent) {
        // INFO: oki do not support graph for now.
        NCCLCHECKGOTO(pcclCeOkiAllGather(comm, args, stream), ret, fail);
        break;
      }
#endif
      NCCLCHECKGOTO(ncclCeAllGather(comm, args, stream), ret, fail);
      break;
    case ncclFuncAlltoAll:
      NCCLCHECKGOTO(ncclCeAlltoAll(comm, args, stream), ret, fail);
      break;
    case ncclFuncScatter:
      NCCLCHECKGOTO(ncclCeScatter(comm, args, stream), ret, fail);
      break;
    case ncclFuncGather:
      NCCLCHECKGOTO(ncclCeGather(comm, args, stream), ret, fail);
      break;
    case ncclFuncAllReduce:
#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
      if (comm->okiSupport && !plan->persistent) {
        // INFO: oki do not support graph for now.
        NCCLCHECKGOTO(pcclCeOkiAllReduce(comm, args, stream), ret, fail);
        break;
      }
#endif
      NCCLCHECKGOTO(pcclCeAllReduce(comm, args, stream), ret, fail);
      break;
    case ncclFuncReduceScatter:
      NCCLCHECKGOTO(pcclCeReduceScatter(comm, args, stream), ret, fail);
      break;
    default:
      ret = ncclInvalidUsage;
  }

exit:
  // Stop CE collective profiling - always attempt if started, even on error
  ncclProfilerStopCeCollEvent(comm, args, stream);
  return ret;
fail:
  goto exit;
}
