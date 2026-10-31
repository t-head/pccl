/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "pccl.h"
#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "param.h"
#include "dev_runtime.h"
#include "rma/rma.h"
#include "rma/rma_ce.h"

ncclResult_t ncclRmaCeInit(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  uint64_t* signalsDevBase = nullptr;
  uint64_t* ackInitHost = nullptr;
  size_t signalSlots = 0;
  size_t depth = NCCL_RMA_CE_GRAPH_DEPTH;
  size_t graphSlots = 0;
  struct ncclRmaCeState* rmaCeState = &comm->rmaState.rmaCeState;

  if (rmaCeState->initialized || rmaCeState->initInProgress) return ncclSuccess;
  rmaCeState->initInProgress = true;

  // Ensure symmetric memory runtime is initialized
  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);

  rmaCeState->rmaCeCtxCount = comm->config.numRmaCtx;

  // CE only targets intra-node (LSA) peers, so signals are indexed by LSA-local rank.
  signalSlots = (size_t)comm->devrState.lsaSize * comm->config.numRmaSig;
  graphSlots = depth * signalSlots;
  NCCLCHECKGOTO(ncclCalloc(&ackInitHost, graphSlots), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&rmaCeState->rmaCeCtxs, rmaCeState->rmaCeCtxCount), ret, fail);
  for (int i = 0; i < rmaCeState->rmaCeCtxCount; i++) {
    // Allocate the RMA CE context
    struct ncclRmaCeCtx* ceCtx;
    NCCLCHECKGOTO(ncclCalloc(&ceCtx, 1), ret, fail);
    rmaCeState->rmaCeCtxs[i] = ceCtx;

    // Initialize context
    ceCtx->comm = comm;

    // Allocate and register symmetric memory for signals
    size_t signalsBufSize = (1 + 2 * depth) * signalSlots * sizeof(uint64_t);
    ncclWindow_vidmem* signalsWinDev;
    ncclWindow_vidmem* signalsWinDevHost;

    NCCLCHECKGOTO(pcclMemAlloc((void**)&signalsDevBase, signalsBufSize), ret, fail);
    HGGCCHECKGOTO(hggcMemset(signalsDevBase, 0, signalsBufSize), ret, fail);
    NCCLCHECKGOTO(ncclDevrWindowRegisterInGroup(comm, signalsDevBase, signalsBufSize, NCCL_WIN_COLL_SYMMETRIC,
                                                &signalsWinDev),
                  ret, fail);
    NCCLCHECKGOTO(ncclShadowPoolToHost(&comm->devrState.shadows, signalsWinDev, &signalsWinDevHost), ret, fail);

    // Get the ncclDevrWindow from the winHost field
    ceCtx->signalsWin = (struct ncclDevrWindow*)signalsWinDevHost->winHost;
    ceCtx->signalsDev = signalsDevBase;
    ceCtx->graphSignalsDev = signalsDevBase + signalSlots;
    ceCtx->graphAckDev = signalsDevBase + (1 + depth) * signalSlots;
    signalsDevBase = nullptr;
    ceCtx->signalOffset = 0;
    ceCtx->graphSignalOffset = signalSlots * sizeof(uint64_t);
    ceCtx->graphAckOffset = (1 + depth) * signalSlots * sizeof(uint64_t);

    // Initialize all graph ack flags to 1
    for (size_t r = 0; r < graphSlots; r++) ackInitHost[r] = 1;
    HGGCCHECKGOTO(hggcMemcpy(ceCtx->graphAckDev, ackInitHost, graphSlots * sizeof(uint64_t),
                             hggcMemcpyHostToDevice),
                  ret, fail);

    // Allocate device-resident constants for graph-safe D2D signal/ack writes
    NCCLCHECKGOTO(pcclMemAlloc((void**)&ceCtx->signalConstDev, 2 * sizeof(uint64_t)), ret, fail);
    ceCtx->signalConstZeroDev = &ceCtx->signalConstDev[0];
    ceCtx->signalConstOneDev = &ceCtx->signalConstDev[1];
    {
      uint64_t zeroone[] = {0, 1};
      HGGCCHECKGOTO(hggcMemcpy(ceCtx->signalConstDev, zeroone, 2 * sizeof(uint64_t), hggcMemcpyHostToDevice),
                    ret, fail);
    }

    // Allocate host buffer to track expected non-graph signal values
    NCCLCHECKGOTO(ncclCalloc(&ceCtx->signalsHost, signalSlots), ret, fail);

    // Allocate host per-(sigIdx, rank) sequence counters.
    NCCLCHECKGOTO(ncclCalloc(&ceCtx->signalOpSeqs, signalSlots), ret, fail);
    NCCLCHECKGOTO(pcclMemAlloc((void**)&ceCtx->signalOpSeqsDev, comm->nRanks * sizeof(uint64_t)), ret, fail);

    NCCLCHECKGOTO(ncclCalloc(&ceCtx->graphPutCount, signalSlots), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&ceCtx->graphWaitCount, signalSlots), ret, fail);
  }

  INFO(NCCL_INIT, "Rank %d: finished init RMA CE contexts, numRmaCeCtxs %d", comm->rank, comm->config.numRmaCtx);

  // Create CE stream for parallel execution
  HGGCCHECKGOTO(hggcStreamCreateWithFlags(&rmaCeState->ceStream, hggcStreamNonBlocking), ret, fail);

  // Create event for synchronization
  HGGCCHECKGOTO(hggcEventCreateWithFlags(&rmaCeState->ceEvent, hggcEventDisableTiming), ret, fail);

  rmaCeState->initialized = true;

exit:
  free(ackInitHost);
  rmaCeState->initInProgress = false;
  return ret;
fail:
  if (signalsDevBase) (void)pcclMemFree(signalsDevBase);
  goto exit;
}

ncclResult_t ncclRmaCeFinalize(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclRmaCeState* rmaCeState = &comm->rmaState.rmaCeState;

  // Destroy CE stream and event
  if (rmaCeState->ceStream != NULL) {
    HGGCCHECKGOTO(hggcStreamDestroy(rmaCeState->ceStream), ret, fail);
    rmaCeState->ceStream = NULL;
  }

  if (rmaCeState->ceEvent != NULL) {
    HGGCCHECKGOTO(hggcEventDestroy(rmaCeState->ceEvent), ret, fail);
    rmaCeState->ceEvent = NULL;
  }

  for (int i = 0; rmaCeState->rmaCeCtxs != NULL && i < rmaCeState->rmaCeCtxCount; i++) {
    struct ncclRmaCeCtx* ceCtx = (struct ncclRmaCeCtx*)rmaCeState->rmaCeCtxs[i];
    if (ceCtx == NULL) continue;

    // Free per-rank operation sequence counters
    if (ceCtx->signalOpSeqs) free(ceCtx->signalOpSeqs);
    if (ceCtx->signalOpSeqsDev) NCCLCHECKGOTO(pcclMemFree(ceCtx->signalOpSeqsDev), ret, fail);

    // Free capture-time counters
    if (ceCtx->graphPutCount) free(ceCtx->graphPutCount);
    if (ceCtx->graphWaitCount) free(ceCtx->graphWaitCount);

    // Free host signals buffer
    if (ceCtx->signalsHost) free(ceCtx->signalsHost);

    // Free device-resident constants
    if (ceCtx->signalConstDev) NCCLCHECKGOTO(pcclMemFree(ceCtx->signalConstDev), ret, fail);

    // Deregister and free signal window
    if (ceCtx->signalsWin) NCCLCHECKGOTO(ncclCommWindowDeregister(comm, ceCtx->signalsWin->vidmem), ret, fail);

    // Free signal device memory
    if (ceCtx->signalsDev) NCCLCHECKGOTO(pcclMemFree(ceCtx->signalsDev), ret, fail);

    // Free the context itself
    free(ceCtx);
    rmaCeState->rmaCeCtxs[i] = NULL;
  }

  // Reset the number of contexts and initialized flag
  rmaCeState->rmaCeCtxCount = 0;
  rmaCeState->initialized = false;

  free(rmaCeState->rmaCeCtxs);
  rmaCeState->rmaCeCtxs = NULL;

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t ncclRmaCeGraphSession(hggcStream_t stream, struct ncclRmaCeCtx* ceCtx, size_t signalSlots) {
  hggcStreamCaptureStatus status = hggcStreamCaptureStatusNone;
  hggcGraph_t graph = NULL;

  HGGCCHECK(hggcStreamGetCaptureInfo_v2(stream, &status, NULL, &graph, NULL, NULL));
  if (status == hggcStreamCaptureStatusNone || graph == ceCtx->graphCaptureId) return ncclSuccess;

  memset(ceCtx->graphPutCount, 0, signalSlots * sizeof(uint64_t));
  memset(ceCtx->graphWaitCount, 0, signalSlots * sizeof(uint64_t));
  ceCtx->graphCaptureId = graph;
  return ncclSuccess;
}

static ncclResult_t ncclRmaCePutLaunchPersist(ncclComm_t comm, struct ncclKernelPlan* plan, hggcStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  int nRmaTasksCe = plan->rmaArgs->nRmaTasksCe;

  int lsaSize = comm->devrState.lsaSize;
  int lsaSelf = comm->devrState.lsaSelf;

  for (int i = 0; i < nRmaTasksCe; i++) {
    struct ncclTaskRma* task = ncclIntruQueueHead(&plan->rmaTaskQueueCe);
    ncclIntruQueueDequeue(&plan->rmaTaskQueueCe);
    // Per-task context: a put-with-signal on ctx c raises the peer's ctx-c signal.
    struct ncclRmaCeCtx* ceCtx = (struct ncclRmaCeCtx*)comm->rmaState.rmaCeState.rmaCeCtxs[task->ctx];

    int peerLsaRank;
    NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, task->peer, &peerLsaRank), ret, fail);

    size_t bytes = task->count * ncclTypeSize(task->datatype);

    size_t graphSet = 0;
    if (task->signalMode != NCCL_SIGNAL_NONE) {
      size_t signalSlots = (size_t)lsaSize * comm->config.numRmaSig;
      NCCLCHECKGOTO(ncclRmaCeGraphSession(stream, ceCtx, signalSlots), ret, fail);
      size_t signalSlot = ncclRmaSignalSlot(lsaSize, task->signalIdx, peerLsaRank);
      uint64_t opIdx = ceCtx->graphPutCount[signalSlot];
      ceCtx->graphPutCount[signalSlot] = opIdx + 1;
      graphSet = (size_t)(opIdx % NCCL_RMA_CE_GRAPH_DEPTH);

      uint64_t* ackAddr = &ceCtx->graphAckDev[graphSet * signalSlots + signalSlot];
      HGCHECKGOTO(hgStreamWaitValue64(stream, (HGdeviceptr)ackAddr, 1, HG_STREAM_WAIT_VALUE_GEQ), ret, fail);
      HGCHECKGOTO(hgStreamWriteValue64(stream, (HGdeviceptr)ackAddr, 0, HG_STREAM_WRITE_VALUE_DEFAULT), ret, fail);
    }

    // Data movement; the signal write below orders after it on the stream.
    if (bytes > 0) {
      void* peerBuff;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, task->peerWinHost, task->peerWinOffset, peerLsaRank, &peerBuff), ret,
                    fail);
      if (peerBuff == NULL) {
        WARN("RMA CE: peerBuff is NULL after ncclDevrGetLsaRankPtr");
        ret = ncclInvalidArgument;
        goto fail;
      }
      HGGCCHECKGOTO(hggcMemcpyAsync(peerBuff, (void*)task->srcBuff, bytes, hggcMemcpyDeviceToDevice, stream), ret,
                    fail);
    }

    // Graph: raise the signal in the same slot set on the peer's side
    if (task->signalMode != NCCL_SIGNAL_NONE) {
      size_t signalSlots = (size_t)lsaSize * comm->config.numRmaSig;
      size_t writeSlot = graphSet * signalSlots + ncclRmaSignalSlot(lsaSize, task->signalIdx, lsaSelf);
      void* peerGraphSignal;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceCtx->signalsWin,
                                          ceCtx->graphSignalOffset + writeSlot * sizeof(uint64_t), peerLsaRank,
                                          &peerGraphSignal),
                    ret, fail);
      HGGCCHECKGOTO(hggcMemcpyAsync(peerGraphSignal, ceCtx->signalConstOneDev, sizeof(uint64_t),
                                    hggcMemcpyDeviceToDevice, stream),
                    ret, fail);
    }

    // Free the task after processing
    ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t ncclRmaCePutLaunchNonPersist(ncclComm_t comm, struct ncclKernelPlan* plan, hggcStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  int nRmaTasksCe = plan->rmaArgs->nRmaTasksCe;

  int lsaSize = comm->devrState.lsaSize;
  int lsaSelf = comm->devrState.lsaSelf;

  // We can not have signal operations to the same physical address in the same
  // batch as batched mem copy does not guarantee order of execution, so signal
  // staging slots are used at most once per round.
  struct ncclIntruQueue<struct ncclTaskRma, &ncclTaskRma::next>* peerTaskQueues = nullptr;
  int* activePeers = nullptr;
  struct ncclTaskRma* currentTask = nullptr;
  int nActivePeers = 0;

  if (nRmaTasksCe == 0) goto exit;

  NCCLCHECKGOTO(ncclCalloc(&peerTaskQueues, comm->nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&activePeers, comm->nRanks), ret, fail);

  for (int i = 0; i < nRmaTasksCe; i++) {
    struct ncclTaskRma* task = ncclIntruQueueDequeue(&plan->rmaTaskQueueCe);
    int peer = task->peer;
    if (ncclIntruQueueEmpty(&peerTaskQueues[peer])) {
      activePeers[nActivePeers++] = peer;
    }
    ncclIntruQueueEnqueue(&peerTaskQueues[peer], task);
  }

  while (nActivePeers > 0) {
    int nNextActivePeers = 0;
    int nSeqStageOps = 0;

    for (int i = 0; i < nActivePeers; i++) {
      int peer = activePeers[i];
      currentTask = ncclIntruQueueDequeue(&peerTaskQueues[peer]);
      // Per-task context: each task's signal goes to the peer's own-context buffer.
      struct ncclRmaCeCtx* ceCtx = (struct ncclRmaCeCtx*)comm->rmaState.rmaCeState.rmaCeCtxs[currentTask->ctx];

      int peerLsaRank;
      NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, currentTask->peer, &peerLsaRank), ret, fail);

      size_t bytes = currentTask->count * ncclTypeSize(currentTask->datatype);

      // Data movement
      if (bytes > 0) {
        void* peerBuff;
        NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, currentTask->peerWinHost, currentTask->peerWinOffset, peerLsaRank,
                                            &peerBuff),
                      ret, fail);
        if (peerBuff == NULL) {
          WARN("RMA CE: peerBuff is NULL after ncclDevrGetLsaRankPtr");
          ret = ncclInvalidArgument;
          goto fail;
        }
        HGGCCHECKGOTO(hggcMemcpyAsync(peerBuff, (void*)currentTask->srcBuff, bytes, hggcMemcpyDeviceToDevice, stream),
                      ret, fail);
      }

      // Non-graph: write incrementing sequence to peer's signalsDev.
      // Each round has at most one task per peer, so each staging slot is
      // written at most once per round.
      if (currentTask->signalMode != NCCL_SIGNAL_NONE) {
        if (nSeqStageOps >= comm->nRanks) {
          WARN("RMA CE: staged signal ops (%d) exceed staging capacity (%d)", nSeqStageOps, comm->nRanks);
          ret = ncclInternalError;
          goto fail;
        }
        size_t signalOffset = ncclRmaSignalOffset(lsaSize, currentTask->signalIdx, lsaSelf);
        size_t signalSlot = ncclRmaSignalSlot(lsaSize, currentTask->signalIdx, peerLsaRank);
        void* peerSignal;
        NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceCtx->signalsWin, ceCtx->signalOffset + signalOffset, peerLsaRank,
                                            &peerSignal),
                      ret, fail);
        ceCtx->signalOpSeqs[signalSlot]++;

        // Stage the sequence value through a device slot with a memop, then copy
        // it to the peer's signal slot; stream order serializes write before copy.
        HGCHECKGOTO(hgStreamWriteValue64(stream, (HGdeviceptr)&ceCtx->signalOpSeqsDev[nSeqStageOps],
                                          ceCtx->signalOpSeqs[signalSlot], HG_STREAM_WRITE_VALUE_DEFAULT),
                    ret, fail);
        HGGCCHECKGOTO(hggcMemcpyAsync(peerSignal, &ceCtx->signalOpSeqsDev[nSeqStageOps], sizeof(uint64_t),
                                      hggcMemcpyDeviceToDevice, stream),
                      ret, fail);

        nSeqStageOps++;
      }

      ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, currentTask);
      currentTask = nullptr;

      if (!ncclIntruQueueEmpty(&peerTaskQueues[peer])) {
        activePeers[nNextActivePeers++] = peer;
      }
    }
    nActivePeers = nNextActivePeers;
  }

exit:
  if (currentTask != nullptr) ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, currentTask);
  if (peerTaskQueues != nullptr) {
    for (int peer = 0; peer < comm->nRanks; peer++) {
      while (!ncclIntruQueueEmpty(&peerTaskQueues[peer])) {
        struct ncclTaskRma* task = ncclIntruQueueDequeue(&peerTaskQueues[peer]);
        ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);
      }
    }
  }
  free(peerTaskQueues);
  free(activePeers);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclRmaCePutLaunch(ncclComm_t comm, struct ncclKernelPlan* plan, hggcStream_t stream) {
  if (!comm->rmaState.rmaCeState.initialized) {
    WARN("RMA CE is not initialized");
    return ncclInternalError;
  }

  if (plan->persistent) {
    return ncclRmaCePutLaunchPersist(comm, plan, stream);
  }
  return ncclRmaCePutLaunchNonPersist(comm, plan, stream);
}

ncclResult_t ncclRmaCeWaitLaunch(ncclComm_t comm, struct ncclKernelPlan* plan, hggcStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // Make sure the RMA CE is initialized
  if (!comm->rmaState.rmaCeState.initialized) {
    WARN("RMA CE is not initialized");
    return ncclInternalError;
  }

  bool persistent = plan->persistent;

  int lsaSize = comm->devrState.lsaSize;
  int lsaSelf = comm->devrState.lsaSelf;

  struct ncclTaskRma* task = ncclIntruQueueHead(&plan->rmaTaskQueueCe);
  ncclIntruQueueDequeue(&plan->rmaTaskQueueCe);

  // A WaitSignal plan is single-context; use the wait task's own context.
  // (Declared before the goto-based checks so they don't cross its initialization.)
  struct ncclRmaCeCtx* ceCtx = (struct ncclRmaCeCtx*)comm->rmaState.rmaCeState.rmaCeCtxs[task->ctx];

  if (task->func != ncclFuncWaitSignal) {
    WARN("RMA CE task function is %d, expected %d", task->func, ncclFuncWaitSignal);
    goto invalid_task;
  }
  if (plan->rmaArgs->nRmaTasksCe != 1) {
    WARN("RMA CE task count is %d, expected 1", plan->rmaArgs->nRmaTasksCe);
    goto invalid_task;
  }

  if (task->signalMode == NCCL_SIGNAL) {
    if (!persistent) {
      // Non-graph: one wait memop per peer with incrementing expected value.
      for (int i = 0; i < task->npeers; i++) {
        int peerRank = task->peers[i];
        int peerLsaRank;
        NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, peerRank, &peerLsaRank), ret, fail);
        size_t signalSlot = ncclRmaSignalSlot(lsaSize, task->signalIdxs[i], peerLsaRank);
        uint64_t waitValue = ceCtx->signalsHost[signalSlot] + task->nsignals[i];
        ceCtx->signalsHost[signalSlot] = waitValue;

        uint64_t* signalAddr = &ceCtx->signalsDev[signalSlot];
        HGCHECKGOTO(hgStreamWaitValue64(stream, (HGdeviceptr)signalAddr, waitValue, HG_STREAM_WAIT_VALUE_GEQ), ret,
                    fail);
      }
    } else {
      size_t signalSlots = (size_t)lsaSize * comm->config.numRmaSig;
      NCCLCHECKGOTO(ncclRmaCeGraphSession(stream, ceCtx, signalSlots), ret, fail);
      for (int i = 0; i < task->npeers; i++) {
        int peerRank = task->peers[i];
        int peerLsaRank;
        NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, peerRank, &peerLsaRank), ret, fail);

        for (int s = 0; s < task->nsignals[i]; s++) {
          size_t signalSlot = ncclRmaSignalSlot(lsaSize, task->signalIdxs[i], peerLsaRank);
          uint64_t opIdx = ceCtx->graphWaitCount[signalSlot];
          ceCtx->graphWaitCount[signalSlot] = opIdx + 1;
          size_t graphSet = (size_t)(opIdx % NCCL_RMA_CE_GRAPH_DEPTH);

          uint64_t* graphSignalAddr = &ceCtx->graphSignalsDev[graphSet * signalSlots + signalSlot];
          HGCHECKGOTO(hgStreamWaitValue64(stream, (HGdeviceptr)graphSignalAddr, 1, HG_STREAM_WAIT_VALUE_GEQ), ret,
                      fail);
          HGCHECKGOTO(hgStreamWriteValue64(stream, (HGdeviceptr)graphSignalAddr, 0, HG_STREAM_WRITE_VALUE_DEFAULT),
                      ret, fail);

          void* peerAck;
          size_t ackSlot = graphSet * signalSlots + ncclRmaSignalSlot(lsaSize, task->signalIdxs[i], lsaSelf);
          NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceCtx->signalsWin,
                                              ceCtx->graphAckOffset + ackSlot * sizeof(uint64_t), peerLsaRank,
                                              &peerAck),
                        ret, fail);
          HGGCCHECKGOTO(hggcMemcpyAsync(peerAck, ceCtx->signalConstOneDev, sizeof(uint64_t),
                                        hggcMemcpyDeviceToDevice, stream),
                        ret, fail);
        }
      }
    }
  }

  // Free the task
  ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);

exit:
  return ret;
invalid_task:
  ret = ncclInternalError;
  ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);
fail:
  goto exit;
}
