/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

  // Ensure symmetric memory runtime is initialized
  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);

  comm->rmaState.rmaCeState.rmaCeCtxCount = comm->config.numRmaCtx;

  NCCLCHECKGOTO(ncclCalloc(&ackInitHost, comm->nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->rmaState.rmaCeState.rmaCeCtxs, comm->rmaState.rmaCeState.rmaCeCtxCount), ret, fail);
  for (int i = 0; i < comm->rmaState.rmaCeState.rmaCeCtxCount; i++) {
    // Allocate the RMA CE context
    struct ncclRmaCeCtx* ceCtx;
    NCCLCHECKGOTO(ncclCalloc(&ceCtx, 1), ret, fail);
    comm->rmaState.rmaCeState.rmaCeCtxs[i] = ceCtx;

    // Initialize context
    ceCtx->comm = comm;

    // Allocate and register symmetric memory for signals
    int nRanks = comm->nRanks;
    size_t signalsBufSize = (3 * nRanks + 2) * sizeof(uint64_t);
    ncclWindow_vidmem* signalsWinDev;
    ncclWindow_vidmem* signalsWinDevHost;

    NCCLCHECKGOTO(pcclMemAlloc((void**)&signalsDevBase, signalsBufSize), ret, fail);
    NCCLCHECKGOTO(ncclHggcMemset(signalsDevBase, 0, 3 * nRanks + 2), ret, fail);
    NCCLCHECKGOTO(ncclDevrWindowRegisterInGroup(comm, signalsDevBase, signalsBufSize, NCCL_WIN_COLL_SYMMETRIC,
                                                &signalsWinDev),
                  ret, fail);
    NCCLCHECKGOTO(ncclShadowPoolToHost(&comm->devrState.shadows, signalsWinDev, &signalsWinDevHost), ret, fail);

    // Get the ncclDevrWindow from the winHost field
    ceCtx->signalsWin = (struct ncclDevrWindow*)signalsWinDevHost->winHost;
    ceCtx->signalsDev = signalsDevBase;
    ceCtx->graphSignalsDev = signalsDevBase + nRanks + 1;
    ceCtx->graphAckDev = signalsDevBase + 2 * nRanks + 2;
    signalsDevBase = nullptr;
    ceCtx->signalOffset = 0;
    ceCtx->graphSignalOffset = (nRanks + 1) * sizeof(uint64_t);
    ceCtx->graphAckOffset = (2 * nRanks + 2) * sizeof(uint64_t);

    // Initialize ack flags to 1
    for (int r = 0; r < nRanks; r++) ackInitHost[r] = 1;
    NCCLCHECKGOTO(ncclHggcMemcpy(ceCtx->graphAckDev, ackInitHost, nRanks), ret, fail);

    // Allocate device-resident constants for graph-safe D2D signal/ack writes
    NCCLCHECKGOTO(ncclHggcCalloc(&ceCtx->signalConstDev, 2, nullptr, comm->memManager), ret, fail);
    ceCtx->signalConstZeroDev = &ceCtx->signalConstDev[0];
    ceCtx->signalConstOneDev = &ceCtx->signalConstDev[1];
    {
      uint64_t zeroone[] = {0, 1};
      NCCLCHECKGOTO(ncclHggcMemcpy(ceCtx->signalConstDev, zeroone, 2), ret, fail);
    }

    // Allocate host buffer to track expected non-graph signal values
    NCCLCHECKGOTO(ncclCalloc(&ceCtx->signalsHost, nRanks + 1), ret, fail);

    // Allocate per-rank operation sequence counters
    NCCLCHECKGOTO(ncclCalloc(&ceCtx->signalOpSeqs, comm->nRanks), ret, fail);
    NCCLCHECKGOTO(ncclHggcCalloc(&ceCtx->signalOpSeqsDev, 1, nullptr, comm->memManager), ret, fail);
  }

  INFO(NCCL_INIT, "Rank %d: finished init RMA CE contexts, numRmaCeCtxs %d", comm->rank, comm->config.numRmaCtx);

  // Create CE stream for parallel execution
  HGGCCHECKGOTO(hggcStreamCreateWithFlags(&comm->rmaState.rmaCeState.ceStream, hggcStreamNonBlocking), ret, fail);

  // Create event for synchronization
  HGGCCHECKGOTO(hggcEventCreateWithFlags(&comm->rmaState.rmaCeState.ceEvent, hggcEventDisableTiming), ret, fail);

  comm->rmaState.rmaCeState.initialized = true;

exit:
  ncclFree(ackInitHost);
  return ret;
fail:
  if (signalsDevBase) pcclMemFree(signalsDevBase);
  goto exit;
}

ncclResult_t ncclRmaCeFinalize(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclRmaCeState* rmaCeState = &comm->rmaState.rmaCeState;

  // Clean up rmaCeInitTaskQueue
  while (!ncclIntruQueueEmpty(&comm->rmaCeInitTaskQueue)) {
    struct ncclRmaCeInitTask* task = ncclIntruQueueDequeue(&comm->rmaCeInitTaskQueue);
    ncclFree(task);
  }

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
    if (ceCtx->signalOpSeqs) ncclFree(ceCtx->signalOpSeqs);
    if (ceCtx->signalOpSeqsDev)
      NCCLCHECKGOTO(ncclHggcFree(ceCtx->signalOpSeqsDev, MEM_NORMAL, comm->memManager), ret, fail);

    // Free host signals buffer
    if (ceCtx->signalsHost) ncclFree(ceCtx->signalsHost);

    // Free device-resident constants
    if (ceCtx->signalConstDev)
      NCCLCHECKGOTO(ncclHggcFree(ceCtx->signalConstDev, MEM_NORMAL, comm->memManager), ret, fail);

    // Deregister and free signal window
    if (ceCtx->signalsWin) NCCLCHECKGOTO(ncclCommWindowDeregister(comm, ceCtx->signalsWin->vidmem), ret, fail);

    // Free signal device memory
    if (ceCtx->signalsDev) NCCLCHECKGOTO(pcclMemFree(ceCtx->signalsDev), ret, fail);

    // Free the context itself
    ncclFree(ceCtx);
    rmaCeState->rmaCeCtxs[i] = NULL;
  }

  // Reset the number of contexts and initialized flag
  rmaCeState->rmaCeCtxCount = 0;
  rmaCeState->initialized = false;

  ncclFree(rmaCeState->rmaCeCtxs);
  rmaCeState->rmaCeCtxs = NULL;

exit:
  return ret;
fail:
  goto exit;
}

// Wrapper for hgStreamBatchMemOp that handles the 255 operations per call limit
static ncclResult_t ncclRmaCeBatchMemOp(hggcStream_t stream, unsigned int numOps,
                                     HGstreamBatchMemOpParams* batchParams) {
  ncclResult_t ret = ncclSuccess;
  const unsigned int maxOpsPerBatch = 255;

  for (unsigned int offset = 0; offset < numOps; offset += maxOpsPerBatch) {
    unsigned int opsInThisChunk = (numOps - offset < maxOpsPerBatch) ? (numOps - offset) : maxOpsPerBatch;
    HGCHECKGOTO(hgStreamBatchMemOp(stream, opsInThisChunk, &batchParams[offset], 0), ret, fail);
  }

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclRmaCePutLaunch(ncclComm_t comm, struct ncclKernelPlan* plan, hggcStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // Make sure the RMA CE is initialized
  if (!comm->rmaState.rmaCeState.initialized) {
    ERROR("RMA CE is not initialized");
    return ncclInternalError;
  }

  bool persistent = plan->persistent;
  int nRmaTasksCe = plan->rmaArgs->nRmaTasksCe;
  int ctx = plan->rmaArgs->ctx;
  struct ncclRmaCeCtx* ceCtx = (struct ncclRmaCeCtx*)comm->rmaState.rmaCeState.rmaCeCtxs[ctx];

  for (int i = 0; i < nRmaTasksCe; i++) {
    struct ncclTaskRma* task = ncclIntruQueueHead(&plan->rmaTaskQueueCe);
    ncclIntruQueueDequeue(&plan->rmaTaskQueueCe);

    int peerLsaRank;
    NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, task->peer, &peerLsaRank), ret, fail);

    size_t bytes = task->count * ncclTypeSize(task->datatype);

    // Graph: wait for receiver's ack, then reset ack flag.
    // Only signaling tasks consume ack; earlier Put chunks do not signal.
    if (persistent && task->signalMode != NCCL_SIGNAL_NONE) {
      HGdeviceptr ackAddr = (HGdeviceptr)&ceCtx->graphAckDev[task->peer];
      HGstreamBatchMemOpParams ackOps[2] = {};
      ackOps[0].waitValue.operation = HG_STREAM_MEM_OP_WAIT_VALUE_64;
      ackOps[0].waitValue.address = ackAddr;
      ackOps[0].waitValue.value64 = 1;
      ackOps[0].waitValue.flags = HG_STREAM_WAIT_VALUE_GEQ;
      ackOps[1].writeValue.operation = HG_STREAM_MEM_OP_WRITE_VALUE_64;
      ackOps[1].writeValue.address = ackAddr;
      ackOps[1].writeValue.value64 = 0;
      ackOps[1].writeValue.flags = HG_STREAM_WRITE_VALUE_DEFAULT;
      NCCLCHECKGOTO(ncclRmaCeBatchMemOp(stream, 2, ackOps), ret, fail);
    }

    if (bytes > 0) {
      // Get the peer buffer from the peer window
      void* peerBuff;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, task->peerWinHost, task->peerWinOffset, peerLsaRank, &peerBuff),
                    ret, fail);

      // Validate peer buffer
      if (peerBuff == NULL) {
        ERROR("RMA CE: peerBuff is NULL after ncclDevrGetLsaRankPtr");
        ret = ncclInvalidArgument;
        goto fail;
      }

      // Copy the data to the peer buffer
      HGGCCHECKGOTO(hggcMemcpyAsync(peerBuff, (void*)task->srcBuff, bytes, hggcMemcpyDeviceToDevice, stream), ret, fail);
    }

    // Write signal if needed for the target rank
    // CE over NVL only supports distinct signal
    if (task->signalMode != NCCL_SIGNAL_NONE) {
      size_t rankSlot = comm->rank * sizeof(uint64_t);

      if (!persistent) {
        // Non-graph: write incrementing sequence to peer's signalsDev
        void* peerSignal;
        NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceCtx->signalsWin, ceCtx->signalOffset + rankSlot,
                                          peerLsaRank, &peerSignal), ret, fail);
        ceCtx->signalOpSeqs[task->peer]++;
        HGCHECKGOTO(hgStreamWriteValue64(stream, (HGdeviceptr)ceCtx->signalOpSeqsDev,
                                       ceCtx->signalOpSeqs[task->peer], HG_STREAM_WRITE_VALUE_DEFAULT), ret, fail);
        HGGCCHECKGOTO(hggcMemcpyAsync(peerSignal, ceCtx->signalOpSeqsDev, sizeof(uint64_t),
                                    hggcMemcpyDeviceToDevice, stream), ret, fail);
      } else {
        // Graph: write signal=1 to peer's graphSignalsDev (separate from non-graph signals)
        void* peerGraphSignal;
        NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceCtx->signalsWin, ceCtx->graphSignalOffset + rankSlot,
                                          peerLsaRank, &peerGraphSignal), ret, fail);
        HGGCCHECKGOTO(hggcMemcpyAsync(peerGraphSignal, ceCtx->signalConstOneDev, sizeof(uint64_t),
                                    hggcMemcpyDeviceToDevice, stream), ret, fail);
      }
    }

    // Free the task after processing
    ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);
  }

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclRmaCeWaitLaunch(ncclComm_t comm, struct ncclKernelPlan* plan, hggcStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  HGstreamBatchMemOpParams* batchParams = nullptr;

  // Make sure the RMA CE is initialized
  if (!comm->rmaState.rmaCeState.initialized) {
    ERROR("RMA CE is not initialized");
    return ncclInternalError;
  }

  bool persistent = plan->persistent;
  int ctx = plan->rmaArgs->ctx;
  struct ncclTaskRma* task = ncclIntruQueueHead(&plan->rmaTaskQueueCe);
  if (task == nullptr || task->func != ncclFuncWaitSignal ||
      task->ctx != ctx || plan->rmaArgs->nRmaTasksCe != 1) {
    ERROR("RMA CE WaitSignal: invalid task or plan");
    return ncclInternalError;
  }
  ncclIntruQueueDequeue(&plan->rmaTaskQueueCe);
  struct ncclRmaCeCtx* ceCtx = (struct ncclRmaCeCtx*)comm->rmaState.rmaCeState.rmaCeCtxs[ctx];

  if (task->signalMode == NCCL_SIGNAL) {
    if (!persistent) {
      // Non-graph: batch one cuStreamWaitValue per peer with incrementing expected value
      NCCLCHECKGOTO(ncclCalloc(&batchParams, task->npeers), ret, fail);
      size_t opIdx = 0;
      for (int i = 0; i < task->npeers; i++) {
        int peerRank = task->peers[i];
        uint64_t waitValue = ceCtx->signalsHost[peerRank] + task->nsignals[i];
        ceCtx->signalsHost[peerRank] = waitValue;

        HGdeviceptr signalAddr = (HGdeviceptr)&ceCtx->signalsDev[peerRank];
        batchParams[opIdx].waitValue.operation = HG_STREAM_MEM_OP_WAIT_VALUE_64;
        batchParams[opIdx].waitValue.address = signalAddr;
        batchParams[opIdx].waitValue.value64 = waitValue;
        batchParams[opIdx].waitValue.flags = HG_STREAM_WAIT_VALUE_GEQ;
        opIdx++;
      }
      NCCLCHECKGOTO(ncclRmaCeBatchMemOp(stream, opIdx, batchParams), ret, fail);
    } else {
      // Graph: wait-reset-ack cycle using separate graphSignalsDev (isolated from non-graph)
      for (int i = 0; i < task->npeers; i++) {
        int peerRank = task->peers[i];
        int peerLsaRank;
        NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, peerRank, &peerLsaRank), ret, fail);

        for (int s = 0; s < task->nsignals[i]; s++) {
          HGdeviceptr graphSignalAddr = (HGdeviceptr)&ceCtx->graphSignalsDev[peerRank];
          HGstreamBatchMemOpParams signalOps[2] = {};
          signalOps[0].waitValue.operation = HG_STREAM_MEM_OP_WAIT_VALUE_64;
          signalOps[0].waitValue.address = graphSignalAddr;
          signalOps[0].waitValue.value64 = 1;
          signalOps[0].waitValue.flags = HG_STREAM_WAIT_VALUE_GEQ;
          signalOps[1].writeValue.operation = HG_STREAM_MEM_OP_WRITE_VALUE_64;
          signalOps[1].writeValue.address = graphSignalAddr;
          signalOps[1].writeValue.value64 = 0;
          signalOps[1].writeValue.flags = HG_STREAM_WRITE_VALUE_DEFAULT;
          NCCLCHECKGOTO(ncclRmaCeBatchMemOp(stream, 2, signalOps), ret, fail);

          void* peerAck;
          NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceCtx->signalsWin,
                                            ceCtx->graphAckOffset + comm->rank * sizeof(uint64_t),
                                            peerLsaRank, &peerAck), ret, fail);
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
  if (batchParams) ncclFree(batchParams);
  return ret;
fail:
  goto exit;
}
