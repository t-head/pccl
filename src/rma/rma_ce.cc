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

  // Ensure symmetric memory runtime is initialized
  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);

  comm->rmaState.rmaCeState.rmaCeCtxCount = comm->config.numRmaCtx;

  NCCLCHECKGOTO(ncclCalloc(&comm->rmaState.rmaCeState.rmaCeCtxs, comm->rmaState.rmaCeState.rmaCeCtxCount), ret, fail);
  for (int i = 0; i < comm->rmaState.rmaCeState.rmaCeCtxCount; i++) {
    // Allocate the RMA CE context
    struct ncclRmaCeCtx* ceCtx;
    NCCLCHECKGOTO(ncclCalloc(&ceCtx, 1), ret, fail);
    comm->rmaState.rmaCeState.rmaCeCtxs[i] = ceCtx;

    // Initialize context
    ceCtx->comm = comm;

    // Allocate and register symmetric memory for signals
    // Signal buffer layout: [0..nRanks-1] per-rank signals, [nRanks] aggregate signal
    int nRanks = comm->nRanks;
    size_t signalsBufSize = (nRanks + 1) * sizeof(uint64_t);
    ncclWindow_vidmem* signalsWinDev;
    ncclWindow_vidmem* signalsWinDevHost;

    NCCLCHECKGOTO(pcclMemAlloc((void**)&signalsDevBase, signalsBufSize), ret, fail);
    NCCLCHECKGOTO(ncclDevrWindowRegisterInGroup(comm, signalsDevBase, signalsBufSize, NCCL_WIN_COLL_SYMMETRIC,
                                                &signalsWinDev),
                  ret, fail);
    NCCLCHECKGOTO(ncclShadowPoolToHost(&comm->devrState.shadows, signalsWinDev, &signalsWinDevHost), ret, fail);

    // Get the ncclDevrWindow from the winHost field
    ceCtx->signalsWin = (struct ncclDevrWindow*)signalsWinDevHost->winHost;
    ceCtx->signalsDev = signalsDevBase;
    signalsDevBase = nullptr;
    ceCtx->signalOffset = 0;

    // Allocate host buffer to track expected signal values
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
    WARN("RMA CE is not initialized");
    return ncclInternalError;
  }

  int nRmaTasksCe = plan->rmaArgs->nRmaTasksCe;
  int ctx = plan->rmaArgs->ctx;
  struct ncclRmaCeCtx* ceCtx = (struct ncclRmaCeCtx*)comm->rmaState.rmaCeState.rmaCeCtxs[ctx];

  for (int i = 0; i < nRmaTasksCe; i++) {
    struct ncclTaskRma* task = ncclIntruQueueHead(&plan->rmaTaskQueueCe);
    ncclIntruQueueDequeue(&plan->rmaTaskQueueCe);

    int peerLsaRank;
    NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, task->peer, &peerLsaRank), ret, fail);

    size_t bytes = task->count * ncclTypeSize(task->datatype);

    if (bytes > 0) {
      // Get the peer buffer from the peer window
      void* peerBuff;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, task->peerWinHost, task->peerWinOffset, peerLsaRank, &peerBuff),
                    ret, fail);

      // Validate peer buffer
      if (peerBuff == NULL) {
        WARN("RMA CE: peerBuff is NULL after ncclDevrGetLsaRankPtr");
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

      // Write incrementing sequence to peer's signalsDev
      void* peerSignal;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceCtx->signalsWin, ceCtx->signalOffset + rankSlot,
                                        peerLsaRank, &peerSignal), ret, fail);
      ceCtx->signalOpSeqs[task->peer]++;
      HGCHECKGOTO(hgStreamWriteValue64(stream, (HGdeviceptr)ceCtx->signalOpSeqsDev,
                                     ceCtx->signalOpSeqs[task->peer], HG_STREAM_WRITE_VALUE_DEFAULT), ret, fail);
      HGGCCHECKGOTO(hggcMemcpyAsync(peerSignal, ceCtx->signalOpSeqsDev, sizeof(uint64_t),
                                  hggcMemcpyDeviceToDevice, stream), ret, fail);
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
    WARN("RMA CE is not initialized");
    return ncclInternalError;
  }

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
    // Batch one cuStreamWaitValue per peer with incrementing expected value
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
  }

  // Free the task
  ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);

exit:
  if (batchParams) ncclFree(batchParams);
  return ret;
fail:
  goto exit;
}
