/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <assert.h>
#include <new>
#include "pccl.h"
#include "alloc.h"
#include "checks.h"
#include "gdrwrap.h"
#include "comm.h"
#include "compiler.h"
#include "debug.h"
#include "param.h"
#include "dev_runtime.h"
#include "gin/gin_host.h"
#include "rma/rma.h"
#include "rma/rma_proxy.h"

NCCL_PARAM(RmaProxyQueueSize, "RMA_PROXY_QUEUE_SIZE", -1);

static uint64_t isPowerOfTwo(uint64_t n) {
  return (n > 0) && ((n & (n - 1)) == 0);
}

// ---- Context lifecycle ----

ncclResult_t ncclRmaProxySeqAlloc(struct ncclRmaProxyCtx* rmaProxyCtx, struct ncclRmaProxyDesc* desc) {
  struct ncclRmaProxySeqSlab* slab = rmaProxyCtx->seqSlab;
  if (slab == nullptr) return ncclInternalError;
  std::lock_guard<std::mutex> guard(slab->lock);

  if (slab->base == nullptr) {
    NCCLCHECK(ncclHgMemHostAlloc((void**)&slab->base, NCCL_RMA_PROXY_SEQ_SLAB_BYTES, &slab->bytes,
                                 &slab->handle));
    slab->capacity = (uint32_t)(slab->bytes / (2 * sizeof(uint64_t)));
    NCCLCHECK(ncclCalloc(&slab->used, (slab->capacity + 63) / 64));
    slab->nextHint = 0;
  }

  for (uint32_t k = 0; k < slab->capacity; k++) {
    uint32_t slot = (slab->nextHint + k) % slab->capacity;
    uint64_t bit = 1ull << (slot & 63);
    if (slab->used[slot >> 6] & bit) continue;
    slab->used[slot >> 6] |= bit;
    slab->nextHint = slot + 1;
    // The pair is addressed identically by the CPU and by the stream, so the
    // host and device pointers are the same virtual address.
    uint64_t* pair = slab->base + (size_t)slot * 2;
    pair[0] = 0;
    pair[1] = 0;
    desc->readySeq = &pair[0];
    desc->readySeqDev = &pair[0];
    desc->doneSeq = &pair[1];
    desc->doneSeqDev = &pair[1];
    desc->readySeqGdrHandle = nullptr;
    desc->doneSeqGdrHandle = nullptr;
    desc->seqSlot = (int)slot;
    desc->seqCtx = rmaProxyCtx;
    return ncclSuccess;
  }

  WARN("Rank %d: RMA proxy sequence slab exhausted (%u slots)", rmaProxyCtx->comm->rank, slab->capacity);
  return ncclInternalError;
}

ncclResult_t ncclRmaProxySeqFree(struct ncclRmaProxyDesc* desc) {
  struct ncclRmaProxyCtx* rmaProxyCtx = desc->seqCtx;
  if (rmaProxyCtx == nullptr || desc->seqSlot < 0) return ncclSuccess;
  struct ncclRmaProxySeqSlab* slab = rmaProxyCtx->seqSlab;
  if (slab == nullptr) return ncclInternalError;
  {
    std::lock_guard<std::mutex> guard(slab->lock);
    // Clear the pair before publishing the slot: the proxy reads a stored 1 as
    // "submitted", so a stale value would reissue a dead descriptor.
    desc->readySeq[0] = 0;
    desc->doneSeq[0] = 0;
    uint32_t slot = (uint32_t)desc->seqSlot;
    slab->used[slot >> 6] &= ~(1ull << (slot & 63));
  }
  desc->seqSlot = -1;
  desc->seqCtx = nullptr;
  return ncclSuccess;
}

static void ncclRmaProxySeqSlabDestroy(struct ncclRmaProxyCtx* rmaProxyCtx) {
  struct ncclRmaProxySeqSlab* slab = rmaProxyCtx->seqSlab;
  if (slab == nullptr) return;
  if (slab->base != nullptr) {
    (void)hgMemUnmap((HGdeviceptr)slab->base, slab->bytes);
    (void)hgMemRelease(slab->handle);
    (void)hgMemAddressFree((HGdeviceptr)slab->base, slab->bytes);
  }
  free(slab->used);
  delete slab;
  rmaProxyCtx->seqSlab = nullptr;
}

static ncclResult_t ncclRmaProxyCtxAlloc(ncclComm_t comm, struct ncclRmaProxyCtx* rmaProxyCtx) {
  // The clean up in case of failure will be done by the ncclRmaProxyDestroyContext function invoked by the caller.
  struct pcclGinState* ginState = &comm->sharedRes->ginState;

  // Allocate the signals on the GPU and then register the memory region with the GIN backend.
  // Enforcing strong ordering on the signals mr is vital to ensure ordering between puts and signals.
  size_t signalSlots = (size_t)comm->nRanks * comm->config.numRmaSig;
  size_t signalsBufSize = signalSlots * sizeof(uint64_t);
  NCCLCHECK(ncclHgMemAlloc((void**)&rmaProxyCtx->signalsDev, &rmaProxyCtx->signalsCumemhandle,
                           HG_MEM_HANDLE_TYPE_NONE, signalsBufSize, NULL, NULL));
  HGGCCHECK(hggcMemset(rmaProxyCtx->signalsDev, 0, signalsBufSize));
  // The IB proxy backend sets *mhandle and *ginHandle to the same object.
  void* signalsGinHandle = NULL;
  NCCLCHECK(ginState->ncclGin->regMrSym(rmaProxyCtx->rmaCollComm, rmaProxyCtx->signalsDev, signalsBufSize,
                                         NCCL_PTR_HGGC, NCCL_NET_MR_FLAG_FORCE_SO,
                                         &rmaProxyCtx->signalsMhandle, &signalsGinHandle));
  // Allocate the host buffer to track the expected values of the signals
  NCCLCHECK(ncclCalloc(&rmaProxyCtx->signalsHost, signalsBufSize));
  // Allocate the sequence numbers for the per-rank network function descriptors
  // These are allocated as CPU-accessible memory (either GDR or host memory)
  NCCLCHECK(allocMemCPUAccessible(&rmaProxyCtx->opSeqs, &rmaProxyCtx->opSeqsDev, comm->nRanks, 0,
                                  &rmaProxyCtx->opSeqsGdrHandle));
  NCCLCHECK(allocMemCPUAccessible(&rmaProxyCtx->readySeqs, &rmaProxyCtx->readySeqsDev, comm->nRanks, 0,
                                  &rmaProxyCtx->readySeqsGdrHandle));
  NCCLCHECK(allocMemCPUAccessible(&rmaProxyCtx->doneSeqs, &rmaProxyCtx->doneSeqsDev, comm->nRanks, 0,
                                  &rmaProxyCtx->doneSeqsGdrHandle));

  // Sanitize and set up the lock-free circular buffer queue size
  uint64_t queueSize = ncclParamRmaProxyQueueSize();
  uint32_t maxRequests = NCCL_NET_MAX_REQUESTS * ginState->ginProps[rmaProxyCtx->ginCommIdx].maxRecvs;
  rmaProxyCtx->maxInflightRequests = maxRequests;
  if (queueSize == (uint64_t)-1) {
    queueSize = maxRequests;
  }
  if (queueSize > maxRequests) {
    INFO(NCCL_NET,
         "PCCL_RMA_PROXY_QUEUE_SIZE is greater than the maximum outstanding requests (%d), using the "
         "default/maximum value instead",
         maxRequests);
    queueSize = maxRequests;
  }
  if (queueSize < 1) {
    INFO(NCCL_NET, "PCCL_RMA_PROXY_QUEUE_SIZE is less than 1, using the default/maximum value instead");
    queueSize = maxRequests;
  }
  if (!isPowerOfTwo(queueSize)) {
    INFO(NCCL_NET, "PCCL_RMA_PROXY_QUEUE_SIZE is not a power of two, using the default/maximum value instead");
    queueSize = maxRequests;
  }
  rmaProxyCtx->queueSize = queueSize;

  // Allocate lock-free circular buffer for pending Descs
  size_t circularBufLength = comm->nRanks * queueSize;
  NCCLCHECK(ncclCalloc(&rmaProxyCtx->circularBuffers, circularBufLength));
  NCCLCHECK(ncclCalloc(&rmaProxyCtx->pis, comm->nRanks));
  NCCLCHECK(ncclCalloc(&rmaProxyCtx->cis, comm->nRanks));
  NCCLCHECK(ncclCalloc(&rmaProxyCtx->inflightRequests, comm->nRanks));

  // Allocate per-peer InProgress queues (kept as linked list, single consumer)
  rmaProxyCtx->inProgressQueues =
    ncclMemoryStackAlloc<struct ncclIntruQueue<struct ncclRmaProxyDesc, &ncclRmaProxyDesc::next>>(
      &comm->memPermanent, comm->nRanks);
  for (int i = 0; i < comm->nRanks; i++) {
    ncclIntruQueueConstruct(&rmaProxyCtx->inProgressQueues[i]);
  }
  return ncclSuccess;
}

static ncclResult_t ncclRmaProxyCtxAllocGraph(ncclComm_t comm, struct ncclRmaProxyCtx* rmaProxyCtx) {
  // The clean up in case of failure will be done by the ncclRmaProxyDestroyContext function invoked by the caller.
  struct pcclGinState* ginState = &comm->sharedRes->ginState;

  size_t signalSlots = (size_t)comm->nRanks * comm->config.numRmaSig;
  size_t signalsBufSize = signalSlots * sizeof(uint64_t);
  // Allocate the CPU-accessible signal for graph capture and then register the memory region with the GIN backend.
  NCCLCHECK(allocMemCPUAccessible(&rmaProxyCtx->cpuAccessSignals, &rmaProxyCtx->cpuAccessSignalsDev, signalSlots, 0,
                                  &rmaProxyCtx->cpuAccessSignalsGdrHandle));
  int cpuAccessSignalsType = (rmaProxyCtx->cpuAccessSignalsGdrHandle != NULL) ? NCCL_PTR_HGGC : NCCL_PTR_HOST;
  void* cpuAccessSignalsGinHandle = NULL;
  NCCLCHECK(ginState->ncclGin->regMrSym(rmaProxyCtx->rmaCollComm, rmaProxyCtx->cpuAccessSignalsDev, signalsBufSize,
                                         cpuAccessSignalsType, NCCL_NET_MR_FLAG_FORCE_SO,
                                         &rmaProxyCtx->cpuAccessSignalsMhandle, &cpuAccessSignalsGinHandle));
  // Allocate the host buffer to track the expected values of the signals
  NCCLCHECK(ncclCalloc(&rmaProxyCtx->cpuAccessSignalsHost, signalsBufSize));

  // Allocate the flush buffer on the GPU and then register the memory region with the GIN backend.
  size_t flushBufSize = comm->nRanks * sizeof(uint64_t);
  NCCLCHECK(ncclHgMemAlloc((void**)&rmaProxyCtx->flushBufDev, &rmaProxyCtx->flushBufCumemhandle,
                           HG_MEM_HANDLE_TYPE_NONE, flushBufSize, NULL, NULL));
  HGGCCHECK(hggcMemset(rmaProxyCtx->flushBufDev, 0, flushBufSize));
  void* flushBufGinHandle = NULL;
  NCCLCHECK(ginState->ncclGin->regMrSym(rmaProxyCtx->rmaCollComm, rmaProxyCtx->flushBufDev, flushBufSize,
                                         NCCL_PTR_HGGC, NCCL_NET_MR_FLAG_FORCE_SO,
                                         &rmaProxyCtx->flushBufMhandle, &flushBufGinHandle));

  // Allocate and initialize persistent descriptor queue
  rmaProxyCtx->persistentQueues =
    ncclMemoryStackAlloc<struct ncclIntruQueue<struct ncclRmaProxyDesc, &ncclRmaProxyDesc::next>>(
      &comm->memPermanent, comm->nRanks);
  for (int i = 0; i < comm->nRanks; i++) {
    ncclIntruQueueConstruct(&rmaProxyCtx->persistentQueues[i]);
  }
  return ncclSuccess;
}

ncclResult_t ncclRmaProxyCreateContext(ncclComm_t comm, void* collComm, int ginCommIdx,
                                       void** outRmaProxyCtx) {
  ncclResult_t ret = ncclSuccess;
  struct pcclGinState* ginState = &comm->sharedRes->ginState;

  // One context with host-issued requests (queueDepth must stay 0 for the IB
  // proxy backend); the RMA proxy drives everything from its progress thread.
  ncclGinConfig_t config = {};
  config.nContexts = 1;
  config.queueDepth = 0;
  config.trafficClass = comm->config.trafficClass;

  // Allocate the RMA proxy context
  struct ncclRmaProxyCtx* rmaProxyCtx = nullptr;
  NCCLCHECKGOTO(ncclCalloc(&rmaProxyCtx, 1), ret, fail);

  rmaProxyCtx->comm = comm;
  rmaProxyCtx->rmaCollComm = collComm;
  rmaProxyCtx->seqSlab = new (std::nothrow) ncclRmaProxySeqSlab();
  if (rmaProxyCtx->seqSlab == nullptr) {
    ret = ncclSystemError;
    goto fail;
  }
  // Index into ginState->ginComms / ginProps used for this context's data-window handles.
  rmaProxyCtx->ginCommIdx = ginCommIdx;
  // Host-issued requests; no device handle is needed.
  NCCLCHECKGOTO(ginState->ncclGin->createContext(collComm, &config, &rmaProxyCtx->rmaCtx, NULL), ret, fail);

  NCCLCHECKGOTO(ncclRmaProxyCtxAlloc(comm, rmaProxyCtx), ret, fail);
  NCCLCHECKGOTO(ncclRmaProxyCtxAllocGraph(comm, rmaProxyCtx), ret, fail);

  *outRmaProxyCtx = rmaProxyCtx;

  return ncclSuccess;
fail:
  ncclRmaProxyDestroyContext(comm, rmaProxyCtx);
  return ret;
}

ncclResult_t ncclRmaProxyDestroyContext(ncclComm_t comm, void* rmaProxyCtx) {
  if (!rmaProxyCtx) return ncclSuccess;
  struct ncclRmaProxyCtx* ctx = (struct ncclRmaProxyCtx*)rmaProxyCtx;
  struct pcclGinState* ginState = &comm->sharedRes->ginState;

  if (ginState->ncclGin && ctx->rmaCtx) NCCLCHECK(ginState->ncclGin->destroyContext(ctx->rmaCtx));

  if (ctx->circularBuffers) {
    for (int i = 0; ctx->pis != NULL && ctx->cis != NULL && i < ctx->comm->nRanks; i++) {
      uint32_t ci = COMPILER_ATOMIC_LOAD_32(&ctx->cis[i], std::memory_order_relaxed);
      uint32_t pi = COMPILER_ATOMIC_LOAD_32(&ctx->pis[i], std::memory_order_relaxed);
      // Free any remaining pending descriptors
      for (uint32_t j = ci; j != pi; j++) {
        uint32_t idx = j & (ctx->queueSize - 1);
        struct ncclRmaProxyDesc* desc = ctx->circularBuffers[i * ctx->queueSize + idx];
        if (desc != NULL) {
          NCCLCHECK(ncclRmaProxyDestroyDesc(ctx->comm, &desc));
        }
      }
    }
    free(ctx->circularBuffers);
  }

  // Free PI/CI arrays
  free(ctx->pis);
  free(ctx->cis);
  free(ctx->inflightRequests);

  // Free InProgress queues and their Descs
  if (ctx->inProgressQueues) {
    for (int i = 0; i < ctx->comm->nRanks; i++) {
      struct ncclRmaProxyDesc* desc = ncclIntruQueueHead(&ctx->inProgressQueues[i]);
      while (desc != NULL) {
        struct ncclRmaProxyDesc* nextDesc = desc->next;
        ncclIntruQueueDequeue(&ctx->inProgressQueues[i]);
        NCCLCHECK(ncclRmaProxyDestroyDesc(ctx->comm, &desc));
        desc = nextDesc;
      }
    }
  }

  // Free persistent descriptor queue and their Descs
  if (ctx->persistentQueues) {
    for (int i = 0; i < ctx->comm->nRanks; i++) {
      struct ncclRmaProxyDesc* desc = ncclIntruQueueHead(&ctx->persistentQueues[i]);
      while (desc != NULL) {
        struct ncclRmaProxyDesc* nextDesc = desc->next;
        ncclIntruQueueDequeue(&ctx->persistentQueues[i]);
        NCCLCHECK(ncclRmaProxyDestroyDesc(ctx->comm, &desc));
        desc = nextDesc;
      }
    }
  }

  // Free counters (using GDR-aware deallocation)
  if (ctx->opSeqs) NCCLCHECK(freeMemCPUAccessible(ctx->opSeqs, ctx->opSeqsGdrHandle));
  if (ctx->readySeqs) NCCLCHECK(freeMemCPUAccessible(ctx->readySeqs, ctx->readySeqsGdrHandle));
  if (ctx->doneSeqs) NCCLCHECK(freeMemCPUAccessible(ctx->doneSeqs, ctx->doneSeqsGdrHandle));

  // Free signals
  if (ginState->ncclGin && ctx->rmaCollComm && ctx->signalsMhandle) {
    NCCLCHECK(ginState->ncclGin->deregMrSym(ctx->rmaCollComm, ctx->signalsMhandle));
  }
  if (ctx->signalsDev) NCCLCHECK(ncclHgMemFree(ctx->signalsDev));

  // Free flush buffer
  if (ginState->ncclGin && ctx->rmaCollComm && ctx->flushBufMhandle) {
    NCCLCHECK(ginState->ncclGin->deregMrSym(ctx->rmaCollComm, ctx->flushBufMhandle));
  }
  if (ctx->flushBufDev) NCCLCHECK(ncclHgMemFree(ctx->flushBufDev));

  // Free CPU-accessible signals
  if (ginState->ncclGin && ctx->rmaCollComm && ctx->cpuAccessSignalsMhandle) {
    NCCLCHECK(ginState->ncclGin->deregMrSym(ctx->rmaCollComm, ctx->cpuAccessSignalsMhandle));
  }
  if (ctx->cpuAccessSignals) {
    NCCLCHECK(freeMemCPUAccessible(ctx->cpuAccessSignals, ctx->cpuAccessSignalsGdrHandle));
  }

  // Free host signals buffers
  free(ctx->signalsHost);
  free(ctx->cpuAccessSignalsHost);

  ncclRmaProxySeqSlabDestroy(ctx);

  free(ctx);

  return ncclSuccess;
}

// ---- Progress thread and global lifecycle ----

static void ncclRmaProxyReleasePending(struct ncclRmaProxyState* rmaProxyState) {
  for (int c = 0; c < rmaProxyState->rmaProxyCtxCount; c++) {
    struct ncclRmaProxyCtx* ctx = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[c];
    if (ctx == NULL || ctx->comm == NULL) continue;

    for (int peer = 0; peer < ctx->comm->nRanks; peer++) {
      // Descriptors handed over to the proxy but not issued yet.
      if (ctx->circularBuffers != NULL) {
        uint32_t ci = COMPILER_ATOMIC_LOAD_32(&ctx->cis[peer], std::memory_order_relaxed);
        uint32_t pi = COMPILER_ATOMIC_LOAD_32(&ctx->pis[peer], std::memory_order_acquire);
        // The indices wrap at 2^32, walk them until they meet again.
        for (uint32_t j = ci; j != pi; j++) {
          struct ncclRmaProxyDesc* desc = ctx->circularBuffers[peer * ctx->queueSize + (j & (ctx->queueSize - 1))];
          if (desc != NULL) COMPILER_ATOMIC_STORE(desc->doneSeq, desc->opSeq, std::memory_order_release);
        }
      }

      // Descriptors with requests in flight.
      struct ncclRmaProxyDesc* desc = ncclIntruQueueHead(&ctx->inProgressQueues[peer]);
      while (desc != NULL) {
        COMPILER_ATOMIC_STORE(desc->doneSeq, desc->opSeq, std::memory_order_release);
        desc = desc->next;
      }

      // Graph descriptors. Their opSeq is 1, which is what the persistent fences
      // wait for, and the next replay resets doneSeq itself. The producer may still
      // be appending, so walk the links with ACQUIRE like the poller does.
      if (ctx->persistentQueues != NULL) {
        desc = COMPILER_ATOMIC_LOAD(&ctx->persistentQueues[peer].head, std::memory_order_acquire);
        while (desc != NULL) {
          COMPILER_ATOMIC_STORE(desc->doneSeq, desc->opSeq, std::memory_order_release);
          desc = COMPILER_ATOMIC_LOAD(&desc->next, std::memory_order_acquire);
        }
      }
    }
  }
}

void* ncclRmaProxyProgressThread(struct ncclRmaProxyState* rmaProxyState_) {
  struct ncclRmaProxyState* rmaProxyState = (struct ncclRmaProxyState*)rmaProxyState_;
  while (1) {
    std::unique_lock<std::mutex> lock(rmaProxyState->mutex);
    if (rmaProxyState->rmaProgress == 1) {
      lock.unlock();
      for (int n = 0; n < rmaProxyState->rmaProxyCtxCount; n++) {
        ncclResult_t ret = ncclRmaProxyProgress(rmaProxyState->comm, rmaProxyState->rmaProxyCtxs[n]);
        if (ret != ncclSuccess) {
          COMPILER_ATOMIC_STORE_32(&rmaProxyState->asyncResult, ret, std::memory_order_release);
          INFO(NCCL_ALL, "-> %d [RMA Proxy Progress Thread]", ret);
          {
            ncclRmaProxyReleasePending(rmaProxyState);
            std::lock_guard<std::mutex> guard(rmaProxyState->mutex);
            rmaProxyState->rmaProgress = -2;
            rmaProxyState->cond.notify_all();
          }
          return NULL;
        }
      }
      std::this_thread::yield();
    } else if (rmaProxyState->rmaProgress == 2) {
      // Pause requested for reclaim: acknowledge and sleep.
      // Main thread will do the actual freeing while we're paused.
      rmaProxyState->rmaProgress = 0;
      rmaProxyState->cond.notify_one();
      rmaProxyState->cond.wait(lock);
    } else if (rmaProxyState->rmaProgress == -1) {
      return NULL;
    } else if (rmaProxyState->rmaProgress == 0) {
      rmaProxyState->cond.wait(lock);
    } else {
      INFO(NCCL_ALL, "[RMA Proxy Progress Thread] state unknown %d", rmaProxyState->rmaProgress);
      rmaProxyState->rmaProgress = -2;
      return NULL;
    }
  }
}

ncclResult_t ncclRmaProxyConnectOnce(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclRmaProxyState* rmaProxyState = &comm->rmaState.rmaProxyState;
  struct pcclGinState* ginState = &comm->sharedRes->ginState;
  rmaProxyState->comm = comm;
  // The proxy rides on the GIN connections ncclGinConnectOnce() already built.
  if (!ginState->connected || ginState->ncclGin == NULL || ginState->ginCommCount == 0) {
    WARN("RMA proxy requires established GIN connections.");
    return ncclInvalidUsage;
  }
  if (rmaProxyState->connected) return ncclSuccess;

  // Create virtual RMA proxy contexts, round-robin across the physical GIN
  // communicators.
  rmaProxyState->rmaProxyCtxCount = comm->config.numRmaCtx;
  NCCLCHECK(ncclCalloc(&rmaProxyState->rmaProxyCtxs, rmaProxyState->rmaProxyCtxCount));
  for (int n = 0; n < rmaProxyState->rmaProxyCtxCount; n++) {
    int ginCommIdx = n % ginState->ginCommCount;
    NCCLCHECKGOTO(ncclRmaProxyCreateContext(comm, ginState->ginComms[ginCommIdx], ginCommIdx,
                                             &rmaProxyState->rmaProxyCtxs[n]),
                  ret, fail);
  }

  // Start / wake up the progress thread.
  rmaProxyState->rmaProgress = 1;
  rmaProxyState->thread = std::thread(ncclRmaProxyProgressThread, rmaProxyState);
  ncclSetThreadName(rmaProxyState->thread, "PCCL RMA Proxy Progress%2d", comm->hggcDev);

  INFO(NCCL_INIT, "Rank %d ncclRmaProxyConnectOnce: ginCommCount %d rmaProxyCtxCount:%d", comm->rank,
       ginState->ginCommCount, rmaProxyState->rmaProxyCtxCount);

exit:
  if (ret == ncclSuccess) rmaProxyState->connected = true;
  return ret;
fail:
  if (rmaProxyState->rmaProxyCtxs != NULL) {
    for (int n = 0; n < rmaProxyState->rmaProxyCtxCount; n++) {
      if (rmaProxyState->rmaProxyCtxs[n] != NULL) {
        ncclResult_t cleanupRet = ncclRmaProxyDestroyContext(comm, rmaProxyState->rmaProxyCtxs[n]);
        if (cleanupRet != ncclSuccess) WARN("RMA proxy context cleanup failed: %d", cleanupRet);
      }
    }
    free(rmaProxyState->rmaProxyCtxs);
    rmaProxyState->rmaProxyCtxs = NULL;
  }
  rmaProxyState->rmaProxyCtxCount = 0;
  goto exit;
}

ncclResult_t ncclRmaProxyFinalize(ncclComm_t comm) {
  struct ncclRmaProxyState* rmaProxyState = &comm->rmaState.rmaProxyState;
  if (!rmaProxyState->connected) return ncclSuccess;

  {
    std::lock_guard<std::mutex> lock(rmaProxyState->mutex);
    rmaProxyState->rmaProgress = -1;
    rmaProxyState->cond.notify_one();
  }
  rmaProxyState->thread.join();

  // Destroy all virtual RMA proxy contexts
  if (rmaProxyState->rmaProxyCtxs) {
    for (int n = 0; n < rmaProxyState->rmaProxyCtxCount; n++) {
      if (rmaProxyState->rmaProxyCtxs[n] != NULL) {
        NCCLCHECK(ncclRmaProxyDestroyContext(comm, rmaProxyState->rmaProxyCtxs[n]));
        rmaProxyState->rmaProxyCtxs[n] = NULL;
      }
    }
    // Free the dynamically allocated context array
    free(rmaProxyState->rmaProxyCtxs);
    rmaProxyState->rmaProxyCtxs = NULL;
  }

  // The GIN connections are owned by comm->sharedRes->ginState and are closed
  // by ncclGinHostFinalize, which runs after this function.

  memset((void*)rmaProxyState, 0, sizeof(*rmaProxyState));
  return ncclSuccess;
}
