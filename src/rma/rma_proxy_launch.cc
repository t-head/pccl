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
#include "gdrwrap.h"
#include "pccl_internal.h"
#include "comm.h"
#include "compiler.h"
#include "dev_runtime.h"
#include "rma/rma.h"
#include "rma/rma_proxy.h"

// ---- Descriptor build ----

// Build a single put-signal op. Used by ncclRmaProxyPutBuildDesc and ncclRmaProxyPutGroupBuildDesc.
ncclResult_t ncclRmaProxyPutBuildOp(ncclComm_t comm, struct ncclRmaProxyCtx* rmaProxyCtx, int ctx,
                                    bool persistent, struct ncclDevrWindow* srcWin, size_t srcOff,
                                    struct ncclDevrWindow* peerWin, size_t peerOff, size_t size, int peer,
                                    int signalIdx, ncclSignalMode_t signalMode, struct ncclRmaPutSignalOp* op) {
  op->srcOff = ncclDevrGetWinOffset(srcWin) + srcOff;
  // Data-window MR handles are per GIN connection; index by this context's ginCommIdx.
  op->srcHandle = ncclDevrGetGinWin(srcWin, rmaProxyCtx->ginCommIdx);
  op->dstOff = ncclDevrGetWinOffset(peerWin) + peerOff;
  op->dstHandle = ncclDevrGetGinWin(peerWin, rmaProxyCtx->ginCommIdx);
  op->size = size;
  op->targetRank = peer;
  op->request = nullptr;

  if (signalMode == NCCL_SIGNAL_NONE) {
    op->signal.op = 0;
  } else if (signalMode == NCCL_SIGNAL) {
    op->signal.op = NCCL_NET_SIGNAL_OP_ADD;
    op->signal.offset = ncclRmaSignalOffset(comm->nRanks, signalIdx, comm->rank);
    op->signal.signalMhandle = persistent ? rmaProxyCtx->cpuAccessSignalsMhandle : rmaProxyCtx->signalsMhandle;
    op->signal.val = 1;
  }
  return ncclSuccess;
}

// Build the put descriptor
ncclResult_t ncclRmaProxyPutBuildDesc(ncclComm_t comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                      struct ncclKernelPlan* plan, struct ncclDevrWindow* srcWinHost,
                                      size_t srcWinOffset, struct ncclDevrWindow* peerWinHost, size_t peerWinOffset,
                                      size_t size, int peer, int ctx, int signalIdx, ncclSignalMode_t signalMode,
                                      struct ncclRmaProxyDesc* desc) {
  ncclResult_t ret = ncclSuccess;
  bool persistent = plan->persistent;

  desc->rmaDescType = ncclRmaDescTypePutSignal;
  desc->rmaDescState = ncclRmaDescStateReady;
  desc->persistPlan = nullptr;
  desc->persistDescValid = false;

  // Inner-struct fields shared with the group builder.
  NCCLCHECKGOTO(ncclRmaProxyPutBuildOp(comm, rmaProxyCtx, ctx, persistent, srcWinHost, srcWinOffset, peerWinHost,
                                       peerWinOffset, size, peer, signalIdx, signalMode, &desc->putSignal),
                ret, fail);

  if (!persistent) {
    desc->opSeq = ++rmaProxyCtx->opSeqs[peer];
    desc->readySeq = &rmaProxyCtx->readySeqs[peer];
    desc->readySeqDev = &rmaProxyCtx->readySeqsDev[peer];
    desc->readySeqGdrHandle = rmaProxyCtx->readySeqsGdrHandle;
    desc->doneSeq = &rmaProxyCtx->doneSeqs[peer];
    desc->doneSeqDev = &rmaProxyCtx->doneSeqsDev[peer];
    desc->doneSeqGdrHandle = rmaProxyCtx->doneSeqsGdrHandle;
  } else {
    desc->opSeq = 1;
    // Allocation during graph capture, off the execution critical path
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->readySeq, &desc->readySeqDev, 1, 0, &desc->readySeqGdrHandle),
                  ret, fail);
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->doneSeq, &desc->doneSeqDev, 1, 0, &desc->doneSeqGdrHandle),
                  ret, fail);
    desc->persistPlan = plan;
  }
exit:
  return ret;
fail:
  if (desc->readySeq) {
    freeMemCPUAccessible(desc->readySeq, desc->readySeqGdrHandle);
    desc->readySeq = nullptr;
    desc->readySeqDev = nullptr;
    desc->readySeqGdrHandle = nullptr;
  }
  if (desc->doneSeq) {
    freeMemCPUAccessible(desc->doneSeq, desc->doneSeqGdrHandle);
    desc->doneSeq = nullptr;
    desc->doneSeqDev = nullptr;
    desc->doneSeqGdrHandle = nullptr;
  }
  goto exit;
}

// Build a put-signal-group descriptor over a caller-supplied array of ops.
ncclResult_t ncclRmaProxyPutGroupBuildDesc(ncclComm_t comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                           struct ncclKernelPlan* plan, int nOps, struct ncclRmaPutSignalOp** ops,
                                           int ctx, struct ncclRmaProxyDesc* desc) {
  ncclResult_t ret = ncclSuccess;
  bool persistent = plan->persistent;
  int slot = comm->rank;

  desc->rmaDescType = ncclRmaDescTypePutSignalGroup;
  desc->rmaDescState = ncclRmaDescStateReady;
  desc->persistPlan = nullptr;
  desc->persistDescValid = false;

  desc->putSignalGroup.nIssued = 0;
  desc->putSignalGroup.nCompleted = 0;
  *ops = nullptr;

  if (!persistent) {
    desc->opSeq = ++rmaProxyCtx->opSeqs[slot];
    desc->readySeq = &rmaProxyCtx->readySeqs[slot];
    desc->readySeqDev = &rmaProxyCtx->readySeqsDev[slot];
    desc->readySeqGdrHandle = rmaProxyCtx->readySeqsGdrHandle;
    desc->doneSeq = &rmaProxyCtx->doneSeqs[slot];
    desc->doneSeqDev = &rmaProxyCtx->doneSeqsDev[slot];
    desc->doneSeqGdrHandle = rmaProxyCtx->doneSeqsGdrHandle;
  } else {
    desc->opSeq = 1;
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->readySeq, &desc->readySeqDev, 1, 0, &desc->readySeqGdrHandle),
                  ret, fail);
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->doneSeq, &desc->doneSeqDev, 1, 0, &desc->doneSeqGdrHandle),
                  ret, fail);
    desc->persistPlan = plan;
  }
exit:
  return ret;
fail:
  if (desc->readySeq) {
    freeMemCPUAccessible(desc->readySeq, desc->readySeqGdrHandle);
    desc->readySeq = nullptr;
    desc->readySeqDev = nullptr;
    desc->readySeqGdrHandle = nullptr;
  }
  if (desc->doneSeq) {
    freeMemCPUAccessible(desc->doneSeq, desc->doneSeqGdrHandle);
    desc->doneSeq = nullptr;
    desc->doneSeqDev = nullptr;
    desc->doneSeqGdrHandle = nullptr;
  }
  goto exit;
}

static ncclResult_t ncclRmaProxyPutDescFromTask(ncclComm_t comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                                struct ncclKernelPlan* plan, struct ncclTaskRma* task,
                                                struct ncclRmaProxyDesc* desc) {
  return ncclRmaProxyPutBuildDesc(comm, rmaProxyCtx, plan, task->srcWinHost, task->srcWinOffset, task->peerWinHost,
                                  task->peerWinOffset, task->count * ncclTypeSize(task->datatype), task->peer,
                                  task->ctx, task->signalIdx, task->signalMode, desc);
}

// Build a wait-signal descriptor. The task's peers/nsignals/signalIdxs arrays live
// in comm->memScoped and die with the group scope, so the descriptor deep-copies them.
ncclResult_t ncclRmaProxyWaitBuildDesc(ncclComm_t comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                       struct ncclKernelPlan* plan, int npeers, int** peers, int** nsignals,
                                       int** signalIdxs, struct ncclRmaProxyDesc* desc) {
  ncclResult_t ret = ncclSuccess;
  bool persistent = plan->persistent;

  desc->rmaDescType = ncclRmaDescTypeWaitSignal;
  desc->rmaDescState = ncclRmaDescStateReady;
  desc->waitSignal.npeers = npeers;
  NCCLCHECKGOTO(ncclCalloc(&desc->waitSignal.waitPeers, npeers), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&desc->waitSignal.waitSignals, npeers), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&desc->waitSignal.waitSignalIdxs, npeers), ret, fail);
  memcpy(desc->waitSignal.waitPeers, *peers, npeers * sizeof(int));
  memcpy(desc->waitSignal.waitSignals, *nsignals, npeers * sizeof(int));
  memcpy(desc->waitSignal.waitSignalIdxs, *signalIdxs, npeers * sizeof(int));
  desc->persistPlan = nullptr;
  desc->persistDescValid = false;
  if (persistent) {
    desc->opSeq = 1;
    // If GDR is not used, we need to flush the NIC-GPU path to ensure the data is visible in vidmem
    desc->waitSignal.needFlush = (rmaProxyCtx->cpuAccessSignalsGdrHandle == NULL);
    // Allocation during graph capture, off the execution critical path
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->readySeq, &desc->readySeqDev, 1, 0, &desc->readySeqGdrHandle),
                  ret, fail);
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->doneSeq, &desc->doneSeqDev, 1, 0, &desc->doneSeqGdrHandle),
                  ret, fail);
    desc->persistPlan = plan;
  }
exit:
  return ret;
fail:
  if (desc->readySeq) {
    freeMemCPUAccessible(desc->readySeq, desc->readySeqGdrHandle);
    desc->readySeq = nullptr;
    desc->readySeqDev = nullptr;
    desc->readySeqGdrHandle = nullptr;
  }
  if (desc->doneSeq) {
    freeMemCPUAccessible(desc->doneSeq, desc->doneSeqGdrHandle);
    desc->doneSeq = nullptr;
    desc->doneSeqDev = nullptr;
    desc->doneSeqGdrHandle = nullptr;
  }
  goto exit;
}

// Build the wait descriptor from an ncclTaskRma
static ncclResult_t ncclRmaProxyWaitDescFromTask(ncclComm_t comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                                 struct ncclKernelPlan* plan, struct ncclTaskRma* task,
                                                 struct ncclRmaProxyDesc* desc) {
  return ncclRmaProxyWaitBuildDesc(comm, rmaProxyCtx, plan, task->npeers, &task->peers, &task->nsignals,
                                   &task->signalIdxs, desc);
}

// ---- Descriptor destroy ----

static ncclResult_t ncclRmaProxyDestroyDescNonPersistent(struct ncclRmaProxyDesc* desc) {
  if (desc->rmaDescType == ncclRmaDescTypeWaitSignal) {
    free(desc->waitSignal.waitPeers);
    free(desc->waitSignal.waitSignals);
    free(desc->waitSignal.waitSignalIdxs);
  } else if (desc->rmaDescType == ncclRmaDescTypePutSignalGroup) {
    free(desc->putSignalGroup.ops);
  }
  free(desc);
  return ncclSuccess;
}

static ncclResult_t ncclRmaProxyDestroyDescPersistent(ncclComm_t comm, struct ncclRmaProxyDesc* desc) {
  if (desc->readySeqGdrHandle || desc->readySeq) {
    freeMemCPUAccessible(desc->readySeq, desc->readySeqGdrHandle);
  }
  if (desc->doneSeqGdrHandle || desc->doneSeq) {
    freeMemCPUAccessible(desc->doneSeq, desc->doneSeqGdrHandle);
  }
  if (desc->rmaDescType == ncclRmaDescTypeWaitSignal) {
    free(desc->waitSignal.waitPeers);
    free(desc->waitSignal.waitSignals);
    free(desc->waitSignal.waitSignalIdxs);
  } else if (desc->rmaDescType == ncclRmaDescTypePutSignalGroup) {
    free(desc->putSignalGroup.ops);
  }
  free(desc);
  return ncclSuccess;
}

// Destroy a descriptor.
ncclResult_t ncclRmaProxyDestroyDesc(ncclComm_t comm, struct ncclRmaProxyDesc** desc) {
  ncclResult_t ret;
  if ((*desc)->persistPlan == nullptr) {
    ret = ncclRmaProxyDestroyDescNonPersistent(*desc);
  } else {
    ret = ncclRmaProxyDestroyDescPersistent(comm, *desc);
  }
  *desc = nullptr;
  return ret;
}

// ---- Queue management ----

bool ncclRmaProxyCircularBufFull(struct ncclRmaProxyCtx* ctx, int peer) {
  uint32_t pi = COMPILER_ATOMIC_LOAD_32(&ctx->pis[peer], std::memory_order_relaxed);
  uint32_t ci = COMPILER_ATOMIC_LOAD_32(&ctx->cis[peer], std::memory_order_acquire);
  return (pi - ci) >= ctx->queueSize;
}

bool ncclRmaProxyCircularBufEmpty(struct ncclRmaProxyCtx* ctx, int peer) {
  uint32_t ci = COMPILER_ATOMIC_LOAD_32(&ctx->cis[peer], std::memory_order_relaxed);
  uint32_t pi = COMPILER_ATOMIC_LOAD_32(&ctx->pis[peer], std::memory_order_acquire);
  return ci >= pi;
}

// Returns true if the queue this descriptor would enqueue into is full.
bool ncclRmaProxyEnqueueFull(struct ncclRmaProxyCtx* ctx, const struct ncclRmaProxyDesc* desc) {
  // Persistent queues are unbounded.
  if (desc->persistPlan != nullptr || desc->captured) return false;
  // Non-persistent wait has no queue.
  if (desc->rmaDescType == ncclRmaDescTypeWaitSignal) return false;
  // Non-persistent put or put-group: derive peer from the desc.
  int peer;
  switch (desc->rmaDescType) {
  case ncclRmaDescTypePutSignal:
    peer = desc->putSignal.targetRank;
    break;
  case ncclRmaDescTypePutSignalGroup:
    peer = ctx->comm->rank;
    break;
  default:
    WARN("ncclRmaProxyEnqueueFull: unknown desc type %d", desc->rmaDescType);
    return false;
  }
  return ncclRmaProxyCircularBufFull(ctx, peer);
}

static inline ncclResult_t ncclRmaProxyEnqueueNonPersistentDesc(struct ncclRmaProxyCtx* ctx, int peer,
                                                                struct ncclRmaProxyDesc* desc) {
  if (desc->rmaDescType == ncclRmaDescTypeWaitSignal) {
    // Destroy the scratch desc for wait non-persistent
    return ncclRmaProxyDestroyDesc(ctx->comm, &desc);
  }
  // Non-persistent puts use a bounded circular buffer; the caller must have
  // checked ncclRmaProxyEnqueueFull.
  if (ncclRmaProxyCircularBufFull(ctx, peer)) {
    WARN("RMA proxy circular buffer is full for peer %d", peer);
    return ncclInternalError;
  }
  uint32_t pi = COMPILER_ATOMIC_LOAD_32(&ctx->pis[peer], std::memory_order_relaxed);
  uint32_t idx = pi & (ctx->queueSize - 1);
  ctx->circularBuffers[peer * ctx->queueSize + idx] = desc;
  // Advance PI with RELEASE to ensure descriptor write is visible
  COMPILER_ATOMIC_STORE_32(&ctx->pis[peer], pi + 1, std::memory_order_release);
  return ncclSuccess;
}

static inline ncclResult_t ncclRmaProxyEnqueuePersistentDesc(struct ncclRmaProxyCtx* ctx, int peer,
                                                             struct ncclRmaProxyDesc* desc) {
  ncclIntruQueueEnqueue(&ctx->persistentQueues[peer], desc);
  // Transition to persistent with RELEASE to guarantee all preceding writes are visible to proxy
  COMPILER_ATOMIC_STORE(&desc->persistDescValid, true, std::memory_order_release);
  return ncclSuccess;
}

// Enqueue a built descriptor onto its target queue. Non-persistent waits carry
// no queue; their desc is destroyed in place.
ncclResult_t ncclRmaProxyEnqueueDesc(struct ncclRmaProxyCtx* rmaProxyCtx, struct ncclRmaProxyDesc** desc) {
  int peer;
  switch ((*desc)->rmaDescType) {
  case ncclRmaDescTypePutSignal:
    peer = (*desc)->putSignal.targetRank;
    break;
  case ncclRmaDescTypeWaitSignal:
    peer = rmaProxyCtx->comm->rank;
    break;
  case ncclRmaDescTypePutSignalGroup:
    peer = rmaProxyCtx->comm->rank;
    break;
  default:
    WARN("ncclRmaProxyEnqueueDesc: unknown desc type %d", (*desc)->rmaDescType);
    return ncclInternalError;
  }

  // Check if the target queue is full and yield if it is.
  while (ncclRmaProxyEnqueueFull(rmaProxyCtx, *desc)) {
    std::this_thread::yield();
  }

  bool persistent = ((*desc)->persistPlan != nullptr) || (*desc)->captured;
  ncclResult_t ret;
  if (!persistent) {
    ret = ncclRmaProxyEnqueueNonPersistentDesc(rmaProxyCtx, peer, *desc);
  } else {
    ret = ncclRmaProxyEnqueuePersistentDesc(rmaProxyCtx, peer, *desc);
  }
  *desc = nullptr;
  return ret;
}

// ---- Stream memops ----

// Write the ready sequence that triggers the proxy to launch the put.
ncclResult_t ncclRmaProxyPutStart(struct ncclRmaProxyDesc* desc, hggcStream_t stream) {
  if (desc->rmaDescType != ncclRmaDescTypePutSignal) {
    WARN("ncclRmaProxyPutStart: descriptor is not a put-signal type (%d)", desc->rmaDescType);
    return ncclInternalError;
  }
  HGCHECK(hgStreamWriteValue64(stream, (HGdeviceptr)desc->readySeqDev, desc->opSeq, HG_STREAM_WRITE_VALUE_DEFAULT));
  return ncclSuccess;
}

// Fence the stream until the proxy reports the put complete; persistent descs
// additionally reset doneSeq to 0 for the next graph replay.
ncclResult_t ncclRmaProxyPutDone(struct ncclRmaProxyDesc* desc, hggcStream_t stream) {
  if (desc->rmaDescType != ncclRmaDescTypePutSignal) {
    WARN("ncclRmaProxyPutDone: descriptor is not a put-signal type (%d)", desc->rmaDescType);
    return ncclInternalError;
  }
  HGCHECK(hgStreamWaitValue64(stream, (HGdeviceptr)desc->doneSeqDev, desc->opSeq, HG_STREAM_WAIT_VALUE_GEQ));
  bool persistent = (desc->persistPlan != nullptr) || desc->captured;
  if (persistent) {
    HGCHECK(hgStreamWriteValue64(stream, (HGdeviceptr)desc->doneSeqDev, 0, HG_STREAM_WRITE_VALUE_DEFAULT));
  }
  return ncclSuccess;
}

// Write the ready sequence that triggers the proxy to launch the group.
// The group has one shared (readySeq, opSeq) regardless of nOps.
ncclResult_t ncclRmaProxyPutGroupStart(struct ncclRmaProxyDesc* desc, hggcStream_t stream) {
  if (desc->rmaDescType != ncclRmaDescTypePutSignalGroup) {
    WARN("ncclRmaProxyPutGroupStart: descriptor is not a put-signal-group type (%d)", desc->rmaDescType);
    return ncclInternalError;
  }
  HGCHECK(hgStreamWriteValue64(stream, (HGdeviceptr)desc->readySeqDev, desc->opSeq, HG_STREAM_WRITE_VALUE_DEFAULT));
  return ncclSuccess;
}

// Fence the stream until the proxy reports the whole group complete. Same as
// ncclRmaProxyPutDone but with the type guard targeting PutSignalGroup.
ncclResult_t ncclRmaProxyPutGroupDone(struct ncclRmaProxyDesc* desc, hggcStream_t stream) {
  if (desc->rmaDescType != ncclRmaDescTypePutSignalGroup) {
    WARN("ncclRmaProxyPutGroupDone: descriptor is not a put-signal-group type (%d)", desc->rmaDescType);
    return ncclInternalError;
  }
  HGCHECK(hgStreamWaitValue64(stream, (HGdeviceptr)desc->doneSeqDev, desc->opSeq, HG_STREAM_WAIT_VALUE_GEQ));
  bool persistent = (desc->persistPlan != nullptr) || desc->captured;
  if (persistent) {
    HGCHECK(hgStreamWriteValue64(stream, (HGdeviceptr)desc->doneSeqDev, 0, HG_STREAM_WRITE_VALUE_DEFAULT));
  }
  return ncclSuccess;
}

// Emit the wait-signal stream memops. Non-persistent: one per-peer wait on the
// signalsDev slot. Persistent: write readySeq, block on doneSeq, reset doneSeq.
ncclResult_t ncclRmaProxyWaitOnStream(struct ncclRmaProxyCtx* rmaProxyCtx, struct ncclRmaProxyDesc* desc,
                                      hggcStream_t stream) {
  if (desc->rmaDescType != ncclRmaDescTypeWaitSignal) {
    WARN("ncclRmaProxyWaitOnStream: descriptor is not a wait-signal type (%d)", desc->rmaDescType);
    return ncclInternalError;
  }

  bool persistent = (desc->persistPlan != nullptr) || desc->captured;
  if (!persistent) {
    int npeers = desc->waitSignal.npeers;
    int* peers = desc->waitSignal.waitPeers;
    int* nsignals = desc->waitSignal.waitSignals;
    int* signalIdxs = desc->waitSignal.waitSignalIdxs;
    for (int i = 0; i < npeers; i++) {
      size_t signalSlot = ncclRmaSignalSlot(rmaProxyCtx->comm->nRanks, signalIdxs[i], peers[i]);
      uint64_t waitValue = rmaProxyCtx->signalsHost[signalSlot] + nsignals[i];
      rmaProxyCtx->signalsHost[signalSlot] = waitValue;

      HGCHECK(hgStreamWaitValue64(stream, (HGdeviceptr)&rmaProxyCtx->signalsDev[signalSlot], waitValue,
                                  HG_STREAM_WAIT_VALUE_GEQ));
    }
  } else {
    HGCHECK(hgStreamWriteValue64(stream, (HGdeviceptr)desc->readySeqDev, desc->opSeq, HG_STREAM_WRITE_VALUE_DEFAULT));
    HGCHECK(hgStreamWaitValue64(stream, (HGdeviceptr)desc->doneSeqDev, desc->opSeq, HG_STREAM_WAIT_VALUE_GEQ));
    HGCHECK(hgStreamWriteValue64(stream, (HGdeviceptr)desc->doneSeqDev, 0, HG_STREAM_WRITE_VALUE_DEFAULT));
  }

  return ncclSuccess;
}

// ---- Launch-time API ----

ncclResult_t ncclRmaProxyPutLaunch(ncclComm_t comm, struct ncclKernelPlan* plan, hggcStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  if (!comm->rmaState.rmaProxyState.connected) {
    WARN("RMA proxy is not connected");
    return ncclInternalError;
  }

  int nRmaTasksProxy = plan->rmaArgs->nRmaTasksProxy;
  void** rmaProxyCtxs = comm->rmaState.rmaProxyState.rmaProxyCtxs;

  struct ncclRmaProxyDesc** descs = nullptr;
  struct ncclRmaProxyCtx** descCtxs = nullptr;  // per-desc context: tasks may span contexts
  NCCLCHECK(ncclCalloc(&descs, nRmaTasksProxy));
  NCCLCHECK(ncclCalloc(&descCtxs, nRmaTasksProxy));

  // Phase 1: build all descriptors up front; each desc is built on its own
  // task's context.
  for (int i = 0; i < nRmaTasksProxy; i++) {
    struct ncclTaskRma* task = ncclIntruQueueDequeue(&plan->rmaTaskQueueProxy);
    struct ncclRmaProxyCtx* taskCtx = (struct ncclRmaProxyCtx*)rmaProxyCtxs[task->ctx];
    descCtxs[i] = taskCtx;

    NCCLCHECKGOTO(ncclCalloc(&descs[i], 1), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyPutDescFromTask(comm, taskCtx, plan, task, descs[i]), ret, fail);

    INFO(NCCL_COLL,
         "ncclRmaProxyPutLaunch enqueued Desc: rank=%d peer=%d ctx=%d size=%ld signalMode=%d readySeq=%lu doneSeq=%lu "
         "persistent=%d",
         comm->rank, task->peer, task->ctx, task->count * ncclTypeSize(task->datatype), task->signalMode,
         (uint64_t)descs[i]->opSeq, (uint64_t)descs[i]->opSeq, plan->persistent);

    ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);
  }

  // Phase 2: submit and enqueue in chunks bounded by each context's per-peer
  // circular buffer capacity. Within a chunk all start memops are issued before
  // any blocking done fence, and the descs are handed over last: every field the
  // memops read belongs to the caller until EnqueueDesc transfers ownership.
  {
    int count = 0;
    while (count < nRmaTasksProxy) {
      int i = count;
      for (; i < nRmaTasksProxy; i++) {
        if (ncclRmaProxyEnqueueFull(descCtxs[i], descs[i])) {
          break;
        }
      }

      int pending = i - count;
      if (pending == 0) {
        // No descs fit this round; let the proxy drain a slot.
        std::this_thread::yield();
        continue;
      }

      for (int j = count; j < i; j++) {
        NCCLCHECKGOTO(ncclRmaProxyPutStart(descs[j], stream), ret, fail);
      }
      for (int j = count; j < i; j++) {
        NCCLCHECKGOTO(ncclRmaProxyPutDone(descs[j], stream), ret, fail);
      }
      for (int j = count; j < i; j++) {
        NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(descCtxs[j], &descs[j]), ret, fail);
      }
      count = i;
    }
  }

exit:
  free(descCtxs);
  free(descs);
  return ret;
fail:
  if (descs != nullptr) {
    for (int i = 0; i < nRmaTasksProxy; i++) {
      if (descs[i] == nullptr) continue;
      (void)ncclRmaProxyDestroyDesc(comm, &descs[i]);
    }
  }
  goto exit;
}

ncclResult_t ncclRmaProxyWaitLaunch(ncclComm_t comm, struct ncclKernelPlan* plan, hggcStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  if (!comm->rmaState.rmaProxyState.connected) {
    WARN("RMA proxy is not connected");
    return ncclInternalError;
  }

  struct ncclTaskRma* task = ncclIntruQueueHead(&plan->rmaTaskQueueProxy);
  ncclIntruQueueDequeue(&plan->rmaTaskQueueProxy);

  // A WaitSignal plan is single-context; use the wait task's own context.
  struct ncclRmaProxyCtx* rmaProxyCtx = (struct ncclRmaProxyCtx*)comm->rmaState.rmaProxyState.rmaProxyCtxs[task->ctx];

  struct ncclRmaProxyDesc* desc = nullptr;

  if (task->func != ncclFuncWaitSignal) {
    WARN("RMA proxy task function is %d, expected %d", task->func, ncclFuncWaitSignal);
    ret = ncclInternalError;
    goto exit_non_wait_task;
  }
  if (plan->rmaArgs->nRmaTasksProxy != 1) {
    WARN("RMA proxy task count is %d, expected 1", plan->rmaArgs->nRmaTasksProxy);
    goto invalid_task;
  }

  if (task->signalMode == NCCL_SIGNAL) {
    NCCLCHECKGOTO(ncclCalloc(&desc, 1), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyWaitDescFromTask(comm, rmaProxyCtx, plan, task, desc), ret, fail);
    // Issue the wait memops before handing the desc over; EnqueueDesc may free it.
    NCCLCHECKGOTO(ncclRmaProxyWaitOnStream(rmaProxyCtx, desc, stream), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &desc), ret, fail);
  }

exit:
  if (desc != nullptr) {
    (void)ncclRmaProxyDestroyDesc(comm, &desc);
  }
  // task->peers/nsignals/signalIdxs live in comm->memScoped; nothing to free here.
exit_non_wait_task:
  ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);
  return ret;
invalid_task:
  ret = ncclInternalError;
fail:
  goto exit;
}

// ---- Graph reclaim ----

// Free persistent descs belonging to the given plan.
// ncclRmaProxyProgress is paused when this function is called, so no concurrent access to persistentQueues.
static ncclResult_t ncclRmaProxyReclaimPersistDescs(struct ncclRmaProxyState* proxyState, struct ncclKernelPlan* plan) {
  ncclComm_t comm = proxyState->comm;

  for (int c = 0; c < proxyState->rmaProxyCtxCount; c++) {
    struct ncclRmaProxyCtx* proxyCtx = (struct ncclRmaProxyCtx*)proxyState->rmaProxyCtxs[c];
    if (proxyCtx == nullptr) continue;

    for (int peer = 0; peer < proxyCtx->comm->nRanks; peer++) {
      struct ncclRmaProxyDesc* prev = nullptr;
      struct ncclRmaProxyDesc* desc = ncclIntruQueueHead(&proxyCtx->persistentQueues[peer]);
      while (desc != nullptr) {
        struct ncclRmaProxyDesc* next = desc->next;
        if (desc->persistPlan == plan) {
          if (prev == nullptr) {
            proxyCtx->persistentQueues[peer].head = next;
            if (next == nullptr) proxyCtx->persistentQueues[peer].tail = nullptr;
          } else {
            prev->next = next;
            if (next == nullptr) proxyCtx->persistentQueues[peer].tail = prev;
          }

          NCCLCHECK(ncclRmaProxyDestroyDesc(comm, &desc));
        } else {
          prev = desc;
        }
        desc = next;
      }
    }
  }

  return ncclSuccess;
}

// Destroy persistent descs belonging to the plan.
// Called from reclaimPlan when the CUDA graph is destroyed.
// Pauses the proxy thread, frees descs on the main thread, then resumes the proxy.
ncclResult_t ncclRmaProxyReclaimPlan(ncclComm_t comm, struct ncclKernelPlan* plan) {
  struct ncclRmaProxyState* proxyState = &comm->rmaState.rmaProxyState;
  if (!proxyState->connected) return ncclSuccess;

  // Step 1: Request proxy to pause and wait for acknowledgment
  {
    std::unique_lock<std::mutex> lock(proxyState->mutex);
    proxyState->rmaProgress = 2;
    proxyState->cond.notify_one();
    proxyState->cond.wait(lock, [&] { return proxyState->rmaProgress == 0; });
  }

  // Step 2: Free persistent descs on main thread
  NCCLCHECK(ncclRmaProxyReclaimPersistDescs(proxyState, plan));

  // Step 3: Resume proxy
  {
    std::lock_guard<std::mutex> lock(proxyState->mutex);
    proxyState->rmaProgress = 1;
    proxyState->cond.notify_one();
  }

  return ncclSuccess;
}
