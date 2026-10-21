/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "pccl.h"
#include "checks.h"
#include "comm.h"
#include "compiler.h"
#include "debug.h"
#include "gin/gin_host.h"
#include "rma/rma.h"
#include "rma/rma_proxy.h"

// Helper functions to manage request credits
static inline bool ncclRmaProxyCanIssueRequest(struct ncclRmaProxyCtx* ctx, int targetRank) {
  return ctx->inflightRequests[targetRank] < ctx->maxInflightRequests;
}

// Issue one putSignal op via the network.
static ncclResult_t ncclRmaProxyIssuePutSignal(struct pcclGinState* ginState, struct ncclRmaProxyCtx* ctx,
                                               struct ncclRmaPutSignalOp* ps) {
  if (ps->signal.op == 0) {
    NCCLCHECK(ginState->ncclGin->iput(ctx->rmaCtx, 0, ps->srcOff, ps->srcHandle, ps->size, ps->dstOff, ps->dstHandle,
                                      ps->targetRank, &ps->request));
  } else {
    NCCLCHECK(ginState->ncclGin->iputSignal(ctx->rmaCtx, 0, ps->srcOff, ps->srcHandle, ps->size, ps->dstOff,
                                            ps->dstHandle, ps->targetRank, ps->signal.offset,
                                            ps->signal.signalMhandle, ps->signal.val, ps->signal.op, &ps->request));
  }
  // Defensive: RMA proxy iput/iputSignal should return a non-NULL request with inflightRequests checked.
  if (ps->request == nullptr) {
    WARN("RMA proxy iput/iputSignal returned success with NULL request");
    return ncclInternalError;
  }
  ctx->inflightRequests[ps->targetRank]++;
  return ncclSuccess;
}

static ncclResult_t ncclRmaProxyTestPutSignal(struct pcclGinState* ginState, struct ncclRmaProxyCtx* ctx,
                                              struct ncclRmaPutSignalOp* ps, int* done) {
  NCCLCHECK(ginState->ncclGin->test(ctx->rmaCollComm, ps->request, done));
  if (*done) {
    if (ctx->inflightRequests[ps->targetRank] == 0) {
      WARN("RMA proxy completed request with no inflight credit");
      return ncclInternalError;
    }
    ctx->inflightRequests[ps->targetRank]--;
    ps->request = nullptr;
  }
  return ncclSuccess;
}

static ncclResult_t ncclRmaProxyProgressPutSignalGroup(struct pcclGinState* ginState, struct ncclRmaProxyCtx* ctx,
                                                       struct ncclRmaPutSignalGroupOp* group) {
  // Test the completion of the issued ops.
  for (int i = 0; i < group->nIssued; i++) {
    if (group->ops[i].request == nullptr) continue;
    int done = 0;
    NCCLCHECK(ncclRmaProxyTestPutSignal(ginState, ctx, &group->ops[i], &done));
    if (done) group->nCompleted++;
  }

  while (group->nIssued < group->nOps) {
    struct ncclRmaPutSignalOp* op = &group->ops[group->nIssued];
    int targetRank = op->targetRank;
    if (!ncclRmaProxyCanIssueRequest(ctx, targetRank)) break;

    NCCLCHECK(ncclRmaProxyIssuePutSignal(ginState, ctx, op));
    group->nIssued++;
  }
  return ncclSuccess;
}

// Poll and test completion of InProgress Descs for a given peer.
// Returns after testing head Desc (stops on first incomplete to enforce FIFO).
static ncclResult_t ncclRmaProxyPollNonPersistCompletion(struct pcclGinState* ginState, struct ncclRmaProxyCtx* ctx,
                                                         int peer) {
  while (true) {
    struct ncclRmaProxyDesc* head = ncclIntruQueueHead(&ctx->inProgressQueues[peer]);
    if (head == nullptr) break;  // No InProgress Descs

    bool fullyDone = false;
    if (head->rmaDescType == ncclRmaDescTypePutSignal) {
      int done = 0;
      NCCLCHECK(ncclRmaProxyTestPutSignal(ginState, ctx, &head->putSignal, &done));
      fullyDone = (done != 0);
    } else if (head->rmaDescType == ncclRmaDescTypePutSignalGroup) {
      NCCLCHECK(ncclRmaProxyProgressPutSignalGroup(ginState, ctx, &head->putSignalGroup));
      fullyDone = (head->putSignalGroup.nCompleted == head->putSignalGroup.nOps);
    }

    if (!fullyDone) break;  // FIFO at this slot - don't peek behind head

    INFO(NCCL_COLL,
         "Rank %d ncclRmaProxyPollNonPersistCompletion: peer=%d type=%d descSeq=%lu COMPLETED, updating doneSeq",
         ctx->comm->rank, peer, head->rmaDescType, head->opSeq);

    // Update the doneSeq with RELEASE to ensure GPU sees it
    // sync with the custreamWait acquire semantic
    COMPILER_ATOMIC_STORE(head->doneSeq, head->opSeq, std::memory_order_release);
    // Dequeue and free the completed Desc
    ncclIntruQueueDequeue(&ctx->inProgressQueues[peer]);
    NCCLCHECK(ncclRmaProxyDestroyDesc(ctx->comm, &head));
  }
  return ncclSuccess;
}

// Poll and issue ready Pending Descs for a given peer
// Moves ready Descs from pending queue to InProgress queue
static ncclResult_t ncclRmaProxyPollNonPersistDesc(struct pcclGinState* ginState, struct ncclRmaProxyCtx* ctx,
                                                   int peer) {
  while (true) {
    if (ncclRmaProxyCircularBufEmpty(ctx, peer)) break;

    uint32_t ci = COMPILER_ATOMIC_LOAD_32(&ctx->cis[peer], std::memory_order_relaxed);
    uint32_t idx = ci & (ctx->queueSize - 1);
    struct ncclRmaProxyDesc* pendingDesc = ctx->circularBuffers[peer * ctx->queueSize + idx];

    // Check if this Desc is ready to be issued
    uint64_t readySeq = COMPILER_ATOMIC_LOAD(pendingDesc->readySeq, std::memory_order_acquire);
    if (readySeq < pendingDesc->opSeq) {
      // ReadySeq not ready yet - stop processing this peer's pending queue to maintain FIFO order
      break;
    }

    if (pendingDesc->rmaDescType == ncclRmaDescTypePutSignal) {
      if (!ncclRmaProxyCanIssueRequest(ctx, pendingDesc->putSignal.targetRank)) break;

      NCCLCHECK(ncclRmaProxyIssuePutSignal(ginState, ctx, &pendingDesc->putSignal));
      INFO(NCCL_COLL,
           "Rank %d ncclRmaProxyPollNonPersistDesc: targetRank=%d descSeq=%lu readySeq=%lu srcOff=%lu srcHandle=%p "
           "dstOff=%lu dstHandle=%p size=%lu - issuing network operation",
           ctx->comm->rank, pendingDesc->putSignal.targetRank, pendingDesc->opSeq, readySeq,
           pendingDesc->putSignal.srcOff, pendingDesc->putSignal.srcHandle, pendingDesc->putSignal.dstOff,
           pendingDesc->putSignal.dstHandle, pendingDesc->putSignal.size);
    } else if (pendingDesc->rmaDescType == ncclRmaDescTypePutSignalGroup) {
      NCCLCHECK(ncclRmaProxyProgressPutSignalGroup(ginState, ctx, &pendingDesc->putSignalGroup));
      if (pendingDesc->putSignalGroup.nIssued < pendingDesc->putSignalGroup.nOps) break;
      INFO(NCCL_COLL,
           "Rank %d ncclRmaProxyPollNonPersistDesc: Group(nOps=%d) descSeq=%lu readySeq=%lu - issuing all ops",
           ctx->comm->rank, pendingDesc->putSignalGroup.nOps, pendingDesc->opSeq, readySeq);
    }

    // Enqueue to InProgress queue (no lock needed - progress thread only)
    ncclIntruQueueEnqueue(&ctx->inProgressQueues[peer], pendingDesc);
    // Advance CI with RELEASE after ownership transfers to the in-progress queue.
    COMPILER_ATOMIC_STORE_32(&ctx->cis[peer], ci + 1, std::memory_order_release);
  }
  return ncclSuccess;
}

// Blocking strict-order loopback iput flush.
// Issues a local iput to the flush buffer registered with strict ordering and
// spins until the NIC reports completion, guaranteeing that all prior data
// written through the NIC-GPU path is visible in vidmem.
static ncclResult_t ncclRmaProxyFlushNicGpuPath(struct pcclGinState* ginState, struct ncclRmaProxyCtx* ctx) {
  void* request = nullptr;
  size_t flushOff = (size_t)ctx->comm->rank * sizeof(uint64_t);
  NCCLCHECK(ginState->ncclGin->iput(ctx->rmaCtx, 0, flushOff, ctx->flushBufMhandle, sizeof(uint64_t), flushOff,
                                    ctx->flushBufMhandle, ctx->comm->rank, &request));
  int done = 0;
  while (!done) {
    NCCLCHECK(ginState->ncclGin->test(ctx->rmaCollComm, request, &done));
  }
  return ncclSuccess;
}

// Poll persistent descriptors for a given peer (graph mode).
static ncclResult_t ncclRmaProxyPollPersistDesc(struct pcclGinState* ginState, struct ncclRmaProxyCtx* ctx, int peer) {
  struct ncclRmaProxyDesc* desc =
    COMPILER_ATOMIC_LOAD(&ctx->persistentQueues[peer].head, std::memory_order_acquire);
  while (desc != nullptr) {
    if (!COMPILER_ATOMIC_LOAD(&desc->persistDescValid, std::memory_order_acquire)) {
      desc = COMPILER_ATOMIC_LOAD(&desc->next, std::memory_order_acquire);
      continue;
    }

    if (desc->rmaDescType == ncclRmaDescTypePutSignal) {
      if (desc->rmaDescState == ncclRmaDescStateReady) {
        uint64_t readyVal = COMPILER_ATOMIC_LOAD(desc->readySeq, std::memory_order_acquire);
        if (readyVal == 1) {
          if (!ncclRmaProxyCanIssueRequest(ctx, desc->putSignal.targetRank)) {
            desc = COMPILER_ATOMIC_LOAD(&desc->next, std::memory_order_acquire);
            continue;
          }

          NCCLCHECK(ncclRmaProxyIssuePutSignal(ginState, ctx, &desc->putSignal));
          COMPILER_ATOMIC_STORE(desc->readySeq, (uint64_t)0, std::memory_order_relaxed);
          desc->rmaDescState = ncclRmaDescStateInProgress;
          INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollPersistDesc: peer=%d PutSignal issued", ctx->comm->rank, peer);
        }
      } else if (desc->rmaDescState == ncclRmaDescStateInProgress) {
        int done = 0;
        NCCLCHECK(ncclRmaProxyTestPutSignal(ginState, ctx, &desc->putSignal, &done));
        if (done) {
          COMPILER_ATOMIC_STORE(desc->doneSeq, (uint64_t)1, std::memory_order_release);
          desc->rmaDescState = ncclRmaDescStateReady;
          INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollPersistDesc: peer=%d PutSignal completed, doneSeq set",
               ctx->comm->rank, peer);
        }
      }
    } else if (desc->rmaDescType == ncclRmaDescTypePutSignalGroup) {
      // State machine mirrors PutSignal but issues / tests nOps requests.
      if (desc->rmaDescState == ncclRmaDescStateReady) {
        uint64_t readyVal = COMPILER_ATOMIC_LOAD(desc->readySeq, std::memory_order_acquire);
        if (readyVal == 1) {
          desc->putSignalGroup.nIssued = 0;
          desc->putSignalGroup.nCompleted = 0;

          NCCLCHECK(ncclRmaProxyProgressPutSignalGroup(ginState, ctx, &desc->putSignalGroup));
          if (desc->putSignalGroup.nIssued > 0) {
            COMPILER_ATOMIC_STORE(desc->readySeq, (uint64_t)0, std::memory_order_relaxed);
            desc->rmaDescState = ncclRmaDescStateInProgress;
            INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollPersistDesc: peer=%d Group(nOps=%d) issued", ctx->comm->rank, peer,
                 desc->putSignalGroup.nOps);
          }
        }
      } else if (desc->rmaDescState == ncclRmaDescStateInProgress) {
        NCCLCHECK(ncclRmaProxyProgressPutSignalGroup(ginState, ctx, &desc->putSignalGroup));
        if (desc->putSignalGroup.nCompleted == desc->putSignalGroup.nOps) {
          COMPILER_ATOMIC_STORE(desc->doneSeq, (uint64_t)1, std::memory_order_release);
          desc->rmaDescState = ncclRmaDescStateReady;
          INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollPersistDesc: peer=%d Group complete, doneSeq set", ctx->comm->rank,
               peer);
        }
      }
    } else if (desc->rmaDescType == ncclRmaDescTypeWaitSignal) {
      if (desc->rmaDescState == ncclRmaDescStateReady) {
        uint64_t readyVal = COMPILER_ATOMIC_LOAD(desc->readySeq, std::memory_order_acquire);
        if (readyVal == 1) {
          COMPILER_ATOMIC_STORE(desc->readySeq, (uint64_t)0, std::memory_order_relaxed);
          desc->rmaDescState = ncclRmaDescStateInProgress;
          INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollPersistDesc: peer=%d WaitSignal ready, start polling signals",
               ctx->comm->rank, peer);
        }
      } else if (desc->rmaDescState == ncclRmaDescStateInProgress) {
        bool signalsAllArrived = true;
        for (int i = 0; i < desc->waitSignal.npeers; i++) {
          int peerRank = desc->waitSignal.waitPeers[i];
          size_t signalSlot = ncclRmaSignalSlot(ctx->comm->nRanks, desc->waitSignal.waitSignalIdxs[i], peerRank);
          uint64_t signalVal = COMPILER_ATOMIC_LOAD(&ctx->cpuAccessSignals[signalSlot], std::memory_order_acquire);
          uint64_t expected = ctx->cpuAccessSignalsHost[signalSlot];
          for (int j = 0; j <= i; j++) {
            if (desc->waitSignal.waitPeers[j] == peerRank &&
                desc->waitSignal.waitSignalIdxs[j] == desc->waitSignal.waitSignalIdxs[i]) {
              expected += desc->waitSignal.waitSignals[j];
            }
          }
          if (signalVal < expected) {
            signalsAllArrived = false;
            break;
          }
        }

        if (signalsAllArrived) {
          if (desc->waitSignal.needFlush) {
            NCCLCHECK(ncclRmaProxyFlushNicGpuPath(ginState, ctx));
          }

          for (int i = 0; i < desc->waitSignal.npeers; i++) {
            int peerRank = desc->waitSignal.waitPeers[i];
            size_t signalSlot = ncclRmaSignalSlot(ctx->comm->nRanks, desc->waitSignal.waitSignalIdxs[i], peerRank);
            ctx->cpuAccessSignalsHost[signalSlot] += desc->waitSignal.waitSignals[i];
          }

          COMPILER_ATOMIC_STORE(desc->doneSeq, (uint64_t)1, std::memory_order_release);
          desc->rmaDescState = ncclRmaDescStateReady;
          INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollPersistDesc: peer=%d WaitSignal all arrived, doneSeq set",
               ctx->comm->rank, peer);
        }
      }
    }

    desc = COMPILER_ATOMIC_LOAD(&desc->next, std::memory_order_acquire);
  }
  return ncclSuccess;
}

// Checks the RMA proxy progress.
ncclResult_t ncclRmaProxyProgress(ncclComm_t comm, void* rmaProxyCtx) {
  struct ncclRmaProxyCtx* ctx = (struct ncclRmaProxyCtx*)rmaProxyCtx;
  struct pcclGinState* ginState = &comm->sharedRes->ginState;

  if (ginState->ncclGin->ginProgress) NCCLCHECK(ginState->ncclGin->ginProgress(ctx->rmaCtx));

  // Loop through each peer's queues
  for (int i = 0; i < ctx->comm->nRanks; i++) {
    // Step 1: Poll completion of InProgress Descs (non-graph)
    NCCLCHECK(ncclRmaProxyPollNonPersistCompletion(ginState, ctx, i));

    // Step 2: Poll and issue ready Pending Descs (non-graph)
    NCCLCHECK(ncclRmaProxyPollNonPersistDesc(ginState, ctx, i));

    // Step 3: Poll persistent descriptors (graph mode)
    NCCLCHECK(ncclRmaProxyPollPersistDesc(ginState, ctx, i));
  }
  return ncclSuccess;
}
