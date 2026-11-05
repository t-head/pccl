/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "pccl.h"
#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "param.h"
#include "dev_runtime.h"
#include "rma/rma.h"

NCCL_PARAM(RMADisable, "RMA_DISABLE", 0);

bool ncclRmaProxyEnabled(ncclComm_t comm) {
  return !ncclDevrIsOneLsaTeam(comm) && comm->config.numRmaCtx > 0 && comm->globalRmaProxySupport &&
         !ncclParamRMADisable();
}

bool ncclRmaInitialized(ncclComm_t comm) {
  if (!comm->hostRmaSupport) return false;
  // CE is set up for every RMA-capable comm at the first window registration.
  if (!comm->rmaState.rmaCeState.initialized) return false;
  // The proxy only has to be connected when there are peers outside the LSA team.
  if (ncclRmaProxyEnabled(comm) && !comm->rmaState.rmaProxyState.connected) return false;
  return true;
}
static bool isLsaAccessible(ncclComm_t comm, int rank) {
  for (int i = 0; i < comm->devrState.lsaSize; i++) {
    if (comm->devrState.lsaRankList[i] == rank) return true;
  }
  return false;
}

static bool isRmaPutOrSignal(ncclFunc_t func) {
  return func == ncclFuncPutSignal || func == ncclFuncSignal;
}

// Schedule comm->planner RMA tasks to the plan and split the RMA tasks into CE and Proxy tasks
// Then seek opportunities to batch tasks, batching checked for consecutive operations targeting the same context
// - ncclFuncWaitSignal does not perform further batching as the API can already batch waitSignal from multiple peers
// - Consecutive put/signal operation can be batched into the same plan
ncclResult_t scheduleRmaTasksToPlan(ncclComm_t comm, struct ncclKernelPlan* plan) {
  struct ncclKernelPlanner* planner = &comm->planner;

  // Find the first non-empty context queue
  int ctx = -1;
  for (int i = 0; i < comm->config.numRmaCtx; i++) {
    if (!ncclIntruQueueEmpty(&planner->rmaTaskQueues[i])) { ctx = i; break; }
  }
  if (ctx == -1) return ncclSuccess;

  auto* ctxQueue = &planner->rmaTaskQueues[ctx];
  struct ncclTaskRma* firstTask = ncclIntruQueueDequeue(ctxQueue);

  plan->isRma = true;
  plan->rmaArgs = ncclMemoryStackAlloc<struct ncclRmaArgs>(&comm->memScoped);
  plan->rmaArgs->func = firstTask->func;
  plan->rmaArgs->nRmaTasks = 0;
  plan->rmaArgs->nRmaTasksProxy = 0;
  plan->rmaArgs->nRmaTasksCe = 0;

  if (firstTask->func == ncclFuncWaitSignal) {
    // One wait can name peers on both sides of the LSA boundary, so the descriptor
    // list is partitioned and each half becomes at most one task.
    int* peersCe = ncclMemoryStackAlloc<int>(&comm->memScoped, firstTask->npeers);
    int* nsignalsCe = ncclMemoryStackAlloc<int>(&comm->memScoped, firstTask->npeers);
    int* signalIdxsCe = ncclMemoryStackAlloc<int>(&comm->memScoped, firstTask->npeers);
    int* peersProxy = ncclMemoryStackAlloc<int>(&comm->memScoped, firstTask->npeers);
    int* nsignalsProxy = ncclMemoryStackAlloc<int>(&comm->memScoped, firstTask->npeers);
    int* signalIdxsProxy = ncclMemoryStackAlloc<int>(&comm->memScoped, firstTask->npeers);
    int npeersCe = 0, npeersProxy = 0;

    for (int i = 0; i < firstTask->npeers; i++) {
      if (isLsaAccessible(comm, firstTask->peers[i])) {
        peersCe[npeersCe] = firstTask->peers[i];
        nsignalsCe[npeersCe] = firstTask->nsignals[i];
        signalIdxsCe[npeersCe] = firstTask->signalIdxs[i];
        npeersCe++;
      } else {
        peersProxy[npeersProxy] = firstTask->peers[i];
        nsignalsProxy[npeersProxy] = firstTask->nsignals[i];
        signalIdxsProxy[npeersProxy] = firstTask->signalIdxs[i];
        npeersProxy++;
      }
    }

    struct { int n; int* peers; int* nsignals; int* signalIdxs;
             ncclIntruQueue<struct ncclTaskRma, &ncclTaskRma::next>* queue; int* counter; } halves[] = {
      { npeersCe,    peersCe,    nsignalsCe,    signalIdxsCe,    &plan->rmaTaskQueueCe,    &plan->rmaArgs->nRmaTasksCe },
      { npeersProxy, peersProxy, nsignalsProxy, signalIdxsProxy, &plan->rmaTaskQueueProxy, &plan->rmaArgs->nRmaTasksProxy },
    };
    for (auto& h : halves) {
      if (h.n == 0) continue;
      struct ncclTaskRma* t =
        ncclMemoryPoolAlloc<struct ncclTaskRma>(&comm->memPool_ncclTaskRma, &comm->memPermanent);
      t->func = ncclFuncWaitSignal;
      t->ctx = firstTask->ctx;
      t->signalMode = firstTask->signalMode;
      t->signalIdx = 0; // irrelevant for waitSignal; the per-peer indices are in signalIdxs
      t->peers = h.peers;
      t->nsignals = h.nsignals;
      t->signalIdxs = h.signalIdxs;
      t->npeers = h.n;
      t->eActivationMask = firstTask->eActivationMask;
      ncclIntruQueueEnqueue(h.queue, t);
      *h.counter = 1;
      plan->rmaArgs->nRmaTasks++;
    }

    planner->nTasksRma -= 1;
    ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, firstTask);
  } else {
    bool lsaAccessible = isLsaAccessible(comm, firstTask->peer);
    plan->rmaArgs->nRmaTasks = 1;
    plan->rmaArgs->nRmaTasksCe = lsaAccessible ? 1 : 0;
    plan->rmaArgs->nRmaTasksProxy = lsaAccessible ? 0 : 1;
    ncclIntruQueueEnqueue(lsaAccessible ? &plan->rmaTaskQueueCe : &plan->rmaTaskQueueProxy, firstTask);
    planner->nTasksRma -= 1;

    // Pull put/signal tasks from every context into this single plan so one launch
    // covers all contexts. Each context's queue is drained in order, only up to its
    // first WaitSignal task, so per-context FIFO is preserved.
    for (int c = 0; c < comm->config.numRmaCtx; c++) {
      auto* q = &planner->rmaTaskQueues[c];
      while (!ncclIntruQueueEmpty(q)) {
        struct ncclTaskRma* task = ncclIntruQueueHead(q);
        if (!isRmaPutOrSignal(task->func)) break; // stop at WaitSignal
        ncclIntruQueueDequeue(q);
        if (isLsaAccessible(comm, task->peer)) {
          ncclIntruQueueEnqueue(&plan->rmaTaskQueueCe, task);
          plan->rmaArgs->nRmaTasksCe++;
        } else {
          ncclIntruQueueEnqueue(&plan->rmaTaskQueueProxy, task);
          plan->rmaArgs->nRmaTasksProxy++;
        }
        plan->rmaArgs->nRmaTasks++;
        planner->nTasksRma -= 1;
      }
    }
  }

  INFO(NCCL_COLL, "scheduleRmaTasksToPlan: rank=%d ctx=%d func=%d nRmaTasks=%d nRmaTasksProxy=%d nRmaTasksCe=%d",
       comm->rank, ctx, plan->rmaArgs->func, plan->rmaArgs->nRmaTasks, plan->rmaArgs->nRmaTasksProxy,
       plan->rmaArgs->nRmaTasksCe);

  return ncclSuccess;
}

ncclResult_t ncclRmaWaitSignal(ncclComm_t comm, struct ncclKernelPlan* plan, hggcStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // If we have both proxy and CE tasks, execute them in parallel
  if (plan->rmaArgs->nRmaTasksProxy > 0 && plan->rmaArgs->nRmaTasksCe > 0) {
    hggcStream_t ceStream = comm->rmaState.rmaCeState.ceStream;
    hggcEvent_t ceEvent = comm->rmaState.rmaCeState.ceEvent;

    // Record event on input stream first to establish dependency
    HGGCCHECKGOTO(hggcEventRecord(ceEvent, stream), ret, fail);

    // Set up CE stream for parallel execution
    HGGCCHECKGOTO(hggcStreamWaitEvent(ceStream, ceEvent, 0), ret, fail);

    // Launch both operations
    NCCLCHECKGOTO(ncclRmaProxyWaitLaunch(comm, plan, stream), ret, fail);
    NCCLCHECKGOTO(ncclRmaCeWaitLaunch(comm, plan, ceStream), ret, fail);

    // Synchronize streams
    HGGCCHECKGOTO(hggcEventRecord(ceEvent, ceStream), ret, fail);
    HGGCCHECKGOTO(hggcStreamWaitEvent(stream, ceEvent, 0), ret, fail);
  } else if (plan->rmaArgs->nRmaTasksProxy > 0) {
    NCCLCHECKGOTO(ncclRmaProxyWaitLaunch(comm, plan, stream), ret, fail);
  } else if (plan->rmaArgs->nRmaTasksCe > 0) {
    NCCLCHECKGOTO(ncclRmaCeWaitLaunch(comm, plan, stream), ret, fail);
  }

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclRmaPut(ncclComm_t comm, struct ncclKernelPlan* plan, hggcStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // If we have both proxy and CE tasks, execute them in parallel
  if (plan->rmaArgs->nRmaTasksProxy > 0 && plan->rmaArgs->nRmaTasksCe > 0) {
    hggcStream_t ceStream = comm->rmaState.rmaCeState.ceStream;
    hggcEvent_t ceEvent = comm->rmaState.rmaCeState.ceEvent;

    // Record event on input stream first to establish dependency
    HGGCCHECKGOTO(hggcEventRecord(ceEvent, stream), ret, fail);

    // Set up CE stream for parallel execution
    HGGCCHECKGOTO(hggcStreamWaitEvent(ceStream, ceEvent, 0), ret, fail);

    // Launch both operations
    NCCLCHECKGOTO(ncclRmaProxyPutLaunch(comm, plan, stream), ret, fail);
    NCCLCHECKGOTO(ncclRmaCePutLaunch(comm, plan, ceStream), ret, fail);

    // Synchronize streams
    HGGCCHECKGOTO(hggcEventRecord(ceEvent, ceStream), ret, fail);
    HGGCCHECKGOTO(hggcStreamWaitEvent(stream, ceEvent, 0), ret, fail);
  } else if (plan->rmaArgs->nRmaTasksProxy > 0) {
    NCCLCHECKGOTO(ncclRmaProxyPutLaunch(comm, plan, stream), ret, fail);
  } else if (plan->rmaArgs->nRmaTasksCe > 0) {
    NCCLCHECKGOTO(ncclRmaCePutLaunch(comm, plan, stream), ret, fail);
  }

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclLaunchRma(ncclComm_t comm, struct ncclKernelPlan* plan) {
  ncclResult_t ret = ncclSuccess;
  hggcStream_t stream = comm->planner.streams->stream;

  switch (plan->rmaArgs->func) {
  case ncclFuncPutSignal:
    NCCLCHECKGOTO(ncclRmaPut(comm, plan, stream), ret, fail);
    break;
  case ncclFuncSignal:
    NCCLCHECKGOTO(ncclRmaPut(comm, plan, stream), ret, fail);
    break;
  case ncclFuncWaitSignal:
    NCCLCHECKGOTO(ncclRmaWaitSignal(comm, plan, stream), ret, fail);
    break;
  default:
    ret = ncclInvalidUsage;
  }

exit:
  return ret;
fail:
  goto exit;
}
