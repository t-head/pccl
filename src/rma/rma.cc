/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <assert.h>
#include "pccl.h"
#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "rma/rma.h"

bool ncclRmaInitialized(struct pcclComm* comm) {
  // Host RMA not supported -> not initialized.
  if (!comm->hostRmaSupport) return false;
  // CE is set up for every RMA-capable comm at the first window registration -> not initialized.
  if (!comm->rmaState.rmaCeState.initialized) return false;
  return true;
}

ncclResult_t ncclRmaWaitSignal(struct pcclComm* comm, struct ncclKernelPlan* plan, hggcStream_t stream){
  ncclResult_t ret = ncclSuccess;

  if (plan->rmaArgs->nRmaTasksCe > 0) {
    NCCLCHECKGOTO(ncclRmaCeWaitLaunch(comm, plan, stream), ret, fail);
  }

exit:
  return ret;
fail:
  goto exit;
}


ncclResult_t ncclRmaPut(struct pcclComm* comm, struct ncclKernelPlan* plan, hggcStream_t stream){
  ncclResult_t ret = ncclSuccess;

  if (plan->rmaArgs->nRmaTasksCe > 0) {
    NCCLCHECKGOTO(ncclRmaCePutLaunch(comm, plan, stream), ret, fail);
  }

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclLaunchRma(struct pcclComm* comm, struct ncclKernelPlan* plan) {
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

static inline bool isRmaPutOrSignal(ncclFunc_t func) {
  return (func == ncclFuncPutSignal || func == ncclFuncSignal);
}

// Check if two RMA tasks can be batched together
static inline bool canBatchRmaTasks(struct ncclTaskRma* task1, struct ncclTaskRma* task2) {
  // Check if the tasks are in the same context
  if (task1->ctx != task2->ctx) return false;

  // Check if the tasks are the same function
  if (task1->func == task2->func) return true;

  // Put/Signal tasks can be batched together
  if (isRmaPutOrSignal(task1->func) && isRmaPutOrSignal(task2->func)) {
    return true;
  }

  return false;
}

// Schedule comm->planner RMA tasks to the plan
// Then seek opportunities to batch tasks, batching checked for consecutive operations targeting the same context
// - ncclFuncWaitSignal does not perform further batching as the API can already batch waitSignal from multiple peers
// - Consecutive put/signal operation can be batched into the same plan
ncclResult_t scheduleRmaTasksToPlan(struct pcclComm* comm, struct ncclKernelPlan* plan) {
  struct ncclKernelPlanner* planner = &comm->planner;

  // Find the first non-empty context queue
  int ctx = -1;
  for (int i = 0; i < comm->config.numRmaCtx; i++) {
    if (!ncclIntruQueueEmpty(&planner->rmaTaskQueues[i])) {
      ctx = i;
      break;
    }
  }

  // No RMA tasks to schedule
  if (ctx == -1) return ncclSuccess;

  struct ncclIntruQueue<struct ncclTaskRma, &ncclTaskRma::next>* ctxQueue = &planner->rmaTaskQueues[ctx];

  // Get the first task to determine the operation category
  struct ncclTaskRma* firstTask = ncclIntruQueueDequeue(ctxQueue);

  // Initialize plan
  plan->isRma = true;
  plan->rmaArgs = ncclMemoryStackAlloc<struct ncclRmaArgs>(&comm->memScoped);
  plan->rmaArgs->ctx = ctx;
  plan->rmaArgs->func = firstTask->func;
  plan->rmaArgs->nRmaTasks = 0;
  plan->rmaArgs->nRmaTasksCe = 0;

  // Host RMA requires a single LSA team, so every peer is reached through CE.

  // WaitSignal tasks
  if (firstTask->func == ncclFuncWaitSignal) {
    ncclIntruQueueEnqueue(&plan->rmaTaskQueueCe, firstTask);
    plan->rmaArgs->nRmaTasks = 1;
    plan->rmaArgs->nRmaTasksCe = 1;
    planner->nTasksRma -= 1;
  }
  // Put/Signal tasks
  else {
    plan->rmaArgs->nRmaTasks = 1;
    plan->rmaArgs->nRmaTasksCe = 1;
    ncclIntruQueueEnqueue(&plan->rmaTaskQueueCe, firstTask);

    planner->nTasksRma -= 1;

    // Batch consecutive tasks from the same context that match operation category
    while (!ncclIntruQueueEmpty(ctxQueue)) {
      struct ncclTaskRma* task = ncclIntruQueueHead(ctxQueue);

      // Check if this task can be batched with the first task
      if (!canBatchRmaTasks(firstTask, task)) {
        break;
      }

      // If the task can be batched, remove from context queue and add to plan
      ncclIntruQueueDequeue(ctxQueue);
      ncclIntruQueueEnqueue(&plan->rmaTaskQueueCe, task);
      plan->rmaArgs->nRmaTasksCe++;
      plan->rmaArgs->nRmaTasks++;
      planner->nTasksRma -= 1;
    }
  }

  INFO(NCCL_COLL, "scheduleRmaTasksToPlan: rank=%d ctx=%d func=%d nRmaTasks=%d nRmaTasksCe=%d",
    comm->rank, ctx, plan->rmaArgs->func, plan->rmaArgs->nRmaTasks, plan->rmaArgs->nRmaTasksCe);

  return ncclSuccess;
}
