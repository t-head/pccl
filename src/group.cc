/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "group.h"
#include "debug.h"
#include "enqueue.h"
#include "transport.h"
#include "symmetric.h"
#include "ce_coll.h"
#include "rma/rma.h"
#include <assert.h>
#include "hgtx.h"
#include "dev_runtime.h"
#include "param/param.h"

thread_local int ncclGroupDepth = 0; // depth of ncclGroupStart nesting
thread_local ncclResult_t ncclGroupError = ncclSuccess;
thread_local ncclComm_t ncclGroupCommHead[ncclGroupTaskTypeNum] = {nullptr};
thread_local ncclComm_t ncclGroupCommPreconnectHead = nullptr;
thread_local struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> ncclAsyncJobs;
thread_local int ncclGroupBlocking = -1; /* default mode */
thread_local bool ncclGroupJobAbortFlag = false;

bool ncclAsyncMode() {
  return ncclGroupDepth > 0;
}

void* ncclAsyncJobMain(void* arg);

ncclResult_t ncclAsyncLaunch(
    struct ncclAsyncJob* job,
    ncclResult_t(*func)(struct ncclAsyncJob*),
    void(*undo)(struct ncclAsyncJob*),
    void(*destructor)(void*), ncclComm_t comm
  ) {
  ncclResult_t ret = ncclSuccess;

  job->destroyFlag = comm->destroyFlag;
  if (ncclGroupDepth == 0) {
    ret = func(job);
    if (ret != ncclSuccess && undo) undo(job);
    if (destructor) destructor(job);
  } else {
    job->func = func;
    job->undo = undo;
    job->destructor = destructor;
    job->abortFlag = comm->abortFlag;
    job->abortFlagDev = comm->abortFlagDev;
    job->childAbortFlag = comm->childAbortFlag;
    job->childAbortFlagDev = comm->childAbortFlagDev;
    job->state = ncclGroupJobRunning;
    job->comm = comm;
    /* check if there are blocking and nonblocking comms at the same time in group. */
    if (comm->destroyFlag) {
      ncclGroupBlocking = 1;
    } else if (ncclGroupBlocking == -1) {
      /* first met communicator */
      ncclGroupBlocking = comm->config.blocking;
    } else if (ncclGroupBlocking != comm->config.blocking) {
      ERROR("Blocking and nonblocking communicators are not allowed in the same group.");
      ret = ncclInvalidArgument;
    }
    if (ret == ncclSuccess) {
      ncclIntruQueueEnqueue(&ncclAsyncJobs, job);
    } else {
      // no need to undo, the job hasn't run
      if (destructor) destructor(job);
    }
  }

  return ret;
}

void* ncclAsyncJobMain(void* arg) {
  struct ncclAsyncJob* job = (struct ncclAsyncJob*)arg;
  job->result = job->func(job);
  if (job->result != ncclSuccess) {
    INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILENAME__, __LINE__, job->result);
  }
  __atomic_store_n(&job->state, ncclGroupJobDone, __ATOMIC_RELEASE);
  return arg;
}

ncclResult_t ncclAsyncJobComplete(struct ncclAsyncJob* job) {
  ncclResult_t ret;
  SYSCHECK(pthread_join(job->thread, NULL), "pthread_join");
  if (job->result != ncclSuccess) {
    ERROR("pcclAsyncJobComplete: job %p failed, job error %d", job, job->result);
  }
  ret = job->result;
  if (job->destructor) job->destructor((void*)job);
  return ret;
}

NCCL_API(ncclResult_t, pcclGroupStart);
ncclResult_t pcclGroupStart() {
  ncclResult_t ret = ncclSuccess;
  PCCL_HGTX3_FUNC_RANGE;

  TRACE_CALL("PCCL API: pcclGroupStart()");
  NCCLCHECK(ncclGroupStartInternal());

  return ret;
}

NCCL_API(ncclResult_t, pcclGroupEnd);
ncclResult_t pcclGroupEnd() {
  ncclResult_t ret = ncclSuccess;
  PCCL_HGTX3_FUNC_RANGE;
  TRACE_CALL("PCCL API: pcclGroupEnd()");
  NCCLCHECKGOTO(ncclGroupEndInternal(), ret, exit);
exit:
  return ret;
}

NCCL_API(ncclResult_t, pcclGroupSimulateEnd, ncclSimInfo_t* simInfo);
ncclResult_t pcclGroupSimulateEnd(ncclSimInfo_t* simInfo) {
  ncclResult_t ret = ncclSuccess;
  PCCL_HGTX3_FUNC_RANGE;
  TRACE_CALL("PCCL API: pcclGroupSimulateEnd()");
  NCCLCHECKGOTO(ncclGroupEndInternal(simInfo), ret, exit);
exit:
  return ret;
}

struct ncclPreconnectJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
  bool* algoNeedConnect;
};

ncclResult_t ncclP2pPreconnectFunc(struct ncclAsyncJob* job_) {
  struct ncclPreconnectJob* job = (struct ncclPreconnectJob*)job_;
  ncclComm_t comm = job->comm;
  HGGCCHECK(hggcSetDevice(comm->hggcDev));
  if (!job_->isThreadMain && CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  NCCLCHECK(ncclTransportP2pSetup(comm, NULL, 1));
  INFO(NCCL_ALLOC, "comm %p totally used %ldB %ldB(sharedRes) device memory after pcclTransportP2pSetup splitShare %d nranks %d rank %d", comm, comm->devMemUsed, comm->sharedRes->devMemUsed, comm->config.splitShare, comm->nRanks, comm->rank);
  INFO(NCCL_ALLOC, "comm %p totally used %ldB %ldB(sharedRes) pinned memory after pcclTransportP2pSetup splitShare %d nranks %d rank %d", comm, comm->pinnedMemUsed, comm->sharedRes->pinnedMemUsed, comm->config.splitShare, comm->nRanks, comm->rank);
  return ncclSuccess;
}

static size_t getP2pChunkSize(size_t totalSize, int minChannels, int maxChannels, size_t minSize, size_t maxSize) {
  size_t size = std::max(minSize, DIVUP(totalSize, minChannels));
  int nChannels = minChannels;
  while (size > maxSize && nChannels <= maxChannels/2) {
    nChannels *= 2;
    size = DIVUP(totalSize, nChannels);
  }
  ALIGN_SIZE(size, minSize);
  return size;
}

static ncclResult_t ncclCollPreconnect(ncclComm_t comm, bool* algoNeedConnect) {
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; ++i) {
    if (algoNeedConnect[i]) {
      switch (i) {
        case NCCL_ALGO_RING: {
          NCCLCHECK(ncclTransportRingConnect(comm));
          break;
        }
        case NCCL_ALGO_TREE: {
          NCCLCHECK(ncclTransportTreeConnect(comm));
          break;
        }
        case NCCL_ALGO_FC: {
          NCCLCHECK(ncclTransportFcConnect(comm));
          break;
        }
        case NCCL_ALGO_PAT: {
          NCCLCHECK(ncclTransportPatConnect(comm));
          break;
        }
        // Yes, it's a dead code.  That's fine...
        // coverity[dead_error_begin]
        default: {
          NCCLCHECK(ncclInternalError);
        }
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclCollPreconnectFunc(struct ncclAsyncJob* job_) {
  struct ncclPreconnectJob* job = (struct ncclPreconnectJob*)job_;
  ncclComm_t comm = job->comm;
  ncclResult_t ret = ncclSuccess;

  if (!job_->isThreadMain) HGGCCHECK(hggcSetDevice(comm->hggcDev));
  if (!job_->isThreadMain && CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  NCCLCHECKGOTO(ncclCollPreconnect(comm, job->algoNeedConnect), ret, fail);
  INFO(NCCL_ALLOC, "comm %p totally used %ldB %ldB(sharedRes) device memory after pcclCollPreconnectFunc splitShare %d nranks %d rank %d", comm, comm->devMemUsed, comm->sharedRes->devMemUsed, comm->config.splitShare, comm->nRanks, comm->rank);
  INFO(NCCL_ALLOC, "comm %p totally used %ldB %ldB(sharedRes) pinned memory after pcclCollPreconnectFunc splitShare %d nranks %d rank %d", comm, comm->pinnedMemUsed, comm->sharedRes->pinnedMemUsed, comm->config.splitShare, comm->nRanks, comm->rank);

exit:
  ncclFree(job->algoNeedConnect);
  job->algoNeedConnect = nullptr;
  return ret;
fail:
  goto exit;
}

struct ncclGroupSymmetricJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
};

ncclResult_t ncclCommGroupRegisterSymmetric(struct ncclAsyncJob* job_) {
  struct ncclGroupSymmetricJob* job = (struct ncclGroupSymmetricJob*)job_;
  ncclComm_t comm = job->comm;
  ncclResult_t ret = ncclSuccess;

  int savedDev;
  bool hasSymmetricTasks = !ncclIntruQueueEmpty(&comm->devrState.regTaskQueue) ||
                           !ncclIntruQueueEmpty(&comm->devrState.commCreateTaskQueue);
  HGGCCHECK(hggcGetDevice(&savedDev));
  HGGCCHECKGOTO(hggcSetDevice(comm->hggcDev), ret, fail);

  while (!ncclIntruQueueEmpty(&comm->devrState.regTaskQueue)) {
    struct ncclDevrRegTask* task = ncclIntruQueueDequeue(&comm->devrState.regTaskQueue);
    ret = ncclDevrWindowRegisterInGroup(comm, task->userPtr, task->userSize, task->winFlags, task->outWinDev);
    ncclFree(task);
    if (ret != ncclSuccess) goto fail;
  }

  // Lazily allocate the p2p-DMA mailbox window the first time this comm
  // registers a symmetric window. This runs inside a per-comm parallel job
  // (asyncJobLaunch), so the bootstrap exchange inside pcclP2pDmaInitOnce is
  // safe here -- it would deadlock if done later at launch time, where
  // doLaunches walks comms sequentially on a single thread.
  if (hasSymmetricTasks && pcclP2pDmaEnabled(comm)) {
    // DMA的执行期epoch依赖CE私有存储；在并行注册job内先满足该资源依赖。
    NCCLCHECKGOTO(ncclCeInit(comm), ret, fail);
    NCCLCHECKGOTO(pcclP2pDmaInitOnce(comm), ret, fail);
  }

  while (!ncclIntruQueueEmpty(&comm->devrState.commCreateTaskQueue)) {
    struct ncclDevrCommCreateTask* task = ncclIntruQueueDequeue(&comm->devrState.commCreateTaskQueue);
    ret = ncclDevrCommCreateInternal(comm, task->reqs, task->outDevComm, /*isInternal=*/false);
    freeDevCommRequirements(task->reqs);
    ncclFree(task);
    if (ret != ncclSuccess) goto fail;
  }

  while (!ncclIntruQueueEmpty(&comm->rmaCeInitTaskQueue)) {
    struct ncclRmaCeInitTask* task = ncclIntruQueueDequeue(&comm->rmaCeInitTaskQueue);
    NCCLCHECKGOTO(ncclRmaCeInit(task->comm), ret, fail);
    ncclFree(task);
  }

  while (!ncclIntruQueueEmpty(&comm->suspendTaskQueue)) {
    struct ncclMemManagerTask* task = ncclIntruQueueDequeue(&comm->suspendTaskQueue);
    ncclComm_t taskComm = task->comm;
    ncclFree(task);
    NCCLCHECKGOTO(ncclCommMemSuspend(taskComm), ret, fail);
  }
  while (!ncclIntruQueueEmpty(&comm->resumeTaskQueue)) {
    struct ncclMemManagerTask* task = ncclIntruQueueDequeue(&comm->resumeTaskQueue);
    ncclComm_t taskComm = task->comm;
    ncclFree(task);
    NCCLCHECKGOTO(ncclCommMemResume(taskComm), ret, fail);
  }

exit:
  HGGCCHECK(hggcSetDevice(savedDev));
  return ret;
fail:
  goto exit;
}

static ncclResult_t doLaunches(ncclComm_t head, bool* groupAbortFlag) {
  ncclResult_t ret = ncclSuccess;
  ncclComm_t comm = head;
  int savedDev;
  bool useBarrier = ncclParamLaunchMode == ncclLaunchModeGroup;
  HGGCCHECK(hggcGetDevice(&savedDev));

  for (comm = head; comm != nullptr; comm = comm->groupNext[ncclGroupTaskTypeCollective]) {
    HGGCCHECKGOTO(hggcSetDevice(comm->hggcDev), ret, end);
    NCCLCHECKGOTO(ncclLaunchPrepare(comm), ret, end);
  }

  while (true) {
    for (comm = head; comm != nullptr; comm = comm->groupNext[ncclGroupTaskTypeCollective]) {
      if (comm->planner.launchFinished) continue;
      if (__atomic_load_n(groupAbortFlag, __ATOMIC_ACQUIRE) ||
          __atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE)) {
        ret = ncclInternalError;
        goto end;
      }
      struct ncclKernelPlan* plan = comm->planner.unlaunchedPlansHead;
      if (plan != nullptr && !comm->planner.launchBeforeDone) {
        bool ready;
        HGGCCHECKGOTO(hggcSetDevice(comm->hggcDev), ret, end);
        NCCLCHECKGOTO(ncclLaunchKernelBefore_NoUncapturedHggc(comm, plan, &ready), ret, end);
        comm->planner.launchBeforeDone = ready;
      }
      if (useBarrier) ncclCommIntraBarrierIn(comm, plan != nullptr ? 1 : 0);
    }
    bool moreRounds = false;
    bool progressed = false;
    for (comm = head; comm != nullptr; comm = comm->groupNext[ncclGroupTaskTypeCollective]) {
      if (comm->planner.launchFinished) continue;
      uint32_t pending = comm->planner.unlaunchedPlansHead != nullptr;
      if (useBarrier) NCCLCHECKGOTO(ncclCommIntraBarrierOut(comm, groupAbortFlag, &pending), ret, end);
      moreRounds |= pending != 0;
      if (pending != 0) {
        struct ncclKernelPlan* plan = comm->planner.unlaunchedPlansHead;
        if (plan != nullptr && comm->planner.launchBeforeDone) {
          bool launched = true;
          HGGCCHECKGOTO(hggcSetDevice(comm->hggcDev), ret, end);
          if (plan->isCeColl) {
            NCCLCHECKGOTO(ncclLaunchCeColl(comm, plan), ret, end);
          } else if (plan->isCeP2p) {
            NCCLCHECKGOTO(pcclP2pDmaLaunch(comm, plan, &launched), ret, end);
          } else if (plan->isRma) {
            NCCLCHECKGOTO(ncclLaunchRma(comm, plan), ret, end);
          } else {
            NCCLCHECKGOTO(ncclLaunchKernel(comm, plan), ret, end);
          }
          if (!launched) continue;
          comm->planner.unlaunchedPlansHead = plan->next;
          comm->planner.launchBeforeDone = false;
          NCCLCHECKGOTO(ncclLaunchKernelAfter_NoHggc(comm, plan), ret, end);
          progressed = true;
        }
      } else {
        HGGCCHECKGOTO(hggcSetDevice(comm->hggcDev), ret, end);
        NCCLCHECKGOTO(ncclLaunchFinish(comm), ret, end);
        NCCLCHECKGOTO(ncclLaunchReset(comm), ret, end);
        comm->planner.launchFinished = true;
        progressed = true;
      }
    }
    if (!moreRounds) break;
    if (!progressed) {
      for (comm = head; comm != nullptr; comm = comm->groupNext[ncclGroupTaskTypeCollective]) {
        if (comm->planner.launchFinished) continue;
        HGGCCHECKGOTO(hggcSetDevice(comm->hggcDev), ret, end);
        NCCLCHECKGOTO(ncclCommPollEventCallbacks(comm, /*waitSome=*/false), ret, end);
      }
      sched_yield();
    }
  }

end:
  if (ret != ncclSuccess) {
    __atomic_store_n(groupAbortFlag, true, __ATOMIC_RELEASE);
    for (comm = head; comm != nullptr; comm = comm->groupNext[ncclGroupTaskTypeCollective]) {
      for (ncclComm_t local = comm->intraComm0; local != nullptr; local = local->intraNext) {
        __atomic_store_n(local->abortFlag, 1, __ATOMIC_RELEASE);
        __atomic_store_n(local->abortFlagDev, 1, __ATOMIC_RELEASE);
      }
      hggcError_t deviceError = hggcSetDevice(comm->hggcDev);
      HGGCCHECKIGNORE(deviceError);
      if (deviceError != hggcSuccess) continue;
      if (comm->planner.hostStreamAcquired) {
        NCCLCHECKIGNORE(ncclStrongStreamRelease(comm->planner.capturingGraph, &comm->sharedRes->hostStream), ret);
        comm->planner.hostStreamAcquired = false;
      }
      if (comm->planner.deviceStreamAcquired) {
        NCCLCHECKIGNORE(ncclStrongStreamRelease(comm->planner.capturingGraph, &comm->sharedRes->deviceStream), ret);
        comm->planner.deviceStreamAcquired = false;
      }
    }
  }
  ncclGroupError = ncclSuccess;
  hggcError_t restoreError = hggcSetDevice(savedDev);
  if (ret == ncclSuccess) {
    HGGCCHECK(restoreError);
  } else {
    HGGCCHECKIGNORE(restoreError);
  }
  return ret;
}

static inline void groupLocalResetJobState() {
  ncclGroupError = ncclSuccess;
  for (int type = 0; type < ncclGroupTaskTypeNum; ++type) ncclGroupCommHead[type] = NULL;
  ncclGroupCommPreconnectHead = NULL;
  ncclGroupBlocking = -1;
  ncclIntruQueueConstruct(&ncclAsyncJobs);
  return;
}

static void groupCleanup(ncclComm_t* groupCommHeadPtr, struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next>* asyncJobsPtr, ncclResult_t error) {
  ncclComm_t comm;
  for (int type = 0; type < ncclGroupTaskTypeNum; ++type) {
    comm = groupCommHeadPtr[type];
    groupCommHeadPtr[type] = nullptr;
    while (comm != nullptr) {
      ncclComm_t next = comm->groupNext[type];
      // We don't know if preconnect succeeded or happened at all, so clear
      // the flags that let `taskAppend()` skip over checking if preconnect
      // is needed.
      if (type == ncclGroupTaskTypeCollective) {
        comm->preconnectNext = reinterpret_cast<ncclComm_t>(0x1);
        while (!ncclIntruQueueEmpty(&comm->argsInfoQueue)) {
          ncclArgsInfo* info = ncclIntruQueueDequeue(&comm->argsInfoQueue);
          ncclFree(info);
        }
        ncclTaskColl* sorted = ncclTaskCollSorterDequeueAll(&comm->planner.collSorter);
        while (sorted != nullptr) {
          ncclTaskColl* nextTask = sorted->next;
          ncclMemoryPoolFree(&comm->memPool_ncclTaskColl, sorted);
          sorted = nextTask;
        }
        for (auto* queue : {&comm->planner.collTaskQueue, &comm->planner.collSymTaskQueue,
                            &comm->planner.collCeTaskQueue}) {
          while (!ncclIntruQueueEmpty(queue))
            ncclMemoryPoolFree(&comm->memPool_ncclTaskColl, ncclIntruQueueDequeue(queue));
        }
        for (int peer = 0; peer < comm->nRanks && comm->planner.peers != nullptr; peer++) {
          for (auto* queue : {&comm->planner.peers[peer].sendQueue, &comm->planner.peers[peer].recvQueue}) {
            while (!ncclIntruQueueEmpty(queue))
              ncclMemoryPoolFree(&comm->memPool_ncclTaskP2p, ncclIntruQueueDequeue(queue));
          }
        }
        for (auto& channel : comm->planner.wipPlan.channels) {
          while (!ncclIntruQueueEmpty(&channel.proxyOpQueue))
            ncclMemoryPoolFree(&comm->memPool_ncclProxyOp, ncclIntruQueueDequeue(&channel.proxyOpQueue));
        }
        while (!ncclIntruQueueEmpty(&comm->planner.collCleanupQueue)) {
          ncclCommCallback* callback = ncclIntruQueueDequeue(&comm->planner.collCleanupQueue);
          ncclResult_t cleanupResult = callback->fn(comm, callback);
          if (cleanupResult != ncclSuccess) ERROR("Abandoned task cleanup failed: %s", pcclGetErrorString(cleanupResult));
        }
        while (!ncclIntruQueueEmpty(&comm->planner.planQueue)) {
          struct ncclKernelPlan* plan = ncclIntruQueueDequeue(&comm->planner.planQueue);
          // A created graph UserObject, queued reclaimer or pending host callback owns its plan.
          // Capturing alone does not transfer ownership when UserObject creation fails.
          if (!plan->graphReclaimOwned && !plan->isHostCbEnq && !plan->reclaimQueued) {
            while (!ncclIntruQueueEmpty(&plan->proxyOpQueue)) {
              struct ncclProxyOp* pxop = ncclIntruQueueDequeue(&plan->proxyOpQueue);
              ncclMemoryPoolFree(&comm->memPool_ncclProxyOp, pxop);
            }
            while (!ncclIntruQueueEmpty(&plan->rmaTaskQueueCe)) {
              ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, ncclIntruQueueDequeue(&plan->rmaTaskQueueCe));
            }
            while (!ncclIntruQueueEmpty(&plan->collTaskQueue))
              ncclMemoryPoolFree(&comm->memPool_ncclTaskColl, ncclIntruQueueDequeue(&plan->collTaskQueue));
            while (!ncclIntruQueueEmpty(&plan->p2pTaskQueue))
              ncclMemoryPoolFree(&comm->memPool_ncclTaskP2p, ncclIntruQueueDequeue(&plan->p2pTaskQueue));
            while (!ncclIntruQueueEmpty(&plan->cleanupQueue)) {
              ncclCommCallback* callback = ncclIntruQueueDequeue(&plan->cleanupQueue);
              ncclResult_t cleanupResult = callback->fn(comm, callback);
              if (cleanupResult != ncclSuccess) ERROR("Abandoned plan cleanup failed: %s", pcclGetErrorString(cleanupResult));
            }
            if (plan->isSymColl) ncclFree(plan->kernelSymkArgs);
            ncclMemoryPoolFree(&comm->memPool_ncclKernelPlan, plan);
          }
        }

        { // Reset comm->planner to empty.
          ncclKernelPlanner::Peer* tmp = comm->planner.peers;
          ncclIntruQueue<struct ncclTaskRma, &ncclTaskRma::next>* tmpRmaQueues = comm->planner.rmaTaskQueues;
          memset(&comm->planner, 0, sizeof(comm->planner));
          comm->planner.peers = tmp;
          if (comm->planner.peers != NULL) memset(comm->planner.peers, 0, comm->nRanks * sizeof(comm->planner.peers[0]));
          comm->planner.rmaTaskQueues = tmpRmaQueues;
          if (tmpRmaQueues != NULL) {
            for (int c = 0; c < comm->config.numRmaCtx; c++) {
              while (!ncclIntruQueueEmpty(&tmpRmaQueues[c])) {
                ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, ncclIntruQueueDequeue(&tmpRmaQueues[c]));
              }
              ncclIntruQueueConstruct(&tmpRmaQueues[c]);
            }
          }
        }
      }

      if (type == ncclGroupTaskTypeSymRegister) {
        while (!ncclIntruQueueEmpty(&comm->devrState.regTaskQueue)) {
          ncclDevrRegTask* task = ncclIntruQueueDequeue(&comm->devrState.regTaskQueue);
          ncclFree(task);
        }
        while (!ncclIntruQueueEmpty(&comm->devrState.commCreateTaskQueue)) {
          ncclDevrCommCreateTask* task = ncclIntruQueueDequeue(&comm->devrState.commCreateTaskQueue);
          freeDevCommRequirements(task->reqs);
          ncclFree(task);
        }
        while (!ncclIntruQueueEmpty(&comm->suspendTaskQueue)) {
          struct ncclMemManagerTask* task = ncclIntruQueueDequeue(&comm->suspendTaskQueue);
          ncclFree(task);
        }
        while (!ncclIntruQueueEmpty(&comm->resumeTaskQueue)) {
          struct ncclMemManagerTask* task = ncclIntruQueueDequeue(&comm->resumeTaskQueue);
          ncclFree(task);
        }
      }
      if (!comm->config.blocking)
        (void) ncclCommSetAsyncError(comm, error);
      (void)ncclGroupCommLeave(comm, type);
      comm = next;
    }
  }

  /* reset everything */
  while (!ncclIntruQueueEmpty(asyncJobsPtr)) {
    struct ncclAsyncJob* job = ncclIntruQueueDequeue(asyncJobsPtr);

    if (!job->destroyFlag) {
      __atomic_store_n(job->abortFlag, 1, __ATOMIC_RELEASE);
      __atomic_store_n(job->abortFlagDev, 1, __ATOMIC_RELEASE);
      if (job->childAbortFlag) {
        __atomic_store_n(job->childAbortFlag, 1, __ATOMIC_RELEASE);
        __atomic_store_n(job->childAbortFlagDev, 1, __ATOMIC_RELEASE);
      }
      if (job->comm && !job->comm->config.blocking)
        (void) ncclCommSetAsyncError(job->comm, error);
    }
    if (job->undo) job->undo(job);
    if (job->destructor) job->destructor((void*)job);
  }

  return;
}

static void ncclPreconnectDeinit(void* args) {
  struct ncclPreconnectJob* job = (struct ncclPreconnectJob*)args;
  ncclFree(job->algoNeedConnect);
  ncclFree(job);
}

static ncclResult_t asyncJobLaunch(struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> *asyncJobsMain, volatile bool *groupAbortFlag) {
  ncclResult_t ret = ncclSuccess;
  bool jobsDone = false;
  bool errorJobAbortFlag = false;
  struct ncclAsyncJob* firstNotStarted = ncclIntruQueueHead(asyncJobsMain);

  if (!ncclIntruQueueEmpty(asyncJobsMain)) {
    struct ncclAsyncJob* job = ncclIntruQueueHead(asyncJobsMain);
    if (job->next == nullptr) {
      job->isThreadMain = true;
      ncclAsyncJobMain(job);
      job->state = ncclGroupJobJoined;
      return job->result;
    }
    do {
      PTHREADCHECKGOTO(pthread_create(&job->thread, nullptr, ncclAsyncJobMain, job),
                       "pcclAsyncJobMain", ret, fail);
      firstNotStarted = job->next;
      job = job->next;
    } while (job != nullptr);

    do {
      jobsDone = true;
      job = ncclIntruQueueHead(asyncJobsMain);
      do {
        ncclGroupJobState_t state = __atomic_load_n(&job->state, __ATOMIC_ACQUIRE);
        if (state == ncclGroupJobRunning) {
          jobsDone = false;
        } else if (state == ncclGroupJobDone) {
          if (pthread_join(job->thread, nullptr) != 0) {
            ERROR("Error waiting for pthread_join : %s", strerror(errno));
            ret = ncclSystemError;
          }
          job->state = ncclGroupJobJoined;
          if (job->result != ncclSuccess) {
            ret = job->result;
            errorJobAbortFlag = true;
          }
        } else {
          /* safety check */
          if (state != ncclGroupJobJoined) {
            ERROR("Async job state is %d, expected %d", state, ncclGroupJobJoined);
            if (ret == ncclSuccess) ret = ncclInternalError;
            errorJobAbortFlag = true;
          }
        }

        if (!job->destroyFlag && (__atomic_load_n(groupAbortFlag, __ATOMIC_ACQUIRE) || errorJobAbortFlag == true)) {
          __atomic_store_n(job->abortFlag, 1, __ATOMIC_RELEASE);
          __atomic_store_n(job->abortFlagDev, 1, __ATOMIC_RELEASE);
          if (job->childAbortFlag) {
            __atomic_store_n(job->childAbortFlag, 1, __ATOMIC_RELEASE);
            __atomic_store_n(job->childAbortFlagDev, 1, __ATOMIC_RELEASE);
          }
          ret = ncclInternalError;
        }

        job = job->next;
      } while (job != nullptr);
    } while (jobsDone == false);

    if (ret != ncclSuccess) goto fail;
  }

exit:
  return ret;
fail:
  if (firstNotStarted != nullptr) {
    for (auto* job = ncclIntruQueueHead(asyncJobsMain); job != nullptr; job = job->next) {
      if (!job->destroyFlag) {
        __atomic_store_n(job->abortFlag, 1, __ATOMIC_RELEASE);
        __atomic_store_n(job->abortFlagDev, 1, __ATOMIC_RELEASE);
        if (job->childAbortFlag) {
          __atomic_store_n(job->childAbortFlag, 1, __ATOMIC_RELEASE);
          __atomic_store_n(job->childAbortFlagDev, 1, __ATOMIC_RELEASE);
        }
      }
    }
    for (auto* job = ncclIntruQueueHead(asyncJobsMain); job != firstNotStarted; job = job->next) {
      int err = pthread_join(job->thread, nullptr);
      if (err != 0) ERROR("Failed to join partially launched async job: %s", strerror(err));
      job->state = ncclGroupJobJoined;
    }
  }
  goto exit;
}

DEFINE_PCCL_PARAM(pcclParamSingleProcMemRegEnable, int64_t, PCCL_SINGLE_PROC_MEM_REG_ENABLE, 0,
                  PCCL_PARAM_FLAG_CACHED, PCCL_PARAM_DEFAULT, "Single-process memory registration");
int64_t ncclParamSingleProcMemRegEnable() { return pcclParamSingleProcMemRegEnable(); }

struct ncclPrepareTasksAndCollPreconnectJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
  ncclSimInfo_t* simInfo;
};

static void ncclPrepareTasksAndCollPreconnectJobFree(void* job) {
  ncclFree(job);
}

static ncclResult_t ncclPrepareTasksAndCollPreconnectFunc(struct ncclAsyncJob* base) {
  auto* job = reinterpret_cast<ncclPrepareTasksAndCollPreconnectJob*>(base);
  bool algoNeedConnect[NCCL_NUM_ALGORITHMS] = {};
  bool needConnect = false;
  int savedDev;
  ncclResult_t ret = ncclSuccess;
  HGGCCHECK(hggcGetDevice(&savedDev));
  HGGCCHECK(hggcSetDevice(job->comm->hggcDev));
  NCCLCHECKGOTO(ncclPrepareTasks(job->comm, algoNeedConnect, &needConnect, job->simInfo), ret, exit);
  if (job->comm->cuMemSupport && needConnect) {
    NCCLCHECKGOTO(ncclCollPreconnect(job->comm, algoNeedConnect), ret, exit);
  }
exit:
  HGGCCHECK(hggcSetDevice(savedDev));
  return ret;
}

static ncclResult_t ncclPrepareTasksAndCollPreconnect(ncclComm_t comm, ncclSimInfo_t* simInfo, struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next>* asyncCollJobs) {
  if (ncclParamSingleProcMemRegEnable()) {
    ncclPrepareTasksAndCollPreconnectJob* job;
    NCCLCHECK(ncclCalloc(&job, 1));
    job->base.func = ncclPrepareTasksAndCollPreconnectFunc;
    job->base.destructor = ncclPrepareTasksAndCollPreconnectJobFree;
    job->base.state = ncclGroupJobRunning;
    job->base.abortFlag = comm->abortFlag;
    job->base.abortFlagDev = comm->abortFlagDev;
    job->comm = comm;
    job->simInfo = simInfo;
    ncclIntruQueueEnqueue(asyncCollJobs, &job->base);
    return ncclSuccess;
  }
  bool needConnect = false;
  bool algoNeedConnect[NCCL_NUM_ALGORITHMS];
  memset(algoNeedConnect, 0, sizeof(bool) * NCCL_NUM_ALGORITHMS);

  HGGCCHECK(hggcSetDevice(comm->hggcDev));
  NCCLCHECK(ncclPrepareTasks(comm, algoNeedConnect, &needConnect, simInfo));

  if (comm->cuMemSupport && needConnect) {
    ncclResult_t ret;
    struct ncclPreconnectJob* job;
    NCCLCHECK(ncclCalloc(&job, 1));
    job->base.func = ncclCollPreconnectFunc;
    job->base.undo = nullptr;
    job->base.destructor = ncclPreconnectDeinit;
    job->base.state = ncclGroupJobRunning;
    job->base.abortFlag = comm->abortFlag;
    job->base.abortFlagDev = comm->abortFlagDev;
    job->comm = comm;
    if ((ret = ncclCalloc(&job->algoNeedConnect, NCCL_NUM_ALGORITHMS))) {
      ncclFree(job);
      NCCLCHECK(ret);
    }
    memcpy(job->algoNeedConnect, algoNeedConnect, sizeof(bool) * NCCL_NUM_ALGORITHMS);
    ncclIntruQueueEnqueue(asyncCollJobs, &job->base);

    // coverity[leaked_storage]
  }

  return ncclSuccess;
}


struct ncclCeInitJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
};

static ncclResult_t ncclCeInitFunc(struct ncclAsyncJob* base) {
  return ncclCeInit(reinterpret_cast<ncclCeInitJob*>(base)->comm);
}

static ncclResult_t ncclCommGroupArgsGlobalCheck(struct ncclAsyncJob* job) {
  ncclComm_t comm = job->comm;
  int savedDev;
  HGGCCHECK(hggcGetDevice(&savedDev));
  ncclResult_t ret = ncclSuccess;
  HGGCCHECKGOTO(hggcSetDevice(comm->hggcDev), ret, exit);
  while (!ncclIntruQueueEmpty(&comm->argsInfoQueue)) {
    ncclArgsInfo* info = ncclIntruQueueDequeue(&comm->argsInfoQueue);
    ret = ncclArgsGlobalCheck(info);
    ncclFree(info);
    if (ret != ncclSuccess) break;
  }
exit:
  HGGCCHECK(hggcSetDevice(savedDev));
  return ret;
}

static ncclResult_t groupLaunch(struct ncclAsyncJob *job_, ncclSimInfo_t* simInfo = NULL) {
  int savedDev;
  ncclResult_t ret = ncclSuccess;
  struct ncclGroupJob *gjob = (struct ncclGroupJob*) job_;
  ncclComm_t *groupCommHeadMain = gjob->groupCommHead;
  ncclComm_t groupCommPreconnectHeadMain = gjob->groupCommPreconnectHead;
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> *asyncJobsMain = &gjob->asyncJobs;
  bool *groupAbortFlag = &gjob->abortFlag;
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> asyncSymJobs, asyncCollJobs, asyncDebugJobs;
  ncclIntruQueueConstruct(&asyncSymJobs);
  ncclIntruQueueConstruct(&asyncCollJobs);
  ncclIntruQueueConstruct(&asyncDebugJobs);

  HGGCCHECKGOTO(hggcGetDevice(&savedDev), ret, fail);

  if (!simInfo && groupCommPreconnectHeadMain != nullptr) {
    ncclComm_t comm = groupCommPreconnectHeadMain;
    do {
      struct ncclPreconnectJob* job;
      NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
      job->base.func = ncclP2pPreconnectFunc;
      job->base.undo = nullptr;
      job->base.destructor = ncclPreconnectDeinit;
      job->base.state = ncclGroupJobRunning;
      job->base.abortFlag = comm->abortFlag;
      job->base.abortFlagDev = comm->abortFlagDev;
      job->comm = comm;
      ncclIntruQueueEnqueue(asyncJobsMain, &job->base);

      ncclComm_t next = comm->preconnectNext;
      comm->preconnectNext = reinterpret_cast<ncclComm_t>(0x1);
      comm = next;
      // job ownership is transferred to asyncJobsMain; it is drained and freed via its destructor.
      // coverity[leaked_storage]
    } while (comm != nullptr);
  }

  NCCLCHECKGOTO(asyncJobLaunch(asyncJobsMain, groupAbortFlag), ret, fail);

  // only loop through sym alloc and register tasks
  for (int type = ncclGroupTaskTypeSymRegister; type <= ncclGroupTaskTypeSymRegister; ++type) {
    if (groupCommHeadMain[type]) {
      ncclComm_t cliqueHead = groupCommHeadMain[type];
      ncclComm_t comm = NULL;
      do {
        comm = cliqueHead;
        do {
          struct ncclGroupSymmetricJob* job;
          NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
          job->base.func = ncclCommGroupRegisterSymmetric;
          job->base.undo = nullptr;
          job->base.destructor = free;
          job->base.state = ncclGroupJobRunning;
          job->base.abortFlag = comm->abortFlag;
          job->base.abortFlagDev = comm->abortFlagDev;
          job->comm = comm;
          ncclIntruQueueEnqueue(&asyncSymJobs, (struct ncclAsyncJob*)job);
          comm = comm->groupNext[type];
        } while (comm != nullptr && comm->intraComm0 == cliqueHead->intraComm0);
        NCCLCHECKGOTO(asyncJobLaunch(&asyncSymJobs, groupAbortFlag), ret, fail);
        while (!ncclIntruQueueEmpty(&asyncSymJobs)) {
          struct ncclAsyncJob* job = ncclIntruQueueDequeue(&asyncSymJobs);
          if (job->destructor) job->destructor((void*)job);
        }
        cliqueHead = comm;
      } while (cliqueHead != nullptr);
    }
  }

  // we do getAlgoInfo in prepare task for two purposes:
  // 1. we can do runtime tree/ring connection by algo choosing instead of init connect all ring/tree.
  // 2. for next local ipc register.
  // in nccl register for ring/tree peer.
  // in pccl register for all local peer
  if (groupCommHeadMain[ncclGroupTaskTypeCollective] != nullptr) {
    ncclComm_t cliqueHead = groupCommHeadMain[ncclGroupTaskTypeCollective];
    ncclComm_t comm = NULL;
    do {
      // We need to preconnect connections for collectives clique by clique to avoid
      // race condition for split shared comms which can connect the same connections
      // at the same time.
      comm = cliqueHead;
      do {
        NCCLCHECKGOTO(ncclPrepareTasksAndCollPreconnect(comm, simInfo, &asyncCollJobs), ret, fail);
        comm = comm->groupNext[ncclGroupTaskTypeCollective];
      } while (comm != nullptr && comm->intraComm0 == cliqueHead->intraComm0);
      // connect
      NCCLCHECKGOTO(asyncJobLaunch(&asyncCollJobs, groupAbortFlag), ret, fail);
      while (!ncclIntruQueueEmpty(&asyncCollJobs)) {
        struct ncclAsyncJob* job = ncclIntruQueueDequeue(&asyncCollJobs);
        if (job->destructor) job->destructor((void*)job);
      }
      cliqueHead = comm;
    } while (cliqueHead != nullptr);

    // done with all buffer allocation, start registration and enqueue
    comm = groupCommHeadMain[ncclGroupTaskTypeCollective];
    do {
      HGGCCHECKGOTO(hggcSetDevice(comm->hggcDev), ret, fail);
      NCCLCHECKGOTO(ncclTasksRegAndEnqueue(comm), ret, fail);
      comm = comm->groupNext[ncclGroupTaskTypeCollective];
    } while (comm);

    cliqueHead = groupCommHeadMain[ncclGroupTaskTypeCollective];
    do {
      comm = cliqueHead;
      do {
        if (comm->checkMode == ncclCheckModeDebugGlobal && !ncclIntruQueueEmpty(&comm->argsInfoQueue)) {
          ncclAsyncJob* job;
          NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
          job->func = ncclCommGroupArgsGlobalCheck;
          job->destructor = free;
          job->state = ncclGroupJobRunning;
          job->abortFlag = comm->abortFlag;
          job->abortFlagDev = comm->abortFlagDev;
          job->comm = comm;
          ncclIntruQueueEnqueue(&asyncDebugJobs, job);
        }
        comm = comm->groupNext[ncclGroupTaskTypeCollective];
      } while (comm != nullptr && comm->intraComm0 == cliqueHead->intraComm0);
      NCCLCHECKGOTO(asyncJobLaunch(&asyncDebugJobs, groupAbortFlag), ret, fail);
      while (!ncclIntruQueueEmpty(&asyncDebugJobs)) {
        ncclAsyncJob* job = ncclIntruQueueDequeue(&asyncDebugJobs);
        if (job->destructor) job->destructor(job);
      }
      cliqueHead = comm;
    } while (cliqueHead != nullptr);

    if (!simInfo) {
      for (comm = groupCommHeadMain[ncclGroupTaskTypeCollective]; comm != nullptr;
           comm = comm->groupNext[ncclGroupTaskTypeCollective]) {
        if (comm->ceColl.initialized || ncclIntruQueueEmpty(&comm->planner.collCeTaskQueue)) continue;
        ncclCeInitJob* job;
        NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
        job->base.func = ncclCeInitFunc;
        job->base.destructor = free;
        job->base.state = ncclGroupJobRunning;
        job->base.abortFlag = comm->abortFlag;
        job->base.abortFlagDev = comm->abortFlagDev;
        job->base.comm = comm;
        job->comm = comm;
        ncclIntruQueueEnqueue(&asyncCollJobs, &job->base);
      }
      NCCLCHECKGOTO(asyncJobLaunch(&asyncCollJobs, groupAbortFlag), ret, fail);
      while (!ncclIntruQueueEmpty(&asyncCollJobs)) {
        ncclAsyncJob* job = ncclIntruQueueDequeue(&asyncCollJobs);
        if (job->destructor) job->destructor(job);
      }
    }
  }

  if ((!simInfo) && groupCommHeadMain[ncclGroupTaskTypeCollective] != nullptr) {
    NCCLCHECKGOTO(doLaunches(groupCommHeadMain[ncclGroupTaskTypeCollective], groupAbortFlag), ret, fail);
  }

  while (!ncclIntruQueueEmpty(asyncJobsMain)) {
    struct ncclAsyncJob* job = ncclIntruQueueDequeue(asyncJobsMain);
    /* destroy jobs may have freed their comm; do not touch it. */
    if (!job->destroyFlag && job->comm && !job->comm->config.blocking)
      (void) ncclCommSetAsyncError(job->comm, ret);
    if (job->destructor) job->destructor((void*)job);
  }

  for (int type = 0; type < ncclGroupTaskTypeNum; ++type) {
    while (groupCommHeadMain[type] != nullptr) {
      ncclComm_t comm = groupCommHeadMain[type];
      ncclComm_t next = comm->groupNext[type];
      // Poll for callbacks sent to us from other threads. Typically these free
      // resources from to our memory pools and UB
      NCCLCHECKGOTO(ncclCommPollCallbacks(comm, /*waitSome=*/false), ret, fail);
      (void) ncclGroupCommLeave(comm, type);
      if (!comm->config.blocking) {
        (void) ncclCommSetAsyncError(comm, ret);
      }
      groupCommHeadMain[type] = next;
    }
  }

  HGGCCHECK(hggcSetDevice(savedDev));

exit:
  return ret;
fail:
  __atomic_store_n(groupAbortFlag, true, __ATOMIC_RELEASE);
  for (ncclComm_t comm = groupCommHeadMain[ncclGroupTaskTypeCollective]; comm != nullptr;
       comm = comm->groupNext[ncclGroupTaskTypeCollective]) {
    __atomic_store_n(comm->abortFlag, 1, __ATOMIC_RELEASE);
    __atomic_store_n(comm->abortFlagDev, 1, __ATOMIC_RELEASE);
    for (ncclComm_t local = comm->intraComm0; local != nullptr; local = local->intraNext) {
      __atomic_store_n(local->abortFlag, 1, __ATOMIC_RELEASE);
      __atomic_store_n(local->abortFlagDev, 1, __ATOMIC_RELEASE);
    }
  }
  for (auto* jobs : {&asyncSymJobs, &asyncCollJobs, &asyncDebugJobs}) {
    while (!ncclIntruQueueEmpty(jobs)) {
      struct ncclAsyncJob* job = ncclIntruQueueDequeue(jobs);
      if (job->undo) job->undo(job);
      if (job->destructor) job->destructor((void*)job);
    }
  }
  groupCleanup(gjob->groupCommHead, &gjob->asyncJobs, ret);
  goto exit;
}

static ncclResult_t groupLaunchNonBlocking(struct ncclAsyncJob *job_) {
  return groupLaunch(job_ /* estimatedTime = NULL */);
}

ncclResult_t ncclGroupEndInternal(ncclSimInfo_t* simInfo) {
  ncclResult_t ret = ncclSuccess;
  ncclSimInfo_t internalSimInfo = PCCL_SIM_INFO_INITIALIZER;
  ncclSimInfo_t* internalSimInfoPtr = NULL;
  size_t realSize = 0;
  ncclGroupJob* groupJob = NULL;
  bool hasCommHead = false;
  internalSimInfo.magic = 0;

  if (ncclGroupDepth == 0) {
    ERROR("pcclGroupEnd: not in a group call.");
    ret = ncclInvalidUsage;
    goto exit;
  }

  if (ncclProfilerApiState.profilerGroupDepth > 0) {
    ncclProfilerApiState.profilerGroupDepth--;
  }
  if (ncclProfilerApiState.profilerGroupDepth == 0) {
    NCCLCHECK(ncclProfilerRecordGroupApiEventState(ncclProfilerGroupEndApiStart));
  }

  if ((--ncclGroupDepth) > 0) goto exit;

  if ((ret = ncclGroupError) != ncclSuccess) goto fail;

  if (simInfo) {
    memcpy((void*)&realSize, (void*)&simInfo->size, sizeof(size_t));
    realSize = realSize > sizeof(ncclSimInfo_t) ? sizeof(ncclSimInfo_t) : realSize;
    memcpy((void*)&internalSimInfo, (void*)simInfo, realSize);
    if (internalSimInfo.magic != 0x74685283) {
      WARN("pcclSimInfo_t argument not initialized via PCCL_SIM_INFO_INITIALIZER");
      ret = ncclInvalidArgument;
      goto fail;
    }
    internalSimInfoPtr = &internalSimInfo;
  }

  for (int type = 0; type < ncclGroupTaskTypeNum; ++type) {
    if (ncclGroupCommHead[type]) {
      hasCommHead = true;
      break;
    }
  }

  NCCLCHECKGOTO(ncclCalloc(&groupJob, 1), ret, fail);
  ncclIntruQueueConstruct(&groupJob->asyncJobs);
  groupJob->groupRefCount = 0;
  groupJob->nonBlockingInit = false;
  memcpy(groupJob->groupCommHead, ncclGroupCommHead, sizeof(ncclGroupCommHead));
  groupJob->groupCommPreconnectHead = ncclGroupCommPreconnectHead;
  groupJob->groupError = ncclSuccess;
  groupJob->abortFlag = false;
  groupJob->joined = false;
  ncclIntruQueueTransfer(&groupJob->asyncJobs, &ncclAsyncJobs);

  if (hasCommHead || !ncclIntruQueueEmpty(&groupJob->asyncJobs) || ncclGroupCommPreconnectHead != nullptr) {
    /* make sure ncclGroupBlocking has been set. */
    if (ncclGroupBlocking != 0 && ncclGroupBlocking != 1) {
      ERROR("Invalid group blocking state %d", ncclGroupBlocking);
      ret = ncclInternalError;
      goto fail;
    }
    if (ncclGroupBlocking == 0) {
      /* nonblocking group */
      if (!ncclIntruQueueEmpty(&groupJob->asyncJobs)) {
        ncclAsyncJob* job = ncclIntruQueueHead(&groupJob->asyncJobs);
        do {
          NCCLCHECKGOTO(ncclCommSetAsyncError(job->comm, ncclInProgress), ret, fail);
          if (job->comm->groupJob == NULL) {
            job->comm->groupJob = groupJob;
            groupJob->groupRefCount++;
          }
          job = job->next;
        } while (job);
      }

      for (int type = 0; type < ncclGroupTaskTypeNum; ++type) {
        if (ncclGroupCommHead[type]) {
          ncclComm_t comm = ncclGroupCommHead[type];
          do {
            NCCLCHECKGOTO(ncclCommSetAsyncError(comm, ncclInProgress), ret, fail);
            /* link group job to communicators. */
            if (comm->groupJob == NULL) {
              comm->groupJob = groupJob;
              groupJob->groupRefCount++;
            }
            comm = comm->groupNext[type];
          } while (comm);
        }
      }

      groupJob->base.func = groupLaunchNonBlocking;
      SYSCHECKGOTO(pthread_create(&groupJob->base.thread, NULL, ncclAsyncJobMain, (void*)&groupJob->base), "pcclAsyncJobMain", ret, fail);
      groupJob->nonBlockingInit = true;
      ret = ncclInProgress;
    } else {
      /* blocking group */
      NCCLCHECKGOTO(groupLaunch(&groupJob->base, internalSimInfoPtr), ret, fail);
      if (simInfo) memcpy((void*)simInfo, (void*)internalSimInfoPtr, realSize);
      ncclFree(groupJob);
    }
  } else {
    ncclFree(groupJob);
  }

  /* Reset the job state for the next group call. */
  groupLocalResetJobState();

exit:
  // coverity[resource_leak]
  // Profiler group API start is called inside taskAppend to get graph capture information for the event
  NCCLCHECK(ncclProfilerStopGroupApiEvent());
  return ret;
fail:
  if (groupJob) {
    groupCleanup(groupJob->groupCommHead, &groupJob->asyncJobs, ret);
    ncclFree(groupJob);
  } else {
    groupCleanup(ncclGroupCommHead, &ncclAsyncJobs, ret);
  }
  groupLocalResetJobState();
  goto exit;
}

ncclResult_t ncclGroupJobComplete(struct ncclGroupJob* groupJob) {
  ncclResult_t ret = ncclSuccess;
  if (groupJob && groupJob->nonBlockingInit) {
    if (ncclAtomicRefCountDecrement(&groupJob->groupRefCount) == 0) {
      if (!__atomic_exchange_n(&groupJob->joined, true, __ATOMIC_ACQ_REL)) {
        ret = ncclAsyncJobComplete(&groupJob->base);
        ncclFree(groupJob);
      }
    }
  }
  return ret;
}

ncclResult_t ncclGroupJobAbort(struct ncclGroupJob* groupJob) {
  if (groupJob && groupJob->nonBlockingInit) {
    if (!__atomic_exchange_n(&groupJob->joined, true, __ATOMIC_ACQ_REL)) {
      __atomic_store_n(&groupJob->abortFlag, true, __ATOMIC_RELAXED);
      ncclAsyncJobComplete(&groupJob->base);
    }
    if (ncclAtomicRefCountDecrement(&groupJob->groupRefCount) == 0) {
      ncclFree(groupJob);
    }
  }
  return ncclSuccess;
}
