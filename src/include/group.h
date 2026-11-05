/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_GROUP_H_
#define NCCL_GROUP_H_

#include "pccl_internal.h"
#include "comm.h"
#include "allocator.h"

bool ncclAsyncMode();
void ncclGroupCommJoin(ncclComm_t comm, int type);
void ncclGroupCommPreconnect(ncclComm_t comm);
ncclResult_t ncclGroupJobAbort(struct ncclGroupJob *job);
ncclResult_t ncclGroupJobComplete(struct ncclGroupJob *job);

ncclResult_t ncclGroupErrCheck(ncclResult_t ret);
ncclResult_t ncclGroupCommLeave(ncclComm_t comm, int type);

typedef enum ncclGroupJobState {
  ncclGroupJobRunning = 0,
  ncclGroupJobDone    = 1,
  ncclGroupJobJoined  = 2,
} ncclGroupJobState_t;

struct ncclAsyncJob {
  struct ncclAsyncJob* next;
  pthread_t thread;
  ncclResult_t result;
  ncclResult_t(*func)(struct ncclAsyncJob*);
  void(*undo)(struct ncclAsyncJob*);
  void(*destructor)(void*);
  ncclGroupJobState_t state;
  uint32_t *abortFlag; /* point to comm abortFlag */
  uint32_t *abortFlagDev; /* point to comm abortFlagDev */
  uint32_t* childAbortFlag; /* point to child abortFlag */
  uint32_t* childAbortFlagDev; /* point to child abortFlagDev */
  ncclComm_t comm;
  int destroyFlag;
  bool isThreadMain;
};

ncclResult_t ncclAsyncLaunch(
  struct ncclAsyncJob* job,
  ncclResult_t(*func)(struct ncclAsyncJob*),
  void(*undo)(struct ncclAsyncJob*),
  void(*destructor)(void*), ncclComm_t comm
);

struct ncclGroupJob {
  struct ncclAsyncJob base;
  int groupRefCount;
  bool nonBlockingInit;
  bool joined;
  ncclComm_t groupCommHead[ncclGroupTaskTypeNum];
  ncclComm_t groupCommPreconnectHead;
  ncclResult_t groupError;
  bool abortFlag;
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> asyncJobs;
};

ncclResult_t ncclGroupStartInternal();
ncclResult_t ncclGroupEndInternal(ncclSimInfo_t* simInfo = NULL);
ncclResult_t ncclAsyncJobComplete(struct ncclAsyncJob* job);

////////////////////////////////////////////////////////////////////////////////
extern thread_local int ncclGroupDepth; // depth of ncclGroupStart nesting
extern thread_local ncclResult_t ncclGroupError;
extern thread_local ncclComm_t ncclGroupCommHead[ncclGroupTaskTypeNum];
extern thread_local ncclComm_t ncclGroupCommPreconnectHead;
extern thread_local int ncclGroupBlocking;

inline ncclResult_t ncclGroupStartInternal() {
  ncclGroupDepth++;
  return ncclSuccess;
}

inline ncclResult_t ncclGroupErrCheck(ncclResult_t ret) {
  if (ncclGroupDepth > 0) {
    if (ret != ncclSuccess && ret != ncclInProgress) ncclGroupError = ret;
  }
  return ret;
}

// Add comm to this thread's group
inline void ncclGroupCommJoin(ncclComm_t comm, int type) {
  if (comm->groupNext[type] == reinterpret_cast<ncclComm_t>(0x1)) {
    // Insert comm into ncclGroupCommHead adjacent to sibling comms. This preserves
    // the users program order yet insures siblings occur consecutively. This
    // is required by doLaunches() in "group.cc".
    ncclComm_t* pp = &ncclGroupCommHead[type];
    while (*pp != nullptr && comm->intraComm0 != (*pp)->intraComm0)
      pp = &(*pp)->groupNext[type];
    comm->groupNext[type] = *pp;
    *pp = comm;
    // Comms gets a new memory stack scope upon joining. Each task batched for
    // this comm is allocated there.
    ncclMemoryStackPush(&comm->memScoped);
    if (type == ncclGroupTaskTypeCollective) {
      // Initialize planner. peers and rmaTaskQueues are permanent allocations, so
      // they survive the memset.
      ncclKernelPlanner::Peer* tmp = comm->planner.peers;
      ncclIntruQueue<struct ncclTaskRma, &ncclTaskRma::next>* tmpRma = comm->planner.rmaTaskQueues;
      memset(&comm->planner, 0, sizeof(comm->planner));
      comm->planner.peers = tmp;
      comm->planner.rmaTaskQueues = tmpRma;
      if (tmpRma != NULL) {
        for (int c = 0; c < comm->config.numRmaCtx; c++) ncclIntruQueueConstruct(&tmpRma[c]);
      }
    }
  }

  ncclGroupBlocking = comm->config.blocking;
}

// Add comm to this thread's group needing preconnect
inline void ncclGroupCommPreconnect(ncclComm_t comm) {
  if (comm->preconnectNext == reinterpret_cast<ncclComm_t>(0x1)) {
    comm->preconnectNext = ncclGroupCommPreconnectHead;
    ncclGroupCommPreconnectHead = comm;
  }
}

// Comm has left group
inline ncclResult_t ncclGroupCommLeave(ncclComm_t comm, int type) {
  comm->groupNext[type] = reinterpret_cast<ncclComm_t>(0x1);
  ncclMemoryStackPop(&comm->memScoped);
  return ncclSuccess;
}


#endif
