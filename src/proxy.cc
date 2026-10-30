/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "info.h"
#include "collectives.h"
#include "socket.h"
#include "shm.h"
#include "transport.h"
#include "profiler.h"
#include "alloc.h"
#include "drv_extension.h"
#define ENABLE_TIMER 0
#include "timer.h"

static bool NeedProxy(int type, int pattern, int root, struct ncclRing* ring, int nranks) {
  if (pattern == ncclPatternRing || pattern == ncclPatternRingTwice) return true;

  /* In chains, one rank does not need a proxy. Let's figure out which one it is */
  /* Which index in the reorganized rings should we compare root against */
  const int myrank = 0, nextrank = 1, prevrank = nranks-1;
  int index = pattern == ncclPatternPipelineFrom ?
      /*                            no recv /  no send    if root = */
      /* bcast  */ (type == proxyRecv ?   myrank : nextrank ):
      /* reduce */ (type == proxyRecv ? prevrank :   myrank );
  int rank = ring->userRanks[index];
  return (root != rank);
}

static void expectedProxyResponseFree(struct ncclProxyState* state) {
  struct ncclExpectedProxyResponse* elem = state->expectedResponses;
  struct ncclExpectedProxyResponse* prev = NULL;

  while (elem) {
    prev = elem;
    elem = elem->next;
    if (prev && prev->respBuff) free(prev->respBuff);
    if (prev) free(prev);
  }
}

static ncclResult_t expectedProxyResponseStore(struct ncclProxyState* state, void* opId, void* respBuff, int respSize, ncclResult_t res) {
  struct ncclExpectedProxyResponse* elem = state->expectedResponses;
  while (elem) {
    if (elem->opId == opId) {
      if (respSize != elem->respSize) {
        ERROR("Mismatched response size for opId=%p: got %d, expected %d", opId, respSize, elem->respSize);
        return ncclInternalError;
      }

      if (elem->done) {
        ERROR("Storing response for already completed opId=%p", opId);
        return ncclInternalError;
      }

      if (respSize > 0) {
        memcpy(elem->respBuff, respBuff, respSize);
        NCCLCHECK(ncclFree(respBuff));
      }
      elem->done = true;
      elem->res  = res;
      return ncclSuccess;
    }
    elem = elem->next;
  }

  ERROR("Proxy response for opId=%p doesn't match any expected response", opId);
  return ncclInternalError;
}

static ncclResult_t expectedProxyResponseEnqueue(struct ncclProxyState* state, void* opId, int respSize) {
  struct ncclExpectedProxyResponse* ex;
  NCCLCHECK(ncclCalloc(&ex, 1));
  ex->opId = opId;

  // Pre-alloc response buffer
  ex->respBuff = malloc(respSize);
  ex->respSize = respSize;
  ex->res      = ncclInternalError;
  ex->done     = false;

  // Enqueue
  struct ncclExpectedProxyResponse* list = state->expectedResponses;
  if (list == NULL) {
    state->expectedResponses = ex;
    return ncclSuccess;
  }
  while (list->next) list = list->next;
  list->next = ex;
  return ncclSuccess;
}

static ncclResult_t expectedProxyResponseDequeue(struct ncclProxyState* state, void* opId, void* respBuff, int* found) {
  struct ncclExpectedProxyResponse* elem = state->expectedResponses;
  struct ncclExpectedProxyResponse* prev = NULL;
  *found = 0;
  while (elem) {
    if ((elem->opId == opId) && elem->done) {
      if (prev == NULL) {
        state->expectedResponses = elem->next;
      } else {
        prev->next = elem->next;
      }
      memcpy(respBuff, elem->respBuff, elem->respSize);
      ncclResult_t res = elem->res;
      NCCLCHECK(ncclFree(elem->respBuff));
      NCCLCHECK(ncclFree(elem));
      *found = 1;
      return res;
    }
    prev = elem;
    elem = elem->next;
  }
  return ncclSuccess;
}

static ncclResult_t expectedProxyResponseRemove(struct ncclProxyState* state, void* opId) {
  struct ncclExpectedProxyResponse* elem = state->expectedResponses;
  struct ncclExpectedProxyResponse* prev = NULL;
  while (elem) {
    if (elem->opId == opId) {
      if (prev == NULL) {
        state->expectedResponses = elem->next;
      } else {
        prev->next = elem->next;
      }
      NCCLCHECK(ncclFree(elem->respBuff));
      NCCLCHECK(ncclFree(elem));
      return ncclSuccess;
    }
    prev = elem;
    elem = elem->next;
  }
  ERROR("Couldn't find opId=%p", opId);
  return ncclInternalError;
}

static ncclResult_t asyncProxyOpEnqueue(struct ncclProxyLocalPeer* peer, ncclProxyAsyncOp* op) {
  ncclProxyAsyncOp* list = peer->asyncOps;
  if (list == NULL) {
    peer->asyncOps = op;
    return ncclSuccess;
  }
  while (list->next) list = list->next;
  list->next = op;
  return ncclSuccess;
}

static ncclResult_t asyncProxyOpDequeue(struct ncclProxyLocalPeer* peer, ncclProxyAsyncOp* op) {
  struct ncclProxyAsyncOp* elem = peer->asyncOps;
  struct ncclProxyAsyncOp* prev = NULL;
  while (elem) {
    if (elem->opId == op->opId) {
      if (prev == NULL) {
        peer->asyncOps = elem->next;
      } else {
        prev->next = elem->next;
      }

      if (elem->reqBuff) {
        NCCLCHECK(ncclFree(elem->reqBuff));
      }
      if (elem->respBuff) {
        NCCLCHECK(ncclFree(elem->respBuff));
      }
      NCCLCHECK(ncclFree(elem));

      return ncclSuccess;
    }
    prev = elem;
    elem = elem->next;
  }
  if (op) {
    WARN("Attempting to dequeue nonexistent async opId=%p", op->opId);
  } else {
    WARN("Attempting to dequeue null operation");
  }
  return ncclInternalError;
}

#define PROXYARGS_ALLOCATE_SIZE NCCL_MAX_OPS
struct ncclProxyPool {
  struct ncclProxyPool *next;
  struct ncclProxyArgs elems[PROXYARGS_ALLOCATE_SIZE];
};

static ncclResult_t allocateArgs(struct ncclProxyProgressState* state, struct ncclProxyArgs** argsptr) {
  struct ncclProxyArgs* elem;
  if (state->pool == NULL) {
    // Allocate a new pool of elements. Make sure we allocate the memory close
    // to the network thread
    struct ncclProxyPool* newPool;
    NCCLCHECK(ncclCalloc(&newPool, 1));

    struct ncclProxyArgs* newElems = newPool->elems;
    // Chain newly allocated elements
    for (int i=0; i<PROXYARGS_ALLOCATE_SIZE; i++) {
      if (i+1 < PROXYARGS_ALLOCATE_SIZE) newElems[i].next = newElems+i+1;
    }
    // Add them all to the pool list
    state->pool = newElems;
    // Save the pool memory block for later resource release
    newPool->next = state->pools;
    state->pools = newPool;
  }
  elem = state->pool;
  state->pool = state->pool->next;
  elem->next = elem->nextPeer = NULL;
  *argsptr = elem;
  return ncclSuccess;
}

//#define DEBUG_PROXY 1
#ifdef DEBUG_PROXY
#define DEBUG_PROXY_PRINT printf
#else
#define DEBUG_PROXY_PRINT(...)
#endif

#define OP_INDEX(op) ((op) ? (op)-state->pools->elems : -1)
#define OP_SEEN 0x100000

ncclResult_t getOpIndex(struct ncclProxyArgs* op, struct ncclProxyProgressState* state, int* poolIndex, int* opIndex) {
  struct ncclProxyPool* pool = state->pools;
  int p = 0;
  while (pool) {
    uint64_t o = op-pool->elems;
    if (o < PROXYARGS_ALLOCATE_SIZE) {
      *opIndex = o;
      *poolIndex = p;
      return ncclSuccess;
    }
    pool = pool->next;
    p++;
  }
  WARN("Could not find pool of op %p\n", op);
  return ncclInternalError;
}

ncclResult_t printProxyOp(FILE *dumpFile, struct ncclProxyArgs* op, int poolIndex, int opIndex) {
  fprintf(dumpFile, "[%d-%d|%ld| %s", poolIndex, opIndex, op->opCount, op->pattern == ncclPatternSend ? "Send" : op->pattern == ncclPatternRecv ? "Recv" : "Coll");
  for (int s=0; s<op->nsubs; s++) {
    struct ncclProxySubArgs* sub = op->subs+s;
    if (op->state == ncclProxyOpProgress) {
      char status = ' ';
      if (sub->sendrecv == proxyRecv) {
        if (sub->posted < sub->nsteps && sub->posted < sub->done + NCCL_STEPS) status = 'I'; // Init
        else if (sub->received < sub->posted) status = 'R'; // Receiving
        else if (sub->received < sub->transmitted) status = 'R'; // Receiving
        else if (sub->transmitted < sub->received) status = 'F'; // Flushing
        else if (sub->done < sub->transmitted) status = 'G'; // Waiting on GPU
        else status = 'D'; // Done
      } else if (sub->sendrecv == proxySend) {
        if (sub->posted < sub->nsteps && sub->posted < sub->done + NCCL_STEPS) status = 'I'; // Init
        else if (sub->transmitted < sub->posted) status = 'G'; // Waiting on GPU
        else if (sub->done < sub->transmitted) status = 'S'; // Sending
        else status = 'D'; // Done
      }
      fprintf(dumpFile, " %d%c/%d", sub->peer, status, sub->channelId);
    } else {
      fprintf(dumpFile, " %d/%d", sub->peer, sub->channelId);
    }
  }
  fprintf(dumpFile, "]");
  return ncclSuccess;
}
ncclResult_t pcclDumpProxyState(struct ncclProxyState* proxyState, FILE *dumpFile) {
  ncclResult_t res = ncclSuccess;
  struct ncclProxyProgressState* state = &proxyState->progressState;
  struct ncclProxyArgs* op = state->active;
  if (!op) {
    INFO(NCCL_DUMP, "comm %lx, rank %d, empty proxy state", proxyState->ownerComm->commHash, proxyState->ownerComm->rank);
    return ncclSuccess;
  }
  if (!dumpFile) {
    int offset = 0;
    char filename[PATH_MAX+1] = {0};
    const char *str = pcclGetEnvWithDualPrefix("PCCL_PROXY_DUMP_FILE");

    if (createOrFillDumpDir(filename, PATH_MAX, "proxy_dump", &offset) != ncclSuccess) {
      WARN("comm %lx, rank %d, error creating proxy dump file", proxyState->ownerComm->commHash, proxyState->ownerComm->rank);
      return ncclInternalError;
    }
    if (str) {
      offset += fillConfigName(filename+offset, PATH_MAX-offset, proxyState->ownerComm->commHash, proxyState->ownerComm->rank, proxyState->ownerComm->nRanks, proxyState->ownerComm->hggcDev, proxyState->ownerComm->hgmlDev, str, "log");
    } else {
      offset += fillConfigName(filename+offset, PATH_MAX-offset, proxyState->ownerComm->commHash, proxyState->ownerComm->rank, proxyState->ownerComm->nRanks, proxyState->ownerComm->hggcDev, proxyState->ownerComm->hgmlDev, "pccl_proxy_dump", "log");
    }
    dumpFile = fopen(filename, "w");
    if (!dumpFile) {
      WARN("comm %lx, rank %d, error opening proxy dump file", proxyState->ownerComm->commHash, proxyState->ownerComm->rank);
      return ncclInternalError;
    }
    INFO(NCCL_DUMP, "using proxy dump file %s", filename);
  }

  int poolIndex, opIndex;
  struct ncclProxyArgs* nextOp;
  fprintf(dumpFile, "ACTIVE OPS\n");
  while (op) {
    NCCLCHECKGOTO(getOpIndex(op, state, &poolIndex, &opIndex), res, end);
    if (op->state & OP_SEEN) {
      WARN("List loop at element %d-%d", poolIndex, opIndex);
    }
    NCCLCHECKGOTO(printProxyOp(dumpFile, op, poolIndex, opIndex), res, end);
    op->state |= OP_SEEN;
    fprintf(dumpFile, "\n");
    nextOp = op->nextPeer;
    while (nextOp) {
      NCCLCHECKGOTO(getOpIndex(nextOp, state, &poolIndex, &opIndex), res, end);
      if (nextOp->state & OP_SEEN) {
        WARN("List loop at element %d-%d", poolIndex, opIndex);
      }
      fprintf(dumpFile, "| `-> ");
      NCCLCHECKGOTO(printProxyOp(dumpFile, nextOp, poolIndex, opIndex), res, end);
      nextOp->state |= OP_SEEN;
      fprintf(dumpFile, "\n");
      if (nextOp->next) {
        WARN("Inactive op has next set!\n");
      }
      nextOp = nextOp->nextPeer;
    }
    if (op->nextPeer == NULL) fprintf(dumpFile, "|\n");
    op = op->next;
    fprintf(dumpFile, "v\n");
  }
  fprintf(dumpFile, "[X]\n");

# if 0
  fprintf(dumpFile, "FREE OPS\n");
  op = state->pool;
  while (op) {
    NCCLCHECKGOTO(getOpIndex(op, state, &poolIndex, &opIndex), res, end);
    if (op->state & OP_SEEN) {
      WARN("List loop at element %d-%d", poolIndex, opIndex);
    }
    NCCLCHECKGOTO(printProxyOp(dumpFile, op, poolIndex, opIndex), res, end);
    op->state |= OP_SEEN;
    fprintf(dumpFile, "->");
    op = op->next;
  }
  fprintf(dumpFile, "[X]\n");
#else
  op = state->pool;
  while (op) {
    NCCLCHECKGOTO(getOpIndex(op, state, &poolIndex, &opIndex), res, end);
    if (op->state & OP_SEEN) {
      WARN("List loop at element %d-%d", poolIndex, opIndex);
    }
    op->state |= OP_SEEN;
    op = op->next;
  }
#endif

  struct ncclProxyPool* pool;
  pool = state->pools;
  poolIndex = 0;
  while (pool) {
    struct ncclProxyArgs* elem = pool->elems;
    for (int e=0; e<PROXYARGS_ALLOCATE_SIZE; e++, elem++) {
      if ((elem->state & OP_SEEN) == 0) {
        fprintf(dumpFile, "Elem %d-%d is not in any list:\n", poolIndex, e);
        NCCLCHECKGOTO(printProxyOp(dumpFile, elem, poolIndex, e), res, end);
        fprintf(dumpFile, "\n");
      } else {
        elem->state -= OP_SEEN;
      }
    }
    pool = pool->next;
    poolIndex++;
  }

end:
  INFO(NCCL_DUMP, "comm %lx, rank %d, proxy dump finished", proxyState->ownerComm->commHash, proxyState->ownerComm->rank);
  if (dumpFile) {
    fclose(dumpFile);
  }
  return ncclSuccess;
}

static ncclResult_t ncclProxyOpToArgs(struct ncclProxyOp* op, struct ncclProxyArgs* args, int subIndex) {
  struct ncclProxySubArgs* sub = args->subs+subIndex;
  if (subIndex >= NCCL_PROXY_MAX_SUBS) {
    WARN("Proxy append out of bounds");
    return ncclInternalError;
  }

  //memset(sub, 0, sizeof(struct ncclProxySubArgs));
  sub->connection = op->connection;
  sub->channelId = op->channelId;
  sub->a2aOptimized = op->a2aOptimized;
  sub->a2aAutoAsyncFlush = op->a2aAutoAsyncFlush;
  sub->a2aSingleQp = op->a2aOptimized ? op->a2aSingleQp : 0;
  sub->a2aAsyncFlush = 0; // Pooled arguments must not inherit the previous task.
  // Only AlltoAll P2P sends produce a preselected window; other operations
  // retain the original depth and never read the optional field.
  sub->a2aSendWindow = op->a2aOptimized && op->pattern == ncclPatternSend
      ? op->a2aSendWindow : NCCL_STEPS;
  sub->nsteps = op->nsteps;
  sub->nbytes = op->nbytes;
  sub->peer = op->peer;
  sub->sendrecv = op->sendrecv;
  args->nsubs = subIndex+1;
  sub->sendbuff = op->sendbuff;
  sub->recvbuff = op->recvbuff;
  sub->eActivationMask = op->eActivationMask;
  sub->taskEventHandle = op->taskEventHandle;
  sub->rank = op->rank;
  sub->pid = op->pid;
  sub->profilerContext = op->profilerContext;
  sub->workCounter = op->workCounter;
  if (subIndex) {
    if ((args->sliceSteps != op->sliceSteps) ||
        (args->chunkSteps != op->chunkSteps) ||
        (args->protocol != op->protocol) ||
        (args->dtype != op->dtype) ||
        (args->redOp != op->redOp)) {
      ERROR("Proxy append mismatch");
      return ncclInternalError;
    }
    if (args->state != ncclProxyOpReady) {
      ERROR("Proxy append on running operation");
      return ncclInternalError;
    }
    goto exit;
  }
  //memset(&args->progress, 0, sizeof(struct ncclProxyArgs)-offsetof(struct ncclProxyArgs, progress));
  args->perfComm = op->perfComm;
  args->done = 0;
  args->opCount = op->opCount;
  args->sliceSteps = op->sliceSteps;
  args->chunkSteps = op->chunkSteps;
  args->chunkSize = op->chunkSize;
  args->dtype = op->dtype;
  args->redOp = op->redOp;
  args->pattern = op->pattern;
  args->protocol = op->protocol;
  args->state = ncclProxyOpReady;
  args->progress = op->connection->tcomm->proxyProgress;
  args->proxyAppendPtr = op->connection->proxyAppendPtr;
exit:
  if (args->pattern != ncclPatternProfiler) ncclProfilerStartProxyOpEvent(subIndex, args);
  return ncclSuccess;
}

static ncclResult_t ProxyAppend(struct ncclProxyProgressState* state, struct ncclProxyOp* op) {
  struct ncclProxyConnection* connection = op->connection;
  int shared = connection->shared;
  struct ncclProxyArgs* args = *connection->proxyAppendPtr;

  if (args) {
    if (shared && args->opCount == op->opCount) {
      NCCLCHECK(ncclProxyOpToArgs(op, args, args->nsubs));
      DEBUG_PROXY_PRINT("Insert (%d/%5ld/%5ld) as group with %5ld\n", shared, args->opCount, op->opCount, OP_INDEX(args));
    } else {
      struct ncclProxyArgs* prevArgs = args;
      NCCLCHECK(allocateArgs(state, &args));
      NCCLCHECK(ncclProxyOpToArgs(op, args, 0));
      prevArgs->nextPeer = args;
      DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld/%5ld) as nextPeer of %5ld\n", OP_INDEX(args), shared, prevArgs->opCount, args->opCount, OP_INDEX(prevArgs));
      *(args->proxyAppendPtr) = args;
    }
  } else {
    // Nothing running for that peer. Add to the list
    NCCLCHECK(allocateArgs(state, &args));
    NCCLCHECK(ncclProxyOpToArgs(op, args, 0));
    if (state->active == NULL) {
      // Create the list
      DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld) as first element\n", OP_INDEX(args), shared, args->opCount);
      state->active = args;
    } else {
      // Append element at the end of the list
      struct ncclProxyArgs* last = state->active;
      while (last->next) last = last->next;
      last->next = args;
      DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld) as last element\n", OP_INDEX(args), shared, args->opCount);
    }
    *(args->proxyAppendPtr) = args;
  }
  return ncclSuccess;
}

ncclResult_t ncclProxyPost(struct ncclProxyOpsPool* pool, int nextOps, int nextOpsEnd) {
  pthread_mutex_lock(&pool->mutex);
  if (pool->nextOps == -1) {
    pool->nextOps = nextOps;
    pthread_cond_signal(&pool->cond);
  } else {
    pool->ops[pool->nextOpsEnd].next = nextOps;
  }
  pool->nextOpsEnd = nextOpsEnd;
  pthread_mutex_unlock(&pool->mutex);
  return ncclSuccess;
}

ncclResult_t ncclLocalOpAppend(ncclComm_t comm, struct ncclProxyConnector* proxyConn, struct ncclProxyOp* proxyOp) {
  int tpLocalRank = comm->topParentLocalRanks[comm->localRank];
  struct ncclProxyOps* proxyOps = comm->proxyState->proxyOps;
  if (proxyOps == NULL) return ncclInternalError;
  proxyOps += proxyConn->tpLocalRank;
  struct ncclProxyOpsPool* pool = proxyOps->pool;
  TIME_START(0);
  int opIndex = proxyOps->freeOp;
  struct ncclProxyOp* op;
  if (opIndex != -1) {
    op = pool->ops+opIndex;
    proxyOps->freeOp = op->next;
  } else {
    int freeOp;
    while ((freeOp = pool->freeOps[tpLocalRank]) == -1) sched_yield();
    int freeOpNew;
    while ((freeOpNew = __sync_val_compare_and_swap(pool->freeOps+tpLocalRank, freeOp, -1)) != freeOp) freeOp = freeOpNew;
    opIndex = freeOp;
    op = pool->ops+opIndex;
    proxyOps->freeOp = op->next;
  }
  if (op->next != -1) __builtin_prefetch(pool->ops+op->next); // Prefetch next free op
  memcpy(op, proxyOp, sizeof(struct ncclProxyOp));
  op->next = -1;
  op->connection = proxyConn->connection;
  if (proxyOps->nextOps == -1) {
    proxyOps->nextOps = proxyOps->nextOpsEnd = opIndex;
  } else {
    pool->ops[proxyOps->nextOpsEnd].next = opIndex;
    proxyOps->nextOpsEnd = opIndex;
  }
  if (++proxyOps->count == MAX_OPS_PER_PEER) {
    // Post what we have so far to free some ops in the pool
    // Do not post last operations as we could have more coming with the same opCount, and posting
    // them in different batches would break proxyArgs aggregation with subs.
    uint64_t lastOpCount = pool->ops[proxyOps->nextOpsEnd].opCount;
    int lastOp = -1;
    int toSend = 0;
    int ops = 0;
    for (int op= proxyOps->nextOps; op != proxyOps->nextOpsEnd; op=pool->ops[op].next) {
      ops++;
      if (pool->ops[op].opCount != lastOpCount) {
        lastOp = op;
        toSend = ops;
      }
    }
    if (lastOp == -1) {
      ERROR("Unable to post incomplete proxy op chain %d..%d (opCount %ld)\n", proxyOps->nextOps, proxyOps->nextOpsEnd, lastOpCount);
      return ncclInternalError;
    }
    // Cut chain at lastOp
    int nextOps = proxyOps->nextOps;
    proxyOps->nextOps = pool->ops[lastOp].next;
    pool->ops[lastOp].next = -1;
    NCCLCHECK(ncclProxyPost(proxyOps->pool, nextOps, lastOp));
    proxyOps->count -= toSend;
  }
  TIME_STOP(0);
  return ncclSuccess;
}

static void incWorkCounter(ncclComm_t comm, struct ncclProxyOp* op) {
  op->workCounter = (op->incWorkCounter) ? ++comm->profiler.workCounter[op->channelId] : comm->profiler.workCounter[op->channelId];
}

static ncclResult_t SaveProxyProfiler(ncclComm_t comm, struct ncclProxyOp* op, bool* justInquire) {
  struct ncclProxyConnector* proxyConn = (op->coll == ncclFuncRecv) ? &comm->profiler.recvProxyConn[op->channelId] : &comm->profiler.sendProxyConn[op->channelId];
  if (justInquire) {
    *justInquire = true;
    if (!comm->planner.persistent) incWorkCounter(comm, op);
  } else {
    op->sendbuff = (uint8_t *)comm->profiler.workStarted;
    op->recvbuff = (uint8_t *)comm->profiler.workCompleted;
    // Ensure that in graph capturing the proxy workCounter is incremented to keep up with kernel workCounter
    if (comm->planner.persistent) incWorkCounter(comm, op);
    NCCLCHECK(ncclLocalOpAppend(comm, proxyConn, op));
  }
  return ncclSuccess;
}

static ncclResult_t SaveProxy(ncclComm_t comm, struct ncclChannel* channel, int type, int peer, struct ncclProxyOp* op, int connIndex, bool* justInquire) {
  if (peer < 0) return ncclSuccess;

  struct ncclChannelPeer* peerComm = channel->peers[peer];
  struct ncclConnector* connector = type == proxyRecv ? peerComm->recv + connIndex : peerComm->send + connIndex;
  if (connector->transportComm == NULL) {
    ERROR("Rank %d has no transport for %s peer %d on channel %d/%d", comm->rank,
        type == proxyRecv ? "recv" : "send", peer, channel->id, connIndex);
    return ncclInternalError;
  }
  if (connector->transportComm->proxyProgress == NULL) return ncclSuccess;

  if (justInquire) {
    *justInquire = true;
  } else {
    op->sendrecv = type;
    op->peer = peer;
    NCCLCHECK(ncclLocalOpAppend(comm, &connector->proxyConn, op));
  }
  return ncclSuccess;
}

// justInquire != nullptr means don't actually do anything, just ascertain need of
// ncclProxySaveOp for this op.
ncclResult_t ncclProxySaveOp(ncclComm_t comm, struct ncclProxyOp* op, bool* justInquire) {
  struct ncclChannel* channel = &comm->channels[op->channelId];
  if (justInquire) *justInquire = false;
  switch (op->pattern) {
    case ncclPatternRing:
    case ncclPatternRingTwice:
    case ncclPatternPipelineFrom:
    case ncclPatternPipelineTo: {
      struct ncclRing* ring = &channel->ring;
      if (NeedProxy(proxyRecv, op->pattern, op->root, ring, comm->nRanks))
        NCCLCHECK(SaveProxy(comm, channel, proxyRecv, ring->prev, op, 0, justInquire));
      if (NeedProxy(proxySend, op->pattern, op->root, ring, comm->nRanks))
        NCCLCHECK(SaveProxy(comm, channel, proxySend, ring->next, op, 0, justInquire));
    } break;
    case ncclPatternTreeUp:
    case ncclPatternTreeDown:
    case ncclPatternTreeUpDown: {
      if (op->pattern != ncclPatternTreeDown) { // Tree up
        struct ncclTree* tree = &channel->tree;
        for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) {
          NCCLCHECK(SaveProxy(comm, channel, proxyRecv, tree->down[i], op, 0, justInquire));
        }
        NCCLCHECK(SaveProxy(comm, channel, proxySend, tree->up, op, 0, justInquire));
      }
      if (op->pattern != ncclPatternTreeUp) { // Tree down
        struct ncclTree* tree = &channel->tree;
        for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) {
          NCCLCHECK(SaveProxy(comm, channel, proxySend, tree->down[i], op, 0, justInquire));
        }
        NCCLCHECK(SaveProxy(comm, channel, proxyRecv, tree->up, op, 0, justInquire));
      }
    } break;
    case ncclPatternCollTreeUp: {
      struct ncclTree* tree = &channel->collTree;
      NCCLCHECK(SaveProxy(comm, channel, proxyRecv, tree->down[0], op, 0, justInquire));
      NCCLCHECK(SaveProxy(comm, channel, proxySend, tree->up,      op, 0, justInquire));
    } break;
    case ncclPatternCollTreeDown: {
      struct ncclTree* tree = &channel->collTree;
      NCCLCHECK(SaveProxy(comm, channel, proxySend, tree->down[0], op, 0, justInquire));
      NCCLCHECK(SaveProxy(comm, channel, proxyRecv, tree->up,      op, 0, justInquire));
    } break;
    case ncclPatternPatUp: {
      // Run full algorithm to count the number of steps for each peer.
      ncclResult_t result = ncclSuccess;
      const ssize_t size = op->nbytes/comm->nRanks;
      const int rank = comm->rank, nranks = comm->nRanks;
      int *nstepsSend = NULL, *nstepsRecv = NULL;
      PatRSAlgorithm<char> algo(op->chunkSize, NCCL_STEPS, 16, 0, size, size, op->chunkSize, rank, nranks);
      struct ncclPatStep ps = {0};
      NCCLCHECKGOTO(ncclCalloc(&nstepsSend, log2Up(nranks)), result, exit_pat_up);
      NCCLCHECKGOTO(ncclCalloc(&nstepsRecv, log2Up(nranks)), result, exit_pat_up);

      do {
        algo.getNextOp(&ps);
        if (ps.flags & PatSkipped) continue;
        if (ps.recvDim != -1 && ps.postRecv) nstepsRecv[ps.recvDim]++;
        if (ps.sendDim != -1 && ps.postSend) nstepsSend[ps.sendDim]++;
      } while (ps.last != 2);
      for (int i=0; i<log2Up(nranks); i++) {
        if (nstepsSend[i]) {
          int sendPeer = (rank + (1<<i)) % nranks;
          op->nsteps = nstepsSend[i];
          NCCLCHECKGOTO(SaveProxy(comm, channel, proxySend, sendPeer, op, 0, justInquire), result, exit_pat_up);
        }
        if (nstepsRecv[i]) {
          int recvPeer = (rank - (1<<i) + nranks) % nranks;
          op->nsteps = nstepsRecv[i];
          NCCLCHECKGOTO(SaveProxy(comm, channel, proxyRecv, recvPeer, op, 0, justInquire), result, exit_pat_up);
        }
      }
    exit_pat_up:
      free(nstepsSend);
      free(nstepsRecv);
      NCCLCHECK(result);
    } break;
    case ncclPatternPatDown:{
      // Run full algorithm to count the number of steps for each peer.
      ncclResult_t result = ncclSuccess;
      const ssize_t size = op->nbytes/comm->nRanks;
      const int rank = comm->rank, nranks = comm->nRanks;
      int *nstepsSend = NULL, *nstepsRecv = NULL;
      PatAGAlgorithm<char> algo(op->chunkSize, NCCL_STEPS, 16, 0, size, size, op->chunkSize, rank, nranks);
      struct ncclPatStep ps = {0};
      NCCLCHECKGOTO(ncclCalloc(&nstepsSend, log2Up(nranks)), result, exit_pat_down);
      NCCLCHECKGOTO(ncclCalloc(&nstepsRecv, log2Up(nranks)), result, exit_pat_down);

      do {
        algo.getNextOp(&ps);
        if (ps.flags & PatSkipped) continue;
        if (ps.recvDim != -1 && ps.postRecv) nstepsRecv[ps.recvDim]++;
        if (ps.sendDim != -1 && ps.postSend) nstepsSend[ps.sendDim]++;
      } while (ps.last != 2);
      for (int i=0; i<log2Up(nranks); i++) {
        if (nstepsSend[i]) {
          int sendPeer = (rank - (1<<i) + nranks) % nranks;
          op->nsteps = nstepsSend[i];
          NCCLCHECKGOTO(SaveProxy(comm, channel, proxySend, sendPeer, op, 0, justInquire), result, exit_pat_down);
        }
        if (nstepsRecv[i]) {
          int recvPeer = (rank + (1<<i)) % nranks;
          op->nsteps = nstepsRecv[i];
          NCCLCHECKGOTO(SaveProxy(comm, channel, proxyRecv, recvPeer, op, 0, justInquire), result, exit_pat_down);
        }
      }
    exit_pat_down:
      free(nstepsSend);
      free(nstepsRecv);
      NCCLCHECK(result);
    } break;
    case ncclPatternSend:
    case ncclPatternRecv: {
      if (op->root == comm->rank) return ncclSuccess;
      NCCLCHECK(SaveProxy(comm, channel,
                          op->pattern == ncclPatternSend ? proxySend : proxyRecv,
                          op->root, op, 1, justInquire));
    } break;
    case ncclPatternProfiler: {
      if (ncclProfilerNeedsProxy(comm, op)) NCCLCHECK(SaveProxyProfiler(comm, op, justInquire));
      else incWorkCounter(comm, op);
    } break;
  }
  return ncclSuccess;
}

static ncclResult_t removeOp(struct ncclProxyProgressState* state, struct ncclProxyArgs** opPtr, struct ncclProxyArgs** prevOpPtr) {
  struct ncclProxyArgs* freeOp = *opPtr;
  struct ncclProxyArgs* next = freeOp->next;
  DEBUG_PROXY_PRINT("Remove %ld -> %ld -> %ld\n", OP_INDEX(*prevOpPtr), OP_INDEX(freeOp), OP_INDEX(next));
  *opPtr = next;
  if (freeOp->nextPeer) {
    // replace op by nextPeer
    struct ncclProxyArgs* nextPeer = freeOp->nextPeer;
    if (*prevOpPtr) {
      (*prevOpPtr)->next = nextPeer;
    } else {
      state->active = nextPeer;
    }
    nextPeer->next = next;
    *(prevOpPtr) = nextPeer;
  } else {
    *(freeOp->proxyAppendPtr) = NULL;
    if (*prevOpPtr) {
      (*prevOpPtr)->next = next;
    } else {
      state->active = next;
    }
  }
  freeOp->next = state->pool;
  state->pool = freeOp;
  DEBUG_PROXY_PRINT("Removed %5ld (%5ld) : ", OP_INDEX(freeOp), OP_INDEX(*freeOp->proxyAppendPtr));
  return ncclSuccess;
}

static ncclResult_t progressOps(struct ncclProxyState* proxyState, struct ncclProxyProgressState* state, struct ncclProxyArgs* opStart, int* idle) {
  struct ncclProxyArgs* prevOp = NULL;
  struct ncclProxyArgs* op = opStart;
  while (op) {
    if (op->state == ncclProxyOpNone) return ncclInternalError;
    TIME_START(0); TIME_START(1);
    NCCLCHECK(op->progress(proxyState, op));
    if (op->idle) { TIME_STOP(1); TIME_CANCEL(0); } else { TIME_CANCEL(1); TIME_STOP(0); }
    *idle &= op->idle;
    if (op->state == ncclProxyOpNone) {
      TIME_START(2);
      NCCLCHECK(removeOp(state, &op, &prevOp));
#ifdef DEBUG_PROXY
      pcclDumpProxyState(proxyState, stdout);
#endif
      TIME_STOP(2);
    } else {
      prevOp = op;
      op = op->next;
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclProxyGetPostedOps(struct ncclProxyState* proxyState, int* added) {
  struct ncclProxyProgressState* state = &proxyState->progressState;
  if (state->opsPool == NULL) return ncclInternalError;
  struct ncclProxyOpsPool* pool = state->opsPool;

  struct ncclProxyArgs profArgs;
  profArgs.perfComm = proxyState->ownerComm; // Only used for Asys pccl proxy profiling purposes

  if (state->nextOps != -1) goto process_nextops;

  void* eHandle;
  // If we have ops to progress, no need to block waiting for something to arrive or even wait for the lock
  // to be available. Exit, continue progress, and come back later.
  if (state->active != NULL && (pool->nextOps == -1 || pthread_mutex_trylock(&pool->mutex) != 0)) return ncclSuccess;
  if (state->active == NULL) {
    pthread_mutex_lock(&pool->mutex);
    while (pool->nextOps == -1 && !state->stop) {
      ncclProfilerStartProxyCtrlEvent(proxyState->profilerContext, &eHandle);
      ncclProfilerRecordProxyCtrlEventState(eHandle, 0, ncclProfilerProxyCtrlSleep);
      profArgs.perfComm = proxyState->ownerComm;
      pcclAsysProxyProfilingRecord(&profArgs, 0, 0, ncclProxyProfileSleep);
      pthread_cond_wait(&pool->cond, &pool->mutex);
      pcclAsysProxyProfilingRecord(&profArgs, 0, 0, ncclProxyProfileWakeup);
      ncclProfilerRecordProxyCtrlEventState(eHandle, 0, ncclProfilerProxyCtrlWakeup);
      ncclProfilerStopProxyCtrlEvent(eHandle);
    }
    if (state->stop) { // We might have been woken up to stop.
      pthread_mutex_unlock(&pool->mutex);
      return ncclSuccess;
    }
  }

  state->nextOps = pool->nextOps;
  pool->nextOps = pool->nextOpsEnd = -1;
  pthread_mutex_unlock(&pool->mutex);
  if (state->nextOps == -1) return ncclInternalError;

process_nextops:
  pcclAsysProxyProfilingRecord(&profArgs, 0, 0, ncclProxyProfileAppend);
  ncclProfilerStartProxyCtrlEvent(proxyState->profilerContext, &eHandle);
  ncclProfilerRecordProxyCtrlEventState(eHandle, 0, ncclProfilerProxyCtrlAppend);
  TIME_START(2);
  int freeOp[NCCL_MAX_PROXY_CONNECTIONS];
  int freeOpEnd[NCCL_MAX_PROXY_CONNECTIONS];
  for (int i=0; i<proxyState->tpLocalnRanks; i++) freeOp[i] = -1;

  for (int opIndex = state->nextOps; opIndex != -1;) {
    struct ncclProxyOp* peerOp = pool->ops+opIndex;
    peerOp->perfComm = proxyState->ownerComm;
    int peer = opIndex / MAX_OPS_PER_PEER;
    if (peerOp->connection == NULL) return ncclInternalError;
    if (peerOp->next != -1) __builtin_prefetch(pool->ops+peerOp->next);
    NCCLCHECK(ProxyAppend(state, peerOp));
    (*added)++;
    int lastOpIndex = opIndex;
    opIndex = peerOp->next;
    // Return op to peer pool
    if (freeOp[peer] == -1) {
      freeOpEnd[peer] = lastOpIndex;
    } else {
      peerOp->next = freeOp[peer];
    }
    freeOp[peer] = lastOpIndex;
    state->nextOps = opIndex;
  }

  for (int i=0; i<proxyState->tpLocalnRanks; i++) {
    if (freeOp[i] == -1) continue;
    int newFree = freeOp[i];
    int oldFree = pool->freeOps[i];
    pool->ops[freeOpEnd[i]].next = oldFree;
    if (oldFree == -1) {
      // Nothing for the main thread to consume, we can set it.
      pool->freeOps[i] = newFree;
    } else {
      // The main thread may recycle free ops at any time, replace the freeOps value atomically and check it worked.
      int swap = __sync_val_compare_and_swap(pool->freeOps+i, oldFree, newFree);
      if (swap != oldFree) {
        if (swap != -1) return ncclInternalError;
        // Ops were recycled while we were trying to swap, just set the value directly now.
        pool->ops[freeOpEnd[i]].next = -1;
        pool->freeOps[i] = newFree;
      }
    }
  }
  ncclProfilerRecordProxyCtrlEventState(eHandle, *added, ncclProfilerProxyCtrlAppendEnd);
  ncclProfilerStopProxyCtrlEvent(eHandle);

  profArgs.opCount = *added;
  pcclAsysProxyProfilingRecord(&profArgs, 0, 0, ncclProxyProfileAppendEnd);
  TIME_STOP(2);
  return ncclSuccess;
}

#include <signal.h>
static ncclProxyState* ncclProxyStates[NCCL_MAX_LOCAL_RANKS] = {NULL};
void ncclDumpProxyState(int signal) {
  for (int i = 0; i < NCCL_MAX_LOCAL_RANKS; i++) {
    ncclProxyState* proxyState = ncclProxyStates[i];
    if (!proxyState) continue;
    pcclDumpProxyState(proxyState);
  }
}

NCCL_PARAM(CreateThreadContext, "CREATE_THREAD_CONTEXT", 0);
static int setProxyThreadContext(struct ncclProxyState* proxyState) {
// TODO: wait HGPFN ready and confirm drv support.
#if 0
  static int createThreadContext = -1;
  if (createThreadContext == -1) {
    createThreadContext = ncclParamCreateThreadContext();
    if (createThreadContext) {
      if (HGPFN(hgCtxCreate) == nullptr || HGPFN(hgCtxDestroy) == nullptr || HGPFN(hgCtxSetCurrent) == nullptr) {
        WARN("Unable to create thread context due to old driver, disabling.");
        createThreadContext = 0;
        goto exit;
      }
    }
  }
  if (createThreadContext) {
    if (proxyState->hggcCtx == NULL) {
      if (HGPFN(hgCtxCreate(&proxyState->hggcCtx,
                            NULL, 0, HG_CTX_SCHED_SPIN|HG_CTX_MAP_HOST, proxyState->hggcDev) != HGGC_SUCCESS) {
        WARN("Failed to create HGGC context on device %d", proxyState->hggcDev);
        createThreadContext = 0;
        goto exit;
      }
    } else {
      if (HGPFN(hgCtxSetCurrent(proxyState->hggcCtx)) != HGGC_SUCCESS) {
        WARN("Failed to set HGGC context on device %d", proxyState->hggcDev);
        goto exit;
      }
    }
    return 1;
  }
exit:
#endif
  return 0;
}


// Set to SIGUSR1 or SIGUSR2 to help debug proxy state during hangs
NCCL_PARAM(ProxyDumpSignal, "PROXY_DUMP_SIGNAL", SIGUSR1);
NCCL_PARAM(ProgressAppendOpFreq, "PROGRESS_APPENDOP_FREQ", 8);
// Idle backoff for the async CPU flush thread. After the last trigger the
// thread busy-waits (PAUSE) up to this many spins hoping for another trigger,
// then yields once so it does not monopolize its core. 0 = pure busy-poll
// (never yield), <0 = yield immediately.
PCCL_PARAM(CpuFlushSpin, "CPU_FLUSH_SPIN", 1024);
// Core to pin the async CPU flush thread to, for run-to-run stability (a busy
// flush thread co-located with the progress thread or another rank's busy
// thread drags that whole run). -2 (default) = auto: pick a core from this
// thread's allowed set (NUMA-local via inherited proxy affinity) offset by
// localRank so ranks do not collide; -1 = no binding (inherit); >=0 = pin to
// exactly that core.
PCCL_PARAM(CpuFlushCore, "CPU_FLUSH_CORE", -2);
// Async CPU flush consumer thread: one per proxyState (per device).
// Demand-driven coalescing read backing the recv-side async flush ticket: waits
// (spin-then-yield) for a new ticket, then issues one PCI-E ordering read that
// covers all pending tickets, on a dedicated thread-owned device scratch
// (host-locked), avoiding any per-connection UAF.
static void* pcclCpuFlushThreadMain(void* proxyState_) {
  struct ncclProxyState* proxyState = (struct ncclProxyState*)proxyState_;
  struct ncclProxyProgressState* state = &proxyState->progressState;
  if (setProxyThreadContext(proxyState)) {
    INFO(NCCL_INIT, "[CpuFlush] Set HGGC context on device %d", proxyState->hggcDev);
  } else if (hggcSetDevice(proxyState->hggcDev) != hggcSuccess) {
    WARN("[CpuFlush] Failed to set HGGC device %d", proxyState->hggcDev);
  }
  // Pin this flush thread to a dedicated core for run-to-run stability.
  // See PCCL_CPU_FLUSH_CORE above. The chosen core is logged via INFO only
  // (no stdout printf on any path, so it never perturbs perf measurements).
  {
    const int64_t coreParam = pcclParamCpuFlushCore();
    const int localRank = proxyState->localRank;
    int chosen = -1;
    if (coreParam == -2) {
      // Auto: enumerate this thread's allowed CPUs (NUMA-local via the inherited
      // proxy affinity) and pick one from the tail, offset by localRank, so each
      // rank's flush thread lands on a distinct core.
      cpu_set_t allowed;
      CPU_ZERO(&allowed);
      if (sched_getaffinity(0, sizeof(cpu_set_t), &allowed) == 0) {
        int n = CPU_COUNT(&allowed);
        if (n > 0) {
          int want = n - 1 - (localRank % n);
          int idx = 0;
          for (int c = 0; c < CPU_SETSIZE; c++) {
            if (CPU_ISSET(c, &allowed) && idx++ == want) { chosen = c; break; }
          }
        }
      }
    } else if (coreParam >= 0) {
      chosen = (int)coreParam;
    }
    if (chosen >= 0) {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(chosen, &set);
      if (sched_setaffinity(0, sizeof(cpu_set_t), &set) == 0) {
        INFO(NCCL_INIT, "[CpuFlush] dev %d localRank %d bound to cpu %d", proxyState->hggcDev, localRank, chosen);
      } else {
        WARN("[CpuFlush] dev %d localRank %d failed to bind to cpu %d", proxyState->hggcDev, localRank, chosen);
      }
    }
  }
  // Dedicated flush scratch (1x uint64 device mem + host-lock); lifetime == this thread.
  uint64_t* gpuPtr = NULL;
  uint64_t* cpuPtr = NULL;
  if (ncclHggcCalloc(&gpuPtr, 1, proxyState->pDevMemUsed) == ncclSuccess
      && hggcEitInterface::getInstance().wrapHggcMemLock((HGdeviceptr)gpuPtr, (void**)&cpuPtr) == hggcSuccess) {
    state->flushScratchDesc = gpuPtr;
    // Release-publish the scratch pointer so the recv progress thread that
    // acquire-loads it also observes the fully set-up host-locked mapping.
    __atomic_store_n(&state->flushScratch, cpuPtr, __ATOMIC_RELEASE);
  } else {
    // Without a dedicated scratch the flush thread cannot issue the ordering
    // read, so it never publishes flushDoneSeq. The recv progress path detects
    // the NULL scratch and stays on the synchronous flush, so this degrades
    // safely rather than signalling completion without a real flush.
    WARN("[CpuFlush] async scratch alloc failed, falling back to synchronous CPU flush");
    goto out;
  }
  // Demand-driven coalescing read: wait (spin-then-yield) until the recv
  // progress thread has taken a new ticket (flushReqSeq advanced past
  // flushDoneSeq), then issue one ordering read that satisfies every pending
  // ticket <= s. The mov reads flushReqSeq>=ticket via an acquire load, so it is
  // ordered after the progress thread's ticket bump (which happens after the
  // step's data arrived) -- a true happens-before, no timing assumption.
  // Scoped in its own block so `goto out` above never crosses these
  // initialized locals (ill-formed in C++).
  {
    const int64_t spinLimit = pcclParamCpuFlushSpin();
    int64_t idle = 0;
    while (1) {
      uint64_t s = __atomic_load_n(&state->flushReqSeq, __ATOMIC_ACQUIRE);
      if (s == __atomic_load_n(&state->flushDoneSeq, __ATOMIC_RELAXED)) {
        if (__atomic_load_n(&state->flushThreadStop, __ATOMIC_ACQUIRE)) break;
        if (spinLimit >= 0) {
          // Busy-wait after the last trigger, then yield once so an idle flush
          // thread does not monopolize its core. spinLimit==0 => pure busy-poll.
#if defined(__x86_64__)
          __builtin_ia32_pause();
#endif
          if (spinLimit > 0 && ++idle >= spinLimit) { idle = 0; sched_yield(); }
        } else {
          sched_yield(); // always yield immediately
        }
        continue;
      }
      idle = 0;
#if defined(__x86_64__)
      asm volatile ("mov (%0), %%eax" :: "l"(state->flushScratch) : "%eax");
#endif
      // One read covers all tickets <= s. RELEASE so a progress-thread ACQUIRE
      // load observing flushDoneSeq>=ticket implies this read's mov completed.
      __atomic_store_n(&state->flushDoneSeq, s, __ATOMIC_RELEASE);
    }
  }
out:
  if (state->flushScratchDesc) {
    // calloc + memLock both succeeded: unlock then free.
    hggcEitInterface::getInstance().wrapHggcMemUnlock((HGdeviceptr)state->flushScratchDesc);
    NCCLVERIFY(ncclHggcFree(state->flushScratchDesc));
    state->flushScratch = NULL;
    state->flushScratchDesc = NULL;
  } else if (gpuPtr) {
    // calloc succeeded but memLock failed: nothing was locked, just free the scratch.
    NCCLVERIFY(ncclHggcFree(gpuPtr));
  }
  return NULL;
}

void* ncclProxyProgress(void *proxyState_) {
  struct ncclProxyState* proxyState = (struct ncclProxyState*)proxyState_;
  struct ncclProxyProgressState* state = &proxyState->progressState;
  state->nextOps = -1;
  const int sig = ncclParamProxyDumpSignal();
  if (sig != -1) signal(sig, ncclDumpProxyState);
  ncclProxyStates[proxyState->localRank] = proxyState;
  char threadName[NCCL_THREAD_NAMELEN];
  snprintf(threadName, NCCL_THREAD_NAMELEN, "PCCL Progress%2d", proxyState->hggcDev);
  hgtxNameOsThreadA(syscall(SYS_gettid), threadName);
  if (setProxyThreadContext(proxyState)) {
    INFO(NCCL_INIT, "[Proxy Progress] Set HGGC context on device %d", proxyState->hggcDev);
  } else if (hggcSetDevice(proxyState->hggcDev) != hggcSuccess) {
    WARN("[pcclProxyProgress] Failed to set HGGC device %d", proxyState->hggcDev);
  }

  int lastIdle = 0;
  /* Too frequent call of ncclProxyGetPostedOps() will result in perf regression for small message
  * communication. proxyOpAppendCounter is a counter that helps us decide if we need to append proxy ops.
  * After each progress, proxyOpAppendCounter will increase by 1 and compare with environment variable
  * ncclParamProgressAppendOpFreq(). If they are equal, we will append proxy ops. This will decrease the
  * frequency of calling ncclProxyGetPostedOps() and reduce the perf impact. */
  int proxyOpAppendCounter = 0;
  struct ncclProxyArgs profArgs;
  profArgs.perfComm = proxyState->ownerComm;
  while (state->stop == 0 && __atomic_load_n(proxyState->abortFlag, __ATOMIC_ACQUIRE) == 0) {
    int idle = 1;
    ncclResult_t ret = progressOps(proxyState, state, state->active, &idle);
    if (ret != ncclSuccess) {
      __atomic_store_n(&proxyState->asyncResult, ret, __ATOMIC_RELEASE);
      INFO(NCCL_ALL,"%s:%d -> %d [Proxy Thread]", __FILENAME__, __LINE__, ret);
      return NULL;
    }
    if ((lastIdle == 0 && idle == 1) || (lastIdle == 1 && idle == 0)) {
      void* eHandle;
      ncclProfilerStartProxyCtrlEvent(proxyState->profilerContext, &eHandle);
      if (lastIdle == 0 && idle == 1) {
        ncclProfilerRecordProxyCtrlEventState(eHandle, 0, ncclProfilerProxyCtrlIdle);
        pcclAsysProxyProfilingRecord(&profArgs, 0, 0, ncclProxyProfileIdle);
      }
      if (lastIdle == 1 && idle == 0) {
        ncclProfilerRecordProxyCtrlEventState(eHandle, 0, ncclProfilerProxyCtrlActive);
        pcclAsysProxyProfilingRecord(&profArgs, 0, 0, ncclProxyProfileActive);
      }
      ncclProfilerStopProxyCtrlEvent(eHandle);
    }
    if (idle || !state->active || (++proxyOpAppendCounter == ncclParamProgressAppendOpFreq())) {
      int added = 0;
      proxyOpAppendCounter = 0;
      TIME_START(3);
      ret = ncclProxyGetPostedOps(proxyState, &added);
      if (added) { TIME_STOP(3); } else { TIME_CANCEL(3); }
      if (ret != ncclSuccess) {
        __atomic_store_n(&proxyState->asyncResult, ret, __ATOMIC_RELEASE);
        INFO(NCCL_ALL,"%s:%d -> %d [Proxy Thread]", __FILENAME__, __LINE__, ret);
      }
      if (added == 0) {
        sched_yield(); // No request progressed. Let others run.
      }
    }
    lastIdle = idle;
  }
  return NULL;
}

ncclResult_t ncclProxyStart(ncclComm_t comm) {
  struct ncclProxyOps* proxyOps = comm->proxyState->proxyOps;
  if (proxyOps == NULL) return ncclSuccess;
  TIME_START(1);
  for (int r=0; r<comm->sharedRes->tpNLocalRanks; r++) {
    struct ncclProxyOps* ops = proxyOps+r;
    if (ops->pool == NULL || ops->nextOps == -1) continue;
    NCCLCHECK(ncclProxyPost(ops->pool, ops->nextOps, ops->nextOpsEnd));
    ops->nextOps = ops->nextOpsEnd = -1;
    ops->count = 0;
  }
  comm->opCount++;
  TIME_STOP(1);
  return ncclSuccess;
}

// Only one caller owns startup: the service thread for the public option,
// or the progress thread for task-local AlltoAll use when that option is off.
static void startCpuFlushThread(struct ncclProxyState* proxyState) {
  struct ncclProxyProgressState* state = &proxyState->progressState;
#if defined(__x86_64__)
  // The flush thread's ordering read and the recv-side async flush path are both
  // x86-only, so only spin up the thread there. On other arches the thread would
  // busy-wait and hold a scratch allocation without ever performing a real flush.
  if (!state->flushThreadStarted) {
    state->flushThreadStop = 0;
    state->flushReqSeq = 0;
    state->flushDoneSeq = 0;
    state->flushScratch = NULL;
    state->flushScratchDesc = NULL;
    if (pthread_create(&state->flushThread, NULL, pcclCpuFlushThreadMain, proxyState) == 0) {
      ncclSetThreadName(state->flushThread, "PCCL CpuFlush%2d", proxyState->hggcDev);
      state->flushThreadStarted = 1;
    } else {
      WARN("NET: failed to create async CPU flush thread");
    }
  }
#endif
}

void ncclProxyEnsureA2aCpuFlush(struct ncclProxyState* proxyState) {
  // The main startup path owns the thread when the public option is enabled.
  // Do not race it or reinterpret its process-wide configuration.
  if (!pcclParamCpuFlushAsync()) startCpuFlushThread(proxyState);
}

ncclResult_t ncclProxyProgressCreate(struct ncclProxyState* proxyState) {
  struct ncclProxyProgressState* state = &proxyState->progressState;
  if (!state->thread) {
    pthread_create(&state->thread, NULL, ncclProxyProgress, proxyState);
    ncclSetThreadName(state->thread, "PCCL Progress%2d", proxyState->hggcDev);
  }
  if (pcclParamCpuFlushAsync()) startCpuFlushThread(proxyState);
  return ncclSuccess;
}

ncclResult_t ncclProxyProgressDestroy(struct ncclProxyState* proxyState) {
  struct ncclProxyProgressState* state = &proxyState->progressState;

  // Request the proxy to stop and then wake it
  if (state->opsPool) {
    pthread_mutex_lock(&state->opsPool->mutex);
    state->stop = true;
    pthread_cond_signal(&state->opsPool->cond);
    pthread_mutex_unlock(&state->opsPool->mutex);
    pthread_join(state->thread, NULL);
  }

  if (state->flushThreadStarted) {
    __atomic_store_n(&state->flushThreadStop, 1, __ATOMIC_RELEASE);
    pthread_join(state->flushThread, NULL);
    state->flushThreadStarted = 0;
  }

  // Free off any memory allocated for the proxy arg pools
  while (state->pools != NULL) {
    struct ncclProxyPool *next = state->pools->next;
    NCCLCHECK(ncclFree(state->pools));
    state->pools = next;
  }

  TIME_PRINT("Proxy");
  return ncclSuccess;
}

#define NCCL_PROXY_CONN_POOL_SIZE_POW2 7
#define NCCL_PROXY_CONN_POOL_SIZE (1<<(NCCL_PROXY_CONN_POOL_SIZE_POW2))
#define NCCL_PROXY_CONN_POOL_MASK ((NCCL_PROXY_CONN_POOL_SIZE)-1)
struct ncclProxyConnectionPool {
  struct ncclProxyConnection** pools;
  int banks;
  int offset;
  struct ncclProxyAsyncOp* ops;
};

static ncclResult_t ncclProxyNewConnection(struct ncclProxyConnectionPool* pool, int* id) {
  if (pool->offset == NCCL_PROXY_CONN_POOL_SIZE) {
    NCCLCHECK(ncclRealloc(&pool->pools, pool->banks, pool->banks+1));
    NCCLCHECK(ncclCalloc(pool->pools+pool->banks, NCCL_PROXY_CONN_POOL_SIZE));
    pool->banks++;
    pool->offset = 0;
  }
  *id = ((pool->banks-1) << NCCL_PROXY_CONN_POOL_SIZE_POW2) + pool->offset;
  pool->offset++;
  return ncclSuccess;
}

static ncclResult_t ncclProxyGetConnection(struct ncclProxyConnectionPool* pool, int id, struct ncclProxyConnection** conn) {
  int bank = id>>NCCL_PROXY_CONN_POOL_SIZE_POW2;
  int offset = id&NCCL_PROXY_CONN_POOL_MASK;
  if ((pool->pools == NULL) || (bank > pool->banks) || (pool->pools[bank] == NULL)) return ncclInternalError;
  *conn = pool->pools[bank]+offset;
  return ncclSuccess;
}

static ncclResult_t proxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  if (connection->send) {
    if (ncclTransports[connection->transport]->send.proxyFree) {
      NCCLCHECK(ncclTransports[connection->transport]->send.proxyFree(connection, proxyState));
    }
  } else {
    if (ncclTransports[connection->transport]->recv.proxyFree) {
      NCCLCHECK(ncclTransports[connection->transport]->recv.proxyFree(connection, proxyState));
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclProxyFreeConnections(struct ncclProxyConnectionPool* pool, struct ncclProxyState* proxyState) {
  for (int b=0; b<pool->banks; b++) {
    int max = b == pool->banks-1 ? pool->offset : NCCL_PROXY_CONN_POOL_SIZE;
    for (int i=0; i<max; i++) {
      ncclProxyConnection *connection = pool->pools[b]+i;
      if (connection->state != connUninitialized) {
        NCCLCHECK(proxyFree(connection, proxyState));
      }
    }
    NCCLCHECK(ncclFree(pool->pools[b]));
  }
  NCCLCHECK(ncclFree(pool->pools));
  return ncclSuccess;
}

#include "transport.h"

struct ncclProxyInitReq {
  int transport;
  int send;
  int tpLocalRank;
  int tpRank;
  int sameProcess;
};

struct ncclProxyInitResp {
  ncclProxyConnection* connection;
  char devShmPath[6]; // "XXXXXX" - May or may not be set
};

ncclResult_t ncclProxyConnect(ncclComm_t comm, int transport, int send, int proxyRank, struct ncclProxyConnector* proxyConn) {
  struct ncclProxyState* sharedProxyState = comm->proxyState;
  int tpProxyRank = comm->topParentRanks[proxyRank];
  int ready = 0;

  proxyConn->sameProcess = PCCL_SAME_PID(comm->peerInfo+proxyRank, comm->peerInfo+comm->rank) ? 1 : 0;
  // Keep one connection per local rank
  proxyConn->connection = NULL;
  proxyConn->tpRank = tpProxyRank;
  proxyConn->rank = proxyRank;
  if (sharedProxyState->peerSocks == NULL) {
    NCCLCHECK(ncclCalloc(&sharedProxyState->peerSocks, comm->sharedRes->tpNLocalRanks));
    NCCLCHECK(ncclCalloc(&sharedProxyState->proxyOps, comm->sharedRes->tpNLocalRanks));
    NCCLCHECK(ncclCalloc(&sharedProxyState->sharedDevMems, comm->sharedRes->tpNLocalRanks));
    NCCLCHECK(ncclCalloc(&sharedProxyState->sharedDevMemModes, comm->sharedRes->tpNLocalRanks));
    for (int r=0; r<comm->sharedRes->tpNLocalRanks; r++) {
      NCCLCHECK(ncclSocketSetFd(-1, &comm->proxyState->peerSocks[r]));
      sharedProxyState->sharedDevMemModes[r] = MEM_NORMAL;
    }
  }

  proxyConn->tpLocalRank = comm->sharedRes->tpRankToLocalRank[proxyConn->tpRank];
  struct ncclSocket* sock = sharedProxyState->peerSocks+proxyConn->tpLocalRank;
  NCCLCHECK(ncclSocketReady(sock, &ready));
  if (!ready) {
    NCCLCHECK(ncclSocketInit(sock, sharedProxyState->peerAddresses+proxyConn->tpRank, comm->sharedRes->magic, ncclSocketTypeProxy, NULL));
    NCCLCHECK(ncclSocketConnect(sock));
  }

  struct ncclProxyInitReq req = {0};
  req.transport = transport;
  req.send = send;
  req.tpLocalRank = comm->topParentLocalRanks[comm->localRank];
  req.tpRank = comm->topParentRanks[comm->rank];
  req.sameProcess = proxyConn->sameProcess;

  struct ncclProxyInitResp resp = {0};
  // This usually sends proxyConn->connection to identify which connection this is.
  // However, this is part of the response and therefore is ignored
  NCCLCHECK(ncclProxyCallBlocking(comm, proxyConn, ncclProxyMsgInit, &req, sizeof(req), &resp, sizeof(resp)));
  proxyConn->connection = resp.connection;

  // If we need proxy progress, map progress ops
  struct ncclTransportComm* tcomm = send ? &ncclTransports[transport]->send : &ncclTransports[transport]->recv;
  if (tcomm->proxyProgress) {
    char poolPath[] = "/dev/shm/pccl-XXXXXX";
    strncpy(poolPath+sizeof("/dev/shm/pccl-")-1, resp.devShmPath, sizeof("XXXXXX")-1);
    struct ncclProxyOps* proxyOps = sharedProxyState->proxyOps + proxyConn->tpLocalRank;
    if (proxyOps->pool == NULL) {
      NCCLCHECK(ncclShmOpen(poolPath, sizeof(struct ncclProxyOpsPool), (void**)(&proxyOps->pool), NULL, 0));
      proxyOps->nextOps = proxyOps->nextOpsEnd = proxyOps->freeOp = -1;
    }
  }
  proxyConn->initialized = true;
  INFO(NCCL_NET|NCCL_PROXY, "Connected to proxy tpLocalRank %d -> connection %p", proxyConn->tpLocalRank, proxyConn->connection);
  return ncclSuccess;
}

// UDS support
ncclResult_t ncclProxyCallBlockingUDS(ncclComm_t comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, void* respBuff, int respSize, int* reqFd, int *respFd) {
  ncclResult_t res = ncclSuccess;
  struct ncclIpcSocket ipcSock = { 0 };
  void *opId;
  NCCLCHECK(getRandomData(&opId, sizeof(opId)));
  int reqFdtmp = -1;

  int rank = comm->topParentLocalRanks[comm->localRank];
  struct ncclProxyState* sharedProxyState = comm->proxyState;
  uint64_t pidHash = sharedProxyState->peerAddressesUDS[proxyConn->tpRank];

  INFO(NCCL_PROXY, "ProxyCall UDS comm %p rank %d tpRank %d(%lx) reqSize %d respSize %d respFd %p opId %p",
       comm, rank, proxyConn->tpRank, pidHash, reqSize, respSize, respFd, opId);

  // cuMem: Create a UDS socket to receive the response
  NCCLCHECK(ncclIpcSocketInit(&ipcSock, rank, (uint64_t)opId, comm->abortFlag));

  if (reqFd) {
    reqFdtmp = *reqFd;
  } else {
    // give a dummy fd for the other side of UDS socket
    NCCLCHECK(ncclIpcSocketGetFd(&ipcSock, &reqFdtmp));
  }

  ncclIpcHdr hdr;
  memset(&hdr, '\0', sizeof(hdr));
  hdr.type = type;
  hdr.rank = rank;
  hdr.reqSize = reqSize;
  hdr.respSize = respSize;
  hdr.opId = opId;

  if (reqSize > sizeof(hdr.data)) {
    ERROR("Invalid UDS proxy request size %d, expected at most %zu", reqSize, sizeof(hdr.data));
    res = ncclInternalError;
    goto error;
  }
  memcpy(&hdr.data, reqBuff, reqSize);
  NCCLCHECKGOTO(ncclIpcSocketSendMsg(&ipcSock, &hdr, sizeof(hdr), reqFdtmp, proxyConn->tpRank, pidHash), res, error);
  NCCLCHECKGOTO(ncclIpcSocketRecvMsg(&ipcSock, respBuff, respSize, respFd), res, error);
  NCCLCHECKGOTO(ncclIpcSocketClose(&ipcSock), res, error);

  INFO(NCCL_PROXY, "ProxyCall UDS comm %p rank %d tpRank %d(%lx) reqSize %d respSize %d respFd %d opId %p - DONE",
       comm, rank, proxyConn->tpRank, pidHash, reqSize, respSize, (respFd ? *respFd : -1), opId);

  return res;

error:
  NCCLCHECK(ncclIpcSocketClose(&ipcSock));
  ERROR("pcclProxyCallBlockingUDS call to tpRank %d(%lx) failed : %d", proxyConn->tpRank, pidHash, res);
  return res;
}

// hgMem API support
// The response is sent out-of-band using ncclIpcSocket for this specific command
ncclResult_t ncclProxyClientGetFdBlocking(ncclComm_t comm, int proxyRank, void *handle, int* convertedFd) {
  ncclResult_t ret = ncclSuccess;
  // Request the allocation of a UDS fd for the handle
  if (comm->gproxyConn[proxyRank].initialized == false) {
    NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_P2P, 1, proxyRank, &comm->gproxyConn[proxyRank]), ret, error);
  }
  NCCLCHECKGOTO(ncclProxyCallBlockingUDS(comm, &comm->gproxyConn[proxyRank], ncclProxyMsgGetFd, handle, sizeof(HGmemGenericAllocationHandle), NULL, 0, NULL, convertedFd), ret, error);

  // Receive converted fd over UDS
  VERBOSE(NCCL_PROXY, "UDS: ClientGetFd handle 0x%lx tpRank %d returned fd %d sameProcess %d", *(uint64_t*)handle, comm->topParentRanks[proxyRank], *convertedFd, comm->gproxyConn[proxyRank].sameProcess);
  return ret;

error:
  ERROR("pcclProxyClientGetFd call to tpRank %d handle 0x%lx failed : %d", comm->topParentRanks[proxyRank], *(uint64_t*)handle, ret);
  return ret;
}


ncclResult_t ncclProxyClientQueryFdBlocking(ncclComm_t comm, struct ncclProxyConnector* proxyConn, int localFd, int* rmtFd) {
  ncclResult_t ret = ncclSuccess;
  NCCLCHECKGOTO(ncclProxyCallBlockingUDS(comm, proxyConn, ncclProxyMsgQueryFd, NULL, 0, (void*)rmtFd, sizeof(int), &localFd, NULL), ret, fail);
exit:
  // We have now received the converted fd over UDS
  INFO(NCCL_PROXY, "UDS: ClientQueryFd localFd %d tpRank %d remote fd %d sameProcess %d", localFd, proxyConn->tpRank, *rmtFd, proxyConn->sameProcess);
  return ret;
fail:
  ERROR("pcclProxyClientQueryFdBlocking call to tpRank %d localFd %d failed : %d", proxyConn->tpRank, localFd, ret);
  goto exit;
}

const char* ncclProxyMsgTypeStr[] = { "Unknown", "Init", "SharedInit", "Setup", "Connect", "Start", "Close", "Abort", "Stop", "GetFd", "QueryFd", "Register", "Deregister" };
ncclResult_t ncclProxyCallAsync(ncclComm_t comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, int respSize, void* opId) {
  struct ncclSocket* sock;
  ncclResult_t ret = ncclSuccess;
  struct ncclProxyState* sharedProxyState = comm->proxyState;

  if (sharedProxyState->peerSocks == NULL) return ncclInternalError;

  sock = sharedProxyState->peerSocks + proxyConn->tpLocalRank;

  NCCLCHECKGOTO(ncclSocketSend(sock, &type, sizeof(int)), ret, error);
  NCCLCHECKGOTO(ncclSocketSend(sock, &proxyConn->connection, sizeof(void*)), ret, error);
  NCCLCHECKGOTO(ncclSocketSend(sock, &reqSize, sizeof(int)), ret, error);
  NCCLCHECKGOTO(ncclSocketSend(sock, &respSize, sizeof(int)), ret, error);
  if (reqSize) NCCLCHECKGOTO(ncclSocketSend(sock, reqBuff, reqSize), ret, error);

  // Send opId to proxy
  NCCLCHECKGOTO(ncclSocketSend(sock, &opId, sizeof(opId)), ret, error);

  // Add proxyOp to expected response queue
  NCCLCHECK(expectedProxyResponseEnqueue(sharedProxyState, opId, respSize));

  return ncclSuccess;
error:
  ERROR("Proxy Call to rank %d failed (%s)", comm->localRankToRank[proxyConn->tpLocalRank], ncclProxyMsgTypeStr[type]);
  return ret;
}

ncclResult_t ncclPollProxyResponse(ncclComm_t comm, struct ncclProxyConnector* proxyConn, void* respBuff, void* opId) {
  struct ncclProxyState* sharedProxyState = comm->proxyState;
  // Receive the connection pointer from the Proxy
  if (__atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE)) {
    ERROR("Comm %p is in abort state", comm);
    return ncclInternalError;
  }
  if (sharedProxyState->peerSocks == NULL) return ncclInternalError;

  // Check response queue
  int found = 0;
  ncclResult_t res = expectedProxyResponseDequeue(sharedProxyState, opId, respBuff, &found);
  if (found == 0) {
    // Attempt to read in a new response header from the proxy thread
    struct ncclSocket* sock = sharedProxyState->peerSocks + proxyConn->tpLocalRank;
    ncclProxyRpcResponseHeader resp = {0};
    int offset = 0;
    if (ncclSuccess != ncclSocketProgress(NCCL_SOCKET_RECV, sock, &resp, sizeof(resp), &offset)) {
      ERROR("Socket recv failed while polling for opId=%p", opId);
      return ncclInternalError;
    }

    if (offset == 0) {
      return ncclInProgress;
    // If we've returned a partial response, block to receive the rest of it
    } else if (offset < sizeof(resp)) {
      while (offset < sizeof(resp))
        NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, sock, &resp, sizeof(resp), &offset));
    }

    INFO(NCCL_PROXY, "pcclPollProxyResponse Received new opId=%p", resp.opId);

    // If there's a respSize to recv
    if (resp.respSize > 0) {
      if (resp.opId != opId) {
        // Unexpected response, need to buffer the socket data
        respBuff = malloc(resp.respSize);
        if (respBuff == NULL) {
          ERROR("Could not allocate proxy response buffer for response opId=%p expected opId=%p response size %d",
               resp.opId, opId, resp.respSize);
          return ncclSystemError;
        }
      }
      if (respBuff == NULL) {
        ERROR("Proxy response buffer is NULL for response opId=%p expected opId=%p response size %d", resp.opId, opId,
             resp.respSize);
        return ncclInternalError;
      }
      NCCLCHECK(ncclSocketRecv(sock, respBuff, resp.respSize));
    }

    if (resp.opId == opId) {
      INFO(NCCL_PROXY, "resp.opId=%p matches expected opId=%p", resp.opId, opId);
      NCCLCHECK(expectedProxyResponseRemove(sharedProxyState, resp.opId));
      return resp.res;
    } else {
      INFO(NCCL_PROXY, "Queuing opId=%p respBuff=%p respSize=%d", resp.opId, respBuff, resp.respSize);
      // Store the result and mark response as completed
      NCCLCHECK(expectedProxyResponseStore(sharedProxyState, resp.opId, respBuff, resp.respSize, resp.res));
      return ncclInProgress;
    }
  } else {
    INFO(NCCL_PROXY, "pcclPollProxyResponse Dequeued cached opId=%p", opId);
  }

  return res;
}

ncclResult_t ncclProxyCallBlocking(ncclComm_t comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, void* respBuff, int respSize) {
  // Alloc some memory to act as a handle
  ncclResult_t res = ncclSuccess;
  void* opId = malloc(1);

  NCCLCHECKGOTO(ncclProxyCallAsync(comm, proxyConn, type, reqBuff, reqSize, respSize, opId), res, fail);

  do {
    res = ncclPollProxyResponse(comm, proxyConn, respBuff, opId);
  } while (res == ncclInProgress);

exit:
  NCCLCHECK(ncclFree(opId));
  return res;
fail:
  goto exit;
}

static ncclResult_t proxyProgressInit(struct ncclProxyState* proxyState) {
  struct ncclProxyProgressState* state = &proxyState->progressState;
  if (state->opsPool == NULL) {
    int size = sizeof(struct ncclProxyOpsPool);
    struct ncclProxyOpsPool* pool = NULL;

    char shmPath[sizeof("/dev/shm/pccl-XXXXXX")];
    shmPath[0] = '\0';
    NCCLCHECK(ncclShmOpen(shmPath, size, (void**)&pool, NULL, 1));

    // Init pool
    pool->nextOps = -1;

    for (int r=0; r<proxyState->tpLocalnRanks; r++) {
      pool->freeOps[r] = r*MAX_OPS_PER_PEER;
      for (int i=0; i<MAX_OPS_PER_PEER-1; i++) pool->ops[r*MAX_OPS_PER_PEER+i].next = r*MAX_OPS_PER_PEER+i+1;
      pool->ops[(r+1)*MAX_OPS_PER_PEER-1].next = -1;
    }

    // Setup mutex/cond to work inter-process
    pthread_mutexattr_t mutexAttr;
    pthread_mutexattr_init(&mutexAttr);
    pthread_mutexattr_setpshared(&mutexAttr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&pool->mutex, &mutexAttr);
    pthread_condattr_t condAttr;
    pthread_condattr_setpshared(&condAttr, PTHREAD_PROCESS_SHARED);
    pthread_cond_init(&pool->cond, &condAttr);
    state->opsPool = pool;

    memcpy(state->opsPoolShmSuffix, shmPath+sizeof("/dev/shm/pccl-")-1, sizeof("XXXXXX")-1);

    // All ops structures are created, we can start the progress thread
    NCCLCHECK(ncclProxyProgressCreate(proxyState));
  }
  return ncclSuccess;
}

static ncclResult_t proxyOpsPoolUnlink(struct ncclProxyProgressState* state) {
  if (state->opsPool == NULL) return ncclSuccess;

  char shmPath[] = "/dev/shm/pccl-XXXXXX";
  memcpy(shmPath+sizeof("/dev/shm/pccl-")-1, state->opsPoolShmSuffix, sizeof("XXXXXX")-1);
  return ncclShmUnlink(shmPath);
}

static void proxyOpsFree(struct ncclProxyState* proxyState) {
  struct ncclProxyProgressState* state = &proxyState->progressState;
  // The ops pool shm is only unlinked at the end of init when cuMem is disabled;
  // with cuMem or splitShare, late proxy connections may still open it by name,
  // so it must stay linked while the service is alive. The service has exited at
  // this point and no new attach can happen: unlink it here so the file does not
  // leak in /dev/shm. ENOENT (already unlinked at init) is tolerated.
  if (proxyOpsPoolUnlink(state) != ncclSuccess) {
    WARN("[Service thread] shm unlink failed");
  }
  if (ncclShmClose(state->opsPool, NULL, sizeof(struct ncclProxyOpsPool)) != ncclSuccess) {
    WARN("[Service thread] shm close failed");
  }
}

ncclResult_t ncclProxyShmUnlink(ncclComm_t comm) {
  struct ncclProxyProgressState* state = &comm->proxyState->progressState;
  if (proxyOpsPoolUnlink(state) != ncclSuccess) {
    WARN("[Service thread] shm unlink failed");
  }
  return ncclSuccess;
}

static ncclResult_t proxyConnInit(struct ncclProxyLocalPeer* peer, struct ncclProxyConnectionPool* connectionPool, struct ncclProxyState* proxyState, ncclProxyInitReq* req, ncclProxyInitResp* resp, struct ncclProxyConnection** connection) {
  int id;
  NCCLCHECK(ncclProxyNewConnection(connectionPool, &id));
  NCCLCHECK(ncclProxyGetConnection(connectionPool, id, connection));

  (*connection)->sock = &peer->sock;
  (*connection)->transport = req->transport;
  (*connection)->send = req->send;
  (*connection)->tpLocalRank = req->tpLocalRank;
  (*connection)->sameProcess = req->sameProcess;
  peer->tpLocalRank = req->tpLocalRank;
  peer->tpRank = req->tpRank;

  resp->connection = *connection;

  (*connection)->tcomm = (*connection)->send ? &ncclTransports[(*connection)->transport]->send : &ncclTransports[(*connection)->transport]->recv;
  // If we need proxy progress, let's allocate ops and start the thread
  if ((*connection)->tcomm->proxyProgress) {
    NCCLCHECK(proxyProgressInit(proxyState));
    struct ncclProxyProgressState* state = &proxyState->progressState;
    memcpy(resp->devShmPath, state->opsPoolShmSuffix, sizeof(resp->devShmPath));
  }
  INFO(NCCL_NET|NCCL_PROXY, "New proxy %s connection %d from local rank %d, transport %d", (*connection)->send ? "send":"recv", id, (*connection)->tpLocalRank, (*connection)->transport);
  __atomic_store_n(&(*connection)->state, connInitialized, __ATOMIC_RELEASE);
  return ncclSuccess;
}

static ncclResult_t proxyQueryFd(struct ncclProxyState* proxyState, int rank, void *opId, int rmtFd) {
  struct ncclIpcSocket ipcSock = { 0 };
  uint64_t hash = (uint64_t) opId;
  ncclResult_t ret = ncclSuccess;

  NCCLCHECKGOTO(ncclIpcSocketInit(&ipcSock, proxyState->tpRank, hash^1, proxyState->abortFlag), ret, exit);
  NCCLCHECKGOTO(ncclIpcSocketSendMsg(&ipcSock, &rmtFd, sizeof(int), -1, rank, hash), ret, exit);
exit:
  NCCLCHECK(ncclIpcSocketClose(&ipcSock));
  return ncclSuccess;
}


static ncclResult_t proxyGetFd(struct ncclProxyState* proxyState, int rank, void *opId, uint64_t handle) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIpcSocket ipcSock = { 0 };
  uint64_t hash = (uint64_t) opId;
  INFO(NCCL_PROXY, "UDS proxyGetFd received handle 0x%lx peer %d opId %lx", handle, rank, hash);

  HGmemAllocationHandleType type = HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  int fd = -1;

  HGCHECK(hgMemExportToShareableHandle(&fd, handle, type, 0));
  // Send back the converted fd using UDS
  NCCLCHECKGOTO(ncclIpcSocketInit(&ipcSock, proxyState->tpRank, hash^1, proxyState->abortFlag), ret, error);
  NCCLCHECKGOTO(ncclIpcSocketSendFd(&ipcSock, fd, rank, hash), ret, error);
error:
  NCCLCHECK(ncclIpcSocketClose(&ipcSock));
  // We can now safely close the exported fd
  SYSCHECK(close(fd), "close");
  return ret;
}

static ncclResult_t proxyProgressAsync(struct ncclProxyAsyncOp* op, struct ncclProxyState* proxyState, int* asyncOpCount, struct ncclProxyLocalPeer* peer, struct ncclProxyConnectionPool* connectionPool) {
  int done = 1;
  ncclResult_t res = ncclInternalError;
  if (op->type == ncclProxyMsgSetup) {
    TRACE(NCCL_PROXY, "proxyProgressAsync::proxySetup() opId=%p", op->opId);
    res = op->connection->tcomm->proxySetup(op->connection, proxyState, op->reqBuff, op->reqSize, op->respBuff, op->respSize, &done);
  } else if (op->type == ncclProxyMsgConnect) {
    TRACE(NCCL_PROXY, "proxyProgressAsync::proxyConnect() opId=%p op.reqBuff=%p", op->opId, op->reqBuff);
    res = op->connection->tcomm->proxyConnect(op->connection, proxyState, op->reqBuff, op->reqSize, op->respBuff, op->respSize, &done);
  } else if (op->type == ncclProxyMsgSharedInit) {
    int nChannels = (int) *op->reqBuff;
    TRACE(NCCL_PROXY, "proxyProgressAsync::ncclProxyMsgSharedInit opId=%p op.reqBuff=%p nChannels=%d", op->opId, op->reqBuff, nChannels);
    if (op->connection->tcomm->proxySharedInit) res = op->connection->tcomm->proxySharedInit(op->connection, proxyState, nChannels);
    __atomic_store_n(&op->connection->state, connSharedInitialized, __ATOMIC_RELEASE);
  } else if (op->type == ncclProxyMsgInit) {
    TRACE(NCCL_PROXY, "proxyProgressAsync::ncclProxyMsgInit opId=%p op.reqBuff=%p", op->opId, op->reqBuff);
    res = proxyConnInit(peer, connectionPool, proxyState, (ncclProxyInitReq*) op->reqBuff, (ncclProxyInitResp*) op->respBuff, &op->connection);
  } else if (op->type == ncclProxyMsgRegister) {
    TRACE(NCCL_PROXY, "proxyProgressAsync::ncclProxyMsgRegister opId=%p op.reqBuff=%p, op->reqSize=%d, op->respSize=%d", op->opId, op->reqBuff, op->reqSize, op->respSize);
    res = op->connection->tcomm->proxyRegister(op->connection, proxyState, op->reqBuff, op->reqSize, op->respBuff, op->respSize, &done);
  } else if (op->type == ncclProxyMsgDeregister) {
    TRACE(NCCL_PROXY, "proxyProgressAsync::ncclProxyMsgDeregister opId=%p op.reqBuff=%p, op->reqSize=%d, op->respSize=%d", op->opId, op->reqBuff, op->reqSize, op->respSize);
    res = op->connection->tcomm->proxyDeregister(op->connection, proxyState, op->reqBuff, op->reqSize, &done);
  } else return ncclInternalError;

  if (done) {
    INFO(NCCL_PROXY, "proxyProgressAsync opId=%p op.type=%d op.reqBuff=%p op.respSize=%d done", op->opId, op->type, op->reqBuff, op->respSize);
    if (op->type == ncclProxyMsgSetup)
      __atomic_store_n(&op->connection->state, connSetupDone, __ATOMIC_RELEASE);
    else if (op->type == ncclProxyMsgConnect)
      __atomic_store_n(&op->connection->state, connConnected, __ATOMIC_RELEASE);

    /* if setup or connect is done, we should not return any error at this point since
     * ncclSocketSend might already send the respBuff to the requester. If we still choose
     * to abort and close the connection, it can cause segfault if the requester is using
     * the respBuff. */
    ncclProxyRpcResponseHeader resp = {op->opId, res, op->respSize};

    // Send the opId for referencing async operation
    NCCLCHECK(ncclSocketSend(op->connection->sock, &resp, sizeof(resp)));

    if (op->respSize) {
      // Send the response
      NCCLCHECK(ncclSocketSend(op->connection->sock, op->respBuff, op->respSize));
    }

    asyncProxyOpDequeue(peer, op);
    (*asyncOpCount)--;
    return ncclSuccess;
  }
  return ncclInProgress;
}

static ncclResult_t proxyServiceInitOp(int type, struct ncclProxyLocalPeer* peer, struct ncclProxyConnectionPool* connectionPool, struct ncclProxyState* proxyState, int* asyncOpCount) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket* sock = &peer->sock;
  struct ncclProxyAsyncOp* asyncOp;
  NCCLCHECK(ncclCalloc(&asyncOp, 1));

  asyncOp->type = type;
  NCCLCHECKGOTO(ncclSocketRecv(sock, &asyncOp->connection, sizeof(void*)), ret, fail);

  NCCLCHECKGOTO(ncclSocketRecv(sock, &asyncOp->reqSize, sizeof(int)), ret, fail);
  NCCLCHECKGOTO(ncclSocketRecv(sock, &asyncOp->respSize, sizeof(int)), ret, fail);
  if (asyncOp->reqSize) {
    NCCLCHECKGOTO(ncclCalloc(&asyncOp->reqBuff, asyncOp->reqSize), ret, fail);
    NCCLCHECKGOTO(ncclSocketRecv(sock, asyncOp->reqBuff, asyncOp->reqSize), ret, fail);
  }

  // Store opId for completion response
  NCCLCHECKGOTO(ncclSocketRecv(sock, &asyncOp->opId, sizeof(asyncOp->opId)), ret, fail);

  if (asyncOp->respSize) NCCLCHECKGOTO(ncclCalloc(&asyncOp->respBuff, asyncOp->respSize), ret, fail);

  NCCLCHECKGOTO(asyncProxyOpEnqueue(peer, asyncOp), ret, fail);

  (*asyncOpCount)++;
  NCCLCHECK(proxyProgressAsync(asyncOp, proxyState, asyncOpCount, peer, connectionPool));
exit:
  return ret;
fail:
  NCCLCHECK(ncclFree(asyncOp->reqBuff));
  NCCLCHECK(ncclFree(asyncOp->respBuff));
  NCCLCHECK(ncclFree(asyncOp));
  goto exit;
}

#include <poll.h>

static bool proxyMatchOpType(int type) {
  switch (type) {
    case ncclProxyMsgInit:
    case ncclProxyMsgSharedInit:
    case ncclProxyMsgSetup:
    case ncclProxyMsgConnect:
    case ncclProxyMsgGetFd:
    case ncclProxyMsgRegister:
    case ncclProxyMsgDeregister:
      return true;
    default:
      return false;
  }
}

enum {
  PROXY_RUNNING = 0,
  PROXY_STOP = 1,
  PROXY_ABORT = 2
};

void* ncclProxyService(void* _args) {
  struct ncclProxyState* proxyState =  (struct ncclProxyState*) _args;
  if (setProxyThreadContext(proxyState)) {
    INFO(NCCL_INIT, "[Proxy Progress] Set HGGC context on device %d", proxyState->hggcDev);
  } else if (hggcSetDevice(proxyState->hggcDev) != hggcSuccess) {
    WARN("[Proxy Service] Failed to set HGGC device %d", proxyState->hggcDev);
  }
  if (CPU_COUNT(&proxyState->ownerComm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &proxyState->ownerComm->cpuAffinity);

  // Prepare poll descriptor
  struct ncclProxyConnectionPool connectionPool;
  connectionPool.pools = NULL;
  connectionPool.banks = 0;
  connectionPool.offset = NCCL_PROXY_CONN_POOL_SIZE;

  struct pollfd pollfds[NCCL_MAX_PROXY_CONNECTIONS+1];
  struct ncclProxyLocalPeer peers[NCCL_MAX_PROXY_CONNECTIONS];
  memset(&peers, 0, sizeof(struct ncclProxyLocalPeer)*NCCL_MAX_PROXY_CONNECTIONS);
  for (int s=0; s<NCCL_MAX_PROXY_CONNECTIONS; s++) {
    peers[s].sock.fd = pollfds[s].fd = -1;
    pollfds[s].events = POLLHUP|POLLIN;
  }

  if (ncclSocketGetFd(proxyState->listenSock, &pollfds[NCCL_MAX_PROXY_CONNECTIONS].fd) != ncclSuccess) {
    WARN("[Proxy Service] Get listenSock fd fails\n");
    return NULL;
  };
  pollfds[NCCL_MAX_PROXY_CONNECTIONS].events = POLLIN;

  int maxnpeers = 0;
  int npeers = 0;
  int stop = PROXY_RUNNING;
  int asyncOpCount = 0;
  while (stop == PROXY_RUNNING || npeers > 0) {
    if (__atomic_load_n(proxyState->abortFlag, __ATOMIC_ACQUIRE) != 0) stop = PROXY_ABORT;
    /* never let proxy service thread blocks in poll, or it cannot receive abortFlag. */
    int ret;
    do {
      // poll all fds including the listenSock
      ret = poll(pollfds, NCCL_MAX_PROXY_CONNECTIONS+1, asyncOpCount ? 0 : 500);
    } while (ret < 0 && errno == EINTR);
    if (ret < 0) {
      ERROR("[Proxy Service] Poll failed: %s", strerror(errno));
      return NULL;
    }
    if (pollfds[NCCL_MAX_PROXY_CONNECTIONS].revents) {
      int s = 0;
      while (s < NCCL_MAX_PROXY_CONNECTIONS && peers[s].sock.fd != -1) s++;
      if (s == NCCL_MAX_PROXY_CONNECTIONS) {
        ERROR("[Proxy service] Too many connections (%d max)", NCCL_MAX_PROXY_CONNECTIONS);
        return NULL;
      }
      if (maxnpeers < s+1) maxnpeers = s+1;
      if (ncclSocketInit(&peers[s].sock) != ncclSuccess) {
        WARN("[Service thread] Initialize peers[%d].sock fails\n", s);
        return NULL;
      }
      if (ncclSocketAccept(&peers[s].sock, proxyState->listenSock) != ncclSuccess) {
        WARN("[Service thread] Accept failed %s", strerror(errno));
      } else {
        if (ncclSocketGetFd(&peers[s].sock, &pollfds[s].fd) != ncclSuccess) {
          WARN("[Service thread] Get peers[%d].sock fd fails\n", s);
          return NULL;
        }
        npeers++;
        peers[s].tpLocalRank = -1;
      }
    }
    for (int s=0; s<maxnpeers; s++) {
      struct ncclProxyLocalPeer* peer = peers+s;
      struct ncclSocket* sock = &peer->sock;
      int closeConn = 0;
      int type = 0;
      ncclResult_t res = ncclSuccess;
      if (pollfds[s].fd == -1) continue;

      // Progress all ops for this ncclProxyLocalPeer
      if (stop == PROXY_ABORT && ncclCuMemEnable() && !proxyState->directMode && __atomic_load_n(&proxyState->stop, __ATOMIC_ACQUIRE)) closeConn = 1;
      struct ncclProxyAsyncOp* op = peer->asyncOps;
      while (op != nullptr) {
        ncclProxyAsyncOp* opNext = op->next; /* in case op is freed in proxyProgressAsync */
        type = op->type;
        res = proxyProgressAsync(op, proxyState, &asyncOpCount, peer, &connectionPool);
        if (res == ncclSuccess || res == ncclInProgress) {
          op = opNext;
        } else {
          // Res is a bad result
          closeConn = 1;
          ERROR("[Service thread] Error encountered progressing operation=%s, res=%d, closing connection", ncclProxyMsgTypeStr[type], res);
          break;
        }
      }

      if (pollfds[s].revents & POLLIN) {
        int closed;
        res = ncclSocketTryRecv(sock, &type, sizeof(int), &closed, false /*blocking*/);
        if (res != ncclSuccess && res != ncclInProgress) {
          WARN("[Service thread] Could not receive type from localRank %d", peer->tpLocalRank);
          closeConn = 1;
        } else if (closed) {
          INFO(NCCL_DESTROY|NCCL_NET, "[Service thread] Connection closed by localRank %d", peer->tpLocalRank);
          closeConn = 1;
        } else if (res == ncclSuccess) {
          if (type == ncclProxyMsgStop) {
            stop = PROXY_STOP;
            closeConn = 1;
          } else if (type == ncclProxyMsgClose) {
            closeConn = 1;
          } else if (proxyMatchOpType(type)) {
            res = proxyServiceInitOp(type, peers+s, &connectionPool, proxyState, &asyncOpCount);
          } else {
            ERROR("[Service thread] Unknown command %d from localRank %d\n", type, peer->tpLocalRank);
            closeConn = 1;
          }

          INFO(NCCL_PROXY, "Received and initiated operation=%s res=%d", ncclProxyMsgTypeStr[type], res);
        }
      } else if (pollfds[s].revents & POLLHUP) {
        closeConn = 1;
      }
      if (res != ncclSuccess && res != ncclInProgress) {
        if (!__atomic_load_n(proxyState->abortFlag, __ATOMIC_RELAXED)) {
          ERROR("[Proxy Service %d] Failed to execute operation %s from rank %d, retcode %d", proxyState->tpRank, ncclProxyMsgTypeStr[type], peer->tpRank, res);
        }
        closeConn = 1;
      }
      if (closeConn) {
        ncclSocketClose(sock);

        if (op != nullptr) {
          asyncProxyOpDequeue(peer, op);
          asyncOpCount--;
        }
        pollfds[s].fd = -1;
        npeers--;
      }
    }
  }
  // Wait for all operations to complete and stop progress thread before freeing any resource
  if (ncclProxyProgressDestroy(proxyState) != ncclSuccess) {
    ERROR("[Proxy Service] proxyDestroy failed");
  }
  for (int s=0; s<maxnpeers; s++) {
    if (peers[s].sock.fd != -1) close(peers[s].sock.fd);
  }
  ncclProxyFreeConnections(&connectionPool, proxyState);
  close(proxyState->listenSock->fd);
  ncclFree(proxyState->listenSock);
  proxyOpsFree(proxyState);
  return NULL;
}

// Process a request on the UDS socket
static ncclResult_t proxyUDSRecvReq(struct ncclProxyState* proxyState, int reqFd) {
  ncclIpcHdr hdr;
  int rmtFd = -1;

  NCCLCHECK(ncclIpcSocketRecvMsg(&proxyState->ipcSock, &hdr, sizeof(hdr), &rmtFd));
  if (hdr.type == ncclProxyMsgGetFd) {
    // cuMem API support for non-UB case, and rmtFd is not used since UDS proxy thread need to export
    // fd from handle and send it back to the main thread to import the buffer. We just need to close
    // this dummy rmtFd.
    uint64_t handle = *(uint64_t*)hdr.data;
    INFO(NCCL_PROXY, "proxyUDSRecvReq::pcclProxyMsgGetFd rank %d opId %p handle=0x%lx", hdr.rank, hdr.opId, handle);
    close(rmtFd);
    return proxyGetFd(proxyState, hdr.rank, hdr.opId, handle);
  } else if (hdr.type == ncclProxyMsgQueryFd) {
    // remote main thread registers buffer into this rank, it queries rmtFd of this rank through UDS
    // and the rmtFd is returned unchanged back to remote main thread which will use rmtFd to call into
    // proxy service thread for buffer registration.
    INFO(NCCL_PROXY, "proxyUDSRecvReq::proxyQueryFd rank %d opId %p rmtFd %d", hdr.rank, hdr.opId, rmtFd);
    return proxyQueryFd(proxyState, hdr.rank, hdr.opId, rmtFd);
  }

  return ncclInternalError;
}

// UDS fd handle support
void* ncclProxyServiceUDS(void* _args) {
  struct ncclProxyState* proxyState =  (struct ncclProxyState*) _args;
  struct pollfd pollfds[1];

  if (setProxyThreadContext(proxyState)) {
    INFO(NCCL_INIT, "[Proxy Service UDS] Set HGGC context on device %d", proxyState->hggcDev);
  } else if (hggcSetDevice(proxyState->hggcDev) != hggcSuccess) {
    WARN("[Proxy Service UDS] Failed to set HGGC device %d", proxyState->hggcDev);
  }

  INFO(NCCL_INIT, "[Proxy Service UDS] Device %d CPU core %d", proxyState->hggcDev, sched_getcpu());

  if (ncclIpcSocketGetFd(&proxyState->ipcSock, &pollfds[0].fd) != ncclSuccess) {
    WARN("[Proxy Service UDS] Get listenSock fd fails");
    return NULL;
  };
  pollfds[0].events = POLLIN|POLLHUP;

  while (1) {
    /* never let proxy service thread blocks in poll, or it cannot receive abortFlag. */
    int ret;
    do {
      ret = poll(pollfds, 1, 500);
    } while (ret < 0 && errno == EINTR);
    if (ret < 0) {
      WARN("[Proxy Service UDS] Poll failed: %s", strerror(errno));
      return NULL;
    }

    // Check for stop/abort
    if (__atomic_load_n(&proxyState->stop, __ATOMIC_ACQUIRE) || __atomic_load_n(proxyState->abortFlag, __ATOMIC_ACQUIRE)) break;

    if (pollfds[0].revents) {
      // A request was seen on the UDS fd
      proxyUDSRecvReq(proxyState, pollfds[0].fd);
    }
  }

  (void)ncclIpcSocketClose(&proxyState->ipcSock);
  INFO(NCCL_PROXY, "[Proxy Service UDS] exit: stop %d abortFlag %d", proxyState->stop, *proxyState->abortFlag);
  return NULL;
}


ncclResult_t ncclProxyInit(ncclComm_t comm, struct ncclSocket* sock, union ncclSocketAddress* peerAddresses, uint64_t* peerAddressesUDS) {
  if (comm->sharedRes->proxyState != NULL) {
    ERROR("Proxy state is already initialized");
    return ncclInternalError;
  }
  NCCLCHECK(ncclCalloc(&comm->sharedRes->proxyState, 1));
  comm->proxyState = comm->sharedRes->proxyState;
  comm->proxyState->refCount = 1;
  comm->proxyState->localRank = -1; // set before starting proxy threads
  comm->proxyState->listenSock = sock;
  comm->proxyState->peerAddresses = peerAddresses;
  comm->proxyState->peerAddressesUDS = peerAddressesUDS;

  // UDS support
  NCCLCHECK(ncclIpcSocketInit(&comm->proxyState->ipcSock, comm->rank, peerAddressesUDS[comm->rank], comm->abortFlag));
  return ncclSuccess;
}

ncclResult_t ncclProxyCreate(ncclComm_t comm) {
  /* proxyState is shared among parent comm and split comms. comm->proxyState->thread is
   * pthread_join()'d by commFree() in init.cc when the refCount reduces down to 0. */
  struct ncclProxyState* proxyState = comm->proxyState;
  if (proxyState->refCount == 1) {
    /* we have to make sure all following fields in comm have been initialized. */
    proxyState->tpRank = comm->rank;
    proxyState->tpnRanks = comm->nRanks;
    proxyState->tpLocalnRanks = comm->localRanks;
    proxyState->hggcDev = comm->hggcDev;
    proxyState->abortFlag = comm->abortFlag;
    proxyState->p2pnChannels = comm->p2pnChannels;
    proxyState->nChannels = comm->nChannels;
    proxyState->allocP2pNetLLBuffers = comm->allocP2pNetLLBuffers;
    proxyState->directMode = comm->directMode;
    proxyState->profilerContext = comm->profilerContext;
    memcpy(proxyState->buffSizes, comm->buffSizes, sizeof(comm->buffSizes));

    proxyState->pDevMemUsed = &comm->sharedRes->devMemUsed;
    proxyState->pPinnedMemUsed = &comm->sharedRes->pinnedMemUsed;
    proxyState->ownerComm = comm;
    proxyState->localRank = comm->localRank;
    proxyState->memManager = comm->memManager;

    PTHREADCHECK(pthread_create(&comm->proxyState->thread, NULL, ncclProxyService, comm->proxyState), "pthread_create");
    ncclSetThreadName(comm->proxyState->thread, "PCCL Service %2d", comm->hggcDev);

    // UDS support
    INFO(NCCL_PROXY, "UDS: Creating service thread comm %p rank %d", comm, comm->rank);
    PTHREADCHECK(pthread_create(&comm->proxyState->threadUDS, NULL, ncclProxyServiceUDS, comm->proxyState), "pthread_create");
    ncclSetThreadName(comm->proxyState->threadUDS, "PCCL UDS Service %2d", comm->hggcDev);
  }
  return ncclSuccess;
}

ncclResult_t ncclProxyStop(ncclComm_t comm) {
  if (comm->sharedRes && comm->sharedRes->proxyState) {
    struct ncclProxyState* sharedProxyState = comm->sharedRes->proxyState;
    TRACE_CALL("pcclProxyStop sharedProxyState->refCount %d comm %p rank %d nRanks %d", sharedProxyState->refCount, comm, comm->rank, comm->nRanks);
    if ((comm->proxyRefCountOld = ncclAtomicRefCountDecrement(&sharedProxyState->refCount)) == 0) {
      int saveDevice;
      HGGCCHECK(hggcGetDevice(&saveDevice));
      HGGCCHECK(hggcSetDevice(comm->hggcDev));
      if (__atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE) == 0 && sharedProxyState->peerAddresses) {
        // We need to send a ncclProxyMsgStop message to our own proxy
        struct ncclSocket sock;
        int type = ncclProxyMsgStop;
        NCCLCHECK(ncclSocketInit(&sock, sharedProxyState->peerAddresses+comm->topParentRanks[comm->rank], comm->sharedRes->magic, ncclSocketTypeProxy, NULL));
        if (ncclSocketConnect(&sock) == ncclSuccess) {
          (void)ncclSocketSend(&sock, &type, sizeof(int));
        }
        (void)ncclSocketClose(&sock);
      }

      if (sharedProxyState->peerSocks) {
        int tplocalRanks = comm->sharedRes->tpNLocalRanks;
        for (int i = 0; i< tplocalRanks; i++) {
          if (sharedProxyState->peerSocks[i].fd != -1) {
            if (sharedProxyState->proxyOps[i].pool) {
              NCCLCHECK(ncclShmClose(sharedProxyState->proxyOps[i].pool, NULL, sizeof(struct ncclProxyOpsPool)));
            }
            if (sharedProxyState->sharedDevMems[i]) {
              if (!ncclCuMemEnable() || sharedProxyState->sharedDevMemModes[i] != MEM_NORMAL) {
                HGGCCHECK(hggcIpcCloseMemHandle(sharedProxyState->sharedDevMems[i]));
              } else {
                NCCLCHECK(ncclHgMemFree(sharedProxyState->sharedDevMems[i], sharedProxyState->memManager));
              }
            }
            int type = ncclProxyMsgClose;
            (void)ncclSocketSend(sharedProxyState->peerSocks+i, &type, sizeof(int));
            NCCLCHECK(ncclSocketClose(&sharedProxyState->peerSocks[i]));
          }
        }
      }
      HGGCCHECK(hggcSetDevice(saveDevice));
      // Now we notify proxy service and UDS thread to exit.
      __atomic_store_n(&comm->proxyState->stop, 1, __ATOMIC_RELEASE);
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclProxyDestroy(ncclComm_t comm) {
  struct ncclProxyState* sharedProxyState = comm->sharedRes->proxyState;

  if (sharedProxyState) {
    if (sharedProxyState->refCount != 0) {
      ERROR("Proxy state refCount is %d, expected 0", sharedProxyState->refCount);
      return ncclInternalError;
    }
    // Clear the global dump slot before freeing to prevent use-after-free in ncclDumpProxyState
    if (sharedProxyState->localRank >= 0 &&
        ncclProxyStates[sharedProxyState->localRank] == sharedProxyState) {
      ncclProxyStates[sharedProxyState->localRank] = NULL;
    }
    NCCLCHECK(ncclFree(sharedProxyState->peerAddresses));
    NCCLCHECK(ncclFree(sharedProxyState->peerAddressesUDS));
    NCCLCHECK(ncclFree(sharedProxyState->peerSocks));
    NCCLCHECK(ncclFree(sharedProxyState->proxyOps));
    NCCLCHECK(ncclFree(sharedProxyState->sharedDevMems));

    NCCLCHECK(ncclFree(sharedProxyState));
  }

  return ncclSuccess;
}
