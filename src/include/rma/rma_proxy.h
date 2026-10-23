/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_RMA_PROXY_H_
#define PCCL_RMA_PROXY_H_

#include "pccl_internal.h"
#include "pccl_common.h"
#include "alloc.h"
#include "utils.h"
#include "strongstream.h"
#include "dev_runtime.h"
#include "rma/rma_types.h"
#include <thread>
#include <mutex>
#include <condition_variable>

struct ncclRmaArgs;
struct ncclKernelPlan;

struct ncclRmaSignal_t {
  void* signalMhandle;
  uint64_t offset;
  uint64_t val;
  uint32_t op;
};

typedef enum ncclRmaDescState_t {
  ncclRmaDescStateInit = 0,
  ncclRmaDescStateReady = 1,
  ncclRmaDescStateInProgress = 2,
} ncclRmaDescState_t;

typedef enum ncclRmaDescType_t {
  ncclRmaDescTypePutSignal = 0,
  ncclRmaDescTypeWaitSignal,
  ncclRmaDescTypePutSignalGroup,
} ncclRmaDescType_t;

struct ncclRmaPutSignalOp {
  // Network function descriptor
  uint64_t srcOff;
  void* srcHandle;
  uint64_t dstOff;
  void* dstHandle;
  size_t size;
  int targetRank;
  ncclRmaSignal_t signal;
  // Request handle for the network operation
  void* request;
};

struct ncclRmaWaitSignalOp {
  int npeers;
  int* waitPeers;
  int* waitSignals;
  int* waitSignalIdxs;
  // Local flush in graph mode
  int needFlush;
};

struct ncclRmaPutSignalGroupOp {
  int nOps;
  struct ncclRmaPutSignalOp* ops;
  int nIssued;
  int nCompleted;
};

struct ncclRmaProxyDesc {
  struct ncclRmaProxyDesc* next;
  ncclRmaDescType_t rmaDescType;
  ncclRmaDescState_t rmaDescState;

  union {
    struct ncclRmaPutSignalOp putSignal;
    struct ncclRmaWaitSignalOp waitSignal;
    struct ncclRmaPutSignalGroupOp putSignalGroup;
  };

  // Non graph mode, desc does not own the sequence allocations but points to the ctx's sequence allocations
  // Graph mode, desc owns the per-descriptor sequence allocations and this needs to be freed when the desc is
  // destroyed
  uint64_t opSeq;
  uint64_t* readySeq;
  uint64_t* readySeqDev;
  void* readySeqGdrHandle;
  uint64_t* doneSeq;
  uint64_t* doneSeqDev;
  void* doneSeqGdrHandle;

  // Graph capture fields
  struct ncclKernelPlan* persistPlan; // Back reference to persistent plan during clean up
  bool captured;
  struct ncclCudaGraph graph;
  bool persistDescValid; // Persistent descriptor is valid
};

struct ncclRmaProxyCtx {
  ncclComm_t comm;

  // GIN context for the RMA proxy context
  void* rmaCollComm;
  void* rmaCtx;
  // Index into comm->sharedRes->ginState.ginComms / ginProps used for this context's
  // data-window handles.
  int ginCommIdx;

  //---------Non-graph descriptor queues and synchronization---------

  // Lock-free circular buffer for pending Descs
  size_t queueSize;  // Power of 2 size for pending queue
  struct ncclRmaProxyDesc** circularBuffers;  // Lock-free circular buffer per peer
  uint32_t* pis;  // Producer Indices per peer
  uint32_t* cis;  // Consumer Indices per peer

  // Per-rank inProgressQueues: Descs with issued network operations waiting for completion
  struct ncclIntruQueue<struct ncclRmaProxyDesc, &ncclRmaProxyDesc::next>* inProgressQueues;

  // Per-target-rank request credits. Each target rank maps to one RMA send comm request pool.
  uint32_t maxInflightRequests;
  uint32_t* inflightRequests;

  // Per-rank sequence number and counters
  uint64_t* opSeqs;
  uint64_t* opSeqsDev;
  void* opSeqsGdrHandle;
  uint64_t* readySeqs;
  uint64_t* readySeqsDev;
  void* readySeqsGdrHandle;
  uint64_t* doneSeqs;
  uint64_t* doneSeqsDev;
  void* doneSeqsGdrHandle;

  // Signal memory layout and management
  // Each RMA context allocates a signal buffer with the following layout:
  // - One set of nRanks per-rank signal counters per signal index
  // - Slot within this context is signalIdx * nRanks + rank
  // Total signal buffer size: (numRmaSig * nRanks) * 8 bytes
  HGmemGenericAllocationHandle signalsCumemhandle;
  void* signalsMhandle;
  uint64_t* signalsDev;
  uint64_t* signalsHost; // Host buffer to track the expected values of the signals

  //---------Graph descriptor queues and synchronization---------

  // Per-rank persistent descriptor queue: Descs from all live graphs
  struct ncclIntruQueue<struct ncclRmaProxyDesc, &ncclRmaProxyDesc::next>* persistentQueues;

  // CPU-accessible signal is required as proxy needs to poll on the signal values
  // Graph/persistent signals use the same per-(signalIdx, rank) layout.
  void* cpuAccessSignalsGdrHandle;
  void* cpuAccessSignalsMhandle;
  uint64_t* cpuAccessSignals;
  uint64_t* cpuAccessSignalsDev;
  uint64_t* cpuAccessSignalsHost; // Host buffer to track the expected values of the signals

  // Local flush buffer
  HGmemGenericAllocationHandle flushBufCumemhandle;
  void* flushBufMhandle;
  uint64_t* flushBufDev;
};

struct ncclRmaProxyState {
  ncclComm_t comm;
  bool connected;

  // Virtual RMA proxy contexts
  int rmaProxyCtxCount;
  void** rmaProxyCtxs;
  int rmaProgress;         // RMA progress is enabled
  std::thread thread;
  std::mutex mutex;
  std::condition_variable cond;
  ncclResult_t asyncResult;
};

// Proxy-specific function declarations
ncclResult_t ncclRmaProxyPutLaunch(ncclComm_t comm, struct ncclKernelPlan* plan, hggcStream_t stream);
ncclResult_t ncclRmaProxyWaitLaunch(ncclComm_t comm, struct ncclKernelPlan* plan, hggcStream_t stream);
ncclResult_t ncclRmaProxyReclaimPlan(ncclComm_t comm, struct ncclKernelPlan* plan);

// RMA Proxy lifecycle functions
ncclResult_t ncclRmaProxyConnectOnce(ncclComm_t comm);
ncclResult_t ncclRmaProxyFinalize(ncclComm_t comm);

// RMA Proxy context management
ncclResult_t ncclRmaProxyCreateContext(ncclComm_t comm, void* collComm, int ginCommIdx, void** outRmaProxyCtx);
ncclResult_t ncclRmaProxyDestroyContext(ncclComm_t comm, void* rmaProxyCtx);
ncclResult_t ncclRmaProxyProgress(ncclComm_t comm, void* rmaProxyCtx);
void* ncclRmaProxyProgressThread(struct ncclRmaProxyState* rmaProxyState);

// Circular buffer helpers
bool ncclRmaProxyCircularBufFull(struct ncclRmaProxyCtx* ctx, int peer);
bool ncclRmaProxyCircularBufEmpty(struct ncclRmaProxyCtx* ctx, int peer);

// Returns true if the queue this descriptor would enqueue into is full.
bool ncclRmaProxyEnqueueFull(struct ncclRmaProxyCtx* ctx, const struct ncclRmaProxyDesc* desc);

// Descriptor API: build the desc, issue its memops on the user stream, then
// hand it over with EnqueueDesc. Issue the memops before EnqueueDesc: it may
// free the desc (non-persistent wait), and the memops read its fields.

// ---- Descriptor builders ----

// Build a single put-signal op. Used by ncclRmaProxyPutBuildDesc and
// ncclRmaProxyPutGroupBuildDesc.
ncclResult_t ncclRmaProxyPutBuildOp(ncclComm_t comm, struct ncclRmaProxyCtx* rmaProxyCtx, int ctx,
                                    bool persistent, struct ncclDevrWindow* srcWin, size_t srcOff,
                                    struct ncclDevrWindow* peerWin, size_t peerOff, size_t size, int peer,
                                    int signalIdx, ncclSignalMode_t signalMode, struct ncclRmaPutSignalOp* op);

// Build a single put descriptor.
ncclResult_t ncclRmaProxyPutBuildDesc(ncclComm_t comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                      struct ncclKernelPlan* plan, struct ncclDevrWindow* srcWinHost,
                                      size_t srcWinOffset, struct ncclDevrWindow* peerWinHost, size_t peerWinOffset,
                                      size_t size, int peer, int ctx, int signalIdx, ncclSignalMode_t signalMode,
                                      struct ncclRmaProxyDesc* desc);

// Build a put-signal-group descriptor over an array of pre-filled ops.
// Takes ownership of *ops and nulls the caller's slot on success.
ncclResult_t ncclRmaProxyPutGroupBuildDesc(ncclComm_t comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                           struct ncclKernelPlan* plan, int nOps, struct ncclRmaPutSignalOp** ops,
                                           int ctx, struct ncclRmaProxyDesc* desc);

// Build a wait-signal descriptor.
// Deep-copies the task's peers/nsignals/signalIdxs arrays (they live in
// comm->memScoped and die with the group scope).
ncclResult_t ncclRmaProxyWaitBuildDesc(ncclComm_t comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                       struct ncclKernelPlan* plan, int npeers, int** peers, int** nsignals,
                                       int** signalIdxs, struct ncclRmaProxyDesc* desc);

// ---- Stream memop issue ----

ncclResult_t ncclRmaProxyPutStart(struct ncclRmaProxyDesc* desc, hggcStream_t stream);
ncclResult_t ncclRmaProxyPutDone(struct ncclRmaProxyDesc* desc, hggcStream_t stream);
ncclResult_t ncclRmaProxyPutGroupStart(struct ncclRmaProxyDesc* desc, hggcStream_t stream);
ncclResult_t ncclRmaProxyPutGroupDone(struct ncclRmaProxyDesc* desc, hggcStream_t stream);
ncclResult_t ncclRmaProxyWaitOnStream(struct ncclRmaProxyCtx* rmaProxyCtx, struct ncclRmaProxyDesc* desc,
                                      hggcStream_t stream);

// Descriptor enqueue dispatcher.
ncclResult_t ncclRmaProxyEnqueueDesc(struct ncclRmaProxyCtx* rmaProxyCtx, struct ncclRmaProxyDesc** desc);

// Descriptor destruction. Takes desc** and nulls *desc after free.
ncclResult_t ncclRmaProxyDestroyDesc(ncclComm_t comm, struct ncclRmaProxyDesc** desc);

#endif
