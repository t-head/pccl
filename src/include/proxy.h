/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_PROXY_H_
#define NCCL_PROXY_H_

#include "devcomm.h"
#include "info.h"
#include "socket.h"
#include "ipcsocket.h"
#include <pthread.h>
#include "p2p.h"

enum { proxyRecv=0, proxySend=1 };
enum ncclProxyOpState { ncclProxyOpNone, ncclProxyOpReady, ncclProxyOpProgress };

struct ncclProxyArgs;
typedef ncclResult_t (*proxyProgressFunc_t)(struct ncclProxyState*, struct ncclProxyArgs*);

#define NCCL_PROXY_MAX_SUBS MAXCHANNELS
static_assert(NCCL_MAX_DEV_WORK_P2P_PER_BATCH <= MAXCHANNELS, "Not enough sub space for max work elements");

struct ncclProxyOp {
  struct ncclProxyConnection* connection;
  int channelId;
  int nsteps;
  ssize_t nbytes;
  int root;
  int next;

  uint64_t opCount;
  int sliceSteps;
  int chunkSteps;
  int chunkSize;
  ncclDataType_t dtype;
  ncclRedOp_t redOp;
  uint8_t /*ncclFunc_t*/ coll;
  uint8_t /*ncclFunc_t*/ collAPI;
  uint8_t /*ncclPattern_t*/ pattern;
  uint8_t protocol;
  int sendrecv;
  ncclComm_t perfComm;
  uint8_t* sendbuff;
  uint8_t* recvbuff;

  // Profiler plugin
  union {
      struct ncclTaskColl* coll;
      struct ncclTaskP2p* p2p;
  } task;

  // Profiler work counter increment flag. Set to 'true' if the profiler work counter for this channel needs increment.
  // Always 'true' for collective operations. Grouped p2p operations are fused into one <send, recv> pair in the GPU kernel,
  // meaning the GPU profiler code increments the work counter for the pair rather than the individual p2p. For this
  // reason, the incWorkCounter flag is used to avoid incrementing the work counter twice in the host code. This is done
  // by setting incWorkCounter to 'true' only for one of the p2ps in the pair during enqueue.
  bool incWorkCounter;
  int eActivationMask;
  void* taskEventHandle;
  int rank;
  int peer;
  pid_t pid;
  void* profilerContext;
  uint64_t workCounter;

  struct ncclProxyOp *enqNext;
};

// TODO: current p2p enqueue & kernel can't separate p2p send/recv ncclProxyOp
// static_assert(sizeof(struct ncclProxyOp) == 64, "Keep ProxyOp aligned with cache lines for effective prefetch");

struct ncclProxyEventHandle {
  void* stepEventHandle;
  struct ncclProxySubArgs* subArgPtr;
};

struct ncclProxySubArgs {
  struct ncclProxyConnection* connection;
  uint8_t* sendbuff;
  uint8_t* recvbuff;
  int channelId;
  int nsteps;
  ssize_t nbytes;
  int peer;
  int sendrecv;

  int groupSize; // Number of consecutive sub operations sharing the same recvComm
  uint64_t base;
  uint64_t posted;
  uint64_t received;
  uint64_t flushed;
  uint64_t transmitted;
  uint64_t done;
  uint64_t end;
  void* requests[NCCL_STEPS];
  uint64_t asyncFlushTicket[NCCL_STEPS]; // async CPU flush: per-step request ticket
  void* profilingEvents[NCCL_STEPS];

  // Profiler plugin
  int eActivationMask;
  int rank;
  pid_t pid;
  void* profilerContext;
  void* taskEventHandle;
  void* opEventHandle;
  void* kernelEventHandle;
  struct ncclProxyEventHandle pHandles[NCCL_STEPS];
  size_t transSize;
  uint64_t workCounter;
};

struct ncclProxyArgs {
  struct ncclProxySubArgs subs[NCCL_PROXY_MAX_SUBS];
  proxyProgressFunc_t progress;
  int nsubs;
  int done;
  uint64_t opCount;
  int sliceSteps;
  int chunkSteps;
  int chunkSize;
  ncclDataType_t dtype;
  ncclRedOp_t redOp;
  uint8_t /*ncclPattern_t*/ pattern;
  uint8_t protocol;
  int state;
  char* sharedBuff[NCCL_STEPS];
  int sharedSize[NCCL_STEPS];

  int idle;
#ifdef NET_BYPASS
  int bypass;
#endif
  ncclComm_t perfComm;

  // Element linking
  struct ncclProxyArgs* next;
  struct ncclProxyArgs* nextPeer;
  struct ncclProxyArgs** proxyAppendPtr;
};
#define NCCL_MAX_NETDEVS 128

// ProxyOps are used to communicate between main thread and service thread
// Make sure we have enough to store two full rounds of operations on all channels.
// Otherwise we'd be unable to post half of them to free new elements.
#define MAX_OPS_PER_PEER (2*MAXCHANNELS*NCCL_MAX_DEV_WORK_P2P_PER_BATCH)
#define NCCL_MAX_PROXY_CONNECTIONS (NCCL_MAX_LOCAL_RANKS + 1) // 16 + 1 (one for destroy)
struct ncclProxyOpsPool {
  struct ncclProxyOp ops[MAX_OPS_PER_PEER*NCCL_MAX_PROXY_CONNECTIONS];
  volatile int nextOps;
  volatile int nextOpsEnd;
  volatile int freeOps[NCCL_MAX_PROXY_CONNECTIONS];
  pthread_mutex_t mutex;
  pthread_cond_t cond;
};

struct ncclProxyOps {
  ncclProxyOpsPool* pool;
  int count;
  int freeOp;
  int nextOps;
  int nextOpsEnd;
};

struct ncclProxySharedP2p {
  int refcount;
  int size;
  char* hggcBuff;
  char* hostBuff;
  // HGGC IPC
  ncclIpcDesc ipcDesc;
  struct ncclProxyArgs* proxyAppend[MAXCHANNELS]; // Separate send and recv
};

struct ncclProxySharedCollNet {
  int size;
  char* hggcBuff;
  char* hostBuff;
  struct ncclProxyArgs* proxyAppend[2*NCCL_MAX_NETDEVS];
  void* resources;
};

struct ncclProxyPeer {
  struct ncclProxySharedP2p send;
  struct ncclProxySharedP2p recv;
};

struct ncclSharedNetComms {
  int activeConnect[MAXCHANNELS];
  int activeAccept[MAXCHANNELS];
  void* sendComm[MAXCHANNELS];
  void* recvComm[MAXCHANNELS];
  int sendRefCount[MAXCHANNELS];
  int recvRefCount[MAXCHANNELS];
};

struct ncclProxyPool;
int64_t pcclParamCpuFlushAsync();
struct ncclProxyProgressState {
  // Used by main threads to send work to progress thread
  struct ncclProxyOpsPool* opsPool;
  char opsPoolShmSuffix[6];

  pthread_t thread;
  bool stop;
  struct ncclProxyPeer** localPeers;
  struct ncclSharedNetComms* netComms[NCCL_MAX_NETDEVS];
  struct ncclProxySharedCollNet collNet;
  struct ncclProxyArgs* active;
  struct ncclProxyArgs* pool;
  struct ncclProxyPool* pools;
  int nextOps;

  // Async CPU flush (only active when PCCL_CPU_FLUSH_ASYNC=1).
  // Cold fields: set once at thread create/destroy.
  pthread_t flushThread;
  int flushThreadStarted;
  volatile int flushThreadStop;
  volatile uint64_t* flushScratch;
  void* flushScratchDesc;
  // Hot cross-core counters. Isolate each on its own cache line to avoid false
  // sharing. flushReqSeq is bumped (atomic add) only by the recv progress thread
  // when it takes an async flush ticket; flushDoneSeq is written only by the
  // flush thread after a coalescing read. A step's ticket completes once
  // flushDoneSeq>=ticket: the read that advanced flushDoneSeq observed
  // flushReqSeq>=ticket, so it was issued after the ticket (after the step's
  // data arrived) -- a happens-before, not a timing assumption.
  alignas(64) volatile uint64_t flushReqSeq;
  alignas(64) volatile uint64_t flushDoneSeq;
  char flushDonePad[56]; // fill the rest of flushDoneSeq's cache line
};

// Expected proxy response fifo
struct ncclExpectedProxyResponse {
  void* opId;
  int respSize;
  bool done;
  void* respBuff;
  ncclResult_t res;
  struct ncclExpectedProxyResponse* next;
};

struct ncclProxyAsyncOp {
  int type;
  struct ncclProxyConnection* connection;
  int reqSize, respSize;
  char *reqBuff, *respBuff;
  void* opId;
  ncclProxyAsyncOp* next;
};

struct ncclProxyLocalPeer {
  struct ncclSocket sock;
  int tpRank;
  int tpLocalRank;
  struct ncclProxyAsyncOp* asyncOps;
  int asyncOpCounter;
};

// Common response header for all proxyOps
// We pack this into a struct to reduce the number of blocking send and recv calls
struct ncclProxyRpcResponseHeader {
  void* opId;
  ncclResult_t res;
  int respSize;
};

// UDS support
struct ncclIpcHdr {
  int type;
  int rank;
  int reqSize;
  int respSize;
  void *opId;
  uint64_t data[16]; // 128-bytes
};

struct ncclProxyState {
  struct ncclMemManager* memManager;
  int refCount;
  int tpRank;
  int tpnRanks;
  int tpLocalnRanks;
  int hggcDev;
  int p2pnChannels;
  int nChannels;
  bool allocP2pNetLLBuffers;
  int buffSizes[NCCL_NUM_PROTOCOLS];
  volatile uint32_t* abortFlag;
  bool directMode;
  // Service thread
  pthread_t thread;
  pthread_t threadUDS;
  struct ncclSocket* listenSock;
  struct ncclIpcSocket ipcSock;
  int stop;
  HGcontext hggcCtx;

  // Used by main thread
  union ncclSocketAddress* peerAddresses;
  struct ncclSocket* peerSocks;
  struct ncclProxyOps* proxyOps;
  void** sharedDevMems;
  int* sharedDevMemModes;
  uint64_t* peerAddressesUDS;

  // Profiler plugin
  void* profilerContext;

  // Progress thread
  struct ncclProxyProgressState progressState;
  ncclResult_t asyncResult;
  pcclComm_t ownerComm;
  int localRank; // local rank index used to register/unregister in ncclProxyStates[]
  uint64_t* pDevMemUsed;
  uint64_t* pPinnedMemUsed;

  // Queue of expected responses from the proxy
  struct ncclExpectedProxyResponse* expectedResponses;
};

enum proxyConnectState {
  connUninitialized     = 0,
  connInitialized       = 1,
  connSharedInitialized = 2,
  connSetupDone         = 3,
  connConnected         = 4,
  numConnStates         = 5
};

struct ncclProxyConnection {
  int send, transport, shared;
  int tpLocalRank, sameProcess;
  struct ncclSocket* sock;
  struct ncclTransportComm* tcomm;
  struct ncclProxyArgs *proxyAppend;
  struct ncclProxyArgs **proxyAppendPtr;
  void* transportResources;
  proxyConnectState state;
};

typedef ncclResult_t (*threadFunc_t)(struct ncclProxyArgs*);

enum proxyMode {
  proxyRing = 0,
  proxyFrom = 1,
  proxyTo = 2
};

ncclResult_t ncclProxySaveOp(ncclComm_t comm, struct ncclProxyOp* op, bool* justInquire);
ncclResult_t ncclProxyStart(ncclComm_t comm);
ncclResult_t ncclProxyInit(ncclComm_t comm, struct ncclSocket* sock, union ncclSocketAddress* peerAddresses, uint64_t* peerAddressesUDS);
ncclResult_t ncclProxyCreate(ncclComm_t comm);
ncclResult_t ncclProxyConnect(ncclComm_t comm, int transport, int send, int proxyRank, struct ncclProxyConnector* proxyConn);
enum ncclProxyMsgType {
  ncclProxyMsgInit = 1,
  ncclProxyMsgSharedInit = 2,
  ncclProxyMsgSetup = 3,
  ncclProxyMsgConnect = 4,
  ncclProxyMsgStart = 5,
  ncclProxyMsgClose = 6,
  ncclProxyMsgAbort = 7,
  ncclProxyMsgStop = 8,
  ncclProxyMsgGetFd = 9, // cuMem API support (UDS)
  ncclProxyMsgQueryFd = 10, // cuMem API support (UDS)
  ncclProxyMsgRegister = 11 ,
  ncclProxyMsgDeregister = 12
};

// This function is called by a client of the proxy that needs to invoke any of the non-progress proxyOp types
// Call this function on the client, supplying a locally unique opId. Then, poll on the return value of
// ncclPollProxyResponse(), supplying the same opId to confirm the operation has completed
ncclResult_t ncclProxyCallAsync(ncclComm_t comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, int respSize, void* opId);

ncclResult_t ncclPollProxyResponse(ncclComm_t comm, struct ncclProxyConnector* proxyConn, void* respBuff, void* opId);

// This function will internally call ncclProxyCallAsync() and spin until ncclPollProxyResponse() confirms the result is received
ncclResult_t ncclProxyCallBlocking(ncclComm_t comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, void* respBuff, int respSize);

ncclResult_t ncclProxyClientGetFdBlocking(ncclComm_t comm, int proxyRank, void *handle, int* convertedFd);

ncclResult_t ncclProxyClientQueryFdBlocking(ncclComm_t comm, struct ncclProxyConnector* proxyConn, int localFd, int* rmtFd);

ncclResult_t ncclProxyStop(ncclComm_t comm);
ncclResult_t ncclProxyShmUnlink(ncclComm_t comm);
ncclResult_t ncclProxyDestroy(ncclComm_t comm);

ncclResult_t pcclDumpProxyState(struct ncclProxyState* proxyState, FILE* dumpFile=NULL);
#endif
