/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_COMM_H_
#define NCCL_COMM_H_

#include "p2p_info.h"
#include "tuner.h"
#include "ext_kernel_plugin.h"
#include "proxy.h"
#include "register.h"
#include "graph.h"
#include "profiler.h"
#include "strongstream.h"
#include "ce_coll.h"
#include "p2p_dma.h"
#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
#include "lt_oki.h"
#endif

#include "dev_runtime.h"
#include "sym_kernels.h"
#include "gin/gin_host.h"

#define CACHE_LINE_SIZE 128
#define MEM_ALIGN 4096
#define HGGC_IPC_MIN 2097152UL

#define MAX_GPUS_PER_NODE 16

#include "mem_manager.h"

// Channels / LL tuning
#define NCCL_LL_THREAD_THRESHOLD 8
#define NCCL_LL128_THREAD_THRESHOLD 8
#define NCCL_SIMPLE_THREAD_THRESHOLD 64
#define NCCL_SIGNAL_THREAD_THRESHOLD 64
#define NCCL_SIMPLE_BULK_THREAD_THRESHOLD 64
#define NCCL_SIGNAL_BULK_THREAD_THRESHOLD 64
#define NCCL_SIMPLE_BULK_AR_THREAD_THRESHOLD 64

#define MAX_USER_OBJS 1024

// 16384 char supports max 64 peers alltoall
#define MAX_HGTX_MSG_LENGTH 16384
#define ONE_HGTX_MSG_LENGTH 128

struct ncclSendMem {
  union {
    struct {
      uint64_t head;
      char pad1[CACHE_LINE_SIZE-sizeof(uint64_t)];
      void* ptrExchange;
      char pad2[CACHE_LINE_SIZE-sizeof(void*)];
    };
    char pad3[MEM_ALIGN];
  };
  char buff[1]; // Actually larger than that
};

struct ncclCommCallback {
  struct ncclCommCallback* next;
  ncclResult_t(*fn)(ncclComm_t comm, struct ncclCommCallback* cb);
};

struct ncclCommEventCallback {
  struct ncclCommEventCallback* next;
  hggcEvent_t event;
  ncclResult_t(*fn)(ncclComm_t comm, struct ncclCommEventCallback* cb);
};

struct pcclExtKernelPluginWorkspace {
  void* devPtr;
  ncclIpcDesc ipcDesc;
  ssize_t size;
  bool sameProc[NCCL_MAX_LOCAL_RANKS];
  int peerIpcPtrNum;
  void* peerIpcPtrs[NCCL_MAX_LOCAL_RANKS];

  void* extKernelPluginParams;
};

struct ncclRecvMem {
  union {
    struct {
      uint64_t tail;
      char pad1[CACHE_LINE_SIZE-sizeof(uint64_t)];
      uint32_t signalMarker[NCCL_STEPS];
      char pad2[CACHE_LINE_SIZE-sizeof(uint32_t)*NCCL_STEPS];
      int sizesFifo[NCCL_STEPS];
      int offsFifo[NCCL_STEPS];
      int flush; // For GDRCopy-based flush
    };
    char pad4[MEM_ALIGN];
  };
  char buff[1]; // Actually larger than that
};

struct ncclUserRedOp {
  int freeNext; // -1=allocated, otherwise index of next free entry in array
  ncclDataType_t datatype;
  ncclDevRedOpFull opFull;
};

struct ncclNodeRanks {
  int localRanks;
  int* localRankToRank;
};

struct cliqueInfo {
  int id;
  int size;
  int *ranks;
};

struct ncclSharedResources {
  int refCount;
  ncclComm_t owner; /* comm which creates this shared res. */
  struct ncclChannelPeer* peers[MAXCHANNELS];
  struct ncclDevChannelPeer* devPeers[MAXCHANNELS];
  /* P2P operation counter, one per channel */
  uint64_t p2pOpCount[MAXCHANNELS];
  /* Collective operation counter */
  uint64_t collOpCount;
  int tpNRanks;
  int tpNLocalRanks;
  int tpNChannels;
  int tpP2pNChannels;
  int tpP2pChunkSize;
  uint64_t magic;

  // top parent rank to localRank translation table
  int* tpRankToLocalRank;
  // Internal streams
  struct ncclStrongStream deviceStream, hostStream;
  int persistentRefs;
  hggcEvent_t launchEvent, scratchEvent;

  /* proxy related shared res */
  struct ncclProxyState* proxyState;

  uint64_t devMemUsed;
  uint64_t pinnedMemUsed;

  struct pcclGinState ginState;
};

struct ncclBuffRegInfo {
  uintptr_t sendbuffOffset;
  uintptr_t recvbuffOffset;
  uintptr_t* sendbuffRmtAddrs;
  uintptr_t* recvbuffRmtAddrs;
  int nBuffs;
};

struct ncclWorkBatchList {
  struct ncclWorkBatchList* next;
  struct ncclDevWorkBatch batch;
};

struct alignas(16) ncclWorkList {
  struct ncclWorkList* next;
  enum ncclDevWorkType workType;
  int size; // Size of struct following this node
  // ncclDevWorkColl, ncclDevWorkColLReg, ncclDevWorkP2p[]...
};

struct ncclTaskColl {
  struct ncclTaskColl* next;
  ncclFunc_t func;
  void const* sendbuff;
  void* recvbuff;
  size_t count;
  int root;
  ncclDataType_t datatype;
  ncclRedOp_t opHost;
  struct ncclDevRedOpFull opDev;
  int chunkSteps;
  int sliceSteps;

  int algorithm;
  int protocol;
  ncclPattern_t pattern;
  int nThreads;
  size_t nBytes;
  uint32_t devFuncId;

  bool isExtKernel;
  int regUsed;
  int regBufType;
  uintptr_t recvbuffOffset;
  uintptr_t sendbuffOffset;
  uintptr_t* recvbuffRmtAddrs;
  uintptr_t* sendbuffRmtAddrs;
  // number of elements in planner->ipcMemQueue associated with this collective
  int nCleanupQueueElts;

  // Profiler plugin
  int eActivationMask;
  void* groupApiEventHandle;
  void* collApiEventHandle;
  void* eventHandle;
  uint8_t nChannels;

  // Symmetric kernel (symk) path fields
  struct ncclDevrWindow* sendWin;
  struct ncclDevrWindow* recvWin;
  ncclSymRegType_t winRegType;
  void* sendMhandle;
  void* recvMhandle;
  void** sendNetHandles;
  void** recvNetHandles;
  void** srecvNetHandles;
  uint32_t isSymLast:1;  // last task in multi-task symmetric group
  int nMaxChannels;      // kernel-selected max channels for this task group
  int nWarps;            // kernel-selected warp count for this task group
};

// Precomputed settings for one channel portion and one protocol.
struct ncclA2aPartPolicy {
  uint8_t sendWindow;
  uint8_t pollFlags;
  uint8_t singleQp[2]; // [0] dedicated buffer, [1] shared buffer.
};

// One AlltoAll call owns this CPU-only object in comm->memScoped. Its tasks
// borrow the pointer; connections never retain it. The layout is prepared at
// entry, request settings after preconnect, then both are read-only.
struct ncclA2aCallPolicy {
  size_t peerBytes; // This invocation's payload for one peer.
  int partProtocol; // Base protocol used for its channel split.
  uint64_t netSendConnections; // Stops at minNetConnections; not a full connection count.
  uint64_t minNetConnections; // Per-call snapshot of the enable threshold.
  bool connectionsReady; // Finalized once, after preconnect and before planning.
  ssize_t partBytes; // Exact result of the existing channel-split helper.
  int partChannels;
  // [full portion / short tail][base protocol / optional LL].
  ncclA2aPartPolicy parts[2][2];

  int64_t sendWindowMinBytes; // Per-network-task byte threshold, not per-peer bytes.
  uint8_t sendWindow; // Zero: no reduced window; tasks retain NCCL_STEPS.
  uint32_t pollTicks; // Base timer ticks at pollReference active sends.
  uint16_t pollReference; // Calibrated concurrency; zero disables waiting.
  uint8_t pollFlags; // PCCL_A2A_POLL_* bits defined in a2a_send_poll.h.
  bool optimized; // Candidate at entry, final decision when connectionsReady is true.
};

struct ncclTaskP2p {
  ncclTaskP2p *next;
  ncclFunc_t func;
  ncclFunc_t collAPI;
  ncclA2aCallPolicy* a2aCallPolicy; // Shared per-call CPU decision, or null.
  bool a2aOptimized; // Request-local eligibility; never a permanent connection flag.
  bool a2aAutoAsyncFlush; // Independent of the request-policy connection threshold.
  void *buff;
  size_t count;
  ncclDataType_t datatype;
  int root;
  size_t bytes;

  // Profiler plugin
  int eActivationMask;
  void* groupApiEventHandle;
  void* p2pApiEventHandle;
  void* eventHandle;
  uint8_t nChannels;
};

struct ncclKernelPlan {
  // A kernel plan is also a callback that reclaims itself. Hence this must
  // be the first member.
  struct ncclCommCallback reclaimer;

  ncclComm_t comm;
  struct ncclKernelPlan* next;

  bool persistent; // aka captured in a graph
  bool isHostCbEnq;
  bool isSymColl;
  bool isCeColl;
  bool isCeP2p;
  enum ncclDevWorkStorageType workStorageType;
  void *kernelFn;
  uint16_t funcIndex;
  uint16_t symFuncIndex;

  void* graphWorkBase;
  union {
    struct ncclDevKernelArgs* kernelArgs;
    void* kernelSymkArgs;
    struct ncclCeCollArgs ceCollArgs;
  };
  size_t kernelArgsSize;
  uint64_t channelMask; // bitset of which channels are present
  bool hasProxyOps; // does any channel have a non-empty proxyOpQueue
  int threadPerBlock;

  int collOpCount; // Number of collectives in this plan.
  int nWorkBatches; // Number of work batches.
  size_t workBytes; // Sum size of all work (in the fifo) in bytes.
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> workQueue;
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next> cleanupQueue;
  void* workBufPersistent;

  struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> p2pTaskQueue;
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collTaskQueue;
  struct ncclIntruQueue<struct ncclProxyOp, &ncclProxyOp::enqNext> proxyOpQueue;

  bool usingExtKernel;

  // Profiler plugin
  void* groupApiEventHandle;
  void* kernelLaunchEventHandle;
  void* groupEventHandle;

  uint64_t channelWorks[MAXCHANNELS];
};

struct ncclCudaStreamList {
  struct ncclCudaStreamList *next;
  hggcStream_t stream;
};

struct ncclKernelPlanner {
  struct Peer {
    bool sendSeen, recvSeen;
    struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> sendQueue;
    struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> recvQueue;
  };

  size_t collBytesTotal;
  struct Peer* peers/*[nRanks]*/;
  int nTasksColl;
  int nTasksP2p;
  int nTasksP2pAlltoall; // subset of nTasksP2p coming from the Alltoall API
  bool a2aHasCallPolicies; // Group-scoped: skip policy walks when every call fell back early.
  bool a2aHasAutoAsyncFlush; // Also check mixed-group fallback for small AlltoAll calls.
  bool persistent;
  bool isCeColl;
  bool isExtKernel;
  // Sequential counter used to stamp a distinct kernelIdx per plan in uploadWork.
  // Reset to 0 at the start of each launch (ncclLaunchPrepare) and advanced once
  // per uploadWork call. A group is split into multiple plans (each plan == one
  // launch == one kernel); Phase B runs uploadWork for every plan before Phase C
  // launches any, so comm->kernelNum stays fixed during Phase B and a naive
  // kernelNum+1 stamps the same value on every plan. Adding this per-launch offset
  // makes plan i get kernelNum+1+i, matching the launch order (== FIFO slot+1), so
  // the PCCL_DUMP_WORK_ELEMS trace shows sequential kernelIdx instead of duplicates.
  uint64_t uploadKernelSeq;

  // The list of user streams aggregated over all tasks present.
  ncclCudaStreamList* streams;
  // The most recent user stream. Ignored if streams==nullptr
  hggcStream_t streamRecent;
  // The graph capturing all user streams or invalid if none. Thus we restrict the
  // user that all streams must be captured in the same graph or not captured
  // at all. Technically we could probably relax this, but that would mean
  // collecting a different `ncclKernelPlanner` per graph and one for non-graph.
  struct ncclCudaGraph capturingGraph;

  //////////////////////////////////////////////////////////////////////////////
  // Lists of tasks to be assembled into plans.
  //////////////////////////////////////////////////////////////////////////////

  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collTaskQueue;
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collCeTaskQueue;
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collSymTaskQueue;
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> collWorkQueue;
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next> collCleanupQueue;

  //////////////////////////////////////////////////////////////////////////////
  // State for building current (Work-In-Progress) plan:
  //////////////////////////////////////////////////////////////////////////////

  struct WipPlan {
    struct Channel {
      struct {
        int workBytes; // Sum size of work metadata referenced by this batch.
        int nP2ps; // Number of p2p works in this batch
        int p2pRounds[NCCL_MAX_DEV_WORK_P2P_PER_BATCH]; // which rounds are present in this batch.
        int p2pEpoch; // which p2p epoch (schedule sweep) this batch belongs to.
      } wipBatch; // work-in-progress batch which will be next tail of workBatchQueue
      int nWorkBatchesP2p; // number of p2p batches for this channel.
      struct ncclIntruQueue<struct ncclWorkBatchList, &ncclWorkBatchList::next> workBatchQueue;
      struct ncclIntruQueue<struct ncclProxyOp, &ncclProxyOp::enqNext> proxyOpQueue;
    } channels[MAXCHANNELS];
  } wipPlan;

  //////////////////////////////////////////////////////////////////////////////
  // State for launching built plans:
  //////////////////////////////////////////////////////////////////////////////

  // List of kernel plans built form tasks.
  struct ncclIntruQueue<struct ncclKernelPlan, &ncclKernelPlan::next> planQueue;
  // First of the unlaunched kernels in `planQueue`
  struct ncclKernelPlan* unlaunchedPlansHead;
};

typedef enum ncclGroupTaskType {
  ncclGroupTaskTypeCollective = 0,
  ncclGroupTaskTypeSymRegister = 1,
  ncclGroupTaskTypeNum = 2,
} ncclGroupTaskType_t;

struct KernelInfoMonitor {
  ncclDevKernelArgs1K kernelArgs;
};

struct pcclComm {
  struct ncclMemoryStack memPermanent;
  struct ncclMemoryStack memScoped;
  struct ncclSharedResources* sharedRes;
  /* map to top parent ranks. */
  int* topParentRanks;
  int* topParentLocalRanks;
  struct ncclChannel channels[MAXCHANNELS];
  uint64_t commHash;
  // number of splits performed on this comm, used to derive unique child comm hashes
  int splitCount;
  bool directMode;    // used to disble register for legecyIPC when at least two ranks in one process.

  struct ncclPeerInfo* peerInfo;
  struct ncclTopoSystem* topo;
  struct ncclProxyConnector* gproxyConn;
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next> legacyRegCleanupQueue;

  ncclConn_t maxConnType;
  bool peerInfoValid;

  void* bootstrap;
  // Bitmasks for ncclTransportP2pSetup
  uint64_t* connectSend;
  uint64_t* connectRecv;
  struct ncclTopoGraph graphs[NCCL_NUM_ALGORITHMS];
  bool initAlgoChannels[NCCL_NUM_ALGORITHMS];
  bool runtimeConn; // if dynamic connection is supported
  int cuMemSupport;

  uint64_t magic; // Magic number for all network communication. Not a security key -- only goal is to detect mismatches.

  int rank;    // my rank in the communicator
  int nRanks;  // number of GPUs in communicator
  int hggcDev; // my hggc device index
  int hgmlDev; // my HGML device index
  int compCap; // compute capability of the PPU
  int minCompCap, maxCompCap; // min/max compute capability in the communicator
  PPU_TYPE ppuType; // PPU model type
  pcclICNRemoteDeviceType_t icnRmtType; // ICN remote device type
  int64_t busId;   // my PCI bus ID in int format
  cpu_set_t cpuAffinity; // CPU affinity of the GPU
  int hgmlDeviceCount;
  int node;
  int nNodes;
  int localRank;
  int localRanks;
  int maxLocalRanks;
  int* rankToNode;
  int* rankToLocalRank;
  int* localRankToRank;
  // localRanks and localRanktoRank for all nodes
  struct ncclNodeRanks* nodeRanks;

  // MNNVL: Multi-Node NVLink
  int MNNVL; // true when MNNVL is available
  struct cliqueInfo clique; // Our MNNVL clique information
  int cliqueRank; // Our rank within the MNNVL clique
  int nvlDomainSize; // Number of ranks in the same NVL domain (same clusterUuid)

  bool checkPointers;

  // Counter to make sure collectives match (needed for bcast/reduce
  // where syncs are not symmetric).
  uint64_t opCount;
  // // Collective operation counter
  uint64_t collOpCount;

  // Channels for collectives
  int nChannels;
  int nGraphChannels;
  // Channels for collectives using fc algo
  int fcnChannels;
  // Channels (per peer) for p2p
  int p2pnChannels;
  int p2pnChannelsPerPeer;
  int p2pSchedGroupSize;
  int p2pChannels[MAXCHANNELS];
  // Channel shuffle
  bool channelShuffle;
  // Buffer sizes
  int buffSizes[NCCL_NUM_PROTOCOLS];

  // Algorithm/Protocols thresholds
  ssize_t threadThresholds[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float latencies[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float bandwidths[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float ringbdw[NCCL_NUM_FUNCTIONS][NCCL_NUM_PROTOCOLS];
  int maxThreads[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  // P2P Protocols
  int p2pProtoEnable[NCCL_NUM_PROTOCOLS];
  bool allocP2pNetLLBuffers;

  /* This attribute can indicate the states of communicators and return code of
   * asynchronous NCCL operations. */
  ncclResult_t asyncResult;

  // Flag to ask NCCL kernels to abort
  uint32_t *abortFlag;
  uint32_t *abortFlagDev;
  int* abortFlagRefCount;
  uint32_t* childAbortFlag;
  uint32_t* childAbortFlagDev;
  uint32_t destroyFlag;

  // Device side of the communicator (for hggcFree's)
  struct ncclKernelComm* devComm; // actually = &ncclKernelCommAndChannels::comm
  // Host copy of the devComm (to free HGGC allocs)
  struct ncclKernelCommAndChannels hostDevComm;

  uint32_t workArgsBytes; // max size of kernel args
  uint32_t workFifoBytes; // size of workFifoBuf, power of 2
  void* workFifoBuf;
  void* workFifoBufDev;
  void* workFifoBufGdrHandle;

  // Monotonic number of bytes (mod 1<<32) sent to fifo.
  uint32_t workFifoProduced;
  uint32_t workFifoProducedLastRecorded;
  uint32_t workFifoConsumed;

  // Intra-process sync
  struct pcclComm *intraComm0;
  struct pcclComm* intraNext; // next of intra-process comms, intraComm0 is head
  int intraRank;
  int intraRanks;
  uint32_t intraBarrierPhase;
  uint64_t intraBarrierCounter; // only used if this is intraComm0
  uint64_t intraBarrierGate; // only used if this is intraComm0

  struct ncclProxyState* proxyState;
  int proxyRefCountOld; /* store proxy post-atomic-sub refcount */

  // Whether this communicator uses collNet
  int collNetSupport;

  bool localP2pSupport;       // direct p2p capability, for local ranks only
  bool globalP2pSupport;      // direct p2p capability, for all ranks
  bool globalIcnP2pSupport;   // direct p2p through icnlink capalibity, for all ranks

  // Store info of async operations
  struct ncclInfo* asyncOps;
  int asyncOpCount;
  size_t asyncTotalSize;
  int lastChannel;

  int localPersistentRefs;

  hggcMemPool_t memPool;
  // Queue of events and associated callbacks for cleaning up asynchronous work.
  // Using this is preferable to using CUDA host callbacks because host callbacks
  // won't allow the work following the callback to run until the callback completes,
  // which comes at expense to perf.
  struct ncclIntruQueue<struct ncclCommEventCallback, &ncclCommEventCallback::next> eventCallbackQueue;

  // Queue of things for the main thread to do
  struct ncclIntruQueueMpsc<struct ncclCommCallback, &ncclCommCallback::next> callbackQueue;

  int algoEnable[NCCL_NUM_FUNCTIONS*NCCL_NUM_ALGORITHMS];
  ncclTuner_t* tuner;

  int hgtxMessageIndex;
  char* hgtxMessage;

  pcclExtKernel_t* extKernel;
  struct pcclExtKernelPluginWorkspace extKernelPluginWorkspace;
  int ceStreamsNum;
  hggcStream_t* ceStreams;
  hggcEvent_t* ceEvents;
  hggcEvent_t ceSyncEvent;
  int64_t ceReduceStep;
  struct pcclCeCollConfig ceCollConfigs;
  struct ncclCeColl ceColl;

  uint64_t devMemUsed;
  uint64_t pinnedMemUsed;
  uint64_t graphDevMemUsed;
  uint64_t graphPinnedMemUsed;

  // Profiler plugin
  void* profilerContext;
  uint64_t seqNumber[NCCL_NUM_FUNCTIONS];
  struct ncclProfilerProxy profiler;

#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
  bool okiSupport;
  int64_t ceOkiStep;

  ltOkiDevice_t okiDevice;
  ltOkiQP_t*    okiQps;          // Host-allocated QPs, accessible from both CPU and GPU
  int           okiQpCount;
  uint64_t*     okiDependencyAddr;  // PCCL-allocated dependency fence address
                                         // used to create dependency between normal kernel and okiQP
#endif

  // buffer registration cache
  struct ncclRegCache regCache;

  // list of func dispatch strategy.
  struct smResourceConfig_t smResConfig;
  struct ncclFuncDispatchStrategy *funcStrategy;
  struct ncclDispatchStrategy defaultStrategy;
  ncclFunc_t collType;
  size_t nBytes;

  // user-created reduction ops
  int userRedOpCapacity, userRedOpFreeHead;
  ncclUserRedOp *userRedOps;

  pcclConfig_t config;
  // Next comm in this thread's active ncclGroup[Start|End](). Holds "0x1" when
  // this comm is not yet in a group.
  struct pcclComm* groupNext[ncclGroupTaskTypeNum];
  // Subset of those in groupNext list. Holds 0x1 if not needing preconnect.
  struct pcclComm* preconnectNext;
  struct P2pSchedulePair {
    int sendRank;
    int recvRank;
  }* p2pSchedule;

  // pools backed by comm->memPermanent
  struct ncclMemoryPool memPool_ncclTaskColl;
  struct ncclMemoryPool memPool_ncclTaskP2p;
  struct ncclMemoryPool memPool_ncclProxyOp;
  struct ncclMemoryPool memPool_ncclKernelPlan;
  struct ncclMemoryPool memPool_kernelSymkArgs;
  struct ncclKernelPlanner planner;

  // initState is to more conveniently reclaim resources when errors happen.
  ncclResult_t initState;
  // flag to indicate if ncclCommFinalize() is called
  bool finalizeCalled;
  // shared structures for finalization
  int finalizeRankCnt;
  // group job to support multi-thread FT
  struct ncclGroupJob *groupJob;

  // Flag indicating if this communicator shares resources with parent or children
  bool shareResources;

  // Flag indicating this comm holds a reference on the net plugin,
  // taken by ncclNetRetain() in commAlloc and released by ncclNetFinalize() in commFree
  bool netPluginRetained;

  bool symmetricSupport;

  struct ncclDevrState devrState;
  struct ncclSymkState symkState;
  struct pcclP2pDmaState p2pDmaState;
  pcclGinConnectionType_t globalGinSupport; // negotiated across all ranks; NONE if no GIN hw

  void* ginContext;       // per-comm GIN plugin instance context
  int ginPluginIndex;     // index into ginPluginLibs[] in plugin/gin.cc

  KernelInfoMonitor* kernelInfoMonitorPtr = NULL;
  uint64_t kernelNum = 0;
  uint64_t channelKernelIdx[MAXCHANNELS]; // begin from 1
  uint64_t channelWorks[MAXCHANNELS];

  struct ncclMemManager* memManager; // tracks device memory for suspend/resume
  struct ncclIntruQueue<struct ncclMemManagerTask, &ncclMemManagerTask::next> suspendTaskQueue;
  struct ncclIntruQueue<struct ncclMemManagerTask, &ncclMemManagerTask::next> resumeTaskQueue;
};

enum ncclLaunchMode {
  ncclLaunchModeInvalid=0,
  ncclLaunchModeParallel,
  ncclLaunchModeGroup
};
extern enum ncclLaunchMode ncclParamLaunchMode;

typedef struct pcclComm ncclComm;

inline ncclResult_t ncclCommPollCallbacks(ncclComm_t comm, bool waitSome) {
  ncclResult_t result = ncclSuccess;
  struct ncclCommCallback* cb = ncclIntruQueueMpscDequeueAll(&comm->callbackQueue, waitSome);
  while (cb != nullptr) {
    struct ncclCommCallback* next = cb->next;
    ncclResult_t res1 = cb->fn(comm, cb); // may reclaim memory of cb
    if (res1 != ncclSuccess) result = res1;
    cb = next;
  }
  NCCLCHECK(result);
  return ncclSuccess;
}

inline ncclResult_t ncclCommPollEventCallbacks(ncclComm_t comm, bool waitSome) {
  ncclResult_t result = ncclSuccess;
  hggcStreamCaptureMode mode = hggcStreamCaptureModeRelaxed;
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
  while (true) {
    struct ncclCommEventCallback* cb = ncclIntruQueueHead(&comm->eventCallbackQueue);
    if (cb == nullptr) break;
    hggcError_t ok;
    if (waitSome) {
      ok = hggcEventSynchronize(cb->event);
      waitSome = false;
    } else {
      ok = hggcEventQuery(cb->event);
      if (ok == hggcErrorNotReady) break;
    }
    ncclIntruQueueDequeue(&comm->eventCallbackQueue);
    if (ok == hggcSuccess) {
      NCCLCHECKGOTO(cb->fn(comm, cb), result, finish);
    } else {
      HGGCCHECKGOTO(ok, result, finish);
    }
  }
finish:
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
  return ncclSuccess;
}

inline void ncclCommIntraBarrierIn(ncclComm_t comm, uint32_t x) {
  int phase = comm->intraBarrierPhase;
  if (comm->intraRanks == 1) {
    // Release everyone (just me).
    comm->intraBarrierGate = (uint64_t(x)<<32) | (phase^1);
  } else {
    ncclComm_t comm0 = comm->intraComm0;
    uint64_t count = __atomic_add_fetch(&comm0->intraBarrierCounter, (uint64_t(x)<<32) + 1, __ATOMIC_RELEASE);
    if (uint32_t(count) == uint32_t(comm->intraRanks)) {
      // Reset.
      __atomic_store_n(&comm0->intraBarrierCounter, 0, __ATOMIC_RELAXED);
      // Release everyone.
      __atomic_store_n(&comm0->intraBarrierGate, (count>>32<<32) | (phase^1), __ATOMIC_RELEASE);
    }
  }
}

// returns sum of x values contributed to ncclCommIntraBarrierIn(comm, x)
inline uint32_t ncclCommIntraBarrierOut(ncclComm_t comm) {
  ncclComm_t comm0 = comm->intraComm0;
  comm->intraBarrierPhase ^= 1;
  uint32_t phase = comm->intraBarrierPhase;
  uint64_t gate = __atomic_load_n(&comm0->intraBarrierGate, __ATOMIC_RELAXED);
  if ((gate & 1) != phase) {
    uint64_t t0 = clockNano();
    do {
      // Spin vigorously for first 5us.
      if (clockNano()-t0 >= 5*1000) sched_yield();
      gate = __atomic_load_n(&comm0->intraBarrierGate, __ATOMIC_RELAXED);
    } while ((gate & 1) != phase);
  }
  if (comm->intraRanks != 1) __atomic_thread_fence(__ATOMIC_ACQUIRE);
  return gate>>32;
}

// Scrambles the bits of non-builtin values of ncclRedOp_t according to the
// communicator memory address. Used to catch bugs so that integer handles
// associated with this communicator won't collide with handles of other
// communicatrs. This function is its own inverse.
static inline ncclRedOp_t ncclUserRedOpMangle(ncclComm *comm, ncclRedOp_t op) {
  // Preserve the built-in values.
  if(int(op) < int(ncclNumOps))
    return op;
  uint64_t h = reinterpret_cast<uint64_t>(comm);
  h ^= h >> 32;
  h *= 0x9e3779b97f4a7c13u; // Knuth's 64-bit magical hash constant
  h >>= 32; // h is now an excellent 32-bit hash of the comm pointer
  h &= int(ncclMaxRedOp); // ncclMaxRedOp is a power of 2 minus 1
  int op1 = int(h) ^ int(op);
  // Since builtin values are preserved, we also have to preserve their preimage.
  return op1 < int(ncclNumOps) ? op : ncclRedOp_t(op1);
}

inline ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState) {
  if (nextState < 0 || nextState >= ncclNumResults || comm == NULL) {
    ERROR("pcclCommSetAsyncError: error comm %p sets state %d", comm, nextState);
    return ncclInvalidArgument;
  }

  __atomic_store_n(&comm->asyncResult, nextState, __ATOMIC_RELEASE);
  return ncclSuccess;
}

bool pcclEnableGlobalIcnP2pThroughSwitchOpt(ncclComm_t comm);

ncclResult_t ncclCommEnsureReady(ncclComm_t comm);

#endif
