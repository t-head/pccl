/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_H_
#define NCCL_DEVICE_H_

#include "pccl_internal.h"
#include "bitops.h"
#include "pccl_common.h"
#include <stdint.h>
#include <sys/types.h>

#define NCCL_MAX_NET_SIZE (1024*1024*1024L) // Rather than send INT_MAX which is 2G-1, send a power of two.

#define NCCL_MAX_LOCAL_RANKS 64

extern const char* ncclFuncStr[NCCL_NUM_FUNCTIONS+1];
extern const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS];
extern const char* ncclProtoStr[NCCL_NUM_PROTOCOLS];
extern const bool ncclSendRecvProtoSupport[NCCL_NUM_PROTOCOLS];

#define NCCL_MAX_OPS 2048
#define NCCL_STEPS 8

typedef enum { CONN_NORMAL, CONN_ICN, CONN_PCIE, CONN_SHM, CONN_GDRDMA, CONN_RDMA, CONN_SOCKET, CONN_NET_EXT, CONN_NUM } ncclConn_t;

union ncclLLFifoLine {
  /* Flags have to be *after* data, because otherwise, an incomplete receive
     from the network may receive the flag but not the data.
     Note this is assuming that either we receive contiguous chunks of data
     (sockets) or data is written with an atomicity of 8 bytes (IB/RDMA). */
  struct {
    uint32_t data1;
    uint32_t flag1;
    uint32_t data2;
    uint32_t flag2;
  };
  uint64_t v[2];
  uint4 i4;
};

#define WARP_SIZE 32
#define MAXCHANNELS 64
#define MAXCHANNELS_PPU0015 24
#define NCCL_MAX_NTHREADS 640
#define NCCL_MIN_NTHREADS (4*WARP_SIZE)
#define NCCL_SIMPLE_MAX_NTHREADS 512
#define NCCL_LL_MAX_NTHREADS 512
#define NCCL_LL_LINES_PER_THREAD 8
#ifdef TEST_LL_CLEANUP
#define NCCL_LL_CLEAN_MASK 0x078 // Set to 0x100 to disable cleanup
#define NCCL_LL_FLAG_MAX   0x100
#define NCCL_LL_FLAG(a) ((uint32_t)((a) % NCCL_LL_FLAG_MAX))
#else
#define NCCL_LL_CLEAN_MASK 0x7ffffff8
#define NCCL_LL_FLAG(a) ((uint32_t)(a))
#endif
// Make sure the clean mask will last for at least NCCL_NSTEPS
static_assert(NCCL_LL_CLEAN_MASK % NCCL_STEPS == 0, "Invalid NCCL_LL_CLEAN_MASK value");

#define NCCL_LL128_LINESIZE 128
#define NCCL_LL128_LINEELEMS (NCCL_LL128_LINESIZE/sizeof(uint64_t))
#define NCCL_LL128_DATAELEMS (NCCL_LL128_LINEELEMS-1)

#define NCCL_LL128_MAX_NTHREADS 640
#define NCCL_LL128_ELEMS_PER_THREAD 120

// Receiving from up to 3 sources is more compute intensive than sending
// to 3 dests. Use 70% for reduce and 30% for bcast.
#define NCCL_LL128_SPLIT(nt) ((nt*7/(10*32))*32)

#define NCCL_LL128_SHMEM_ELEMS_PER_THREAD 8
#define NCCL_LL128_SHMEM_SIZE (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS)
#define NCCL_PAT_SHMEM_SIZE 8192

#define NCCL_DIRECT_GPU 0x01
#define NCCL_DIRECT_NIC 0x10

// Number of named barriers supported by CUDA
#define NCCL_MAX_GROUPS 16

enum ncclDevRedOp_t {
  ncclDevSum, ncclDevProd, ncclDevMax, ncclDevMin,
  ncclDevPreMulSum, ncclDevSumPostDiv,
  ncclNumDevRedOps
};
struct ncclDevRedOpFull {
  ncclDevRedOp_t op;
  bool scalarArgIsPtr;
  uint64_t scalarArg;
};

// Single-rank reduction launcher (defined in collectives/device/onerank.cu).
ncclResult_t ncclLaunchOneRank(void* dst, void const* src, size_t nElts, struct ncclDevRedOpFull redOp, ncclDataType_t eltType, hggcStream_t stream);

struct ncclConnInfo {
  // Regular comm mechanism
  char *buffs[NCCL_NUM_PROTOCOLS]; // Local for recv, remote for send
  uint64_t *tail;     // Local for recv, remote for send
  uint64_t *head;     // Local for send, remote for recv

  int direct;         // Direct communication
  int shared;         // Buffers are shared
  uint32_t minPathNum;     // minimum physical link path number between 2 devices
  uint32_t* signalMarker;  // signal marker memory used for signal fence remote atomic add
  void **ptrExchange; // Pointer exchange for direct communication

  int *sizesFifo;     // Sizes fifo from GPU to proxy
  int *offsFifo;      // Offset fifo from proxy to GPU

  uint64_t step;      // Keep where we are
  uint64_t llLastCleaning;
};

struct ncclProxyConnector {
  bool initialized;
  int rank;
  int tpRank;
  int tpLocalRank;
  int sameProcess;
  struct ncclProxyConnection* connection;
};

struct ncclConnector {
  int connected;
  struct ncclProxyConnector proxyConn;
  struct ncclTransportComm* transportComm;
  void* transportResources; // Host-side resources
  struct ncclConnInfo conn;
};

struct ncclRing {
  // Shortcuts for userRanks[1] and userRanks[n-1]
  int prev;
  int next;

  // Maps an internal nccl index to user-specified rank order. This is necessary
  // since we need to know how the user expects data to be ordered across
  // devices. Ordered from current device.
  int* userRanks;
  int* devUserRanks;
  int nRanks;
  ncclConn_t sendConnType;
  ncclConn_t recvConnType;
};

#define NCCL_MAX_TREE_ARITY 3
struct ncclTree {
  int depth;
  int up;
  int down[NCCL_MAX_TREE_ARITY];
  ncclConn_t upConnType;
  ncclConn_t downConnType;
};

#define NCCL_MAX_FC_ARITY 4
struct ncclFC {
  int rank;
  int nPeers;
  int peerRanks[NCCL_MAX_FC_ARITY];
  ncclConn_t connType;
};

#define NCCL_MAX_CONNS 2
#define MAX_PROFILER_EVENTS_PER_CHANNEL 64
struct ncclDevProfiler {
    struct {
        uint64_t counter;
        uint64_t timestamp;
    } data[MAX_PROFILER_EVENTS_PER_CHANNEL];
};

struct ncclChannelPeer {
  struct ncclConnector send[NCCL_MAX_CONNS];
  struct ncclConnector recv[NCCL_MAX_CONNS];
  int refCount;
};

struct ncclDevChannelPeer {
  // Stripped version of ncclChannelPeer where we only keep the ncclConnInfo
  // instead of the full ncclConnector.
  struct ncclConnInfo send[NCCL_MAX_CONNS];
  struct ncclConnInfo recv[NCCL_MAX_CONNS];
};

struct alignas(16) ncclDevChannel {
  struct ncclDevChannelPeer** peers;
  struct ncclRing ring;
  struct ncclTree tree;
  struct ncclTree collTree;
  struct ncclFC fc;
  uint64_t perKernelBatchCounter;
  uint64_t workCounter;
};

struct ncclKernelComm;

struct alignas(16) ncclDevWorkP2p {
  // Header
  struct ncclKernelComm* comm;
  uint8_t sendRegUsed:1, recvRegUsed:1, sendProtoLL:1, recvProtoLL:1, profilerEnabled:1;

  const void * sendbuff;
  void * recvbuff;

  ssize_t sendCount, recvCount;
  int sendRank, recvRank;
  int sendChunkSize, recvChunkSize;
};

struct alignas(16) ncclDevWorkColl {
  // Running on channels [channelLo..channelHi], hi is inclusive.
  //   nChannels == (channelHi - channelLo) + 1
  uint32_t channelLo:8, channelHi:8;
  uint32_t nWarps:8;
  uint32_t redOpArgIsPtr:1, sendRegUsed:1, recvRegUsed:1;
  uint32_t profilerEnabled:1;
  uint32_t root;
  void* recvbuff;
  const void* sendbuff;
  uintptr_t sendbuffOffset;
  uintptr_t recvbuffOffset;
  uintptr_t* sendbuffRmtAddrs;
  uintptr_t* recvbuffRmtAddrs;
  union {
    // Continuous-byte-distribution scheduling. The lo and hi channels are of
    // different size than the channels in the middle.
    struct {
      size_t countLo, countMid, countHi;
      // Chunk counts where units are ncclProtoGrainSize(protocol) bytes
      uint64_t chunkGrainsLo:21, chunkGrainsMid:21, chunkGrainsHi:21;
    } cbd;
    // Existing non-Ring algorithms retain their count and chunk semantics.
    struct {
      size_t count; // Total size, not divided per channel.
      size_t chunkCount;
    } collnet;
  };
  uint64_t redOpArg;
#ifdef NO_TSM
  // Without thread shared memory, device primitives locate comm via work.
  struct ncclKernelComm* comm;
#endif
};
static_assert(MAXCHANNELS <= 256, "Collective channel interval must fit in eight bits");
static_assert(NCCL_MAX_NTHREADS / WARP_SIZE < 256, "Collective nWarps must fit in eight bits");
#ifndef NO_TSM
static_assert(sizeof(ncclDevWorkColl) == 96, "Collective work must remain 96 bytes");
#else
static_assert(sizeof(ncclDevWorkColl) == 112, "NO_TSM collective work includes comm");
#endif

__host__ __device__ constexpr int ncclProtoGrainSize(int proto) {
  return proto == NCCL_PROTO_LL ? 16 :
      proto == NCCL_PROTO_LL128 ? WARP_SIZE * NCCL_LL128_SHMEM_ELEMS_PER_THREAD /
          NCCL_LL128_LINEELEMS * NCCL_LL128_DATAELEMS * sizeof(uint64_t) : 512;
}

template<typename Int>
__host__ __device__ inline void ncclCollCbdPart(
    const struct ncclDevWorkColl* work, uint32_t channelId, int proto, int eltSize,
    Int* count, Int* partOffset, Int* partCount, Int* chunkCount) {
  const int eltPerGrain = ncclProtoGrainSize(proto) / eltSize;
  const int nMidChannels = int(work->channelHi) - int(work->channelLo) - 1;
  if (count != nullptr) {
    *count = work->cbd.countLo + work->cbd.countMid * nMidChannels + work->cbd.countHi;
  }
  if (channelId == work->channelLo) {
    *partOffset = 0;
    *partCount = work->cbd.countLo;
    *chunkCount = size_t(work->cbd.chunkGrainsLo) * eltPerGrain;
  } else if (channelId == work->channelHi) {
    *partOffset = work->cbd.countLo + nMidChannels * work->cbd.countMid;
    *partCount = work->cbd.countHi;
    *chunkCount = size_t(work->cbd.chunkGrainsHi) * eltPerGrain;
  } else {
    *partOffset = work->cbd.countLo + (channelId - work->channelLo - 1) * work->cbd.countMid;
    *partCount = work->cbd.countMid;
    *chunkCount = size_t(work->cbd.chunkGrainsMid) * eltPerGrain;
  }
}

// Preserve PPU's measured AllReduce tail policy without extending the work ABI.
// Proto and eltSize are compile-time constants in each device specialization.
__host__ __device__ inline size_t pcclAllReduceTailGrain(
    const struct ncclDevWorkColl* work, int proto, int eltSize) {
  int nthreads = work->nWarps * WARP_SIZE;
  if (proto == NCCL_PROTO_LL) return nthreads * sizeof(uint64_t) / eltSize;
  if (proto == NCCL_PROTO_LL128) {
    return size_t(nthreads) * NCCL_LL128_SHMEM_ELEMS_PER_THREAD *
        NCCL_LL128_DATAELEMS * sizeof(uint64_t) / NCCL_LL128_LINEELEMS / 2 / eltSize;
  }
#ifndef NO_EXTRA_WARP
  nthreads -= WARP_SIZE;
#endif
  return (work->channelLo == work->channelHi ? nthreads * sizeof(uint64_t) : 16) / eltSize;
}

// unused now
struct alignas(16) ncclDevWorkCollReg {
  struct ncclDevWorkColl coll;
};

enum ncclDevWorkType: uint8_t {
  ncclDevWorkTypeP2p,
  ncclDevWorkTypeColl,
  ncclDevWorkTypeCollReg
};

constexpr size_t ncclDevWorkSize(enum ncclDevWorkType type) {
  return type == ncclDevWorkTypeP2p ? sizeof(ncclDevWorkP2p) :
         type == ncclDevWorkTypeColl ? sizeof(ncclDevWorkColl) : sizeof(ncclDevWorkCollReg);
}

#define NCCL_MAX_DEV_WORK_BATCH_BYTES 1024
#define NCCL_MAX_DEV_WORK_BATCH_COLLS (NCCL_MAX_DEV_WORK_BATCH_BYTES/sizeof(ncclDevWorkColl))
#define NCCL_MAX_DEV_WORK_P2P_PER_BATCH 8
#define NCCL_MAX_BIT_IDX_IN_OFFSET_BITSET 63
struct alignas(16) ncclDevWorkBatch {
  union {
    struct {
      // nextExtends: should next one be merged into this one.
      // nextJump=0: end of this channel's batch list
      // nextJump>0: batches[thisIndex+nextJump] is next batch in this list
      uint32_t nextJump:14, nextExtends:1;
      uint32_t workType:2, funcId:15;
    };
    // Unioning bitfields with underlying type hints compiler to emit the best
    // SASS LD/ST accesses.
    uint32_t flags;
  };
  // Rolling offset in fifo where this batch's work structs begin
  uint32_t offsetBase;
  // Set of relative offsets from offsetBase for this channel's subset of the batch:
  // For each bit index i in offsetMask, find work at fifo offset: offsetBase + i*sizeof(WorkStructType)
  uint64_t offsetBitset;
};

#define NCCL_MAX_SLICE_PER_CHUNK 2

#if __HGGC_ARCH__ >= 800
#define NCCL_MAX_DEV_ARITY (NCCL_MAX_TREE_ARITY-1)  // Using balanced tree instead of split tree
#else
#define NCCL_MAX_DEV_ARITY NCCL_MAX_TREE_ARITY
#endif

__device__ __forceinline__ constexpr int ncclCollUnroll() {
#if defined(__HGGC_ARCH__)
#if __HGGC_ARCH__ >= 150
  return 4;
#else
  return 2;
#endif
#else
  return 8;
#endif
}

__device__ __forceinline__ constexpr int ncclSendRecvLocalUnroll() {
#if defined(__HGGC_ARCH__)
  return 2;
#else
  return 8;
#endif
}

#define NCCL_REG_ELEM_FACTOR 2

struct ncclShmemGroup {
  void* srcs[NCCL_MAX_FC_ARITY];
  void* dsts[NCCL_MAX_FC_ARITY];
  int realSizes[NCCL_MAX_FC_ARITY];
  int totalSendSize[NCCL_MAX_SLICE_PER_CHUNK];
};

struct alignas(16) ncclKernelComm {
  int rank;
  int nRanks;
  int buffSizes[NCCL_NUM_PROTOCOLS];
  int* rankToLocalRank;

  // Flag to ask NCCL kernels to abort
  volatile uint32_t *abortFlag;
  uint8_t stateMonitorLevel;

  // Channels, device side
  struct ncclChannel* channels;

  // Profiler counters
  struct ncclDevProfiler* workStarted/*[MAXCHANNELS]*/;
  struct ncclDevProfiler* workCompleted/*[MAXCHANNELS]*/;

  uint8_t profilerGlobalEnabled;
};

struct alignas(16) ncclKernelCommAndChannels {
  struct ncclKernelComm comm;
  struct ncclDevChannel channels[MAXCHANNELS];
};

enum ncclDevWorkStorageType: uint8_t {
  ncclDevWorkStorageTypeArgs=0,
  ncclDevWorkStorageTypeFifo=1,
  ncclDevWorkStorageTypePersistent=2
};

struct alignas(16) pcclDevKernelArgs {
  struct ncclKernelComm* comm;
  uint64_t channelMask;
  uint64_t kernelIdx; // begin from 1
  uint32_t workMask;
  enum ncclDevWorkStorageType workStorageType;
  bool persistent;
  void* workBuf;
  // A channel's first batch is at `blosckIdx.x`. Use `nextJump` to follow rest of list.
  // struct ncclDevWorkBatch batches[];
};
#define ncclDevKernelArgs pcclDevKernelArgs

__host__ __device__ constexpr int ncclMaxKernelArgsSize() {
  return 1<<10;
}

template<size_t capacity>
struct alignas(16) pcclDevKernelArgsStorage {
  union {
    struct ncclDevKernelArgs args;
    ulong2 storage[capacity/sizeof(ulong2)];
  };
};
#define ncclDevKernelArgsStorage pcclDevKernelArgsStorage

typedef ncclDevKernelArgsStorage<(1<<10)> ncclDevKernelArgs1K;

template<typename T>
__host__ __device__ constexpr T min_constexpr(T a) { return a; }
template<typename T, typename ...Ts>
__host__ __device__ constexpr T min_constexpr(T a, T b, Ts ...c) {
  return min_constexpr<T>((a < b ? a : b), c...);
}

template<typename T>
__host__ __device__ constexpr T max_constexpr(T a) { return a; }
template<typename T, typename ...Ts>
__host__ __device__ constexpr T max_constexpr(T a, T b, Ts ...c) {
  return max_constexpr<T>((a > b ? a : b), c...);
}

constexpr int ncclDevMaxChannelsForArgsBytes(size_t argsBytes) {
  return min_constexpr<size_t>(MAXCHANNELS, (argsBytes - sizeof(struct ncclDevKernelArgs))/sizeof(struct ncclDevWorkBatch));
}

// The amount of dynamic shmem per warp
__host__ __device__ constexpr int ncclShmemScratchWarpSize() {
#ifdef NO_TSM
  return 0;
#else
  // PAT stores ncclPatShmem across the whole dynamic per-warp scratch region
  // (accessed via ncclScratchForWarp(0)), so the total must be large enough to
  // hold NCCL_PAT_SHMEM_SIZE. Otherwise only existing small users need 64B/warp.
  return max_constexpr<int>(
           64,
           ((( NCCL_PAT_SHMEM_SIZE + (NCCL_MAX_NTHREADS/WARP_SIZE) - 1 ) / (NCCL_MAX_NTHREADS/WARP_SIZE)) + 15) & -16);
#endif
}

// The amount of dynamic shmem per block
__host__ __device__ constexpr int ncclShmemDynamicSize() {
  return ncclShmemScratchWarpSize()*(NCCL_MAX_NTHREADS/WARP_SIZE);
}

struct ncclShmemData {
  struct ncclDevKernelArgs args;
  int channelId;
  int aborted;
  alignas(16) struct ncclKernelComm comm;
  alignas(16) struct ncclDevChannel channel;

  int batchIx, nextBatchIx;
  enum ncclDevWorkType workType;
  uint8_t directMode;
  uint16_t funcId;
  int nWorks;
  int workSize;
  union {
    volatile uint64_t data[NCCL_LL128_SHMEM_SIZE];
    struct ncclShmemGroup groups[NCCL_MAX_GROUPS];
  };

  alignas(16) char workStorage[1024];
};

struct ncclChannel {
  union {
    struct {
      struct ncclRing ring;
      struct ncclTree tree;
      struct ncclTree collTree;
      struct ncclFC fc;

      int id;

      // Communication structures
      struct ncclChannelPeer** peers;
      struct ncclDevChannelPeer** devPeers;
      /* devPeer pointer array used for host side access */
      struct ncclDevChannelPeer** devPeersHostPtr;

#ifdef NO_TSM
      struct ncclShmemData* shmem;
      // For the PAT algorithm
      void* patShmem;
#endif

      uint64_t workCounter;
      uint32_t nSpinsBeforeReportHang; // Only used by GPU
      uint64_t *hostDevIndexLevel1;   // Only write by GPU and read by CPU
      uint64_t *hostDevIndexLevel2;   // Only write by GPU and read by CPU
      uint64_t *devIndexLevel1;   // Only write by GPU and read by CPU
      uint64_t *devPerKernelBatchCounter;   // Only write by GPU and read by CPU
    };
    int data[0x80];
  };
};
static_assert(sizeof(struct ncclChannel) == 0x80*sizeof(int), "ncclChannel must have a pow2 size");

#endif
