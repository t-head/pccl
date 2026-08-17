/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_SYMMETRIC_H_
#define NCCL_DEVICE_SYMMETRIC_H_

#include "pccl_internal.h"
#include "pccl_common.h"
#include "bitops.h"
#include "utils.h"
#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
#include "lt_oki.h"
#endif

// ce AR only support max two dies.
#define PCCL_CE_MAX_DIE_NUM 2

constexpr int ncclSymMaxBlocks = 64;
constexpr int ncclSymMaxThreads = 512;
constexpr int ncclSymMinBlocksPerCU = 2;
constexpr int ncclSymLLMaxEltSize = 64;

// every transfer size should be 2_MB at least to ensure transfer efficiency
// for BTV1.5 2Die design, chunk size should be 4_MB
// use pingpong buffer to pipeline transfer and compute, midBuffer size should be 8_MB per peer rank.
constexpr int pcclCeReduceChunkSize = 4_MB;
constexpr int pcclCeReduceMaxSect = 2;
constexpr int pcclSymMidBufferPerRank = (pcclCeReduceChunkSize * pcclCeReduceMaxSect);

// The per-rank DMA pipelines drift apart on long transfers, which costs bandwidth. Every this
// many reduce chunks the OKI AR cmd kernel inserts one full sync: that chunk waits on every
// peer's tail instead of only the target rank's. Pipelines no longer than one interval
// (4MB chunk on 8 ranks, i.e. up to 128MB total) keep the original single-tail wait.
constexpr int pcclCeOkiWaitAllTailsInterval = 4;

constexpr int pcclCeReduceMaxBlocks = 4;
constexpr int pcclCeReduceMaxThreads = 1024;
constexpr int bulkBytesPerThread = 16;

constexpr __host__ __device__ int ncclSymLLMaxSlots(int eltSize = ncclSymLLMaxEltSize) {
  return ncclSymMaxThreads*ncclSymLLMaxEltSize/eltSize;
}

constexpr __host__ __device__ int ncclSymLLEpochSize(int nRanks) {
  return /*LL Overhead*/2 * maxval(ncclSymMaxThreads*nRanks*8, ncclSymLLMaxSlots(ncclSymLLMaxEltSize)*ncclSymLLMaxEltSize);
}

struct ncclCeGridSyncMem {
  uint32_t gridSyncCounter;
  uint32_t gridSyncGeneration;
};

constexpr size_t pcclCeReduceStaBufSize(int nRanks) {
  return pcclSymMidBufferPerRank * nRanks;
}

constexpr size_t pcclCeReduceSyncBufSize(int nRanks) {
  return alignUp(nRanks * PCCL_CE_MAX_DIE_NUM * sizeof(uint64_t), 16);
}

constexpr size_t pcclCeReduceGridSyncBufSize() {
  return alignUp(sizeof(ncclCeGridSyncMem), 16);
}

struct alignas(16) ncclSymDevBase {
  uint32_t llEpoch[ncclSymMaxBlocks];
  uint32_t barEpochUc[ncclSymMaxBlocks];
  uint32_t barInboxPerPeer[];

  // symk-only prefix (header + barInboxPerPeer + LL epoch buffer).
  static constexpr size_t size_symk(int nRanks) {
    return sizeof(ncclSymDevBase) +
           alignUp(ncclSymMaxBlocks*nRanks*sizeof(uint32_t), 16) +
           ncclSymMaxBlocks * /*epochs=*/2 * ncclSymLLEpochSize(nRanks);
  }
  // Full layout including CE-private tail (sta/tail/head/gridSync), indexed
  // by the get* accessors below.
  static constexpr size_t size(int nRanks) {
    return size_symk(nRanks) +
           pcclCeReduceStaBufSize(nRanks) +
           pcclCeReduceSyncBufSize(nRanks) + // tail mem
           pcclCeReduceSyncBufSize(nRanks) + // head mem
           pcclCeReduceGridSyncBufSize() +
           alignUp(sizeof(uint64_t) * nRanks * PCCL_CE_MAX_DIE_NUM, 16); // oki remote fence
  }
};

static void* ncclSymDevBase_getStaBuf(struct ncclSymDevBase* base, int nRanks) {
  // Get pointer to buffer trailing the header struct.
  char* ans = (char*)(base + 1);
  ans += (alignUp(ncclSymMaxBlocks*nRanks*sizeof(uint32_t), 16) +
         ncclSymMaxBlocks * /*epochs=*/2 * ncclSymLLEpochSize(nRanks));
  return (void*)ans;
}

static void* ncclSymDevBase_getCeTailBase(struct ncclSymDevBase* base, int nRanks) {
  char* ans = (char*)(base + 1);
  ans += (alignUp(ncclSymMaxBlocks*nRanks*sizeof(uint32_t), 16) +
         ncclSymMaxBlocks * /*epochs=*/2 * ncclSymLLEpochSize(nRanks) +
         pcclCeReduceStaBufSize(nRanks));
  return (void*)ans;
}

static void* ncclSymDevBase_getCeHeadBase(struct ncclSymDevBase* base, int nRanks) {
  char* ans = (char*)(base + 1);
  ans += (alignUp(ncclSymMaxBlocks*nRanks*sizeof(uint32_t), 16) +
         ncclSymMaxBlocks * /*epochs=*/2 * ncclSymLLEpochSize(nRanks) +
         pcclCeReduceStaBufSize(nRanks) +
         pcclCeReduceSyncBufSize(nRanks));
  return (void*)ans;
}

static void* ncclSymDevBase_getGridSyncBase(struct ncclSymDevBase* base, int nRanks) {
  char* ans = (char*)(base + 1);
  ans += (alignUp(ncclSymMaxBlocks*nRanks*sizeof(uint32_t), 16) +
         ncclSymMaxBlocks * /*epochs=*/2 * ncclSymLLEpochSize(nRanks) +
         pcclCeReduceStaBufSize(nRanks) +
         pcclCeReduceSyncBufSize(nRanks) +  // tail mem
         pcclCeReduceSyncBufSize(nRanks));  // head mem
  return (void*)ans;
}

static void* ncclSymDevBase_getKuiRemoteFenceAddr(struct ncclSymDevBase* base, int nRanks) {
  char* ans = (char*)(base + 1);
  ans += (alignUp(ncclSymMaxBlocks*nRanks*sizeof(uint32_t), 16) +
         ncclSymMaxBlocks * /*epochs=*/2 * ncclSymLLEpochSize(nRanks) +
         pcclCeReduceStaBufSize(nRanks) +
         pcclCeReduceSyncBufSize(nRanks) +  // tail mem
         pcclCeReduceSyncBufSize(nRanks)) +  // head mem
         pcclCeReduceGridSyncBufSize(); // head mem
  return (void*)ans;
}

static __device__ uint4* ncclSymDevBase_getLLBuf(struct ncclSymDevBase* base, int nRanks, int block, uint32_t epoch) {
  // Get pointer to buffer trailing the header struct.
  char* ans = (char*)(base + 1);
  // Skip over barInboxPerPeer[]
  ans += alignUp(ncclSymMaxBlocks*nRanks*sizeof(uint32_t), 16);
  // Skip to our block
  int epochSize = ncclSymLLEpochSize(nRanks);
  ans += block * /*epochs=*/2 * epochSize;
  ans += (epoch & 1)*epochSize;
  return (uint4*)ans;
}

struct ncclSymDevComm {
  ncclSymDevBase* base;
  uint32_t stride4G;
  int nRanks, rank;
  uint32_t nRanks_rcp32; // idivRcp32(nRanks)
};

struct alignas(16) pcclSymDevArgs {
  struct ncclSymDevComm comm;
  int rootRank;
  int nBlocks;
  uint64_t redOpArg; // must be collectively uniform
  size_t nElts;
  char* input;
  char* output;
  bool incPrecision;
};
#define ncclSymDevArgs pcclSymDevArgs

// Symmetric kernel selection and dispatch live in sym_kernels.h (the multi-task
// symk path); this header exposes only the device-side types still used by the
// CE path and by the legacy ncclSymRun_* inline kernel bodies that the
// generated __global__ wrappers translate into.

__global__ void gpuBarrier(ncclSymDevArgs const args);
#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
__global__ void gpuBarrierWithTrigger(ncclSymDevArgs const args, uint64_t* addr, uint64_t value);
#endif
__global__ void gpuGEQ(volatile uint64_t* ptr, uint64_t data);
__global__ void gpuPreDma(uint64_t* tailPtr, uint64_t flag, int32_t nDies);
__global__ void gpuPostDma(uint64_t* headPtr, uint64_t flag);

template<typename T, int NRANKS>
__global__ void ceLocalReduce(T* src, T* dst, ssize_t eltsTotal, uint32_t myRank, uint32_t blockNum, uint32_t nDies, uint64_t* tailBase, uint64_t* headBase, uint32_t iters, uint32_t stride4G, void* gridSyncBase, int64_t baseStep
#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
  , ltOkiQP_t* qps, uint64_t okiStep
#endif
);

#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
template<int DIE_NUM>
__global__ void okiMemcpyBatchAsyncForAR(
  ltOkiQP_t* qps,
  uint64_t* dependencyFenceAddr,
  void* sendBuff,
  uint8_t* myRecvBuff,
  uint8_t* rank0RecvBuff,
  uint8_t* rank0StaBuff,
  int myRank, int nRanks,
  uint64_t nElts,
  int eltSize,
  int loopCnt,
  uint64_t* rank0RemoteFenceAddrs,
  uint64_t okiBaseStep,
  uint64_t* tailPtr,
  uint64_t* headPtr,
  uint64_t ceReduceStepBase,
  int stride4G
);
#endif

#define PCCL_OKI_MAX_LOCAL_RANKS 64
struct pcclCeCollKernelLaunchArgs {
  void* dsts[PCCL_OKI_MAX_LOCAL_RANKS];
  void* srcs[PCCL_OKI_MAX_LOCAL_RANKS];
  size_t sizes[PCCL_OKI_MAX_LOCAL_RANKS];
  int targets[PCCL_OKI_MAX_LOCAL_RANKS];
  int nOps;
  uint64_t* rank0RemoteFenceAddrs;
  uint64_t waitFenceValue;
  uint64_t writeFenceValue;
};
#define ncclCeCollKernelLaunchArgs pcclCeCollKernelLaunchArgs

#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
template<int DIE_NUM>
__global__ void okiMemcpyBatchAsync(
  struct ncclCeCollKernelLaunchArgs args, // use struct to prevent compiler opt array to pointer.
  ltOkiQP_t* ibs,
  uint64_t* dependencyFenceAddr,
  uint64_t step,
  int myRank,
  int stride4G,
  int numQpGroups
);

template<int DIE_NUM>
__global__ void okiBatchSynchronize(
  struct ncclCeCollKernelLaunchArgs args, // use struct to prevent compiler opt array to pointer.
  ltOkiQP_t* qps,
  int myRank,
  uint64_t pollValue,
  int numQpGroups
);
#endif

#endif
