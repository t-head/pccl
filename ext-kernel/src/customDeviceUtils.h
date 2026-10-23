/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef CUSTOM_DEVICE_UTILS_HOST_H_
#define CUSTOM_DEVICE_UTILS_HOST_H_
#pragma once

#include "pccl/ext_kernel.h"
#include "pccl/ext_kernel_dispatch.h"
#include <sys/types.h>

// ============================================================================
// Device memory layout — region-based model
// ============================================================================
//
// All cross-rank kernel state is described as a flat list of "regions". Each
// region is independently configurable along two axes:
//
//   * enabled       — allocate a per-rank data area (elementSize bytes)
//   * peerVisible   — allocate a peer pointer table (EXT_KERNEL_PTRS_SIZE bytes)
//                     that holds the base address of this region on every peer.
//
// Most regions are { enabled, peerVisible } (data + peer table). LL_EPOCHS is
// { enabled, !peerVisible } (local-only, no peer indexing). BUFFER_D when
// double buffering is off is { !enabled, peerVisible } (no extra data; the
// pointer table is aliased to BUFFER so kernel indexing stays uniform).
//
// The full per-rank device buffer looks like:
//
//   +------------------+  <- basePtr (aligned to RMT_BULK_ALIGN_SIZE)
//   | data areas       |   one per enabled region, packed in enum order
//   |   BUFFER         |
//   |   BUFFER_D       |   (0 bytes if doubleBuffer disabled)
//   |   BARRIER_*      |
//   |   PEER_RMT_ADDR  |
//   |   LL_BUFFER      |
//   |   LL_EPOCHS      |   (no peer table — local-only)
//   +------------------+
//   | pointer tables   |   one per peerVisible region (LL_EPOCHS skipped)
//   |   BUFFER     ptrs|   peerPtrTables[BUFFER][peer] -> peer's BUFFER base
//   |   ...            |
//   |   LL_BUFFER  ptrs|
//   +------------------+
//
// FOR_EACH_DEV_REGION(X) is the single source of truth: adding/removing an
// entry here auto-updates the DevRegion enum, the regionConfig table size,
// and the per-rank pointer-table set. Plugin code then only needs to:
//   * fill regionConfig[NEW_REGION] in configHostParams() (enabled + peerVisible + size)
//   * for peer-visible regions: read peers via devParams.peerPtrTables[NEW_REGION]
//   * for local-only regions: read the local base from dataBases[NEW_REGION]
//     during pluginConfigWorkspace and stash it in a typed devParams field
//     (see llEpochs).
//
// Notes on quirks:
//   * BUFFER_D with enabled==false still has a pointer table; pluginConfigWorkspace
//     aliases its dataBase to BUFFER's so kernels can use the same indexing
//     code regardless of whether double buffering is on.
//   * Regions are packed back-to-back without per-region alignment padding.
//     LL_BUFFER needs 16-byte alignment for uint4 loads; this is currently
//     satisfied incidentally by the barrier/peerRmtAddr sizing for all
//     supported nRanks ∈ {2,4,8,16}. Adding a new region (or changing one of
//     these sizes) must preserve that property — add an assert if unsure.
// ============================================================================
#define FOR_EACH_DEV_REGION(X)       \
    X(BUFFER)                         \
    X(BUFFER_D)                       \
    X(BARRIER_IN)                     \
    X(BARRIER_PRE)                    \
    X(BARRIER_REDUCE)                 \
    X(BARRIER_POST)                   \
    X(BARRIER_OUT)                    \
    X(PEER_RMT_ADDR)                  \
    X(LL_BUFFER)                      \
    X(LL_EPOCHS)

#define GEN_DEV_REGION_ENUM(id) DEV_REGION_##id,
typedef enum {
    FOR_EACH_DEV_REGION(GEN_DEV_REGION_ENUM)
    DEV_REGION_COUNT
} devRegion;

struct devRegionConfig {
    bool enabled;
    bool peerVisible;
    size_t elementSize;
};

typedef enum {
  ONESHOT = 0,
  TWOSHOT = 1,
  ONESHOT_REG = 2,
  TWOSHOT_REG = 3,
  ONESHOT_LL = 4,
  STRATEGY_NUM = 5
} extKernelStrategyType;

struct extKernelDevParams {
  int32_t eltsTotal;
  uint32_t eltsPerBlock;
  uint32_t buffEltsPerBlock;
  uint8_t myRank;
  uint32_t* abortFlag;

  void* peerPtrTables[DEV_REGION_COUNT];
  void* llEpochs;

  void* sendbuff;
  void* recvbuff;

  uintptr_t sendbuffOffset;
  uintptr_t recvbuffOffset;
  uintptr_t* sendbuffRmtAddrs;
  uintptr_t* recvbuffRmtAddrs;
};

struct extKernelHostParams {
  uint32_t nRanks;
  int32_t boundLow;
  int32_t boundHigh;
  int32_t agBoundHigh;           // AG-specific upper bound (from topoConfigs column 6)
  int32_t rsBoundHigh;           // RS-specific upper bound (from topoConfigs)
  int32_t twoShotSwitchPoint;
  uint8_t doubleBuffer;
  int compCap;

  size_t bufSize;
  size_t barrierSize;
  size_t peerRmtAddrSize;
  struct devRegionConfig regionConfig[DEV_REGION_COUNT];

  int32_t oneShotLLBoundHigh;
  size_t llBufferSize;

  pcclDataType_t dtype;
  uint8_t dtypeSize;
  extKernelStrategyType strategy;
  int funcIndex;

  int maxBlockNum;
  int maxThreadNum;
  bool useUsrConfigBlockNum;
  bool useUsrConfigThreadNum;

  struct ncclFuncDispatchStrategy *funcStrategy;
  struct ncclDispatchStrategy defaultStrategy;
  ncclFunc_t currentFunc;
};

struct extKernelParams {
  struct extKernelDevParams* devParams;
  struct extKernelHostParams hostParams;

  // for dump convenient
  uint32_t* myBarrierIn;
  uint32_t* myBarrierPre;
  uint32_t* myBarrierReduce;
  uint32_t* myBarrierPost;
  uint32_t* myBarrierOut;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Shared constants and index helpers — used by both AllReduce and AllGather
////////////////////////////////////////////////////////////////////////////////////////////////////
#define WARP_SIZE 32
#define PACKED_ELEMS_SIZE 16
#define RMT_BULK_ALIGN_SIZE 512

// LL protocol constants (shared between AR and AG device code)
#define EXT_KERNEL_LL_MAX_BLOCKS 8
#define EXT_KERNEL_LL_MAX_THREADS 1024
#define EXT_KERNEL_LL_PACK_SIZE 8       // 8 bytes data per LL slot (uint4 = 16 bytes: {data_lo, epoch, data_hi, epoch})
#define EXT_KERNEL_LL_MAX_ELT_SIZE 64
#define LL_EFFICIENCY (1/2)

// Template parameter space dimensions shared across collectives
#define CUSTOM_NRANKS_NUM 5
#define CUSTOM_ALIGN_NUM 4

static inline int ranksToIndex(const int nRanks) {
  switch(nRanks) {
    case 2:  return 0;
    case 4:  return 1;
    case 8:  return 2;
    case 16: return 3;
    case 32: return 4;
    default: return -1;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// oneShotDevParams — shared by oneShot AllReduce and oneShot AllGather kernels
////////////////////////////////////////////////////////////////////////////////////////////////////
struct oneShotDevParams {
  int32_t eltsTotal;
  uint32_t eltsPerBlock;
  uint32_t buffEltsPerBlock;
  uint32_t barrierFlag;
  uint8_t myRank;
  uint32_t* abortFlag;

  volatile uint32_t** peerBarrierIn;
  volatile uint32_t** peerBarrierOut;
  volatile uint32_t** peerBarrierReduce;
  void** peerBuffers;

  void* sendbuff;
  void* recvbuff;

  oneShotDevParams(struct extKernelDevParams& params) {
    eltsTotal = params.eltsTotal;
    eltsPerBlock = params.eltsPerBlock;
    buffEltsPerBlock = params.buffEltsPerBlock;
    myRank = params.myRank;
    abortFlag = params.abortFlag;
    peerBarrierIn = (volatile uint32_t**)params.peerPtrTables[DEV_REGION_BARRIER_IN];
    peerBarrierOut = (volatile uint32_t**)params.peerPtrTables[DEV_REGION_BARRIER_OUT];
    peerBarrierReduce = (volatile uint32_t**)params.peerPtrTables[DEV_REGION_BARRIER_REDUCE];
    peerBuffers = (void**)params.peerPtrTables[DEV_REGION_BUFFER];
    sendbuff = params.sendbuff;
    recvbuff = params.recvbuff;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Host-side dispatch functions (defined in customDeviceUtils.cu)
////////////////////////////////////////////////////////////////////////////////////////////////////
ncclResult_t extKernelSetDispatchStrategy(const void* func, const char* funcName, extKernelHostParams& hostParams, int blocksPerGrid, ncclFunc_t collFunc);
ncclResult_t kernelLaunchConfig(extKernelDevParams& devParams, extKernelHostParams& hostParams, size_t eltsPerThread, int* blocksPerGrid, int* threadsPerBlock);

#endif  // CUSTOM_DEVICE_UTILS_HOST_H_
