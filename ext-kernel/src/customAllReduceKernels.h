/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef CUSTOM_ALL_REDUCE_H_
#define CUSTOM_ALL_REDUCE_H_
#pragma once

#include "pccl/ext_kernel.h"
#include "pccl/ext_kernel_dispatch.h"
#include "customDeviceUtils.h"
#include <assert.h>
#include <hggc_bf16.h>
#include <hggc_fp16.h>
#include <sys/types.h>

#define MAX_ALL_REDUCE_BLOCKS 128
#define MAX_BLOCK_SIZE 1024

static inline int extKernelLLMaxSlots(int eltSize) {
  return EXT_KERNEL_LL_MAX_THREADS * EXT_KERNEL_LL_MAX_ELT_SIZE / eltSize;
}

static inline int extKernelLLEpochSize(int nRanks) {
  int a = EXT_KERNEL_LL_MAX_THREADS * nRanks * EXT_KERNEL_LL_PACK_SIZE;
  int b = extKernelLLMaxSlots(EXT_KERNEL_LL_MAX_ELT_SIZE) * EXT_KERNEL_LL_MAX_ELT_SIZE;
  return 2 * (a > b ? a : b);
}

// Per-region peer pointer table size: one void* per local rank slot. All
// regions that are peer-visible share the same table size, so this is a
// single compile-time constant rather than per-region state.
#define EXT_KERNEL_PTRS_SIZE (NCCL_MAX_LOCAL_RANKS * sizeof(void*))


static inline void* alignPtr(void* ptr, size_t alignSize) {
  if (((uintptr_t)ptr & (RMT_BULK_ALIGN_SIZE - 1)) != 0) {
    size_t offset = RMT_BULK_ALIGN_SIZE - (((uintptr_t)ptr) % alignSize);
    return (void*)((uintptr_t)ptr + offset);
  }
  return ptr;
}

// Compute total device memory needed for all regions (enabled data areas + peer-visible pointer tables).
static inline size_t computeDevMemTotalSize(const struct devRegionConfig config[], int nRegions) {
  size_t total = 0;
  for (int i = 0; i < nRegions; i++) {
    if (config[i].enabled)     total += config[i].elementSize;
    if (config[i].peerVisible) total += EXT_KERNEL_PTRS_SIZE;
  }
  return total + RMT_BULK_ALIGN_SIZE;
}

// Compute base addresses for each region's data area and (if peerVisible) pointer table.
// dataBases[region] = start of data area (undefined when !enabled)
// ptrTables[region] = start of pointer table (undefined when !peerVisible)
static inline void computeDevMemLayout(void* basePtr, const struct devRegionConfig config[],
                                       int nRegions,
                                       void* dataBases[], void* ptrTables[]) {
  uintptr_t addr = (uintptr_t)alignPtr(basePtr, RMT_BULK_ALIGN_SIZE);
  for (int i = 0; i < nRegions; i++) {
    dataBases[i] = (void*)addr;
    if (config[i].enabled) addr += config[i].elementSize;
  }
  for (int i = 0; i < nRegions; i++) {
    ptrTables[i] = (void*)addr;
    if (config[i].peerVisible) addr += EXT_KERNEL_PTRS_SIZE;
  }
}

struct twoShotDevParams{
  int32_t eltsTotal;
  uint32_t eltsPerBlock;
  uint32_t buffEltsPerBlock;
  uint32_t barrierFlag;
  uint8_t myRank;

  volatile uint32_t** peerBarrierIn;
  volatile uint32_t** peerBarrierPre;
  volatile uint32_t** peerBarrierReduce;
  volatile uint32_t** peerBarrierPost;
  volatile uint32_t** peerBarrierOut;

  void** peerBuffers;
  void** peerBuffersD;

  void* sendbuff;
  void* recvbuff;
  twoShotDevParams(struct extKernelDevParams& params) {
    eltsTotal = params.eltsTotal;
    eltsPerBlock = params.eltsPerBlock;
    buffEltsPerBlock = params.buffEltsPerBlock;
    myRank = params.myRank;
    peerBarrierIn = (volatile uint32_t**)params.peerPtrTables[DEV_REGION_BARRIER_IN];
    peerBarrierPre = (volatile uint32_t**)params.peerPtrTables[DEV_REGION_BARRIER_PRE];
    peerBarrierReduce = (volatile uint32_t**)params.peerPtrTables[DEV_REGION_BARRIER_REDUCE];
    peerBarrierPost = (volatile uint32_t**)params.peerPtrTables[DEV_REGION_BARRIER_POST];
    peerBarrierOut = (volatile uint32_t**)params.peerPtrTables[DEV_REGION_BARRIER_OUT];
    peerBuffers = (void**)params.peerPtrTables[DEV_REGION_BUFFER];
    peerBuffersD = (void**)params.peerPtrTables[DEV_REGION_BUFFER_D];
    sendbuff = params.sendbuff;
    recvbuff = params.recvbuff;
  }
};

struct twoShotRegDevParams{
  int32_t eltsTotal;
  uint32_t eltsPerBlock;
  uint32_t buffEltsPerBlock;
  uint32_t barrierFlag;
  uint8_t myRank;

  volatile uint32_t** peerBarrierIn;
  volatile uint32_t** peerBarrierPre;
  volatile uint32_t** peerBarrierOut;

  volatile void** peerRmtAddrs;
  void** peerBuffers;

  void* sendbuff;
  void* recvbuff;
  uintptr_t sendbuffOffset;
  uintptr_t recvbuffOffset;
  uintptr_t* sendbuffRmtAddrs;
  uintptr_t* recvbuffRmtAddrs;

  twoShotRegDevParams(struct extKernelDevParams& params) {
    eltsTotal = params.eltsTotal;
    eltsPerBlock = params.eltsPerBlock;
    buffEltsPerBlock = params.buffEltsPerBlock;
    myRank = params.myRank;
    peerBarrierIn = (volatile uint32_t**)params.peerPtrTables[DEV_REGION_BARRIER_IN];
    peerBarrierPre = (volatile uint32_t**)params.peerPtrTables[DEV_REGION_BARRIER_PRE];
    peerBarrierOut = (volatile uint32_t**)params.peerPtrTables[DEV_REGION_BARRIER_OUT];
    peerBuffers = (void**)params.peerPtrTables[DEV_REGION_BUFFER];
    sendbuff = params.sendbuff;
    recvbuff = params.recvbuff;
    sendbuffOffset = params.sendbuffOffset;
    recvbuffOffset = params.recvbuffOffset;
    sendbuffRmtAddrs = params.sendbuffRmtAddrs;
    recvbuffRmtAddrs = params.recvbuffRmtAddrs;

    peerRmtAddrs = (volatile void**)params.peerPtrTables[DEV_REGION_PEER_RMT_ADDR];
  }
};

struct oneShotLLDevParams {
  int32_t eltsTotal;
  uint32_t eltsPerBlock;
  uint8_t myRank;

  void** peerLLBufferPtrs;
  uint32_t* llEpochs;

  void* sendbuff;
  void* recvbuff;

  oneShotLLDevParams(struct extKernelDevParams& params) {
    eltsTotal = params.eltsTotal;
    eltsPerBlock = params.eltsPerBlock;
    myRank = params.myRank;
    peerLLBufferPtrs = (void**)params.peerPtrTables[DEV_REGION_LL_BUFFER];
    llEpochs = (uint32_t*)params.llEpochs;
    sendbuff = params.sendbuff;
    recvbuff = params.recvbuff;
  }
};

// Forward declarations of the AllReduce kernel templates. The definitions and
// explicit instantiations live in customAllReduceKernels.cu; these declarations
// let the unified customKernels[] array (customKernels.cu) reference them.
template <typename T, int NRANKS, int INPUT_ALIGN, int OUTPUT_ALIGN>
__global__ void pcclKernel_oneShotAllReduceKernel(struct oneShotDevParams params);
template <typename T, int NRANKS, int INPUT_ALIGN, int OUTPUT_ALIGN>
__global__ void pcclKernel_oneShotLLAllReduceKernel(struct oneShotLLDevParams params);
template <typename T, int NRANKS, int DOUBLE_BUFFER, int INPUT_ALIGN, int OUTPUT_ALIGN>
__global__ void pcclKernel_twoShotAllReduceKernel(struct twoShotDevParams params);
template <typename T, int NRANKS, int INPUT_ALIGN, int OUTPUT_ALIGN>
__global__ void pcclKernel_twoShotRegAllReduceKernel(struct twoShotRegDevParams params);

template <typename T>
pcclResult_t invokeOneOrTwoShotAllReduceKernel(struct extKernelDevParams& devParams, extKernelStrategyType strat, hggcStream_t stream);

pcclResult_t customAllReduceImpl(struct extKernelDevParams& devParams, struct extKernelHostParams& hostParams, hggcStream_t stream);

#define ONE_SHOT_FUNCS_NAME(input_align, output_align, nRanks, dtype) \
  (void*)pcclKernel_oneShotAllReduceKernel<dtype, nRanks, input_align, output_align>,

#define ONE_SHOT_FUNCS_ALIGN(input_align, output_align, nRanks, dtype) \
  ONE_SHOT_FUNCS_NAME(input_align, output_align, nRanks, dtype)

#define ONE_SHOT_FUNCS_RANK(nRanks, dtype) \
  ONE_SHOT_FUNCS_ALIGN(0, 0, nRanks, dtype) \
  ONE_SHOT_FUNCS_ALIGN(0, 1, nRanks, dtype) \
  ONE_SHOT_FUNCS_ALIGN(1, 0, nRanks, dtype) \
  ONE_SHOT_FUNCS_ALIGN(1, 1, nRanks, dtype)

#define ONE_SHOT_FUNCS_DTYPE(dtype) \
  ONE_SHOT_FUNCS_RANK(2, dtype) \
  ONE_SHOT_FUNCS_RANK(4, dtype) \
  ONE_SHOT_FUNCS_RANK(8, dtype) \
  ONE_SHOT_FUNCS_RANK(16, dtype)

// Generate all oneShot kernels
#define ONE_SHOT_FUNCS() \
  ONE_SHOT_FUNCS_DTYPE(half) \
  ONE_SHOT_FUNCS_DTYPE(__ppu_bfloat16)  \
  ONE_SHOT_FUNCS_DTYPE(float)


#define TWO_SHOT_FUNCS_NAME(db ,input_align, output_align, nRanks, dtype) \
  (void*)pcclKernel_twoShotAllReduceKernel<dtype, nRanks, db, input_align, output_align>,

#define TWO_SHOT_FUNCS_DB(db ,input_align, output_align, nRanks, dtype) \
  TWO_SHOT_FUNCS_NAME(db ,input_align, output_align, nRanks, dtype)

#define TWO_SHOT_FUNCS_ALIGN(input_align, output_align, nRanks, dtype) \
  TWO_SHOT_FUNCS_DB(0, input_align, output_align, nRanks, dtype) \
  TWO_SHOT_FUNCS_DB(1, input_align, output_align, nRanks, dtype)

#define TWO_SHOT_FUNCS_RANK(nRanks, dtype) \
  TWO_SHOT_FUNCS_ALIGN(0, 0, nRanks, dtype) \
  TWO_SHOT_FUNCS_ALIGN(0, 1, nRanks, dtype) \
  TWO_SHOT_FUNCS_ALIGN(1, 0, nRanks, dtype) \
  TWO_SHOT_FUNCS_ALIGN(1, 1, nRanks, dtype)

#define TWO_SHOT_FUNCS_DTYPE(dtype) \
  TWO_SHOT_FUNCS_RANK(2,  dtype) \
  TWO_SHOT_FUNCS_RANK(4,  dtype) \
  TWO_SHOT_FUNCS_RANK(8,  dtype) \
  TWO_SHOT_FUNCS_RANK(16, dtype)

// Generate all oneShot kernels
#define TWO_SHOT_FUNCS() \
  TWO_SHOT_FUNCS_DTYPE(half) \
  TWO_SHOT_FUNCS_DTYPE(__ppu_bfloat16)  \
  TWO_SHOT_FUNCS_DTYPE(float)

#define TWO_SHOT_REG_FUNCS_NAME(input_align, output_align, nRanks, dtype) \
  (void*)pcclKernel_twoShotRegAllReduceKernel<dtype, nRanks, input_align, output_align>,

#define TWO_SHOT_REG_FUNCS_ALIGN(input_align, output_align, nRanks, dtype) \
  TWO_SHOT_REG_FUNCS_NAME(input_align, output_align, nRanks, dtype)

#define TWO_SHOT_REG_FUNCS_RANK(nRanks, dtype) \
  TWO_SHOT_REG_FUNCS_ALIGN(0, 0, nRanks, dtype) \
  TWO_SHOT_REG_FUNCS_ALIGN(0, 1, nRanks, dtype) \
  TWO_SHOT_REG_FUNCS_ALIGN(1, 0, nRanks, dtype) \
  TWO_SHOT_REG_FUNCS_ALIGN(1, 1, nRanks, dtype)

#define TWO_SHOT_REG_FUNCS_DTYPE(dtype) \
  TWO_SHOT_REG_FUNCS_RANK(2,  dtype) \
  TWO_SHOT_REG_FUNCS_RANK(4,  dtype) \
  TWO_SHOT_REG_FUNCS_RANK(8,  dtype) \
  TWO_SHOT_REG_FUNCS_RANK(16, dtype)

// Generate all twoShotReg kernels
#define TWO_SHOT_REG_FUNCS() \
  TWO_SHOT_REG_FUNCS_DTYPE(half) \
  TWO_SHOT_REG_FUNCS_DTYPE(__ppu_bfloat16)  \
  TWO_SHOT_REG_FUNCS_DTYPE(float)

#define ONESHOT_LL_FUNCS_NAME(input_align, output_align, nRanks, dtype) \
  (void*)pcclKernel_oneShotLLAllReduceKernel<dtype, nRanks, input_align, output_align>,

#define ONESHOT_LL_FUNCS_ALIGN(input_align, output_align, nRanks, dtype) \
  ONESHOT_LL_FUNCS_NAME(input_align, output_align, nRanks, dtype)

#define ONESHOT_LL_FUNCS_RANK(nRanks, dtype) \
  ONESHOT_LL_FUNCS_ALIGN(0, 0, nRanks, dtype) \
  ONESHOT_LL_FUNCS_ALIGN(0, 1, nRanks, dtype) \
  ONESHOT_LL_FUNCS_ALIGN(1, 0, nRanks, dtype) \
  ONESHOT_LL_FUNCS_ALIGN(1, 1, nRanks, dtype)

#define ONESHOT_LL_FUNCS_DTYPE(dtype) \
  ONESHOT_LL_FUNCS_RANK(2, dtype) \
  ONESHOT_LL_FUNCS_RANK(4, dtype) \
  ONESHOT_LL_FUNCS_RANK(8, dtype) \
  ONESHOT_LL_FUNCS_RANK(16, dtype)

// Generate all oneShotLL kernels
#define ONESHOT_LL_FUNCS() \
  ONESHOT_LL_FUNCS_DTYPE(half) \
  ONESHOT_LL_FUNCS_DTYPE(__ppu_bfloat16) \
  ONESHOT_LL_FUNCS_DTYPE(float)


////////////////////////////////////////////////////////////////////////////////////////////////////
// AR-only index helper (dtypeToIndex is AR-specific)
////////////////////////////////////////////////////////////////////////////////////////////////////
static inline int dtypeToIndex(pcclDataType_t dtype) {
  if (dtype == pcclHalf) return 0;
  else if (dtype == pcclBfloat16) return 1;
  else return 2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Template parameter space constants — aligned with ONESHOT_*/TWO_SHOT_* macro expansion
// Note: CUSTOM_NRANKS_NUM and CUSTOM_ALIGN_NUM are defined in customDeviceUtils.h
////////////////////////////////////////////////////////////////////////////////////////////////////
#define CUSTOM_AR_DTYPES_NUM 3
#define CUSTOM_AR_DB_NUM 2

// Kernel variant counts used across collectives
#define ONESHOT_FUNCS_NUM (CUSTOM_AR_DTYPES_NUM * CUSTOM_NRANKS_NUM * CUSTOM_ALIGN_NUM)
#define ONESHOT_LL_FUNCS_NUM (CUSTOM_AR_DTYPES_NUM * CUSTOM_NRANKS_NUM * CUSTOM_ALIGN_NUM)
#define TWOSHOT_FUNCS_NUM (CUSTOM_AR_DTYPES_NUM * CUSTOM_NRANKS_NUM * CUSTOM_ALIGN_NUM * CUSTOM_AR_DB_NUM)
#define TWOSHOT_REG_FUNCS_NUM (CUSTOM_AR_DTYPES_NUM * CUSTOM_NRANKS_NUM * CUSTOM_ALIGN_NUM)
#define AR_EXT_FUNCS_NUM (ONESHOT_FUNCS_NUM + TWOSHOT_FUNCS_NUM + TWOSHOT_REG_FUNCS_NUM + ONESHOT_LL_FUNCS_NUM)

// Index calculation functions
static int getOneShotKernelIndex(pcclDataType_t dataType, int nRanks, int inputAlign, int outputAlign) {
  const int alignNum = CUSTOM_ALIGN_NUM;
  const int rankNum = CUSTOM_NRANKS_NUM * alignNum;

  int nRanksIdx = ranksToIndex(nRanks);
  int dtypeIndex = dtypeToIndex(dataType);
  int idx = dtypeIndex * rankNum +
            nRanksIdx * alignNum +
            (inputAlign * 2 + outputAlign);
  return idx;
}

static int getTwoShotKernelIndex(pcclDataType_t dataType, int nRanks, int doubleBuffer, int inputAlign, int outputAlign) {
  const int dbNum = CUSTOM_AR_DB_NUM;
  const int alignNum = CUSTOM_ALIGN_NUM * dbNum;
  const int rankNum = CUSTOM_NRANKS_NUM * alignNum;

  int nRanksIdx = ranksToIndex(nRanks);
  int dtypeIndex = dtypeToIndex(dataType) ;
  int idx = dtypeIndex * rankNum +
            nRanksIdx * alignNum +
            (inputAlign * 2 + outputAlign) * dbNum +
            doubleBuffer;
  return idx;
}

static int getTwoShotRegKernelIndex(pcclDataType_t dataType, int nRanks, int inputAlign, int outputAlign) {
  const int alignNum = CUSTOM_ALIGN_NUM;
  const int rankNum = CUSTOM_NRANKS_NUM * alignNum;

  int nRanksIdx = ranksToIndex(nRanks);
  int dtypeIndex = dtypeToIndex(dataType) ;
  int idx = dtypeIndex * rankNum +
            nRanksIdx * alignNum +
            (inputAlign * 2 + outputAlign);
  return idx;
}

static inline int getOneShotLLKernelIndex(pcclDataType_t dataType, int nRanks, int inputAlign, int outputAlign) {
  return getOneShotKernelIndex(dataType, nRanks, inputAlign, outputAlign);
}

static int extKernelGetFuncIndex(extKernelStrategyType strategy, pcclDataType_t dataType, int nRanks, int inputAlign, int outputAlign, int doubleBuffer) {
  if (strategy == ONESHOT || strategy == ONESHOT_REG) {
    return getOneShotKernelIndex(dataType, nRanks, inputAlign, outputAlign);
  }
  if (strategy == TWOSHOT) {
    return (ONESHOT_FUNCS_NUM + getTwoShotKernelIndex(dataType, nRanks, doubleBuffer, inputAlign, outputAlign));
  }
  if (strategy == TWOSHOT_REG) {
    return (ONESHOT_FUNCS_NUM +
            TWOSHOT_FUNCS_NUM +
            getTwoShotRegKernelIndex(dataType, nRanks, inputAlign, outputAlign));
  }
  if (strategy == ONESHOT_LL) {
    return (ONESHOT_FUNCS_NUM +
            TWOSHOT_FUNCS_NUM +
            TWOSHOT_REG_FUNCS_NUM +
            getOneShotLLKernelIndex(dataType, nRanks, inputAlign, outputAlign));
  }
  return -1;
}

#endif  // CUSTOM_ALL_REDUCE_H_
