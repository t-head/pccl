/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
// sym_kernels.h: Symmetric kernel state, device work structures, and scheduling API.

#ifndef NCCL_SYM_KERNELS_H_
#define NCCL_SYM_KERNELS_H_

#include "pccl_internal.h"
#include "nccl_device/core.h"
#include "../device/symmetric/gin_scratch.h"
#include "dev_runtime.h"
#include "bitops.h"

////////////////////////////////////////////////////////////////////////////////
// ncclSymk[Foo]: Kernels built on the device API

#define NCCL_SYM_KERNEL_CELL_SIZE 1024 // no less than 16 bytes minimal cell size

constexpr int ncclSymkMaxBlocks = 64;
constexpr int ncclSymkMaxThreads = 512;
constexpr int ncclSymkLLMaxEltSize = 8;

constexpr __host__ __device__ int ncclSymkLLMaxSlots(int eltSize = ncclSymkLLMaxEltSize) {
  return ncclSymkMaxThreads * ncclSymkLLMaxEltSize / eltSize;
}

enum ncclSymkKernelId {
  ncclSymkKernelId_AllReduce_AGxLL_R,
  ncclSymkKernelId_AllReduce_RSxLD_AGxST,

  ncclSymkKernelId_AllGather_LL,
  ncclSymkKernelId_AllGather_ST,
  ncclSymkKernelId_AllGather_ST_Ring,

  ncclSymkKernelId_ReduceScatter_LL,
  ncclSymkKernelId_ReduceScatter_LD,

  ncclSymkKernelId_Count
};

struct ncclSymDevBase;

// Extended DevComm for symmetric kernels.
//   base = lsa-flat self-rank slot of symk->symkWin; ncclSymPrims's
//          peerPtr(peer, base) = add4G(base, (peer-rank)*stride4G).
struct ncclSymkDevComm {
  struct ncclDevComm devComm;
  struct ncclSymDevBase* base;
  struct ncclLLA2AHandle lsaLLA2A;
  struct ncclGinOutboxHandle ginOutbox;
  ncclGinCounter_t ginCounterPerBlock;
  struct ncclGinInboxA2AHandle ginInboxRail;
  struct ncclGinSyncHandle ginSyncHandle;
};

struct ncclSymkState {
  bool initialized;
  bool initInProgress;
  int maxGinInboxBlocks;
  struct ncclSymkDevComm kcomm;
  struct ncclDevrWindow* symkWin;
};

struct pcclSymkChannelWorkRange {
  uint16_t workHi;
  uint16_t fracHi;
};

struct alignas(16) pcclSymkDevWork {
  uint64_t redOpArg;
  size_t nElts;
  ncclDevWin_t inputWin, outputWin;
  size_t inputOff, outputOff;
  int rootRank;
  uint64_t sChannelId:16, nChannels:16, padding:32;
};

struct alignas(16) pcclSymkDevWorkArgs {
  struct ncclSymkDevComm kcomm;
  int nMaxChannels;

  __host__ static constexpr size_t calcArgsSize(int nChannels, int nWorks) {
    return alignUp(sizeof(struct pcclSymkDevWorkArgs), 16)
         + alignUp(nChannels * sizeof(struct pcclSymkChannelWorkRange), 16)
         + nWorks * sizeof(struct pcclSymkDevWork);
  }
  __host__ __device__ struct pcclSymkChannelWorkRange* getWorkRange() const {
    return (struct pcclSymkChannelWorkRange*)((uint8_t*)this + alignUp(sizeof(struct pcclSymkDevWorkArgs), 16));
  }
  __host__ __device__ struct pcclSymkDevWork* getWorks(int nChannels) const {
    return (struct pcclSymkDevWork*)((uint8_t*)this->getWorkRange() + alignUp(nChannels * sizeof(struct pcclSymkChannelWorkRange), 16));
  }
};

// 4KB by-value: hgLaunchKernelExAD memcpys the full kernel-signature size.
union pcclSymkDevWorkArgs4K {
  struct pcclSymkDevWorkArgs args;
  char buf4K[4096];
};

// Registration type for symmetric windows
typedef enum {
  ncclSymSendNonregRecvNonreg = 0,
  ncclSymSendNonregRecvReg = 1,
  ncclSymSendRegRecvNonreg = 2,
  ncclSymSendRegRecvReg = 3,
  ncclNumSymRegTypes = 4
} ncclSymRegType_t;

////////////////////////////////////////////////////////////////////////////////
// Host API

// Initialize symmetric kernel state (lazy, on first symk-eligible task)
ncclResult_t ncclSymkInitOnce(struct pcclComm* comm);
ncclResult_t ncclSymkFinalize(struct pcclComm* comm);

// Check if a collective operation can use the symmetric kernel path
bool ncclSymkAvailable(struct pcclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red,
                       ncclDataType_t ty, size_t nElts);

// Select the best kernel for a set of symmetric operations
ncclResult_t ncclSymkPickKernel(struct pcclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                                size_t nEltsTotal, size_t nEltsMax, int nWorks, ncclSymRegType_t winRegType,
                                float* estTimeUs, ncclSymkKernelId* kernelId, int* nBlocks, int* nWarps, bool* forced);

// Fill a device work descriptor from a task
struct ncclTaskColl;
ncclResult_t ncclSymkMakeDevWork(struct pcclComm* comm, struct ncclTaskColl* task, struct pcclSymkDevWork* outDevWork);

// ncclSymkKernelList[]: multi-task wrappers (4KB pcclSymkDevWorkArgs4K).
// ncclSymkKernelList_fast[]: single-task wrappers (64B ncclSymDevArgs).
// Scheduler selects via PCCL_PARAM(SymkUseMultiTask); default fast-path.
extern int const ncclSymkKernelCount;
extern void* ncclSymkKernelList[];
extern void* ncclSymkKernelList_fast[];
extern int ncclSymkKernelRequirements[];

// Kernel index lookup
int ncclSymkGetKernelIndex(ncclSymkKernelId kernelId, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty);
const char* ncclSymkKernelIdToString(int kernelId);

// Determine registration type from send/recv windows
ncclResult_t ncclGetSymRegType(struct ncclDevrWindow* sendWin, struct ncclDevrWindow* recvWin, ncclSymRegType_t* winRegType);

int ncclSymkLLKernelMask();

#endif // NCCL_SYM_KERNELS_H_
