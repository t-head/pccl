/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/


#ifndef NCCL_CE_COLL_H_
#define NCCL_CE_COLL_H_

#include "pccl_internal.h"
// #include "nccl_common.h"
#include "bitops.h"
#include "symmetric.h"
#include "devcomm.h"
#include "sym_kernels.h"
#include <mutex>

struct pcclCeAllGatherStrategy {
  int peerNumStage1;
  int peerNumStage2;
  int icnConnectedPeers[NCCL_MAX_LOCAL_RANKS];
};

struct pcclCeCollConfig {
  // Strategy lifecycle management: Initialized lazily at runtime on first use
  // and destroyed when the communicator is finalized to ensure proper resource cleanup
  struct pcclCeAllGatherStrategy* agOneStageStrategy;
  struct pcclCeAllGatherStrategy* agTwoStageStrategy;
};

// CE-private symmetric storage, devr-backed.
//   ceSyncBase   = ceSyncWin->userPtr (cumem view; backs ncclSymDevBase_get*)
//   symDevComm   = passed to gpuBarrier kernel; .base is the lsa-flat
//                  self-rank slot so ncclSymPrims::peerPtr lands on peer slots.
struct ncclCeColl {
  bool initialized;
  struct ncclDevrWindow* ceSyncWin;
  uint8_t* ceSyncBase;
  size_t ceSyncSize;
  struct ncclSymDevComm symDevComm;
};

struct alignas(16) ncclCeCollArgs {
  struct ncclSymDevComm comm;
  ncclFunc_t func;
  int rootRank;
  size_t nElts;
  size_t eltSize;
  pcclDataType_t datatype;
  // Reduction op, only meaningful for the reduce family (AllReduce, ReduceScatter).
  pcclRedOp_t redOp;
  uint8_t* sendBuff;
  uint8_t* recvBuff;
  // Devr windows hosting sendBuff/recvBuff; resolved in enqueue.cc.
  struct ncclDevrWindow* sendWin;
  struct ncclDevrWindow* recvWin;
  int nBlocks;
  void* collApiEventHandle;  // Parent API event handle for profiler hierarchy
  void* ceCollProfHandle;     // CE collective profiler event handle
};

struct ncclCeBatchOpsParams {
  void** dsts;
  void** srcs;
  size_t* sizes;
  int32_t* targets;
  size_t numOps;
#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
  hggcMemcpyAttributes* attrs;
  size_t* attrIdxs;
  size_t numAttrs;
#endif
};

ncclResult_t ncclCeInitBatchOpsParams(struct ncclCeBatchOpsParams* params, int nRanks);
ncclResult_t ncclCeFreeBatchOpsParams(struct ncclCeBatchOpsParams* params);

static inline int ceArDtypeToIndex(pcclDataType_t dtype) {
  if (dtype == pcclHalf) return 0;
  else if (dtype == pcclBfloat16) return 1;
  else if (dtype == pcclFloat32) return 2;
  else return -1;
}

static inline int ceArRanksToIndex(const int nRanks) {
  switch(nRanks) {
    case 2:  return 0;
    case 4:  return 1;
    case 8:  return 2;
    default: return -1;
  }
}

// Ops the CE reduce kernels are instantiated for. avg is implemented by pre-scaling
// each source value by 1/nRanks inside the local reduce.
static inline int ceArRedOpToIndex(int/*ncclRedOp_t*/ red) {
  switch (red) {
    case ncclSum: return 0;
    case ncclAvg: return 1;
    default:      return -1;
  }
}

static int localReduceFuncIndex(pcclDataType_t dtype, int nRanks, int/*ncclRedOp_t*/ red) {
  constexpr int dtypeNum = 3;
  constexpr int nRanksNum = 3;
  int nRanksIdx = ceArRanksToIndex(nRanks);
  int dtypeIndex = ceArDtypeToIndex(dtype);
  int redIdx = ceArRedOpToIndex(red);
  if (nRanksIdx < 0 || dtypeIndex < 0 || redIdx < 0) return -1;
  int idx = redIdx * (nRanksNum * dtypeNum) + nRanksIdx * dtypeNum + dtypeIndex;
  return idx;
}

bool ncclCeOkiRequested();
bool ncclCeReduceEnabled(ncclFunc_t coll);
bool ncclCeImplemented(ncclFunc_t coll, int/*ncclRedOp_t*/ red, ncclDataType_t ty, bool okiSupport);
bool ncclCeAvailable(ncclComm_t comm, ncclFunc_t coll, int/*ncclRedOp_t*/ red, ncclDataType_t ty, size_t count, ncclSymRegType_t winRegType, const void* recvbuff, hggcStream_t stream);

ncclResult_t ncclCeInit(ncclComm_t comm);
ncclResult_t ncclCeFinalize(ncclComm_t comm);

ncclResult_t ncclMemOpSync(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream);
#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
ncclResult_t ncclMemOpSyncWithTrigger(struct ncclCeCollArgs* args, hggcStream_t stream, uint64_t* triggerPtr, uint64_t trigger);
#endif

ncclResult_t ncclLaunchCeColl(ncclComm_t comm, struct ncclKernelPlan* plan);

ncclResult_t ncclCeAllGather(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream);

ncclResult_t ncclCeScatter(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream);

ncclResult_t ncclCeGather(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream);

ncclResult_t ncclCeAlltoAll(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream);

ncclResult_t pcclInitCeAllGatherStrategy(struct pcclCeAllGatherStrategy* strategy, ncclComm_t comm, int twoStage);
#endif /* NCCL_CE_COLL_H_ */
