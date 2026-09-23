/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "enqueue.h"
#include "argcheck.h"
#include "coll_net.h"
#include "gdrwrap.h"
#include "channel.h"
#include "pccl_internal.h"
#include "transport.h"
#include "register_inline.h"
#include "hgtx.h"
#include "scheduler.h"
#include "dev_runtime.h"
#include "sym_kernels.h"
#include "ce_coll.h"
#include "comm_dumper.h"

// Only generate inline kernels for LL
#define NCCL_FUNC5(func, algo, devredop, dtype) \
  (void*)NCCL_KERN_NAME(func, algo, LL, devredop, dtype), \
  (void*)NCCL_KERN_NAME(func, algo, LL, devredop, dtype), \
  (void*)NCCL_KERN_NAME(func, algo, LL, devredop, dtype), \
  (void*)NCCL_KERN_NAME(func, algo, LL, devredop, dtype), \
  (void*)NCCL_KERN_NAME(func, algo, LL, devredop, dtype), \
  (void*)NCCL_KERN_NAME(func, algo, LL, devredop, dtype), \
  (void*)NCCL_KERN_NAME(func, algo, LL, devredop, dtype)

#define NCCL_FUNC4(func, devredop, type) \
  (void*)NCCL_FUNC5(func, TREE,    devredop, type), \
  (void*)NCCL_FUNC5(func, RING,    devredop, type), \
  (void*)NCCL_FUNC5(func, COLLNET, devredop, type), \
  (void*)NCCL_FUNC5(func, FC,      devredop, type), \
  (void*)NCCL_FUNC5(func, PAT,     devredop, type)

// Must be consistent with ncclDataType_t
#define NCCL_FUNCS3A(func, devredop) \
  (void*)NCCL_FUNC4(func, devredop, int8_t), \
  (void*)NCCL_FUNC4(func, devredop, uint8_t), \
  (void*)NCCL_FUNC4(func, devredop, int32_t), \
  (void*)NCCL_FUNC4(func, devredop, uint32_t), \
  (void*)NCCL_FUNC4(func, devredop, int64_t), \
  (void*)NCCL_FUNC4(func, devredop, uint64_t), \
  (void*)NCCL_FUNC4(func, devredop, half), \
  (void*)NCCL_FUNC4(func, devredop, float), \
  (void*)NCCL_FUNC4(func, devredop, double), \
  (void*)NCCL_FUNC4(func, devredop, __ppu_bfloat16), \
  (void*)NCCL_FUNC4(func, devredop, __hg_fp8_e4m3), \
  (void*)NCCL_FUNC4(func, devredop, __hg_fp8_e5m2)
#define NCCL_FUNCS3B(func, devredop) \
  (void*)NCCL_FUNC4(func, devredop, int8_t), \
  (void*)NCCL_FUNC4(func, devredop, int8_t), \
  (void*)NCCL_FUNC4(func, devredop, int8_t), \
  (void*)NCCL_FUNC4(func, devredop, int8_t), \
  (void*)NCCL_FUNC4(func, devredop, int8_t), \
  (void*)NCCL_FUNC4(func, devredop, int8_t), \
  (void*)NCCL_FUNC4(func, devredop, int8_t), \
  (void*)NCCL_FUNC4(func, devredop, int8_t), \
  (void*)NCCL_FUNC4(func, devredop, int8_t), \
  (void*)NCCL_FUNC4(func, devredop, int8_t), \
  (void*)NCCL_FUNC4(func, devredop, int8_t), \
  (void*)NCCL_FUNC4(func, devredop, int8_t)

// Must be consistent with ncclRedOp_t -- but we only generate kernel for sums.
#define NCCL_FUNCS2A(func) \
  NCCL_FUNCS3A(func, Sum), /*Sum*/  \
  NCCL_FUNCS3A(func, Sum), /*Prod*/ \
  NCCL_FUNCS3A(func, Sum), /*Max*/  \
  NCCL_FUNCS3A(func, Sum), /*Min*/  \
  NCCL_FUNCS3A(func, Sum), /*PreMulSum*/ \
  NCCL_FUNCS3A(func, Sum)  /*SumPostDiv*/
#define NCCL_FUNCS2B(func) \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum)

// Must be consistent with the ncclFuncSet enum
static void* const ncclKerns[NUM_P2P_FUNC+NCCL_NUM_FUNCTIONS*ncclNumDevRedOps*ncclNumTypes*NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS] = {
  (void*)NCCL_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t),
  (void*)NCCL_KERN_NAME(SendRecv, RING, SIMPLE_BULK, Sum, int8_t),
  (void*)NCCL_KERN_NAME(SendRecv, RING, SIGNAL, Sum, int8_t),
  (void*)NCCL_KERN_NAME(SendRecv, RING, SIGNAL_BULK, Sum, int8_t),
  (void*)NCCL_KERN_NAME(SendRecv, RING, SIMPLE_BULK_AR, Sum, int8_t),
  NCCL_FUNCS2B(Broadcast),
  NCCL_FUNCS2A(Reduce),
  NCCL_FUNCS2B(AllGather),
  NCCL_FUNCS2A(ReduceScatter),
  NCCL_FUNCS2A(AllReduce)
};

enum { P2P_Segment=1, Coll_Segment=2 };

static ncclResult_t ncclScanKernelMaxStackSize(void* const* kernels, int count, void* lru[2], size_t* maxStackSize) {
  ncclResult_t result = ncclSuccess;
  for (int i = 0; i < count; i++) {
    void* fn = kernels[i];
    if (fn == nullptr || fn == lru[0] || fn == lru[1]) continue;
    lru[1] = lru[0];
    lru[0] = fn;
    if (maxStackSize) {
      hggcFuncAttributes attr = {0};
      HGGCCHECKGOTO(hggcFuncGetAttributes(&attr, fn), result, ignore);
      if (attr.localSizeBytes > *maxStackSize) *maxStackSize = attr.localSizeBytes;
    ignore:;
    }
  }
  return result;
}

// Returns maximum kernel stack size of all HGGC kernels
ncclResult_t ncclInitKernelsForDevice(size_t* maxStackSize) {
  constexpr int KernelCount = sizeof(ncclKerns)/sizeof(ncclKerns[0]);
  if (maxStackSize) *maxStackSize = 0;
  // Keep track if we already visited a function pointer.
  void* lru[2] = {nullptr, nullptr};
  NCCLCHECK(ncclScanKernelMaxStackSize(ncclKerns, KernelCount, lru, maxStackSize));
  NCCLCHECK(ncclScanKernelMaxStackSize(ncclSymkKernelList, ncclSymkKernelCount, lru, maxStackSize));
  return ncclSuccess;
}

static inline void ncclTaskCollToExtTaskColl(const struct ncclTaskColl* src, struct ncclExtTaskColl* dst) {
  dst->func = src->func;
  dst->op = src->opDev.op;
  dst->datatype = src->datatype;
  dst->count = src->count;
  dst->sendbuff = src->sendbuff;
  dst->recvbuff = src->recvbuff;
  dst->regUsed = src->regUsed;
  dst->sendbuffOffset = src->sendbuffOffset;
  dst->recvbuffOffset = src->recvbuffOffset;
  dst->sendbuffRmtAddrs = src->sendbuffRmtAddrs;
  dst->recvbuffRmtAddrs = src->recvbuffRmtAddrs;
}

/*****************************************************************************/
/*       Launch system : synchronization and HGGC kernel launch              */
/*****************************************************************************/

static ncclResult_t ncclSetDispatchStrategy(const void* ncclKernFunc, ncclComm_t comm, struct dispatchStrategyExAD &launchConfig, int funcIndex, int nChannels, bool isSymmetric) {
  int savedDev;
  HGGCCHECK(hggcGetDevice(&savedDev));
  HGGCCHECK(hggcSetDevice(comm->hggcDev));
  char funcStr[128] = {0};
  if (isSymmetric) {
    const char* symFuncName = ncclSymkKernelIdToString(funcIndex);
    sprintf(funcStr, "%s", symFuncName);
  } else {
    getKerFuncStr(funcIndex, funcStr);
  }
  funcIndex = isSymmetric ? funcIndex + TOTAL_FUNCS_NUM : funcIndex;
  dispatchInfo dInfo = {comm->collType, comm->nNodes, nChannels, comm->nBytes};
  NCCLCHECK(setNativeDispatchStrategy(ncclKernFunc, comm->funcStrategy+funcIndex, comm->defaultStrategy, funcStr, comm->compCap, dInfo));
  launchConfig = comm->funcStrategy[funcIndex].launchConfig;
  HGGCCHECK(hggcSetDevice(savedDev));
  return ncclSuccess;
}

ncclResult_t dumpP2pWork(char* lineBuff, struct ncclDevWorkP2p* work, uint16_t index, int nRanks, int rank) {
  const char* indentStr = " ";
  int offset = strlen(lineBuff);
  offset += sprintf(lineBuff + offset, "%s%d: ", indentStr, index);
  offset += sprintf(lineBuff + offset, "sendRegUsed:%d, recvRegUsed:%d sendbuff %p recvbuff %p ",
    work->sendRegUsed, work->recvRegUsed, work->sendbuff, work->recvbuff);
  offset += sprintf(lineBuff + offset, "p2p, ");
  offset += sprintf(lineBuff + offset, "sendRank:%d, recvRank:%d, sendCount:%ld, recvCount:%ld, sendChunkSize:%d, recvChunkSize:%d, ",
    work->sendRank, work->recvRank, work->sendCount, work->recvCount, work->sendChunkSize, work->recvChunkSize);
  offset += sprintf(lineBuff + offset, "\n");
  return ncclSuccess;
}

ncclResult_t dumpCollWork(char* lineBuff, struct ncclDevWorkColl* work, uint16_t index) {
  const char* indentStr = " ";
  int offset = strlen(lineBuff);
  offset += sprintf(lineBuff + offset, "%s%d: ", indentStr, index);
  offset += sprintf(lineBuff + offset, "work nThreads:%d ", work->nThreads);
  offset += sprintf(lineBuff + offset, "sendRegUsed:%d, recvRegUsed:%d sendbuff %p recvbuff %p ",
    work->sendRegUsed, work->recvRegUsed, work->sendbuff, work->recvbuff);
  offset += sprintf(lineBuff + offset, "coll, count:%ld, lastChunkSize:%ld, root:%d, startChannelId:%d, nChannels:%d, commNChannels:%d, ",
    work->count, work->lastChunkSize, work->root, work->startChannelId, work->nChannels, work->commNChannels);
  offset += sprintf(lineBuff + offset, "\n");
  return ncclSuccess;
}

ncclResult_t dumpCollRegWork(char* lineBuff, struct ncclDevWorkCollReg* work, uint16_t index) {
  return ncclSuccess;
}

ncclResult_t dumpWork(char* lineBuff, uint32_t workType, uint64_t offsetBitset, const struct DevWorkStorageRef& storage, int nRanks, int rank) {
  int workIdx = 0;;
  alignas(16) char localWork[sizeof(struct ncclDevWorkCollReg)];
  const size_t workSize = ncclDevWorkSize((enum ncclDevWorkType)workType);
  for (uint16_t i = 0; i < (NCCL_MAX_BIT_IDX_IN_OFFSET_BITSET + 1); i++) {
    if (offsetBitset & (1L << i)) {
      uint32_t workOffset = storage.offsetBase + i * workSize;
      char* workPtr = copyDevWorkFromStorage(localWork, storage, workOffset, workSize);
      if (workPtr == nullptr) continue;
      switch (workType) {
        case ncclDevWorkTypeP2p: {
          ncclDevWorkP2p* p2pWork = (ncclDevWorkP2p*)workPtr;
          NCCLCHECK(dumpP2pWork(lineBuff, p2pWork, workIdx, nRanks, rank));
          break;
        }
        case ncclDevWorkTypeColl: {
          ncclDevWorkColl* collWork = (ncclDevWorkColl*)workPtr;
          NCCLCHECK(dumpCollWork(lineBuff, collWork, workIdx));
          break;
        }
        case ncclDevWorkTypeCollReg:
        default: {
          ncclDevWorkCollReg* collRegWork = (ncclDevWorkCollReg*)workPtr;
          NCCLCHECK(dumpCollRegWork(lineBuff, collRegWork, workIdx));
          break;
        }
      }
      workIdx++;
    }
  }
  return ncclSuccess;
}

static ncclResult_t dumpChannelWorkBatch(ncclComm_t comm, struct ncclKernelPlan* plan, uint16_t channelId, uint16_t blkIdx, void* graphFifoBufHost) {
  uint16_t batchIx = blkIdx;
  constexpr int HeaderSize = 128;
  constexpr int PrintBuffSize = 32768;
  char lineHeader[HeaderSize];

  sprintf(lineHeader, "channel: %u, blkIdx %u, rank: %d, kernelIdx: %lu, channelMask %lx channelWorks %lu totalPreChannelWorks %lu",
   channelId, blkIdx, comm->rank, plan->kernelArgs->kernelIdx, plan->channelMask, plan->channelWorks[channelId], comm->channelWorks[channelId]);
  while (true) {
    char lineBuff[PrintBuffSize];
    int offset = 0;
    struct ncclDevWorkBatch batch = ((struct ncclDevWorkBatch*)(plan->kernelArgs+1))[batchIx];
    char func_str[128] = {0};
    getKerFuncStr(batch.funcId, func_str, NULL, NULL);
    offset += sprintf(lineBuff + offset, "%s batchIx %u workType:%u, funcId:%u(%s), nextJump:%u, nextExtends:%u ",
      lineHeader, batchIx, batch.workType, batch.funcId, func_str, batch.nextJump, batch.nextExtends);
    offset += sprintf(lineBuff + offset, "offsetBase:%x, offsetBitset:%lx ", batch.offsetBase, batch.offsetBitset);
    offset += sprintf(lineBuff + offset, "\n");

    // kernelArgs->workBuf is device addr if comm->workFifoBuf/comm->workFifoBufDev is allocated by ncclGdrCudaCalloc
    // kernelArgs->workBuf is equal to comm->workFifoBufDev if workStorageType is ncclDevWorkStorageTypeFifo.
    // The device masks offsets with workMask for Fifo storage, so the dumper must do the same
    // (offsetBase can exceed workFifoBytes once the circular Fifo wraps).
    char* workPtrBase = (char*)comm->workFifoBuf;
    if (plan->kernelArgs->workStorageType == ncclDevWorkStorageTypeArgs) {
      workPtrBase = (char*)plan->kernelArgs;
    } else if (plan->kernelArgs->workStorageType == ncclDevWorkStorageTypePersistent) {
      workPtrBase = (char*)graphFifoBufHost;
    }

    struct DevWorkStorageRef storage = { workPtrBase, batch.offsetBase, plan->kernelArgs->workMask, plan->kernelArgs->workStorageType };
    NCCLCHECK(dumpWork(lineBuff, batch.workType, batch.offsetBitset, storage, comm->nRanks, comm->rank));

    if (batch.nextExtends) {
      batchIx += batch.nextJump;
    } else {
      if (batch.nextJump == 0) {
        INFO(NCCL_COLL|NCCL_P2P, "\n%s", lineBuff);
        break;
      }
      batchIx += batch.nextJump;
    }
    INFO(NCCL_COLL|NCCL_P2P, "\n%s", lineBuff);
  }
  return ncclSuccess;
}

static ncclResult_t dumpNcclPlan(ncclComm_t comm, struct ncclKernelPlan* plan, void* graphFifoBufHost) {
  uint16_t blkIdx = 0;

  static_assert(MAXCHANNELS <= 64, "MAXCHANNELS must be <= 64");
  for (uint16_t i = 0; i < MAXCHANNELS; i++) {
    if (plan->channelMask & (1L << i)) {
      // coverity[overrun-buffer-arg]
      NCCLCHECK(dumpChannelWorkBatch(comm, plan, i, blkIdx, graphFifoBufHost));
      blkIdx++;
    }
  }
  return ncclSuccess;
}

struct ncclKernelPlanBudget {
  ssize_t inArgsBytes; // Space available within kernel args struct
  ssize_t outArgsBytes; // Space available outside of args struct (fifo or persistent buf)
};

static bool testBudget(
    struct ncclKernelPlanBudget* budget, int nWorkBatches, ssize_t workBytes
  ) {
  ssize_t batchBytes = nWorkBatches*sizeof(struct ncclDevWorkBatch);
  bool ok = false;
  ok |= (batchBytes + workBytes <= budget->inArgsBytes);
  ok |= (batchBytes <= budget->inArgsBytes) && (workBytes <= budget->outArgsBytes);
  return ok;
}

ncclResult_t ncclTasksRegAndEnqueue(ncclComm_t comm) {
  struct ncclKernelPlanner* planner = &comm->planner;
  struct ncclTaskColl *task = ncclIntruQueueHead(&planner->collTaskQueue);
  while (task != nullptr) {
    void* regBufSend[NCCL_MAX_LOCAL_RANKS];
    void* regBufRecv[NCCL_MAX_LOCAL_RANKS];

    struct ncclWorkList* workNode = NULL;
    struct ncclDevWorkColl devWork = {};

    NCCLCHECK(ncclRegisterCollBuffers(comm, task, regBufSend, regBufRecv, &comm->planner.collCleanupQueue));
    devWork.sendbuff = (void*)task->sendbuff;
    devWork.recvbuff = (void*)task->recvbuff;
    devWork.sendbuffOffset = task->sendbuffOffset;
    devWork.recvbuffOffset = task->recvbuffOffset;
    devWork.sendbuffRmtAddrs = task->sendbuffRmtAddrs;
    devWork.recvbuffRmtAddrs = task->recvbuffRmtAddrs;
    devWork.root = task->root;
    devWork.profilerEnabled = ncclProfilerPluginLoaded() && (task->eActivationMask & ncclProfileKernelCh);

    if (task->regBufType & NCCL_IPC_REG_BUFFER) {
      // task->regUsed is used by kernelPluginBuildParams
      task->regUsed = 1;
      devWork.sendRegUsed = 1;
      devWork.recvRegUsed = 1;
    }

    workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkColl>(&comm->memScoped, 1);
    workNode->workType = ncclDevWorkTypeColl;
    workNode->size = sizeof(struct ncclDevWorkColl);
    memcpy((void*)(workNode+1), (void*)&devWork, workNode->size);

    ncclIntruQueueEnqueue(&planner->collWorkQueue, workNode);

    // inject register info into ext kernel.
    if (task->isExtKernel) {
      struct ncclExtTaskColl extTask;
      ncclTaskCollToExtTaskColl(task, &extTask);
      NCCLCHECK(comm->extKernel->kernelPluginBuildParams(&extTask, comm->extKernelPluginWorkspace.extKernelPluginParams));
    }
    task = task->next;
  }
  return ncclSuccess;
}

ncclResult_t saveDevKernelArgs(ncclComm_t comm, struct ncclKernelPlan* plan) {
  KernelInfoMonitor* kernelInfoMonitorPtr = comm->kernelInfoMonitorPtr + comm->kernelNum % pcclDevKernelArgsFifoDepth();
  memcpy(&(kernelInfoMonitorPtr->kernelArgs), plan->kernelArgs, plan->kernelArgsSize);
  comm->kernelNum++;
  for (int i = 0; i < MAXCHANNELS; i++) {
    if (plan->channelMask & (1L << i)) {
      comm->channelKernelIdx[i] = comm->kernelNum;
    }
  }
  return ncclSuccess;
}

PCCL_PARAM(DumpWorkElems, "DUMP_WORK_ELEMS", 0);
ncclResult_t ncclLaunchKernel(ncclComm_t comm, struct ncclKernelPlan* plan) {
  struct ncclKernelPlanner* planner = &comm->planner;
  int nChannels = countOneBits(plan->channelMask);
  dim3 grid = {(unsigned)nChannels, 1, 1};
  dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
  int smem = ncclShmemDynamicSize();
  hggcStream_t launchStream = planner->streams->stream;
  NCCLCHECK(ncclProfilerStartKernelLaunchEvent(plan, launchStream));
  // todo: save hgtxMessage per ncclKernelPlan, need to confirm patpro uses.
  // now only one ncclKernelPlan for pccl group ops, so we save hgtxMessage per group.
  struct pcclHgtxInfo hgtxKernelInfo = pcclHgtxInfo(comm->hgtxMessage);

  if (plan->usingExtKernel) {
    NCCLCHECK(comm->extKernel->kernelPluginExecute(comm->extKernelPluginWorkspace.extKernelPluginParams, launchStream));
    NCCLCHECK(ncclProfilerStopKernelLaunchEvent(plan));
    return ncclSuccess;
  }

  if (nChannels == 0) return ncclSuccess;

  void *args[1] = {plan->kernelArgs};
  if (plan->isSymColl) args[0] = plan->kernelSymkArgs;
  struct dispatchStrategyExAD launchConfig = {0};
  if (!plan->isSymColl) {
    NCCLCHECK(ncclSetDispatchStrategy(plan->kernelFn, comm, launchConfig, plan->funcIndex, nChannels, false));
  } else {
    NCCLCHECK(ncclSetDispatchStrategy(plan->kernelFn, comm, launchConfig, plan->symFuncIndex, nChannels, true));
  }

  LAUNCH_KERNEL(plan->kernelFn, grid, block, args, smem, launchStream, 1, launchConfig);

  NCCLCHECK(ncclProfilerStopKernelLaunchEvent(plan));

  if (comm->hostDevComm.comm.stateMonitorLevel) {
    NCCLCHECK(saveDevKernelArgs(comm, plan));
  }

  return ncclSuccess;
}

ncclResult_t ncclAddProxyOpIfNeeded(ncclComm_t comm, struct ncclProxyOp* op) {
  bool needed = true;
  NCCLCHECK(ncclProxySaveOp(comm, op, &needed));
  if (needed) {
    struct ncclProxyOp* q = ncclMemoryPoolAlloc<struct ncclProxyOp>(
            &comm->memPool_ncclProxyOp, &comm->memPermanent);
    *q = *op; // C++ struct assignment
    ncclIntruQueueEnqueue(&comm->planner.wipPlan.channels[op->channelId].proxyOpQueue, q);
  }
  return ncclSuccess;
}

// When enabled, p2p operations from different epochs (schedule sweeps) are never
// merged into the same work batch. This enforces uniform batching across ranks and
// avoids hangs, matching NCCL. On by default (as in NCCL); set PCCL_P2P_EPOCH_ENABLE=0
// to disable.
PCCL_PARAM(P2pEpochEnable, "P2P_EPOCH_ENABLE", 1);

static void addWorkBatchToPlan(
    ncclComm_t comm, struct ncclKernelPlan* plan, int channelId,
    enum ncclDevWorkType workType, int devFuncId, uint32_t workOffset,
    int p2pRound = -1, int maxP2pWorkPerBatch = NCCL_MAX_DEV_WORK_P2P_PER_BATCH,
    int p2pEpoch = 0
  ) {
  ncclKernelPlanner::WipPlan::Channel* chan = &comm->planner.wipPlan.channels[channelId];
  size_t workSize = ncclDevWorkSize(workType);
  // Conditions causing us to create a new blank batch.
  bool newBatch = (chan->workBatchQueue.tail == nullptr);
  struct ncclDevWorkBatch* batch = nullptr;
  if (!newBatch) {
    batch = &chan->workBatchQueue.tail->batch;
    // All of the conditions that prevent us from appending to current batch.
    newBatch |= batch->workType != (uint8_t)workType;
    newBatch |= batch->funcId != devFuncId;
    // The following ensure the device can handle a batch this large. They have to
    // account for all extension batches being fused together which is why
    // wipBatch.workBytes and wipBatch.nP2ps aren't reset to 0 for a new extension
    // batch further down.
    newBatch |= NCCL_MAX_DEV_WORK_BATCH_BYTES < chan->wipBatch.workBytes + workSize;
    if (workType == ncclDevWorkTypeP2p) {
      // Never merge p2p ops from different epochs (schedule sweeps) into one batch.
      // Enforces uniform batching across ranks; gated by param, off by default.
      if (pcclParamP2pEpochEnable()) newBatch |= chan->wipBatch.p2pEpoch != p2pEpoch;
      // We only allow NCCL_MAX_DEV_WORK_P2P_PER_BATCH ops per batch.
      newBatch |= chan->wipBatch.nP2ps == NCCL_MAX_DEV_WORK_P2P_PER_BATCH;
      newBatch |= chan->wipBatch.nP2ps >= maxP2pWorkPerBatch && chan->wipBatch.p2pRounds[chan->wipBatch.nP2ps - 1] != 0;

      for (int i = 0; i < chan->wipBatch.nP2ps; i++) {
        // Do not allow the same round twice in the same batch.
        newBatch |= p2pRound == chan->wipBatch.p2pRounds[i];
        // Make sure we only aggregate p2p operations within the same p2p round
        // epoch (one epoch is NCCL_MAX_DEV_WORK_P2P_PER_BATCH ops). This enforces
        // uniform batching across ranks in the communicator and prevents hangs.
        newBatch |= (p2pRound / NCCL_MAX_DEV_WORK_P2P_PER_BATCH) !=
                    (chan->wipBatch.p2pRounds[i] / NCCL_MAX_DEV_WORK_P2P_PER_BATCH);
      }
    }
  }

  // Conditions causing us to create an extension batch (prev->nextExtends=1)
  uint32_t offset = newBatch ? 0 : (workOffset - batch->offsetBase);
  bool extendBatch = NCCL_MAX_BIT_IDX_IN_OFFSET_BITSET * workSize < offset;
  extendBatch |= 0 != offset % workSize;
  if (newBatch || extendBatch) {
    if (!newBatch) batch->nextExtends = extendBatch; // Extending the previous batch.
    struct ncclWorkBatchList* batchNode = ncclMemoryStackAlloc<ncclWorkBatchList>(&comm->memScoped);
    // Coverity thinks that ncclIntruQueueEnqueue will access chan->workBatchQueue->tail, which might
    // be NULL.  But that code is guarded by chan->workBatchQueue->head not being NULL, in which
    // case tail won't be NULL either.
    // coverity[var_deref_model:FALSE]
    ncclIntruQueueEnqueue(&chan->workBatchQueue, batchNode);
    batch = &batchNode->batch;
    batch->nextExtends = 0;
    batch->workType = (uint32_t)workType;
    batch->funcId = devFuncId;
    batch->offsetBase = workOffset;
    batch->offsetBitset = 0;
    offset = 0;
    if (newBatch) {
      // Since extension batches are fused together on the device, and these values
      // account for constraints on the fused batch, we only reset the values on
      // a new batch
      chan->wipBatch.workBytes = 0;
      chan->wipBatch.nP2ps = 0;
      // We don't count extension batches since this is used to derive a proxyOpCount,
      // and we wan't all ops which are fused together to have the same value.
      chan->nWorkBatchesP2p += (workType == ncclDevWorkTypeP2p ? 1 : 0);
    }
    plan->nWorkBatches += 1;
  }
  batch->offsetBitset |= 1ull<<(offset/workSize);
  chan->wipBatch.workBytes += workSize;
  if (workType == ncclDevWorkTypeP2p) {
    // We need to ensure that a single batch doesn't have multiple p2p's
    // of the same round since they would use the same connections.
    chan->wipBatch.p2pRounds[chan->wipBatch.nP2ps++] = p2pRound;
    chan->wipBatch.p2pEpoch = p2pEpoch;
  }
}

static ncclResult_t uploadProxyOps(ncclComm_t comm, struct ncclKernelPlan* plan) {
  uint64_t collOpCount = comm->sharedRes->collOpCount;
  uint64_t p2pOpBump[MAXCHANNELS] = {/*0...*/};
  // Advance comm's collOpCount by number of colls in this plan.
  int hasp2p = 0;
  comm->sharedRes->collOpCount += plan->collOpCount;
  comm->collOpCount += plan->collOpCount;

  struct ncclProxyOp* op = ncclIntruQueueHead(&plan->proxyOpQueue);
  while (op != nullptr) {
    op->profilerContext = comm->profilerContext;
    op->eActivationMask = op->coll <= ncclFuncAllReduce ? op->task.coll->eActivationMask : op->task.p2p->eActivationMask;
    op->taskEventHandle = op->coll <= ncclFuncAllReduce ? op->task.coll->eventHandle : op->task.p2p->eventHandle;
    ncclProfilerAddPidToProxyOp(op);
    uint64_t oldId = op->opCount;
    // Ignoring the bottom tag bit, opCount's are zero-based within plan so
    // translate them to the tip of the comm's history.
    if (oldId & 1) { // p2p
      // opCount is monotonic increasing within a plan's channel so just
      // remember last value to compute max.
      p2pOpBump[op->channelId] = (oldId>>1) + 1; // +1 to ensure next plan doesn't collide
      op->opCount = (comm->sharedRes->p2pOpCount[op->channelId]<<1) + oldId;
      hasp2p = 1;
    } else { // coll
      op->opCount = (collOpCount<<1) + oldId;
    }

    NCCLCHECK(ncclProxySaveOp(comm, op, nullptr));
    op->opCount = oldId; // Restore for next uploadProxyOps()
    op = op->enqNext;
  }

  if (hasp2p) {
    for (int c=0; c < MAXCHANNELS; c++) {
      // Advance channel's p2pOpCount by number of p2p's in this plan channel.
      comm->sharedRes->p2pOpCount[c] += p2pOpBump[c];
    }
  }
  return ncclSuccess;
}

static ncclResult_t hostStreamPlanTask(ncclComm_t comm, struct ncclKernelPlan* plan) {
  NCCLCHECK(ncclProfilerStartGroupEvent(plan));
  NCCLCHECK(ncclProfilerStartTaskEvents(plan));
  NCCLCHECK(uploadProxyOps(comm, plan));
  NCCLCHECK(ncclProxyStart(comm));

  NCCLCHECK(ncclProfilerStopTaskEvents(plan));
  NCCLCHECK(ncclProfilerStopGroupEvent(plan));
  if (!plan->persistent) {
    // Notify main thread of our reclaiming. This will reclaim eqInfo concurrently.
    ncclIntruQueueMpscEnqueue(&comm->callbackQueue, &plan->reclaimer);
  }
  return ncclSuccess;
}

static void HGGCRT_CB hostStreamPlanCallback(void *plan_) {
  PCCL_HGTX3_FUNC_RANGE;
  struct ncclKernelPlan* plan = (struct ncclKernelPlan*)plan_;
  ncclResult_t result = hostStreamPlanTask(plan->comm, plan);
  if (result != ncclSuccess) {
    ERROR("hostStreamPlanCallback() failed : %s", pcclGetErrorString(result));
  }
}

ncclResult_t ncclLaunchKernelAfter_NoHggc(ncclComm_t comm, struct ncclKernelPlan* plan) {
  if (!plan->isHostCbEnq) {
    // We are not using the host stream for proxy ops and reclaimation submission.
    NCCLCHECK(hostStreamPlanTask(comm, plan));
  }
  return ncclSuccess;
}

namespace {
  struct KernelFinishCallback {
    struct ncclCommEventCallback base;
    uint32_t workFifoConsumed;
  };
  ncclResult_t KernelFinishCallback_fn(
      ncclComm_t comm, struct ncclCommEventCallback* cb
    ) {
    struct KernelFinishCallback* me = (struct KernelFinishCallback*)cb;
    comm->workFifoConsumed = me->workFifoConsumed;
    HGGCCHECK(hggcEventDestroy(me->base.event));
    free(me);
    return ncclSuccess;
  }
}

ncclResult_t ncclLaunchFinish(ncclComm_t comm) {
  ncclResult_t result = ncclSuccess;
  struct ncclKernelPlanner* planner = &(comm->planner);
  planner->collBytesTotal = 0; // Just in case subtraction during scheduleCollTasksToPlan() doesn't get to 0

  if (ncclIntruQueueEmpty(&planner->planQueue)) {
    return ncclSuccess;
  }

  // Reset queue to empty without destroying plans since those will be sent
  // back to us for reclaiming via callbackQueue.
  ncclIntruQueueConstruct(&planner->planQueue);

  hggcStream_t launchStream = planner->streams->stream; // First user stream gets launch

  if (comm->workFifoProduced - comm->workFifoProducedLastRecorded > comm->workFifoBytes/8) {
    hggcEvent_t finishedEvent = comm->sharedRes->scratchEvent;
    HGGCCHECK(hggcEventRecord(finishedEvent, launchStream));

    comm->workFifoProducedLastRecorded = comm->workFifoProduced;
    struct KernelFinishCallback* cb;
    NCCLCHECK(ncclCalloc(&cb, 1));
    cb->base.event = finishedEvent;
    cb->base.fn = KernelFinishCallback_fn;
    cb->workFifoConsumed = comm->workFifoProduced;
    ncclIntruQueueEnqueue(&comm->eventCallbackQueue, &cb->base);

    // We just stole scratchEvent so must create a new one.
    // coverity[leaked_storage]
    HGGCCHECK(hggcEventCreateWithFlags(&comm->sharedRes->scratchEvent, hggcEventDisableTiming));

    // coverity[leaked_storage]
  }

  // Enqueue event after NCCL kernel

  // Create dependency for deviceStream on launchStream. We know that deviceStream
  // hasn't been modified since launchStream waited on it (in ncclLaunchPrepare),
  // so we can say that launchStream subsumes it.
  NCCLCHECKGOTO(ncclStrongStreamWaitStream(planner->capturingGraph, &comm->sharedRes->deviceStream, launchStream, /*b_subsumes_a=*/true), result, resume1);
  resume1:
    // Create dependency for other user streams (skip launch stream) on deviceStream.
    // Again, the user streams haven't been touched since deviceStream waited on them
    // so we can say they are subsumed by deviceStream.
    struct ncclCudaStreamList* sl = planner->streams->next;
    while (sl != nullptr) {
      NCCLCHECKGOTO(ncclStrongStreamWaitStream(planner->capturingGraph, sl->stream, &comm->sharedRes->deviceStream, /*b_subsumes_a=*/true), result, resume2);
    resume2:
      sl = sl->next;
    }
    // Release device stream as acquired in ncclLaunchPrepare()

    NCCLCHECKGOTO(ncclStrongStreamRelease(planner->capturingGraph, &comm->sharedRes->deviceStream), result, resume3);
  resume3:;
  return result;
}

ncclResult_t ncclLaunchReset(ncclComm_t comm) {
  comm->hgtxMessageIndex = 0;
  return ncclSuccess;
}

/*****************************************************************************/
/* Enqueueing system : computation of kernel and proxy operations parameters */
/*****************************************************************************/

PCCL_PARAM(UseStaticLaunchGrid, "USE_STATIC_LAUNCH_GRID", -2);

static bool algoSupportsRedOp(ncclTaskColl* info, int algorithm) {
  if (algorithm == NCCL_ALGO_PAT && info->func == ncclFuncReduceScatter &&
      (info->opDev.op == ncclDevPreMulSum || info->opDev.op == ncclDevSumPostDiv)) return false;
  return true;
}

static ncclResult_t defaultGetAlgoInfo(ncclComm_t comm, ncclTaskColl* info) {
  float minTime = 3600000000.0; // Hopefully no operation will take an hour to complete.
  // Find algorithm / protocol.
  info->algorithm = NCCL_ALGO_UNDEF;
  info->protocol = NCCL_PROTO_UNDEF;
  // Check collNet support
  int collNetTypeSupport = 0;
  if (comm->collNetSupport)
    NCCLCHECK(collNetReduceSupport(info->datatype, info->opHost, &collNetTypeSupport));
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    if (a == NCCL_ALGO_COLLNET && collNetTypeSupport != 1) continue;
    /* PAT reduceScatter doesn't support scaling yet */
    if (!algoSupportsRedOp(info, a)) continue;
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      float time;
      NCCLCHECK(ncclTopoGetAlgoTime(comm, info->func, a, p, info->nBytes, info->nChannels, &time));
      if (time >= 0 && time < minTime) {
        info->algorithm = a;
        info->protocol = p;
        minTime = time;
      }
    }
  }

  if (info->algorithm == NCCL_ALGO_UNDEF || info->protocol == NCCL_PROTO_UNDEF) {
    ERROR("Error : no algorithm/protocol available");
    return ncclInternalError;
  }

  if (comm->rank == 0) TRACE(NCCL_TUNING, "%ld Bytes -> Algo %d proto %d time %f", info->nBytes, info->algorithm, info->protocol, minTime);
  TRACE(NCCL_COLL, "%ld Bytes -> Algo %d proto %d time %f", info->nBytes, info->algorithm, info->protocol, minTime);

  // CollNet uses one channel for up and one channel for down
  int nc = (info->nChannels > 0) ? info->nChannels :
           (info->algorithm == NCCL_ALGO_COLLNET) ? comm->nChannels/2 :
           (info->algorithm == NCCL_ALGO_FC) ? comm->fcnChannels : comm->nChannels;
  static int useStaticLaunchGrid = pcclParamUseStaticLaunchGrid();

  int nt = comm->maxThreads[info->algorithm][info->protocol];
  int threadThreshold = comm->threadThresholds[info->algorithm][info->protocol];
  if (useStaticLaunchGrid != 1 ){
     while (info->nBytes < nc*nt*threadThreshold) {
       if (info->algorithm != NCCL_ALGO_COLLNET && nc >= 2) nc--;
       else if ((nt % 128) == 0) nt/=2;
       else break;
     }
  }
#ifndef NO_EXTRA_WARP
  if (info->protocol == NCCL_PROTO_SIMPLE || info->protocol == NCCL_PROTO_SIGNAL
      || info->protocol == NCCL_PROTO_SIMPLE_BULK || info->protocol == NCCL_PROTO_SIGNAL_BULK
      || info->protocol == NCCL_PROTO_SIMPLE_BULK_AR) {
    nt += WARP_SIZE; // Extra warp for sync
  }
  if ((info->protocol == NCCL_PROTO_SIMPLE || info->protocol == NCCL_PROTO_SIGNAL
        || info->protocol == NCCL_PROTO_SIMPLE_BULK || info->protocol == NCCL_PROTO_SIGNAL_BULK
        || info->protocol == NCCL_PROTO_SIMPLE_BULK_AR) &&
      info->algorithm == NCCL_ALGO_TREE) {
    nt += WARP_SIZE;
  }
#endif
  if (info->algorithm == NCCL_ALGO_PAT) nt = NCCL_MAX_NTHREADS;
  info->nChannels = nc;
  info->nThreads = nt;
  return ncclSuccess;
}

// Use the default topo-based tuner if tuner plugin is not successful.
// Call the plugin first. Let it set algo+proto, and/or nChannels.
// Then, defaultGetAlgoInfo will set algo/proto if not set, then nChannels and nThreads based on algo/proto.
// Finally, nChannels will be overriden by the plugin setting.
static ncclResult_t getAlgoInfo(ncclComm_t comm, ncclTaskColl* info, ncclSimInfo_t* simInfo = NULL) {
  if (comm->nRanks == 1) {
    // Single-rank: no inter-rank communication is performed, so the estimated
    // time is 0. Keep this aligned with NCCL upstream, where the single-rank
    // cost is explicitly set to 0.0 in updateCollCostTable.
    if (simInfo) simInfo->estimatedTime = 0.f;
    return ncclSuccess;
  }

  info->algorithm = NCCL_ALGO_RING;
  info->protocol = NCCL_PROTO_LL;
  int nChannels = 0;

  if (comm->tuner != NULL) {
    NCCLCHECK(comm->tuner->getCollInfo(
          info->func, comm->nRanks, info->nBytes,
          0, 0, 0,
          &info->algorithm, &info->protocol, &nChannels, &info->nThreads));

    if (nChannels > 0 && !algoSupportsRedOp(info, info->algorithm)) {
      WARN("Tuner plugin selected %s for %s, which does not support the requested reduction op; using the internal tuner instead",
          ncclAlgoStr[info->algorithm], ncclFuncStr[info->func]);
      nChannels = 0;
    }
  }
  // if plugin do not provide valid config, use default;
  if (nChannels <= 0) {
    NCCLCHECK(defaultGetAlgoInfo(comm, info));
  } else {
    info->nChannels = nChannels;
  }
  // record dispatch strategy tuning info
  comm->collType = info->func;
  comm->nBytes = info->nBytes;

  if (simInfo) {
    // Estimate the time of the finally chosen (algorithm, protocol) by reusing
    // the same internal model as defaultGetAlgoInfo (ncclTopoGetAlgoTime).
    // This guarantees a sensible non-zero value regardless of whether the
    // tuner plugin participated in algo selection, and keeps the estimate
    // self-consistent with how algo/proto were selected.
    float time = -1.f;
    if (ncclTopoGetAlgoTime(comm, info->func, info->algorithm, info->protocol,
                            info->nBytes, info->nChannels, &time) == ncclSuccess
        && time >= 0.f) {
      simInfo->estimatedTime = time;
    } else {
      simInfo->estimatedTime = 0.f;
    }
  }
  return ncclSuccess;
}

static ncclResult_t getPatternInfo(struct ncclInfo* info) {
  switch (info->coll) {
    case ncclFuncBroadcast:
      info->pattern = info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeDown : ncclPatternPipelineFrom; break;
    case ncclFuncReduce:
      info->pattern = info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUp : ncclPatternPipelineTo; break;
    case ncclFuncReduceScatter:
      info->pattern = info->algorithm == NCCL_ALGO_PAT ? ncclPatternPatUp : ncclPatternRing; break;
    case ncclFuncAllGather:
      info->pattern = info->algorithm == NCCL_ALGO_PAT ? ncclPatternPatDown : ncclPatternRing; break;
    case ncclFuncAllReduce:
      info->pattern = info->algorithm == NCCL_ALGO_COLLNET ? ncclPatternCollTreeUp : info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUpDown : ncclPatternRingTwice; break;
    default:
      ERROR("Unknown pattern for collective %d algorithm %d", info->coll, info->algorithm);
      return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t getLoopInfo(struct ncclInfo* info) {
  switch (info->pattern) {
    case ncclPatternTreeUp:
    case ncclPatternTreeDown:
    case ncclPatternTreeUpDown:
    case ncclPatternPipelineFrom:
    case ncclPatternPipelineTo:
    case ncclPatternCollTreeUp:
    case ncclPatternCollTreeDown:
    case ncclPatternPatUp:
    case ncclPatternPatDown:
      info->nstepsPerLoop = info->nchunksPerLoop = 1; break;
    case ncclPatternRing:
      info->nstepsPerLoop = info->comm->nRanks-1; info->nchunksPerLoop = info->comm->nRanks; break;
    case ncclPatternRingTwice:
      info->nstepsPerLoop = 2*(info->comm->nRanks-1); info->nchunksPerLoop = info->comm->nRanks; break;
    default:
      ERROR("Unknown pattern %d", info->pattern);
      return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t computeColl(struct ncclInfo* info /* input */, struct ncclDevWorkColl* work, struct ncclProxyOp* proxyOp /* output */) {
  work->comm = info->comm->devComm;

  // Set nstepsPerLoop and nchunksPerLoop
  NCCLCHECK(getPatternInfo(info));
  NCCLCHECK(getLoopInfo(info));

  work->sendbuff = info->sendbuff;
  work->recvbuff = info->recvbuff;
  work->root = info->root;
  work->count = (ssize_t)info->count;
  work->nChannels = info->nChannels;
  work->nThreads = info->nThreads;
  work->redOpArgIsPtr = info->opFull.scalarArgIsPtr;
  work->redOpArg = info->opFull.scalarArg;

  int stepSize   = info->comm->buffSizes[info->protocol]/NCCL_STEPS;
  int chunkSteps = ((info->protocol == NCCL_PROTO_SIMPLE || info->protocol == NCCL_PROTO_SIGNAL
        || info->protocol == NCCL_PROTO_SIMPLE_BULK || info->protocol == NCCL_PROTO_SIGNAL_BULK
        || info->protocol == NCCL_PROTO_SIMPLE_BULK_AR)
      && info->algorithm == NCCL_ALGO_RING) ? info->chunkSteps : 1;
  int sliceSteps = ((info->protocol == NCCL_PROTO_SIMPLE || info->protocol == NCCL_PROTO_SIGNAL
        || info->protocol == NCCL_PROTO_SIMPLE_BULK || info->protocol == NCCL_PROTO_SIGNAL_BULK
        || info->protocol == NCCL_PROTO_SIMPLE_BULK_AR)
      && info->algorithm == NCCL_ALGO_RING) ? info->sliceSteps : 1;
  int chunkSize  = stepSize*chunkSteps;

  // Compute lastChunkSize
  if (info->algorithm == NCCL_ALGO_TREE &&
      (info->protocol == NCCL_PROTO_SIMPLE || info->protocol == NCCL_PROTO_SIGNAL
       || info->protocol == NCCL_PROTO_SIMPLE_BULK || info->protocol == NCCL_PROTO_SIGNAL_BULK
       || info->protocol == NCCL_PROTO_SIMPLE_BULK_AR)) {
    if (info->pattern == ncclPatternTreeUpDown) {
      // Optimize chunkSize / nSteps
      while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].tree.depth*8 && chunkSize > 131072) chunkSize /= 2;
      while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].tree.depth*4 && chunkSize > 65536) chunkSize /= 2;
      while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].tree.depth && chunkSize > 32768) chunkSize /= 2;
    }
    // Use lastChunkSize as chunkSize
    work->lastChunkSize = chunkSize / ncclTypeSize(info->datatype);
  } else if (info->algorithm == NCCL_ALGO_COLLNET &&
      (info->protocol == NCCL_PROTO_SIMPLE || info->protocol == NCCL_PROTO_SIGNAL
       || info->protocol == NCCL_PROTO_SIMPLE_BULK || info->protocol == NCCL_PROTO_SIGNAL_BULK
       || info->protocol == NCCL_PROTO_SIMPLE_BULK_AR)) {
    // Optimize chunkSize / nSteps
    while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].collTree.depth*16 && chunkSize > 131072) chunkSize /= 2;
    while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].collTree.depth*4 && chunkSize > 65536) chunkSize /= 2;
    while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].collTree.depth && chunkSize > 32768) chunkSize /= 2;
    // Use lastChunkSize as chunkSize
    work->lastChunkSize = chunkSize / ncclTypeSize(info->datatype);
  } else if (info->algorithm == NCCL_ALGO_PAT &&
      (info->protocol == NCCL_PROTO_SIMPLE || info->protocol == NCCL_PROTO_SIGNAL
       || info->protocol == NCCL_PROTO_SIMPLE_BULK || info->protocol == NCCL_PROTO_SIGNAL_BULK
       || info->protocol == NCCL_PROTO_SIMPLE_BULK_AR)) {
    // Optimize chunkSize / nSteps for PAT
    while (chunkSize*info->nChannels*32 > info->nBytes && chunkSize > 65536) chunkSize /= 2;
    // Use lastChunkSize as chunkSize
    work->lastChunkSize = chunkSize / ncclTypeSize(info->datatype);
  } else if (info->protocol == NCCL_PROTO_LL) {
    const ssize_t sliceSize = stepSize*sizeof(uint64_t)/sizeof(union ncclLLFifoLine);
    const ssize_t loopSize = info->nChannels*info->nchunksPerLoop*(ssize_t)sliceSize;
    work->lastChunkSize = DIVUP((info->nBytes-(info->nBytes/loopSize)*loopSize), info->nChannels*info->nchunksPerLoop);
    ALIGN_SIZE(work->lastChunkSize, info->nThreads*sizeof(uint64_t));
    work->lastChunkSize /= ncclTypeSize(info->datatype);
  } else if (info->algorithm == NCCL_ALGO_TREE && info->protocol == NCCL_PROTO_LL128) {
    int nNodes = info->comm->nNodes;
    float ppn = info->comm->nRanks / (float)nNodes;
    float nstepsLL128 = 1+log2i(nNodes) + 0.1*ppn;
    while (info->nBytes / (info->nChannels*chunkSize) < nstepsLL128*64/ppn && chunkSize > 131072) chunkSize /= 2;
    while (info->nBytes / (info->nChannels*chunkSize) < nstepsLL128*16/ppn && chunkSize > 32768) chunkSize /= 2;
    // Use lastChunkSize as chunkSize
    work->lastChunkSize = chunkSize*NCCL_LL128_DATAELEMS/(NCCL_LL128_LINEELEMS*ncclTypeSize(info->datatype));
  }

  // Compute nSteps for proxies
  int chunkEffectiveSize = chunkSize;
  if (info->protocol == NCCL_PROTO_LL) chunkEffectiveSize /= 2;
  if (info->protocol == NCCL_PROTO_LL128) chunkEffectiveSize = (chunkSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;
  if (info->comm->rank == 0) VERBOSE(NCCL_COLL, "Coll %d, size %ld -> nChannels: %d nThreads: %d, chunkSize %d (algo %d proto %d)\n",
      info->coll, info->nBytes, info->nChannels, info->nThreads, chunkSize, info->algorithm, info->protocol);
  int nLoops = (int)(DIVUP(info->nBytes, (((size_t)(info->nChannels))*info->nchunksPerLoop*chunkEffectiveSize)));
  proxyOp->nsteps = info->nstepsPerLoop * nLoops * chunkSteps;
  proxyOp->sliceSteps = sliceSteps;
  proxyOp->chunkSteps = chunkSteps;
  proxyOp->chunkSize = chunkSize;
  proxyOp->protocol = info->protocol;
  proxyOp->dtype = info->datatype;
  proxyOp->redOp = info->algorithm != NCCL_ALGO_COLLNET ? ncclNumOps : // Only set redOp when using CollNet
                     info->opFull.op==ncclDevPreMulSum || info->opFull.op==ncclDevSumPostDiv ? ncclSum : // Network sees avg as sum
                     info->op;
  proxyOp->pattern = info->pattern;
  proxyOp->root = info->root;
  proxyOp->peer = info->root;
  // This is used by P2P to reduce the receive buffer size. We don't use it in collectives
  // because some protocols need to transmit more than the total size, plus they sometimes
  // round up
  if (info->pattern == ncclPatternPatUp || info->pattern == ncclPatternPatDown)
    proxyOp->nbytes = DIVUP(info->nBytes, info->nChannels);
  else
    proxyOp->nbytes = stepSize*proxyOp->sliceSteps;

  TRACE(NCCL_COLL,"opCount %lx slicesteps %d spl %d cpl %d nbytes %zi -> protocol %d nchannels %d nthreads %d, nloops %d nsteps %d chunksize %d comm %p",
      proxyOp->opCount, sliceSteps, info->nstepsPerLoop, info->nchunksPerLoop, info->nBytes, info->protocol, info->nChannels, info->nThreads,
      nLoops, proxyOp->nsteps, chunkSize, info->comm);
  return ncclSuccess;
}

#define RMT_BULK_V4_ALIGN 128
ncclResult_t ncclUserBuffAlignmentCheck(struct ncclInfo* info) {
  // check kernel input buffers alignment conditions
  uint64_t sendbuffAligned = (uint64_t)(info->sendbuff) % RMT_BULK_V4_ALIGN;
  uint64_t recvbuffAligned = (uint64_t)(info->recvbuff) % RMT_BULK_V4_ALIGN;
  if (sendbuffAligned != 0 || recvbuffAligned != 0) {
    TRACE(NCCL_COLL|NCCL_P2P, "sendbuff %p, recvbuff %p not both 128B aligned", info->sendbuff, info->recvbuff);
  }

  return ncclSuccess;
}

static ncclResult_t addProfilerProxyOpIfNeeded(ncclComm_t comm, struct ncclProxyOp* op) {
  int tmp = op->pattern;
  op->pattern = ncclPatternProfiler;
  ncclResult_t ret = ncclAddProxyOpIfNeeded(comm, op);
  op->pattern = tmp;
  return ret;
}

#define NCCL_MIN_CHANNEL_SIZE (NCCL_LL_THREAD_THRESHOLD*64)
#define NCCL_AGG_CHANNEL_SIZE (1LL << 21) /* 2 MiB, ideal per-channel size to fully utilize bandwidth */

ncclResult_t ncclComputeP2pProto(ncclComm_t comm, int* protocol) {
  ncclResult_t ret = ncclSuccess;
  // fixed priority
  if (comm->p2pProtoEnable[NCCL_PROTO_SIMPLE_BULK]) *protocol = NCCL_PROTO_SIMPLE_BULK;
  else if (comm->p2pProtoEnable[NCCL_PROTO_SIMPLE_BULK_AR]) *protocol = NCCL_PROTO_SIMPLE_BULK_AR;
  else if (comm->p2pProtoEnable[NCCL_PROTO_SIMPLE]) *protocol = NCCL_PROTO_SIMPLE;
  else if (comm->p2pProtoEnable[NCCL_PROTO_SIGNAL_BULK]) *protocol = NCCL_PROTO_SIGNAL_BULK;
  else if (comm->p2pProtoEnable[NCCL_PROTO_SIGNAL]) *protocol = NCCL_PROTO_SIGNAL;
  else *protocol = NCCL_PROTO_SIMPLE;

  if (usingMemcpy() && NCCL_PROTO_SIMPLE != *protocol) {
    INFO(NCCL_P2P, "p2p use memcpy, fallback to Simple");
    *protocol = NCCL_PROTO_SIMPLE;
  }
  return ret;
}

uint16_t ncclComputeP2pFuncIndex(int protocol) {
  uint16_t funcIndex = FUNC_INDEX_P2P_SIMPLE;
  switch (protocol) {
    case NCCL_PROTO_SIMPLE_BULK:
      funcIndex = FUNC_INDEX_P2P_SIMPLE_BULK;
      break;
    case NCCL_PROTO_SIGNAL:
      funcIndex = FUNC_INDEX_P2P_SIGNAL;
      break;
    case NCCL_PROTO_SIGNAL_BULK:
      funcIndex = FUNC_INDEX_P2P_SIGNAL_BULK;
      break;
    case NCCL_PROTO_SIMPLE_BULK_AR:
      funcIndex = FUNC_INDEX_P2P_SIMPLE_BULK_AR;
      break;
    default:
      funcIndex = FUNC_INDEX_P2P_SIMPLE;
      break;
  }
  return funcIndex;
}

// Spin until its safe to increase comm->workFifoProduced to desiredProduced.
static ncclResult_t waitWorkFifoAvailable(ncclComm_t comm, uint32_t desiredProduced) {
  bool hasRoom = (desiredProduced - comm->workFifoConsumed) <= comm->workFifoBytes;
  if (!hasRoom) {
    while (true) {
      NCCLCHECK(ncclCommPollEventCallbacks(comm, /*waitSome=*/true));
      hasRoom = (desiredProduced - comm->workFifoConsumed) <= comm->workFifoBytes;
      if (hasRoom) break;
      sched_yield();
    }
  }
  return ncclSuccess;
}

namespace {
  struct uploadWork_cleanup_t {
    struct ncclCommEventCallback base;
    void *hostBuf;
  };
  ncclResult_t uploadWork_cleanup_fn(
      ncclComm_t comm, struct ncclCommEventCallback* cb
    ) {
    struct uploadWork_cleanup_t* me = (struct uploadWork_cleanup_t*)cb;
    NCCLCHECK(ncclFree(me->hostBuf));
    HGGCCHECK(hggcEventDestroy(me->base.event));
    NCCLCHECK(ncclFree(me));
    return ncclSuccess;
  }
}

static ncclResult_t uploadWork(ncclComm_t comm, struct ncclKernelPlan* plan) {
  // isCeP2p plans carry no ncclDevKernelArgs at all (kernelArgs is null);
  // their device-visible work list was already reserved from the work FIFO
  // and filled by pcclP2pDmaBuildPlan.
  if (plan->isSymColl || plan->isCeColl || plan->isCeP2p) return ncclSuccess;

  size_t workBytes = plan->workBytes;
  size_t batchBytes = plan->nWorkBatches*sizeof(struct ncclDevWorkBatch);
  void* fifoBufHost;
  uint32_t fifoCursor, fifoMask;

  switch (plan->workStorageType) {
  case ncclDevWorkStorageTypeArgs:
    plan->kernelArgs->workBuf = nullptr;
    fifoBufHost = (void*)plan->kernelArgs;
    fifoCursor = sizeof(ncclDevKernelArgs) + batchBytes;
    fifoMask = ~0u;
    break;
  case ncclDevWorkStorageTypeFifo:
    fifoBufHost = comm->workFifoBuf;
    fifoCursor = comm->workFifoProduced;
    fifoMask = comm->workFifoBytes-1;
    NCCLCHECK(waitWorkFifoAvailable(comm, fifoCursor + workBytes));
    plan->kernelArgs->workBuf = comm->workFifoBufDev;
    break;
  case ncclDevWorkStorageTypePersistent:
    // We rely on 16-byte alignment
    #if __cplusplus >= 201103L
    fifoBufHost = aligned_alloc(16, ROUNDUP(workBytes, 16));
    #else
    static_assert(16 <= alignof(max_align_t), "We rely on 16-byte alignment.");
    fifoBufHost = malloc(workBytes);
    #endif
    fifoCursor = 0;
    fifoMask = ~0u;
    break;
  default:
    return ncclInternalError;
  }
  plan->kernelArgs->workMask = fifoMask;

  // Batches were placed after kernelArgs by finishPlan(). Only thing left to
  // do is translate the work offset from zero based (in plan) to:
  //  ncclDevWorkStorageTypeArgs: offset from beginning of kernel args
  //  ncclDevWorkStorageTypeFifo: offset from base of fifo
  //  ncclDevWorkStorageTypePersistent: no translation since our dedicated buffer will also begin at zero.
  struct ncclDevWorkBatch* batchZero = (struct ncclDevWorkBatch*)(plan->kernelArgs+1);
  for (int b=0; b < plan->nWorkBatches; b++) {
    batchZero[b].offsetBase += fifoCursor;
  }

  // Write the channel-shared work structs.
  struct ncclWorkList* workNode = ncclIntruQueueHead(&plan->workQueue);
  while (workNode != nullptr) {
    char* dst = (char*)fifoBufHost;
    char* src = (char*)(workNode+1);
    for (int n = workNode->size; n != 0; n -= 16) {
      memcpy(
        __builtin_assume_aligned(dst + (fifoCursor & fifoMask), 16),
        __builtin_assume_aligned(src, 16),
        16
      );
      fifoCursor += 16;
      src += 16;
    }
    workNode = workNode->next;
  }

  plan->kernelArgs->kernelIdx = comm->kernelNum + 1 + comm->planner.uploadKernelSeq++;
  if (pcclParamDumpWorkElems() == 1) {
    for(int i = 0; i < MAXCHANNELS; i++) {
      comm->channelWorks[i] += plan->channelWorks[i];
    }
    NCCLCHECK(dumpNcclPlan(comm, plan, fifoBufHost));
  }

  switch (plan->workStorageType) {
  case ncclDevWorkStorageTypeFifo:
    comm->workFifoProduced = fifoCursor;
    if (comm->workFifoBufGdrHandle != nullptr) wc_store_fence();
    break;
  case ncclDevWorkStorageTypePersistent:
    { ncclResult_t result = ncclSuccess;
      struct uploadWork_cleanup_t* cleanup = nullptr;
      hggcStreamCaptureMode mode = hggcStreamCaptureModeRelaxed;
      void* fifoBufDev = nullptr;
      hggcStream_t deviceStream = comm->sharedRes->deviceStream.hggcStream; // todo

      HGGCCHECKGOTO(hggcThreadExchangeStreamCaptureMode(&mode), result, fail);

      // Acquire deviceStream. Since the user's graph will be launched later and it also
      // acquires the deviceStream, it will observe this upload.
      NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(), &comm->sharedRes->deviceStream), result, fail);

      HGGCCHECKGOTO(hggcMallocAsync(&fifoBufDev, workBytes, comm->memPool, deviceStream), result, fail);
      plan->workBufPersistent = fifoBufDev;
      plan->kernelArgs->workBuf = fifoBufDev;

      // coverity[uninit_use_in_call:FALSE] => fifoBufHost is never NULL
      HGGCCHECKGOTO(hggcMemcpyAsync(fifoBufDev, fifoBufHost, workBytes, hggcMemcpyDefault, deviceStream), result, fail);
      hggcEvent_t memcpyDone;
      HGGCCHECKGOTO(hggcEventCreateWithFlags(&memcpyDone, hggcEventDisableTiming), result, fail);
      HGGCCHECKGOTO(hggcEventRecord(memcpyDone, deviceStream), result, fail);

      NCCLCHECKGOTO(ncclCalloc(&cleanup, 1), result, fail);
      cleanup->base.fn = uploadWork_cleanup_fn;
      cleanup->base.event = memcpyDone;
      cleanup->hostBuf = fifoBufHost;
      ncclIntruQueueEnqueue(&comm->eventCallbackQueue, (struct ncclCommEventCallback *)cleanup);

      NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->deviceStream), result, fail);
      NCCLCHECKGOTO(ncclCommPollEventCallbacks(comm, /*waitSome=*/false), result, fail);

    finish_scope:
      if (mode != hggcStreamCaptureModeRelaxed) (void)hggcThreadExchangeStreamCaptureMode(&mode);
      return result;
    fail:
      if (!cleanup) free(fifoBufHost);
      goto finish_scope;
    } break;
  default: break;
  }
  return ncclSuccess;
}

ncclResult_t ncclLaunchKernelBefore_NoUncapturedHggc(ncclComm_t comm, struct ncclKernelPlan* plan) {
  // This code is called after we've checked in to the intra-process barrier
  // but before launching the kernel. We are not allowed to call HGGC unless the
  // kernel launch is captured.
  NCCLCHECK(uploadWork(comm, plan));
  return ncclSuccess;
}

static ncclResult_t reclaimPlan(ncclComm_t comm, struct ncclCommCallback* me) {
  struct ncclKernelPlan* plan = (struct ncclKernelPlan*)me; // cast from first member `reclaim`
  if (plan->persistent) {
    comm->sharedRes->persistentRefs -= 1;
    comm->localPersistentRefs -= 1;
    if (plan->workStorageType == ncclDevWorkStorageTypePersistent) {
      hggcStreamCaptureMode mode = hggcStreamCaptureModeRelaxed;
      HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
      HGGCCHECK(hggcFree(plan->workBufPersistent));
      HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
    }
  }
  // Free coll tasks
  struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
  while (ct != nullptr) {
    struct ncclTaskColl* ct1 = ct->next;
    ncclMemoryPoolFree(&comm->memPool_ncclTaskColl, ct);
    ct = ct1;
  }
  // Free p2p tasks
  struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
  while (pt != nullptr) {
    struct ncclTaskP2p* pt1 = pt->next;
    ncclMemoryPoolFree(&comm->memPool_ncclTaskP2p, pt);
    pt = pt1;
  }
  // Free proxy ops

  struct ncclProxyOp* q = ncclIntruQueueHead(&plan->proxyOpQueue);
  while (q != nullptr) {
    struct ncclProxyOp* q1 = q->enqNext;
    ncclMemoryPoolFree(&comm->memPool_ncclProxyOp, q);
    q = q1;
  }

  // Run other free callbacks
  ncclResult_t result = ncclSuccess;
  while (!ncclIntruQueueEmpty(&plan->cleanupQueue)) {
    struct ncclCommCallback* cb = ncclIntruQueueDequeue(&plan->cleanupQueue);
    ncclResult_t res1 = cb->fn(comm, cb); // Expect to reclaim memory of cb
    if (res1 != ncclSuccess) result = res1;
  }
  NCCLCHECK(result);

  // Multi-task path pool-allocates 4KB args; fast-path uses memScoped.
  if (plan->isSymColl && plan->workStorageType == ncclDevWorkStorageTypeArgs
      && plan->kernelSymkArgs != nullptr
      && plan->kernelArgsSize == sizeof(pcclSymkDevWorkArgs4K)) {
    ncclMemoryPoolFree<pcclSymkDevWorkArgs4K>(&comm->memPool_kernelSymkArgs,
        reinterpret_cast<pcclSymkDevWorkArgs4K*>(plan->kernelSymkArgs));
    plan->kernelSymkArgs = nullptr;
  }

  // // Free plan struct
  ncclMemoryPoolFree(&comm->memPool_ncclKernelPlan, plan);
  return ncclSuccess;
}

static void persistentDestructor(void* plans_) {
  struct ncclKernelPlan* plan = (struct ncclKernelPlan*)plans_;
  ncclComm_t comm = plan->comm;
  while (plan != nullptr) {
    struct ncclKernelPlan* next = plan->next;
    ncclIntruQueueMpscEnqueue(&comm->callbackQueue, &plan->reclaimer);
    plan = next;
  }
}

#define PCCL_DEFAULT_LL_THRESHOLD 8192
#define PCCL_DEFAULT_LL_THRESHOLD_PPU0015 16384
NCCL_PARAM(P2pLLThreshold, "P2P_LL_THRESHOLD", -2);
static ncclResult_t addP2pToPlan(
    ncclComm_t comm, struct ncclKernelPlan* plan, int baseProtocol, int funcIndex,
    int p2pRound, int p2pEpoch, int channelId,
    int sendRank, void* sendAddr, ssize_t sendBytes,
    int recvRank, void* recvAddr, ssize_t recvBytes,
    ssize_t totSendBytes, ssize_t totRecvBytes,
    struct ncclTaskP2p** p2pTasks, int maxP2pWorkPerBatch
  ) {
  constexpr int connIndex = 1;
  bool selfSend = (sendRank == comm->rank);
  if (selfSend && totSendBytes != totRecvBytes) return ncclInvalidUsage;
  // recv: dir=0, send: dir=1
  void* addrs[2] = {recvAddr, sendAddr};
  ssize_t bytes[2] = {recvBytes, sendBytes};
  const ssize_t totBytes[2] = {totRecvBytes, totSendBytes};
  bool protoLL[2] = {!selfSend, !selfSend};
  bool network[2] = {false, false};
  struct ncclProxyOp proxyOps[2] = {};
  int nProxyOps = selfSend ? 0 : 2;
  if (!selfSend) {
    struct ncclChannelPeer** channelPeers = comm->channels[channelId].peers;
    for (int dir = 0; dir <= 1; dir++) {
      int peerRank = dir ? sendRank : recvRank;
      struct ncclConnector* conn = dir ? &channelPeers[peerRank]->send[connIndex]
                                       : &channelPeers[peerRank]->recv[connIndex];
      protoLL[dir] &= conn->conn.buffs[NCCL_PROTO_LL] != nullptr;
      network[dir] |= conn->transportComm == (dir ? &netTransport.send : &netTransport.recv);
    }
  }

  // Arrays indexed by dir where recv=0, send=1:
  int protocol[2];
  int stepSize[2];
  int chunkSize[2];
  int chunkDataSize[2];
  bool ipcRegistered[2] = {false, false};
  ssize_t llThreshold = ncclParamP2pLLThreshold() >= 0 ? ncclParamP2pLLThreshold() :
                                                         (comm->compCap < 89 ? PCCL_DEFAULT_LL_THRESHOLD : PCCL_DEFAULT_LL_THRESHOLD_PPU0015);
  for (int dir = 0; dir < 2; dir++) { // 0=recv, 1=send
    // Only SIMPLE/SIMPLE_BULK kernels host the runtime LL switch.
    protoLL[dir] &= (baseProtocol == NCCL_PROTO_SIMPLE || baseProtocol == NCCL_PROTO_SIMPLE_BULK || baseProtocol == NCCL_PROTO_SIMPLE_BULK_AR);
    // Runtime LL switch gated by NCCL_P2P_LL_THRESHOLD on the total p2p payload across channels.
    protoLL[dir] &= totBytes[dir] <= llThreshold;
    protocol[dir] = protoLL[dir] ? NCCL_PROTO_LL : baseProtocol;

    stepSize[dir]  = comm->buffSizes[protocol[dir]] / NCCL_STEPS / SENDRECV_SLICEFACTOR;
    chunkSize[dir] = stepSize[dir];
    if (network[dir]) {
      // Tune chunk size for the network
      if (protocol[dir] != NCCL_PROTO_LL && bytes[dir] < stepSize[dir]) chunkSize[dir] /= 4;
      else if (bytes[dir] < 8 * stepSize[dir]) chunkSize[dir] /= 2;
    }
    chunkDataSize[dir] = chunkSize[dir];
    if (protocol[dir] == NCCL_PROTO_LL) chunkDataSize[dir] /= 2;

    if (bytes[dir] > 0 && addrs[dir] && PROTO_IS_SIMPLE_CLASS(protocol[dir]) && !selfSend) {
      int peerRank = dir ? sendRank : recvRank;
      int regFlag = 0;
      struct ncclChannelPeer** channelPeers = comm->channels[channelId].peers;
      struct ncclConnector* conn = dir ? &channelPeers[peerRank]->send[connIndex]
        : &channelPeers[peerRank]->recv[connIndex];
      void* regAddr = NULL;
      if (conn->conn.direct & NCCL_DIRECT_GPU) {
        // We require users registering buffers on both sides
        // as we do not use direct_read for now, send side only need set regUsed flag.
        NCCLCHECK(ncclRegisterP2pIpcBuffer(comm, addrs[dir], bytes[dir], peerRank, &regFlag, &regAddr, &plan->cleanupQueue));
        if (regFlag) {
          if (dir == 0) recvAddr = regAddr;
        }
      }
      ipcRegistered[dir] = regFlag ? true : false;
    }
  }

  VERBOSE(NCCL_P2P, "P2P rank=%d ch=%d recv[peer=%d bytes=%ld/%ld proto=%s] send[peer=%d bytes=%ld/%ld proto=%s]",
          comm->rank, channelId,
          recvRank, bytes[0], totBytes[0], ncclProtoStr[protocol[0]],
          sendRank, bytes[1], totBytes[1], ncclProtoStr[protocol[1]]);

  struct ncclWorkList* workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkP2p>(&comm->memScoped, 1);
  workNode->workType = ncclDevWorkTypeP2p;
  workNode->size = sizeof(struct ncclDevWorkP2p);
  ncclIntruQueueEnqueue(&plan->workQueue, workNode);
  uint32_t workOffset = plan->workBytes;
  plan->workBytes += sizeof(struct ncclDevWorkP2p);

  struct ncclDevWorkP2p* work = (struct ncclDevWorkP2p*)(workNode+1);
  work->comm = comm->devComm;
  work->sendbuff = sendAddr;
  work->sendChunkSize = chunkSize[1];
  work->sendCount = sendBytes;
  work->sendRegUsed = ipcRegistered[1];
  work->sendProtoLL = protoLL[1];
  work->sendRank = sendRank;
  work->recvbuff = recvAddr;
  work->recvChunkSize = chunkSize[0];
  work->recvCount = recvBytes;
  work->recvRank = recvRank;
  work->recvRegUsed = ipcRegistered[0];
  work->recvProtoLL = protoLL[0];
  work->profilerEnabled = ncclProfilerPluginLoaded() && ((p2pTasks[0] ? p2pTasks[0] : p2pTasks[1])->eActivationMask & ncclProfileKernelCh);

  plan->channelMask |= uint64_t(1)<<channelId;
  // Add batch first.
  addWorkBatchToPlan(comm, plan, channelId, ncclDevWorkTypeP2p, funcIndex, workOffset, p2pRound, maxP2pWorkPerBatch, p2pEpoch);
  plan->channelWorks[channelId]++;
  plan->threadPerBlock = std::max<unsigned>(plan->threadPerBlock, NCCL_MAX_NTHREADS);

  // Record the first kernel to launch
  // Just for HGGC kernel to know this is a P2P operation
  // The HGGC kernel does not use the inlined first work element as fastpath argument
  if (plan->kernelFn == NULL) {
    plan->kernelFn = ncclKerns[funcIndex];
    plan->funcIndex = funcIndex;
  }

  for (int dir = 0; dir < nProxyOps; dir++) {
    struct ncclProxyOp* op = &proxyOps[dir];
    op->root = dir ? sendRank : recvRank;
    op->sliceSteps = 1;
    op->chunkSteps = 1;
    op->dtype = ncclInt8;
    op->redOp = ncclSum;
    op->protocol = protocol[dir];
    op->pattern = dir ? ncclPatternSend : ncclPatternRecv;
    op->chunkSize = chunkSize[dir];
    op->coll = p2pTasks[dir] ? p2pTasks[dir]->func : 0;
    op->collAPI = p2pTasks[dir] ? p2pTasks[dir]->collAPI : 0;
    op->task.p2p = p2pTasks[dir];
    op->rank = comm->rank;
    op->eActivationMask = p2pTasks[dir] ? p2pTasks[dir]->eActivationMask : 0;
    // The following are modified per channel below:
    // op->channelId, op->opCount, op->nsteps, op->nbytes.
  }

  int incWorkCounter = -1;
  for (int dir = 0; dir < nProxyOps; dir++) {
    ssize_t bytes = dir ? work->sendCount : work->recvCount;
    if (bytes < 0) {
      proxyOps[dir].nsteps = 0;
    } else if (bytes == 0) {
      proxyOps[dir].nsteps = 1;
      proxyOps[dir].nbytes = 0;
    } else {
      proxyOps[dir].nsteps = DIVUP(bytes, chunkDataSize[dir]);
      proxyOps[dir].nbytes = bytes;
      if (proxyOps[dir].protocol == NCCL_PROTO_LL) {
        proxyOps[dir].nbytes *= 2;
        proxyOps[dir].nbytes = ROUNDUP(proxyOps[dir].nbytes, sizeof(union ncclLLFifoLine));
      }
    }

    // Increment work counter for <send, recv> pair rather than individual p2p
    if (proxyOps[dir].nsteps && incWorkCounter < 0) {
      proxyOps[dir].incWorkCounter = true;
      incWorkCounter = dir;
    }

    if (proxyOps[dir].nsteps != 0) {
      // Calculate the opCount after adding batch since then the batch count will
      // equal one plus the batch index this p2p settled in.
      proxyOps[dir].channelId = channelId;
      proxyOps[dir].opCount = uint64_t(comm->planner.wipPlan.channels[channelId].nWorkBatchesP2p)<<1 | 1;
      NCCLCHECK(ncclAddProxyOpIfNeeded(comm, &proxyOps[dir]));
      NCCLCHECK(addProfilerProxyOpIfNeeded(comm, &proxyOps[dir]));
    }
  }

  return ncclSuccess;
}

PCCL_PARAM(P2pSwitchSerialThreshold, "P2P_SWITCH_SERIAL_THRESHOLD", 65536);
PCCL_PARAM(P2pSwitchSerialFactor, "P2P_SWITCH_SERIAL_FACTOR", -2);
PCCL_PARAM(P2pSwitchMaxWorkPerBatch, "P2P_SWITCH_MAX_WORK_PER_BATCH", -2);

struct P2pSwitchAutoConfig {
  int nRanks;
  size_t minPeerBytes;
  size_t maxPeerBytes;
  int serialFactor;       // -1 keeps the old factor of 8
  int maxWorkPerBatch;    // -1 keeps the old limit of 1
};

static constexpr P2pSwitchAutoConfig p2pSwitchAutoConfigs[] = {
  // nRanks, minPeerBytes, maxPeerBytes, serialFactor, maxWorkPerBatch
  {16,  64_MB, 128_MB,  2, 2},
  {32,  64_MB, 128_MB,  2, 2},
  {32, 256_MB, 256_MB, -1, 2},
  {64,  64_MB,  64_MB,  1, 2},
  {64, 128_MB, 256_MB,  2, 2},
};

static const P2pSwitchAutoConfig* getP2pSwitchAutoConfig(
    ncclComm_t comm, size_t peerBytes) {
  for (const P2pSwitchAutoConfig& config : p2pSwitchAutoConfigs) {
    if (comm->nRanks == config.nRanks &&
        peerBytes >= config.minPeerBytes && peerBytes <= config.maxPeerBytes) {
      return &config;
    }
  }
  return nullptr;
}

static size_t getP2pSwitchSerialFactor(ncclComm_t comm, size_t peerBytes) {
  const int64_t configuredFactor = pcclParamP2pSwitchSerialFactor();
  if (configuredFactor == -2) {
    const P2pSwitchAutoConfig* config = getP2pSwitchAutoConfig(comm, peerBytes);
    return config == nullptr || config->serialFactor == -1 ?
        8 : static_cast<size_t>(config->serialFactor);
  }
  return static_cast<size_t>(std::max<int64_t>(1, configuredFactor));
}

static int getP2pSwitchMaxWorkPerBatch(ncclComm_t comm, ssize_t maxPeerBytes) {
  const int64_t configuredMaxWorks = pcclParamP2pSwitchMaxWorkPerBatch();
  if (configuredMaxWorks == -2) {
    const P2pSwitchAutoConfig* config = maxPeerBytes < 0 ? nullptr :
        getP2pSwitchAutoConfig(comm, static_cast<size_t>(maxPeerBytes));
    return config == nullptr || config->maxWorkPerBatch == -1 ?
        1 : config->maxWorkPerBatch;
  }
  return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(
      NCCL_MAX_DEV_WORK_P2P_PER_BATCH, configuredMaxWorks)));
}

static size_t getP2pChunkSize(ncclComm_t comm, size_t totalSize, int minChannels, int maxChannels, size_t minSize, size_t maxSize, int *nChannelsPtr) {
  size_t size = std::max(minSize, DIVUP(totalSize, minChannels));
  int nChannels = minChannels;
  while (size > maxSize && nChannels <= maxChannels/2) {
    nChannels *= 2;
    size = DIVUP(totalSize, nChannels);
  }
  ALIGN_SIZE(size, minSize);
  if (pcclEnableGlobalIcnP2pThroughSwitchOpt(comm)) {
    // switch topo: try to use all channels per peer for large size greater than threshold (per channel)
    size_t serialFactor = comm->nRanks > 8 ? getP2pSwitchSerialFactor(comm, totalSize) : 1;
    size_t thresholdSize = pcclParamP2pSwitchSerialThreshold() * comm->nRanks * serialFactor;
    size_t tileSize = DIVUP(totalSize, *nChannelsPtr);
    if (tileSize >= thresholdSize && nChannels < (*nChannelsPtr)) {
      ALIGN_SIZE(tileSize, thresholdSize);
      ALIGN_SIZE(tileSize, minSize);
      int nChannelsTile = DIVUP(totalSize, tileSize);
      if (nChannelsTile == (*nChannelsPtr)) {
        nChannels = nChannelsTile;
        size = tileSize;
      }
    }
  }
  *nChannelsPtr = nChannels;
  return size;
}

static ncclResult_t scheduleP2pTasksToPlan(ncclComm_t comm, int* p2pEpoch, int* p2pRound,
                                           bool pureAlltoallGroup,
                                           struct ncclKernelPlan* plan, struct ncclKernelPlanBudget* budget) {
  struct ncclKernelPlanner* planner = &comm->planner;
  int nRanks = comm->nRanks;
  struct ncclKernelPlanner::Peer* peers = planner->peers;

  int protocol;
  NCCLCHECK(ncclComputeP2pProto(comm, &protocol));
  int funcIndex = ncclComputeP2pFuncIndex(protocol);

  // Compute how much to split operations
  // Natural step size matching buffer steps.
  ssize_t stepSize = comm->buffSizes[protocol] / NCCL_STEPS;
  // Try to use all channels
  int nChannelsMax = comm->p2pnChannelsPerPeer;
  int nChannelsMin = nChannelsMax;
  // Try to use all channels, but one channel per operation.
  while (nChannelsMin*comm->nRanks > comm->p2pnChannels && nChannelsMin > 1) nChannelsMin /= 2;
  // Avoid overloading channels with 8+ operations as we loose the sync warp, hence a bit of bandwidth.
  while (nChannelsMax*comm->nRanks > comm->p2pnChannels*4 && nChannelsMax > 1) nChannelsMax /= 2;

  // Traverse the pre-computed p2p schedule. p2pRound (0..nRanks-1) and p2pEpoch
  // persist across plans via pointer args, so after a budget-triggered split the
  // next plan resumes exactly where we left off. The schedule is swept repeatedly
  // (p2pRound resets to 0, p2pEpoch increments) until all tasks are drained,
  // supporting queue depth > 1 (multiple send/recv to the same peer).
  while (planner->nTasksP2p != 0) {
    for (; *p2pRound < nRanks; (*p2pRound)++) {
      int sendRank = comm->p2pSchedule[*p2pRound].sendRank;
      int recvRank = comm->p2pSchedule[*p2pRound].recvRank;
      struct ncclTaskP2p* send = ncclIntruQueueHead(&peers[sendRank].sendQueue);
      struct ncclTaskP2p* recv = ncclIntruQueueHead(&peers[recvRank].recvQueue);
      if (send == NULL && recv == NULL) continue;

      if (sendRank == comm->rank) {
        if (send != NULL && recv == NULL) {
          WARN("Trying to send to self without a matching recv");
          return ncclInvalidUsage;
        }
        if (send == NULL && recv != NULL) {
          WARN("Trying to recv to self without a matching send");
          return ncclInvalidUsage;
        }
      }
      ssize_t totSendBytes = -1, totRecvBytes = -1;
      if (send != NULL) totSendBytes = send->bytes;
      if (recv != NULL) totRecvBytes = recv->bytes;
      int nChannelsSend = comm->p2pnChannelsPerPeer, nChannelsRecv = comm->p2pnChannelsPerPeer;
      ssize_t sendChunkSize = getP2pChunkSize(comm, totSendBytes, nChannelsMin, nChannelsMax, stepSize, SENDRECV_SLICEFACTOR*stepSize, &nChannelsSend);
      ssize_t recvChunkSize = getP2pChunkSize(comm, totRecvBytes, nChannelsMin, nChannelsMax, stepSize, SENDRECV_SLICEFACTOR*stepSize, &nChannelsRecv);
      // Ensure room for worst case of one new batch per channel.
      int maxNBatches = 0;
      if (send && (nChannelsSend > maxNBatches)) {
        maxNBatches = nChannelsSend;
      }
      if (recv && (nChannelsRecv > maxNBatches)) {
        maxNBatches = nChannelsRecv;
      }
      if (!testBudget(budget, plan->nWorkBatches+maxNBatches, plan->workBytes + maxNBatches * sizeof(struct ncclDevWorkP2p))) {
        return ncclSuccess;
      }
      ssize_t sendOffset = 0;
      ssize_t recvOffset = 0;
      int sendRemaining = 1, recvRemaining = 1;
      int chunk = 0;
      do {
        int channelId = pcclP2pChannelId(comm, *p2pRound, chunk);
        ssize_t sendBytes = totSendBytes - sendOffset;
        ssize_t recvBytes = totRecvBytes - recvOffset;
        if (sendBytes > sendChunkSize) { sendBytes = sendChunkSize; } else { sendRemaining = 0; }
        if (recvBytes > recvChunkSize) { recvBytes = recvChunkSize; } else { recvRemaining = 0; }
        // 0-bytes send/recv are considered as syncs. Make sure we only add syncs when requested
        // (total size == 0), otherwise set size to -1 so that the kernel skips the operation.
        if (sendBytes == 0 && totSendBytes != 0) sendBytes = -1;
        if (recvBytes == 0 && totRecvBytes != 0) recvBytes = -1;
        if (sendBytes >= 0 || recvBytes >= 0) {
          struct ncclTaskP2p* p2pTasks[2] = { recv, send };
          int maxP2pWorkPerBatch = NCCL_MAX_DEV_WORK_P2P_PER_BATCH;
          if (pcclEnableGlobalIcnP2pThroughSwitchOpt(comm) &&
              (nChannelsRecv == comm->p2pnChannelsPerPeer || nChannelsSend == comm->p2pnChannelsPerPeer)) {
            // Only native AllToAll has communicator-wide symmetric peer sizes.
            maxP2pWorkPerBatch = pureAlltoallGroup ?
                getP2pSwitchMaxWorkPerBatch(comm, std::max(totSendBytes, totRecvBytes)) : 1;
          }
          NCCLCHECK(addP2pToPlan(comm, plan, protocol, funcIndex, *p2pRound, *p2pEpoch, channelId,
                sendRank, send ? ((char*)(send->buff)) + sendOffset : NULL, sendBytes,
                recvRank, recv ? ((char*)(recv->buff)) + recvOffset : NULL, recvBytes,
                totSendBytes, totRecvBytes,
                p2pTasks, maxP2pWorkPerBatch));
        }
        sendOffset += sendChunkSize;
        recvOffset += recvChunkSize;
        chunk++;
      } while (sendRemaining || recvRemaining);
      if (send) {
        // Profiler - We can overwrite groupAPI event handles here since all operations here belong to the same group
        plan->groupApiEventHandle = send->groupApiEventHandle;
        ncclIntruQueueDequeue(&peers[sendRank].sendQueue);
        ncclIntruQueueEnqueue(&plan->p2pTaskQueue, send);
        planner->nTasksP2p -= 1;
      }
      if (recv) {
        // Profiler - We can overwrite groupAPI event handles here since all operations here belong to the same group
        plan->groupApiEventHandle = recv->groupApiEventHandle;
        ncclIntruQueueDequeue(&peers[recvRank].recvQueue);
        ncclIntruQueueEnqueue(&plan->p2pTaskQueue, recv);
        planner->nTasksP2p -= 1;
      }
    }
    *p2pRound = 0;
    (*p2pEpoch)++;
  }
  return ncclSuccess;
}

static ncclResult_t scheduleCollTasksToPlan(ncclComm_t comm, struct ncclKernelPlan* plan, struct ncclKernelPlanBudget* budget) {
  struct ncclKernelPlanner* planner = &comm->planner;
  // Aggregation
  size_t channelSize = NCCL_AGG_CHANNEL_SIZE * comm->nRanks;  // scale channel size based on nranks as latency increases
  // Reduce the per-channel size if we cannot fully utilize the channels
  while (planner->collBytesTotal < channelSize * comm->nChannels && channelSize > NCCL_MIN_CHANNEL_SIZE) channelSize /= 2;
  int channelUsed = 0;
  while (!ncclIntruQueueEmpty(&planner->collTaskQueue)) {
    struct ncclTaskColl* head = ncclIntruQueueHead(&planner->collTaskQueue);
    struct ncclWorkList* workNode = ncclIntruQueueHead(&planner->collWorkQueue);
    struct ncclDevWorkColl* devWork = (struct ncclDevWorkColl*)(workNode+1);
    struct ncclInfo info = {};
    info.comm = comm;
    info.coll = head->func;
    info.sendbuff = head->sendbuff;
    info.recvbuff = head->recvbuff;
    info.count = head->count;
    info.root = head->root;
    info.datatype = head->datatype;
    info.op = head->opHost;
    info.opFull = head->opDev;
    info.chunkSteps = head->chunkSteps;
    info.sliceSteps = head->sliceSteps;
    info.nThreads = head->nThreads;
    info.nChannels = head->nChannels;
    info.algorithm = head->algorithm;
    info.protocol = head->protocol;
    info.nBytes = head->nBytes;

    if (planner->nTasksColl > 1 || channelUsed > 0) {
      info.nChannels = std::min(std::max(1, (int)DIVUP(head->nBytes, channelSize)), comm->nChannels); // assign number of channels
      channelUsed += info.nChannels;
    }
    if (!plan->usingExtKernel) {
      // Ensure room for worst case of one new batch per channel
      if (!testBudget(budget, plan->nWorkBatches + info.nChannels, plan->workBytes + workNode->size)) {
        return ncclSuccess;
      }

      NCCLCHECK(ncclUserBuffAlignmentCheck(&info));

      struct ncclProxyOp proxyOp = {};
      info.funcIndex = FUNC_INDEX(info.coll, info.opFull.op, info.datatype, info.algorithm, info.protocol);
      NCCLCHECK(computeColl(&info, devWork, &proxyOp));

      int nChannels = devWork->nChannels;
      int nSubChannels = (proxyOp.pattern == ncclPatternCollTreeUp || proxyOp.pattern == ncclPatternCollTreeDown) ? 2 : 1;

      uint64_t opCount = uint64_t(plan->collOpCount++)<<1 | 0;

      devWork->startChannelId = comm->lastChannel % comm->nChannels;
      devWork->commNChannels = comm->nChannels;
      if (nChannels*nSubChannels > comm->nChannels) {
        ERROR("nChannels(%d)*nSubChannels(%d) is greater than comm->nChannels(%d)", nChannels, nSubChannels, comm->nChannels);
        return ncclInternalError;
      }
      for (int bid=0; bid<nChannels*nSubChannels; bid++) {
        int channelId = comm->lastChannel % comm->nChannels;
        // Proxy
        proxyOp.task.coll = head;
        proxyOp.coll = info.coll;
        proxyOp.rank = comm->rank;
        proxyOp.eActivationMask = head->eActivationMask;
        proxyOp.incWorkCounter = true;
        proxyOp.channelId = channelId;
        // Adjust pattern for CollNet based on channel index
        if (nSubChannels == 2) {
          proxyOp.pattern = (channelId < comm->nChannels/nSubChannels) ? ncclPatternCollTreeUp : ncclPatternCollTreeDown;
        }

        if (proxyOp.nsteps) {
          proxyOp.opCount = opCount;
          NCCLCHECK(ncclAddProxyOpIfNeeded(comm, &proxyOp));
          NCCLCHECK(addProfilerProxyOpIfNeeded(comm, &proxyOp));
        }

        addWorkBatchToPlan(comm, plan, channelId, workNode->workType, info.funcIndex, plan->workBytes);
        plan->channelWorks[channelId]++;

        comm->lastChannel++;
        plan->channelMask |= uint64_t(1)<<channelId;
      }

      plan->threadPerBlock = std::max<unsigned>(plan->threadPerBlock, info.nThreads);

    #ifdef KERNEL_LOOPBACK
      if (comm->nRanks == 1) {
        // above compute will got zero block when nRanks = 1.
        plan->channelMask = 1;
      }
    #endif

      // Inline the first kernel
      if (plan->kernelFn == NULL) {
        plan->kernelFn = ncclKerns[info.funcIndex];
        plan->funcIndex = info.funcIndex;
      }
    }

    // Profiler
    plan->groupApiEventHandle = head->groupApiEventHandle;

    for (int i=0; i < head->nCleanupQueueElts; i++) {
      ncclIntruQueueEnqueue(&plan->cleanupQueue, ncclIntruQueueDequeue(&planner->collCleanupQueue));
    }

    planner->collBytesTotal -= info.nBytes;
    ncclIntruQueueDequeue(&planner->collTaskQueue);
    ncclIntruQueueDequeue(&planner->collWorkQueue);

    planner->nTasksColl -= 1;
    ncclIntruQueueEnqueue(&plan->collTaskQueue, head);
    ncclIntruQueueEnqueue(&plan->workQueue, workNode);
    plan->workBytes += workNode->size;
  }

  comm->lastChannel = 0;
  return ncclSuccess;
}

static ncclResult_t finishPlan(ncclComm_t comm, struct ncclKernelPlan* plan) {
  ncclKernelPlanner::WipPlan::Channel* wipChannels = comm->planner.wipPlan.channels;
  size_t workBytes = plan->workBytes;
  size_t batchBytes = plan->nWorkBatches*sizeof(struct ncclDevWorkBatch);

  if (plan->isSymColl) return ncclSuccess;
  plan->threadPerBlock = std::max(plan->threadPerBlock, NCCL_MIN_NTHREADS);

  // Debug: dump the sizes that decide whether we promote to Args mode, and
  // guard against the batch table alone overflowing the fixed kernel-args
  // budget (device-side loadWorkBatchToShmem indexes batches without any
  // bounds check, so a too-large table would silently corrupt following
  // args/work bytes).
  constexpr size_t kMaxKernelArgsBytes = ncclMaxKernelArgsSize(); // 1<<10 = 1024
  VERBOSE(NCCL_COLL,
       "finishPlan sizes: sizeof(pcclDevKernelArgs)=%zu batchBytes=%zu workBytes=%zu "
       "nWorkBatches=%d workArgsBytes=%zu kernelArgsBudget=%zu",
       sizeof(ncclDevKernelArgs), batchBytes, workBytes, plan->nWorkBatches,
       (size_t)comm->workArgsBytes, kMaxKernelArgsBytes);
  if (sizeof(ncclDevKernelArgs) + batchBytes > kMaxKernelArgsBytes) {
    ERROR("finishPlan: sizeof(pcclDevKernelArgs)(%zu) + batchBytes(%zu) = %zu exceeds "
          "the %zu-byte kernel-args budget (nWorkBatches=%d). "
          "Device-side batch indexing would OOB; aborting launch.",
          sizeof(ncclDevKernelArgs), batchBytes,
          sizeof(ncclDevKernelArgs) + batchBytes, kMaxKernelArgsBytes,
          plan->nWorkBatches);
    return ncclInternalError;
  }

  // If we can fit everything into the kernel args we do so.
  if (sizeof(ncclDevKernelArgs) + batchBytes + workBytes <= comm->workArgsBytes) {
    plan->workStorageType = ncclDevWorkStorageTypeArgs;
  }

  plan->kernelArgsSize = sizeof(struct ncclDevKernelArgs) + batchBytes;
  plan->kernelArgsSize += (plan->workStorageType == ncclDevWorkStorageTypeArgs) ? workBytes : 0;
  plan->kernelArgsSize = alignUp(plan->kernelArgsSize, 16);
  plan->kernelArgs = (struct ncclDevKernelArgs*)ncclMemoryStackAlloc(&comm->memScoped, plan->kernelArgsSize, /*align=*/16);
  plan->kernelArgs->comm = comm->devComm;
  plan->kernelArgs->channelMask = plan->channelMask;
  plan->kernelArgs->workStorageType = plan->workStorageType;
  plan->kernelArgs->persistent = plan->persistent;
  INFO(NCCL_COLL, "workStorageType %d, persistent %d", plan->workStorageType, plan->persistent);

  // Put batches into the kernel arguments. The first batch for each channel
  // must be located at batchZero[blockIdx.x]. To achieve this we round robin
  // over the channels in ascending order until they're exhausted.
  uint64_t hasBatchMask = plan->channelMask;
  struct ncclDevWorkBatch* batchPrev[MAXCHANNELS] = {}; // {0...}
  struct ncclDevWorkBatch* batchZero = (struct ncclDevWorkBatch*)(plan->kernelArgs+1);
  int batchIx = 0;
  while (hasBatchMask != 0) {
    uint64_t tmpMask = hasBatchMask; // channels with a batch for this round.
    do {
      int c = popFirstOneBit(&tmpMask);
      if (!ncclIntruQueueEmpty(&wipChannels[c].workBatchQueue)) {
        struct ncclWorkBatchList* batchNode = ncclIntruQueueDequeue(&wipChannels[c].workBatchQueue);
        if (batchPrev[c] != nullptr) {
          batchPrev[c]->nextJump = int(&batchZero[batchIx] - batchPrev[c]);
        }
        batchPrev[c] = &batchZero[batchIx];
        batchZero[batchIx++] = batchNode->batch;
      }
      if (ncclIntruQueueEmpty(&wipChannels[c].workBatchQueue)) {
        hasBatchMask ^= 1ull<<c;
      }
    } while (tmpMask != 0);
  }

  // Merge-sort per-channel proxy-op lists by opCount when merging them into plan->proxyOpQueue
  // Phase 1: scan first op of each channel, store opCount in headIds[c].
  uint64_t headIds[MAXCHANNELS];
  int nHeads = 0;
  int channelUbound = 0;
  for (int c=0; c < MAXCHANNELS; c++) {
    struct ncclProxyOp* op = ncclIntruQueueHead(&wipChannels[c].proxyOpQueue);
    headIds[c] = op ? op->opCount : uint64_t(-1);
    if (op) nHeads += 1;
    if (op) plan->hasProxyOps = true;
    if (op) channelUbound = c+1;
  }
  // Phase 2: Dequeue from planner->channels[c], enqueue in merged order to plan
  while (nHeads != 0) {
    int c = -1;
    uint64_t minId = uint64_t(-1);
    // Find channel with least proxy-op id. We store the heads[c]->opCount in
    // headIds[c] to remove indirect loads from this loop.
    for (int c1=0; c1 < channelUbound; c1++) {
      uint64_t id = headIds[c1];
      id = (id>>1 | id<<63); // Move tag bit to order collectives before p2p's
      if (id < minId) { c = c1; minId = id; }
    }
    struct ncclProxyOp* op = ncclIntruQueueDequeue(&wipChannels[c].proxyOpQueue);
    struct ncclProxyOp* opNext = ncclIntruQueueHead(&wipChannels[c].proxyOpQueue);
    headIds[c] = opNext ? opNext->opCount : uint64_t(-1);
    nHeads -= opNext ? 0 : 1;
    ncclIntruQueueEnqueue(&plan->proxyOpQueue, op);
  }
  return ncclSuccess;
}

PCCL_PARAM(SymOpAccPrecisionInc, "SYM_OP_ACC_PRECISION_INC", 0);
ncclResult_t ncclLaunchPrepare(ncclComm_t comm) {
  ncclResult_t result = ncclSuccess;
  struct ncclKernelPlanner* planner = &comm->planner;
  bool persistent = ncclCudaGraphValid(planner->capturingGraph);
  planner->persistent = persistent;
  planner->uploadKernelSeq = 0;
  int nPlans = 0;
  hggcStream_t launchStream = planner->streams->stream;
  struct ncclKernelPlan* planHead = nullptr;

  if (planner->nTasksColl + planner->nTasksP2p != 0 ||
      !ncclIntruQueueEmpty(&planner->collSymTaskQueue) ||
      !ncclIntruQueueEmpty(&planner->collCeTaskQueue)) {
    // p2pEpoch and p2pRound persist across plans within one launch (passed by
    // pointer to scheduleP2pTasksToPlan), so after a budget-triggered split the
    // next plan resumes from where we left off.
    int p2pEpoch = 0, p2pRound = 0;
    // All p2p tasks came from Alltoall, so every rank picks the same batch limit.
    bool pureAlltoallGroup = planner->nTasksP2p == planner->nTasksP2pAlltoall;
    do {
      struct ncclKernelPlan* plan = ncclMemoryPoolAlloc<struct ncclKernelPlan>(&comm->memPool_ncclKernelPlan, &comm->memPermanent);
      plan->comm = comm;
      plan->reclaimer.fn = reclaimPlan;
      plan->persistent = persistent;
      // finishPlan() promotes ncclDevWorkStorageType[Fifo|Persistent]->Args if the work can fit.
      plan->workStorageType = persistent ? ncclDevWorkStorageTypePersistent
                                         : ncclDevWorkStorageTypeFifo;

      if (!ncclIntruQueueEmpty(&planner->collCeTaskQueue)) {
        struct ncclTaskColl* task = ncclIntruQueueHead(&planner->collCeTaskQueue);
        plan->isCeColl = true;
        plan->ceCollArgs.rootRank = task->root;
        plan->ceCollArgs.nElts = task->count;
        plan->ceCollArgs.eltSize = ncclTypeSize(task->datatype);
        plan->ceCollArgs.datatype = task->datatype;
        // opHost, not opDev: opDev rewrites ncclAvg into PreMulSum/SumPostDiv for the generic
        // kernels, while the CE reduce path implements avg itself by pre-scaling.
        plan->ceCollArgs.redOp = task->opHost;
        plan->ceCollArgs.sendBuff = (uint8_t*)task->sendbuff;
        plan->ceCollArgs.recvBuff = (uint8_t*)task->recvbuff;
        plan->ceCollArgs.func = task->func;
        plan->ceCollArgs.comm = comm->ceColl.symDevComm;
        plan->ceCollArgs.nBlocks = 1;
        plan->ceCollArgs.collApiEventHandle = task->collApiEventHandle;
        plan->ceCollArgs.sendWin = task->sendWin;
        plan->ceCollArgs.recvWin = task->recvWin;

        ncclIntruQueueEnqueue(&planner->planQueue, plan);
        ncclIntruQueueDequeue(&planner->collCeTaskQueue);
        ncclMemoryPoolFree(&comm->memPool_ncclTaskColl, task);
        nPlans += 1;
      } else if (planner->nTasksP2p != 0 && ncclIntruQueueEmpty(&planner->collSymTaskQueue)
                 && !persistent && pcclP2pDmaEligible(comm)) {
        // A DMA-carried p2p batch is its own plan kind, isolated from the
        // generic kernel-arg/budget machinery below, the same way isCeColl
        // is: pcclP2pDmaBuildPlan fills in everything the plan needs
        // (including dequeuing the tasks it consumes) and we enqueue as-is.
        // Any p2p left over (more than pcclP2pDmaOpsPerPeer per peer) stays
        // queued, so the outer do-while keeps building batches.
        // !persistent: the mailbox seq numbers and host CE exchange are
        // captured by value, so a replayed graph would silently reuse stale
        // handshakes -- under capture we fall through to the kernel p2p path.
        NCCLCHECKGOTO(pcclP2pDmaBuildPlan(comm, plan), result, failure);
        ncclIntruQueueEnqueue(&planner->planQueue, plan);
        nPlans += 1;
      } else {
        if (!ncclIntruQueueEmpty(&planner->collSymTaskQueue)) {
          NCCLCHECKGOTO(ncclSymmetricTaskScheduler(comm, &planner->collSymTaskQueue, plan), result, failure);
        }
        else {
          if (planner->isExtKernel) {
            plan->usingExtKernel = true;
          }

          struct ncclKernelPlanBudget budget;
          budget.inArgsBytes = comm->workArgsBytes - sizeof(struct ncclDevKernelArgs);
          // Non-persistent kernels fill up at most half of our fifo per kernel.
          budget.outArgsBytes = plan->persistent ? (1<<30) : comm->workFifoBytes/2;

          if (planner->nTasksP2p != 0) {
            NCCLCHECKGOTO(scheduleP2pTasksToPlan(comm, &p2pEpoch, &p2pRound, pureAlltoallGroup, plan, &budget), result, failure);
          }

          if (planner->nTasksP2p == 0 && planner->nTasksColl != 0) {
            NCCLCHECKGOTO(scheduleCollTasksToPlan(comm, plan, &budget), result, failure);
          }
        }

        NCCLCHECKGOTO(finishPlan(comm, plan), result, failure);
        if (plan->workBytes != 0) {
          ncclIntruQueueEnqueue(&planner->planQueue, plan);
          nPlans += 1;
        }
      }
    } while ((planner->nTasksColl + planner->nTasksP2p != 0) ||
             !ncclIntruQueueEmpty(&planner->collSymTaskQueue) ||
             !ncclIntruQueueEmpty(&planner->collCeTaskQueue));
  }

  planHead = ncclIntruQueueHead(&planner->planQueue);
  planner->unlaunchedPlansHead = planHead;


  // Semantically we want these dependencies for the kernels launched:
  //   1. Launch host task on hostStream.
  //   2. Launch kernel, depends on all of {deviceStream, hostStream, userStream[i]...}
  //   3. {deviceStream, userStream[i]...} depend on kernel.
  // We achieve this by:
  //   1. deviceStream waits on each of userStream[1...]
  //   2. userStream[0] waits on deviceStream
  //   3. host task launch on hostStream
  //   4. userStream[0] waits on hostStream
  //   5. kernel launch on userStream[0]
  //   6. deviceStream waits on userStream[0]
  //   7. userStream[1...] each waits on deviceStream
  // The two-level fan-in fan-out is because ncclStrongStreamWaitStream() requires
  // at least one of the two streams to be strong-stream.

  NCCLCHECKGOTO(ncclStrongStreamAcquire(planner->capturingGraph, &comm->sharedRes->deviceStream), result, failure);

  // Create dependency for device stream on user streams. First from extra user
  // streams to deviceStream. Then deviceStream to first user stream.
  for (struct ncclCudaStreamList* l = planner->streams->next; l != nullptr; l = l->next) {
    NCCLCHECKGOTO(ncclStrongStreamWaitStream(planner->capturingGraph, &comm->sharedRes->deviceStream, l->stream), result, failure);
  }

  NCCLCHECKGOTO(ncclStrongStreamWaitStream(planner->capturingGraph, launchStream, &comm->sharedRes->deviceStream), result, failure);

  if (persistent || ncclCudaLaunchBlocking || comm->sharedRes->persistentRefs != 0) {
    // We have to launch host tasks to push proxy args. We are careful to only
    // do this if necessary since host tasks impose a high performance cost in HGGC.

    for (struct ncclKernelPlan* plan=planHead; plan != nullptr; plan = plan->next) {
      if (plan->hasProxyOps) {
        NCCLCHECKGOTO(ncclStrongStreamAcquire(planner->capturingGraph, &comm->sharedRes->hostStream), result, failure);
        plan->isHostCbEnq = true;
        NCCLCHECKGOTO(ncclStrongStreamLaunchHost(planner->capturingGraph, &comm->sharedRes->hostStream, hostStreamPlanCallback, plan), result, failure);
        // Make to-be-launched kernels dependent on just-launched host stream tasks.
        NCCLCHECKGOTO(ncclStrongStreamWaitStream(planner->capturingGraph, launchStream, &comm->sharedRes->hostStream), result, failure);
        NCCLCHECKGOTO(ncclStrongStreamRelease(planner->capturingGraph, &comm->sharedRes->hostStream), result, failure);
      }
    }
  }

  if (persistent) {
    comm->sharedRes->persistentRefs += nPlans;
    comm->localPersistentRefs += nPlans;
    NCCLCHECKGOTO(ncclCudaGraphAddDestructor(planner->capturingGraph, persistentDestructor, (void*)planHead), result, failure);
  }

failure:
  return result;
}

const char* opInfoFormatStr = "pccl:%s, p:%d, c:%zi, d:%d, r:%d, w:%d, h:%lx, t:%d, b:%p/%p.";
ncclResult_t constructHgtxInfo(struct ncclInfo* info) {
  // hgtxMessage buffer size should be equal or bigger than MAX_HGTX_MSG_LENGTH + ONE_HGTX_MSG_LENGTH(max one op message size).
  if (info->comm->hgtxMessageIndex < MAX_HGTX_MSG_LENGTH) {
    int opHgtxMessageSize = snprintf(
    info->comm->hgtxMessage+info->comm->hgtxMessageIndex, MAX_HGTX_MSG_LENGTH-info->comm->hgtxMessageIndex,
      opInfoFormatStr, info->opName, info->root, info->count, info->datatype, info->comm->rank,
      info->comm->nRanks, info->comm->commHash, info->comm->maxConnType, info->sendbuff, info->recvbuff);
    info->comm->hgtxMessageIndex += opHgtxMessageSize;
    if (opHgtxMessageSize > ONE_HGTX_MSG_LENGTH) {
      WARN("one op's hgtx info size(%d) excced pre defined max message size(%d)", opHgtxMessageSize, ONE_HGTX_MSG_LENGTH);
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclPlannerSetCapturingGraph(ncclComm_t comm, struct ncclInfo* info) {
  struct ncclKernelPlanner *planner = &comm->planner;
  if (info->stream != planner->streamRecent || planner->streams == nullptr) {
    planner->streamRecent = info->stream;
    struct ncclCudaStreamList* l = planner->streams;
    while (true) {
      if (l == nullptr) { // Got to the end, this must be a new stream.
        struct ncclCudaGraph graph;
        NCCLCHECK(ncclCudaGetCapturingGraph(&graph, info->stream));
        if (planner->streams != nullptr && !ncclCudaGraphSame(planner->capturingGraph, graph)) {
          ERROR("Streams given to a communicator within a PCCL group must either be all uncaptured or all captured by the same graph.");
          return ncclInvalidUsage;
        }
        planner->capturingGraph = graph; // C++ struct assignment
        // Add stream to list
        l = ncclMemoryStackAlloc<struct ncclCudaStreamList>(&comm->memScoped);
        l->stream = info->stream;
        l->next = planner->streams;
        planner->streams = l;
        break;
      }
      if (l->stream == info->stream)
        break; // Already seen stream.
      l = l->next;
    }
  }
  return ncclSuccess;
}

static ncclResult_t p2pTaskAppend(
    ncclComm_t comm,
    struct ncclInfo* info,
    ncclFunc_t coll, // for alltoall/scatter/gather, extra info needed belong info.
    ncclFunc_t collAPI, // reserved, used for determine net strategy(algo/proto) in net.cc for (coll vs p2p[sendrecv/alltoall/scatter/gather])
    void* buff,
    size_t count,
    ncclDataType_t datatype,
    int peer) {
  struct ncclKernelPlanner *planner = &comm->planner;

  // Determine peer and basic parameters.
  ssize_t nBytes = count * ncclTypeSize(datatype);
  bool isSendNotRecv = coll == ncclFuncSend;

  // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`.
  ncclGroupCommJoin(comm, ncclGroupTaskTypeCollective);
  // Set capturing graph. Called here so that profiler can emit a group API event with this information
  NCCLCHECK(ncclPlannerSetCapturingGraph(comm, info));
  bool isGraphCaptured = ncclCudaGraphValid(planner->capturingGraph);
  NCCLCHECK(ncclProfilerStartGroupApiEvent(info, isGraphCaptured));
  NCCLCHECK(ncclProfilerRecordGroupApiEventState(ncclProfilerGroupStartApiStop));

  NCCLCHECK(ncclProfilerStartP2pApiEvent(info, isGraphCaptured));

  struct ncclTaskP2p* p2p = ncclMemoryPoolAlloc<struct ncclTaskP2p>(&comm->memPool_ncclTaskP2p, &comm->memPermanent);
  p2p->func = coll;
  p2p->collAPI = collAPI;
  p2p->buff = buff;
  p2p->count = count;
  p2p->datatype = datatype;
  p2p->root = peer;
  p2p->bytes = nBytes;
  p2p->eActivationMask = ncclProfilerApiState.eActivationMask;
  p2p->groupApiEventHandle = ncclProfilerApiState.groupApiEventHandle;
  p2p->p2pApiEventHandle = ncclProfilerApiState.p2pApiEventHandle;
  p2p->nChannels = comm->p2pnChannels;
  ncclIntruQueueEnqueue(
    isSendNotRecv ? &planner->peers[peer].sendQueue : &planner->peers[peer].recvQueue,
    p2p);
  planner->nTasksP2p += 1;
  if (collAPI == ncclFuncAlltoAll) planner->nTasksP2pAlltoall += 1;

  // Mark channels that need pre-connect
  if (comm->rank != peer) {
    int round = 0;
    while (peer != (isSendNotRecv ? comm->p2pSchedule[round].sendRank
                                  : comm->p2pSchedule[round].recvRank)) {
      round += 1;
    }
    for (int c = 0; c < comm->p2pnChannelsPerPeer; c++) {
      int channelId = pcclP2pChannelId(comm, round, c);
      if (isSendNotRecv) {
        if (comm->channels[channelId].peers[peer]->send[1].connected == 0) {
          comm->connectSend[peer] |= (1UL<<channelId);
          ncclGroupCommPreconnect(comm);
        }
      }
      else {
        if (comm->channels[channelId].peers[peer]->recv[1].connected == 0) {
          comm->connectRecv[peer] |= (1UL<<channelId);
          ncclGroupCommPreconnect(comm);
        }
      }
    }
  }
  ncclProfilerStopP2pApiEvent();
  // Mark channels that need pre-connect
  return ncclSuccess;
}

static ncclResult_t collTaskAppend(
    ncclComm_t comm,
    struct ncclInfo* info,
    struct ncclDevRedOpFull opDev) {
  struct ncclKernelPlanner *planner = &comm->planner;

  // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`.
  ncclGroupCommJoin(info->comm, ncclGroupTaskTypeCollective);
  // Set capturing graph. Called here so that profiler can emit a group API event with this information
  NCCLCHECK(ncclPlannerSetCapturingGraph(comm, info));
  bool isGraphCaptured = ncclCudaGraphValid(planner->capturingGraph);
  NCCLCHECK(ncclProfilerStartGroupApiEvent(info, isGraphCaptured));
  NCCLCHECK(ncclProfilerRecordGroupApiEventState(ncclProfilerGroupStartApiStop));
  NCCLCHECK(ncclProfilerStartCollApiEvent(info, isGraphCaptured));

  struct ncclTaskColl* t = ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, &comm->memPermanent);
  t->func = info->coll;
  t->sendbuff = info->sendbuff;
  t->recvbuff = info->recvbuff;
  t->count = info->count;
  t->root = info->root;
  t->datatype = info->datatype;
  t->opHost = info->op;
  t->opDev = opDev; // C++ struct assignment
  t->nBytes = info->nBytes;
  t->chunkSteps = info->chunkSteps;
  t->sliceSteps = info->sliceSteps;
  t->eActivationMask = ncclProfilerApiState.eActivationMask;
  t->groupApiEventHandle = ncclProfilerApiState.groupApiEventHandle;
  t->collApiEventHandle = ncclProfilerApiState.collApiEventHandle;

  ncclIntruQueueEnqueue(&planner->collTaskQueue, t);
  planner->collBytesTotal += t->count*ncclTypeSize(t->datatype);
  planner->nTasksColl += 1;

  ncclProfilerStopCollApiEvent();
  return ncclSuccess;
}

static ncclResult_t ceCollTaskAppend(
    ncclComm_t comm,
    struct ncclInfo* info,
    struct ncclDevrWindow* sendWin,
    struct ncclDevrWindow* recvWin,
    struct ncclDevRedOpFull opDev) {
  struct ncclKernelPlanner *planner = &comm->planner;

  // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`.
  ncclGroupCommJoin(info->comm, ncclGroupTaskTypeCollective);
  NCCLCHECK(ncclPlannerSetCapturingGraph(comm, info));
  struct ncclTaskColl* t = ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, &comm->memPermanent);

  t->func = info->coll;
  t->sendbuff = info->sendbuff;
  t->recvbuff = info->recvbuff;
  t->count = info->count;
  t->root = info->root;
  t->datatype = info->datatype;
  t->opHost = info->op;
  t->opDev = opDev; // C++ struct assignment
  t->chunkSteps = info->chunkSteps;
  t->sliceSteps = info->sliceSteps;
  t->eActivationMask = __atomic_load_n(&ncclProfilerEventMask, __ATOMIC_RELAXED);
  t->sendWin = sendWin;
  t->recvWin = recvWin;

  ncclIntruQueueEnqueue(&planner->collCeTaskQueue, t);

  return ncclSuccess;
}

static ncclResult_t hostToDevRedOp(
    ncclDevRedOpFull *opFull, ncclRedOp_t op, ncclDataType_t datatype, ncclComm *comm
  ) {
  union {
    int8_t   i8; uint8_t   u8;
    int32_t i32; uint32_t u32;
    int64_t i64; uint64_t u64;
    __half f16; float f32; double f64;
    __ppu_bfloat16 bf16;
    __hg_fp8_storage_t f8;
    void *ptr;
  };
  u64 = 0;
  opFull->scalarArgIsPtr = false;

  switch (int(op)) {
  case ncclSum:  opFull->op = ncclDevSum;  break;
  case ncclProd: opFull->op = ncclDevProd; break;
  case ncclMax:  opFull->op = ncclDevMax;  break;
  case ncclMin:  opFull->op = ncclDevMin;  break;
  case ncclAvg:
    switch ((int)datatype) {
    case ncclInt8:  case ncclInt32:  case ncclInt64:
    case ncclUint8: case ncclUint32: case ncclUint64:
      opFull->op = ncclDevSumPostDiv;
      u64 = comm->nRanks;
      break;
#if defined(__HGGC_FP8_TYPES_EXIST__)
    case ncclFloat8e4m3:
      opFull->op = ncclDevPreMulSum;
      f8 = __hg_cvt_float_to_fp8(float(1.0/comm->nRanks), __HG_SATFINITE, __HG_E4M3);
      break;
    case ncclFloat8e5m2:
      opFull->op = ncclDevPreMulSum;
      f8 = __hg_cvt_float_to_fp8(float(1.0/comm->nRanks), __HG_SATFINITE, __HG_E5M2);
      break;
#endif
    case ncclFloat16:
      opFull->op = ncclDevPreMulSum;
      f16 = __float2half(float(1.0/comm->nRanks));
      break;
    case ncclBfloat16:
      opFull->op = ncclDevPreMulSum;
      bf16 = __float2bfloat16(float(1.0/comm->nRanks));
      break;
    case ncclFloat32:
      opFull->op = ncclDevPreMulSum;
      f32 = float(1.0/comm->nRanks);
      break;
    case ncclFloat64:
      opFull->op = ncclDevPreMulSum;
      f64 = 1.0/comm->nRanks;
      break;
    }
    opFull->scalarArgIsPtr = false;
    opFull->scalarArg = u64;
    break;
  default: // user created
    int ix = int(ncclUserRedOpMangle(comm, op)) - int(ncclNumOps);
    ncclUserRedOp *user = &comm->userRedOps[ix];
    if (datatype != user->datatype) {
      ERROR("Data type supplied to user-created pcclRedOp_t does not match type "
           "given to reduction operation");
      return ncclInvalidArgument;
    }
    *opFull = user->opFull;
    break;
  }
  return ncclSuccess;
}

// Converts `info` to a task and adds it to `comm->planner`. The exception is with
// single rank communicators, collectives are issued as `ncclMemcpyAsync`s and
// thus don't need a task.
static ncclResult_t taskAppend(ncclComm_t comm, struct ncclInfo* info) {
  ncclFunc_t collAPI = info->coll;

  if (info->coll == ncclFuncSend || info->coll == ncclFuncRecv) { //p2p stored separately
    NCCLCHECK(p2pTaskAppend(comm, info, info->coll, collAPI, (void*)info->recvbuff, info->count, info->datatype, info->root));
  } else {
    if (info->datatype == ncclFloat8e4m3 || info->datatype == ncclFloat8e5m2) {
      if (ncclHggcCompCap() < 89 && info->coll != ncclFuncAllGather && info->coll != ncclFuncBroadcast && info->coll != ncclFuncAlltoAll && info->coll != ncclFuncScatter && info->coll != ncclFuncGather) {
        WARN("FP8 reduction support begins with sm89 capable devices.");
        return ncclInvalidArgument;
      }
    }

    // Copy reduction op state from op handle into info struct here since the
    // op handle may be destroyed before ncclGroupEnd().
    struct ncclDevRedOpFull opDev = {};
    NCCLCHECK(hostToDevRedOp(&opDev, info->op, info->datatype, comm));

#ifndef KERNEL_LOOPBACK
    if (info->comm->nRanks == 1) {
      struct pcclHgtxInfo hgtxKernelInfo = pcclHgtxInfo(comm->hgtxMessage);
      NCCLCHECK(ncclLaunchOneRank(info->recvbuff, info->sendbuff, info->count, opDev, info->datatype, info->stream));
      NCCLCHECK(ncclLaunchReset(info->comm));
      return ncclSuccess;
    }
#endif

    struct ncclDevrWindow* sendWin;
    struct ncclDevrWindow* recvWin;
    ncclDevrFindWindow(comm, info->sendbuff, &sendWin);
    ncclDevrFindWindow(comm, info->recvbuff, &recvWin);
    // Append CE collective task if CE is supported and requested by user
    ncclSymRegType_t winRegType;
    NCCLCHECK(ncclGetSymRegType(sendWin, recvWin, &winRegType));
    bool ceAvailable = ncclCeAvailable(comm, info->coll, info->op, info->datatype, info->count, winRegType, info->recvbuff, info->stream);
    bool hasSysmemSegment = ncclDevrWindowHasSysmemSegment(sendWin) || ncclDevrWindowHasSysmemSegment(recvWin);

    if ((comm->config.CTAPolicy & PCCL_CTA_POLICY_ZERO) && ceAvailable && !hasSysmemSegment) {
      NCCLCHECK(ceCollTaskAppend(comm, info, sendWin, recvWin, opDev));
    } else {
      if (info->coll == ncclFuncAlltoAll) {
        for (int r=0; r<comm->nRanks; r++) {
          NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncSend, collAPI, (void*)((char*)info->sendbuff+r*info->count*ncclTypeSize(info->datatype)), info->count, info->datatype, r));
          NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncRecv, collAPI, (void*)((char*)info->recvbuff+r*info->count*ncclTypeSize(info->datatype)), info->count, info->datatype, r));
        }
      } else if (info->coll == ncclFuncGather){
        size_t offset = 0;
        NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncSend, collAPI, (void*)info->sendbuff, info->count, info->datatype, info->root));
        if (comm->rank == info->root) {
          for (int r=0; r<comm->nRanks; r++) {
            void* buff = (void*)((char*)info->recvbuff + offset);
            NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncRecv, collAPI, buff, info->count, info->datatype, r));
            offset += info->count * ncclTypeSize(info->datatype);
          }
        }
      } else if (info->coll == ncclFuncScatter) {
        size_t offset = 0;
        if (comm->rank == info->root) {
          for (int r = 0; r < comm->nRanks; r++) {
            void* buff = (void*)((char*)info->sendbuff + offset);
            NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncSend, collAPI, buff, info->count, info->datatype, r));
            offset += info->count * ncclTypeSize(info->datatype);
          }
        }
        NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncRecv, collAPI, (void*)info->recvbuff, info->count, info->datatype, info->root));
      } else {
        NCCLCHECK(collTaskAppend(comm, info, opDev));
      }
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclEnqueueCheck(struct ncclInfo* info) {
  NCCLCHECK(ncclGroupStartInternal());
  ncclResult_t ret = ncclSuccess;
  int savedDev = -1;
  // Check arguments
  NCCLCHECK(PtrCheck(info->comm, info->opName, "comm"));
  // Check whether communicator is ready to communicate
  NCCLCHECKGOTO(ncclCommEnsureReady(info->comm), ret, fail);
  if (info->comm->checkPointers) {
    HGGCCHECKGOTO(hggcGetDevice(&savedDev), ret, fail);
    HGGCCHECKGOTO(hggcSetDevice(info->comm->hggcDev), ret, fail);
  }
  // assign info init elems.
  info->nChannels = 0;
  NCCLCHECKGOTO(ArgsCheck(info), ret, fail);

  INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d op %d root %d comm %p [nranks=%d] stream %p rank %d",
      info->opName, info->comm->opCount, info->sendbuff, info->recvbuff, info->count,
      info->datatype, info->op, info->root, info->comm, info->comm->nRanks, info->stream, info->comm->rank);
  NCCLCHECKGOTO(constructHgtxInfo(info), ret, fail);

  NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);

exit:
  if (savedDev != -1) HGGCCHECK(hggcSetDevice(savedDev));
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  /* if depth is 1, ncclGroupEndInternal() will trigger group ops. The state can change
   * so we have to check state here. */
  if (info->comm && !info->comm->config.blocking) { NCCLCHECK(pcclCommGetAsyncError(info->comm, &ret)); };
  return ret;
fail:
  if (info->comm && !info->comm->config.blocking) (void) ncclCommSetAsyncError(info->comm, ret);
  goto exit;
}

// Called once per ncclGroup to organize the user submitted tasks in
// comm->planner so that they can be peeled off into plans.
ncclResult_t ncclPrepareTasks(ncclComm_t comm, bool* algoNeedConnect, bool* needConnect, ncclSimInfo_t* simInfo) {
  struct ncclKernelPlanner* planner = &comm->planner;
  planner->persistent = ncclCudaGraphValid(planner->capturingGraph);
  ncclTaskColl* task = ncclIntruQueueHead(&planner->collTaskQueue);
  planner->isExtKernel = false;

  // Symmetric kernel path
  if (comm->symmetricSupport) {
    struct ncclTaskColl* remainTasksHead = nullptr;
    NCCLCHECK(ncclMakeSymmetricTaskList(comm, &planner->collTaskQueue, &planner->collSymTaskQueue, &remainTasksHead));
    // Re-enqueue non-symk-eligible tasks back into collTaskQueue
    while (remainTasksHead != nullptr) {
      struct ncclTaskColl* next = remainTasksHead->next;
      remainTasksHead->next = nullptr;
      ncclIntruQueueEnqueue(&planner->collTaskQueue, remainTasksHead);
      remainTasksHead = next;
    }
  }

  // If all tasks went to symk path, nothing left for standard scheduling
  if (planner->nTasksColl == 0) return ncclSuccess;

  task = ncclIntruQueueHead(&planner->collTaskQueue);

  if (planner->nTasksColl == 1 && planner->nTasksP2p == 0) {
    if (comm->extKernel != NULL && !simInfo) {
      struct ncclExtTaskColl extTask;
      ncclTaskCollToExtTaskColl(task, &extTask);
      NCCLCHECK(comm->extKernel->kernelPluginAvailCheck(&extTask, comm->extKernelPluginWorkspace.extKernelPluginParams, &task->isExtKernel));
    }
    if (!task->isExtKernel) {
      NCCLCHECK(getAlgoInfo(comm, task, simInfo));
      if (comm->runtimeConn && comm->initAlgoChannels[task->algorithm] == false) {
        comm->initAlgoChannels[task->algorithm] = true;
        algoNeedConnect[task->algorithm] = true;
        *needConnect = true;
      }
    } else {
      if (simInfo) {
        WARN("pcclGroupSimulateEnd() is not currently supported for external kernels.");
      }
    }
    planner->isExtKernel = task->isExtKernel;
    return ncclSuccess;
  }

  while(task) {
    NCCLCHECK(getAlgoInfo(comm, task, simInfo));
    task = task->next;
  }

  task = ncclIntruQueueHead(&planner->collTaskQueue);
  while(task) {
    if (comm->runtimeConn && comm->initAlgoChannels[task->algorithm] == false) {
      comm->initAlgoChannels[task->algorithm] = true;
      algoNeedConnect[task->algorithm] = true;
      *needConnect = true;
    }
    task = task->next;
  }
  return ncclSuccess;
}

NCCL_API(ncclResult_t, pcclRedOpCreatePreMulSum, ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm);
ncclResult_t pcclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm) {
  INFO_CALL("PCCL API: pcclRedOpCreatePreMulSum(op=%d, scalar=%p, datatype=%d, residence=%d, comm=%p)", *op, scalar, datatype, residence, comm);

  NCCLCHECK(CommCheck(comm, "pcclRedOpCreatePreMulSum", "comm"));
  /* join init thread before creating PreMulSum op. */
  NCCLCHECK(ncclCommEnsureReady(comm));

  if (comm->userRedOpFreeHead == comm->userRedOpCapacity) {
    // double capacity and resize
    int cap = 2*comm->userRedOpCapacity;
    if (cap < 4) cap = 4;
    ncclUserRedOp *ops = new ncclUserRedOp[cap];
    if (comm->userRedOpCapacity > 0)
      std::memcpy(ops, comm->userRedOps, comm->userRedOpCapacity*sizeof(ncclUserRedOp));
    for(int ix=comm->userRedOpCapacity; ix < cap; ix++)
      ops[ix].freeNext = ix + 1;
    delete[] comm->userRedOps;
    comm->userRedOps = ops;
    comm->userRedOpCapacity = cap;
  }
  // pop from free list
  int ix = comm->userRedOpFreeHead;
  ncclUserRedOp *user = &comm->userRedOps[ix];
  comm->userRedOpFreeHead = user->freeNext;

  user->freeNext = -1; // allocated
  user->datatype = datatype;
  user->opFull.op = ncclDevPreMulSum;
  if (residence == ncclScalarHostImmediate) {
    int size = ncclTypeSize(datatype);
    if (size < 1) return ncclInternalError;
    user->opFull.scalarArgIsPtr = false;
    std::memcpy(&user->opFull.scalarArg, scalar, size);
  } else {
    user->opFull.scalarArgIsPtr = true;
    user->opFull.scalarArg = reinterpret_cast<uint64_t>(scalar);
  }
  *op = ncclRedOp_t(int(ncclNumOps) + ix);
  *op = ncclUserRedOpMangle(comm, *op);

  return ncclSuccess;
}

NCCL_API(ncclResult_t, pcclRedOpDestroy, ncclRedOp_t op, ncclComm_t comm);
ncclResult_t pcclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  INFO_CALL("PCCL API: pcclRedOpDestroy(op=%d, comm=%p)", op, comm);

  if (0 <= int(op) && int(op) < int(ncclNumOps)) {
    ERROR("pcclRedOpDestroy : operator is a PCCL builtin.");
    return ncclInvalidArgument;
  }
  // int(ncclMaxRedOp) < int(op) will always be false due to the sizes of
  // the datatypes involved, and that's by design.  We keep the check though
  // just as a reminder.
  // coverity[result_independent_of_operands]
  if (int(op) < 0 || int(ncclMaxRedOp) < int(op)) {
    ERROR("pcclRedOpDestroy :  operator is garbage.");
    return ncclInvalidArgument;
  }
  if (comm == NULL) {
    ERROR("pcclRedOpDestroy : invalid communicator passed.");
    return ncclInvalidArgument;
  }

  int ix = int(ncclUserRedOpMangle(comm, op)) - int(ncclNumOps);
  if (comm->userRedOpCapacity <= ix || comm->userRedOps[ix].freeNext != -1) {
    ERROR("pcclRedOpDestroy : operator unknown to this communicator.");
    return ncclInvalidArgument;
  }
  // push to free list
  comm->userRedOps[ix].freeNext = comm->userRedOpFreeHead;
  comm->userRedOpFreeHead = ix;

  return ncclSuccess;
}
