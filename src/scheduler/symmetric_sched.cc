/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
// symmetric_sched.cc: Multi-task symmetric kernel scheduling.

#include "scheduler.h"
#include "enqueue.h"
#include "sym_kernels.h"
#include "dev_runtime.h"
#include "comm.h"
#include "core.h"
#include "param.h"
#include "bitops.h"
#include "profiler.h"
#include "allocator.h"
#include <cstring>

#include <hggc_fp16.h>

NCCL_PARAM(SymNoWinEnable, "SYM_NOWIN_ENABLE", 0);
PCCL_PARAM(SymOpAccPrecisionInc, "SYM_OP_ACC_PRECISION_INC", 0);
PCCL_PARAM(SymkUseMultiTask, "SYMK_USE_MULTI_TASK", 0);

////////////////////////////////////////////////////////////////////////////////
// Helper: Convert ncclRedOp_t to the reduction op used by symmetric kernels

static ncclDevRedOp_t symkRedOp(ncclRedOp_t redOp, ncclDevRedOp_t devRedOp) {
  if (redOp == ncclAvg) {
    return ncclDevSumPostDiv;
  }
  return devRedOp;
}

////////////////////////////////////////////////////////////////////////////////
// Helper: Convert collective task to symmetric task format.
// Adjusts the scalar argument for SumPostDiv (average) operations.

static void convertCollTaskToSymmetricTask(struct pcclComm* comm, struct ncclTaskColl* task) {
  task->opDev.op = symkRedOp(task->opHost, task->opDev.op);
  if (task->opDev.op == ncclDevSumPostDiv) {
    union {
      __half f16;
      float f32;
      uint64_t u64;
      void *ptr;
    };
    u64 = 0;
    switch (task->datatype) {
      // 16-bit floats use float accumulator
      case ncclFloat16:
      case ncclBfloat16:
        f32 = float(1.0/comm->nRanks);
        task->opDev.scalarArg = u64;
        return;
      case ncclFloat8e4m3:
      case ncclFloat8e5m2:
        f16 = __float2half(float(1.0/comm->nRanks));
        task->opDev.scalarArg = u64;
        return;
      default:
        break;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// ncclMakeSymmetricTaskList: Classify tasks into symmetric-eligible and remaining.

ncclResult_t ncclMakeSymmetricTaskList(
    struct pcclComm* comm,
    struct ncclTaskColl* task,
    struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue,
    struct ncclTaskColl** remainTasksHead
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclKernelPlanner* planner = &comm->planner;
  struct ncclTaskColl* remainTasksTail = nullptr;
  bool useMulti = pcclParamSymkUseMultiTask() != 0;
  *remainTasksHead = nullptr;

  if (task) {
    if (comm->symkWorkArgsBytes < ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 1)) {
      INFO(NCCL_TUNING, "PCCL_WORK_ARGS_BYTES=%u is too small for symmetric work metadata; using legacy",
           comm->symkWorkArgsBytes);
      *remainTasksHead = task;
      goto exit;
    }
    NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);
  }

  // Consume the linked input FIFO. Only adjacent compatible tasks may share a
  // batch; disabling multi-task caps it at one work without changing the ABI.
  while (task != nullptr) {
    struct ncclTaskColl* headTask = task;
    ncclDevRedOp_t symkOp = symkRedOp(task->opHost, task->opDev.op);
    ncclSymkKernelId kernelId = ncclSymkKernelId_Count;
    int nChannels = MAXCHANNELS;
    int nWarps = 0;
    int nWorks = 0;
    float estTimeUs = 1.e18f;
    size_t countTotal = 0, countMax = 0, countActualMax = 0;
    size_t cellCount = NCCL_SYM_KERNEL_CELL_SIZE / ncclTypeSize(headTask->datatype);
    bool forced = false;
    bool needFallback = false;

    for (ncclTaskColl* candidate = headTask; candidate != nullptr; candidate = candidate->next) {
      if (candidate->func != headTask->func || candidate->datatype != headTask->datatype ||
          symkRedOp(candidate->opHost, candidate->opDev.op) != symkOp) break;
      if (!ncclSymkAvailable(comm, candidate->func, symkOp, candidate->datatype, candidate->count)) break;
      NCCLCHECKGOTO(ncclDevrFindWindow(comm, candidate->sendbuff, &candidate->sendWin), ret, fail);
      NCCLCHECKGOTO(ncclDevrFindWindow(comm, candidate->recvbuff, &candidate->recvWin), ret, fail);
      NCCLCHECKGOTO(ncclGetSymRegType(candidate->sendWin, candidate->recvWin, &candidate->winRegType), ret, fail);
      if (candidate->winRegType != headTask->winRegType) break;

      // Registration is conservative even when LL or NOWIN is forced. Do not
      // initialize symmetric resources or convert an unregistered multi-GPU task.
      if (comm->intraRanks > 1 && candidate->winRegType == ncclSymSendNonregRecvNonreg) break;
      needFallback |= ncclDevrWindowHasSysmemSegment(candidate->sendWin) ||
                      ncclDevrWindowHasSysmemSegment(candidate->recvWin);
      nWorks++;
      size_t count = alignUp(candidate->count, cellCount);
      countTotal += count;
      if (count > countMax) countMax = count;
      if (candidate->count > countActualMax) countActualMax = candidate->count;
      if (!useMulti || ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, nWorks + 1) > comm->symkWorkArgsBytes) break;
    }

    if (nWorks != 0 && !needFallback) {
      NCCLCHECKGOTO(ncclSymkPickKernel(comm, headTask->func, symkOp, headTask->datatype,
                                       countTotal, countMax, countActualMax, nWorks, headTask->winRegType,
                                       &estTimeUs, &kernelId, &nChannels, &nWarps, &forced), ret, fail);
      bool isLLKernel = kernelId != ncclSymkKernelId_Count &&
                        ((1u << (int)kernelId) & ncclSymkLLKernelMask());
      if (isLLKernel && headTask->winRegType == ncclSymSendNonregRecvNonreg) {
        needFallback = headTask->algorithm != NCCL_ALGO_UNDEF && !forced && !ncclParamSymNoWinEnable();
        if (!needFallback) NCCLCHECKGOTO(ncclSymkInitOnce(comm), ret, fail);
      }
    }

    // Leave fallback tasks' reduction descriptors untouched. The planner count
    // changes only when ownership transfers to the symmetric queue.
    if (nWorks == 0) nWorks = 1;
    for (int work = 0; work < nWorks; work++) {
      struct ncclTaskColl* next = task->next;
      if (kernelId == ncclSymkKernelId_Count || needFallback) {
        task->next = nullptr;
        if (remainTasksTail) remainTasksTail->next = task;
        else *remainTasksHead = task;
        remainTasksTail = task;
      } else {
        task->devFuncId = (uint32_t)kernelId;
        task->nMaxChannels = nChannels;
        task->nWarps = nWarps;
        task->isSymLast = work == nWorks - 1;
        convertCollTaskToSymmetricTask(comm, task);
        ncclIntruQueueEnqueue(symTaskQueue, task);
        planner->nTasksColl--;
      }
      task = next;
    }
  }

exit:
  return ret;
fail:
  // The current batch has not been detached or converted; return it and the
  // untouched suffix to the caller for normal cleanup.
  if (remainTasksTail) remainTasksTail->next = task;
  else *remainTasksHead = task;
  goto exit;
}

ncclResult_t ncclSymmetricTaskScheduler(
    struct pcclComm* comm,
    struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue,
    struct ncclKernelPlan* plan
  ) {
  struct ncclTaskColl* headTask = ncclIntruQueueHead(symTaskQueue);
  if (headTask == nullptr) return ncclInternalError;
  uint32_t devFuncId = headTask->devFuncId;
  struct ncclTaskColl* task = nullptr;
  ssize_t totalCount = 0;
  ssize_t logCount = 0;
  ssize_t remainCell = 0;
  ssize_t cellPerChannel = 0;
  int workCount = 0, workIndex = 0;
  size_t cellCount = NCCL_SYM_KERNEL_CELL_SIZE / ncclTypeSize(headTask->datatype);
  ncclResult_t ret = ncclSuccess;
  int curChannel = 0;
  int curChannelWork = 0;
  int nMaxChannels = headTask->nMaxChannels;
  ncclDataType_t datatype = headTask->datatype;
  struct ncclSymkDevWork* workBufPtr = nullptr;
  struct ncclSymkChannelWorkRange* workRangePtr = nullptr;
  const char* funcName = ncclFuncToString(headTask->func);
  const char* kName = ncclSymkKernelIdToString(headTask->devFuncId);
  struct ncclSymkDevWorkArgs* argsBuf = nullptr;

  plan->isSymColl = true;
  plan->threadPerBlock = headTask->nWarps * WARP_SIZE;
  plan->hasProxyOps = false;
  ncclSymkKernelId kernelId = (ncclSymkKernelId)headTask->devFuncId;
  plan->symFuncIndex = (uint16_t)kernelId;
  int kernelIndex = ncclSymkGetKernelIndex(kernelId, headTask->opDev.op, headTask->datatype);
  if (kernelIndex < 0 || kernelIndex >= ncclSymkKernelCount || ncclSymkKernelList[kernelIndex] == nullptr ||
      ncclSymkKernelRequirements[kernelIndex] > comm->minCompCap) return ncclInternalError;
  plan->kernelFn = ncclSymkKernelList[kernelIndex];
  if (nMaxChannels < 1 || nMaxChannels > ncclSymkMaxBlocks) return ncclInternalError;

  task = headTask;
  while (task != nullptr && task->devFuncId == devFuncId) {
    workCount++;
    totalCount += alignUp(task->count, cellCount);
    logCount += task->count;
    if (task->isSymLast == 1) break;
    task = task->next;
  }

  size_t actualSize = ncclSymkDevWorkArgs::calcArgsSize(nMaxChannels, workCount);
  if (actualSize > sizeof(ncclSymkDevWorkArgs4K) || actualSize > comm->symkWorkArgsBytes) return ncclInternalError;
  plan->kernelArgsSize = actualSize;
  NCCLCHECK(ncclCalloc(reinterpret_cast<char**>(&argsBuf), actualSize));

  remainCell = cellPerChannel = DIVUP(DIVUP(totalCount, nMaxChannels), (ssize_t)cellCount);
  workRangePtr = argsBuf->getWorkRange();
  workBufPtr = argsBuf->getWorks(nMaxChannels);
  argsBuf->nMaxChannels = nMaxChannels;
  argsBuf->reserved = 0;
  bool incPrecision = pcclParamSymOpAccPrecisionInc() != 0;

  while (!ncclIntruQueueEmpty(symTaskQueue)) {
    struct ncclSymkDevWork devWork = {};
    size_t cellLeft = 0, taskCell = 0;
    uint8_t isSymLast = 0;

    if (ncclIntruQueueHead(symTaskQueue)->devFuncId != devFuncId) break;

    task = ncclIntruQueueDequeue(symTaskQueue);
    isSymLast = task->isSymLast;

    ret = ncclSymkMakeDevWork(comm, task, &devWork);
    if (ret != ncclSuccess) {
      ncclMemoryPoolFree(&comm->memPool_ncclTaskColl, task);
      goto fail;
    }
    devWork.pcclAccPrecisionInc = incPrecision;

    cellLeft = taskCell = DIVUP(task->count, cellCount);
    for (; curChannel < nMaxChannels;) {
      workRangePtr[curChannel].workHi = workIndex;
      if (curChannelWork == 0) {
        if (devWork.nChannels == 0) {
          devWork.sChannelId = curChannel;
          devWork.nChannels = 1;
        } else if ((ssize_t)cellLeft <= remainCell) {
          if ((remainCell - (ssize_t)cellLeft) * NCCL_SYM_KERNEL_CELL_SIZE <= (1 << 10) || ncclIntruQueueEmpty(symTaskQueue)) {
            devWork.nChannels++;
          }
        } else {
          devWork.nChannels++;
        }
      } else {
        if ((ssize_t)taskCell <= remainCell) {
          devWork.sChannelId = curChannel;
          devWork.nChannels = 1;
        }
      }
      if ((ssize_t)cellLeft < remainCell) {
        workRangePtr[curChannel].fracHi = uint16_t(0x10000UL - 1);
        remainCell -= cellLeft;
        curChannelWork++;
        break;
      } else if ((ssize_t)cellLeft == remainCell) {
        workRangePtr[curChannel].fracHi = uint16_t(0x10000UL - 1);
        remainCell = cellPerChannel;
        curChannel++;
        curChannelWork = 0;
        break;
      } else {
        cellLeft -= remainCell;
        workRangePtr[curChannel].fracHi = uint16_t(DIVUP(0x10000L * (ssize_t)(taskCell - cellLeft), (ssize_t)taskCell) - 1);
        remainCell = cellPerChannel;
        curChannel++;
        curChannelWork = 0;
      }
    }
    memcpy(workBufPtr + workIndex, &devWork, sizeof(struct ncclSymkDevWork));
    workIndex++;

    plan->groupApiEventHandle = task->groupApiEventHandle;

    ncclMemoryPoolFree(&comm->memPool_ncclTaskColl, task);
    if (isSymLast == 1) break;
    if (curChannel == nMaxChannels) {
      WARN("pcclSymmetricTaskScheduler ran out of channel space (nMaxChannels=%d, workCount=%d, workIndex=%d)",
           nMaxChannels, workCount, workIndex);
      ret = ncclInternalError;
      goto fail;
    }
  }
  if (remainCell < cellPerChannel) curChannel++;

  memcpy(&argsBuf->kcomm, &comm->symkState.kcomm, sizeof(comm->symkState.kcomm));
  plan->workBytes = totalCount * ncclTypeSize(datatype);
  plan->channelMask = uint64_t(-1) >> (64 - curChannel);
  plan->kernelSymkArgs = (void*)argsBuf;
  plan->workStorageType = ncclDevWorkStorageTypeArgs;

  if (comm->rank == 0) {
    INFO(NCCL_TUNING, "%s [Symmetric]: %ld Bytes -> Kernel %s nchannels %d nthreads %d nWorks %d",
         funcName, (long)(logCount * ncclTypeSize(datatype)),
         kName, curChannel, plan->threadPerBlock, workCount);
  }

exit:
  return ret;
fail:
  if (argsBuf) ncclFree(argsBuf);
  goto exit;
}
