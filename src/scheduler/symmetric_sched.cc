/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
// symmetric_sched.cc: Multi-task symmetric kernel scheduling.

#include "scheduler.h"
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

PCCL_PARAM(SymkUseMultiTask, "SYMK_USE_MULTI_TASK", 0);

extern int64_t pcclParamSymOpAccPrecisionInc();

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
    struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* collTaskQueue,
    struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue,
    struct ncclTaskColl** remainTasksHead
  ) {
  ncclResult_t ret = ncclSuccess;
  int fnOpTySymCount = 0;
  struct ncclTaskColl* tasksSymByFnOpTy[ncclNumFuncs * ncclNumDevRedOps * ncclNumTypes * ncclNumSymRegTypes];
  int fnOpTySymIndices[ncclNumFuncs * ncclNumDevRedOps * ncclNumTypes * ncclNumSymRegTypes];
  struct ncclKernelPlanner* planner = &comm->planner;
  struct ncclTaskColl* remainTasksTail = nullptr;
  bool foundSymm = false;

  memset(tasksSymByFnOpTy, 0, sizeof(tasksSymByFnOpTy));
  *remainTasksHead = nullptr;

  if (!ncclIntruQueueEmpty(collTaskQueue)) {
    NCCLCHECK(ncclDevrInitOnce(comm));
  }

  // Dequeue all tasks from collTaskQueue and classify them
  while (!ncclIntruQueueEmpty(collTaskQueue)) {
    struct ncclTaskColl* task = ncclIntruQueueDequeue(collTaskQueue);
    ncclDevRedOp_t symkOp = symkRedOp(task->opHost, task->opDev.op);
    bool symAvailable = ncclSymkAvailable(comm, task->func, symkOp, task->datatype, task->count);

    if (symAvailable) {
      NCCLCHECK(ncclDevrFindWindow(comm, task->sendbuff, &task->sendWin));
      NCCLCHECK(ncclDevrFindWindow(comm, task->recvbuff, &task->recvWin));
      NCCLCHECK(ncclGetSymRegType(task->sendWin, task->recvWin, &task->winRegType));

      int index = (((int)task->func * ncclNumDevRedOps + symkOp) * ncclNumTypes + (int)task->datatype) * ncclNumSymRegTypes + (int)task->winRegType;
      if (tasksSymByFnOpTy[index] == nullptr) fnOpTySymIndices[fnOpTySymCount++] = index;
      task->next = tasksSymByFnOpTy[index];
      tasksSymByFnOpTy[index] = task;
      planner->nTasksColl--;
      foundSymm = true;
    } else {
      task->next = nullptr;
      if (*remainTasksHead) {
        remainTasksTail->next = task;
        remainTasksTail = task;
      } else {
        *remainTasksHead = remainTasksTail = task;
      }
    }
  }
  if (!foundSymm) goto exit;

  if (pcclParamSymkUseMultiTask() &&
      comm->workArgsBytes < pcclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 1)) {
    WARN("workArgsBytes (%u) too small for symmetric kernel args", comm->workArgsBytes);
    ret = ncclInternalError;
    goto exit;
  }

  // Group tasks by (func, op, type, regType). Fast-path mode emits one plan
  // per task (nWorks=1); multi-task mode batches until args-buffer is full.
  for (int cursor = 0; cursor < fnOpTySymCount; cursor++) {
    struct ncclTaskColl* task = tasksSymByFnOpTy[fnOpTySymIndices[cursor]];
    bool useMulti = pcclParamSymkUseMultiTask() != 0;
    while (task != nullptr) {
      ncclSymkKernelId kernelId = ncclSymkKernelId_Count;
      int nChannels = MAXCHANNELS;
      int nWarps = 0;
      int nWorks = 0;
      float estTimeUs = 1.e18f;
      size_t countTotal = 0, countMax = 0;
      struct ncclTaskColl* headTask = task;
      size_t cellCount = NCCL_SYM_KERNEL_CELL_SIZE / ncclTypeSize(headTask->datatype);
      bool forced = false;
      ncclDevRedOp_t symkOp = symkRedOp(task->opHost, task->opDev.op);

      while (task != nullptr) {
        size_t count;
        nWorks++;
        count = alignUp(task->count, cellCount);
        countTotal += count;
        if (count > countMax) countMax = count;
        bool overflow = useMulti && (pcclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, nWorks + 1) > comm->workArgsBytes);
        if (!useMulti || overflow || task->next == nullptr) {
          task->isSymLast = 1;
          break;
        }
        task = task->next;
      }

      NCCLCHECK(ncclSymkPickKernel(comm, headTask->func, symkOp, headTask->datatype,
                                   countTotal, countMax, nWorks, headTask->winRegType,
                                   &estTimeUs, &kernelId, &nChannels, &nWarps, &forced));
      task = headTask;
      bool isLLKernel = (1u << (int)kernelId) & ncclSymkLLKernelMask();
      bool needFallback = false;

      // Fallback logic for symmetric LL kernels:
      // - If both src and dst are registered, we don't fall back if a symmetric kernel is available.
      // - Otherwise, we have to fall back to a legacy kernel if running the selected symmetric LL
      //   kernel is not possible (if the buffers are not registered and we manage multiple GPUs
      //   in a single process, i.e. intraRanks > 1).
      // - If the user forced a symmetric kernel via PCCL_SYM_KERNEL or requested preference for
      //   using symmetric kernels even without symmetric buffers via NCCL_SYM_NOWIN_ENABLE, we
      //   respect that.
      if (headTask->winRegType == ncclSymSendRegRecvReg || headTask->algorithm == NCCL_ALGO_UNDEF) {
        needFallback = false;
      } else if (isLLKernel) {
        needFallback = comm->intraRanks > 1 && headTask->winRegType == ncclSymSendNonregRecvNonreg;
        if (!needFallback && !forced) {
          needFallback = !ncclParamSymNoWinEnable() && headTask->winRegType == ncclSymSendNonregRecvNonreg;
        }
      }

      // Override fallback when buffers contain sysmem segments
      if (!needFallback) {
        bool hasSysmemSegment = ncclDevrWindowHasSysmemSegment(headTask->sendWin) ||
                                ncclDevrWindowHasSysmemSegment(headTask->recvWin);
        needFallback = hasSysmemSegment;
      }

      if (kernelId == ncclSymkKernelId_Count || needFallback) {
        // Cannot find appropriate symmetric kernel; fallback to legacy
        while (task != nullptr) {
          struct ncclTaskColl* next = task->next;
          int isSymLast = task->isSymLast;
          task->next = nullptr;
          if (*remainTasksHead) {
            remainTasksTail->next = task;
            remainTasksTail = task;
          } else {
            *remainTasksHead = remainTasksTail = task;
          }
          planner->nTasksColl++;
          task = next;
          if (isSymLast) break;
        }
        continue;
      }

      // Initialize symmetric objects for LL kernels without registered buffers
      if (isLLKernel && headTask->winRegType == ncclSymSendNonregRecvNonreg) {
        NCCLCHECK(ncclSymkInitOnce(comm));
      }

      // Set all symmetric tasks in this group to the selected kernel
      while (task != nullptr) {
        struct ncclTaskColl* next = task->next;
        int isSymLast = task->isSymLast;
        task->devFuncId = (uint32_t)kernelId;
        task->nMaxChannels = nChannels;
        task->nWarps = nWarps;
        convertCollTaskToSymmetricTask(comm, task);
        ncclIntruQueueEnqueue(symTaskQueue, task);
        task = next;
        if (isSymLast) break;
      }
    }
  }

exit:
  return ret;
}

static ncclResult_t ncclSymmetricTaskSchedulerFast(
    struct pcclComm* comm,
    struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue,
    struct ncclKernelPlan* plan
  ) {
  struct ncclTaskColl* task = ncclIntruQueueDequeue(symTaskQueue);
  ncclSymkKernelId kernelId = (ncclSymkKernelId)task->devFuncId;
  int kernelIndex = ncclSymkGetKernelIndex(kernelId, task->opDev.op, task->datatype);

  plan->isSymColl = true;
  plan->threadPerBlock = task->nWarps * WARP_SIZE;
  plan->hasProxyOps = false;
  plan->symFuncIndex = (uint16_t)kernelId;
  plan->kernelFn = ncclSymkKernelList_fast[kernelIndex];

  plan->kernelArgsSize = sizeof(ncclSymDevArgs);

  struct pcclSymkDevWork devWork = {};
  ncclResult_t ret = ncclSymkMakeDevWork(comm, task, &devWork);
  if (ret != ncclSuccess) {
    ncclMemoryPoolFree(&comm->memPool_ncclTaskColl, task);
    return ret;
  }

  // Resolve LSA-flat base on host (shadow pool gives the host copy of vidmem).
  // NonregNonreg LL path: windows are NULL, fall back to raw user pointers
  // and use symk window stride4G.
  struct ncclWindow_vidmem* inWinHost = nullptr;
  struct ncclWindow_vidmem* outWinHost = nullptr;
  if (devWork.inputWin)
    NCCLCHECK(ncclShadowPoolToHost(&comm->devrState.shadows, devWork.inputWin, &inWinHost));
  if (devWork.outputWin)
    NCCLCHECK(ncclShadowPoolToHost(&comm->devrState.shadows, devWork.outputWin, &outWinHost));

  uint32_t stride4G = inWinHost  ? inWinHost->stride4G
                    : outWinHost ? outWinHost->stride4G
                    : (uint32_t)(comm->devrState.bigSize >> 32);

  auto* args = ncclMemoryStackAlloc<ncclSymDevArgs>(&comm->memScoped);
  args->comm.base         = comm->symkState.kcomm.base;
  args->comm.stride4G     = stride4G;
  args->comm.nRanks       = comm->symkState.kcomm.devComm.lsaSize;
  args->comm.rank         = comm->symkState.kcomm.devComm.lsaRank;
  args->comm.nRanks_rcp32 = comm->symkState.kcomm.devComm.lsaSize_rcp32;
  args->rootRank          = devWork.rootRank;
  args->nBlocks           = task->nMaxChannels;
  args->redOpArg          = devWork.redOpArg;
  args->nElts             = devWork.nElts;
  args->input  = inWinHost
                 ? (char*)::add4G((void*)inWinHost->lsaFlatBase,
                                  (int)(args->comm.rank * inWinHost->stride4G)) + devWork.inputOff
                 : (char*)devWork.inputOff;
  args->output = outWinHost
                 ? (char*)::add4G((void*)outWinHost->lsaFlatBase,
                                  (int)(args->comm.rank * outWinHost->stride4G)) + devWork.outputOff
                 : (char*)devWork.outputOff;
  args->incPrecision = pcclParamSymOpAccPrecisionInc();

  plan->kernelSymkArgs = (void*)args;
  plan->workStorageType = ncclDevWorkStorageTypeArgs;
  plan->workBytes = devWork.nElts * ncclTypeSize(task->datatype);
  plan->channelMask = uint64_t(-1) >> (64 - task->nMaxChannels);
  plan->groupApiEventHandle = task->groupApiEventHandle;

  if (comm->rank == 0) {
    INFO(NCCL_TUNING, "%s [Symmetric]: %ld Bytes -> Kernel %s nchannels %d nthreads %d",
         ncclFuncToString(task->func), (long)(task->count * ncclTypeSize(task->datatype)),
         ncclSymkKernelIdToString(task->devFuncId), task->nMaxChannels, plan->threadPerBlock);
  }

  ncclMemoryPoolFree(&comm->memPool_ncclTaskColl, task);
  return ncclSuccess;
}

static ncclResult_t ncclSymmetricTaskSchedulerMulti(
    struct pcclComm* comm,
    struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue,
    struct ncclKernelPlan* plan
  ) {
  struct ncclTaskColl* headTask = ncclIntruQueueHead(symTaskQueue);
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
  struct pcclSymkDevWork* workBufPtr = nullptr;
  struct pcclSymkChannelWorkRange* workRangePtr = nullptr;
  const char* funcName = ncclFuncToString(headTask->func);
  const char* kName = ncclSymkKernelIdToString(headTask->devFuncId);
  struct pcclSymkDevWorkArgs* argsBuf = nullptr;

  plan->isSymColl = true;
  plan->threadPerBlock = headTask->nWarps * WARP_SIZE;
  plan->hasProxyOps = false;
  ncclSymkKernelId kernelId = (ncclSymkKernelId)headTask->devFuncId;
  plan->symFuncIndex = (uint16_t)kernelId;
  int kernelIndex = ncclSymkGetKernelIndex(kernelId, headTask->opDev.op, headTask->datatype);
  plan->kernelFn = ncclSymkKernelList[kernelIndex];


  task = headTask;
  while (task != nullptr && task->devFuncId == devFuncId) {
    workCount++;
    totalCount += alignUp(task->count, cellCount);
    logCount += task->count;
    if (task->isSymLast == 1) break;
    task = task->next;
  }

  size_t actualSize = pcclSymkDevWorkArgs::calcArgsSize(nMaxChannels, workCount);
  if (actualSize > sizeof(pcclSymkDevWorkArgs4K)) {
    WARN("Symmetric kernel args (%zu bytes) exceed 4KB by-value capacity (nMaxChannels=%d, workCount=%d)",
         actualSize, nMaxChannels, workCount);
    return ncclInternalError;
  }
  plan->kernelArgsSize = sizeof(pcclSymkDevWorkArgs4K);
  argsBuf = (struct pcclSymkDevWorkArgs*)
      ncclMemoryPoolAlloc<pcclSymkDevWorkArgs4K>(&comm->memPool_kernelSymkArgs, &comm->memPermanent);
  if (!argsBuf) {
    WARN("Failed to allocate symmetric kernel args buffer (%zu bytes)", plan->kernelArgsSize);
    return ncclSystemError;
  }

  remainCell = cellPerChannel = DIVUP(DIVUP(totalCount, nMaxChannels), (ssize_t)cellCount);
  workRangePtr = argsBuf->getWorkRange();
  workBufPtr = argsBuf->getWorks(nMaxChannels);
  argsBuf->nMaxChannels = nMaxChannels;

  while (!ncclIntruQueueEmpty(symTaskQueue)) {
    struct pcclSymkDevWork devWork = {};
    size_t cellLeft = 0, taskCell = 0;
    uint8_t isSymLast = 0;

    if (ncclIntruQueueHead(symTaskQueue)->devFuncId != devFuncId) break;

    task = ncclIntruQueueDequeue(symTaskQueue);
    isSymLast = task->isSymLast;

    NCCLCHECKGOTO(ncclSymkMakeDevWork(comm, task, &devWork), ret, fail);

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
    memcpy(workBufPtr + workIndex, &devWork, sizeof(struct pcclSymkDevWork));
    workIndex++;

    plan->groupApiEventHandle = task->groupApiEventHandle;

    ncclMemoryPoolFree(&comm->memPool_ncclTaskColl, task);
    if (isSymLast == 1) break;
    if (curChannel == nMaxChannels) {
      WARN("pcclSymmetricTaskScheduler ran out of channel space (nMaxChannels=%d, workCount=%d, workIndex=%d)",
           nMaxChannels, workCount, workIndex);
      goto fail;
    }
  }
  if (remainCell < cellPerChannel) curChannel++;

  memcpy(&argsBuf->kcomm, &comm->symkState.kcomm, sizeof(comm->symkState.kcomm));
  plan->workBytes = totalCount * ncclTypeSize(headTask->datatype);
  plan->channelMask = uint64_t(-1) >> (64 - curChannel);
  plan->kernelSymkArgs = (void*)argsBuf;
  plan->workStorageType = ncclDevWorkStorageTypeArgs;

  if (comm->rank == 0) {
    INFO(NCCL_TUNING, "%s [Symmetric]: %ld Bytes -> Kernel %s nchannels %d nthreads %d nWorks %d",
         funcName, (long)(logCount * ncclTypeSize(headTask->datatype)),
         kName, curChannel, plan->threadPerBlock, workCount);
  }

exit:
  return ret;
fail:
  if (argsBuf) {
    ncclMemoryPoolFree<pcclSymkDevWorkArgs4K>(&comm->memPool_kernelSymkArgs,
        reinterpret_cast<pcclSymkDevWorkArgs4K*>(argsBuf));
  }
  goto exit;
}

ncclResult_t ncclSymmetricTaskScheduler(
    struct pcclComm* comm,
    struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue,
    struct ncclKernelPlan* plan
  ) {
  if (pcclParamSymkUseMultiTask())
    return ncclSymmetricTaskSchedulerMulti(comm, symTaskQueue, plan);
  return ncclSymmetricTaskSchedulerFast(comm, symTaskQueue, plan);
}
