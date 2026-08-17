//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include <algorithm>
#include <cstdio>
#include "pccl/ext_kernel_utils.h"
#include "pccl/ext_kernel_launch.h"
#include "customAllReduceKernels.h"
#include "customDeviceUtils.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
ncclResult_t kernelLaunchConfig(extKernelDevParams& devParams, extKernelHostParams& hostParams, size_t eltsPerThread, int* blocksPerGrid, int* threadsPerBlock) {
  if (devParams.eltsTotal % eltsPerThread != 0) {
    ERROR("EXT: eltsTotal %zu is not a multiple of %d", devParams.eltsTotal, eltsPerThread);
    return ncclInvalidUsage;
  }
  int nBlocks = 0;
  int nThreads = 0;

  int maxBlockNum = hostParams.maxBlockNum;
  int maxThreadNum = hostParams.maxThreadNum;
  extKernelStrategyType strategy = hostParams.strategy;
  switch (strategy) {
    case ONESHOT:
    case ONESHOT_REG: {

      nBlocks = 1;
      nThreads = maxThreadNum;

      const size_t totalThreads = devParams.eltsTotal / eltsPerThread;
      if (totalThreads <= maxThreadNum) { // local reduce
        nThreads = WARP_SIZE * divUp(totalThreads, WARP_SIZE);
        nBlocks = 1;
      } else { // local reduce
        nThreads = maxThreadNum;
        nBlocks = divUp(totalThreads, maxThreadNum);
        nBlocks = std::min(static_cast<int>(maxBlockNum), nBlocks);
      }
      devParams.eltsPerBlock = divUp(devParams.eltsTotal, nBlocks);
      devParams.eltsPerBlock = ROUNDUP(devParams.eltsPerBlock, eltsPerThread*nThreads);
      devParams.eltsPerBlock = std::min(devParams.eltsPerBlock, (uint32_t)devParams.eltsTotal);
      break;
    }
    case TWOSHOT:
    case TWOSHOT_REG: {
      nThreads = maxThreadNum;
      int eltsPerBlockPerIter = nThreads * eltsPerThread;

      if (!hostParams.useUsrConfigBlockNum) {
        nBlocks = divUp(devParams.eltsTotal, eltsPerBlockPerIter);
        nBlocks = std::min(nBlocks, maxBlockNum);
      } else {
        nBlocks = maxBlockNum;
      }

      devParams.eltsPerBlock = divUp(devParams.eltsTotal, nBlocks);
      devParams.eltsPerBlock = ROUNDUP(devParams.eltsPerBlock, eltsPerBlockPerIter);
      break;
    }
    case ONESHOT_LL: {
      nThreads = std::min(maxThreadNum, EXT_KERNEL_LL_MAX_THREADS);
      const int llEltsPerThread = eltsPerThread / 2;

      const size_t totalThreads = devParams.eltsTotal / llEltsPerThread;
      if (totalThreads <= (size_t)nThreads) {
        nThreads = WARP_SIZE * divUp(totalThreads, WARP_SIZE);
        nBlocks = 1;
      } else {
        nBlocks = divUp(totalThreads, nThreads);
        nBlocks = std::min(EXT_KERNEL_LL_MAX_BLOCKS, nBlocks);
      }
      devParams.eltsPerBlock = divUp(devParams.eltsTotal, nBlocks);
      devParams.eltsPerBlock = ROUNDUP(devParams.eltsPerBlock, llEltsPerThread * nThreads);
      devParams.eltsPerBlock = std::min(devParams.eltsPerBlock, (uint32_t)devParams.eltsTotal);
      break;
    }
    default: {
      ERROR("EXT: strategy %d not supported here.", strategy);
      return ncclInvalidUsage;
    }
  }

  *blocksPerGrid = nBlocks;
  *threadsPerBlock = nThreads;
  if (strategy != ONESHOT_LL) {
    devParams.buffEltsPerBlock = hostParams.bufSize / hostParams.dtypeSize / maxBlockNum;
    if (devParams.buffEltsPerBlock % eltsPerThread != 0) {
      ERROR("buffEltsPerBlock %d must be aligned to %d (eltsPerThread)", devParams.buffEltsPerBlock, (int)eltsPerThread);
      return ncclInternalError;
    }
  }

  INFO(PCCL_EXT, "EXT: blocksPerGrid:%d threadsPerBlock:%d, eltsPerBlock:%u, eltsTotal:%u, bufEltsPerBlock: %u",
                   nBlocks, nThreads, devParams.eltsPerBlock, devParams.eltsTotal, devParams.buffEltsPerBlock);

  return ncclSuccess;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
ncclResult_t extKernelSetDispatchStrategy(const void* func, const char* funcName, extKernelHostParams& hostParams, int blocksPerGrid, ncclFunc_t collFunc) {
  int funcIndex = hostParams.funcIndex;
  dispatchInfo dInfo = {collFunc, 1, blocksPerGrid, 0};
  NCCLCHECK(setDispatchStrategy(func, &hostParams.funcStrategy[funcIndex], hostParams.defaultStrategy, funcName, hostParams.compCap, dInfo));
  return ncclSuccess;
}
