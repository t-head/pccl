//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include <stdarg.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <atomic>
#include "errno.h"
#include <hggc.h>

#include "pccl/ext_kernel.h"
#include "pccl/ext_kernel_eit.h"
#include "pccl/ext_kernel_utils.h"
#include "ext_topo.h"
#include "customAllReduceKernels.h"
#include "customAllGatherKernels.h"
#include "customReduceScatterKernels.h"
#include "customKernels.h"

ncclDebugLogger_t logFn;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

PCCL_PARAM(CustomAllReduceDoubleBuffer, "CUSTOM_ALLREDUCE_DOUBLE_BUFFER", -2);
PCCL_PARAM(CustomAllReduceBlocks, "CUSTOM_ALLREDUCE_BLOCK_NUM", -2);
PCCL_PARAM(CustomAllReduceThreads, "CUSTOM_ALLREDUCE_THREAD_NUM", -2);
PCCL_PARAM(CustomAllReduceLLBound, "CUSTOM_ALLREDUCE_LL_BOUND", -2);
static_assert((RMT_BULK_ALIGN_SIZE & (RMT_BULK_ALIGN_SIZE-1)) == 0, "RMT_BULK_ALIGN_SIZE should be power of 2");

static ssize_t getFlagsSize(int nRanks) {
  ssize_t static constexpr flagsSize = (MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t);
  // keep x2 size although we do not use the later part.
  return  flagsSize*nRanks*2;
}

ncclResult_t configHostParams(int nRanks, extTopo topo, struct extKernelHostParams* hostParams) {
  ncclResult_t result = ncclSuccess;
  size_t llEpochsSize;
  size_t bufAllocSize;

  if (topoConfigs[topo].topoId != topo) {
    ERROR("EXT KERNEL: topo config id should match it's index exactly. topo id:%d", topo);
    return ncclInternalError;
  }
  hostParams->boundLow = topoConfigs[topo].boundLow;
  hostParams->boundHigh = topoConfigs[topo].boundHigh;
  hostParams->agBoundHigh = topoConfigs[topo].agBoundHigh;
  hostParams->rsBoundHigh = topoConfigs[topo].rsBoundHigh;
  hostParams->oneShotLLBoundHigh = topoConfigs[topo].oneShotLLBoundHigh;
  hostParams->twoShotSwitchPoint = topoConfigs[topo].twoShotSwitchPoint;
  hostParams->maxBlockNum = topoConfigs[topo].maxBlockNum;
  hostParams->maxThreadNum = MAX_BLOCK_SIZE;
  hostParams->useUsrConfigBlockNum = false;
  hostParams->useUsrConfigThreadNum = false;
  NCCLCHECKGOTO(extCalloc(&hostParams->funcStrategy, EXT_TOTAL_FUNCS_NUM), result, fail);
  if (hostParams->twoShotSwitchPoint < 0) hostParams->twoShotSwitchPoint = hostParams->boundHigh;

  // process user config;
  if (pcclParamCustomAllReduceBlocks() > 0) {
    hostParams->useUsrConfigBlockNum = true;
    hostParams->maxBlockNum = pcclParamCustomAllReduceBlocks();
  }

  if (pcclParamCustomAllReduceThreads() > 0) {
    hostParams->useUsrConfigThreadNum = true;
    hostParams->maxThreadNum = pcclParamCustomAllReduceThreads();
  }

  if (pcclParamCustomAllReduceDoubleBuffer() > 0) {
    hostParams->doubleBuffer = 1;
  } else {
    hostParams->doubleBuffer = 0;
  }

  if (hostParams->useUsrConfigBlockNum || hostParams->useUsrConfigThreadNum) {
    // adjust upper bound to fit block num;
    size_t highBoundIntElts = hostParams->boundHigh/sizeof(int);
    int warpNum = hostParams->maxThreadNum;
    int bulkElts = RMT_BULK_ALIGN_SIZE / sizeof(int);
    size_t maxBlockEltsPerIter = warpNum * hostParams->maxBlockNum * bulkElts;
    highBoundIntElts = ROUNDUP(highBoundIntElts, maxBlockEltsPerIter);
    hostParams->boundHigh = highBoundIntElts * sizeof(int);
  }

  // bufSize must stay padding-free: buffEltsPerBlock = bufSize/dtypeSize/maxBlockNum
  // relies on it being a multiple of dtypeSize*eltsPerThread*maxBlockNum for AR/AG.
  hostParams->bufSize = hostParams->boundHigh;
  if (hostParams->agBoundHigh > (int32_t)hostParams->bufSize) hostParams->bufSize = hostParams->agBoundHigh;
  if (hostParams->rsBoundHigh > (int32_t)hostParams->bufSize) hostParams->bufSize = hostParams->rsBoundHigh;
  hostParams->peerRmtAddrSize = nRanks * MAX_ALL_REDUCE_BLOCKS * 2 * sizeof(void*); // 2 for sendbuff and recvbuff
  hostParams->barrierSize = getFlagsSize(nRanks);

  // LL buffer and epoch configuration
  hostParams->llBufferSize = EXT_KERNEL_LL_MAX_BLOCKS * 2 * extKernelLLEpochSize(nRanks);
  llEpochsSize = ROUNDUP(EXT_KERNEL_LL_MAX_BLOCKS * sizeof(uint32_t), 16);
  if (pcclParamCustomAllReduceLLBound() > 0) {
    hostParams->oneShotLLBoundHigh = pcclParamCustomAllReduceLLBound();
  }

  // Build region config table from computed sizes.
  // Quirks: BUFFER_D when double buffering is off is { !enabled, peerVisible }
  // (no data; pointer table aliased to BUFFER in pluginConfigWorkspace).
  // LL_EPOCHS is { enabled, !peerVisible } (local-only, no peer table).
  // RS chunked-copy places each rank's chunk at a 512-byte-aligned stride in the
  // peer buffer; padding per chunk is at most RMT_BULK_ALIGN_SIZE-1 bytes.
  // Reserve the extra space in the allocation only, not in bufSize.
  bufAllocSize = hostParams->bufSize + nRanks * RMT_BULK_ALIGN_SIZE;
  hostParams->regionConfig[DEV_REGION_BUFFER]         = { true,                       true,  bufAllocSize };
  hostParams->regionConfig[DEV_REGION_BUFFER_D]       = { hostParams->doubleBuffer != 0, true,  bufAllocSize };
  hostParams->regionConfig[DEV_REGION_BARRIER_IN]     = { true,                       true,  hostParams->barrierSize };
  hostParams->regionConfig[DEV_REGION_BARRIER_PRE]    = { true,                       true,  hostParams->barrierSize };
  hostParams->regionConfig[DEV_REGION_BARRIER_REDUCE] = { true,                       true,  hostParams->barrierSize };
  hostParams->regionConfig[DEV_REGION_BARRIER_POST]   = { true,                       true,  hostParams->barrierSize };
  hostParams->regionConfig[DEV_REGION_BARRIER_OUT]    = { true,                       true,  hostParams->barrierSize };
  hostParams->regionConfig[DEV_REGION_PEER_RMT_ADDR]  = { true,                       true,  hostParams->peerRmtAddrSize };
  hostParams->regionConfig[DEV_REGION_LL_BUFFER]      = { true,                       true,  hostParams->llBufferSize };
  hostParams->regionConfig[DEV_REGION_LL_EPOCHS]      = { true,                       false, llEpochsSize };

  return ncclSuccess;
fail:
  extFree(hostParams->funcStrategy);
  return result;
}

ncclResult_t pluginInit(const struct pcclExtKernelInitInfo* info, void** params, ssize_t* devMemSize, ncclDebugLogger_t logfn) {
  pthread_mutex_lock(&lock);
  logFn = logfn;
  pthread_mutex_unlock(&lock);
  ncclResult_t result = ncclSuccess;
  int rank = info->rank;
  int nRanks = info->nRanks;
  *devMemSize = 0;

  extTopo topo = UNSUPPORTED_TOPO;
  struct extKernelParams* extKernelParams = NULL;
  struct extKernelDevParams* devParams = NULL;
  NCCLCHECKGOTO(parseTopo(info->ppuType, info->icnRmtType, info->localRanks, info->topoArray, &topo), result, cleanup);
  if (topo == UNSUPPORTED_TOPO) {
    *devMemSize = -1;  // info invoker do not support, unload plugin.
    INFO(NCCL_EXT, "EXT: skip ext kernel init due to topo unsupported");
    goto cleanup;
  }

  NCCLCHECKGOTO(extCalloc(&extKernelParams, 1), result, cleanup);
  NCCLCHECKGOTO(extHggcHostCalloc(&devParams, 1), result, cleanup);

  extKernelParams->devParams = devParams;
  extKernelParams->hostParams.nRanks = nRanks;
  extKernelParams->hostParams.compCap = info->compCap;
  extKernelParams->hostParams.defaultStrategy = info->defaultStrategy;

  devParams->myRank = rank;
  *params = reinterpret_cast<void*>(extKernelParams);
  NCCLCHECKGOTO(configHostParams(nRanks, topo, &extKernelParams->hostParams), result, cleanup);

  // devMemSize computed from region config table (single source of truth).
  *devMemSize = computeDevMemTotalSize(extKernelParams->hostParams.regionConfig,
                                       DEV_REGION_COUNT);

exit:
  return result;
cleanup:
  NCCLCHECK(extFree(extKernelParams));
  if (devParams) {
    HGGCCHECK(hggcFreeHost(devParams));
    devParams = NULL;
  }
  goto exit;
}

PCCL_PARAM(CustomAllReduceStrategy, "CUSTOM_ALLREDUCE_STRATEGY", -2);

ncclResult_t pluginAvailCheck(const struct ncclExtTaskColl* task, const void* params, bool* usingExtKernel) {
  *usingExtKernel = false;

  if (task->func != ncclFuncAllReduce && task->func != ncclFuncAllGather && task->func != ncclFuncReduceScatter)
    return ncclSuccess;
  if (task->func == ncclFuncAllReduce && task->op != ncclDevSum)
    return ncclSuccess;
  if (task->func == ncclFuncReduceScatter && task->op != ncclDevSum)
    return ncclSuccess;

  const struct extKernelHostParams* hostParams = reinterpret_cast<const struct extKernelHostParams*>(&reinterpret_cast<const struct extKernelParams*>(params)->hostParams);

  if (task->func == ncclFuncAllGather) {
    // PCCL core's argcheck.cc always rewrites AllGather to ncclInt8 + count-as-per-rank-bytes.
    // The kernel does byte-level memcpy via Bulk*128, so it needs PACKED_ELEMS_SIZE alignment.
    if (task->datatype != pcclInt8) return ncclSuccess;
    if (task->count == 0) return ncclSuccess;
    if (task->count % PACKED_ELEMS_SIZE != 0) return ncclSuccess;
    size_t totalBytes = task->count * hostParams->nRanks;
    if (totalBytes <= (size_t)hostParams->agBoundHigh) {
      *usingExtKernel = 1;
    }
    return ncclSuccess;
  }

  if (task->func == ncclFuncReduceScatter) {
    // Supports half, bfloat16, float with Sum reduction.
    if (task->datatype != pcclHalf && task->datatype != pcclFloat && task->datatype != pcclBfloat16)
      return ncclSuccess;
    int dtypeSize;
    if (task->datatype == pcclHalf || task->datatype == pcclBfloat16)
      dtypeSize = sizeof(uint16_t);
    else
      dtypeSize = sizeof(float);
    size_t totalBytes = task->count * dtypeSize * hostParams->nRanks;
    if (totalBytes == 0) return ncclSuccess;
    if (totalBytes <= (size_t)hostParams->rsBoundHigh) {
      *usingExtKernel = 1;
    }
    return ncclSuccess;
  }

  // AllReduce path
  if (task->datatype != pcclHalf && task->datatype != pcclFloat && task->datatype != pcclBfloat16)
    return ncclSuccess;

  ssize_t eltsPerThread;
  size_t nBytes;
  if (task->datatype == pcclHalf || task->datatype == pcclBfloat16) {
    eltsPerThread = PACKED_ELEMS_SIZE / sizeof(uint16_t);
    nBytes = task->count * sizeof(uint16_t);
  } else {
    eltsPerThread = PACKED_ELEMS_SIZE / sizeof(float);
    nBytes = task->count * sizeof(float);
  }
  if (nBytes == 0) return ncclSuccess;

  if (nBytes >= (size_t)hostParams->boundLow &&
      nBytes <= (size_t)hostParams->boundHigh &&
      task->count % eltsPerThread == 0) {
    *usingExtKernel = 1;
  }
  return ncclSuccess;
}

ncclResult_t pluginBuildParams(const struct ncclExtTaskColl* task, void* params) {
  struct extKernelDevParams* devParams = reinterpret_cast<struct extKernelParams*>(params)->devParams;
  struct extKernelHostParams* hostParams = &reinterpret_cast<struct extKernelParams*>(params)->hostParams;

  hostParams->currentFunc = task->func;
  devParams->sendbuff = const_cast<void*>(task->sendbuff);
  devParams->recvbuff = task->recvbuff;

  static int64_t pcclCustomAllReduceStrategy = pcclParamCustomAllReduceStrategy();

  if (task->func == ncclFuncAllGather) {
    // After argcheck.cc: datatype=ncclInt8, count=per-rank bytes.
    // The kernel treats data as int8 byte stream, so dtype/dtypeSize/eltsTotal map directly.
    size_t totalBytes = task->count * hostParams->nRanks;
    hostParams->dtype = pcclInt8;
    hostParams->dtypeSize = 1;
    devParams->eltsTotal = totalBytes;

    hostParams->strategy = ONESHOT;
  } else if (task->func == ncclFuncReduceScatter) {
    // task->count = recvcount (per-rank element count); eltsTotal = count * nRanks.
    int dtypeSize;
    if (task->datatype == pcclHalf || task->datatype == pcclBfloat16)
      dtypeSize = sizeof(uint16_t);
    else
      dtypeSize = sizeof(float);

    hostParams->dtype = task->datatype;
    hostParams->dtypeSize = dtypeSize;
    devParams->eltsTotal = task->count * hostParams->nRanks;

    hostParams->strategy = ONESHOT;
  } else {
    int dtypeSize;
    size_t nBytes;
    if (task->datatype == pcclHalf || task->datatype == pcclBfloat16) {
      dtypeSize = sizeof(uint16_t);
    } else {
      dtypeSize = sizeof(float);
    }
    nBytes = task->count * dtypeSize;

    hostParams->dtype = task->datatype;
    hostParams->dtypeSize = dtypeSize;
    devParams->eltsTotal = task->count;

    if (nBytes <= (size_t)hostParams->twoShotSwitchPoint) {
      if (nBytes <= (size_t)hostParams->oneShotLLBoundHigh) {
        hostParams->strategy = ONESHOT_LL;
      } else {
        hostParams->strategy = task->regUsed ? ONESHOT_REG : ONESHOT;
      }
    } else {
      hostParams->strategy = task->regUsed ? TWOSHOT_REG : TWOSHOT;
    }
    if (pcclCustomAllReduceStrategy >= ONESHOT && pcclCustomAllReduceStrategy < STRATEGY_NUM) {
      hostParams->strategy = (extKernelStrategyType)pcclCustomAllReduceStrategy;
    }

    if (task->regUsed) {
      devParams->sendbuffOffset = task->sendbuffOffset;
      devParams->recvbuffOffset = task->recvbuffOffset;
      devParams->sendbuffRmtAddrs = task->sendbuffRmtAddrs;
      devParams->recvbuffRmtAddrs = task->recvbuffRmtAddrs;
    }
  }

  const char* collName = (task->func == ncclFuncAllGather) ? "AllGather" :
                          (task->func == ncclFuncReduceScatter) ? "ReduceScatter" : "AllReduce";
  INFO(NCCL_COLL | NCCL_EXT, "EXT: Coll %s, count %zu, datatype %d, ext kernel, strategy:%d", collName, task->count, task->datatype, hostParams->strategy);
  return ncclSuccess;
}

ncclResult_t pluginConfigWorkspace(void** buffers, int nGpus, void* params) {
  struct extKernelParams* extParams = reinterpret_cast<struct extKernelParams*>(params);
  struct extKernelDevParams* devParams = extParams->devParams;
  struct extKernelHostParams* hostParams = &extParams->hostParams;

  int32_t myRank = devParams->myRank;

  // peerAddrs[gpu][region] = base address of region data on that GPU
  void* peerAddrs[NCCL_MAX_LOCAL_RANKS][DEV_REGION_COUNT];

  for (int i = 0; i < nGpus; ++i) {
    void* dataBases[DEV_REGION_COUNT];
    void* ptrTables[DEV_REGION_COUNT];
    computeDevMemLayout(buffers[i], hostParams->regionConfig, DEV_REGION_COUNT,
                        dataBases, ptrTables);

    // When double buffering is disabled, BUFFER_D aliases BUFFER's data area
    if (!hostParams->regionConfig[DEV_REGION_BUFFER_D].enabled)
      dataBases[DEV_REGION_BUFFER_D] = dataBases[DEV_REGION_BUFFER];

    for (int r = 0; r < DEV_REGION_COUNT; r++)
      peerAddrs[i][r] = dataBases[r];

    if (i == myRank) {
      for (int r = 0; r < DEV_REGION_COUNT; r++)
        devParams->peerPtrTables[r] = ptrTables[r];
      // Local-only region (no peer table): expose the data base directly.
      devParams->llEpochs = dataBases[DEV_REGION_LL_EPOCHS];
    }
  }

  // Copy each peer-visible region's per-GPU address array to its device pointer table
  for (int r = 0; r < DEV_REGION_COUNT; r++) {
    if (!hostParams->regionConfig[r].peerVisible) continue;
    void* perGpuAddrs[NCCL_MAX_LOCAL_RANKS];
    for (int i = 0; i < nGpus; i++)
      perGpuAddrs[i] = peerAddrs[i][r];
    HGGCCHECK(hggcMemcpy(devParams->peerPtrTables[r], perGpuAddrs, EXT_KERNEL_PTRS_SIZE, hggcMemcpyHostToDevice));
  }

  // Zero-initialize LL epochs on the local rank's device memory
  HGGCCHECK(hggcMemset(devParams->llEpochs, 0, hostParams->regionConfig[DEV_REGION_LL_EPOCHS].elementSize));
  // Zero-initialize LL buffer to prevent spurious epoch flag matches on first launch
  HGGCCHECK(hggcMemset(peerAddrs[myRank][DEV_REGION_LL_BUFFER], 0, hostParams->llBufferSize));

  // for dump
  extParams->myBarrierIn      = (uint32_t*)peerAddrs[myRank][DEV_REGION_BARRIER_IN];
  extParams->myBarrierPre     = (uint32_t*)peerAddrs[myRank][DEV_REGION_BARRIER_PRE];
  extParams->myBarrierReduce  = (uint32_t*)peerAddrs[myRank][DEV_REGION_BARRIER_REDUCE];
  extParams->myBarrierPost    = (uint32_t*)peerAddrs[myRank][DEV_REGION_BARRIER_POST];
  extParams->myBarrierOut     = (uint32_t*)peerAddrs[myRank][DEV_REGION_BARRIER_OUT];

  return ncclSuccess;
}

ncclResult_t pluginExecute(void* params, hggcStream_t stream) {
  struct extKernelDevParams* devParams = reinterpret_cast<struct extKernelParams*>(params)->devParams;
  struct extKernelHostParams* hostParams = &reinterpret_cast<struct extKernelParams*>(params)->hostParams;
  switch (hostParams->currentFunc) {
    case ncclFuncAllGather: {
      NCCLCHECK(customAllGatherImpl(*devParams, *hostParams, stream));
      break;
    }
    case ncclFuncReduceScatter: {
      NCCLCHECK(customReduceScatterImpl(*devParams, *hostParams, stream));
      break;
    }
    case ncclFuncAllReduce: {
      NCCLCHECK(customAllReduceImpl(*devParams, *hostParams, stream));
      break;
    }
    default:
      ERROR("EXT: unsupported collective func %d", hostParams->currentFunc);
      return ncclInvalidUsage;
  }
  return ncclSuccess;
}


void dumpBarrierToFile(FILE* file, const char* barrierName, uint32_t* hostBarriers, uint32_t* devBarriers, ssize_t flagSize, int nRanks) {
  fprintf(file, "%s Barrier:\n", barrierName);
  ExtKernelEit::getInstance().memDump(hostBarriers, (HGdeviceptr)(devBarriers), flagSize);
  for (int b = 0; b < MAX_ALL_REDUCE_BLOCKS + 1; ++b) {
    fprintf(file, "block %d: ", b);
    for (int n = 0; n < nRanks; ++n) {
      fprintf(file, "%u ", hostBarriers[b *nRanks + n]);
    }
    fprintf(file, "\n");
  }
}

ncclResult_t pluginDump(void* params, const char* filename) {
  FILE* dumpFile = fopen(filename, "a");
  if (dumpFile == NULL) return ncclSuccess;
  INFO(PCCL_DUMP, "EXT: dump params:%p to %s", params, filename);
  fprintf(dumpFile, "\n///////////////////////////////////\n");
  fprintf(dumpFile, "//////////  Ext Kernel Status /////\n");
  fprintf(dumpFile, "///////////////////////////////////\n");

  struct extKernelParams* extParams = reinterpret_cast<struct extKernelParams*>(params);
  struct extKernelHostParams* hostParams = &extParams->hostParams;
  ncclResult_t result = ncclSuccess;
  uint32_t* hostBarrierBuffer = NULL;
  ssize_t barrierSize = hostParams->barrierSize;
  NCCLCHECKGOTO(extCalloc(&hostBarrierBuffer, barrierSize / sizeof(int32_t)), result, cleanup);

  dumpBarrierToFile(dumpFile, "In", hostBarrierBuffer, extParams->myBarrierIn, barrierSize, hostParams->nRanks);
  dumpBarrierToFile(dumpFile, "Pre", hostBarrierBuffer, extParams->myBarrierPre, barrierSize, hostParams->nRanks);
  dumpBarrierToFile(dumpFile, "Reduce", hostBarrierBuffer, extParams->myBarrierReduce, barrierSize, hostParams->nRanks);
  dumpBarrierToFile(dumpFile, "Post", hostBarrierBuffer, extParams->myBarrierPost, barrierSize, hostParams->nRanks);
  dumpBarrierToFile(dumpFile, "Out", hostBarrierBuffer, extParams->myBarrierOut, barrierSize, hostParams->nRanks);

  fprintf(dumpFile, "///////////////////////////////////\n");
  result = ncclSuccess;

cleanup:
  if(dumpFile) fclose(dumpFile);
  NCCLCHECK(extFree(hostBarrierBuffer));
  return result;
}

ncclResult_t pluginDestroy(void* params) {
  if (params == NULL) return ncclSuccess;
  struct extKernelDevParams* devParams = reinterpret_cast<struct extKernelParams*>(params)->devParams;
  struct extKernelHostParams* hostParams = &(reinterpret_cast<struct extKernelParams*>(params)->hostParams);
  if (devParams) {
    hggcFreeHost(devParams);
    devParams = NULL;
  }
  NCCLCHECK(extFree(hostParams->funcStrategy));
  NCCLCHECK(extFree(params));
  return ncclSuccess;
}

#define PLUGIN_NAME "CustomCollectives"

__attribute__((visibility("default")))
pcclExtKernel_v1_t pcclExtKernelPlugin_v1 = {
  .name = PLUGIN_NAME,
  .kernelPluginInit = pluginInit,
  .kernelPluginConfigWorkspace = pluginConfigWorkspace,
  .kernelPluginAvailCheck = pluginAvailCheck,
  .kernelPluginBuildParams = pluginBuildParams,
  .kernelPluginExecute = pluginExecute,
  .kernelPluginDump = pluginDump,
  .kernelPluginDestroy = pluginDestroy
};
