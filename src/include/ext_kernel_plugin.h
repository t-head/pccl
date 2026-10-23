/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INT_EXT_KERNEL_H_
#define NCCL_INT_EXT_KERNEL_H_

#include "pccl_internal.h"
#include "devcomm.h"
#include <sys/types.h>

struct pcclExtKernelInitInfo {
  int rank;
  int nRanks;
  int localRanks;
  int compCap;
  PPU_TYPE ppuType;
  pcclICNRemoteDeviceType_t icnRmtType;
  struct ncclDispatchStrategy defaultStrategy;
  const int* topoArray;  // localRanks x localRanks, pre-filled by core
  // Device-visible comm abort flag, set by pcclCommAbort
  uint32_t* abortFlagDev;
};

/*========================================================================
 * pcclExtTaskColl — lightweight task descriptor passed to ext-kernel plugin.
 * The pccl core converts from its internal ncclTaskColl to this struct
 * before calling plugin functions.
 *========================================================================*/
struct pcclExtTaskColl {
  ncclFunc_t func;
  ncclDevRedOp_t op;
  ncclDataType_t datatype;
  size_t count;
  void const* sendbuff;
  void* recvbuff;
  int regUsed;
  uintptr_t sendbuffOffset;
  uintptr_t recvbuffOffset;
  uintptr_t* sendbuffRmtAddrs;
  uintptr_t* recvbuffRmtAddrs;
};

typedef struct {
  const char* name;

  pcclResult_t (*kernelPluginInit)(const struct pcclExtKernelInitInfo* info, void** params, ssize_t* workspaceSize, ncclDebugLogger_t logfn);
  pcclResult_t (*kernelPluginConfigWorkspace)(void** ptrs, int nGpus, void* params);
  pcclResult_t (*kernelPluginAvailCheck)(const struct pcclExtTaskColl* task, const void* params, bool* usingExtKernel);
  pcclResult_t (*kernelPluginBuildParams)(const struct pcclExtTaskColl* task, void* params);
  pcclResult_t (*kernelPluginExecute)(void* params, hggcStream_t stream);
  pcclResult_t (*kernelPluginDump)(void* params, const char* filename);
  pcclResult_t (*kernelPluginDestroy)(void* params);
} pcclExtKernel_v1_t;

typedef pcclExtKernel_v1_t pcclExtKernel_t;

#define PCCL_EXT_KERNEL_PLUGIN_SYMBOL "pcclExtKernelPlugin_v1"

ncclResult_t pcclExtKernelPluginLoad(ncclComm_t comm);

ncclResult_t pcclExtKernelPluginUnload(ncclComm_t comm);
#endif
