/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_GIN_HOST_H_
#define _PCCL_GIN_HOST_H_

#include "allocator.h"
#include "pccl_internal.h"
#include "pccl_net_internal.h"
#include "pccl_gin.h"
#include "pccl_device/gin/gin_device_host_common.h"
#include "pccl_device/gin/gin_gdaki_device_host_common.h"
#include <thread>
#include <mutex>
#include <condition_variable>

struct pcclGinStateDevComm {
  int contextCount;
  void* ginCtx[NCCL_GIN_MAX_CONNECTIONS];
  ncclNetDeviceHandle_t* devHandles[NCCL_GIN_MAX_CONNECTIONS];
  struct pcclGinStateDevComm* next;
};

struct pcclGinState {
  ncclGin_t* ncclGin;
  void* ginInstance;
  bool connected;
  ncclGinType_t ginType;
  int ginCommCount;
  void* ginComms[NCCL_GIN_MAX_CONNECTIONS];
  // GIN uses the v12-shaped properties (netDeviceType, vProps, railId, planeId)
  // returned by the v13 ABI getProperties callback. Not the legacy v6 net type.
  ncclNetProperties_v12_t ginProps[NCCL_GIN_MAX_CONNECTIONS];
  int needsProxyProgress;  // Whether we need to progress GIN operations with the proxy
  int ginProgress;         // GIN progress is enabled
  std::thread thread;
  std::mutex mutex;
  std::condition_variable cond;
  ncclResult_t asyncResult;
  int ginVersion;

  struct pcclGinStateDevComm* devComms;
  ncclGinConnectionType_t ginConnectionType;
};

// Providers own the descriptor allocated by CreateHandle.
ncclResult_t ncclGinGdakiCreateHandle(pcclGinGdakiProvider provider, uint32_t providerVersion,
                                      void* contexts, ncclNetDeviceHandle_t* handle);

extern int64_t ncclParamGinType();

// Sets the local GIN type for comm. The GIN type that is set for comm is the
// GIN type supported by the call process itself, without taking into account
// (1) GIN support of other ranks, and (2) additional local constraints like
// cross-NIC
ncclResult_t setLocalGinType(struct pcclComm* comm);
// Get the GIN type from comm. ginType is set to the GIN type that can be used
// by the comm to communicate with other nodes.
ncclResult_t getGlobalGinType(struct pcclComm* comm, ncclGinType_t* ginType);
ncclResult_t getGlobalRailedGinType(struct pcclComm* comm, ncclGinType_t* ginType);

ncclResult_t ncclGinConnectOnce(struct pcclComm* comm);
ncclResult_t ncclGinHostFinalize(struct pcclComm* comm);
ncclResult_t ncclGinDevCommSetup(struct pcclComm* comm, struct ncclDevCommRequirements const* reqs, struct ncclDevComm* devComm);
ncclResult_t ncclGinDevCommFree(struct pcclComm* comm, struct ncclDevComm const* devComm);
ncclResult_t ncclGinRegister(struct pcclComm* comm, void* address, size_t size,
                             void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS],
                             ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS], int winFlags,
                             bool multiSegment = false, int memType = NCCL_PTR_HGGC);
ncclResult_t ncclGinDeregister(struct pcclComm* comm, void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS]);

ncclResult_t ncclGinQueryLastError(struct pcclGinState* ginState, bool* hasError);

#endif
