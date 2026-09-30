/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Windows-only stub for pccl_device/gin.h. Provides minimal types so that
 * barrier/gin_barrier headers compile without pulling in real GIN device headers
 * (gin/gin_device_common.h etc.). Used only when building on Windows.
 *************************************************************************/

#ifndef _PCCL_DEVICE_GIN_SESSION_H_
#define _PCCL_DEVICE_GIN_SESSION_H_

#if !defined(PCCL_OS_WINDOWS)
#error "gin_win_stub.h is for Windows builds only"
#endif

#include "core.h"
#include "net_device.h"

/* No GIN backends on Windows */
#define PCCL_GIN_BACKEND_MASK_ALL 0u

#if PCCL_CHECK_HGGCCC
template<unsigned backendMask>
struct pcclGin_BackendMask {
  pcclDevComm const& comm;
  uint32_t nConnections:8, connectionId:8, _ginBackend:8;
  uint32_t contextId;
  PCCL_DEVICE_INLINE pcclGin_BackendMask(pcclDevComm const& c, int contextIndex)
    : comm(c), nConnections(0), connectionId(0), _ginBackend(0), contextId(0) {}
};

template<pcclNetDeviceType backend>
using pcclGin_BackendOne = pcclGin_BackendMask<(1u<<(int)backend)>;

using pcclGin = pcclGin_BackendMask<PCCL_GIN_BACKEND_MASK_ALL>;
#endif

/* GIN scratch types (mirrors gin_scratch.h / gin_scratch__types.h) */
struct pcclGinOutboxHandle {
  pcclDevResourceHandle bufHandle;
  pcclGinCounter_t counter0;
  uint32_t size_log2;
};
struct pcclGinInboxA2AHandle {
  pcclDevResourceHandle bufHandle;
  pcclGinSignal_t signals;
  uint32_t size_log2;
  uint32_t nPeers_rcp32;
};
struct pcclGinSyncHandle {
  pcclGinSignal_t railSignals;
};

PCCL_EXTERN_C __host__ pcclResult_t pcclGinOutboxCreateRequirement(
  int nBlocks, int size_log2,
  pcclGinOutboxHandle* outHandle, pcclDevResourceRequirements* outReq
);
PCCL_EXTERN_C __host__ pcclResult_t pcclGinInboxA2ACreateRequirement(
  pcclTeam peers, int nBlocks, int size_log2,
  pcclGinInboxA2AHandle* outHandle, pcclDevResourceRequirements* outReq
);

#endif /* _PCCL_DEVICE_GIN_SESSION_H_ */
