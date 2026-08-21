/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_GIN_BARRIER_H_
#define _PCCL_DEVICE_GIN_BARRIER_H_
#include "core.h"
#if defined(PCCL_OS_WINDOWS)
#include "gin_win_stub.h"
#else
#include "gin.h"
#endif

struct pcclGinBarrierHandle;

PCCL_EXTERN_C __host__ pcclResult_t pcclGinBarrierCreateRequirement(pcclComm_t, pcclTeam_t, int nBarriers, pcclGinBarrierHandle_t* outHandle, pcclDevResourceRequirements_t* outReq);

#if PCCL_CHECK_HGGCCC
enum class pcclGinFenceLevel {
  Relaxed
};

template<typename Coop>
struct pcclGinBarrierSession_internal;

template<typename Coop>
struct pcclGinBarrierSession: pcclGinBarrierSession_internal<Coop> {
  PCCL_DEVICE_INLINE pcclGinBarrierSession(Coop, pcclGin, pcclTeam, pcclGinBarrierHandle, uint32_t index);
  PCCL_DEVICE_INLINE pcclGinBarrierSession(Coop, pcclGin, pcclTeamTagRail, uint32_t index);
  PCCL_DEVICE_INLINE pcclGinBarrierSession(Coop, pcclGin, pcclTeamTagWorld, uint32_t index);

  PCCL_DEVICE_INLINE ~pcclGinBarrierSession();

  pcclGinBarrierSession(pcclGinBarrierSession const&) = delete; // Sessions are not copyable

  PCCL_DEVICE_INLINE void sync(Coop, hggc::memory_order, pcclGinFenceLevel);
  PCCL_DEVICE_INLINE pcclResult_t sync(Coop, hggc::memory_order, pcclGinFenceLevel, uint64_t timeoutCycles);
};
#endif // PCCL_CHECK_HGGCCC

#endif // _PCCL_DEVICE_GIN_BARRIER_H_
