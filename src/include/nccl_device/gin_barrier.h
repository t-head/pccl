/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_BARRIER_H_
#define _NCCL_DEVICE_GIN_BARRIER_H_
#include "core.h"
#if defined(NCCL_OS_WINDOWS)
#include "gin_win_stub.h"
#else
#include "gin.h"
#endif

struct ncclGinBarrierHandle;
struct ncclGinInboxA2AHandle;
struct ncclGinOutboxHandle;

NCCL_EXTERN_C __host__ pcclResult_t ncclGinBarrierCreateRequirement(pcclComm_t, ncclTeam_t, int nBarriers, ncclGinBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq);

// GIN scratch (inbox/outbox) requirement declarations.
NCCL_EXTERN_C __host__ pcclResult_t ncclGinOutboxCreateRequirement(
    int nBlocks, int size_log2,
    struct ncclGinOutboxHandle* outHandle,
    ncclDevResourceRequirements_t* outReq);

NCCL_EXTERN_C __host__ pcclResult_t ncclGinInboxA2ACreateRequirement(
    ncclTeam_t peers, int nBlocks, int size_log2,
    struct ncclGinInboxA2AHandle* outHandle,
    ncclDevResourceRequirements_t* outReq);

enum class ncclGinFenceLevel {
  Relaxed
};

#if NCCL_CHECK_HGGCCC
template<typename Coop>
struct ncclGinBarrierSession_internal;

template<typename Coop>
struct ncclGinBarrierSession: ncclGinBarrierSession_internal<Coop> {
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGin, ncclTeam, ncclGinBarrierHandle, uint32_t index);
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGin, ncclTeamTagRail, uint32_t index);
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGin, ncclTeamTagWorld, uint32_t index);

  NCCL_DEVICE_INLINE ~ncclGinBarrierSession();

  ncclGinBarrierSession(ncclGinBarrierSession const&) = delete; // Sessions are not copyable

  NCCL_DEVICE_INLINE void sync(Coop, hggc::memory_order, ncclGinFenceLevel);
  NCCL_DEVICE_INLINE pcclResult_t sync(Coop, hggc::memory_order, ncclGinFenceLevel, uint64_t timeoutCycles);
};
#endif // NCCL_CHECK_HGGCCC

#endif // _NCCL_DEVICE_GIN_BARRIER_H_
