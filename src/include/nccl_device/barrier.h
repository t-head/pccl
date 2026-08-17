/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_BARRIER_H_
#define _NCCL_DEVICE_BARRIER_H_
#include "impl/core__types.h"
#include "impl/lsa_barrier__types.h"
#include "impl/gin_barrier__types.h"

#if NCCL_CHECK_HGGCCC
template<typename Coop>
struct ncclBarrierSession_internal;

template<typename Coop>
struct ncclBarrierSession: ncclBarrierSession_internal<Coop> {
  // Full featured constructor:
  NCCL_DEVICE_INLINE ncclBarrierSession(
    Coop, ncclTeam innerTeam, ncclTeam outerTeam, ncclGin,
    ncclLsaBarrierHandle innerBarHandle,
    ncclGinBarrierHandle outerBarHandle,
    uint32_t index,
    bool multimem=false, ncclMultimemHandle innerMmHandle={}
  );
  // Convenience constructors for baked in teams:
  NCCL_DEVICE_INLINE ncclBarrierSession(
    Coop, ncclTeamTagWorld, ncclGin, uint32_t index, bool multimem=false
  );
  NCCL_DEVICE_INLINE ncclBarrierSession(
    Coop, ncclTeamTagLsa, ncclDevComm const&, uint32_t index, bool multimem=false
  );
  NCCL_DEVICE_INLINE ncclBarrierSession(
    Coop, ncclTeamTagRail, ncclGin, uint32_t index
  );

  ncclBarrierSession(ncclBarrierSession const&) = delete; // Sessions are not copyable

  NCCL_DEVICE_INLINE ncclLsaBarrierSession<Coop>& lsaBarrier();
  NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>& ginBarrier();

  NCCL_DEVICE_INLINE void sync(Coop, hggc::memory_order, ncclGinFenceLevel);
  NCCL_DEVICE_INLINE pcclResult_t sync(Coop, hggc::memory_order, ncclGinFenceLevel, uint64_t timeoutCycles);
};
#endif // NCCL_CHECK_HGGCCC

#endif // _NCCL_DEVICE_BARRIER_H_
