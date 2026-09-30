/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_BARRIER_H_
#define _PCCL_DEVICE_BARRIER_H_
#include "impl/core__types.h"
#include "impl/lsa_barrier__types.h"
#include "impl/gin_barrier__types.h"

#if PCCL_CHECK_HGGCCC
template<typename Coop>
struct pcclBarrierSession_internal;

template<typename Coop>
struct pcclBarrierSession: pcclBarrierSession_internal<Coop> {
  // Full featured constructor:
  PCCL_DEVICE_INLINE pcclBarrierSession(
    Coop, pcclTeam innerTeam, pcclTeam outerTeam, pcclGin,
    pcclLsaBarrierHandle innerBarHandle,
    pcclGinBarrierHandle outerBarHandle,
    uint32_t index,
    bool multimem=false, pcclMultimemHandle innerMmHandle={}
  );
  // Convenience constructors for baked in teams:
  PCCL_DEVICE_INLINE pcclBarrierSession(
    Coop, pcclTeamTagWorld, pcclGin, uint32_t index, bool multimem=false
  );
  PCCL_DEVICE_INLINE pcclBarrierSession(
    Coop, pcclTeamTagLsa, pcclDevComm const&, uint32_t index, bool multimem=false
  );
  PCCL_DEVICE_INLINE pcclBarrierSession(
    Coop, pcclTeamTagRail, pcclGin, uint32_t index
  );

  pcclBarrierSession(pcclBarrierSession const&) = delete; // Sessions are not copyable

  PCCL_DEVICE_INLINE pcclLsaBarrierSession<Coop>& lsaBarrier();
  PCCL_DEVICE_INLINE pcclGinBarrierSession<Coop>& ginBarrier();

  PCCL_DEVICE_INLINE void sync(Coop, hggc::memory_order, pcclGinFenceLevel);
  PCCL_DEVICE_INLINE pcclResult_t sync(Coop, hggc::memory_order, pcclGinFenceLevel, uint64_t timeoutCycles);
};
#endif // PCCL_CHECK_HGGCCC

#endif // _PCCL_DEVICE_BARRIER_H_
