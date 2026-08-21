/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_MEM_BARRIER_H_
#define _PCCL_DEVICE_MEM_BARRIER_H_
#include "impl/core__types.h"

struct pcclLsaBarrierHandle;

PCCL_EXTERN_C __host__ pcclResult_t pcclLsaBarrierCreateRequirement(pcclTeam_t, int nBarriers, pcclLsaBarrierHandle_t* outHandle, pcclDevResourceRequirements_t* outReq);
#if PCCL_CHECK_HGGCCC
template<typename Coop>
struct pcclLsaBarrierSession_internal;

template<typename Coop>
struct pcclLsaBarrierSession: pcclLsaBarrierSession_internal<Coop> {
  PCCL_DEVICE_INLINE pcclLsaBarrierSession(Coop, pcclDevComm const&, pcclTeam, pcclLsaBarrierHandle, uint32_t index, bool multimem=false, pcclMultimemHandle mmHandle={});

  PCCL_DEVICE_INLINE pcclLsaBarrierSession(Coop, pcclDevComm const&, pcclTeamTagLsa, uint32_t index, bool multimem=false);

  PCCL_DEVICE_INLINE ~pcclLsaBarrierSession();

  pcclLsaBarrierSession(pcclLsaBarrierSession const&) = delete; // Sessions are not copyable

  PCCL_DEVICE_INLINE void arrive(Coop, hggc::memory_order);
  PCCL_DEVICE_INLINE void wait(Coop, hggc::memory_order);
  PCCL_DEVICE_INLINE void sync(Coop, hggc::memory_order);
  PCCL_DEVICE_INLINE pcclResult_t wait(Coop, hggc::memory_order, uint64_t timeoutCycles);
  PCCL_DEVICE_INLINE pcclResult_t sync(Coop, hggc::memory_order, uint64_t timeoutCycles);
};
#endif // PCCL_CHECK_HGGCCC

#endif // _PCCL_DEVICE_MEM_BARRIER_H_
