/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_MEM_BARRIER_H_
#define _NCCL_DEVICE_MEM_BARRIER_H_
#include "impl/core__types.h"

struct ncclLsaBarrierHandle;

NCCL_EXTERN_C __host__ pcclResult_t ncclLsaBarrierCreateRequirement(ncclTeam_t, int nBarriers, ncclLsaBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq);
#if NCCL_CHECK_HGGCCC
template<typename Coop>
struct ncclLsaBarrierSession_internal;

template<typename Coop>
struct ncclLsaBarrierSession: ncclLsaBarrierSession_internal<Coop> {
  NCCL_DEVICE_INLINE ncclLsaBarrierSession(Coop, ncclDevComm const&, ncclTeam, ncclLsaBarrierHandle, uint32_t index, bool multimem=false, ncclMultimemHandle mmHandle={});

  NCCL_DEVICE_INLINE ncclLsaBarrierSession(Coop, ncclDevComm const&, ncclTeamTagLsa, uint32_t index, bool multimem=false);

  NCCL_DEVICE_INLINE ~ncclLsaBarrierSession();

  ncclLsaBarrierSession(ncclLsaBarrierSession const&) = delete; // Sessions are not copyable

  NCCL_DEVICE_INLINE void arrive(Coop, hggc::memory_order);
  NCCL_DEVICE_INLINE void wait(Coop, hggc::memory_order);
  NCCL_DEVICE_INLINE void sync(Coop, hggc::memory_order);
  NCCL_DEVICE_INLINE pcclResult_t wait(Coop, hggc::memory_order, uint64_t timeoutCycles);
  NCCL_DEVICE_INLINE pcclResult_t sync(Coop, hggc::memory_order, uint64_t timeoutCycles);
};
#endif // NCCL_CHECK_HGGCCC

#endif // _NCCL_DEVICE_MEM_BARRIER_H_
