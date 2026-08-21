/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_MEM_BARRIER__TYPES_H_
#define _PCCL_DEVICE_MEM_BARRIER__TYPES_H_
#include "../lsa_barrier.h"
#include "core__types.h"

struct pcclLsaBarrierHandle {
  pcclDevResourceHandle_t bufHandle;
  int nBarriers;
};

#if PCCL_CHECK_HGGCCC
template<typename Coop>
struct pcclLsaBarrierSession_internal {
  Coop coop;
  pcclDevComm const& comm;
  pcclTeam team;
  pcclLsaBarrierHandle handle;
  int index;
  bool multimem;
  pcclMultimemHandle mmHandle;
  uint32_t epoch;

  PCCL_DEVICE_INLINE uint32_t* ucInbox(int owner, int peer) {
    uint32_t* state = (uint32_t*)pcclGetResourceBufferPeerPointer(comm, handle.bufHandle, team, owner);
    return state + 3*handle.nBarriers + index*team.nRanks + peer;
  }

  template<bool EnableTimeout>
  PCCL_DEVICE_INLINE pcclResult_t waitInternal(Coop, hggc::memory_order order, uint64_t timeoutCycles);
};
#endif // PCCL_CHECK_HGGCCC

#endif // _PCCL_DEVICE_MEM_BARRIER__TYPES_H_
