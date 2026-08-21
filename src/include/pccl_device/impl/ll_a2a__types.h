/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_LL_A2A__TYPES_H_
#define _PCCL_DEVICE_LL_A2A__TYPES_H_
#include "../ll_a2a.h"
#include "core__types.h"

struct pcclLLA2AHandle {
  pcclDevResourceHandle_t bufHandle;
  uint32_t nSlots;
};

#if PCCL_CHECK_HGGCCC
template<typename Coop>
struct pcclLLA2ASession_internal {
  Coop coop;
  pcclDevComm const& comm;
  pcclTeam team;
  pcclLLA2AHandle handle;
  int block;
  int pitch;
  bool multimem;
  pcclMultimemHandle mmHandle;
  uint32_t epoch;
  uint32_t slotsOffset;

  PCCL_DEVICE_INLINE uint32_t calcSlotOffset() const {
    return block*(1 + 2*handle.nSlots) + 1 + (epoch & 1)*handle.nSlots;
  }
};
#endif

#endif // _PCCL_DEVICE_LL_A2A__TYPES_H_
