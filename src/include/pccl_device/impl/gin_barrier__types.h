/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_GIN_BARRIER__TYPES_H_
#define _PCCL_DEVICE_GIN_BARRIER__TYPES_H_
#include "../gin_barrier.h"
#include "core__types.h"
#include "gin__types.h"

struct pcclGinBarrierHandle {
  pcclGinSignal_t signal0;
  pcclDevResourceHandle_t unused;
};

#if PCCL_CHECK_HGGCCC
template<typename Coop>
struct pcclGinBarrierSession_internal {
  Coop coop;
  pcclGin net;
  pcclTeam team;
  pcclGinBarrierHandle handle;
  int index;
  pcclGinSignal_t signal;

  template<bool EnableTimeout>
  PCCL_DEVICE_INLINE pcclResult_t syncInternal(Coop, hggc::memory_order ord, pcclGinFenceLevel fence,
                                               uint64_t timeoutCycles);
};
#endif // PCCL_CHECK_HGGCCC

#endif // _PCCL_DEVICE_GIN_BARRIER__TYPES_H_
