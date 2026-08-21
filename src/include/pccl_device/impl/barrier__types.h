/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_BARRIER__TYPES_H_
#define _PCCL_DEVICE_BARRIER__TYPES_H_
#include "../barrier.h"
#include "../utility.h"

#if PCCL_CHECK_HGGCCC
template<typename Coop>
struct pcclBarrierSession_internal {
  Coop coop;
  pccl::utility::Optional<pcclGin> gin;
  pccl::utility::Optional<pcclLsaBarrierSession<Coop>> innerLsaBar;
  pccl::utility::Optional<pcclGinBarrierSession<Coop>> outerGinBar;

  template<typename GinInit, typename InnerInit, typename OuterInit>
  PCCL_DEVICE_INLINE pcclBarrierSession_internal(
      Coop coop, GinInit ginInit, InnerInit innerInit, OuterInit outerInit
    ):
    coop(coop), gin{ginInit}, innerLsaBar{innerInit}, outerGinBar{outerInit} {
  }
};
#endif // PCCL_CHECK_HGGCCC

#endif // _PCCL_DEVICE_BARRIER__TYPES_H_
