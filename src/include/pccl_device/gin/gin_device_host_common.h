/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_GIN_DEVICE_HOST_COMMON_H_
#define _PCCL_GIN_DEVICE_HOST_COMMON_H_

#include <hggc.h>
#include "../net_device.h"
#include "../core.h"  // for pcclGin{Signal|Counter}_t

#define PCCL_GIN_MAX_CONNECTIONS 4

typedef struct pcclGinGpuCtx *pcclGinGpuCtx_t;
typedef void *pcclGinWindow_t;

typedef enum pcclGinSignalOp_t {
  pcclGinSignalInc = 0,
  pcclGinSignalAdd,
} pcclGinSignalOp_t;

#endif
