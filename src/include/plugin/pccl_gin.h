/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_GIN_H_
#define NCCL_GIN_H_

#include "pccl_internal.h"
#include "pccl_common.h"
#include "pccl_device/net_device.h"
#include <stdint.h>

#define NCCL_GIN_HANDLE_MAXSIZE 128
#define MAX_GIN_SIZE (1024*1024*1024L) // Rather than send INT_MAX which is 2G-1, send a power of two.

// Max number of ncclNet objects which can live in the same process
#ifndef NCCL_GIN_MAX_PLUGINS
#define NCCL_GIN_MAX_PLUGINS 16
#endif

#define NCCL_GIN_SIGNAL_OP_INC 0x1
#define NCCL_GIN_SIGNAL_OP_ADD 0x2

#define NCCL_NET_ID_UNDEF -1

#include "gin/gin_v13.h"
#include "gin/gin_v12.h"
#include "gin/gin_v11.h"

typedef ncclGin_v13_t ncclGin_t;
typedef ncclGinConfig_v13_t ncclGinConfig_t;

#define NCCL_GIN_PLUGIN_SYMBOL pcclGinPlugin_v13

#endif // end include guard
