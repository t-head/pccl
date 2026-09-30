/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_NET_INTTERNAL_H_
#define PCCL_NET_INTTERNAL_H_

#include "pccl_net.h"
#include "pccl_internal.h"
#include <stdint.h>

#define NCCL_NET_HANDLE_MAXSIZE 128

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_HGGC 0x2
#define NCCL_PTR_DMABUF 0x4

#define NCCL_NET_MR_FLAG_FORCE_SO 0x1

#define NCCL_NET_SIGNAL_OP_INC 0x1
#define NCCL_NET_SIGNAL_OP_ADD 0x2

// Maximum number of requests per comm object
#define NCCL_NET_MAX_REQUESTS 8

typedef pcclNetProperties_v6_t ncclNetProperties_v6_t;
typedef pcclNetProperties_v4_t ncclNetProperties_v4_t;
typedef ncclNetProperties_v6_t ncclNetProperties_t;
typedef pcclNet_v6_t ncclNet_v6_t;
typedef pcclNet_v5_t ncclNet_v5_t;
typedef pcclNet_v4_t ncclNet_v4_t;
typedef ncclNet_v6_t ncclNet_t;
#define NCCL_PLUGIN_SYMBOL pcclNetPlugin_v6
typedef pcclCollNet_v6_t ncclCollNet_v6_t;
typedef pcclCollNet_v5_t ncclCollNet_v5_t;
typedef pcclCollNet_v4_t ncclCollNet_v4_t;
typedef ncclCollNet_v6_t ncclCollNet_t;
#define NCCL_COLLNET_PLUGIN_SYMBOL pcclCollNetPlugin_v6

#endif // end include guard
