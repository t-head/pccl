/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INT_NET_H_
#define NCCL_INT_NET_H_

#include <sys/socket.h>
#include "pccl_internal.h"
#include "pccl_net_internal.h"
#include "checks.h"

#define EIC_PLUGIN_NAME "accl-lite"
extern bool nccl_use_eic;
extern int nccl_use_u2mm_host;
extern int nccl_use_u2mm_dev;
typedef char ncclNetHandle_t[NCCL_NET_HANDLE_MAXSIZE];

ncclResult_t ncclNetInit(ncclComm_t comm);
ncclResult_t ncclNetFinalize(ncclComm_t comm);
int ncclNetVersion(ncclComm_t comm);
bool ncclEicNet(ncclNet_t* net);

// Test whether the current GPU support GPU Direct RDMA.
ncclResult_t ncclGpuGdrSupport(int* gdrSupport);

extern ncclNet_t ncclNetIb;
extern ncclNet_t ncclNetSocket;

#endif
