/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INT_GIN_H_
#define NCCL_INT_GIN_H_

#include "pccl_gin.h"

ncclResult_t ncclGinInit(struct pcclComm* comm);
ncclResult_t ncclGinInitFromParent(struct pcclComm* comm, struct pcclComm* parent);
ncclResult_t ncclGinGetDevCount(int ginPluginIndex, int* nPhysDev, int* nVirtDev);
ncclResult_t ncclGinFinalize(struct pcclComm* comm);

extern ncclGin_t ncclGinIb;
extern ncclGin_t ncclGinIbGdaki;
extern ncclGin_t ncclGinIbProxy;

#endif
