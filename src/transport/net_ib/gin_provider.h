/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PCCL_NET_IB_GIN_PROVIDER_H_
#define PCCL_NET_IB_GIN_PROVIDER_H_

#include "common.h"
#include "pccl_gin.h"

// Built-in GDA provider interface. Each provider defines its QP/CQ layout and
// device implementation. External GIN plugins use the separate plugin interface.
struct ncclGinIbProviderOps {
  enum ncclIbProvider provider;
  ncclGin_t* gin;
  ncclResult_t (*probe)();
};

const ncclGinIbProviderOps* ncclGinIbProviderOpsFor(enum ncclIbProvider provider);
struct ncclGinIbCollComm;
ncclResult_t ncclGinIbCheckPeerProvider(ncclGinIbCollComm* comm, ncclIbProvider provider);
int ncclGinIbProviderDeviceCount(enum ncclIbProvider provider);
int ncclGinIbProviderDeviceIndex(enum ncclIbProvider provider, int ginDev);

#endif
