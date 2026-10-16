/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PCCL_NET_IB_GIN_PROVIDER_H_
#define PCCL_NET_IB_GIN_PROVIDER_H_

#include "common.h"
#include "pccl_gin.h"

// Host-side registration of a GPU-direct implementation. GDA is the mode;
// the provider selects the actual QP/CQ layout and device backend. An external
// GIN plugin remains independent of this built-in IB registry.
struct ncclGinIbProviderOps {
  enum ncclIbProvider provider;
  ncclGin_t* gin;
  bool needsPeerMem;
};

const ncclGinIbProviderOps* ncclGinIbProviderOpsFor(enum ncclIbProvider provider);
struct ncclGinIbCollComm;
ncclResult_t ncclGinIbCheckPeerProvider(ncclGinIbCollComm* comm, ncclIbProvider provider);
int ncclGinIbProviderDeviceCount(enum ncclIbProvider provider);
int ncclGinIbProviderDeviceIndex(enum ncclIbProvider provider, int ginDev);

#endif
