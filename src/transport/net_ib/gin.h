/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_NET_IB_GIN_H_
#define _PCCL_NET_IB_GIN_H_

#include <stddef.h>
#include <stdint.h>
#include "pccl_internal.h"
#include "pccl_gin.h"
#include "common.h"

struct ncclGinIbCollComm {
  void*         ctx;
  int           rank;
  int           nranks;
  void*         recvComm;
  void*         sendComm;
  int           dev;
  struct {
    struct ibv_context* context;
    struct ibv_pd *pd;
  }ib;
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v6_t* props);
  ncclResult_t (*allGather)(struct ncclGinIbCollComm *cComm, void *srcBuf, void *recvBuf, size_t len);
  ncclResult_t (*allToAll)(struct ncclGinIbCollComm *cComm, void *srcBuf, void *recvBuf, size_t len);
  ncclResult_t (*getGidIndex)(struct ibv_context *context, uint8_t portNum, struct ibv_port_attr* portAttr, int *gidIndex);
};

ncclResult_t ncclGinIbInitType(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction,
                               int ginType, ncclGin_t* ginIb, ncclIbProvider requiredProvider = IB_PROVIDER_UNKNOWN);
ncclResult_t ncclGinIbGdrGpuSupport();
void ncclGinIbFillV12Properties(const ncclNetProperties_v6_t& src, ncclNetProperties_v12_t* dst);
ncclResult_t ncclGinIbConnect(void* ctx, void* handles[], int nranks, int rank,
                              void* listenComm, void** collComm);
ncclResult_t ncclGinIbCloseColl(void* collComm);
ncclResult_t ncclGinIbCloseListen(void* listenComm);
ncclResult_t ncclGinIbFinalize(void* ctx);

#endif
