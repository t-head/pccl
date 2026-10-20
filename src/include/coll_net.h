/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef COLL_NET_H_
#define COLL_NET_H_

#include "pccl_net_internal.h"
#include "pccl_internal.h"

typedef char collNetHandle_t[NCCL_NET_HANDLE_MAXSIZE];

// Translation to external API
static const char* collNetName(ncclComm_t comm) { return comm->ncclCollNet->name; }
static ncclResult_t collNetDevices(ncclComm_t comm, int* ndev) { NCCLCHECK(comm->ncclCollNet->devices(ndev)); return ncclSuccess; }
static ncclResult_t collNetGetProperties(ncclComm_t comm, int dev, ncclNetProperties_t* props) { NCCLCHECK(comm->ncclCollNet->getProperties(dev, props)); return ncclSuccess; }
static ncclResult_t collNetListen(ncclComm_t comm, int dev, void* handle, void** listenComm) { NCCLCHECK(comm->ncclCollNet->listen(dev, handle, listenComm)); return ncclSuccess; }
static ncclResult_t collNetConnect(ncclComm_t comm, void* handles[], int nranks, int rank, void* listenComm, void** collComm) { NCCLCHECK(comm->ncclCollNet->connect(handles, nranks, rank, listenComm, collComm)); return ncclSuccess; }
static ncclResult_t collNetReduceSupport(ncclComm_t comm, ncclDataType_t dataType, ncclRedOp_t redOp, int* supported) { NCCLCHECK(comm->ncclCollNet->reduceSupport(dataType, redOp, supported)); return ncclSuccess; }
static ncclResult_t collNetRegMr(ncclComm_t comm, void* collComm, void* data, int size, int type, void** mhandle) { NCCLCHECK(comm->ncclCollNet->regMr(collComm, data, size, type, mhandle)); return ncclSuccess; }
static ncclResult_t collNetDeregMr(ncclComm_t comm, void* collComm, void* mhandle) { NCCLCHECK(comm->ncclCollNet->deregMr(collComm, mhandle)); return ncclSuccess; }
static ncclResult_t collNetIallreduce(ncclComm_t comm, void* collComm, void* sendData, void* recvData, int count, ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle,  void** request) {
  NCCLCHECK(comm->ncclCollNet->iallreduce(collComm, sendData, recvData, count, dataType, redOp, sendMhandle, recvMhandle, request)); return ncclSuccess; }
static ncclResult_t collNetIflush(ncclComm_t comm, void* collComm, void* data, int size, void* mhandle, void** request) { NCCLCHECK(comm->ncclCollNet->iflush(collComm, data, size, mhandle, request)); return ncclSuccess; }
static ncclResult_t collNetTest(ncclComm_t comm, void* request, int* done, int* size) { NCCLCHECK(comm->ncclCollNet->test(request, done, size)); return ncclSuccess; }
static ncclResult_t collNetCloseColl(ncclComm_t comm, void* collComm) { NCCLCHECK(comm->ncclCollNet->closeColl(collComm)); return ncclSuccess; }
static ncclResult_t collNetCloseListen(ncclComm_t comm, void* listenComm) { NCCLCHECK(comm->ncclCollNet->closeListen(listenComm)); return ncclSuccess; }

static int collNetSupport(ncclComm_t comm) { return comm->ncclCollNet != nullptr ? 1 : 0; }

#endif
