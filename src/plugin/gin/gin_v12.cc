/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl_gin.h"
#include "comm.h"
#include "proxy.h"
#include <dlfcn.h>

static ncclGin_v12_t* ncclGin_v12;
static ncclGin_t ncclGinShim;

static ncclResult_t ncclGin_connect(void* ctx, void* handles[], int nranks, int rank,
  void* listenComm, void** collComm) {
  return ncclGin_v12->connect(ctx, handles, nranks, rank, 1, 0, listenComm, collComm);
}

static ncclResult_t ncclGin_createContext(void* collComm, ncclGinConfig_t* config,
    void** ginCtx, ncclNetDeviceHandle_t** devHandle) {
  if (ncclGin_v12->createContext == NULL) {
    if (config->nContexts > 1) {
      WARN("GIN plugin v12 does not support multiple contexts");
      return ncclInvalidUsage;
    }
    *ginCtx = collComm;
    return ncclSuccess;
  } else {
    NCCLCHECK(ncclGin_v12->createContext(collComm, config->nSignals, config->nCounters, config->nContexts, ginCtx, devHandle));
  }
  return ncclSuccess;
}

static ncclResult_t ncclGin_destroyContext(void* ginCtx) {
  if (ncclGin_v12->destroyContext) {
    NCCLCHECK(ncclGin_v12->destroyContext(ginCtx));
  }
  return ncclSuccess;
}

static ncclResult_t ncclGin_iflush(void* ginCtx, int context, void* mhandle, uint32_t rank, void** request) {
  *request = NULL;
  return ncclSuccess;
}

static ncclResult_t ncclGin_iput(void* ginCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
    uint64_t dstOff, void* dstMhandle, uint32_t rank, void** request) {
  if (context != 0) {
    WARN("GIN plugin v12 does not support multiple contexts");
    return ncclInvalidUsage;
  }
  return ncclGin_v12->iput(ginCtx, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, 0, request);
}

static ncclResult_t ncclGin_iputSignal(void* ginCtx, int context, uint64_t srcOff, void* srcMhandle,
    size_t size, uint64_t dstOff, void* dstMhandle, uint32_t rank, uint64_t signalOff, void *signalMhandle,
    uint64_t signalValue, uint32_t signalOp, void** request) {
  if (context != 0) {
    WARN("GIN plugin v12 does not support multiple connections");
    return ncclInvalidUsage;
  }
  return ncclGin_v12->iputSignal(ginCtx, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, signalOff, signalMhandle, signalValue, signalOp, 0, request);
}

static ncclResult_t ncclGin_getProperties(int dev, ncclNetProperties_v12_t* props) {
  ncclNetProperties_v11_t props_v11;
  NCCLCHECK(ncclGin_v12->getProperties(dev, &props_v11));
  props->name = props_v11.name;
  props->pciPath = props_v11.pciPath;
  props->guid = props_v11.guid;
  props->ptrSupport = props_v11.ptrSupport;
  props->regIsGlobal = props_v11.regIsGlobal;
  props->forceFlush = props_v11.forceFlush;
  props->speed = props_v11.speed;
  props->port = props_v11.port;
  props->latency = props_v11.latency;
  props->maxComms = props_v11.maxComms;
  props->maxRecvs = props_v11.maxRecvs;
  props->netDeviceType = props_v11.netDeviceType;
  props->netDeviceVersion = props_v11.netDeviceVersion;
  props->vProps.ndevs = props_v11.vProps.ndevs;
  for (int i = 0; i < props_v11.vProps.ndevs; i++)
    props->vProps.devs[i] = props_v11.vProps.devs[i];
  props->maxP2pBytes = props_v11.maxP2pBytes;
  props->maxCollBytes = props_v11.maxCollBytes;
  props->maxMultiRequestSize = props_v11.maxMultiRequestSize;
  props->railId = NCCL_NET_ID_UNDEF;
  props->planeId = NCCL_NET_ID_UNDEF;
  return ncclSuccess;
}

ncclGin_t* getNcclGin_v12(void* lib) {
  ncclGin_v12 = (ncclGin_v12_t*)dlsym(lib, "ncclGinPlugin_v12");
  if (ncclGin_v12) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded gin plugin %s (v12)", ncclGin_v12->name);
    ncclGinShim.name = ncclGin_v12->name;
    ncclGinShim.init = ncclGin_v12->init;
    ncclGinShim.devices = ncclGin_v12->devices;
    ncclGinShim.getProperties = ncclGin_getProperties;
    ncclGinShim.listen = ncclGin_v12->listen;
    ncclGinShim.connect = ncclGin_connect;
    ncclGinShim.createContext = ncclGin_createContext;
    ncclGinShim.regMrSym = ncclGin_v12->regMrSym;
    ncclGinShim.regMrSymDmaBuf = ncclGin_v12->regMrSymDmaBuf;
    ncclGinShim.deregMrSym = ncclGin_v12->deregMrSym;
    ncclGinShim.destroyContext = ncclGin_destroyContext;
    ncclGinShim.closeColl = ncclGin_v12->closeColl;
    ncclGinShim.closeListen = ncclGin_v12->closeListen;
    ncclGinShim.iput = ncclGin_iput;
    ncclGinShim.iputSignal = ncclGin_iputSignal;
    ncclGinShim.iget = NULL;
    ncclGinShim.iflush = ncclGin_iflush;
    ncclGinShim.test = ncclGin_v12->test;
    ncclGinShim.ginProgress = ncclGin_v12->ginProgress;
    ncclGinShim.queryLastError = ncclGin_v12->queryLastError;
    ncclGinShim.finalize = ncclGin_v12->finalize;
    return &ncclGinShim;
  }
  return nullptr;
}
