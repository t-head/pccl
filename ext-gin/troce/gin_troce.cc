/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "transport/net_ib/common.h"
#include "transport/net_ib/gin.h"
#include "alloc.h"
#include "gin_host_troce.h"
#include "pccl_device/gin/troce/gin_troce_device_host_common.h"
#include "trocedvwrap.h"
#include <mutex>

extern int64_t ncclParamGinType();

static std::mutex ncclGinIbTroceLockMutex;
static int ncclGinIbTroceNDevs = -1;
static int ncclGinIbTroceRefCount;
static int ncclGinIbTroceDevIndexes[MAX_IB_DEVS];

static int ncclGinIbTroceGetIbDevIndex(int dev) {
  std::lock_guard<std::mutex> lock(ncclGinIbTroceLockMutex);
  return dev >= 0 && dev < ncclGinIbTroceNDevs ? ncclGinIbTroceDevIndexes[dev] : -1;
}

static ncclResult_t ncclGinIbTroceInit(void** ctx, uint64_t /*commId*/, ncclDebugLogger_t logFunction) {
  *ctx = nullptr;
  int type = ncclParamGinType();
  if (type != -1 && type != PCCL_GIN_TYPE_TROCE) return ncclInvalidUsage;

  NCCLCHECK(ncclIbInitDevices(logFunction, nullptr));
  ncclResult_t result = ncclSuccess;
  ncclTroceFunctionTable* functions = nullptr;
  ncclNetCommConfig_t* config = nullptr;

  std::lock_guard<std::mutex> lock(ncclGinIbTroceLockMutex);
  if (ncclGinIbTroceNDevs == -1) {
    int ndevs = 0;
    for (int dev = 0; dev < ncclNIbDevs; dev++) {
      if (ncclIbDevs[dev].ibProvider == IB_PROVIDER_TROCE) {
        ncclGinIbTroceDevIndexes[ndevs++] = dev;
      }
    }
    ncclGinIbTroceNDevs = ndevs;
  }
  if (ncclGinIbTroceNDevs == 0) result = ncclInvalidUsage;

  if (result != ncclSuccess) goto fail;
  NCCLCHECKGOTO(ncclTroceFtableInit(&functions), result, fail);
  NCCLCHECKGOTO(ncclCalloc(&config, 1), result, fail);

  config->trafficClass = NCCL_NET_TRAFFIC_CLASS_UNDEF;
  *ctx = config;
  ncclGinIbTroceRefCount++;
  return ncclSuccess;

fail:
  if (ncclGinIbTroceRefCount == 0) ncclTroceFtableFini();
  NCCLCHECKIGNORE(ncclIbFinalizeDevices(), result);
  return result;
}

static ncclResult_t ncclGinIbTroceFinalize(void* ctx) {
  std::lock_guard<std::mutex> lock(ncclGinIbTroceLockMutex);
  NCCLCHECK(ncclGinIbFinalize(ctx));
  if (--ncclGinIbTroceRefCount == 0) ncclTroceFtableFini();
  return ncclSuccess;
}

static ncclResult_t ncclGinIbTroceDevices(int* count) {
  std::lock_guard<std::mutex> lock(ncclGinIbTroceLockMutex);
  *count = ncclGinIbTroceNDevs;
  return ncclSuccess;
}

static ncclResult_t ncclGinIbTroceGetProperties(int dev, ncclNetProperties_v12_t* props) {
  int ibDev = ncclGinIbTroceGetIbDevIndex(dev);
  if (ibDev < 0) return ncclInvalidArgument;

  ncclNetProperties_v6_t source = {};
  NCCLCHECK(ncclIbGetPhysProperties(ibDev, &source));

  memset(props, 0, sizeof(*props));
  ncclGinIbFillV12Properties(source, props);
  props->netDeviceType = PCCL_NET_DEVICE_GIN_TROCE;
  props->netDeviceVersion = PCCL_GIN_TROCE_VERSION;
  props->vProps.devs[0] = dev;

  return ncclSuccess;
}

static ncclResult_t ncclGinIbTroceListen(void* /*ctx*/, int dev, void* handle, void** output) {
  int ibDev = ncclGinIbTroceGetIbDevIndex(dev);
  if (ibDev < 0) return ncclInvalidArgument;
  return ncclNetIb.listen(ibDev, handle, output);
}

static ncclResult_t ncclGinIbTroceConnect(void* ctx, void** handles, int ranks, int rank,
                                          void* listener, void** output) {
  NCCLCHECK(ncclGinIbGdrGpuSupport(ncclIbGdrSupport() == ncclSuccess));
  NCCLCHECK(ncclGinIbConnect(ctx, handles, ranks, rank, listener, output));

  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)*output;
  int* providers = nullptr;
  ncclResult_t result = ncclSuccess;
  int provider = ncclIbDevs[cComm->dev].ibProvider;

  NCCLCHECKGOTO(ncclIbMalloc((void**)&providers, ranks * sizeof(int)), result, exit);
  NCCLCHECKGOTO(cComm->allGather(cComm, &provider, providers, sizeof(int)), result, exit);

  for (int peer = 0; peer < ranks; peer++) {
    if (providers[peer] != IB_PROVIDER_TROCE) {
      WARN("GIN/TROCE cannot exchange device resources with a different peer backend");
      result = ncclInvalidUsage;
      break;
    }
  }

exit:
  NCCLCHECKIGNORE(ncclFree(providers), result);

  if (result != ncclSuccess) {
    NCCLCHECKIGNORE(ncclGinIbCloseColl(*output), result);
    *output = nullptr;
  }

  return result;
}

static ncclResult_t ncclGinIbTroceCreateContext(void* comm, ncclGinConfig_t* config,
                                                void** context, ncclNetDeviceHandle_v11_t** handle) {
  *context = nullptr;
  *handle = nullptr;

  if (config->nCounters != 0) {
    WARN("GIN/TROCE does not support counter resources");
    return ncclInvalidUsage;
  }

  if (config->nContexts <= 0 || config->nSignals < 0 || config->queueDepth < 0) return ncclInvalidArgument;

  return ncclGinTroceCreateContext(comm, config->nSignals, 0, config->nContexts,
                                   config->queueDepth, config->trafficClass, context, handle);
}

extern "C" {
ncclGin_v13_t pcclGinPlugin_v13 = {
  "GIN_IB_TROCE",
  ncclGinIbTroceInit,
  ncclGinIbTroceDevices,
  ncclGinIbTroceGetProperties,
  ncclGinIbTroceListen,
  ncclGinIbTroceConnect,
  ncclGinIbTroceCreateContext,
  ncclGinTroceRegMrSym,
  nullptr,
  ncclGinTroceDeregMrSym,
  ncclGinTroceDestroyContext,
  ncclGinIbCloseColl,
  ncclGinIbCloseListen,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  ncclGinTroceProgress,
  ncclGinTroceQueryLastError,
  ncclGinIbTroceFinalize
};
}
