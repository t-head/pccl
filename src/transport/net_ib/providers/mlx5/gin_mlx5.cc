/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../../common.h"
#include "../../gin.h"
#include "../../gin_provider.h"
#include "gin/gin_host.h"
#include "param.h"
#include "gin_host_mlx5.h"

// ---------------------- MLX5 GDA backend ---------------------------------
// The MLX5 provider dispatches every device-facing entry point into
// gin_host_mlx5.cc, which drives the DOCA-GPUNetIO host library.

ncclResult_t ncclGinIbGdakiMlxInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  return ncclGinIbInitType(ctx, commId, logFunction, NCCL_GIN_TYPE_GDAKI, NULL, IB_PROVIDER_MLX5);
}

ncclResult_t ncclGinIbGdakiMlxDevices(int* ndev) {
  *ndev = ncclGinIbProviderDeviceCount(IB_PROVIDER_MLX5);
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiMlxGetProperties(int dev, ncclNetProperties_v12_t* props) {
  int ibDev = ncclGinIbProviderDeviceIndex(IB_PROVIDER_MLX5, dev);
  if (ibDev < 0) {
    WARN("NET/IB : Requested properties for GIN GDAKI NIC %d, only %d GIN GDAKI NICs have been created",
         dev, ncclGinIbProviderDeviceCount(IB_PROVIDER_MLX5));
    return ncclInvalidUsage;
  }
  ncclNetProperties_v6_t physProps = {};
  NCCLCHECK(ncclIbGetPhysProperties(ibDev, &physProps));
  ncclGinIbFillV12Properties(physProps, props);
  props->netDeviceType = NCCL_NET_DEVICE_GIN_GDAKI;
  props->netDeviceVersion = PCCL_GIN_GDAKI_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiMlxListen(void* /*ctx*/, int dev, void* opaqueHandle, void** listenComm) {
  int ibDev = ncclGinIbProviderDeviceIndex(IB_PROVIDER_MLX5, dev);
  if (ibDev < 0) return ncclInvalidUsage;
  return ncclNetIb.listen(ibDev, opaqueHandle, listenComm);
}

ncclResult_t ncclGinIbGdakiMlxConnect(void* ctx, void* handles[], int nranks, int rank,
                                   void* listenComm, void** collComm) {
  NCCLCHECK(ncclGinIbGdrGpuSupport(/*gdaki*/ true));
  NCCLCHECK(ncclGinIbConnect(ctx, handles, nranks, rank, listenComm, collComm));

  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)*collComm;
  ncclResult_t result = ncclGinIbCheckPeerProvider(cComm, IB_PROVIDER_MLX5);
  if (result != ncclSuccess) {
    ncclGinIbCloseColl(cComm);
    *collComm = nullptr;
    return result;
  }
  cComm->getProperties = (ncclResult_t(*)(int, void*))ncclGinIbGdakiMlxGetProperties;
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiMlxCreateContext(void* collComm, ncclGinConfig_v13_t* config,
                                         void** ginCtx, ncclNetDeviceHandle_v11_t** devHandle) {
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;
  NCCLCHECK(ncclGinGdakiMlxCreateContext(cComm, config->nSignals, config->nCounters,
                                      config->nContexts, config->queueDepth,
                                      config->trafficClass, ginCtx, devHandle));
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiMlxRegMrSym(void* collComm, void* data, size_t size, int type,
                                    uint64_t mr_flags, void** mhandle, void** ginHandle) {
  return ncclGinGdakiMlxRegMrSym((struct ncclGinIbCollComm*)collComm, data, size, type,
                              mr_flags, mhandle, ginHandle);
}

ncclResult_t ncclGinIbGdakiMlxDeregMrSym(void* collComm, void* mhandle) {
  return ncclGinGdakiMlxDeregMrSym((struct ncclGinIbCollComm*)collComm, mhandle);
}

ncclResult_t ncclGinIbGdakiMlxDestroyContext(void* ginCtx) {
  return ncclGinGdakiMlxDestroyContext(ginCtx);
}

ncclResult_t ncclGinIbGdakiMlxProgress(void* collComm) {
  return ncclGinGdakiMlxProgress(collComm);
}

ncclResult_t ncclGinIbGdakiMlxQueryLastError(void* ginCtx, bool* hasError) {
  return ncclGinGdakiMlxQueryLastError(ginCtx, hasError);
}

ncclGin_t ncclGinIbGdakiMlx = {
  "GIN_IB_GDAKI_MLX",
  ncclGinIbGdakiMlxInit,
  ncclGinIbGdakiMlxDevices,
  ncclGinIbGdakiMlxGetProperties,
  ncclGinIbGdakiMlxListen,
  ncclGinIbGdakiMlxConnect,
  ncclGinIbGdakiMlxCreateContext,
  ncclGinIbGdakiMlxRegMrSym,
  NULL,
  ncclGinIbGdakiMlxDeregMrSym,
  ncclGinIbGdakiMlxDestroyContext,
  ncclGinIbCloseColl,
  ncclGinIbCloseListen,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  ncclGinIbGdakiMlxProgress,
  ncclGinIbGdakiMlxQueryLastError,
  ncclGinIbFinalize
};
