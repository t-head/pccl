/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../../common.h"
#include "../../gin.h"
#include "../../gin_provider.h"
#include "gin/gin_host.h"
#include "param.h"
#include "gin_host_troce.h"

// ---------------------- TROCE GDA backend --------------------------------
// The SNIC provider dispatches every device-facing entry point into
// gin_host_troce.cc, which drives troce queue pairs directly.

ncclResult_t ncclGinIbGdakiTroceInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  return ncclGinIbInitType(ctx, commId, logFunction, PCCL_GIN_TYPE_GDAKI, NULL, IB_PROVIDER_TROCE);
}

ncclResult_t ncclGinIbGdakiTroceDevices(int* ndev) {
  *ndev = ncclGinIbProviderDeviceCount(IB_PROVIDER_TROCE);
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiTroceGetProperties(int dev, ncclNetProperties_v12_t* props) {
  int ibDev = ncclGinIbProviderDeviceIndex(IB_PROVIDER_TROCE, dev);
  if (ibDev < 0) {
    WARN("NET/IB : Requested properties for GIN GDAKI_TROCE NIC %d, only %d GIN GDAKI_TROCE NICs have been created",
         dev, ncclGinIbProviderDeviceCount(IB_PROVIDER_TROCE));
    return ncclInvalidUsage;
  }
  ncclNetProperties_v6_t physProps;
  NCCLCHECK(ncclIbGetPhysProperties(ibDev, &physProps));
  ncclGinIbFillV12Properties(physProps, props);
  props->netDeviceType = PCCL_NET_DEVICE_GIN_GDAKI;
  props->netDeviceVersion = PCCL_GIN_GDAKI_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiTroceListen(void* /*ctx*/, int dev, void* opaqueHandle, void** listenComm) {
  int ibDev = ncclGinIbProviderDeviceIndex(IB_PROVIDER_TROCE, dev);
  if (ibDev < 0) return ncclInvalidUsage;
  return ncclNetIb.listen(ibDev, opaqueHandle, listenComm);
}

ncclResult_t ncclGinIbGdakiTroceConnect(void* ctx, void* handles[], int nranks, int rank,
                                   void* listenComm, void** collComm) {
  NCCLCHECK(ncclGinIbGdrGpuSupport(/*gdaki*/ false));
  NCCLCHECK(ncclGinIbConnect(ctx, handles, nranks, rank, listenComm, collComm));

  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)*collComm;
  ncclResult_t result = ncclGinIbCheckPeerProvider(cComm, IB_PROVIDER_TROCE);
  if (result != ncclSuccess) {
    ncclGinIbCloseColl(cComm);
    *collComm = nullptr;
    return result;
  }
  cComm->getProperties = (ncclResult_t(*)(int, void*))ncclGinIbGdakiTroceGetProperties;
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiTroceCreateContext(void* collComm, ncclGinConfig_v13_t* config,
                                         void** ginCtx, ncclNetDeviceHandle_v11_t** devHandle) {
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;
  NCCLCHECK(ncclGinGdakiTroceCreateContext(cComm, config->nSignals, config->nCounters,
                                      config->nContexts, config->queueDepth,
                                      config->trafficClass, ginCtx, devHandle));
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiTroceRegMrSym(void* collComm, void* data, size_t size, int type,
                                    uint64_t mr_flags, void** mhandle, void** ginHandle) {
  return ncclGinGdakiTroceRegMrSym((struct ncclGinIbCollComm*)collComm, data, size, type,
                              mr_flags, mhandle, ginHandle);
}

ncclResult_t ncclGinIbGdakiTroceDeregMrSym(void* collComm, void* mhandle) {
  return ncclGinGdakiTroceDeregMrSym((struct ncclGinIbCollComm*)collComm, mhandle);
}

ncclResult_t ncclGinIbGdakiTroceDestroyContext(void* ginCtx) {
  return ncclGinGdakiTroceDestroyContext(ginCtx);
}

ncclResult_t ncclGinIbGdakiTroceProgress(void* ginCtx) {
  return ncclGinGdakiTroceProgress(ginCtx);
}

ncclResult_t ncclGinIbGdakiTroceQueryLastError(void* ginCtx, bool* hasError) {
  return ncclGinGdakiTroceQueryLastError(ginCtx, hasError);
}

ncclGin_t ncclGinIbGdakiTroce = {
  "GIN_IB_GDAKI_TROCE",
  ncclGinIbGdakiTroceInit,
  ncclGinIbGdakiTroceDevices,
  ncclGinIbGdakiTroceGetProperties,
  ncclGinIbGdakiTroceListen,
  ncclGinIbGdakiTroceConnect,
  ncclGinIbGdakiTroceCreateContext,
  ncclGinIbGdakiTroceRegMrSym,
  NULL,
  ncclGinIbGdakiTroceDeregMrSym,
  ncclGinIbGdakiTroceDestroyContext,
  ncclGinIbCloseColl,
  ncclGinIbCloseListen,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  ncclGinIbGdakiTroceProgress,
  ncclGinIbGdakiTroceQueryLastError,
  ncclGinIbFinalize
};
