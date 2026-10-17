/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gin/gin_host.h"
#include "checks.h"
#include <stdio.h>
#include "pccl_device/gin/providers/mlx5/gin_mlx5_device_host_common.h"
#include "pccl_device/gin/providers/troce/gin_troce_device_host_common.h"

static bool ginGdakiSupported(const pcclGinGdakiHandle& gda) {
  if (gda.contexts == nullptr) return false;
  switch (gda.provider) {
    case PCCL_GIN_GDAKI_PROVIDER_MLX5: return gda.providerVersion == PCCL_GIN_MLX5_VERSION;
    case PCCL_GIN_GDAKI_PROVIDER_TROCE: return gda.providerVersion == PCCL_GIN_TROCE_VERSION;
    default: return false;
  }
}

ncclResult_t ncclGinGdakiCreateHandle(pcclGinGdakiProvider provider, uint32_t providerVersion,
                                     void* contexts, ncclNetDeviceHandle_t* handle) {
  pcclGinGdakiHandle gda = {provider, providerVersion, contexts};
  if (!handle || !ginGdakiSupported(gda)) return ncclInvalidUsage;
  void* gpuHandle = nullptr;
  HGGCCHECK(hggcMalloc(&gpuHandle, sizeof(gda)));
  hggcError_t copyStatus = hggcMemcpy(gpuHandle, &gda, sizeof(gda), hggcMemcpyHostToDevice);
  if (copyStatus != hggcSuccess) {
    hggcFree(gpuHandle);
    HGGCCHECK(copyStatus);
  }
  handle->netDeviceType = PCCL_NET_DEVICE_GIN_GDAKI;
  handle->netDeviceVersion = PCCL_GIN_GDAKI_VERSION;
  handle->handle = gpuHandle;
  handle->size = 0;
  return ncclSuccess;
}

ncclResult_t ncclGinGdakiPrepareHandle(const ncclNetDeviceHandle_t* handle, void** gpuHandle, bool* owned) {
  *gpuHandle = nullptr;
  *owned = false;
  if (!handle || !handle->handle || handle->netDeviceType != PCCL_NET_DEVICE_GIN_GDAKI)
    return ncclInvalidUsage;
  if (handle->netDeviceVersion == PCCL_GIN_MLX5_VERSION) {
    // Version 200 explicitly denotes the old DOCA/MLX5 context-array ABI.
    // Do not guess a provider for unknown plugin versions.
    ncclNetDeviceHandle_t adapter = {};
    NCCLCHECK(ncclGinGdakiCreateHandle(PCCL_GIN_GDAKI_PROVIDER_MLX5, PCCL_GIN_MLX5_VERSION,
                                      handle->handle, &adapter));
    *gpuHandle = adapter.handle;
    *owned = true;
    return ncclSuccess;
  }
  if (handle->netDeviceVersion != PCCL_GIN_GDAKI_VERSION) {
    WARN("Unsupported GDAKI handle version %d (expected %d)", handle->netDeviceVersion, PCCL_GIN_GDAKI_VERSION);
    return ncclInvalidUsage;
  }
  pcclGinGdakiHandle gda = {};
  HGGCCHECK(hggcMemcpy(&gda, handle->handle, sizeof(gda), hggcMemcpyDeviceToHost));
  if (!ginGdakiSupported(gda)) {
    WARN("Unsupported GDAKI provider/context version %u/%u", (unsigned)gda.provider, gda.providerVersion);
    return ncclInvalidUsage;
  }
  *gpuHandle = handle->handle;
  return ncclSuccess;
}

void ncclGinGdakiDump(void* gpuHandle) {
  pcclGinGdakiHandle gda = {};
  if (hggcMemcpy(&gda, gpuHandle, sizeof(gda), hggcMemcpyDeviceToHost) == hggcSuccess)
    printf("    GDAKI provider %u context version %u contexts %p\n",
           (unsigned)gda.provider, gda.providerVersion, gda.contexts);
}
