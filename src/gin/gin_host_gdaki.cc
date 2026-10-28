/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "alloc.h"
#include "gin/gin_host.h"
#include "checks.h"
#include "pccl_device/gin/providers/mlx5/gin_mlx5_device_host_common.h"
#include "pccl_device/gin/providers/troce/gin_troce_device_host_common.h"

static bool ginGdakiSupported(const pcclGinGdakiHandle& gda) {
  if (gda.contexts == nullptr) {
    return false;
  }
  switch (gda.provider) {
    case PCCL_GIN_GDAKI_PROVIDER_MLX5:
      return gda.providerVersion == PCCL_GIN_MLX5_VERSION;
    case PCCL_GIN_GDAKI_PROVIDER_TROCE:
      return gda.providerVersion == PCCL_GIN_TROCE_VERSION;
    default:
      return false;
  }
}

ncclResult_t ncclGinGdakiCreateHandle(pcclGinGdakiProvider provider, uint32_t providerVersion,
                                      void* contexts, ncclNetDeviceHandle_t* handle) {
  pcclGinGdakiHandle gda = {provider, providerVersion, contexts};
  if (!handle || !ginGdakiSupported(gda)) {
    return ncclInvalidUsage;
  }
  pcclGinGdakiHandle* gpuHandle = nullptr;
  ncclResult_t result = ncclHggcCalloc(&gpuHandle, 1, nullptr);
  if (result == ncclSuccess) {
    result = ncclHggcMemcpy(gpuHandle, &gda, 1);
  }
  if (result != ncclSuccess) {
    ncclResult_t cleanupResult = ncclHggcFree(gpuHandle);
    if (cleanupResult != ncclSuccess) {
      WARN("Failed to release GDAKI descriptor after initialization error: %d", cleanupResult);
    }
    return result;
  }
  handle->netDeviceType = PCCL_NET_DEVICE_GIN_GDAKI;
  handle->netDeviceVersion = PCCL_GIN_GDAKI_VERSION;
  handle->handle = gpuHandle;
  handle->size = 0;
  return ncclSuccess;
}
