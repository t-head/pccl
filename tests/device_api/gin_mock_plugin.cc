/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *************************************************************************/
#include <pccl.h>
#include "pccl_common.h"
typedef pcclResult_t ncclResult_t;
typedef pcclDebugLogger_t ncclDebugLogger_t;
#include "plugin/gin/gin_v13.h"
#include <cstdlib>
#include <cstring>

namespace {
int connectionCalls = 0, contextCalls = 0;
bool mode(const char* name) {
  const char* value = getenv("PCCL_GIN_TEST_MODE");
  return value && strcmp(value, name) == 0;
}
pcclResult_t init(void** context, uint64_t, ncclDebugLogger_t) { *context = nullptr; return pcclSuccess; }
pcclResult_t devices(int* count) { *count = 2; return pcclSuccess; }
pcclResult_t properties(int, ncclNetProperties_v12_t* props) {
  memset(props, 0, sizeof(*props));
  props->name = const_cast<char*>("mock-gin");
  props->netDeviceType = PCCL_NET_DEVICE_GIN_GDAKI;
  // Transport properties must not determine the accepted device-handle ABI.
  props->netDeviceVersion = mode("bad-version") ? 201 : 999;
  props->ptrSupport = 3;
  props->vProps.ndevs = 1;
  return pcclSuccess;
}
pcclResult_t listen(void*, int, void*, void** result) { *result = malloc(1); return pcclSuccess; }
pcclResult_t connect(void*, void**, int, int, void*, void** result) {
  if (mode("retry") && connectionCalls++ == 0) return pcclSystemError;
  *result = malloc(1);
  return *result ? pcclSuccess : pcclSystemError;
}
pcclResult_t create(void*, ncclGinConfig_v13_t*, void** context, pcclNetDeviceHandle_v11_t** output) {
  auto* handle = static_cast<pcclNetDeviceHandle_v11_t*>(calloc(1, sizeof(pcclNetDeviceHandle_v11_t)));
  if (!handle) return pcclSystemError;
  int index = contextCalls++;
  handle->netDeviceType = PCCL_NET_DEVICE_GIN_GDAKI;
  handle->netDeviceVersion = mode("bad-version") ? 201 : 200;
  handle->handle = handle;
  handle->needsProxyProgress = mode("partial-progress") && index == 0;
  if (mode("partial-progress") && index == 1) handle->netDeviceType = PCCL_NET_DEVICE_HOST;
  *context = *output = handle;
  return pcclSuccess;
}
pcclResult_t release(void* pointer) { free(pointer); return pcclSuccess; }
pcclResult_t progress(void*) { return pcclSuccess; }
pcclResult_t finalize(void*) { return pcclSuccess; }
} // namespace

extern "C" {
ncclGin_v13_t pcclGinPlugin_v13 = {};
}
__attribute__((constructor)) static void initializePlugin() {
  auto& plugin = pcclGinPlugin_v13;
  plugin.name = "mock-gin";
  plugin.init = init;
  plugin.devices = devices;
  plugin.getProperties = properties;
  plugin.listen = listen;
  plugin.connect = connect;
  plugin.createContext = create;
  plugin.destroyContext = release;
  plugin.closeColl = release;
  plugin.closeListen = release;
  plugin.ginProgress = progress;
  plugin.finalize = finalize;
}
