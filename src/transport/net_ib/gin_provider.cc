/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "gin_provider.h"
#include "gin.h"
#include <string.h>

extern ncclGin_t ncclGinIbGdakiMlx;
extern ncclGin_t ncclGinIbGdakiTroce;

// Add a built-in GPU-direct provider here and implement its ncclGin_t in its
// own file. Unsupported providers remain eligible for the generic IB proxy.
static const ncclGinIbProviderOps providers[] = {
  {IB_PROVIDER_MLX5, &ncclGinIbGdakiMlx, true},
  {IB_PROVIDER_TROCE, &ncclGinIbGdakiTroce, false},
};

enum ncclIbProvider ncclIbProviderFromName(const char* name) {
  if (name == nullptr) return IB_PROVIDER_UNKNOWN;
  if (strncmp(name, "mlx5", 4) == 0 || strncmp(name, "ibp", 3) == 0) return IB_PROVIDER_MLX5;
  if (strncmp(name, "troce", 5) == 0) return IB_PROVIDER_TROCE;
  if (strncmp(name, "fic2", 4) == 0 || strncmp(name, "vsolar", 6) == 0) return IB_PROVIDER_EIC;
  if (strncmp(name, "tsoe", 4) == 0) return IB_PROVIDER_TSOE;
  return IB_PROVIDER_UNKNOWN;
}

const char* ncclIbProviderName(enum ncclIbProvider provider) {
  switch (provider) {
    case IB_PROVIDER_MLX5: return "mlx5";
    case IB_PROVIDER_TROCE: return "troce";
    case IB_PROVIDER_EIC: return "eic";
    case IB_PROVIDER_TSOE: return "tsoe";
    default: return "unknown";
  }
}

const ncclGinIbProviderOps* ncclGinIbProviderOpsFor(enum ncclIbProvider provider) {
  for (const auto& ops : providers) if (ops.provider == provider) return &ops;
  return nullptr;
}

int ncclGinIbProviderDeviceCount(enum ncclIbProvider provider) {
  int count = 0;
  for (int i = 0; i < ncclNIbDevs; ++i) count += ncclIbDevs[i].ibProvider == provider;
  return count;
}

int ncclGinIbProviderDeviceIndex(enum ncclIbProvider provider, int ginDev) {
  if (ginDev < 0) return -1;
  for (int i = 0; i < ncclNIbDevs; ++i) {
    if (ncclIbDevs[i].ibProvider == provider && ginDev-- == 0) return i;
  }
  return -1;
}

// GDAKI is a common mode, but private QP exchange layouts are not compatible
// across providers. Reject a mixed-provider group before exchanging them.
ncclResult_t ncclGinIbCheckPeerProvider(ncclGinIbCollComm* comm, ncclIbProvider provider) {
  int* peers = (int*)calloc(comm->nranks, sizeof(int));
  if (!peers) return ncclSystemError;
  int localProvider = (int)provider;
  ncclResult_t result = comm->allGather(comm, &localProvider, peers, sizeof(int));
  if (result == ncclSuccess) {
    for (int r = 0; r < comm->nranks; ++r) {
      if (peers[r] != (int)provider) {
        WARN("GIN GDAKI providers differ: local %s, peer %d provider %d",
             ncclIbProviderName(provider), r, peers[r]);
        result = ncclInvalidUsage;
        break;
      }
    }
  }
  free(peers);
  return result;
}
