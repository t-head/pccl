/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2024-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "net.h"
#include "comm.h"
#include "transport.h"
#include "param.h"
// #include "checks.h"

#include <string.h>
#include <strings.h>
// #include <errno.h>
#include <dlfcn.h>
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <unistd.h>

static ncclNet_v6_t ncclNet_v4_as_v6;
static ncclNet_v6_t ncclNet_v5_as_v6;
static ncclNet_v4_t *ncclNet_v4;
static ncclNet_v5_t *ncclNet_v5;
static ncclCollNet_v6_t ncclCollNet_v4_as_v6;
static ncclCollNet_v6_t ncclCollNet_v5_as_v6;
static ncclCollNet_v4_t *ncclCollNet_v4;
static ncclCollNet_v5_t *ncclCollNet_v5;

static ncclResult_t ncclNet_v4_as_v6_getProperties(int dev, ncclNetProperties_v6_t* props) {
  ncclNetProperties_v4_t p4;
  ncclResult_t ans = ncclNet_v4->getProperties(dev, &p4);
  if (ans != ncclSuccess) return ans;
  props->name = p4.name;
  props->pciPath = p4.pciPath;
  props->guid = p4.guid;
  props->ptrSupport = p4.ptrSupport;
  props->speed = p4.speed;
  props->port = p4.port;
  props->maxComms = p4.maxComms;
  props->maxRecvs = 1;
  props->latency = 0;
  return ncclSuccess;
}

static ncclResult_t ncclNet_v4_as_v6_isend(void* sendComm, void* data, int size, int tag, void* mhandle, void** request) {
  return ncclNet_v4->isend(sendComm, data, size, mhandle, request);
}

static ncclResult_t ncclNet_v4_as_v6_irecv(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request) {
  if (n == 0) return ncclSuccess;
  if (n != 1) return ncclInvalidArgument;
  return ncclNet_v4->irecv(recvComm, data[0], sizes[0], mhandles[0], request);
}

static ncclResult_t ncclNet_v4_as_v6_iflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  if (n == 0) return ncclSuccess;
  if (n != 1) return ncclInvalidArgument;
  return ncclNet_v4->iflush(recvComm, data[0], sizes[0], mhandles[0], request);
}

// We use a wrapper around the v4 init to copy over the struct contents
// post-init since they may not be initialized before hand.
static ncclResult_t ncclNet_v4_as_v6_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclNet_v4->init(logfn));
  ncclNet_v4_as_v6.name = ncclNet_v4->name;
  ncclNet_v4_as_v6.devices = ncclNet_v4->devices;
  ncclNet_v4_as_v6.getProperties = ncclNet_v4_as_v6_getProperties;
  ncclNet_v4_as_v6.listen = ncclNet_v4->listen;
  ncclNet_v4_as_v6.connect = ncclNet_v4->connect;
  ncclNet_v4_as_v6.accept = ncclNet_v4->accept;
  ncclNet_v4_as_v6.regMr = ncclNet_v4->regMr;
  ncclNet_v4_as_v6.regMrDmaBuf = NULL;
  ncclNet_v4_as_v6.deregMr = ncclNet_v4->deregMr;
  ncclNet_v4_as_v6.isend = ncclNet_v4_as_v6_isend;
  ncclNet_v4_as_v6.irecv = ncclNet_v4_as_v6_irecv;
  ncclNet_v4_as_v6.iflush = ncclNet_v4_as_v6_iflush;
  ncclNet_v4_as_v6.test = ncclNet_v4->test;
  ncclNet_v4_as_v6.closeSend = ncclNet_v4->closeSend;
  ncclNet_v4_as_v6.closeRecv = ncclNet_v4->closeRecv;
  ncclNet_v4_as_v6.closeListen = ncclNet_v4->closeListen;
  return ncclSuccess;
}

// We use a wrapper around the v5 init to copy over the struct contents
// post-init since they may not be initialized before hand.
static ncclResult_t ncclNet_v5_as_v6_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclNet_v5->init(logfn));
  ncclNet_v5_as_v6.name = ncclNet_v5->name;
  ncclNet_v5_as_v6.devices = ncclNet_v5->devices;
  ncclNet_v5_as_v6.getProperties = ncclNet_v5->getProperties;
  ncclNet_v5_as_v6.listen = ncclNet_v5->listen;
  ncclNet_v5_as_v6.connect = ncclNet_v5->connect;
  ncclNet_v5_as_v6.accept = ncclNet_v5->accept;
  ncclNet_v5_as_v6.regMr = ncclNet_v5->regMr;
  ncclNet_v5_as_v6.regMrDmaBuf = NULL;
  ncclNet_v5_as_v6.deregMr = ncclNet_v5->deregMr;
  ncclNet_v5_as_v6.isend = ncclNet_v5->isend;
  ncclNet_v5_as_v6.irecv = ncclNet_v5->irecv;
  ncclNet_v5_as_v6.iflush = ncclNet_v5->iflush;
  ncclNet_v5_as_v6.test = ncclNet_v5->test;
  ncclNet_v5_as_v6.closeSend = ncclNet_v5->closeSend;
  ncclNet_v5_as_v6.closeRecv = ncclNet_v5->closeRecv;
  ncclNet_v5_as_v6.closeListen = ncclNet_v5->closeListen;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_v4_as_v6_getProperties(int dev, ncclNetProperties_v6_t* props) {
  ncclNetProperties_v4_t p4;
  ncclResult_t ans = ncclCollNet_v4->getProperties(dev, &p4);
  if (ans != ncclSuccess) return ans;
  props->name = p4.name;
  props->pciPath = p4.pciPath;
  props->guid = p4.guid;
  props->ptrSupport = p4.ptrSupport;
  props->speed = p4.speed;
  props->port = p4.port;
  props->maxComms = p4.maxComms;
  props->maxRecvs = 1;
  props->latency = 0;
  return ncclSuccess;
}

// We use a wrapper around the v4 init to copy over the struct contents
// post-init since they may not be initialized before hand.
static ncclResult_t ncclCollNet_v4_as_v6_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclCollNet_v4->init(logfn));
  ncclCollNet_v4_as_v6.name = ncclCollNet_v4->name;
  ncclCollNet_v4_as_v6.devices = ncclCollNet_v4->devices;
  ncclCollNet_v4_as_v6.getProperties = ncclCollNet_v4_as_v6_getProperties;
  ncclCollNet_v4_as_v6.listen = ncclCollNet_v4->listen;
  ncclCollNet_v4_as_v6.connect = ncclCollNet_v4->connect;
  ncclCollNet_v4_as_v6.reduceSupport = ncclCollNet_v4->reduceSupport;
  ncclCollNet_v4_as_v6.regMr = ncclCollNet_v4->regMr;
  ncclCollNet_v4_as_v6.regMrDmaBuf = NULL;
  ncclCollNet_v4_as_v6.deregMr = ncclCollNet_v4->deregMr;
  ncclCollNet_v4_as_v6.iallreduce = ncclCollNet_v4->iallreduce;
  ncclCollNet_v4_as_v6.iflush = ncclCollNet_v4->iflush;
  ncclCollNet_v4_as_v6.test = ncclCollNet_v4->test;
  ncclCollNet_v4_as_v6.closeColl = ncclCollNet_v4->closeColl;
  ncclCollNet_v4_as_v6.closeListen = ncclCollNet_v4->closeListen;
  return ncclSuccess;
}

// We use a wrapper around the v5 init to copy over the struct contents
// post-init since they may not be initialized before hand.
static ncclResult_t ncclCollNet_v5_as_v6_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclCollNet_v5->init(logfn));
  ncclCollNet_v5_as_v6.name = ncclCollNet_v5->name;
  ncclCollNet_v5_as_v6.devices = ncclCollNet_v5->devices;
  ncclCollNet_v5_as_v6.getProperties = ncclCollNet_v5->getProperties;
  ncclCollNet_v5_as_v6.listen = ncclCollNet_v5->listen;
  ncclCollNet_v5_as_v6.connect = ncclCollNet_v5->connect;
  ncclCollNet_v5_as_v6.reduceSupport = ncclCollNet_v5->reduceSupport;
  ncclCollNet_v5_as_v6.regMr = ncclCollNet_v5->regMr;
  ncclCollNet_v5_as_v6.regMrDmaBuf = NULL;
  ncclCollNet_v5_as_v6.deregMr = ncclCollNet_v5->deregMr;
  ncclCollNet_v5_as_v6.iallreduce = ncclCollNet_v5->iallreduce;
  ncclCollNet_v5_as_v6.iflush = ncclCollNet_v5->iflush;
  ncclCollNet_v5_as_v6.test = ncclCollNet_v5->test;
  ncclCollNet_v5_as_v6.closeColl = ncclCollNet_v5->closeColl;
  ncclCollNet_v5_as_v6.closeListen = ncclCollNet_v5->closeListen;
  return ncclSuccess;
}

static pthread_mutex_t netPluginLock = PTHREAD_MUTEX_INITIALIZER;
// All live communicators retain the plugin library, including users of built-in networks.
// Plugin ABI flags can also be used outside the selected network's callbacks.
static int netPluginRefCount = 0;
static void* netPluginLib = NULL;
static ncclNet_t* ncclNets[3] = { nullptr, &ncclNetIb, &ncclNetSocket };
static ncclCollNet_t* ncclCollNets[3] = { nullptr, nullptr, nullptr };

enum {
  ncclNetStateInit = 0,
  ncclNetStateEnabled = 1,
  ncclNetStateDisabled = 2,
};

static int ncclNetStates[3] = { ncclNetStateInit, ncclNetStateInit, ncclNetStateInit };
static int ncclCollNetStates[3] = { ncclNetStateInit, ncclNetStateInit, ncclNetStateInit };

enum {
  netPluginLoadFailed  = -1,
  netPluginLoadReady   =  0,
  netPluginLoadSuccess =  1,
};

static int netPluginStatus = netPluginLoadReady;

PCCL_PARAM(NetPluginEnable, "NET_PLUGIN_ENABLE", 1);

// Called with netPluginLock held and no live communicator references.
static ncclResult_t ncclNetPluginUnload() {
  if (netPluginLib != nullptr) {
    INFO(NCCL_NET, "NET/Plugin: Unload net plugin");
    if (dlclose(netPluginLib) != 0) {
      WARN("NET/Plugin: Failed to unload net plugin: %s", dlerror());
      return ncclSystemError;
    }
    netPluginLib = nullptr;
  }
  ncclNets[0] = nullptr;
  ncclCollNets[0] = nullptr;
  ncclNet_v4 = nullptr;
  ncclNet_v5 = nullptr;
  ncclCollNet_v4 = nullptr;
  ncclCollNet_v5 = nullptr;
  memset(&ncclNet_v4_as_v6, 0, sizeof(ncclNet_v4_as_v6));
  memset(&ncclNet_v5_as_v6, 0, sizeof(ncclNet_v5_as_v6));
  memset(&ncclCollNet_v4_as_v6, 0, sizeof(ncclCollNet_v4_as_v6));
  memset(&ncclCollNet_v5_as_v6, 0, sizeof(ncclCollNet_v5_as_v6));
  for (int i = 0; i < 3; i++) {
    ncclNetStates[i] = ncclCollNetStates[i] = ncclNetStateInit;
  }
  nccl_use_u2mm_host = nccl_use_u2mm_dev = 0;
  nccl_eic_gpu_offset = 0;
  netPluginStatus = netPluginLoadReady;
  return ncclSuccess;
}

// Loading and candidate initialization are serialized with communicator retain/release.
static ncclResult_t ncclNetPluginLoad() {
  ncclResult_t ret = ncclSuccess;
  ncclNet_v6_t** net = &ncclNets[0];
  ncclCollNet_v6_t** collnet = &ncclCollNets[0];

  char ncclNetPluginName[128];
  const char* envPluginName = pcclGetEnvWithDualPrefix("PCCL_NET_PLUGIN");

  if (netPluginLoadFailed == netPluginStatus) {
    goto exit;
  }
  if (netPluginLoadSuccess == netPluginStatus) {
    goto exit;
  }

  if (envPluginName && strlen(envPluginName)) {
    snprintf(ncclNetPluginName, 128, "libpccl-net-%s.so", envPluginName);
    INFO(NCCL_INIT, "Plugin name set by env to %s\n", ncclNetPluginName);
  } else {
    if (pcclParamNetPluginEnable() == 0) {
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: external plugin loading disabled by PCCL_NET_PLUGIN_ENABLE");
      goto fail;
    }
    sprintf(ncclNetPluginName, "libpccl-net.so");
  }
  netPluginLib = dlopen(ncclNetPluginName, RTLD_NOW | RTLD_LOCAL);
  if (netPluginLib == nullptr) {
    // dlopen does not guarantee to set errno, but dlerror only gives us a
    // string, so checking errno doesn't hurt to try to provide a better
    // error message
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin : Plugin (%s) load returned %d : %s, using internal implementation", ncclNetPluginName, errno, dlerror());
    goto fail;
  }

  *net = (ncclNet_v6_t*)dlsym(netPluginLib, "pcclNetPlugin_v6");
  if (*net == nullptr) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find pcclNetPlugin_v6 symbol.");
    // Try v5 plugin
    ncclNet_v5 = (ncclNet_v5_t*)dlsym(netPluginLib, "pcclNetPlugin_v5");
    if (ncclNet_v5 == nullptr) {
      ncclNet_v4 = (ncclNet_v4_t*)dlsym(netPluginLib, "pcclNetPlugin_v4");
      if (ncclNet_v4 == nullptr) {
        INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find pcclNetPlugin symbol (v4 or v5).");
        goto fail;
      }
      *net = &ncclNet_v4_as_v6;
      ncclNet_v4_as_v6.init = ncclNet_v4_as_v6_init;
      // Set the name right away to allow for NCCL_NET=... to work
      ncclNet_v4_as_v6.name = ncclNet_v4->name;
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v4)", (*net)->name);
    } else {
      *net = &ncclNet_v5_as_v6;
      ncclNet_v5_as_v6.init = ncclNet_v5_as_v6_init;
      // Set the name right away to allow for NCCL_NET=... to work
      ncclNet_v5_as_v6.name = ncclNet_v5->name;
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v5)", (*net)->name);
    }
  } else {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v6)", (*net)->name);
  }

  // Check for CollNet
  *collnet = (ncclCollNet_v6_t*)dlsym(netPluginLib, "pcclCollNetPlugin_v6");
  if (*collnet == nullptr) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find pcclCollNetPlugin_v6 symbol.");
    ncclCollNet_v5 = (ncclCollNet_v5_t*)dlsym(netPluginLib, "pcclCollNetPlugin_v5");
    if (ncclCollNet_v5 == nullptr) {
      ncclCollNet_v4 = (ncclCollNet_v4_t*)dlsym(netPluginLib, "pcclCollNetPlugin_v4");
      if (ncclCollNet_v4 == nullptr) {
        INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find pcclCollNetPlugin symbol (v4 or v5).");
      } else {
        *collnet = &ncclCollNet_v4_as_v6;
        ncclCollNet_v4_as_v6.init = ncclCollNet_v4_as_v6_init;
        ncclCollNet_v4_as_v6.name = ncclCollNet_v4->name;
        INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded coll plugin %s (v4)", (*collnet)->name);
      }
    } else {
      *collnet = &ncclCollNet_v5_as_v6;
      ncclCollNet_v5_as_v6.init = ncclCollNet_v5_as_v6_init;
      ncclCollNet_v5_as_v6.name = ncclCollNet_v5->name;
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded coll plugin %s (v5)", (*collnet)->name);
    }
  }

  netPluginStatus = netPluginLoadSuccess;

exit:
  return ret;

fail:
  ret = ncclNetPluginUnload();
  netPluginStatus = netPluginLoadFailed;
  goto exit;
}

ncclResult_t ncclNetInit(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  const char* netName = comm->config.netName;
  pthread_mutex_lock(&netPluginLock);
  if (comm->netPluginRetained) goto exit;
  comm->ncclNet = nullptr;
  comm->ncclCollNet = nullptr;
  NCCLCHECKGOTO(ncclNetPluginLoad(), ret, exit);

  for (int i = 0; i < 3; i++) {
    if (ncclNets[i] == nullptr) continue;
    if (netName && strcasecmp(netName, ncclNets[i]->name) != 0) continue;

    if (ncclNetStates[i] == ncclNetStateInit) {
      int ndev = 0;
      ncclNetStates[i] = ncclNetStateDisabled;
      if (ncclNets[i]->init(ncclDebugLogForPlugin) == ncclSuccess &&
          ncclNets[i]->devices(&ndev) == ncclSuccess && ndev > 0) {
        ncclNetStates[i] = ncclNetStateEnabled;
      }
    }
    if (ncclNetStates[i] != ncclNetStateEnabled) continue;

    if (ncclCollNets[i] && ncclCollNetStates[i] == ncclNetStateInit) {
      int ndev = 0;
      ncclCollNetStates[i] = ncclNetStateDisabled;
      if (ncclCollNets[i]->init(ncclDebugLogForPlugin) == ncclSuccess &&
          ncclCollNets[i]->devices(&ndev) == ncclSuccess && ndev > 0) {
        ncclCollNetStates[i] = ncclNetStateEnabled;
      }
    }
    comm->ncclNet = ncclNets[i];
    if (ncclCollNetStates[i] == ncclNetStateEnabled) comm->ncclCollNet = ncclCollNets[i];
    netPluginRefCount++;
    comm->netPluginRetained = true;
    goto exit;
  }

  WARN("Error: network %s not found.", netName ? netName : "");
  ret = ncclInvalidUsage;
  if (netPluginRefCount == 0) NCCLCHECKIGNORE(ncclNetPluginUnload(), ret);
exit:
  pthread_mutex_unlock(&netPluginLock);
  return ret;
}

ncclResult_t ncclNetFinalize(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  pthread_mutex_lock(&netPluginLock);
  comm->ncclNet = nullptr;
  comm->ncclCollNet = nullptr;
  if (comm->netPluginRetained) {
    comm->netPluginRetained = false;
    if (--netPluginRefCount == 0) ret = ncclNetPluginUnload();
  }
  pthread_mutex_unlock(&netPluginLock);
  return ret;
}

ncclResult_t ncclGpuGdrSupport(int* gdrSupport) {
  // In CUDA 11.3 and later we can now query the cudaDevAttrGPUDirectRDMASupported attribute
  int hggcDev, attr = 0;
  HGGCCHECK(hggcGetDevice(&hggcDev));
  HGGCCHECK(hggcDeviceGetAttribute(&attr, hggcDevAttrGPUDirectRDMASupported, hggcDev));
  *gdrSupport = attr;
  return ncclSuccess;
}

int ncclNetVersion(ncclComm_t comm) {
  return (comm->ncclNet == &ncclNet_v4_as_v6) ? 4 : ((comm->ncclNet == &ncclNet_v5_as_v6) ? 5 : 6);
}

bool ncclEicNet(ncclNet_t* net) {
  return net && strcmp(net->name, EIC_PLUGIN_NAME) == 0;
}
