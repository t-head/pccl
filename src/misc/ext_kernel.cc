//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include <dlfcn.h>
#include <errno.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "core.h"
#include "transport.h"
#include "ext_kernel_plugin.h"

pthread_mutex_t extKernelPluginLock = PTHREAD_MUTEX_INITIALIZER;
static int extKernelPluginRefCount;
static void* extKernelPluginLib = NULL;
pcclExtKernel_t* extKernelSymbol = NULL;

PCCL_PARAM(EnableExtKernel, "ENABLE_EXT_KERNEL", 1);

#define MAX_STR_LEN 255
static void* tryOpenLib(char* name, int* err, char* errStr) {
  *err = 0;
  if (NULL == name || strlen(name) == 0) {
    return NULL;
  }

  if (strncasecmp(name, "STATIC_PLUGIN", strlen(name)) == 0) {
    name = NULL;
  }

  void *handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
  if (NULL == handle) {
    strncpy(errStr, dlerror(), MAX_STR_LEN);
    errStr[MAX_STR_LEN] = '\0';
    if (strstr(errStr, name) && strstr(errStr, "No such file or directory")) {
      *err = ENOENT;
    }
  }
  return handle;
}

static char* tryOpenLibCheck(int openErr, char* openErrStr, char* nameList, int *nameListLen, char* name) {
  if (openErr == ENOENT) {
    snprintf(nameList, *nameListLen, " %s", name);
    nameList += strlen(name) + 1;
    *nameListLen -= strlen(name) + 1;
    return nameList;
  }
  INFO(NCCL_INIT|NCCL_EXT, "ExtKernel/Plugin: %s", openErrStr);
  return nameList;
}

static void* openExtKernelPluginLib(char* couldNotFindNames, int len) {
  int openErr;
  void *pluginLib;
  char extKernelPluginLibName[PATH_MAX];
  char openErrStr[MAX_STR_LEN + 1] = { 0 };
  const char *envExtKernelPluginName = pcclGetEnvWithDualPrefix("PCCL_EXT_KERNEL_PLUGIN");
  if (envExtKernelPluginName && strlen(envExtKernelPluginName)) {
    snprintf(extKernelPluginLibName, PATH_MAX, "%s", envExtKernelPluginName);
    pluginLib = tryOpenLib(extKernelPluginLibName, &openErr, openErrStr);
    if (pluginLib) {
      INFO(NCCL_INIT|NCCL_EXT, "ExtKernel/Plugin: Plugin name set by env to %s", extKernelPluginLibName);
      return pluginLib;
    }
    couldNotFindNames = tryOpenLibCheck(openErr, openErrStr, couldNotFindNames, &len, extKernelPluginLibName);
  } else {
    snprintf(extKernelPluginLibName, PATH_MAX, "libpccl-ext-kernel.so");
    pluginLib = tryOpenLib(extKernelPluginLibName, &openErr, openErrStr);
    if (pluginLib) {
      return pluginLib;
    }
    couldNotFindNames = tryOpenLibCheck(openErr, openErrStr, couldNotFindNames, &len, extKernelPluginLibName);
  }
  return NULL;
}

enum {
  extKernelPluginLoadFailed  = -1,
  extKernelPluginLoadReady   =  0,
  extKernelPluginLoadSuccess =  1,
};

// for ext kernel, MAX_PLUGIN_LOAD=2 just used to protect the snprintf
#define MAX_PLUGIN_LOAD 2

static int extKernelPluginStatus = extKernelPluginLoadReady;
ncclResult_t pcclExtKernelPluginLoad(ncclComm_t comm) {
  // Initialize to NULL by default if plugin extKernel cannot be loaded.
  comm->extKernel = NULL;
  if (pcclParamEnableExtKernel() == 0) return ncclSuccess;
  const char* algoStr = pcclGetEnvWithDualPrefix("PCCL_ALGO");
  const char* protoStr = pcclGetEnvWithDualPrefix("PCCL_PROTO");
  if (algoStr != NULL || protoStr != NULL) {
    INFO(NCCL_EXT, "Skip ext kernel init due to PCCL_ALGO or PCCL_PROTO set");
    return ncclSuccess;
  }

  if (comm->nNodes > 1) {
    INFO(NCCL_EXT, "Skip ext kernel init due to nNodes(%d)>1", comm->nNodes);
    return ncclSuccess;
  }

  if ((comm->nRanks != 2 && comm->nRanks != 4 && comm->nRanks != 8 &&
       comm->nRanks != 16 && comm->nRanks != 32)) {
    INFO(NCCL_EXT, "Skip ext kernel init due to nRanks(%d) not 2/4/8/16/32", comm->nRanks);
    return ncclSuccess;
  }

  if (!comm->globalIcnP2pSupport) {
    INFO(NCCL_EXT, "Skip ext kernel init due to not fully icn connection");
    return ncclSuccess;
  }

  char couldNotFindNames[MAX_PLUGIN_LOAD*PATH_MAX] = { 0 };
  if (extKernelPluginLoadFailed == extKernelPluginStatus) {
    return ncclSuccess;
  }
  pthread_mutex_lock(&extKernelPluginLock);
  if (extKernelPluginLoadSuccess == extKernelPluginStatus) {
    comm->extKernel = extKernelSymbol;
    ++extKernelPluginRefCount;
    goto exit;
  }

  extKernelPluginLib = openExtKernelPluginLib(couldNotFindNames, MAX_PLUGIN_LOAD*PATH_MAX);
  if (extKernelPluginLib == NULL) {
    if (strlen(couldNotFindNames)) {
      INFO(NCCL_INIT|NCCL_EXT, "ExtKernel/Plugin: Could not find:%s. Using default pccl kernel.", couldNotFindNames);
    } else {
      INFO(NCCL_INIT|NCCL_EXT, "ExtKernel/Plugin: Using default pccl kernel.");
    }
    goto fail;
  }

  extKernelSymbol = (pcclExtKernel_t*)dlsym(extKernelPluginLib, PCCL_EXT_KERNEL_PLUGIN_SYMBOL);
  if (extKernelSymbol == NULL) {
    INFO(NCCL_INIT|NCCL_EXT, "ExtKernel/Plugin: Failed to find " PCCL_EXT_KERNEL_PLUGIN_SYMBOL " symbol in plugin, Using default pccl kernel");
    goto fail;
  }

  VERBOSE(NCCL_EXT, "ExtKernel/Plugin: Using extKernel plugin %s", extKernelSymbol->name);
  comm->extKernel = extKernelSymbol;
  ++extKernelPluginRefCount;
  extKernelPluginStatus = extKernelPluginLoadSuccess;

exit:
  pthread_mutex_unlock(&extKernelPluginLock);
  return ncclSuccess;

fail:
  if (extKernelPluginLib) dlclose(extKernelPluginLib);
  extKernelPluginLib = NULL;
  extKernelPluginStatus = extKernelPluginLoadFailed;
  goto exit;
}

ncclResult_t pcclExtKernelPluginUnload(ncclComm_t comm) {
  pthread_mutex_lock(&extKernelPluginLock);
  comm->extKernel = NULL;
  if (0 == (--extKernelPluginRefCount) && extKernelPluginLib != NULL) {
    VERBOSE(NCCL_EXT, "Closing extKernel: '%s'", extKernelSymbol->name);
    dlclose(extKernelPluginLib);
    extKernelPluginLib = NULL;
    extKernelSymbol = NULL;
    extKernelPluginStatus = extKernelPluginLoadReady;
  }
  pthread_mutex_unlock(&extKernelPluginLock);
  return ncclSuccess;
}
