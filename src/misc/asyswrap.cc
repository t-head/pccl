//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include "asyswrap.h"
#include <dlfcn.h>
#include "core.h"

static enum { asysUninitialized, asysInitialized, asysError } asysState = asysUninitialized;
static int32_t (*pcclInjection)(void);
static bool asysPcclInjected = false;

ncclResult_t wrapAsysSymbols(void) {
  // per process
  if (asysState == asysInitialized)
    return ncclSuccess;

  if (asysState == asysError)
    return ncclSystemError;

  const char* asysDlName = pcclGetEnvWithDualPrefix("PCCL_INJECTION64_PATH");
  if (!asysDlName) return pcclInternalError;

  static void* asysHandle = NULL;
  void* tmp;
  void** cast;

  asysHandle = dlopen(asysDlName, RTLD_NOW);
  if (!asysHandle) {
    WARN("Failed to open %s", asysDlName);
    goto teardown;
  }

#define LOAD_SYM(handle, symbol, funcptr) do {           \
    cast = (void**)&funcptr;                             \
    tmp = dlsym(handle, symbol);                         \
    if (tmp == NULL) {                                   \
      WARN("dlsym failed on %s - %s", symbol, dlerror());\
      goto teardown;                                     \
    }                                                    \
    *cast = tmp;                                         \
  } while (0)

  LOAD_SYM(asysHandle, "InitializePcclInjection", pcclInjection);

  asysState = asysInitialized;
  return ncclSuccess;

teardown:
  pcclInjection = NULL;
  if (asysHandle != NULL) dlclose(asysHandle);
  asysState = asysError;
  return ncclSystemError;
}

ncclResult_t wrapPcclPcclInjection(void) {
  if (asysPcclInjected) return ncclSuccess;

  if (pcclInjection == NULL) {
    ERROR("asys lib wrapper not initialized.");
    return ncclInternalError;
  }

  int32_t ret = pcclInjection();
  if (ret) {
    WARN("pcclInjection() failed !");
    return ncclSystemError;
  }

  asysPcclInjected = true;
  return ncclSuccess;
}
