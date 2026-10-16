/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <dlfcn.h>
#include <string.h>
#include <mutex>

#include "pccl_internal.h"
#include "debug.h"
#include "trocedvwrap.h"

static std::mutex ncclTroceHandleMutex;
static void *ncclTroceHandle = nullptr;
static struct ncclTroceFunctionTable ncclTroceFtable;
static int ncclTroceFtableState = 0;  // 0: untried, 1: ready, -1: unavailable

#define TROCE_LOAD_SYM(handle, name, fn)                              \
  do {                                                                \
    void *sym = dlsym(handle, name);                                  \
    if (sym == nullptr) {                                             \
      INFO(NCCL_NET, "GIN/GDAKI_TROCE: libtroce is missing symbol %s", name); \
      return false;                                                   \
    }                                                                 \
    fn = decltype(fn)(sym);                                           \
  } while (0)

static bool ncclTroceLoadSymbols(void *handle) {
  TROCE_LOAD_SYM(handle, "troce_create_qp_with_context", ncclTroceFtable.troce_create_qp_with_context);
  TROCE_LOAD_SYM(handle, "troce_destroy_qp_with_context", ncclTroceFtable.troce_destroy_qp_with_context);
  TROCE_LOAD_SYM(handle, "troce_create_cq_with_context", ncclTroceFtable.troce_create_cq_with_context);
  TROCE_LOAD_SYM(handle, "troce_destroy_cq_with_context", ncclTroceFtable.troce_destroy_cq_with_context);
  TROCE_LOAD_SYM(handle, "trocedv_query_qp_attr", ncclTroceFtable.trocedv_query_qp_attr);
  return true;
}

ncclResult_t ncclTroceFtableInit(struct ncclTroceFunctionTable **ftable) {
  std::lock_guard<std::mutex> lock(ncclTroceHandleMutex);

  if (ncclTroceFtableState == 1) {
    *ftable = &ncclTroceFtable;
    return ncclSuccess;
  }
  if (ncclTroceFtableState == -1) return ncclInvalidUsage;

  memset(&ncclTroceFtable, 0, sizeof(ncclTroceFtable));

  ncclTroceHandle = dlopen("libtroce.so", RTLD_LAZY);
  if (ncclTroceHandle == nullptr) ncclTroceHandle = dlopen("libtroce.so.1", RTLD_LAZY);
  if (ncclTroceHandle == nullptr) {
    INFO(NCCL_NET, "GIN/GDAKI_TROCE: cannot open libtroce.so[.1]: %s", dlerror());
    ncclTroceFtableState = -1;
    return ncclInvalidUsage;
  }

  if (!ncclTroceLoadSymbols(ncclTroceHandle)) {
    memset(&ncclTroceFtable, 0, sizeof(ncclTroceFtable));
    dlclose(ncclTroceHandle);
    ncclTroceHandle = nullptr;
    ncclTroceFtableState = -1;
    return ncclInvalidUsage;
  }

  INFO(NCCL_INIT | NCCL_NET, "GIN/GDAKI_TROCE: loaded libtroce GPU-queue interface");
  ncclTroceFtableState = 1;
  *ftable = &ncclTroceFtable;
  return ncclSuccess;
}

void ncclTroceFtableFini() {
  std::lock_guard<std::mutex> lock(ncclTroceHandleMutex);
  if (ncclTroceHandle) {
    dlclose(ncclTroceHandle);
    ncclTroceHandle = nullptr;
  }
  memset(&ncclTroceFtable, 0, sizeof(ncclTroceFtable));
  ncclTroceFtableState = 0;
}
