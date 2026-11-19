/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "afedwrap.h"
#include <dlfcn.h>
#include <errno.h>
#include <mutex>

static int (*afed_internal_query_num_lag_ports)(struct ibv_context*, uint8_t*);
static int (*afed_internal_set_qp_tx_affinity)(struct ibv_qp*, uint8_t, uint8_t);
static std::once_flag initOnceFlag;
static ncclResult_t initResult;

static ncclResult_t loadAfedSymbols(void) {
  static void* afedhandle = NULL;
  void* tmp;
  void** cast;
  Dl_info info = {};

  afedhandle = dlopen("libafed.so", RTLD_NOW);
  if (!afedhandle) afedhandle = dlopen("libafed.so.1", RTLD_NOW);
  if (!afedhandle) {
    INFO(NCCL_INIT | NCCL_NET, "NET/IB: TX affinity unavailable: %s", dlerror());
    goto teardown;
  }

#define LOAD_SYM(handle, symbol, funcptr) do { \
    cast = (void**)&funcptr; \
    tmp = dlsym(handle, symbol); \
    if (tmp == NULL) { \
      INFO(NCCL_INIT | NCCL_NET, "NET/IB: TX affinity unavailable: dlsym failed on %s - %s", symbol, dlerror()); \
      goto teardown; \
    } \
    *cast = tmp; \
  } while (0)

  LOAD_SYM(afedhandle, "afed_query_num_lag_ports", afed_internal_query_num_lag_ports);
  LOAD_SYM(afedhandle, "afed_set_qp_tx_affinity", afed_internal_set_qp_tx_affinity);
#undef LOAD_SYM

  // Retain the library for the process lifetime, as with the verbs wrappers.
  if (dladdr(tmp, &info))
    INFO(NCCL_NET, "NET/IB: TX affinity loaded from %s", info.dli_fname);
  return ncclSuccess;

teardown:
  afed_internal_query_num_lag_ports = NULL;
  afed_internal_set_qp_tx_affinity = NULL;
  if (afedhandle != NULL) dlclose(afedhandle);
  afedhandle = NULL;
  return ncclSystemError;
}

ncclResult_t wrap_afed_symbols(void) {
  std::call_once(initOnceFlag, [](){ initResult = loadAfedSymbols(); });
  return initResult;
}

int wrap_afed_query_num_lag_ports(struct ibv_context* context, uint8_t* num_lag_ports) {
  if (wrap_afed_symbols() != ncclSuccess) return ENOSYS;
  return afed_internal_query_num_lag_ports(context, num_lag_ports);
}

int wrap_afed_set_qp_tx_affinity(struct ibv_qp* qp, uint8_t num_lag_ports, uint8_t expect_egress) {
  if (wrap_afed_symbols() != ncclSuccess) return ENOSYS;
  return afed_internal_set_qp_tx_affinity(qp, num_lag_ports, expect_egress);
}
