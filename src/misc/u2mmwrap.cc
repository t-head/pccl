//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include "core.h"
#include "u2mmwrap.h"

#define U2MM_SO_NAME "libu2mm.so"

int (*u2mm_set_log_level)(enum u2mm_log_level level);
int (*u2mm_alloc_init)(u2mm_alloc_config_t *config);
u2mm_alloc_config_t (*u2mm_get_default_alloc_config)(void);
void *(*u2mm_alloc)(size_t size, uint64_t flags);
int (*u2mm_free)(void *addr);
void (*u2mm_alloc_cleanup)(void);
int (*u2mm_shm_share)(void *addr, size_t len, char **mkey);
int (*u2mm_shm_unshare)(char *mkey);
int (*u2mm_shm_map)(const char *mkey, void **addr, size_t len);
int (*u2mm_shm_unmap)(void *addr);

ncclResult_t wrap_u2mm_symbols(void) {
  static void *u2mm_lib = NULL;
  if (u2mm_lib) {
    return ncclSuccess;
  }

  u2mm_lib = dlopen(U2MM_SO_NAME, RTLD_NOW);
  if (u2mm_lib == NULL) {
    WARN("Failed to open %s: %s.", U2MM_SO_NAME, dlerror());
    goto teardown;
  }

#define LOAD_SYM(lib, func) do { \
    func = (decltype(func))dlsym(lib, #func); \
    if (func == NULL) { \
      WARN("Failed to load symbol %s: %s.", #func, dlerror()); \
      goto teardown; \
    } \
  } while(0)

  LOAD_SYM(u2mm_lib, u2mm_set_log_level);
  LOAD_SYM(u2mm_lib, u2mm_alloc_init);
  LOAD_SYM(u2mm_lib, u2mm_get_default_alloc_config);
  LOAD_SYM(u2mm_lib, u2mm_alloc);
  LOAD_SYM(u2mm_lib, u2mm_free);
  LOAD_SYM(u2mm_lib, u2mm_alloc_cleanup);
  LOAD_SYM(u2mm_lib, u2mm_shm_share);
  LOAD_SYM(u2mm_lib, u2mm_shm_unshare);
  LOAD_SYM(u2mm_lib, u2mm_shm_map);
  LOAD_SYM(u2mm_lib, u2mm_shm_unmap);

  return ncclSuccess;
teardown:
  u2mm_set_log_level = NULL;
  u2mm_alloc_init = NULL;
  u2mm_get_default_alloc_config = NULL;
  u2mm_alloc = NULL;
  u2mm_free = NULL;
  u2mm_alloc_cleanup = NULL;
  u2mm_shm_share = NULL;
  u2mm_shm_unshare = NULL;
  u2mm_shm_map = NULL;
  u2mm_shm_unmap = NULL;
  if (u2mm_lib) {
    dlclose(u2mm_lib);
    u2mm_lib = NULL;
  }
  return ncclInternalError;
}
