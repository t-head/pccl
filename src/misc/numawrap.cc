//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include "numawrap.h"
#include "core.h"

#define NUMA_SO_NAME "libnuma.so.1"

static void (*numa_set_membind)(struct bitmask *nodemask);
static struct bitmask *(*numa_get_membind)(void);
static struct bitmask *(*numa_allocate_nodemask)(void);
static struct bitmask *(*numa_bitmask_setbit)(struct bitmask *bmp, unsigned int n);
static void (*numa_bitmask_free)(struct bitmask *bmp);

ncclResult_t wrap_numa_symbols(void) {
  static void *numahandle = NULL;
  if (numahandle) {
    return ncclSuccess;
  }

  numahandle = dlopen(NUMA_SO_NAME, RTLD_NOW);
  if (numahandle == NULL) {
    WARN("Failed to open %s: %s.", NUMA_SO_NAME, dlerror());
    goto teardown;
  }

#define LOAD_SYM(lib, func) do {          \
    func = (decltype(func))dlsym(lib, #func);     \
    if (func == NULL) {           \
      WARN("Failed to load symbol %s: %s.", #func, dlerror());\
      goto teardown;           \
    }               \
  } while(0)

  LOAD_SYM(numahandle, numa_set_membind);
  LOAD_SYM(numahandle, numa_get_membind);
  LOAD_SYM(numahandle, numa_allocate_nodemask);
  LOAD_SYM(numahandle, numa_bitmask_setbit);
  LOAD_SYM(numahandle, numa_bitmask_free);

  return ncclSuccess;

teardown:
  numa_set_membind = NULL;
  numa_get_membind = NULL;
  numa_allocate_nodemask = NULL;
  numa_bitmask_setbit = NULL;
  numa_bitmask_free = NULL;
  if (numahandle) {
    dlclose(numahandle);
    numahandle = NULL;
  }
  return ncclInternalError;
}

ncclResult_t wrap_numa_set_membind(struct bitmask *nodemask) {
  numa_set_membind(nodemask);
  return ncclSuccess;
}

ncclResult_t wrap_numa_get_membind(struct bitmask **nodemask) {
  *nodemask = numa_get_membind();
  return ncclSuccess;
}

ncclResult_t wrap_numa_allocate_nodemask(struct bitmask **nodemask) {
  *nodemask = numa_allocate_nodemask();
  return ncclSuccess;
}

ncclResult_t wrap_numa_bitmask_setbit(struct bitmask *nodemask, unsigned int n, struct bitmask **oldnodemask) {
  *oldnodemask = numa_bitmask_setbit(nodemask, n);
  return ncclSuccess;
}

ncclResult_t wrap_numa_bitmask_free(struct bitmask *nodemask) {
  numa_bitmask_free(nodemask);
  return ncclSuccess;
}

ncclResult_t wrap_numa_free_nodemask(struct bitmask *nodemask) {
  numa_bitmask_free(nodemask);
  return ncclSuccess;
}
