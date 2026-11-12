/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"

// size_t so GIN symmetric windows above 2GiB can be registered. NCCL is size_t throughout.
ncclResult_t ncclIbRegMrInternal(void* comm, void* data, size_t size, int type, uint64_t mrFlags,
                                 void** mhandle) {
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*)comm;
  struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, 0);
  uintptr_t addr = (uintptr_t)data;
  if (size == 0) {
    ERROR("Invalid size %zu for memory registration.", size);
    return ncclInvalidArgument;
  }

  long pageSize = sysconf(_SC_PAGESIZE);
  uintptr_t regAddr = addr - addr % (size_t)pageSize;
  size_t pageOffset = addr - regAddr;
  size_t regSize = ((size + pageOffset + pageSize - 1) / pageSize) * pageSize;
  unsigned int flags = IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ|IBV_ACCESS_REMOTE_ATOMIC;
  bool relaxedOrdering = ncclIbRelaxedOrderingEnabled && (mrFlags & NCCL_NET_MR_FLAG_FORCE_SO) == 0;
  if (relaxedOrdering) flags |= IBV_ACCESS_RELAXED_ORDERING;

  struct ncclIbDev* dev = ncclIbDevs + devComm->ibDevN;
  struct ncclIbMrCache* cache = &dev->mrCache;
  std::lock_guard<std::mutex> lock(dev->mutex);
  for (size_t i = 0; i < cache->population; i++) {
    struct ncclIbMr* entry = cache->slots + i;
    if (entry->type == type && entry->flags == flags && regAddr >= entry->addr &&
        regAddr - entry->addr <= entry->size && regSize <= entry->size - (regAddr - entry->addr)) {
      entry->refs++;
      *mhandle = entry->mr;
      return ncclSuccess;
    }
  }

  if (cache->population == cache->capacity) {
    size_t capacity = cache->capacity ? cache->capacity * 2 : 32;
    if (cache->slots == NULL) {
      NCCLCHECK(ncclCalloc(&cache->slots, capacity));
    } else {
      NCCLCHECK(ncclRealloc(&cache->slots, cache->population, capacity));
    }
    cache->capacity = capacity;
  }

  struct ibv_mr* mr;
  ncclResult_t result = relaxedOrdering
    ? wrap_ibv_reg_mr_iova2(&mr, devComm->pd, (void*)regAddr, regSize, regAddr, flags)
    : wrap_ibv_reg_mr(&mr, devComm->pd, (void*)regAddr, regSize, flags);
  if (result != ncclSuccess) {
    if (cache->population == 0) {
      free(cache->slots);
      cache->slots = NULL;
      cache->capacity = 0;
    }
    return result;
  }
  struct ncclIbMr* entry = cache->slots + cache->population++;
  entry->addr = regAddr;
  entry->size = regSize;
  entry->flags = flags;
  entry->type = type;
  entry->refs = 1;
  entry->mr = mr;
  *mhandle = (void*)mr;
  TRACE(NCCL_INIT,"regAddr %lx size %zu rkey %x", (unsigned long)regAddr, regSize, mr->rkey);
  return ncclSuccess;
}

// pcclNet_t entry point; the plugin ABI passes size as int. Larger callers use the size_t form.
ncclResult_t ncclIbRegMr(void* comm, void* data, int size, int type, void** mhandle) {
  if (size <= 0) {
    ERROR("Invalid size %d for memory registration.", size);
    return ncclInvalidArgument;
  }
  return ncclIbRegMrInternal(comm, data, (size_t)size, type, 0, mhandle);
}

ncclResult_t ncclIbDeregMr(void* comm, void* mhandle) {
  if (mhandle == NULL) return ncclSuccess;
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*)comm;
  struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, 0);
  struct ncclIbDev* dev = ncclIbDevs + devComm->ibDevN;
  struct ncclIbMrCache* cache = &dev->mrCache;
  std::lock_guard<std::mutex> lock(dev->mutex);
  for (size_t i = 0; i < cache->population; i++) {
    struct ncclIbMr* entry = cache->slots + i;
    if (entry->mr != mhandle) continue;
    if (entry->refs > 1) {
      entry->refs--;
    } else {
      NCCLCHECK(wrap_ibv_dereg_mr(entry->mr));
      cache->slots[i] = cache->slots[--cache->population];
      if (cache->population == 0) {
        free(cache->slots);
        cache->slots = NULL;
        cache->capacity = 0;
      }
    }
    return ncclSuccess;
  }
  WARN("NET/IB: Could not find MR %p in cache", mhandle);
  return ncclInternalError;
}
