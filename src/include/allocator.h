/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_ALLOCATOR_H_
#define NCCL_ALLOCATOR_H_

#include "pccl_internal.h"
#include "hggc_ad.h"

////////////////////////////////////////////////////////////////////////////////
// ncclSpace: Allocates contiguous segments of non-negative integers. Useful
// as a memory allocator when we can't put allocator state within the memory
// being allocated (e.g., device memory managed from host side).

struct ncclSpace {
  int count;
  int capacity;
  int64_t* cuts;
};

void ncclSpaceConstruct(struct ncclSpace* a);
void ncclSpaceDestruct(struct ncclSpace* a);
ncclResult_t ncclSpaceAlloc(struct ncclSpace* a, int64_t spaceLimit, int64_t objSize, int objAlign, int64_t* outObjOffset);
ncclResult_t ncclSpaceFree(struct ncclSpace* a, int64_t objOffset, int64_t objSize);

////////////////////////////////////////////////////////////////////////////////
// ncclShadowPool: Allocates device-side objects, their host-side shadows, and
// maintains the device->host object address mapping.
//
// The shadow pool is used to maintain host-side copies of device-resident
// structures (like ncclWindow_vidmem). The device allocations come from an
// hggcMemPool (async pool allocator), and each device object gets a host-side
// "shadow" that can be read/written without device synchronization.

struct ncclShadowObject;
struct ncclShadowPage;
struct ncclShadowPool {
  int count, hbits;
  struct ncclShadowObject** table;
  hggcMemPool_t memPool;
  struct ncclShadowPage* pages;
};

void ncclShadowPoolConstruct(struct ncclShadowPool*);
ncclResult_t ncclShadowPoolDestruct(struct ncclShadowPool*);
ncclResult_t ncclShadowPoolAlloc(struct ncclShadowPool*, size_t size, void** outDevObj, void** outHostObj, hggcStream_t stream);
ncclResult_t ncclShadowPoolFree(struct ncclShadowPool*, void* devObj, hggcStream_t stream);
ncclResult_t ncclShadowPoolToHost(struct ncclShadowPool*, void* devObj, void** outHostObj);

template<typename T>
static inline ncclResult_t ncclShadowPoolAlloc(struct ncclShadowPool* pool, T** outDevObj, T** outHostObj, hggcStream_t stream) {
  void* devObj;
  void* hostObj;
  ncclResult_t got = ncclShadowPoolAlloc(pool, sizeof(T), &devObj, &hostObj, stream);
  if (outDevObj) *outDevObj = (T*)devObj;
  if (outHostObj) *outHostObj = (T*)hostObj;
  return got;
}

template<typename T>
static inline ncclResult_t ncclShadowPoolToHost(struct ncclShadowPool* pool, T* devObj, T** hostObj) {
  return ncclShadowPoolToHost(pool, (void*)devObj, (void**)hostObj);
}

#endif
