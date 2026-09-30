/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "transport.h"
#include "group.h"
#include "hgtx.h"
#include "utils.h"
#include "bitops.h"
#include "allocator.h"

NCCL_API(ncclResult_t, pcclMemAlloc, void **ptr, size_t size);
ncclResult_t  pcclMemAlloc(void **ptr, size_t size) {
  INFO_CALL("PCCL API: pcclMemAlloc(ptr=%p, size=%ld)", ptr, size);
  PCCL_HGTX3_FUNC_RANGE;
  ncclResult_t ret = ncclSuccess;

  size_t memGran = 0;
  HGdevice currentDev;
  HGmemAllocationProp memprop = {};
  HGmemAccessDesc accessDesc = {};
  HGmemGenericAllocationHandle handle;
  int hggcDev;
  int flag;
  int dcnt;

  if (ptr == NULL || size == 0) goto fallback;

  if (ncclHggcLibraryInit() != ncclSuccess) goto fallback;

  HGGCCHECK(hggcGetDevice(&hggcDev));
  HGCHECK(hgDeviceGet(&currentDev, hggcDev));

  if (ncclCuMemEnable()) {
    size_t granularityVir = 0;
    size_t granularityPhy = 0;
    size_t sizeAlignedVir = 0;
    size_t sizeAlignedPhy = 0;

    int requestedHandleTypes = HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    // Query device to see if FABRIC handle support is available
    flag = 0;
    (void) hgDeviceGetAttribute(&flag, HG_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDev);
    if (flag) requestedHandleTypes |= HG_MEM_HANDLE_TYPE_FABRIC;
    memprop.type = HG_MEM_ALLOCATION_TYPE_PINNED;
    memprop.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
    memprop.requestedHandleTypes = (HGmemAllocationHandleType) requestedHandleTypes;
    memprop.location.id = currentDev;
    // Query device to see if RDMA support is available
    flag = 0;
    HGCHECK(hgDeviceGetAttribute(&flag, HG_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_HGGC_VMM_SUPPORTED, currentDev));
    if (flag) memprop.allocFlags.gpuDirectRDMACapable = 1;
    HGCHECK(hgMemGetAllocationGranularity(&granularityVir, &memprop, HG_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    HGCHECK(hggcEitInterface::getInstance().wrapHgMemGetAllocationGranularityEx(&granularityPhy, &memprop, HG_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    ALIGN_SIZE_WITH_RTN(size, granularityPhy, sizeAlignedPhy);
    ALIGN_SIZE_WITH_RTN(size, granularityVir, sizeAlignedVir);

    HGGCCHECK(hggcGetDeviceCount(&dcnt));

    if (requestedHandleTypes & HG_MEM_HANDLE_TYPE_FABRIC) {
      /* First try cuMemCreate() with FABRIC handle support and then remove if it fails */
      HGresult err = hggcEitInterface::getInstance().wrapHgMemCreateEx(&handle, sizeAlignedPhy, &memprop, HG_MEM_CREATE_MINI_GRANULARITY);
      if (err == HGGC_ERROR_NOT_PERMITTED || err == HGGC_ERROR_NOT_SUPPORTED) {
        requestedHandleTypes &= ~HG_MEM_HANDLE_TYPE_FABRIC;
        memprop.requestedHandleTypes = (HGmemAllocationHandleType) requestedHandleTypes;
        /* Allocate the physical memory on the device */
        HGCHECK(hggcEitInterface::getInstance().wrapHgMemCreateEx(&handle, sizeAlignedPhy, &memprop, HG_MEM_CREATE_MINI_GRANULARITY));
      }
    } else {
      /* Allocate the physical memory on the device */
      HGCHECK(hggcEitInterface::getInstance().wrapHgMemCreateEx(&handle, sizeAlignedPhy, &memprop, HG_MEM_CREATE_MINI_GRANULARITY));
    }
    /* Reserve a virtual address range */
    HGCHECK(hgMemAddressReserve((HGdeviceptr*)ptr, sizeAlignedVir, memGran, 0, 0));
    /* Map the virtual address range to the physical allocation */
    HGCHECK(hgMemMap((HGdeviceptr)*ptr, sizeAlignedPhy, 0, handle, 0));
    /* Now allow RW access to the newly mapped memory */
    for (int i = 0; i < dcnt; ++i) {
      int p2p = 0;
      if (i == hggcDev || ((hggcDeviceCanAccessPeer(&p2p, hggcDev, i) == hggcSuccess) && p2p)) {
        accessDesc.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = i;
        accessDesc.flags = HG_MEM_ACCESS_FLAGS_PROT_READWRITE;
        HGCHECK(hgMemSetAccess((HGdeviceptr)*ptr, sizeAlignedPhy, &accessDesc, 1));
      }
      if (0 == p2p && i != hggcDev) INFO(NCCL_ALLOC, "P2P not supported between GPU%d and GPU%d", hggcDev, i);
    }
    goto exit;
  }

fallback:
  // Coverity is right to complain that we may pass a NULL ptr to cudaMalloc.  That's deliberate though:
  // we want CUDA to return an error to the caller.
  // coverity[var_deref_model]
  HGGCCHECKGOTO(hggcMalloc(ptr, size), ret, fail);
exit:
  return ret;
fail:
  goto exit;
}

NCCL_API(ncclResult_t, pcclMemFree, void *ptr);
ncclResult_t  pcclMemFree(void *ptr) {
  INFO_CALL("PCCL API: pcclMemFree(ptr=%p)", ptr);
  PCCL_HGTX3_FUNC_RANGE;
  ncclResult_t ret = ncclSuccess;
  int saveDevice;

  HGGCCHECK(hggcGetDevice(&saveDevice));
  HGdevice ptrDev = 0;

  if (ptr == NULL) goto fallback;
  if (ncclHggcLibraryInit() != ncclSuccess) goto fallback;

  HGCHECKGOTO(hgPointerGetAttribute((void*)&ptrDev, HG_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (HGdeviceptr)ptr), ret, fail);
  HGGCCHECKGOTO(hggcSetDevice((int)ptrDev), ret, fail);
  if (ncclCuMemEnable()) {
    NCCLCHECKGOTO(ncclHgMemFree(ptr), ret, fail);
    goto exit;
  }

fallback:
  HGGCCHECKGOTO(hggcFree(ptr), ret, fail);

exit:
  HGGCCHECK(hggcSetDevice(saveDevice));
  return ret;
fail:
  goto exit;
}


////////////////////////////////////////////////////////////////////////////////
// ncclSpace:
//
// This datastructure "cuts" the line of non-negative integers into segments
// which alternate between "full" (allocated) and "empty" (not allocated). The
// cuts are sorted ascending. The segment after the last cut must be empty
// (the unallocated frontier). Knowing this we can deduce whether the segment
// ending at cut[i] is full or empty with this formula:
//   isFull(i) = (i%2 != ncuts%2)

void ncclSpaceConstruct(struct ncclSpace* a) {
  memset(a, 0, sizeof(*a));
}

void ncclSpaceDestruct(struct ncclSpace* a) {
  free(a->cuts);
}

static void insertSegment(struct ncclSpace* a, int index, int64_t lo, int64_t hi) {
  // Insert space for two cuts in a->cuts[] before index.
  if (a->count + 2 > a->capacity) {
    a->capacity *= 2;
    if (a->capacity == 0) a->capacity = 16;
    int64_t* cuts1 = (int64_t*)malloc(a->capacity * sizeof(int64_t));
    for (int i = 0; i < index; i++) cuts1[i] = a->cuts[i];
    for (int i = index; i < a->count; i++) cuts1[i + 2] = a->cuts[i];
    free(a->cuts);
    a->cuts = cuts1;
  } else {
    for (int i = a->count - 1; index <= i; i--) a->cuts[i + 2] = a->cuts[i];
  }
  a->cuts[index + 0] = lo;
  a->cuts[index + 1] = hi;
  a->count += 2;

  // Filter pairs of adjacent repeated values from cuts[]. Since these mark
  // boundaries where segments transition between full<->empty, dropping such a
  // pair fuses two adjacent segments together.
  int r = index, w = index;
  int64_t prev = r == 0 ? 0 : a->cuts[r - 1];
  while (r < a->count) {
    int64_t cur = a->cuts[r++];
    a->cuts[w++] = cur;
    if (prev == cur) {
      w -= w == 1 ? 1 : 2;
      cur = 0;
    }
    prev = cur;
  }
  a->count = w;
}

ncclResult_t ncclSpaceAlloc(
    struct ncclSpace* a, int64_t limit, int64_t size, int align,
    int64_t* outOffset
  ) {
  // Try to locate the first empty segment which can hold the allocation.
  int i = a->count % 2; // First empty segment ends at cuts[i]
  size_t off;
  while (i <= a->count) {
    size_t lo = i == 0 ? 0 : a->cuts[i - 1];
    size_t hi = i == a->count ? limit : a->cuts[i];
    off = alignUp(lo, align);
    if (off + size <= hi) {
      *outOffset = off;
      if (i == 0 || off + size == hi) {
        insertSegment(a, i, off, off + size);
      } else {
        a->cuts[i - 1] = off + size;
      }
      return ncclSuccess;
    }
    i += 2;
  }
  WARN("Allocation failed. No suitable space found to accommodate size=0x%lx within limit=0x%lx", (long)size, (long)limit);
  return ncclInternalError;
}

ncclResult_t ncclSpaceFree(struct ncclSpace* a, int64_t offset, int64_t size) {
  if (a->count == 0 || a->cuts[a->count - 1] <= offset) {
    WARN("No allocation found at offset=0x%lx", (long)offset);
    return ncclInternalError;
  }

  int i = 1 - a->count % 2; // First full segment ends at cuts[i]
  while (a->cuts[i] <= offset) i += 2;

  int64_t lo = i == 0 ? 0 : a->cuts[i - 1];
  int64_t hi = a->cuts[i];

  if (offset < lo || hi < offset + size) {
    WARN("Given size=0x%lx extends beyond allocation.", (long)size);
    return ncclInternalError;
  }

  if (i != 0 && lo == offset && offset + size != hi) {
    a->cuts[i - 1] = offset + size;
  } else if (lo != offset && offset + size == hi) {
    a->cuts[i] = offset;
  } else {
    insertSegment(a, i, offset, offset + size);
  }
  return ncclSuccess;
}

////////////////////////////////////////////////////////////////////////////////
// ncclShadowPool:

struct ncclShadowPage {
  struct ncclShadowPage* next;
  int objSize;
  uint64_t freeMask;
  void* devObjs;
};

struct ncclShadowObject {
  struct ncclShadowObject* next;
  void* devObj;
  void* hostObj;
  struct ncclShadowPage* page; // null if not allocated in page but directly from mempool
};

void ncclShadowPoolConstruct(struct ncclShadowPool* pool) {
  pool->hbits = 0;
  pool->count = 0;
  pool->table = nullptr;
  pool->pages = nullptr;
}

ncclResult_t ncclShadowPoolDestruct(struct ncclShadowPool* pool) {
  if (pool->hbits != 0) {
    hggcStream_t stream;
    HGGCCHECK(hggcStreamCreateWithFlags(&stream, hggcStreamNonBlocking));

    if (pool->count != 0) {
      for (int i = 0; i < 1 << pool->hbits; i++) {
        struct ncclShadowObject* obj = pool->table[i];
        while (obj != nullptr) {
          struct ncclShadowPage* page = obj->page;
          if (page != nullptr) {
            if (page->freeMask == 0) {
              page->freeMask = 1;
              page->next = pool->pages;
              pool->pages = page;
            }
          } else {
            hggcFreeAsync(obj->devObj, stream);
          }
          struct ncclShadowObject* next = obj->next;
          free(obj);
          obj = next;
        }
      }
    }
    free(pool->table);

    while (pool->pages != nullptr) {
      hggcFreeAsync(pool->pages->devObjs, stream);
      struct ncclShadowPage* next = pool->pages->next;
      free(pool->pages);
      pool->pages = next;
    }

    hggcStreamSynchronize(stream);
    hggcStreamDestroy(stream);
    hggcMemPoolDestroy(pool->memPool);
  }
  return ncclSuccess;
}

static void shadowHashInsert(struct ncclShadowPool* pool, struct ncclShadowObject* obj) {
  uint64_t b = ncclHashPointer(pool->hbits, obj->devObj);
  obj->next = pool->table[b];
  pool->table[b] = obj;
}

ncclResult_t ncclShadowPoolAlloc(
    struct ncclShadowPool* pool, size_t size, void** outDevObj, void** outHostObj,
    hggcStream_t stream
  ) {
  if (size == 0) {
    if (outDevObj) *outDevObj = nullptr;
    if (outHostObj) *outHostObj = nullptr;
    return ncclSuccess;
  }

  int hbits = pool->hbits;
  if (hbits == 0) {
    hggcMemPoolProps props = {};
    props.allocType = hggcMemAllocationTypePinned;
    props.handleTypes = hggcMemHandleTypeNone;
    props.location.type = hggcMemLocationTypeDevice;
    hggcGetDevice(&props.location.id);
    HGGCCHECK(hggcMemPoolCreate(&pool->memPool, &props));

    pool->hbits = hbits = 4;
    pool->table = (struct ncclShadowObject**)malloc(sizeof(struct ncclShadowObject*) << hbits);
    for (int i = 0; i < 1 << hbits; i++) pool->table[i] = nullptr;
  }

  // Check for hash table size increase. Maintain 2:1 object:bucket ratio.
  if (pool->count + 1 > 2 << hbits) {
    struct ncclShadowObject** table0 = pool->table;
    struct ncclShadowObject** table1 = (struct ncclShadowObject**)malloc(sizeof(struct ncclShadowObject*) << (hbits + 1));
    pool->table = table1;
    pool->hbits = hbits + 1;
    for (int i1 = 0; i1 < 2 << hbits; i1++) table1[i1] = nullptr;
    for (int i0 = 0; i0 < 1 << hbits; i0++) {
      struct ncclShadowObject* obj = table0[i0];
      while (obj) {
        struct ncclShadowObject* next = obj->next;
        shadowHashInsert(pool, obj);
        obj = next;
      }
    }
    hbits += 1;
    free(table0);
  }

  struct ncclShadowPage* page;
  void* devObj;
  if ((64 << 10) / size >= 3) {
    int shift = std::max<int>(0, (int)log2Down(size) + 1 - 4);
    int pageObjSize = ((size + (1 << shift) - 1) >> shift) << shift;
    struct ncclShadowPage** pagePtr = &pool->pages;
    while (true) {
      page = *pagePtr;
      if (page == nullptr) {
        size_t pageSize = std::min<size_t>(64 << 10, 64 * pageObjSize);
        page = (struct ncclShadowPage*)malloc(sizeof(struct ncclShadowPage));
        page->objSize = pageObjSize;
        page->freeMask = uint64_t(-1) >> (64 - pageSize / pageObjSize);
        page->next = pool->pages;
        pool->pages = page;
        HGGCCHECK(hggcMallocFromPoolAsync(&page->devObjs, pageSize, pool->memPool, stream));
        HGGCCHECK(hggcMemsetAsync(page->devObjs, 0, pageSize, stream));
        // fall through...
      }
      if (page->objSize == pageObjSize) {
        int slot = popFirstOneBit(&page->freeMask);
        devObj = (char*)page->devObjs + slot * pageObjSize;
        if (page->freeMask == 0) *pagePtr = page->next;
        break;
      }
      pagePtr = &page->next;
    }
  } else {
    page = nullptr;
    HGGCCHECK(hggcMallocFromPoolAsync(&devObj, size, pool->memPool, stream));
    HGGCCHECK(hggcMemsetAsync(devObj, 0, size, stream));
  }

  struct ncclShadowObject* obj = (struct ncclShadowObject*)malloc(
    sizeof(struct ncclShadowObject) + alignof(max_align_t) - 1 + size
  );
  obj->page = page;
  obj->devObj = devObj;
  obj->hostObj = (void*)alignUp((uintptr_t)(obj + 1), (int)alignof(max_align_t));
  memset(obj->hostObj, 0, size);
  shadowHashInsert(pool, obj);
  pool->count += 1;
  if (outDevObj) *outDevObj = devObj;
  if (outHostObj) *outHostObj = obj->hostObj;
  return ncclSuccess;
}

ncclResult_t ncclShadowPoolFree(struct ncclShadowPool* pool, void* devObj, hggcStream_t stream) {
  if (devObj == nullptr) return ncclSuccess;

  uint64_t b = ncclHashPointer(pool->hbits, devObj);
  struct ncclShadowObject** pobj = &pool->table[b];
  while (true) {
    if (*pobj == nullptr) {
      WARN("Device object does not exist in shadow pool.");
      return ncclInternalError;
    }
    if ((*pobj)->devObj == devObj) break;
    pobj = &(*pobj)->next;
  }
  struct ncclShadowObject* obj = *pobj;
  *pobj = obj->next;
  if (obj->page != nullptr) {
    if (obj->page->freeMask == 0) {
      obj->page->next = pool->pages;
      pool->pages = obj->page;
    }
    int slot = ((char*)obj->devObj - (char*)obj->page->devObjs) / obj->page->objSize;
    obj->page->freeMask |= uint64_t(1) << slot;
  } else {
    HGGCCHECK(hggcFreeAsync(devObj, stream));
  }
  free(obj);
  pool->count -= 1;
  return ncclSuccess;
}

ncclResult_t ncclShadowPoolToHost(struct ncclShadowPool* pool, void* devObj, void** hostObj) {
  if (devObj == nullptr) {
    *hostObj = nullptr;
    return ncclSuccess;
  }

  uint64_t b = ncclHashPointer(pool->hbits, devObj);
  struct ncclShadowObject* obj = pool->table[b];
  while (true) {
    if (obj == nullptr) {
      WARN("Device object does not exist in shadow pool.");
      return ncclInternalError;
    }
    if (obj->devObj == devObj) break;
    obj = obj->next;
  }
  *hostObj = obj->hostObj;
  return ncclSuccess;
}
