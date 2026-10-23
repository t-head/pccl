/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_ALLOC_H_
#define NCCL_ALLOC_H_

#include "pccl_internal.h"
#include "checks.h"
#include "bitops.h"
#include "param.h"
#include "u2mm.h"
#include "drv_extension.h"
#include <hggc.h>
#include "cudawrap.h"

#include "assert.h"
#include <unistd.h>
#include <stdio.h>
#include <sys/mman.h>
#include <iostream>

enum {
  HW_PPU = 0,             // ppu mode.
  HW_BEHAVIOR = 1,        // behavior mode.
  HW_BIT_ACCURATE = 2,    // bit accurate.
  HW_PERF = 3,            // perf mode.
};
extern int gModelType;

extern int ncclPerfmodelSpeedup;
extern void* ncclPerfmodelHostPtr[8];
extern void* ncclPerfmodelDevicePtr[8];
extern size_t ncclPerfmodelMemOffset[8];
extern const size_t ncclPerfmodelTotalDeviceMemSize;
extern const size_t ncclPerfmodelDevMemAlignSize;
extern int ncclPerfmodelNeedCopy[8];

template<typename T>
constexpr size_t ncclSizeOfT() { return sizeof(T); }
template<>
constexpr size_t ncclSizeOfT<void>() { return 1; }

enum {
  MEM_NORMAL = 0,       // normal ppu mem alloc & free mode.
  MEM_U2MM = 1,         // u2mm mem alloc & free mode.
  MEM_ALIGN_2M = 2,     // u2mm host mem alloc for old rund arch
  MEM_U2MM_DIRECT = 3,  // direct alloc u2mm cuda mem
  MEM_LEGACY = 4        // runtime device allocation; caller initializes memory
};

// use allocFlag to check if we need other memory allocation method like u2mm alloc
template <typename T>
static ncclResult_t ncclHggcHostCallocU2mmDebug(T** ptr, size_t nelem, uint64_t* pPinnedMemUsed, int memMode, int dev, const char *filefunc, int line) {
  ncclResult_t result = ncclSuccess;
  hggcStreamCaptureMode mode = hggcStreamCaptureModeRelaxed;
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));

  if (memMode == MEM_U2MM_DIRECT) {
    ERROR("Unsupported to alloc u2mm direct host memory.");
    result = ncclInternalError;
    goto finish;
  }

  if (memMode == MEM_U2MM) {
    NCCLCHECKGOTO(nccl_u2mm_host_calloc(ptr, nelem, dev), result, finish);
    goto finish;
  }

  if (memMode == MEM_ALIGN_2M) {
#define PAGE_SIZE_2M (1 << 21)
    if (posix_memalign((void **)ptr, PAGE_SIZE_2M, ROUNDUP(nelem * sizeof(T), PAGE_SIZE_2M))) {
#undef PAGE_SIZE_2M
      ERROR("Failed to alloc %zu host memory.", nelem * sizeof(T));
      result = ncclInternalError;
      goto finish;
    }
    if (hggcHostRegister(*ptr, nelem * sizeof(T), hggcHostRegisterMapped) != hggcSuccess) {
      ERROR("Failed to register: %p size: %zu.", *ptr, nelem * sizeof(T));
      free(*ptr);
      *ptr = NULL;
      result = ncclInternalError;
      goto finish;
    }
    VERBOSE(NCCL_ALLOC, "Alloc host memory pointer: %p, size: %zu.", *ptr, nelem * sizeof(T));
    memset(*ptr, 0, nelem * sizeof(T));
  } else {
    VERBOSE(NCCL_ALLOC, "hggcHostAlloc(ptr=%p)", *ptr);
    HGGCCHECKGOTO(hggcHostAlloc(ptr, nelem*sizeof(T), hggcHostAllocMapped), result, finish);
    memset(*ptr, 0, nelem*sizeof(T));
    VERBOSE(NCCL_ALLOC, "%s:%d Hggc Host Alloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), *ptr);
  }

finish:
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
  if (result == ncclSuccess && pPinnedMemUsed) *pPinnedMemUsed += nelem * sizeof(T);
  return result;
}

#define ncclHggcHostCallocU2mm(...) ncclHggcHostCallocU2mmDebug(__VA_ARGS__, __FILENAME__, __LINE__)

template <typename T>
static ncclResult_t ncclHggcHostCallocDebug(T** ptr, size_t nelem, uint64_t* pPinnedMemUsed, const char *filefunc, int line) {
  return ncclHggcHostCallocU2mmDebug(ptr, nelem, pPinnedMemUsed, MEM_NORMAL, 0, filefunc, line);
}

#define ncclHggcHostCalloc(...) ncclHggcHostCallocDebug(__VA_ARGS__, __FILENAME__, __LINE__)

// ncclHgMemAllocAddr takes memory handle and size and returns the mapped address pointer
static inline ncclResult_t ncclHgMemAllocAddr(void **ptr, HGmemGenericAllocationHandle *handleIn, size_t size, int* rootDev) {
  ncclResult_t result = ncclSuccess;
  size_t granularityVir = 0;
  size_t granularityPhy = 0;
  size_t sizeAlignedVir = 0;
  size_t sizeAlignedPhy = 0;
  HGmemAllocationProp prop = {};
  HGmemAccessDesc accessDesc = {};
  int hggcDev = -1;
  VERBOSE(NCCL_ALLOC, "pcclHgMemAllocAddr(ptr=%p, handle=%llx, size=%zu)", ptr, *handleIn, size);
  HGGCCHECK(hggcGetDevice(&hggcDev));
  HGCHECK(hgMemGetAllocationPropertiesFromHandle(&prop, *handleIn));
  HGCHECK(hgMemGetAllocationGranularity(&granularityVir, &prop, HG_MEM_ALLOC_GRANULARITY_MINIMUM));
  HGCHECK(hggcEitInterface::getInstance().wrapHgMemGetAllocationGranularityEx(&granularityPhy, &prop, HG_MEM_ALLOC_GRANULARITY_MINIMUM));
  ALIGN_SIZE_WITH_RTN(size, granularityPhy, sizeAlignedPhy);
  ALIGN_SIZE_WITH_RTN(size, granularityVir, sizeAlignedVir);
  /* Reserve a virtual address range */
  HGCHECK(hgMemAddressReserve((HGdeviceptr *)ptr, sizeAlignedVir, granularityVir, 0, 0));
  /* Map the virtual address range to the physical allocation */
  HGCHECK(hgMemMap((HGdeviceptr)*ptr, sizeAlignedPhy, 0, *handleIn, 0));
  /* Now allow RW access to the newly mapped memory */
  accessDesc.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = hggcDev;
  accessDesc.flags = HG_MEM_ACCESS_FLAGS_PROT_READWRITE;
  HGCHECK(hgMemSetAccess((HGdeviceptr)*ptr, sizeAlignedPhy, &accessDesc, 1));

  VERBOSE(NCCL_ALLOC, "HgMem Map Size %zu pointer %p handle %llx", size, *ptr, *handleIn);
  return result;
}

static inline ncclResult_t ncclHgMemFreeAddr(void *ptr) {
  if (ptr == NULL) return ncclSuccess;
  ncclResult_t result = ncclSuccess;
  HGmemGenericAllocationHandle handle;
  size_t sizeAlignedVir = 0;
  size_t sizeAlignedPhy = 0;
  size_t granularityVir = 0;
  HGmemAllocationProp prop = {};

  VERBOSE(NCCL_ALLOC, "pcclHgMemFreeAddr(ptr=%p)", ptr);
  HGCHECK(hgMemRetainAllocationHandle(&handle, ptr));
  HGCHECK(hgMemRelease(handle));
  HGCHECK(hgMemGetAllocationPropertiesFromHandle(&prop, handle));
  HGCHECK(hgMemGetAllocationGranularity(&granularityVir, &prop, HG_MEM_ALLOC_GRANULARITY_MINIMUM));

  HGCHECK(hgMemGetAddressRange(NULL, &sizeAlignedPhy, (HGdeviceptr)ptr));
  ALIGN_SIZE_WITH_RTN(sizeAlignedPhy, granularityVir, sizeAlignedVir);

  HGCHECK(hgMemUnmap((HGdeviceptr)ptr, sizeAlignedPhy));
  HGCHECK(hgMemAddressFree((HGdeviceptr)ptr, sizeAlignedVir));
  return result;
}

static inline ncclResult_t ncclHgMemAlloc(void **ptr, HGmemGenericAllocationHandle *handlep, HGmemAllocationHandleType type, size_t size,
  uint64_t *pDevMemUsed, size_t *pSizeAlignedPhy, int memMode = MEM_NORMAL) {
  ncclResult_t result = ncclSuccess;
  size_t granularityVir = 0;
  size_t granularityPhy = 0;
  size_t sizeAlignedVir = 0;
  size_t sizeAlignedPhy = 0;
  HGdevice currentDev;
  HGmemAllocationProp prop = {};
  HGmemAccessDesc accessDesc = {};
  HGmemGenericAllocationHandle handle;
  int hggcDev;
  int flag = 0;
  VERBOSE(NCCL_ALLOC, "pcclHgMemAlloc(ptr=%p, size=%zu)", ptr, size);
  HGGCCHECK(hggcGetDevice(&hggcDev));
  HGCHECK(hgDeviceGet(&currentDev, hggcDev));
  prop.type = HG_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
  prop.requestedHandleTypes = type;
  prop.location.id = currentDev;
  // Query device to see if RDMA support is available
  HGCHECK(hgDeviceGetAttribute(&flag, HG_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, currentDev));
  if (flag) prop.allocFlags.gpuDirectRDMACapable = 1;
  HGCHECK(hgMemGetAllocationGranularity(&granularityVir, &prop, HG_MEM_ALLOC_GRANULARITY_MINIMUM));
  HGCHECK(hggcEitInterface::getInstance().wrapHgMemGetAllocationGranularityEx(&granularityPhy, &prop, HG_MEM_ALLOC_GRANULARITY_MINIMUM));
  ALIGN_SIZE_WITH_RTN(size, granularityPhy, sizeAlignedPhy);
  ALIGN_SIZE_WITH_RTN(size, granularityVir, sizeAlignedVir);
  /* Allocate the physical memory on the device */
  HGCHECK(hggcEitInterface::getInstance().wrapHgMemCreateEx(&handle, sizeAlignedPhy, &prop, HG_MEM_CREATE_MINI_GRANULARITY));
  /* Reserve a virtual address range */
  HGCHECK(hgMemAddressReserve((HGdeviceptr *)ptr, sizeAlignedVir, 0, 0, 0));
  /* Map the virtual address range to the physical allocation */
  HGCHECK(hgMemMap((HGdeviceptr)*ptr, sizeAlignedPhy, 0, handle, 0));
  /* Now allow RW access to the newly mapped memory */
  accessDesc.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = currentDev;
  accessDesc.flags = HG_MEM_ACCESS_FLAGS_PROT_READWRITE;
  HGCHECK(hgMemSetAccess((HGdeviceptr)*ptr, sizeAlignedPhy, &accessDesc, 1));
  if (handlep) *handlep = handle;
  if (result == ncclSuccess && pDevMemUsed) *pDevMemUsed += sizeAlignedPhy;
  if (pSizeAlignedPhy) *pSizeAlignedPhy = sizeAlignedPhy;
  VERBOSE(NCCL_ALLOC, "HgMem Alloc Size %zi sizeAlignedPhy %zi sizeAlignedVir %zi pointer %p handle %llx", size, sizeAlignedPhy, sizeAlignedVir, *ptr, handle);
  return result;
}

static inline ncclResult_t ncclHgMemFree(void *ptr) {
  if (ptr == NULL) {
    return ncclSuccess;
  }
  ncclResult_t result = ncclSuccess;
  HGmemGenericAllocationHandle handle;
  size_t sizeAlignedVir = 0;
  size_t sizeAlignedPhy = 0;
  size_t granularityVir = 0;
  HGmemAllocationProp prop = {};

  VERBOSE(NCCL_ALLOC, "pcclHgMemFree(ptr=%p)", ptr);
  HGCHECK(hgMemRetainAllocationHandle(&handle, ptr));
  HGCHECK(hgMemRelease(handle));
  HGCHECK(hgMemGetAllocationPropertiesFromHandle(&prop, handle));
  HGCHECK(hgMemGetAllocationGranularity(&granularityVir, &prop, HG_MEM_ALLOC_GRANULARITY_MINIMUM));

  HGCHECK(hgMemGetAddressRange(NULL, &sizeAlignedPhy, (HGdeviceptr)ptr));
  ALIGN_SIZE_WITH_RTN(sizeAlignedPhy, granularityVir, sizeAlignedVir);

  VERBOSE(NCCL_ALLOC, "HgMem Free sizeAlignedPhy %zi sizeAlignedVir %zi pointer %p handle 0x%llx", sizeAlignedPhy, sizeAlignedVir, ptr, handle);

  HGCHECK(hgMemUnmap((HGdeviceptr)ptr, sizeAlignedPhy));
  HGCHECK(hgMemRelease(handle));

  HGCHECK(hgMemAddressFree((HGdeviceptr)ptr, sizeAlignedVir));
  return result;
}

template <typename T>
ncclResult_t ncclHggcCallocAsyncDebug(T** ptr, size_t nelem, uint64_t* pDevMemUsed, hggcStream_t stream, const char *filefunc, int line) {
  ncclResult_t result = ncclSuccess;
  size_t sizeAligned = nelem * sizeof(T);
  hggcStreamCaptureMode mode = hggcStreamCaptureModeRelaxed;
  *ptr = nullptr;
  VERBOSE(NCCL_ALLOC, "pcclHggcCallocAsync(ptr=%p, nelem=%zu, stream=%p)", ptr, nelem, stream);
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
  if (ncclCuMemEnable()) {
    NCCLCHECKGOTO(ncclHgMemAlloc((void **)ptr, NULL, ncclHgMemHandleType, nelem*sizeof(T), pDevMemUsed, &sizeAligned), result, finish);
  } else {
    HGGCCHECKGOTO(hggcMalloc(ptr, nelem*sizeof(T)), result, finish);
  }
  HGGCCHECKGOTO(hggcMemsetAsync(*ptr, 0, nelem*sizeof(T), stream), result, finish);
finish:
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr) ERROR("Failed to HGGC calloc async %ld bytes", nelem*sizeof(T));
  if (result == ncclSuccess && pDevMemUsed) *pDevMemUsed += sizeAligned;
  VERBOSE(NCCL_ALLOC, "%s:%d Hggc Alloc Size %zi sizeAligned %zi pointer %p", filefunc, line, nelem*sizeof(T), sizeAligned, *ptr);
  return result;
}
#define ncclHggcCallocAsync(...) ncclHggcCallocAsyncDebug(__VA_ARGS__, __FILENAME__, __LINE__)


static ncclResult_t ncclHggcHostFree(void* ptr, int memMode=MEM_NORMAL) {
  ncclResult_t result = ncclSuccess;
  if (ptr == NULL) return result;
  hggcStreamCaptureMode mode = hggcStreamCaptureModeRelaxed;
  VERBOSE(NCCL_ALLOC, "Hggc Host Free pointer %p", ptr);
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));

  if (memMode == MEM_U2MM_DIRECT) {
    ERROR("Unsupport to free u2mm direct host memory.");
    result = ncclInternalError;
    goto finish;
  }

  if (memMode == MEM_U2MM) {
    NCCLCHECKGOTO(nccl_u2mm_host_free(ptr), result, finish);
  } else if (memMode == MEM_ALIGN_2M) {
    if (hggcHostUnregister(ptr) != hggcSuccess) {
      ERROR("Failed to hggc host unregister: %p", ptr);
      free(ptr);
      ptr = NULL;
      result = ncclInternalError;
      goto finish;
    }
    free(ptr);
    ptr = NULL;
  } else {
    HGGCCHECKGOTO(hggcFreeHost(ptr), result, finish);
    ptr = NULL;
  }

finish:
  if (result != ncclSuccess) WARN("%s: free %p failed", __func__, ptr);
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
  return result;
}

template <typename T>
ncclResult_t ncclCallocDebug(T** ptr, size_t nelem, const char *filefunc, int line) {
  if (nelem > 0) {
    void* p = malloc(nelem*ncclSizeOfT<T>());
    if (p == NULL) {
      WARN("Failed to malloc %ld bytes", nelem*ncclSizeOfT<T>());
      return ncclSystemError;
    }
    //TRACE(NCCL_ALLOC, "%s:%d malloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), p);
    memset(p, 0, nelem*ncclSizeOfT<T>());
    *ptr = (T*)p;
  } else {
    *ptr = NULL;
  }
  return ncclSuccess;
}
#define ncclCalloc(...) ncclCallocDebug(__VA_ARGS__, __FILENAME__, __LINE__)

template <typename T>
static ncclResult_t ncclMemset(T* ptr, size_t nelem) {
  void* p = ptr;
  if (p == NULL) {
    ERROR("Failed to memset %ld bytes", nelem*sizeof(T));
    return ncclSystemError;
  }
  memset(p, 0, nelem*sizeof(T));
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclFree(T*& ptr) {
  if(ptr != NULL) free(ptr);
  ptr = NULL;
  return ncclSuccess;
}

extern int ncclParamNoMemsetFlag;

template <typename T>
static ncclResult_t ncclHggcCallocOrigin(T** ptr, size_t nelem, size_t *sizeAligned, int memMode) {
  if (nelem == 0) {
    return ncclSuccess;
  }
  if (memMode == MEM_U2MM || memMode == MEM_U2MM_DIRECT) {
    return nccl_u2mm_gpu_calloc(ptr, nelem, memMode == MEM_U2MM_DIRECT);
  }

  hggcStream_t stream;
  VERBOSE(NCCL_ALLOC, "pcclHggcCallocOrigin(ptr=%p, nelem=%zu)", ptr, nelem);
  HGGCCHECK(hggcEitInterface::getInstance().wrapHggcStreamCreateForCopyEngine(&stream));
  if (ncclCuMemEnable()) {
    NCCLCHECK(ncclHgMemAlloc((void **)ptr, NULL, ncclHgMemHandleType, nelem*sizeof(T), nullptr, sizeAligned, memMode));
  } else {
    HGGCCHECK(hggcMalloc(ptr, nelem*sizeof(T)));
  }
  if (ncclParamNoMemsetFlag == 0) {
    // for bit accurate mode.
    if (gModelType == HW_PERF) {
        T* host_ptr = NULL;
        ncclCalloc(&host_ptr, nelem);
        HGGCCHECK(hggcMemcpyAsync(*ptr, host_ptr, nelem*sizeof(T), hggcMemcpyDefault, stream));
        free((void*)host_ptr);
    } else {
      HGGCCHECK(hggcMemsetAsync(*ptr, 0, nelem*ncclSizeOfT<T>(), stream));
    }
  }
  HGGCCHECK(hggcStreamSynchronize(stream));
  HGGCCHECK(hggcStreamDestroy(stream));
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclRealloc(T** ptr, size_t oldNelem, size_t nelem) {
  if (nelem < oldNelem) return ncclInternalError;
  if (nelem == oldNelem) return ncclSuccess;

  T* oldp = *ptr;
  T* p = (T*)malloc(nelem*sizeof(T));
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem*sizeof(T));
    return ncclSystemError;
  }
  memcpy(p, oldp, oldNelem*sizeof(T));
  NCCLCHECK(ncclFree(oldp));
  memset(p+oldNelem, 0, (nelem-oldNelem)*sizeof(T));
  *ptr = (T*)p;
  //VERBOSE(NCCL_ALLOC, "Mem Realloc old size %ld, new size %ld pointer %p", oldNelem*sizeof(T), nelem*sizeof(T), *ptr);
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclHggcCallocPerfmodelSpeedup(T** ptr, size_t nelem, size_t *sizeAligned) {
  if (ncclCuMemEnable()) {
    NCCLCHECK(ncclHgMemAlloc((void **)ptr, NULL, ncclHgMemHandleType, nelem*sizeof(T), nullptr, sizeAligned));
  } else {
    VERBOSE(NCCL_ALLOC, "hggcMalloc(ptr=%p)", *ptr);
    HGGCCHECK(hggcMalloc(ptr, nelem*sizeof(T)));
  }

  int dev;
  HGGCCHECK(hggcGetDevice(&dev));
  *ptr = (T*)ncclPerfmodelDevicePtr[dev] + ncclPerfmodelMemOffset[dev]/sizeof(T);
  int align_size = (nelem*sizeof(T) + ncclPerfmodelDevMemAlignSize - 1) / ncclPerfmodelDevMemAlignSize * ncclPerfmodelDevMemAlignSize;
  ncclPerfmodelMemOffset[dev] += align_size;
  ncclPerfmodelNeedCopy[dev] = 1;
  if (ncclPerfmodelMemOffset[dev] > ncclPerfmodelTotalDeviceMemSize) {
    ERROR("hggc device memory exceed %lx, you need to increase the total memory size", ncclPerfmodelTotalDeviceMemSize);
    return ncclSystemError;
  }

  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclHggcCallocU2mmDebug(T** ptr, size_t nelem, uint64_t* pDevMemUsed, int memMode, const char *filefunc, int line) {
  ncclResult_t result = ncclSuccess;
  size_t sizeAligned = nelem * sizeof(T);
  hggcStreamCaptureMode mode = hggcStreamCaptureModeRelaxed;
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));

  if (memMode == MEM_LEGACY) {
    HGGCCHECKGOTO(hggcMalloc(ptr, nelem*sizeof(T)), result, finish);
  } else if (ncclPerfmodelSpeedup) {
    assert(memMode == MEM_NORMAL);
    NCCLCHECKGOTO(ncclHggcCallocPerfmodelSpeedup(ptr, nelem, &sizeAligned), result, finish);
  } else {
    // use memMode to specify other kinds of memory alloc
    NCCLCHECKGOTO(ncclHggcCallocOrigin(ptr, nelem, &sizeAligned, memMode), result, finish);
  }

finish:
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
  if (result == ncclSuccess && pDevMemUsed) *pDevMemUsed += sizeAligned;
  VERBOSE(NCCL_ALLOC, "%s:%d Hggc Alloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), *ptr);
  return result;
}

#define ncclHggcCallocU2mm(...) ncclHggcCallocU2mmDebug(__VA_ARGS__, __FILENAME__, __LINE__)

template <typename T>
static ncclResult_t ncclHggcCallocDebug(T** ptr, size_t nelem, uint64_t* pDevMemUsed, const char *filefunc, int line) {
  return ncclHggcCallocU2mmDebug(ptr, nelem, pDevMemUsed, MEM_NORMAL, filefunc, line);
}

#define ncclHggcCalloc(...) ncclHggcCallocDebug(__VA_ARGS__, __FILENAME__, __LINE__)

template <typename T>
static ncclResult_t ncclHggcMemcpy(T* dst, T* src, size_t nelem) {
  ncclResult_t result = ncclSuccess;
  hggcStreamCaptureMode mode = hggcStreamCaptureModeRelaxed;
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));

  if (ncclPerfmodelSpeedup) {
    int dev;
    HGGCCHECKGOTO(hggcGetDevice(&dev), result, finish);
    size_t off = (char*)dst - (char*)ncclPerfmodelDevicePtr[dev];
    memcpy((T*)ncclPerfmodelHostPtr[dev]+off/sizeof(T), src, nelem*sizeof(T));
    ncclPerfmodelNeedCopy[dev] = 1;
  } else {
    HGGCCHECKGOTO(hggcMemcpy(dst, src, nelem*sizeof(T), hggcMemcpyDefault), result, finish);
  }

finish:
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
  return result;
}

template <typename T>
static ncclResult_t ncclHggcMemset(T* ptr, int val, size_t nelem) {
  ncclResult_t result = ncclSuccess;
  hggcStream_t stream;
  hggcStreamCaptureMode mode = hggcStreamCaptureModeRelaxed;
  VERBOSE(NCCL_ALLOC, "pcclHggcMemset(ptr=%p, nelem=%zu)", ptr, nelem);
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));

  // use memMode to specify other kinds of memory alloc
  HGGCCHECKGOTO(hggcEitInterface::getInstance().wrapHggcStreamCreateForCopyEngine(&stream), result, finish);
  HGGCCHECKGOTO(hggcMemsetAsync(ptr, val, nelem*ncclSizeOfT<T>(), stream), result, finish);
  HGGCCHECKGOTO(hggcStreamSynchronize(stream), result, finish);
  HGGCCHECKGOTO(hggcStreamDestroy(stream), result, finish);

finish:
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
  return result;
}

template <typename T, int needSwitchCaptureMode=1>
ncclResult_t ncclHggcMemcpyAsync(T* dst, T* src, size_t nelem, hggcStream_t stream) {
  ncclResult_t result = ncclSuccess;
  hggcStreamCaptureMode mode = hggcStreamCaptureModeRelaxed;
  VERBOSE(NCCL_ALLOC, "pcclHggcMemcpyAsync(dst=%p, src=%p, stream=%p)", dst, src, stream);
  if (needSwitchCaptureMode) HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
  HGGCCHECKGOTO(hggcMemcpyAsync(dst, src, nelem*ncclSizeOfT<T>(), hggcMemcpyDefault, stream), result, finish);
finish:
  if (needSwitchCaptureMode) HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
  return result;
}

template <typename T, int needSwitchCaptureMode=1>
static ncclResult_t ncclHggcMemsetAsync(T* ptr, int val, size_t nelem, hggcStream_t stream) {
  ncclResult_t result = ncclSuccess;
  hggcStreamCaptureMode mode = hggcStreamCaptureModeRelaxed;
  VERBOSE(NCCL_ALLOC, "pcclHggcMemsetAsync(ptr=%p, nelem=%zu, stream=%p)", ptr, nelem, stream);
  if (needSwitchCaptureMode) HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));

  // use memMode to specify other kinds of memory alloc
  HGGCCHECKGOTO(hggcMemsetAsync(ptr, val, nelem*ncclSizeOfT<T>(), stream), result, finish);

finish:
  if (needSwitchCaptureMode) HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
  return result;
}

template <typename T>
static ncclResult_t ncclHggcFreeDebug(const char *filefunc, int line, T* ptr, int memMode=MEM_NORMAL) {
  ncclResult_t result = ncclSuccess;
  hggcStreamCaptureMode mode = hggcStreamCaptureModeRelaxed;
  VERBOSE(NCCL_ALLOC, "%s:%d Hggc Free pointer %p", filefunc, line, ptr);
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));

  if (memMode == MEM_LEGACY) {
    HGGCCHECKGOTO(hggcFree(ptr), result, finish);
  } else if (ncclPerfmodelSpeedup) {
    int dev;
    HGGCCHECKGOTO(hggcGetDevice(&dev), result, finish);
    size_t off = (size_t)((char*)ptr - (char*)ncclPerfmodelDevicePtr[dev]);
    if (off < 0 || off >= ncclPerfmodelTotalDeviceMemSize) {
      INFO(NCCL_INIT, "perf acc mode, ptr:%p out of the pcclPerfmodelDevicePtr(%p):+0x%lx range", ptr, ncclPerfmodelDevicePtr[dev], ncclPerfmodelTotalDeviceMemSize);
      if (ncclCuMemEnable()) {
        NCCLCHECKGOTO(ncclHgMemFree((void *)ptr), result, finish);
      } else {
        HGGCCHECKGOTO(hggcFree(ptr), result, finish);
      }
    }
  } else {
    if (memMode == MEM_U2MM) {
      NCCLCHECKGOTO(nccl_u2mm_gpu_free(ptr), result, finish);
    } else {
      if (ncclCuMemEnable()) {
        NCCLCHECKGOTO(ncclHgMemFree((void *)ptr), result, finish);
      } else {
        HGGCCHECKGOTO(hggcFree(ptr), result, finish);
      }
    }
  }
  ptr = NULL;

finish:
  if (result != ncclSuccess) WARN("%s: free %p failed", __func__, ptr);
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
  return result;
}

#define ncclHggcFree(...) ncclHggcFreeDebug(__FILENAME__, __LINE__, __VA_ARGS__)

// Allocate memory to be potentially ibv_reg_mr'd. This needs to be
// allocated on separate pages as those pages will be marked DONTFORK
// and if they are shared, that could cause a crash in a child process
static ncclResult_t ncclIbMallocDebug(void** ptr, size_t size, const char *filefunc, int line) {
  size_t page_size = sysconf(_SC_PAGESIZE);
  void* p;
  int size_aligned = ROUNDUP(size, page_size);
  int ret = posix_memalign(&p, page_size, size_aligned);
  if (ret != 0) return ncclSystemError;
  memset(p, 0, size);
  *ptr = p;

  VERBOSE(NCCL_ALLOC, "%s:%d Ib Alloc Size %ld pointer %p", filefunc, line, size, *ptr);
  return ncclSuccess;
}

#define ncclIbMalloc(...) ncclIbMallocDebug(__VA_ARGS__, __FILENAME__, __LINE__)

// hggc hggcGraphRetainUserObject hack
#define HGGC_GRAPH_HACK
#ifdef HGGC_GRAPH_HACK
template <typename T>
static ncclResult_t ncclHggcCallocOriginLegacy(T** ptr, size_t nelem) {
  if (nelem == 0) {
    return ncclSuccess;
  }

  VERBOSE(NCCL_ALLOC, "pcclHggcCallocOriginLegacy(ptr=%p, nelem=%zu)", ptr, nelem);
  HGGCCHECK(hggcMalloc(ptr, nelem*sizeof(T)));

  if (ncclParamNoMemsetFlag == 0) {
    HGGCCHECK(hggcMemset(*ptr, 0, nelem*ncclSizeOfT<T>()));
  }
  HGGCCHECK(hggcStreamSynchronize(0));
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclHggcCallocDebugLegacy(T** ptr, size_t nelem, uint64_t* pDevMemUsed, const char *filefunc, int line) {
  ncclResult_t result = ncclSuccess;
  hggcStreamCaptureMode mode = hggcStreamCaptureModeRelaxed;
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));

  NCCLCHECKGOTO(ncclHggcCallocOriginLegacy(ptr, nelem), result, finish);

finish:
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
  if (result == ncclSuccess && pDevMemUsed) *pDevMemUsed += nelem * sizeof(T);
  VERBOSE(NCCL_ALLOC, "%s:%d Hggc Alloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), *ptr);
  return result;
}

#define ncclHggcCallocLegacy(...) ncclHggcCallocDebugLegacy(__VA_ARGS__, __FILENAME__, __LINE__)

template <typename T>
ncclResult_t ncclHggcCallocAsyncDebugLegacy(T** ptr, size_t nelem, uint64_t* pDevMemUsed, hggcStream_t stream, const char *filefunc, int line) {
  ncclResult_t result = ncclSuccess;
  hggcStreamCaptureMode mode = hggcStreamCaptureModeRelaxed;
  *ptr = nullptr;
  VERBOSE(NCCL_ALLOC, "pcclHggcCallocAsyncLegacy(ptr=%p, nelem=%zu, stream=%p)", ptr, nelem, stream);
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
  HGGCCHECKGOTO(hggcMalloc(ptr, nelem*sizeof(T)), result, finish);
  HGGCCHECKGOTO(hggcMemsetAsync(*ptr, 0, nelem*sizeof(T), stream), result, finish);
finish:
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr) ERROR("Failed to HGGC calloc async %ld bytes", nelem*sizeof(T));
  if (result == ncclSuccess && pDevMemUsed) *pDevMemUsed += nelem * sizeof(T);
  INFO(NCCL_ALLOC, "%s:%d Hggc Alloc Size %ld pointer %p", filefunc, line, nelem*sizeof(T), *ptr);
  return result;
}
#define ncclHggcCallocAsyncLegacy(...) ncclHggcCallocAsyncDebugLegacy(__VA_ARGS__, __FILENAME__, __LINE__)

template <typename T>
static ncclResult_t ncclHggcFreeDebugLegacy(const char *filefunc, int line, T* ptr, int memMode=MEM_NORMAL) {
  ncclResult_t result = ncclSuccess;
  hggcStreamCaptureMode mode = hggcStreamCaptureModeRelaxed;
  VERBOSE(NCCL_ALLOC, "%s:%d Hggc Free pointer %p", filefunc, line, ptr);
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));

  HGGCCHECKGOTO(hggcFree(ptr), result, finish);
  ptr = NULL;

finish:
  if (result != ncclSuccess) WARN("%s: free %p failed", __func__, ptr);
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
  return result;
}

#define ncclHggcFreeLegacy(...) ncclHggcFreeDebugLegacy(__FILENAME__, __LINE__, __VA_ARGS__)

// Walks all physical segments backing a virtual address range, returning
// the base address, total mapped size, segment count, and whether any
// segment is host-resident.
static inline ncclResult_t ncclHgMemGetAddressRange(
    HGdeviceptr userBuff, size_t userBuffSize,
    HGdeviceptr* mappedPtrBase, size_t* totalMappedBufferSize,
    int* numSegments, bool* hasSysmemSegment = nullptr
  ) {
  *totalMappedBufferSize = 0;
  *mappedPtrBase = 0;
  if (numSegments) *numSegments = 0;
  HGdeviceptr userBuffStart = userBuff;
  HGdeviceptr userBuffEnd = userBuffStart + userBuffSize;
  HGdeviceptr mappedPtrEnd = userBuffStart;
  HGdeviceptr baseSend = 0;
  size_t baseSendSize = 0;

  if (hasSysmemSegment != nullptr) {
    *hasSysmemSegment = false;
  }

  while (mappedPtrEnd < userBuffEnd) {
    HGCHECK(hgMemGetAddressRange(&baseSend, &baseSendSize, mappedPtrEnd));

    if (hasSysmemSegment != nullptr && *hasSysmemSegment == false) {
      HGmemGenericAllocationHandle handle;
      HGmemAllocationProp prop;
      HGCHECK(hgMemRetainAllocationHandle(&handle, (void*)mappedPtrEnd));
      HGCHECK(hgMemGetAllocationPropertiesFromHandle(&prop, handle));
      if (prop.location.type == HG_MEM_LOCATION_TYPE_HOST_NUMA) {
        *hasSysmemSegment = true;
      }
      HGCHECK(hgMemRelease(handle));
    }

    if (*totalMappedBufferSize == 0) {
      if (baseSend == 0) {
        *mappedPtrBase = userBuffStart;
      } else {
        *mappedPtrBase = baseSend;
      }
    }
    *totalMappedBufferSize += baseSendSize;
    if (baseSend == 0) {
      mappedPtrEnd = mappedPtrEnd + baseSendSize;
    } else {
      mappedPtrEnd = baseSend + baseSendSize;
    }

    if (numSegments) *numSegments = *numSegments + 1;
  }
  return ncclSuccess;
}

#endif
#undef HGGC_GRAPH_HACK

#endif
