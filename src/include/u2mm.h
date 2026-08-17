/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_U2MM_H_
#define NCCL_U2MM_H_

#include "pccl_internal.h"
#include "net.h"
#include "numawrap.h"
#include "u2mmwrap.h"
#include <pthread.h>

#define GPU_MAX_NUM 32

ncclResult_t nccl_u2mm_init(void);

extern pthread_mutex_t nccl_u2mm_lock;

static ncclResult_t u2mmNumaAlloc(void** ptr, size_t size, int netdev) {
  ncclNetProperties_t props;
  ncclNetGetProperties(netdev, &props);

  FILE *fp = NULL;
  char path[512];
  int numa_node = -1;
  struct bitmask *orig = NULL, *cur = NULL, *old = NULL;
  *ptr = NULL;

  snprintf(path, sizeof(path), "%s/numa_node", props.pciPath);
  fp = fopen(path, "r");
  if (fp == NULL) {
    WARN("[U2MM HOST] failed to get numa node of netdev %d", netdev);
    goto alloc;
  }

  if (fscanf(fp, "%d", &numa_node) != 1) numa_node = -1;
  // INFO(NCCL_INIT, "[U2MM HOST] netdev %d on numa node %d", netdev, numa_node);
  fclose(fp);
  if (numa_node < 0) goto alloc;

  NCCLCHECK(wrap_numa_get_membind(&orig));
  if (orig == NULL) {
    WARN("[U2MM HOST] failed to get membind");
    goto alloc;
  }

  NCCLCHECK(wrap_numa_allocate_nodemask(&cur));
  if (cur == NULL) {
    WARN("[U2MM HOST] failed to allocate nodemask");
    goto alloc;
  }

  NCCLCHECK(wrap_numa_bitmask_setbit(cur, numa_node, &old));
  NCCLCHECK(wrap_numa_set_membind(cur));
  NCCLCHECK(wrap_numa_free_nodemask(cur));

alloc:
  pthread_mutex_lock(&nccl_u2mm_lock);
  *ptr = u2mm_alloc(size, U2MM_DEV_FLAG_HOST | U2MM_PAGE_FLAG_1G | U2MM_MEM_SHARED_FLAG);
  pthread_mutex_unlock(&nccl_u2mm_lock);
  if (*ptr == NULL) {
    ERROR("[U2MM HOST] failed to alloc %zu from host.", size);
    return ncclSystemError;
  }

  memset(*ptr, 0, size);

  if (orig) {
    NCCLCHECK(wrap_numa_set_membind(orig));
    NCCLCHECK(wrap_numa_free_nodemask(orig));
  }

  return ncclSuccess;
}

template <typename T>
ncclResult_t nccl_u2mm_host_calloc_debug(T** ptr, size_t nelem, int netdev, const char *filefunc, int line) {
  ncclResult_t res = ncclSuccess;
  NCCLCHECKGOTO(u2mmNumaAlloc((void**)ptr, nelem * sizeof(T), netdev), res, fail);
  if (hggcHostRegister(*ptr, nelem * sizeof(T), hggcHostRegisterMapped) != hggcSuccess) {
    ERROR("[U2MM HOST] failed to register: %p size: %zu", *ptr, nelem * sizeof(T));
    res = ncclUnhandledHggcError;
    goto fail;
  }

  INFO(NCCL_ALLOC, "%s:%d [U2MM HOST] alloc size: %zu pointer: %p", filefunc, line, nelem * sizeof(T), *ptr);
  return ncclSuccess;

fail:
  if (*ptr) {
    pthread_mutex_lock(&nccl_u2mm_lock);
    int ret = u2mm_free(*ptr);
    pthread_mutex_unlock(&nccl_u2mm_lock);
    if (ret != 0) {
      ERROR("[U2MM HOST] failed to free: %p", *ptr);
      return ncclInternalError;
    }
  }

  return res;
}
#define nccl_u2mm_host_calloc(...) nccl_u2mm_host_calloc_debug(__VA_ARGS__, __FILENAME__, __LINE__)

template<typename T>
ncclResult_t nccl_u2mm_host_free(T *ptr) {
  if (ptr == NULL) {
    return ncclSuccess;
  }

  INFO(NCCL_ALLOC, "[U2MM HOST] free: %p", ptr);

  if (hggcHostUnregister((void *)ptr) != hggcSuccess) {
    ERROR("[U2MM HOST] failed to unregister: %p", ptr);
    return ncclInternalError;
  }

  pthread_mutex_lock(&nccl_u2mm_lock);
  int ret = u2mm_free((void *)ptr);
  pthread_mutex_unlock(&nccl_u2mm_lock);
  if (ret != 0) {
    ERROR("[U2MM HOST] failed to free: %p", ptr);
    return ncclInternalError;
  }
  return ncclSuccess;
}

template <typename T>
ncclResult_t nccl_u2mm_gpu_calloc_debug(T** ptr, size_t nelem, bool direct, const char *filefunc, int line) {
  int gpu;
  HGGCCHECK(hggcGetDevice(&gpu));
  ncclResult_t result = ncclSuccess;
  hggcStreamCaptureMode mode = hggcStreamCaptureModeRelaxed;
  *ptr = nullptr;
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
  // Need a side stream so as not to interfere with graph capture.
  hggcStream_t stream;
  HGGCCHECK(hggcStreamCreateWithFlags(&stream, hggcStreamNonBlocking));
  pthread_mutex_lock(&nccl_u2mm_lock);
  uint64_t flag = U2MM_DEV_FLAG_PPU | gpu;
  if (direct) flag |= U2MM_DIRECT_ALLOC_FLAG;
  *ptr = (T *)u2mm_alloc(nelem * sizeof(T), flag);
  pthread_mutex_unlock(&nccl_u2mm_lock);
  if (*ptr == NULL) {
    ERROR("[U2MM GPU%d]%s failed to alloc %zu from ppu %d", gpu, direct ? "[direct]" : "", nelem * sizeof(T), gpu);
    result = ncclInternalError;
    goto finish;
  }
  HGGCCHECKGOTO(hggcMemsetAsync(*ptr, 0, nelem*sizeof(T), stream), result, finish);
  HGGCCHECKGOTO(hggcStreamSynchronize(stream), result, finish);
  HGGCCHECKGOTO(hggcStreamDestroy(stream), result, finish);
  INFO(NCCL_ALLOC, "%s:%d [U2MM GPU%d]%s alloc size: %zu pointer: %p", filefunc, line, gpu, direct ? "[direct]" : "", nelem * sizeof(T), *ptr);
finish:
  HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
  return result;
}
#define nccl_u2mm_gpu_calloc(...) nccl_u2mm_gpu_calloc_debug(__VA_ARGS__, __FILENAME__, __LINE__)

template<typename T>
ncclResult_t nccl_u2mm_gpu_free(T *ptr) {
  if (ptr == NULL) {
    return ncclSuccess;
  }

  int ret = 0;
  INFO(NCCL_ALLOC, "[U2MM] free: %p", ptr);

  pthread_mutex_lock(&nccl_u2mm_lock);
  ret = u2mm_free((void *)ptr);
  pthread_mutex_unlock(&nccl_u2mm_lock);
  if (ret != 0) {
    ERROR("[U2MM] failed to free: %p", ptr);
    return ncclInternalError;
  }

  return ncclSuccess;
}

static ncclResult_t ncclU2mmShmDestroy(char* mkey, void* shmPtr) {
  // INFO(NCCL_INIT, "[U2MM SHM] destroy mkey %s (%p) addr %p", mkey, mkey, shmPtr);
  INFO(NCCL_INIT, "[U2MM SHM] destroy mkey %p addr %p", mkey, shmPtr);
  int ret = 0;
  ncclResult_t res = ncclSuccess;
  if (mkey) {
    ret = u2mm_shm_unshare(mkey);
    if (ret != 0) {
      // WARN("[U2MM SHM] failed to unshare shm with mkey %s (%p), ret %d error %d (%s)", mkey, mkey, ret, errno, strerror(errno));
      WARN("[U2MM SHM] failed to unshare shm with mkey %p, ret %d error %d (%s)", mkey, ret, errno, strerror(errno));
      res = ncclSystemError;
    }
  }

  if (shmPtr) {
    pthread_mutex_lock(&nccl_u2mm_lock);
    ret = u2mm_free(shmPtr);
    pthread_mutex_unlock(&nccl_u2mm_lock);

    if (ret != 0) {
      WARN("[U2MM SHM] failed to free shm with addr %p, ret %d error %d (%s)",
        shmPtr, ret, errno, strerror(errno));
      res = ncclSystemError;
    }
  }

  return res;
}

static ncclResult_t ncclU2mmShmClose(void* shmPtr, void* devShmPtr) {
  if (shmPtr == NULL) return ncclSuccess;
  if (devShmPtr) HGGCCHECK(hggcHostUnregister(shmPtr));
  int ret = u2mm_shm_unmap(shmPtr);
  if (ret != 0) {
    WARN("[U2MM SHM] failed to unmap shm with addr %p, ret %d error %d (%s)",
      shmPtr, ret, errno, strerror(errno));
    return ncclSystemError;
  }
  INFO(NCCL_INIT, "[U2MM SHM] close shm hptr %p dptr %p", shmPtr, devShmPtr);

  return ncclSuccess;
}

static ncclResult_t ncclU2mmShmCreate(char** mkeyRet, const int size, void** shmPtr, int netdev) {
  ncclResult_t res = ncclSuccess;
  void* ptr = NULL;
  char* mkey = NULL;
  int ret = 0;
  NCCLCHECKGOTO(u2mmNumaAlloc(&ptr, size, netdev), res, fail);

  pthread_mutex_lock(&nccl_u2mm_lock);
  ret = u2mm_shm_share(ptr, size, &mkey);
  pthread_mutex_unlock(&nccl_u2mm_lock);

  if (ret != 0) {
    WARN("[U2MM SHM] failed to share shm with addr %p len %d, ret %d error %d (%s)",
        ptr, size, ret, errno, strerror(errno));
    res = ncclSystemError;
    goto fail;
  }

  // INFO(NCCL_INIT, "[U2MM SHM] Allocated %ld bytes of shared memory, addr %p mkey %s (%p)", size, ptr, mkey, mkey);
  INFO(NCCL_ALLOC, "[U2MM SHM] Allocated %ld bytes of shared memory, addr %p mkey %p", size, ptr, mkey);

exit:
  *mkeyRet = mkey;
  *shmPtr = ptr;
  return res;

fail:
  ERROR("[U2MM SHM] Error while allocating shared memory segment (size %d)", size);
  ncclU2mmShmDestroy(mkey, ptr);
  mkey = NULL;
  ptr = NULL;
  goto exit;
}

static ncclResult_t ncclU2mmShmOpen(char* mkey, const int size, void** shmPtr, void** devShmPtr) {
  ncclResult_t res = ncclSuccess;
  void* hptr = NULL;
  void* dptr = NULL;

  pthread_mutex_lock(&nccl_u2mm_lock);
  int ret = u2mm_shm_map(mkey, &hptr, size);
  pthread_mutex_unlock(&nccl_u2mm_lock);

  if (ret != 0) {
    // WARN("[U2MM SHM] failed to map shm with mkey %s (%p) size %d, ret %d error %d (%s)", mkey, mkey, size, ret, errno, strerror(errno));
    WARN("[U2MM SHM] failed to map shm with mkey %p size %d, ret %d error %d (%s)", mkey, size, ret, errno, strerror(errno));
    res = ncclSystemError;
    goto fail;
  }

  if (devShmPtr) {
    HGGCCHECKGOTO(hggcHostRegister(hptr, size, hggcHostRegisterMapped), res, fail);
    HGGCCHECKGOTO(hggcHostGetDevicePointer(&dptr, hptr, 0), res, fail);
  }

  // INFO(NCCL_INIT, "[U2MM SHM] open shm with mkey %s (%p) size %d, hptr %p dptr %p", mkey, mkey, size, hptr, dptr);
  INFO(NCCL_INIT, "[U2MM SHM] open shm with mkey %p size %d, hptr %p dptr %p", mkey, size, hptr, dptr);

exit:
  *shmPtr = hptr;
  if (devShmPtr) *devShmPtr = dptr;
  return res;

fail:
  // ERROR("[U2MM SHM] Error while attaching to shared memory segment, mkey %s (%p) size %d hptr %p dptr %p", mkey, mkey, size, hptr, dptr);
  ERROR("[U2MM SHM] Error while attaching to shared memory segment, mkey %s size %d hptr %p dptr %p)", mkey, size, hptr, dptr);
  ncclU2mmShmClose(hptr, dptr);
  hptr = NULL;
  dptr = NULL;
  goto exit;
}

#endif
