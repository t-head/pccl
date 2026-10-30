/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2024-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "shm.h"
#include "checks.h"
#include "utils.h"

struct shmHandleInternal {
  int fd;
  char* shmPath;
  char* shmPtr;
  void* devShmPtr;
  size_t shmSize;
  size_t realShmSize;
  int* refcount;
};

static void shmHandleInit(int fd, char* shmPath, size_t shmSize, size_t realShmSize, char* hptr, void* dptr, bool create, struct shmHandleInternal* handle) {
  handle->fd = fd;
  handle->shmPtr = hptr;
  handle->devShmPtr = dptr;
  handle->shmSize = shmSize;
  handle->realShmSize = realShmSize;
  handle->refcount = (hptr != NULL) ? (int*)(hptr + shmSize) : NULL;
  if (create) {
    int slen = strlen(shmPath);
    handle->shmPath = (char*)malloc(slen + 1);
    memcpy(handle->shmPath, shmPath, slen + 1);
    if (hptr) memset(hptr, 0, shmSize);
  } else {
    handle->shmPath = NULL;
  }
  return;
}

ncclResult_t ncclShmOpen(char* shmPath, size_t shmPathSize, size_t shmSize, void** shmPtr, void** devShmPtr, int refcount, ncclShmHandle_t* handle) {
  int fd = -1;
  char* hptr = NULL;
  void* dptr = NULL;
  ncclResult_t ret = ncclSuccess;
  struct shmHandleInternal* tmphandle;
  bool create = refcount > 0 ? true : false;
  const size_t refSize = sizeof(int); /* extra sizeof(int) bytes for reference count */
  const size_t realShmSize = shmSize + refSize;

  *handle = *shmPtr = NULL; /* assume shmPtr and handle always set correctly by users. */
  EQCHECKGOTO(tmphandle = (struct shmHandleInternal*)calloc(1, sizeof(struct shmHandleInternal)), NULL, ret, fail);
  if (create) {
    /* Positive refcount creates the segment; peers decrement on attachment and unlink at zero. */
    if (shmPath[0] == '\0') {
      snprintf(shmPath, shmPathSize, "/dev/shm/pccl-XXXXXX");
    retry_mkstemp:
      fd = mkstemp(shmPath);
      if (fd < 0) {
        if (errno == EINTR) {
          INFO(NCCL_ALL, "mkstemp: Failed to create %s, error: %s (%d) - retrying", shmPath, strerror(errno), errno);
          goto retry_mkstemp;
        }
        WARN("Error: failed to create shared memory file %s, error %s (%d)", shmPath, strerror(errno), errno);
        ret = ncclSystemError;
        goto fail;
      }
    } else {
      SYSCHECKGOTO(fd = open(shmPath, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR), "open", ret, fail);
    }

  retry_fallocate:
    if (fallocate(fd, 0, 0, realShmSize) != 0) {
      if (errno == EINTR) {
        INFO(NCCL_ALL, "fallocate: Failed to extend %s to %ld bytes, error: %s (%d) - retrying", shmPath, realShmSize, strerror(errno), errno);
        goto retry_fallocate;
      }
      WARN("Error: failed to extend %s to %ld bytes, error: %s (%d)", shmPath, realShmSize, strerror(errno), errno);
      ret = ncclSystemError;
      goto fail;
    }
    INFO(NCCL_ALLOC, "Allocated %ld bytes of shared memory in %s", realShmSize, shmPath);
  } else {
    SYSCHECKGOTO(fd = open(shmPath, O_RDWR, S_IRUSR | S_IWUSR), "open", ret, fail);
  }

  hptr = (char*)mmap(NULL, realShmSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (hptr == MAP_FAILED) {
    WARN("Error: Could not map %s size %zu, error: %s (%d)", shmPath, realShmSize, strerror(errno), errno);
    ret = ncclSystemError;
    hptr = NULL;
    goto fail;
  }

  if (create) {
    *(int*)(hptr + shmSize) = refcount;
  } else {
    int remref = ncclAtomicRefCountDecrement((int*)(hptr + shmSize));
    if (remref == 0) {
      /* the last peer has completed attachment, it should unlink the shm mem file. */
      if (unlink(shmPath) != 0) {
        INFO(NCCL_ALLOC, "unlink shared memory %s failed, error: %s (%d)", shmPath, strerror(errno), errno);
      }
    }
  }

  if (devShmPtr) {
    HGGCCHECKGOTO(hggcHostRegister((void*)hptr, realShmSize, hggcHostRegisterPortable | hggcHostRegisterMapped), ret, fail);
    HGGCCHECKGOTO(hggcHostGetDevicePointer(&dptr, (void*)hptr, 0), ret, fail);
  }

  shmHandleInit(fd, shmPath, shmSize, realShmSize, hptr, dptr, create, tmphandle);
exit:
  *shmPtr = hptr;
  if (devShmPtr) *devShmPtr = dptr;
  *handle = (ncclShmHandle_t)tmphandle;
  return ret;
fail:
  WARN("Error while %s shared memory segment %s (size %ld), error: %s (%d)", create ? "creating" : "attaching to",
       shmPath, shmSize, strerror(errno), errno);
  if (tmphandle) {
    shmHandleInit(fd, shmPath, shmSize, realShmSize, hptr, dptr, create, tmphandle);
    (void)ncclShmClose((ncclShmHandle_t)tmphandle);
    tmphandle = NULL;
  }
  hptr = NULL;
  dptr = NULL;
  goto exit;
}

ncclResult_t ncclShmClose(ncclShmHandle_t handle) {
  ncclResult_t ret = ncclSuccess;
  struct shmHandleInternal* tmphandle = (struct shmHandleInternal*)handle;
  if (tmphandle) {
    if (tmphandle->fd >= 0) {
      close(tmphandle->fd);
      if (tmphandle->shmPath != NULL && tmphandle->refcount != NULL && *tmphandle->refcount > 0) {
        if (unlink(tmphandle->shmPath) != 0) {
          WARN("unlink shared memory %s failed, error: %s (%d)", tmphandle->shmPath, strerror(errno), errno);
          ret = ncclSystemError;
        }
      }
      free(tmphandle->shmPath);
    }

    if (tmphandle->shmPtr) {
      if (tmphandle->devShmPtr) HGGCCHECK(hggcHostUnregister(tmphandle->shmPtr));
      if (munmap(tmphandle->shmPtr, tmphandle->realShmSize) != 0) {
        WARN("munmap of shared memory %p size %ld failed, error: %s (%d)", tmphandle->shmPtr, tmphandle->realShmSize, strerror(errno), errno);
        ret = ncclSystemError;
      }
    }
    free(tmphandle);
  }
  return ret;
}

static ncclResult_t ncclShmSetup(char* shmPath, const int shmSize, int* fd, void** ptr, int create) {
  if (create) {
    if (shmPath[0] == '\0') {
      sprintf(shmPath, "/dev/shm/pccl-XXXXXX");
      *fd = mkstemp(shmPath);
    } else {
      SYSCHECKVAL(open(shmPath, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR), "open", *fd);
    }
    if (ftruncate(*fd, shmSize) != 0) {
      ERROR("Error: failed to extend %s to %d bytes", shmPath, shmSize);
      return ncclSystemError;
    }
    // Reserve the backing pages now: ftruncate only sets the logical size, and
    // running out of tmpfs space would otherwise surface as a SIGBUS on first
    // touch instead of a clean error here. posix_fallocate returns the error
    // code directly and leaves errno untouched.
    int err = posix_fallocate(*fd, 0, shmSize);
    if (err != 0) {
      ERROR("Error: failed to allocate %d bytes for %s : %s%s", shmSize, shmPath, strerror(err),
            err == ENOSPC ? " (out of space, check the size of /dev/shm)" : "");
      return ncclSystemError;
    }
  } else {
    SYSCHECKVAL(open(shmPath, O_RDWR, S_IRUSR | S_IWUSR), "open", *fd);
  }
  *ptr = (char*)mmap(NULL, shmSize, PROT_READ|PROT_WRITE, MAP_SHARED, *fd, 0);
  if (*ptr == MAP_FAILED) {
    ERROR("Could not map %s, size %d : %s", shmPath, shmSize, strerror(errno));
    return ncclSystemError;
  }
  close(*fd);
  *fd = -1;
  if (create) memset(*ptr, 0, shmSize);
  return ncclSuccess;
}

ncclResult_t ncclShmOpen(char* shmPath, const int shmSize, void** shmPtr, void** devShmPtr, int create) {
  int fd = -1;
  void* ptr = MAP_FAILED;
  ncclResult_t res = ncclSuccess;

  NCCLCHECKGOTO(ncclShmSetup(shmPath, shmSize, &fd, &ptr, create), res, sysError);
  if (devShmPtr) {
    hggcStreamCaptureMode mode = hggcStreamCaptureModeRelaxed;
    HGGCCHECKGOTO(hggcThreadExchangeStreamCaptureMode(&mode), res, cudaError);
    HGGCCHECKGOTO(hggcHostRegister(ptr, shmSize, hggcHostRegisterMapped), res, cudaError);
    HGGCCHECKGOTO(hggcHostGetDevicePointer(devShmPtr, ptr, 0), res, cudaError);
    HGGCCHECKGOTO(hggcThreadExchangeStreamCaptureMode(&mode), res, cudaError);
  }

  *shmPtr = ptr;
  return ncclSuccess;
sysError:
  ERROR("Error while %s shared memory segment %s (size %d)", create ? "creating" : "attaching to", shmPath, shmSize);
cudaError:
  if (fd != -1) close(fd);
  // The file was created via open/mkstemp with a full path, so it must be
  // removed with unlink(); shm_unlink() expects a "/name" and would fail here.
  if (create) unlink(shmPath);
  if (ptr != MAP_FAILED) munmap(ptr, shmSize);
  *shmPtr = NULL;
  return res;
}

ncclResult_t ncclShmUnlink(const char* shmPath) {
  // Tolerate ENOENT so unlink stays idempotent: the ops pool may already have
  // been unlinked at init time (non-cuMem path) before the destroy-time unlink.
  if (shmPath != NULL && unlink(shmPath) != 0 && errno != ENOENT) {
    ERROR("Error: failed to unlink %s : %s", shmPath, strerror(errno));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t ncclShmClose(void* shmPtr, void* devShmPtr, const int shmSize) {
  if (devShmPtr) HGGCCHECK(hggcHostUnregister(shmPtr));
  if (shmPtr && munmap(shmPtr, shmSize) != 0) {
    ERROR("munmap of shared memory failed");
    return ncclSystemError;
  }
  return ncclSuccess;
}
