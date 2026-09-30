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
