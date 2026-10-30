/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_SHM_H_
#define NCCL_SHM_H_

#include "pccl_internal.h"

ncclResult_t ncclShmOpen(char* shmPath, const int shmSize, void** shmPtr, void** devShmPtr, int create);
ncclResult_t ncclShmUnlink(const char* shmname);
ncclResult_t ncclShmClose(void* shmPtr, void* devShmPtr, const int shmSize);

typedef void* ncclShmHandle_t;
ncclResult_t ncclShmOpen(char* shmPath, size_t shmPathSize, size_t shmSize, void** shmPtr, void** devShmPtr, int refcount, ncclShmHandle_t* handle);
ncclResult_t ncclShmClose(ncclShmHandle_t handle);

struct shmLegacyIpc {
  char shmSuffix[7];
  ncclShmHandle_t handle;
  size_t shmSize;
};

struct shmIpcDesc {
  struct shmLegacyIpc shmli;
  bool legacy;
};

typedef struct shmIpcDesc ncclShmIpcDesc_t;

ncclResult_t ncclShmAllocateShareableBuffer(size_t size, bool legacy, ncclShmIpcDesc_t *descOut, void **hptr, void **dptr);
ncclResult_t ncclShmImportShareableBuffer(ncclComm_t comm, int proxyRank, ncclShmIpcDesc_t *desc, void **hptr, void **dptr, ncclShmIpcDesc_t *descOut);
ncclResult_t ncclShmIpcClose(ncclShmIpcDesc_t *desc);
#endif
