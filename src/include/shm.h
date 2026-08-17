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
#endif
