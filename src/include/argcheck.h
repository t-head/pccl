/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_ARGCHECK_H_
#define NCCL_ARGCHECK_H_

#include "core.h"
#include "info.h"

ncclResult_t PtrCheck(const void* ptr, const char* opname, const char* ptrname);
ncclResult_t CommCheck(ncclComm_t comm, const char* opname, const char* ptrname);
ncclResult_t ArgsCheck(struct ncclInfo* info);
ncclResult_t HggcPtrCheck(const void* pointer, ncclComm_t comm, const char* ptrname, const char* opname);

#endif
