/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"

#define REG_ALIGN (4096)

// size_t so GIN symmetric windows above 2GiB can be registered. NCCL is size_t throughout.
ncclResult_t ncclIbRegMrInternal(void* comm, void* data, size_t size, int type, uint64_t mrFlags,
                                 void** mhandle) {
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*)comm;
  struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, 0);
  uint64_t addr = (uint64_t)data;
  if (size == 0) {
    ERROR("Invalid size %zu for memory registration.", size);
    return ncclInvalidArgument;
  }

  // Deregister / register
  uint64_t regAddr = addr & (~(REG_ALIGN-1));
  uint64_t regSize = addr + size - regAddr;
  regSize = ((regSize + REG_ALIGN - 1) / REG_ALIGN ) * REG_ALIGN;
  struct ibv_mr* mr;
  unsigned int flags = IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ|IBV_ACCESS_REMOTE_ATOMIC;
  bool relaxedOrdering = ncclIbRelaxedOrderingEnabled && (mrFlags & NCCL_NET_MR_FLAG_FORCE_SO) == 0;
  if (relaxedOrdering) {
    flags |= IBV_ACCESS_RELAXED_ORDERING;
    NCCLCHECK(wrap_ibv_reg_mr_iova2(&mr, devComm->pd, (void*)regAddr, regSize, regAddr, flags));
  } else {
    NCCLCHECK(wrap_ibv_reg_mr(&mr, devComm->pd, (void*)regAddr, regSize, flags));
  }
  *mhandle = (void*)mr;
  TRACE(NCCL_INIT,"regAddr %lx size %ld rkey %x", regAddr, regSize, mr->rkey);
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
  NCCLCHECK(wrap_ibv_dereg_mr((struct ibv_mr*)mhandle));
  return ncclSuccess;
}
