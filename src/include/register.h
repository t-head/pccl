/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2024-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_REGISTER_H_
#define NCCL_REGISTER_H_

#include "devcomm.h"
#include "pccl_device/core.h"
#include <hggc.h>
#include <stdint.h>

int64_t ncclParamLocalRegister();
int64_t ncclParamGraphRegister();

enum {
  IPC_REG_COMPLETE = 0x20
};

#define NCCL_REGULAR_BUFFER 0x00
#define NCCL_IPC_REG_BUFFER 0x01

struct ncclPeerRegIpcAddr {
  uintptr_t* devPeerRmtAddrs;
  uintptr_t* hostPeerRmtAddrs;
};

struct ncclReg {
  // common attributes
  uintptr_t begAddr, endAddr; // page aligned
  size_t pages;
  int localRefs;
  int graphRefs;
  uintptr_t addr;
  uint32_t state;

  // general ipc reg
  struct ncclPeerRegIpcAddr regIpcAddrs;
  struct ncclIpcRegInfo* ipcInfos[NCCL_MAX_LOCAL_RANKS];
  // symmetric reg
  void* baseSymPtr;
  size_t symSize;
  int winFlags;
};

struct ncclRegCache {
  struct ncclReg **slots;
  int capacity, population;
  uintptr_t pageSize;
};

ncclResult_t ncclRegCleanup(ncclComm_t comm);
ncclResult_t ncclCommGraphRegister(const ncclComm_t comm, void* buff, size_t size, void** handle);
ncclResult_t ncclCommGraphDeregister(const ncclComm_t comm, struct ncclReg *handle);
ncclResult_t ncclRegLocalIsValid(struct ncclReg *reg, bool *isValid);

#endif
