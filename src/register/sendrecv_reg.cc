/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "register.h"
#include "transport.h"

ncclResult_t ncclRegisterP2pIpcBuffer(ncclComm_t comm, void* userbuff, size_t size, int peerRank, int* regFlag, void** regAddr, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue) {
  ncclResult_t ret = ncclSuccess;
  uintptr_t offset = 0;
  uintptr_t* peerRmtAddrs = NULL;

  *regFlag = 0;
  if (comm->planner.persistent && ncclParamGraphRegister()) {
    ncclIpcGraphRegisterBuffer(comm, userbuff, size, &peerRank, 1, NCCL_IPC_SENDRECV, regFlag, &offset, &peerRmtAddrs, reinterpret_cast<void*>(cleanupQueue), NULL);
  }
  if (*regFlag == 0 && ncclParamLocalRegister()) {
    ncclIpcLocalRegisterBuffer(comm, userbuff, size, &peerRank, 1, NCCL_IPC_SENDRECV, regFlag, &offset, &peerRmtAddrs);
  }

  if (*regFlag)
    *regAddr = (void*)((uintptr_t)peerRmtAddrs + offset);
  return ret;
}
