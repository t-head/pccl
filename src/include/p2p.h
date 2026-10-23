/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <stdlib.h>

#ifndef NCCL_P2P_H_
#define NCCL_P2P_H_

#include <hggc.h>
#include "core.h"
#include "mem_manager.h"

extern int useMemcpy;
static int usingMemcpy() {
  return useMemcpy;
}

enum ncclIpcRegType {
  NCCL_IPC_SENDRECV = 0,
  NCCL_IPC_COLLECTIVE = 1
};

struct ncclIpcImpInfo {
  void* rmtRegAddr;
  bool legacyIpcCap;
  uintptr_t offset;
};

struct ncclIpcRegInfo {
  int peerRank;
  void* baseAddr;
  struct ncclProxyConnector* ipcProxyconn;
  struct ncclIpcImpInfo impInfo;
};

typedef union {
  uint64_t data; // Needs to hold a CUmemGenericAllocationHandle for UDS fd support
  HGmemFabricHandle handle;
} ncclHgDesc;

typedef union {
  // Legacy HGGC IPC
  hggcIpcMemHandle_t devIpc;
  // hgMem API support
  struct {
    ncclHgDesc hgDesc;
    HGmemGenericAllocationHandle memHandle;
  };
} ncclIpcDesc;

ncclResult_t ncclP2pAllocateShareableBuffer(size_t size, int refcount, ncclIpcDesc *ipcDesc, void **ptr,
  uint64_t* pDevMemUsed, int memMode=MEM_NORMAL, int peerRank=-1,
  struct ncclMemManager* manager=nullptr, ncclMemType_t memType=ncclMemPersist);
// Allocate on this rank's local proxy; its connection owns the returned buffer.
// cpuPtrRef optionally returns a same-process slot for a mapping that the owner must unlock before free.
ncclResult_t pcclP2pAllocateProxyBuffer(ncclComm_t comm, size_t size, ncclIpcDesc* ipcDesc,
  void** ptr, int memMode=MEM_NORMAL, void*** cpuPtrRef=nullptr, ncclMemType_t memType=ncclMemPersist);
ncclResult_t ncclP2pFreeShareableBuffer(ncclIpcDesc *ipcDesc);
ncclResult_t ncclP2pImportShareableBuffer(ncclComm_t comm, int tpPeer, size_t size, ncclIpcDesc *ipcDesc,
  void **devMemPtr, int memMode=MEM_NORMAL,
  void* ownerPtr=nullptr, ncclMemType_t memType=ncclMemPersist);

ncclResult_t ncclIpcLocalRegisterBuffer(ncclComm_t comm, const void* userbuff, size_t buffSize, int* peerRanks, int nPeers, ncclIpcRegType type, int* regBufFlag, uintptr_t* offsetOut, uintptr_t** peerRmtAddrsOut);
ncclResult_t ncclIpcGraphRegisterBuffer(ncclComm_t comm, const void* userbuff, size_t buffSize, int* peerRanks, int nPeers, ncclIpcRegType type, int* regBufFlag, uintptr_t* offsetOut, uintptr_t** peerRmtAddrsOut, void* cleanupQueuePtr, int* nCleanupQueueElts);

ncclResult_t ncclIpcDeregBuffer(ncclComm_t comm, struct ncclIpcRegInfo* regInfo);
#endif
