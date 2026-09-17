/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
#include "argcheck.h" // Need some checks here since we access comm
#include "pccl_internal.h"
#include "comm.h"
#include "net.h"
#include "register.h"
#include "transport.h"
#include "group.h"
#include "dev_runtime.h"

NCCL_PARAM(LocalRegister, "LOCAL_REGISTER", 1);

ncclResult_t ncclRegLocalIsValid(struct ncclReg *reg, bool *isValid) {
  if (reg && isValid) {
    if (reg->localRefs) *isValid = true;
    else *isValid = false;
  }
  return ncclSuccess;
}

ncclResult_t ncclRegister(ncclComm_t comm, void* data, size_t size, bool isGraph, void** handle) {
  NCCLCHECK(CommCheck(comm, "pcclCommRegister", "comm"));
  struct ncclRegCache* cache = &comm->regCache;
  uintptr_t pageSize = cache->pageSize;
  uintptr_t begAddr = (uintptr_t)data & -pageSize;
  uintptr_t endAddr = ((uintptr_t)data + size + pageSize-1) & -pageSize;

  if (comm->checkPointers) NCCLCHECK(HggcPtrCheck(data, comm, "buff", "pcclCommRegister"));
  INFO(NCCL_REG, "register comm %p buffer %p size %zi", comm, data, size);

  for (int slot=0; /*true*/; slot++) {
    if ((slot == cache->population) || (begAddr < cache->slots[slot]->begAddr)) {
      if (cache->population == cache->capacity) { // must grow cache
        cache->capacity = cache->capacity < 32 ? 32 : 2*cache->capacity;
        NCCLCHECK(ncclRealloc(&cache->slots, cache->population, cache->capacity));
      }
      memmove(cache->slots+slot+1, cache->slots+slot, (cache->population-slot)*sizeof(struct ncclReg*));
      NCCLCHECK(ncclCalloc(cache->slots+slot, 1));
      struct ncclReg* regSlot = cache->slots[slot];
      regSlot->begAddr = begAddr;
      regSlot->endAddr = endAddr;
      if (isGraph) regSlot->graphRefs = 1;
      else regSlot->localRefs = 1;
      cache->population += 1;
      *handle = regSlot;
      goto exit;
    } else if ((cache->slots[slot]->begAddr <= begAddr) &&
               (cache->slots[slot]->endAddr >= endAddr)) {
      if (isGraph) cache->slots[slot]->graphRefs++;
      else cache->slots[slot]->localRefs++;
      *handle = cache->slots[slot];
      goto exit;
    }
  }

exit:
  return ncclSuccess;
}

static ncclResult_t regCleanup(ncclComm_t comm, struct ncclReg* reg) {
  if (reg->state & IPC_REG_COMPLETE) {
    for (int i = 0; i < NCCL_MAX_LOCAL_RANKS; ++i)
      if (reg->ipcInfos[i]) {
        if (ncclIpcDeregBuffer(comm, reg->ipcInfos[i]) != ncclSuccess) {
          WARN("rank %d deregister IPC buffer %p peerRank %d failed", comm->rank, reg->ipcInfos[i]->baseAddr, reg->ipcInfos[i]->peerRank);
        }
        NCCLCHECK(ncclFree(reg->ipcInfos[i]));
      }
    NCCLCHECK(ncclFree(reg->regIpcAddrs.hostPeerRmtAddrs));
    if (reg->regIpcAddrs.devPeerRmtAddrs) NCCLCHECK(ncclHggcFree(reg->regIpcAddrs.devPeerRmtAddrs, MEM_NORMAL, comm->memManager));
  }
  return ncclSuccess;
}

ncclResult_t ncclRegCleanup(ncclComm_t comm) {
  struct ncclRegCache* cache = &comm->regCache;
  for (int i = 0; i < cache->population; i++) {
    struct ncclReg* reg = cache->slots[i];
    INFO(NCCL_INIT, "Cleanup buffer %p pages %lx", (void*)reg->addr, reg->pages);
    NCCLCHECK(regCleanup(comm, reg));
    NCCLCHECK(ncclFree(reg));
  }
  NCCLCHECK(ncclFree(cache->slots));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, pcclCommRegister, const ncclComm_t comm, void* buff, size_t size, void** handle);
ncclResult_t pcclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle) {
  INFO_CALL("PCCL API: pcclCommRegister(comm=%p, buff=%p, size=%zu, handle=%p)", comm, buff, size, handle);
  if (!ncclParamLocalRegister())
    *handle = NULL;
  else
    NCCLCHECK(ncclRegister(comm, buff, size, false, handle));
  return ncclSuccess;
}

ncclResult_t ncclCommGraphRegister(const ncclComm_t comm, void* buff, size_t size, void** handle) {
  NCCLCHECK(ncclRegister(comm, buff, size, true, handle));
  return ncclSuccess;
}

static ncclResult_t commDeregister(ncclComm_t comm, bool isGraph, struct ncclReg* reg) {
  NCCLCHECK(CommCheck(comm, "pcclCommDeregister", "comm"));
  struct ncclRegCache* cache = &comm->regCache;
  int slot;
  int saveDev;
  if (reg == NULL) goto exit;
  HGGCCHECK(hggcGetDevice(&saveDev));
  HGGCCHECK(hggcSetDevice(comm->hggcDev));
  for (slot = 0; slot < cache->population && cache->slots[slot] != reg; slot++);
  if (slot == cache->population) {
    WARN("Deregister: Could not find handle");
    return ncclInvalidUsage;
  }
  if (isGraph) --reg->graphRefs;
  else --reg->localRefs;
  if (reg->localRefs || reg->graphRefs) return ncclSuccess;
  NCCLCHECK(regCleanup(comm, reg));
  NCCLCHECK(ncclFree(reg));
  memmove(cache->slots + slot, cache->slots + slot + 1, (cache->population - slot - 1) * sizeof(struct ncclReg*));
  cache->population -= 1;
  HGGCCHECK(hggcSetDevice(saveDev));
exit:
  return ncclSuccess;
}

NCCL_API(ncclResult_t, pcclCommDeregister, const ncclComm_t comm, void* handle);
ncclResult_t pcclCommDeregister(const ncclComm_t comm, void *handle) {
  INFO_CALL("PCCL API: pcclCommDeregister(comm=%p, handle=%p)", comm, handle);
  NCCLCHECK(commDeregister(comm, false, (struct ncclReg*)handle));
  return ncclSuccess;
}

ncclResult_t ncclCommGraphDeregister(const ncclComm_t comm, struct ncclReg *handle) {
  NCCLCHECK(commDeregister(comm, true, handle));
  return ncclSuccess;
}

