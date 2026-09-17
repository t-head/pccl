/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_MEM_MANAGER_H_
#define NCCL_MEM_MANAGER_H_

#include "pccl_internal.h"
#include <hggc.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <mutex>

#ifdef __cplusplus
extern "C" {
#endif

// The manager hangs off a communicator, but comm.h includes this header, so the
// communicator type is only forward declared here (NCCL: struct ncclComm).
struct pcclComm;

// Initial capacity for exported peers array.
#define NCCL_MEM_EXPORT_PEERS_INIT 8

// Memory type for PCCL internal allocations.
typedef enum {
  ncclMemPersist  = 0,  // Persistent memory: track stats only.
  ncclMemScratch  = 1,  // Scratch memory: releasable without data backup.
  ncclMemOffload  = 2   // Offload memory: data must be preserved for suspend/resume.
} ncclMemType_t;

// Dynamic memory entry state.
typedef enum {
  ncclDynMemStateActive   = 0,
  ncclDynMemStateReleased = 1
} ncclDynMemState_t;

// Local owned memory descriptor.
typedef struct ncclDynMemLocalDesc {
  // Shareable handle for P2P exports
  // TODO: Remove the 'fd' field - POSIX FD handles are converted on-demand via proxy
  // (ncclProxyClientGetFdBlocking), so we no longer export them upfront. Only FABRIC
  // handles need upfront export since they can be shared directly via messaging.
  union {
    int               fd;           // For POSIX_FILE_DESCRIPTOR (unused)
    HGmemFabricHandle fabricHandle; // For FABRIC
  } shareableHandle;
  bool shareableHandleValid;
  int numExportedPeers;
  int exportedPeersCapacity;
  int* exportedPeerRanks;
} ncclDynMemLocalDesc;

// Imported memory descriptor.
typedef struct ncclDynMemImportDesc {
  int ownerRank;
  int ownerDev;
  void* ownerPtr;
} ncclDynMemImportDesc;

// Individual tracked dynamic memory entry.
typedef struct ncclDynMemEntry {
  void*                       ptr;
  size_t                      size;
  size_t                      reservationSize;
  HGmemGenericAllocationHandle handle;
  HGmemAllocationHandleType    handleType;
  ncclMemType_t               memType;
  ncclDynMemState_t           state;
  int                         hggcDev;
  void*                       cpuBackup;
  bool                        isImportedFromPeer;
  union {
    ncclDynMemLocalDesc  local;
    ncclDynMemImportDesc imported;
  } desc;
  struct ncclDynMemEntry*     next;
} ncclDynMemEntry;

// P2P handle exchange structure for dynamic memory resume.
typedef struct ncclDynMemP2pHandleInfo {
  void*  ptr;
  int    ownerRank;
  int    ownerDev;
  size_t size;
  int    handleType;
  union {
    uint64_t          handleData;
    HGmemFabricHandle fabricHandle;
  };
} ncclDynMemP2pHandleInfo;

// Memory manager attached to ncclComm.
typedef struct ncclMemManager {
  ncclDynMemEntry* entries;
  int numEntries;
  std::mutex lock;
  int released;
  int initialized;
  int refCount;

  size_t totalPersist;
  size_t totalPersistImported;
  size_t totalScratch;
  size_t totalScratchImported;
  size_t totalOffload;
  size_t totalOffloadImported;
  size_t cpuBackupUsage;

  int commHggcDev;
} ncclMemManager;

struct ncclMemManagerTask {
  struct ncclMemManagerTask* next;
  ncclComm_t comm;
};

ncclResult_t ncclMemManagerInit(ncclComm_t comm);
ncclResult_t ncclMemManagerDestroy(ncclComm_t comm);

ncclResult_t ncclMemTrack(
  struct ncclMemManager* manager,
  void* ptr,
  size_t size,
  HGmemGenericAllocationHandle handle,
  HGmemAllocationHandleType handleType,
  ncclMemType_t memType,
  size_t reservationSize = 0
);

ncclResult_t ncclMemTrackImportFromPeer(
  struct ncclMemManager* manager,
  void* ptr,
  size_t size,
  HGmemGenericAllocationHandle handle,
  HGmemAllocationHandleType handleType,
  ncclMemType_t memType,
  int ownerRank,
  int ownerDev,
  void* ownerPtr,
  size_t reservationSize = 0
);

struct ncclMemUntrackInfo {
  ncclMemType_t memType;
  ncclDynMemState_t dynMemState;
  size_t dynMemSize;
  size_t reservationSize;
};

ncclResult_t ncclMemUntrack(struct ncclMemManager* manager, void* ptr, size_t size,
                          struct ncclMemUntrackInfo* info = nullptr);
ncclResult_t ncclDynMemMarkExportToPeer(struct ncclMemManager* manager, void* ptr, int peerRank);

ncclResult_t ncclCommMemSuspend(ncclComm_t comm);
ncclResult_t ncclCommMemResume(ncclComm_t comm);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif /* NCCL_MEM_MANAGER_H_ */
