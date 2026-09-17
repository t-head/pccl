/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "core.h"
#include "comm.h"
#include "alloc.h"
#include "checks.h"
#include "argcheck.h"
#include "cudawrap.h"
#include "debug.h"
#include "bootstrap.h"
#include "proxy.h"
#include "transport.h"
#include "param.h"
#include "mem_manager.h"
#include "group.h"
#include <hggc.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <mutex>
#include <new>

// Internal parameter to disable memory manager for testing
NCCL_PARAM(MemManagerDisable, "DISABLE_MEM_MANAGER", 0);

// Initialize memory manager
ncclResult_t ncclMemManagerInit(ncclComm_t comm) {
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (comm == nullptr) return ncclInvalidArgument;

  ncclMemManager* mgr;
  NCCLCHECK(ncclCalloc(&mgr, 1));
  // Explicitly construct std::mutex using placement new
  new (&mgr->lock) std::mutex();

  mgr->entries = nullptr;
  mgr->numEntries = 0;
  mgr->released = 0;
  mgr->refCount = 1;
  mgr->totalPersist = 0;
  mgr->totalPersistImported = 0;
  mgr->totalScratch = 0;
  mgr->totalScratchImported = 0;
  mgr->totalOffload = 0;
  mgr->totalOffloadImported = 0;
  mgr->cpuBackupUsage = 0;
  mgr->commHggcDev = comm->hggcDev;

  __atomic_store_n(&mgr->initialized, 1, __ATOMIC_RELEASE);
  comm->memManager = mgr;

  INFO(NCCL_ALLOC, "MemManager: Initialized for device %d", comm->hggcDev);
  return ncclSuccess;
}

// Destroy memory manager and free all resources
ncclResult_t ncclMemManagerDestroy(ncclComm_t comm) {
  if (comm == nullptr) return ncclInvalidArgument;
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (comm->memManager == nullptr) return ncclSuccess;

  ncclMemManager* mgr = comm->memManager;
  if (!__atomic_load_n(&mgr->initialized, __ATOMIC_ACQUIRE)) {
    comm->memManager = nullptr;
    return ncclSuccess;
  }

  // Decrement reference count
  int refCount = ncclAtomicRefCountDecrement(&mgr->refCount);

  if (refCount > 0) {
    INFO(NCCL_ALLOC, "MemManager: Decremented refCount to %d", refCount);
    comm->memManager = nullptr;
    return ncclSuccess;
  }

  INFO(NCCL_ALLOC, "MemManager: Destroying (refCount=0)");
  __atomic_store_n(&mgr->initialized, 0, __ATOMIC_RELEASE);

  ncclDynMemEntry* entry = mgr->entries;
  while (entry != nullptr) {
    ncclDynMemEntry* next = entry->next;

    if (entry->cpuBackup != nullptr) {
      NCCLCHECK(ncclHggcHostFree(entry->cpuBackup));
    }

    // Close shareable FD if valid (defensive cleanup for POSIX FD handle type)
    if (!entry->isImportedFromPeer &&
        entry->desc.local.shareableHandleValid &&
        entry->handleType == HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
        entry->desc.local.shareableHandle.fd >= 0) {
      close(entry->desc.local.shareableHandle.fd);
      entry->desc.local.shareableHandle.fd = -1;
      entry->desc.local.shareableHandleValid = false;
    }

    // Only local entries have exportedPeerRanks (imported entries use desc.imported union member)
    if (!entry->isImportedFromPeer && entry->desc.local.exportedPeerRanks != nullptr) {
      ncclFree(entry->desc.local.exportedPeerRanks);
    }

    // Free the entry itself
    ncclFree(entry);
    entry = next;
  }

  mgr->entries = nullptr;
  mgr->numEntries = 0;

  // Explicitly call destructor for std::mutex
  mgr->lock.~mutex();
  // Free the manager struct
  ncclFree(mgr);
  comm->memManager = nullptr;

  INFO(NCCL_ALLOC, "MemManager: Destroyed");
  return ncclSuccess;
}

// Internal helper to create and track memory entry
static ncclResult_t ncclMemTrackInternal(
  struct ncclMemManager* manager,
  void* ptr,
  size_t size,
  HGmemGenericAllocationHandle handle,
  HGmemAllocationHandleType handleType,
  ncclMemType_t memType,
  bool isImportedFromPeer,
  int ownerRank,
  int ownerDev,
  void* ownerPtr,
  size_t reservationSize
) {
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (manager == nullptr || ptr == nullptr) return ncclInternalError;
  if (!__atomic_load_n(&manager->initialized, __ATOMIC_ACQUIRE)) {
    ERROR("MemManager: Cannot track allocation ptr=%p, manager not initialized", ptr);
    return ncclInternalError;
  }

  ncclDynMemEntry* entry = nullptr;
  NCCLCHECK(ncclCalloc(&entry, 1));

  // Initialize common fields
  entry->ptr = ptr;
  entry->size = size;
  entry->reservationSize = reservationSize;
  entry->handle = handle;
  entry->handleType = handleType;
  entry->memType = memType;
  entry->state = ncclDynMemStateActive;
  entry->hggcDev = manager->commHggcDev;
  entry->cpuBackup = nullptr;
  entry->isImportedFromPeer = isImportedFromPeer;

  if (isImportedFromPeer) {
    entry->desc.imported.ownerRank = ownerRank;
    entry->desc.imported.ownerDev = ownerDev;
    entry->desc.imported.ownerPtr = ownerPtr;
  } else {
    if (handleType == HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      entry->desc.local.shareableHandle.fd = -1;
    }
    entry->desc.local.shareableHandleValid = false;
    entry->desc.local.numExportedPeers = 0;
    entry->desc.local.exportedPeersCapacity = 0;
    entry->desc.local.exportedPeerRanks = nullptr;
  }

  { // lock the mutex to add the entry to the linked list
    std::lock_guard<std::mutex> lock(manager->lock);
    // Add to linked list (prepend)
    entry->next = manager->entries;
    manager->entries = entry;
    manager->numEntries++;
  } // lock_guard automatically releases mutex

  // Update statistics
  if (isImportedFromPeer) {
    if (memType == ncclMemPersist) {
      __atomic_fetch_add(&manager->totalPersistImported, size, __ATOMIC_RELAXED);
    } else if (memType == ncclMemScratch) {
      __atomic_fetch_add(&manager->totalScratchImported, size, __ATOMIC_RELAXED);
    } else if (memType == ncclMemOffload) {
      __atomic_fetch_add(&manager->totalOffloadImported, size, __ATOMIC_RELAXED);
    }
    TRACE(NCCL_ALLOC, "MemManager: Track imported ptr=%p size=%zu type=%d from rank=%d entries=%d",
          ptr, size, memType, ownerRank, manager->numEntries);
  } else {
    if (memType == ncclMemPersist) {
      __atomic_fetch_add(&manager->totalPersist, size, __ATOMIC_RELAXED);
    } else if (memType == ncclMemScratch) {
      __atomic_fetch_add(&manager->totalScratch, size, __ATOMIC_RELAXED);
    } else if (memType == ncclMemOffload) {
      __atomic_fetch_add(&manager->totalOffload, size, __ATOMIC_RELAXED);
    }
    TRACE(NCCL_ALLOC, "MemManager: Track ptr=%p size=%zu type=%d dev=%d entries=%d",
          ptr, size, memType, manager->commHggcDev, manager->numEntries);
  }
  return ncclSuccess;
}

// Track a new allocation
ncclResult_t ncclMemTrack(
  struct ncclMemManager* manager,
  void* ptr,
  size_t size,
  HGmemGenericAllocationHandle handle,
  HGmemAllocationHandleType handleType,
  ncclMemType_t memType,
  size_t reservationSize
) {
  return ncclMemTrackInternal(manager, ptr, size, handle, handleType, memType,
                              false, -1, -1, nullptr, reservationSize);
}

// Track imported allocation from peer
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
  size_t reservationSize
) {
  return ncclMemTrackInternal(manager, ptr, size, handle, handleType, memType,
                              true, ownerRank, ownerDev, ownerPtr, reservationSize);
}

// Untrack allocation
ncclResult_t ncclMemUntrack(struct ncclMemManager* manager, void* ptr, size_t size,
                          struct ncclMemUntrackInfo* info) {
  if (info != nullptr) {
    info->memType = ncclMemPersist;
    info->dynMemState = ncclDynMemStateActive;
    info->dynMemSize = 0;
    info->reservationSize = 0;
  }
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (manager == nullptr || ptr == nullptr) return ncclInternalError;

  // Atomic check to avoid locking destroyed mutex
  if (!__atomic_load_n(&manager->initialized, __ATOMIC_ACQUIRE)) {
    ERROR("MemManager: Cannot untrack allocation ptr=%p, manager not initialized", ptr);
    return ncclInternalError;
  }

  // Variables to save values before releasing lock
  size_t entrySize = 0;
  int numEntries __attribute__((unused)) = 0; // May be unused if TRACE compiled out
  bool isImportedFromPeer = false;
  ncclMemType_t memType = ncclMemScratch;

  {
    std::lock_guard<std::mutex> lock(manager->lock);
    ncclDynMemEntry* prev = nullptr;
    ncclDynMemEntry* entry = manager->entries;
    while (entry != nullptr) {
      if (entry->ptr == ptr) {
        // Remove from linked list
        if (prev == nullptr) {
          manager->entries = entry->next;
        }
        else {
          prev->next = entry->next;
        }
        manager->numEntries--;

        // Free CPU backup if exists
        if (entry->cpuBackup != nullptr) {
          manager->cpuBackupUsage -= entry->size;
          NCCLCHECK(ncclHggcHostFree(entry->cpuBackup));
        }

        // Close shareable FD if valid (defensive cleanup for POSIX FD handle type)
        if (!entry->isImportedFromPeer &&
            entry->desc.local.shareableHandleValid &&
            entry->handleType == HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
            entry->desc.local.shareableHandle.fd >= 0) {
          close(entry->desc.local.shareableHandle.fd);
          entry->desc.local.shareableHandle.fd = -1;
          entry->desc.local.shareableHandleValid = false;
        }

        // Only local entries have exportedPeerRanks (imported entries use desc.imported union member)
        if (!entry->isImportedFromPeer && entry->desc.local.exportedPeerRanks != nullptr) {
          ncclFree(entry->desc.local.exportedPeerRanks);
        }

        // Save values before unlock for logging (may be unused if TRACE is compiled out)
        entrySize = entry->size;
        numEntries = manager->numEntries;
        isImportedFromPeer = entry->isImportedFromPeer;
        memType = entry->memType;
        if (info != nullptr) {
          info->memType = entry->memType;
          info->dynMemState = entry->state;
          info->dynMemSize = entry->size;
          info->reservationSize = entry->reservationSize;
        }

        // Safety check: log if tracked size doesn't match passed size
        if (size != 0 && entrySize != size) {
          INFO(NCCL_ALLOC, "MemManager: Untrack size mismatch ptr=%p tracked=%zu passed=%zu",
               ptr, entrySize, size);
        }
        ncclFree(entry);
        break;
      }
      prev = entry;
      entry = entry->next;
    }
  } // lock_guard automatically releases mutex

  // Update statistics
  if (entrySize > 0) {
    if (isImportedFromPeer) {
      if (memType == ncclMemPersist) {
        __atomic_fetch_sub(&manager->totalPersistImported, entrySize, __ATOMIC_RELAXED);
      } else if (memType == ncclMemScratch) {
        __atomic_fetch_sub(&manager->totalScratchImported, entrySize, __ATOMIC_RELAXED);
      } else if (memType == ncclMemOffload) {
        __atomic_fetch_sub(&manager->totalOffloadImported, entrySize, __ATOMIC_RELAXED);
      }
    } else {
      if (memType == ncclMemPersist) {
        __atomic_fetch_sub(&manager->totalPersist, entrySize, __ATOMIC_RELAXED);
      } else if (memType == ncclMemScratch) {
        __atomic_fetch_sub(&manager->totalScratch, entrySize, __ATOMIC_RELAXED);
      } else if (memType == ncclMemOffload) {
        __atomic_fetch_sub(&manager->totalOffload, entrySize, __ATOMIC_RELAXED);
      }
    }
    TRACE(NCCL_ALLOC, "MemManager: Untrack ptr=%p size=%zu entries=%d", ptr, entrySize, numEntries);
  }
  return ncclSuccess;
}

// Track that a buffer is being shared with a peer (for suspend/resume coordination)
// NOTE: Only for dynamic memory (scratch/offload) that's in the linked list.
// Persistent memory doesn't need export tracking since it's never suspended.
// Call this after allocating dynamic memory and the peer imports it.
ncclResult_t ncclDynMemMarkExportToPeer(struct ncclMemManager* manager, void* ptr, int peerRank) {
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (manager == nullptr || ptr == nullptr) return ncclInternalError;
  if (!__atomic_load_n(&manager->initialized, __ATOMIC_ACQUIRE)) {
    ERROR("MemManager: Cannot mark export for ptr=%p, manager not initialized", ptr);
    return ncclInternalError;
  }
  std::lock_guard<std::mutex> lock(manager->lock);

  ncclDynMemEntry* entry = manager->entries;
  while (entry != nullptr && entry->ptr != ptr) {
    entry = entry->next;
  }

  if (entry == nullptr) {
    ERROR("MemManager: Cannot mark export for ptr=%p - not found in tracked entries", ptr);
    return ncclInternalError;
  }

  // Verify this is a local entry, not an imported one
  if (entry->isImportedFromPeer) {
    ERROR("MemManager: Cannot mark export for ptr=%p - this is an imported buffer", ptr);
    return ncclInternalError;
  }

  // Check if peer already exists
  for (int i = 0; i < entry->desc.local.numExportedPeers; i++) {
    if (entry->desc.local.exportedPeerRanks[i] == peerRank) {
      ERROR("MemManager: Buffer ptr=%p already exported to peer rank %d", ptr, peerRank);
      return ncclInternalError;
    }
  }

  if (entry->desc.local.numExportedPeers >= entry->desc.local.exportedPeersCapacity) {
    int newCapacity = entry->desc.local.exportedPeersCapacity == 0 ? NCCL_MEM_EXPORT_PEERS_INIT :
                      entry->desc.local.exportedPeersCapacity * 2;
    ncclResult_t ret = ncclRealloc(&entry->desc.local.exportedPeerRanks,
                                   entry->desc.local.exportedPeersCapacity, newCapacity);
    if (ret != ncclSuccess) {
      WARN("MemManager: Failed to grow exportedPeerRanks array for ptr=%p", ptr);
      return ret;
    }
    entry->desc.local.exportedPeersCapacity = newCapacity;
  }
  entry->desc.local.exportedPeerRanks[entry->desc.local.numExportedPeers++] = peerRank;
  TRACE(NCCL_ALLOC, "MemManager: ExportToPeer ptr=%p peerRank=%d numExportedPeers=%d",
        ptr, peerRank, entry->desc.local.numExportedPeers);
  return ncclSuccess;
}

/*
 * Internal: Suspend all dynamic memory with P2P coordination
 *
 * Order of operations:
 * 1. First pass: Unmap all peer-imported buffers
 * 2. Second pass: Offload local buffers to CPU and suspend physical memory
 */
ncclResult_t ncclCommMemSuspend(ncclComm_t comm) {
  if (comm == nullptr) return ncclInvalidArgument;
  if (comm->memManager == nullptr) return ncclInvalidUsage;
  ncclMemManager* manager = comm->memManager;

  if (manager->released) {
    WARN("MemManager: Already suspended");
    return ncclInvalidUsage;
  }

  ncclResult_t ret = ncclSuccess;
  size_t releasedScratch = 0, releasedOffload = 0, releasedPeerImport = 0;
  int releasedCount = 0, peerImportCount = 0;
  ncclDynMemEntry* entry = nullptr;

  HGGCCHECK(hggcDeviceSynchronize());
  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0xBEEF), ret, fail);

  // Step 1: Unmap all peer-imported buffers first
  entry = manager->entries;
  while (entry != nullptr) {
    if (entry->memType != ncclMemPersist && entry->isImportedFromPeer && entry->state == ncclDynMemStateActive) {
      TRACE(NCCL_ALLOC, "MemManager: Unmapping peer-imported buffer ptr=%p from rank %d",
            entry->ptr, entry->desc.imported.ownerRank);
      // Unmap our local mapping of the peer's memory
      HGCHECKIGNORE(hgMemUnmap((HGdeviceptr)entry->ptr, entry->size));

      // Release our reference to the peer's handle
      if (entry->handle != 0) {
        HGCHECKIGNORE(hgMemRelease(entry->handle));
      }
      entry->handle = 0;  // Clear invalid handle

      entry->state = ncclDynMemStateReleased;
      releasedPeerImport += entry->size;
      peerImportCount++;
    }
    entry = entry->next;
  }

  // Step 2: Offload and release local memory
  entry = manager->entries;
  while (entry != nullptr) {
    if (entry->memType == ncclMemPersist || entry->isImportedFromPeer || entry->state == ncclDynMemStateReleased) {
      entry = entry->next;
      continue;
    }

    if (entry->memType == ncclMemOffload) {
      NCCLCHECKGOTO(ncclHggcHostCalloc((char**)&entry->cpuBackup, entry->size, nullptr), ret, fail);
      if (entry->cpuBackup == nullptr) {
        WARN("MemManager: Failed to allocate CPU backup for offload");
        ret = ncclSystemError;
        goto fail;
      }

      hggcError_t err = hggcMemcpy(entry->cpuBackup, entry->ptr, entry->size, hggcMemcpyDeviceToHost);
      if (err != hggcSuccess) {
        ncclHggcHostFree(entry->cpuBackup);
        entry->cpuBackup = nullptr;
        ret = ncclUnhandledHggcError;
        goto fail;
      }
      manager->cpuBackupUsage += entry->size;
      releasedOffload += entry->size;
    } else {
      releasedScratch += entry->size;
    }

    if (entry->desc.local.shareableHandleValid &&
        entry->handleType == HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
        entry->desc.local.shareableHandle.fd >= 0) {
      close(entry->desc.local.shareableHandle.fd);
      entry->desc.local.shareableHandle.fd = -1;
      entry->desc.local.shareableHandleValid = false;
    }
    // Unmap physical memory but keep virtual address reservation
    HGCHECKIGNORE(hgMemUnmap((HGdeviceptr)entry->ptr, entry->size));

    // Release physical memory handle
    HGCHECKIGNORE(hgMemRelease(entry->handle));
    entry->handle = 0;  // Clear invalid handle

    entry->state = ncclDynMemStateReleased;
    releasedCount++;
    entry = entry->next;
  }

  manager->released = 1;
  INFO(NCCL_ALLOC, "MemManager: rank %d suspended %d local + %d peer entries (scratch=%zu, offload=%zu, peerImport=%zu, cpuBackup=%zu)",
       comm->rank, releasedCount, peerImportCount, releasedScratch, releasedOffload,
       releasedPeerImport, manager->cpuBackupUsage);
  return ncclSuccess;
fail:
  return ret;
}

/*
 * Internal: Resume previously suspended dynamic memory with P2P coordination
 *
 * Order of operations:
 * 1. First pass: Resume local memory (re-allocate, re-map, restore offloaded data)
 * 2. Exchange: AllGather new handle info for P2P buffers
 * 3. Second pass: Re-import peer buffers using new handles
 */
ncclResult_t ncclCommMemResume(ncclComm_t comm) {
  if (comm == nullptr) return ncclInvalidArgument;
  if (comm->memManager == nullptr) return ncclInvalidUsage;
  ncclMemManager* manager = comm->memManager;
  if (!manager->released) {
    WARN("MemManager: Not in suspended state");
    return ncclInvalidUsage;
  }

  ncclResult_t ret = ncclSuccess;
  int restoredLocalCount = 0, restoredPeerCount = 0;
  size_t restoredLocalBytes = 0, restoredPeerBytes = 0;
  int localBroadcastCount = 0, totalInfoCount = 0;
  int* allCounts = nullptr;
  int* offsets = nullptr;
  ncclDynMemP2pHandleInfo* localInfos = nullptr;
  ncclDynMemP2pHandleInfo* allInfos = nullptr;
  ncclDynMemEntry* pendingEntry = nullptr;
  bool pendingMapped = false;

  // Step 1: Restore all local memory
  ncclDynMemEntry* entry = manager->entries;
  while (entry != nullptr) {
    if (entry->isImportedFromPeer || entry->state != ncclDynMemStateReleased) {
      entry = entry->next;
      continue;
    }

    HGdevice currentDev;
    HGmemAllocationProp prop = {};
    prop.type = HG_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
    HGCHECKGOTO(hgDeviceGet(&currentDev, entry->hggcDev), ret, fail);
    prop.location.id = currentDev;
    prop.requestedHandleTypes = entry->handleType;

    HGmemGenericAllocationHandle newHandle;
    HGCHECKGOTO(hggcEitInterface::getInstance().wrapHgMemCreateEx(
                  &newHandle, entry->size, &prop, HG_MEM_CREATE_MINI_GRANULARITY), ret, fail);

    // Re-map to the same virtual address and set access permissions for local device
    pendingEntry = entry;
    entry->handle = newHandle;
    NCCLCHECKGOTO(ncclHgMemMapAndSetAccess(entry->ptr, entry->size, newHandle, entry->hggcDev, &pendingMapped), ret, fail);

    for (int i = 0; i < entry->desc.local.numExportedPeers; i++) {
      int peerRank = entry->desc.local.exportedPeerRanks[i];
      // Only set access for peers on the same node and same process
      if (peerRank >= 0 && peerRank < comm->nRanks &&
          comm->peerInfo[peerRank].pidHash == comm->peerInfo[comm->rank].pidHash &&
          comm->peerInfo[peerRank].hostHash == comm->peerInfo[comm->rank].hostHash) {
        int peerDev = comm->peerInfo[peerRank].hggcDev;
        HGdevice peerDevice;
        HGCHECKGOTO(hgDeviceGet(&peerDevice, peerDev), ret, fail);
        HGmemAccessDesc peerAccessDesc = {};
        peerAccessDesc.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
        peerAccessDesc.location.id = peerDevice;
        peerAccessDesc.flags = HG_MEM_ACCESS_FLAGS_PROT_READWRITE;
        HGCHECKIGNORE(hgMemSetAccess((HGdeviceptr)entry->ptr, entry->size, &peerAccessDesc, 1));
        TRACE(NCCL_ALLOC, "MemManager: Restored peer access for ptr=%p to rank %d dev %d",
              entry->ptr, peerRank, peerDev);
      }
    }

    // For OFFLOAD type: restore data from CPU backup
    if (entry->memType == ncclMemOffload && entry->cpuBackup != nullptr) {
      HGGCCHECKGOTO(hggcMemcpy(entry->ptr, entry->cpuBackup, entry->size, hggcMemcpyHostToDevice), ret, fail);
    }

    entry->desc.local.shareableHandleValid = false;
    if (entry->handleType == HG_MEM_HANDLE_TYPE_FABRIC) {
      HGresult exportRet = hgMemExportToShareableHandle(&entry->desc.local.shareableHandle.fabricHandle, newHandle,
                                                        HG_MEM_HANDLE_TYPE_FABRIC, 0);
      if (exportRet != HGGC_SUCCESS) {
        WARN("MemManager: hgMemExportToShareableHandle (FABRIC) failed for ptr=%p", entry->ptr);
        HGCHECKIGNORE(hgMemUnmap((HGdeviceptr)entry->ptr, entry->size));
        HGCHECKIGNORE(hgMemRelease(newHandle));
        entry->handle = 0;
        pendingEntry = nullptr;
        pendingMapped = false;
        ret = ncclUnhandledHggcError;
        goto fail;
      }
      entry->desc.local.shareableHandleValid = true;
    }

    if (entry->cpuBackup != nullptr) {
      NCCLCHECKGOTO(ncclHggcHostFree(entry->cpuBackup), ret, fail);
      entry->cpuBackup = nullptr;
      manager->cpuBackupUsage -= entry->size;
    }
    entry->state = ncclDynMemStateActive;
    pendingEntry = nullptr;
    pendingMapped = false;
    restoredLocalCount++;
    restoredLocalBytes += entry->size;
    TRACE(NCCL_ALLOC, "MemManager: Resumed local buffer ptr=%p size=%zu numExportedPeers=%d",
          entry->ptr, entry->size, entry->desc.local.numExportedPeers);
    entry = entry->next;
  }

  // Step 2: Barrier to ensure all ranks have resumed their local memory
  if (comm->bootstrap != nullptr) {
    INFO(NCCL_ALLOC, "MemManager: rank %d resumed %d local entries, waiting at barrier",
         comm->rank, restoredLocalCount);
    ret = bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0xBEEF);
    if (ret != ncclSuccess) {
      WARN("MemManager: Barrier failed during resume");
      return ret;
    }
  }

  // Step 3: Exchange new handle info for P2P coordination
  entry = manager->entries;
  while (entry != nullptr) {
    if (!entry->isImportedFromPeer && entry->desc.local.numExportedPeers > 0 && entry->state == ncclDynMemStateActive) {
      localBroadcastCount++;
    }
    entry = entry->next;
  }

  if (comm->bootstrap != nullptr && comm->nRanks > 1) {
    NCCLCHECKGOTO(ncclCalloc(&allCounts, comm->nRanks), ret, fail);
    allCounts[comm->rank] = localBroadcastCount;
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allCounts, sizeof(int)), ret, fail);

    NCCLCHECKGOTO(ncclCalloc(&offsets, comm->nRanks), ret, fail);
    for (int r = 0; r < comm->nRanks; r++) {
      offsets[r] = totalInfoCount;
      totalInfoCount += allCounts[r];
    }

    if (totalInfoCount > 0) {
      int localAllocCount = localBroadcastCount > 0 ? localBroadcastCount : 1;
      NCCLCHECKGOTO(ncclCalloc(&localInfos, localAllocCount), ret, exchange_fail);
      NCCLCHECKGOTO(ncclCalloc(&allInfos, totalInfoCount), ret, exchange_fail);

      int idx = 0;
      entry = manager->entries;
      while (entry != nullptr && idx < localBroadcastCount) {
        if (!entry->isImportedFromPeer && entry->desc.local.numExportedPeers > 0 && entry->state == ncclDynMemStateActive) {
          localInfos[idx].ptr = entry->ptr;
          localInfos[idx].ownerRank = comm->rank;
          localInfos[idx].ownerDev = entry->hggcDev;
          localInfos[idx].size = entry->size;
          localInfos[idx].handleType = entry->handleType;

          if (entry->handleType == HG_MEM_HANDLE_TYPE_FABRIC) {
            // For FABRIC: copy the exported fabric handle (can be shared directly)
            if (entry->desc.local.shareableHandleValid) {
              memcpy(&localInfos[idx].fabricHandle, &entry->desc.local.shareableHandle.fabricHandle, sizeof(HGmemFabricHandle));
            } else {
              WARN("MemManager: FABRIC handle not valid for entry ptr=%p", entry->ptr);
            }
          } else {
            // For POSIX FD: store the hgMem handle (for FD conversion via proxy)
            memcpy(&localInfos[idx].handleData, &entry->handle, sizeof(HGmemGenericAllocationHandle));
          }
          idx++;
        }
        entry = entry->next;
      }

      if (localBroadcastCount > 0) {
        memcpy(allInfos + offsets[comm->rank], localInfos, localBroadcastCount * sizeof(ncclDynMemP2pHandleInfo));
      }
      for (int r = 0; r < comm->nRanks; r++) {
        if (r != comm->rank && localBroadcastCount > 0) {
          NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, r, 0xFEED, localInfos, localBroadcastCount * sizeof(ncclDynMemP2pHandleInfo)), ret, exchange_fail);
        }
      }
      for (int r = 0; r < comm->nRanks; r++) {
        if (r != comm->rank && allCounts[r] > 0) {
          NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, r, 0xFEED, allInfos + offsets[r], allCounts[r] * sizeof(ncclDynMemP2pHandleInfo)), ret, exchange_fail);
        }
      }
    }
    ncclFree(offsets);
    goto exchange_done;
exchange_fail:
    ncclFree(offsets);
    goto fail;
  }
exchange_done:

  // Step 4: Re-import peer buffers using exchanged handle info
  entry = manager->entries;
  while (entry != nullptr) {
    if (entry->isImportedFromPeer && entry->state == ncclDynMemStateReleased) {
      ncclDynMemP2pHandleInfo* matchedInfo = nullptr;
      for (int i = 0; allInfos != nullptr && i < totalInfoCount; i++) {
        if (allInfos[i].ownerRank == entry->desc.imported.ownerRank &&
            allInfos[i].ptr == entry->desc.imported.ownerPtr &&
            allInfos[i].size == entry->size) {
          matchedInfo = &allInfos[i];
          break;
        }
      }
      if (matchedInfo == nullptr) {
        WARN("MemManager: Could not find matching handle info for ptr=%p from rank %d",
             entry->ptr, entry->desc.imported.ownerRank);
        entry = entry->next;
        continue;
      }

      TRACE(NCCL_ALLOC, "MemManager: Re-importing peer buffer ptr=%p from rank %d (owner ptr=%p)",
            entry->ptr, entry->desc.imported.ownerRank, matchedInfo->ptr);

      HGmemGenericAllocationHandle newHandle;
      HGresult hgret;
      if (matchedInfo->handleType == HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
        // POSIX FD handles only work within the same node - check hostHash
        if (comm->peerInfo && comm->peerInfoValid &&
            comm->peerInfo[entry->desc.imported.ownerRank].hostHash !=
            comm->peerInfo[comm->rank].hostHash) {
          WARN("MemManager: Cannot re-import peer buffer from rank %d (different node) using POSIX FD - skipping",
               entry->desc.imported.ownerRank);
          entry = entry->next;
          continue;
        }
        // For POSIX FD: use the proxy to convert the owner's hgMem handle to an FD
        int fd = -1;
        ret = ncclProxyClientGetFdBlocking(comm, entry->desc.imported.ownerRank,
                                           &matchedInfo->handleData, &fd);
        if (ret != ncclSuccess || fd < 0) {
          WARN("MemManager: Failed to get FD from rank %d for ptr=%p",
               entry->desc.imported.ownerRank, entry->ptr);
          entry = entry->next;
          continue;
        }
        hgret = hgMemImportFromShareableHandle(&newHandle, (void*)(uintptr_t)fd,
                                                HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
        close(fd);
      } else if (matchedInfo->handleType == HG_MEM_HANDLE_TYPE_FABRIC) {
        // For FABRIC: Import directly using the fabric handle
        hgret = hgMemImportFromShareableHandle(&newHandle, &matchedInfo->fabricHandle,
                                                HG_MEM_HANDLE_TYPE_FABRIC);
      } else {
        WARN("MemManager: Unknown handle type %d for peer import", matchedInfo->handleType);
        entry = entry->next;
        continue;
      }

      if (hgret != HGGC_SUCCESS) {
        WARN("MemManager: hgMemImportFromShareableHandle failed for ptr=%p (ret=%d)",
             entry->ptr, hgret);
        entry = entry->next;
        continue;
      }

      // Re-map to the same virtual address and set access permissions
      bool mapped = false;
      ncclResult_t mapResult = ncclHgMemMapAndSetAccess(
        entry->ptr, entry->size, newHandle, comm->hggcDev, &mapped);
      if (mapResult != ncclSuccess) {
        if (mapped && hgMemUnmap((HGdeviceptr)entry->ptr, entry->size) != HGGC_SUCCESS) {
          entry->handle = newHandle;
          entry->state = ncclDynMemStateActive;
          ret = mapResult;
          goto fail;
        }
        HGCHECKIGNORE(hgMemRelease(newHandle));
        entry->handle = 0;
        WARN("MemManager: ncclHgMemMapAndSetAccess failed for re-imported ptr=%p", entry->ptr);
        entry = entry->next;
        continue;
      }

      entry->handle = newHandle;
      entry->state = ncclDynMemStateActive;
      restoredPeerCount++;
      restoredPeerBytes += entry->size;

      TRACE(NCCL_ALLOC, "MemManager: Successfully re-imported peer buffer ptr=%p from rank %d",
            entry->ptr, entry->desc.imported.ownerRank);
    }
    entry = entry->next;
  }

  manager->released = 0;

  // Final barrier to ensure all ranks have completed peer import setup
  if (comm->bootstrap != nullptr) {
    INFO(NCCL_ALLOC, "MemManager: rank %d resumed %d local + %d peer entries (%zu + %zu bytes)",
         comm->rank, restoredLocalCount, restoredPeerCount, restoredLocalBytes, restoredPeerBytes);
    NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0xCAFE), ret, fail);
  }

  // Cleanup
  ncclFree(offsets);
  ncclFree(allCounts);
  ncclFree(localInfos);
  ncclFree(allInfos);
  return ncclSuccess;

fail:
  if (pendingEntry != nullptr) {
    if (pendingMapped && hgMemUnmap((HGdeviceptr)pendingEntry->ptr, pendingEntry->size) != HGGC_SUCCESS) {
      pendingEntry->state = ncclDynMemStateActive;
    } else {
      HGCHECKIGNORE(hgMemRelease(pendingEntry->handle));
      pendingEntry->handle = 0;
      pendingEntry->desc.local.shareableHandleValid = false;
    }
  }
  ncclFree(offsets);
  ncclFree(allCounts);
  ncclFree(localInfos);
  ncclFree(allInfos);
  return ret;
}

/*
 * Public Communicator Suspend/Resume APIs
 */
NCCL_API(ncclResult_t, pcclCommSuspend, pcclComm_t comm, int flags);
ncclResult_t pcclCommSuspend(pcclComm_t comm, int flags) {
  NCCLCHECK(CommCheck(comm, "pcclCommSuspend", "comm"));
  NCCLCHECK(ncclCommEnsureReady(comm));
  ncclResult_t ret = ncclSuccess;
  ncclResult_t groupRet;
  int savedDev;
  struct ncclMemManagerTask* task = nullptr;
  HGGCCHECK(hggcGetDevice(&savedDev));
  NCCLCHECK(ncclGroupStartInternal());
  HGGCCHECKGOTO(hggcSetDevice(comm->hggcDev), ret, fail);

  if (flags & PCCL_SUSPEND_MEM) {
    if (ncclParamMemManagerDisable()) {
      WARN("MemManager: Suspend not supported, memory manager is disabled");
      ret = ncclInvalidUsage;
      goto fail;
    }
    if (comm->memManager && comm->memManager->refCount > 1) {
      WARN("Memory suspend not supported with shared communicators (refCount=%d)", comm->memManager->refCount);
      ret = ncclInvalidUsage;
      goto fail;
    }
    NCCLCHECKGOTO(ncclCalloc(&task, 1), ret, fail);
    task->comm = comm;
    ncclIntruQueueEnqueue(&comm->suspendTaskQueue, task);
    ncclGroupCommJoin(comm, ncclGroupTaskTypeSymRegister);
  }

exit:
  ncclGroupErrCheck(ret);
  groupRet = ncclGroupEndInternal();
  if (ret == ncclSuccess) ret = groupRet;
  if (ret == ncclSuccess && !comm->config.blocking) {
    ncclResult_t asyncRet = pcclCommGetAsyncError(comm, &ret);
    if (asyncRet != ncclSuccess) ret = asyncRet;
  }
  HGGCCHECK(hggcSetDevice(savedDev));
  return ret;
fail:
  ncclFree(task);
  goto exit;
}

NCCL_API(ncclResult_t, pcclCommResume, pcclComm_t comm);
ncclResult_t pcclCommResume(pcclComm_t comm) {
  NCCLCHECK(CommCheck(comm, "pcclCommResume", "comm"));
  NCCLCHECK(ncclCommEnsureReady(comm));
  ncclResult_t ret = ncclSuccess;
  ncclResult_t groupRet;
  int savedDev;
  struct ncclMemManagerTask* task = nullptr;
  HGGCCHECK(hggcGetDevice(&savedDev));
  NCCLCHECK(ncclGroupStartInternal());
  HGGCCHECKGOTO(hggcSetDevice(comm->hggcDev), ret, fail);

  if (ncclParamMemManagerDisable()) {
    WARN("MemManager: Resume not supported, memory manager is disabled");
    ret = ncclInvalidUsage;
    goto fail;
  }
  if (comm->memManager && comm->memManager->refCount > 1) {
    WARN("Memory resume not supported with shared communicators (refCount=%d)", comm->memManager->refCount);
    ret = ncclInvalidUsage;
    goto fail;
  }
  NCCLCHECKGOTO(ncclCalloc(&task, 1), ret, fail);
  task->comm = comm;
  ncclIntruQueueEnqueue(&comm->resumeTaskQueue, task);
  ncclGroupCommJoin(comm, ncclGroupTaskTypeSymRegister);

exit:
  ncclGroupErrCheck(ret);
  groupRet = ncclGroupEndInternal();
  if (ret == ncclSuccess) ret = groupRet;
  if (ret == ncclSuccess && !comm->config.blocking) {
    ncclResult_t asyncRet = pcclCommGetAsyncError(comm, &ret);
    if (asyncRet != ncclSuccess) ret = asyncRet;
  }
  HGGCCHECK(hggcSetDevice(savedDev));
  return ret;
fail:
  ncclFree(task);
  goto exit;
}

NCCL_API(ncclResult_t, pcclCommMemStats, pcclComm_t comm, pcclCommMemStat_t stat, uint64_t* value);
ncclResult_t pcclCommMemStats(pcclComm_t comm, pcclCommMemStat_t stat, uint64_t* value) {
  NCCLCHECK(CommCheck(comm, "pcclCommMemStats", "comm"));
  NCCLCHECK(ncclCommEnsureReady(comm));
  if (value == nullptr) return ncclInvalidArgument;

  if (ncclParamMemManagerDisable()) {
    WARN("MemManager: MemStats not supported, memory manager is disabled");
    return ncclInvalidUsage;
  }

  if (comm->memManager == nullptr) {
    *value = 0;
    return ncclSuccess;
  }

  ncclMemManager* manager = comm->memManager;
  switch (stat) {
    case pcclStatGpuMemTotal:
      *value = __atomic_load_n(&manager->totalPersist, __ATOMIC_RELAXED) +
               __atomic_load_n(&manager->totalScratch, __ATOMIC_RELAXED) +
               __atomic_load_n(&manager->totalOffload, __ATOMIC_RELAXED);
      return ncclSuccess;
    case pcclStatGpuMemPersist:
      *value = __atomic_load_n(&manager->totalPersist, __ATOMIC_RELAXED);
      return ncclSuccess;
    case pcclStatGpuMemSuspend:
      *value = __atomic_load_n(&manager->totalScratch, __ATOMIC_RELAXED) +
               __atomic_load_n(&manager->totalOffload, __ATOMIC_RELAXED);
      return ncclSuccess;
    case pcclStatGpuMemSuspended:
      // Boolean: 0=active, 1=suspended
      *value = manager->released ? 1 : 0;
      return ncclSuccess;
    default:
      return ncclInvalidArgument;
  }
}
