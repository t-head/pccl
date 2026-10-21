/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
// dev_runtime.h: Device runtime state and function declarations.

#ifndef NCCL_DEV_RUNTIME_H_
#define NCCL_DEV_RUNTIME_H_

#include "pccl_internal.h"
#include "pccl_device/core.h"
#include "allocator.h"
#include "bitops.h"
#include "utils.h"

////////////////////////////////////////////////////////////////////////////////
// ncclDevr[_]: runtime implementations for symmetric API.

struct ncclDevrMemory;

struct ncclDevrWindow {
  struct ncclDevrMemory* memory;
  void* userPtr;
  size_t size;
  size_t bigOffset;         // Offset in big VA space
  int winFlags;
  void* localRegHandle;
  bool globalSameMemOffset;  // true if every rank registered this window at the same allocation offset
  struct ncclWindow_vidmem* vidmem;  // key for intrusive address map
  struct ncclDevrWindow* next;       // next for intrusive address map
  struct pcclComm* comm;             // comm for window <> comm lookup
};

struct ncclDevrWindowSorted;
struct ncclDevrTeam;

struct ncclDevrRegTask {
  struct ncclDevrRegTask* next;
  void* userPtr;
  size_t userSize;
  int winFlags;
  ncclWindow_t* outWinDev;   // Output: pointer to device window location
};

struct ncclDevrCommCreateTask {
  struct ncclDevrCommCreateTask* next;
  struct ncclDevCommRequirements* reqs;
  struct ncclDevComm* outDevComm;
};

struct ncclDevrState {
  // LSA (Local Shared Addressing) topology:
  // "lsa" ranks must be consecutive in the world and all lsa subsets
  // have the same number of ranks. If any condition is false then the
  // lsa team is just the singleton of self.
  int lsaSelf;
  int lsaSize;
  int* lsaRankList;
  int nLsaTeams;

  size_t granularity;         // hgMemGetAllocationGranularity
  bool ginEnabled;
  struct ncclDevrMemory* memHead;
  struct ncclDevrWindowSorted* winSorted;
  int winSortedCapacity, winSortedCount;
  struct ncclDevrTeam* teamHead;
  size_t bigSize;             // size of our big logical space (128GB?)
  struct ncclSpace bigSpace;  // allocates our big VA space
  void* lsaFlatBase;         // base ptr for all lsa ranks' big VA's concatenated
  struct ncclShadowPool shadows;
  struct ncclDevCommWindowTable* windowTable;

  struct ncclIntruQueue<struct ncclDevrRegTask, &ncclDevrRegTask::next> regTaskQueue;
  struct ncclIntruQueue<struct ncclDevrCommCreateTask, &ncclDevrCommCreateTask::next> commCreateTaskQueue;
};

// Check if GIN resources have been requested as part of `reqs`.
bool ncclGinResourcesRequested(struct ncclDevCommRequirements const* reqs);

// Check if there is only one LSA team.
bool ncclDevrIsOneLsaTeam(struct pcclComm* comm);

// Initialize/finalize the devrState for a communicator.
ncclResult_t ncclDevrInitOnce(struct pcclComm* comm);
ncclResult_t ncclDevrFinalize(struct pcclComm* comm);

// Symmetric VA stride (NCCL_WIN_STRIDE). -1 means auto-compute from peerInfo.
int64_t ncclParamWinStride();

// Find a registered window by user pointer address.
// If found, *outWin points to the host window; otherwise *outWin is nullptr.
ncclResult_t ncclDevrFindWindow(struct pcclComm* comm, void const* userPtr, struct ncclDevrWindow** outWin);

// Register a user buffer as a symmetric window (called during group commit).
ncclResult_t ncclDevrWindowRegisterInGroup(
  struct pcclComm* comm, void* ptr, size_t size, int winFlags, ncclWindow_t* outWinDev
);

// Create a device communicator with the given resource requirements.
ncclResult_t ncclDevrCommCreateInternal(
  struct pcclComm* comm, struct ncclDevCommRequirements* reqs, struct ncclDevComm* outDevComm,
  bool isInternal = false
);

// Free deep-copied requirements
void freeDevCommRequirements(struct ncclDevCommRequirements* reqs);

// Query if a window has multi-segment or sysmem segments
bool ncclDevrWindowIsMultiSegment(struct ncclDevrWindow* win);
bool ncclDevrWindowHasSysmemSegment(struct ncclDevrWindow* win);

// Get the data-window MR handle registered on GIN connection `ginCommIdx`.
// The handles are filled by ncclGinRegister() and consumed by the RMA proxy
// launch layer.
void* ncclDevrGetGinWin(struct ncclDevrWindow* winHost, int ginCommIdx);

// Get the byte offset of a window within its backing memory allocation.
size_t ncclDevrGetWinOffset(struct ncclDevrWindow* winHost);

// Get pointer into another LSA rank's symmetric memory
ncclResult_t ncclDevrGetLsaRankPtr(struct pcclComm* comm, struct ncclDevrWindow* winHost, size_t offset, int lsaRank, void** outPtr);

// Convert world rank to LSA rank
ncclResult_t ncclDevrWorldToLsaRank(struct pcclComm* comm, int peerWorldRank, int* peerLsaRank);

// Copy DevComm LSA data (barrier state etc) between ranks
void ncclDevCommCopyLsaData(void* dstRankPtr, void const* srcRankPtr);

// Dump DevComm info for debugging
void ncclDevCommDump(struct ncclDevComm* devComm);

// Get user pointer from a device window. Public API: declared via NCCL_API
// in dev_runtime.cc (extern "C"). Mirror that linkage here so internal callers
// see a matching declaration.
extern "C" ncclResult_t ncclWinGetUserPtr(struct pcclComm* comm, struct ncclWindow_vidmem* win, void** outUserPtr);

// Get LSA/peer device pointers from a device window
ncclResult_t ncclGetLsaDevicePointer(ncclWindow_t window, size_t offset, int lsaRank, void** outPtr);
ncclResult_t ncclGetPeerDevicePointer(ncclWindow_t window, size_t offset, int peer, void** outPtr);

////////////////////////////////////////////////////////////////////////////////
// Team API: Host-side functions (take ncclComm_t / pcclComm*)

ncclTeam_t ncclTeamWorld(ncclComm_t comm);
ncclTeam_t ncclTeamLsa(ncclComm_t comm);
ncclTeam_t ncclTeamRail(ncclComm_t comm);
int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank);
int ncclTeamRankToLsa(ncclComm_t comm, ncclTeam_t team, int rank);

// Inline team math (host + device compatible, operate on ncclTeam_t only) is
// provided by pccl_device/impl/core__funcs.h in the canonical NCCL form. The
// duplicate static-inline copies that previously lived here have been removed.

#endif // NCCL_DEV_RUNTIME_H_
