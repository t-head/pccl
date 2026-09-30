/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "dev_runtime.h"
#include "comm.h"
#include "pccl_device/core.h"
#include "pccl_device/gin_barrier.h"
#include "pccl_device/lsa_barrier.h"
#include "devcomm.h"
#include "sym_kernels.h"
#include "transport.h"
#include "group.h"
#include "utils.h"
#include "argcheck.h"
#include "gin/gin_host.h"
#include <mutex>

NCCL_PARAM(WinStride, "WIN_STRIDE", -1);
NCCL_PARAM(ElasticBufferRegister, "ELASTIC_BUFFER_REGISTER", 1);
NCCL_PARAM(SymReuseSysmemHandles, "SYM_REUSE_SYSMEM_HANDLES", 0);

// Global window map using intrusive address map
// Uses ncclDevrWindow directly (vidmem as key, next pointer embedded in struct)
static std::mutex ncclWindowMapMutex;
static ncclIntruAddressMap<ncclDevrWindow, struct ncclWindow_vidmem*, &ncclDevrWindow::vidmem, &ncclDevrWindow::next> ncclWindowMap;
static ncclResult_t symWindowDestroy(struct pcclComm* comm, struct ncclWindow_vidmem* winDev, hggcStream_t stream);

struct multiSegmentGinInfo {
  void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS];
  ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS];
  HGmemLocationType memType;
  size_t segmentSize;
};

// Complete types from src/include/dev_runtime.h
struct ncclDevrMemory {
  int refCount;
  struct ncclDevrMemory* next;
  HGmemGenericAllocationHandle* memHandles;
  void* primaryAddr; // What we hope is the VA of this memory's first mapping.
  size_t size;
  size_t bigOffset; // offset in big VA space
  void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS];
  ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS];
  int winFlags;
  // Per-rank info derived from this rank's own allocation
  int numSegments;         // number of physical segments backing this rank's buffer
  bool hasSysmemSegment;   // true if any segment is CPU-backed (HOST_NUMA memory type)
  size_t* segmentSizes;    // size of each segment, length numSegments
  // Communicator-wide aggregates over all nRanks, populated via bootstrapAllGather
  int maxGlobalNumSegments;    // max(numSegments) across all communicator ranks
  bool globalHasSysmemSegment; // true if any communicator rank has a sysmem segment
  bool globalSameMemOffset;    // true if every rank registered this buffer at the same allocation offset
  // LSA-team aggregates, derived from a global allgather
  int* lsaNumSegments;   // numSegments for each LSA rank, length lsaSize
  // GIN registration state
  int numGinSegments;                          // 1 if !globalHasSysmemSegment; else == numSegments
  struct multiSegmentGinInfo* ginSegmentInfos; // per-GIN-segment info; valid only when numGinSegments > 1
};

struct ncclDevrWindowSorted {
  uintptr_t userAddr;
  size_t size;
  struct ncclDevrWindow* win;
};

struct ncclDevrTeam {
  struct ncclDevrTeam* next;
  struct ncclTeam team;
  int worldRankList[];
};

////////////////////////////////////////////////////////////////////////////////
// Helpers at the bottom:

// Find least index such that `arg < sorted[i].key` (least upper bound)
template<typename Obj, typename Key>
static int listFindSortedLub(Key Obj::*key, Obj* sorted, int count, Key arg);

template<typename Obj>
static void listInsert(Obj** list, int* capacity, int* count, int index, Obj val);

template<typename Obj>
static void listRemove(Obj* list, int* count, int index);

////////////////////////////////////////////////////////////////////////////////

NCCL_PARAM(LsaTeamSize, "LSA_TEAM_SIZE", 0)

// Compute the LSA team size from the comm topology without any side effects.
static int computeLsaSize(struct pcclComm* comm) {
  if (comm->devrState.bigSize != 0) return comm->devrState.lsaSize;

  // LSA needs to be the same size for all ranks, and it needs to represent
  // a consecutive set of ranks.
  int lsaSize = ncclParamLsaTeamSize();
  // Standard node-based gcd LSA calculation
  int nodeSize = 1;
  for (int r=1; r < comm->nRanks; r++) {
    if (comm->rankToNode[r] == comm->rankToNode[r-1]) {
      nodeSize += 1;
    } else {
      lsaSize = gcd(lsaSize, nodeSize);
      nodeSize = 1;
    }
  }
  lsaSize = gcd(lsaSize, nodeSize);

  return lsaSize;
}

bool ncclDevrIsOneLsaTeam(struct pcclComm* comm) {
  int lsaSize = computeLsaSize(comm);
  return lsaSize == comm->nRanks; // Same as comm->nRanks / comm->devrState.lsaSize == 1
}

ncclResult_t ncclDevrInitOnce(struct pcclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  if (devr->bigSize != 0) return ncclSuccess;

  // LSA needs to be the same size for all ranks, and it needs to represent
  // a consecutive set of ranks.
  int lsaSize = computeLsaSize(comm);
  devr->lsaSize = lsaSize;
  devr->lsaSelf = comm->rank % lsaSize;
  devr->lsaRankList = (int*)malloc(devr->lsaSize*sizeof(int));
  for (int i=0; i < devr->lsaSize; i++) {
    devr->lsaRankList[i] = comm->rank + (i - devr->lsaSelf);
  }
  devr->nLsaTeams = comm->nRanks / devr->lsaSize;

  HGmemAllocationProp memProp = {};
  memProp.type = HG_MEM_ALLOCATION_TYPE_PINNED;
  memProp.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
  memProp.requestedHandleTypes = ncclHgMemHandleType;
  memProp.location.id = comm->hggcDev;
  HGCHECKGOTO(hgMemGetAllocationGranularity(&devr->granularity, &memProp, HG_MEM_ALLOC_GRANULARITY_RECOMMENDED), ret, fail_lsaRankList);

  devr->bigSize = ncclParamWinStride();
  if (-devr->bigSize <= 1) {
    devr->bigSize = 1;
    for (int r=0; r < comm->nRanks; ++r) {
      devr->bigSize = std::max<size_t>(devr->bigSize, comm->peerInfo[r].totalGlobalMem);
    }
  }
  devr->bigSize = alignUp(devr->bigSize, size_t(1)<<32);
  INFO(NCCL_INIT, "Symmetric VA size=%ldGB", (long)devr->bigSize>>30);

  ncclSpaceConstruct(&devr->bigSpace);
  ncclShadowPoolConstruct(&devr->shadows);
  return ncclSuccess;

fail_lsaRankList:
  free(devr->lsaRankList);
  return ret;
}

static void symTeamDestroyAll(struct pcclComm* comm); // Further down

ncclResult_t ncclDevrFinalize(struct pcclComm* comm) {
  struct ncclDevrState* devr = &comm->devrState;
  hggcStream_t stream;
  ncclResult_t ret = ncclSuccess;
  if (devr->bigSize == 0) return ncclSuccess;

  while (!ncclIntruQueueEmpty(&devr->regTaskQueue)) {
    struct ncclDevrRegTask* task = ncclIntruQueueDequeue(&devr->regTaskQueue);
    free(task);
  }

  // During abort or any other cases, users might not call deregister API for
  // symmetric window objects, we need to destroy all remaining window objects
  // that are not deregistered by user to avoid memory leaks here.
  HGGCCHECKIGNORE(hggcStreamCreateWithFlags(&stream, hggcStreamNonBlocking));
  while (devr->winSortedCount > 0) {
    struct ncclDevrWindow* win = devr->winSorted[0].win;
    NCCLCHECKIGNORE(symWindowDestroy(comm, win->vidmem, stream), ret);
  }
  HGGCCHECKIGNORE(hggcStreamSynchronize(stream));
  HGGCCHECKIGNORE(hggcStreamDestroy(stream));

  symTeamDestroyAll(comm);
  { // delete windowTable
    hggcStream_t stream;
    if (HGGCSUCCESS(hggcStreamCreateWithFlags(&stream, hggcStreamNonBlocking))) {
      struct ncclDevCommWindowTable* tableDev = devr->windowTable;
      while (tableDev != nullptr) {
        struct ncclDevCommWindowTable* tableHost;
        if (ncclSuccess != ncclShadowPoolToHost(&devr->shadows, tableDev, &tableHost)) break;
        struct ncclDevCommWindowTable* next = tableHost->next;
        ncclShadowPoolFree(&devr->shadows, tableDev, stream);
        tableDev = next;
      }
      hggcStreamSynchronize(stream);
      hggcStreamDestroy(stream);
    }
  }
  if (devr->lsaFlatBase != nullptr) {
    HGdeviceptr flatAddr = reinterpret_cast<HGdeviceptr>(devr->lsaFlatBase);
    HGCHECKIGNORE(hgMemUnmap(flatAddr, devr->lsaSize*devr->bigSize));
    HGCHECKIGNORE(hgMemAddressFree(flatAddr, devr->lsaSize*devr->bigSize));
  }
  ncclShadowPoolDestruct(&devr->shadows);
  ncclSpaceDestruct(&devr->bigSpace);
  free(devr->lsaRankList);
  free(devr->winSorted);
  return ncclSuccess;
}

////////////////////////////////////////////////////////////////////////////////

// Message layout used for LSA team all-gather (one per rank per segment).
struct symLsaMessage {
  union {
    HGmemGenericAllocationHandle memHandle;
    HGmemFabricHandle fabricHandle;
  };
  HGmemLocationType type;
  size_t segmentSize;
};

static ncclResult_t symMemorySetAccessForVASegment(struct pcclComm* comm, symLsaMessage* message, HGdeviceptr addr) {
  HGmemAccessDesc accessDesc;
  accessDesc.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = comm->hggcDev;
  accessDesc.flags = HG_MEM_ACCESS_FLAGS_PROT_READWRITE;
  HGCHECK(hgMemSetAccess(addr, message->segmentSize, &accessDesc, 1));
  return ncclSuccess;
}

static ncclResult_t symMemoryExportSegmentHandle(
    struct pcclComm* comm, symLsaMessage* msg,
    HGmemGenericAllocationHandle memHandle, size_t segmentSize
  ) {
  ncclResult_t ret = ncclSuccess;
  HGmemAllocationProp prop;
  HGCHECKGOTO(hgMemGetAllocationPropertiesFromHandle(&prop, memHandle), ret, fail);
  msg->type = prop.location.type;
  msg->segmentSize = segmentSize;
  if (ncclHgMemHandleType == HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    msg->memHandle = memHandle;
  } else {
    HGCHECKGOTO(hgMemExportToShareableHandle(&msg->fabricHandle, memHandle, ncclHgMemHandleType, 0), ret, fail);
  }
fail:
  return ret;
}

static ncclResult_t symMemoryImportAndMapSegmentHandle(
    struct pcclComm* comm, int r, HGdeviceptr addr, symLsaMessage* msg,
    HGmemGenericAllocationHandle memHandle, bool reuseLocal
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  HGmemGenericAllocationHandle impHandle;
  if (reuseLocal) {
    impHandle = memHandle;
  } else {
    if (ncclHgMemHandleType == HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      int fd = -1;
      NCCLCHECKGOTO(ncclProxyClientGetFdBlocking(comm, devr->lsaRankList[r], msg, &fd), ret, fail);
      HGCHECKGOTO(hgMemImportFromShareableHandle(&impHandle, reinterpret_cast<void*>((uintptr_t)fd), ncclHgMemHandleType), ret, fail);
      SYSCHECKGOTO(close(fd), "close", ret, fail);
    } else {
      HGCHECKGOTO(hgMemImportFromShareableHandle(&impHandle, (void*)&msg->fabricHandle, ncclHgMemHandleType), ret, fail);
    }
  }
  HGCHECKGOTO(hgMemMap(addr, msg->segmentSize, 0, impHandle, 0), ret, fail);
  NCCLCHECKGOTO(symMemorySetAccessForVASegment(comm, msg, addr), ret, fail);
  if (!reuseLocal) {
    HGCHECKGOTO(hgMemRelease(impHandle), ret, fail);
  }
fail:
  return ret;
}

static ncclResult_t symMemoryImportAndMapSegmentsForRank(
    struct pcclComm* comm, int r, symLsaMessage* messages, int maxSegments,
    int numSegments, HGmemGenericAllocationHandle* memHandles, size_t bigOffset
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  HGdeviceptr addr = reinterpret_cast<uintptr_t>((char*)devr->lsaFlatBase + r * devr->bigSize + bigOffset);
  for (int segment = 0; segment < numSegments; segment++) {
    symLsaMessage* msg = messages + r * maxSegments + segment;
    bool reuseLocal = (r == devr->lsaSelf) ||
                      (ncclParamSymReuseSysmemHandles() && msg->type == HG_MEM_LOCATION_TYPE_HOST_NUMA);
    HGmemGenericAllocationHandle handle = reuseLocal ? memHandles[segment] : (HGmemGenericAllocationHandle) 0ULL;
    NCCLCHECKGOTO(symMemoryImportAndMapSegmentHandle(comm, r, addr, msg, handle, reuseLocal), ret, fail);
    addr += msg->segmentSize;
  }
fail:
  return ret;
}

static ncclResult_t symMemoryMapLsaTeam(struct pcclComm* comm, struct ncclDevrMemory* mem) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  symLsaMessage* messages = nullptr;
  int* segmentCounts = mem->lsaNumSegments;  // filled from global allgather in symMemoryObtain
  int maxSegments = 0;
  const int numSegments = mem->numSegments;
  size_t* segmentSizes = mem->segmentSizes;

  for (int rank = 0; rank < devr->lsaSize; rank++) {
    maxSegments = std::max(maxSegments, segmentCounts[rank]);
  }

  NCCLCHECKGOTO(ncclCalloc(&messages, (size_t)devr->lsaSize * maxSegments), ret, fail);

  for (int segment = 0; segment < numSegments; segment++) {
    symLsaMessage* msg = messages + devr->lsaSelf * maxSegments + segment;
    NCCLCHECKGOTO(symMemoryExportSegmentHandle(comm, msg, mem->memHandles[segment], segmentSizes[segment]), ret, fail);
    INFO(NCCL_REG, "[%d] Segment %d, Type : %d, numSegments : %d, Segment size : %ld, memHandle : %lld", devr->lsaSelf, segment, msg->type, numSegments, msg->segmentSize, msg->memHandle);
  }

  NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, devr->lsaRankList, devr->lsaSelf, devr->lsaSize, messages, sizeof(symLsaMessage) * maxSegments), ret, fail);

  if (devr->lsaFlatBase == nullptr) { // Create on first need.
    HGdeviceptr addr;
    HGCHECKGOTO(hgMemAddressReserve(&addr, devr->lsaSize*devr->bigSize, NCCL_MAX_PAGE_SIZE, 0, 0), ret, fail);
    devr->lsaFlatBase = reinterpret_cast<void*>(addr);
  }

  for (int r = 0; r < devr->lsaSize; r++) {
    NCCLCHECKGOTO(symMemoryImportAndMapSegmentsForRank(comm, r, messages, maxSegments, segmentCounts[r], mem->memHandles, mem->bigOffset), ret, fail);
  }
  // Ensure everyone has imported my mem handles.
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, devr->lsaRankList, devr->lsaSelf, devr->lsaSize, 0xbeef), ret, fail);
leave:
  free(messages);
  return ret;
fail:
  goto leave;
}

// Caller must barrier the team afterward.
static ncclResult_t symTeamObtain(
    struct pcclComm* comm, struct ncclTeam team,
    struct ncclDevrTeam** outTeam
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  struct ncclDevrTeam* t = devr->teamHead;
  bool teamIsNew = false;
  while (true) {
    if (t == nullptr) {
      teamIsNew = true;
      t = (struct ncclDevrTeam*)malloc(sizeof(struct ncclDevrTeam) + team.nRanks*sizeof(int));
      if (t == nullptr) return ncclSystemError;
      t->team = team;
      for (int i=0; i < team.nRanks; i++) {
        t->worldRankList[i] = comm->rank + (i - team.rank)*team.stride;
      }
      break;
    } else if (t->team.rank == team.rank && t->team.nRanks == team.nRanks && t->team.stride == team.stride) {
      if (outTeam) *outTeam = t;
      return ncclSuccess;
    } else {
      t = t->next;
    }
  }

  if (teamIsNew) {
     // Add to list
    t->next = devr->teamHead;
    devr->teamHead = t;
  }
  if (outTeam) *outTeam = t;
  return ret;
}

static void symTeamDestroyAll(struct pcclComm* comm) {
  struct ncclDevrState* devr = &comm->devrState;
  while (devr->teamHead != nullptr) {
    struct ncclDevrTeam* t = devr->teamHead;
    devr->teamHead = t->next;
    free(t);
  }
}

static ncclResult_t symMemoryRegisterGin(struct pcclComm* comm, struct ncclDevrMemory* mem) {
  ncclResult_t ret = ncclSuccess;
  HGmemLocationType* segmentTypes = nullptr;
  size_t* paddedSegmentSizes = nullptr;
  size_t* globalPaddedSegmentSizes = nullptr;
  int numSegmentsRegistered = 0;

  if (!mem->globalHasSysmemSegment) {
    NCCLCHECK(ncclGinRegister(comm, mem->primaryAddr, mem->size, mem->ginHostWins, mem->ginDevWins,
                              mem->winFlags, mem->maxGlobalNumSegments > 1, NCCL_PTR_HGGC));
    NCCLCHECK(ncclCalloc(&mem->ginSegmentInfos, 1));
    mem->ginSegmentInfos[0].memType = HG_MEM_LOCATION_TYPE_DEVICE;
    mem->ginSegmentInfos[0].segmentSize = mem->size;
    for (int i = 0; i < NCCL_GIN_MAX_CONNECTIONS; i++) {
      mem->ginSegmentInfos[0].ginDevWins[i] = mem->ginDevWins[i];
      mem->ginSegmentInfos[0].ginHostWins[i] = mem->ginHostWins[i];
    }
    mem->numGinSegments = 1;
  } else {
    if (mem->maxGlobalNumSegments != mem->numSegments) {
      WARN("Elastic GIN: rank %d has %d segments but another rank has %d segments; all ranks must have identical segment configurations",
           comm->rank, mem->numSegments, mem->maxGlobalNumSegments);
      return ncclInvalidUsage;
    }

    NCCLCHECKGOTO(ncclCalloc(&segmentTypes, mem->numSegments), ret, fail);
    for (int segment = 0; segment < mem->numSegments; segment++) {
      HGmemAllocationProp prop;
      HGCHECKGOTO(hgMemGetAllocationPropertiesFromHandle(&prop, mem->memHandles[segment]), ret, fail);
      segmentTypes[segment] = prop.location.type;
    }

    // Gather segment sizes from all ranks to verify they are identical.
    NCCLCHECKGOTO(ncclCalloc(&paddedSegmentSizes, mem->maxGlobalNumSegments), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&globalPaddedSegmentSizes, (size_t)mem->maxGlobalNumSegments * comm->nRanks), ret, fail);

    memcpy(paddedSegmentSizes, mem->segmentSizes, mem->numSegments * sizeof(size_t));
    memcpy(&globalPaddedSegmentSizes[(size_t)comm->rank * mem->maxGlobalNumSegments],
           paddedSegmentSizes, sizeof(size_t) * mem->maxGlobalNumSegments);

    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, globalPaddedSegmentSizes,
                                     sizeof(size_t) * mem->maxGlobalNumSegments), ret, fail);

    for (int rank = 0; rank < comm->nRanks; rank++) {
      for (int seg = 0; seg < mem->numSegments; seg++) {
        if (globalPaddedSegmentSizes[(size_t)rank * mem->maxGlobalNumSegments + seg] != mem->segmentSizes[seg]) {
          WARN("Elastic GIN: rank %d segment %d size %zu differs from rank %d segment %d size %zu; all ranks must have identical segment configurations",
               rank, seg, globalPaddedSegmentSizes[(size_t)rank * mem->maxGlobalNumSegments + seg],
               comm->rank, seg, mem->segmentSizes[seg]);
          ret = ncclInvalidUsage;
          goto fail;
        }
      }
    }

    NCCLCHECKGOTO(ncclCalloc(&mem->ginSegmentInfos, mem->numSegments), ret, fail);

    size_t offset = 0;
    for (int segment = 0; segment < mem->numSegments; segment++) {
      HGmemLocationType locType = segmentTypes[segment];
      bool segHasSysmem = (locType == HG_MEM_LOCATION_TYPE_HOST_NUMA);
      int ptrType = segHasSysmem ? NCCL_PTR_HOST : NCCL_PTR_HGGC;
      NCCLCHECKGOTO(ncclGinRegister(comm, (char*)mem->primaryAddr + offset, mem->segmentSizes[segment],
                                    mem->ginSegmentInfos[segment].ginHostWins,
                                    mem->ginSegmentInfos[segment].ginDevWins,
                                    mem->winFlags, mem->maxGlobalNumSegments > 1, ptrType), ret, fail);
      mem->ginSegmentInfos[segment].segmentSize = mem->segmentSizes[segment];
      mem->ginSegmentInfos[segment].memType = locType;
      numSegmentsRegistered++;
      offset += mem->segmentSizes[segment];
    }
    mem->numGinSegments = mem->numSegments;
  }
exit:
  free(segmentTypes);
  free(paddedSegmentSizes);
  free(globalPaddedSegmentSizes);
  return ret;
fail:
  for (int i = 0; i < numSegmentsRegistered; i++) {
    ncclGinDeregister(comm, mem->ginSegmentInfos[i].ginHostWins);
  }
  free(mem->ginSegmentInfos);
  mem->ginSegmentInfos = nullptr;
  goto exit;
}

// On success we take caller's reference on memHandle.
// Due to multicast binds for each pre-exiting team, this function requires
// caller do a world barrier before returning to user.
static ncclResult_t symMemoryObtain(
    struct pcclComm* comm, HGmemGenericAllocationHandle* memHandles, int numSegments, void* memAddr, size_t size, int winFlags,
    size_t memOffset, struct ncclDevrMemory** outMem, bool hasSysmemSegment = false
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  int64_t bigOffset = 0;
  struct segmentInfo { int numSegments; bool hasSysmemSegment; size_t memOffset; };
  struct segmentInfo* globalSegmentInfo = nullptr;
  const int globalLsaTeamBaseIdx = devr->lsaSize * (comm->rank / devr->lsaSize);

  struct ncclDevrMemory* mem = devr->memHead;
  while (mem != nullptr) {
    if (mem->primaryAddr == memAddr && mem->size == size && mem->numSegments == numSegments) {
      // Check if all memHandles that [memAddr, memAddr + size] spans also match
      bool allMatch = true;
      for (int segment = 0; segment < mem->numSegments; segment++) {
        if (mem->memHandles[segment] != memHandles[segment]) {
          allMatch = false;
          break;
        }
      }
      if (allMatch) {
        for (int segment = 0; segment < mem->numSegments; segment++) {
          HGCHECKIGNORE(hgMemRelease(memHandles[segment]));
        }
        goto leave;
      }
    }
    mem = mem->next;
  }

  // New memory.
  NCCLCHECKGOTO(ncclCalloc(&mem, 1), ret, fail_mem);
  NCCLCHECKGOTO(ncclCalloc(&mem->memHandles, numSegments), ret, fail_mem);
  memcpy(mem->memHandles, memHandles, sizeof(*mem->memHandles) * numSegments);
  mem->primaryAddr = memAddr;
  mem->size = size;
  mem->winFlags = winFlags;
  mem->hasSysmemSegment = hasSysmemSegment;
  mem->numSegments = numSegments;

  NCCLCHECKGOTO(ncclCalloc(&mem->segmentSizes, numSegments), ret, fail_mem);
  NCCLCHECKGOTO(ncclCalloc(&globalSegmentInfo, comm->nRanks), ret, fail_mem);

  if (numSegments > 1) {
    size_t offset = 0;
    for (int segment = 0; segment < numSegments; segment++) {
      size_t baseSendSize = 0;
      HGdeviceptr segmentStart = reinterpret_cast<HGdeviceptr>(reinterpret_cast<char*>(mem->primaryAddr) + offset);
      HGCHECKGOTO(hgMemGetAddressRange(NULL, &baseSendSize, segmentStart), ret, fail_mem);
      mem->segmentSizes[segment] = baseSendSize;
      offset += baseSendSize;
    }
  } else {
    mem->segmentSizes[0] = size;
  }

  // We need max segments and global sysmem info to selectively disable some features
  globalSegmentInfo[comm->rank].numSegments = numSegments;
  globalSegmentInfo[comm->rank].hasSysmemSegment = hasSysmemSegment;
  globalSegmentInfo[comm->rank].memOffset = memOffset;
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, globalSegmentInfo, sizeof(*globalSegmentInfo)), ret, fail_mem);
  mem->globalHasSysmemSegment = false;
  mem->globalSameMemOffset = true;
  for (int r = 0; r < comm->nRanks; r++) {
    if (mem->maxGlobalNumSegments < globalSegmentInfo[r].numSegments) {
      mem->maxGlobalNumSegments = globalSegmentInfo[r].numSegments;
    }
    if (globalSegmentInfo[r].hasSysmemSegment) mem->globalHasSysmemSegment = true;
    // Every rank must agree on the offset for the flat-address arithmetic in the
    // symmetric p2p DMA path to resolve to the same buffer on both endpoints.
    if (globalSegmentInfo[r].memOffset != memOffset) mem->globalSameMemOffset = false;
  }

  NCCLCHECKGOTO(ncclCalloc(&mem->lsaNumSegments, devr->lsaSize), ret, fail_mem);
  for (int r = 0; r < devr->lsaSize; r++) {
    mem->lsaNumSegments[r] = globalSegmentInfo[globalLsaTeamBaseIdx + r].numSegments;
  }

  // Grab offset in the big space.
  NCCLCHECKGOTO(ncclSpaceAlloc(&devr->bigSpace, devr->bigSize, size, devr->granularity, &bigOffset), ret, fail_mem);
  mem->bigOffset = bigOffset;

  // Map unicast addresses into flat VA space for lsa team.
  NCCLCHECKGOTO(symMemoryMapLsaTeam(comm, mem), ret, fail_mem_space);

  // If our caller doesn't have a VA then we'll use the LSA mapping.
  if (mem->primaryAddr == nullptr) {
    mem->primaryAddr = (char*)devr->lsaFlatBase + devr->lsaSelf*devr->bigSize + mem->bigOffset;
  }

  if (devr->ginEnabled) {
    NCCLCHECKGOTO(symMemoryRegisterGin(comm, mem), ret, fail_mem_space);
  } else {
    // Default to single segment when GIN is not yet enabled.
    // This will be recomputed in ncclDevrCommCreateInternal when GIN is activated.
    mem->numGinSegments = 1;
  }

  // Add to list of mems.
  mem->next = devr->memHead;
  devr->memHead = mem;

leave:
  mem->refCount += 1;
  *outMem = mem;
  free(globalSegmentInfo);
  return ret;

fail_mem_space:
  ncclSpaceFree(&devr->bigSpace, bigOffset, size);
fail_mem:
  if (mem != nullptr) {
    free(mem->memHandles);
    free(mem->segmentSizes);
    free(mem->lsaNumSegments);
  }
  free(mem);
  free(globalSegmentInfo);
//fail:
  return ret;
}

static void symMemoryDropRef(
    struct pcclComm* comm, struct ncclDevrMemory* mem
  ) {
  if (mem != nullptr && 0 == --mem->refCount) {
    struct ncclDevrState* devr = &comm->devrState;
    if (devr->ginEnabled) {
      if (mem->numGinSegments > 1) {
        for (int i = 0; i < mem->numGinSegments; i++) {
          ncclGinDeregister(comm, mem->ginSegmentInfos[i].ginHostWins);
        }
      } else {
        ncclGinDeregister(comm, mem->ginHostWins);
      }
    }
    for (int r = 0; r < devr->lsaSize; r++) {
      HGdeviceptr addr = reinterpret_cast<uintptr_t>((char*)devr->lsaFlatBase + r*devr->bigSize + mem->bigOffset);
      for (int idx = 0; idx < mem->lsaNumSegments[r]; idx++) {
        HGdeviceptr tmpBase;
        size_t tmpBaseSize;
        HGCHECKIGNORE(hgMemGetAddressRange(&tmpBase, &tmpBaseSize, addr));
        HGCHECKIGNORE(hgMemUnmap(addr, tmpBaseSize));
        addr = addr + tmpBaseSize;
      }
    }

    ncclSpaceFree(&devr->bigSpace, mem->bigOffset, mem->size);
    for (int segment = 0; segment < mem->numSegments; segment++) {
      HGCHECKIGNORE(hgMemRelease(mem->memHandles[segment]));
    }

    struct ncclDevrMemory** ptr = &devr->memHead;
    while (*ptr != mem) ptr = &(*ptr)->next;
    *ptr = mem->next; // Remove from list.

    free(mem->ginSegmentInfos);
    free(mem->lsaNumSegments);
    free(mem->segmentSizes);
    free(mem->memHandles);
    free(mem);
  }
}

static ncclResult_t symWindowTableInitOnce(struct pcclComm* comm, hggcStream_t stream) {
  struct ncclDevrState* devr = &comm->devrState;
  struct ncclDevCommWindowTable* tableDev = devr->windowTable;
  if (tableDev == nullptr) { // Create on first need.
    NCCLCHECK(ncclShadowPoolAlloc<ncclDevCommWindowTable>(&devr->shadows, &tableDev, nullptr, stream));
    devr->windowTable = tableDev;
  }
  return ncclSuccess;
}

// Segment windows are used in the multi-segment codepath and need their own shadow pool
// as they're variable in size.
static ncclResult_t allocAndPopulateSegmentWindows(
    struct ncclDevrState* devr, struct ncclDevrMemory* mem,
    hggcStream_t stream, struct ncclSegmentWindow** outSegmentWindowsDev,
    struct ncclSegmentWindow** outSegmentWindowsHost) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSegmentWindow* segmentWindowsDev = nullptr;
  struct ncclSegmentWindow* segmentWindowsHost = nullptr;

  NCCLCHECKGOTO(ncclShadowPoolAlloc(&devr->shadows,
      sizeof(struct ncclSegmentWindow) * mem->numGinSegments,
      (void**)&segmentWindowsDev, (void**)&segmentWindowsHost, stream), ret, fail);

  if (devr->ginEnabled) {
    for (int segment = 0; segment < mem->numGinSegments; segment++) {
      segmentWindowsHost[segment].memType = mem->ginSegmentInfos[segment].memType;
      segmentWindowsHost[segment].segmentSize = mem->ginSegmentInfos[segment].segmentSize;
      for (int i = 0; i < NCCL_GIN_MAX_CONNECTIONS; i++) {
        segmentWindowsHost[segment].ginWins[i] = mem->ginSegmentInfos[segment].ginDevWins[i];
      }
    }
    HGGCCHECKGOTO(hggcMemcpyAsync(segmentWindowsDev, segmentWindowsHost,
        sizeof(struct ncclSegmentWindow) * mem->numGinSegments,
        hggcMemcpyHostToDevice, stream), ret, fail);
  }

  *outSegmentWindowsDev = segmentWindowsDev;
  *outSegmentWindowsHost = segmentWindowsHost;

exit:
  return ret;
fail:
  if (segmentWindowsDev != nullptr)
    ncclShadowPoolFree(&devr->shadows, segmentWindowsDev, stream);
  goto exit;
}

// On success we take callers reference on `mem`.
static ncclResult_t symWindowCreate(
    struct pcclComm* comm, struct ncclDevrMemory* mem,
    size_t memOffset, void* userPtr, size_t userSize, int winFlags, void* localReg,
    bool globalSameMemOffset,
    struct ncclWindow_vidmem** outWinDev, struct ncclDevrWindow** outWin,
    hggcStream_t stream
  ) {
  uintptr_t userAddr = reinterpret_cast<uintptr_t>(userPtr);
  struct ncclDevrState* devr = &comm->devrState;
  struct ncclDevrWindow* win;
  struct ncclWindow_vidmem* winDev;
  struct ncclWindow_vidmem* winDevHost;

  NCCLCHECK(ncclCalloc(&win, 1));
  ncclResult_t ret = ncclShadowPoolAlloc(&devr->shadows, &winDev, &winDevHost, stream);
  if (ret != ncclSuccess) {
    free(win);
    return ret;
  }
  win->memory = mem;
  win->size = userSize;
  win->bigOffset = mem->bigOffset + memOffset;
  win->winFlags = winFlags;
  win->localRegHandle = localReg;
  win->globalSameMemOffset = globalSameMemOffset;
  win->vidmem = winDev;
  if (userPtr == nullptr) {
    // Null means caller has no VA and will use the lsa team flat VA address.
    win->userPtr = userPtr = (char*)devr->lsaFlatBase + (devr->lsaSelf*devr->bigSize) + mem->bigOffset;
    userAddr = reinterpret_cast<uintptr_t>(userPtr);
  } else {
    win->userPtr = userPtr;
  }
  winDevHost->lsaFlatBase = (char*)devr->lsaFlatBase + win->bigOffset;
  winDevHost->stride4G = devr->bigSize>>32;
  winDevHost->lsaRank = devr->lsaSelf;
  winDevHost->worldRank = comm->rank;
  winDevHost->winHost = (void*)win;
  winDevHost->ginOffset4K = memOffset>>12;
  winDevHost->numSegments = mem->numGinSegments;
  for (int i=0; i < NCCL_GIN_MAX_CONNECTIONS; i++) {
    winDevHost->ginWins[i] = mem->ginDevWins[i];
  }
  struct ncclSegmentWindow* segmentWindowsDev;
  struct ncclSegmentWindow* segmentWindowsHost;
  NCCLCHECK(allocAndPopulateSegmentWindows(devr, mem, stream, &segmentWindowsDev, &segmentWindowsHost));
  winDevHost->ginMultiSegmentWins = segmentWindowsDev;
  HGGCCHECK(hggcMemcpyAsync(winDev, winDevHost, sizeof(struct ncclWindow_vidmem), hggcMemcpyHostToDevice, stream));

  NCCLCHECK(symWindowTableInitOnce(comm, stream)); // ensure devr->windowTable exists
  struct ncclDevCommWindowTable* tableDev = devr->windowTable;
  while (true) {
    struct ncclDevCommWindowTable* tableHost;
    NCCLCHECK(ncclShadowPoolToHost(&devr->shadows, tableDev, &tableHost));
    int i = 0;
    while (i < 32 && tableHost->entries[i].window != nullptr) i += 1;
    if (i < 32) {
      tableHost->entries[i].base = userAddr;
      tableHost->entries[i].size = userSize;
      tableHost->entries[i].window = winDev;
      HGGCCHECK(hggcMemcpyAsync(&tableDev->entries[i], &tableHost->entries[i], sizeof(tableHost->entries[i]), hggcMemcpyHostToDevice, stream));
      break;
    }
    if (tableHost->next == nullptr) {
      NCCLCHECK(ncclShadowPoolAlloc<ncclDevCommWindowTable>(&devr->shadows, &tableHost->next, nullptr, stream));
      HGGCCHECK(hggcMemcpyAsync(&tableDev->next, &tableHost->next, sizeof(tableHost->next), hggcMemcpyHostToDevice, stream));
    }
    tableDev = tableHost->next;
  }

  { // insert into winSorted[]
    int i = listFindSortedLub(&ncclDevrWindowSorted::userAddr, devr->winSorted, devr->winSortedCount, userAddr);
    struct ncclDevrWindowSorted winSort;
    winSort.userAddr = userAddr;
    winSort.size = userSize;
    winSort.win = win;
    listInsert(&devr->winSorted, &devr->winSortedCapacity, &devr->winSortedCount, i, winSort);
  }

  if (outWinDev) *outWinDev = winDev;
  if (outWin) *outWin = win;
  return ncclSuccess;
}

static ncclResult_t symWindowDestroy(struct pcclComm* comm, struct ncclWindow_vidmem* winDev, hggcStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  struct ncclWindow_vidmem* winDevHost;
  struct ncclDevrWindow* winHost;

  NCCLCHECKGOTO(ncclShadowPoolToHost(&devr->shadows, winDev, &winDevHost), ret, fail);
  winHost = (struct ncclDevrWindow*)winDevHost->winHost;

  symMemoryDropRef(comm, winHost->memory);

  { struct ncclDevCommWindowTable* tableDev = devr->windowTable;
    while (true) {
      struct ncclDevCommWindowTable* tableHost;
      NCCLCHECKGOTO(ncclShadowPoolToHost(&devr->shadows, tableDev, &tableHost), ret, remove_winSorted);
      int i = 0;
      while (i < 32 && tableHost->entries[i].window != winDev) i += 1;
      if (i < 32) {
        memset(&tableHost->entries[i], 0, sizeof(tableHost->entries[i]));
        HGGCCHECKGOTO(hggcMemsetAsync(&tableDev->entries[i], 0, sizeof(tableDev->entries[i]), stream), ret, remove_winSorted);
        break;
      }
      if (tableHost->next == nullptr) break; // Error didn't find window in table
      tableDev = tableHost->next;
    }
  }

  if (winDevHost->ginMultiSegmentWins != nullptr) {
    NCCLCHECKGOTO(ncclShadowPoolFree(&devr->shadows, winDevHost->ginMultiSegmentWins, stream), ret, remove_winSorted);
  }

  NCCLCHECKGOTO(ncclShadowPoolFree(&devr->shadows, winDev, stream), ret, remove_winSorted);

  NCCLCHECKGOTO(ncclCommDeregister(comm, winHost->localRegHandle), ret, remove_winSorted);

remove_winSorted:
  { int i = listFindSortedLub(&ncclDevrWindowSorted::userAddr, devr->winSorted, devr->winSortedCount, reinterpret_cast<uintptr_t>(winHost->userPtr));
    i -= 1; // least upper bound is just after ours.
    listRemove(devr->winSorted, &devr->winSortedCount, i);
  }
  // Remove the just deallocated window from the table storing the communicator pointer
  {
    std::lock_guard<std::mutex> lock(ncclWindowMapMutex);
    NCCLCHECKGOTO(ncclIntruAddressMapRemove(&ncclWindowMap, winDev), ret, fail);
  }

  free(winHost);
fail:
  return ret;
}

ncclResult_t ncclDevrWindowRegisterInGroup(
    struct pcclComm* comm,
    void* userPtr, size_t userSize, int winFlags, ncclWindow_t* outWinDev
  ) {
  ncclResult_t ret = ncclSuccess;
  HGdeviceptr memAddr = 0;
  size_t memSize = 0;
  HGmemGenericAllocationHandle* memHandles = nullptr;
  size_t memOffset;
  struct ncclDevrMemory* mem = nullptr;
  hggcStream_t stream = nullptr;
  void* localRegHandle = nullptr;
  struct ncclDevrWindow* winHost = nullptr;
  int numSegments = 0;
  size_t offset = 0;
  bool hasSysmemSegment = false;
  bool globalSameMemOffset = true;

  NCCLCHECKGOTO(ncclCommRegister(comm, userPtr, userSize, &localRegHandle), ret, fail);

  if (winFlags & NCCL_WIN_COLL_SYMMETRIC) {
    // Defer symmetric kernel init until at least one window with that flag exists.
    NCCLCHECKGOTO(ncclSymkInitOnce(comm), ret, fail_locReg);
  }

  // Get underlying cumem base address and number of mapped physical segments that userPtr spans
  NCCLCHECKGOTO(ncclHgMemGetAddressRange(reinterpret_cast<HGdeviceptr>(userPtr), userSize, &memAddr, &memSize, &numSegments, &hasSysmemSegment), ret, fail_locReg);
  NCCLCHECKGOTO(ncclCalloc(&memHandles, numSegments), ret, fail_locReg);

  if (hasSysmemSegment) {
    if (!ncclParamElasticBufferRegister()) {
      WARN("VA represented by {userPtr = %p, size = %zu} contains CPU-backed physical segments, but PCCL_ELASTIC_BUFFER_REGISTER is set to 0. Please set PCCL_ELASTIC_BUFFER_REGISTER=1 and retry window registration", userPtr, userSize);
      ret = ncclInvalidArgument;
      goto fail_locReg_memHandle;
    } else if (comm->MNNVL) {
        int multiNodeLsaSupported = 0;
        HGCHECKGOTO(hgDeviceGetAttribute(&multiNodeLsaSupported, HG_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED, comm->hggcDev), ret, fail_locReg_memHandle);
        if (!multiNodeLsaSupported) {
          WARN("VA represented by {userPtr = %p, size = %zu} contains CPU-backed physical segments, but the LSA team does not support multi-node IPC on CPU-backed buffers. Please retry by setting PCCL_MNNVL_ENABLE=0", userPtr, userSize);
          ret = ncclInvalidArgument;
          goto fail_locReg_memHandle;
        }
    }
  }

  memOffset = reinterpret_cast<HGdeviceptr>(userPtr) - memAddr;
  if (memOffset%NCCL_WIN_REQUIRED_ALIGNMENT != 0) {
    WARN("Window address must be suitably aligned.");
    ret = ncclInvalidArgument;
    goto fail_locReg_memHandle;
  }

  // Retain all handles and validate segment location types
  for (int segment = 0; segment < numSegments; segment++) {
    size_t baseSendSize;
    HGCHECKGOTO(hgMemGetAddressRange(nullptr, &baseSendSize, memAddr + offset), ret, fail_locReg_memHandle);
    HGCHECKGOTO(hgMemRetainAllocationHandle(&memHandles[segment], (void *) (reinterpret_cast<char*>(memAddr) + offset)), ret, fail_locReg_memHandle);
    HGmemAllocationProp prop;
    HGCHECKGOTO(hgMemGetAllocationPropertiesFromHandle(&prop, memHandles[segment]), ret, fail_locReg_memHandle);
    if (prop.location.type != HG_MEM_LOCATION_TYPE_HOST_NUMA && prop.location.type != HG_MEM_LOCATION_TYPE_DEVICE) {
      WARN("Segment %d has unsupported location type %d. Symmetric memory currently only supports HG_MEM_LOCATION_TYPE_HOST_NUMA and HG_MEM_LOCATION_TYPE_DEVICE.", segment, (int)prop.location.type);
      ret = ncclInvalidArgument;
      goto fail_locReg_memHandle;
    }
    offset += baseSendSize;
  }

  // Trade cumem handles for ncclDevrMemory*. symMemoryObtain folds the memOffset
  // exchange into its own segment-info allgather and records whether every rank
  // agreed on the offset, so no separate bootstrap round is needed here.
  NCCLCHECKGOTO(symMemoryObtain(comm, memHandles, numSegments, (void*)memAddr, memSize, winFlags, memOffset, &mem, hasSysmemSegment), ret, fail_locReg_memHandle);
  memset(memHandles, 0, numSegments * sizeof(*memHandles)); // symMemoryObtain took our reference
  globalSameMemOffset = mem->globalSameMemOffset;

  HGGCCHECKGOTO(hggcStreamCreateWithFlags(&stream, hggcStreamNonBlocking), ret, fail_locReg_memHandle_mem);

  NCCLCHECKGOTO(symWindowCreate(
      comm, mem, memOffset, userPtr, userSize, winFlags, localRegHandle, globalSameMemOffset, outWinDev, &winHost, stream
    ), ret, fail_locReg_memHandle_mem_stream);
  mem = nullptr; // symWindowCreate took our reference

  HGGCCHECKGOTO(hggcStreamSynchronize(stream), ret, fail_locReg_memHandle_mem_stream_win);

  // symWindowCreate needs barrier.
  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0xbeef), ret, fail_locReg_memHandle_mem_stream_win);

  {
    std::lock_guard<std::mutex> lock(ncclWindowMapMutex);
    // Set intrusive map fields directly on winHost
    winHost->comm = comm;
    winHost->next = nullptr;  // Initialize next pointer
    // Since windows are unique and belong to a single communicator,
    // it is not necessary to check if the insert would overwrite existing entries
    NCCLCHECKGOTO(ncclIntruAddressMapInsert(&ncclWindowMap, *outWinDev, winHost), ret, fail_locReg_memHandle_mem_stream_win);
    INFO(NCCL_ALLOC, "Inserted window %p into address map, ret=%d", *outWinDev, ret);
  }

  hggcStreamDestroy(stream);
  free(memHandles);
  return ret;

fail_locReg_memHandle_mem_stream_win:
  symWindowDestroy(comm, *outWinDev, stream);
  *outWinDev = nullptr;
  hggcStreamSynchronize(stream);
fail_locReg_memHandle_mem_stream:
  hggcStreamDestroy(stream);
fail_locReg_memHandle_mem:
  symMemoryDropRef(comm, mem);
fail_locReg_memHandle:
  for (int idx = 0; idx < numSegments; idx++) {
    if (memHandles[idx] != 0x0ULL) { HGCHECKIGNORE(hgMemRelease(memHandles[idx])); }
  }
  free(memHandles);
fail_locReg:
  ncclCommDeregister(comm, localRegHandle);
fail:
  *outWinDev = nullptr;
  return ret;
}

static ncclResult_t deepCopyDevCommRequirements(
    struct ncclDevCommRequirements const* src,
    struct ncclDevCommRequirements** dst
) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevResourceRequirements **dstRes;
  struct ncclTeamRequirements **dstTeam;

  NCCLCHECK(ncclCalloc(dst, 1));
  **dst = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;

  // Copy the entire struct now and update linked lists later.  Because of backwards compatibility, the source may
  // actually be smaller than the type would imply.
  memcpy(*dst, src, src->size);

  dstRes = &(*dst)->resourceRequirementsList;
  for (struct ncclDevResourceRequirements* rr = src->resourceRequirementsList; rr != nullptr; rr = rr->next) {
    NCCLCHECKGOTO(ncclCalloc(dstRes, 1), ret, fail);
    (*dstRes)->bufferSize = rr->bufferSize;
    (*dstRes)->bufferAlign = rr->bufferAlign;
    (*dstRes)->outBufferHandle = rr->outBufferHandle;
    dstRes = &(*dstRes)->next;
  }

  dstTeam = &(*dst)->teamRequirementsList;
  for (struct ncclTeamRequirements* tr = src->teamRequirementsList; tr != nullptr; tr = tr->next) {
    NCCLCHECKGOTO(ncclCalloc(dstTeam, 1), ret, fail);
    (*dstTeam)->team = tr->team;
    (*dstTeam)->multimem = tr->multimem;
    (*dstTeam)->outMultimemHandle = tr->outMultimemHandle;
    dstTeam = &(*dstTeam)->next;
  }

exit:
  return ret;
fail:
  freeDevCommRequirements(*dst);
  *dst = nullptr;
  goto exit;
}

void freeDevCommRequirements(
    struct ncclDevCommRequirements* reqs
) {
  if (reqs) {
    while (reqs->resourceRequirementsList) {
      struct ncclDevResourceRequirements* rr_next = reqs->resourceRequirementsList->next;
      free(reqs->resourceRequirementsList);
      reqs->resourceRequirementsList = rr_next;
    }

    while (reqs->teamRequirementsList) {
      struct ncclTeamRequirements* tr_next = reqs->teamRequirementsList->next;
      free(reqs->teamRequirementsList);
      reqs->teamRequirementsList = tr_next;
    }

    free(reqs);
  }
}

bool ncclGinResourcesRequested(struct ncclDevCommRequirements const* reqs) {
  bool requestedGinResources = reqs->ginSignalCount > 0 || reqs->ginCounterCount > 0 ||
                               reqs->barrierCount > 0 || reqs->railGinBarrierCount > 0 ||
                               reqs->worldGinBarrierCount > 0;

  struct ncclDevResourceRequirements* node = reqs->resourceRequirementsList;
  while (!requestedGinResources && node != nullptr) {
    requestedGinResources = node->ginSignalCount > 0 || node->ginCounterCount > 0;
    node = node->next;
  }

  return requestedGinResources;
}

#include "pccl_device/gin/gdaki/gin_gdaki_device_host_common.h"
static void ncclDevCommGdakiDump(void* handle) {
  struct ncclGinGdakiGPUContext ctx;
  if (hggcMemcpy(&ctx, handle, sizeof(ctx), hggcMemcpyDeviceToHost) == hggcSuccess) {
    printf("    GDAKI qp %p companion qp %p sink buffer lkey %x\n", ctx.gdqp, ctx.companion_gdqp, ctx.sink_buffer_lkey);
    printf("    GDAKI counters %p rkeys %p lkey %x offset %d\n", ctx.counters_table.buffer,
        ctx.counters_table.rkeys, ctx.counters_table.lkey, ctx.counters_table.offset);
    printf("    GDAKI signals  %p rkeys %p lkey %x offset %d\n", ctx.signals_table.buffer,
        ctx.signals_table.rkeys, ctx.signals_table.lkey, ctx.signals_table.offset);
  }
}

#include "pccl_device/gin/proxy/gin_proxy_device_host_common.h"
static void ncclDevCommProxyDump(void* handle) {
  ncclGinProxyGpuCtx_t ctx;
  if (hggcMemcpy(&ctx, handle, sizeof(ctx), hggcMemcpyDeviceToHost) == hggcSuccess) {
    printf("    PROXY nranks %d queue size %d queues %p\n", ctx.nranks, ctx.queueSize, ctx.queues);
    printf("    PROXY pis %p cis %p counters %p signals %p\n", ctx.pis, ctx.cis, ctx.counters, ctx.signals);
  }
}

void ncclDevCommDump(struct ncclDevComm* devComm) {
  printf("**** Dev Comm Dump %p ****\n", devComm);
  printf(" Rank %d/%d CPC32 %d\n", devComm->rank, devComm->nRanks, devComm->nRanks_rcp32);
  printf(" LSA Rank %d/%d CPC32 %d\n", devComm->lsaRank, devComm->lsaSize, devComm->lsaSize_rcp32);
  printf("\n");
  printf(" GIN\n");
  printf("  Mode %s\n", devComm->ginIsRailed ? "Rail" : "Full");
  printf("  Connections %d\n", devComm->ginConnectionCount);
  for (int c=0; c<devComm->ginConnectionCount; c++) {
    printf("   [%d] %d %p\n", c, devComm->ginNetDeviceTypes[c], devComm->ginHandles[c]);
    if (devComm->ginNetDeviceTypes[c] == NCCL_GIN_TYPE_GDAKI) ncclDevCommGdakiDump(devComm->ginHandles[c]);
    if (devComm->ginNetDeviceTypes[c] == NCCL_GIN_TYPE_PROXY) ncclDevCommProxyDump(devComm->ginHandles[c]);
  }
  printf("  Signals  %d shadows %p\n", devComm->ginSignalCount, devComm->ginSignalShadows);
  printf("  Contexts %d\n", devComm->ginContextCount);
  printf("\n");
  printf(" Abort flag %p\n", devComm->abortFlag);
  printf(" LSA Barriers count %d handle %d\n", devComm->lsaBarrier.nBarriers, devComm->lsaBarrier.bufHandle);
  printf(" Hybrid Barriers count %d LSA handle %d GIN Rail Barrier signal0 %d\n", devComm->hybridLsaBarrier.nBarriers,
         devComm->hybridLsaBarrier.bufHandle, devComm->hybridRailGinBarrier.signal0);
  printf(" GIN Rail Barrier signal0 %d\n", devComm->railGinBarrier.signal0);
  printf(" GIN World Barrier signal0 %d\n", devComm->worldGinBarrier.signal0);
}

ncclResult_t ncclDevrCommCreateInternal(
    struct pcclComm* comm,
    struct ncclDevCommRequirements* reqs, struct ncclDevComm* outDevComm, bool isInternal
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  struct ncclTeam world = ncclTeamWorld(comm);
  struct ncclTeam lsa = ncclTeamInnerFactor(world, devr->lsaSize);
  bool ginActivated = false;
  struct ncclDevrTeam* tmLsa;
  size_t bufSizeTotal;
  int nGinContexts = reqs->ginContextCount;
  int ginSignalTotal = 0, ginCounterTotal = 0;
  struct ncclDevResourceRequirements* resReqsHead = reqs->resourceRequirementsList;
  struct ncclDevResourceRequirements lsaBarReq;
  struct ncclDevResourceRequirements hybridLsaBarrierReq;
  hggcStream_t stream = nullptr;
  struct ncclDevResourceRequirements railGinBarrierReq;
  struct ncclDevResourceRequirements hybridRailGinBarrierReq;
  struct ncclDevResourceRequirements worldGinBarrierReq;
  HGmemGenericAllocationHandle memHandle = 0x0;
  struct ncclDevrMemory* mem = nullptr;
  struct ncclDevrWindow* win = nullptr;
  struct ncclWindow_vidmem* winHost = nullptr;
  size_t ginSignalShadowsOffset = 0;

  // This function always operates on the current version of the ncclDevResourceRequirements structure, thanks
  // to the deepCopyDevCommRequirements() function, so version checks are not needed.  The data in reqs can also
  // be changed by this function, which is again OK thanks to the fact that it's not a user-supplied structure.
  ncclGinConnectionType_t requestedConnectionType = reqs->ginConnectionType;

  if (reqs->ginForceEnable) {
    INFO(NCCL_INIT,
         "ginForceEnable set to true, defaulting ginConnectionType to PCCL_GIN_CONNECTION_FULL");
    INFO(NCCL_INIT,
         "ginForceEnable is being deprecated in favor of explicitly setting ginConnectionType!");
    requestedConnectionType = NCCL_GIN_CONNECTION_FULL;
  }

  bool requestedGinResources = ncclGinResourcesRequested(reqs);
  if (requestedGinResources && requestedConnectionType == NCCL_GIN_CONNECTION_NONE) {
    WARN("User requested GIN resources but did not request GIN to be enabled!");
    return ncclInvalidArgument;
  }

  if (requestedConnectionType != NCCL_GIN_CONNECTION_NONE) {
    if (comm->globalGinSupport == NCCL_GIN_CONNECTION_NONE) {
      WARN("User requested GIN but not all ranks in the communicator support GIN");
      return ncclInvalidArgument;
    }
    if (requestedConnectionType == NCCL_GIN_CONNECTION_FULL) {
      if (comm->globalGinSupport == NCCL_GIN_CONNECTION_RAIL) {
        WARN("User requested GIN connection type PCCL_GIN_CONNECTION_FULL but the communicator supports only PCCL_GIN_CONNECTION_RAIL");
        return ncclInvalidArgument;
      }
    }

    ginActivated = !devr->ginEnabled;
    devr->ginEnabled = true;
  }

  if (ginActivated) {
    NCCLCHECKGOTO(ncclGinConnectOnce(comm), ret, fail);
    // Register all preexisting memories with GIN. Update the windows later when
    // we have a stream.
    for (struct ncclDevrMemory* mem = devr->memHead; mem != nullptr; mem = mem->next) {
      NCCLCHECKGOTO(symMemoryRegisterGin(comm, mem), ret, fail);
    }
  }

  memset(outDevComm, 0, sizeof(*outDevComm));
  outDevComm->magic = PCCL_API_MAGIC;
  outDevComm->version = reqs->version;
  outDevComm->rank = comm->rank;
  outDevComm->nRanks = comm->nRanks;
  outDevComm->nRanks_rcp32 = idivRcp32(comm->nRanks);
  outDevComm->lsaRank = devr->lsaSelf;
  outDevComm->lsaSize = devr->lsaSize;
  outDevComm->lsaSize_rcp32 = idivRcp32(devr->lsaSize);
  outDevComm->ginIsRailed = comm->sharedRes->ginState.ginConnectionType == NCCL_GIN_CONNECTION_RAIL; // false if FULL or NONE
  outDevComm->lsaMultimem.mcBasePtr = nullptr; // PPU does not support multimem
  if (isInternal) outDevComm->abortFlag = comm->abortFlagDev;

  // PPU does not support multimem; error if requested
  if (reqs->lsaMultimem) {
    WARN("lsaMultimem is not supported on PPU");
    return ncclInvalidArgument;
  }
  for (struct ncclTeamRequirements* tr = reqs->teamRequirementsList; tr != nullptr; tr = tr->next) {
    if (tr->multimem) {
      WARN("Team multimem is not supported on PPU");
      return ncclInvalidArgument;
    }
  }

  NCCLCHECKGOTO(symTeamObtain(comm, lsa, &tmLsa), ret, fail);

  resReqsHead = reqs->resourceRequirementsList;

  // Initialize resources for the hybrid barrier
  ncclLsaBarrierCreateRequirement(lsa, reqs->barrierCount, &outDevComm->hybridLsaBarrier, &hybridLsaBarrierReq);
  hybridLsaBarrierReq.next = resReqsHead;
  ncclGinBarrierCreateRequirement(comm, ncclTeamRail(comm), reqs->barrierCount, &outDevComm->hybridRailGinBarrier, &hybridRailGinBarrierReq);
  hybridRailGinBarrierReq.next = &hybridLsaBarrierReq;
  resReqsHead = &hybridRailGinBarrierReq;

  ncclLsaBarrierCreateRequirement(lsa, reqs->lsaBarrierCount, &outDevComm->lsaBarrier, &lsaBarReq);
  lsaBarReq.next = resReqsHead;
  resReqsHead = &lsaBarReq;

  ncclGinBarrierCreateRequirement(comm, ncclTeamRail(comm), reqs->railGinBarrierCount, &outDevComm->railGinBarrier, &railGinBarrierReq);
  railGinBarrierReq.next = resReqsHead;
  resReqsHead = &railGinBarrierReq;

  ncclGinBarrierCreateRequirement(comm, ncclTeamWorld(comm), reqs->worldGinBarrierCount, &outDevComm->worldGinBarrier, &worldGinBarrierReq);
  worldGinBarrierReq.next = resReqsHead;
  resReqsHead = &worldGinBarrierReq;

  { struct ncclDevResourceRequirements* rr = resReqsHead;
    bufSizeTotal = 0;
    ginSignalTotal = reqs->ginSignalCount;
    ginCounterTotal = reqs->ginCounterCount;
    while (rr != nullptr) {
      bufSizeTotal = alignUp(bufSizeTotal, std::max<size_t>(128, rr->bufferAlign));
      if (rr->outBufferHandle != nullptr) *rr->outBufferHandle = bufSizeTotal/128;
      if (rr->outGinSignalStart != nullptr) *rr->outGinSignalStart = ginSignalTotal;
      if (rr->outGinCounterStart != nullptr) *rr->outGinCounterStart = ginCounterTotal;
      bufSizeTotal += rr->bufferSize;
      ginSignalTotal += rr->ginSignalCount;
      ginCounterTotal += rr->ginCounterCount;
      rr = rr->next;
    }
    bufSizeTotal= alignUp(bufSizeTotal, 128);
    ginSignalShadowsOffset = bufSizeTotal;
    bufSizeTotal += nGinContexts * ginSignalTotal * sizeof(uint64_t); // include signal shadows
    bufSizeTotal = alignUp(bufSizeTotal, devr->granularity);
  }

  if (devr->ginEnabled) {
    reqs->ginSignalCount = ginSignalTotal;
    reqs->ginCounterCount = ginCounterTotal;
    NCCLCHECK(ncclGinDevCommSetup(comm, reqs, outDevComm));
  }

  HGGCCHECKGOTO(hggcStreamCreateWithFlags(&stream, hggcStreamNonBlocking), ret, fail);

  if (ginActivated) {
    // Now update the GIN handles in all existing windows. Registration of memories happened above.
    for (int i=0; i < devr->winSortedCount; i++) {
      struct ncclDevrWindow* win = devr->winSorted[i].win;
      struct ncclWindow_vidmem* winHost;
      NCCLCHECKGOTO(ncclShadowPoolToHost(&devr->shadows, win->vidmem, &winHost), ret, fail_stream);
      winHost->ginOffset4K = (win->bigOffset - win->memory->bigOffset)>>12;
      for (int c=0; c < NCCL_GIN_MAX_CONNECTIONS; c++) {
        winHost->ginWins[c] = win->memory->ginDevWins[c];
      }
      winHost->numSegments = win->memory->numGinSegments;

      if (win->memory->numGinSegments > 1) {
        // When a window is created, numGinSegments is always set to `1`.  As we now
        // know that there are multiple segments, we need to reallocate ginMultiSegmentWins

        struct ncclSegmentWindow* segmentWindowsDev;
        struct ncclSegmentWindow* segmentWindowsHost;
        NCCLCHECKGOTO(ncclShadowPoolFree(&devr->shadows, winHost->ginMultiSegmentWins, stream), ret, fail_stream);
        NCCLCHECKGOTO(allocAndPopulateSegmentWindows(devr, win->memory, stream,
            &segmentWindowsDev, &segmentWindowsHost), ret, fail_stream);
        winHost->ginMultiSegmentWins = segmentWindowsDev;
      }
      HGGCCHECKGOTO(hggcMemcpyAsync(win->vidmem, winHost, sizeof(struct ncclWindow_vidmem), hggcMemcpyHostToDevice, stream), ret, fail_stream);
    }
  }

  NCCLCHECKGOTO(symWindowTableInitOnce(comm, stream), ret, fail_stream); // ensure devr->windowTable exists
  outDevComm->windowTable = devr->windowTable;

  if (bufSizeTotal == 0) {
    outDevComm->resourceWindow = nullptr;
    outDevComm->resourceWindow_inlined = {};
  } else {
    HGmemAllocationProp memProp = {};
    memProp.type = HG_MEM_ALLOCATION_TYPE_PINNED;
    memProp.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
    memProp.requestedHandleTypes = ncclHgMemHandleType;
    memProp.allocFlags.gpuDirectRDMACapable = comm->sharedRes->ginState.ncclGin != nullptr ? 1 : 0;
    memProp.location.id = comm->hggcDev;

    HGCHECKGOTO(hgMemCreate(&memHandle, bufSizeTotal, &memProp, 0), ret, fail_stream);

    NCCLCHECKGOTO(symMemoryObtain(comm, &memHandle, 1, NULL, bufSizeTotal, /*winFlags=*/0, /*memOffset=*/0, &mem), ret, fail_stream_mem);
    memHandle = 0x0; // Reference given to symMemoryObtain

    NCCLCHECKGOTO(symWindowCreate( // Requires world barrier afterward.
      comm, mem, /*memOffset=*/0, nullptr, bufSizeTotal, /*winFlags=*/0,
      /*localReg=*/nullptr, /*globalSameMemOffset=*/true, &outDevComm->resourceWindow, &win,
      stream), ret, fail_stream_mem);
    mem = nullptr; // Reference given to symWindowCreate
    NCCLCHECKGOTO(ncclShadowPoolToHost(&devr->shadows, win->vidmem, &winHost), ret, fail_stream_mem_win);
    outDevComm->resourceWindow_inlined.lsaFlatBase = winHost->lsaFlatBase;
    outDevComm->resourceWindow_inlined.stride4G = winHost->stride4G;
    outDevComm->ginSignalShadows = (uint64_t*)add4G((char*)winHost->lsaFlatBase + ginSignalShadowsOffset, winHost->lsaRank*winHost->stride4G);

    HGGCCHECKGOTO(hggcMemsetAsync(win->userPtr, 0, bufSizeTotal, stream), ret, fail_stream_mem_win);
  }

  HGGCCHECKGOTO(hggcStreamSynchronize(stream), ret, fail_stream_mem_win);

  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0xbeef), ret, fail_stream_mem_win);
  HGGCCHECKGOTO(hggcStreamDestroy(stream), ret, fail_stream_mem_win);

  //ncclDevCommDump(outDevComm);
  return ret;

fail_stream_mem_win:
  symWindowDestroy(comm, win->vidmem, stream);
  hggcStreamSynchronize(stream);
fail_stream_mem:
  if (memHandle != 0x0) { HGCHECKIGNORE(hgMemRelease(memHandle)); }
  symMemoryDropRef(comm, mem);
fail_stream:
  hggcStreamDestroy(stream);
fail:
  return ret;
}

////////////////////////////////////////////////////////////////////////////////

NCCL_API(ncclResult_t, pcclCommWindowRegister, ncclComm_t comm, void* buff, size_t size, ncclWindow_t* win, int winFlags);
ncclResult_t pcclCommWindowRegister(ncclComm_t comm, void* buff, size_t size, ncclWindow_t* win, int winFlags) {
  NCCLCHECK(CommCheck(comm, __func__, "comm"));
  NCCLCHECK(PtrCheck(win, __func__, "win"));
  *win = nullptr;
  if (buff == nullptr || size <= 0) {
    WARN("%s: invalid pointer %p / size %zu", __func__, buff, size);
    return ncclInvalidArgument;
  }

  if (!comm->symmetricSupport) {
    return ncclSuccess;
  }

  ncclResult_t ret = ncclSuccess;
  int saveDev;
  struct ncclDevrRegTask* task;

  HGGCCHECK(hggcGetDevice(&saveDev));
  NCCLCHECK(ncclGroupStartInternal());

  NCCLCHECKGOTO(ncclCommEnsureReady(comm), ret, fail);
  HGGCCHECKGOTO(hggcSetDevice(comm->hggcDev), ret, fail);

  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);

  NCCLCHECKGOTO(ncclCalloc(&task, 1), ret, fail);
  task->userPtr = buff;
  task->userSize = size;
  task->winFlags = winFlags;
  task->outWinDev = win;
  ncclIntruQueueEnqueue(&comm->devrState.regTaskQueue, task);
  ncclGroupCommJoin(comm, ncclGroupTaskTypeSymRegister);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  hggcSetDevice(saveDev);
  return ret;
fail:
  goto exit;
}

NCCL_API(ncclResult_t, pcclCommWindowDeregister, ncclComm_t comm, ncclWindow_t win);
ncclResult_t pcclCommWindowDeregister(struct pcclComm* comm, struct ncclWindow_vidmem* winDev) {
  NCCLCHECK(CommCheck(comm, __func__, "comm"));
  ncclResult_t ret = ncclSuccess;
  int saveDev;
  hggcStream_t stream;

  if (winDev == nullptr) goto exit;

  HGGCCHECKGOTO(hggcGetDevice(&saveDev), ret, fail);
  HGGCCHECKGOTO(hggcSetDevice(comm->hggcDev), ret, fail);
  HGGCCHECKGOTO(hggcStreamCreateWithFlags(&stream, hggcStreamNonBlocking), ret, fail_dev);
  NCCLCHECKGOTO(symWindowDestroy(comm, winDev, stream), ret, fail_dev_stream);
fail_dev_stream:
  hggcStreamSynchronize(stream);
  hggcStreamDestroy(stream);
fail_dev:
  hggcSetDevice(saveDev);
fail:
exit:
  return ret;
}

ncclResult_t ncclDevrFindWindow(
    struct pcclComm* comm, void const* userPtr, struct ncclDevrWindow** outWin
  ) {
  struct ncclDevrState* devr = &comm->devrState;
  uintptr_t userAddr = reinterpret_cast<uintptr_t>(userPtr);
  int i = listFindSortedLub(&ncclDevrWindowSorted::userAddr, devr->winSorted, devr->winSortedCount, userAddr);
  if (0 < i && (userAddr - devr->winSorted[i-1].userAddr < devr->winSorted[i-1].size)) {
    *outWin = devr->winSorted[i-1].win;
  } else {
    *outWin = nullptr;
  }
  return ncclSuccess;
}

bool ncclDevrWindowIsMultiSegment(struct ncclDevrWindow* win) {
  return win != NULL && win->memory->maxGlobalNumSegments > 1;
}

bool ncclDevrWindowHasSysmemSegment(struct ncclDevrWindow* win) {
  return win != NULL && win->memory->globalHasSysmemSegment;
}

void ncclDevCommCopyLsaData(void* dstRankPtr, void const* srcRankPtr) {
  memcpy(dstRankPtr, srcRankPtr, offsetof(struct ncclDevComm, railGinBarrier) - offsetof(struct ncclDevComm, rank));
}

NCCL_API(ncclResult_t, pcclCommQueryProperties, ncclComm_t, pcclCommProperties_t*);
ncclResult_t pcclCommQueryProperties(ncclComm_t comm, ncclCommProperties_t* props) {
  NCCLCHECK(CommCheck(comm, __func__, "comm"));
  NCCLCHECK(PtrCheck(props, __func__, "props"));

  NCCLCHECK(ncclCommEnsureReady(comm));

  if (props->magic != PCCL_API_MAGIC) {
    WARN("Cannot get communicator properties: pcclCommProperties_t argument must be initialized via PCCL_COMM_PROPERTIES_INITIALIZER");
    return ncclInvalidUsage;
  }

  props->rank = comm->rank;
  props->nRanks = comm->nRanks;
  props->hggcDev = comm->hggcDev;
  props->hgmlDev = comm->hgmlDev;
  props->deviceApiSupport = comm->symmetricSupport;
  props->multimemSupport = false; // PPU does not support multimem/NVLS
  props->hostRmaSupport = false;  // PPU does not support host RMA
  NCCLCHECK(getGlobalGinType(comm, &props->ginType));
  NCCLCHECK(getGlobalRailedGinType(comm, &props->railedGinType));

  // Preferring to call ncclDevrInitOnce directly instead to calling ncclTeam* functions because
  // we can propagate the result of ncclDevrInitOnce back to the caller.
  NCCLCHECK(ncclDevrInitOnce(comm));
  props->nLsaTeams = comm->devrState.nLsaTeams;

  return ncclSuccess;
}

NCCL_API(ncclResult_t, pcclDevCommCreate, ncclComm_t comm, pcclDevCommRequirements_t const* reqs, pcclDevComm_t* outDevComm);
ncclResult_t ncclDevCommCreate(
    ncclComm_t comm, struct ncclDevCommRequirements const* reqs,
    struct ncclDevComm* outDevComm
  ) {
  NCCLCHECK(CommCheck(comm, __func__, "comm"));
  NCCLCHECK(PtrCheck(reqs, __func__, "reqs"));
  if (reqs->magic != PCCL_API_MAGIC) {
    WARN("Cannot create device communicator: pcclDevCommRequirements_t argument must be initialized via PCCL_DEV_COMM_REQUIREMENTS_INITIALIZER");
    return ncclInvalidUsage;
  }

  ncclResult_t ret = ncclSuccess;
  int saveDev;
  struct ncclDevrCommCreateTask* task = nullptr;

  HGGCCHECK(hggcGetDevice(&saveDev));
  NCCLCHECK(ncclGroupStartInternal());

  if (!comm->symmetricSupport) {
    WARN("Communicator does not support symmetric memory!");
    ret = ncclInvalidUsage;
    goto fail;
  }

  NCCLCHECKGOTO(ncclCommEnsureReady(comm), ret, fail);
  HGGCCHECKGOTO(hggcSetDevice(comm->hggcDev), ret, fail);

  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);

  NCCLCHECKGOTO(ncclCalloc(&task, 1), ret, fail);
  // reqs must be deep copied to the task so background threads can safely access it
  NCCLCHECKGOTO(deepCopyDevCommRequirements(reqs, &task->reqs), ret, fail);
  task->outDevComm = outDevComm;
  ncclIntruQueueEnqueue(&comm->devrState.commCreateTaskQueue, task);
  ncclGroupCommJoin(comm, ncclGroupTaskTypeSymRegister);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  hggcSetDevice(saveDev);
  return ret;
fail:
  free(task);
  goto exit;
}

NCCL_API(ncclResult_t, pcclDevCommDestroy, ncclComm_t comm, pcclDevComm_t const* devComm);
ncclResult_t ncclDevCommDestroy(
    struct pcclComm* comm, struct ncclDevComm const* devComm
  ) {
  NCCLCHECK(CommCheck(comm, __func__, "comm"));
  NCCLCHECK(PtrCheck(devComm, __func__, "devComm"));
  int saveDev;
  ncclResult_t ret = ncclSuccess;

  HGGCCHECK(hggcGetDevice(&saveDev));
  HGGCCHECK(hggcSetDevice(comm->hggcDev));

  if (devComm->resourceWindow != nullptr) {
    NCCLCHECKGOTO(ncclCommWindowDeregister(comm, devComm->resourceWindow), ret, end);
  }
  if (devComm->ginContextCount) {
    NCCLCHECKGOTO(ncclGinDevCommFree(comm, devComm), ret, end);
  }

end:
  hggcSetDevice(saveDev);
  return ret;
}

NCCL_API(ncclResult_t, pcclWinGetUserPtr, ncclComm_t comm, ncclWindow_t win, void** outUserPtr);
ncclResult_t pcclWinGetUserPtr(struct pcclComm* comm, struct ncclWindow_vidmem* win, void** outUserPtr) {
  NCCLCHECK(CommCheck(comm, __func__, "comm"));
  NCCLCHECK(PtrCheck(outUserPtr, __func__, "outUserPtr"));

  if (!comm->symmetricSupport) {
    INFO(NCCL_INIT, "Symmetric registration is not supported in this communicator.");
    *outUserPtr = nullptr;
    return ncclSuccess;
  }

  NCCLCHECK(PtrCheck(win, __func__, "win"));

  struct ncclDevrWindow* winHost = nullptr;
  struct ncclWindow_vidmem* winDevHost = nullptr;
  NCCLCHECK(ncclShadowPoolToHost(&comm->devrState.shadows, win, &winDevHost));

  winHost = (struct ncclDevrWindow*)winDevHost->winHost;
  if (winHost == nullptr) {
    WARN("window has a NULL user pointer");
    return ncclInternalError;
  }

  *outUserPtr = winHost->userPtr;
  return ncclSuccess;
}

ncclResult_t ncclDevrWorldToLsaRank(struct pcclComm* comm, int peerWorldRank, int* peerLsaRank) {
  ncclTeam_t worldTeam = ncclTeamWorld(comm);
  ncclTeam_t lsaTeam = ncclTeamLsa(comm);
  if (!ncclTeamRankIsMember(lsaTeam, worldTeam, peerWorldRank)) {
    WARN("pcclDevrWorldToLsaRank: world rank %d is not a member of the LSA team", peerWorldRank);
    return ncclInternalError;
  }
  *peerLsaRank = ncclTeamRankToTeam(lsaTeam, worldTeam, peerWorldRank);
  return ncclSuccess;
}

// Get the corresponding pointer in another lsa rank's symmetric memory window
ncclResult_t ncclDevrGetLsaRankPtr(struct pcclComm* comm, struct ncclDevrWindow* winHost, size_t offset, int lsaRank, void** outPtr) {
  NCCLCHECK(CommCheck(comm, __func__, "comm"));
  NCCLCHECK(PtrCheck(outPtr, __func__, "outPtr"));

  struct ncclDevrState* devr = &comm->devrState;

  // Validate lsaRank is within bounds
  if (lsaRank < 0 || lsaRank >= devr->lsaSize) {
    return ncclInvalidArgument;
  }

  // Validate offset is within bounds
  if (offset < 0 || offset >= winHost->size) {
    return ncclInvalidArgument;
  }

  // Calculate the address with offset for the specified lsa rank
  *outPtr = (void*)((uintptr_t)devr->lsaFlatBase + lsaRank * devr->bigSize + winHost->bigOffset + offset);
  return ncclSuccess;
}


static ncclResult_t findCommAndHostWindowFromDeviceWindow(ncclWindow_t devWindow, ncclComm_t* foundComm, ncclDevrWindow** hostWindow) {
  struct ncclDevrWindow* winHost = nullptr;
  std::lock_guard<std::mutex> lock(ncclWindowMapMutex);
  NCCLCHECK(ncclIntruAddressMapFind(&ncclWindowMap, devWindow, &winHost));
  if (winHost == nullptr) {
    WARN("Could not find communicator matching window %p (map hbits=%d count=%d)",
         devWindow, ncclWindowMap.base.hbits, ncclWindowMap.base.count);
    return ncclInvalidArgument;
  }

  *foundComm = winHost->comm;
  *hostWindow = winHost;

  return ncclSuccess;
}

NCCL_API(ncclResult_t, pcclGetLsaDevicePointer, ncclWindow_t window, size_t offset, int lsaRank, void** outPtr);
ncclResult_t pcclGetLsaDevicePointer(ncclWindow_t window, size_t offset, int lsaRank, void** outPtr) {
  NCCLCHECK(PtrCheck(window, __func__, "window"));
  NCCLCHECK(PtrCheck(outPtr, __func__, "outPtr"));

  ncclComm_t comm = nullptr;
  struct ncclDevrState* devr;
  struct ncclDevrWindow* winHost = nullptr;

  // Get the host version of the device window
  NCCLCHECK(findCommAndHostWindowFromDeviceWindow(window, &comm, &winHost));

  devr = &comm->devrState;
  if (lsaRank < 0 || lsaRank >= devr->lsaSize) {
    WARN("The provided lsaRank %d is not in the valid lsaSize of [0,%d] for the provided window %p.",
         lsaRank, devr->lsaSize, window);
    return ncclInvalidArgument; // In this case the user should know what the lsa size is.
  }

  NCCLCHECK(ncclDevrGetLsaRankPtr(comm, winHost, offset, lsaRank, outPtr));

  return ncclSuccess;
}

NCCL_API(ncclResult_t, pcclGetPeerDevicePointer, ncclWindow_t window, size_t offset, int peer, void** outPtr);
ncclResult_t pcclGetPeerDevicePointer(ncclWindow_t window, size_t offset, int peer, void** outPtr) {
  NCCLCHECK(PtrCheck(window, __func__, "window"));
  NCCLCHECK(PtrCheck(outPtr, __func__, "outPtr"));

  ncclComm_t comm = nullptr;
  struct ncclDevrState* devr;
  struct ncclDevrWindow* winHost = nullptr;
  int lsaRank;
  ncclTeam_t worldTeam;
  ncclTeam_t lsaTeam;

  // Get the host version of the device window
  NCCLCHECK(findCommAndHostWindowFromDeviceWindow(window, &comm, &winHost));
  // Validate peer rank is within bounds
  if (peer < 0 || peer >= comm->nRanks) {
    WARN("peer %d is not within valid range of ranks %d.", peer, comm->nRanks);
    return ncclInvalidArgument;
  }

  devr = &comm->devrState;
  worldTeam = ncclTeamWorld(comm);
  lsaTeam = ncclTeamLsa(comm);

  // Convert world rank to LSA team rank
  lsaRank = ncclTeamRankToTeam(lsaTeam, worldTeam, peer);

  // Validate the converted LSA rank is within bounds
  if (lsaRank < 0 || lsaRank >= devr->lsaSize) {
    // We return a nullptr if peer is not reachable. Same as device side
    *outPtr = nullptr;
    return ncclSuccess;
  }

  NCCLCHECK(ncclDevrGetLsaRankPtr(comm, winHost, offset, lsaRank, outPtr));

  return ncclSuccess;
}
////////////////////////////////////////////////////////////////////////////////

// Multimem API stubs - PPU does not support multimem/NVLS
NCCL_API(ncclResult_t, pcclGetMultimemDevicePointer, ncclWindow_t window, size_t offset, pcclMultimemHandle_t multimem, void** outPtr);
ncclResult_t pcclGetMultimemDevicePointer(ncclWindow_t window, size_t offset, ncclMultimemHandle_t multimem, void** outPtr) {
  WARN("pcclGetMultimemDevicePointer is not supported on PPU");
  return ncclInvalidUsage;
}

NCCL_API(ncclResult_t, pcclGetLsaMultimemDevicePointer, ncclWindow_t window, size_t offset, void** outPtr);
ncclResult_t pcclGetLsaMultimemDevicePointer(ncclWindow_t window, size_t offset, void** outPtr) {
  WARN("pcclGetLsaMultimemDevicePointer is not supported on PPU");
  return ncclInvalidUsage;
}

////////////////////////////////////////////////////////////////////////////////

// Find the least index strictly greater than arg.
template<typename Obj, typename Key>
static int listFindSortedLub(Key Obj::*key, Obj* sorted, int count, Key arg) {
  int lo = 0, hi = count;
  while (lo + 16 < hi) {
    int i = (lo + hi)/2;
    if (sorted[i].*key <= arg) lo = i+1;
    else hi = i;
  }
  int i = lo;
  while (i < hi && sorted[i].*key <= arg) i++;
  return i;
}

template<typename Obj>
static void listInsert(Obj** list, int* capacity, int* count, int index, Obj val) {
  if (*capacity < *count + 1) {
    *capacity *= 2;
    if (*capacity == 0) *capacity = 16;
    *list = (Obj*)realloc(*list, (*capacity)*sizeof(Obj));
  }
  for (int j = *count; j != index; j--) {
    (*list)[j] = (*list)[j-1];
  }
  (*list)[index] = val;
  *count += 1;
}

template<typename Obj>
static void listRemove(Obj* list, int* count, int index) {
  for (int i = index; i+1 < *count; i++) {
    list[i] = list[i+1];
  }
  *count -= 1;
}
