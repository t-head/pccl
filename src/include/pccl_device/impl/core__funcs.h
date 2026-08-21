/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_CORE__FUNCS_H_
#define _PCCL_DEVICE_CORE__FUNCS_H_
#include "core__types.h"
#include "comm__types.h"
#include "ptr__types.h"
#include <cassert>

#if __cplusplus
PCCL_HOST_DEVICE_INLINE pcclTeam pcclTeamWorld(pcclDevComm const &comm) {
  pcclTeam ans;
  ans.nRanks = comm.nRanks;
  ans.rank = comm.rank;
  ans.stride = 1;
  return ans;
}
#endif

#if __cplusplus
PCCL_HOST_DEVICE_INLINE pcclTeam pcclTeamLsa(pcclDevComm const &comm) {
  pcclTeam ans;
  ans.nRanks = comm.lsaSize;
  ans.rank = comm.lsaRank;
  ans.stride = 1;
  return ans;
}
#endif

#if __cplusplus
PCCL_HOST_DEVICE_INLINE pcclTeam pcclTeamRail(pcclDevComm const& comm) {
  pcclTeam ans;
  ans.nRanks = pccl::utility::idivFast32(comm.nRanks, comm.lsaSize, comm.lsaSize_rcp32);
  ans.rank = pccl::utility::idivFast32(comm.rank, comm.lsaSize, comm.lsaSize_rcp32);
  ans.stride = comm.lsaSize;
  return ans;
}
#endif

PCCL_HOST_DEVICE_INLINE bool pcclTeamRankIsMember(pcclTeam_t a, pcclTeam_t b, int brank) {
  int wrank = (brank - b.rank)*b.stride;
  uint32_t adelta = wrank/a.stride;
  uint32_t amod = wrank%a.stride;
  int arank = a.rank + adelta;
  return 0 <= arank && arank < a.nRanks && amod == 0;
}

PCCL_HOST_DEVICE_INLINE int pcclTeamRankToTeam(pcclTeam_t a, pcclTeam_t b, int brank) {
  int wrank = (brank - b.rank)*b.stride;
  uint32_t adelta = wrank/a.stride;
  //uint32_t amod = wrank%a.stride;
  int arank = a.rank + adelta;
  return arank;
}

#if __cplusplus
PCCL_HOST_DEVICE_INLINE int pcclTeamRankToWorld(pcclDevComm const& comm, pcclTeam tm, int rank) {
  return comm.rank + (rank - tm.rank)*tm.stride;
}
#endif

#if __cplusplus
PCCL_HOST_DEVICE_INLINE int pcclTeamRankToLsa(pcclDevComm const& comm, pcclTeam tm, int rank) {
  return comm.lsaRank + (rank - tm.rank)*tm.stride;
}
#endif

PCCL_HOST_DEVICE_INLINE pcclTeam_t pcclTeamInnerFactor(pcclTeam_t parent, int innerSize) {
  pcclTeam_t ans;
  ans.nRanks = innerSize;
  ans.rank = parent.rank%innerSize;
  ans.stride = parent.stride;
  return ans;
}

PCCL_HOST_DEVICE_INLINE pcclTeam_t pcclTeamOuterFactor(pcclTeam_t parent, int innerSize) {
  pcclTeam_t ans;
  ans.nRanks = parent.nRanks/innerSize;
  ans.rank = parent.rank/innerSize;
  ans.stride = parent.stride*innerSize;
  return ans;
}

PCCL_HOST_DEVICE_INLINE int pcclTeamRankInDifference(pcclTeam_t parent, pcclTeam_t subset, int index) {
  int stride = subset.stride/parent.stride;
  int below = parent.rank - subset.rank*stride;
  if (stride < 0) {
    stride = -stride;
    below -= (subset.nRanks-1)*stride;
  }
  if (index < below) {
    return index;
  } else if (index-below < (subset.nRanks-1)*(stride-1)) {
    return below + 1 + ((index-below)/(stride-1))*stride + (index-below)%(stride-1);
  } else {
    return below + 1 + (subset.nRanks-1)*stride + (index - below - (subset.nRanks-1)*(stride-1));
  }
}

#if PCCL_CHECK_HGGCCC
PCCL_DEVICE_INLINE void* pcclGetLocalPointer(pcclWindow_t w, size_t offset) {
  char* base = pccl::utility::loadConst(&w->lsaFlatBase);
  uint32_t stride4G = pccl::utility::loadConst(&w->stride4G);
  int i = pccl::utility::loadConst(&w->lsaRank);
  return (void*)(pccl::utility::add4G(base, i*stride4G) + offset);
}

PCCL_DEVICE_INLINE void* pcclGetLsaPointer(pcclWindow_t w, size_t offset, int peer) {
  char* base = pccl::utility::loadConst(&w->lsaFlatBase);
  uint32_t stride4G = pccl::utility::loadConst(&w->stride4G);
  int i = peer;
  return (void*)(pccl::utility::add4G(base, i*stride4G) + offset);
}

PCCL_DEVICE_INLINE void* pcclGetPeerPointer(pcclWindow_t w, size_t offset, int peer) {
  char* base = pccl::utility::loadConst(&w->lsaFlatBase);
  uint32_t stride4G = pccl::utility::loadConst(&w->stride4G);
  int worldRank = pccl::utility::loadConst(&w->worldRank);
  int lsaRank = pccl::utility::loadConst(&w->lsaRank);
  int i = lsaRank + (peer - worldRank);
  return (void*)(pccl::utility::add4G(base, i*stride4G) + offset);
}

PCCL_DEVICE_INLINE void* pcclGetPeerPointer(pcclWindow_t w, size_t offset, pcclTeam tm, int peer) {
  char* base = pccl::utility::loadConst(&w->lsaFlatBase);
  uint32_t stride4G = pccl::utility::loadConst(&w->stride4G);
  int lsaRank = pccl::utility::loadConst(&w->lsaRank);
  int i = lsaRank + (peer - tm.rank)*tm.stride;
  return (void*)(pccl::utility::add4G(base, i*stride4G) + offset);
}

PCCL_DEVICE_INLINE void* pcclGetMultimemPointer(pcclWindow_t w, size_t offset, pcclMultimemHandle mm) {
  assert(false && "multimem is not supported on PPU");
  void* ptr = mm.mcBasePtr;
  ptr = reinterpret_cast<char(*)[4096]>(ptr) + pccl::utility::loadConst(&w->mcOffset4K);
  return (void*)((char*)ptr + offset);
}

PCCL_DEVICE_INLINE void* pcclGetLsaMultimemPointer(pcclWindow_t w, size_t offset, pcclDevComm const& comm) {
  assert(false && "multimem is not supported on PPU");
  return pcclGetMultimemPointer(w, offset, comm.lsaMultimem);
}

template<typename Coop>
PCCL_DEVICE_INLINE pcclWindow_t pcclFindWindow(Coop coop, pcclDevComm const& comm, void const *ptr) {
  using pccl::utility::loadConst;
  auto coalesced = pcclCoopCoalesced(coop);
  pcclDevCommWindowTable* t = comm.windowTable;
  while (true) {
    bool found = false;
    int index = coalesced.thread_rank();
    #pragma unroll 1
    while (index < 32) {
      uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
      pcclDevCommWindowTable::Entry e = loadConst(&t->entries[index]);
      if ((e.base != 0) && (e.size != 0) && (e.window != 0)) {
        if (uptr - uintptr_t(e.base) < uintptr_t(e.size)) {
          found = true;
          break;
        }
      }
      index += coalesced.size();
    }
    uint32_t mask = __ballot_sync(pcclCoopGetLaneMask(coalesced), found);
    if (mask != 0) {
      int source = __popc(mask-1);
      index = __shfl_sync(pcclCoopGetLaneMask(coalesced), index, source);
      return loadConst(&t->entries[index].window);
    }
    t = loadConst(&t->next);
  }
}
#endif // PCCL_CHECK_HGGCCC

PCCL_HOST_DEVICE_INLINE size_t pcclGetResourceBufferOffset(pcclDevResourceHandle_t h) {
  return ((size_t)h)*128;
}

#if PCCL_CHECK_HGGCCC
PCCL_DEVICE_INLINE void* pcclGetResourceBufferLocalPointer(pcclDevComm const& comm, pcclDevResourceHandle h) {
  void* lsaFlatBase = comm.resourceWindow_inlined.lsaFlatBase;
  uint32_t stride4G = comm.resourceWindow_inlined.stride4G;
  void* local = pccl::utility::add4G(lsaFlatBase, comm.lsaRank*stride4G);
  return (void*)(reinterpret_cast<char(*)[128]>(local) + h);
}

PCCL_DEVICE_INLINE void* pcclGetResourceBufferLsaPointer(pcclDevComm const& comm, pcclDevResourceHandle h, int peer) {
  int r = peer;
  void* lsaFlatBase = comm.resourceWindow_inlined.lsaFlatBase;
  uint32_t stride4G = comm.resourceWindow_inlined.stride4G;
  void* local = pccl::utility::add4G(lsaFlatBase, r*stride4G);
  return (void*)(reinterpret_cast<char(*)[128]>(local) + h);
}

PCCL_DEVICE_INLINE void* pcclGetResourceBufferPeerPointer(pcclDevComm const& comm, pcclDevResourceHandle h, pcclTeam team, int peer) {
  int r = comm.lsaRank + (peer - team.rank)*team.stride;
  void* lsaFlatBase = comm.resourceWindow_inlined.lsaFlatBase;
  uint32_t stride4G = comm.resourceWindow_inlined.stride4G;
  void* local = pccl::utility::add4G(lsaFlatBase, r*stride4G);
  return (void*)(reinterpret_cast<char(*)[128]>(local) + h);
}

PCCL_DEVICE_INLINE void* pcclGetResourceBufferMultimemPointer(pcclDevComm const& comm, pcclDevResourceHandle h, pcclMultimemHandle mm) {
  assert(false && "multimem is not supported on PPU");
  void* ptr = mm.mcBasePtr;
  ptr = reinterpret_cast<char(*)[4096]>(ptr) + comm.resourceWindow_inlined.mcOffset4K;
  ptr = reinterpret_cast<char(*)[128]>(ptr) + h;
  return ptr;
}

PCCL_DEVICE_INLINE void* pcclGetResourceBufferLsaMultimemPointer(pcclDevComm const& comm, pcclDevResourceHandle h) {
  assert(false && "multimem is not supported on PPU");
  return pcclGetResourceBufferMultimemPointer(comm, h, comm.lsaMultimem);
}

PCCL_DEVICE_INLINE pcclSymPtr<char> pcclGetResourceBuffer(pcclDevComm const& comm, pcclDevResourceHandle h) {
  return pcclSymPtr<char>(comm.resourceWindow, size_t(h)*128);
}
#endif // PCCL_CHECK_HGGCCC

#endif
