/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_SYMMETRIC_PRIMITIVES_H_
#define NCCL_DEVICE_SYMMETRIC_PRIMITIVES_H_

#include "symmetric.h"
#include "sym_kernels.h"
#include "bitops.h"
#include "sym_reduce_kernel.h"
#include "op128.h"
#include "pccl_device.h"
#include "pccl_device/core.h"
#include "pccl_device/ptr.h"
#include "pccl_device/ll_a2a.h"
#include "pccl_device/lsa_barrier.h"

// align to 128B could get enough icn performance
#define BULK_ALIGN_BYTES 128u
#define WARP_SIZE 32

// flattenIx(pos0, dim0, pos1, dim1, pos2, dim2, ...)
// Given a position vector `pos` in a rectangular index space with lengths in the `dim`
// vector, flatten that down to a linear index. The fastest moving dimension is given first.
__device__ __forceinline__ int flattenIx() { return 0; }

template<typename Int0, typename Int1, typename ...Ints>
static __device__ Int0 flattenIx(Int0 pos, Int1 size, Ints ...more) {
  return pos + size*flattenIx(more...);
}

// Precomputed integer reciprocals for denominator values 1..64 inclusive.
// Pass these to idivFast64() for fast division on the GPU.
static __device__ uint64_t idivRcp64_upto64(int x) {
  static constexpr uint64_t table[65] = {
    idivRcp64(0x01), idivRcp64(0x01), idivRcp64(0x02), idivRcp64(0x03),
    idivRcp64(0x04), idivRcp64(0x05), idivRcp64(0x06), idivRcp64(0x07),
    idivRcp64(0x08), idivRcp64(0x09), idivRcp64(0x0a), idivRcp64(0x0b),
    idivRcp64(0x0c), idivRcp64(0x0d), idivRcp64(0x0e), idivRcp64(0x0f),
    idivRcp64(0x10), idivRcp64(0x11), idivRcp64(0x12), idivRcp64(0x13),
    idivRcp64(0x14), idivRcp64(0x15), idivRcp64(0x16), idivRcp64(0x17),
    idivRcp64(0x18), idivRcp64(0x19), idivRcp64(0x1a), idivRcp64(0x1b),
    idivRcp64(0x1c), idivRcp64(0x1d), idivRcp64(0x1e), idivRcp64(0x1f),
    idivRcp64(0x20), idivRcp64(0x21), idivRcp64(0x22), idivRcp64(0x23),
    idivRcp64(0x24), idivRcp64(0x25), idivRcp64(0x26), idivRcp64(0x27),
    idivRcp64(0x28), idivRcp64(0x29), idivRcp64(0x2a), idivRcp64(0x2b),
    idivRcp64(0x2c), idivRcp64(0x2d), idivRcp64(0x2e), idivRcp64(0x2f),
    idivRcp64(0x30), idivRcp64(0x31), idivRcp64(0x32), idivRcp64(0x33),
    idivRcp64(0x34), idivRcp64(0x35), idivRcp64(0x36), idivRcp64(0x37),
    idivRcp64(0x38), idivRcp64(0x39), idivRcp64(0x3a), idivRcp64(0x3b),
    idivRcp64(0x3c), idivRcp64(0x3d), idivRcp64(0x3e), idivRcp64(0x3f),
    idivRcp64(0x40)
  };
  return table[x];
}

static __device__ uint32_t idivRcp32_upto64(int x) {
  return idivRcp64_upto64(x)>>32;
}

namespace {
template<typename Coop>
using ncclLsaBarrierSession = pcclLsaBarrierSession<Coop>;
template<typename Coop>
using ncclLLA2ASession = pcclLLA2ASession<Coop>;

struct ncclSymkArgsHandler {
  ncclDevComm const& comm;
  ncclLLA2AHandle const& lsaLLA2A;
  ncclGinOutboxHandle const& ginOutbox;
  ncclGinInboxA2AHandle const& ginInboxRail;
  ncclGinCounter_t ginCounterPerBlock;
  ncclGinSyncHandle const& ginSyncHandle;
  struct ncclSymkChannelWorkRange* channelWorkRange;
  struct ncclSymkDevWork* devWork;
  struct ncclSymkDevWork const* currentWork;
  uint32_t nRanks_rcp32;

  __device__ ncclSymkArgsHandler(ncclSymkDevWorkArgs const* args):
    comm(args->kcomm.devComm),
    lsaLLA2A(args->kcomm.lsaLLA2A),
    ginOutbox(args->kcomm.ginOutbox),
    ginInboxRail(args->kcomm.ginInboxRail),
    ginCounterPerBlock(args->kcomm.ginCounterPerBlock),
    ginSyncHandle(args->kcomm.ginSyncHandle) {
    channelWorkRange = args->getWorkRange();
    devWork = args->getWorks(args->nMaxChannels);
    currentWork = nullptr;
    nRanks_rcp32 = comm.nRanks_rcp32;
  }

  __device__ uint64_t redOpArg() const { return currentWork->redOpArg; }
  __device__ bool incPrecision() const { return (currentWork->padding & 1) != 0; }

  template<typename T>
  __device__ void getWorkRange(int block, uint16_t& workLo, size_t& indexLo, uint16_t& workHi, size_t& indexHi) {
    constexpr int EltPerCell = NCCL_SYM_KERNEL_CELL_SIZE / sizeof(T);
    uint32_t fracLo, fracHi;

    workLo = (block==0) ? 0 : channelWorkRange[block-1].workHi;
    fracLo = (block==0) ? 0 : channelWorkRange[block-1].fracHi + 1;
    if (fracLo == 0x10000) {
      workLo++;
      fracLo = 0;
    }
    struct ncclSymkDevWork const& dwLo = devWork[workLo];
    indexLo = ((fracLo * divUp(dwLo.nElts, EltPerCell)) >> 16) * EltPerCell;

    workHi = channelWorkRange[block].workHi;
    fracHi = channelWorkRange[block].fracHi + 1;
    struct ncclSymkDevWork const& dwHi = devWork[workHi];
    indexHi = min(((fracHi * divUp(dwHi.nElts, EltPerCell)) >> 16) * EltPerCell, dwHi.nElts);
  }

  template<typename T>
  __device__ void getWorkRangeFused(int blockIdx, int w, int& block, int& nBlocks, size_t& indexLo, size_t& indexHi) {
    constexpr int EltPerCell = NCCL_SYM_KERNEL_CELL_SIZE / sizeof(T);
    struct ncclSymkDevWork const& dw = devWork[w];
    uint32_t fracLo, fracHi;
    int lastBlock;

    block = blockIdx - dw.sChannelId;
    nBlocks = dw.nChannels;
    lastBlock = dw.sChannelId+dw.nChannels-1;

    fracLo = (dw.sChannelId>0 && channelWorkRange[dw.sChannelId-1].workHi == w)
           ? ((channelWorkRange[dw.sChannelId-1].fracHi + 1) & 0xFFFF) : 0;
    indexLo = ((fracLo * divUp(dw.nElts, EltPerCell)) >> 16) * EltPerCell;
    fracHi = (channelWorkRange[lastBlock].workHi == w) ? channelWorkRange[lastBlock].fracHi + 1 : 0x10000;
    indexHi = min(((fracHi * divUp(dw.nElts, EltPerCell)) >> 16) * EltPerCell, dw.nElts);
  }

  template<typename T, typename Fn>
  __device__ void forEachWork(Fn const& fn) {
    constexpr int EltPerCell = NCCL_SYM_KERNEL_CELL_SIZE / sizeof(T);
    uint16_t workLo, workHi;
    uint32_t fracLo, fracHi;

    workLo = (blockIdx.x==0) ? 0 : channelWorkRange[blockIdx.x-1].workHi;
    fracLo = (blockIdx.x==0) ? 0 : channelWorkRange[blockIdx.x-1].fracHi + 1;
    if (fracLo == 0x10000) {
      workLo++;
      fracLo = 0;
    }
    workHi = channelWorkRange[blockIdx.x].workHi;
    fracHi = channelWorkRange[blockIdx.x].fracHi + 1;

    #pragma unroll 1
    for (int w = workLo; w <= workHi; w++) {
      struct ncclSymkDevWork const& dw = devWork[w];
      size_t const& nAllElts = dw.nElts;
      size_t currentIndexLo, currentIndexHi;
      int block, nBlocks;
      if (blockIdx.x >= dw.sChannelId && blockIdx.x < dw.sChannelId + dw.nChannels) {
        getWorkRangeFused<T>(blockIdx.x, w, block, nBlocks, currentIndexLo, currentIndexHi);
      } else {
        currentIndexLo = (w > workLo) ? 0 : ((fracLo * divUp(nAllElts, EltPerCell)) >> 16) * EltPerCell;
        currentIndexHi = (w < workHi) ? nAllElts : min(((fracHi * divUp(nAllElts, EltPerCell)) >> 16) * EltPerCell, nAllElts);
        block = 0;
        nBlocks = 1;
      }

      currentWork = &dw;
      fn(block, nBlocks, currentIndexHi - currentIndexLo, nAllElts,
         ncclSymPtr<T>(dw.inputWin, dw.inputOff) + currentIndexLo,
         ncclSymPtr<T>(dw.outputWin, dw.outputOff) + currentIndexLo);

      currentIndexLo = 0;
    }
  }

  template<typename T, typename Fn>
  __device__ void singleWork(Fn const& fn) {
    constexpr int EltPerCell = NCCL_SYM_KERNEL_CELL_SIZE / sizeof(T);
    // The single-work contract makes work index zero invariant.
    struct ncclSymkDevWork const& dw = devWork[0];
    size_t nCells = divUp(dw.nElts, EltPerCell);
    uint32_t fracLo = blockIdx.x == 0 ? 0 : channelWorkRange[blockIdx.x-1].fracHi + 1;
    uint32_t fracHi = channelWorkRange[blockIdx.x].fracHi + 1;
    size_t indexLo = ((fracLo * nCells) >> 16) * EltPerCell;
    size_t indexHi = min(((fracHi * nCells) >> 16) * EltPerCell, dw.nElts);
    currentWork = &dw;
    fn(indexHi - indexLo, dw.nElts,
       ncclSymPtr<T>(dw.inputWin, dw.inputOff) + indexLo,
       ncclSymPtr<T>(dw.outputWin, dw.outputOff) + indexLo);
  }

  template<typename T, typename Fn>
  __device__ void forEachWorkNoFusion(Fn const& fn) {
    uint16_t workLo, workHi;
    size_t indexLo, indexHi;

    getWorkRange<T>(blockIdx.x, workLo, indexLo, workHi, indexHi);

    #pragma unroll 1
    for (int w = workLo; w <= workHi; w++) {
      struct ncclSymkDevWork const& dw = devWork[w];
      size_t const& nAllElts = dw.nElts;
      size_t currentIndexLo, currentIndexHi;
      currentIndexLo = (w > workLo) ? 0 : indexLo;
      currentIndexHi = (w < workHi) ? nAllElts : indexHi;

      currentWork = &dw;
      fn(currentIndexHi - currentIndexLo, nAllElts,
         ncclSymPtr<T>(dw.inputWin, dw.inputOff) + currentIndexLo,
         ncclSymPtr<T>(dw.outputWin, dw.outputOff) + currentIndexLo);
    }
  }
};

static constexpr int ncclSymPrims_UseBarrier = 1;
struct ncclSymPrims {
  int flags;
  int const &rank;
  int const &nRanks;
  uint32_t const &nRanks_rcp32;
  int block, nBlocks;
  uint32_t nBlocks_rcp32;
  uint32_t nBlocks_nWarps_rcp32;
  uint32_t nRanks_nBlocks_rcp32;
  uint32_t nWarpPerRank, nWarpPerRank_rcp32;
  struct ncclSymDevBase* const &base;

  uint32_t const &stride4G;
  uint32_t barEpoch;

  __device__ ncclSymPrims(ncclSymDevComm const &comm, int numBlocks, int flags):
    flags(flags),
    rank(comm.rank),
    nRanks(comm.nRanks),
    nRanks_rcp32(comm.nRanks_rcp32),
    block(blockIdx.x),
    nBlocks(numBlocks),
    nBlocks_rcp32(idivRcp32_upto64(nBlocks)),
    nBlocks_nWarps_rcp32(imulRcp32(nBlocks, nBlocks_rcp32, blockDim.x/32, idivRcp32_upto64(blockDim.x/32))),
    nRanks_nBlocks_rcp32(imulRcp32(nRanks, nRanks_rcp32, numBlocks, nBlocks_rcp32)),
    nWarpPerRank(idivFast32(nBlocks*blockDim.x/32, nRanks, nRanks_rcp32)),
    nWarpPerRank_rcp32(idivRcp32_upto64(nWarpPerRank)),
    base(comm.base),
    stride4G(comm.stride4G) {
    if ((flags & ncclSymPrims_UseBarrier) && threadIdx.x < nRanks) {
      barEpoch = base->barEpochUc[block];
    }
  }
  __device__  ~ncclSymPrims() {
    if (threadIdx.x == 0) {
      if (flags & ncclSymPrims_UseBarrier) {
        base->barEpochUc[block] = barEpoch;
      }
    }
  }

  template<typename T>
  __device__ T* peerPtr(int peer, T* selfPtr) {
    return add4G(selfPtr, (peer-rank)*stride4G);
  }

  __device__  void barrierArrive(ncclCoopCta cta, bool release) {
    cta.sync();
    int r = cta.thread_rank();
    if (r != rank && r < nRanks) {
      uint32_t* inbox = &peerPtr(r, base)->barInboxPerPeer[block*nRanks + rank];
      if (release) {
        asm volatile("ppu.st.release.sys.u32 [%0],%1;" :: "l"(inbox), "r"(barEpoch+1));
      } else {
        asm volatile("ppu.st.relaxed.sys.u32 [%0],%1;" :: "l"(inbox), "r"(barEpoch+1));
      }
    }
  }

  __device__  void barrierWait(ncclCoopCta cta, bool acquire) {
    int r = cta.thread_rank();
    if (r != rank && r < nRanks) {
      uint32_t* inbox = &base->barInboxPerPeer[block*nRanks + r];
      while (true) {
        uint32_t got;
        if (acquire) {
          asm volatile("ppu.ld.acquire.sys.u32 %0,[%1];" : "=r"(got) : "l"(inbox));
        } else {
          asm volatile("ppu.ld.relaxed.sys.u32 %0,[%1];" : "=r"(got) : "l"(inbox));
        }
        if (got-(barEpoch+1) <= uint32_t(-1)>>1) break;
      }
    }
    barEpoch += 1;
    cta.sync();
  }

  // Barrier restricted to a contiguous range of ranks [groupBase, groupBase+groupSize).
  // Reuses the all-rank barInboxPerPeer slots, touching only the in-group ones.
  //
  // Contract: every rank must issue the same total number of arrive/wait calls
  // (full or group) so barEpoch stays in lockstep. Inbox values only increase, so
  // peers skipped by a group barrier are still satisfied by a later full barrier
  // writing a strictly larger epoch.
  __device__ void groupBarrierArrive(ncclCoopCta cta, int groupBase, int groupSize, bool release) {
    cta.sync();
    int r = cta.thread_rank();
    if (r != rank && r >= groupBase && r < groupBase + groupSize && r < nRanks) {
      uint32_t* inbox = &peerPtr(r, base)->barInboxPerPeer[block*nRanks + rank];
      if (release) {
        asm volatile("ppu.st.release.sys.u32 [%0],%1;" :: "l"(inbox), "r"(barEpoch+1));
      } else {
        asm volatile("ppu.st.relaxed.sys.u32 [%0],%1;" :: "l"(inbox), "r"(barEpoch+1));
      }
    }
  }

  __device__ void groupBarrierWait(ncclCoopCta cta, int groupBase, int groupSize, bool acquire) {
    int r = cta.thread_rank();
    if (r != rank && r >= groupBase && r < groupBase + groupSize && r < nRanks) {
      uint32_t* inbox = &base->barInboxPerPeer[block*nRanks + r];
      while (true) {
        uint32_t got;
        if (acquire) {
          asm volatile("ppu.ld.acquire.sys.u32 %0,[%1];" : "=r"(got) : "l"(inbox));
        } else {
          asm volatile("ppu.ld.relaxed.sys.u32 %0,[%1];" : "=r"(got) : "l"(inbox));
        }
        if (got-(barEpoch+1) <= uint32_t(-1)>>1) break;
      }
    }
    barEpoch += 1;
    cta.sync();
  }
};

struct ncclSymkPrims {
  int const &rank;
  int const &nRanks;
  uint32_t const &nRanks_rcp32;
  int block, nBlocks;
  uint32_t nBlocks_rcp32;
  uint32_t nBlocks_nWarps_rcp32;
  uint32_t nRanks_nBlocks_rcp32;
  uint32_t nWarpPerRank, nWarpPerRank_rcp32;
  uint32_t stride4G;

private:
  ncclLsaBarrierSession<ncclCoopCta>* barSession;
  ncclLLA2ASession<ncclCoopCta>* llSession;

  __device__ ncclSymkPrims(ncclDevComm const &comm, int block, int nBlocks, uint32_t stride4G):
    rank(comm.rank),
    nRanks(comm.nRanks),
    nRanks_rcp32(comm.nRanks_rcp32),
    block(block),
    nBlocks(nBlocks),
    nBlocks_rcp32(idivRcp32_upto64(nBlocks)),
    nBlocks_nWarps_rcp32(imulRcp32(nBlocks, nBlocks_rcp32, blockDim.x/32, idivRcp32_upto64(blockDim.x/32))),
    nRanks_nBlocks_rcp32(imulRcp32(nRanks, nRanks_rcp32, nBlocks, nBlocks_rcp32)),
    nWarpPerRank(idivFast32(nBlocks*blockDim.x/32, nRanks, nRanks_rcp32)),
    nWarpPerRank_rcp32(idivRcp32_upto64(nWarpPerRank)),
    stride4G(stride4G),
    barSession(nullptr),
    llSession(nullptr) {
  }

  __device__ uint32_t* barInbox(int owner, int peer) {
    uint32_t* state = (uint32_t*)pcclGetResourceBufferLsaPointer(
      barSession->comm, barSession->handle.bufHandle, owner);
    return state + 3*barSession->handle.nBarriers + barSession->index*barSession->comm.lsaSize + peer;
  }

public:
  __device__ ncclSymkPrims(ncclDevComm const &comm, int block, int nBlocks, uint32_t stride4G,
                         ncclLsaBarrierSession<ncclCoopCta>& session):
    ncclSymkPrims(comm, block, nBlocks, stride4G) {
    barSession = &session;
  }

  __device__ ncclSymkPrims(ncclDevComm const &comm, int block, int nBlocks, uint32_t stride4G,
                         ncclLLA2ASession<ncclCoopCta>& session):
    ncclSymkPrims(comm, block, nBlocks, stride4G) {
    llSession = &session;
  }

  template<typename T>
  __device__ T* peerPtr(int peer, T* selfPtr) {
    return add4G(selfPtr, (peer-rank)*stride4G);
  }

  __device__ void barrierArrive(ncclCoopCta cta, bool release) {
    cta.sync();
    int r = cta.thread_rank();
    if (r != barSession->comm.lsaRank && r < barSession->comm.lsaSize) {
      uint32_t* inbox = barInbox(r, barSession->comm.lsaRank);
      if (release) {
        asm volatile("ppu.st.release.sys.u32 [%0],%1;" :: "l"(inbox), "r"(barSession->epoch+1));
      } else {
        asm volatile("ppu.st.relaxed.sys.u32 [%0],%1;" :: "l"(inbox), "r"(barSession->epoch+1));
      }
    }
  }

  __device__ void barrierWait(ncclCoopCta cta, bool acquire) {
    int r = cta.thread_rank();
    if (r != barSession->comm.lsaRank && r < barSession->comm.lsaSize) {
      uint32_t* inbox = barInbox(barSession->comm.lsaRank, r);
      while (true) {
        uint32_t got;
        if (acquire) {
          asm volatile("ppu.ld.acquire.sys.u32 %0,[%1];" : "=r"(got) : "l"(inbox));
        } else {
          asm volatile("ppu.ld.relaxed.sys.u32 %0,[%1];" : "=r"(got) : "l"(inbox));
        }
        if (got-(barSession->epoch+1) <= uint32_t(-1)>>1) break;
      }
    }
    barSession->epoch += 1;
    cta.sync();
  }

  __device__ void groupBarrierArrive(ncclCoopCta cta, int groupBase, int groupSize, bool release) {
    cta.sync();
    int r = cta.thread_rank();
    if (r != barSession->comm.lsaRank && r >= groupBase && r < groupBase + groupSize &&
        r < barSession->comm.lsaSize) {
      uint32_t* inbox = barInbox(r, barSession->comm.lsaRank);
      if (release) {
        asm volatile("ppu.st.release.sys.u32 [%0],%1;" :: "l"(inbox), "r"(barSession->epoch+1));
      } else {
        asm volatile("ppu.st.relaxed.sys.u32 [%0],%1;" :: "l"(inbox), "r"(barSession->epoch+1));
      }
    }
  }

  __device__ void groupBarrierWait(ncclCoopCta cta, int groupBase, int groupSize, bool acquire) {
    int r = cta.thread_rank();
    if (r != barSession->comm.lsaRank && r >= groupBase && r < groupBase + groupSize &&
        r < barSession->comm.lsaSize) {
      uint32_t* inbox = barInbox(barSession->comm.lsaRank, r);
      while (true) {
        uint32_t got;
        if (acquire) {
          asm volatile("ppu.ld.acquire.sys.u32 %0,[%1];" : "=r"(got) : "l"(inbox));
        } else {
          asm volatile("ppu.ld.relaxed.sys.u32 %0,[%1];" : "=r"(got) : "l"(inbox));
        }
        if (got-(barSession->epoch+1) <= uint32_t(-1)>>1) break;
      }
    }
    barSession->epoch += 1;
    cta.sync();
  }

  __device__ void endLL(ncclCoopCta cta) {
    llSession->endEpoch(cta);
  }

  template<typename T>
  __device__ void sendLL(int peer, int slot, T val) {
    llSession->send(peer, slot, val);
  }

  template<typename T>
  __device__ void bcastLL(int slot, T val) {
    union { T tmp; uint32_t u32[divUp(sizeof(T),8)][2]; };
    tmp = val;
    uint4* buf0 = (uint4*)pcclGetResourceBufferPeerPointer(
      llSession->comm, llSession->handle.bufHandle, llSession->team, 0);
    buf0 += llSession->slotsOffset + slot;
    uint32_t const peerStride4G = llSession->team.stride*stride4G;
    uint32_t const epoch = llSession->epoch;
    int const pitch = llSession->pitch;
    int const nRanks = llSession->team.nRanks;
    int dr = 0;
    int r = llSession->team.rank;
    #pragma unroll 1
    for (; dr+8 <= nRanks; dr += 8) {
      #pragma unroll
      for (int ur=0; ur < 8; ur++) {
        uint4* buf = add4G(buf0, r*peerStride4G);
        #pragma unroll
        for (int u=0; u < divUp(sizeof(T),8); u++) {
          asm volatile("st.relaxed.sys.v4.u32 [%0],{%1,%3,%2,%3};" ::
            "l"(buf + u*pitch), "r"(u32[u][0]), "r"(u32[u][1]), "r"(epoch) : "memory");
        }
        r += 1;
        if (r == nRanks) r = 0;
      }
    }
    #pragma unroll
    for (int ur=0; ur < 8; ur++, dr++) {
      if (dr == nRanks) break;
      uint4* buf = add4G(buf0, r*peerStride4G);
      #pragma unroll
      for (int u=0; u < divUp(sizeof(T),8); u++) {
        asm volatile("st.relaxed.sys.v4.u32 [%0],{%1,%3,%2,%3};" ::
          "l"(buf + u*pitch), "r"(u32[u][0]), "r"(u32[u][1]), "r"(epoch) : "memory");
      }
      r += 1;
      if (r == nRanks) r = 0;
    }
  }

  template<int nSlotsMin, int nSlotsMax, typename T>
  __device__ void recvLL(int slot0, int nSlots, int stride, T(&elts)[nSlotsMax]) {
    llSession->template recvUnrolled<nSlotsMin, nSlotsMax, T>(slot0, nSlots, stride, elts);
  }

  template<typename Pack, typename T, typename Red, int Unroll=8>
  __device__ Pack recvReduceLL(int slot, int stride, Red red) {
    using Acc = typename Red::EltType;
    using AccPack = BytePack<sizeof(Pack)*sizeof(Acc)/sizeof(T)>;
    AccPack acc;
    bool first = true;
    int r = 0;
    #pragma unroll 1
    for (; r+Unroll <= nRanks; r += Unroll) {
      Pack got[Unroll];
      this->template recvLL</*Min=*/Unroll>(slot + r*stride, Unroll, stride, got);
      AccPack acc0 = applyCast<T, Acc>(got[0]);
      acc = first ? acc0 : applyReduce(red, acc, acc0);
      first = false;
      #pragma unroll
      for (int i=1; i < Unroll; i++) acc = applyReduce(red, acc, applyCast<T, Acc>(got[i]));
    }
    if (r < nRanks) {
      Pack got[Unroll];
      this->template recvLL</*Min=*/1>(slot + r*stride, nRanks-r, stride, got);
      AccPack acc0 = applyCast<T, Acc>(got[0]);
      acc = first ? acc0 : applyReduce(red, acc, acc0);
      #pragma unroll
      for (int i=1; i < Unroll-1; i++) {
        if (r+i < nRanks) acc = applyReduce(red, acc, applyCast<T, Acc>(got[i]));
      }
    }
    return applyCast<Acc, T>(acc);
  }

  template<typename T>
  __device__ T recvLL(int slot) {
    return llSession->template recv<T>(slot);
  }
};
}

__device__ __forceinline__ uint64_t read_firstlane_u64(uint64_t val) {
    uint32_t lowPart = (uint32_t)val;
    uint32_t highPart = (uint32_t)(val >> 32);

    uint32_t lowPartPerWarp = __ppu_read_firstlane(lowPart);
    uint32_t highPartPerWarp = __ppu_read_firstlane(highPart);

    return ((uint64_t)highPartPerWarp << 32) | lowPartPerWarp;
}

template<template<typename> class Red, typename T, bool incPrecision = false>
struct ncclSymAccumType { using Type = T; };

// Only Red's whose opArg is invariant w.r.t. the datatype can have a different
// accumulator type. At the moment this excludes integer min/max, sumpostdiv,
// and premulsum.
template<> struct ncclSymAccumType<FuncSum, __half, true> { using Type = float; };
template<> struct ncclSymAccumType<FuncSum, __ppu_bfloat16, true> { using Type = float; };
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150) && defined(__HGGC_FP8_TYPES_EXIST__)
template<> struct ncclSymAccumType<FuncSum, __hg_fp8_e4m3, true> { using Type = float; };
template<> struct ncclSymAccumType<FuncSum, __hg_fp8_e5m2, true> { using Type = float; };
#endif
#endif
