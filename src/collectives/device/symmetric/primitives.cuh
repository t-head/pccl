/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_SYMMETRIC_PRIMITIVES_H_
#define NCCL_DEVICE_SYMMETRIC_PRIMITIVES_H_

#include "sym_kernels.h"
#include "pccl_device.h"
#include "bitops.h"
#include "sym_reduce_kernel.h"
#include "op128.h"
#include "pccl_device/coop.h"

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

template<bool val> struct BoolTag { static constexpr bool value = val; };
template<typename T> struct ncclDecayType { using Type = T; };
template<typename T> struct ncclDecayType<T&> { using Type = T; };
template<typename T> struct ncclDecayType<T&&> { using Type = T; };
template<typename T> struct ncclDecayType<T const> { using Type = T; };
template<typename T> struct ncclDecayType<T volatile> { using Type = T; };
template<typename T> using ncclDecayType_t = typename ncclDecayType<T>::Type;

template<typename T>
static __device__ void partitionElts(unsigned nParts, unsigned part, size_t* nElts,
                                      ncclSymPtr<T>* inPtr, ncclSymPtr<T>* outPtr) {
  constexpr int eltPerB16 = 16/sizeof(T);
  size_t nB16 = (*nElts + eltPerB16-1)/eltPerB16;
  size_t beginB16 = part*(nB16/nParts) + min(part, uint32_t(nB16%nParts));
  *inPtr += beginB16*eltPerB16;
  *outPtr += beginB16*eltPerB16;
  if (part < nParts-1) {
    nB16 = nB16/nParts + (part < nB16%nParts ? 1 : 0);
    *nElts = nB16*eltPerB16;
  } else {
    *nElts = *nElts - beginB16*eltPerB16;
  }
}

namespace {
struct pcclSymkPrims {
  ncclDevComm const& comm;
  int const& rank;
  int const& nRanks;
  uint32_t const& stride4G;
  ncclLsaBarrierSession<ncclCoopCta>& bar;

  __device__ pcclSymkPrims(ncclDevComm const& comm, ncclLsaBarrierSession<ncclCoopCta>& bar):
    comm(comm), rank(comm.rank), nRanks(comm.nRanks), stride4G(comm.resourceWindow_inlined.stride4G), bar(bar) {}

  __device__ void barrierWait(ncclCoopCta cta, bool acquire) {
    bar.wait(cta, acquire ? hggc::memory_order_acquire : hggc::memory_order_relaxed);
  }

  // Barrier restricted to a contiguous range of ranks [groupBase, groupBase+groupSize).
  // Reuses the full-team inbox slots, touching only the in-group ones.
  //
  // Contract: every rank must issue the same total number of arrive/wait calls
  // (full or group) so bar.epoch stays in lockstep. Inbox values only increase, so
  // peers skipped by a group barrier are still satisfied by a later full barrier
  // writing a strictly larger epoch.
  __device__ void groupBarrierArrive(ncclCoopCta cta, int groupBase, int groupSize, bool release) {
    cta.sync();
    int r = cta.thread_rank();
    if (r != rank && r >= groupBase && r < groupBase + groupSize && r < nRanks) {
      uint32_t* inbox = bar.ucInbox(r, rank);
      if (release) {
        asm volatile("ppu.st.release.sys.u32 [%0],%1;" :: "l"(inbox), "r"(bar.epoch+1));
      } else {
        asm volatile("ppu.st.relaxed.sys.u32 [%0],%1;" :: "l"(inbox), "r"(bar.epoch+1));
      }
    }
  }

  __device__ void groupBarrierWait(ncclCoopCta cta, int groupBase, int groupSize, bool acquire) {
    int r = cta.thread_rank();
    if (r != rank && r >= groupBase && r < groupBase + groupSize && r < nRanks) {
      uint32_t* inbox = bar.ucInbox(rank, r);
      uint32_t steps = 0;
      while (true) {
        uint32_t got;
        if (acquire) {
          asm volatile("ppu.ld.acquire.sys.u32 %0,[%1];" : "=r"(got) : "l"(inbox));
        } else {
          asm volatile("ppu.ld.relaxed.sys.u32 %0,[%1];" : "=r"(got) : "l"(inbox));
        }
        if (got-(bar.epoch+1) <= uint32_t(-1)>>1) break;
        if (pccl::utility::testAbort(comm.abortFlag, steps)) break;
      }
    }
    bar.epoch += 1;
    cta.sync();
  }
};

// One resource handle per CTA keeps metadata separate from aligned data slices.
struct pcclSymkLLa2ASession : ncclLLA2ASession<ncclCoopCta> {
  uint32_t dataOffset;

  __device__ pcclSymkLLa2ASession(ncclDevComm const& comm, ncclLLA2AHandle handle, uint32_t block):
    ncclLLA2ASession<ncclCoopCta>(ncclCoopCta(), comm, ncclTeamLsa(comm),
      ncclLLA2AHandle{handle.bufHandle + block, handle.nSlots}, 0, ncclSymkMaxThreads),
    dataOffset(ncclSymkMaxBlocks*(pcclSymkLLMetadataBytesPerBlock/16) +
               block*(2*handle.nSlots - pcclSymkLLMetadataBytesPerBlock/16)) {
    this->slotsOffset = dataOffset + (this->epoch & 1)*this->handle.nSlots;
  }

  __device__ void endEpoch(ncclCoopCta coop) {
    ncclLLA2ASession<ncclCoopCta>::endEpoch(coop);
    this->slotsOffset = dataOffset + (this->epoch & 1)*this->handle.nSlots;
  }
};
}

// Work partitioning of NCCL v2.30.4-1 ncclSymkArgsHandler (getWorkRange, getWorkRangeFused, forEachWork).
namespace {
struct ncclSymkArgsHandler {
  ncclDevComm const& comm;
  ncclLLA2AHandle const& lsaLLA2A;
  struct ncclSymkChannelWorkRange* channelWorkRange;
  struct ncclSymkDevWork* devWork;

  __device__ ncclSymkArgsHandler(ncclSymkDevWorkArgs const* args):
    comm(args->kcomm.devComm), lsaLLA2A(args->kcomm.lsaLLA2A) {
    channelWorkRange = args->getWorkRange();
    devWork = args->getWorks(args->nMaxChannels);
  }

  template<typename T>
  __device__ void getWorkRange(int block, uint16_t& workLo, size_t& indexLo, uint16_t& workHi, size_t& indexHi) {
    constexpr int EltPerCell = NCCL_SYM_KERNEL_CELL_SIZE / sizeof(T);
    uint32_t fracLo, fracHi;

    // Where the work begins
    workLo = (block==0) ? 0 : channelWorkRange[block-1].workHi; // start where predecessor ends
    fracLo = (block==0) ? 0 : channelWorkRange[block-1].fracHi + 1;
    // If the predecessor ended on the work boundary, then we step to the beginning of the next work.
    // This ensures we never have empty parts.
    if (fracLo == 0x10000) {
      workLo++;
      fracLo = 0;
    }
    struct ncclSymkDevWork const& dwLo = devWork[workLo];
    indexLo = ((fracLo * divUp(dwLo.nElts, EltPerCell)) >> 16) * EltPerCell;

    // Where the work ends
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

    // Where the work begins
    fracLo = (dw.sChannelId>0 && channelWorkRange[dw.sChannelId-1].workHi == w) ? ((channelWorkRange[dw.sChannelId-1].fracHi + 1) & 0xFFFF) : 0;
    indexLo = ((fracLo * divUp(dw.nElts, EltPerCell)) >> 16) * EltPerCell;
    fracHi = (channelWorkRange[lastBlock].workHi == w) ? channelWorkRange[lastBlock].fracHi + 1 : 0x10000;
    indexHi = min(((fracHi * divUp(dw.nElts, EltPerCell)) >> 16) * EltPerCell, dw.nElts);
  }

  template<typename T, typename Fn>
  __device__ void forEachWork(Fn const& fn) {
    uint16_t workLo, workHi;
    size_t indexLo, indexHi;

    getWorkRange<T>(blockIdx.x, workLo, indexLo, workHi, indexHi);

    #pragma unroll 1
    for (int w = workLo; w <= workHi; w++) {
      struct ncclSymkDevWork const& dw = devWork[w];
      size_t const& nAllElts = dw.nElts;
      size_t currentIndexLo, currentIndexHi;
      int block, nBlocks;
      if (blockIdx.x >= dw.sChannelId && blockIdx.x < dw.sChannelId + dw.nChannels) {
        getWorkRangeFused<T>(blockIdx.x, w, block, nBlocks, currentIndexLo, currentIndexHi);
      } else {
        currentIndexLo = (w > workLo) ? 0 : indexLo;
        currentIndexHi = (w < workHi) ? nAllElts : indexHi;
        block = 0;
        nBlocks = 1;
      }

      fn(block, nBlocks, currentIndexHi - currentIndexLo, nAllElts,
         ncclSymPtr<T>(dw.inputWin, dw.inputOff) + currentIndexLo,
         ncclSymPtr<T>(dw.outputWin, dw.outputOff) + currentIndexLo);

      currentIndexLo = 0;
    }
  }

  template<typename T, typename Fn>
  __device__ void singleWork(Fn const& fn) {
    uint16_t w = 0;
    size_t indexLo = 0, indexHi;
    if (gridDim.x == 1) {
      indexHi = devWork[0].nElts;
    } else {
      getWorkRange<T>(blockIdx.x, w, indexLo, w, indexHi);
    }
    ncclSymkDevWork const& dw = devWork[w];
    fn(indexHi - indexLo, dw.nElts,
       ncclSymPtr<T>(dw.inputWin, dw.inputOff) + indexLo,
       ncclSymPtr<T>(dw.outputWin, dw.outputOff) + indexLo);
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

template<template<typename> class Red, typename T, bool nvls = false>
struct ncclSymkAccumType { using Type = T; };
template<> struct ncclSymkAccumType<FuncSum, __half, false> { using Type = float; };
template<> struct ncclSymkAccumType<FuncSumPostDiv, __half, false> { using Type = float; };
template<> struct ncclSymkAccumType<FuncSum, __ppu_bfloat16, false> { using Type = float; };
template<> struct ncclSymkAccumType<FuncSumPostDiv, __ppu_bfloat16, false> { using Type = float; };
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150) && defined(__HGGC_FP8_TYPES_EXIST__)
template<> struct ncclSymkAccumType<FuncSum, __hg_fp8_e4m3, false> { using Type = float; };
template<> struct ncclSymkAccumType<FuncSum, __hg_fp8_e5m2, false> { using Type = float; };
template<> struct ncclSymkAccumType<FuncSumPostDiv, __hg_fp8_e4m3, false> { using Type = __half; };
template<> struct ncclSymkAccumType<FuncSumPostDiv, __hg_fp8_e5m2, false> { using Type = __half; };
#endif

template<template<typename> class Red, typename T>
struct pcclSymkAccumType : ncclSymkAccumType<Red, T> {};
template<typename T>
struct pcclSymkAccumType<FuncSum, T> { using Type = T; };
#endif
