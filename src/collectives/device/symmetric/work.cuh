/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_DEVICE_SYMMETRIC_WORK_H_
#define PCCL_DEVICE_SYMMETRIC_WORK_H_

#include "symmetric.h"
#include "sym_kernels.h"

namespace {
struct pcclSymkArgsHandler {
  ncclSymDevComm comm;
  bool incPrecision;
  const pcclSymkChannelWorkRange* channelWorkRange;
  const pcclSymkDevWork* devWork;

  __device__ explicit pcclSymkArgsHandler(pcclSymkDevWorkArgs const* args):
    comm{args->kcomm.base, args->stride4G, args->kcomm.devComm.lsaSize,
         args->kcomm.devComm.lsaRank, args->kcomm.devComm.lsaSize_rcp32},
    incPrecision(args->incPrecision), channelWorkRange(args->getWorkRange()),
    devWork(args->getWorks(args->nMaxChannels)) {}

  template<typename T>
  __device__ T* localPtr(ncclWindow_vidmem* win, size_t offset) const {
    if (!win) return reinterpret_cast<T*>(offset);
    char* base = (char*)::add4G(win->lsaFlatBase, comm.rank * win->stride4G);
    return reinterpret_cast<T*>(base + offset);
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
    struct pcclSymkDevWork const& dwLo = devWork[workLo];
    indexLo = ((fracLo * divUp(dwLo.nElts, EltPerCell)) >> 16) * EltPerCell;

    // Where the work ends
    workHi = channelWorkRange[block].workHi;
    fracHi = channelWorkRange[block].fracHi + 1;
    struct pcclSymkDevWork const& dwHi = devWork[workHi];
    indexHi = minval<size_t>(((fracHi * divUp(dwHi.nElts, EltPerCell)) >> 16) * EltPerCell, dwHi.nElts);
  }

  template<typename T>
  __device__ void getWorkRangeFused(int blockIdx, int w, int& block, int& nBlocks, size_t& indexLo, size_t& indexHi) {
    constexpr int EltPerCell = NCCL_SYM_KERNEL_CELL_SIZE / sizeof(T);
    struct pcclSymkDevWork const& dw = devWork[w];
    uint32_t fracLo, fracHi;
    int lastBlock;

    block = blockIdx - dw.sChannelId;
    nBlocks = dw.nChannels;
    lastBlock = dw.sChannelId+dw.nChannels-1;

    // Where the work begins
    fracLo = (dw.sChannelId>0 && channelWorkRange[dw.sChannelId-1].workHi == w) ? ((channelWorkRange[dw.sChannelId-1].fracHi + 1) & 0xFFFF) : 0;
    indexLo = ((fracLo * divUp(dw.nElts, EltPerCell)) >> 16) * EltPerCell;
    fracHi = (channelWorkRange[lastBlock].workHi == w) ? channelWorkRange[lastBlock].fracHi + 1 : 0x10000;
    indexHi = minval<size_t>(((fracHi * divUp(dw.nElts, EltPerCell)) >> 16) * EltPerCell, dw.nElts);
  }

  template<typename T, typename Fn>
  __device__ void forEachWork(Fn const& fn) {
    uint16_t workLo, workHi;
    size_t indexLo, indexHi;

    getWorkRange<T>(blockIdx.x, workLo, indexLo, workHi, indexHi);

    #pragma unroll 1
    for (int w = workLo; w <= workHi; w++) {
      struct pcclSymkDevWork const& dw = devWork[w];
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
         localPtr<T>(dw.inputWin, dw.inputOff) + currentIndexLo,
         localPtr<T>(dw.outputWin, dw.outputOff) + currentIndexLo, dw.redOpArg);

    }
  }

};

// The existing single-task ABI already describes one whole work.
struct pcclSymSingleArgsHandler {
  ncclSymDevArgs const* args;
  ncclSymDevComm const& comm;
  bool incPrecision;
  __device__ explicit pcclSymSingleArgsHandler(ncclSymDevArgs const* a):
    args(a), comm(a->comm), incPrecision(a->incPrecision) {}
  template<typename T, typename Fn>
  __device__ void forEachWork(Fn const& fn) {
    fn(blockIdx.x, args->nBlocks, args->nElts, args->nElts,
       reinterpret_cast<T*>(args->input), reinterpret_cast<T*>(args->output), args->redOpArg);
  }
};
} // namespace
#endif
