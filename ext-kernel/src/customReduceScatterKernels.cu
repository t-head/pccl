//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include <algorithm>
#include <cstdio>
#include "pccl/ext_kernel_utils.h"
#include "pccl/ext_kernel_launch.h"
#include "customReduceScatterKernels.h"
#include "customAllReduceKernels.h"
#include "customKernels.h"
#include "customDeviceUtils.h"
#include "customDeviceUtils.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////////
// OneShot ReduceScatter Kernel — Warp-Group Approach (mirrors AllReduce oneShot)
//
// eltsTotal = count * nRanks (total input). sendbuff has NRANKS contiguous chunks of recvCount.
//
//   BUFFER layout per rank (one "aligned cycle" = NRANKS * chunkStride elements):
//     chunk[0]: BUFFER[0 .. recvCount)                            — 512-aligned start
//     chunk[1]: BUFFER[chunkStride .. chunkStride + recvCount)    — 512-aligned start
//     ...
//     chunk[r]: BUFFER[r * chunkStride .. r * chunkStride + recvCount)
//
//   chunkStride = ROUNDUP(recvCount, RMT_BULK_ALIGN_SIZE / sizeof(T))
//
// Phase 1 (warp-group): Divide nWarps into NRANKS groups.  Group r (warpGroupSize warps)
//   cooperatively copies chunk r: sendbuff[r*recvCount ..] → localBuf[r*chunkStride ..]
//   Uses memcpyPackWise when recvCount*sizeof(T) % 16 == 0, else memcpyElementWise.
//
// Phase 2 (all warps, aligned decomposition): All warps reduce chunk myRank from
//   all peers' buffers into recvbuff.  recvCount is split across blocks with
//   alignElts granularity so RemoteBulkLoad128 addresses are always 512-aligned.
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, int NRANKS, int INPUT_ALIGN, int OUTPUT_ALIGN, int ELT_WISE_TAIL>
__global__ void pcclKernel_oneShotReduceScatterKernel(struct oneShotDevParams params) {
  static constexpr int PACKED_ELTS_PER_THREAD = PACKED_ELEMS_SIZE / sizeof(T);
  static constexpr int PACKED_ELTS_PER_WARP = PACKED_ELTS_PER_THREAD * WARP_SIZE;

  const int32_t bidx = blockIdx.x;
  const int32_t tidx = threadIdx.x;
  const int32_t wid = tidx / WARP_SIZE;
  const int32_t nWarps = blockDim.x / WARP_SIZE;
  const int32_t laneId = getLaneId();

  const int32_t recvCount = params.eltsTotal / NRANKS;
  // Aligned stride between consecutive chunks in the peer buffer (elements).
  // Guaranteed: chunkStride * sizeof(T) is a multiple of RMT_BULK_ALIGN_SIZE (512).
  constexpr int32_t alignElts = RMT_BULK_ALIGN_SIZE / sizeof(T);
  const int32_t chunkStride = ((recvCount + alignElts - 1) / alignElts) * alignElts;

  volatile T* localBuf = (volatile T*)params.peerBuffers[params.myRank];

  // ---- Phase 1: Warp-group copy to local peer buffer (multi-block collaboration) ----
  // All blocks participate in the copy to maximize parallelism (like AllReduce oneShot).
  // Each block handles a slice of each chunk, decomposed with PACKED_ELTS_PER_WARP alignment.
  // Source alignment (sendbuff + r * recvCount) may not be 16B-aligned for r>0
  // when recvCount * sizeof(T) % 16 != 0 — fall back to memcpyElementWise then.
  {
    int32_t nBlocks = (int32_t)gridDim.x;
    int32_t sliceSize = ROUNDUP(DIVUP(recvCount, nBlocks), PACKED_ELTS_PER_WARP);
    int32_t blockOffsetInChunk = bidx * sliceSize;
    int32_t blockElts = min(sliceSize, max(0, recvCount - blockOffsetInChunk));

    if (blockElts > 0) {
      // nWarps is a multiple of NRANKS (enforced host-side), so warpGroupSize divides
      // nWarps evenly and peerRank = wid / warpGroupSize is always in [0, NRANKS-1].
      // No peerRank < NRANKS guard needed (matches the AllReduce twoShot kernel).
      const int32_t warpGroupSize = nWarps / NRANKS;
      const int32_t peerRank      = wid / warpGroupSize;
      const int32_t widInGroup    = wid % warpGroupSize;

      volatile T* src = (volatile T*)params.sendbuff + peerRank * recvCount + blockOffsetInChunk;
      volatile T* dst = localBuf + peerRank * chunkStride + blockOffsetInChunk;

      const bool srcAligned = INPUT_ALIGN && ((((size_t)peerRank * recvCount + blockOffsetInChunk) * sizeof(T)) % PACKED_ELEMS_SIZE == 0);

      if (srcAligned) {
        int32_t Npack = blockElts / PACKED_ELTS_PER_WARP;
        int32_t exElts = blockElts % PACKED_ELTS_PER_WARP;
        // Forward ELT_WISE_TAIL: with per-peer srcAligned, an aligned peer (e.g. peer0)
        // can enter the pack path even when recvCount is not a pack multiple, so the
        // sub-pack tail must be copied element-wise instead of over-reading a full pack.
        memcpyPackWise<T, 1, 1, ELT_WISE_TAIL>(dst, src, Npack, exElts, warpGroupSize, widInGroup, laneId);
      } else {
        memcpyElementWise<T, 1, 1>(dst, src, blockElts, warpGroupSize, widInGroup, laneId);
      }
    }
  }

  uint32_t barrierFlag = getBarrierFlag<NRANKS>(params.peerBarrierIn, params.myRank, tidx, bidx) + 1;

  __syncthreads();
  blockBarrier<NRANKS>(params.peerBarrierIn, barrierFlag, params.myRank, tidx, bidx);

  // The last block processes the remainder (recvCount - bidx * chunksPerWarpIter).
  {
    int32_t nBlocks = (int32_t)gridDim.x;
    int32_t chunksPerWarpIter = ((recvCount + nBlocks - 1) / nBlocks + alignElts - 1) / alignElts * alignElts;
    int32_t blockOffsetInChunk = bidx * chunksPerWarpIter;
    int32_t myChunkElts = min(chunksPerWarpIter, max(0, recvCount - blockOffsetInChunk));

    if (myChunkElts > 0) {
      volatile T* srcs[NRANKS];
      #pragma unroll
      for (int ii = 0; ii < NRANKS; ++ii) {
        int rank = (params.myRank + ii) % NRANKS;
        srcs[ii] = (volatile T*)params.peerBuffers[rank] + params.myRank * chunkStride + blockOffsetInChunk;
      }

      volatile T* recvbuff = (volatile T*)params.recvbuff + blockOffsetInChunk;

      int32_t reduceNpack = myChunkElts / PACKED_ELTS_PER_WARP;
      int32_t reduceExElts = myChunkElts % PACKED_ELTS_PER_WARP;

      if (OUTPUT_ALIGN)
        reduceFromMultiSource<T, NRANKS, 1, ELT_WISE_TAIL>(srcs, &recvbuff, reduceNpack, reduceExElts,
                                                            nWarps, wid, laneId);
      else
        reduceFromMultiSourceElemWise<T, NRANKS, 1>(srcs, &recvbuff, myChunkElts,
                                                     nWarps, wid, laneId);
    }
  }

  // Trailing barrier: ensure all ranks have finished reading from peer buffers
  // before any rank reuses them in the next call.
  __syncthreads();
  blockBarrier<NRANKS>(params.peerBarrierOut, barrierFlag, params.myRank, tidx, bidx);
}

RS_ONE_SHOT_FUNCS_IMPL()

static char* getReduceScatterExtFuncName(char* funcName, int maxLength, int nRanks) {
  int offset = 0;
  offset += snprintf(funcName, maxLength-offset, "pcclKernel_oneShotReduceScatterKernel");
  offset += snprintf(funcName+offset, maxLength-offset, "_%d", nRanks);
  return funcName;
}

ncclResult_t dispatchReduceScatterKernels(extKernelHostParams& hostParams, extKernelDevParams& devParams,
                                          int blocksPerGrid, int threadsPerBlock, hggcStream_t stream) {
  char funcName[128];
  int inputAlign = ((uintptr_t)(devParams.sendbuff) % RMT_BULK_ALIGN_SIZE == 0 ? 1 : 0);
  int outputAlign = ((uintptr_t)(devParams.recvbuff) % RMT_BULK_ALIGN_SIZE == 0 ? 1 : 0);

  uint32_t packedEltsPerThread = PACKED_ELEMS_SIZE / hostParams.dtypeSize;
  uint32_t recvCount = devParams.eltsTotal / hostParams.nRanks;
  int eltWiseTail = (recvCount % packedEltsPerThread != 0) ? 1 : 0;

  // Force element-wise path when recvCount < one packed warp
  uint32_t packedPerWarp = packedEltsPerThread * WARP_SIZE;
  if (recvCount < packedPerWarp) {
    inputAlign = 0;
    outputAlign = 0;
  }

  int localIndex = getReduceScatterKernelIndex(hostParams.dtype, hostParams.nRanks, inputAlign, outputAlign, eltWiseTail);

  if (localIndex < 0 || localIndex >= RS_ONESHOT_FUNCS_NUM) {
    ERROR("[EXT] ReduceScatter: Unknown funcIndex: %d, legal range[0, %d)", localIndex, RS_ONESHOT_FUNCS_NUM);
    return ncclInternalError;
  }
  hostParams.funcIndex = EXT_FUNC_RS_BASE + localIndex;
  void* func = customKernels[hostParams.funcIndex];

  if (func != NULL) {
    NCCLCHECK(extKernelSetDispatchStrategy((const void*)(func), getReduceScatterExtFuncName(funcName, 128, hostParams.nRanks),
                                           hostParams, blocksPerGrid, ncclFuncReduceScatter));
    dim3 gridDim = blocksPerGrid;
    dim3 blockDim = threadsPerBlock;
    struct oneShotDevParams arg(devParams);
    void* args[] = {&arg};
    LAUNCH_KERNEL(func, gridDim, blockDim, args, 0, stream, 0, hostParams.funcStrategy[hostParams.funcIndex].launchConfig);
  }

  return ncclSuccess;
}

ncclResult_t customReduceScatterImpl(struct extKernelDevParams& devParams, struct extKernelHostParams& hostParams, hggcStream_t stream) {
  // Threads: maximize parallelism for both Phase 1 (warp-group copy) and Phase 2 (reduce).
  // nWarps must be a multiple of NRANKS so that warpGroupSize = nWarps/NRANKS is integer.
  // Use maxThreadNum as the target, rounded down to a multiple of NRANKS warps.
  int maxThreadNum = (int)hostParams.maxThreadNum;
  int nRanks = (int)hostParams.nRanks;
  int nWarps = std::max(1, (maxThreadNum / WARP_SIZE / nRanks)) * nRanks;
  int threadsPerBlock = nWarps * WARP_SIZE;

  // Grid: drive block count by recvCount (the per-chunk work), not eltsTotal.
  // Each block handles one aligned slice of recvCount in Phase 2.
  // Small data (recvCount fits in one block) naturally gets 1 block, avoiding
  // the Phase 2 alignment bug that occurs when blocks straddle chunk boundaries.
  uint32_t alignElts = RMT_BULK_ALIGN_SIZE / hostParams.dtypeSize;
  uint32_t recvCount = devParams.eltsTotal / hostParams.nRanks;
  int blocksPerGrid = divUp((int)recvCount, (int)alignElts);
  blocksPerGrid = std::min(blocksPerGrid, hostParams.maxBlockNum);

  // eltsPerBlock is not used by the kernel (kernel uses gridDim.x for nBlocks).
  // Set to recvCount so that eltsTotal / eltsPerBlock = nRanks, which is a reasonable
  // semantic if any dispatch code references it.
  devParams.eltsPerBlock = recvCount;

  NCCLCHECK(dispatchReduceScatterKernels(hostParams, devParams, blocksPerGrid, threadsPerBlock, stream));

  return ncclSuccess;
}
