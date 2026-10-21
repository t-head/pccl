//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include <algorithm>
#include <cstdio>
#include "pccl/ext_kernel_utils.h"
#include "pccl/ext_kernel_launch.h"
#include "customAllReduceKernels.h"
#include "customKernels.h"
#include "customDeviceUtils.h"
#include "customDeviceUtils.cuh"

/*#define DEBUG_CUSTOM_ALL_REDUCE 1*/

template<typename T, int NRANKS, int DIRECT_OUT>
__inline__ __device__ void reduce(volatile T* myBuff, volatile T* dstPtr, int32_t myRank, int32_t eltsPerRank, int32_t Npack, int32_t warpGroupSize, int32_t exElts, int32_t w, int32_t laneId) {
  static constexpr int PACKED_ELTS_PER_THREAD = PACKED_ELEMS_SIZE / sizeof(T);
  static constexpr int PACKED_ELTS_PER_WARP = PACKED_ELTS_PER_THREAD * WARP_SIZE;
  using PackedStruct = typename PackedOn16Bytes<T>::Type;
  const int32_t eltsPerGroup = warpGroupSize * PACKED_ELTS_PER_WARP;

  PackedStruct vals[NRANKS];
  volatile T* buff = myBuff + w * PACKED_ELTS_PER_WARP;
  int32_t myRankBase = myRank * eltsPerRank;

  volatile T* dst = NULL;
  if (DIRECT_OUT) dst = dstPtr + w * PACKED_ELTS_PER_WARP;
  else dst = buff + myRankBase;

  while (w < Npack) {
    #pragma unroll
    for (int ii = 0; ii < NRANKS; ++ii) {
      BulkLoad128(vals[ii].upacked, (const uint4*)(buff + ii * eltsPerRank));
    }

    // Sum the values from the different ranks.
    #pragma unroll
    for (int ii = 1; ii < NRANKS; ++ii) {
      vals[0].packed = add128b(vals[0], vals[ii]);
    }

    BulkStore128(vals[0].upacked, (uint4*)(dst));

    buff += eltsPerGroup;
    dst += eltsPerGroup;
    w += warpGroupSize;
  }

  if (exElts > 0 && w == Npack) {
    int32_t offset = laneId * PACKED_ELTS_PER_THREAD;
    if (offset < exElts)  {
      buff += offset;
      dst += offset;
      #pragma unroll
      for (int ii = 0; ii < NRANKS; ++ii) {
        Load128(vals[ii].upacked, (const uint4*)(buff + ii * eltsPerRank));
      }
      // Sum the values from the different ranks.
      #pragma unroll
      for (int ii = 1; ii < NRANKS; ++ii) {
        vals[0].packed = add128b(vals[0], vals[ii]);
      }
      Store128(vals[0].upacked, (uint4*)(dst));
    }
  }
}

template <typename T, int NRANKS, int INPUT_ALIGN, int OUTPUT_ALIGN>
__global__ void pcclKernel_oneShotAllReduceKernel(struct oneShotDevParams params) {
  static constexpr int PACKED_ELTS_PER_THREAD = PACKED_ELEMS_SIZE / sizeof(T);
  static constexpr int PACKED_ELTS_PER_WARP = PACKED_ELTS_PER_THREAD * WARP_SIZE;

  const int32_t bidx = blockIdx.x;
  const int32_t tidx = threadIdx.x;
  const int32_t wid = tidx / WARP_SIZE;
  const int32_t nWarps = blockDim.x / WARP_SIZE;
  const int32_t laneId = getLaneId();

  const int32_t buffEltsOffset = params.buffEltsPerBlock * bidx;
  const int32_t eltsOffset = params.eltsPerBlock * bidx;

  volatile T* srcPtrs[NRANKS];
  #pragma unroll
  for (int ii = 0; ii < NRANKS; ++ii) {
    int rank = (params.myRank + ii) % NRANKS;
    srcPtrs[ii] = (T*)params.peerBuffers[rank] + buffEltsOffset;
  }

  volatile T* sendbuff = (T*)params.sendbuff + eltsOffset;
  volatile T* recvbuff = (T*)params.recvbuff + eltsOffset;

  int32_t eltsPerBlock = params.eltsTotal - eltsOffset;
  eltsPerBlock = min(eltsPerBlock, params.eltsPerBlock);
  int32_t exElts = eltsPerBlock % (PACKED_ELTS_PER_WARP);
  int32_t Npack = eltsPerBlock / PACKED_ELTS_PER_WARP;

  if (INPUT_ALIGN) memcpyPackWise<T, 1, 1>(srcPtrs[0], sendbuff, Npack, exElts, nWarps, wid, laneId);
  else memcpyElementWise<T, 1, 1>(srcPtrs[0], sendbuff, eltsPerBlock, nWarps, wid, laneId);

  uint32_t barrierFlag = getBarrierFlag<NRANKS>(params.peerBarrierIn, params.myRank, tidx, bidx) + 1;

  __syncthreads();
  blockBarrier<NRANKS>(params.peerBarrierIn, barrierFlag, params.myRank, tidx, bidx);

  if (OUTPUT_ALIGN) reduceFromMultiSource<T, NRANKS, 1>(srcPtrs, &recvbuff, Npack, exElts, nWarps, wid, laneId);
  else reduceFromMultiSourceElemWise<T, NRANKS, 1>(srcPtrs, &recvbuff, eltsPerBlock, nWarps, wid, laneId);

  __syncthreads();
  blockBarrier<NRANKS>(params.peerBarrierOut, barrierFlag, params.myRank, tidx, bidx);
}

template <typename T, int NRANKS, int USE_DOUBLE_BUFFER, int INPUT_ALIGN, int OUTPUT_ALIGN>
__global__ void pcclKernel_twoShotAllReduceKernel(struct twoShotDevParams params) {
  // The number of elements packed into one for comms
  static constexpr int PACKED_ELTS_PER_THREAD = PACKED_ELEMS_SIZE / sizeof(T);
  static constexpr int PACKED_ELTS_PER_WARP = PACKED_ELTS_PER_THREAD * WARP_SIZE;

  const int32_t bidx = blockIdx.x;
  const int32_t tidx = threadIdx.x;
  const int32_t laneId = getLaneId();
  const int32_t nWarps = blockDim.x / WARP_SIZE;
  const int32_t wid = tidx / WARP_SIZE;
  const int32_t warpGroupSize = nWarps / NRANKS;   // warpGroupSize: warp number for one peer.
  const int32_t peerRank = wid / warpGroupSize;
  const int32_t widInGroup = wid % warpGroupSize;  // warp offset in group
  const int32_t myRank = params.myRank;

  volatile T* myBuf = (T*)params.peerBuffers[myRank] + params.buffEltsPerBlock * bidx;
  volatile T* peerBuf = (T*)params.peerBuffers[peerRank] + params.buffEltsPerBlock * bidx;
  const int32_t buffEltsPerRank = params.buffEltsPerBlock / NRANKS;
  T* myBufD = NULL;
  T* peerBufD = NULL;
  if (USE_DOUBLE_BUFFER) myBufD = (T*)params.peerBuffersD[myRank] + params.buffEltsPerBlock * bidx;
  if (USE_DOUBLE_BUFFER) peerBufD = (T*)params.peerBuffersD[peerRank] + params.buffEltsPerBlock * bidx;

  volatile T* sendbuff = (T*)params.sendbuff + params.eltsPerBlock * bidx;
  volatile T* recvbuff = (T*)params.recvbuff + params.eltsPerBlock * bidx;

  // calculate element number that should be handle by this block.
  int32_t eltsPerBlock = params.eltsTotal - params.eltsPerBlock * bidx;
  if (eltsPerBlock <= 0) return;
  eltsPerBlock = min(eltsPerBlock, params.eltsPerBlock);

  int32_t eltsPerRank = divUp(eltsPerBlock, NRANKS);
  // split point for ranks must be aligned to 512B as we want every rank can do bulk ld/st first.
  eltsPerRank = ROUNDUP(eltsPerRank, PACKED_ELTS_PER_WARP);

  int32_t peerRankElts = eltsPerBlock - eltsPerRank * peerRank;
  peerRankElts = min(peerRankElts, eltsPerRank);

  int32_t peerRankExElts = peerRankElts % (PACKED_ELTS_PER_WARP);
  int32_t peerRankNpack = peerRankElts / PACKED_ELTS_PER_WARP;

  int32_t peerRankOffset = peerRank * eltsPerRank;
  int32_t peerBuffOffset = peerRank * buffEltsPerRank;
  int32_t myRankOffset = myRank * eltsPerRank;
  int32_t myBuffOffset = myRank * buffEltsPerRank;
  uint32_t barrierFlag = getBarrierFlag<NRANKS>(params.peerBarrierIn, params.myRank, tidx, bidx) + 1;

  __syncthreads();
  blockBarrier<NRANKS>(params.peerBarrierIn, barrierFlag, myRank, tidx, bidx);

#ifdef DEBUG_CUSTOM_ALL_REDUCE
  if (laneId == 0) printf("SC:rank:%2d, <%4d, %4d>, peer:%2d, offset:%4d, eltsTotal:%4d, eltsPerRank:%4d, exElts:%4d, Npack:%4d, warpGroupSize:%4d\n", myRank, bidx, tidx, peerRank, peerRankOffset, peerRankElts, eltsPerRank, peerRankExElts, peerRankNpack, warpGroupSize);
#endif

  volatile T* src = sendbuff + peerRankOffset;
  volatile T* dst = peerBuf + myBuffOffset;

  if (INPUT_ALIGN) {
    if (peerRank == myRank) memcpyPackWise<T, 1, 1>(dst, src, peerRankNpack, peerRankExElts, warpGroupSize, widInGroup, laneId);
    else memcpyPackWise<T, 1, 0>(dst, src, peerRankNpack, peerRankExElts, warpGroupSize, widInGroup, laneId);
  } else {
    if (peerRank == myRank) memcpyElementWise<T, 1, 1>(dst, src, peerRankElts, warpGroupSize, widInGroup, laneId);
    else memcpyElementWise<T, 1, 0>(dst, src, peerRankElts, warpGroupSize, widInGroup, laneId);
  }

  __syncthreads();
  blockBarrier<NRANKS>(params.peerBarrierPre, barrierFlag, myRank, tidx, bidx);

  int32_t myRankElts = eltsPerBlock - myRank * eltsPerRank;
  myRankElts = min(myRankElts, eltsPerRank);

  int32_t myRankExElts = myRankElts % (PACKED_ELTS_PER_WARP);
  int32_t myRankNpack = myRankElts / PACKED_ELTS_PER_WARP;

  reduce<T, NRANKS, 0>(myBuf, NULL, myRank, buffEltsPerRank, myRankNpack, nWarps, myRankExElts, wid, laneId);
  if (USE_DOUBLE_BUFFER) {
    __syncthreads();  // wait pre reduce done.

#ifdef DEBUG_CUSTOM_ALL_REDUCE
    if (laneId == 0) printf("R1:rank:%2d, <%4d, %4d>, peer:%2d, offset:%4d, eltsTotal:%4d, eltsPerRank:%4d, exElts:%4d, Npack:%4d, warpGroupSize:%4d\n", myRank, bidx, tidx, peerRank, myRankOffset, myRankElts, eltsPerRank, myRankExElts, myRankNpack, nWarps);
#endif

    // every warpGroup read the same data but send to different peer;
    volatile T* src = myBuf + myBuffOffset;
    volatile T* dst = NULL;
    if (peerRank == myRank) {
      dst = recvbuff + myRankOffset;
      if (OUTPUT_ALIGN) memcpyPackWise<T, 1, 1>(dst, src, myRankNpack, myRankExElts, warpGroupSize, widInGroup, laneId);
      else memcpyElementWise<T, 1, 1>(dst, src, myRankElts, warpGroupSize, widInGroup, laneId);
    } else {
      dst = peerBufD + myBuffOffset;
      memcpyPackWise<T, 1, 0>(dst, src, myRankNpack, myRankExElts, warpGroupSize, widInGroup, laneId);
    }
    // sync threads to make sure all block threads have the sums
    // keep this sync as we must ensure all threads complete the calculation when previous several thread write flag out(we need copy remote to local later)
    __syncthreads();
    blockBarrier<NRANKS>(params.peerBarrierReduce, barrierFlag, myRank, tidx, bidx);
    #ifdef DEBUG_CUSTOM_ALL_REDUCE
      if (laneId == 0) printf("FI:rank:%2d, <%3d, %3d>, peer:%2d, offset:%5d, eltsTotal:%4d, eltsPerRank:%4d, exElts:%3d, Npack:%2d, warpGroupSize:%d\n", myRank, bidx, tidx, peerRank, peerRankOffset, peerRankElts, eltsPerRank, peerRankExElts, peerRankNpack, warpGroupSize);
    #endif

    dst = recvbuff + peerRankOffset;
    src = myBufD + peerBuffOffset;
    if (peerRank != myRank) {
      if (OUTPUT_ALIGN) memcpyPackWise<T, 1, 1>(dst, src, peerRankNpack, peerRankExElts, warpGroupSize, widInGroup, laneId);
      else memcpyElementWise<T, 1, 1>(dst, src, peerRankElts, warpGroupSize, widInGroup, laneId);
    }
  } else {
    __syncthreads();  // wait pre reduce done.
    blockBarrier<NRANKS>(params.peerBarrierReduce, barrierFlag, myRank, tidx, bidx);
    // every warpGroup read the same data but send to different peer;
    volatile T* src = myBuf + myBuffOffset;

    volatile T* dst = NULL;
    if (peerRank == myRank) {
      dst = recvbuff + myRankOffset;
      if (OUTPUT_ALIGN) memcpyPackWise<T, 1, 1>(dst, src, myRankNpack, myRankExElts, warpGroupSize, widInGroup, laneId);
      else memcpyElementWise<T, 1, 1>(dst, src, myRankElts, warpGroupSize, widInGroup, laneId);
    } else {
      dst = peerBuf + myBuffOffset;
      memcpyPackWise<T, 1, 0>(dst, src, myRankNpack, myRankExElts, warpGroupSize, widInGroup, laneId);
    }
    __syncthreads();
    blockBarrier<NRANKS>(params.peerBarrierPost, barrierFlag, myRank, tidx, bidx);
    #ifdef DEBUG_CUSTOM_ALL_REDUCE
      if (laneId == 0) printf("FI:rank:%2d, <%3d, %3d>, peer:%2d, offset:%5d, eltsTotal:%4d, eltsPerRank:%4d, exElts:%3d, Npack:%2d, warpGroupSize:%d\n", myRank, bidx, tidx, peerRank, peerRankOffset, peerRankElts, eltsPerRank, peerRankExElts, peerRankNpack, warpGroupSize);
    #endif

    src = myBuf + peerBuffOffset;
    dst = recvbuff + peerRankOffset;
    if (peerRank != myRank) {
      if (OUTPUT_ALIGN) memcpyPackWise<T, 1, 1>(dst, src, peerRankNpack, peerRankExElts, warpGroupSize, widInGroup, laneId);
      else memcpyElementWise<T, 1, 1>(dst, src, peerRankElts, warpGroupSize, widInGroup, laneId);
    }
  }
}

template <typename T, int NRANKS, int INPUT_ALIGN, int OUTPUT_ALIGN>
__global__ void pcclKernel_twoShotRegAllReduceKernel(struct twoShotRegDevParams params) {
  // The number of elements packed into one for comms
  static constexpr int PACKED_ELTS_PER_THREAD = PACKED_ELEMS_SIZE / sizeof(T);
  static constexpr int PACKED_ELTS_PER_WARP = PACKED_ELTS_PER_THREAD * WARP_SIZE;

  const int32_t bidx = blockIdx.x;
  const int32_t tidx = threadIdx.x;
  const int32_t laneId = getLaneId();
  const int32_t nWarps = blockDim.x / WARP_SIZE;
  const int32_t wid = tidx / WARP_SIZE;
  const int32_t warpGroupSize = nWarps / NRANKS;   // warpGroupSize: warp number for one peer.
  const int32_t peerRank = wid / warpGroupSize;
  const int32_t widInGroup = wid % warpGroupSize;  // warp offset in group
  const int32_t myRank = params.myRank;

  volatile T* myBuf = (T*)params.peerBuffers[myRank] + params.buffEltsPerBlock * bidx;
  volatile T* peerBuf = (T*)params.peerBuffers[peerRank] + params.buffEltsPerBlock * bidx;
  const int32_t buffEltsPerRank = params.buffEltsPerBlock / NRANKS;

  volatile T* sendbuff = (T*)params.sendbuff + params.eltsPerBlock * bidx;
  volatile T* recvbuff = (T*)params.recvbuff + params.eltsPerBlock * bidx;

  // calculate element number that should be handled by this block.
  int32_t eltsPerBlock = params.eltsTotal - params.eltsPerBlock * bidx;
  if (eltsPerBlock <= 0) return;
  eltsPerBlock = min(eltsPerBlock, params.eltsPerBlock);

  int32_t eltsPerRank = divUp(eltsPerBlock, NRANKS);
  // split point for ranks must be aligned to 512B as we want every rank can do bulk ld/st first.
  eltsPerRank = ROUNDUP(eltsPerRank, PACKED_ELTS_PER_WARP);

  int32_t peerRankElts = eltsPerBlock - eltsPerRank * peerRank;
  peerRankElts = min(peerRankElts, eltsPerRank);

  int32_t peerRankExElts = peerRankElts % (PACKED_ELTS_PER_WARP);
  int32_t peerRankNpack = peerRankElts / PACKED_ELTS_PER_WARP;

  int32_t peerRankOffset = peerRank * eltsPerRank;
  int32_t peerBuffOffset = peerRank * buffEltsPerRank;
  int32_t myRankOffset = myRank * eltsPerRank;
  int32_t myBuffOffset = myRank * buffEltsPerRank;
  uint32_t barrierFlag = getBarrierFlag<NRANKS>(params.peerBarrierIn, params.myRank, tidx, bidx) + 1;
  exchangePtrs<NRANKS>(params.peerRmtAddrs, params.sendbuffOffset, params.recvbuffOffset, params.sendbuffRmtAddrs, params.recvbuffRmtAddrs, myRank, tidx, bidx);

  __syncthreads();
  blockBarrier<NRANKS>(params.peerBarrierIn, barrierFlag, myRank, tidx, bidx);

#ifdef DEBUG_CUSTOM_ALL_REDUCE
  if (laneId == 0) printf("SC:rank:%2d, <%4d, %4d>, peer:%2d, offset:%4d, eltsTotal:%4d, eltsPerRank:%4d, exElts:%4d, Npack:%4d, warpGroupSize:%4d\n", myRank, bidx, tidx, peerRank, peerRankOffset, peerRankElts, eltsPerRank, peerRankExElts, peerRankNpack, warpGroupSize);
#endif

  volatile T* src = sendbuff + peerRankOffset;
  volatile T* dst = peerBuf + myBuffOffset;

  if (INPUT_ALIGN) {
    if (peerRank == myRank) memcpyPackWise<T, 1, 1>(dst, src, peerRankNpack, peerRankExElts, warpGroupSize, widInGroup, laneId);
    else memcpyPackWise<T, 1, 0>(dst, src, peerRankNpack, peerRankExElts, warpGroupSize, widInGroup, laneId);
  } else {
    if (peerRank == myRank) memcpyElementWise<T, 1, 1>(dst, src, peerRankElts, warpGroupSize, widInGroup, laneId);
    else memcpyElementWise<T, 1, 0>(dst, src, peerRankElts, warpGroupSize, widInGroup, laneId);
  }

  __syncthreads();
  blockBarrier<NRANKS>(params.peerBarrierPre, barrierFlag, myRank, tidx, bidx);

  int32_t myRankElts = eltsPerBlock - myRank * eltsPerRank;
  myRankElts = min(myRankElts, eltsPerRank);

  int32_t myRankExElts = myRankElts % (PACKED_ELTS_PER_WARP);
  int32_t myRankNpack = myRankElts / PACKED_ELTS_PER_WARP;

  reduce<T, NRANKS, 0>(myBuf, NULL, myRank, buffEltsPerRank, myRankNpack, nWarps, myRankExElts, wid, laneId);
  __syncthreads();  // wait pre reduce done.

  // every warpGroup read the same data but send to different peer;
  src = myBuf + myBuffOffset;
  dst = NULL;
  uint32_t blockOffset = bidx * NRANKS;
  if (peerRank == myRank) {
    dst = recvbuff + myRankOffset;
    if (OUTPUT_ALIGN) memcpyPackWise<T, 1, 1>(dst, src, myRankNpack, myRankExElts, warpGroupSize, widInGroup, laneId);
    else memcpyElementWise<T, 1, 1>(dst, src, myRankElts, warpGroupSize, widInGroup, laneId);
  } else {
    void* peerAddr = (void*)((uintptr_t)params.peerRmtAddrs[myRank] + (blockOffset + peerRank) * 2 * sizeof(uintptr_t));
    PackedUintptr_t peerRmtPtrs;
    peerRmtPtrs.upacked = __ppu_global_load_volatile_b32x4(peerAddr);
    // only recvbuff was registered for now, register sendbuff when necessary.
    dst = (T*)peerRmtPtrs.unpacked[1] + myRankOffset + params.eltsPerBlock * bidx;
    memcpyPackWise<T, 1, 0>(dst, src, myRankNpack, myRankExElts, warpGroupSize, widInGroup, laneId);
  }
  __syncthreads();
  blockBarrier<NRANKS>(params.peerBarrierOut, barrierFlag, myRank, tidx, bidx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// oneShotLL AllReduce Kernel: barrier-free low-latency protocol using epoch-based LL slots
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, int NRANKS, int INPUT_ALIGN, int OUTPUT_ALIGN>
__global__ void pcclKernel_oneShotLLAllReduceKernel(struct oneShotLLDevParams params) {
  constexpr int LL_PACK_SIZE = EXT_KERNEL_LL_PACK_SIZE;
  constexpr int EltPerPack = LL_PACK_SIZE / sizeof(T);
  using LLPack = typename PackedLLOn8Bytes<T>::Type;

  const int32_t bidx = blockIdx.x;
  const int32_t tidx = threadIdx.x;
  const int32_t nThreads = blockDim.x;
  const int32_t myRank = params.myRank;

  void** peerLLBufs = params.peerLLBufferPtrs;

  // Load persistent epoch (gap of 2 from last used epoch)
  uint32_t llEpoch = params.llEpochs[bidx] + 2;

  // Per-block work range
  const int32_t eltsOffset = params.eltsPerBlock * bidx;
  int32_t eltsPerBlock = params.eltsTotal - eltsOffset;
  if (eltsPerBlock <= 0) return;
  eltsPerBlock = min(eltsPerBlock, (int32_t)params.eltsPerBlock);

  int nPacks = (eltsPerBlock + EltPerPack - 1) / EltPerPack;
  T* input = (T*)params.sendbuff + eltsOffset;
  T* output = (T*)params.recvbuff + eltsOffset;

  #pragma unroll 1
  while (nPacks > 0) {
    int nIterPacks = min(nPacks, nThreads);

    if (tidx < nIterPacks) {
      LLPack inp;
      llLoadPack<T, INPUT_ALIGN, EltPerPack>(inp, input, tidx, eltsPerBlock);

      // ---- bcastLL: write to ALL peers' LL buffers ----
      int slot = nIterPacks * myRank + tidx;
      #pragma unroll
      for (int r = 0; r < NRANKS; r++) {
        uint4* buf = extGetLLBuf<NRANKS>(peerLLBufs, r, bidx, llEpoch);
        LLStore(buf + slot, inp.u32[0], inp.u32[1], llEpoch);
      }

      // ---- recvReduceLL: read from own LL buffer and reduce ----
      uint4* myBuf = extGetLLBuf<NRANKS>(peerLLBufs, myRank, bidx, llEpoch);
      LLPack acc;
      #pragma unroll
      for (int r = 0; r < NRANKS; r++) {
        int readSlot = r * nIterPacks + tidx;
        uint4 tmp;
        // Spin until epoch flags match (both .y and .w must equal llEpoch)
        while (true) {
          LLLoad(tmp, myBuf + readSlot);
          if (tmp.y == llEpoch && tmp.w == llEpoch) break;
        }
        LLPack val;
        val.u32[0] = tmp.x;
        val.u32[1] = tmp.z;
        if (r == 0) {
          acc.packed = val.packed;
        } else {
          addLLPack(acc, val);
        }
      }

      // ---- Store 8 bytes result to recvbuff ----
      if (OUTPUT_ALIGN) {
        // Pack into lower 8 bytes of a uint4 for Store128
        // For LL we only have 8 bytes, write directly
        *(uint2*)(output + tidx * EltPerPack) = acc.packed;
      } else {
        // Safe element-wise store for unaligned case
        T* dst = output + tidx * EltPerPack;
        T elts[EltPerPack];
        __builtin_memcpy(elts, &acc, sizeof(LLPack));
        #pragma unroll
        for (int e = 0; e < EltPerPack; e++) {
          if (tidx * EltPerPack + e < eltsPerBlock)
            dst[e] = elts[e];
        }
      }
    }

    // ---- endLL: handle rare epoch overflow (every ~4 billion iterations) ----
    if (__builtin_expect(llEpoch >= 0xFFFFFFFEu, false)) {
      __syncthreads();
      uint4* buf = extGetLLBuf<NRANKS>(peerLLBufs, myRank, bidx, llEpoch);
      int epochSize = devExtKernelLLEpochSize<NRANKS>();
      #pragma unroll 4
      for (int i = tidx; i * 16 < epochSize; i += nThreads) {
        uint4 zero = make_uint4(0, 0, 0, 0);
        Store128(zero, buf + i);
      }
    }
    __syncthreads();
    llEpoch += (llEpoch == 0xFFFFFFFFu) ? 3 : 1;

    input += nThreads * EltPerPack;
    output += nThreads * EltPerPack;
    nPacks -= nThreads;
    eltsPerBlock -= nThreads * EltPerPack;
  }

  // Store epoch back (only thread 0 per block)
  if (tidx == 0) {
    params.llEpochs[bidx] = llEpoch - 2;
  }
}

// Explicit template instantiations using nested macros like functions.cu
#define ONE_SHOT_FUNCS_IMPL_NAME(input_align, output_align, nRanks, dtype) \
  template __global__ void pcclKernel_oneShotAllReduceKernel<dtype, nRanks, input_align, output_align>(struct oneShotDevParams params);

#define ONE_SHOT_FUNCS_IMPL_ALIGN(input_align, output_align, nRanks, dtype) \
  ONE_SHOT_FUNCS_IMPL_NAME(input_align, output_align, nRanks, dtype)

#define ONE_SHOT_FUNCS_IMPL_RANK(nRanks, dtype) \
  ONE_SHOT_FUNCS_IMPL_ALIGN(0, 0, nRanks, dtype) \
  ONE_SHOT_FUNCS_IMPL_ALIGN(0, 1, nRanks, dtype) \
  ONE_SHOT_FUNCS_IMPL_ALIGN(1, 0, nRanks, dtype) \
  ONE_SHOT_FUNCS_IMPL_ALIGN(1, 1, nRanks, dtype)

#define ONE_SHOT_FUNCS_IMPL_DTYPE(dtype) \
  ONE_SHOT_FUNCS_IMPL_RANK(2, dtype) \
  ONE_SHOT_FUNCS_IMPL_RANK(4, dtype) \
  ONE_SHOT_FUNCS_IMPL_RANK(8, dtype) \
  ONE_SHOT_FUNCS_IMPL_RANK(16, dtype) \
  ONE_SHOT_FUNCS_IMPL_RANK(32, dtype)

// Generate all oneShot kernels
#define ONE_SHOT_FUNCS_IMPL() \
  ONE_SHOT_FUNCS_IMPL_DTYPE(half) \
  ONE_SHOT_FUNCS_IMPL_DTYPE(__ppu_bfloat16)  \
  ONE_SHOT_FUNCS_IMPL_DTYPE(float)


#define TWO_SHOT_FUNCS_IMPL_NAME(db ,input_align, output_align, nRanks, dtype) \
  template __global__ void pcclKernel_twoShotAllReduceKernel<dtype, nRanks, db, input_align, output_align>(struct twoShotDevParams params);

#define TWO_SHOT_FUNCS_IMPL_DB(db ,input_align, output_align, nRanks, dtype) \
  TWO_SHOT_FUNCS_IMPL_NAME(db ,input_align, output_align, nRanks, dtype)

#define TWO_SHOT_FUNCS_IMPL_ALIGN(input_align, output_align, nRanks, dtype) \
  TWO_SHOT_FUNCS_IMPL_DB(0, input_align, output_align, nRanks, dtype) \
  TWO_SHOT_FUNCS_IMPL_DB(1, input_align, output_align, nRanks, dtype)

#define TWO_SHOT_FUNCS_IMPL_RANK(nRanks, dtype) \
  TWO_SHOT_FUNCS_IMPL_ALIGN(0, 0, nRanks, dtype) \
  TWO_SHOT_FUNCS_IMPL_ALIGN(0, 1, nRanks, dtype) \
  TWO_SHOT_FUNCS_IMPL_ALIGN(1, 0, nRanks, dtype) \
  TWO_SHOT_FUNCS_IMPL_ALIGN(1, 1, nRanks, dtype)

#define TWO_SHOT_FUNCS_IMPL_DTYPE(dtype) \
  TWO_SHOT_FUNCS_IMPL_RANK(2,  dtype) \
  TWO_SHOT_FUNCS_IMPL_RANK(4,  dtype) \
  TWO_SHOT_FUNCS_IMPL_RANK(8,  dtype) \
  TWO_SHOT_FUNCS_IMPL_RANK(16, dtype) \
  TWO_SHOT_FUNCS_IMPL_RANK(32, dtype)

// Generate all oneShot kernels
#define TWO_SHOT_FUNCS_IMPL() \
  TWO_SHOT_FUNCS_IMPL_DTYPE(half) \
  TWO_SHOT_FUNCS_IMPL_DTYPE(__ppu_bfloat16)  \
  TWO_SHOT_FUNCS_IMPL_DTYPE(float)

#define TWO_SHOT_REG_FUNCS_IMPL_NAME(input_align, output_align, nRanks, dtype) \
  template __global__ void pcclKernel_twoShotRegAllReduceKernel<dtype, nRanks, input_align, output_align>(struct twoShotRegDevParams params);

#define TWO_SHOT_REG_FUNCS_IMPL_ALIGN(input_align, output_align, nRanks, dtype) \
  TWO_SHOT_REG_FUNCS_IMPL_NAME(input_align, output_align, nRanks, dtype)

#define TWO_SHOT_REG_FUNCS_IMPL_RANK(nRanks, dtype) \
  TWO_SHOT_REG_FUNCS_IMPL_ALIGN(0, 0, nRanks, dtype) \
  TWO_SHOT_REG_FUNCS_IMPL_ALIGN(0, 1, nRanks, dtype) \
  TWO_SHOT_REG_FUNCS_IMPL_ALIGN(1, 0, nRanks, dtype) \
  TWO_SHOT_REG_FUNCS_IMPL_ALIGN(1, 1, nRanks, dtype)

#define TWO_SHOT_REG_FUNCS_IMPL_DTYPE(dtype) \
  TWO_SHOT_REG_FUNCS_IMPL_RANK(2,  dtype) \
  TWO_SHOT_REG_FUNCS_IMPL_RANK(4,  dtype) \
  TWO_SHOT_REG_FUNCS_IMPL_RANK(8,  dtype) \
  TWO_SHOT_REG_FUNCS_IMPL_RANK(16, dtype) \
  TWO_SHOT_REG_FUNCS_IMPL_RANK(32, dtype)

// Generate all twoShotReg kernels
#define TWO_SHOT_REG_FUNCS_IMPL() \
  TWO_SHOT_REG_FUNCS_IMPL_DTYPE(half) \
  TWO_SHOT_REG_FUNCS_IMPL_DTYPE(__ppu_bfloat16)  \
  TWO_SHOT_REG_FUNCS_IMPL_DTYPE(float)

#define ONESHOT_LL_FUNCS_IMPL_NAME(input_align, output_align, nRanks, dtype) \
  template __global__ void pcclKernel_oneShotLLAllReduceKernel<dtype, nRanks, input_align, output_align>(struct oneShotLLDevParams params);

#define ONESHOT_LL_FUNCS_IMPL_ALIGN(input_align, output_align, nRanks, dtype) \
  ONESHOT_LL_FUNCS_IMPL_NAME(input_align, output_align, nRanks, dtype)

#define ONESHOT_LL_FUNCS_IMPL_RANK(nRanks, dtype) \
  ONESHOT_LL_FUNCS_IMPL_ALIGN(0, 0, nRanks, dtype) \
  ONESHOT_LL_FUNCS_IMPL_ALIGN(0, 1, nRanks, dtype) \
  ONESHOT_LL_FUNCS_IMPL_ALIGN(1, 0, nRanks, dtype) \
  ONESHOT_LL_FUNCS_IMPL_ALIGN(1, 1, nRanks, dtype)

#define ONESHOT_LL_FUNCS_IMPL_DTYPE(dtype) \
  ONESHOT_LL_FUNCS_IMPL_RANK(2, dtype) \
  ONESHOT_LL_FUNCS_IMPL_RANK(4, dtype) \
  ONESHOT_LL_FUNCS_IMPL_RANK(8, dtype) \
  ONESHOT_LL_FUNCS_IMPL_RANK(16, dtype) \
  ONESHOT_LL_FUNCS_IMPL_RANK(32, dtype)

// Generate all oneShotLL kernels
#define ONESHOT_LL_FUNCS_IMPL() \
  ONESHOT_LL_FUNCS_IMPL_DTYPE(half) \
  ONESHOT_LL_FUNCS_IMPL_DTYPE(__ppu_bfloat16) \
  ONESHOT_LL_FUNCS_IMPL_DTYPE(float)

TWO_SHOT_REG_FUNCS_IMPL()
TWO_SHOT_FUNCS_IMPL()
ONE_SHOT_FUNCS_IMPL()
ONESHOT_LL_FUNCS_IMPL()

////////////////////////////////////////////////////////////////////////////////////////////////////
static char* getExtFuncName(char* funcName, int maxLength, extKernelStrategyType strategy, pcclDataType_t dtype, int nRanks) {
  int offset = 0;
  if (strategy == ONESHOT || strategy == ONESHOT_REG) {
    offset += snprintf(funcName, maxLength-offset, "pcclKernel_oneShotAllReduceKernel");
  } else if (strategy == TWOSHOT || strategy == TWOSHOT_REG) {
    offset += snprintf(funcName, maxLength-offset, "pcclKernel_twoShotAllReduceKernel");
  } else if (strategy == ONESHOT_LL) {
    offset += snprintf(funcName, maxLength-offset, "pcclKernel_oneShotLLAllReduceKernel");
  }
  offset += snprintf(funcName+offset, maxLength-offset, "_%d", nRanks);
  if (dtype == pcclFloat32) {
    offset += snprintf(funcName+offset, maxLength-offset, "_Float32");
  } else if (dtype == pcclFloat16) {
    offset += snprintf(funcName+offset, maxLength-offset, "_Float16");
  } else if (dtype == pcclBfloat16) {
    offset += snprintf(funcName+offset, maxLength-offset, "_Bfloat16");
  }
  return funcName;
}


ncclResult_t dispatchAllReduceKernels(extKernelHostParams& hostParams, extKernelDevParams& devParams, int blocksPerGrid,
                                      int threadsPerBlock, hggcStream_t stream) {
  char funcName[128];
  void* func = NULL;
  // TODO: use the branch code to choose the template function, late add a more elegant way.
  int inputAlign = ((uintptr_t)(devParams.sendbuff) % 4 == 0 ? 1 : 0);
  int outputAlign = ((uintptr_t)(devParams.recvbuff) % 4 == 0 ? 1 : 0);
  int localIndex = extKernelGetFuncIndex(hostParams.strategy, hostParams.dtype, hostParams.nRanks, inputAlign, outputAlign, hostParams.doubleBuffer);
  if (localIndex < 0 || localIndex >= AR_EXT_FUNCS_NUM) {
    ERROR("[EXT] Unknown funcIndex: %d, legal range[%d, %d)", localIndex, 0, AR_EXT_FUNCS_NUM);
    return ncclInternalError;
  }
  hostParams.funcIndex = EXT_FUNC_AR_BASE + localIndex;
  func = customKernels[hostParams.funcIndex];

  if (func != NULL) {
    NCCLCHECK(extKernelSetDispatchStrategy((const void*)(func), getExtFuncName(funcName, 128, hostParams.strategy, hostParams.dtype, hostParams.nRanks), hostParams, blocksPerGrid, ncclFuncAllReduce));
    dim3 gridDim = blocksPerGrid;
    dim3 blockDim = threadsPerBlock;
    switch (hostParams.strategy) {
      case ONESHOT:
      case ONESHOT_REG: {
        struct oneShotDevParams arg(devParams);
        void* args[] = {&arg};
        LAUNCH_KERNEL(func, gridDim, blockDim, args, 0, stream, 0, hostParams.funcStrategy[hostParams.funcIndex].launchConfig);
        break;
      }
      case TWOSHOT: {
        struct twoShotDevParams arg(devParams);
        void* args[] = {&arg};
        // TODO: analyze the performance drop reason after enable sub dispatch
        LAUNCH_KERNEL(func, gridDim, blockDim, args, 0, stream, 0, hostParams.funcStrategy[hostParams.funcIndex].launchConfig);
        break;
      }
      case TWOSHOT_REG: {
        struct twoShotRegDevParams arg(devParams);
        void* args[] = {&arg};
        LAUNCH_KERNEL(func, gridDim, blockDim, args, 0, stream, 0, hostParams.funcStrategy[hostParams.funcIndex].launchConfig);
        break;
      }
      case ONESHOT_LL: {
        struct oneShotLLDevParams arg(devParams);
        void* args[] = {&arg};
        LAUNCH_KERNEL(func, gridDim, blockDim, args, 0, stream, 0, hostParams.funcStrategy[hostParams.funcIndex].launchConfig);
        break;
      }
      default: {
        ERROR("[EXT]: invalid func");
        return ncclInvalidUsage;
      }
    }
  }

  return ncclSuccess;
}

ncclResult_t customAllReduceImpl(struct extKernelDevParams& devParams, struct extKernelHostParams& hostParams, hggcStream_t stream) {
  size_t eltsPerThread = PACKED_ELEMS_SIZE / hostParams.dtypeSize;
  int blocksPerGrid = 1;
  int threadsPerBlock = WARP_SIZE;
  NCCLCHECK(kernelLaunchConfig(devParams, hostParams, eltsPerThread, &blocksPerGrid, &threadsPerBlock));
  NCCLCHECK(dispatchAllReduceKernels(hostParams, devParams, blocksPerGrid, threadsPerBlock, stream));
  return ncclSuccess;
}
