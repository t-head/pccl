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
#include "customAllGatherKernels.h"
#include "customKernels.h"
#include "customDeviceUtils.h"
#include "customDeviceUtils.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////////
// OneShot AllGather Kernel
// Phase 1: Broadcast sendbuff to ALL peers' buffers at myRank offset.
//          Local copy (r==0) uses BulkLoad/BulkStore; remote writes use RemoteBulkStore.
// Phase 2: Gather all ranks' data from the LOCAL buffer (srcPtrs[0]) into recvbuff.
//          After the barrier every rank's data is present locally, so all reads are local.
// PPU constraint: RemoteBulk ops on local addresses cause illegal memory access.
// NOTE: pluginBuildParams adjusts sendbuff to recvbuff + myRank*perRankBytes (in-place
// position), so sendbuff already points to this rank's per-rank input slice.
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, int NRANKS, int INPUT_ALIGN, int OUTPUT_ALIGN>
__global__ void pcclKernel_oneShotAllGatherKernel(struct oneShotDevParams params) {
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

  // NCCL passes sendbuff as the per-rank slice; eltsOffset/NRANKS selects this
  // block's range; params.myRank * eltsPerRank selects this rank's data.
  // sendbuff is computed after eltsPerRank is defined (see below)
  volatile T* recvbuff = (T*)params.recvbuff;

  int32_t eltsPerBlock = (eltsOffset >= params.eltsTotal)
      ? 0 : min((int32_t)(params.eltsTotal - eltsOffset), (int32_t)params.eltsPerBlock);

  // Phase 1: Write sendbuff to LOCAL peer buffer at myRank offset.
  // pluginBuildParams/pluginExecute adjust sendbuff to recvbuff + myRank*perRankBytes,
  // so sendbuff already points to this rank's per-rank input slice. No extra rank offset needed.
  int32_t buffEltsPerRank = params.buffEltsPerBlock / NRANKS;
  int32_t eltsPerRank = eltsPerBlock / NRANKS;
  volatile T* sendbuff = (T*)params.sendbuff + eltsOffset / NRANKS;
  int32_t myBuffOffset = params.myRank * buffEltsPerRank;

  // Phase 1: Broadcast sendbuff to ALL peers' buffers at myRank offset.
  // Load each 128-bit pack from sendbuff once, fan out to all N peer buffers.
  // dstPtrs[0] is local (BulkStore), dstPtrs[1..N-1] are remote (RemoteBulkStore).
  {
    int32_t copyNpack = eltsPerRank / PACKED_ELTS_PER_WARP;
    int32_t copyExElts = eltsPerRank % PACKED_ELTS_PER_WARP;
    volatile T* bcastDsts[NRANKS];
    #pragma unroll
    for (int r = 0; r < NRANKS; r++)
      bcastDsts[r] = srcPtrs[r] + myBuffOffset;

    if (INPUT_ALIGN)
      broadcastToAllPeers<T, NRANKS>(bcastDsts, sendbuff, copyNpack, copyExElts, nWarps, wid, laneId);
    else
      broadcastToAllPeersElemWise<T, NRANKS>(bcastDsts, sendbuff, eltsPerRank, nWarps, wid, laneId);
  }

  uint32_t barrierFlag = getBarrierFlag<NRANKS>(params.peerBarrierIn, params.myRank, tidx, bidx) + 1;

  __syncthreads();
  blockBarrier<NRANKS>(params.peerBarrierIn, barrierFlag, params.myRank, tidx, bidx, params.abortFlag);

  // Phase 2: gather from local buffer to recvbuff (warps grouped by peer rank).
  // After barrier, srcPtrs[0] contains all ranks' data. Each warp group
  // handles one rank's slice in parallel (same pattern as twoShotAllReduceKernel).
  // All reads are local — avoids RemoteBulk loads that fault on PPU.
  int32_t totalPerRank = params.eltsTotal / NRANKS;
  int32_t perRankOffset = eltsOffset / NRANKS;

  const int32_t warpGroupSize = nWarps / NRANKS;
  const int32_t peerRank      = wid / warpGroupSize;
  const int32_t widInGroup    = wid % warpGroupSize;

  int32_t peerRankElts = eltsPerBlock - eltsPerRank * peerRank;
  peerRankElts = min(peerRankElts, eltsPerRank);
  int32_t peerRankNpack  = peerRankElts / PACKED_ELTS_PER_WARP;
  int32_t peerRankExElts = peerRankElts % PACKED_ELTS_PER_WARP;

  if (OUTPUT_ALIGN)
    memcpyPackWise<T, 1, 1>(
        recvbuff + (int32_t)peerRank * totalPerRank + perRankOffset,
        srcPtrs[0] + (int32_t)peerRank * buffEltsPerRank,
        peerRankNpack, peerRankExElts, warpGroupSize, widInGroup, laneId);
  else
    memcpyElementWise<T, 1, 1>(
        recvbuff + (int32_t)peerRank * totalPerRank + perRankOffset,
        srcPtrs[0] + (int32_t)peerRank * buffEltsPerRank,
        peerRankElts, warpGroupSize, widInGroup, laneId);

  // Make sure every rank is done reading this round before the next round
  // starts overwriting the shared staging buffers.
  __syncthreads();
  blockBarrier<NRANKS>(params.peerBarrierOut, barrierFlag, params.myRank, tidx, bidx, params.abortFlag);
}

// Template instantiations (macros defined in header)
AG_ONE_SHOT_FUNCS_IMPL()

////////////////////////////////////////////////////////////////////////////////////////////////////
// AllGather dispatch and impl
////////////////////////////////////////////////////////////////////////////////////////////////////
// AllGather reuses extKernelStrategyType — strategy enum is defined in customAllReduceKernels.h.
static char* getAllGatherExtFuncName(char* funcName, int maxLength, int nRanks) {
  int offset = 0;
  offset += snprintf(funcName, maxLength-offset, "pcclKernel_oneShotAllGatherKernel");
  offset += snprintf(funcName+offset, maxLength-offset, "_%d_Int8", nRanks);
  return funcName;
}

ncclResult_t dispatchAllGatherKernels(extKernelHostParams& hostParams, extKernelDevParams& devParams, int blocksPerGrid,
                                      int threadsPerBlock, hggcStream_t stream) {
  char funcName[128];
  int inputAlign = ((uintptr_t)(devParams.sendbuff) % RMT_BULK_ALIGN_SIZE == 0 ? 1 : 0);
  int outputAlign = ((uintptr_t)(devParams.recvbuff) % RMT_BULK_ALIGN_SIZE == 0 ? 1 : 0);
  // The bulk copy path (memcpyPackWise) requires at least one
  // full pack-warp per rank.  When eltsPerRank < PACKED_ELTS_PER_WARP the Npack
  // count drops to zero and only a single lane of a single warp handles the
  // excess elements, silently dropping data.  Force the element-wise path for
  // these small sizes.
  {
    uint32_t eltsPerRank = devParams.eltsTotal / hostParams.nRanks;
    uint32_t packedPerWarp = (PACKED_ELEMS_SIZE / hostParams.dtypeSize) * WARP_SIZE;
    if (eltsPerRank < packedPerWarp) {
      inputAlign = 0;
      outputAlign = 0;
    }
  }
  int localIndex = getAllGatherKernelIndex(hostParams.nRanks, inputAlign, outputAlign);

  if (localIndex < 0 || localIndex >= AG_ONESHOT_FUNCS_NUM) {
    ERROR("[EXT] AllGather: Unknown funcIndex: %d, legal range[0, %d)", localIndex, AG_ONESHOT_FUNCS_NUM);
    return ncclInternalError;
  }
  hostParams.funcIndex = EXT_FUNC_AG_BASE + localIndex;
  void* func = customKernels[hostParams.funcIndex];

  if (func != NULL) {
    NCCLCHECK(extKernelSetDispatchStrategy((const void*)(func), getAllGatherExtFuncName(funcName, 128, hostParams.nRanks), hostParams, blocksPerGrid, ncclFuncAllGather));
    dim3 gridDim = blocksPerGrid;
    dim3 blockDim = threadsPerBlock;
    struct oneShotDevParams arg(devParams);
    void* args[] = {&arg};
    LAUNCH_KERNEL(func, gridDim, blockDim, args, 0, stream, 0, hostParams.funcStrategy[hostParams.funcIndex].launchConfig);
  }

  return ncclSuccess;
}

ncclResult_t customAllGatherImpl(struct extKernelDevParams& devParams, struct extKernelHostParams& hostParams, hggcStream_t stream) {
  size_t eltsPerThread = PACKED_ELEMS_SIZE / hostParams.dtypeSize;
  int blocksPerGrid = 1;
  int threadsPerBlock = WARP_SIZE;
  NCCLCHECK(kernelLaunchConfig(devParams, hostParams, eltsPerThread, &blocksPerGrid, &threadsPerBlock));

  // Phase 2 warps-grouped-by-rank pattern requires nWarps >= NRANKS.
  int minThreads = hostParams.nRanks * WARP_SIZE;
  if (threadsPerBlock < minThreads) {
    threadsPerBlock = minThreads;
    devParams.eltsPerBlock = divUp(devParams.eltsTotal, blocksPerGrid);
    devParams.eltsPerBlock = ROUNDUP(devParams.eltsPerBlock, eltsPerThread * threadsPerBlock);
    devParams.eltsPerBlock = std::min(devParams.eltsPerBlock, (uint32_t)devParams.eltsTotal);
  }

  // Ensure eltsPerBlock is a multiple of NRANKS for exact per-rank partition
  devParams.eltsPerBlock = ROUNDUP(devParams.eltsPerBlock, hostParams.nRanks);
  devParams.eltsPerBlock = std::min(devParams.eltsPerBlock, (uint32_t)devParams.eltsTotal);

  NCCLCHECK(dispatchAllGatherKernels(hostParams, devParams, blocksPerGrid, threadsPerBlock, stream));

  return ncclSuccess;
}
