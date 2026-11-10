/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "devcomm.h"
#include "collectives.h"
#include "primitives.h"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRing(struct ncclDevWorkColl* work) {
    const int tid = threadIdx.x;
    int nworkers;
    if (Proto::Id == NCCL_PROTO_LL || Proto::Id == NCCL_PROTO_LL128) {
      nworkers = work->nThreads;
    } else {
#ifndef NO_EXTRA_WARP
      nworkers = work->nThreads-WARP_SIZE;
#else
      nworkers = work->nThreads;
#endif
    }
    const int nthreads = ncclPrimsThreadsFromWorkers<Proto, 1>(nworkers);
    const int nChannels = work->nChannels;
    int bid = (blockIdx.x >= work->startChannelId) ?
      (blockIdx.x - work->startChannelId) :
      (blockIdx.x + work->commNChannels - work->startChannelId);
    bid = bid % nChannels;
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    struct ncclRing* ring = &channel->ring;
    const int nranks = comm->nRanks;
    const int* ringRanks = ring->devUserRanks;
    int stepSize;
    ssize_t chunkSize;
    ssize_t minChunkSize;
    if (Proto::Id == NCCL_PROTO_LL) {
      stepSize = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
      chunkSize = stepSize * sizeof(uint64_t) / sizeof(T);
    } else if (Proto::Id == NCCL_PROTO_LL128) {
      stepSize = comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
      chunkSize = stepSize*NCCL_LL128_DATAELEMS*sizeof(uint64_t) / (NCCL_LL128_LINEELEMS*sizeof(T));
      // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
      minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nworkers*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2;
    } else {
      stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
      chunkSize = stepSize * REDUCESCATTER_CHUNKSTEPS;
    }
    const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
    const ssize_t count = work->count;
    ssize_t offset;
    ssize_t dataOffset;
    int nelem;
    int rankDest;

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)work->sendbuff;
    T * __restrict__ thisOutput = (T*)work->recvbuff;

    Primitives<T, RedOp, FanAsymmetric<1, 1>, 0, Proto, 0>
      prims(tid, nthreads, &ring->prev, &ring->next, NULL, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);
    prims.setDataPtrs(thisInput, thisOutput);

    for (ssize_t gridOffset = 0; gridOffset < count; gridOffset += loopSize) {
      int realChunkSize;
      if (Proto::Id == NCCL_PROTO_LL) {
        if (count-gridOffset < loopSize) {
          chunkSize = work->lastChunkSize;
        }
        realChunkSize = chunkSize;
      } else if (Proto::Id == NCCL_PROTO_LL128) {
        chunkSize = min(DIVUP(count-gridOffset, nChannels*minChunkSize)*minChunkSize, chunkSize);
        realChunkSize = chunkSize;
      } else {
        realChunkSize = min(chunkSize, DIVUP(count-gridOffset,nChannels));
        ALIGN_SIZE(realChunkSize, nworkers*sizeof(uint64_t)/sizeof(T));
      }
      dataOffset = gridOffset + bid*realChunkSize;
      nelem = (int)min(realChunkSize, count-dataOffset);
      /////////////// begin ReduceScatter steps ///////////////
      // step 0: push data to next GPU
      rankDest = ringRanks[nranks-1];
      offset = dataOffset + rankDest * count;

      prims.send(offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        rankDest = ringRanks[nranks-j];
        offset = dataOffset + rankDest * count;

        prims.recvReduceSend(offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final result
      rankDest = ringRanks[0];
      offset = dataOffset + rankDest * count;

      prims.recvReduceCopy(offset, dataOffset, nelem, /*postOp=*/true);
    }
  }

  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runFC(struct ncclDevWorkColl* work) {
    const int tid = threadIdx.x;
    int nworkers;
    if (Proto::Id == NCCL_PROTO_LL || Proto::Id == NCCL_PROTO_LL128) {
      nworkers = work->nThreads;
    } else {
#ifndef NO_EXTRA_WARP
      nworkers = work->nThreads - WARP_SIZE;
#else
      nworkers = work->nThreads;
#endif
    }
    const int nthreads = ncclPrimsThreadsFromWorkers<Proto, NCCL_MAX_FC_ARITY-1>(nworkers);
    const int nChannels = work->nChannels;
    int bid = (blockIdx.x >= work->startChannelId) ?
      (blockIdx.x - work->startChannelId) :
      (blockIdx.x + work->commNChannels - work->startChannelId);
    bid = bid % nChannels;
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    struct ncclFC* fc = &channel->fc;

    const int rank = comm->rank;
    const int nranks = comm->nRanks;
    int stepSize;
    ssize_t chunkSize;
    ssize_t minChunkSize;
    if (Proto::Id == NCCL_PROTO_LL) {
      stepSize = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
      chunkSize = stepSize * sizeof(uint64_t) / sizeof(T);
    } else if (Proto::Id == NCCL_PROTO_LL128) {
      stepSize = comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
      chunkSize = stepSize*NCCL_LL128_DATAELEMS*sizeof(uint64_t) / (NCCL_LL128_LINEELEMS*sizeof(T));
      minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nworkers*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T));
    } else {
      stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T) * NCCL_STEPS);
      chunkSize = stepSize * REDUCESCATTER_CHUNKSTEPS;
    }
    const ssize_t loopSize = nChannels * chunkSize;
    const ssize_t size = work->count;

    const T* __restrict__ thisInput = (const T*)work->sendbuff;
    T* __restrict__ thisOutput = (T*)work->recvbuff;

    Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_FC_ARITY-1, NCCL_MAX_FC_ARITY-1>, 0, Proto, 0>
      prims(tid, nthreads, fc->peerRanks, fc->peerRanks, NULL, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);
    prims.setDataPtrs(thisInput, thisOutput);

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nChannels));
      ALIGN_SIZE(realChunkSize, nworkers*sizeof(uint64_t)/sizeof(T));
      ssize_t chunkOffset = gridOffset + bid * realChunkSize;
      int nelem = min(realChunkSize, size-chunkOffset);
      int totalElem = nelem * nranks;
      prims.scatter(chunkOffset, totalElem, nelem, size);
      prims.recvReduceCopy(rank*size+chunkOffset, chunkOffset, nelem, /*postOp=*/true);
    }
  }
}

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduceScatter, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      using Proto = ProtoSimple<REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS, REDUCESCATTER_SLICESTEPS, UNROLL>;
      runRing<T, RedOp, Proto>(work);
    }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduceScatter, NCCL_ALGO_FC, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      using Proto = ProtoSimple<REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS, REDUCESCATTER_SLICESTEPS, UNROLL>;
      runFC<T, RedOp, Proto>(work);
    }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduceScatter, NCCL_ALGO_RING, NCCL_PROTO_LL, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      runRing<T, RedOp, ProtoLL>(work);
    }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduceScatter, NCCL_ALGO_FC, NCCL_PROTO_LL, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      runFC<T, RedOp, ProtoLL>(work);
    }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduceScatter, NCCL_ALGO_RING, NCCL_PROTO_LL128, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      runRing<T, RedOp, ProtoLL128>(work);
    }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduceScatter, NCCL_ALGO_RING, NCCL_PROTO_SIGNAL, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      using Proto = ProtoSignal<REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS, REDUCESCATTER_SLICESTEPS, UNROLL>;
      runRing<T, RedOp, Proto>(work);
    }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduceScatter, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE_BULK, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      struct ncclKernelComm* comm = getShmemComm(work);
      struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
      const ncclConn_t sendConnType = channel->ring.sendConnType;
      const ncclConn_t recvConnType = channel->ring.recvConnType;

      if (sendConnType == CONN_ICN) {
        using Proto = ProtoSimpleBulk<REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS, REDUCESCATTER_SLICESTEPS, UNROLL, CONN_ICN, CONN_ICN>;
        runRing<T, RedOp, Proto>(work);
      } else {
        using Proto = ProtoSimpleBulk<REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS, REDUCESCATTER_SLICESTEPS, UNROLL, CONN_NORMAL, CONN_ICN>;
        runRing<T, RedOp, Proto>(work);
      }
    }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduceScatter, NCCL_ALGO_FC, NCCL_PROTO_SIMPLE_BULK, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      struct ncclKernelComm* comm = getShmemComm(work);
      struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
      const ncclConn_t connType = channel->fc.connType;
      if (connType != CONN_ICN) {
        using Proto = ProtoSimpleBulk<REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS, REDUCESCATTER_SLICESTEPS, UNROLL, CONN_NORMAL, CONN_ICN>;
        runFC<T, RedOp, Proto>(work);
      } else {
        using Proto = ProtoSimpleBulk<REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS, REDUCESCATTER_SLICESTEPS, UNROLL, CONN_ICN, CONN_ICN>;
        runFC<T, RedOp, Proto>(work);
      }
    }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduceScatter, NCCL_ALGO_RING, NCCL_PROTO_SIGNAL_BULK, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      using Proto = ProtoSignalBulk<REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS, REDUCESCATTER_SLICESTEPS, UNROLL>;
      runRing<T, RedOp, Proto>(work);
    }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduceScatter, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE_BULK_AR, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
      using Proto = ProtoSimpleBulkAR<REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS, REDUCESCATTER_SLICESTEPS, UNROLL, CONN_ICN, CONN_ICN>;
      runRing<T, RedOp, Proto>(work);
#endif
    }
};

#if defined(__HGGC_ARCH__)
namespace {
  template<class RedOp, typename T, int Unroll>
  __device__ __forceinline__ void runPAT(struct ncclDevWorkColl* work) {
    struct ncclKernelComm* comm = getShmemComm(work);
    const int nranks = comm->nRanks;
    const int rank = comm->rank;

    const int tid = threadIdx.x;
    const int nthreads = work->nThreads;

    const int nChannels = work->nChannels;
    int bid = (blockIdx.x >= work->startChannelId) ?
      (blockIdx.x - work->startChannelId) :
      (blockIdx.x + work->commNChannels - work->startChannelId);
    bid = bid % nChannels;

    const ssize_t count = work->count;
    ssize_t channelCount = (count + nChannels - 1) / nChannels;
    ssize_t channelOffset = bid * channelCount;
    if (channelOffset + channelCount > count) channelCount = count - channelOffset;

    const int chunkCount = work->lastChunkSize;

    static constexpr int nworkers = NCCL_PAT_NWORKERS;
    struct ncclPatShmem* shmem = ncclPatShmemPtr(comm);
    uint64_t pollCount = 0;

    __syncthreads();
    for (int i = tid; i < NCCL_SHMEM_PAT_STEPS; i += nthreads) shmem->patSteps[i].flags = 0;
    if (tid == 0) shmem->localAccSize = 0;
    if (tid == nworkers) shmem->parallelFactor = 0;
    __syncthreads();

    if (tid == nworkers) { // Algo computation thread
      PatRSAlgorithm<T> patAlgo(chunkCount*sizeof(T), NCCL_STEPS, NCCL_PAT_NWORKERS/WARP_SIZE, channelOffset, channelOffset + channelCount, count, chunkCount, rank, nranks);
      shmem->parallelFactor = patAlgo.getParallelFactor();
      int step = 0;
      while (1) {
        struct ncclPatStep* ps = shmem->patSteps+(step%NCCL_SHMEM_PAT_STEPS);
        volatile int* poll = &ps->flags;
        while (patStepFlagsLoadAcquire(poll) != 0) pollCount++; // Wait for workers to be done with step 'step-NCCL_SHMEM_PAT_STEPS'
        patAlgo.getNextOp(ps);
        int last = ps->last;
        step++;
        if (last == 2) break;
      }
    } else if (tid < nworkers) { // Worker threads
      T* inputBuf = (T*)work->sendbuff;
      T* outputBuf = (T*)work->recvbuff;
      int parallelFactor = 0;
      volatile int* pfPtr = &shmem->parallelFactor;
      while (parallelFactor == 0) parallelFactor = *pfPtr;

      int groupSize = nworkers/(WARP_SIZE*parallelFactor) * WARP_SIZE;
      int group = tid / groupSize;
      int nGroups = nworkers / groupSize;
      int tidInGroup = tid - group*groupSize;
      // We don't use recvPeers/sendPeers so let's pass shmem structs instead
      Primitives<T, RedOp, FanAsymmetric<1, 1>, 0, ProtoSimple<1, 1, Unroll>, 0> prims
        (tidInGroup, groupSize, (int*)shmem->recvDims, (int*)shmem->sendDims,
         nullptr /*directBuff*/, comm, work->redOpArg, getShmemPtr(comm)->groups,
         group, 0, 0, nullptr, nullptr, 0 /*stepSize_*/, NCCL_STEPS, false,
         inputBuf, outputBuf, primsModePatRs);

      int step = group;
      while (1) {
        struct ncclPatStep* ps = shmem->patSteps+(step%NCCL_SHMEM_PAT_STEPS);
        volatile int* poll = &ps->flags;
        while (patStepFlagsLoadAcquire(poll) == 0) pollCount++; // Wait for compute thread
        int last = ps->last;
        prims.patReduce(ps, shmem);
        if (tidInGroup == 0) patStepFlagsStoreRelease(poll, 0); // Return element to compute thread
        if (last) break;
        step += nGroups;
      }
    }
  }
}

template<class RedOp, typename T, int Unroll>
class RunWorkColl<ncclFuncReduceScatter, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE, RedOp, T, Unroll> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      runPAT<RedOp, T, Unroll>(work);
    }
};

#endif

template<int PROTO, typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduceScatter, NCCL_ALGO_TREE, PROTO, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {}
};

template<int PROTO, typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduceScatter, NCCL_ALGO_COLLNET, PROTO, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {}
};
