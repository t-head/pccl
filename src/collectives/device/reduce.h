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
    const int rank = ringRanks[0];
    const int prevRank = ringRanks[nranks-1];
    const int root = work->root;
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
      stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
      chunkSize = stepSize * REDUCE_CHUNKSTEPS;
    }
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = work->count;
    ssize_t offset;
    int nelem;

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)work->sendbuff;
    T * __restrict__ thisOutput = (T*)work->recvbuff;

    Primitives<T, RedOp, FanAsymmetric<1, 1>, 0, Proto, 0>
      prims(tid, nthreads, &ring->prev, &ring->next, NULL, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);
    prims.setDataPtrs(thisInput, thisOutput);

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      int realChunkSize;
      if (Proto::Id == NCCL_PROTO_LL) {
        if (size-gridOffset < loopSize) {
          chunkSize = work->lastChunkSize;
        }
        realChunkSize = chunkSize;
      } else if (Proto::Id == NCCL_PROTO_LL128) {
        chunkSize = min(DIVUP(size-gridOffset, nChannels*minChunkSize)*minChunkSize, chunkSize);
        realChunkSize = chunkSize;
      } else {
        realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nChannels));
        ALIGN_SIZE(realChunkSize, nworkers*sizeof(uint64_t)/sizeof(T));
      }
      offset = gridOffset + bid*realChunkSize;
      nelem = (int)min(realChunkSize, size-offset);
      if (prevRank == root) {
        prims.send(offset, nelem);
      } else if (rank == root) {
        prims.recvReduceCopy(offset, offset, nelem, /*postOp=*/true);
      } else {
        prims.recvReduceSend(offset, nelem);
      }
    }
  }


  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runFC(struct ncclDevWorkColl* work) {
    const int tid = threadIdx.x;
    const int nChannels = work->nChannels;
    int bid = (blockIdx.x >= work->startChannelId) ?
      (blockIdx.x - work->startChannelId) :
      (blockIdx.x + work->commNChannels - work->startChannelId);
    bid = bid % nChannels;
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    struct ncclFC* fc = &channel->fc;
    const int rank = comm->rank;
    int root = work->root;

    int nroleWorkers = work->nThreads, nworkers = work->nThreads;
#ifndef NO_EXTRA_WARP
    if (Proto::Id != NCCL_PROTO_LL && Proto::Id != NCCL_PROTO_LL128) {
      nroleWorkers = (rank != root) ? (nroleWorkers - WARP_SIZE) : nroleWorkers;
      nworkers -= WARP_SIZE;
    }
#endif

    int stepSize;
    ssize_t chunkSize;
    if (Proto::Id == NCCL_PROTO_LL) {
      stepSize = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
      chunkSize = stepSize * sizeof(uint64_t) / sizeof(T);
    } else if (Proto::Id == NCCL_PROTO_LL128) {

    } else {
      stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T) * NCCL_STEPS);
      chunkSize = stepSize * REDUCE_CHUNKSTEPS;
    }
    const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
    const ssize_t size = work->count;

    const T* __restrict__ thisInput = (const T*)work->sendbuff;
    T* __restrict__ thisOutput = (T*)work->recvbuff;

    if (rank == root) {
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_FC_ARITY-1, 0>, 0, Proto, 0>
        prims(tid, ncclPrimsThreadsFromWorkers<Proto, 0>(nroleWorkers), fc->peerRanks, NULL, NULL, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);
      prims.setDataPtrs(thisInput, thisOutput);

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset, nChannels));
        ALIGN_SIZE(realChunkSize, nworkers*sizeof(uint64_t)/sizeof(T));
        ssize_t offset = gridOffset + bid * realChunkSize;
        int nelem = min(realChunkSize, size-offset);
        prims.recvReduceCopy(offset, offset, nelem, /*postOp=*/true);
      }
    } else {
      Primitives<T, RedOp, FanAsymmetric<0, 1>, 0, Proto, 0>
        prims(tid, ncclPrimsThreadsFromWorkers<Proto, 1>(nroleWorkers), NULL, &root, NULL, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);
      prims.setDataPtrs(thisInput, thisOutput);

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset, nChannels));
        ALIGN_SIZE(realChunkSize, nworkers*sizeof(uint64_t)/sizeof(T));
        ssize_t offset = gridOffset + bid * realChunkSize;
        int nelem = min(realChunkSize, size-offset);
        prims.send(offset, nelem);
      }
    }
  }
}

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduce, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      using Proto = ProtoSimple<REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS, UNROLL>;
      runRing<T, RedOp, Proto>(work);
    }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduce, NCCL_ALGO_RING, NCCL_PROTO_LL, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      runRing<T, RedOp, ProtoLL>(work);
    }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduce, NCCL_ALGO_RING, NCCL_PROTO_LL128, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      runRing<T, RedOp, ProtoLL128>(work);
    }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduce, NCCL_ALGO_RING, NCCL_PROTO_SIGNAL, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      using Proto = ProtoSignal<REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS, UNROLL>;
      runRing<T, RedOp, Proto>(work);
    }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduce, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE_BULK, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      struct ncclKernelComm* comm = getShmemComm(work);
      struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
      const ncclConn_t sendConnType = channel->ring.sendConnType;
      const ncclConn_t recvConnType = channel->ring.recvConnType;

      if (sendConnType == CONN_ICN) {
        using Proto = ProtoSimpleBulk<REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS, UNROLL, CONN_ICN, CONN_NORMAL>;
        runRing<T, RedOp, Proto>(work);
      } else {
        using Proto = ProtoSimpleBulk<REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS, UNROLL, CONN_NORMAL, CONN_ICN>;
        runRing<T, RedOp, Proto>(work);
      }
    }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduce, NCCL_ALGO_RING, NCCL_PROTO_SIGNAL_BULK, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      using Proto = ProtoSignalBulk<REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS, UNROLL>;
      runRing<T, RedOp, Proto>(work);
    }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduce, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE_BULK_AR, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
      using Proto = ProtoSimpleBulkAR<REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS, UNROLL, CONN_ICN, CONN_NORMAL>;
      runRing<T, RedOp, Proto>(work);
#endif
    }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduce, NCCL_ALGO_FC, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      using Proto = ProtoSimple<REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS, UNROLL>;
      runFC<T, RedOp, Proto>(work);
    }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduce, NCCL_ALGO_FC, NCCL_PROTO_SIMPLE_BULK, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      struct ncclKernelComm* comm = getShmemComm(work);
      struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
      const ncclConn_t connType = channel->fc.connType;
      if (connType == CONN_ICN) {
        using Proto = ProtoSimpleBulk<REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS, UNROLL, CONN_ICN, CONN_ICN>;
        runFC<T, RedOp, Proto>(work);
      } else {
        using Proto = ProtoSimpleBulk<REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS, UNROLL, CONN_NORMAL, CONN_ICN>;
        runFC<T, RedOp, Proto>(work);
      }
    }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduce, NCCL_ALGO_FC, NCCL_PROTO_LL, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      runFC<T, RedOp, ProtoLL>(work);
    }
};

template<int PROTO, typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduce, NCCL_ALGO_TREE, PROTO, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {}
};

template<int PROTO, typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduce, NCCL_ALGO_COLLNET, PROTO, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {}
};
