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
  template<class FUNC, typename T, int UNROLL, typename Proto>
  __device__ __forceinline__ void runRing(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL, int* inputUserRanks=NULL) {
    const int tid = threadIdx.x;
    int nthreads;
    if (Proto::Id == NCCL_PROTO_LL || Proto::Id == NCCL_PROTO_LL128) {
      nthreads = work->nThreads;
    } else {
#ifndef NO_EXTRA_WARP
      nthreads = work->nThreads-WARP_SIZE;
#else
      nthreads = work->nThreads;
#endif
    }
    const int nChannels = work->nChannels;
    int bid = (blockIdx.x >= work->startChannelId) ?
      (blockIdx.x - work->startChannelId) :
      (blockIdx.x + work->commNChannels - work->startChannelId);
    bid = bid % nChannels;
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    struct ncclRing* ring = (inputRing == NULL) ? &channel->ring : inputRing;
    int stepSize;
    ssize_t chunkSize;
    ssize_t minChunkSize;
    if (Proto::Id == NCCL_PROTO_LL) {
      stepSize = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
      chunkSize = stepSize * sizeof(uint64_t) / sizeof(T);
    } else if (Proto::Id == NCCL_PROTO_LL128) {
      stepSize = comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
      chunkSize = stepSize*NCCL_LL128_DATAELEMS*sizeof(uint64_t) / (NCCL_LL128_LINEELEMS*sizeof(T));
      minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2;
    } else {
      stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
      chunkSize = stepSize * ALLGATHER_CHUNKSTEPS;
    }
    const int nranks = (inputRing == NULL) ? comm->nRanks : inputRing->nRanks;
    const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
    const ssize_t size = work->count;

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)work->sendbuff;
    T * __restrict__ thisOutput = (T*)work->recvbuff;

    // DIRECT=1, LL/LL128 do not use
    Primitives<T, FUNC, Proto, 1, 1, 1, 0> prims(tid, nthreads, &ring->prev, &ring->next, thisOutput, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, NULL);

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
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
      }

      ssize_t chunkOffset = gridOffset + bid*realChunkSize;
      int nelem = min(realChunkSize, size-chunkOffset);

      /////////////// begin AllGather steps ///////////////

      // step 0: push data to next GPU
      int rankDest = (inputRing == NULL) ? ring->devUserRanks[0] : inputUserRanks[ring->devUserRanks[0]]; // inputUsersRanks not shift
      ssize_t offset = chunkOffset + rankDest * size;

      if (thisInput + chunkOffset == thisOutput + offset) { // In place
        prims.directSend(thisInput+chunkOffset, offset, nelem);
      } else {
        prims.directCopySend(thisInput+chunkOffset, thisOutput+offset, offset, nelem);
      }

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        rankDest = (inputRing == NULL) ? ring->devUserRanks[nranks-j] : inputUserRanks[ring->devUserRanks[nranks-j]]; // inputUsersRanks not shift
        offset = chunkOffset + rankDest * size;

        prims.directRecvCopySend(thisOutput+offset, offset, nelem);
      }

      // Make final copy from buffer to dest.
      rankDest = (inputRing == NULL) ? ring->devUserRanks[1] : inputUserRanks[ring->devUserRanks[1]]; // inputUsersRanks not shift
      offset = chunkOffset + rankDest * size;

      // Final wait/copy.
      prims.directRecv(thisOutput+offset, offset, nelem);
    }
  }

  template<class FUNC, typename T, int UNROLL, typename Proto>
  __device__ __forceinline__ void runFC(struct ncclDevWorkColl* work) {
    const int tid = threadIdx.x;
    int nthreads;
    if (Proto::Id == NCCL_PROTO_LL || Proto::Id == NCCL_PROTO_LL128) {
      nthreads = work->nThreads;
    } else {
#ifndef NO_EXTRA_WARP
      nthreads = work->nThreads-WARP_SIZE;
#else
      nthreads = work->nThreads;
#endif
    }
    const int nChannels = work->nChannels;
    int bid = (blockIdx.x >= work->startChannelId) ?
      (blockIdx.x - work->startChannelId) :
      (blockIdx.x + work->commNChannels - work->startChannelId);
    bid = bid % nChannels;
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    struct ncclFC* fc = &channel->fc;

    int stepSize;
    ssize_t chunkSize;
    if (Proto::Id == NCCL_PROTO_LL) {
      stepSize = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine) * NCCL_STEPS);
      chunkSize = stepSize* sizeof(uint64_t) / sizeof(T);
    } else if (Proto::Id == NCCL_PROTO_LL128) {
      static_assert(!(Proto::Id == NCCL_PROTO_LL128), "not ready");
    } else {
      stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T) * NCCL_STEPS);
      chunkSize = stepSize * ALLGATHER_CHUNKSTEPS;
    }

    const int nranks = comm->nRanks;
    const int rank = comm->rank;
    const ssize_t loopSize = nChannels * chunkSize;
    const ssize_t size = work->count;

    const T* __restrict__ thisInput = (const T*)work->sendbuff;
    T* __restrict__ thisOutput = (T*)work->recvbuff;

    Primitives<T, FUNC, Proto, NCCL_MAX_FC_ARITY-1, NCCL_MAX_FC_ARITY-1, 1, 0>
      prims(tid, nthreads, fc->peerRanks, fc->peerRanks, thisOutput, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, NULL);

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      ssize_t realChunkSize;
      if (Proto::Id == NCCL_PROTO_LL) {
        if (size-gridOffset < loopSize) {
          chunkSize = work->lastChunkSize;
        }
        realChunkSize = chunkSize;
      } else if (Proto::Id == NCCL_PROTO_LL128) {

      } else {
        realChunkSize = min(chunkSize, DIVUP(size-gridOffset, nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
      }

      ssize_t chunkOffset = gridOffset + bid * realChunkSize;

      ssize_t dstOffset = rank * size + chunkOffset;
      int nelem = min(realChunkSize, size-chunkOffset);
      prims.directCopySend(thisInput+chunkOffset, thisOutput+dstOffset, dstOffset, nelem);
      int totalElem = nranks * nelem;
      prims.directGather(thisOutput+chunkOffset, totalElem, nelem, size, chunkOffset);
    }

  }
}

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL, int* inputUserRanks=NULL) {
      using Proto = ProtoSimple<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS>;
      runRing<FUNC, T, UNROLL, Proto>(work, inputRing, inputUserRanks);
    }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllGather, NCCL_ALGO_FC, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS>;
    runFC<FUNC, T, UNROLL, Proto>(work);
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL, int* inputUserRanks=NULL) {
      runRing<FUNC, T, UNROLL, ProtoLL>(work, inputRing, inputUserRanks);
    }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllGather, NCCL_ALGO_FC, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      runFC<FUNC, T, UNROLL, ProtoLL>(work);
    }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL, int* inputUserRanks=NULL) {
      runRing<FUNC, T, UNROLL, ProtoLL128>(work, inputRing, inputUserRanks);
    }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_SIGNAL, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL, int* inputUserRanks=NULL) {
      using Proto = ProtoSignal<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS>;
      runRing<FUNC, T, UNROLL, Proto>(work, inputRing, inputUserRanks);
    }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE_BULK, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL, int* inputUserRanks=NULL) {
      struct ncclKernelComm* comm = getShmemComm(work);
      const ncclConn_t sendConnType = getShmemPtr(comm)->channel.ring.sendConnType;

      // Compute pointers

      if (sendConnType == CONN_ICN) {
        using Proto = ProtoSimpleBulk<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS, CONN_ICN, CONN_ICN>;
        runRing<FUNC, T, UNROLL, Proto>(work, inputRing, inputUserRanks);
      } else {
        using Proto = ProtoSimpleBulk<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS, CONN_NORMAL, CONN_ICN>;
        runRing<FUNC, T, UNROLL, Proto> (work, inputRing, inputUserRanks);
      }
    }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllGather, NCCL_ALGO_FC, NCCL_PROTO_SIMPLE_BULK, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    struct ncclKernelComm* comm = getShmemComm(work);
    const ncclConn_t connType = getShmemPtr(comm)->channel.fc.connType;
    if (connType == CONN_ICN) {
      using Proto = ProtoSimpleBulk<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS, CONN_ICN, CONN_ICN>;
      runFC<FUNC, T, UNROLL, Proto>(work);
    } else {
      using Proto = ProtoSimpleBulk<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS, CONN_NORMAL, CONN_ICN>;
      runFC<FUNC, T, UNROLL, Proto>(work);
    }
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_SIGNAL_BULK, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL, int* inputUserRanks=NULL) {
      using Proto = ProtoSignalBulk<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS>;
      runRing<FUNC, T, UNROLL, Proto>(work, inputRing, inputUserRanks);
    }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE_BULK_AR, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL, int* inputUserRanks=NULL) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
      using Proto = ProtoSimpleBulkAR<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS, CONN_ICN, CONN_ICN>;
      runRing<FUNC, T, UNROLL, Proto>(work, inputRing, inputUserRanks);
#endif
    }
};

template<int PROTO, class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllGather, NCCL_ALGO_TREE, PROTO, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {}
};

template<int PROTO, class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllGather, NCCL_ALGO_COLLNET, PROTO, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {}
};
