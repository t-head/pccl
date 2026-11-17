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
  __device__ __forceinline__ void runRing(struct ncclDevWorkColl* work) {
    const int tid = threadIdx.x;
    int nthreads;
    if (Proto::Id == NCCL_PROTO_LL || Proto::Id == NCCL_PROTO_LL128) {
      nthreads = (work->nWarps * WARP_SIZE);
    } else {
      nthreads = (work->nWarps * WARP_SIZE)-WARP_SIZE;
    }
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    struct ncclRing* ring = &channel->ring;

    const int nranks = comm->nRanks;
    ssize_t size, partOffset, partCount, chunkSize;
    ncclCollCbdPart(work, getShmemPtr(comm)->channelId, Proto::Id, sizeof(T), &size, &partOffset, &partCount, &chunkSize);
    const int rank = ring->devUserRanks[0];
    const int prevRank = ring->devUserRanks[nranks-1];
    const int root = work->root;

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)work->sendbuff;
    T * __restrict__ thisOutput = (T*)work->recvbuff;

    Primitives<T, FUNC, Proto, 1, 1, 0, 0>
      prims(tid, nthreads, &ring->prev, &ring->next, NULL, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);

    for (ssize_t elemOffset = 0; elemOffset < partCount; elemOffset += chunkSize) {
      const ssize_t offset = partOffset + elemOffset;
      const int nelem = min(chunkSize, partCount - elemOffset);
      if (prevRank == root) {
        prims.send(thisInput+offset, nelem);
      } else if (rank == root) {
        prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem, /*postOp=*/true);
      } else {
        prims.recvReduceSend(thisInput+offset, nelem);
      }
    }
  }


  template<class FUNC, typename T, int UNROLL, typename Proto>
  __device__ __forceinline__ void runFC(struct ncclDevWorkColl* work) {
    const int tid = threadIdx.x;
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    struct ncclFC* fc = &channel->fc;
    const int rank = comm->rank;
    int root = work->root;

    int nthreads = (work->nWarps * WARP_SIZE);
#ifndef NO_EXTRA_WARP
    if (Proto::Id != NCCL_PROTO_LL && Proto::Id != NCCL_PROTO_LL128) {
      // root recvs only, other ranks' nthreads will increase later
      nthreads = (rank != root) ? (nthreads - WARP_SIZE) : nthreads;
    }
#endif

    ssize_t partOffset, partCount, chunkSize;
    ncclCollCbdPart(work, getShmemPtr(comm)->channelId, Proto::Id, sizeof(T), (ssize_t*)nullptr, &partOffset, &partCount, &chunkSize);

    const T* __restrict__ thisInput = (const T*)work->sendbuff;
    T* __restrict__ thisOutput = (T*)work->recvbuff;

    if (rank == root) {
      Primitives<T, FUNC, Proto, NCCL_MAX_FC_ARITY-1, 0, 0, 0>
        prims(tid, nthreads, fc->peerRanks, NULL, NULL, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);

      for (ssize_t elemOffset = 0; elemOffset < partCount; elemOffset += chunkSize) {
        const ssize_t offset = partOffset + elemOffset;
        int nelem = min(chunkSize, partCount - elemOffset);
        prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem, /*postOp=*/true);
      }
    } else {
      Primitives<T, FUNC, Proto, 0, 1, 0, 0>
        prims(tid, nthreads, NULL, &root, NULL, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);

      for (ssize_t elemOffset = 0; elemOffset < partCount; elemOffset += chunkSize) {
        const ssize_t offset = partOffset + elemOffset;
        int nelem = min(chunkSize, partCount - elemOffset);
        prims.send(thisInput+offset, nelem);
      }
    }
  }
}

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduce, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      using Proto = ProtoSimple<UNROLL, REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS>;
      runRing<FUNC, T, UNROLL, Proto>(work);
    }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduce, NCCL_ALGO_RING, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      runRing<FUNC, T, UNROLL, ProtoLL>(work);
    }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduce, NCCL_ALGO_RING, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      runRing<FUNC, T, UNROLL, ProtoLL128>(work);
    }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduce, NCCL_ALGO_RING, NCCL_PROTO_SIGNAL, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      using Progo = ProtoSignal<UNROLL, REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS>;
      runRing<FUNC, T, UNROLL, Progo>(work);
    }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduce, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE_BULK, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      struct ncclKernelComm* comm = getShmemComm(work);
      struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
      const ncclConn_t sendConnType = channel->ring.sendConnType;
      const ncclConn_t recvConnType = channel->ring.recvConnType;

      if (sendConnType == CONN_ICN) {
        using Proto = ProtoSimpleBulk<UNROLL, REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS, CONN_ICN, CONN_NORMAL>;
        runRing<FUNC, T, UNROLL, Proto>(work);
      } else {
        using Proto = ProtoSimpleBulk<UNROLL, REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS, CONN_NORMAL, CONN_ICN>;
        runRing<FUNC, T, UNROLL, Proto>(work);
      }
    }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduce, NCCL_ALGO_RING, NCCL_PROTO_SIGNAL_BULK, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      using Proto = ProtoSignalBulk<UNROLL, REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS>;
      runRing<FUNC, T, UNROLL, Proto>(work);
    }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduce, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE_BULK_AR, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
      using Proto = ProtoSimpleBulkAR<UNROLL, REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS, CONN_ICN, CONN_NORMAL>;
      runRing<FUNC, T, UNROLL, Proto>(work);
#endif
    }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduce, NCCL_ALGO_FC, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      using Proto = ProtoSimple<UNROLL, REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS>;
      runFC<FUNC, T, UNROLL, Proto>(work);
    }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduce, NCCL_ALGO_FC, NCCL_PROTO_SIMPLE_BULK, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      struct ncclKernelComm* comm = getShmemComm(work);
      struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
      const ncclConn_t connType = channel->fc.connType;
      if (connType == CONN_ICN) {
        using Proto = ProtoSimpleBulk<UNROLL, REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS, CONN_ICN, CONN_ICN>;
        runFC<FUNC, T, UNROLL, Proto>(work);
      } else {
        using Proto = ProtoSimpleBulk<UNROLL, REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS, CONN_NORMAL, CONN_ICN>;
        runFC<FUNC, T, UNROLL, Proto>(work);
      }
    }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduce, NCCL_ALGO_FC, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      runFC<FUNC, T, UNROLL, ProtoLL>(work);
    }
};

template<int PROTO, class REDOP, typename T, int UNROLL>
class RunWorkColl<ncclFuncReduce, NCCL_ALGO_TREE, PROTO, REDOP, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {}
};
