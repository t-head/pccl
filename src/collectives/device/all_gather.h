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
#ifndef NO_EXTRA_WARP
      nthreads = (work->nWarps * WARP_SIZE)-WARP_SIZE;
#else
      nthreads = (work->nWarps * WARP_SIZE);
#endif
    }
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    struct ncclRing* ring = &channel->ring;
    const int nranks = comm->nRanks;
    ssize_t count, partOffset, partCount, chunkCount;
    ncclCollCbdPart(work, getShmemPtr(comm)->channelId, Proto::Id, sizeof(T), &count, &partOffset, &partCount, &chunkCount);

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)work->sendbuff;
    T * __restrict__ thisOutput = (T*)work->recvbuff;

    // DIRECT=1, LL/LL128 do not use
    Primitives<T, FUNC, Proto, 1, 1, 1, 0> prims(tid, nthreads, &ring->prev, &ring->next, thisOutput, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, NULL);

    for (size_t elemOffset = 0; elemOffset < partCount; elemOffset += chunkCount) {
      const int nelem = min(chunkCount, partCount - elemOffset);
      const ssize_t dataOffset = partOffset + elemOffset;

      /////////////// begin AllGather steps ///////////////

      // step 0: push data to next GPU
      int rankDest = ring->devUserRanks[0];
      ssize_t offset = dataOffset + rankDest * count;

      if (thisInput + dataOffset == thisOutput + offset) { // In place
        prims.directSend(thisInput+dataOffset, offset, nelem);
      } else {
        prims.directCopySend(thisInput+dataOffset, thisOutput+offset, offset, nelem);
      }

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        rankDest = ring->devUserRanks[nranks-j];
        offset = dataOffset + rankDest * count;

        prims.directRecvCopySend(thisOutput+offset, offset, nelem);
      }

      // Make final copy from buffer to dest.
      rankDest = ring->devUserRanks[1];
      offset = dataOffset + rankDest * count;

      // Final wait/copy.
      prims.directRecv(thisOutput+offset, offset, nelem);
    }
  }

  template<class FUNC, typename T, int UNROLL, typename Proto>
  __device__ __forceinline__ void runFC(struct ncclDevWorkColl* work) {
    const int tid = threadIdx.x;
    int nthreads;
    if (Proto::Id == NCCL_PROTO_LL || Proto::Id == NCCL_PROTO_LL128) {
      nthreads = (work->nWarps * WARP_SIZE);
    } else {
#ifndef NO_EXTRA_WARP
      nthreads = (work->nWarps * WARP_SIZE)-WARP_SIZE;
#else
      nthreads = (work->nWarps * WARP_SIZE);
#endif
    }
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    struct ncclFC* fc = &channel->fc;

    const int nranks = comm->nRanks;
    const int rank = comm->rank;
    ssize_t size, partOffset, partCount, chunkSize;
    ncclCollCbdPart(work, getShmemPtr(comm)->channelId, Proto::Id, sizeof(T), &size, &partOffset, &partCount, &chunkSize);

    const T* __restrict__ thisInput = (const T*)work->sendbuff;
    T* __restrict__ thisOutput = (T*)work->recvbuff;

    Primitives<T, FUNC, Proto, NCCL_MAX_FC_ARITY-1, NCCL_MAX_FC_ARITY-1, 1, 0>
      prims(tid, nthreads, fc->peerRanks, fc->peerRanks, thisOutput, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, NULL);

    for (ssize_t elemOffset = 0; elemOffset < partCount; elemOffset += chunkSize) {
      const ssize_t chunkOffset = partOffset + elemOffset;
      ssize_t dstOffset = rank * size + chunkOffset;
      int nelem = min(chunkSize, partCount - elemOffset);
      prims.directCopySend(thisInput+chunkOffset, thisOutput+dstOffset, dstOffset, nelem);
      int totalElem = nranks * nelem;
      prims.directGather(thisOutput+chunkOffset, totalElem, nelem, size, chunkOffset);
    }

  }
}

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      using Proto = ProtoSimple<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS>;
      runRing<FUNC, T, UNROLL, Proto>(work);
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
    __device__ void run(struct ncclDevWorkColl* work) {
      runRing<FUNC, T, UNROLL, ProtoLL>(work);
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
    __device__ void run(struct ncclDevWorkColl* work) {
      runRing<FUNC, T, UNROLL, ProtoLL128>(work);
    }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_SIGNAL, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      using Proto = ProtoSignal<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS>;
      runRing<FUNC, T, UNROLL, Proto>(work);
    }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE_BULK, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      struct ncclKernelComm* comm = getShmemComm(work);
      const ncclConn_t sendConnType = getShmemPtr(comm)->channel.ring.sendConnType;

      // Compute pointers

      if (sendConnType == CONN_ICN) {
        using Proto = ProtoSimpleBulk<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS, CONN_ICN, CONN_ICN>;
        runRing<FUNC, T, UNROLL, Proto>(work);
      } else {
        using Proto = ProtoSimpleBulk<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS, CONN_NORMAL, CONN_ICN>;
        runRing<FUNC, T, UNROLL, Proto>(work);
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
    __device__ void run(struct ncclDevWorkColl* work) {
      using Proto = ProtoSignalBulk<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS>;
      runRing<FUNC, T, UNROLL, Proto>(work);
    }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE_BULK_AR, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
      using Proto = ProtoSimpleBulkAR<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS, CONN_ICN, CONN_ICN>;
      runRing<FUNC, T, UNROLL, Proto>(work);
#endif
    }
};

#if defined(__HGGC_ARCH__)
namespace {
  template<class FUNC, typename T, int UNROLL>
  __device__ __forceinline__ void runPAT(struct ncclDevWorkColl* work) {
    struct ncclKernelComm* comm = getShmemComm(work);
    const int nranks = comm->nRanks;
    const int rank = comm->rank;

    const int tid = threadIdx.x;
    const int nthreads = (work->nWarps * WARP_SIZE);
    
    ssize_t count, channelOffset, channelCount, chunkCount;
    ncclCollCbdPart(work, getShmemPtr(comm)->channelId, NCCL_PROTO_SIMPLE, sizeof(T),
        &count, &channelOffset, &channelCount, &chunkCount);

    static constexpr int nworkers = NCCL_PAT_NWORKERS;
    struct ncclPatShmem* shmem = ncclPatShmemPtr(comm);
    uint64_t pollCount = 0;
    __syncthreads();
    for (int i = tid; i < NCCL_SHMEM_PAT_STEPS; i += nthreads) shmem->patSteps[i].flags = 0;
    if (tid == 0) shmem->localAccSize = 0;
    if (tid == nworkers) shmem->parallelFactor = 0;
    __syncthreads();

    if (tid == nworkers) { // Algo computation thread
      PatAGAlgorithm<T> patAlgo(chunkCount*sizeof(T), NCCL_STEPS, NCCL_PAT_NWORKERS/WARP_SIZE, channelOffset, channelOffset + channelCount, count, chunkCount, rank, nranks);
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
      Primitives<T, FUNC, ProtoSimple<UNROLL, 1, 1>, 1, 1, 0, 0> prims
        (tidInGroup, groupSize, (int*)shmem->recvDims, (int*)shmem->sendDims,
         nullptr /*directBuff*/, comm, work->redOpArg, getShmemPtr(comm)->groups,
         group, 0, 0, nullptr, nullptr, NCCL_STEPS, false,
         inputBuf, outputBuf, primsModePatAg);

      int step = group;
      while (1) {
        struct ncclPatStep* ps = shmem->patSteps+(step%NCCL_SHMEM_PAT_STEPS);
        volatile int* poll = &ps->flags;
        while (patStepFlagsLoadAcquire(poll) == 0) pollCount++; // Wait for compute thread
        int last = ps->last;
        prims.patCopy(ps, shmem);
        if (tidInGroup == 0) patStepFlagsStoreRelease(poll, 0); // Return element to compute thread
        if (last) break;
        step += nGroups;
      }
    }
  }
}

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllGather, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {
      runPAT<FUNC, T, UNROLL>(work);
    }
};

#endif

template<int PROTO, class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllGather, NCCL_ALGO_TREE, PROTO, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkColl* work) {}
};
