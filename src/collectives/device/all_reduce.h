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
  __device__ __forceinline__ void runRing(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL) {
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
    const int nranks = (inputRing == NULL) ? comm->nRanks : inputRing->nRanks;

    int stepSize;
    ssize_t chunkSize;
    ssize_t minChunkSize;
    if (Proto::Id == NCCL_PROTO_LL) {
      stepSize = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
      chunkSize = stepSize * sizeof(uint64_t) / sizeof(T);
      minChunkSize = nthreads * (sizeof(uint64_t)) / sizeof(T);
    } else if (Proto::Id == NCCL_PROTO_LL128) {
      stepSize = comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
      chunkSize = stepSize*NCCL_LL128_DATAELEMS*sizeof(uint64_t) / (NCCL_LL128_LINEELEMS*sizeof(T));
      // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
      minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2;
    } else {
      stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
      chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
    }

    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = work->count;

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)work->sendbuff;
    T * __restrict__ thisOutput = (T*)work->recvbuff;

    Primitives<T, FUNC, Proto, 1, 1, 1, 0>
      prims(tid, nthreads, &ring->prev, &ring->next, thisOutput, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
      ssize_t realChunkSize;
      ssize_t offset;
      int nelem;
      int chunk;
      ssize_t chunkOffset;
      if (Proto::Id == NCCL_PROTO_LL || Proto::Id == NCCL_PROTO_LL128) {
        chunkSize = min(DIVUP(size-gridOffset, nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);
        chunk = ring->devUserRanks[nranks-1];
        offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
        nelem = min(chunkSize, size-offset);
      } else {
        realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
        chunkOffset = gridOffset + bid*nranks*realChunkSize;
        // step 0: push data to next GPU
        chunk = ring->devUserRanks[nranks-1];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);
      }
      /////////////// begin AllReduce steps ///////////////

#ifdef KERNEL_LOOPBACK
      if (ring->prev == -1 && ring->next == -1) {
        prims.sendRecvLoopback(thisInput+offset, thisOutput+offset, nelem);
        continue;
      }
#endif

      prims.send(thisInput+offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        if (Proto::Id == NCCL_PROTO_LL || Proto::Id == NCCL_PROTO_LL128) {
          offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
          nelem = min(chunkSize, size-offset);
        } else {
          offset = chunkOffset + chunk * realChunkSize;
          nelem = min(realChunkSize, size-offset);
        }

        prims.recvReduceSend(thisInput+offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ring->devUserRanks[0];
      if (Proto::Id == NCCL_PROTO_LL || Proto::Id == NCCL_PROTO_LL128) {
        offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
        nelem = min(chunkSize, size-offset);
      } else {
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);
      }

      prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem, /*postOp=*/true);

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        if (Proto::Id == NCCL_PROTO_LL || Proto::Id == NCCL_PROTO_LL128) {
          offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
          nelem = min(chunkSize, size-offset);
        } else {
          offset = chunkOffset + chunk * realChunkSize;
          nelem = min(realChunkSize, size-offset);
        }

        prims.directRecvCopySend(thisOutput+offset, offset, nelem);
      }

      // Make final copy from buffer to dest.
      chunk = ring->devUserRanks[1];
      if (Proto::Id == NCCL_PROTO_LL || Proto::Id == NCCL_PROTO_LL128) {
        offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
        nelem = min(chunkSize, size-offset);
      } else {
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);
      }

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
      chunkSize = stepSize * sizeof(uint64_t) / sizeof(T);
    } else if (Proto::Id == NCCL_PROTO_LL128) {
      stepSize = comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
      chunkSize = stepSize*NCCL_LL128_DATAELEMS*sizeof(uint64_t) / (NCCL_LL128_LINEELEMS*sizeof(T));
      ALIGN_SIZE_DOWN(chunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    } else {
      stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
      chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
    }
    const int nranks = comm->nRanks;
    const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
    const ssize_t size = work->count;
    int chunk = fc->rank;

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)work->sendbuff;
    T * __restrict__ thisOutput = (T*)work->recvbuff;

    Primitives<T, FUNC, Proto, NCCL_MAX_FC_ARITY-1, NCCL_MAX_FC_ARITY-1, 1, 0>
      prims(tid, nthreads, fc->peerRanks, fc->peerRanks, thisOutput, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);

    if (Proto::Id == NCCL_PROTO_LL) {
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
        ssize_t chunkOffset = gridOffset + bid*realChunkSize;
        int nelem = min(realChunkSize, size-chunkOffset);
        prims.send(thisInput+chunkOffset, nelem);
        prims.recvReduceCopy(thisInput+chunkOffset, thisOutput+chunkOffset, nelem, /*postOp=*/true);
      }
    } else {
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
        ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
        ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

        /////////////// begin AllReduce steps ///////////////
        ssize_t offset;
        int nelem;
        int peerElem;

        // step 0: scatter data with different offsets to all other peer GPUs
        nelem = min(realChunkSize * nranks, size-chunkOffset);
        peerElem = DIVUP(nelem, nranks);
        prims.scatter(thisInput+chunkOffset, nelem, peerElem, peerElem);

        // step 1: reduce chunk offset data with all other peer GPUs
        offset = chunkOffset + chunk * peerElem;
        prims.directRecvReduceCopySend(thisInput+offset,
            thisOutput+offset, offset, min(peerElem, size-offset), /*postOp=*/true, peerElem);

        // step 2: final wait/copy.
        prims.directGather(thisOutput+chunkOffset, nelem, peerElem, peerElem, chunkOffset);
      }
    }
  }

  template<class FUNC, typename T, int UNROLL, typename upProto, typename downProto>
  __device__ __forceinline__ void runTreeUpDown(struct ncclDevWorkColl* work) {
    const int tid = threadIdx.x;
#ifndef NO_EXTRA_WARP
    const int nthreads = work->nThreads-2*WARP_SIZE;
#else
    const int nthreads = work->nThreads;
#endif
    const int nChannels = work->nChannels;
    int bid = (blockIdx.x >= work->startChannelId) ?
      (blockIdx.x - work->startChannelId) :
      (blockIdx.x + work->commNChannels - work->startChannelId);
    bid = bid % nChannels;
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    struct ncclTree* tree = &channel->tree;
    const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
    int chunkSize = work->lastChunkSize;
    const ssize_t minChunkSize = nthreads*8*sizeof(uint64_t) / sizeof(T);
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = work->count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)work->sendbuff;
    T * __restrict__ thisOutput = (T*)work->recvbuff;

#ifndef NO_EXTRA_WARP
    if (tid < nthreads+WARP_SIZE) {
#else
    if (tid < nthreads) {
#endif
      // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      Primitives<T, FUNC, upProto, NCCL_MAX_DEV_ARITY, 1, 0, 0>
        prims(tid, nthreads, tree->down, &tree->up, NULL, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Up
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);

#ifdef KERNEL_LOOPBACK
        if (tree->up == -1 && tree->down[0] == -1) {
          prims.sendRecvLoopback(thisInput+offset, thisOutput+offset, nelem);
          continue;
        }
#endif
        if (tree->up == -1) {
          prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem, /*postOp=*/true);
        } else if (tree->down[0] == -1) {
          prims.send(thisInput+offset, nelem);
        } else {
          prims.recvReduceSend(thisInput+offset, nelem);
        }
      }
    }

#ifndef KERNEL_LOOPBACK
#ifndef NO_EXTRA_WARP
    if (tid < nthreads+WARP_SIZE) {
#else
    if (tid < nthreads) {
#endif
      // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      Primitives<T, FUNC, downProto, 1, NCCL_MAX_DEV_ARITY, 1, 0>
        prims(tid, nthreads, &tree->up, tree->down, thisOutput, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work, NULL, NCCL_STEPS, true);

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Down
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          prims.directSend(thisOutput+offset, offset, nelem);
        } else if (tree->down[0] == -1) {
          prims.directRecv(thisOutput+offset, offset, nelem);
        } else {
          prims.directRecvCopySend(thisOutput+offset, offset, nelem);
        }
      }
    }
#endif
  }

  template<class FUNC, typename T, int UNROLL, typename upProto, typename downProto>
  __device__ __forceinline__ void runSplitTree(struct ncclDevWorkColl* work) {
    const int tid = threadIdx.x;
    int nthreads;
#ifndef NO_EXTRA_WARP
    if (upProto::Id == NCCL_PROTO_LL || upProto::Id == NCCL_PROTO_LL128) {
      nthreads = work->nThreads;
    } else {
      nthreads = work->nThreads-2*WARP_SIZE;
    }
#else
    nthreads = work->nThreads;
#endif
    const int nChannels = work->nChannels;
    int bid = (blockIdx.x >= work->startChannelId) ?
      (blockIdx.x - work->startChannelId) :
      (blockIdx.x + work->commNChannels - work->startChannelId);
    bid = bid % nChannels;
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    struct ncclTree* tree = &channel->tree;
    int stepSize;
    ssize_t chunkSize;
    ssize_t minChunkSize;
    static_assert(upProto::Id != NCCL_PROTO_LL, "do not use split tree for LL now.");
    if (upProto::Id == NCCL_PROTO_LL) {
      const int stepSize= comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
      chunkSize = stepSize * sizeof(uint64_t) / sizeof(T);
      minChunkSize = nthreads*sizeof(uint64_t) / sizeof(T);
    } else if (upProto::Id == NCCL_PROTO_LL128) {
      stepSize = comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
      chunkSize = work->lastChunkSize;
      minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/8;
    } else {
      stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
      chunkSize = work->lastChunkSize;
      minChunkSize = nthreads*8*sizeof(uint64_t) / sizeof(T);
    }

    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = work->count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)work->sendbuff;
    T * __restrict__ thisOutput = (T*)work->recvbuff;

    int nthreadsSplit;
    if (upProto::Id == NCCL_PROTO_LL128) {
      nthreadsSplit = NCCL_LL128_SPLIT(nthreads);
    } else {
      nthreadsSplit = nthreads/2;
      if (nthreadsSplit == 256) nthreadsSplit += 64;
    }
    if (tree->up == -1) {
#ifndef NO_EXTRA_WARP
      if (tid < nthreads+WARP_SIZE || upProto::Id == NCCL_PROTO_LL128) {
#else
      if (tid < nthreads) {
#endif
        // ReduceAndBroadcast : max number of recv is 3, max number of send is 3
        Primitives<T, FUNC, downProto,  NCCL_MAX_DEV_ARITY, NCCL_MAX_DEV_ARITY, 1, 0>
          prims(tid, nthreads, tree->down, tree->down, thisOutput, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);

        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem, /*postOp=*/true);
        }
      }
    } else {
#ifndef NO_EXTRA_WARP
      if (tid < nthreadsSplit + (upProto::Id == NCCL_PROTO_LL128 ? 0 : WARP_SIZE)) {
#else
      if (tid < nthreadsSplit) {
#endif
        // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
        Primitives<T, FUNC, upProto,  NCCL_MAX_DEV_ARITY, 1, 1, 0>
          prims(tid, nthreadsSplit, tree->down, &tree->up, thisOutput, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);

        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          // Up
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          if (tree->down[0] == -1) {
            prims.send(thisInput+offset, nelem);
          } else {
            prims.recvReduceSend(thisInput+offset, nelem);
          }
        }
      } else {
        // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
        Primitives<T, FUNC, downProto, 1, NCCL_MAX_DEV_ARITY, 1, 0>
#ifndef NO_EXTRA_WARP
          prims(tid-nthreadsSplit-(downProto::Id == NCCL_PROTO_LL128 ? 0 : WARP_SIZE), nthreads-nthreadsSplit,
                &tree->up, tree->down, thisOutput, comm, work->redOpArg, getShmemPtr(comm)->groups, 2, 0, 0, work);
#else
          prims(tid-nthreadsSplit, nthreads-nthreadsSplit, &tree->up, tree->down, thisOutput, comm, work->redOpArg, getShmemPtr(comm)->groups, 2, 0, 0, work);
#endif

        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          // Down
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          if (tree->down[0] == -1) {
            prims.directRecv(thisOutput+offset, offset, nelem);
          } else {
            prims.directRecvCopySend(thisOutput+offset, offset, nelem);
          }
        }
      }
    }
  }
}

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL) {
    using Proto = ProtoSimple<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    runRing<FUNC, T, UNROLL, Proto>(work, inputRing);
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<UNROLL, 1, 1>;
#if 0
    runTreeUpDown<FUNC, T, UNROLL, Proto, Proto>(work);
#else
    runSplitTree<FUNC, T, UNROLL, Proto, Proto>(work);
#endif
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    const int tid = threadIdx.x;
#ifndef NO_EXTRA_WARP
    const int nthreads = work->nThreads-WARP_SIZE;
#else
    const int nthreads = work->nThreads;
#endif
    const int nChannels = work->nChannels;
    int bid = (blockIdx.x >= work->startChannelId) ?
      (blockIdx.x - work->startChannelId) :
      (blockIdx.x + work->commNChannels - work->startChannelId);
    bid = bid % nChannels;
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    struct ncclTree* tree = &channel->collTree;
    const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
    int chunkSize = work->lastChunkSize;
    const ssize_t minChunkSize = nthreads*8*sizeof(uint64_t) / sizeof(T);
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = work->count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)work->sendbuff;
    T * __restrict__ thisOutput = (T*)work->recvbuff;

    if (blockIdx.x < nChannels) { // first half of the channels do reduce
      Primitives<T, FUNC, ProtoSimple<UNROLL, 1, 1>, 1, 1, 0, 0>
        prims(tid, nthreads, tree->down, &tree->up, NULL, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Up
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
        } else if (tree->down[0] == -1) {
          prims.send(thisInput+offset, nelem);
        } else {
          prims.recvReduceSend(thisInput+offset, nelem);
        }
      }
    }

    if (blockIdx.x >= nChannels) { // second half of the channels do broadcast
      Primitives<T, FUNC, ProtoSimple<UNROLL, 1, 1>, 1, 1, 0, 0>
        prims(tid, nthreads, &tree->up, tree->down, NULL, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Down
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          prims.send(thisOutput+offset, nelem);
        } else if (tree->down[0] == -1) {
          prims.recv(thisOutput+offset, nelem);
        } else {
          prims.recvCopySend(thisOutput+offset, nelem);
        }
      }
    }
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_FC, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    runFC<FUNC, T, UNROLL, Proto>(work);
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL) {
    runRing<FUNC, T, UNROLL, ProtoLL>(work, inputRing);
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    const int tid = threadIdx.x;
    const int nthreads = work->nThreads;
    const int nChannels = work->nChannels;
    int bid = (blockIdx.x >= work->startChannelId) ?
      (blockIdx.x - work->startChannelId) :
      (blockIdx.x + work->commNChannels - work->startChannelId);
    bid = bid % nChannels;
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    struct ncclTree* tree = &channel->tree;
    const int stepLines = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
    ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
    const ssize_t minChunkSize = nthreads*sizeof(uint64_t) / sizeof(T);
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = work->count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)work->sendbuff;
    T * __restrict__ thisOutput = (T*)work->recvbuff;

    do {
      // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      Primitives<T, FUNC, ProtoLL, NCCL_MAX_DEV_ARITY, 1, 0, 0> LLprims(tid, nthreads, tree->down, &tree->up, NULL, comm, work->redOpArg);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Up
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);

#ifdef KERNEL_LOOPBACK
        if (tree->up == -1 && tree->down[0] == -1) {
          LLprims.sendRecvLoopback(thisInput+offset, thisOutput+offset, nelem);
          continue;
        }
#endif
        if (tree->up == -1) {
          LLprims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem, /*postOp=*/true);
        } else if (tree->down[0] == -1) {
          LLprims.send(thisInput+offset, nelem);
        } else {
          LLprims.recvReduceSend(thisInput+offset, nelem);
        }
      }
    } while(0);

#ifndef KERNEL_LOOPBACK
    do {
      // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      Primitives<T, FUNC, ProtoLL, 1, NCCL_MAX_DEV_ARITY, 0, 0> LLprims(tid, nthreads, &tree->up, tree->down, NULL, comm, work->redOpArg, NULL, 0, 0, 0, work, nullptr, NCCL_STEPS, true);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Down
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          LLprims.send(thisOutput+offset, nelem);
        } else if (tree->down[0] == -1) {
          LLprims.recv(thisOutput+offset, nelem);
        } else {
          LLprims.recvCopySend(thisOutput+offset, nelem);
        }
      }
    } while(0);
#endif
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    const int tid = threadIdx.x;
    const int nthreads = work->nThreads;
    const int nChannels = work->nChannels;
    int bid = (blockIdx.x >= work->startChannelId) ?
      (blockIdx.x - work->startChannelId) :
      (blockIdx.x + work->commNChannels - work->startChannelId);
    bid = bid % nChannels;
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    struct ncclTree* tree = &channel->collTree;
    const int stepLines = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
    ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
    const ssize_t minChunkSize = nthreads*sizeof(uint64_t) / sizeof(T);
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = work->count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)work->sendbuff;
    T * __restrict__ thisOutput = (T*)work->recvbuff;

    if (blockIdx.x < nChannels) { // first half of the channels do reduce
      Primitives<T, FUNC, ProtoLL, 1, 1, 0, 0> LLprims(tid, nthreads, tree->down, &tree->up, NULL, comm, work->redOpArg);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Up
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          LLprims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
        } else if (tree->down[0] == -1) {
          LLprims.send(thisInput+offset, nelem);
        } else {
          LLprims.recvReduceSend(thisInput+offset, nelem);
        }
      }
    }

    if (blockIdx.x >= nChannels) { // second half of the channels do broadcast
      Primitives<T, FUNC, ProtoLL, 1, 1, 0, 0> LLprims(tid, nthreads, &tree->up, tree->down, NULL, comm, work->redOpArg);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Down
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          LLprims.send(thisOutput+offset, nelem);
        } else if (tree->down[0] == -1) {
          LLprims.recv(thisOutput+offset, nelem);
        } else {
          LLprims.recvCopySend(thisOutput+offset, nelem);
        }
      }
    }
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_FC, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    runFC<FUNC, T, UNROLL, ProtoLL>(work);
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL) {
    runRing<FUNC, T, UNROLL, ProtoLL128>(work, inputRing);
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    runSplitTree<FUNC, T, UNROLL, ProtoLL128, ProtoLL128>(work);
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
__device__ void run(struct ncclDevWorkColl* work) { }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_FC, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
__device__ void run(struct ncclDevWorkColl* work) {
    runFC<FUNC, T, UNROLL, ProtoLL128>(work);
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_SIGNAL, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL) {
    using Proto = ProtoSignal<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    runRing<FUNC, T, UNROLL, Proto>(work, inputRing);
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_SIGNAL, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    using Proto = ProtoSignal<UNROLL, 1, 1>;

#if 1
    runTreeUpDown<FUNC, T, UNROLL, Proto, Proto>(work);
#else
    runSplitTree<FUNC, T, UNROLL, Proto, Proto>(work);
#endif
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_SIGNAL, FUNC, T, UNROLL> {
  public:
__device__ void run(struct ncclDevWorkColl* work) { }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE_BULK, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL) {
    struct ncclKernelComm* comm = getShmemComm(work);
    const ncclConn_t sendConnType = getShmemPtr(comm)->channel.ring.sendConnType;

    if (sendConnType == CONN_ICN) {
      using Proto = ProtoSimpleBulk<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, CONN_ICN, CONN_ICN>;
      runRing<FUNC, T, UNROLL, Proto>(work, inputRing);
    } else {
      using Proto = ProtoSimpleBulk<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, CONN_NORMAL, CONN_ICN>;
      runRing<FUNC, T, UNROLL, Proto>(work, inputRing);
    }
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE_BULK_AR, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
    struct ncclKernelComm* comm = getShmemComm(work);
    const ncclConn_t sendConnType = getShmemPtr(comm)->channel.ring.sendConnType;

    if (sendConnType == CONN_ICN) {
      using Proto = ProtoSimpleBulkAR<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, CONN_ICN, CONN_ICN>;
      runRing<FUNC, T, UNROLL, Proto>(work, inputRing);
    } else {
      using Proto = ProtoSimpleBulkAR<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, CONN_NORMAL, CONN_ICN>;
      runRing<FUNC, T, UNROLL, Proto>(work, inputRing);
    }
#endif
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_FC, NCCL_PROTO_SIGNAL, FUNC, T, UNROLL> {
  public:
__device__ void run(struct ncclDevWorkColl* work) {
    using Proto = ProtoSignal<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    runFC<FUNC, T, UNROLL, Proto>(work);
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE_BULK, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    const ncclConn_t upConnType = channel->tree.upConnType;
    const ncclConn_t downConnType = channel->tree.downConnType;

#if 1
    if (upConnType == CONN_ICN && downConnType == CONN_ICN) {
      using upProto = ProtoSimpleBulk<UNROLL, 1, 1, CONN_ICN, CONN_ICN>;
      using downProto = ProtoSimpleBulk<UNROLL, 1, 1, CONN_ICN, CONN_ICN>;
      runTreeUpDown<FUNC, T, UNROLL, upProto, downProto>(work);
    } else if (upConnType == CONN_ICN && downConnType != CONN_ICN) {
      using upProto = ProtoSimpleBulk<UNROLL, 1, 1, CONN_ICN, CONN_ICN>;
      using downProto = ProtoSimpleBulk<UNROLL, 1, 1, CONN_NORMAL, CONN_ICN>;
      runTreeUpDown<FUNC, T, UNROLL, upProto, downProto>(work);
    } else if (upConnType != CONN_ICN && downConnType == CONN_ICN) {
      using upProto = ProtoSimpleBulk<UNROLL, 1, 1, CONN_NORMAL, CONN_ICN>;
      using downProto = ProtoSimpleBulk<UNROLL, 1, 1, CONN_ICN, CONN_ICN>;
      runTreeUpDown<FUNC, T, UNROLL, upProto, downProto>(work);
    } else {
      using upProto = ProtoSimpleBulk<UNROLL, 1, 1, CONN_NORMAL, CONN_ICN>;
      using downProto = ProtoSimpleBulk<UNROLL, 1, 1, CONN_NORMAL, CONN_ICN>;
      runTreeUpDown<FUNC, T, UNROLL, upProto, downProto>(work);
    }
#else
    if (upConnType == CONN_ICN && downConnType == CONN_ICN) {
      using upProto = ProtoSimpleBulk<UNROLL, 1, 1, CONN_ICN, CONN_ICN>;
      using downProto = ProtoSimpleBulk<UNROLL, 1, 1, CONN_ICN, CONN_ICN>;
      runSplitTree<FUNC, T, UNROLL, upProto, downProto>(work);
    } else if (upConnType == CONN_ICN && downConnType != CONN_ICN) {
      using upProto = ProtoSimpleBulk<UNROLL, 1, 1, CONN_ICN, CONN_ICN>;
      using downProto = ProtoSimpleBulk<UNROLL, 1, 1, CONN_NORMAL, CONN_ICN>;
      runSplitTree<FUNC, T, UNROLL, upProto, downProto>(work);
    } else if (upConnType != CONN_ICN && downConnType == CONN_ICN) {
      using upProto = ProtoSimpleBulk<UNROLL, 1, 1, CONN_NORMAL, CONN_ICN>;
      using downProto = ProtoSimpleBulk<UNROLL, 1, 1, CONN_ICN, CONN_ICN>;
      runSplitTree<FUNC, T, UNROLL, upProto, downProto>(work);
    } else {
      using upProto = ProtoSimpleBulk<UNROLL, 1, 1, CONN_NORMAL, CONN_ICN>;
      using downProto = ProtoSimpleBulk<UNROLL, 1, 1, CONN_NORMAL, CONN_ICN>;
      runSplitTree<FUNC, T, UNROLL, upProto, downProto>(work);
    }
#endif
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_SIMPLE_BULK, FUNC, T, UNROLL> {
  public:
__device__ void run(struct ncclDevWorkColl* work) { }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_SIGNAL_BULK, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL) {
    using Proto = ProtoSignalBulk<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    runRing<FUNC, T, UNROLL, Proto>(work, inputRing);
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_FC, NCCL_PROTO_SIMPLE_BULK, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    const ncclConn_t connType = channel->fc.connType;
    if (connType == CONN_ICN) {
      using Proto = ProtoSimpleBulk<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, CONN_ICN, CONN_ICN>;
      runFC<FUNC, T, UNROLL, Proto>(work);
    } else {
      using Proto = ProtoSimpleBulk<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, CONN_NORMAL, CONN_ICN>;
      runFC<FUNC, T, UNROLL, Proto>(work);
    }
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_SIGNAL_BULK, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    const ncclConn_t upConnType = channel->tree.upConnType;
    const ncclConn_t downConnType = channel->tree.downConnType;

#if 1
    if (upConnType == CONN_ICN && downConnType == CONN_ICN) {
      using upProto = ProtoSignalBulk<UNROLL, 1, 1, CONN_ICN, CONN_ICN>;
      using downProto = ProtoSignalBulk<UNROLL, 1, 1, CONN_ICN, CONN_ICN>;
      runTreeUpDown<FUNC, T, UNROLL, upProto, downProto>(work);
    } else if (upConnType == CONN_ICN && downConnType != CONN_ICN) {
      using upProto = ProtoSignalBulk<UNROLL, 1, 1, CONN_ICN, CONN_ICN>;
      using downProto = ProtoSignalBulk<UNROLL, 1, 1, CONN_NORMAL, CONN_ICN>;
      runTreeUpDown<FUNC, T, UNROLL, upProto, downProto>(work);
    } else if (upConnType != CONN_ICN && downConnType == CONN_ICN) {
      using upProto = ProtoSignalBulk<UNROLL, 1, 1, CONN_NORMAL, CONN_ICN>;
      using downProto = ProtoSignalBulk<UNROLL, 1, 1, CONN_ICN, CONN_ICN>;
      runTreeUpDown<FUNC, T, UNROLL, upProto, downProto>(work);
    } else {
      using upProto = ProtoSignalBulk<UNROLL, 1, 1, CONN_NORMAL, CONN_ICN>;
      using downProto = ProtoSignalBulk<UNROLL, 1, 1, CONN_NORMAL, CONN_ICN>;
      runTreeUpDown<FUNC, T, UNROLL, upProto, downProto>(work);
    }
#else
    if (upConnType == CONN_ICN && downConnType == CONN_ICN) {
      using upProto = ProtoSignalBulk<UNROLL, 1, 1, CONN_ICN, CONN_ICN>;
      using downProto = ProtoSignalBulk<UNROLL, 1, 1, CONN_ICN, CONN_ICN>;
      runSplitTree<FUNC, T, UNROLL, upProto, downProto>(work);
    } else if (upConnType == CONN_ICN && downConnType != CONN_ICN) {
      using upProto = ProtoSignalBulk<UNROLL, 1, 1, CONN_ICN, CONN_ICN>;
      using downProto = ProtoSignalBulk<UNROLL, 1, 1, CONN_NORMAL, CONN_ICN>;
      runSplitTree<FUNC, T, UNROLL, upProto, downProto>(work);
    } else if (upConnType != CONN_ICN && downConnType == CONN_ICN) {
      using upProto = ProtoSignalBulk<UNROLL, 1, 1, CONN_NORMAL, CONN_ICN>;
      using downProto = ProtoSignalBulk<UNROLL, 1, 1, CONN_ICN, CONN_ICN>;
      runSplitTree<FUNC, T, UNROLL, upProto, downProto>(work);
    } else {
      using upProto = ProtoSignalBulk<UNROLL, 1, 1, CONN_NORMAL, CONN_ICN>;
      using downProto = ProtoSignalBulk<UNROLL, 1, 1, CONN_NORMAL, CONN_ICN>;
      runSplitTree<FUNC, T, UNROLL, upProto, downProto>(work);
    }
#endif
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_SIGNAL_BULK, FUNC, T, UNROLL> {
  public:
__device__ void run(struct ncclDevWorkColl* work) { }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_FC, NCCL_PROTO_SIGNAL_BULK, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    const ncclConn_t connType = channel->fc.connType;
    if (connType == CONN_ICN) {
      using Proto = ProtoSignalBulk<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, CONN_ICN, CONN_ICN>;
      runFC<FUNC, T, UNROLL, Proto>(work);
    } else {
      using Proto = ProtoSignalBulk<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, CONN_NORMAL, CONN_ICN>;
      runFC<FUNC, T, UNROLL, Proto>(work);
    }
  }
};
