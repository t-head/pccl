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
  __device__ __forceinline__ int modRanks(int r, int nranks) {
    return r - (r >= nranks ? nranks : 0);
  }

  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRing(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL) {
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
    struct ncclRing* ring = (inputRing == NULL) ? &channel->ring : inputRing;
    const int nranks = (inputRing == NULL) ? comm->nRanks : inputRing->nRanks;
    const int ringIx = ring->index;

    int stepSize;
    ssize_t chunkSize;
    ssize_t minChunkSize;
    if (Proto::Id == NCCL_PROTO_LL) {
      stepSize = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
      chunkSize = stepSize * sizeof(uint64_t) / sizeof(T);
      minChunkSize = nworkers * (sizeof(uint64_t)) / sizeof(T);
    } else if (Proto::Id == NCCL_PROTO_LL128) {
      stepSize = comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
      chunkSize = stepSize*NCCL_LL128_DATAELEMS*sizeof(uint64_t) / (NCCL_LL128_LINEELEMS*sizeof(T));
      // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
      minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nworkers*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2;
    } else {
      stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
      chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
    }

    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = work->count;

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)work->sendbuff;
    T * __restrict__ thisOutput = (T*)work->recvbuff;

    Primitives<T, RedOp, FanAsymmetric<1, 1>, 1, Proto, 0>
      prims(tid, nthreads, &ring->prev, &ring->next, thisOutput, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);
    prims.setDataPtrs(thisInput, thisOutput);

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
      ssize_t realChunkSize;
      ssize_t offset;
      int nelem;
      int chunk;
      ssize_t chunkOffset;
      if (Proto::Id == NCCL_PROTO_LL || Proto::Id == NCCL_PROTO_LL128) {
        chunkSize = min(DIVUP(size-gridOffset, nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);
        chunk = modRanks(ringIx + nranks-1, nranks);
        offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
        nelem = min(chunkSize, size-offset);
      } else {
        realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*nChannels));
        ALIGN_SIZE(realChunkSize, nworkers*sizeof(uint64_t)/sizeof(T));
        chunkOffset = gridOffset + bid*nranks*realChunkSize;
        // step 0: push data to next GPU
        chunk = modRanks(ringIx + nranks-1, nranks);
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);
      }
      /////////////// begin AllReduce steps ///////////////

#ifdef KERNEL_LOOPBACK
      if (ring->prev == -1 && ring->next == -1) {
        prims.sendRecvLoopback(offset, offset, nelem);
        continue;
      }
#endif

      prims.directSend(offset, offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        chunk = modRanks(ringIx + nranks-j, nranks);
        if (Proto::Id == NCCL_PROTO_LL || Proto::Id == NCCL_PROTO_LL128) {
          offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
          nelem = min(chunkSize, size-offset);
        } else {
          offset = chunkOffset + chunk * realChunkSize;
          nelem = min(realChunkSize, size-offset);
        }

        prims.directRecvReduceDirectSend(offset, offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ringIx;
      if (Proto::Id == NCCL_PROTO_LL || Proto::Id == NCCL_PROTO_LL128) {
        offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
        nelem = min(chunkSize, size-offset);
      } else {
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);
      }

      prims.directRecvReduceCopyDirectSend(offset, offset, nelem, /*postOp=*/true);

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        chunk = modRanks(ringIx + nranks-j, nranks);
        if (Proto::Id == NCCL_PROTO_LL || Proto::Id == NCCL_PROTO_LL128) {
          offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
          nelem = min(chunkSize, size-offset);
        } else {
          offset = chunkOffset + chunk * realChunkSize;
          nelem = min(realChunkSize, size-offset);
        }

        prims.directRecvCopyDirectSend(offset, offset, nelem);
      }

      // Make final copy from buffer to dest.
      chunk = modRanks(ringIx + 1, nranks);
      if (Proto::Id == NCCL_PROTO_LL || Proto::Id == NCCL_PROTO_LL128) {
        offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
        nelem = min(chunkSize, size-offset);
      } else {
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);
      }

      // Final wait/copy.
      prims.directRecv(offset, nelem);
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
      nworkers = work->nThreads-WARP_SIZE;
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

    int stepSize;
    ssize_t chunkSize;
    if (Proto::Id == NCCL_PROTO_LL) {
      stepSize = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine) * NCCL_STEPS);
      chunkSize = stepSize * sizeof(uint64_t) / sizeof(T);
    } else if (Proto::Id == NCCL_PROTO_LL128) {
      stepSize = comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
      chunkSize = stepSize*NCCL_LL128_DATAELEMS*sizeof(uint64_t) / (NCCL_LL128_LINEELEMS*sizeof(T));
      ALIGN_SIZE_DOWN(chunkSize, nworkers*sizeof(uint64_t)/sizeof(T));
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

    Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_FC_ARITY-1, NCCL_MAX_FC_ARITY-1>, 1, Proto, 0>
      prims(tid, nthreads, fc->peerRanks, fc->peerRanks, thisOutput, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);
    prims.setDataPtrs(thisInput, thisOutput);

    if (Proto::Id == NCCL_PROTO_LL) {
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nChannels));
        ALIGN_SIZE(realChunkSize, nworkers*sizeof(uint64_t)/sizeof(T));
        ssize_t chunkOffset = gridOffset + bid*realChunkSize;
        int nelem = min(realChunkSize, size-chunkOffset);
        prims.send(chunkOffset, nelem);
        prims.recvReduceCopy(chunkOffset, chunkOffset, nelem);
      }
    } else {
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
        ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*nChannels));
        ALIGN_SIZE(realChunkSize, nworkers*sizeof(uint64_t)/sizeof(T));
        ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

        /////////////// begin AllReduce steps ///////////////
        ssize_t offset;
        int nelem;
        int peerElem;

        // step 0: scatter data with different offsets to all other peer GPUs
        nelem = min(realChunkSize * nranks, size-chunkOffset);
        peerElem = DIVUP(nelem, nranks);
        prims.scatter(chunkOffset, nelem, peerElem, peerElem);

        // step 1: reduce chunk offset data with all other peer GPUs
        offset = chunkOffset + chunk * peerElem;
        prims.recvReduceCopyDirectSend(offset,
            offset, min(peerElem, size-offset), /*postOp=*/true, peerElem);

        // step 2: final wait/copy.
        prims.directGather(chunkOffset, nelem, peerElem, peerElem);
      }
    }
  }

  template<typename T, typename RedOp, typename upProto, typename downProto>
  __device__ __forceinline__ void runTreeUpDown(struct ncclDevWorkColl* work) {
    const int tid = threadIdx.x;
#ifndef NO_EXTRA_WARP
    const int nworkers = work->nThreads-2*WARP_SIZE;
#else
    const int nworkers = work->nThreads;
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
    const ssize_t minChunkSize = nworkers*8*sizeof(uint64_t) / sizeof(T);
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = work->count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    const T * __restrict__ thisInput = (const T*)work->sendbuff;
    T * __restrict__ thisOutput = (T*)work->recvbuff;

#ifndef NO_EXTRA_WARP
    if (tid < nworkers+WARP_SIZE) {
#else
    if (tid < nworkers) {
#endif
      // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DEV_ARITY, 1>, 1, upProto, 0>
        prims(tid, ncclPrimsThreadsFromWorkers<upProto, 1>(nworkers), tree->down, &tree->up, thisOutput, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);
      prims.setDataPtrs(thisInput, thisOutput);

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Up
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
#ifdef KERNEL_LOOPBACK
        if (tree->up == -1 && tree->down[0] == -1) {
          prims.sendRecvLoopback(offset, offset, nelem);
          continue;
        }
#endif
        if (tree->up == -1) {
          prims.directRecvReduceCopy(offset, offset, nelem, /*postOp=*/true);
        } else if (tree->down[0] == -1) {
          prims.directSend(offset, offset, nelem);
        } else {
          prims.directRecvReduceDirectSend(offset, offset, nelem);
        }
      }
    }

#ifndef KERNEL_LOOPBACK
#ifndef NO_EXTRA_WARP
    if (tid < nworkers+WARP_SIZE) {
#else
    if (tid < nworkers) {
#endif
      // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DEV_ARITY>, 1, downProto, 0>
        prims(tid, ncclPrimsThreadsFromWorkers<downProto, NCCL_MAX_DEV_ARITY>(nworkers), &tree->up, tree->down, thisOutput, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work, NULL, 0, NCCL_STEPS, true);
      prims.setDataPtrs(thisInput, thisOutput);

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Down
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          // The root re-broadcasts the result it already reduced into the output
          // buffer, so it reads FromOutput.
          prims.directSendFromOutput(offset, nelem);
        } else if (tree->down[0] == -1) {
          prims.directRecv(offset, nelem);
        } else {
          prims.directRecvCopyDirectSend(offset, offset, nelem);
        }
      }
    }
#endif
  }

  template<typename T, typename RedOp, typename upProto, typename downProto>
  __device__ __forceinline__ void runSplitTree(struct ncclDevWorkColl* work) {
    const int tid = threadIdx.x;
    int nworkers;
#ifndef NO_EXTRA_WARP
    if (upProto::Id == NCCL_PROTO_LL || upProto::Id == NCCL_PROTO_LL128) {
      nworkers = work->nThreads;
    } else {
      nworkers = work->nThreads-2*WARP_SIZE;
    }
#else
    nworkers = work->nThreads;
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
      minChunkSize = nworkers*sizeof(uint64_t) / sizeof(T);
    } else if (upProto::Id == NCCL_PROTO_LL128) {
      stepSize = comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
      chunkSize = work->lastChunkSize;
      minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nworkers*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/8;
    } else {
      stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
      chunkSize = work->lastChunkSize;
      minChunkSize = nworkers*8*sizeof(uint64_t) / sizeof(T);
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
      nthreadsSplit = NCCL_LL128_SPLIT(nworkers);
    } else {
      nthreadsSplit = nworkers/2;
      if (nthreadsSplit == 256) nthreadsSplit += 64;
    }
    if (tree->up == -1) {
#ifndef NO_EXTRA_WARP
      if (tid < nworkers+WARP_SIZE || upProto::Id == NCCL_PROTO_LL128) {
#else
      if (tid < nworkers) {
#endif
        // ReduceAndBroadcast : max number of recv is 3, max number of send is 3
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DEV_ARITY, NCCL_MAX_DEV_ARITY>, 1, downProto, 0>
          prims(tid, ncclPrimsThreadsFromWorkers<downProto, NCCL_MAX_DEV_ARITY>(nworkers), tree->down, tree->down, thisOutput, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);
        prims.setDataPtrs(thisInput, thisOutput);

        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.directRecvReduceCopyDirectSend(offset, offset, nelem, /*postOp=*/true);
        }
      }
    } else {
#ifndef NO_EXTRA_WARP
      if (tid < nthreadsSplit + (upProto::Id == NCCL_PROTO_LL128 ? 0 : WARP_SIZE)) {
#else
      if (tid < nthreadsSplit) {
#endif
        /* Reduce up. Max number of recv is 3, max number of send is 1 (binary tree + local).
         * Why Direct=1????
         * Answer: Because despite not performing any direct operations, the ctor
         * must assume Direct so that it can exchange direct pointers with remote ctors
         * that are Direct, otherwise it hangs. A cleaner solution would be to seperate
         * into DirectRecv and DirectSend capabilities, this ctor would have both=0,
         * but the ctor above for tree roots would be DirectRecv=0 DirectSend=1.
         */
        // Coverity reports that the callee treats &tree->up as an array.  However, due to the use of
        // FanAsymmetric<n, 1>, only the first element is ever accessed, so it's fine.
        // coverity[callee_ptr_arith:FALSE]
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DEV_ARITY, 1>, 1, upProto, 0>
          prims(tid, ncclPrimsThreadsFromWorkers<upProto, 1>(nthreadsSplit), tree->down, &tree->up, thisOutput, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);
        prims.setDataPtrs(thisInput, thisOutput);

        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          // Up
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          if (tree->down[0] == -1) {
            prims.directSend(offset, offset, nelem);
          } else {
            prims.directRecvReduceDirectSend(offset, offset, nelem);
          }
        }
      } else {
        // Broadcast down. Max number of recv is 1, max number of send is 3 (binary tree + local)
        // Coverity reports that the callee treats &tree->up as an array.  However, due to the use of
        // FanAsymmetric<1, n>, only the first element is ever accessed, so it's fine.
        // coverity[callee_ptr_arith:FALSE]
        Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DEV_ARITY>, 1, downProto, 0>
#ifndef NO_EXTRA_WARP
          prims(tid-nthreadsSplit-(downProto::Id == NCCL_PROTO_LL128 ? 0 : WARP_SIZE),
                ncclPrimsThreadsFromWorkers<downProto, NCCL_MAX_DEV_ARITY>(nworkers-nthreadsSplit),
                &tree->up, tree->down, thisOutput, comm, work->redOpArg, getShmemPtr(comm)->groups, 2, 0, 0, work);
#else
          prims(tid-nthreadsSplit, ncclPrimsThreadsFromWorkers<downProto, NCCL_MAX_DEV_ARITY>(nworkers-nthreadsSplit), &tree->up, tree->down, thisOutput, comm, work->redOpArg, getShmemPtr(comm)->groups, 2, 0, 0, work);
#endif
        prims.setDataPtrs(thisInput, thisOutput);

        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          // Down
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          if (tree->down[0] == -1) {
            prims.directRecv(offset, nelem);
          } else {
            prims.directRecvCopyDirectSend(offset, offset, nelem);
          }
        }
      }
    }
  }
}

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL) {
    using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, UNROLL>;
    runRing<T, RedOp, Proto>(work, inputRing);
  }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<1, 1, UNROLL>;
#if 0
    runTreeUpDown<T, RedOp, Proto, Proto>(work);
#else
    runSplitTree<T, RedOp, Proto, Proto>(work);
#endif
  }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    const int tid = threadIdx.x;
#ifndef NO_EXTRA_WARP
    const int nworkers = work->nThreads-WARP_SIZE;
#else
    const int nworkers = work->nThreads;
#endif
    const int nthreads = ncclPrimsThreadsFromWorkers<ProtoSimple<1, 1, UNROLL>, 1>(nworkers);
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
    const ssize_t minChunkSize = nworkers*8*sizeof(uint64_t) / sizeof(T);
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = work->count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)work->sendbuff;
    T * __restrict__ thisOutput = (T*)work->recvbuff;

    if (blockIdx.x < nChannels) { // first half of the channels do reduce
      Primitives<T, RedOp, FanAsymmetric<1, 1>, 0, ProtoSimple<1, 1, UNROLL>, 0>
        prims(tid, nthreads, tree->down, &tree->up, NULL, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);
      prims.setDataPtrs(thisInput, thisOutput);

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Up
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          prims.recvReduceCopy(offset, offset, nelem);
        } else if (tree->down[0] == -1) {
          prims.send(offset, nelem);
        } else {
          prims.recvReduceSend(offset, nelem);
        }
      }
    }

    if (blockIdx.x >= nChannels) { // second half of the channels do broadcast
      Primitives<T, RedOp, FanAsymmetric<1, 1>, 0, ProtoSimple<1, 1, UNROLL>, 0>
        prims(tid, nthreads, &tree->up, tree->down, NULL, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);
      prims.setDataPtrs(thisInput, thisOutput);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Down
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          prims.sendFromOutput(offset, nelem);
        } else if (tree->down[0] == -1) {
          prims.recv(offset, nelem);
        } else {
          prims.recvCopySend(offset, nelem);
        }
      }
    }
  }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_FC, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, UNROLL>;
    runFC<T, RedOp, Proto>(work);
  }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_LL, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL) {
    runRing<T, RedOp, ProtoLL>(work, inputRing);
  }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_LL, RedOp, T, UNROLL> {
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
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DEV_ARITY, 1>, 0, ProtoLL, 0> LLprims(tid, nthreads, tree->down, &tree->up, NULL, comm, work->redOpArg);
      LLprims.setDataPtrs(thisInput, thisOutput);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Up
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);

#ifdef KERNEL_LOOPBACK
        if (tree->up == -1 && tree->down[0] == -1) {
          LLprims.sendRecvLoopback(offset, offset, nelem);
          continue;
        }
#endif
        if (tree->up == -1) {
          LLprims.recvReduceCopy(offset, offset, nelem, /*postOp=*/true);
        } else if (tree->down[0] == -1) {
          LLprims.send(offset, nelem);
        } else {
          LLprims.recvReduceSend(offset, nelem);
        }
      }
    } while(0);

#ifndef KERNEL_LOOPBACK
    do {
      // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DEV_ARITY>, 0, ProtoLL, 0> LLprims(tid, nthreads, &tree->up, tree->down, NULL, comm, work->redOpArg, NULL, 0, 0, 0, work, nullptr, 0, NCCL_STEPS, true);
      LLprims.setDataPtrs(thisInput, thisOutput);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Down
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          // Root re-broadcasts what it already reduced into the output buffer.
          LLprims.sendFromOutput(offset, nelem);
        } else if (tree->down[0] == -1) {
          LLprims.recv(offset, nelem);
        } else {
          LLprims.recvCopySend(offset, nelem);
        }
      }
    } while(0);
#endif
  }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_LL, RedOp, T, UNROLL> {
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
      Primitives<T, RedOp, FanAsymmetric<1, 1>, 0, ProtoLL, 0> LLprims(tid, nthreads, tree->down, &tree->up, NULL, comm, work->redOpArg);
      LLprims.setDataPtrs(thisInput, thisOutput);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Up
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          LLprims.recvReduceCopy(offset, offset, nelem);
        } else if (tree->down[0] == -1) {
          LLprims.send(offset, nelem);
        } else {
          LLprims.recvReduceSend(offset, nelem);
        }
      }
    }

    if (blockIdx.x >= nChannels) { // second half of the channels do broadcast
      Primitives<T, RedOp, FanAsymmetric<1, 1>, 0, ProtoLL, 0> LLprims(tid, nthreads, &tree->up, tree->down, NULL, comm, work->redOpArg);
      LLprims.setDataPtrs(thisInput, thisOutput);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Down
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          // Root re-broadcasts what it already reduced into the output buffer.
          LLprims.sendFromOutput(offset, nelem);
        } else if (tree->down[0] == -1) {
          LLprims.recv(offset, nelem);
        } else {
          LLprims.recvCopySend(offset, nelem);
        }
      }
    }
  }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_FC, NCCL_PROTO_LL, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    runFC<T, RedOp, ProtoLL>(work);
  }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_LL128, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL) {
    runRing<T, RedOp, ProtoLL128>(work, inputRing);
  }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_LL128, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    runSplitTree<T, RedOp, ProtoLL128, ProtoLL128>(work);
  }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_LL128, RedOp, T, UNROLL> {
  public:
__device__ void run(struct ncclDevWorkColl* work) { }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_FC, NCCL_PROTO_LL128, RedOp, T, UNROLL> {
  public:
__device__ void run(struct ncclDevWorkColl* work) {
    runFC<T, RedOp, ProtoLL128>(work);
  }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_SIGNAL, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL) {
    using Proto = ProtoSignal<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, UNROLL>;
    runRing<T, RedOp, Proto>(work, inputRing);
  }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_SIGNAL, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    using Proto = ProtoSignal<1, 1, UNROLL>;

#if 1
    runTreeUpDown<T, RedOp, Proto, Proto>(work);
#else
    runSplitTree<T, RedOp, Proto, Proto>(work);
#endif
  }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_SIGNAL, RedOp, T, UNROLL> {
  public:
__device__ void run(struct ncclDevWorkColl* work) { }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE_BULK, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL) {
    struct ncclKernelComm* comm = getShmemComm(work);
    const ncclConn_t sendConnType = getShmemPtr(comm)->channel.ring.sendConnType;

    if (sendConnType == CONN_ICN) {
      using Proto = ProtoSimpleBulk<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, UNROLL, CONN_ICN, CONN_ICN>;
      runRing<T, RedOp, Proto>(work, inputRing);
    } else {
      using Proto = ProtoSimpleBulk<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, UNROLL, CONN_NORMAL, CONN_ICN>;
      runRing<T, RedOp, Proto>(work, inputRing);
    }
  }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE_BULK_AR, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
    struct ncclKernelComm* comm = getShmemComm(work);
    const ncclConn_t sendConnType = getShmemPtr(comm)->channel.ring.sendConnType;

    if (sendConnType == CONN_ICN) {
      using Proto = ProtoSimpleBulkAR<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, UNROLL, CONN_ICN, CONN_ICN>;
      runRing<T, RedOp, Proto>(work, inputRing);
    } else {
      using Proto = ProtoSimpleBulkAR<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, UNROLL, CONN_NORMAL, CONN_ICN>;
      runRing<T, RedOp, Proto>(work, inputRing);
    }
#endif
  }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_FC, NCCL_PROTO_SIGNAL, RedOp, T, UNROLL> {
  public:
__device__ void run(struct ncclDevWorkColl* work) {
    using Proto = ProtoSignal<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, UNROLL>;
    runFC<T, RedOp, Proto>(work);
  }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE_BULK, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    const ncclConn_t upConnType = channel->tree.upConnType;
    const ncclConn_t downConnType = channel->tree.downConnType;

#if 1
    if (upConnType == CONN_ICN && downConnType == CONN_ICN) {
      using upProto = ProtoSimpleBulk<1, 1, UNROLL, CONN_ICN, CONN_ICN>;
      using downProto = ProtoSimpleBulk<1, 1, UNROLL, CONN_ICN, CONN_ICN>;
      runTreeUpDown<T, RedOp, upProto, downProto>(work);
    } else if (upConnType == CONN_ICN && downConnType != CONN_ICN) {
      using upProto = ProtoSimpleBulk<1, 1, UNROLL, CONN_ICN, CONN_ICN>;
      using downProto = ProtoSimpleBulk<1, 1, UNROLL, CONN_NORMAL, CONN_ICN>;
      runTreeUpDown<T, RedOp, upProto, downProto>(work);
    } else if (upConnType != CONN_ICN && downConnType == CONN_ICN) {
      using upProto = ProtoSimpleBulk<1, 1, UNROLL, CONN_NORMAL, CONN_ICN>;
      using downProto = ProtoSimpleBulk<1, 1, UNROLL, CONN_ICN, CONN_ICN>;
      runTreeUpDown<T, RedOp, upProto, downProto>(work);
    } else {
      using upProto = ProtoSimpleBulk<1, 1, UNROLL, CONN_NORMAL, CONN_ICN>;
      using downProto = ProtoSimpleBulk<1, 1, UNROLL, CONN_NORMAL, CONN_ICN>;
      runTreeUpDown<T, RedOp, upProto, downProto>(work);
    }
#else
    if (upConnType == CONN_ICN && downConnType == CONN_ICN) {
      using upProto = ProtoSimpleBulk<1, 1, UNROLL, CONN_ICN, CONN_ICN>;
      using downProto = ProtoSimpleBulk<1, 1, UNROLL, CONN_ICN, CONN_ICN>;
      runSplitTree<T, RedOp, upProto, downProto>(work);
    } else if (upConnType == CONN_ICN && downConnType != CONN_ICN) {
      using upProto = ProtoSimpleBulk<1, 1, UNROLL, CONN_ICN, CONN_ICN>;
      using downProto = ProtoSimpleBulk<1, 1, UNROLL, CONN_NORMAL, CONN_ICN>;
      runSplitTree<T, RedOp, upProto, downProto>(work);
    } else if (upConnType != CONN_ICN && downConnType == CONN_ICN) {
      using upProto = ProtoSimpleBulk<1, 1, UNROLL, CONN_NORMAL, CONN_ICN>;
      using downProto = ProtoSimpleBulk<1, 1, UNROLL, CONN_ICN, CONN_ICN>;
      runSplitTree<T, RedOp, upProto, downProto>(work);
    } else {
      using upProto = ProtoSimpleBulk<1, 1, UNROLL, CONN_NORMAL, CONN_ICN>;
      using downProto = ProtoSimpleBulk<1, 1, UNROLL, CONN_NORMAL, CONN_ICN>;
      runSplitTree<T, RedOp, upProto, downProto>(work);
    }
#endif
  }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_SIMPLE_BULK, RedOp, T, UNROLL> {
  public:
__device__ void run(struct ncclDevWorkColl* work) { }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_SIGNAL_BULK, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work, struct ncclRing* inputRing=NULL) {
    using Proto = ProtoSignalBulk<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, UNROLL>;
    runRing<T, RedOp, Proto>(work, inputRing);
  }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_FC, NCCL_PROTO_SIMPLE_BULK, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    const ncclConn_t connType = channel->fc.connType;
    if (connType == CONN_ICN) {
      using Proto = ProtoSimpleBulk<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, UNROLL, CONN_ICN, CONN_ICN>;
      runFC<T, RedOp, Proto>(work);
    } else {
      using Proto = ProtoSimpleBulk<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, UNROLL, CONN_NORMAL, CONN_ICN>;
      runFC<T, RedOp, Proto>(work);
    }
  }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_SIGNAL_BULK, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    const ncclConn_t upConnType = channel->tree.upConnType;
    const ncclConn_t downConnType = channel->tree.downConnType;

#if 1
    if (upConnType == CONN_ICN && downConnType == CONN_ICN) {
      using upProto = ProtoSignalBulk<1, 1, UNROLL, CONN_ICN, CONN_ICN>;
      using downProto = ProtoSignalBulk<1, 1, UNROLL, CONN_ICN, CONN_ICN>;
      runTreeUpDown<T, RedOp, upProto, downProto>(work);
    } else if (upConnType == CONN_ICN && downConnType != CONN_ICN) {
      using upProto = ProtoSignalBulk<1, 1, UNROLL, CONN_ICN, CONN_ICN>;
      using downProto = ProtoSignalBulk<1, 1, UNROLL, CONN_NORMAL, CONN_ICN>;
      runTreeUpDown<T, RedOp, upProto, downProto>(work);
    } else if (upConnType != CONN_ICN && downConnType == CONN_ICN) {
      using upProto = ProtoSignalBulk<1, 1, UNROLL, CONN_NORMAL, CONN_ICN>;
      using downProto = ProtoSignalBulk<1, 1, UNROLL, CONN_ICN, CONN_ICN>;
      runTreeUpDown<T, RedOp, upProto, downProto>(work);
    } else {
      using upProto = ProtoSignalBulk<1, 1, UNROLL, CONN_NORMAL, CONN_ICN>;
      using downProto = ProtoSignalBulk<1, 1, UNROLL, CONN_NORMAL, CONN_ICN>;
      runTreeUpDown<T, RedOp, upProto, downProto>(work);
    }
#else
    if (upConnType == CONN_ICN && downConnType == CONN_ICN) {
      using upProto = ProtoSignalBulk<1, 1, UNROLL, CONN_ICN, CONN_ICN>;
      using downProto = ProtoSignalBulk<1, 1, UNROLL, CONN_ICN, CONN_ICN>;
      runSplitTree<T, RedOp, upProto, downProto>(work);
    } else if (upConnType == CONN_ICN && downConnType != CONN_ICN) {
      using upProto = ProtoSignalBulk<1, 1, UNROLL, CONN_ICN, CONN_ICN>;
      using downProto = ProtoSignalBulk<1, 1, UNROLL, CONN_NORMAL, CONN_ICN>;
      runSplitTree<T, RedOp, upProto, downProto>(work);
    } else if (upConnType != CONN_ICN && downConnType == CONN_ICN) {
      using upProto = ProtoSignalBulk<1, 1, UNROLL, CONN_NORMAL, CONN_ICN>;
      using downProto = ProtoSignalBulk<1, 1, UNROLL, CONN_ICN, CONN_ICN>;
      runSplitTree<T, RedOp, upProto, downProto>(work);
    } else {
      using upProto = ProtoSignalBulk<1, 1, UNROLL, CONN_NORMAL, CONN_ICN>;
      using downProto = ProtoSignalBulk<1, 1, UNROLL, CONN_NORMAL, CONN_ICN>;
      runSplitTree<T, RedOp, upProto, downProto>(work);
    }
#endif
  }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_SIGNAL_BULK, RedOp, T, UNROLL> {
  public:
__device__ void run(struct ncclDevWorkColl* work) { }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_FC, NCCL_PROTO_SIGNAL_BULK, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    const ncclConn_t connType = channel->fc.connType;
    if (connType == CONN_ICN) {
      using Proto = ProtoSignalBulk<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, UNROLL, CONN_ICN, CONN_ICN>;
      runFC<T, RedOp, Proto>(work);
    } else {
      using Proto = ProtoSignalBulk<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, UNROLL, CONN_NORMAL, CONN_ICN>;
      runFC<T, RedOp, Proto>(work);
    }
  }
};
