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

    ssize_t size, partOffset, partCount, chunkSize;
    ncclCollCbdPart(work, getShmemPtr(comm)->channelId, Proto::Id, sizeof(T), &size, &partOffset, &partCount, &chunkSize);
    const ssize_t loopSize = nranks * chunkSize;

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)work->sendbuff;
    T * __restrict__ thisOutput = (T*)work->recvbuff;

    Primitives<T, FUNC, Proto, 1, 1, 1, 0>
      prims(tid, nthreads, &ring->prev, &ring->next, thisOutput, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);

    for (ssize_t elemOffset = 0; elemOffset < partCount; elemOffset += loopSize) {
      const ssize_t remaining = partCount - elemOffset;
      ssize_t chunkCount = chunkSize;
      if (remaining < loopSize) {
        chunkCount = DIVUP(remaining, nranks);
        ALIGN_SIZE(chunkCount, 16 / sizeof(T));
      }

      ssize_t offset;
      int nelem;
      auto selectChunk = [&](int ringIndex) {
        // Empty rank chunks still participate in both ring phases.
        const ssize_t chunkOffset = min(ring->devUserRanks[ringIndex] * chunkCount, remaining);
        offset = partOffset + elemOffset + chunkOffset;
        nelem = min(chunkCount, remaining - chunkOffset);
      };
      selectChunk(nranks-1);
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
        selectChunk(nranks-j);

        prims.recvReduceSend(thisInput+offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      selectChunk(0);

      prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem, /*postOp=*/true);

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        selectChunk(nranks-j);

        prims.directRecvCopySend(thisOutput+offset, offset, nelem);
      }

      // Make final copy from buffer to dest.
      selectChunk(1);

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
    ssize_t partOffset, partCount, chunkSize;
    ncclCollCbdPart(work, getShmemPtr(comm)->channelId, Proto::Id, sizeof(T), (ssize_t*)nullptr, &partOffset, &partCount, &chunkSize);
    int chunk = fc->rank;

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)work->sendbuff;
    T * __restrict__ thisOutput = (T*)work->recvbuff;

    Primitives<T, FUNC, Proto, NCCL_MAX_FC_ARITY-1, NCCL_MAX_FC_ARITY-1, 1, 0>
      prims(tid, nthreads, fc->peerRanks, fc->peerRanks, thisOutput, comm, work->redOpArg, getShmemPtr(comm)->groups, 0, 0, 0, work);

    if (Proto::Id == NCCL_PROTO_LL) {
      for (ssize_t elemOffset = 0; elemOffset < partCount; elemOffset += chunkSize) {
        const ssize_t chunkOffset = partOffset + elemOffset;
        int nelem = min(chunkSize, partCount - elemOffset);
        prims.send(thisInput+chunkOffset, nelem);
        prims.recvReduceCopy(thisInput+chunkOffset, thisOutput+chunkOffset, nelem, /*postOp=*/true);
      }
    } else {
      for (ssize_t elemOffset = 0; elemOffset < partCount; elemOffset += nranks*chunkSize) {
        const ssize_t chunkOffset = partOffset + elemOffset;

        /////////////// begin AllReduce steps ///////////////
        ssize_t offset;
        int nelem;
        int peerElem;

        // step 0: scatter data with different offsets to all other peer GPUs
        nelem = min(chunkSize * nranks, partCount - elemOffset);
        peerElem = DIVUP(nelem, nranks);
        prims.scatter(thisInput+chunkOffset, nelem, peerElem, peerElem);

        // step 1: reduce chunk offset data with all other peer GPUs
        offset = chunkOffset + chunk * peerElem;
        prims.directRecvReduceCopySend(thisInput+offset,
            thisOutput+offset, offset, min(peerElem, partCount - elemOffset - chunk * peerElem), /*postOp=*/true, peerElem);

        // step 2: final wait/copy.
        prims.directGather(thisOutput+chunkOffset, nelem, peerElem, peerElem, chunkOffset);
      }
    }
  }

  template<class FUNC, typename T, int UNROLL, typename upProto, typename downProto>
  __device__ __forceinline__ void runTreeUpDown(struct ncclDevWorkColl* work) {
    const int tid = threadIdx.x;
#ifndef NO_EXTRA_WARP
    const int nthreads = (work->nWarps * WARP_SIZE)-2*WARP_SIZE;
#else
    const int nthreads = (work->nWarps * WARP_SIZE);
#endif
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    struct ncclTree* tree = &channel->tree;
    ssize_t partOffset, partCount, chunkSize;
    ncclCollCbdPart(work, getShmemPtr(comm)->channelId, upProto::Id, sizeof(T), (ssize_t*)nullptr, &partOffset, &partCount, &chunkSize);

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

      for (ssize_t elemOffset = 0; elemOffset < partCount; elemOffset += chunkSize) {
        // Up
        const ssize_t offset = partOffset + elemOffset;
        int nelem = min(chunkSize, partCount - elemOffset);

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

      for (ssize_t elemOffset = 0; elemOffset < partCount; elemOffset += chunkSize) {
        // Down
        const ssize_t offset = partOffset + elemOffset;
        int nelem = min(chunkSize, partCount - elemOffset);
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
      nthreads = (work->nWarps * WARP_SIZE);
    } else {
      nthreads = (work->nWarps * WARP_SIZE)-2*WARP_SIZE;
    }
#else
    nthreads = (work->nWarps * WARP_SIZE);
#endif
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    struct ncclTree* tree = &channel->tree;
    ssize_t partOffset, partCount, chunkSize;
    ncclCollCbdPart(work, getShmemPtr(comm)->channelId, upProto::Id, sizeof(T), (ssize_t*)nullptr, &partOffset, &partCount, &chunkSize);

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

        for (ssize_t elemOffset = 0; elemOffset < partCount; elemOffset += chunkSize) {
          const ssize_t offset = partOffset + elemOffset;
          int nelem = min(chunkSize, partCount - elemOffset);
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

        for (ssize_t elemOffset = 0; elemOffset < partCount; elemOffset += chunkSize) {
          // Up
          const ssize_t offset = partOffset + elemOffset;
          int nelem = min(chunkSize, partCount - elemOffset);
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

        for (ssize_t elemOffset = 0; elemOffset < partCount; elemOffset += chunkSize) {
          // Down
          const ssize_t offset = partOffset + elemOffset;
          int nelem = min(chunkSize, partCount - elemOffset);
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
  __device__ void run(struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    runRing<FUNC, T, UNROLL, Proto>(work);
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
  __device__ void run(struct ncclDevWorkColl* work) {
    runRing<FUNC, T, UNROLL, ProtoLL>(work);
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    const int tid = threadIdx.x;
    const int nthreads = (work->nWarps * WARP_SIZE);
    struct ncclKernelComm* comm = getShmemComm(work);
    struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
    struct ncclTree* tree = &channel->tree;
    ssize_t partOffset, partCount, chunkSize;
    ncclCollCbdPart(work, getShmemPtr(comm)->channelId, NCCL_PROTO_LL, sizeof(T), (ssize_t*)nullptr, &partOffset, &partCount, &chunkSize);

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)work->sendbuff;
    T * __restrict__ thisOutput = (T*)work->recvbuff;

    do {
      // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      Primitives<T, FUNC, ProtoLL, NCCL_MAX_DEV_ARITY, 1, 0, 0> LLprims(tid, nthreads, tree->down, &tree->up, NULL, comm, work->redOpArg);
      for (ssize_t elemOffset = 0; elemOffset < partCount; elemOffset += chunkSize) {
        // Up
        const ssize_t offset = partOffset + elemOffset;
        int nelem = min(chunkSize, partCount - elemOffset);

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
      for (ssize_t elemOffset = 0; elemOffset < partCount; elemOffset += chunkSize) {
        // Down
        const ssize_t offset = partOffset + elemOffset;
        int nelem = min(chunkSize, partCount - elemOffset);
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
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_FC, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    runFC<FUNC, T, UNROLL, ProtoLL>(work);
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    runRing<FUNC, T, UNROLL, ProtoLL128>(work);
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
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_FC, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
__device__ void run(struct ncclDevWorkColl* work) {
    runFC<FUNC, T, UNROLL, ProtoLL128>(work);
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_SIGNAL, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    using Proto = ProtoSignal<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    runRing<FUNC, T, UNROLL, Proto>(work);
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
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE_BULK, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    struct ncclKernelComm* comm = getShmemComm(work);
    const ncclConn_t sendConnType = getShmemPtr(comm)->channel.ring.sendConnType;

    if (sendConnType == CONN_ICN) {
      using Proto = ProtoSimpleBulk<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, CONN_ICN, CONN_ICN>;
      runRing<FUNC, T, UNROLL, Proto>(work);
    } else {
      using Proto = ProtoSimpleBulk<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, CONN_NORMAL, CONN_ICN>;
      runRing<FUNC, T, UNROLL, Proto>(work);
    }
  }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE_BULK_AR, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
    struct ncclKernelComm* comm = getShmemComm(work);
    const ncclConn_t sendConnType = getShmemPtr(comm)->channel.ring.sendConnType;

    if (sendConnType == CONN_ICN) {
      using Proto = ProtoSimpleBulkAR<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, CONN_ICN, CONN_ICN>;
      runRing<FUNC, T, UNROLL, Proto>(work);
    } else {
      using Proto = ProtoSimpleBulkAR<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, CONN_NORMAL, CONN_ICN>;
      runRing<FUNC, T, UNROLL, Proto>(work);
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
class RunWorkColl<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_SIGNAL_BULK, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {
    using Proto = ProtoSignalBulk<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    runRing<FUNC, T, UNROLL, Proto>(work);
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
