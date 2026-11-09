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

// Per-direction runners. Proto is the actual proto used for this direction — caller passes either
// the plan-wide base proto or ProtoLL after the per-pair LL upgrade. Pulled out so the dispatch
// site is a single ternary instead of two near-identical 12-line blocks.
template<class FUNC, typename T, typename Proto>
__device__ __forceinline__ void runRecv(
    int tid, int nt, int peer, T* recvbuff, ssize_t recvCount, int chunkSize,
    struct ncclKernelComm* comm, int group, struct ncclDevWorkP2p* work) {
  Primitives<T, FUNC, Proto, 1, 0, 1, 1>
    prims(tid, nt, &peer, NULL, recvbuff, comm, /*redOpArg(ignored)=*/0, getShmemPtr(comm)->groups, group, 1, 1, NULL, work);
  if (recvCount == 0) {
    prims.recv(recvbuff, 0);
  } else for (ssize_t offset = 0; offset < recvCount; offset += chunkSize) {
    // Match the proxy's exact chunk bound, including non-power-of-two sizes.
    int nelem = min(chunkSize, recvCount-offset);
    prims.directRecv(recvbuff+offset, offset, nelem);
  }
}

template<class FUNC, typename T, typename Proto>
__device__ __forceinline__ void runSend(
    int tid, int nt, int peer, const T* sendbuff, T* recvbuff, ssize_t sendCount, int chunkSize,
    struct ncclKernelComm* comm, int group, struct ncclDevWorkP2p* work) {
  Primitives<T, FUNC, Proto, 0, 1, 1, 1>
    prims(tid, nt, NULL, &peer, recvbuff, comm, /*redOpArg(ignored)=*/0, getShmemPtr(comm)->groups, group, 1, 1, NULL, work);
  if (sendCount == 0) {
    prims.send(sendbuff, 0);
  } else for (ssize_t offset = 0; offset < sendCount; offset += chunkSize) {
    // Match the proxy's exact chunk bound, including non-power-of-two sizes.
    int nelem = min(chunkSize, sendCount-offset);
    prims.directSend(sendbuff+offset, offset, nelem);
  }
}

template<typename Proto>
__device__ __noinline__ void preprocessP2pWork(struct ncclDevWorkP2p* work, struct ncclShmemData* shmem) {
  struct ncclKernelComm* comm = getShmemComm(shmem);
  // Like NCCL, partition this channel's shared-memory work copies once before
  // assigning threads to send/recv. Keep the host's stride and channel mapping.
  // Each lane owns one direction; private works and absent directions need no changes.
  // Active directions retain their channel count; only an out-of-range part clears it.
  if (threadIdx.x < WARP_SIZE) {
    int workIx = threadIdx.x % (WARP_SIZE/2);
    bool isSend = threadIdx.x >= WARP_SIZE/2;
    if (workIx < shmem->nWorks) {
      ncclDevWorkP2p* partWork = &work[workIx];
      if (partWork->sharedWork) {
        const int part = ncclP2pChannelToPart(shmem->args.comm, partWork->nP2pChannels, partWork->channelBase, shmem->channelId);
        const size_t step = comm->buffSizes[Proto::Id] / NCCL_STEPS;
        size_t offset;
        if (isSend && partWork->nSendChannels != 0) {
          ssize_t bytes = ncclP2pPartBytes(partWork->sendBytes, step * partWork->sendPartSteps,
              partWork->nSendChannels, part, &offset);
          partWork->sendBytes = bytes < 0 ? 0 : bytes;
          if (bytes < 0) partWork->nSendChannels = 0;
          if (offset) partWork->sendAddr = (const char*)partWork->sendAddr + offset;
        } else if (!isSend && partWork->nRecvChannels != 0) {
          ssize_t bytes = ncclP2pPartBytes(partWork->recvBytes, step * partWork->recvPartSteps,
              partWork->nRecvChannels, part, &offset);
          partWork->recvBytes = bytes < 0 ? 0 : bytes;
          if (bytes < 0) partWork->nRecvChannels = 0;
          if (offset) partWork->recvAddr = (char*)partWork->recvAddr + offset;
        }
      }
    }
  }
  __syncthreads(); // Publish the sliced work before any execution thread reads it.
}

template<class FUNC, typename T, int UNROLL, typename Proto>
__device__ __forceinline__ void runSendRecv(struct ncclDevWorkP2p* work, struct ncclShmemData* shmem) {
  static_assert(sizeof(T) == 1, "P2P partition strides are in bytes");
  static_assert(NCCL_MAX_DEV_WORK_P2P_PER_BATCH <= WARP_SIZE/2, "One warp partitions both directions");
  if (shmem->args.p2pPreprocess) preprocessP2pWork<Proto>(work, shmem);
  // Execution state is read after preprocessing, not kept live across its barrier.
  int tid = threadIdx.x;
  struct ncclKernelComm* comm = getShmemComm(shmem);
  int nWorks = shmem->nWorks;

  int group = 0;
  constexpr bool kProtoNeedsExtraWarp = (Proto::Id != NCCL_PROTO_LL);
  constexpr bool kAllowLLSwitch = (Proto::Id == NCCL_PROTO_SIMPLE || Proto::Id == NCCL_PROTO_SIMPLE_BULK || Proto::Id == NCCL_PROTO_SIMPLE_BULK_AR);
  int nThreadsSegment = 16 * WARP_SIZE;
  while (nWorks*nThreadsSegment > 512) nThreadsSegment /= 2;
#ifndef NO_EXTRA_WARP
  if (kProtoNeedsExtraWarp && nThreadsSegment >= 128) nThreadsSegment += WARP_SIZE;
#endif
  for (int s=0; s<nWorks; s++) {
    int groupRecv = group;
    group += 1;
    int groupSend = group;
    group += nThreadsSegment > 128 ? 2 : 1;
    if (tid < nThreadsSegment) {
      const int nThreads = (kProtoNeedsExtraWarp && nThreadsSegment > 128) ? nThreadsSegment-WARP_SIZE : nThreadsSegment;

      // Compute pointers
      const T* sendbuff = (const T*)work->sendAddr;
      T* recvbuff = (T*)work->recvAddr;
      const ssize_t sendCount = work->sendBytes;
      const ssize_t recvCount = work->recvBytes;

      if (work->sendRank == comm->rank) {
        if (tid < nThreads && sendbuff != recvbuff) {
          // local copy : ReduceOrCopyMulti takes an int as number of elements,
          // so we split it in blocks of 1G elements.
          int blockSize = 1<<30;
          for (ssize_t offset=0; offset<sendCount; offset += blockSize) {
            size_t remaining = sendCount - offset;
            if (remaining < blockSize) blockSize = remaining;
            ReduceOrCopyMultiBulk<ncclSendRecvLocalUnroll(), FUNC, T, 1, 1, 1, 1, 1, 1, Proto::RTMD, CONN_ICN, CONN_ICN>(tid, nThreads, FUNC(), false, false, 1, &sendbuff, 1, &recvbuff, blockSize);
            sendbuff += blockSize; recvbuff += blockSize;
          }
        }
      } else {
        int nThreadsSplit = nThreads/2;
        const bool recvLL = kAllowLLSwitch && work->recvProtoLL;
        const bool sendLL = kAllowLLSwitch && work->sendProtoLL;
        if ((tid < nThreadsSplit) && work->nRecvChannels != 0) {
          const int chunkSize = work->recvIpcReg ? NCCL_MAX_NET_SIZE : work->recvChunkSize()/sizeof(T);
          int peer = work->recvRank;
          int nt = nThreadsSplit;
          if (recvLL) {
            runRecv<FUNC, T, ProtoLL>(tid, nt, peer, recvbuff, recvCount, chunkSize, comm, groupRecv, work);
          } else {
            runRecv<FUNC, T, Proto>(tid, nt, peer, recvbuff, recvCount, chunkSize, comm, groupRecv, work);
          }
        }
        if ((tid >= nThreadsSplit) && work->nSendChannels != 0) {
          const int chunkSize = work->sendIpcReg ? NCCL_MAX_NET_SIZE : work->sendChunkSize()/sizeof(T);
          int peer = work->sendRank;
          int nt = sendLL ? (nThreadsSegment - nThreadsSplit) : (nThreads - nThreadsSplit);
          if (sendLL) {
            runSend<FUNC, T, ProtoLL>(tid-nThreadsSplit, nt, peer, sendbuff, recvbuff, sendCount, chunkSize, comm, groupSend, work);
          } else {
            runSend<FUNC, T, Proto>(tid-nThreadsSplit, nt, peer, sendbuff, recvbuff, sendCount, chunkSize, comm, groupSend, work);
          }
        }
      }
    }
    tid -= nThreadsSegment;
    if (tid < 0) break;
    work++;
  }
}

}

template<class FUNC, typename T, int UNROLL>
class RunWorkP2p<ncclFuncSendRecv, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkP2p* work, struct ncclShmemData* shmem) {
      using Proto = ProtoSimple<UNROLL, 1, 1>;
      runSendRecv<FUNC, T, UNROLL, Proto>(work, shmem);
    }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkP2p<ncclFuncSendRecv, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE_BULK, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkP2p* work, struct ncclShmemData* shmem) {
      using Proto = ProtoSimpleBulk<UNROLL, 1, 1>;
      runSendRecv<FUNC, T, UNROLL, Proto>(work, shmem);
    }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkP2p<ncclFuncSendRecv, NCCL_ALGO_RING, NCCL_PROTO_SIGNAL, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkP2p* work, struct ncclShmemData* shmem) {
      using Proto = ProtoSignal<UNROLL, 1, 1>;
      runSendRecv<FUNC, T, UNROLL, Proto>(work, shmem);
    }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkP2p<ncclFuncSendRecv, NCCL_ALGO_RING, NCCL_PROTO_SIGNAL_BULK, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkP2p* work, struct ncclShmemData* shmem) {
      using Proto = ProtoSignalBulk<UNROLL, 1, 1>;
      runSendRecv<FUNC, T, UNROLL, Proto>(work, shmem);
    }
};

template<class FUNC, typename T, int UNROLL>
class RunWorkP2p<ncclFuncSendRecv, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE_BULK_AR, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclDevWorkP2p* work, struct ncclShmemData* shmem) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
      using Proto = ProtoSimpleBulkAR<UNROLL, 1, 1>;
      runSendRecv<FUNC, T, UNROLL, Proto>(work, shmem);
#endif
    }
};
