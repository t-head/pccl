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
    int realChunkSize = min(chunkSize, recvCount-offset);
    ALIGN_SIZE(realChunkSize, nt*sizeof(uint64_t)/sizeof(T));
    int nelem = min(realChunkSize, recvCount-offset);
    prims.directRecv(recvbuff+offset, offset, nelem);
  }
}

template<class FUNC, typename T, typename Proto, bool NET_POLL = false>
__device__ __forceinline__ void runSend(
    int tid, int nt, int peer, const T* sendbuff, T* recvbuff, ssize_t sendCount, int chunkSize,
    struct ncclKernelComm* comm, int group, struct ncclDevWorkP2p* work) {
  Primitives<T, FUNC, Proto, 0, 1, 1, 1, NET_POLL>
    prims(tid, nt, NULL, &peer, recvbuff, comm, /*redOpArg(ignored)=*/0, getShmemPtr(comm)->groups, group, 1, 1, NULL, work);
  if constexpr (NET_POLL) prims.beginSendPoll(work);
  if (sendCount == 0) {
    prims.send(sendbuff, 0);
  } else for (ssize_t offset = 0; offset < sendCount; offset += chunkSize) {
    int realChunkSize = min(chunkSize, sendCount-offset);
    ALIGN_SIZE(realChunkSize, nt*sizeof(uint64_t)/sizeof(T));
    int nelem = min(realChunkSize, sendCount-offset);
    prims.directSend(sendbuff+offset, offset, nelem);
  }
}

template<class FUNC, typename T, int UNROLL, typename Proto, bool NET_POLL>
__device__ __forceinline__ void runSendRecv(struct ncclDevWorkP2p* work) {
  int tid = threadIdx.x;
  struct ncclKernelComm* comm = getShmemComm(work);
  int nWorks = getShmemPtr(comm)->nWorks;
  struct ncclDevChannel* channel = &(getShmemPtr(comm)->channel);
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
      const T* sendbuff = (const T*)work->sendbuff;
      T* recvbuff = (T*)work->recvbuff;
      const ssize_t sendCount = work->sendCount;
      const ssize_t recvCount = work->recvCount;

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
        struct ncclKernelComm* comm = getShmemComm(work);

        const int stepSize = comm->buffSizes[Proto::Id]/(sizeof(T)*NCCL_STEPS);

        int nThreadsSplit = nThreads/2;
        const bool recvLL = kAllowLLSwitch && work->recvProtoLL;
        const bool sendLL = kAllowLLSwitch && work->sendProtoLL;
        if ((tid < nThreadsSplit) && recvCount >= 0) {
          const int chunkSize = work->recvRegUsed ? NCCL_MAX_NET_SIZE : work->recvChunkSize/sizeof(T);
          int peer = work->recvRank;
          int nt = nThreadsSplit;
          if (recvLL) {
            runRecv<FUNC, T, ProtoLL>(tid, nt, peer, recvbuff, recvCount, chunkSize, comm, groupRecv, work);
          } else {
            runRecv<FUNC, T, Proto>(tid, nt, peer, recvbuff, recvCount, chunkSize, comm, groupRecv, work);
          }
        }
        if ((tid >= nThreadsSplit) && sendCount >= 0) {
          const int chunkSize = work->sendRegUsed ? NCCL_MAX_NET_SIZE : work->sendChunkSize/sizeof(T);
          int peer = work->sendRank;
          int nt = sendLL ? (nThreadsSegment - nThreadsSplit) : (nThreads - nThreadsSplit);
          if (sendLL) {
            runSend<FUNC, T, ProtoLL>(tid-nThreadsSplit, nt, peer, sendbuff, recvbuff, sendCount, chunkSize, comm, groupSend, work);
          } else {
            runSend<FUNC, T, Proto, NET_POLL>(tid-nThreadsSplit, nt, peer, sendbuff, recvbuff, sendCount, chunkSize, comm, groupSend, work);
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

template<class FUNC, typename T, int UNROLL, bool NET_POLL>
class RunWorkP2p<ncclFuncSendRecv, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL, NET_POLL> {
  public:
    __device__ void run(struct ncclDevWorkP2p* work) {
      using Proto = ProtoSimple<UNROLL, 1, 1>;
      runSendRecv<FUNC, T, UNROLL, Proto, NET_POLL>(work);
    }
};

template<class FUNC, typename T, int UNROLL, bool NET_POLL>
class RunWorkP2p<ncclFuncSendRecv, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE_BULK, FUNC, T, UNROLL, NET_POLL> {
  public:
    __device__ void run(struct ncclDevWorkP2p* work) {
      using Proto = ProtoSimpleBulk<UNROLL, 1, 1>;
      runSendRecv<FUNC, T, UNROLL, Proto, NET_POLL>(work);
    }
};

template<class FUNC, typename T, int UNROLL, bool NET_POLL>
class RunWorkP2p<ncclFuncSendRecv, NCCL_ALGO_RING, NCCL_PROTO_SIGNAL, FUNC, T, UNROLL, NET_POLL> {
  public:
    __device__ void run(struct ncclDevWorkP2p* work) {
      using Proto = ProtoSignal<UNROLL, 1, 1>;
      runSendRecv<FUNC, T, UNROLL, Proto, NET_POLL>(work);
    }
};

template<class FUNC, typename T, int UNROLL, bool NET_POLL>
class RunWorkP2p<ncclFuncSendRecv, NCCL_ALGO_RING, NCCL_PROTO_SIGNAL_BULK, FUNC, T, UNROLL, NET_POLL> {
  public:
    __device__ void run(struct ncclDevWorkP2p* work) {
      using Proto = ProtoSignalBulk<UNROLL, 1, 1>;
      runSendRecv<FUNC, T, UNROLL, Proto, NET_POLL>(work);
    }
};

template<class FUNC, typename T, int UNROLL, bool NET_POLL>
class RunWorkP2p<ncclFuncSendRecv, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE_BULK_AR, FUNC, T, UNROLL, NET_POLL> {
  public:
    __device__ void run(struct ncclDevWorkP2p* work) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
      using Proto = ProtoSimpleBulkAR<UNROLL, 1, 1>;
      runSendRecv<FUNC, T, UNROLL, Proto, NET_POLL>(work);
#endif
    }
};
