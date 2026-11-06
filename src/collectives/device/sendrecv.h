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

__device__ __forceinline__ struct ncclShmemData* p2pRunShmem(void* args) {
#ifdef NO_TSM
  return (struct ncclShmemData*)args;
#else
  (void)args;
  return &pcclShmem;
#endif
}

template<typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runRecv(int tid, int workers, int peer, T* recvbuff,
    ssize_t recvCount, int chunkSize, struct ncclKernelComm* comm, int group, struct ncclDevWorkP2p* work) {
  int total = ncclPrimsThreadsFromWorkers<Proto, 0>(workers);
  Primitives<T, RedOp, FanAsymmetric<1, 0>, 1, Proto, 1> prims(
      tid, total, &peer, nullptr, recvbuff, comm, 0, getShmemPtr(comm)->groups, group, 1, 1, nullptr, work);
  prims.setDataPtrs(nullptr, recvbuff);
  if (recvCount == 0) {
    prims.recv(0, 0);
  } else for (ssize_t offset = 0; offset < recvCount; offset += chunkSize) {
    int realChunkSize = min(chunkSize, recvCount-offset);
    ALIGN_SIZE(realChunkSize, workers*sizeof(uint64_t)/sizeof(T));
    int nelem = min(realChunkSize, recvCount-offset);
    prims.directRecv(offset, nelem);
  }
}

template<typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runSend(int tid, int workers, int peer, const T* sendbuff, T* recvbuff,
    ssize_t sendCount, int chunkSize, struct ncclKernelComm* comm, int group, struct ncclDevWorkP2p* work) {
  int total = ncclPrimsThreadsFromWorkers<Proto, 1>(workers);
  Primitives<T, RedOp, FanAsymmetric<0, 1>, 1, Proto, 1> prims(
      tid, total, nullptr, &peer, recvbuff, comm, 0, getShmemPtr(comm)->groups, group, 1, 1, nullptr, work);
  prims.setDataPtrs(sendbuff, nullptr);
  if (sendCount == 0) {
    prims.send(0, 0);
  } else for (ssize_t offset = 0; offset < sendCount; offset += chunkSize) {
    int realChunkSize = min(chunkSize, sendCount-offset);
    ALIGN_SIZE(realChunkSize, workers*sizeof(uint64_t)/sizeof(T));
    int nelem = min(realChunkSize, sendCount-offset);
    prims.directSend(offset, offset, nelem);
  }
}

template<typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runSendRecv(void* args) {
  static_assert(sizeof(T) == 1, "SendRecv only works on single byte types T.");
  struct ncclShmemData* shmem = p2pRunShmem(args);
  struct ncclDevWorkP2p* work = (struct ncclDevWorkP2p*)shmem->workStorage;
  int tid = threadIdx.x;
  int nWorks = shmem->nWorks;
  constexpr bool kProtoNeedsExtraWarp = Proto::Id != NCCL_PROTO_LL;
  constexpr bool kAllowLLSwitch = Proto::Id == NCCL_PROTO_SIMPLE || Proto::Id == NCCL_PROTO_SIMPLE_BULK || Proto::Id == NCCL_PROTO_SIMPLE_BULK_AR;
  int nThreadsSegment = 16 * WARP_SIZE;
  while (nWorks*nThreadsSegment > 512) nThreadsSegment /= 2;
#ifndef NO_EXTRA_WARP
  if (kProtoNeedsExtraWarp && nThreadsSegment >= 128) nThreadsSegment += WARP_SIZE;
#endif
  int group = 0;
  for (int s=0; s<nWorks; s++) {
    int groupRecv = group++;
    int groupSend = group;
    group += nThreadsSegment > 128 ? 2 : 1;
    if (tid < nThreadsSegment) {
      int nThreads = nThreadsSegment;
#ifndef NO_EXTRA_WARP
      if (kProtoNeedsExtraWarp && nThreadsSegment > 128) nThreads -= WARP_SIZE;
#endif
      struct ncclKernelComm* comm = getShmemComm(work);
      const T* sendbuff = (const T*)work->sendbuff;
      T* recvbuff = (T*)work->recvbuff;
      const ssize_t sendCount = work->sendCount;
      const ssize_t recvCount = work->recvCount;

      if (work->sendRank == comm->rank) {
        if (tid < nThreads && sendbuff != recvbuff) {
          // Same local unroll, 1G element blocking and same-address skip as the
          // old path.
          const int blockSize = 1 << 30;
          for (ssize_t offset=0; offset<sendCount; offset+=blockSize) {
            int count = min(blockSize, sendCount-offset);
            ReduceOrCopyMultiBulk<ncclSendRecvLocalUnroll(), RedOp, T, 1, 1, 1, 1, 1, 1, Proto::RTMD, CONN_ICN, CONN_ICN>(
                tid, nThreads, RedOp(), false, false, 1, &sendbuff, 1, &recvbuff, count);
            sendbuff += count;
            recvbuff += count;
          }
        }
      } else {
        int nThreadsSplit = nThreads/2;
        const bool recvLL = kAllowLLSwitch && work->recvProtoLL;
        const bool sendLL = kAllowLLSwitch && work->sendProtoLL;
        if ((tid < nThreadsSplit) && recvCount >= 0) {
          const int chunkSize = work->recvRegUsed ? NCCL_MAX_NET_SIZE : work->recvChunkSize/sizeof(T);
          int peer = work->recvRank;
          int nt = nThreadsSplit;
          if (recvLL) {
            runRecv<T, RedOp, ProtoLL>(tid, nt, peer, recvbuff, recvCount, chunkSize, comm, groupRecv, work);
          } else {
            runRecv<T, RedOp, Proto>(tid, nt, peer, recvbuff, recvCount, chunkSize, comm, groupRecv, work);
          }
        }
        if ((tid >= nThreadsSplit) && sendCount >= 0) {
          const int chunkSize = work->sendRegUsed ? NCCL_MAX_NET_SIZE : work->sendChunkSize/sizeof(T);
          int peer = work->sendRank;
          int nt = sendLL ? (nThreadsSegment - nThreadsSplit) : (nThreads - nThreadsSplit);
          if (sendLL) {
            runSend<T, RedOp, ProtoLL>(tid-nThreadsSplit, nt, peer, sendbuff, recvbuff, sendCount, chunkSize, comm, groupSend, work);
          } else {
            runSend<T, RedOp, Proto>(tid-nThreadsSplit, nt, peer, sendbuff, recvbuff, sendCount, chunkSize, comm, groupSend, work);
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

template<typename RedOp, typename T, int UNROLL>
class RunWorkBatch<ncclFuncSendRecv, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
    __device__ void run(void* args) {
      using Proto = ProtoSimple<1, 1, UNROLL>;
      runSendRecv<T, RedOp, Proto>(args);
    }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkBatch<ncclFuncSendRecv, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE_BULK, RedOp, T, UNROLL> {
  public:
    __device__ void run(void* args) {
      using Proto = ProtoSimpleBulk<1, 1, UNROLL>;
      runSendRecv<T, RedOp, Proto>(args);
    }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkBatch<ncclFuncSendRecv, NCCL_ALGO_RING, NCCL_PROTO_SIGNAL, RedOp, T, UNROLL> {
  public:
    __device__ void run(void* args) {
      using Proto = ProtoSignal<1, 1, UNROLL>;
      runSendRecv<T, RedOp, Proto>(args);
    }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkBatch<ncclFuncSendRecv, NCCL_ALGO_RING, NCCL_PROTO_SIGNAL_BULK, RedOp, T, UNROLL> {
  public:
    __device__ void run(void* args) {
      using Proto = ProtoSignalBulk<1, 1, UNROLL>;
      runSendRecv<T, RedOp, Proto>(args);
    }
};

template<typename RedOp, typename T, int UNROLL>
class RunWorkBatch<ncclFuncSendRecv, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE_BULK_AR, RedOp, T, UNROLL> {
  public:
    __device__ void run(void* args) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
      using Proto = ProtoSimpleBulkAR<1, 1, UNROLL>;
      runSendRecv<T, RedOp, Proto>(args);
#else
      (void)args;
#endif
    }
};
