/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_PRIMS_SIMPLE_BASE_H_
#define NCCL_PRIMS_SIMPLE_BASE_H_

enum primsMode { primsModeDefault = 0, primsModePatRs = 1, primsModePatAg = 2 };

// Implementation of primitive types
template <typename T, int NRECV, int NSEND, class FUNC, int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, int DIRECT, typename Proto, class RealPrimitives, int P2P>
class PrimitivesSimpleBase {
 public:
  struct ncclConnInfo* conn = NULL;
  int* connOffsFifoPtr = NULL;
  volatile uint64_t* connTailPtr = NULL;
  uint64_t connTailCache; // Cache last seen value
  int index; // Peer index I'm responsible for
  ssize_t peer = -1;
  int role = 0;
  uint64_t step;
  const T** srcs;
  T* buff;

  uint32_t spins = 0;
  uint32_t hangSpins = 0;
  uint32_t abort = 0;

  inline __device__ int internalCheckAbort(uint32_t& spins, uint32_t& hangSpins) {
    return checkAbort(comm, spins, abort, hangSpins);
  }

  template <int DIRECTPTR>
  inline __device__ T* directPtr(ssize_t directOffset) {
    return DIRECTPTR && direct ? direct+directOffset : buff+(step%ncclSteps)*stepSize;
  }

 private:
  const int tid;
  int nthreads;
  int nworkers;
  const int stepSize;
  int nrecv = 0;
  int nsend = 0;
  volatile int* connSizesFifoPtr = NULL;
  volatile uint64_t* connHeadPtr = NULL;
  uint64_t connHeadCache; // Cache last seen value
  const uint64_t ncclSteps;

  int group;
  T* direct = NULL;
  struct ncclKernelComm* comm;

  T** dsts;
  int *realSizes;

  int* totalSendSize;
  FUNC redOp;

 protected:
  __device__ __forceinline__ int getTid() const { return tid; }
  __device__ __forceinline__ int getNthreads() const { return nthreads; }
  __device__ __forceinline__ int getGroup() const { return group; }
  __device__ __forceinline__ struct ncclKernelComm* getComm() const { return comm; }
  __device__ __forceinline__ FUNC getRedOp() const { return redOp; }

 private:
  const uintptr_t ptrExchangeMagicNum = 0x5a5a5a5a5a5a5a5a;

  // Don't use barrier 0 as it's used by the final sync
  inline __device__ void barrier() {
    if (nthreads == WARP_SIZE) __syncwarp();
    else __ppu_barrier_sync(group+1, nthreads);
  }
  inline __device__ void subBarrier() {
    if (nworkers == nthreads) barrier();
    else __ppu_barrier_sync(group+2, nworkers);
  }

  template <int DST, int DIRECTSEND>
  inline __device__ void waitSend(ssize_t directOffset, int nbytes) {
    hangSpins = spins = 0;
    while (connHeadCache + ncclSteps < step + SLICESTEPS) {
      connHeadCache = volatileLoadU64(connHeadPtr);
      if (checkAbort(comm, spins, abort, hangSpins)) break;
    }
    if (connSizesFifoPtr) {
      connSizesFifoPtr[step%ncclSteps] = nbytes;
    }

    if (connOffsFifoPtr) dsts[DST+index] = buff + loadInt(connOffsFifoPtr + (step%NCCL_STEPS))/sizeof(T);
    else dsts[DST+index] = directPtr<DIRECTSEND>(directOffset);
    step += SLICESTEPS;
  }

  template <int SRC, int DIRECTRECV>
  inline __device__ void waitRecv(ssize_t directOffset) {
    hangSpins = spins = 0;
    while (connTailCache < step + SLICESTEPS) {
      connTailCache = volatileLoadU64(connTailPtr);
      if (checkAbort(comm, spins, abort, hangSpins)) break;
    }
    if (connOffsFifoPtr) srcs[SRC+index] = buff + loadInt(connOffsFifoPtr + (step%NCCL_STEPS))/sizeof(T);
    else srcs[SRC+index] = directPtr<DIRECTRECV>(directOffset);
    step += SLICESTEPS;
  }

  inline __device__ void postRecv() {
    step += SLICESTEPS;
    volatileStoreU64(connHeadPtr, step);
  }

  inline __device__ void postSend() {
    step += SLICESTEPS;
    volatileStoreU64(connTailPtr, step);
  }

  template<int SEND>
  __device__ __forceinline__ void memoryFence(int size) {
    if (SEND && (role & ROLE_POST_SEND) && size > 0 && index == 0) {
      __ppu_threadfence_system_uncache();
    }
  }

  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST>
  __device__ __forceinline__ void DirectReduceOrCopyMulti(bool postOp, int realSize) {
    // (1-SEND) is only there to avoid compilation errors in case NSEND=0 (and SEND=0).
    if (Proto::Id == NCCL_PROTO_SIMPLE || Proto::Id == NCCL_PROTO_SIGNAL) {
      ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, (1-SEND)+NSEND>(tid, nworkers, redOp, false, false, 1, srcs, nsend, dsts+1, realSize);
    } else {
      ReduceOrCopyMultiBulk<UNROLL, FUNC, T, 1, 1, 1, (1-SEND)+NSEND, 0, 0, Proto::RTMD, Proto::sendConnType, Proto::recvConnType>(tid, nworkers, redOp, false, false, 1, srcs, nsend, dsts+1, realSize);
    }

  }

  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST>
  __device__ __forceinline__ void IndirectReduceOrCopyMulti(bool postOp, int realSize) {
    if (Proto::Id == NCCL_PROTO_SIMPLE || Proto::Id == NCCL_PROTO_SIGNAL) {
      ReduceOrCopyMulti<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST>(tid, nworkers, redOp, SRC, postOp, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, realSize);
    } else {
      ReduceOrCopyMultiBulk<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST, SRC, DST, Proto::RTMD, Proto::sendConnType, Proto::recvConnType>(tid, nworkers, redOp, SRC, postOp, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, realSize);
    }
  }

  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST>
  inline __device__ void
  GenericOp(const T* srcPtr, T* dstPtr, int nelem, ssize_t directOffset,
      bool postOp, int reqChunkSize=0) {
    int offset = 0;
    int sliceSize = stepSize*SLICESTEPS;
    int dataSize = max(DIVUP(nelem, 16*SLICESPERCHUNK)*16, sliceSize/32);
    if (reqChunkSize > 0) {
      dataSize = max(DIVUP(reqChunkSize, 16*SLICESPERCHUNK)*16, sliceSize/32);
    }

    #pragma unroll
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(dataSize, nelem-offset));
      if (tid < nworkers) {
        if (SRC && (role & ROLE_SRC)) srcs[0] = srcPtr+offset;
        if (RECV && (role & ROLE_WAIT_RECV)) static_cast<RealPrimitives*>(this)->template waitRecv<SRC, DIRECTRECV>(directOffset+offset);
        if (DST && (role & ROLE_DST)) dsts[0] = dstPtr+offset;
        if (SEND && (role & ROLE_WAIT_SEND)) waitSend<DST, DIRECTSEND>(directOffset+offset, realSize*sizeof(T));
        if (realSize > 0) {
          subBarrier();
          if (DIRECTRECV && srcs[0] == dsts[0]) {
            // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
            if (SEND) {
              static_cast<RealPrimitives*>(this)->template DirectReduceOrCopyMulti<DIRECTRECV, DIRECTSEND, RECV, SEND, SRC, DST>(postOp, realSize);
            }
          } else {
            static_cast<RealPrimitives*>(this)->template IndirectReduceOrCopyMulti<DIRECTRECV, DIRECTSEND, RECV, SEND, SRC, DST>(postOp, realSize);
          }
        }
      }
      barrier();
      static_cast<RealPrimitives*>(this)->template memoryFence<SEND>(realSize);
      __syncwarp();
      if (SEND && (role & ROLE_POST_SEND)) postSend();
      if (RECV && (role & ROLE_POST_RECV)) postRecv();
      offset += realSize;
    }
  }

  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND>
  __device__ __forceinline__ void ScatterImpl(int* realSizes) {
    if (Proto::Id == NCCL_PROTO_SIMPLE || Proto::Id == NCCL_PROTO_SIGNAL) {
      ScatterGatherMulti<UNROLL, T, NSEND, FUNC, 1>(tid, nworkers, redOp, srcs, dsts, realSizes, nsend);
    } else {
      ScatterGatherMultiBulk<UNROLL, T, NSEND, SEND, FUNC, 1, Proto::RTMD, Proto::sendConnType>(tid, nworkers, redOp, srcs, dsts, realSizes, nsend);
    }
  }

  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND>
  __device__ __forceinline__ void GatherImpl(int* realSizes) {
    if (Proto::Id == NCCL_PROTO_SIMPLE || Proto::Id == NCCL_PROTO_SIGNAL) {
      ScatterGatherMulti<UNROLL, T, NRECV, FUNC, 0>(tid, nworkers, redOp, srcs, dsts, realSizes, nrecv);
    } else {
      ScatterGatherMultiBulk<UNROLL, T, NRECV, SEND, FUNC, 0, Proto::RTMD, Proto::sendConnType>(tid, nworkers, redOp, srcs, dsts, realSizes, nrecv);
    }
  }

  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND>
  __device__ __forceinline__ void
  ScatterGatherOp(const T* srcPtr, T* dstPtr, int totalElem, int peerElem, ssize_t rankOffset, ssize_t directOffset) {
    int offset = 0; // slice offset
    int sliceSize = stepSize*SLICESTEPS;
    int dataSize = max(DIVUP(peerElem, 16*SLICESPERCHUNK)*16, sliceSize/32);  // per-peer slice size

    #pragma unroll
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(dataSize, peerElem-offset));
      if (tid < nworkers) {
        if (SEND) {
          if (tid == 0) totalSendSize[slice] = 0;
          if (role & ROLE_WAIT_SEND) {
            srcs[index] = srcPtr + peer * rankOffset + offset;
            int realPeerSize = max(0, min(realSize, totalElem-peer*peerElem-offset));
            realSizes[index] = realPeerSize;
            waitSend<0, DIRECTSEND>(directOffset+offset, realPeerSize*sizeof(T));
          }
          subBarrier();
          if (realSizes[0] > 0) {
            ScatterImpl<DIRECTRECV, DIRECTSEND, RECV, SEND>(realSizes);
            if (tid == 0) totalSendSize[slice] += realSizes[0];
          }
        } else if (RECV) {
          if (role & ROLE_WAIT_RECV) {
            dsts[index] = dstPtr + peer * rankOffset + offset;
            int realPeerSize = max(0, min(realSize, totalElem-peer*peerElem-offset));
            realSizes[index] = realPeerSize;
            static_cast<RealPrimitives*>(this)->template waitRecv<0, DIRECTRECV>(DIRECTRECV*(directOffset+peer*rankOffset)+offset);
          }
          subBarrier();
          if (DIRECTRECV) {
            int doRecv = 0;
            for (int i=0; i<nrecv; i++) if (srcs[i] != dsts[i]) { doRecv++; break;}
            if (doRecv > 0) {
              GatherImpl<DIRECTRECV, DIRECTSEND, RECV, SEND>(realSizes);
            }
          } else {
            GatherImpl<DIRECTRECV, DIRECTSEND, RECV, SEND>(realSizes);
          }
        }
      }
      barrier();
      static_cast<RealPrimitives*>(this)->template memoryFence<SEND>(totalSendSize[slice]);
      __syncwarp();
      if (SEND && (role & ROLE_POST_SEND)) postSend();
      if (RECV && (role & ROLE_POST_RECV)) postRecv();
      offset += realSize;
    }
  }

  __device__ __forceinline__ void loadRecvConn(ncclDevChannelPeer *devChannelPeer, T* directBuff, int connIndex, struct ncclDevWorkColl* collWork, struct ncclDevWorkP2p* p2pWork) {
    // Caller already gates this with (role & (ROLE_WAIT_RECV|ROLE_POST_RECV)),
    // so devChannelPeer is guaranteed non-null here.
    conn = &devChannelPeer->recv[connIndex];
    step = conn->step;
    step = ROUNDUP(step, SLICESPERCHUNK*SLICESTEPS);
    if (role & ROLE_POST_RECV) {
      connHeadPtr = conn->head;
      // Return credits in case we rounded up.
      volatileStoreU64(connHeadPtr, step);
    }
    if (role & ROLE_WAIT_RECV) {
      buff = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
      if (DIRECT && ((P2P ? ((p2pWork != nullptr && p2pWork->recvRegUsed)) : (collWork != nullptr && collWork->recvRegUsed)) && (conn->direct & NCCL_DIRECT_GPU))) {
        hangSpins = spins = 0;
        void **slot = conn->ptrExchange;
        void *ptr;
        loadPtr(slot, ptr);
        // Wait for consumer to consume previous value before trampling it.
        while (ptr != NULL && !checkAbort(comm, spins, abort, hangSpins)) {
          loadPtr(slot, ptr);
        }
        direct = directBuff;
        // Encode pointer by XOR'ing against some address they definitely wouldn't send
        // since we want to allow them sending us nullptr while not colliding with
        // the empty slot value.
        uintptr_t exchange = reinterpret_cast<uintptr_t>(directBuff);
        if (!P2P) {
          int localPeer = comm->rankToLocalRank[peer];
          int regUsed = collWork != nullptr ? collWork->recvRegUsed : 0;
          if (regUsed) exchange = collWork->recvbuffOffset + collWork->recvbuffRmtAddrs[localPeer];
        }
        storePtr(slot, reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(exchange) ^ ptrExchangeMagicNum));
      }
      connTailPtr = conn->tail;
      connTailCache = volatileLoadU64(connTailPtr);
      connOffsFifoPtr = conn->offsFifo;
    }
  }

  __device__ __forceinline__ void loadSendConn(ncclDevChannelPeer *devChannelPeer, int connIndex, struct ncclDevWorkColl* collWork, struct ncclDevWorkP2p* p2pWork) {
    // Caller already gates this with (role & (ROLE_WAIT_SEND|ROLE_POST_SEND)),
    // so devChannelPeer is guaranteed non-null here.
    conn = &devChannelPeer->send[connIndex];
    step = conn->step;
    step = ROUNDUP(step, SLICESPERCHUNK*SLICESTEPS);
    if (role & ROLE_POST_SEND) {
      connTailPtr = conn->tail;
    }
    if (role & ROLE_WAIT_SEND) {
      buff = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
      if (DIRECT && ((P2P ? ((p2pWork != nullptr && p2pWork->sendRegUsed)) : (collWork != nullptr && collWork->sendRegUsed)) && (conn->direct & NCCL_DIRECT_GPU))) {
        hangSpins = spins = 0;
        void **slot = conn->ptrExchange;
        void *ptr = NULL;
        loadPtr(slot, ptr);
        while (ptr == NULL && !checkAbort(comm, spins, abort, hangSpins)) {
          loadPtr(slot, ptr);
        }
        direct = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(ptr) ^ ptrExchangeMagicNum);
        storePtr(slot, (T*)NULL);
      }
      connHeadPtr = conn->head;
      connHeadCache = volatileLoadU64(connHeadPtr);
      connSizesFifoPtr = conn->sizesFifo;
      connOffsFifoPtr = conn->offsFifo;
    }
  }

 public:
  __device__ __forceinline__
  PrimitivesSimpleBase(
      const int tid, const int nworkers, int* recvPeers, int* sendPeers, T* directBuff,
      struct ncclKernelComm* comm, uint64_t redOpArg, struct ncclShmemGroup* ptrs, int group, uint8_t connIndexRecv = 0, uint8_t connIndexSend = 0,
      struct ncclDevWorkColl* collWork = nullptr, struct ncclDevWorkP2p* p2pWork = nullptr, const uint64_t ncclSteps=NCCL_STEPS, bool noRed=false,
      int mode = primsModeDefault
      ):
    // WR: fix potential double preOp for up/down tree.
    redOp(noRed ? RedOpIdentityArg<FUNC>::value() : redOpArg),
    comm(comm), tid(tid), nworkers(nworkers), stepSize(comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/sizeof(T)),
    srcs((const T**)ptrs[group].srcs), dsts((T**)ptrs[group].dsts), realSizes(ptrs[group].realSizes), group(group),
    totalSendSize(ptrs[group].totalSendSize), ncclSteps(ncclSteps) {
    nthreads = nworkers;
    if (mode == primsModeDefault) {
#ifndef NO_EXTRA_WARP
      // For send operations, we need an extra warp to overlap the threadfence and the copy
      int postThreads = NSEND && nworkers >= 64 ? WARP_SIZE : 0;
      nthreads += postThreads;
#endif

      // Make sure step is updated before we read it.
      barrier();

      for (int i=0; i<NRECV; i++) if (recvPeers[i] != -1) nrecv++;
      for (int i=0; i<NSEND; i++) if (sendPeers[i] != -1) nsend++;

      #define SYNC_GROUP 8
      static_assert(NSEND < SYNC_GROUP && NRECV < SYNC_GROUP, "Not enough threads to cover all peers");

      int g = tid / SYNC_GROUP;
      int ng = nthreads / SYNC_GROUP;
      index = tid % SYNC_GROUP;

      if (g == 0) {
        if (index < nrecv) role |= ROLE_WAIT_RECV;
        if (index == nrecv) role |= ROLE_SRC;
      } else if (g == 1) {
        if (index < nsend) role |= ROLE_WAIT_SEND;
        if (index == nsend) role |= ROLE_DST;
      } else if (g == ng - 2) {
        if (index < nrecv) role |= ROLE_POST_RECV;
      } else if (g == ng - 1) {
        if (index < nsend) role |= ROLE_POST_SEND;
      }

      if (role & (ROLE_WAIT_RECV|ROLE_POST_RECV)) peer = recvPeers[index];
      if (role & (ROLE_WAIT_SEND|ROLE_POST_SEND)) peer = sendPeers[index];

      if (role & (ROLE_WAIT_RECV|ROLE_POST_RECV))
        loadRecvConn(getShmemPtr(comm)->channel.peers[peer], directBuff, connIndexRecv, collWork, p2pWork);
      if (role & (ROLE_WAIT_SEND|ROLE_POST_SEND))
        loadSendConn(getShmemPtr(comm)->channel.peers[peer], connIndexSend, collWork, p2pWork);
    }
  }

  __device__ __forceinline__ void
  send(const T* src, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 0>(src, NULL, nelem, 0, false);
  }
  __device__ __forceinline__ void
  directSend(const T* src, ssize_t directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 0>(src, NULL, nelem, directOffset, false);
  }

  __device__ __forceinline__ void
  recv(T* dst, int nelem, bool postOp=false) {
    GenericOp<0, 0, 1, 0, 0, 1>(NULL, dst, nelem, 0, postOp);
  }
  __device__ __forceinline__ void
  directRecv(T* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 0, 1, 0, 0, 1>(NULL, dst, nelem, directOffset, false);
  }

  __device__ __forceinline__ void
  copySend(const T* src, T* dst, int nelem, bool postOp=false) {
    GenericOp<0, 0, 0, 1, 1, 1>(src, dst, nelem, 0, postOp);
  }
  __device__ __forceinline__ void
  directCopySend(const T* src, T* dst, ssize_t directOffset, int nelem, bool postOp=false) {
    GenericOp<0, 1, 0, 1, 1, 1>(src, dst, nelem, directOffset, postOp);
  }

  __device__ __forceinline__ void
  recvCopySend(T* dst, int nelem, bool postOp=false) {
    GenericOp<0, 0, 1, 1, 0, 1>(NULL, dst, nelem, 0, postOp);
  }
  __device__ __forceinline__ void
  directRecvCopySend(T* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 1, 1, 1, 0, 1>(NULL, dst, nelem, directOffset, false);
  }

  __device__ __forceinline__ void
  recvReduceCopy(const T* src, T* dst, int nelem, bool postOp=false) {
    GenericOp<0, 0, 1, 0, 1, 1>(src, dst, nelem, 0, postOp);
  }

#ifdef KERNEL_LOOPBACK
  __device__ __forceinline__ void
  sendRecvLoopback(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 0, 0, 1, 1>(src, dst, nelem, 0, false);
  }
#endif

  __device__ __forceinline__ void
  recvReduceSend(const T* src, int nelem, bool postOp=false) {
    GenericOp<0, 0, 1, 1, 1, 0>(src, NULL, nelem, 0, postOp);
  }

  __device__ __forceinline__ void
  recvReduceCopySend(const T* src, T* dst, int nelem, bool postOp=false) {
    GenericOp<0, 0, 1, 1, 1, 1>(src, dst, nelem, 0, postOp);
  }
  __device__ __forceinline__ void
  directRecvReduceCopySend(const T* src, T* dst, ssize_t directOffset, int nelem,
      bool postOp=false, int reqChunkSize=0) {
    // Direct is only for the send part
    GenericOp<0, 1, 1, 1, 1, 1>(src, dst, nelem, directOffset, postOp, reqChunkSize);
  }

  __device__ __forceinline__ void
  scatter(const T* src, int totalElem, int peerElem, ssize_t rankOffset) {
    ScatterGatherOp<0, 0, 0, 1>(src, NULL, totalElem, peerElem, rankOffset, 0);
  }

  __device__ __forceinline__ void
  gather(T* dst, int totalElem, int peerElem, ssize_t rankOffset) {
    ScatterGatherOp<0, 0, 1, 0>(NULL, dst, totalElem, peerElem, rankOffset, 0);
  }

  __device__ __forceinline__ void
  directGather(T* dst, int totalElem, int peerElem, ssize_t rankOffset, ssize_t directOffset) {
    ScatterGatherOp<1, 0, 1, 0>(NULL, dst, totalElem, peerElem, rankOffset, directOffset);
  }

  __device__ __forceinline__ ~PrimitivesSimpleBase() {
    // Save steps for the next operation
    if (role & (ROLE_POST_SEND|ROLE_POST_RECV))
      volatileStoreU64(reinterpret_cast<volatile uint64_t*>(&conn->step), step);
    barrier();
  }
};

#endif
