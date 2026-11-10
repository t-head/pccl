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
template<typename T, typename Fan, typename RedOp, int SlicePerChunk, int StepPerSlice, int Unroll, int Direct, typename Proto, class RealPrimitives, int P2p>
class PrimitivesSimpleBase {
  static constexpr int MaxRecv = Fan::MaxRecv, MaxSend = Fan::MaxSend;
  static constexpr int Input=0, Output=1;
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

  template <int DirectPtr>
  inline __device__ T* directPtr(ssize_t directOffset) {
    return DirectPtr && direct ? direct+directOffset : buff+(step%NCCL_STEPS)*buffStepSize;
  }

 private:
  const int tid;
  int nthreads;
  int nworkers;
  const int stepSize;
  const int buffStepSize;
  Fan fan;
  volatile int* connSizesFifoPtr = NULL;
  volatile uint64_t* connHeadPtr = NULL;
  uint64_t connHeadCache; // Cache last seen value

  int group;
  T* direct = NULL;
  struct ncclKernelComm* comm;

  T** dsts;
  int *realSizes;

  int* totalSendSize;
  RedOp redOp;

 protected:
  __device__ __forceinline__ int getTid() const { return tid; }
  __device__ __forceinline__ int getNthreads() const { return nthreads; }
  __device__ __forceinline__ int getGroup() const { return group; }
  __device__ __forceinline__ struct ncclKernelComm* getComm() const { return comm; }
  __device__ __forceinline__ RedOp getRedOp() const { return redOp; }

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

  template <int Dst, int DirectSend>
  inline __device__ void waitSend(ssize_t directOffset, int nbytes) {
    hangSpins = spins = 0;
    while (connHeadCache + NCCL_STEPS < step + StepPerSlice) {
      connHeadCache = volatileLoadU64(connHeadPtr);
      if (checkAbort(comm, spins, abort, hangSpins)) break;
    }
    if (connSizesFifoPtr) {
      connSizesFifoPtr[step%NCCL_STEPS] = nbytes;
    }

    if (connOffsFifoPtr) dsts[Dst+index] = buff + loadInt(connOffsFifoPtr + (step%NCCL_STEPS))/sizeof(T);
    else dsts[Dst+index] = directPtr<DirectSend>(directOffset);
    step += StepPerSlice;
  }

  template <int Src, int DirectRecv>
  inline __device__ void waitRecv(ssize_t directOffset) {
    hangSpins = spins = 0;
    while (connTailCache < step + StepPerSlice) {
      connTailCache = volatileLoadU64(connTailPtr);
      if (checkAbort(comm, spins, abort, hangSpins)) break;
    }
    if (connOffsFifoPtr) srcs[Src+index] = buff + loadInt(connOffsFifoPtr + (step%NCCL_STEPS))/sizeof(T);
    else srcs[Src+index] = directPtr<DirectRecv>(directOffset);
    step += StepPerSlice;
  }

  inline __device__ void postRecv() {
    step += StepPerSlice;
    volatileStoreU64(connHeadPtr, step);
  }

  inline __device__ void postSend() {
    step += StepPerSlice;
    volatileStoreU64(connTailPtr, step);
  }

  template<int Send>
  __device__ __forceinline__ void memoryFence(int size) {
    if (Send && (role & ROLE_POST_SEND) && size > 0 && index == 0) {
      __ppu_threadfence_system_uncache();
    }
  }

  template <int DirectRecv, int DirectSend, int Recv, int Send, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void DirectReduceOrCopyMulti(bool postOp, int realSize) {
    if constexpr (Proto::Id == NCCL_PROTO_SIMPLE || Proto::Id == NCCL_PROTO_SIGNAL) {
      reduceCopy<Unroll, RedOp, T, 0, 1, 1, 0, 1, (1-Send)+MaxSend, 0, int>
        (tid, nworkers, getShmemPtr(comm)->groups[group].redOpArgs, /*preOpArgs=*/&getShmemPtr(comm)->groups[group].redOpArgs, /*postOp=*/false,
         1, (void**)srcs, fan.nsend(), (void**)(dsts+1), realSize);
    } else {
      ReduceOrCopyMultiBulk<Unroll, RedOp, T, 1, 1, 1, (1-Send)+MaxSend, 0, 0, Proto::RTMD, Proto::sendConnType, Proto::recvConnType>(tid, nworkers, redOp, false, false, 1, srcs, fan.nsend(), dsts+1, realSize);
    }

  }

  template <int DirectRecv, int DirectSend, int Recv, int Send, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void IndirectReduceOrCopyMulti(bool postOp, int realSize) {
    constexpr int Src = SrcBuf != -1 ? 1 : 0;
    constexpr int Dst = DstBuf != -1 ? 1 : 0;
    constexpr int PreOpSrcs = SrcBuf != Input ? 0 : 1;
    if constexpr (Proto::Id == NCCL_PROTO_SIMPLE || Proto::Id == NCCL_PROTO_SIGNAL) {
      reduceCopy<Unroll, RedOp, T, 0, Recv+Src, Recv*MaxRecv+Src, 0, Send+Dst, Send*MaxSend+Dst, PreOpSrcs, int>
        (tid, nworkers, getShmemPtr(comm)->groups[group].redOpArgs, /*preOpArgs=*/&getShmemPtr(comm)->groups[group].redOpArgs, postOp,
         Recv*fan.nrecv()+Src, (void**)srcs, Send*fan.nsend()+Dst, (void**)dsts, realSize);
    } else {
      ReduceOrCopyMultiBulk<Unroll, RedOp, T, Recv+Src, Recv*MaxRecv+Src, Send+Dst, Send*MaxSend+Dst, Src, Dst, Proto::RTMD, Proto::sendConnType, Proto::recvConnType>(tid, nworkers, redOp, PreOpSrcs, postOp, Recv*fan.nrecv()+Src, srcs, Send*fan.nsend()+Dst, dsts, realSize);
    }
  }

  template <int DirectRecv1, int DirectSend1, int Recv, int Send, int SrcBuf, int DstBuf>
  inline __device__ void
  genericOp(intptr_t srcIx, intptr_t dstIx, int nelem,
      bool postOp, int reqChunkSize=0) {
    constexpr int DirectRecv = 1 && Direct && DirectRecv1;
    constexpr int DirectSend = 1 && Direct && DirectSend1;
    constexpr int Src = SrcBuf != -1;
    constexpr int Dst = DstBuf != -1;
    T* userIn = (T*)getShmemPtr(comm)->groups[group].userInput;
    T* userOut = (T*)getShmemPtr(comm)->groups[group].userOutput;
    const T* srcPtr = SrcBuf == -1 ? nullptr : (SrcBuf == Input ? userIn : userOut) + srcIx;
    T* dstPtr = DstBuf == -1 ? nullptr : (DstBuf == Input ? userIn : userOut) + dstIx;
    nelem = nelem < 0 ? 0 : nelem;
    int sliceSize = stepSize*StepPerSlice;
    sliceSize = max(divUp(nelem, 16*SlicePerChunk)*16, sliceSize/32);
    if (reqChunkSize > 0) {
      sliceSize = max(divUp(reqChunkSize, 16*SlicePerChunk)*16, stepSize*StepPerSlice/32);
    }
    int slice = 0;
    int offset = 0;

    if (tid < nworkers && offset < nelem) {
      // Worker-only loop over the non-empty slices: non-workers and
      // empty slices are handled by the loop below. Splitting pulls two branches
      // out of the critical path (perf_orig = 2*numslices, perf_new = 2+numslices).
      #pragma unroll 1
      do {
        int realSize = min(sliceSize, nelem-offset);
        if (Src && (role & ROLE_SRC)) srcs[0] = srcPtr+offset;
        if (Recv && (role & ROLE_WAIT_RECV)) static_cast<RealPrimitives*>(this)->template waitRecv<Src, DirectRecv>(dstIx+offset);
        if (Dst && (role & ROLE_DST)) dsts[0] = dstPtr+offset;
        if (Send && (role & ROLE_WAIT_SEND)) waitSend<Dst, DirectSend>(dstIx+offset, realSize*sizeof(T));
        subBarrier();
        if (DirectRecv && srcs[0] == dsts[0] && !Src) {
          if (Send && Dst && srcs[0] != dsts[1]) {
            static_cast<RealPrimitives*>(this)->template DirectReduceOrCopyMulti<DirectRecv, DirectSend, Recv, Send, SrcBuf, DstBuf>(postOp, realSize);
          }
        } else {
          static_cast<RealPrimitives*>(this)->template IndirectReduceOrCopyMulti<DirectRecv, DirectSend, Recv, Send, SrcBuf, DstBuf>(postOp, realSize);
        }
        barrier(); // This barrier has a counterpart in the following loop
        static_cast<RealPrimitives*>(this)->template memoryFence<Send>(realSize);
        __syncwarp();
        if (Send && (role & ROLE_POST_SEND)) postSend();
        if (Recv && (role & ROLE_POST_RECV)) postRecv();
        offset += realSize;
        slice += 1;
      } while (slice < SlicePerChunk && offset < nelem);
    }

    // Non-workers arrive here directly, as do workers once every remaining slice
    // is empty.
    #pragma unroll 1
    while (slice < SlicePerChunk) {
      int realSize = max(0, min(sliceSize, nelem-offset));
      { // Only workers hold a wait role and their slices are empty here, so just
        // claim the step.
        if (Recv && (role & ROLE_WAIT_RECV)) static_cast<RealPrimitives*>(this)->template waitRecv<Src, DirectRecv>(dstIx+offset);
        if (Send && (role & ROLE_WAIT_SEND)) waitSend<Dst, DirectSend>(dstIx+offset, realSize*sizeof(T));
      }
      barrier(); // Has a counterpart in the preceding worker-only loop
      static_cast<RealPrimitives*>(this)->template memoryFence<Send>(realSize);
      __syncwarp();
      if (Send && (role & ROLE_POST_SEND)) postSend();
      if (Recv && (role & ROLE_POST_RECV)) postRecv();
      offset += realSize;
      slice += 1;
    }
  }

  template <int DirectRecv, int DirectSend, int Recv, int Send>
  __device__ __forceinline__ void ScatterImpl(int* realSizes) {
    if (Proto::Id == NCCL_PROTO_SIMPLE || Proto::Id == NCCL_PROTO_SIGNAL) {
      ScatterGatherMulti<Unroll, T, MaxSend, RedOp, 1>(tid, nworkers, redOp, srcs, dsts, realSizes, fan.nsend());
    } else {
      ScatterGatherMultiBulk<Unroll, T, MaxSend, Send, RedOp, 1, Proto::RTMD, Proto::sendConnType>(tid, nworkers, redOp, srcs, dsts, realSizes, fan.nsend());
    }
  }

  template <int DirectRecv, int DirectSend, int Recv, int Send>
  __device__ __forceinline__ void GatherImpl(int* realSizes) {
    if (Proto::Id == NCCL_PROTO_SIMPLE || Proto::Id == NCCL_PROTO_SIGNAL) {
      ScatterGatherMulti<Unroll, T, MaxRecv, RedOp, 0>(tid, nworkers, redOp, srcs, dsts, realSizes, fan.nrecv());
    } else {
      ScatterGatherMultiBulk<Unroll, T, MaxRecv, Send, RedOp, 0, Proto::RTMD, Proto::sendConnType>(tid, nworkers, redOp, srcs, dsts, realSizes, fan.nrecv());
    }
  }

  template <int DirectRecv1, int DirectSend1, int Recv, int Send>
  __device__ __forceinline__ void
  ScatterGatherOp(intptr_t inpIx, intptr_t outIx, int totalElem, int peerElem, ssize_t rankOffset) {
    constexpr int DirectRecv = 1 && Direct && DirectRecv1;
    constexpr int DirectSend = 1 && Direct && DirectSend1;
    const T* srcPtr = Send ? (const T*)getShmemPtr(comm)->groups[group].userInput + inpIx : nullptr;
    T* dstPtr = Recv ? (T*)getShmemPtr(comm)->groups[group].userOutput + outIx : nullptr;
    int offset = 0; // slice offset
    int sliceSize = stepSize*StepPerSlice;
    int dataSize = max(DIVUP(peerElem, 16*SlicePerChunk)*16, sliceSize/32);  // per-peer slice size

    #pragma unroll
    for (int slice=0; slice<SlicePerChunk; ++slice) {
      int realSize = max(0, min(dataSize, peerElem-offset));
      if (tid < nworkers) {
        if (Send) {
          if (tid == 0) totalSendSize[slice] = 0;
          if (role & ROLE_WAIT_SEND) {
            srcs[index] = srcPtr + peer * rankOffset + offset;
            int realPeerSize = max(0, min(realSize, totalElem-peer*peerElem-offset));
            realSizes[index] = realPeerSize;
            waitSend<0, DirectSend>(outIx+offset, realPeerSize*sizeof(T));
          }
          subBarrier();
          if (realSizes[0] > 0) {
            ScatterImpl<DirectRecv, DirectSend, Recv, Send>(realSizes);
            if (tid == 0) totalSendSize[slice] += realSizes[0];
          }
        } else if (Recv) {
          if (role & ROLE_WAIT_RECV) {
            dsts[index] = dstPtr + peer * rankOffset + offset;
            int realPeerSize = max(0, min(realSize, totalElem-peer*peerElem-offset));
            realSizes[index] = realPeerSize;
            static_cast<RealPrimitives*>(this)->template waitRecv<0, DirectRecv>(DirectRecv*(outIx+peer*rankOffset)+offset);
          }
          subBarrier();
          if (DirectRecv) {
            int doRecv = 0;
            for (int i=0; i<fan.nrecv(); i++) if (srcs[i] != dsts[i]) { doRecv++; break;}
            if (doRecv > 0) {
              GatherImpl<DirectRecv, DirectSend, Recv, Send>(realSizes);
            }
          } else {
            GatherImpl<DirectRecv, DirectSend, Recv, Send>(realSizes);
          }
        }
      }
      barrier();
      static_cast<RealPrimitives*>(this)->template memoryFence<Send>(totalSendSize[slice]);
      __syncwarp();
      if (Send && (role & ROLE_POST_SEND)) postSend();
      if (Recv && (role & ROLE_POST_RECV)) postRecv();
      offset += realSize;
    }
  }

  __device__ __forceinline__ void loadRecvConn(ncclDevChannelPeer *devChannelPeer, T* directBuff, int connIndex, struct ncclDevWorkColl* collWork, struct ncclDevWorkP2p* p2pWork) {
    // Caller already gates this with (role & (ROLE_WAIT_RECV|ROLE_POST_RECV)),
    // so devChannelPeer is guaranteed non-null here.
    conn = &devChannelPeer->recv[connIndex];
    step = conn->step;
    step = ROUNDUP(step, SlicePerChunk*StepPerSlice);
    if (role & ROLE_POST_RECV) {
      connHeadPtr = conn->head;
      // Return credits in case we rounded up.
      volatileStoreU64(connHeadPtr, step);
    }
    if (role & ROLE_WAIT_RECV) {
      buff = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
      if (Direct && ((P2p ? ((p2pWork != nullptr && p2pWork->recvRegUsed)) : (collWork != nullptr && collWork->recvRegUsed)) && (conn->direct & NCCL_DIRECT_GPU))) {
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
        if (!P2p) {
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
    step = ROUNDUP(step, SlicePerChunk*StepPerSlice);
    if (role & ROLE_POST_SEND) {
      connTailPtr = conn->tail;
    }
    if (role & ROLE_WAIT_SEND) {
      buff = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
      if (Direct && ((P2p ? ((p2pWork != nullptr && p2pWork->sendRegUsed)) : (collWork != nullptr && collWork->sendRegUsed)) && (conn->direct & NCCL_DIRECT_GPU))) {
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
  static constexpr int workersFromThreads(int nthreads) {
#ifdef NO_EXTRA_WARP
    return nthreads;
#else
    return nthreads - (MaxSend > 0 && nthreads >= NCCL_SIMPLE_EXTRA_GROUP_IF_NTHREADS_GE ? WARP_SIZE : 0);
#endif
  }

  __device__ __forceinline__
  PrimitivesSimpleBase(
      const int tid, const int nthreads, int* recvPeers, int* sendPeers, T* directBuff,
      struct ncclKernelComm* comm, uint64_t redOpArg, struct ncclShmemGroup* ptrs, int group, uint8_t connIndexRecv = 0, uint8_t connIndexSend = 0,
      struct ncclDevWorkColl* collWork = nullptr, struct ncclDevWorkP2p* p2pWork = nullptr, int stepSize_ = 0, const uint64_t ncclSteps=NCCL_STEPS, bool noRed=false,
      int mode = primsModeDefault
      ):
    redOp(noRed ? RedOpIdentityArg<RedOp>::value() : redOpArg),
    comm(comm), tid(tid), nthreads(nthreads),
    stepSize(stepSize_ == 0 ? comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/sizeof(T) : stepSize_),
    buffStepSize(comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/sizeof(T)),
    srcs((const T**)ptrs[group].srcs), dsts((T**)ptrs[group].dsts), realSizes(ptrs[group].realSizes), group(group),
    totalSendSize(ptrs[group].totalSendSize) {
    ptrs[group].redOpArgs = noRed ? RedOpIdentityArg<RedOp>::value() : redOpArg;

    if (mode == primsModeDefault) {
      nworkers = workersFromThreads(nthreads);

      // Make sure step is updated before we read it.
      barrier();

      int nrecv = 0, nsend = 0;
      for (int i=0; i<MaxRecv; i++) if (recvPeers[i] != -1) nrecv++;
      for (int i=0; i<MaxSend; i++) if (sendPeers[i] != -1) nsend++;
      this->fan = Fan(nrecv, nsend);

      constexpr int ThreadPerSync =
        MaxSend >= 16 || MaxRecv >= 16 ? 32 :
        MaxSend >= 8  || MaxRecv >= 8  ? 16 :
                                         8; // All four roles fit in a single warp
      static_assert(MaxSend < ThreadPerSync && MaxRecv < ThreadPerSync, "Not enough threads to cover all peers");

      int g = tid / ThreadPerSync;
      int ng = nthreads / ThreadPerSync;
      index = tid % ThreadPerSync;

      if (g == 0) {
        if (index < fan.nrecv()) role |= ROLE_WAIT_RECV;
        if (index == fan.nrecv()) role |= ROLE_SRC;
      } else if (g == 1) {
        if (index < fan.nsend()) role |= ROLE_WAIT_SEND;
        if (index == fan.nsend()) role |= ROLE_DST;
      } else if (g == ng - 2) {
        if (index < fan.nrecv()) role |= ROLE_POST_RECV;
      } else if (g == ng - 1) {
        if (index < fan.nsend()) role |= ROLE_POST_SEND;
      }

      if (role & (ROLE_WAIT_RECV|ROLE_POST_RECV)) peer = recvPeers[index];
      if (role & (ROLE_WAIT_SEND|ROLE_POST_SEND)) peer = sendPeers[index];

      // Gated here rather than inside loadRecvConn/loadSendConn: the peer index
      // is only meaningful for the threads that hold the matching role, so an
      // unconditioned call would index channel.peers[] with a stale peer.
      if (role & (ROLE_WAIT_RECV|ROLE_POST_RECV))
        loadRecvConn(getShmemPtr(comm)->channel.peers[peer], directBuff, connIndexRecv, collWork, p2pWork);
      if (role & (ROLE_WAIT_SEND|ROLE_POST_SEND))
        loadSendConn(getShmemPtr(comm)->channel.peers[peer], connIndexSend, collWork, p2pWork);
    } else {
      // PAT drives every thread as a worker, so nothing is carved out for the
      // post warp (patBarrier() names NCCL_PAT_NWORKERS explicitly) and the
      // peers come from the PAT steps rather than the channel table (they are
      // loaded by the primitives' own ctor).
      nworkers = nthreads;
      this->fan = Fan(MaxRecv, MaxSend);
    }
  }

  __device__ __forceinline__ void setDataPtrs(void const* inputBuf, void* outputBuf) {
    if (tid == 0) {
      getShmemPtr(comm)->groups[group].userInput = (void*)inputBuf;
      getShmemPtr(comm)->groups[group].userOutput = outputBuf;
    }
    barrier();
  }

  // Same publication rule as setDataPtrs.
  __device__ __forceinline__ void moveDataPtrs(intptr_t delta) {
    if (tid == 0) {
      getShmemPtr(comm)->groups[group].userInput = (T*)getShmemPtr(comm)->groups[group].userInput + delta;
      getShmemPtr(comm)->groups[group].userOutput = (T*)getShmemPtr(comm)->groups[group].userOutput + delta;
    }
    barrier();
  }

  __device__ __forceinline__ void
  send(intptr_t inpIx, int eltN) {
    genericOp<0, 0, 0, 1, Input, -1>(inpIx, -1, eltN, false);
  }

  __device__ __forceinline__ void
  sendFromOutput(intptr_t outIx, int eltN) {
    genericOp<0, 0, 0, 1, Output, -1>(outIx, -1, eltN, false);
  }
  __device__ __forceinline__ void
  directSend(intptr_t inpIx, intptr_t outIx, int eltN) {
    genericOp<0, 1, 0, 1, Input, -1>(inpIx, outIx, eltN, false);
  }
  __device__ __forceinline__ void
  directSendFromOutput(intptr_t outIx, int eltN) {
    genericOp<0, 1, 0, 1, Output, -1>(outIx, outIx, eltN, false);
  }

  __device__ __forceinline__ void
  recv(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 0, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void
  directRecv(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 0, -1, Output>(outIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void
  copySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 0, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void
  directCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 1, 0, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void
  recvCopySend(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void
  directRecvCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 1, 1, 1, -1, Output>(inpIx, outIx, eltN, postOp);
  }
  // Direct recv, buffered send: the recv leg reads through directPtr, the send
  // leg stages through the conn buffer.
  __device__ __forceinline__ void
  directRecvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void
  directRecvCopySend(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 1, -1, Output>(outIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void
  recvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 0, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void
  directRecvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 0, Input, Output>(inpIx, outIx, eltN, postOp);
  }

#ifdef KERNEL_LOOPBACK
  __device__ __forceinline__ void
  sendRecvLoopback(intptr_t inpIx, intptr_t outIx, int nelem) {
    genericOp<0, 0, 0, 0, Input, Output>(inpIx, outIx, nelem, false);
  }
#endif

  __device__ __forceinline__ void
  recvReduceSend(intptr_t inpIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, Input, -1>(inpIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void
  directRecvReduceDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN,
      bool postOp=false, int reqChunkSize=0) {
    genericOp<1, 1, 1, 1, Input, -1>(inpIx, outIx, eltN, postOp, reqChunkSize);
  }

  __device__ __forceinline__ void
  recvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void
  recvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN,
      bool postOp=false, int reqChunkSize=0) {
    // Direct is only for the send part
    genericOp<0, 1, 1, 1, Input, Output>(inpIx, outIx, eltN, postOp, reqChunkSize);
  }
  __device__ __forceinline__ void
  directRecvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN,
      bool postOp=false, int reqChunkSize=0) {
    genericOp<1, 1, 1, 1, Input, Output>(inpIx, outIx, eltN, postOp, reqChunkSize);
  }

  __device__ __forceinline__ void
  scatter(intptr_t inpIx, int totalElem, int peerElem, ssize_t rankOffset) {
    ScatterGatherOp<0, 0, 0, 1>(inpIx, -1, totalElem, peerElem, rankOffset);
  }

  __device__ __forceinline__ void
  gather(intptr_t outIx, int totalElem, int peerElem, ssize_t rankOffset) {
    ScatterGatherOp<0, 0, 1, 0>(-1, outIx, totalElem, peerElem, rankOffset);
  }

  __device__ __forceinline__ void
  directGather(intptr_t outIx, int totalElem, int peerElem, ssize_t rankOffset) {
    ScatterGatherOp<1, 0, 1, 0>(-1, outIx, totalElem, peerElem, rankOffset);
  }

  __device__ __forceinline__ ~PrimitivesSimpleBase() {
    // Save steps for the next operation
    if (role & (ROLE_POST_SEND|ROLE_POST_RECV))
      volatileStoreU64(reinterpret_cast<volatile uint64_t*>(&conn->step), step);
    barrier();
  }
};

#endif
