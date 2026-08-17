/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

template <typename T, class FUNC, int NRECV, int NSEND, int DIRECT, int P2P>
class Primitives<T, FUNC, ProtoLL, NRECV, NSEND, DIRECT, P2P>:
    public PrimitivesWithoutDirect<T, Primitives<T, FUNC, ProtoLL, NRECV, NSEND, DIRECT, P2P>> {

 private:
  const int tid;
  const int nthreads;
  const int wid;
  const int group;
  const int stepLines;
  int nrecv = 0;
  int nsend = 0;
  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead = 0;

  struct ncclConnInfo* sendConn = NULL;
  volatile int* sendConnFifoPtr = NULL;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache; // Cache last seen value

  int recvPeer[NRECV];
  int sendPeer[NSEND];
  uint64_t recvStep[NRECV];
  uint64_t sendStep[NSEND];
  union ncclLLFifoLine* recvBuff[NRECV];
  union ncclLLFifoLine* sendBuff[NSEND];
  struct ncclKernelComm* comm;
  FUNC redOp;

  inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*stepLines; }
  inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*stepLines; }
  inline __device__ union ncclLLFifoLine* recvPtr(int i) { return recvBuff[i]+recvOffset(i); }
  inline __device__ union ncclLLFifoLine* sendPtr(int i) { return sendBuff[i]+sendOffset(i); }
  inline __device__ uint32_t recvFlag(int i) { return NCCL_LL_FLAG(recvStep[i]+1); }
  inline __device__ uint32_t sendFlag(int i) { return NCCL_LL_FLAG(sendStep[i]+1); }

  inline __device__ void barrier() {
    if (nthreads == WARP_SIZE) __syncwarp();
    else __ppu_barrier_sync(group+1, nthreads);
  }

  uint32_t spins = 0;
  uint32_t hangSpins = 0;
  uint32_t abort = 0;

  inline __device__ void waitSend(int nbytes) {
    hangSpins = spins = 0;
    if (sendConnHeadPtr) {
      while (sendConnHeadCache + NCCL_STEPS < sendConnHead + 1) {
        sendConnHeadCache = volatileLoadU64(sendConnHeadPtr);
        if (checkAbort(comm, spins, abort, hangSpins)) break;
      }
      if (sendConnFifoPtr) {
        int size = ((sendConnHead & NCCL_LL_CLEAN_MASK) == NCCL_LL_CLEAN_MASK) ? stepLines*sizeof(union ncclLLFifoLine) : nbytes;
        sendConnFifoPtr[sendConnHead%NCCL_STEPS] = size;
      }
      sendConnHead += 1;
    }
    barrier();
  }

  inline __device__ void incRecv(int i) {
    recvStep[i] += 1;
  }
  inline __device__ void postRecv() {
    barrier();
    if (recvConnHeadPtr) {
      recvConnHead += 1;
      volatileStoreU64(recvConnHeadPtr, recvConnHead);
    }
  }

  inline __device__ void incSend(int i, int offset) {
    // LL Cleanup : write all flags in the slice to make sure we don't have
    // data corruption when flag loops over.
    if ((sendStep[i] & NCCL_LL_CLEAN_MASK) == NCCL_LL_CLEAN_MASK) {
      for (int o = offset; o<stepLines; o+=nthreads) storeLL(sendPtr(i)+o, 0, sendFlag(i));
    }
    sendStep[i]++;
  }

  __device__ uint64_t readLL(int i, int offset) {
#ifndef LOAD_BYPASS
    volatile union ncclLLFifoLine* src = recvPtr(i) + offset;
    uint32_t flag = recvFlag(i);
    hangSpins = spins = 0;
    union ncclLLFifoLine read;
    do {
      read.i4 = __ppu_global_load_volatile_b32x4(&src->i4);
      // read.i4 = __ppu_global_load_bulk_volatile_b32x4(&src->i4);
      if (checkAbort(comm, spins, abort, hangSpins)) break;
    } while ((read.flag1 != flag) || (read.flag2 != flag));
    uint64_t val64 = read.data1 + (((uint64_t)read.data2) << 32);
    return val64;
#else
    return 0UL;
#endif
  }

  __device__ void storeLL(union ncclLLFifoLine* dst, uint64_t val, uint32_t flag) {
#ifndef STORE_BYPASS
    union ncclLLFifoLine data;
    data.data1 = (uint32_t)val;
    data.flag1 = flag;
    data.data2 = (uint32_t)(val >> 32);
    data.flag2 = flag;
    __ppu_global_store_volatile_b32x4((volatile void*)(&dst->i4), data.i4);
    // __ppu_global_store_bulk_volatile_b32x4((volatile void*)(&dst->i4), data.i4);
#endif
  }

  static constexpr int EltPerLine = sizeof(uint64_t)/sizeof(T);

  template<typename U>
  __device__ static U load(U *src) {
    union {
      U elt;
      uint16_t u2;
      uint32_t u4;
      uint64_t u8;
    };
    if(sizeof(U) == 1)
      asm("ppu.ld.volatile.global.b8 %0,[%1];" : "=r"(u4) : "l"(src));
    else if(sizeof(U) == 2)
      asm("ppu.ld.volatile.global.b16 %0,[%1];" : "=h"(u2) : "l"(src));
    else if(sizeof(U) == 4)
      asm("ppu.ld.volatile.global.b32 %0,[%1];" : "=r"(u4) : "l"(src));
    else
      asm("ppu.ld.volatile.global.b64 %0,[%1];" : "=l"(u8) : "l"(src));
    return elt;
  }

  struct DataLoader {
    int misalign;
    union {
      uint32_t u4[sizeof(T) <= 2 ? 3 : 2];
      uint64_t u8;
      T elt[EltPerLine];
    };

    __device__ void loadBegin(T *src, int eltN) {
      if (sizeof(T) <= 2) {
        misalign = reinterpret_cast<uintptr_t>(src)%4;
        uint32_t *p = reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(src) & -uintptr_t(4));
        u4[0] = load(p+0);
        u4[1] = misalign + eltN*sizeof(T) > 4 ? load(p+1) : 0;
        // u4[2] would be simpler, but that throws warnings on some compilers
        u4[sizeof(T) <= 2 ? 2 : 0] = misalign + eltN*sizeof(T) > 8 ? load(p+2) : 0;
      } else {
        #pragma unroll
        for(int i=0; i < EltPerLine; i++) {
          if(i==0 || i < eltN)
            elt[i] = load(src + i);
        }
      }
    }

    __device__ uint64_t loadFinish() {
      if (sizeof(T) <= 2) {
        u4[0] = __funnelshift_r(u4[0], u4[1], 8*misalign);
        // u4[2] would be simpler, but that throws warnings on some compilers
        u4[1] = __funnelshift_r(u4[1], u4[sizeof(T) <= 2 ? 2 : 0], 8*misalign);
      }
      return u8;
    }
  };

  __device__ uint64_t readAL(T* src, int eltN) {
    uint64_t val;
#ifndef LOAD_BYPASS
    DataLoader dl;
    dl.loadBegin(src, eltN);
    return dl.loadFinish();
#else
    val = 0UL;
#endif
    return val;
  }

  __device__ void storeAL(T* dst, uint64_t val, uint32_t eltN) {
#ifndef STORE_BYPASS
    union {
      uint64_t u8;
      T elt[EltPerLine];
    };
    u8 = val;
    #pragma unroll
    for(int i=0; i < EltPerLine; i++) {
      if (i==0 || i < eltN)
        //store(dst+i, elt[i]);
        dst[i] = elt[i];
    }
#endif
  }

  template <int RECV, int SEND, int SRC, int DST>
  __device__ void LLGenericOp(const T* srcPtr, T* dstPtr, int nelem, bool postOp) {
    nelem = nelem < 0 ? 0 : nelem;
    uint32_t nbytes = nelem*sizeof(T);
    uint32_t npack = DIVUP(nbytes, sizeof(uint64_t));
    T *srcElts = (T*)srcPtr;
    T *dstElts = dstPtr;

    // Always waitSend in case of cleanup
    if (SEND) waitSend(npack*sizeof(union ncclLLFifoLine));

    nelem -= tid*EltPerLine;
    srcElts += tid*EltPerLine;
    dstElts += tid*EltPerLine;
    int offset = tid;
    int eltPerTrip = nthreads*EltPerLine;

    // Do multiples of 64 bits
    // TODO: need to find other way to do variable unrolling while PPU path
    while(nelem>0) {
    // for (; offset<npack; offset+=nthreads) {
      // Recv : local, then intra-node, then inter-node
      int eltInLine = EltPerLine < nelem ? EltPerLine : nelem;
      int64_t val;
      val = SRC ? MULTI<FUNC, T>().preOp(redOp, readAL(srcElts, eltInLine)) : readLL(0, offset);
      srcElts += eltPerTrip;
      if (RECV) {
        if (SRC) val = MULTI<FUNC, T>()(redOp, readLL(0, offset), val);
        for (int i=1; i<NRECV && i<nrecv; i++) {
          val = MULTI<FUNC, T>()(redOp, readLL(i, offset), val);
        }
      }

      if(postOp) val = MULTI<FUNC, T>().postOp(redOp, val);

      // Send : inter-node, then intra-node, then local
      if (SEND) {
        for (int i=1; i<NSEND && i<nsend; i++) storeLL(sendPtr(i)+offset, val, sendFlag(i));
        storeLL(sendPtr(0)+offset, val, sendFlag(0));
      }
      if (DST) {
        storeAL(dstElts, val, eltInLine);
        dstElts += eltPerTrip;
      }

      nelem -= eltPerTrip;
      offset += nthreads;
    }
    FOR_RECV(incRecv); if (RECV) postRecv();
    FOR_SEND(incSend, offset);
  }

  template <int RECV, int SEND>
  __device__ void ScatterGatherOp(const T* srcPtr, T* dstPtr, int totalElem, int peerElem, ssize_t rankOffset) {
    totalElem = totalElem < 0 ? 0 : totalElem;
    peerElem = peerElem < 0 ? 0 : peerElem;
    int peerNbytes = peerElem < 0 ? 0 : peerElem * sizeof(T);
    int peerNpacks = DIVUP(peerNbytes, sizeof(uint64_t));
    int totalNpacks = 0;

    // Always waitSend in case of cleanup
    if (SEND) waitSend(peerNpacks*sizeof(union ncclLLFifoLine));

    // calculate real address offset and nbytes for each peer
    ssize_t peerOffsets[NSEND > NRECV ? NSEND : NRECV];
    int realPeerNelem[NSEND > NRECV ? NSEND : NRECV];
    if (SEND) {
      #pragma unroll
      for (int i=0; i<NSEND; i++) {
        peerOffsets[i] = sendPeer[i] * rankOffset;
        realPeerNelem[i] = max(0, min(peerElem, totalElem - sendPeer[i] * peerElem));
        totalNpacks += DIVUP(realPeerNelem[i] * sizeof(T), sizeof(uint64_t));
      }
    } else if (RECV) {
      #pragma unroll
      for (int i=0; i<NRECV; i++) {
        peerOffsets[i] = recvPeer[i] * rankOffset;
        realPeerNelem[i] = max(0, min(peerElem, totalElem - recvPeer[i] * peerElem));
        totalNpacks += DIVUP(realPeerNelem[i] * sizeof(T), sizeof(uint64_t));
      }
    }

    int offset = tid;
    for (; offset < totalNpacks; offset+=nthreads) {
      if (SEND) {
        int index = (offset / peerNpacks) % nsend;
        int pack = (offset % peerNpacks);
        int elem = pack * EltPerLine;
        int eltInLine = min(EltPerLine, (realPeerNelem[index] - elem));
        T* src = (T*)(srcPtr + peerOffsets[index] + elem);
        union ncclLLFifoLine* dst = sendPtr(index) + pack;
        uint64_t val = readAL(src, eltInLine);
        storeLL(dst, val, sendFlag(index));
      } else if (RECV) {
        int index = (offset / peerNpacks) % nrecv;
        int pack = offset % peerNpacks;
        int elem = pack * EltPerLine;
        int eltInLine = min(EltPerLine, (realPeerNelem[index] - elem));
        union ncclLLFifoLine* src = recvPtr(index) + pack;
        T* dst = dstPtr + peerOffsets[index] + elem;
        uint64_t val = readLL(index, pack);
        storeAL(dst, val, eltInLine);
      }
    }
    FOR_RECV(incRecv); if (RECV) postRecv();
    FOR_SEND(incSend, offset);
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i) {
    recvBuff[i] = (union ncclLLFifoLine*)conn->buffs[NCCL_PROTO_LL];
    recvStep[i] = conn->step;
    if (wid == i) recvConn = conn;
    nrecv++;
  }
  __device__ __forceinline__ void loadRecvSync() {
    if (tid >= nthreads-WARP_SIZE && wid < nrecv) {
      recvConnHeadPtr = recvConn->head;
      recvConnHead = recvConn->step;
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i) {
    sendBuff[i] = (union ncclLLFifoLine*)conn->buffs[NCCL_PROTO_LL];
    sendStep[i] = conn->step;
    if (wid == i) sendConn = conn;
    nsend++;
  }
  __device__ __forceinline__ void loadSendSync() {
    if (tid < nsend) {
      sendConnHeadPtr = sendConn->head;
      sendConnHeadCache = volatileLoadU64(sendConnHeadPtr);
      sendConnHead = sendConn->step;
      sendConnFifoPtr = sendConn->sizesFifo;
    }
  }

 public:
  // p2pWork: unused by LL itself, kept so sendrecv.h's runRecv/runSend can pass the
  // same arglist regardless of whether Proto is ProtoLL or a base (SIMPLE/...) proto.
  __device__ __forceinline__
  Primitives(
      const int tid, const int nthreads, int* recvPeers, int* sendPeers,
      T* directBuff, struct ncclKernelComm* comm, uint64_t redOpArg, struct ncclShmemGroup* ptrs = NULL,
      int group = 0, uint8_t connIndexRecv = 0, uint8_t connIndexSend = 0, struct ncclDevWorkColl* collWork = nullptr,
      struct ncclDevWorkP2p* p2pWork = nullptr, const int ncclSteps = NCCL_STEPS, bool noRed = false
    ):
    // WR: fix potential double preOp for up/down tree.
    redOp(noRed ? RedOpIdentityArg<FUNC>::value() : redOpArg),
    comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), group(group),
    stepLines(comm->buffSizes[NCCL_PROTO_LL]/NCCL_STEPS/sizeof(ncclLLFifoLine)) {
    // Make sure step is updated before we read it.
    barrier();

    for (int i=0; i<NRECV && recvPeers[i] >= 0; i++) {
      recvPeer[i] = recvPeers[i];
      loadRecvConn(&getShmemPtr(comm)->channel.peers[recvPeers[i]]->recv[connIndexRecv], i);
    }
    for (int i=0; i<NSEND && sendPeers[i] >= 0; i++) {
      sendPeer[i] = sendPeers[i];
      loadSendConn(&getShmemPtr(comm)->channel.peers[sendPeers[i]]->send[connIndexSend], i);
    }
    loadRecvSync();
    loadSendSync();
  }

  __device__ void send(const T* src, int nelem) {
    return LLGenericOp<0, 1, 1, 0>(src, NULL, nelem, false);
  }

  __device__ void recv(T* dst, int nelem, bool postOp=false) {
    return LLGenericOp<1, 0, 0, 1>(NULL, dst, nelem, postOp);
  }

  __device__ void recvReduceSend(const T* src, int nelem) {
    return LLGenericOp<1, 1, 1, 0>(src, NULL, nelem, false);
  }

  __device__ void recvReduceCopy(const T* src, T* dst, int nelem, bool postOp=false) {
    return LLGenericOp<1, 0, 1, 1>(src, dst, nelem, postOp);
  }

#ifdef KERNEL_LOOPBACK
  __device__ void sendRecvLoopback(const T* src, T* dst, int nelem) {
    return LLGenericOp<0, 0, 1, 1>(src, dst, nelem, false);
  }
#endif

  __device__ void copySend(const T* src, T* dst, int nelem, bool postOp=false) {
    return LLGenericOp<0, 1, 1, 1>(src, dst, nelem, postOp);
  }

  __device__ void recvCopySend(T* dst, int nelem, bool postOp=false) {
    return LLGenericOp<1, 1, 0, 1>(NULL, dst, nelem, postOp);
  }

  __device__ void recvReduceCopySend(const T* src, T* dst, int nelem, bool postOp=false) {
    return LLGenericOp<1, 1, 1, 1>(src, dst, nelem, postOp);
  }

  __device__ __forceinline__ void
  scatter(const T* src, int totalElem, int peerElem, ssize_t rankOffset) {
    ScatterGatherOp<0, 1>(src, NULL, totalElem, peerElem, rankOffset);
  }

  __device__ __forceinline__ void
  gather(T* dst, int totalElem, int peerElem, ssize_t rankOffset) {
    ScatterGatherOp<1, 0>(NULL, dst, totalElem, peerElem, rankOffset);
  }

  __device__ __forceinline__ ~Primitives() {
    // Save steps for the next operation
    if (tid >= nthreads-WARP_SIZE && wid < nrecv)
      volatileStoreU64(reinterpret_cast<volatile uint64_t*>(&recvConn->step), recvConnHead);
    if (tid < nsend)
      volatileStoreU64(reinterpret_cast<volatile uint64_t*>(&sendConn->step), sendConnHead);
    barrier();
  }
};
