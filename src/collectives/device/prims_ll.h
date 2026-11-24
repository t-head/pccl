/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

template<typename T, typename RedOp, typename Fan, int Direct, int P2p, bool isNetOffload>
class Primitives<T, RedOp, Fan, Direct, ProtoLL, P2p, isNetOffload>:
  public PrimitivesWithoutDirect<Primitives<T, RedOp, Fan, Direct, ProtoLL, P2p, isNetOffload>> {

 private:
  // In the case of Fan::MaxRecv == 0, we need to force MaxRecv to 1 for this to compile
  // This is because of a recv buffer which is allocated to MaxRecv length in send-only cases
  static constexpr int MaxRecv = Fan::MaxRecv > 1 ? Fan::MaxRecv : 1;
  static constexpr int MaxSend = Fan::MaxSend;
  static constexpr int Input = 0, Output = 1;

  static_assert(!isNetOffload, "isNetOffload is not implemented on PPU");

  RedOp redOp;
  const int tid;
  const int nthreads;
  const int wid;
  const int group;
  const int stepLines;
  Fan fan;
  T* userBufs[2];
  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead;

  struct ncclConnInfo* sendConn = NULL;
  volatile int* sendConnFifoPtr = NULL;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache; // Cache last seen value

  uint64_t recvStep[MaxRecv];
  uint64_t sendStep[MaxSend];
  union ncclLLFifoLine* recvBuff[MaxRecv];
  union ncclLLFifoLine* sendBuff[MaxSend];

  struct ncclKernelComm* comm;
  int recvPeer[MaxRecv];
  int sendPeer[MaxSend];

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
    if (sendConnHeadPtr) {
      hangSpins = spins = 0;
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

  __device__ uint64_t readLL(int offset, int i) {
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

  template<int BeginIx>
  __device__ void readLLBeginAll(int offset, ncclLLFifoLine(&line)[MaxRecv]) {
#ifndef LOAD_BYPASS
    #pragma unroll
    for (int i=BeginIx; i < MaxRecv; i++) {
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_line]
      if (i < fan.nrecv()) {
        volatile union ncclLLFifoLine* src = recvPtr(i) + offset;
        line[i].i4 = __ppu_global_load_volatile_b32x4(&src->i4);
      }
    }
#endif
  }

  __device__ uint64_t readLLFinish(int offset, ncclLLFifoLine(&line)[MaxRecv], int i) {
#ifndef LOAD_BYPASS
    volatile union ncclLLFifoLine* src = recvPtr(i) + offset;
    uint32_t flag = recvFlag(i);
    hangSpins = spins = 0;
    while (line[i].flag1 != flag || line[i].flag2 != flag) {
      line[i].i4 = __ppu_global_load_volatile_b32x4(&src->i4);
      if (checkAbort(comm, spins, abort, hangSpins)) break;
    }
    uint64_t val64 = line[i].data1 + (((uint64_t)line[i].data2) << 32);
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

  template<typename U>
  __device__ static void store(U *dst, U val) {
    union {
      U elt;
      uint16_t u2;
      uint32_t u4;
      uint64_t u8;
    };
    elt = val;
    if(sizeof(U) == 1)
      asm volatile("ppu.st.volatile.global.b8 [%0],%1;" :: "l"(dst), "r"(u4) : "memory");
    else if(sizeof(U) == 2)
      asm volatile("ppu.st.volatile.global.b16 [%0],%1;" :: "l"(dst), "h"(u2) : "memory");
    else if(sizeof(U) == 4)
      asm volatile("ppu.st.volatile.global.b32 [%0],%1;" :: "l"(dst), "r"(u4) : "memory");
    else
      asm volatile("ppu.st.volatile.global.b64 [%0],%1;" :: "l"(dst), "l"(u8) : "memory");
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
      }
      else {
        #pragma unroll
        for(int i=0; i < EltPerLine; i++) {
          // Yes, for some template arguments this code will be unreachable.  That's fine.
          // coverity[dead_error_line]
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

  __device__ void storeData(T* dst, uint64_t val, int eltN) {
#ifndef STORE_BYPASS
    union {
      uint64_t u8;
      T elt[EltPerLine];
    };
    u8 = val;
    #pragma unroll
    for(int i=0; i < EltPerLine; i++) {
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_line]
      if (i==0 || i < eltN)
        //store(dst+i, elt[i]);
        dst[i] = elt[i];
    }
#endif
  }

  template <int RECV, int SEND, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void LLGenericOp(intptr_t srcIx, intptr_t dstIx, int nelem, bool postOp) {
    constexpr int SRC = SrcBuf != -1 ? 1 : 0;
    constexpr int DST = DstBuf != -1 ? 1 : 0;
    T *srcElts = SrcBuf == -1 ? nullptr : userBufs[SrcBuf] + srcIx;
    T *dstElts = DstBuf == -1 ? nullptr : userBufs[DstBuf] + dstIx;

    // Always waitSend in case of cleanup
    nelem = nelem < 0 ? 0 : nelem;
    if (SEND) waitSend(DIVUP(nelem, EltPerLine)*sizeof(union ncclLLFifoLine));

    nelem -= tid*EltPerLine;
    srcElts += tid*EltPerLine;
    dstElts += tid*EltPerLine;
    int offset = tid;
    int eltPerTrip = nthreads*EltPerLine;

    while (nelem > 0) {
      int eltInLine = EltPerLine < nelem ? EltPerLine : nelem;

      DataLoader dl;
      ncclLLFifoLine line[MaxRecv];
      uint64_t data, peerData;
      if (SRC) {
        dl.loadBegin(srcElts, eltInLine);
        srcElts += eltPerTrip;
      }
      if (RECV) {
        readLLBeginAll<1>(offset, line);
        peerData = readLL(offset, 0);
      }
      if (SRC) {
        data = dl.loadFinish();
        if (SrcBuf == Input) data = applyPreOp(redOp, data);
      }
      if (RECV) {
        data = !SRC ? peerData : applyReduce(redOp, peerData, data);
        #pragma unroll MaxRecv
        // Yes, for some template arguments this code will be unreachable.  That's fine.
        // coverity[dead_error_line]
        for (int i=1; i < MaxRecv && i < fan.nrecv(); i++) {
          peerData = readLLFinish(offset, line, i);
          data = applyReduce(redOp, peerData, data);
        }
      }

      if (postOp) data = applyPostOp(redOp, data);

      // Send : inter-node, then intra-node, then local
      if (SEND) {
        // Yes, for some template arguments this code will be unreachable.  That's fine.
        // coverity[dead_error_line]
        for (int i=1; i < MaxSend && i < fan.nsend(); i++)
          storeLL(sendPtr(i)+offset, data, sendFlag(i));
        storeLL(sendPtr(0)+offset, data, sendFlag(0));
      }
      if (DST) {
        storeData(dstElts, data, eltInLine);
        dstElts += eltPerTrip;
      }

      nelem -= eltPerTrip;
      offset += nthreads;
    }

    if (RECV) {
      for (int i=0; i<MaxRecv; i++) incRecv(i);
      postRecv();
    }
    if (SEND) {
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_line]
      for (int i=1; i < MaxSend && i < fan.nsend(); i++)
        incSend(i, offset);
      incSend(0, offset);
    }
  }

  template <int RECV, int SEND>
  __device__ void ScatterGatherOp(intptr_t inpIx, intptr_t outIx, int totalElem, int peerElem, ssize_t rankOffset) {
    const T* srcPtr = SEND ? userBufs[Input] + inpIx : nullptr;
    T* dstPtr = RECV ? userBufs[Output] + outIx : nullptr;
    totalElem = totalElem < 0 ? 0 : totalElem;
    peerElem = peerElem < 0 ? 0 : peerElem;
    int peerNbytes = peerElem < 0 ? 0 : peerElem * sizeof(T);
    int peerNpacks = DIVUP(peerNbytes, sizeof(uint64_t));
    int totalNpacks = 0;

    // Always waitSend in case of cleanup
    if (SEND) waitSend(peerNpacks*sizeof(union ncclLLFifoLine));

    // calculate real address offset and nbytes for each peer
    ssize_t peerOffsets[MaxSend > MaxRecv ? MaxSend : MaxRecv];
    int realPeerNelem[MaxSend > MaxRecv ? MaxSend : MaxRecv];
    if (SEND) {
      #pragma unroll
      for (int i=0; i<MaxSend; i++) {
        peerOffsets[i] = sendPeer[i] * rankOffset;
        realPeerNelem[i] = max(0, min(peerElem, totalElem - sendPeer[i] * peerElem));
        totalNpacks += DIVUP(realPeerNelem[i] * sizeof(T), sizeof(uint64_t));
      }
    } else if (RECV) {
      #pragma unroll
      for (int i=0; i<MaxRecv; i++) {
        peerOffsets[i] = recvPeer[i] * rankOffset;
        realPeerNelem[i] = max(0, min(peerElem, totalElem - recvPeer[i] * peerElem));
        totalNpacks += DIVUP(realPeerNelem[i] * sizeof(T), sizeof(uint64_t));
      }
    }

    int offset = tid;
    for (; offset < totalNpacks; offset+=nthreads) {
      if (SEND) {
        int index = (offset / peerNpacks) % fan.nsend();
        int pack = (offset % peerNpacks);
        int elem = pack * EltPerLine;
        int eltInLine = min(EltPerLine, (realPeerNelem[index] - elem));
        T* src = (T*)(srcPtr + peerOffsets[index] + elem);
        union ncclLLFifoLine* dst = sendPtr(index) + pack;
        uint64_t val = MULTI<RedOp, T>().preOp(redOp, readAL(src, eltInLine));
        storeLL(dst, val, sendFlag(index));
      } else if (RECV) {
        int index = (offset / peerNpacks) % fan.nrecv();
        int pack = offset % peerNpacks;
        int elem = pack * EltPerLine;
        int eltInLine = min(EltPerLine, (realPeerNelem[index] - elem));
        union ncclLLFifoLine* src = recvPtr(index) + pack;
        T* dst = dstPtr + peerOffsets[index] + elem;
        uint64_t val = readLL(pack, index);
        storeData(dst, val, eltInLine);
      }
    }
    if (RECV) {
      for (int i=0; i<MaxRecv; i++) incRecv(i);
      postRecv();
    }
    if (SEND) {
      for (int i=1; i<MaxSend && i<fan.nsend(); i++) incSend(i, offset);
      incSend(0, offset);
    }
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i) {
    recvBuff[i] = (union ncclLLFifoLine*)conn->buffs[NCCL_PROTO_LL];
    recvStep[i] = conn->step;
    if (wid == i) recvConn = conn;
  }
  __device__ __forceinline__ void loadRecvSync() {
    if (tid >= nthreads-WARP_SIZE && wid < fan.nrecv()) {
      recvConnHeadPtr = recvConn->head;
      recvConnHead = recvConn->step;
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i) {
    sendBuff[i] = (union ncclLLFifoLine*)conn->buffs[NCCL_PROTO_LL];
    sendStep[i] = conn->step;
    if (wid == i) sendConn = conn;
  }
  __device__ __forceinline__ void loadSendSync() {
    if (tid < fan.nsend()) {
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
      struct ncclDevWorkP2p* p2pWork = nullptr, int stepSize_ = 0, const int ncclSteps = NCCL_STEPS, bool noRed = false
    ):
    // WR: fix potential double preOp for up/down tree.
    redOp(noRed ? RedOpIdentityArg<RedOp>::value() : redOpArg),
    tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), group(group),
    stepLines(comm->buffSizes[NCCL_PROTO_LL]/NCCL_STEPS/sizeof(ncclLLFifoLine)), comm(comm) {
    // Make sure step is updated before we read it.
    barrier();

    // If we are going to support oneshot collNet + LL, then we would need to add connector index here
    int nrecv=0, nsend=0;
    // We compare with Fan::MaxRecv here because this->MaxRecv is always at least 1
    // Yes, for some template arguments this code will be unreachable.  That's fine.
    // coverity[dead_error_line]
    while (nrecv < Fan::MaxRecv && recvPeers[nrecv] >= 0) {
      recvPeer[nrecv] = recvPeers[nrecv];
      loadRecvConn(&getShmemPtr(comm)->channel.peers[recvPeers[nrecv]]->recv[connIndexRecv], nrecv);
      nrecv++;
    }
    // coverity[dead_error_line]
    while (nsend < MaxSend && sendPeers[nsend] >= 0) {
      sendPeer[nsend] = sendPeers[nsend];
      loadSendConn(&getShmemPtr(comm)->channel.peers[sendPeers[nsend]]->send[connIndexSend], nsend);
      nsend++;
    }
    this->fan = Fan(nrecv, nsend);
    // Coverity reports recvConn and sendConn being possibly NULL at this point but that won't actually
    // happen given the two "while" loops just above.
    // coverity[var_deref_model:FALSE]
    loadRecvSync();
    // coverity[var_deref_model:FALSE]
    loadSendSync();
  }

  __device__ void setDataPtrs(void const* inputBuf, void* outputBuf) {
    userBufs[Input] = (T*)inputBuf;
    userBufs[Output] = (T*)outputBuf;
  }

  __device__ void moveDataPtrs(intptr_t delta) {
    userBufs[Input] += delta;
    userBufs[Output] += delta;
  }

  __device__ void send(intptr_t inpIx, int eltN) {
    return LLGenericOp<0, 1, Input, -1>(inpIx, -1, eltN, false);
  }
 
  __device__ void sendFromOutput(intptr_t outIx, int eltN) {
    return LLGenericOp<0, 1, Output, -1>(outIx, -1, eltN, false);
  }
  __device__ void recv(intptr_t outIx, int eltN, bool postOp=false) {
    return LLGenericOp<1, 0, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ void recvReduceSend(intptr_t inpIx, int eltN) {
    return LLGenericOp<1, 1, Input, -1>(inpIx, -1, eltN, false);
  }
  __device__ void recvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    return LLGenericOp<1, 0, Input, Output>(inpIx, outIx, eltN, postOp);
  }

#ifdef KERNEL_LOOPBACK
  __device__ void sendRecvLoopback(intptr_t inpIx, intptr_t outIx, int nelem) {
    return LLGenericOp<0, 0, Input, Output>(inpIx, outIx, nelem, false);
  }
#endif

  __device__ void copySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    return LLGenericOp<0, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ void recvCopySend(intptr_t outIx, int eltN, bool postOp=false) {
    return LLGenericOp<1, 1, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ void recvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    return LLGenericOp<1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void
  scatter(intptr_t inpIx, int totalElem, int peerElem, ssize_t rankOffset) {
    ScatterGatherOp<0, 1>(inpIx, -1, totalElem, peerElem, rankOffset);
  }

  __device__ __forceinline__ void
  gather(intptr_t outIx, int totalElem, int peerElem, ssize_t rankOffset) {
    ScatterGatherOp<1, 0>(-1, outIx, totalElem, peerElem, rankOffset);
  }

  __device__ __forceinline__ ~Primitives() {
    // Save steps for the next operation
    if (tid >= nthreads-WARP_SIZE && wid < fan.nrecv())
      volatileStoreU64(reinterpret_cast<volatile uint64_t*>(&recvConn->step), recvConnHead);
    if (tid < fan.nsend())
      volatileStoreU64(reinterpret_cast<volatile uint64_t*>(&sendConn->step), sendConnHead);
    // Ensure all steps written back
    barrier();
  }
};
