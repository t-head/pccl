/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "op128.h"

#define NCCL_LL128_FLAGTHREAD (NCCL_LL128_LINEELEMS-1)
#define REG_PHASE_NUM 2

template <typename T, class FUNC, int NRECV, int NSEND, int DIRECT, int P2P>
class Primitives<T, FUNC, ProtoLL128, NRECV, NSEND, DIRECT, P2P>:
    public PrimitivesWithoutDirect<T, Primitives<T, FUNC, ProtoLL128, NRECV, NSEND, DIRECT, P2P>> {
 private:
  const int tid;
  const int nthreads;
  const int wid;
  const int stepSize;
  const int warp;
  const bool flagThread;
  int nrecv = 0;
  int nsend = 0;
  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead = 0;

  struct ncclConnInfo* sendConn = NULL;
  volatile int* sendConnFifoPtr = NULL;
  volatile uint64_t* sendConnTailPtr = NULL;
  uint64_t sendConnTail;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache; // Cache last seen value

  int recvPeer[NRECV];
  int sendPeer[NSEND];
  uint64_t recvStep[NRECV];
  uint64_t sendStep[NSEND];
  uint64_t* recvBuff[NRECV];
  uint64_t* sendBuff[NSEND];
  struct ncclKernelComm* comm;

  volatile uint64_t* shmem;
  FUNC redOp;

  inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ uint64_t* recvPtr(int i) { return recvBuff[i]+recvOffset(i); }
  inline __device__ uint64_t* sendPtr(int i) { return sendBuff[i]+sendOffset(i); }
  inline __device__ uint64_t recvFlag(int i) { return recvStep[i]+1; }
  inline __device__ uint64_t sendFlag(int i) { return sendStep[i]+1; }

    inline __device__ void barrier() {
      if (NSEND>NRECV) {
        __ppu_barrier_sync(1, nthreads);
      } else {
        __ppu_barrier_sync(2, nthreads);
      }
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
        sendConnFifoPtr[sendStep[wid]%NCCL_STEPS] = nbytes;
      }
      sendConnHead += 1;
    }
  }

  inline __device__ void incRecv(int i) {
    recvStep[i] += 1;
  }
  inline __device__ void postRecv() {
    if (recvConnHeadPtr) {
      recvConnHead += 1;
      volatileStoreU64(recvConnHeadPtr, recvConnHead);
    }
  }

  inline __device__ void incSend(int i) {
    sendStep[i] += 1;
  }
  inline __device__ void postSend() {
    if (sendConnTailPtr) {
      __threadfence();
      sendConnTail += 1;
      volatileStoreU64(sendConnTailPtr, sendConnTail);
    }
  }

  template<int WordPerThread>
  __device__ __forceinline__ void loadRegsBegin(uint64_t(&regs)[WordPerThread], T const *src, int eltN) {
    constexpr int EltPer16B = 16/sizeof(T);
    if(reinterpret_cast<uintptr_t>(src)%16 == 0) {
      /* We are aligned to 16 bytes, so load directly to registers no shmem.
       * Flag threads load half as much data which gets shuffled to the even
       * registers during Finish. The point of splitting into two phases is to
       * defer that shuffle, which incurs a dependency stall, until after other
       * memops are launched by the caller.
       */
      #pragma unroll
      for(int g=0; g < WordPerThread/2; g++) {
        int ix = g*WARP_SIZE - 4*(g/2) + wid - (g%2)*(wid/8);
        if(!flagThread || g%2==0) {
          if(ix*EltPer16B < eltN)
            load128((uint64_t*)(src + ix*EltPer16B), regs[2*g+0], regs[2*g+1]);
        }
      }
    } else {
      // Not aligned. Stage the smallest 16 byte aligned region subsuming the
      // buffer into shmem.
      int misalignment = reinterpret_cast<uintptr_t>(src) % 16;
      uint64_t *src8 = reinterpret_cast<uint64_t*>(reinterpret_cast<uintptr_t>(src) & -uintptr_t(16));

      uint64_t *shm8 = (uint64_t *)shmem;

      #pragma unroll
      for(int g=0; g < WordPerThread/2; g++)
        if((g*WARP_SIZE + wid)*16 < misalignment + eltN*sizeof(T))
          load128(src8 + 2*(g*WARP_SIZE + wid), regs[2*g+0], regs[2*g+1]);
      #pragma unroll
      for(int g=0; g < WordPerThread/2; g++) {
#ifdef NO_TSM
        store128(shm8 + 2*(g*WARP_SIZE + wid), regs[2*g+0], regs[2*g+1]);
#else
        storeShmem128(shm8 + 2*(g*WARP_SIZE + wid), regs[2*g+0], regs[2*g+1]);
#endif
      }

      __syncwarp();

      // Now load from shmem stage to regs. Preserve the same pre-shuffled layout
      // as the aligned case since Finish() will be applied regardless.
      T *shm = (T*)shm8 + misalignment/sizeof(T);
      #pragma unroll
      for(int g=0; g < WordPerThread/2; g++) {
        int ix = g*WARP_SIZE - 4*(g/2) + wid - (g%2)*(wid/8);
        if(!flagThread || g%2==0) {
          if(ix*EltPer16B < eltN) {
#ifdef NO_TSM
            loadMisaligned128(shm + ix*EltPer16B, regs[2*g+0], regs[2*g+1]);
#else
            loadShmemMisaligned128(shm + ix*EltPer16B, regs[2*g+0], regs[2*g+1]);
#endif
          }
        }
      }
    }
  }

  template<int WordPerThread>
  __device__ __forceinline__ void loadRegsFinish(uint64_t(&regs)[WordPerThread]) {
    // Move data out of flag registers into the vacant registers.
    #pragma unroll
    for (int g=1; g < WordPerThread/2; g+=2) {
      if (flagThread) regs[2*g] = regs[2*g-1];
    }
  }

  template<int WordPerThread>
  __device__ __forceinline__ void storeRegs(T *dst, uint64_t(&regs)[WordPerThread], int eltN) {
    constexpr int EltPer16B = 16/sizeof(T);
    // Reverse Finish() register permuatation.
    #pragma unroll
    for (int g=1; g < WordPerThread/2; g+=2) {
      if (flagThread) regs[2*g-1] = regs[2*g];
    }
    // Write to dst if 16-byte aligned, shmem otherwise.
    int misalignment = reinterpret_cast<uintptr_t>(dst)%16;
    uint64_t *shm8 = (uint64_t *)shmem;
    #pragma unroll
    for(int g=0; g < WordPerThread/2; g++) {
      int ix = g*WARP_SIZE - 4*(g/2) + wid - (g%2)*(wid/8);
      if (!flagThread || g%2==0) {
        if(misalignment == 0 && (ix+1)*EltPer16B <= eltN)
          store128((uint64_t*)(dst + ix*EltPer16B), regs[2*g+0], regs[2*g+1]);
        else
          storeShmem128(shm8+2*ix, regs[2*g+0], regs[2*g+1]);
      }
    }
    __syncwarp();
    // Write rest from shmem to dst. No need to coalesce stores to 16-bytes,
    // the hardware keeps up fine.
    T *shm = (T*)shmem;
    int skip = misalignment == 0 ? eltN & -EltPer16B : 0;
    for(int i=skip+wid; i < eltN; i += WARP_SIZE)
      dst[i] = shm[i];
  }

  #define WARP_MASK 0xffffffff

  template <int ELEMS_PER_THREAD, int RECV, int SEND, int SRC, int DST>
  __device__ __forceinline__ void recvReduceSendCopy(uint64_t(&v)[ELEMS_PER_THREAD], int ll128Offset, bool postOp) {
    uint64_t vr[ELEMS_PER_THREAD];

    __syncwarp();
    /************************ Wait first recv ********************/
    if (RECV) {
      uint64_t* ptr = recvPtr(0)+ll128Offset;
      uint64_t flag = recvFlag(0);
      bool needReload;
      hangSpins = spins = 0;
      do {
        needReload = false;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          load128(ptr+u*WARP_SIZE, vr[u], vr[u+1]);
#ifndef LOAD_BYPASS
          needReload |= flagThread && (vr[u+1] != flag);
#endif
        }
      } while (__any_sync(WARP_MASK, needReload) && checkAbort(comm, spins, abort, hangSpins) == 0);
    }

    /************* Finish register load **************/
    if (SRC) {
      // By deferring register shuffle here we've overlapped spinning on first
      // peer's data with memory loads of src data.
      loadRegsFinish(v);
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        v[u] = MULTI<FUNC, T>().preOp(redOp, v[u]);
        if (!flagThread)
          v[u+1] = MULTI<FUNC, T>().preOp(redOp, v[u+1]);
      }
    }
    /************************ Recv rest *********************/
    if (RECV) {
      { // Consume data from first recv
        uint64_t* ptr = recvPtr(0)+ll128Offset;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          v[u] = SRC ? MULTI<FUNC, T>()(redOp, vr[u], v[u]) : vr[u];
          v[u+1] = SRC ? MULTI<FUNC, T>()(redOp, vr[u+1], v[u+1]) : vr[u+1];
        }
      }

      for (int i=1; i<NRECV && i<nrecv; i++) {
        uint64_t flag = recvFlag(i);
        uint64_t* ptr = recvPtr(i)+ll128Offset;
        bool needReload;
        hangSpins = spins = 0;
        do {
          needReload = false;
          #pragma unroll
          for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
            load128(ptr+u*WARP_SIZE, vr[u], vr[u+1]);
#ifndef LOAD_BYPASS
            needReload |= flagThread && (vr[u+1] != flag);
#endif
          }
        } while (__any_sync(WARP_MASK, needReload) && checkAbort(comm, spins, abort, hangSpins) == 0);

        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          v[u] = MULTI<FUNC, T>()(redOp, vr[u], v[u]);
          v[u+1] = MULTI<FUNC, T>()(redOp, vr[u+1], v[u+1]);
        }
      }
    }
    /********************** End Recv ************************/
    if (postOp && !FuncTraits<FUNC>::IsPostOpIdentity) {
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        v[u]   = MULTI<FUNC, T>().postOp(redOp, v[u]);
        v[u+1] = MULTI<FUNC, T>().postOp(redOp, v[u+1]);
      }
    }

    /************************ Send **************************/
    if (SEND) {
      for (int i=1; i<NSEND && i<nsend; i++) {
        uint64_t flag = sendFlag(i);
        uint64_t* ptr = sendPtr(i)+ll128Offset;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          store128(ptr+u*WARP_SIZE, v[u], flagThread ? flag : v[u+1]);
        }
      }
      uint64_t flag = sendFlag(0);
      uint64_t* ptr = sendPtr(0)+ll128Offset;
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        store128(ptr+u*WARP_SIZE, v[u], flagThread ? flag : v[u+1]);
      }
    }
    /********************** End Send ************************/
  }

  static constexpr int WireWordPerSlice = WARP_SIZE*NCCL_LL128_SHMEM_ELEMS_PER_THREAD;
  static constexpr int DataEltPerSlice = (WireWordPerSlice - WireWordPerSlice/NCCL_LL128_LINEELEMS)*(sizeof(uint64_t)/sizeof(T));

  template <int RECV, int SEND, int SRC, int DST>
  __device__ __forceinline__ void GenericOp(const T* srcPtr, T* dstPtr, int nelem, bool postOp) {
    int wireOffset = WireWordPerSlice*warp + 2*wid;
    const int nwarps = nthreads/WARP_SIZE;
    nelem = nelem < 0 ? 0 : nelem;

    if (SEND) waitSend(divUp(nelem, DataEltPerSlice)*WireWordPerSlice*sizeof(uint64_t));
    barrier();

    nelem -= DataEltPerSlice*warp;
    srcPtr += DataEltPerSlice*warp;
    dstPtr += DataEltPerSlice*warp;
    while (nelem > 0) {
      const int eltInSlice = min(nelem, DataEltPerSlice);
      uint64_t regs[NCCL_LL128_SHMEM_ELEMS_PER_THREAD];
      if (SRC) loadRegsBegin(regs, srcPtr, eltInSlice);
      recvReduceSendCopy<NCCL_LL128_SHMEM_ELEMS_PER_THREAD, RECV, SEND, SRC, DST>(regs, wireOffset, postOp);
      if (DST) storeRegs(dstPtr, regs, eltInSlice);

      wireOffset += WireWordPerSlice*nwarps;
      srcPtr += DataEltPerSlice*nwarps;
      dstPtr += DataEltPerSlice*nwarps;
      nelem -= DataEltPerSlice*nwarps;
    }

    barrier();
    FOR_SEND(incSend); if (SEND) postSend();
    FOR_RECV(incRecv); if (RECV) postRecv();
  }

  template <int RECV, int SEND, int SRC, int DST>
  __device__ __forceinline__ void ScatterGatherOp(const T* srcPtr, T* dstPtr, int totalElem, int peerElem, int rankOffset) {
    const int nwarps = nthreads/WARP_SIZE;
    peerElem = peerElem < 0 ? 0 : peerElem;

    if (SEND) waitSend(divUp(peerElem, DataEltPerSlice)*WireWordPerSlice*sizeof(uint64_t));
    barrier();

    int warpOffset = DataEltPerSlice*warp;
    int wireOffset = WireWordPerSlice*warp + 2*wid;
    uint64_t regs[REG_PHASE_NUM][NCCL_LL128_SHMEM_ELEMS_PER_THREAD];
    while (warpOffset < peerElem) {
      if (SEND) {
        int peerOffset = sendPeer[0] * peerElem + warpOffset;
        int peerRemainSize = min(peerElem - warpOffset, totalElem - peerOffset);
        if (peerRemainSize > 0) {
          const int eltInSlice = min(peerRemainSize, DataEltPerSlice);
          T* srcPtrForPeer = (T*)srcPtr + peerOffset;
          if (SRC) loadRegsBegin(regs[0], srcPtrForPeer, eltInSlice);
          __syncwarp();
          if (SRC) loadRegsFinish(regs[0]);
          uint64_t flag = sendFlag(0);
          uint64_t* dstPtrForPeer = sendPtr(0) + wireOffset;
          #pragma unroll
          for (int u=0; u<NCCL_LL128_SHMEM_ELEMS_PER_THREAD; u+=2) {
            store128(dstPtrForPeer+u*WARP_SIZE, regs[0][u], flagThread ? flag : regs[0][u+1]);
          }
        }

        for (int i = 1; i < NSEND; i++) {
          int peerOffset = sendPeer[i] * peerElem + warpOffset;
          int peerRemainSize = min(peerElem - warpOffset, totalElem - peerOffset);
          if (peerRemainSize > 0) {
            const int eltInSlice = min(peerRemainSize, DataEltPerSlice);
            T* srcPtrForPeer = (T*)srcPtr + peerOffset;
            if (SRC) loadRegsBegin(regs[i%2], srcPtrForPeer, eltInSlice);
            __syncwarp();
            if (SRC) loadRegsFinish(regs[i%2]);
            uint64_t flag = sendFlag(i);
            uint64_t* dstPtrForPeer = sendPtr(i) + wireOffset;
            #pragma unroll
            for (int u=0; u<NCCL_LL128_SHMEM_ELEMS_PER_THREAD; u+=2) {
              store128(dstPtrForPeer+u*WARP_SIZE, regs[i%2][u], flagThread ? flag : regs[i%2][u+1]);
            }
          }
        }
      } else if (RECV) {
        for (int i = 0; i < NRECV; i++) {
          int peerOffset = recvPeer[i] * peerElem + warpOffset;
          int peerRemainSize = min(peerElem - warpOffset, totalElem - peerOffset);
          if (peerRemainSize > 0) {
            const int eltInSlice = min(peerRemainSize, DataEltPerSlice);
            uint64_t* ptr = recvPtr(i) + wireOffset;
            T* dstPtrForPeer = dstPtr + peerOffset;
            uint64_t flag = recvFlag(i);
            bool needReload;
            hangSpins = spins = 0;
            do {
              needReload = false;
              #pragma unroll
              for (int u=0; u<NCCL_LL128_SHMEM_ELEMS_PER_THREAD; u+=2) {
                load128(ptr+u*WARP_SIZE, regs[0][u], regs[0][u+1]);
                needReload |= flagThread && (regs[0][u+1] != flag);
              }
            } while (__any_sync(WARP_MASK, needReload) && checkAbort(comm, spins, abort, hangSpins) == 0);
            if (DST) storeRegs(dstPtrForPeer, regs[0], eltInSlice);
          }
        }
      }
      warpOffset += DataEltPerSlice*nwarps;
      wireOffset += WireWordPerSlice*nwarps;
    }

    barrier();
    FOR_SEND(incSend); if (SEND) postSend();
    FOR_RECV(incRecv); if (RECV) postRecv();
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i) {
    recvBuff[i] = (uint64_t*)conn->buffs[NCCL_PROTO_LL128];
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
    sendBuff[i] = (uint64_t*)conn->buffs[NCCL_PROTO_LL128];
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
    if (tid >= nthreads-WARP_SIZE && wid<nsend) {
      if (sendConn->sizesFifo) {
        sendConnTailPtr = sendConn->tail;
        sendConnTail = sendConn->step;
      }
    }
  }

 public:
  __device__ __forceinline__
  Primitives(
      const int tid, const int nthreads, int* recvPeers, int* sendPeers, T* directBuff,
      struct ncclKernelComm* comm, uint64_t redOpArg, struct ncclShmemGroup* ptrs=NULL,
      int group=0, uint8_t connIndexRecv=0, uint8_t connIndexSend=0, struct ncclDevWorkColl* collWork=nullptr,
      const int ncclSteps=NCCL_STEPS, bool noRed=false
  ):
  // WR: fix potential double preOp for up/down tree.
  redOp(noRed ? RedOpIdentityArg<FUNC>::value() : redOpArg),
  comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), warp(tid/WARP_SIZE),
  flagThread((tid%8)==7), stepSize(comm->buffSizes[NCCL_PROTO_LL128]/NCCL_STEPS/sizeof(uint64_t)),

  shmem(getShmemPtr(comm)->data+(threadIdx.x/WARP_SIZE)*NCCL_LL128_SHMEM_ELEMS_PER_THREAD*WARP_SIZE) {
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
    return GenericOp<0, 1, 1, 0>(src, NULL, nelem, false);
  }

  __device__ void recv(T* dst, int nelem, bool postOp=false) {
    return GenericOp<1, 0, 0, 1>(NULL, dst, nelem, postOp);
  }

  __device__ void recvReduceSend(const T* src, int nelem) {
    return GenericOp<1, 1, 1, 0>(src, NULL, nelem, false);
  }

  __device__ void recvReduceCopy(const T* src, T* dst, int nelem, bool postOp=false) {
    return GenericOp<1, 0, 1, 1>(src, dst, nelem, postOp);
  }

  __device__ void copySend(const T* src, T* dst, int nelem, bool postOp=false) {
    return GenericOp<0, 1, 1, 1>(src, dst, nelem, postOp);
  }

  __device__ void recvCopySend(T* dst, int nelem, bool postOp=false) {
    return GenericOp<1, 1, 0, 1>(NULL, dst, nelem, postOp);
  }

  __device__ void recvReduceCopySend(const T* src, T* dst, int nelem, bool postOp=false) {
    return GenericOp<1, 1, 1, 1>(src, dst, nelem, postOp);
  }

#ifdef KERNEL_LOOPBACK
  __device__ void sendRecvLoopback(const T* src, T* dst, int nelem) {
    return GenericOp<0, 0, 1, 1>(src, dst, nelem, false);
  }
#endif

  __device__ __forceinline__ void
  scatter(const T* src, int totalElem, int peerElem, int rankOffset) {
    ScatterGatherOp<0, 1, 1, 0>(src, NULL, totalElem, peerElem, rankOffset);
  }

  __device__ __forceinline__ void
  gather(T* dst, int totalElem, int peerElem, int rankOffset) {
    ScatterGatherOp<1, 0, 0, 1>(NULL, dst, totalElem, peerElem, rankOffset);
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
