/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_PRIMS_SIMPLE_H_
#define NCCL_PRIMS_SIMPLE_H_
#include "prims_simple_base.h"
// Implementation of primitive types
template <typename T, int NRECV, int NSEND, class FUNC, int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, int DIRECT, int P2P>
class Primitives<T, FUNC, ProtoSimple<UNROLL, SLICESPERCHUNK, SLICESTEPS>, NRECV, NSEND, DIRECT, P2P>:
    public PrimitivesSimpleBase<T, NRECV, NSEND, FUNC, UNROLL, SLICESPERCHUNK, SLICESTEPS, DIRECT,
     ProtoSimple<UNROLL, SLICESPERCHUNK, SLICESTEPS>,
     Primitives<T, FUNC, ProtoSimple<UNROLL, SLICESPERCHUNK, SLICESTEPS>, NRECV, NSEND, DIRECT, P2P>, P2P> {

 private:
  static constexpr int StepPerSlice = 1;
  const T* inputBuf;
  T* outputBuf;
  int flags;

  // PAT uses a single barrier across all groups
  __device__ __forceinline__ void patBarrier() {
    __ppu_barrier_sync(15, NCCL_PAT_NWORKERS);
  }

 public:
  __device__ __forceinline__ Primitives(
      const int tid, const int nworkers, int* recvPeers, int* sendPeers,
      T* directBuff, struct ncclKernelComm* comm, uint64_t redOpArg, struct ncclShmemGroup* ptrs,
      int group, uint8_t connIndexRecv = 0, uint8_t connIndexSend = 0, struct ncclDevWorkColl* collWork = nullptr,
      struct ncclDevWorkP2p* p2pWork=nullptr, const uint64_t ncclSteps=NCCL_STEPS, bool noRed=false,
      const T* inputBuf = nullptr, T* outputBuf = nullptr, int mode = primsModeDefault
    ): 
    PrimitivesSimpleBase<T, NRECV, NSEND, FUNC, UNROLL, SLICESPERCHUNK, SLICESTEPS, DIRECT,
    ProtoSimple<UNROLL, SLICESPERCHUNK, SLICESTEPS>,
    Primitives<T, FUNC, ProtoSimple<UNROLL, SLICESPERCHUNK, SLICESTEPS>, NRECV, NSEND, DIRECT, P2P>, P2P>
    (tid, nworkers, recvPeers, sendPeers, directBuff, comm, redOpArg, ptrs, group, connIndexRecv, connIndexSend, collWork, p2pWork, ncclSteps, noRed, mode),
    inputBuf(inputBuf), outputBuf(outputBuf), flags(0) {
    if (mode == primsModePatRs || mode == primsModePatAg) {
      const int roles[5] = { ROLE_WAIT_RECV, ROLE_POST_RECV, ROLE_WAIT_SEND, ROLE_POST_SEND, ROLE_SRC | ROLE_DST };
      if (tid < 5) flags |= roles[tid];

      const int nranks = comm->nRanks;
      const int rank = comm->rank;
      struct ncclPatPeer* recvDims = (struct ncclPatPeer*)recvPeers;
      struct ncclPatPeer* sendDims = (struct ncclPatPeer*)sendPeers;

      if (tid < 32 && (1u << tid) < (uint32_t)nranks) {
        int dim = tid;
        uint32_t delta = 1u << dim;
        // Load recv peer
        int recvPeer = mode == primsModePatRs ? (rank - delta + nranks) % nranks : (rank + delta) % nranks;
        struct ncclPatPeer* peer = recvDims + dim;
        struct ncclDevChannelPeer* devPeer = getShmemPtr(comm)->channel.peers[recvPeer];
        struct ncclConnInfo* conn = &devPeer->recv[connIndexRecv];
        peer->conn = conn;
        peer->step = conn->step;
        peer->buff = conn->buffs[NCCL_PROTO_SIMPLE];
        peer->tailPtr = conn->tail;
        peer->stepCache = patConnStepLoadVolatile(conn->tail);
        peer->headPtr = conn->head;
        peer->accSize = 0;
        peer->connStepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS / sizeof(T);
        // Load send peer
        int sendPeer = mode == primsModePatAg ? (rank - delta + nranks) % nranks : (rank + delta) % nranks;
        peer = sendDims + dim;
        devPeer = getShmemPtr(comm)->channel.peers[sendPeer];
        conn = &devPeer->send[connIndexSend];
        peer->conn = conn;
        peer->step = conn->step;
        peer->buff = conn->buffs[NCCL_PROTO_SIMPLE];
        peer->sizesFifo = conn->sizesFifo;
        peer->headPtr = conn->head;
        peer->stepCache = patConnStepLoadVolatile(conn->head);
        peer->tailPtr = conn->tail;
        peer->accSize = 0;
        peer->connStepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS / sizeof(T);
      }
      patBarrier();
    }
  }

  __device__ __forceinline__ void patCopy(struct ncclPatStep* ps, struct ncclPatShmem* shmem) {
    if (ps->flags & PatSkipped) { patBarrier(); patBarrier(); return; } //Skipped
    int nelem = ps->nelem < 0 ? 0 : ps->nelem;

    bool recv = ps->recvDim >= 0 && (flags & (ROLE_POST_RECV | ROLE_WAIT_RECV));
    bool send = ps->sendDim >= 0 && (flags & (ROLE_POST_SEND | ROLE_WAIT_SEND));
    bool postRecv = ps->postRecv && recv;
    bool postSend = ps->postSend && send;
    struct ncclPatPeer* peer = NULL;
    uint64_t step = 0;

    if (recv) {
      peer = shmem->recvDims + ps->recvDim;
      step = peer->step;
    }
    if (send) {
      peer = shmem->sendDims + ps->sendDim;
      step = peer->step;
    }

    struct ncclShmemGroup* groupPtr = getShmemPtr(this->getComm())->groups + this->getGroup();
    const T** srcs = (const T**)groupPtr->srcs;
    T** dsts = (T**)groupPtr->dsts;

    if (recv && (flags & ROLE_WAIT_RECV)) {
      peer = shmem->recvDims + ps->recvDim;
      srcs[0] = ((T*)peer->buff) + ((step + ps->stepOffset) % NCCL_STEPS) * peer->connStepSize + ps->recvOffset;
      uint32_t spins = 0, hangSpins = 0;
      while (peer->stepCache < step + ps->stepOffset + StepPerSlice) {
        peer->stepCache = patConnStepLoadVolatile(peer->tailPtr);
        if (checkAbort(this->getComm(), spins, this->abort, hangSpins)) break;
      }
      if (peer->accSize < (long long)(ps->recvOffset + nelem) + (long long)(step + ps->stepOffset) * peer->connStepSize) {
        dsts[1] = outputBuf + ps->outIx;
      } else {
        dsts[1] = (T*)srcs[0];
      }
    }
    if (send && (flags & ROLE_WAIT_SEND)) {
      peer = shmem->sendDims + ps->sendDim;
      uint32_t spins = 0, hangSpins = 0;
      while (peer->stepCache + NCCL_STEPS < step + StepPerSlice) {
        peer->stepCache = patConnStepLoadVolatile(peer->headPtr);
        if (checkAbort(this->getComm(), spins, this->abort, hangSpins)) break;
      }
      dsts[0] = ((T*)peer->buff) + (step % NCCL_STEPS) * peer->connStepSize + ps->sendOffset;
    }
    long long localAccSize = shmem->localAccSize;
    if (ps->recvDim < 0 && (flags & ROLE_SRC)) {
      srcs[0] = inputBuf + ps->inpIx;
      if (localAccSize < ps->inpIx + nelem) {
        dsts[1] = outputBuf + ps->outIx;
        localAccSize = ps->inpIx + nelem;
      } else {
        dsts[1] = (T*)srcs[0];
      }
    }
    patBarrier();
    int nDsts = 2;
    if (ps->sendDim < 0) { dsts++; nDsts--; } // No peer to send to, remove one dest
    if (srcs[0] == (const T*)groupPtr->dsts[1]) nDsts--; // In-place or already done.

    int workSize = getShmemPtr(this->getComm())->aborted ? 0 : nelem;

    ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, 2>
      (this->getTid(), this->getNthreads(), this->getRedOp(), false, false, 1, srcs, nDsts, dsts, workSize);

    // Store conn step here inside the two barriers to make sure next reload will see the update.
    if (postSend && (flags & ROLE_POST_SEND)) {
      if (peer->sizesFifo) {
        reinterpret_cast<volatile int*>(peer->sizesFifo)[step % NCCL_STEPS] = (ps->sendOffset + nelem) * sizeof(T);
      }
      peer->step = step += StepPerSlice;
      patConnStepStoreVolatile(reinterpret_cast<volatile uint64_t*>(&peer->conn->step), step);
    }
    if (postRecv && (flags & ROLE_POST_RECV)) {
      peer->step = step += StepPerSlice;
      patConnStepStoreVolatile(reinterpret_cast<volatile uint64_t*>(&peer->conn->step), step);
    }

    // Update accSize
    if (ps->recvDim < 0 && (flags & ROLE_SRC)) atomicMax(&shmem->localAccSize, localAccSize);
    if (ps->recvDim >= 0 && (flags & ROLE_WAIT_RECV)) atomicMax(&peer->accSize, (long long)(ps->recvOffset + nelem) + (long long)(step+ps->stepOffset)*peer->connStepSize);

    patBarrier();

    if (postSend && (flags & ROLE_POST_SEND)) {
      __ppu_threadfence_system_uncache();
      patConnStepStoreVolatile(peer->tailPtr, step);
    }
    if (postRecv && (flags & ROLE_POST_RECV)) {
      patConnStepStoreVolatile(peer->headPtr, step);
    }
  }

  __device__ __forceinline__ void patReduce(struct ncclPatStep* ps, struct ncclPatShmem* shmem) {
    if (ps->flags & PatSkipped) { patBarrier(); patBarrier(); return; } //Skipped
    int nelem = ps->nelem < 0 ? 0 : ps->nelem;

    bool recv = ps->recvDim >= 0 && (flags & (ROLE_POST_RECV | ROLE_WAIT_RECV));
    bool send = ps->sendDim >= 0 && (flags & (ROLE_POST_SEND | ROLE_WAIT_SEND));
    bool postRecv = ps->postRecv && recv;
    bool postSend = ps->postSend && send;
    struct ncclPatPeer* peer = NULL;
    uint64_t step = 0;

    if (recv) {
      peer = shmem->recvDims + ps->recvDim;
      step = peer->step;
    }
    if (send) {
      peer = shmem->sendDims + ps->sendDim;
      step = peer->step;
    }

    struct ncclShmemGroup* groupPtr = getShmemPtr(this->getComm())->groups + this->getGroup();
    const T** srcs = (const T**)groupPtr->srcs;
    T** dsts = (T**)groupPtr->dsts;

    if (recv && (flags & ROLE_WAIT_RECV)) {
      peer = shmem->recvDims + ps->recvDim;
      srcs[0] = ((T*)peer->buff) + (step % NCCL_STEPS) * peer->connStepSize + ps->recvOffset;
      uint32_t spins = 0, hangSpins = 0;
      while (peer->stepCache < step + StepPerSlice) {
        peer->stepCache = patConnStepLoadVolatile(peer->tailPtr);
        if (checkAbort(this->getComm(), spins, this->abort, hangSpins)) break;
      }
    }
    if (send && (flags & ROLE_WAIT_SEND)) {
      peer = shmem->sendDims + ps->sendDim;
      uint32_t spins = 0, hangSpins = 0;
      while (peer->stepCache + NCCL_STEPS < step + ps->stepOffset + StepPerSlice) {
        peer->stepCache = patConnStepLoadVolatile(peer->headPtr);
        if (checkAbort(this->getComm(), spins, this->abort, hangSpins)) break;
      }
      dsts[0] = ((T*)peer->buff) + ((step + ps->stepOffset) % NCCL_STEPS) * peer->connStepSize + ps->sendOffset;
      if (peer->accSize < (long long)(ps->sendOffset + nelem) + (long long)(step + ps->stepOffset) * peer->connStepSize) {
        // New data, add our own data to it.
        srcs[1] = inputBuf + ps->inpIx;
      } else {
        // There is already data in there, accumulate instead of writing to it.
        srcs[1] = (T*)dsts[0];
      }
    }
    long long localAccSize = shmem->localAccSize;
    if (ps->sendDim < 0 && (flags & ROLE_DST)) {
      dsts[0] = outputBuf + ps->outIx;
      if (localAccSize < ps->outIx + nelem) {
        // New data, add our own data to it.
        srcs[1] = inputBuf + ps->inpIx;
        localAccSize = ps->outIx + nelem;
      } else {
        // There is already data in there, accumulate instead of writing to it.
        srcs[1] = (T*)dsts[0];
      }
    }
    patBarrier();
    int nSrcs = 2;
    if (ps->recvDim < 0) { srcs++; nSrcs--; } // No peer to receive from, drop one source
    int workSize = getShmemPtr(this->getComm())->aborted ? 0 : nelem;

    ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 2, 1, 1>
      (this->getTid(), this->getNthreads(), this->getRedOp(), false, false, nSrcs, srcs, 1, dsts, workSize);

    // Store conn step here inside the two barriers to make sure next reload will see the update.
    if (postSend && (flags & ROLE_POST_SEND)) {
      if (peer->sizesFifo) {
        reinterpret_cast<volatile int*>(peer->sizesFifo)[step % NCCL_STEPS] =
            (ps->sendOffset + nelem) * sizeof(T);
      }
      peer->step = step += StepPerSlice;
      patConnStepStoreVolatile(reinterpret_cast<volatile uint64_t*>(&peer->conn->step), step);
    }
    if (postRecv && (flags & ROLE_POST_RECV)) {
      peer->step = step += StepPerSlice;
      patConnStepStoreVolatile(reinterpret_cast<volatile uint64_t*>(&peer->conn->step), step);
    }

    // Update accSize
    if (ps->sendDim < 0 && (flags & ROLE_DST)) atomicMax(&shmem->localAccSize, localAccSize);
    if (ps->sendDim >= 0 && (flags & ROLE_WAIT_SEND)) atomicMax(&peer->accSize, (long long)(ps->sendOffset + nelem) + (long long)(step + ps->stepOffset) * peer->connStepSize);

    patBarrier();

    if (postSend && (flags & ROLE_POST_SEND)) {
      __ppu_threadfence_system_uncache();
      patConnStepStoreVolatile(peer->tailPtr, step);
    }
    if (postRecv && (flags & ROLE_POST_RECV)) {
      patConnStepStoreVolatile(peer->headPtr, step);
    }
  }
};

#endif
