/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
// p2p_dma.cc: host side of the symmetric point-to-point OKI DMA transport.
// See src/include/p2p_dma.h for the protocol and docs/symmetric_p2p_dma_design.md
// for the full write-up. Independent of, and does not reuse, the kernel-mailbox
// symmetric p2p implementation.

#include "p2p_dma.h"
#include "comm.h"
#include "enqueue.h"
#include "param.h"
#include "bootstrap.h"
#include "dev_runtime.h"
#include "alloc.h"
#include "symmetric.h"
#include "bitops.h"
#include "utils.h"
#include <cstring>
#include <sched.h>

extern int ppuDiesNum;

// Master switch. Off by default: this transport is new and must not change
// behavior for anyone who has not explicitly opted in. The p2p DMA transport
// is OKI-only; the host-CE backend is intentionally disabled.
PCCL_PARAM(P2pDmaEnable, "P2P_DMA_ENABLE", 0);
// Smallest op this transport will carry. Ops below it stay on the kernel p2p
// path, so a run can keep OKI for the sizes where it wins and avoid it where the
// per-queue host-link cost dominates. -1 (default) routes every size here.
//
// The test is per op, which is what makes it safe: for a matched send/recv pair
// both endpoints see the same byte count, so they always reach the same verdict
// and the ops they each pick out of a peer's queue line up one for one. A test
// over the batch as a whole would not be safe -- a rank's local task set differs
// from its peers', so the two ends could choose different transports and the DMA
// side would wait on a declaration that never arrives.
//
// Must be set identically on every rank, like PCCL_P2P_DMA_ENABLE itself.
PCCL_PARAM(P2pDmaMinBytes, "P2P_DMA_MIN_BYTES", -1);
#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
// OKI sends smaller than this use only die 0: each QP operation costs a
// host-link round trip through the OKI device library, so at small sizes the
// second die's submission overhead outweighs its bandwidth share.
PCCL_PARAM(P2pDmaOki1DieMax, "P2P_DMA_OKI_1DIE_MAX", 4 * 1024 * 1024);
#endif

static inline int p2pDmaDies() {
  int d = ppuDiesNum > 1 ? ppuDiesNum : 1;
  if (d > PCCL_CE_MAX_DIE_NUM) d = PCCL_CE_MAX_DIE_NUM;
  return d;
}

// An op is carried by this transport only if it is at least this large. 0 means
// no threshold.
static inline uint64_t p2pDmaMinBytes() {
  int64_t v = pcclParamP2pDmaMinBytes();
  return v < 0 ? 0 : (uint64_t)v;
}

// Bytes of work list one batch can need at worst: every peer contributing a full
// quota in both directions. p2pDmaReserveWorks is deliberately non-wrapping, so a
// run larger than the whole FIFO could never be satisfied and would spin forever
// -- hence this is a precondition, not something to discover at reserve time.
static inline size_t p2pDmaWorstCaseWorkBytes(int nRanks) {
  return alignUp((size_t)2 * nRanks * pcclP2pDmaOpsPerPeer * sizeof(struct pcclP2pDmaWork),
                 (size_t)16);
}

bool pcclP2pDmaEnabled(struct pcclComm* comm) {
  if (!pcclParamP2pDmaEnable()) return false;
  // Gate on the same condition as the CE collective path (ncclCeSupported:
  // symmetric support, single node, CTA policy ZERO). This transport is part of
  // the copy-engine family, so it follows the same two-layer usage: the CE gate
  // opens the copy-engine path, and okiSupport (PCCL_OKI_ENABLE) below is what
  // then carries the payload over OKI. Under any other CTA policy the plain
  // kernel p2p path is used, uniformly with the collectives.
  if (!ncclCeSupported(comm)) return false;
  // The work list is reserved from the comm's work FIFO in one non-wrapping run,
  // so the FIFO must be able to hold a worst-case batch. PCCL_WORK_FIFO_BYTES is
  // user-tunable and could be set below that, which would leave
  // p2pDmaWaitWorkFifoAvailable spinning on a condition that can never become
  // true. Both terms are comm-level, so the verdict stays rank-invariant.
  if (p2pDmaWorstCaseWorkBytes(comm->nRanks) > (size_t)comm->workFifoBytes) {
    WARN_ONCE("P2P DMA disabled: PCCL_WORK_FIFO_BYTES=%u is smaller than the %zu bytes a %d-rank batch can need",
              comm->workFifoBytes, p2pDmaWorstCaseWorkBytes(comm->nRanks), comm->nRanks);
    return false;
  }
#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
  // OKI carries the payload, so there is no transport at all without it.
  if (!comm->okiSupport) return false;
  if (comm->okiQps == nullptr || comm->okiQpCount <= 0) return false;
  // p2pDmaSlotOf folds peers onto the remote slots of the QP array, so a
  // clique wider than the array shares queues instead of running off its end;
  // all that is required is one usable remote slot. okiSupport above already
  // rules out a 1-rank comm, so a remote slot is always the right thing to
  // require here. This stays a comm-level test so the answer is rank-invariant:
  // a per-batch test would let the two ranks of one send/recv pair disagree
  // (their slots come from their own (peer - rank) distance) and the DMA side
  // would wait on a declaration that never arrives.
  if (pcclP2pDmaQpSlots(comm->okiQpCount, p2pDmaDies()) < 1) return false;
  return true;
#else
  return false;
#endif
}

static bool p2pDmaDetectWindows(struct pcclComm* comm, struct ncclDevrWindow** outSendWin,
                                struct ncclDevrWindow** outRecvWin);

// How many ops from `q` this transport may take: every op at or above the size
// threshold, capped at pcclP2pDmaOpsPerPeer. Ops below the threshold are simply
// passed over -- both endpoints of a matched sequence apply the same predicate to
// the same sizes, so the taken subsequences still correspond position for
// position and `k` keeps its meaning on both sides.
static int p2pDmaTakeableFromQueue(struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next>* q,
                                   uint64_t minBytes) {
  int n = 0;
  for (struct ncclTaskP2p* t = ncclIntruQueueHead(q); t && n < pcclP2pDmaOpsPerPeer; t = t->next) {
    if ((uint64_t)t->bytes >= minBytes) n++;
  }
  return n;
}

bool pcclP2pDmaEligible(struct pcclComm* comm) {
  if (!pcclP2pDmaEnabled(comm)) return false;
  // nNodes == 1 is already implied by pcclP2pDmaEnabled -> ncclCeSupported.
  if (!ncclDevrIsOneLsaTeam(comm)) return false; // lsa index == world rank, assumed below
  // With a size threshold set, a group can be entirely below it. Building a plan
  // then would consume nothing, so the planner has to be told to use the kernel
  // path instead of looping on an empty batch.
  uint64_t minBytes = p2pDmaMinBytes();
  if (minBytes != 0) {
    struct ncclKernelPlanner::Peer* peers = comm->planner.peers;
    bool any = false;
    for (int p = 0; p < comm->nRanks && !any; p++) {
      any = p2pDmaTakeableFromQueue(&peers[p].sendQueue, minBytes) != 0
         || p2pDmaTakeableFromQueue(&peers[p].recvQueue, minBytes) != 0;
    }
    if (!any) return false;
  }
  struct ncclDevrWindow *sendWin = nullptr, *recvWin = nullptr;
  return p2pDmaDetectWindows(comm, &sendWin, &recvWin);
}

////////////////////////////////////////////////////////////////////////////////
// Lazy init / finalize

ncclResult_t pcclP2pDmaInitOnce(struct pcclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  struct pcclP2pDmaState* state = &comm->p2pDmaState;
  if (state->initialized) return ncclSuccess;
  NCCLCHECK(ncclDevrInitOnce(comm));

  uint8_t* mboxDev = nullptr;
  size_t mboxSize = pcclP2pDmaMboxWindowSize(comm->nRanks);
  ncclWindow_vidmem* mboxWinDev = nullptr;
  ncclWindow_vidmem* mboxWinDevHost = nullptr;

  NCCLCHECKGOTO(pcclMemAlloc((void**)&mboxDev, mboxSize), ret, fail);
  NCCLCHECKGOTO(ncclDevrWindowRegisterInGroup(comm, mboxDev, mboxSize, NCCL_WIN_COLL_SYMMETRIC, &mboxWinDev), ret, fail);
  NCCLCHECKGOTO(ncclShadowPoolToHost(&comm->devrState.shadows, mboxWinDev, &mboxWinDevHost), ret, fail);
  state->mboxWin = (struct ncclDevrWindow*)mboxWinDevHost->winHost;
  HGGCCHECKGOTO(hggcMemset(state->mboxWin->userPtr, 0, mboxSize), ret, fail);
  state->mboxBase = (void*)mboxWinDevHost->lsaFlatBase;
  NCCLCHECKGOTO(ncclCalloc(&state->peerSeq, comm->nRanks), ret, fail);
  state->initialized = true;
  return ncclSuccess;

fail:
  if (mboxWinDev) (void)ncclCommWindowDeregister(comm, mboxWinDev);
  if (mboxDev) (void)pcclMemFree(mboxDev);
  state->mboxWin = nullptr;
  state->mboxBase = nullptr;
  return ret;
}

ncclResult_t pcclP2pDmaFinalize(struct pcclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  struct pcclP2pDmaState* state = &comm->p2pDmaState;
  if (!state->initialized) return ncclSuccess;
  if (state->mboxWin) {
    void* userPtr = state->mboxWin->userPtr;
    NCCLCHECKIGNORE(ncclCommWindowDeregister(comm, state->mboxWin->vidmem), ret);
    NCCLCHECKIGNORE(pcclMemFree(userPtr), ret);
    state->mboxWin = nullptr;
  }
  if (state->peerSeq) { NCCLCHECK(ncclFree(state->peerSeq)); state->peerSeq = nullptr; }
  state->initialized = false;
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
// Work FIFO reservation for the device-visible p2p work list. This mirrors the
// native uploadWork() FIFO discipline, but the DMA p2p plan is not a normal
// ncclDevKernelArgs plan and therefore must reserve its work list itself.

static ncclResult_t p2pDmaWaitWorkFifoAvailable(ncclComm_t comm, uint32_t desiredProduced) {
  bool hasRoom = (desiredProduced - comm->workFifoConsumed) <= comm->workFifoBytes;
  while (!hasRoom) {
    NCCLCHECK(ncclCommPollEventCallbacks(comm, /*waitSome=*/true));
    hasRoom = (desiredProduced - comm->workFifoConsumed) <= comm->workFifoBytes;
    if (!hasRoom) sched_yield();
  }
  return ncclSuccess;
}

// Carve the device-visible work list out of the comm's existing work FIFO,
// rather than allocating a private buffer: the FIFO is already a host/device
// mirrored ring with produced/consumed accounting, which is exactly what the
// kernel needs to read the list. That is why this transport has a stake in the
// FIFO size (see p2pDmaWorstCaseWorkBytes). The run is non-wrapping so the
// kernel sees one contiguous array.
static ncclResult_t p2pDmaReserveWorks(ncclComm_t comm, size_t bytes, struct pcclP2pDmaWork** outHost,
                                       struct pcclP2pDmaWork** outDev) {
  if (bytes == 0) {
    *outHost = nullptr;
    *outDev = nullptr;
    return ncclSuccess;
  }
  bytes = alignUp(bytes, (size_t)16);
  uint32_t off = alignUp(comm->workFifoProduced, 16);
  NCCLCHECK(p2pDmaWaitWorkFifoAvailable(comm, off + bytes));
  uint32_t mask = comm->workFifoBytes - 1;
  // Keep this simple and non-wrapping. If the aligned run would straddle the
  // ring end, skip to the next lap. workFifoBytes is power-of-two checked at
  // comm init.
  if (((off & mask) + bytes) > comm->workFifoBytes) {
    off = alignUp((off + comm->workFifoBytes) & ~mask, 16);
    NCCLCHECK(p2pDmaWaitWorkFifoAvailable(comm, off + bytes));
  }
  *outHost = (struct pcclP2pDmaWork*)((char*)comm->workFifoBuf + (off & mask));
  *outDev = (struct pcclP2pDmaWork*)((char*)comm->workFifoBufDev + (off & mask));
  comm->workFifoProduced = off + bytes;
  return ncclSuccess;
}

////////////////////////////////////////////////////////////////////////////////
// Window detection: same contract as the symmetric collective path -- one
// window per direction, covering every queued send/recv buffer.
//
// Every buffer must resolve to a registered symmetric window, and all sends of
// a batch must share one window (likewise all recvs): pcclP2pDmaBuildPlan
// stores each op as an offset from a single window base and the device args
// carry one sendFlatBase/recvFlatBase, so a second window in the same batch
// would be addressed off the first window's base. Mixed windows are legal for
// the kernel p2p path, so they fall back instead of failing.

static bool p2pDmaDetectWindows(struct pcclComm* comm, struct ncclDevrWindow** outSendWin,
                                struct ncclDevrWindow** outRecvWin) {
  struct ncclKernelPlanner::Peer* peers = comm->planner.peers;
  struct ncclDevrWindow* sw = nullptr;
  struct ncclDevrWindow* rw = nullptr;
  for (int p = 0; p < comm->nRanks; p++) {
    for (struct ncclTaskP2p* s = ncclIntruQueueHead(&peers[p].sendQueue); s; s = s->next) {
      struct ncclDevrWindow* w = nullptr;
      if (ncclDevrFindWindow(comm, s->buff, &w) != ncclSuccess || w == nullptr) return false;
      if (sw == nullptr) sw = w;
      else if (sw != w) return false;
    }
    for (struct ncclTaskP2p* r = ncclIntruQueueHead(&peers[p].recvQueue); r; r = r->next) {
      struct ncclDevrWindow* w = nullptr;
      if (ncclDevrFindWindow(comm, r->buff, &w) != ncclSuccess || w == nullptr) return false;
      if (rw == nullptr) rw = w;
      else if (rw != w) return false;
    }
  }
  if (sw == nullptr) sw = rw;
  if (rw == nullptr) rw = sw;
  if (sw == nullptr || rw == nullptr) return false;
  ncclSymRegType_t regType;
  if (ncclGetSymRegType(sw, rw, &regType) != ncclSuccess) return false;
  if (regType != ncclSymSendRegRecvReg) return false;
  if (!sw->globalSameMemOffset || !rw->globalSameMemOffset) return false;
  if (ncclDevrWindowHasSysmemSegment(sw) || ncclDevrWindowHasSysmemSegment(rw)) return false;
  *outSendWin = sw;
  *outRecvWin = rw;
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// Plan args: the by-value device args plus the host-only bits pcclP2pDmaLaunch
// needs (die count, thread count). Lives in comm->memScoped, which outlives the
// launch.

struct pcclP2pDmaPlanArgs {
  struct pcclP2pDmaArgs devArgs;
  int nDies;
  int nThreads;
};

ncclResult_t pcclP2pDmaBuildPlan(struct pcclComm* comm, struct ncclKernelPlan* plan) {
  // The mailbox window must already exist: it is registered when the first
  // symmetric window is registered (see ncclCommGroupRegisterSymmetric).
  // Registering it here would deadlock whenever one thread drives more than
  // one comm, because doLaunches walks comms sequentially while the
  // registration carries a blocking bootstrap exchange.
  if (!comm->p2pDmaState.initialized) {
    WARN("P2P DMA path requires at least one symmetric window registered (PCCL_WIN_COLL_SYMMETRIC) before the first send/recv");
    return ncclInvalidUsage;
  }
  struct ncclKernelPlanner* planner = &comm->planner;
  struct pcclP2pDmaState* dmaState = &comm->p2pDmaState;

  struct ncclDevrWindow *sendWin, *recvWin;
  if (!p2pDmaDetectWindows(comm, &sendWin, &recvWin)) return ncclInternalError;

  int numRanks = comm->nRanks;
  int rank = comm->rank;
  int* sendsToTake = ncclMemoryStackAlloc<int>(&comm->memScoped, numRanks);
  int* recvsToTake = ncclMemoryStackAlloc<int>(&comm->memScoped, numRanks);
  uint32_t* batchSeqForPeer = ncclMemoryStackAlloc<uint32_t>(&comm->memScoped, numRanks);
  int numSends = 0, numRecvs = 0;
  uint64_t minBytes = p2pDmaMinBytes();
  for (int peer = 0; peer < numRanks; peer++) {
    int numSendsForPeer = p2pDmaTakeableFromQueue(&planner->peers[peer].sendQueue, minBytes);
    int numRecvsForPeer = p2pDmaTakeableFromQueue(&planner->peers[peer].recvQueue, minBytes);
    if (peer == rank) {
      // Self ops go on the wire list like any other peer, but ride the local D2D
      // queues and skip the mailbox: the receiver is this rank, so the kernel
      // reads the destination straight out of the recv list, pairing the k-th
      // self send with the k-th self recv. Emitting only matched pairs is what
      // makes that lookup always succeed; the counts are equal by API contract.
      int numPairs = numSendsForPeer < numRecvsForPeer ? numSendsForPeer : numRecvsForPeer;
      numSendsForPeer = numRecvsForPeer = numPairs;
      batchSeqForPeer[peer] = 0; // unused: no mailbox cell, no done slot
    } else if (numSendsForPeer != 0 || numRecvsForPeer != 0) {
      uint32_t nextSeq = ++dmaState->peerSeq[peer];
      if (nextSeq == 0) nextSeq = ++dmaState->peerSeq[peer];
      batchSeqForPeer[peer] = nextSeq;
    }
    sendsToTake[peer] = numSendsForPeer;
    recvsToTake[peer] = numRecvsForPeer;
    numSends += numSendsForPeer;
    numRecvs += numRecvsForPeer;
  }
  // A rank that sits out an alltoallv entirely has nTasksP2p == 0, so the caller
  // never enters BuildPlan for it (see the guard in ncclLaunchPrepare) and it
  // simply builds no DMA plan -- its non-participation is supported. Reaching
  // here with an empty batch would therefore be an internal inconsistency: the
  // eligibility gate already guarantees at least one takeable op for this rank.
  if (numSends == 0 && numRecvs == 0) return ncclInternalError;

  struct ncclWindow_vidmem* sendWinHost = nullptr;
  struct ncclWindow_vidmem* recvWinHost = nullptr;
  NCCLCHECK(ncclShadowPoolToHost(&comm->devrState.shadows, sendWin->vidmem, &sendWinHost));
  NCCLCHECK(ncclShadowPoolToHost(&comm->devrState.shadows, recvWin->vidmem, &recvWinHost));
  uint32_t stride4G = sendWinHost->stride4G;

  struct pcclP2pDmaPlanArgs* planArgs = ncclMemoryStackAlloc<struct pcclP2pDmaPlanArgs>(&comm->memScoped);
  struct pcclP2pDmaWork* worksHost = nullptr;
  struct pcclP2pDmaWork* worksDev = nullptr;
  NCCLCHECK(p2pDmaReserveWorks(comm, (size_t)(numSends + numRecvs) * sizeof(struct pcclP2pDmaWork),
                               &worksHost, &worksDev));
  struct pcclP2pDmaWork* sendWorks = worksHost;
  struct pcclP2pDmaWork* recvWorks = worksHost ? worksHost + numSends : nullptr;

  int sendIndex = 0, recvIndex = 0;
  void* apiEvent = nullptr;
  // Walk peers by distance so peerDistance == numRanks lands on this rank: the
  // self ops are drained last and sit at the tail of both work lists.
  for (int peerDistance = 1; peerDistance <= numRanks; peerDistance++) {
    int peer = (rank + peerDistance) % numRanks;
    int32_t peerLocalSlot = 0;
    // Drain the queue, carry the ops at or above the threshold and put the rest
    // back in their original order for a later kernel-p2p plan to pick up. `k`
    // counts only the carried ops, which is exactly what the other endpoint does.
    {
      struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> keep;
      ncclIntruQueueConstruct(&keep);
      int taken = 0;
      while (!ncclIntruQueueEmpty(&planner->peers[peer].sendQueue)) {
        struct ncclTaskP2p* sendTask = ncclIntruQueueDequeue(&planner->peers[peer].sendQueue);
        if (taken == sendsToTake[peer] || (uint64_t)sendTask->bytes < minBytes) {
          ncclIntruQueueEnqueue(&keep, sendTask);
          continue;
        }
        sendWorks[sendIndex].peer = peer;
        sendWorks[sendIndex].k = peerLocalSlot++;
        sendWorks[sendIndex].bytes = (uint64_t)sendTask->bytes;
        sendWorks[sendIndex].off = (uint64_t)((uint8_t*)sendTask->buff - (uint8_t*)sendWin->userPtr);
        sendWorks[sendIndex].seq = batchSeqForPeer[peer];
        sendIndex++;
        taken++;
        apiEvent = sendTask->groupApiEventHandle;
        ncclMemoryPoolFree(&comm->memPool_ncclTaskP2p, sendTask);
        planner->nTasksP2p--;
      }
      ncclIntruQueueTransfer(&planner->peers[peer].sendQueue, &keep);
    }
    peerLocalSlot = 0;
    {
      struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> keep;
      ncclIntruQueueConstruct(&keep);
      int taken = 0;
      while (!ncclIntruQueueEmpty(&planner->peers[peer].recvQueue)) {
        struct ncclTaskP2p* recvTask = ncclIntruQueueDequeue(&planner->peers[peer].recvQueue);
        if (taken == recvsToTake[peer] || (uint64_t)recvTask->bytes < minBytes) {
          ncclIntruQueueEnqueue(&keep, recvTask);
          continue;
        }
        recvWorks[recvIndex].peer = peer;
        recvWorks[recvIndex].k = peerLocalSlot++;
        recvWorks[recvIndex].bytes = (uint64_t)recvTask->bytes;
        recvWorks[recvIndex].off = (uint64_t)((uint8_t*)recvTask->buff - (uint8_t*)recvWin->userPtr);
        recvWorks[recvIndex].seq = batchSeqForPeer[peer];
        recvIndex++;
        taken++;
        apiEvent = recvTask->groupApiEventHandle;
        ncclMemoryPoolFree(&comm->memPool_ncclTaskP2p, recvTask);
        planner->nTasksP2p--;
      }
      ncclIntruQueueTransfer(&planner->peers[peer].recvQueue, &keep);
    }
  }

  planArgs->devArgs.stride4G = stride4G;
  planArgs->devArgs.nRanks = numRanks;
  planArgs->devArgs.rank = rank;
  planArgs->devArgs.sendFlatBase = (uint64_t)((char*)::add4G((void*)sendWinHost->lsaFlatBase, (int)(rank * stride4G)));
  planArgs->devArgs.recvFlatBase = (uint64_t)((char*)::add4G((void*)recvWinHost->lsaFlatBase, (int)(rank * stride4G)));
  planArgs->devArgs.mboxFlatBase = (uint64_t)dmaState->mboxBase;
  planArgs->devArgs.abortFlag = comm->abortFlagDev;
  planArgs->devArgs.nSend = numSends;
  planArgs->devArgs.nRecv = numRecvs;
  planArgs->devArgs.works = worksDev;
  planArgs->nDies = p2pDmaDies();
  planArgs->devArgs.nDoneSignallers = planArgs->nDies;
#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
  planArgs->devArgs.oki1DieMax = (uint32_t)pcclParamP2pDmaOki1DieMax();
  planArgs->devArgs.qpSlots = pcclP2pDmaQpSlots(comm->okiQpCount, planArgs->nDies);
#else
  planArgs->devArgs.oki1DieMax = 0;
  planArgs->devArgs.qpSlots = 0;
#endif

  // Threads cover two independent axes: the per-op phases (publish / done
  // handshake) stride over the work list, and phase 1/2 stride over the QP queues
  // one thread per (slot, die). Size for whichever needs more so a batch never has
  // to walk its queues in several passes -- that serialisation is what made the
  // submission cost show up on the critical path for small, many-peer batches.
  int nOpsMax = numSends > numRecvs ? numSends : numRecvs;
  int nQueues = (planArgs->devArgs.qpSlots + 1) * planArgs->nDies;
  int nUnits = nOpsMax > nQueues ? nOpsMax : nQueues;
  int nThreads = ((nUnits + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  if (nThreads < WARP_SIZE) nThreads = WARP_SIZE;
  if (nThreads > 512) nThreads = 512;
  planArgs->nThreads = nThreads;

  plan->isCeP2p = true;
  plan->kernelSymkArgs = (void*)planArgs;
  plan->threadPerBlock = nThreads;
  plan->workBytes = 0;
  for (int i = 0; i < numSends; i++) plan->workBytes += sendWorks[i].bytes;
  for (int i = 0; i < numRecvs; i++) plan->workBytes += recvWorks[i].bytes;
  if (plan->workBytes == 0) plan->workBytes = 1;
  if (apiEvent) plan->groupApiEventHandle = apiEvent;

  if (comm->rank == 0) {
    INFO(NCCL_TUNING, "SendRecv [Symmetric DMA OKI]: batch nSend %d nRecv %d nSelfSend %d nthreads %d",
         numSends, numRecvs, sendsToTake[rank], nThreads);
  }
  return ncclSuccess;
}

////////////////////////////////////////////////////////////////////////////////
// Submission

#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
static ncclResult_t p2pDmaLaunchOki(struct pcclComm* comm, struct pcclP2pDmaPlanArgs* pa, hggcStream_t stream) {
  ltOkiQP_t* qps = comm->okiQps;
  // The fence value must be drawn from comm->ceOkiStep at launch time, not at
  // plan-build time: all plans of a group are built before any is launched,
  // and the OKI collectives bump the counter only when they launch, so a
  // build-time capture could alias a fence value still in use by another plan.
  uint64_t step = comm->ceOkiStep + 1;
  void* pargs[] = { &pa->devArgs, &qps, &step };
  struct dispatchStrategyExAD launchConfig = {1, 1, 1, -2, /*cooperative=*/1};
  dim3 grid = {1, 1, 1};
  dim3 block = {(uint32_t)pa->nThreads, 1, 1};
  void* roundFn = pa->nDies > 1 ? (void*)pcclP2pDmaOkiRound<2> : (void*)pcclP2pDmaOkiRound<1>;
  // Ordered on the caller stream: no AllGather-style dependency trigger is
  // needed because the kernel itself publishes first and waits on the device
  // mailbox before it writes any WQE.
  LAUNCH_KERNEL(roundFn, grid, block, pargs, (ssize_t)0, stream, 1, launchConfig);
  // Shared with the OKI collectives: this comm has one physical CQ per QP,
  // and the fence value stored there is a plain software counter, not a
  // hardware one -- two independently-incrementing step spaces on the same
  // QP would corrupt each other's completion checks.
  comm->ceOkiStep += 1;
  return ncclSuccess;
}
#endif

ncclResult_t pcclP2pDmaLaunch(struct pcclComm* comm, struct ncclKernelPlan* plan) {
  struct pcclP2pDmaPlanArgs* pa = (struct pcclP2pDmaPlanArgs*)plan->kernelSymkArgs;
  if (pa->devArgs.nSend == 0 && pa->devArgs.nRecv == 0) return ncclSuccess;

#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
  // Fully device-driven, self copies included: a single kernel per batch covers
  // remote-write offset publish, local offset polling, submit, CQ sync and done
  // handshake. A self copy is just a work on the local D2D queues, so it runs
  // concurrently with the peer traffic instead of being serialized in front of
  // it by a host-side memcpy on the same stream.
  return p2pDmaLaunchOki(comm, pa, comm->planner.streams->stream);
#else
  return ncclInternalError;
#endif
}
