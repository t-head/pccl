/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
// p2p_dma_kernel.cuh: device side of the symmetric point-to-point DMA
// transports. See src/include/p2p_dma.h for the protocol and
// docs/symmetric_p2p_dma_design.md for the full write-up.
//
// Every spin loop here polls comm->abortFlagDev periodically instead of
// looping unconditionally, so a user abort always has a way out even if a
// caller mismatched their send/recv pattern.

#ifndef PCCL_P2P_DMA_KERNEL_CUH_
#define PCCL_P2P_DMA_KERNEL_CUH_

#include "p2p_dma.h"
#include "utils.h"
#include "symmetric/p2p_dma_oki.cuh"
#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)

// Spins before the first abort check; chosen so the functional (non-hang)
// case never pays for the check, while a real mismatch is caught quickly.
constexpr int pcclP2pDmaSpinsBeforeAbortCheck = 1 << 16;

static __device__ __forceinline__ bool p2pDmaAborted(uint32_t* abortFlag) {
  return abortFlag != nullptr && *abortFlag != 0;
}

// Receiver declares where it wants op (peer=senderRank, k). The release store on
// `seq` orders the `off` write ahead of it, which is what p2pDmaMboxWaitOffset's
// acquire load pairs with; no separate system fence is needed.
static __device__ __forceinline__ void p2pDmaMboxPublish(struct pcclP2pDmaMboxCell* cell,
                                                          uint64_t off, uint32_t seq) {
  cell->off = off;
  asm volatile("ppu.st.release.sys.u32 [%0],%1;" :: "l"(&cell->seq), "r"(seq));
}

// Sender resolves that declaration. Spins until `seq` matches (never 0, so a
// zero-initialized or stale cell never satisfies the wait), then reads `off`.
static __device__ __forceinline__ uint64_t p2pDmaMboxWaitOffset(struct pcclP2pDmaMboxCell* cell,
                                                                uint32_t seq, uint32_t* abortFlag) {
  uint32_t spins = 0;
  uint32_t got;
  while (true) {
    asm volatile("ppu.ld.acquire.sys.u32 %0,[%1];" : "=r"(got) : "l"(&cell->seq));
    if (got == seq) break;
    if (++spins == pcclP2pDmaSpinsBeforeAbortCheck) {
      spins = 0;
      if (p2pDmaAborted(abortFlag)) break;
    }
  }
  return cell->off;
}

// Done slots use an 8-byte stride even though only the low u32 carries the
// sequence: the OKI ICN fence WQE always writes a full u64 (little-endian, so
// the low word lands at the slot address), and a packed u32 array would let an
// ICN fence clobber the neighbouring slot. The CE exit kernel and the OKI sync
// kernel signal/poll through the low word only.
static __device__ __forceinline__ uint32_t* p2pDmaDoneSlot(uint32_t* doneBase, int senderRank,
                                                           int k, int signaller) {
  return &doneBase[((size_t)senderRank * pcclP2pDmaOpsPerPeer + k) * pcclP2pDmaMaxSignallers * 2
                   + signaller * 2];
}

static __device__ __forceinline__ void p2pDmaDoneSignal(uint32_t* slot, uint32_t seq) {
  asm volatile("ppu.st.release.sys.u32 [%0],%1;" :: "l"(slot), "r"(seq));
}

static __device__ __forceinline__ void p2pDmaWaitDone(uint32_t* slot, uint32_t seq, uint32_t* abortFlag) {
  uint32_t spins = 0;
  uint32_t got;
  while (true) {
    asm volatile("ppu.ld.acquire.sys.u32 %0,[%1];" : "=r"(got) : "l"(slot));
    if (got == seq) break;
    if (++spins == pcclP2pDmaSpinsBeforeAbortCheck) {
      spins = 0;
      if (p2pDmaAborted(abortFlag)) break;
    }
  }
}

// A self op's receiver is this rank, so its declaration is already in this
// batch's own recv list: the k-th self send pairs with the k-th self recv, the
// same rule the mailbox enforces across ranks. The host emits self sends and
// recvs in matched pairs, so this always finds its mate.
static __device__ __forceinline__ uint64_t p2pDmaSelfDstOff(struct pcclP2pDmaWork* recvs, int nRecv,
                                                            int rank, int k) {
  for (int i = 0; i < nRecv; i++) {
    if (recvs[i].peer == rank && recvs[i].k == k) return recvs[i].off;
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
// OKI transport: one kernel carries the whole protocol round -- remote-write
// recv declarations, locally poll offsets, submit WQEs, poll CQs, then run the
// done handshake -- so a batch costs a single kernel launch. The phases are
// separated by __syncthreads(); every cross-rank ordering is enforced by the
// mailbox/done spins, never by kernel boundaries, so merging cannot deadlock:
// every receiver publishes before any sender waits, senders signal done only
// after their own CQs drain, and no phase waits on a peer's later phase.
//
// Self ops (peer == rank) ride the same work list and the same phases, just on
// the local D2D queues (slot 0) instead of the ICN: their destination is known
// without a mailbox round trip and their completion is covered by this kernel's
// own CQ poll, so they skip the publish and the done handshake entirely. Keeping
// them here rather than on a host-side memcpy is what lets a rank's local copy
// run concurrently with its peer traffic instead of serializing ahead of it.

template<int DIE_NUM>
__global__ void pcclP2pDmaOkiRound(struct pcclP2pDmaArgs args, ltOkiQP_t* qps, uint64_t step) {
  int tid = threadIdx.x;
  int wid = tid / WARP_SIZE;
  int laneId = tid % WARP_SIZE;
  int nWarps = blockDim.x / WARP_SIZE;
  int rank = args.rank;
  int nRanks = args.nRanks;
  int qpSlots = args.qpSlots;
  uint32_t stride4G = args.stride4G;
  int opsPerPeer = pcclP2pDmaOpsPerPeer;
  struct pcclP2pDmaMboxCell* mbox = (struct pcclP2pDmaMboxCell*)args.mboxFlatBase;
  uint32_t* doneBox = (uint32_t*)((char*)args.mboxFlatBase + pcclP2pDmaMboxCellBytes(args.nRanks));
  struct pcclP2pDmaWork* sends = args.sends();
  struct pcclP2pDmaWork* recvs = args.recvs();

  // Phase 0: publish every remote recv declaration into the sender's mailbox
  // window. The receiver performs remote writes; the sender only polls its local
  // mailbox in phase 1, avoiding remote polling on the hot path. Self recvs need
  // no cell: phase 1 reads their offset straight out of this work list.
  {
    for (int i = tid; i < args.nRecv; i += blockDim.x) {
      int q = recvs[i].peer;
      if (q == rank) continue;
      struct pcclP2pDmaMboxCell* senderMbox =
          (struct pcclP2pDmaMboxCell*)::add4G((void*)mbox, (int)(q * stride4G));
      p2pDmaMboxPublish(&senderMbox[rank * opsPerPeer + recvs[i].k], recvs[i].off, recvs[i].seq);
    }
  }
  __syncthreads();

  // Phase 1: one *thread* owns one (slot, die) queue and drives it end to end --
  // resolve, reserve, write every WQE, post. Slot 0 is the local D2D queue set
  // that carries the self ops; several remote peers can fold onto one remote slot
  // (p2pDmaSlotOf), and the owning thread walks the folded peers serially.
  //
  // The queue count is small (p2pDmaSlotCount * DIE_NUM, at most 18 here) while a
  // block has at least a full warp, so this hands every queue its own thread in
  // one pass. The previous mapping gave a whole warp to one slot and used only
  // DIE_NUM of its 32 lanes, which meant a batch with few ops -- exactly the
  // 16-rank, one-op-per-peer case -- got a single warp that walked all the slots
  // serially, paying each queue's submission cost end to end on the critical path.
  // Lanes driving distinct QPs is not new: the old mapping already had lane 0 and
  // lane 1 reserve/post on different queues concurrently.
  //
  // Doorbells still ring from lane 0 only (below): ltOkiRingDB from any other lane
  // hangs the engine. That is cheap relative to reserve/post, so it stays serial.
  struct pcclP2pDmaMboxCell* myMbox =
      (struct pcclP2pDmaMboxCell*)::add4G((void*)mbox, (int)(rank * stride4G));
  int nSlots = p2pDmaSlotCount(qpSlots);
  int nQueues = nSlots * DIE_NUM;
  for (int qIdx = tid; qIdx < nQueues; qIdx += blockDim.x) {
    int slot = qIdx / DIE_NUM;
    int lane = qIdx % DIE_NUM; // lane is the die index, as the payload split expects
    ltOkiQP_t qp = qps[okiQpGroupToQid(slot, lane, DIE_NUM)];

    // Size the run, and resolve every declaration this queue needs, before
    // reserving: the reservation window must not contain a spin on a peer.
    // These are local reads -- the receiver remote-wrote into my mailbox, and a
    // self op never had to leave this rank at all.
    uint32_t wqeWanted = 0;
    for (int i = 0; i < args.nSend; i++) {
      size_t size = (size_t)sends[i].bytes;
      if (size == 0) continue; // zero-byte sends never touch a queue
      int activeDies = p2pDmaActiveDies<DIE_NUM>(size, args.oki1DieMax);
      if (lane >= activeDies) continue;
      if (p2pDmaSlotOf(sends[i].peer, rank, nRanks, qpSlots) != slot) continue;
      if (sends[i].peer != rank) {
        (void)p2pDmaMboxWaitOffset(&myMbox[sends[i].peer * opsPerPeer + sends[i].k], sends[i].seq, args.abortFlag);
      }
      wqeWanted += calcDmaSegmentCount(p2pDmaLaneBytes<DIE_NUM>(size, activeDies, lane));
    }
    if (wqeWanted == 0) continue; // this batch puts nothing on this queue

    uint32_t skip;
    uint32_t wqeReserved = wqeWanted + 1 /*tail fence*/;
    uint32_t wqeBase = ltOkiReserveWqeSlots(qp, wqeReserved, skip);
    uint32_t wqeWritten = 0;
    for (int i = 0; i < args.nSend; i++) {
      size_t size = (size_t)sends[i].bytes;
      if (size == 0) continue;
      int activeDies = p2pDmaActiveDies<DIE_NUM>(size, args.oki1DieMax);
      if (lane >= activeDies) continue;
      int peer = sends[i].peer;
      if (p2pDmaSlotOf(peer, rank, nRanks, qpSlots) != slot) continue;
      bool self = (peer == rank);
      uint64_t dstOff = self ? p2pDmaSelfDstOff(recvs, args.nRecv, rank, sends[i].k)
                             : p2pDmaMboxWaitOffset(&myMbox[peer * opsPerPeer + sends[i].k], sends[i].seq, args.abortFlag);
      uint8_t* src = (uint8_t*)args.sendFlatBase + sends[i].off;
      // add4G by (peer - rank) is the identity for a self op, so one expression
      // covers both: the recv window slot of whoever owns the destination.
      uint8_t* dst = (uint8_t*)::add4G((void*)args.recvFlatBase, (int)((peer - rank) * stride4G)) + dstOff;
      wqeWritten += p2pDmaWriteCopy<DIE_NUM>(activeDies, lane, qp, size, src, dst, /*local=*/self);
    }
    // One fence at the queue tail. The queue is in-order, so polling it implies
    // every copy above landed -- for every peer folded onto this queue, which is
    // why the fence is per queue and not per peer.
    ltOkiQPWriteFenceWqe(qp, (uint64_t)ltOkiQPGetCQ(qp), step, lane, true);
    wqeWritten++;
    // Post exactly the number of WQEs written; wqeReserved may have been
    // rounded up by ltOkiReserveWqeSlots and is not what we wrote.
    ltOkiPostSend(qp, wqeBase, wqeWritten);
  }
  __syncthreads();

  // Ring doorbells from lane 0 only, after every queue has been posted, same as
  // the OKI collectives. Recompute which (slot, die) queues this batch used.
  if (laneId == 0) {
    for (int slot = wid; slot < nSlots; slot += nWarps) {
      #pragma unroll
      for (int ii = 0; ii < DIE_NUM; ii++) {
        if (p2pDmaSlotLaneUsed<DIE_NUM>(sends, args.nSend, slot, ii, rank, nRanks, qpSlots, args.oki1DieMax))
          ltOkiRingDB(qps[okiQpGroupToQid(slot, ii, DIE_NUM)]);
      }
    }
  }
  __syncthreads();

  // Phase 2: sender-side completion -- one thread per queue polls the fence it
  // posted, so the waits for every queue overlap instead of being walked by one
  // warp. This is what keeps the stream from reporting the group done while a
  // source buffer is still being read. Only queues that were actually posted may
  // be polled, which is why the predicate must match phase 1's sizing filter --
  // a fence that was never written would never reach `step`.
  for (int qIdx = tid; qIdx < nQueues; qIdx += blockDim.x) {
    int slot = qIdx / DIE_NUM;
    int lane = qIdx % DIE_NUM;
    if (!p2pDmaSlotLaneUsed<DIE_NUM>(sends, args.nSend, slot, lane, rank, nRanks, qpSlots, args.oki1DieMax)) continue;
    ltOkiPollCQ(ltOkiQPGetCQ(qps[okiQpGroupToQid(slot, lane, DIE_NUM)]), step);
  }
  __syncthreads();

  // Phase 3: with all local DMA drained, report every remote send's completion
  // to its receiver (per-die slot). Zero-byte sends never touched a QP; their
  // signal is produced here as well. A self send needs no signal: its receiver
  // is this rank and phase 2 already drained the queue that carried it.
  for (int workId = wid; workId < args.nSend; workId += nWarps) {
    if (laneId >= DIE_NUM) continue;
    int p = sends[workId].peer;
    if (p == rank) continue;
    uint32_t* peerDone = (uint32_t*)::add4G((void*)doneBox, (int)(p * stride4G));
    p2pDmaDoneSignal(p2pDmaDoneSlot(peerDone, rank, sends[workId].k, laneId), sends[workId].seq);
  }
  __syncthreads();

  // Phase 4: receiver-side completion -- every remote recv waits for all per-die
  // done signals. Self recvs are already complete: phase 2 polled the local D2D
  // queue's fence, and that poll precedes this point in program order.
  uint32_t* myDoneBox = (uint32_t*)::add4G((void*)doneBox, (int)(rank * stride4G));
  for (int i = tid; i < args.nRecv; i += blockDim.x) {
    int q = recvs[i].peer;
    if (q == rank) continue;
    for (int sg = 0; sg < DIE_NUM; sg++) {
      p2pDmaWaitDone(p2pDmaDoneSlot(myDoneBox, q, recvs[i].k, sg), recvs[i].seq, args.abortFlag);
    }
  }
}

template
__global__ void pcclP2pDmaOkiRound<1>(struct pcclP2pDmaArgs args, ltOkiQP_t* qps, uint64_t step);
template
__global__ void pcclP2pDmaOkiRound<2>(struct pcclP2pDmaArgs args, ltOkiQP_t* qps, uint64_t step);

#endif // PCCL_HGGC_VERSION >= 2.2.0

#endif // PCCL_P2P_DMA_KERNEL_CUH_
