/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
// Mailbox ownership, batch alignment and reuse invariants are documented in p2p_dma.h.

#ifndef PCCL_P2P_DMA_KERNEL_CUH_
#define PCCL_P2P_DMA_KERNEL_CUH_

#include "p2p_dma.h"
#include "utils.h"
#include "symmetric/p2p_dma_oki.cuh"
#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)

// Amortize abort checks in mailbox and done waits.
constexpr int pcclP2pDmaSpinsBeforeAbortCheck = 1 << 16;

static __device__ __forceinline__ bool p2pDmaAborted(uint32_t* abortFlag) {
  return abortFlag != nullptr && *abortFlag != 0;
}

// Publish off before seq; the sender acquires seq before reading off.
static __device__ __forceinline__ void p2pDmaMboxPublish(struct pcclP2pDmaMboxCell* cell,
                                                          uint64_t off, uint32_t seq) {
  cell->off = off;
  asm volatile("ppu.st.release.sys.u32 [%0],%1;" :: "l"(&cell->seq), "r"(seq));
}

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

// Reserve 8 bytes per done slot because ICN fences write a full u64.
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

// Pair the k-th self send with the k-th self recv.
static __device__ __forceinline__ uint64_t p2pDmaSelfDstOff(struct pcclP2pDmaWork* recvs, int nRecv,
                                                            int rank, int k) {
  for (int i = 0; i < nRecv; i++) {
    if (recvs[i].peer == rank && recvs[i].k == k) return recvs[i].off;
  }
  return 0;
}

// Publish all declarations before waiting; drain CQs before signaling done.

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

  // Publish remote recv offsets into the senders' mailboxes; self offsets come from the work list.
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

  // One thread owns each (slot, die) queue, including all peers folded onto it.
  struct pcclP2pDmaMboxCell* myMbox =
      (struct pcclP2pDmaMboxCell*)::add4G((void*)mbox, (int)(rank * stride4G));
  int nSlots = p2pDmaSlotCount(qpSlots);
  int nQueues = nSlots * DIE_NUM;
  for (int qIdx = tid; qIdx < nQueues; qIdx += blockDim.x) {
    int slot = qIdx / DIE_NUM;
    int lane = qIdx % DIE_NUM; // lane is the die index, as the payload split expects
    ltOkiQP_t qp = qps[okiQpGroupToQid(slot, lane, DIE_NUM)];

    // Resolve remote offsets before reserving WQEs.
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
      uint8_t* dst = (uint8_t*)::add4G((void*)args.recvFlatBase, (int)((peer - rank) * stride4G)) + dstOff;
      wqeWritten += p2pDmaWriteCopy<DIE_NUM>(activeDies, lane, qp, size, src, dst, /*local=*/self);
    }
    // A tail CQ fence covers all peers sharing this queue.
    ltOkiQPWriteFenceWqe(qp, (uint64_t)ltOkiQPGetCQ(qp), step, lane, true);
    wqeWritten++;
    // Post emitted WQEs; the reservation may include padding.
    ltOkiPostSend(qp, wqeBase, wqeWritten);
  }
  __syncthreads();

  // Only lane 0 may ring doorbells, after all WQEs are posted.
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

  // Poll only queues with posted fences, before reporting sender completion.
  for (int qIdx = tid; qIdx < nQueues; qIdx += blockDim.x) {
    int slot = qIdx / DIE_NUM;
    int lane = qIdx % DIE_NUM;
    if (!p2pDmaSlotLaneUsed<DIE_NUM>(sends, args.nSend, slot, lane, rank, nRanks, qpSlots, args.oki1DieMax)) continue;
    ltOkiPollCQ(ltOkiQPGetCQ(qps[okiQpGroupToQid(slot, lane, DIE_NUM)]), step);
  }
  __syncthreads();

  // Signal every remote send after CQ completion, including zero-byte sends and inactive dies.
  for (int workId = wid; workId < args.nSend; workId += nWarps) {
    if (laneId >= DIE_NUM) continue;
    int p = sends[workId].peer;
    if (p == rank) continue;
    uint32_t* peerDone = (uint32_t*)::add4G((void*)doneBox, (int)(p * stride4G));
    p2pDmaDoneSignal(p2pDmaDoneSlot(peerDone, rank, sends[workId].k, laneId), sends[workId].seq);
  }
  __syncthreads();

  // Wait for all remote per-die done signals; self copies already completed at the CQ wait.
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
