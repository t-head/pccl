/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
// p2p_dma.h: symmetric point-to-point send/recv over DMA transports.
//
// This is an independent implementation (no code or structures reused from
// the kernel-mailbox symmetric p2p branch). Payload never moves through a
// GPU kernel's own bulk-copy loop: it moves through device-driven OKI queue
// pairs. See docs/symmetric_p2p_dma_design.md for the full protocol write-up
// and the no-deadlock / no-drop argument.
//
// Protocol summary
// -----------------
// The only thing exchanged between endpoints is the receiver's write offset,
// carried by a small symmetric "mailbox" window (one per comm, allocated
// lazily). The receiver remote-writes {offset, seq} into the sender's mailbox;
// the sender polls only its local mailbox and trusts `off` after `seq` matches.
// The sender reports completion by remote-writing a done flag into the
// receiver's mailbox, which the receiver polls locally.
//
// A cell is identified by (owner, peerRank, k): the window belongs to the
// *sender*, cell[receiverRank][k] is written by that receiver and polled by
// that sender locally. `k` is an occurrence index local to one *batch*
// (protocol round), not a global op counter -- see pcclP2pDmaOpsPerPeer below.
//
// Self ops (peer == rank) share the work list and the kernel with remote ops,
// but need no cell and no done slot: the receiver is this rank, so the k-th self
// send simply reads its destination from the k-th self recv in the same work
// list, and the kernel's own CQ poll covers its completion. They ride the local
// D2D queues (QP slot 0) rather than the ICN, so a rank's local copy proceeds
// alongside its peer traffic instead of ahead of it.
//
// Batching
// --------
// The mailbox only has room for pcclP2pDmaOpsPerPeer cells per peer. A
// group whose per-peer op count exceeds that capacity is not rejected: the
// host scheduler simply builds more than one plan (batch), each consuming up
// to the capacity from the front of the peer's FIFO queue. Because the
// capacity is a compile-time constant and both endpoints apply the same
// "take up to N from my local FIFO" rule to the same underlying matched op
// sequence, the batch boundaries line up 1:1 across ranks without any
// runtime negotiation -- see the design doc for why this cannot desync.
//
// Reusing the same physical cells across batches is safe only because every
// plan for one comm's p2p stream executes strictly after the previous one
// (normal stream-order semantics): a batch's full round trip (publish, read,
// transfer, done-signal, done-wait) completes before the next batch, which
// reuses the same cells, is even launched. The per-batch sequence number is
// a data-integrity backstop (rejects a stale read), not the mechanism that
// prevents cell reuse from racing.

#ifndef PCCL_P2P_DMA_H_
#define PCCL_P2P_DMA_H_

#include "pccl_internal.h"
#include "bitops.h"
#include "dev_runtime.h"
#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
#include "lt_oki.h"
#endif

////////////////////////////////////////////////////////////////////////////////
// Wire protocol constants and layout

// Cells (and done slots) reserved per peer, per protocol batch. Not a
// group-wide limit -- see the batching note above.
constexpr int pcclP2pDmaOpsPerPeer = 64;

// Upper bound on how many independent completion signals one op can have.
// OKI reports one completion signal per die (each die DMAs its own slice).
constexpr int pcclP2pDmaMaxSignallers = 2; // == PCCL_CE_MAX_DIE_NUM

// One mailbox cell: the receiver's declared destination for one op. `seq` is
// written *after* `off` (release order) and read *before* `off` (acquire
// order) by the poller, so a torn or stale read is impossible: the reader
// only trusts `off` once it has observed the matching `seq`.
struct alignas(16) pcclP2pDmaMboxCell {
  uint64_t off;
  uint32_t seq;
  uint32_t pad;
};

// One send/recv op, resolved to (peer, batch-local k, byte range, sequence).
struct alignas(16) pcclP2pDmaWork {
  uint64_t off;    // this rank's own buffer offset (send: src, recv: dst)
  uint64_t bytes;
  int32_t  peer;
  int32_t  k;      // occurrence index within this batch, [0, pcclP2pDmaOpsPerPeer)
  uint32_t seq;    // this batch's sequence number for `peer` (never 0)
  uint32_t pad;
};

constexpr __host__ __device__ size_t pcclP2pDmaMboxCellBytes(int nRanks) {
  return (size_t)nRanks * pcclP2pDmaOpsPerPeer * sizeof(struct pcclP2pDmaMboxCell);
}

// done[senderRank][k][slot], written by the sender and polled by the receiver.
// Each slot occupies 8 bytes even though only the low u32 carries the sequence,
// because an ICN fence writes a full u64.
constexpr __host__ __device__ size_t pcclP2pDmaDoneBytes(int nRanks) {
  return alignUp((size_t)nRanks * pcclP2pDmaOpsPerPeer * pcclP2pDmaMaxSignallers
                     * sizeof(uint64_t),
                 (size_t)16);
}

constexpr __host__ __device__ size_t pcclP2pDmaMboxWindowSize(int nRanks) {
  return pcclP2pDmaMboxCellBytes(nRanks) + pcclP2pDmaDoneBytes(nRanks);
}

// Remote peer slots in the QP array. init.cc lays the array out as [dies local
// D2D QPs][one QP per (peer slot, die)] and clamps the total to
// LT_OKI_MAX_QUEUE_NUM, so this is how many distinct peers can get their own
// queue. A wider clique folds several peers onto one queue: an MNNVL clique
// reaches 16 ranks with nNodes still 1, which would want 32 queues but only
// ever gets 16. The local D2D set is not counted here; in the kernel's slot
// space it is slot 0 and the remote slots are 1..pcclP2pDmaQpSlots.
constexpr __host__ __device__ int pcclP2pDmaQpSlots(int okiQpCount, int dies) {
  return (okiQpCount / dies) - 1;
}

////////////////////////////////////////////////////////////////////////////////
// Device-visible args for the OKI p2p kernel.
//
// Raw fields only (no devComm dependency), so this header stays include-light
// and the by-value kernel signature is easy to reason about.

struct alignas(16) pcclP2pDmaArgs {
  uint32_t stride4G;
  int32_t  nRanks;
  int32_t  rank;
  uint64_t sendFlatBase;     // self send-win lsa-flat slot (team base + rank*stride4G)
  uint64_t recvFlatBase;     // self recv-win lsa-flat slot (team base + rank*stride4G)
  uint64_t mboxFlatBase;     // mailbox TEAM lsa-flat base; rank r's window at +r*stride4G
  uint32_t* abortFlag;       // device-visible abort flag (comm->abortFlagDev)
  int32_t  nSend;
  int32_t  nRecv;
  int32_t  nDoneSignallers;  // nDies for OKI
  uint32_t oki1DieMax;       // OKI: sends smaller than this use only die 0 (fewer QP ops)
  int32_t  qpSlots;          // remote peer slots in the QP array; peers fold onto these
  struct pcclP2pDmaWork* works; // [0,nSend) sends, [nSend,nSend+nRecv) recvs; self ops last in each run

  __host__ __device__ struct pcclP2pDmaWork* sends() const { return works; }
  __host__ __device__ struct pcclP2pDmaWork* recvs() const { return works + nSend; }
};

////////////////////////////////////////////////////////////////////////////////
// Per-comm host state

struct pcclP2pDmaState {
  bool initialized;
  struct ncclDevrWindow* mboxWin;
  void* mboxBase;         // self lsa-flat slot of the mailbox window
  uint32_t* peerSeq;      // host-only, one sequence counter per peer
};

////////////////////////////////////////////////////////////////////////////////
// Host API (defined in p2p_dma.cc)

// Rank-invariant: reads only comm-level state, never the task list. Governs
// whether p2p sendrecv is routed through this transport at all.
bool pcclP2pDmaEnabled(struct pcclComm* comm);

// True when the current pending p2p work can be scheduled onto this
// transport: symmetric support, buffers registered as symmetric windows.
bool pcclP2pDmaEligible(struct pcclComm* comm);

ncclResult_t pcclP2pDmaInitOnce(struct pcclComm* comm);
ncclResult_t pcclP2pDmaFinalize(struct pcclComm* comm);

// Consume up to pcclP2pDmaOpsPerPeer sends/recvs per peer from the planner's
// queues and fill in `plan` (sets plan->isCeP2p, plan->kernelSymkArgs, and
// plan->workBytes). Leaves any remainder queued for the next call, so the
// existing ncclLaunchPrepare loop (which keeps building plans while
// nTasksP2p != 0) naturally produces one plan per batch.
ncclResult_t pcclP2pDmaBuildPlan(struct pcclComm* comm, struct ncclKernelPlan* plan);

// Submit one plan: remote-write offset publish, OKI DMA transfer, and the
// done handshake.
ncclResult_t pcclP2pDmaLaunch(struct pcclComm* comm, struct ncclKernelPlan* plan);

////////////////////////////////////////////////////////////////////////////////
// Device entry points (defined in collectives/device/symmetric/p2p_dma_kernel.cuh)

#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
// One kernel carries a whole OKI protocol round: publish, submit, CQ sync,
// done handshake. See p2p_dma_kernel.cuh for the phase breakdown.
template<int DIE_NUM>
__global__ void pcclP2pDmaOkiRound(struct pcclP2pDmaArgs args, ltOkiQP_t* qps, uint64_t step);
#endif

#endif // PCCL_P2P_DMA_H_
