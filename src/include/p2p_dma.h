/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
// The receiver publishes its destination in the sender-owned cell[receiverRank][k].
// Matched peer FIFOs use the same size filter and per-peer cap, keeping batch boundaries aligned.
// Stream order must complete each batch before cell reuse; seq rejects stale data but does not serialize reuse.
// Self ops read the local recv list and complete at the CQ wait, so they need no mailbox cell or done slot.

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

// Per-peer batch capacity; larger groups are split into multiple plans.
constexpr int pcclP2pDmaOpsPerPeer = 64;

constexpr int pcclP2pDmaMaxSignallers = 2; // == PCCL_CE_MAX_DIE_NUM

// Release seq after off; acquire seq before reading off.
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

// Done slots use 8 bytes for u64 ICN fences; receivers poll the low u32.
constexpr __host__ __device__ size_t pcclP2pDmaDoneBytes(int nRanks) {
  return alignUp((size_t)nRanks * pcclP2pDmaOpsPerPeer * pcclP2pDmaMaxSignallers
                     * sizeof(uint64_t),
                 (size_t)16);
}

constexpr __host__ __device__ size_t pcclP2pDmaMboxWindowSize(int nRanks) {
  return pcclP2pDmaMboxCellBytes(nRanks) + pcclP2pDmaDoneBytes(nRanks);
}

// QPs are indexed by slot * dies + die: slot 0 is local D2D; each slot contains dies QPs.
// Remote peers fold onto slots 1..pcclP2pDmaQpSlots, excluding the local group.
constexpr __host__ __device__ int pcclP2pDmaQpSlots(int okiQpCount, int dies) {
  return (okiQpCount / dies) - 1;
}

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

int64_t pcclParamP2pDmaEnable();

// Admission must depend only on comm-wide state so ranks agree.
bool pcclP2pDmaEnabled(struct pcclComm* comm);

bool pcclP2pDmaEligible(struct pcclComm* comm);

ncclResult_t pcclP2pDmaInitOnce(struct pcclComm* comm);
ncclResult_t pcclP2pDmaFinalize(struct pcclComm* comm);

// Take at most pcclP2pDmaOpsPerPeer operations per peer, leaving the remainder queued.
ncclResult_t pcclP2pDmaBuildPlan(struct pcclComm* comm, struct ncclKernelPlan* plan);

ncclResult_t pcclP2pDmaLaunch(struct pcclComm* comm, struct ncclKernelPlan* plan);

////////////////////////////////////////////////////////////////////////////////
// Device entry points (defined in collectives/device/symmetric/p2p_dma_kernel.cuh)

#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
template<int DIE_NUM>
__global__ void pcclP2pDmaOkiRound(struct pcclP2pDmaArgs args, ltOkiQP_t* qps, uint64_t* stepDev);
#endif

#endif // PCCL_P2P_DMA_H_
