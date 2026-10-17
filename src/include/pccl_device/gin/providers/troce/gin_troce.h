/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_GIN_GDAKI_TROCE_H_
#define _PCCL_DEVICE_GIN_GDAKI_TROCE_H_

#include <cstdint>

#include "../../gin_gdaki_device_common.h"
#include "gin_troce_device_host_common.h"
#include "gin_troce_hw.h"

// Device side of the GDAKI_TROCE GIN backend. Mirrors providers/mlx5/gin_mlx5.h, except
// that the WQE is built here instead of inside a vendor library, so this header
// also carries the SQ/CQ/doorbell primitives.

struct pcclGinGdakiTroceRequest {
  int peer;
  uint64_t wqeIdx;  // ticket: the WQE index to wait for on the send CQ
};
static_assert(sizeof(pcclGinGdakiTroceRequest) <= sizeof(pcclGinRequest_t),
              "pcclGinGdakiTroceRequest must fit in pcclGinRequest_t");

namespace pccl {
namespace gin {
namespace gdaki_troce {

#define PCCL_TROCE_BSWAP32(x) __builtin_bswap32(x)
#define PCCL_TROCE_BSWAP64(x) __builtin_bswap64(x)
#define PCCL_TROCE_HTOBE32(x) __builtin_bswap32(x)
#define PCCL_TROCE_HTOBE64(x) __builtin_bswap64(x)

// Plain volatile load/store plus __threadfence. Stronger primitives (e.g.
// system-scoped atomic_ref) may emit an instruction the NIC BAR cannot service
// when the target is a mapped doorbell register.
PCCL_DEVICE_INLINE void mfence() { __threadfence_block(); }
// NIC-visibility fence on the WQE->doorbell path. Device scope suffices because
// the SQ/CQ live in GPU memory.
PCCL_DEVICE_INLINE void membar() { __threadfence(); }

template <typename T>
PCCL_DEVICE_INLINE void storeRelaxed(T *ptr, T val) {
  *(volatile T *)ptr = val;
}

template <typename T>
PCCL_DEVICE_INLINE void storeRelease(T *ptr, T val) {
  membar();
  *(volatile T *)ptr = val;
}

template <typename T>
PCCL_DEVICE_INLINE T atomicRead(T *ptr) {
  return *(volatile T *)ptr;
}

PCCL_DEVICE_INLINE int troceTryProgressQp(struct pcclGinGdakiTroceQp *qp);

// Ring the shared troce doorbell (SQ or CQ) safely under both SIMT and SIMD
// (--no-simt) warp execution.
//
// The doorbell is a single device-wide, last-write-wins MMIO register. Data is
// only lost when multiple lanes of the SAME warp issue the store in the SAME
// instruction: the SM coalesces those into one transaction and a single lane's
// value wins. Stores from different warps (or from the same lane at different
// times) are independent transactions and are never coalesced, so NO cross-warp
// lock is needed -- a producer-index doorbell is monotonic, and re-notifying a
// stale index is harmless.
//
// We therefore only serialize within the warp: enumerate exactly the lanes that
// called in (via __activemask) and let each issue its store in its own
// iteration, so every doorbell store executes with at most one active lane and
// can never be coalesced. __activemask() is used ONLY to step over the
// participating lanes -- it is never fed to __syncwarp(), which would be unsafe
// when only a subset of the warp calls in.
template <typename Fn>
PCCL_DEVICE_INLINE void troceRingDbSerialized(Fn ring) {
  unsigned active = __activemask();
  int me = pccl::utility::lane();
  for (unsigned m = active; m != 0; m &= (m - 1)) {
    int src = __ffs(m) - 1;  // next lowest participating lane
    if (src == me) ring();
  }
}

// Reserve `n` WQEBB slots. Slots are handed out monotonically; the caller then
// fills them and publishes with markReady().
PCCL_DEVICE_INLINE uint64_t reserveWqeSlots(struct pcclGinGdakiTroceQp *qp, uint32_t n) {
  uint64_t base = atomicAdd((unsigned long long *)qp->resv_head, (unsigned long long)n);
  // Wait until the slot has been freed by a completion (SQ wrap-around).
  if (base + n > qp->nwqes) {
    while (atomicRead(qp->cons_idx) + qp->nwqes < base + n) {
      // A producer must make its own CQ progress; there need not be another
      // thread in flush() or waitCounter() to free the SQ slots.
      troceTryProgressQp(qp);
    }
  }
  return base;
}

PCCL_DEVICE_INLINE void *getWqePtr(struct pcclGinGdakiTroceQp *qp, uint64_t wqeIdx) {
  uint16_t idx = (uint16_t)(wqeIdx & (qp->nwqes - 1));
  return (void *)((uintptr_t)qp->sq_wqe + ((uintptr_t)idx << 6));
}

// ---------------------------------------------------------------------------
// troce doorbell
// ---------------------------------------------------------------------------

// The SQ doorbell is a single device-wide MMIO register: the QP is selected by
// the qid field inside the doorbell value, not by the address, so stores must be
// serialised across QPs (see submit()/troceRingDbSerialized). The collapsed CQ
// (ncqe=1 + IGNORE_OVERRUN) needs no CQ doorbell: the NIC never stalls on
// CQ-full and tracks no consumer index, so there is no CQ ring to update.
PCCL_DEVICE_INLINE void troceRingDb(struct pcclGinGdakiTroceQp *qp, uint16_t prodIdx) {
  pcclTroceSqDb db;
  db.value = 0;
  db.db.dw0.bs.qid = qp->qpn;
  db.db.dw0.bs.q_type = PCCL_PAU_QTYPE_ROCE_SQ;
  db.db.dw1.bs.pi = prodIdx;
  // whether the fabric expects a different class than the QP reports.
  db.db.dw1.bs.tc = qp->traffic_class;
  db.db.dw0.value = PCCL_TROCE_HTOBE32(db.db.dw0.value);
  db.db.dw1.value = PCCL_TROCE_HTOBE32(db.db.dw1.value);
  storeRelease((uint64_t *)qp->sq_db, db.value);
}

// Publish the reserved range and ring the doorbell. `endWqeIdx` is inclusive.
//
// The doorbell is a single device-wide register shared by every QP, but stores
// are only lost to coalescing within a warp, so the ring goes through
// troceRingDbSerialized() (warp-serialized, no lock) rather than a global mutex.
// `ringDb == false` implements pcclGinOptFlagsAggregateRequests: ready_head still
// advances but the doorbell is deferred, so prod_idx must stay behind for a later
// submit (or troceRingPendingDb) to announce the accumulated batch.
PCCL_DEVICE_INLINE void submit(struct pcclGinGdakiTroceQp *qp, uint64_t baseWqeIdx, uint64_t endWqeIdx,
                               bool ringDb) {
  uint64_t newProd = endWqeIdx + 1;
  // Single atomic CAS, not read-spin plus store: two submitters on the same QP can
  // both observe ready_head == their base, and the later store would clobber the
  // earlier publish, leaving one batch unannounced.
  mfence();
  while (atomicCAS((unsigned long long *)qp->ready_head, (unsigned long long)baseWqeIdx,
                   (unsigned long long)newProd) != (unsigned long long)baseWqeIdx)
    continue;
  mfence();

  if (!ringDb) return;

  troceRingDbSerialized([&] __device__() {
    uint64_t old = atomicMax((unsigned long long *)qp->prod_idx, (unsigned long long)newProd);
    if (newProd > old) {
      membar();
      troceRingDb(qp, (uint16_t)newProd);
    }
  });
}

// Announce anything that was published but whose doorbell was deferred by
// AggregateRequests. Must run before waiting on a completion, otherwise a batch
// that ended with an aggregated request would never reach the NIC.
PCCL_DEVICE_INLINE void troceRingPendingDb(struct pcclGinGdakiTroceQp *qp) {
  uint64_t ready = atomicRead(qp->ready_head);
  if (ready == atomicRead(qp->prod_idx)) return;  // nothing deferred
  troceRingDbSerialized([&] __device__() {
    uint64_t old = atomicMax((unsigned long long *)qp->prod_idx, (unsigned long long)ready);
    if (ready > old) {
      membar();
      troceRingDb(qp, (uint16_t)ready);
    }
  });
}

// ---------------------------------------------------------------------------
// troce WQE builders
// ---------------------------------------------------------------------------

PCCL_DEVICE_INLINE void troceWriteRdmaWqe(struct pcclGinGdakiTroceQp *qp, uint64_t laddr, __be32 lkey,
                                          uint64_t raddr, __be32 rkey, uint32_t bytes,
                                          uint16_t wqeIdx, bool needCqe, void *wqePtr,
                                          bool isWrite) {
  pcclTroceCtrlSeg ctrl = {};
  pcclTroceRaddrSeg raddrSeg = {};
  pcclTroceDataSeg dataSeg = {};

  pcclTroceCtrlSeg *ctrlPtr = (pcclTroceCtrlSeg *)wqePtr;
  pcclTroceRaddrSeg *raddrPtr =
    (pcclTroceRaddrSeg *)((uintptr_t)ctrlPtr + sizeof(pcclTroceCtrlSeg));
  uintptr_t after = (uintptr_t)raddrPtr + sizeof(pcclTroceRaddrSeg);

  size_t wqeSize = sizeof(pcclTroceCtrlSeg) + sizeof(pcclTroceRaddrSeg);
  pcclTroceDataSeg *dataPtr = (pcclTroceDataSeg *)(after + PCCL_TROCE_16B_PAD(wqeSize));

  raddrSeg.raddr = PCCL_TROCE_HTOBE64(raddr);
  raddrSeg.rkey = rkey;
  raddrSeg.len = PCCL_TROCE_HTOBE32(bytes);

  if (bytes > 0) {
    dataSeg.len = PCCL_TROCE_HTOBE32(bytes);
    dataSeg.dw3.bs.lkey = PCCL_TROCE_BSWAP32(lkey);
    dataSeg.dw3.value = PCCL_TROCE_HTOBE32(dataSeg.dw3.value);
    dataSeg.addr = PCCL_TROCE_HTOBE64(laddr);
  }

  wqeSize += sizeof(pcclTroceDataSeg);
  wqeSize = PCCL_TROCE_DIV_ROUND_UP(wqeSize, PCCL_TROCE_SEND_WQE_BB);

  ctrl.dw0.bs.qpn = (uint16_t)qp->qpn;
  ctrl.dw0.bs.q_type = PCCL_PAU_QTYPE_ROCE_SQ;
  ctrl.dw0.bs.tc = qp->traffic_class;
  ctrl.dw0.bs.wqe_size = (uint8_t)(wqeSize - 1);
  ctrl.dw0.value = PCCL_TROCE_HTOBE32(ctrl.dw0.value);

  ctrl.dw1.bs.ce = needCqe ? 2 : 0;
  ctrl.dw1.bs.opc = isWrite ? PCCL_TROCE_OPCODE_RDMA_WRITE : PCCL_TROCE_OPCODE_RDMA_READ;
  ctrl.dw1.bs.wqe_idx = wqeIdx;
  if (bytes == 0) ctrl.dw1.bs.in_dat = 1;
  ctrl.dw1.value = PCCL_TROCE_HTOBE32(ctrl.dw1.value);

  ctrl.dw2.bs.local_sge_num = bytes > 0 ? 1 : 0;
  ctrl.dw2.value = PCCL_TROCE_HTOBE32(ctrl.dw2.value);
  ctrl.pld_len = PCCL_TROCE_HTOBE32(bytes);

  uint32_t *dst = (uint32_t *)ctrlPtr;
  uint32_t *src = (uint32_t *)&ctrl;
  for (unsigned i = 0; i < sizeof(*ctrlPtr) / sizeof(uint32_t); ++i) storeRelaxed(&dst[i], src[i]);
  dst = (uint32_t *)raddrPtr;
  src = (uint32_t *)&raddrSeg;
  for (unsigned i = 0; i < sizeof(*raddrPtr) / sizeof(uint32_t); ++i) storeRelaxed(&dst[i], src[i]);
  dst = (uint32_t *)dataPtr;
  src = (uint32_t *)&dataSeg;
  for (unsigned i = 0; i < sizeof(*dataPtr) / sizeof(uint32_t); ++i) storeRelaxed(&dst[i], src[i]);
}

// Inline RDMA_WRITE: the payload travels inside the WQE, so concurrent putValue
// calls need no shared staging buffer.
PCCL_DEVICE_INLINE void troceWriteRdmaInlineWqe(struct pcclGinGdakiTroceQp *qp, const void *val,
                                                uint32_t bytes, uint64_t raddr, __be32 rkey,
                                                uint16_t wqeIdx, bool needCqe, void *wqePtr) {
  pcclTroceCtrlSeg ctrl = {};
  pcclTroceRaddrSeg raddrSeg = {};

  pcclTroceCtrlSeg *ctrlPtr = (pcclTroceCtrlSeg *)wqePtr;
  pcclTroceRaddrSeg *raddrPtr =
    (pcclTroceRaddrSeg *)((uintptr_t)ctrlPtr + sizeof(pcclTroceCtrlSeg));
  void *dataPtr = (void *)((uintptr_t)raddrPtr + sizeof(pcclTroceRaddrSeg));
  const uint32_t hdrSize = (uint32_t)((uintptr_t)dataPtr - (uintptr_t)ctrlPtr);

  size_t total = hdrSize + bytes;
  total += PCCL_TROCE_16B_PAD(total);
  size_t wqeSize = PCCL_TROCE_DIV_ROUND_UP(total, PCCL_TROCE_SEND_WQE_BB);

  raddrSeg.raddr = PCCL_TROCE_HTOBE64(raddr);
  raddrSeg.rkey = rkey;
  raddrSeg.len = PCCL_TROCE_HTOBE32(bytes);

  ctrl.dw0.bs.qpn = (uint16_t)qp->qpn;
  ctrl.dw0.bs.q_type = PCCL_PAU_QTYPE_ROCE_SQ;
  ctrl.dw0.bs.tc = qp->traffic_class;
  ctrl.dw0.bs.wqe_size = (uint8_t)(wqeSize - 1);
  ctrl.dw0.value = PCCL_TROCE_HTOBE32(ctrl.dw0.value);

  ctrl.dw1.bs.ce = needCqe ? 2 : 0;
  ctrl.dw1.bs.opc = PCCL_TROCE_OPCODE_RDMA_WRITE;
  ctrl.dw1.bs.wqe_idx = wqeIdx;
  ctrl.dw1.bs.in_dat = 1;  // payload is inline, not an SGE
  ctrl.dw1.bs.inline_offset = 0;
  ctrl.dw1.value = PCCL_TROCE_HTOBE32(ctrl.dw1.value);

  ctrl.dw2.bs.local_sge_num = 0;
  ctrl.dw2.value = PCCL_TROCE_HTOBE32(ctrl.dw2.value);
  ctrl.pld_len = PCCL_TROCE_HTOBE32(bytes);

  uint32_t *dst = (uint32_t *)ctrlPtr;
  uint32_t *src = (uint32_t *)&ctrl;
  for (unsigned i = 0; i < sizeof(*ctrlPtr) / sizeof(uint32_t); ++i) storeRelaxed(&dst[i], src[i]);
  dst = (uint32_t *)raddrPtr;
  src = (uint32_t *)&raddrSeg;
  for (unsigned i = 0; i < sizeof(*raddrPtr) / sizeof(uint32_t); ++i) storeRelaxed(&dst[i], src[i]);

  // Zero the padded payload area first so the tail bytes are deterministic.
  for (uint32_t o = 0; o < (uint32_t)(total - hdrSize); o += sizeof(uint32_t))
    storeRelaxed((uint32_t *)((uintptr_t)dataPtr + o), (uint32_t)0);

  switch (bytes) {
    case 1: storeRelaxed((uint8_t *)dataPtr, *((const uint8_t *)val)); break;
    case 2: storeRelaxed((uint16_t *)dataPtr, *((const uint16_t *)val)); break;
    case 4: storeRelaxed((uint32_t *)dataPtr, *((const uint32_t *)val)); break;
    case 8:
      storeRelaxed(&(((uint32_t *)dataPtr)[0]), ((const uint32_t *)val)[0]);
      storeRelaxed(&(((uint32_t *)dataPtr)[1]), ((const uint32_t *)val)[1]);
      break;
    default:
      for (uint32_t o = 0; o < bytes; ++o)
        storeRelaxed((uint8_t *)((uintptr_t)dataPtr + o), ((const uint8_t *)val)[o]);
      break;
  }
}

// 8-byte fetch-and-add, used to bump a remote signal or a counter. The fetched
// value lands in the local sink buffer and is never consumed.
PCCL_DEVICE_INLINE void troceWriteAtomicFaWqe(struct pcclGinGdakiTroceQp *qp, uint64_t addVal,
                                              uint64_t laddr, __be32 lkey, uint64_t raddr,
                                              __be32 rkey, uint16_t wqeIdx, bool needCqe,
                                              void *wqePtr) {
  pcclTroceCtrlSeg ctrl = {};
  pcclTroceRaddrSeg raddrSeg = {};
  pcclTroceAtomicSeg atomicSeg = {};
  pcclTroceDataSeg dataSeg = {};

  pcclTroceCtrlSeg *ctrlPtr = (pcclTroceCtrlSeg *)wqePtr;
  pcclTroceRaddrSeg *raddrPtr =
    (pcclTroceRaddrSeg *)((uintptr_t)ctrlPtr + sizeof(pcclTroceCtrlSeg));
  pcclTroceAtomicSeg *atomicPtr =
    (pcclTroceAtomicSeg *)((uintptr_t)raddrPtr + sizeof(pcclTroceRaddrSeg));
  uintptr_t after = (uintptr_t)atomicPtr + sizeof(pcclTroceAtomicSeg);

  size_t wqeSize =
    sizeof(pcclTroceCtrlSeg) + sizeof(pcclTroceRaddrSeg) + sizeof(pcclTroceAtomicSeg);
  // An atomic WQE spans more than one WQEBB, so the trailing data segment can
  // cross the SQ wrap point; in that case it lands at the ring base.
  bool wraps = ((wqeIdx + 1) % qp->nwqes) == 0;
  pcclTroceDataSeg *dataPtr = wraps ? (pcclTroceDataSeg *)qp->sq_wqe
                                    : (pcclTroceDataSeg *)(after + PCCL_TROCE_16B_PAD(wqeSize));

  raddrSeg.raddr = PCCL_TROCE_HTOBE64(raddr);
  raddrSeg.rkey = rkey;
  raddrSeg.len = PCCL_TROCE_HTOBE32(PCCL_TROCE_ATOMIC_SIZE);

  dataSeg.len = PCCL_TROCE_HTOBE32(PCCL_TROCE_ATOMIC_SIZE);
  dataSeg.dw3.bs.lkey = PCCL_TROCE_BSWAP32(lkey);
  dataSeg.dw3.value = PCCL_TROCE_HTOBE32(dataSeg.dw3.value);
  dataSeg.addr = PCCL_TROCE_HTOBE64(laddr);

  atomicSeg.swap_or_add_dat = PCCL_TROCE_HTOBE64(addVal);

  wqeSize += sizeof(pcclTroceDataSeg);
  wqeSize = PCCL_TROCE_DIV_ROUND_UP(wqeSize, PCCL_TROCE_SEND_WQE_BB);

  ctrl.dw0.bs.qpn = (uint16_t)qp->qpn;
  ctrl.dw0.bs.q_type = PCCL_PAU_QTYPE_ROCE_SQ;
  ctrl.dw0.bs.tc = qp->traffic_class;
  ctrl.dw0.bs.wqe_size = (uint8_t)(wqeSize - 1);
  ctrl.dw0.value = PCCL_TROCE_HTOBE32(ctrl.dw0.value);

  ctrl.dw1.bs.ce = needCqe ? 2 : 0;
  ctrl.dw1.bs.opc = PCCL_TROCE_OPCODE_ATOMIC_FA;
  ctrl.dw1.bs.wqe_idx = wqeIdx;
  ctrl.dw1.value = PCCL_TROCE_HTOBE32(ctrl.dw1.value);

  ctrl.dw2.bs.local_sge_num = 1;
  ctrl.dw2.value = PCCL_TROCE_HTOBE32(ctrl.dw2.value);
  ctrl.pld_len = PCCL_TROCE_HTOBE32(PCCL_TROCE_ATOMIC_SIZE);

  uint32_t *dst = (uint32_t *)ctrlPtr;
  uint32_t *src = (uint32_t *)&ctrl;
  for (unsigned i = 0; i < sizeof(*ctrlPtr) / sizeof(uint32_t); ++i) storeRelaxed(&dst[i], src[i]);
  dst = (uint32_t *)raddrPtr;
  src = (uint32_t *)&raddrSeg;
  for (unsigned i = 0; i < sizeof(*raddrPtr) / sizeof(uint32_t); ++i) storeRelaxed(&dst[i], src[i]);
  dst = (uint32_t *)atomicPtr;
  src = (uint32_t *)&atomicSeg;
  for (unsigned i = 0; i < sizeof(*atomicPtr) / sizeof(uint32_t); ++i) storeRelaxed(&dst[i], src[i]);
  dst = (uint32_t *)dataPtr;
  src = (uint32_t *)&dataSeg;
  for (unsigned i = 0; i < sizeof(*dataPtr) / sizeof(uint32_t); ++i) storeRelaxed(&dst[i], src[i]);
}

// ---------------------------------------------------------------------------
// troce CQ progress. The single CQ slot contains the newest completed WQE.
// ---------------------------------------------------------------------------

// This probe never waits for a CQE. The same QP may be progressed by a counter
// reader, a waiter, a flush, or a producer whose SQ is full. Only one of them
// may consume pending_counter[] at a time, and SQ slots cannot be recycled until
// all counters covered by the completion have been updated.
PCCL_DEVICE_INLINE int troceTryProgressQp(struct pcclGinGdakiTroceQp *qp) {
  troceRingPendingDb(qp);
  if (atomicCAS((unsigned long long *)qp->progress_lock, 0ull, 1ull) != 0ull) return 0;

  int status = 0;
  uint64_t oldCons = atomicRead(qp->cons_idx);
  uint64_t prodIdx = atomicRead(qp->prod_idx);
  if (oldCons < prodIdx) {
    struct troce_be_cqe *cqe0 = (struct troce_be_cqe *)qp->cq_cqe;
    uint32_t dw14 = atomicRead((uint32_t *)&cqe0->dw14.value);
    uint16_t wqeCounter = __builtin_bswap16(
      (uint16_t)((dw14 & PCCL_TROCE_CQE_WQE_IDX_MASK) >> PCCL_TROCE_CQE_WQE_IDX_SHIFT));
    // At most nwqes WQEBBs can be outstanding. Relative to oldCons, an older
    // CQE (including the initial all-ones CQ slot) has a distance >= nwqes.
    uint16_t distance = (uint16_t)(wqeCounter - (uint16_t)oldCons);
    if (distance < qp->nwqes) {
      uint64_t newCons = oldCons + (uint64_t)distance + 1;
      if (newCons <= prodIdx) {
        uint32_t dw15 = atomicRead((uint32_t *)&cqe0->dw15.value);
        uint8_t opcode = (dw15 & PCCL_TROCE_CQE_OPCODE_MASK) >> PCCL_TROCE_CQE_OPCODE_SHIFT;
        if (opcode != PCCL_TROCE_CQE_REQ) {
          status = -1;
        } else {
          // A later CQE on this RC QP also retires all earlier WQEs. Process
          // their counter records before making the SQ slots reusable.
          if (qp->pending_counter != nullptr) {
            for (uint64_t w = oldCons; w < newCons; w++) {
              uint32_t slot = (uint32_t)(w & (qp->nwqes - 1));
              uint32_t pending = atomicExch(&qp->pending_counter[slot], 0u);
              if (pending != 0u) {
                atomicAdd((unsigned long long *)&qp->counters_buf[pending - 1u], 1ull);
              }
            }
          }
          membar();
          atomicExch((unsigned long long *)qp->cons_idx, (unsigned long long)newCons);
          status = 1;
        }
      }
    }
  }
  membar();
  atomicExch((unsigned long long *)qp->progress_lock, 0ull);
  return status;
}

PCCL_DEVICE_INLINE int trocePollCqAt(struct pcclGinGdakiTroceQp *qp, uint64_t idx, uint32_t *abortFlag) {
  using pccl::utility::testAbort;
  if (idx == uint64_t(-1)) return 0; // FlushAsync on an empty QP
  uint32_t steps = 0;
  #pragma unroll 1
  while (atomicRead(qp->cons_idx) <= idx) {
    if (troceTryProgressQp(qp) < 0) return -1;
    if (abortFlag && testAbort(abortFlag, steps)) return -1;
  }
  mfence();
  return 0;
}

// Wait for everything currently reserved on this QP.
PCCL_DEVICE_INLINE void troceQuiet(struct pcclGinGdakiTroceQp *qp, uint32_t *abortFlag) {
  uint64_t resv = atomicRead(qp->resv_head);
  if (resv == 0) return;
  trocePollCqAt(qp, resv - 1, abortFlag);
}

// ---------------------------------------------------------------------------
// put / signal
// ---------------------------------------------------------------------------

// An optional RDMA_WRITE followed by an optional remote signal ATOMIC_FA. A
// local counter action is associated with the final signaled WQE; the SNIC
// readCounter/waitCounter paths consume its CQE and increment the GPU counter.
PCCL_DEVICE_INLINE uint64_t putSignal(struct pcclGinGdakiTroceGPUContext *ctx,
                                      struct pcclGinGdakiTroceQp *qp, bool hasData, uint64_t dstOff,
                                      __be32 dstRkey, uint64_t srcOff, __be32 srcLkey,
                                      size_t bytes, bool hasSignal, uint64_t signalOff,
                                      __be32 signalRkey, uint64_t signalVal, bool hasCounter,
                                      uint32_t counterId, bool ringDb) {
  // A signal is an ATOMIC_FA that occupies two WQEBBs, and troce
  // reports the atomic's completion against the FIRST of those two WQEBBs. If an
  // atomic were the last WQE, the tail (resv_head-1) would point at the second
  // WQEBB and the CQE index would never reach it, so quiet() would hang.
  // sailSHMEM handles this by appending one extra WQEBB after the atomic -- a
  // zero-byte RDMA_WRITE that carries the completion -- and leaving the atomics
  // themselves without a CQE. The trailing single-WQEBB write then owns the tail
  // and its CQE index matches.
  const bool hasAtomic = hasSignal;
  uint32_t nWqe = (hasData ? 1u : 0u) + (hasSignal ? 2u : 0u) + (hasAtomic ? 1u : 0u);
  if (nWqe == 0) return 0;

  uint64_t base = reserveWqeSlots(qp, nWqe);
  uint64_t idx = base;

  if (hasData) {
    // When an atomic follows, the trailing zero-byte write owns the CQE, so the
    // data write requests none. For a data-only put the write is the tail and
    // must carry the completion itself.
    troceWriteRdmaWqe(qp, srcOff, srcLkey, dstOff, dstRkey, (uint32_t)bytes, (uint16_t)idx,
                      /*needCqe=*/!hasAtomic, getWqePtr(qp, idx), /*isWrite=*/true);
    idx++;
  }
  if (hasSignal) {
    // Two WQEBBs, no completion requested; the trailer below carries the CQE. The
    // fetched value goes to this QP's private sink slot (qp->sink_addr) so
    // concurrent atomics on other QPs never collide on the writeback.
    troceWriteAtomicFaWqe(qp, signalVal, qp->sink_addr, ctx->sink_buffer_lkey, signalOff,
                          signalRkey, (uint16_t)idx, /*needCqe=*/false, getWqePtr(qp, idx));
    idx += 2;
  }
  if (hasAtomic) {
    // Trailing zero-byte RDMA_WRITE: carries the CQE and occupies exactly one
    // WQEBB so the tail lands on a CQE-reporting index. Targets whichever remote
    // address we already hold a valid rkey for.
    troceWriteRdmaWqe(qp, (uint64_t)0, ctx->sink_buffer_lkey, signalOff, signalRkey, /*bytes=*/0u,
                      (uint16_t)idx, /*needCqe=*/true, getWqePtr(qp, idx), /*isWrite=*/true);
    idx++;
  }

  uint64_t last = idx - 1;
  // `last` is the WQE that requested the CQE, so that is where the poll will look
  // for a pending counter. Registered before the doorbell so the poll can never
  // observe the completion before the registration.
  if (hasCounter) {
    storeRelaxed(&qp->pending_counter[last & (qp->nwqes - 1)], counterId + 1u);
    mfence();
  }
  submit(qp, base, last, ringDb);
  return last;
}

}  // namespace gdaki_troce
}  // namespace gin
}  // namespace pccl

// ---------------------------------------------------------------------------
// GIN device API specializations
// ---------------------------------------------------------------------------

template <>
struct pcclGinGdakiApi_Put<PCCL_GIN_GDAKI_PROVIDER_TROCE> {
  template <typename Coop>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                      pcclGinWindow_t dstWin, size_t dstOff, pcclGinWindow_t srcWin,
                                      size_t srcOff, size_t bytes, pcclGinSignalDescriptor signal,
                                      pcclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                      bool hasCounter, pcclGinCounter_t counterId,
                                      bool hasDescriptor, pcclGinDescriptorSmem *descriptor,
                                      hggc::thread_scope required, hggc::thread_scope given,
                                      uint32_t optFlags) {
    using pccl::utility::loadConst;
    coop.sync();
    if (coop.thread_rank() == 0) {
      pcclGinGdakiTroceGPUContext *gda =
        &((struct pcclGinGdakiTroceGPUContext *)ctx.handle)[ctx.contextId];
      pcclGinGdakiTroceQp *qp = loadConst(&gda->qps) + peer;

      bool hasSignal = signal.type != PCCL_GIN_SIGNAL_TYPE_NONE;
      uint64_t signalOff = 0;
      __be32 signalRkey = 0;
      __be32 dstRkey = 0, srcLkey = 0;

      if (signal.type == PCCL_GIN_SIGNAL_TYPE_INDEXED) {
        signalOff = sizeof(uint64_t) *
                    (signal.indexedSignal.signalId + loadConst(&gda->signals_table.offset));
        signalRkey = loadConst(loadConst(&gda->signals_table.rkeys) + peer);
      } else if (signal.type == PCCL_GIN_SIGNAL_TYPE_VA) {
        pcclGinGdakiTroceMemHandle *mh = (pcclGinGdakiTroceMemHandle *)signal.vaSignal.signalWindow;
        signalRkey = loadConst(loadConst(&mh->rkeys) + peer);
        signalOff = signal.vaSignal.signalOffset;
      }
      uint64_t signalVal = (signalOp == pcclGinSignalInc) ? 1 : signalOpArg;

      if (hasWins) {
        pcclGinGdakiTroceMemHandle *dstMh = (pcclGinGdakiTroceMemHandle *)dstWin;
        pcclGinGdakiTroceMemHandle *srcMh = (pcclGinGdakiTroceMemHandle *)srcWin;
        dstRkey = loadConst(loadConst(&dstMh->rkeys) + peer);
        srcLkey = loadConst(&srcMh->lkey);
      }

      // Make prior GPU writes visible to the NIC before the WQE is posted.
      pccl::gin::gdaki_troce::membar();
      // A zero-byte put with CounterInc still needs a completion WQE.
      pccl::gin::gdaki_troce::putSignal(gda, qp, hasWins && (bytes > 0 || (hasCounter && !hasSignal)),
                                  dstOff, dstRkey, srcOff, srcLkey,
                                  bytes, hasSignal, signalOff, signalRkey, signalVal, hasCounter,
                                  (uint32_t)counterId,
                                  /*ringDb=*/!(optFlags & pcclGinOptFlagsAggregateRequests));
    }
    coop.sync();
    (void)hasDescriptor;
    (void)descriptor;
    (void)required;
    (void)given;
  }
};

template <>
struct pcclGinGdakiApi_Get<PCCL_GIN_GDAKI_PROVIDER_TROCE> {
  template <typename Coop>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, Coop coop, int peer,
                                      pcclGinWindow_t remoteWin, size_t remoteOff,
                                      pcclGinWindow_t localWin, size_t localOff, size_t bytes,
                                      bool hasDescriptor, pcclGinDescriptorSmem *descriptor,
                                      uint32_t optFlags = pcclGinOptFlagsDefault) {
    using pccl::utility::loadConst;
    coop.sync();
    if (coop.thread_rank() == 0) {
      pcclGinGdakiTroceGPUContext *gda =
        &((struct pcclGinGdakiTroceGPUContext *)ctx.handle)[ctx.contextId];
      pcclGinGdakiTroceQp *qp = loadConst(&gda->qps) + peer;
      pcclGinGdakiTroceMemHandle *remoteMh = (pcclGinGdakiTroceMemHandle *)remoteWin;
      pcclGinGdakiTroceMemHandle *localMh = (pcclGinGdakiTroceMemHandle *)localWin;

      uint64_t base = pccl::gin::gdaki_troce::reserveWqeSlots(qp, 1);
      pccl::gin::gdaki_troce::troceWriteRdmaWqe(
        qp, localOff, loadConst(&localMh->lkey), remoteOff,
        loadConst(loadConst(&remoteMh->rkeys) + peer), (uint32_t)bytes, (uint16_t)base,
        /*needCqe=*/true, pccl::gin::gdaki_troce::getWqePtr(qp, base), /*isWrite=*/false);
      pccl::gin::gdaki_troce::submit(qp, base, base,
                               /*ringDb=*/!(optFlags & pcclGinOptFlagsAggregateRequests));
    }
    coop.sync();
  }
};

template <>
struct pcclGinGdakiApi_Flush<PCCL_GIN_GDAKI_PROVIDER_TROCE> {
  template <typename Coop>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, Coop coop, hggc::memory_order ord,
                                      uint32_t *abortFlag) {
    (void)ord;
    using pccl::utility::loadConst;
    // Peers have separate CQs and progress locks, so different lanes can drain
    // their QPs independently. The single-slot CQ needs no CQ doorbell.
    pcclGinGdakiTroceGPUContext *gda = &((struct pcclGinGdakiTroceGPUContext *)ctx.handle)[ctx.contextId];
    pcclGinGdakiTroceQp *qps = loadConst(&gda->qps);
    #pragma unroll 1
    for (int peer = coop.thread_rank(); peer < ctx.nRanks; peer += coop.size()) {
      pccl::gin::gdaki_troce::troceQuiet(qps + peer, abortFlag);
    }
  }
};

template <>
struct pcclGinGdakiApi_FlushAsync<PCCL_GIN_GDAKI_PROVIDER_TROCE> {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, int peer, pcclGinRequest_t *outRequest,
                                      uint32_t optFlags) {
    using pccl::utility::loadConst;
    pcclGinGdakiTroceRequest *req = reinterpret_cast<pcclGinGdakiTroceRequest *>(outRequest);
    pcclGinGdakiTroceGPUContext *gda = &((struct pcclGinGdakiTroceGPUContext *)ctx.handle)[ctx.contextId];
    pcclGinGdakiTroceQp *qp = loadConst(&gda->qps) + peer;
    req->peer = peer;
    uint64_t resv = pccl::gin::gdaki_troce::atomicRead(qp->resv_head);
    req->wqeIdx = resv ? resv - 1 : uint64_t(-1);
  }
};

template <>
struct pcclGinGdakiApi_Wait<PCCL_GIN_GDAKI_PROVIDER_TROCE> {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, pcclGinRequest_t &request, bool hasDescriptor,
                                      pcclGinDescriptorSmem *descriptor, hggc::memory_order ord,
                                      uint32_t *abortFlag) {
    (void)ord;
    using pccl::utility::loadConst;
    pcclGinGdakiTroceRequest &req = reinterpret_cast<pcclGinGdakiTroceRequest &>(request);
    pcclGinGdakiTroceGPUContext *gda = &((struct pcclGinGdakiTroceGPUContext *)ctx.handle)[ctx.contextId];
    pcclGinGdakiTroceQp *qp = loadConst(&gda->qps) + req.peer;
    pccl::gin::gdaki_troce::trocePollCqAt(qp, req.wqeIdx, abortFlag);
    (void)hasDescriptor;
    (void)descriptor;
  }
};

template <>
struct pcclGinGdakiApi_GetSignalPtr<PCCL_GIN_GDAKI_PROVIDER_TROCE> {
  PCCL_DEVICE_INLINE static uint64_t *call(pcclGinCtx ctx, pcclGinSignal_t signalId) {
    using pccl::utility::loadConst;
    pcclGinGdakiTroceGPUContext *gda = &((struct pcclGinGdakiTroceGPUContext *)ctx.handle)[ctx.contextId];
    return loadConst(&gda->signals_table.buffer) + signalId;
  }
};

template <>
struct pcclGinGdakiApi_ResetSignal<PCCL_GIN_GDAKI_PROVIDER_TROCE> {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, pcclGinSignalDescriptor signal) {
    using pccl::utility::loadConst;
    if (signal.type == PCCL_GIN_SIGNAL_TYPE_VA) {
      uint64_t *p =
        (uint64_t *)pcclGetLocalPointer(signal.vaSignal.pcclWindow, signal.vaSignal.signalOffset);
      *p = 0;
    } else {
      pcclGinGdakiTroceGPUContext *gda = &((struct pcclGinGdakiTroceGPUContext *)ctx.handle)[ctx.contextId];
      loadConst(&gda->signals_table.buffer)[signal.indexedSignal.signalId] = 0;
    }
  }
};

template <>
struct pcclGinGdakiApi_GetCounterPtr<PCCL_GIN_GDAKI_PROVIDER_TROCE> {
  PCCL_DEVICE_INLINE static uint64_t *call(pcclGinCtx ctx, pcclGinCounter_t counterId) {
    using pccl::utility::loadConst;
    pcclGinGdakiTroceGPUContext *gda = &((struct pcclGinGdakiTroceGPUContext *)ctx.handle)[ctx.contextId];
    return loadConst(&gda->counters_table.buffer) + counterId;
  }
};

namespace pccl {
namespace gin {
namespace gdaki_troce {

// A counter may collect completions from several peers, so one observation
// probes every QP of this context without blocking on any individual QP.
PCCL_DEVICE_INLINE int troceProgressCounters(pcclGinCtx ctx) {
  using pccl::utility::loadConst;
  pcclGinGdakiTroceGPUContext *gda =
    &((struct pcclGinGdakiTroceGPUContext *)ctx.handle)[ctx.contextId];
  pcclGinGdakiTroceQp *qps = loadConst(&gda->qps);
  int status = 0;
  for (int peer = 0; peer < ctx.nRanks; peer++) {
    if (troceTryProgressQp(qps + peer) < 0) status = -1;
  }
  return status;
}

} // namespace gdaki_troce
} // namespace gin
} // namespace pccl

template <>
struct pcclGinGdakiApi_ReadCounter<PCCL_GIN_GDAKI_PROVIDER_TROCE> {
  PCCL_DEVICE_INLINE static uint64_t call(pcclGinCtx ctx, pcclGinCounter_t counter,
                                          int bits, hggc::memory_order ord) {
    pccl::gin::gdaki_troce::troceProgressCounters(ctx);
    uint64_t *ptr = pcclGinGdakiApi_GetCounterPtr<PCCL_GIN_GDAKI_PROVIDER_TROCE>::call(ctx, counter);
    uint64_t mask = uint64_t(-1) >> (64 - bits);
    return mask & hggc::atomic_ref<uint64_t>{*ptr}.load(ord);
  }
};

template <>
struct pcclGinGdakiApi_WaitCounter<PCCL_GIN_GDAKI_PROVIDER_TROCE> {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, pcclGinCounter_t counter,
                                      uint64_t least, int bits, hggc::memory_order ord,
                                      uint32_t *abortFlag) {
    using pccl::utility::testAbort;
    uint32_t steps = 0;
    uint64_t *ptr = pcclGinGdakiApi_GetCounterPtr<PCCL_GIN_GDAKI_PROVIDER_TROCE>::call(ctx, counter);
    uint64_t got;
    #pragma unroll 1
    do {
      pccl::gin::gdaki_troce::troceProgressCounters(ctx);
      got = hggc::atomic_ref<uint64_t>{*ptr}.load(ord);
    } while (!pccl::utility::rollingLessEq(least, got, bits) && !testAbort(abortFlag, steps));
  }
};

template <>
struct pcclGinGdakiApi_ResetCounter<PCCL_GIN_GDAKI_PROVIDER_TROCE> {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, pcclGinCounter_t counterId) {
    using pccl::utility::loadConst;
    pcclGinGdakiTroceGPUContext *gda = &((struct pcclGinGdakiTroceGPUContext *)ctx.handle)[ctx.contextId];
    loadConst(&gda->counters_table.buffer)[counterId] = 0;
  }
};

template <>
struct pcclGinGdakiApi_PutValue<PCCL_GIN_GDAKI_PROVIDER_TROCE> {
  template <typename Coop, typename T>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, Coop coop, int peer, pcclGinWindow_t dstWin,
                                      size_t dstOff, T srcVal, pcclGinSignalDescriptor signal,
                                      pcclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                      bool hasDescriptor, pcclGinDescriptorSmem *descriptor,
                                      hggc::thread_scope required, hggc::thread_scope given,
                                      uint32_t optFlags = pcclGinOptFlagsDefault) {
    using pccl::utility::loadConst;
    static_assert(sizeof(T) <= 8, "PutValue payload must fit in 8 bytes");
    coop.sync();
    if (coop.thread_rank() == 0) {
      pcclGinGdakiTroceGPUContext *gda =
        &((struct pcclGinGdakiTroceGPUContext *)ctx.handle)[ctx.contextId];
      pcclGinGdakiTroceQp *qp = loadConst(&gda->qps) + peer;

      bool hasSignal = signal.type != PCCL_GIN_SIGNAL_TYPE_NONE;
      uint64_t signalOff = 0;
      __be32 signalRkey = 0;
      if (signal.type == PCCL_GIN_SIGNAL_TYPE_INDEXED) {
        signalOff = sizeof(uint64_t) *
                    (signal.indexedSignal.signalId + loadConst(&gda->signals_table.offset));
        signalRkey = loadConst(loadConst(&gda->signals_table.rkeys) + peer);
      } else if (signal.type == PCCL_GIN_SIGNAL_TYPE_VA) {
        pcclGinGdakiTroceMemHandle *mh = (pcclGinGdakiTroceMemHandle *)signal.vaSignal.signalWindow;
        signalRkey = loadConst(loadConst(&mh->rkeys) + peer);
        signalOff = signal.vaSignal.signalOffset;
      }
      uint64_t signalVal = (signalOp == pcclGinSignalInc) ? 1 : signalOpArg;

      pcclGinGdakiTroceMemHandle *dstMh = (pcclGinGdakiTroceMemHandle *)dstWin;
      __be32 dstRkey = loadConst(loadConst(&dstMh->rkeys) + peer);

      pccl::gin::gdaki_troce::membar();

      // The value goes out as an inline write, then the optional signal atomic
      // plus its CQE-carrying trailer, exactly as in the regular put path.
      const uint32_t nWqe = 1u + (hasSignal ? 3u : 0u);
      uint64_t base = pccl::gin::gdaki_troce::reserveWqeSlots(qp, nWqe);
      uint64_t idx = base;

      pccl::gin::gdaki_troce::troceWriteRdmaInlineWqe(qp, &srcVal, (uint32_t)sizeof(T), dstOff, dstRkey,
                                                (uint16_t)idx, /*needCqe=*/!hasSignal,
                                                pccl::gin::gdaki_troce::getWqePtr(qp, idx));
      idx++;
      if (hasSignal) {
        pccl::gin::gdaki_troce::troceWriteAtomicFaWqe(qp, signalVal, qp->sink_addr,
                                                loadConst(&gda->sink_buffer_lkey), signalOff, signalRkey,
                                                (uint16_t)idx, /*needCqe=*/false,
                                                pccl::gin::gdaki_troce::getWqePtr(qp, idx));
        idx += 2;
        pccl::gin::gdaki_troce::troceWriteRdmaWqe(qp, (uint64_t)0, loadConst(&gda->sink_buffer_lkey), signalOff,
                                            signalRkey, /*bytes=*/0u, (uint16_t)idx,
                                            /*needCqe=*/true, pccl::gin::gdaki_troce::getWqePtr(qp, idx),
                                            /*isWrite=*/true);
        idx++;
      }
      pccl::gin::gdaki_troce::submit(qp, base, idx - 1,
                               /*ringDb=*/!(optFlags & pcclGinOptFlagsAggregateRequests));
    }
    coop.sync();
    (void)hasDescriptor;
    (void)descriptor;
    (void)required;
    (void)given;
  }
};

#endif /* _PCCL_DEVICE_GIN_GDAKI_TROCE_H_ */
