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

// Device-side GIN interface for the TROCE backend.

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

// Memory access helpers used by the TROCE submission path.
PCCL_DEVICE_INLINE void mfence() {
  __threadfence_block();
}
// Device-scope publication fence used by this backend.
PCCL_DEVICE_INLINE void membar() {
  __threadfence();
}

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

// The abort flag resides in mapped host memory, never NIC MMIO.
PCCL_DEVICE_INLINE void troceRequestAbort(uint32_t *abortFlag) {
  if (abortFlag != nullptr) {
    hggc::atomic_ref<uint32_t, hggc::thread_scope_system>{*abortFlag}.store(1u, hggc::memory_order_release);
  }
}

PCCL_DEVICE_INLINE bool troceStopRequested(uint32_t *abortFlag) {
  return abortFlag != nullptr &&
    hggc::atomic_ref<uint32_t, hggc::thread_scope_system>{*abortFlag}.load(hggc::memory_order_relaxed) != 0;
}

PCCL_DEVICE_INLINE int troceTryProgressQp(struct pcclGinGdakiTroceQp *qp, uint32_t *abortFlag);

// Reserve n WQEBBs. Cancellation or error invalidates the reservation and
// stops the GIN instance; callers must not fill or publish a failed reservation.
PCCL_DEVICE_INLINE bool reserveWqeSlots(struct pcclGinGdakiTroceQp *qp, uint32_t n,
                                        uint32_t *abortFlag, uint64_t *outBase) {
  if (troceStopRequested(abortFlag)) {
    return false;
  }
  uint64_t base = atomicAdd((unsigned long long *)qp->resv_head, (unsigned long long)n);
  while (atomicRead(qp->cons_idx) + qp->nwqes < base + n) {
    // Progress completions while waiting for SQ capacity; stop on error or abort.
    if (troceStopRequested(abortFlag) || troceTryProgressQp(qp, abortFlag) < 0) {
      return false;
    }
  }
  if (troceStopRequested(abortFlag)) {
    return false;
  }
  *outBase = base;
  return true;
}

PCCL_DEVICE_INLINE void *getWqePtr(struct pcclGinGdakiTroceQp *qp, uint64_t wqeIdx) {
  uint16_t idx = (uint16_t)(wqeIdx & (qp->nwqes - 1));
  return (void *)((uintptr_t)qp->sq_wqe + ((uintptr_t)idx << 6));
}

// ---------------------------------------------------------------------------
// troce doorbell
// ---------------------------------------------------------------------------

// Only explicit submission paths may call this helper. Callers must hold
// qp->db_lock (see troceRingReady).
PCCL_DEVICE_INLINE void troceRingDb(struct pcclGinGdakiTroceQp *qp, uint16_t prodIdx) {
  pcclTroceSqDb db;
  db.value = 0;
  db.db.dw0.bs.qid = qp->qpn;
  db.db.dw0.bs.q_type = PCCL_PAU_QTYPE_ROCE_SQ;
  db.db.dw1.bs.pi = prodIdx;
  // Use the QP traffic class queried after RTR.
  db.db.dw1.bs.tc = qp->traffic_class;
  db.db.dw0.value = PCCL_TROCE_HTOBE32(db.db.dw0.value);
  db.db.dw1.value = PCCL_TROCE_HTOBE32(db.db.dw1.value);
  storeRelease((uint64_t *)qp->sq_db, db.value);
}

// Explicitly notify a bounded WQEBB tail, never a later ready_head snapshot.
// Ordinary submit and flushAsync are the only callers. A flush may include
// reservations waiting for SQ credits, so announce published prefixes before
// waiting for the entire target. CQ progress itself never submits requests.
//
// Call through troceRingWarpGrouped() to serialize ownership attempts among
// participating lanes. Only the selected lane enters the QP notification path.
// Only the doorbell encoding truncates target to 16 bits; logical comparisons
// remain 64-bit across PI wraparound. Returns false on cancellation/error.
PCCL_DEVICE_INLINE bool troceRingReady(struct pcclGinGdakiTroceQp *qp, uint64_t target,
                                       uint32_t *abortFlag) {
  unsigned long long *lock = (unsigned long long *)qp->db_lock;
  while (true) {
    if (troceStopRequested(abortFlag)) {
      return false;
    }
    while (atomicCAS(lock, 0ull, 1ull) != 0ull) {
      if (troceStopRequested(abortFlag)) {
        return false;
      }
    }
    // Order ready_head and prod_idx reads after acquiring doorbell ownership.
    membar();
    if (troceStopRequested(abortFlag)) {
      atomicExch(lock, 0ull);
      return false;
    }
    const uint64_t ready = atomicRead(qp->ready_head);
    const uint64_t prefix = ready < target ? ready : target;
    const uint64_t notified = atomicRead(qp->prod_idx);
    bool covered = notified >= target;
    if (prefix > notified) {
      troceRingDb(qp, (uint16_t)prefix);
      // Keep MMIO submission and the published tail in one critical section.
      membar();
      atomicExch((unsigned long long *)qp->prod_idx, (unsigned long long)prefix);
      covered = prefix == target;
    }
    membar();
    atomicExch(lock, 0ull);
    if (covered) {
      return true;
    }
    // Release ownership before progressing completions and waiting for more
    // publishers. This lets a credit-blocked reservation reach ready_head.
    if (troceTryProgressQp(qp, abortFlag) < 0) {
      return false;
    }
  }
}

// Notify `target` on behalf of every lane of the warp that entered this helper.
// Lanes are grouped by QP; each group's highest lane notifies its largest
// bounded target when all group reservations have been published.
// Grouping is only used when every target of the group is already published;
// otherwise each lane notifies its own target, one lane at a time, so an earlier
// flush never waits for a later lane's reservations. Members return the result
// of their representative's notification. All participating lanes must call
// this helper convergently; lanes that did not enter it are not synchronized.
PCCL_DEVICE_INLINE bool troceRingWarpGrouped(struct pcclGinGdakiTroceQp *qp, uint64_t target,
                                            uint32_t *abortFlag) {
  const unsigned active = __activemask();
  if ((active & (active - 1)) == 0) {
    return troceRingReady(qp, target, abortFlag);
  }
  const int me = pccl::utility::lane();
  const uint64_t key = (uint64_t)(uintptr_t)qp;
  const unsigned published = __ballot_sync(active, atomicRead(qp->ready_head) >= target);
  unsigned members = 0;
  uint64_t groupTarget = target;
  for (unsigned scan = active; scan != 0; scan &= scan - 1) {
    const int src = __ffs(scan) - 1;
    const uint64_t peerKey = (uint64_t)__shfl_sync(active, (uint32_t)key, src) |
                             ((uint64_t)__shfl_sync(active, (uint32_t)(key >> 32), src) << 32);
    const uint64_t peerTarget = (uint64_t)__shfl_sync(active, (uint32_t)target, src) |
                                ((uint64_t)__shfl_sync(active, (uint32_t)(target >> 32), src) << 32);
    if (peerKey == key) {
      members |= 1u << src;
      if (peerTarget > groupTarget) groupTarget = peerTarget;
    }
  }
  const bool grouped = (members & published) == members;
  const int leader = grouped ? 31 - __clz(members) : me;
  const unsigned leaders = __ballot_sync(active, me == leader);
  bool notified = false;
  for (unsigned pending = leaders; pending != 0; pending &= pending - 1) {
    const int src = __ffs(pending) - 1;
    int result = 0;
    if (me == src) {
      result = troceRingReady(qp, grouped ? groupTarget : target, abortFlag) ? 1 : 0;
    }
    result = __shfl_sync(active, result, src);
    if (leader == src) notified = result != 0;
  }
  return notified;
}

// Publish the reserved range and ring the doorbell. `endWqeIdx` is inclusive.
//
// Make WQEs visible at device scope before advancing ready_head, since another
// warp or CTA may submit them while holding doorbell ownership.
// `ringDb == false` implements pcclGinOptFlagsAggregateRequests: ready_head still
// advances but the doorbell is deferred, so prod_idx must stay behind for a later
// non-aggregate submit or explicit flushAsync to announce the accumulated batch.
// The batch must be closed before it exhausts SQ credits; CQ progress, synchronous
// flush, wait and counter operations do not submit requests.
PCCL_DEVICE_INLINE bool submit(struct pcclGinGdakiTroceQp *qp, uint64_t baseWqeIdx, uint64_t endWqeIdx,
                               bool ringDb, uint32_t *abortFlag) {
  if (troceStopRequested(abortFlag)) {
    return false;
  }
  uint64_t newProd = endWqeIdx + 1;
  // Atomically advance the contiguous published tail so concurrent producers
  // cannot overwrite each other's updates.
  membar();
  while (atomicCAS((unsigned long long *)qp->ready_head, (unsigned long long)baseWqeIdx,
                   (unsigned long long)newProd) != (unsigned long long)baseWqeIdx) {
    if (troceStopRequested(abortFlag)) {
      return false;
    }
  }
  mfence();

  if (!ringDb) {
    return !troceStopRequested(abortFlag);
  }

  // A normal submission covers all earlier published requests on this QP,
  // without extending its notification to later concurrent aggregates.
  return troceRingWarpGrouped(qp, newProd, abortFlag);
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
  if (bytes == 0) {
    ctrl.dw1.bs.in_dat = 1;
  }
  ctrl.dw1.value = PCCL_TROCE_HTOBE32(ctrl.dw1.value);

  ctrl.dw2.bs.local_sge_num = bytes > 0 ? 1 : 0;
  ctrl.dw2.value = PCCL_TROCE_HTOBE32(ctrl.dw2.value);
  ctrl.pld_len = PCCL_TROCE_HTOBE32(bytes);

  uint32_t *dst = (uint32_t *)ctrlPtr;
  uint32_t *src = (uint32_t *)&ctrl;
  for (unsigned i = 0; i < sizeof(*ctrlPtr) / sizeof(uint32_t); ++i) {
    storeRelaxed(&dst[i], src[i]);
  }
  dst = (uint32_t *)raddrPtr;
  src = (uint32_t *)&raddrSeg;
  for (unsigned i = 0; i < sizeof(*raddrPtr) / sizeof(uint32_t); ++i) {
    storeRelaxed(&dst[i], src[i]);
  }
  dst = (uint32_t *)dataPtr;
  src = (uint32_t *)&dataSeg;
  for (unsigned i = 0; i < sizeof(*dataPtr) / sizeof(uint32_t); ++i) {
    storeRelaxed(&dst[i], src[i]);
  }
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
  for (unsigned i = 0; i < sizeof(*ctrlPtr) / sizeof(uint32_t); ++i) {
    storeRelaxed(&dst[i], src[i]);
  }
  dst = (uint32_t *)raddrPtr;
  src = (uint32_t *)&raddrSeg;
  for (unsigned i = 0; i < sizeof(*raddrPtr) / sizeof(uint32_t); ++i) {
    storeRelaxed(&dst[i], src[i]);
  }

  // Zero the padded payload area first so the tail bytes are deterministic.
  for (uint32_t o = 0; o < (uint32_t)(total - hdrSize); o += sizeof(uint32_t)) {
    storeRelaxed((uint32_t *)((uintptr_t)dataPtr + o), (uint32_t)0);
  }

  switch (bytes) {
    case 1:
    storeRelaxed((uint8_t *)dataPtr, *((const uint8_t *)val)); break;
    case 2:
    storeRelaxed((uint16_t *)dataPtr, *((const uint16_t *)val)); break;
    case 4:
    storeRelaxed((uint32_t *)dataPtr, *((const uint32_t *)val)); break;
    case 8:
      storeRelaxed(&(((uint32_t *)dataPtr)[0]), ((const uint32_t *)val)[0]);
      storeRelaxed(&(((uint32_t *)dataPtr)[1]), ((const uint32_t *)val)[1]);
      break;
    default:
      for (uint32_t o = 0; o < bytes; ++o) {
      storeRelaxed((uint8_t *)((uintptr_t)dataPtr + o), ((const uint8_t *)val)[o]);
    }
    break;
  }
}

// Encode an 8-byte remote fetch-and-add. The returned value is written to the
// QP's atomic return storage and is not consumed.
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
  for (unsigned i = 0; i < sizeof(*ctrlPtr) / sizeof(uint32_t); ++i) {
    storeRelaxed(&dst[i], src[i]);
  }
  dst = (uint32_t *)raddrPtr;
  src = (uint32_t *)&raddrSeg;
  for (unsigned i = 0; i < sizeof(*raddrPtr) / sizeof(uint32_t); ++i) {
    storeRelaxed(&dst[i], src[i]);
  }
  dst = (uint32_t *)atomicPtr;
  src = (uint32_t *)&atomicSeg;
  for (unsigned i = 0; i < sizeof(*atomicPtr) / sizeof(uint32_t); ++i) {
    storeRelaxed(&dst[i], src[i]);
  }
  dst = (uint32_t *)dataPtr;
  src = (uint32_t *)&dataSeg;
  for (unsigned i = 0; i < sizeof(*dataPtr) / sizeof(uint32_t); ++i) {
    storeRelaxed(&dst[i], src[i]);
  }
}

// ---------------------------------------------------------------------------
// troce CQ progress. The single CQ slot contains the newest completed WQE.
// ---------------------------------------------------------------------------

// Progress does not wait for a CQE. The same QP may be progressed by a counter
// reader, a waiter, a flush, or a producer whose SQ is full. Only one of them
// may consume pending_counter[] at a time, and SQ slots cannot be recycled until
// all counters covered by the completion have been updated. This is completion
// progress only: it must never ring a doorbell or publish pending requests.
PCCL_DEVICE_INLINE int troceTryProgressQp(struct pcclGinGdakiTroceQp *qp, uint32_t *abortFlag) {
  if (troceStopRequested(abortFlag)) {
    return -1;
  }
  if (atomicCAS((unsigned long long *)qp->progress_lock, 0ull, 1ull) != 0ull) {
    return 0;
  }

  if (troceStopRequested(abortFlag)) {
    atomicExch((unsigned long long *)qp->progress_lock, 0ull);
    return -1;
  }
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
        if (opcode != PCCL_TROCE_CQE_REQ || (dw14 & 0xffu) != 0) {
          troceRequestAbort(abortFlag);
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
  return troceStopRequested(abortFlag) ? -1 : status;
}

PCCL_DEVICE_INLINE int trocePollCqAt(struct pcclGinGdakiTroceQp *qp, uint64_t idx, uint32_t *abortFlag) {
  if (troceStopRequested(abortFlag)) {
    return -1;
  }
  if (idx == uint64_t(-1)) {
    return 0; // FlushAsync on an empty QP
  }
  #pragma unroll 1
  while (atomicRead(qp->cons_idx) <= idx) {
    if (troceTryProgressQp(qp, abortFlag) < 0) {
      return -1;
    }
    if (troceStopRequested(abortFlag)) {
      return -1;
    }
  }
  mfence();
  return troceStopRequested(abortFlag) ? -1 : 0;
}

// Wait for everything currently reserved on this QP, preserving error status.
// This does not submit: the caller must close aggregates with a normal submit or
// flushAsync before waiting. Leaving an aggregate unsubmitted requires abort or
// another explicit submit to make this wait finish.
PCCL_DEVICE_INLINE int troceQuiet(struct pcclGinGdakiTroceQp *qp, uint32_t *abortFlag) {
  uint64_t resv = atomicRead(qp->resv_head);
  return trocePollCqAt(qp, resv ? resv - 1 : uint64_t(-1), abortFlag);
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
                                      uint32_t counterId, bool ringDb, uint32_t *abortFlag) {
  // Signal batches end with a CQE-reporting zero-byte RDMA_WRITE. The final
  // single-WQEBB write owns the batch's completion ticket.
  const bool hasAtomic = hasSignal;
  uint32_t nWqe = (hasData ? 1u : 0u) + (hasSignal ? 2u : 0u) + (hasAtomic ? 1u : 0u);
  if (nWqe == 0) {
    return 0;
  }

  uint64_t base;
  if (!reserveWqeSlots(qp, nWqe, abortFlag, &base)) {
    return uint64_t(-1);
  }
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
    // The trailing zero-byte RDMA_WRITE occupies one WQEBB and requests the
    // batch completion. Reuse the signal MR's valid remote address and key.
    troceWriteRdmaWqe(qp, (uint64_t)0, ctx->sink_buffer_lkey, signalOff, signalRkey, /*bytes=*/0u,
                      (uint16_t)idx, /*needCqe=*/true, getWqePtr(qp, idx), /*isWrite=*/true);
    idx++;
  }

  uint64_t last = idx - 1;
  // Associate the counter with the completion-requesting WQE before submission,
  // so CQ progress can retire the counter record with that completion.
  if (hasCounter) {
    storeRelaxed(&qp->pending_counter[last & (qp->nwqes - 1)], counterId + 1u);
    mfence();
  }
  return submit(qp, base, last, ringDb, abortFlag) ? last : uint64_t(-1);
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
      if (pccl::gin::gdaki_troce::putSignal(gda, qp, hasWins && (bytes > 0 || (hasCounter && !hasSignal)),
                                            dstOff, dstRkey, srcOff, srcLkey,
                                            bytes, hasSignal, signalOff, signalRkey, signalVal, hasCounter,
                                            (uint32_t)counterId,
                                            /*ringDb=*/!(optFlags & pcclGinOptFlagsAggregateRequests),
                                            ctx.abortFlag) == uint64_t(-1)) {
        pccl::gin::gdaki_troce::troceRequestAbort(ctx.abortFlag);
      }
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

      uint64_t base;
      if (pccl::gin::gdaki_troce::reserveWqeSlots(qp, 1, ctx.abortFlag, &base)) {
        pccl::gin::gdaki_troce::troceWriteRdmaWqe(
          qp, localOff, loadConst(&localMh->lkey), remoteOff,
          loadConst(loadConst(&remoteMh->rkeys) + peer), (uint32_t)bytes, (uint16_t)base,
          /*needCqe=*/true, pccl::gin::gdaki_troce::getWqePtr(qp, base), /*isWrite=*/false);
        if (!pccl::gin::gdaki_troce::submit(qp, base, base,
              /*ringDb=*/!(optFlags & pcclGinOptFlagsAggregateRequests), ctx.abortFlag)) {
          pccl::gin::gdaki_troce::troceRequestAbort(ctx.abortFlag);
        }
      }
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
      if (pccl::gin::gdaki_troce::troceQuiet(qps + peer, abortFlag) < 0) {
        pccl::gin::gdaki_troce::troceRequestAbort(abortFlag);
        break;
      }
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
    req->wqeIdx = uint64_t(-1);
    if (pccl::gin::gdaki_troce::troceStopRequested(ctx.abortFlag)) {
      return;
    }
    // Snapshot the reserved tail so the ticket covers earlier requests without
    // depending on later reservations. Operations end with a completion-requesting
    // WQEBB, so no additional flush WQE is needed. Submit published prefixes while
    // earlier reservations wait for SQ capacity.
    const uint64_t target = pccl::gin::gdaki_troce::atomicRead(qp->resv_head);
    if (target == 0) {
      return;  // empty QP: no doorbell, wait(-1) is a no-op
    }
    bool notified = pccl::gin::gdaki_troce::troceRingWarpGrouped(qp, target, ctx.abortFlag);
    if (!notified) {
      pccl::gin::gdaki_troce::troceRequestAbort(ctx.abortFlag);
      return;  // wait observes the error/abort before interpreting the empty ticket
    }
    req->wqeIdx = target - 1;
    (void)optFlags;  // flushAsync always submits, including AggregateRequests
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
    if (pccl::gin::gdaki_troce::trocePollCqAt(qp, req.wqeIdx, abortFlag) < 0) {
      pccl::gin::gdaki_troce::troceRequestAbort(abortFlag);
    }
    (void)hasDescriptor;
    (void)descriptor;
  }
};

template <>
struct pcclGinGdakiApi_GetSignalPtr<PCCL_GIN_GDAKI_PROVIDER_TROCE> {
  PCCL_DEVICE_INLINE static uint64_t *call(pcclGinCtx ctx, int peer, pcclGinSignal_t signalId) {
    (void)peer;
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
  PCCL_DEVICE_INLINE static uint64_t *call(pcclGinCtx ctx, int peer, pcclGinCounter_t counterId) {
    (void)peer;
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
  for (int peer = 0; peer < ctx.nRanks; peer++) {
    if (troceTryProgressQp(qps + peer, ctx.abortFlag) < 0) {
      return -1;
    }
  }
  return 0;
}

} // namespace gdaki_troce
} // namespace gin
} // namespace pccl

template <>
struct pcclGinGdakiApi_ReadCounter<PCCL_GIN_GDAKI_PROVIDER_TROCE> {
  PCCL_DEVICE_INLINE static uint64_t call(pcclGinCtx ctx, pcclGinCounter_t counter,
                                          int bits, hggc::memory_order ord) {
    if (pccl::gin::gdaki_troce::troceProgressCounters(ctx) < 0) {
      pccl::gin::gdaki_troce::troceRequestAbort(ctx.abortFlag);
      return 0;  // no completion is fabricated; the async-error/abort channel is set
    }
    uint64_t *ptr = pcclGinGdakiApi_GetCounterPtr<PCCL_GIN_GDAKI_PROVIDER_TROCE>::call(ctx, ctx.rank, counter);
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
    ctx.abortFlag = abortFlag;
    uint32_t steps = 0;
    uint64_t *ptr = pcclGinGdakiApi_GetCounterPtr<PCCL_GIN_GDAKI_PROVIDER_TROCE>::call(ctx, ctx.rank, counter);
    uint64_t got;
    #pragma unroll 1
    do {
      if (pccl::gin::gdaki_troce::troceProgressCounters(ctx) < 0) {
        pccl::gin::gdaki_troce::troceRequestAbort(abortFlag);
        return;
      }
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

      // Encode the value inline. An optional signal atomic is followed by a
      // zero-byte write that requests the batch completion.
      const uint32_t nWqe = 1u + (hasSignal ? 3u : 0u);
      uint64_t base;
      if (pccl::gin::gdaki_troce::reserveWqeSlots(qp, nWqe, ctx.abortFlag, &base)) {
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
        if (!pccl::gin::gdaki_troce::submit(qp, base, idx - 1,
              /*ringDb=*/!(optFlags & pcclGinOptFlagsAggregateRequests), ctx.abortFlag)) {
          pccl::gin::gdaki_troce::troceRequestAbort(ctx.abortFlag);
        }
      }
    }
    coop.sync();
    (void)hasDescriptor;
    (void)descriptor;
    (void)required;
    (void)given;
  }
};

#endif /* _PCCL_DEVICE_GIN_GDAKI_TROCE_H_ */
