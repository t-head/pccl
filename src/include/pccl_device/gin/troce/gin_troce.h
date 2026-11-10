/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_GIN_TROCE_H_
#define _PCCL_DEVICE_GIN_TROCE_H_

#include <cstdint>
#include <errno.h>

#ifdef PCCL_DEVICE_GIN_TROCE_ENABLE_DEBUG
#include <stdio.h>
#endif

#include "../gin_device_common.h"
#include "gin_troce_device_host_common.h"
#include "gin_troce_hw.h"
#include "gin_troce_utils.h"

// Device-side GIN interface for the TROCE backend.

struct pcclGinTroceRequest {
  int peer;
  uint64_t wqeIdx;  // ticket: the WQE index to wait for on the send CQ
};
static_assert(sizeof(pcclGinTroceRequest) <= sizeof(pcclGinRequest_t),
              "pcclGinTroceRequest must fit in pcclGinRequest_t");

namespace pccl {
namespace gin {
namespace troce {

#define PCCL_TROCE_BSWAP32(x) __builtin_bswap32(x)
#define PCCL_TROCE_BSWAP64(x) __builtin_bswap64(x)
#define PCCL_TROCE_HTOBE32(x) __builtin_bswap32(x)
#define PCCL_TROCE_HTOBE64(x) __builtin_bswap64(x)
#define PCCL_TROCE_POLL_SUCCESS 0

PCCL_DEVICE_INLINE bool tryLock(uint64_t *lockPtr) {
  if (atomicCAS((unsigned long long *)lockPtr, 0ull, 1ull) != 0ull) return false;
  fenceAcquire();
  return true;
}

PCCL_DEVICE_INLINE void lock(uint64_t *lockPtr) {
  while (!tryLock(lockPtr)) continue;
}

PCCL_DEVICE_INLINE void unlock(uint64_t *lockPtr) {
  hggc::atomic_ref<uint64_t, hggc::thread_scope_device> ref(*lockPtr);
  ref.store(0, hggc::memory_order_release);
}

PCCL_DEVICE_INLINE int trocePollCqAt(struct pcclGinTroceQp *qp, uint64_t idx);

// Match GDAKI's reservation path: wait for capacity, then return the first slot.
PCCL_DEVICE_INLINE void submit(struct pcclGinTroceQp *qp, uint64_t prodIdx, bool ringDb);

PCCL_DEVICE_INLINE uint64_t reserveWqeSlots(struct pcclGinTroceQp *qp, uint32_t nwqebbs,
                                         bool submitPending = false) {
  uint64_t base = atomicAdd((unsigned long long *)qp->resv_head, (unsigned long long)nwqebbs);
  uint64_t last = base + nwqebbs - 1;
  if (last >= qp->nwqes) {
    // A flush marker also needs SQ credit when earlier aggregated requests fill
    // the queue. Submit their prefix before waiting for the marker's slot.
    if (submitPending && atomicRead(qp->cons_idx) <= last - qp->nwqes) {
      submit(qp, base, /*ringDb=*/true);
    }
    trocePollCqAt(qp, last - qp->nwqes);
  }
  return base;
}

PCCL_DEVICE_INLINE void *getWqePtr(struct pcclGinTroceQp *qp, uint64_t wqeIdx) {
  uint16_t idx = (uint16_t)(wqeIdx & (qp->nwqes - 1));
  return (void *)((uintptr_t)qp->sq_wqe + ((uintptr_t)idx * PCCL_TROCE_SEND_WQE_BB));
}

// ---------------------------------------------------------------------------
// troce doorbell
// ---------------------------------------------------------------------------

// Only explicit submission paths may call this helper. Callers must hold
// qp->db_lock (see troceRingReady).
PCCL_DEVICE_INLINE void troceRingDb(struct pcclGinTroceQp *qp, uint16_t prodIdx) {
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
// Each calling thread acquires doorbell ownership and checks prod_idx to avoid
// notifying a target that another thread has already submitted.
// Only the doorbell encoding truncates target to 16 bits; logical comparisons
// remain 64-bit across PI wraparound. CQ progress belongs to reservation and
// completion-wait paths.
PCCL_DEVICE_INLINE void troceRingReady(struct pcclGinTroceQp *qp, uint64_t target) {
  while (true) {
    lock(qp->db_lock);
    const uint64_t ready = atomicRead(qp->ready_head);
    const uint64_t prefix = ready < target ? ready : target;
    const uint64_t notified = atomicRead(qp->prod_idx);
    bool covered = notified >= target;
    if (prefix > notified) {
      troceRingDb(qp, (uint16_t)prefix);
      // Keep MMIO submission and the published tail in one critical section.
      fenceRelease();
      atomicExch((unsigned long long *)qp->prod_idx, (unsigned long long)prefix);
      covered = prefix == target;
    }
    unlock(qp->db_lock);
    if (covered) {
      return;
    }
    // Release ownership while waiting for more publishers. Credit-blocked
    // reservations progress completions in reserveWqeSlots().
  }
}

// Publish the reserved range. `endWqeIdx` is inclusive. Another warp or CTA may
// submit this range, so both fences use device scope, as in GDAKI's GPU mode.
PCCL_DEVICE_INLINE void markWqesReady(struct pcclGinTroceQp *qp, uint64_t baseWqeIdx, uint64_t endWqeIdx) {
  uint64_t newProd = endWqeIdx + 1;
  // Atomically advance the contiguous published tail so concurrent producers
  // cannot overwrite each other's updates.
  fenceRelease();
  while (atomicCAS((unsigned long long *)qp->ready_head, (unsigned long long)baseWqeIdx,
                   (unsigned long long)newProd) != (unsigned long long)baseWqeIdx) {
  }
  fenceAcquire();
}

// Notify the exclusive producer index after publishing the WQEs. Aggregation
// defers the doorbell until a non-aggregate submit or explicit flushAsync.
// Close the batch before exhausting SQ credits; CQ progress and waits do not
// submit requests.
PCCL_DEVICE_INLINE void submit(struct pcclGinTroceQp *qp, uint64_t prodIdx, bool ringDb) {
  if (!ringDb) {
    return;
  }

  // A normal submission covers all earlier published requests on this QP,
  // without extending its notification to later concurrent aggregates.
  troceRingReady(qp, prodIdx);
}

// ---------------------------------------------------------------------------
// troce WQE builders
// ---------------------------------------------------------------------------

PCCL_DEVICE_INLINE void troceWriteRdmaWqe(struct pcclGinTroceQp *qp, uint64_t laddr, __be32 lkey,
                                          uint64_t raddr, __be32 rkey, uint32_t bytes,
                                          uint16_t wqeIdx, bool needCqe, void *wqePtr,
                                          bool isWrite, bool fence = false) {
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

  const bool hasSge = bytes > 0;
  if (hasSge) {
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
  ctrl.dw1.bs.fm = fence ? PCCL_TROCE_FENCE : 0;
  if (!hasSge) {
    ctrl.dw1.bs.in_dat = 1;
  }
  ctrl.dw1.value = PCCL_TROCE_HTOBE32(ctrl.dw1.value);

  ctrl.dw2.bs.local_sge_num = hasSge ? 1 : 0;
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
PCCL_DEVICE_INLINE void troceWriteRdmaInlineWqe(struct pcclGinTroceQp *qp, const void *val,
                                                uint32_t bytes, uint64_t raddr, __be32 rkey,
                                                uint16_t wqeIdx, bool needCqe, void *wqePtr,
                                                bool signalSet = false) {
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
  // Flush write carries the address segment inside its inline payload. Its
  // length field follows the TROCE provider's host-endian encoding.
  raddrSeg.len = signalSet ? bytes : PCCL_TROCE_HTOBE32(bytes);

  ctrl.dw0.bs.qpn = (uint16_t)qp->qpn;
  ctrl.dw0.bs.q_type = PCCL_PAU_QTYPE_ROCE_SQ;
  ctrl.dw0.bs.tc = qp->traffic_class;
  ctrl.dw0.bs.wqe_size = (uint8_t)(wqeSize - 1);
  ctrl.dw0.value = PCCL_TROCE_HTOBE32(ctrl.dw0.value);

  ctrl.dw1.bs.ce = needCqe ? 2 : 0;
  ctrl.dw1.bs.opc = signalSet ? PCCL_TROCE_OPCODE_SEND_INVAL : PCCL_TROCE_OPCODE_RDMA_WRITE;
  ctrl.dw1.bs.wqe_idx = wqeIdx;
  ctrl.dw1.bs.in_dat = 1;  // payload is inline, not an SGE
  ctrl.dw1.bs.inline_offset = 0;
  ctrl.dw1.value = PCCL_TROCE_HTOBE32(ctrl.dw1.value);

  ctrl.dw2.bs.local_sge_num = 0;
  ctrl.dw2.value = PCCL_TROCE_HTOBE32(ctrl.dw2.value);
  ctrl.pld_len = PCCL_TROCE_HTOBE32(bytes + (signalSet ? sizeof(pcclTroceRaddrSeg) : 0));

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
      storeRelaxed((uint8_t *)dataPtr, *((const uint8_t *)val));
      break;
    case 2:
      storeRelaxed((uint16_t *)dataPtr, *((const uint16_t *)val));
      break;
    case 4:
      storeRelaxed((uint32_t *)dataPtr, *((const uint32_t *)val));
      break;
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
PCCL_DEVICE_INLINE void troceWriteAtomicFaWqe(struct pcclGinTroceQp *qp, uint64_t addVal,
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

// Check completion without advancing cons_idx or performing a system fence.
// The caller holds progress_lock until it has published the completed tail.
PCCL_DEVICE_INLINE int troceCheckCqAt(struct pcclGinTroceQp *qp, uint64_t idx, uint64_t *completedTail) {
  uint64_t oldCons = atomicRead(qp->cons_idx);
  *completedTail = oldCons;
  if (oldCons > idx) return PCCL_TROCE_POLL_SUCCESS;

  // Pair with submission's publication of prod_idx before reading GPU-written
  // WQE headers. The progress lock keeps their SQ slots from being reused.
  hggc::atomic_ref<uint64_t, hggc::thread_scope_device> prodRef(*qp->prod_idx);
  uint64_t prodIdx = prodRef.load(hggc::memory_order_acquire);
  if (oldCons >= prodIdx) return EBUSY;

  struct troce_be_cqe *cqe0 = (struct troce_be_cqe *)qp->cq_cqe;
  uint32_t dw14 = atomicRead((uint32_t *)&cqe0->dw14.value);
  uint16_t wqeCounter = __builtin_bswap16(
    (uint16_t)((dw14 & PCCL_TROCE_CQE_WQE_IDX_MASK) >> PCCL_TROCE_CQE_WQE_IDX_SHIFT));
  // At most nwqes WQEBBs can be outstanding. Relative to oldCons, an older
  // CQE (including the initial all-ones CQ slot) has a distance >= nwqes.
  uint16_t distance = (uint16_t)(wqeCounter - (uint16_t)oldCons);
  if (distance >= qp->nwqes) return EBUSY;
  uint64_t completedIdx = oldCons + (uint64_t)distance;
  if (completedIdx >= prodIdx) return EBUSY;

  uint32_t dw15 = atomicRead((uint32_t *)&cqe0->dw15.value);
  uint8_t opcode = (dw15 & PCCL_TROCE_CQE_OPCODE_MASK) >> PCCL_TROCE_CQE_OPCODE_SHIFT;
  if (opcode != PCCL_TROCE_CQE_REQ || (dw14 & 0xffu) != 0) {
#ifdef PCCL_DEVICE_GIN_TROCE_ENABLE_DEBUG
    printf("GIN/TROCE: got completion with err: qpn=%#x, cqn=%#x, wqe_counter=%u, "
           "opcode=%#x, status=%#x, dw14=%#x, dw15=%#x\n",
           qp->qpn, qp->cqn, (unsigned)wqeCounter, (unsigned)opcode, dw14 & 0xffu, dw14, dw15);
#endif
    return -EIO;
  }

  // The CQE identifies the first WQEBB. Recover the full WQE length, but only
  // report success when it covers the requested index, as in GDAKI's poll.
  pcclTroceCtrlSeg ctrl = {};
  ctrl.dw0.value = PCCL_TROCE_BSWAP32(atomicRead((uint32_t *)getWqePtr(qp, completedIdx)));
  uint64_t newCons = completedIdx + ctrl.dw0.bs.wqe_size + 1;
  if (newCons <= idx || newCons > prodIdx) return EBUSY;
  *completedTail = newCons;
  return PCCL_TROCE_POLL_SUCCESS;
}

// Match GDAKI: acquire at system scope once on success, then publish completion.
// Pending and error polls neither acquire at system scope nor advance cons_idx.
PCCL_DEVICE_INLINE int trocePollOneCqAt(struct pcclGinTroceQp *qp, uint64_t idx) {
  if (idx == uint64_t(-1)) {
    return PCCL_TROCE_POLL_SUCCESS; // FlushAsync on an empty QP
  }
  uint64_t completedTail = atomicRead(qp->cons_idx);
  int status = PCCL_TROCE_POLL_SUCCESS;
  bool locked = false;
  if (completedTail <= idx) {
    if (!tryLock(qp->progress_lock)) return EBUSY;
    locked = true;
    status = troceCheckCqAt(qp, idx, &completedTail);
  }
  if (status == PCCL_TROCE_POLL_SUCCESS) {
    fenceAcquire<hggc::thread_scope_system>();
    if (locked) {
      atomicExch((unsigned long long *)qp->cons_idx, (unsigned long long)completedTail);
    }
  }
  if (locked) unlock(qp->progress_lock);
  return status;
}

PCCL_DEVICE_INLINE int trocePollCqAt(struct pcclGinTroceQp *qp, uint64_t idx) {
  int status;
  do {
    status = trocePollOneCqAt(qp, idx);
  } while (status == EBUSY);
  return status;
}

PCCL_DEVICE_INLINE void troceWait(struct pcclGinTroceQp *qp, uint64_t idx, uint32_t *abortFlag) {
  using pccl::utility::testAbort;
  if (abortFlag) {
    uint32_t steps = 0;
    int status = EBUSY;
    #pragma unroll 1
    while (status != PCCL_TROCE_POLL_SUCCESS && !testAbort(abortFlag, steps)) {
      status = trocePollOneCqAt(qp, idx);
    }
  } else {
    trocePollCqAt(qp, idx);
  }
}

// Wait for everything currently reserved on this QP.
// This does not submit: the caller must close aggregates with a normal submit or
// flushAsync before waiting. Leaving an aggregate unsubmitted requires abort or
// another explicit submit to make this wait finish.
PCCL_DEVICE_INLINE void troceQuiet(struct pcclGinTroceQp *qp, uint32_t *abortFlag) {
  uint64_t resv = atomicRead(qp->resv_head);
  troceWait(qp, resv ? resv - 1 : uint64_t(-1), abortFlag);
}

// ---------------------------------------------------------------------------
// put / signal
// ---------------------------------------------------------------------------

// An optional RDMA_WRITE followed by an optional remote signal. A
// final completion ticket is consumed by flush and queue-capacity progress.
PCCL_DEVICE_INLINE void putSignal(struct pcclGinTroceGPUContext *ctx,
                                      struct pcclGinTroceQp *qp, bool hasData, uint64_t dstOff,
                                      __be32 dstRkey, uint64_t srcOff, __be32 srcLkey,
                                      size_t bytes, bool hasSignal, uint64_t signalOff,
                                      __be32 signalRkey, uint64_t signalVal,
                                      pcclGinSignalOp_t signalOp, bool ringDb) {
  const bool signalSet = signalOp == pcclGinSignalSet;
  const uint32_t signalWqebbs = signalSet ? 1u : 2u;
  uint32_t nWqe = (hasData ? 1u : 0u) + (hasSignal ? signalWqebbs : 0u);
  if (nWqe == 0) {
    return;
  }

  uint64_t base = reserveWqeSlots(qp, nWqe);
  uint64_t idx = base;

  if (hasData) {
    // The last WQE owns the completion: the signal when present, otherwise
    // the data write itself.
    troceWriteRdmaWqe(qp, srcOff, srcLkey, dstOff, dstRkey, (uint32_t)bytes, (uint16_t)idx,
                      /*needCqe=*/!hasSignal, getWqePtr(qp, idx), /*isWrite=*/true);
    idx++;
  }
  if (hasSignal) {
    if (signalSet) {
      // Flush write settles preceding puts before publishing the absolute flag.
      troceWriteRdmaInlineWqe(qp, &signalVal, sizeof(signalVal), signalOff, signalRkey,
                              (uint16_t)idx, /*needCqe=*/true, getWqePtr(qp, idx), /*signalSet=*/true);
    } else {
      // The fetched value goes to this QP's private sink slot.
      troceWriteAtomicFaWqe(qp, signalVal, qp->sink_addr, ctx->sink_buffer_lkey, signalOff,
                            signalRkey, (uint16_t)idx, /*needCqe=*/true, getWqePtr(qp, idx));
    }
    idx += signalWqebbs;
  }

  uint64_t last = idx - 1;
  markWqesReady(qp, base, last);
  submit(qp, last + 1, ringDb);
}

}  // namespace troce
}  // namespace gin
}  // namespace pccl

// ---------------------------------------------------------------------------
// GIN device API specializations
// ---------------------------------------------------------------------------

template <>
struct pcclGinApi_Put<PCCL_NET_DEVICE_GIN_TROCE> {
  template <typename Coop>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                      pcclGinWindow_t dstWin, size_t dstOff, pcclGinWindow_t srcWin,
                                      size_t srcOff, size_t bytes, pcclGinSignalDescriptor signal,
                                      pcclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                      bool hasCounter, pcclGinCounter_t counterId,
                                      bool hasDescriptor, pcclGinDescriptorSmem *descriptor,
                                      hggc::thread_scope required, hggc::thread_scope given,
                                      uint32_t optFlags) {
    if (hasCounter) {
      __builtin_trap();
    }
    using pccl::utility::loadConst;
    coop.sync();
    if (coop.thread_rank() == 0) {
      pcclGinTroceGPUContext *gda =
        &((struct pcclGinTroceGPUContext *)ctx.handle)[ctx.contextId];
      pcclGinTroceQp *qp = loadConst(&gda->qps) + peer;

      bool hasSignal = signal.type != PCCL_GIN_SIGNAL_TYPE_NONE;
      uint64_t signalOff = 0;
      __be32 signalRkey = 0;
      __be32 dstRkey = 0, srcLkey = 0;

      if (signal.type == PCCL_GIN_SIGNAL_TYPE_INDEXED) {
        signalOff = sizeof(uint64_t) *
                    (signal.indexedSignal.signalId + loadConst(&gda->signals_table.offset));
        signalRkey = loadConst(loadConst(&gda->signals_table.rkeys) + peer);
      } else if (signal.type == PCCL_GIN_SIGNAL_TYPE_VA) {
        pcclGinTroceMemHandle *mh = (pcclGinTroceMemHandle *)signal.vaSignal.signalWindow;
        signalRkey = loadConst(loadConst(&mh->rkeys) + peer);
        signalOff = signal.vaSignal.signalOffset;
      }
      uint64_t signalVal = (signalOp == pcclGinSignalInc) ? 1 : signalOpArg;

      if (hasWins) {
        pcclGinTroceMemHandle *dstMh = (pcclGinTroceMemHandle *)dstWin;
        pcclGinTroceMemHandle *srcMh = (pcclGinTroceMemHandle *)srcWin;
        dstRkey = loadConst(loadConst(&dstMh->rkeys) + peer);
        srcLkey = loadConst(&srcMh->lkey);
      }

      // Make prior GPU writes visible to the NIC before the WQE is posted.
      pccl::gin::troce::fenceRelease();
      pccl::gin::troce::putSignal(gda, qp, hasWins && bytes > 0,
                                  dstOff, dstRkey, srcOff, srcLkey,
                                  bytes, hasSignal, signalOff, signalRkey, signalVal, signalOp,
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
struct pcclGinApi_Get<PCCL_NET_DEVICE_GIN_TROCE> {
  template <typename Coop>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, Coop coop, int peer,
                                      pcclGinWindow_t remoteWin, size_t remoteOff,
                                      pcclGinWindow_t localWin, size_t localOff, size_t bytes,
                                      bool hasDescriptor, pcclGinDescriptorSmem *descriptor,
                                      uint32_t optFlags = pcclGinOptFlagsDefault) {
    using pccl::utility::loadConst;
    coop.sync();
    if (coop.thread_rank() == 0) {
      pcclGinTroceGPUContext *gda =
        &((struct pcclGinTroceGPUContext *)ctx.handle)[ctx.contextId];
      pcclGinTroceQp *qp = loadConst(&gda->qps) + peer;
      pcclGinTroceMemHandle *remoteMh = (pcclGinTroceMemHandle *)remoteWin;
      pcclGinTroceMemHandle *localMh = (pcclGinTroceMemHandle *)localWin;

      uint64_t base = pccl::gin::troce::reserveWqeSlots(qp, 1);
      pccl::gin::troce::troceWriteRdmaWqe(
        qp, localOff, loadConst(&localMh->lkey), remoteOff,
        loadConst(loadConst(&remoteMh->rkeys) + peer), (uint32_t)bytes, (uint16_t)base,
        /*needCqe=*/true, pccl::gin::troce::getWqePtr(qp, base), /*isWrite=*/false);
      pccl::gin::troce::markWqesReady(qp, base, base);
      pccl::gin::troce::submit(qp, base + 1,
                              /*ringDb=*/!(optFlags & pcclGinOptFlagsAggregateRequests));
    }
    coop.sync();
  }
};

template <>
struct pcclGinApi_Flush<PCCL_NET_DEVICE_GIN_TROCE> {
  template <typename Coop>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, Coop coop, hggc::memory_order ord,
                                      uint32_t *abortFlag) {
    (void)ord;
    using pccl::utility::loadConst;
    // Peers have separate CQs and progress locks, so different lanes can drain
    // their QPs independently. The single-slot CQ needs no CQ doorbell.
    pcclGinTroceGPUContext *gda = &((struct pcclGinTroceGPUContext *)ctx.handle)[ctx.contextId];
    pcclGinTroceQp *qps = loadConst(&gda->qps);
    #pragma unroll 1
    for (int peer = coop.thread_rank(); peer < ctx.nRanks; peer += coop.size()) {
      pccl::gin::troce::troceQuiet(qps + peer, abortFlag);
    }
  }
};

template <>
struct pcclGinApi_FlushAsync<PCCL_NET_DEVICE_GIN_TROCE> {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, int peer, pcclGinRequest_t *outRequest,
                                      uint32_t optFlags) {
    using pccl::utility::loadConst;
    pcclGinTroceRequest *req = reinterpret_cast<pcclGinTroceRequest *>(outRequest);
    pcclGinTroceGPUContext *gda = &((struct pcclGinTroceGPUContext *)ctx.handle)[ctx.contextId];
    pcclGinTroceQp *qp = loadConst(&gda->qps) + peer;
    req->peer = peer;
    const uint64_t base = pccl::gin::troce::reserveWqeSlots(qp, 1, /*submitPending=*/true);
    const uint16_t wqeIdx = (uint16_t)base;
    // Fence the zero-byte WRITE behind prior READs/atomics on this QP. Its CQE
    // provides the completion ticket without transferring sink buffer data.
    pccl::gin::troce::troceWriteRdmaWqe(qp, loadConst(&qp->sink_addr),
                                       loadConst(&gda->sink_buffer_lkey),
                                       /*remote sink MR offset=*/0, loadConst(&qp->sink_rkey),
                                       0, wqeIdx, /*needCqe=*/true,
                                       pccl::gin::troce::getWqePtr(qp, base),
                                       /*isWrite=*/true, /*fence=*/true);
    pccl::gin::troce::markWqesReady(qp, base, base);
    pccl::gin::troce::submit(qp, base + 1, /*ringDb=*/true);
    req->wqeIdx = base;
    (void)optFlags;  // flushAsync always submits, including AggregateRequests
  }
};

template <>
struct pcclGinApi_Wait<PCCL_NET_DEVICE_GIN_TROCE> {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, pcclGinRequest_t &request, bool hasDescriptor,
                                      pcclGinDescriptorSmem *descriptor, hggc::memory_order ord,
                                      uint32_t *abortFlag) {
    (void)ord;
    using pccl::utility::loadConst;
    pcclGinTroceRequest &req = reinterpret_cast<pcclGinTroceRequest &>(request);
    pcclGinTroceGPUContext *gda = &((struct pcclGinTroceGPUContext *)ctx.handle)[ctx.contextId];
    pcclGinTroceQp *qp = loadConst(&gda->qps) + req.peer;
    pccl::gin::troce::troceWait(qp, req.wqeIdx, abortFlag);
    (void)hasDescriptor;
    (void)descriptor;
  }
};

template <>
struct pcclGinApi_GetSignalPtr<PCCL_NET_DEVICE_GIN_TROCE> {
  PCCL_DEVICE_INLINE static uint64_t* call(pcclGinCtx ctx, pcclGinSignal_t signalId) {
    using pccl::utility::loadConst;
    pcclGinTroceGPUContext *gda = &((struct pcclGinTroceGPUContext *)ctx.handle)[ctx.contextId];
    return loadConst(&gda->signals_table.buffer) + signalId;
  }
};

template <>
struct pcclGinApi_ResetSignal<PCCL_NET_DEVICE_GIN_TROCE> {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, pcclGinSignalDescriptor signal) {
    using pccl::utility::loadConst;
    if (signal.type == PCCL_GIN_SIGNAL_TYPE_VA) {
      uint64_t *p =
        (uint64_t *)pcclGetLocalPointer(signal.vaSignal.pcclWindow, signal.vaSignal.signalOffset);
      *p = 0;
    } else {
      pcclGinTroceGPUContext *gda = &((struct pcclGinTroceGPUContext *)ctx.handle)[ctx.contextId];
      loadConst(&gda->signals_table.buffer)[signal.indexedSignal.signalId] = 0;
    }
  }
};

// Counter resources are rejected by createContext. These guards also reject
// device-side counter use against a context created without counter resources.
template <>
struct pcclGinApi_GetCounterPtr<PCCL_NET_DEVICE_GIN_TROCE> {
  PCCL_DEVICE_INLINE static uint64_t* call(pcclGinCtx ctx, pcclGinCounter_t) {
    __builtin_trap();
    return nullptr;
  }
};

template <>
struct pcclGinApi_ResetCounter<PCCL_NET_DEVICE_GIN_TROCE> {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, pcclGinCounter_t) {
    __builtin_trap();
  }
};

template <>
struct pcclGinApi_PutValue<PCCL_NET_DEVICE_GIN_TROCE> {
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
      pcclGinTroceGPUContext *gda =
        &((struct pcclGinTroceGPUContext *)ctx.handle)[ctx.contextId];
      pcclGinTroceQp *qp = loadConst(&gda->qps) + peer;

      bool hasSignal = signal.type != PCCL_GIN_SIGNAL_TYPE_NONE;
      uint64_t signalOff = 0;
      __be32 signalRkey = 0;
      if (signal.type == PCCL_GIN_SIGNAL_TYPE_INDEXED) {
        signalOff = sizeof(uint64_t) *
                    (signal.indexedSignal.signalId + loadConst(&gda->signals_table.offset));
        signalRkey = loadConst(loadConst(&gda->signals_table.rkeys) + peer);
      } else if (signal.type == PCCL_GIN_SIGNAL_TYPE_VA) {
        pcclGinTroceMemHandle *mh = (pcclGinTroceMemHandle *)signal.vaSignal.signalWindow;
        signalRkey = loadConst(loadConst(&mh->rkeys) + peer);
        signalOff = signal.vaSignal.signalOffset;
      }
      uint64_t signalVal = (signalOp == pcclGinSignalInc) ? 1 : signalOpArg;

      pcclGinTroceMemHandle *dstMh = (pcclGinTroceMemHandle *)dstWin;
      __be32 dstRkey = loadConst(loadConst(&dstMh->rkeys) + peer);

      pccl::gin::troce::fenceRelease();

      // The optional signal requests the completion; otherwise the
      // inline write requests it directly.
      const bool signalSet = signalOp == pcclGinSignalSet;
      const uint32_t signalWqebbs = signalSet ? 1u : 2u;
      const uint32_t nWqe = 1u + (hasSignal ? signalWqebbs : 0u);
      uint64_t base = pccl::gin::troce::reserveWqeSlots(qp, nWqe);
      uint64_t idx = base;

      pccl::gin::troce::troceWriteRdmaInlineWqe(qp, &srcVal, (uint32_t)sizeof(T), dstOff, dstRkey,
                                              (uint16_t)idx, /*needCqe=*/!hasSignal,
                                              pccl::gin::troce::getWqePtr(qp, idx));
      idx++;
      if (hasSignal) {
        if (signalSet) {
          pccl::gin::troce::troceWriteRdmaInlineWqe(qp, &signalVal, sizeof(signalVal), signalOff, signalRkey,
                                                  (uint16_t)idx, /*needCqe=*/true,
                                                  pccl::gin::troce::getWqePtr(qp, idx), /*signalSet=*/true);
        } else {
          pccl::gin::troce::troceWriteAtomicFaWqe(qp, signalVal, qp->sink_addr,
                                              loadConst(&gda->sink_buffer_lkey), signalOff, signalRkey,
                                              (uint16_t)idx, /*needCqe=*/true,
                                              pccl::gin::troce::getWqePtr(qp, idx));
        }
        idx += signalWqebbs;
      }
      pccl::gin::troce::markWqesReady(qp, base, idx - 1);
      pccl::gin::troce::submit(qp, idx,
                              /*ringDb=*/!(optFlags & pcclGinOptFlagsAggregateRequests));
    }
    coop.sync();
    (void)hasDescriptor;
    (void)descriptor;
    (void)required;
    (void)given;
  }
};

#endif /* _PCCL_DEVICE_GIN_TROCE_H_ */
