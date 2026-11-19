/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCLX_DEVICE_GIN_TROCE_H_
#define _PCCLX_DEVICE_GIN_TROCE_H_

#include <cstdint>
#include <errno.h>

#ifdef PCCLX_DEVICE_GIN_TROCE_ENABLE_DEBUG
#include <stdio.h>
#endif

#include "../gin_device_common.h"
#include "gin_troce_device_host_common.h"
#include "gin_troce_hw.h"
#include "gin_troce_utils.h"

// Device-side GIN interface for the TROCE backend.

struct pcclxGinTroceRequest {
  int peer;
  uint64_t prodIdx;  // exclusive WQEBB tail captured by flushAsync
};
static_assert(sizeof(pcclxGinTroceRequest) <= sizeof(pcclGinRequest_t),
              "pcclxGinTroceRequest must fit in pcclGinRequest_t");

namespace pcclx {
namespace gin {
namespace troce {

#define PCCLX_TROCE_BSWAP32(x) __builtin_bswap32(x)
#define PCCLX_TROCE_BSWAP64(x) __builtin_bswap64(x)
#define PCCLX_TROCE_HTOBE32(x) __builtin_bswap32(x)
#define PCCLX_TROCE_HTOBE64(x) __builtin_bswap64(x)
#define PCCLX_TROCE_POLL_SUCCESS 0

template <hggc::thread_scope scope = hggc::thread_scope_device>
PCCL_DEVICE_INLINE bool tryLock(int *lockPtr) {
  int old = scope == hggc::thread_scope_block ? atomicCAS_block(lockPtr, 0, 1) : atomicCAS(lockPtr, 0, 1);
  if (old != 0) return false;
  fenceAcquire<scope>();
  return true;
}

template <hggc::thread_scope scope = hggc::thread_scope_device>
PCCL_DEVICE_INLINE void unlock(int *lockPtr) {
  hggc::atomic_ref<int, scope> ref(*lockPtr);
  ref.store(0, hggc::memory_order_release);
}

template <hggc::thread_scope scope = hggc::thread_scope_device>
PCCL_DEVICE_INLINE int trocePollCqAt(struct pcclxGinTroceQp *qp, uint64_t idx);

// Wait for SQ capacity by progressing completions.
template <hggc::thread_scope scope = hggc::thread_scope_device>
PCCL_DEVICE_INLINE void waitUntilSlotAvailable(struct pcclxGinTroceQp *qp, uint64_t wqeIdx) {
  if (wqeIdx >= qp->nwqes) {
    trocePollCqAt<scope>(qp, wqeIdx - qp->nwqes);
  }
}

template <hggc::thread_scope scope = hggc::thread_scope_device>
PCCL_DEVICE_INLINE uint64_t reserveWqeSlots(struct pcclxGinTroceQp *qp, uint32_t nwqebbs,
                                             uint32_t optFlags = pcclGinOptFlagsDefault) {
  hggc::atomic_ref<uint64_t, scope> ref(qp->resv_head);
  uint64_t base = ref.fetch_add(nwqebbs, hggc::memory_order_relaxed);
  if (!(optFlags & pcclGinOptFlagsMaySkipCreditCheck))
    waitUntilSlotAvailable<scope>(qp, base + nwqebbs - 1);
  return base;
}

PCCL_DEVICE_INLINE void *getWqePtr(struct pcclxGinTroceQp *qp, uint64_t wqeIdx) {
  uint16_t idx = (uint16_t)(wqeIdx & (qp->nwqes - 1));
  return (void *)((uintptr_t)qp->sq_wqe + ((uintptr_t)idx * PCCLX_TROCE_SEND_WQE_BB));
}

// ---------------------------------------------------------------------------
// troce doorbell
// ---------------------------------------------------------------------------

// Only explicit submission paths may call this helper. Callers must hold
// qp->db_lock (see submit).
PCCL_DEVICE_INLINE void troceRingDb(struct pcclxGinTroceQp *qp, uint16_t prodIdx) {
  pcclxTroceSqDb db;
  db.value = 0;
  db.db.dw0.bs.qid = qp->qpn;
  db.db.dw0.bs.q_type = PCCLX_PAU_QTYPE_ROCE_SQ;
  db.db.dw1.bs.pi = prodIdx;
  // Use the QP traffic class queried after RTR.
  db.db.dw1.bs.tc = qp->traffic_class;
  db.db.dw0.value = PCCLX_TROCE_HTOBE32(db.db.dw0.value);
  db.db.dw1.value = PCCLX_TROCE_HTOBE32(db.db.dw1.value);
  storeRelease((uint64_t *)qp->sq_db, db.value);
}

// Publish the reserved range. `endWqeIdx` is inclusive.
template <hggc::thread_scope scope = hggc::thread_scope_device>
PCCL_DEVICE_INLINE void markWqesReady(struct pcclxGinTroceQp *qp, uint64_t baseWqeIdx, uint64_t endWqeIdx) {
  fenceRelease<scope>();
  if (scope == hggc::thread_scope_block) {
    while (atomicCAS_block((unsigned long long *)&qp->ready_head, baseWqeIdx, endWqeIdx + 1) != baseWqeIdx)
      continue;
  } else {
    while (atomicCAS((unsigned long long *)&qp->ready_head, baseWqeIdx, endWqeIdx + 1) != baseWqeIdx)
      continue;
  }
  fenceAcquire<scope>();
}

// Notify an already-published exclusive producer index. Aggregation defers the
// doorbell until a normal submit or flushAsync. Callers must submit
// before exhausting SQ credits.
template <hggc::thread_scope scope = hggc::thread_scope_device>
PCCL_DEVICE_INLINE void submit(struct pcclxGinTroceQp *qp, uint64_t prodIdx, uint32_t optFlags) {
  if (optFlags & pcclGinOptFlagsAggregateRequests) {
    return;
  }

  while (true) {
    if (tryLock<scope>(&qp->db_lock)) {
      hggc::atomic_ref<uint64_t, scope> ref(qp->prod_idx);
      uint64_t old = ref.fetch_max(prodIdx, hggc::memory_order_acquire);
      if (old < prodIdx) {
        troceRingDb(qp, (uint16_t)prodIdx);
      }
      unlock<scope>(&qp->db_lock);
      break;
    }
  }
}

// ---------------------------------------------------------------------------
// troce WQE builders
// ---------------------------------------------------------------------------

PCCL_DEVICE_INLINE void troceWriteRdmaWqe(struct pcclxGinTroceQp *qp, uint64_t laddr, __be32 lkey,
                                          uint64_t raddr, __be32 rkey, uint32_t bytes,
                                          uint16_t wqeIdx, bool needCqe, void *wqePtr,
                                          bool isWrite, bool fence = false) {
  pcclxTroceCtrlSeg ctrl = {};
  pcclxTroceRaddrSeg raddrSeg = {};
  pcclxTroceDataSeg dataSeg = {};

  pcclxTroceCtrlSeg *ctrlPtr = (pcclxTroceCtrlSeg *)wqePtr;
  pcclxTroceRaddrSeg *raddrPtr =
    (pcclxTroceRaddrSeg *)((uintptr_t)ctrlPtr + sizeof(pcclxTroceCtrlSeg));
  uintptr_t after = (uintptr_t)raddrPtr + sizeof(pcclxTroceRaddrSeg);

  size_t wqeSize = sizeof(pcclxTroceCtrlSeg) + sizeof(pcclxTroceRaddrSeg);
  pcclxTroceDataSeg *dataPtr = (pcclxTroceDataSeg *)(after + PCCLX_TROCE_16B_PAD(wqeSize));

  raddrSeg.raddr = PCCLX_TROCE_HTOBE64(raddr);
  raddrSeg.rkey = rkey;
  raddrSeg.len = PCCLX_TROCE_HTOBE32(bytes);

  const bool hasSge = bytes > 0;
  if (hasSge) {
    dataSeg.len = PCCLX_TROCE_HTOBE32(bytes);
    dataSeg.dw3.bs.lkey = PCCLX_TROCE_BSWAP32(lkey);
    dataSeg.dw3.value = PCCLX_TROCE_HTOBE32(dataSeg.dw3.value);
    dataSeg.addr = PCCLX_TROCE_HTOBE64(laddr);
  }

  wqeSize += sizeof(pcclxTroceDataSeg);
  wqeSize = PCCLX_TROCE_DIV_ROUND_UP(wqeSize, PCCLX_TROCE_SEND_WQE_BB);

  ctrl.dw0.bs.qpn = (uint16_t)qp->qpn;
  ctrl.dw0.bs.q_type = PCCLX_PAU_QTYPE_ROCE_SQ;
  ctrl.dw0.bs.tc = qp->traffic_class;
  ctrl.dw0.bs.wqe_size = (uint8_t)(wqeSize - 1);
  ctrl.dw0.value = PCCLX_TROCE_HTOBE32(ctrl.dw0.value);

  ctrl.dw1.bs.ce = needCqe ? 2 : 0;
  ctrl.dw1.bs.opc = isWrite ? PCCLX_TROCE_OPCODE_RDMA_WRITE : PCCLX_TROCE_OPCODE_RDMA_READ;
  ctrl.dw1.bs.wqe_idx = wqeIdx;
  ctrl.dw1.bs.fm = fence ? PCCLX_TROCE_FENCE : 0;
  if (!hasSge) {
    ctrl.dw1.bs.in_dat = 1;
  }
  ctrl.dw1.value = PCCLX_TROCE_HTOBE32(ctrl.dw1.value);

  ctrl.dw2.bs.local_sge_num = hasSge ? 1 : 0;
  ctrl.dw2.value = PCCLX_TROCE_HTOBE32(ctrl.dw2.value);
  ctrl.pld_len = PCCLX_TROCE_HTOBE32(bytes);

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
PCCL_DEVICE_INLINE void troceWriteRdmaInlineWqe(struct pcclxGinTroceQp *qp, const void *val,
                                                uint32_t bytes, uint64_t raddr, __be32 rkey,
                                                uint16_t wqeIdx, bool needCqe, void *wqePtr) {
  pcclxTroceCtrlSeg ctrl = {};
  pcclxTroceRaddrSeg raddrSeg = {};

  pcclxTroceCtrlSeg *ctrlPtr = (pcclxTroceCtrlSeg *)wqePtr;
  pcclxTroceRaddrSeg *raddrPtr =
    (pcclxTroceRaddrSeg *)((uintptr_t)ctrlPtr + sizeof(pcclxTroceCtrlSeg));
  void *dataPtr = (void *)((uintptr_t)raddrPtr + sizeof(pcclxTroceRaddrSeg));
  const uint32_t hdrSize = (uint32_t)((uintptr_t)dataPtr - (uintptr_t)ctrlPtr);

  size_t total = hdrSize + bytes;
  total += PCCLX_TROCE_16B_PAD(total);
  size_t wqeSize = PCCLX_TROCE_DIV_ROUND_UP(total, PCCLX_TROCE_SEND_WQE_BB);

  raddrSeg.raddr = PCCLX_TROCE_HTOBE64(raddr);
  raddrSeg.rkey = rkey;
  raddrSeg.len = PCCLX_TROCE_HTOBE32(bytes);

  ctrl.dw0.bs.qpn = (uint16_t)qp->qpn;
  ctrl.dw0.bs.q_type = PCCLX_PAU_QTYPE_ROCE_SQ;
  ctrl.dw0.bs.tc = qp->traffic_class;
  ctrl.dw0.bs.wqe_size = (uint8_t)(wqeSize - 1);
  ctrl.dw0.value = PCCLX_TROCE_HTOBE32(ctrl.dw0.value);

  ctrl.dw1.bs.ce = needCqe ? 2 : 0;
  ctrl.dw1.bs.opc = PCCLX_TROCE_OPCODE_RDMA_WRITE;
  ctrl.dw1.bs.wqe_idx = wqeIdx;
  ctrl.dw1.bs.in_dat = 1;  // payload is inline, not an SGE
  ctrl.dw1.bs.inline_offset = 0;
  ctrl.dw1.value = PCCLX_TROCE_HTOBE32(ctrl.dw1.value);

  ctrl.dw2.bs.local_sge_num = 0;
  ctrl.dw2.value = PCCLX_TROCE_HTOBE32(ctrl.dw2.value);
  ctrl.pld_len = PCCLX_TROCE_HTOBE32(bytes);

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

// Publish an 8-byte signal value with flush-write ordering for preceding puts.
PCCL_DEVICE_INLINE void troceWriteFlushWqe(struct pcclxGinTroceQp *qp, uint64_t val,
                                         uint64_t raddr, __be32 rkey, uint16_t wqeIdx,
                                         bool needCqe, void *wqePtr) {
  const uint32_t bytes = sizeof(val);
  pcclxTroceCtrlSeg ctrl = {};
  pcclxTroceRaddrSeg raddrSeg = {};

  pcclxTroceCtrlSeg *ctrlPtr = (pcclxTroceCtrlSeg *)wqePtr;
  pcclxTroceRaddrSeg *raddrPtr =
    (pcclxTroceRaddrSeg *)((uintptr_t)ctrlPtr + sizeof(pcclxTroceCtrlSeg));
  void *dataPtr = (void *)((uintptr_t)raddrPtr + sizeof(pcclxTroceRaddrSeg));
  const uint32_t hdrSize = (uint32_t)((uintptr_t)dataPtr - (uintptr_t)ctrlPtr);

  size_t total = hdrSize + bytes;
  total += PCCLX_TROCE_16B_PAD(total);
  size_t wqeSize = PCCLX_TROCE_DIV_ROUND_UP(total, PCCLX_TROCE_SEND_WQE_BB);

  raddrSeg.raddr = PCCLX_TROCE_HTOBE64(raddr);
  raddrSeg.rkey = rkey;
  // Flush write carries the address segment inside its inline payload. Its
  // length field follows the TROCE provider's host-endian encoding.
  raddrSeg.len = bytes;

  ctrl.dw0.bs.qpn = (uint16_t)qp->qpn;
  ctrl.dw0.bs.q_type = PCCLX_PAU_QTYPE_ROCE_SQ;
  ctrl.dw0.bs.tc = qp->traffic_class;
  ctrl.dw0.bs.wqe_size = (uint8_t)(wqeSize - 1);
  ctrl.dw0.value = PCCLX_TROCE_HTOBE32(ctrl.dw0.value);

  ctrl.dw1.bs.ce = needCqe ? 2 : 0;
  ctrl.dw1.bs.opc = PCCLX_TROCE_OPCODE_SEND_INVAL;
  ctrl.dw1.bs.wqe_idx = wqeIdx;
  ctrl.dw1.bs.in_dat = 1;  // payload is inline, not an SGE
  ctrl.dw1.bs.inline_offset = 0;
  ctrl.dw1.value = PCCLX_TROCE_HTOBE32(ctrl.dw1.value);

  ctrl.dw2.bs.local_sge_num = 0;
  ctrl.dw2.value = PCCLX_TROCE_HTOBE32(ctrl.dw2.value);
  ctrl.pld_len = PCCLX_TROCE_HTOBE32(bytes + sizeof(pcclxTroceRaddrSeg));

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

  storeRelaxed(&(((uint32_t *)dataPtr)[0]), ((const uint32_t *)&val)[0]);
  storeRelaxed(&(((uint32_t *)dataPtr)[1]), ((const uint32_t *)&val)[1]);
}

// Encode an 8-byte remote fetch-and-add. The returned value is written to the
// QP's atomic return storage and is not consumed.
PCCL_DEVICE_INLINE void troceWriteAtomicFaWqe(struct pcclxGinTroceQp *qp, uint64_t addVal,
                                              uint64_t laddr, __be32 lkey, uint64_t raddr,
                                              __be32 rkey, uint16_t wqeIdx, bool needCqe,
                                              void *wqePtr) {
  pcclxTroceCtrlSeg ctrl = {};
  pcclxTroceRaddrSeg raddrSeg = {};
  pcclxTroceAtomicSeg atomicSeg = {};
  pcclxTroceDataSeg dataSeg = {};

  pcclxTroceCtrlSeg *ctrlPtr = (pcclxTroceCtrlSeg *)wqePtr;
  pcclxTroceRaddrSeg *raddrPtr =
    (pcclxTroceRaddrSeg *)((uintptr_t)ctrlPtr + sizeof(pcclxTroceCtrlSeg));
  pcclxTroceAtomicSeg *atomicPtr =
    (pcclxTroceAtomicSeg *)((uintptr_t)raddrPtr + sizeof(pcclxTroceRaddrSeg));
  uintptr_t after = (uintptr_t)atomicPtr + sizeof(pcclxTroceAtomicSeg);

  size_t wqeSize =
    sizeof(pcclxTroceCtrlSeg) + sizeof(pcclxTroceRaddrSeg) + sizeof(pcclxTroceAtomicSeg);
  // An atomic WQE spans more than one WQEBB, so the trailing data segment can
  // cross the SQ wrap point; in that case it lands at the ring base.
  bool wraps = ((wqeIdx + 1) % qp->nwqes) == 0;
  pcclxTroceDataSeg *dataPtr = wraps ? (pcclxTroceDataSeg *)qp->sq_wqe
                                    : (pcclxTroceDataSeg *)(after + PCCLX_TROCE_16B_PAD(wqeSize));

  raddrSeg.raddr = PCCLX_TROCE_HTOBE64(raddr);
  raddrSeg.rkey = rkey;
  raddrSeg.len = PCCLX_TROCE_HTOBE32(PCCLX_TROCE_ATOMIC_SIZE);

  dataSeg.len = PCCLX_TROCE_HTOBE32(PCCLX_TROCE_ATOMIC_SIZE);
  dataSeg.dw3.bs.lkey = PCCLX_TROCE_BSWAP32(lkey);
  dataSeg.dw3.value = PCCLX_TROCE_HTOBE32(dataSeg.dw3.value);
  dataSeg.addr = PCCLX_TROCE_HTOBE64(laddr);

  atomicSeg.swap_or_add_dat = PCCLX_TROCE_HTOBE64(addVal);

  wqeSize += sizeof(pcclxTroceDataSeg);
  wqeSize = PCCLX_TROCE_DIV_ROUND_UP(wqeSize, PCCLX_TROCE_SEND_WQE_BB);

  ctrl.dw0.bs.qpn = (uint16_t)qp->qpn;
  ctrl.dw0.bs.q_type = PCCLX_PAU_QTYPE_ROCE_SQ;
  ctrl.dw0.bs.tc = qp->traffic_class;
  ctrl.dw0.bs.wqe_size = (uint8_t)(wqeSize - 1);
  ctrl.dw0.value = PCCLX_TROCE_HTOBE32(ctrl.dw0.value);

  ctrl.dw1.bs.ce = needCqe ? 2 : 0;
  ctrl.dw1.bs.opc = PCCLX_TROCE_OPCODE_ATOMIC_FA;
  ctrl.dw1.bs.wqe_idx = wqeIdx;
  ctrl.dw1.value = PCCLX_TROCE_HTOBE32(ctrl.dw1.value);

  ctrl.dw2.bs.local_sge_num = 1;
  ctrl.dw2.value = PCCLX_TROCE_HTOBE32(ctrl.dw2.value);
  ctrl.pld_len = PCCLX_TROCE_HTOBE32(PCCLX_TROCE_ATOMIC_SIZE);

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
template <hggc::thread_scope scope = hggc::thread_scope_device>
PCCL_DEVICE_INLINE int troceCheckCqAt(struct pcclxGinTroceQp *qp, uint64_t idx, uint64_t oldCons,
                                      uint64_t *completedTail) {
  *completedTail = oldCons;
  if (oldCons > idx) return PCCLX_TROCE_POLL_SUCCESS;

  // The producer index covers published WQEs.
  hggc::atomic_ref<uint64_t, scope> prodRef(qp->prod_idx);
  uint64_t prodIdx = prodRef.load(hggc::memory_order_acquire);
  if (oldCons >= prodIdx) return EBUSY;

  struct troce_be_cqe *cqe0 = (struct troce_be_cqe *)qp->cq_cqe;
  uint64_t completion = loadRelaxedSysGlobal((uint64_t *)&cqe0->dw14.value);
  uint32_t dw14 = (uint32_t)completion;
  uint32_t dw15 = (uint32_t)(completion >> 32);
  uint16_t wqeCounter = __builtin_bswap16(
    (uint16_t)((dw14 & PCCLX_TROCE_CQE_WQE_IDX_MASK) >> PCCLX_TROCE_CQE_WQE_IDX_SHIFT));
  // At most nwqes WQEBBs can be outstanding. Relative to oldCons, an older
  // CQE (including the initial all-ones CQ slot) has a distance >= nwqes.
  uint16_t distance = (uint16_t)(wqeCounter - (uint16_t)oldCons);
  if (distance >= qp->nwqes) return EBUSY;
  uint64_t completedIdx = oldCons + (uint64_t)distance;
  if (completedIdx >= prodIdx) return EBUSY;

  uint8_t opcode = (dw15 & PCCLX_TROCE_CQE_OPCODE_MASK) >> PCCLX_TROCE_CQE_OPCODE_SHIFT;
  if (opcode != PCCLX_TROCE_CQE_REQ || (dw14 & 0xffu) != 0) {
#ifdef PCCLX_DEVICE_GIN_TROCE_ENABLE_DEBUG
    printf("GIN/TROCE: got completion with err: qpn=%#x, cqn=%#x, wqe_counter=%u, "
           "opcode=%#x, status=%#x, dw14=%#x, dw15=%#x\n",
           qp->qpn, qp->cqn, (unsigned)wqeCounter, (unsigned)opcode, dw14 & 0xffu, dw14, dw15);
#endif
    return -EIO;
  }

  // The CQE identifies the first WQEBB. Recover the full WQE length, but only
  // report success when it covers the requested index.
  pcclxTroceCtrlSeg ctrl = {};
  ctrl.dw0.value = PCCLX_TROCE_BSWAP32(atomicRead((uint32_t *)getWqePtr(qp, completedIdx)));
  uint64_t newCons = completedIdx + ctrl.dw0.bs.wqe_size + 1;
  if (newCons <= idx || newCons > prodIdx) return EBUSY;
  *completedTail = newCons;
  return PCCLX_TROCE_POLL_SUCCESS;
}

template <hggc::thread_scope scope = hggc::thread_scope_device>
PCCL_DEVICE_INLINE int trocePollOneCqAt(struct pcclxGinTroceQp *qp, uint64_t idx) {
  uint64_t oldCons = loadRelaxed<scope>(&qp->cons_idx);
  uint64_t completedTail;
  int status = troceCheckCqAt<scope>(qp, idx, oldCons, &completedTail);
  if (status != PCCLX_TROCE_POLL_SUCCESS) return status;

  fenceAcquire<hggc::thread_scope_system>();
  if (completedTail != oldCons) {
    // Reject the SQ header read if another poller may have released its slot.
    hggc::atomic_ref<uint64_t, scope> ref(qp->cons_idx);
    if (!ref.compare_exchange_strong(oldCons, completedTail,
                                     hggc::memory_order_relaxed, hggc::memory_order_relaxed)) return EBUSY;
  }
  return PCCLX_TROCE_POLL_SUCCESS;
}

template <hggc::thread_scope scope>
PCCL_DEVICE_INLINE int trocePollCqAt(struct pcclxGinTroceQp *qp, uint64_t idx) {
  int status;
  #pragma unroll 1
  do {
    status = trocePollOneCqAt<scope>(qp, idx);
  } while (status == EBUSY);
  return status;
}

PCCL_DEVICE_INLINE bool troceWait(struct pcclxGinTroceQp *qp, uint64_t idx, uint32_t *abortFlag) {
  using pccl::utility::testAbort;
  if (abortFlag) {
    uint32_t steps = 0;
    int status = EBUSY;
    #pragma unroll 1
    while (status != PCCLX_TROCE_POLL_SUCCESS && !testAbort(abortFlag, steps)) {
      status = trocePollOneCqAt(qp, idx);
    }
    return status == PCCLX_TROCE_POLL_SUCCESS;
  } else {
    return trocePollCqAt(qp, idx) == PCCLX_TROCE_POLL_SUCCESS;
  }
}

PCCL_DEVICE_INLINE uint64_t troceFlushAsync(struct pcclxGinTroceQp *qp) {
  hggc::atomic_ref<uint64_t, hggc::thread_scope_device> ref(qp->ready_head);
  uint64_t prodIdx = ref.load(hggc::memory_order_acquire);
  if (prodIdx != 0) submit(qp, prodIdx, pcclGinOptFlagsDefault);
  return prodIdx;
}

PCCL_DEVICE_INLINE bool troceFlushWait(struct pcclxGinTroceGPUContext *gda, struct pcclxGinTroceQp *qp,
                                      struct pcclxGinTroceQp *flushQp, uint64_t prodIdx, uint32_t *abortFlag) {
  using pccl::utility::loadConst;
  if (prodIdx == 0) return true;
  if (!troceWait(qp, prodIdx - 1, abortFlag)) return false;
  if (abortFlag && hggc::atomic_ref<uint32_t, hggc::thread_scope_system>{*abortFlag}
                       .load(hggc::memory_order_relaxed)) return false;

  uint64_t base = reserveWqeSlots(flushQp, 1);
  // Publish a fresh loopback WQE only after the data CQ completion.
  troceWriteRdmaWqe(flushQp, 0, loadConst(&gda->sink_buffer_lkey),
                    0, loadConst(&flushQp->sink_rkey), 0, (uint16_t)base,
                    /*needCqe=*/true, getWqePtr(flushQp, base), /*isWrite=*/true);
  markWqesReady(flushQp, base, base);
  submit(flushQp, base + 1, pcclGinOptFlagsDefault);
  return troceWait(flushQp, base, abortFlag);
}

// Wait for everything currently reserved on this QP.
// This does not submit: the caller must close aggregates with a normal submit or
// flushAsync before waiting. Leaving an aggregate unsubmitted requires abort or
// another explicit submit to make this wait finish.
PCCL_DEVICE_INLINE void troceQuiet(struct pcclxGinTroceQp *qp, uint32_t *abortFlag) {
  uint64_t resv = atomicRead(&qp->resv_head);
  if (resv == 0) return;
  troceWait(qp, resv - 1, abortFlag);
}

// ---------------------------------------------------------------------------
// put / signal
// ---------------------------------------------------------------------------

// An optional RDMA_WRITE followed by an optional remote signal. A
// final completion ticket is consumed by flush and queue-capacity progress.
template <hggc::thread_scope scope = hggc::thread_scope_device>
PCCL_DEVICE_INLINE void putSignal(struct pcclxGinTroceGPUContext *ctx,
                                      struct pcclxGinTroceQp *qp, bool hasData, uint64_t dstOff,
                                      __be32 dstRkey, uint64_t srcOff, __be32 srcLkey,
                                      size_t bytes, bool hasSignal, uint64_t signalOff,
                                      __be32 signalRkey, uint64_t signalVal,
                                      pcclGinSignalOp_t signalOp, uint32_t optFlags) {
  const bool signalSet = signalOp == pcclxGinSignalSet;
  const uint32_t signalWqebbs = signalSet ? 1u : 2u;
  uint32_t numChunks = hasData ? (bytes == 0 ? 1u : uint32_t((bytes - 1) / PCCLX_GIN_TROCE_MAX_TRANSFER_SIZE + 1)) : 0u;
  uint32_t nwqebbs = numChunks + (hasSignal ? signalWqebbs : 0u);
  if (nwqebbs == 0) return;

  uint64_t base = reserveWqeSlots<scope>(qp, nwqebbs, optFlags);
  uint64_t idx = base;
  size_t remaining = bytes;
  #pragma unroll 1
  for (uint32_t i = 0; i < numChunks; i++, idx++) {
    uint32_t chunk = uint32_t(remaining > PCCLX_GIN_TROCE_MAX_TRANSFER_SIZE ?
                              PCCLX_GIN_TROCE_MAX_TRANSFER_SIZE : remaining);
    if (chunk > 0) {
      troceWriteRdmaWqe(qp, srcOff, srcLkey, dstOff, dstRkey, chunk, (uint16_t)idx,
                        /*needCqe=*/true, getWqePtr(qp, idx), /*isWrite=*/true);
    } else {
      // An ordinary zero-byte WRITE provides the empty operation's completion.
      troceWriteRdmaWqe(qp, 0, ctx->sink_buffer_lkey, 0, qp->sink_rkey, 0, (uint16_t)idx,
                        /*needCqe=*/true, getWqePtr(qp, idx), /*isWrite=*/true);
    }
    srcOff += chunk;
    dstOff += chunk;
    remaining -= chunk;
  }
  if (hasSignal) {
    if (signalSet) {
      // Flush write settles preceding puts before publishing the absolute flag.
      troceWriteFlushWqe(qp, signalVal, signalOff, signalRkey, (uint16_t)idx,
                         /*needCqe=*/true, getWqePtr(qp, idx));
    } else {
      // The fetched value is discarded in the shared sink.
      troceWriteAtomicFaWqe(qp, signalVal, 0, ctx->sink_buffer_lkey, signalOff,
                            signalRkey, (uint16_t)idx, /*needCqe=*/true, getWqePtr(qp, idx));
    }
    idx += signalWqebbs;
  }

  uint64_t last = idx - 1;
  markWqesReady<scope>(qp, base, last);
  submit<scope>(qp, last + 1, optFlags);
}

}  // namespace troce
}  // namespace gin
}  // namespace pcclx

// ---------------------------------------------------------------------------
// GIN device API specializations
// ---------------------------------------------------------------------------

template <>
struct pcclGinApi_Put<PCCLX_NET_DEVICE_GIN_TROCE> {
  template <typename Coop>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                      pcclGinWindow_t dstWin, size_t dstOff, pcclGinWindow_t srcWin,
                                      size_t srcOff, size_t bytes, pcclGinSignalDescriptor signal,
                                      pcclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                      bool hasCounter, pcclGinCounter_t counterId,
                                      bool hasDescriptor, pcclGinDescriptorSmem *descriptor,
                                      hggc::thread_scope required, hggc::thread_scope given,
                                      uint32_t optFlags) {
    if (ctx.resourceSharingMode == PCCL_GIN_RESOURCE_SHARING_CTA) {
      callMode<hggc::thread_scope_block>(ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff, bytes, signal,
        signalOp, signalOpArg, hasCounter, counterId, hasDescriptor, descriptor,
        required, given, optFlags);
    } else {
      callMode<hggc::thread_scope_device>(ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff, bytes, signal,
        signalOp, signalOpArg, hasCounter, counterId, hasDescriptor, descriptor,
        required, given, optFlags);
    }
  }

  template <hggc::thread_scope scope, typename Coop>
  PCCL_DEVICE_INLINE static void callMode(pcclGinCtx ctx, Coop coop, int peer, bool hasWins,
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
      pcclxGinTroceGPUContext *gda =
        &((struct pcclxGinTroceGPUContext *)ctx.handle)[ctx.contextId];
      pcclxGinTroceQp *qp = loadConst(&gda->qps) + peer;

      bool hasSignal = signal.type != PCCL_GIN_SIGNAL_TYPE_NONE;
      uint64_t signalOff = 0;
      __be32 signalRkey = 0;
      __be32 dstRkey = 0, srcLkey = 0;

      if (signal.type == PCCL_GIN_SIGNAL_TYPE_INDEXED) {
        signalOff = sizeof(uint64_t) *
                    (signal.indexedSignal.signalId + loadConst(&gda->signals_table.offset));
        signalRkey = loadConst(loadConst(&gda->signals_table.rkeys) + peer);
      } else if (signal.type == PCCL_GIN_SIGNAL_TYPE_VA) {
        pcclxGinTroceMemHandle *mh = (pcclxGinTroceMemHandle *)signal.vaSignal.signalWindow;
        signalRkey = loadConst(loadConst(&mh->rkeys) + peer);
        signalOff = signal.vaSignal.signalOffset;
      }
      uint64_t signalVal = (signalOp == pcclGinSignalInc) ? 1 : signalOpArg;

      if (hasWins) {
        pcclxGinTroceMemHandle *dstMh = (pcclxGinTroceMemHandle *)dstWin;
        pcclxGinTroceMemHandle *srcMh = (pcclxGinTroceMemHandle *)srcWin;
        dstRkey = loadConst(loadConst(&dstMh->rkeys) + peer);
        srcLkey = loadConst(&srcMh->lkey);
      }

      if (required == hggc::thread_scope_system && given > required)
        pcclx::gin::troce::fenceRelease<hggc::thread_scope_system>();
      pcclx::gin::troce::putSignal<scope>(gda, qp, hasWins,
                                  dstOff, dstRkey, srcOff, srcLkey,
                                  bytes, hasSignal, signalOff, signalRkey, signalVal, signalOp,
                                  optFlags);
#ifdef PCCLX_DEVICE_GIN_TROCE_ENABLE_DEBUG
      pcclx::gin::troce::troceQuiet(qp, nullptr);
#endif
    }
    coop.sync();
    (void)hasDescriptor;
    (void)descriptor;
  }
};

template <>
struct pcclGinApi_Get<PCCLX_NET_DEVICE_GIN_TROCE> {
  template <typename Coop>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, Coop coop, int peer,
                                      pcclGinWindow_t remoteWin, size_t remoteOff,
                                      pcclGinWindow_t localWin, size_t localOff, size_t bytes,
                                      bool hasDescriptor, pcclGinDescriptorSmem *descriptor,
                                      uint32_t optFlags = pcclGinOptFlagsDefault) {
    using pccl::utility::loadConst;
    coop.sync();
    if (coop.thread_rank() == 0) {
      pcclxGinTroceGPUContext *gda =
        &((struct pcclxGinTroceGPUContext *)ctx.handle)[ctx.contextId];
      pcclxGinTroceQp *qp = loadConst(&gda->qps) + peer;
      pcclxGinTroceMemHandle *remoteMh = (pcclxGinTroceMemHandle *)remoteWin;
      pcclxGinTroceMemHandle *localMh = (pcclxGinTroceMemHandle *)localWin;

      uint32_t numChunks = bytes == 0 ? 1u : uint32_t((bytes - 1) / PCCLX_GIN_TROCE_MAX_TRANSFER_SIZE + 1);
      uint64_t base = pcclx::gin::troce::reserveWqeSlots(qp, numChunks, optFlags);
      size_t remaining = bytes;
      #pragma unroll 1
      for (uint32_t i = 0; i < numChunks; i++) {
        uint32_t chunk = uint32_t(remaining > PCCLX_GIN_TROCE_MAX_TRANSFER_SIZE ?
                                  PCCLX_GIN_TROCE_MAX_TRANSFER_SIZE : remaining);
        if (chunk > 0) {
          pcclx::gin::troce::troceWriteRdmaWqe(
            qp, localOff, loadConst(&localMh->lkey), remoteOff,
            loadConst(loadConst(&remoteMh->rkeys) + peer), chunk, (uint16_t)(base + i),
            /*needCqe=*/true, pcclx::gin::troce::getWqePtr(qp, base + i), /*isWrite=*/false);
        } else {
          pcclx::gin::troce::troceWriteRdmaWqe(
            qp, 0, loadConst(&gda->sink_buffer_lkey), 0, loadConst(&qp->sink_rkey),
            0, (uint16_t)(base + i), /*needCqe=*/true,
            pcclx::gin::troce::getWqePtr(qp, base + i), /*isWrite=*/true);
        }
        localOff += chunk;
        remoteOff += chunk;
        remaining -= chunk;
      }
      pcclx::gin::troce::markWqesReady(qp, base, base + numChunks - 1);
      pcclx::gin::troce::submit(qp, base + numChunks, optFlags);
    }
    coop.sync();
  }
};

template <>
struct pcclGinApi_Flush<PCCLX_NET_DEVICE_GIN_TROCE> {
  template <typename Coop>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, Coop coop, hggc::memory_order ord,
                                      uint32_t *abortFlag) {
    (void)ord;
    using pccl::utility::loadConst;
    pcclxGinTroceGPUContext *gda = &((struct pcclxGinTroceGPUContext *)ctx.handle)[ctx.contextId];
    pcclxGinTroceQp *qps = loadConst(&gda->qps);
    #pragma unroll 1
    for (int peer = coop.thread_rank(); peer < ctx.nRanks; peer += coop.size()) {
      uint64_t prodIdx = pcclx::gin::troce::troceFlushAsync(qps + peer);
      if (prodIdx != 0 && !pcclx::gin::troce::troceWait(qps + peer, prodIdx - 1, abortFlag)) break;
    }
  }
};

template <>
struct pcclGinApi_FlushAsync<PCCLX_NET_DEVICE_GIN_TROCE> {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, int peer, pcclGinRequest_t *outRequest,
                                      uint32_t optFlags) {
    using pccl::utility::loadConst;
    pcclxGinTroceRequest *req = reinterpret_cast<pcclxGinTroceRequest *>(outRequest);
    pcclxGinTroceGPUContext *gda = &((struct pcclxGinTroceGPUContext *)ctx.handle)[ctx.contextId];
    pcclxGinTroceQp *qp = loadConst(&gda->qps) + peer;
    req->peer = peer;
    req->prodIdx = pcclx::gin::troce::troceFlushAsync(qp);
    (void)optFlags;  // flushAsync always submits, including AggregateRequests
  }
};

template <>
struct pcclGinApi_Wait<PCCLX_NET_DEVICE_GIN_TROCE> {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, pcclGinRequest_t &request, bool hasDescriptor,
                                      pcclGinDescriptorSmem *descriptor, hggc::memory_order ord,
                                      uint32_t *abortFlag) {
    (void)ord;
    using pccl::utility::loadConst;
    pcclxGinTroceRequest &req = reinterpret_cast<pcclxGinTroceRequest &>(request);
    pcclxGinTroceGPUContext *gda = &((struct pcclxGinTroceGPUContext *)ctx.handle)[ctx.contextId];
    pcclxGinTroceQp *qp = loadConst(&gda->qps) + req.peer;
    pcclx::gin::troce::troceFlushWait(gda, qp, loadConst(&gda->qps) + ctx.nRanks, req.prodIdx, abortFlag);
    (void)hasDescriptor;
    (void)descriptor;
  }
};

template <>
struct pcclGinApi_GetSignalPtr<PCCLX_NET_DEVICE_GIN_TROCE> {
  PCCL_DEVICE_INLINE static uint64_t* call(pcclGinCtx ctx, pcclGinSignal_t signalId) {
    using pccl::utility::loadConst;
    pcclxGinTroceGPUContext *gda = &((struct pcclxGinTroceGPUContext *)ctx.handle)[ctx.contextId];
    return loadConst(&gda->signals_table.buffer) + signalId;
  }
};

template <>
struct pcclGinApi_ResetSignal<PCCLX_NET_DEVICE_GIN_TROCE> {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, pcclGinSignalDescriptor signal) {
    using pccl::utility::loadConst;
    if (signal.type == PCCL_GIN_SIGNAL_TYPE_VA) {
      pcclWindow_t window = signal.vaSignal.pcclWindow;
      size_t offset = signal.vaSignal.signalOffset - 4096*size_t(loadConst(&window->ginOffset4K));
      uint64_t *p = (uint64_t *)pcclGetLocalPointer(window, offset);
      *p = 0;
    } else {
      pcclxGinTroceGPUContext *gda = &((struct pcclxGinTroceGPUContext *)ctx.handle)[ctx.contextId];
      loadConst(&gda->signals_table.buffer)[signal.indexedSignal.signalId] = 0;
    }
  }
};

// Counter resources are rejected by createContext. These guards also reject
// device-side counter use against a context created without counter resources.
template <>
struct pcclGinApi_GetCounterPtr<PCCLX_NET_DEVICE_GIN_TROCE> {
  PCCL_DEVICE_INLINE static uint64_t* call(pcclGinCtx ctx, pcclGinCounter_t) {
    __builtin_trap();
    return nullptr;
  }
};

template <>
struct pcclGinApi_ResetCounter<PCCLX_NET_DEVICE_GIN_TROCE> {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, pcclGinCounter_t) {
    __builtin_trap();
  }
};

template <>
struct pcclGinApi_PutValue<PCCLX_NET_DEVICE_GIN_TROCE> {
  template <typename Coop, typename T>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, Coop coop, int peer, pcclGinWindow_t dstWin,
                                      size_t dstOff, T srcVal, pcclGinSignalDescriptor signal,
                                      pcclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                      bool hasDescriptor, pcclGinDescriptorSmem *descriptor,
                                      hggc::thread_scope required, hggc::thread_scope given,
                                      uint32_t optFlags = pcclGinOptFlagsDefault) {
    if (ctx.resourceSharingMode == PCCL_GIN_RESOURCE_SHARING_CTA) {
      callMode<hggc::thread_scope_block>(ctx, coop, peer, dstWin, dstOff, srcVal, signal, signalOp, signalOpArg,
        hasDescriptor, descriptor, required, given, optFlags);
    } else {
      callMode<hggc::thread_scope_device>(ctx, coop, peer, dstWin, dstOff, srcVal, signal, signalOp, signalOpArg,
        hasDescriptor, descriptor, required, given, optFlags);
    }
  }

  template <hggc::thread_scope scope, typename Coop, typename T>
  PCCL_DEVICE_INLINE static void callMode(pcclGinCtx ctx, Coop coop, int peer, pcclGinWindow_t dstWin,
                                      size_t dstOff, T srcVal, pcclGinSignalDescriptor signal,
                                      pcclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                      bool hasDescriptor, pcclGinDescriptorSmem *descriptor,
                                      hggc::thread_scope required, hggc::thread_scope given,
                                      uint32_t optFlags = pcclGinOptFlagsDefault) {
    using pccl::utility::loadConst;
    static_assert(sizeof(T) <= 8, "PutValue payload must fit in 8 bytes");
    coop.sync();
    if (coop.thread_rank() == 0) {
      pcclxGinTroceGPUContext *gda =
        &((struct pcclxGinTroceGPUContext *)ctx.handle)[ctx.contextId];
      pcclxGinTroceQp *qp = loadConst(&gda->qps) + peer;

      bool hasSignal = signal.type != PCCL_GIN_SIGNAL_TYPE_NONE;
      uint64_t signalOff = 0;
      __be32 signalRkey = 0;
      if (signal.type == PCCL_GIN_SIGNAL_TYPE_INDEXED) {
        signalOff = sizeof(uint64_t) *
                    (signal.indexedSignal.signalId + loadConst(&gda->signals_table.offset));
        signalRkey = loadConst(loadConst(&gda->signals_table.rkeys) + peer);
      } else if (signal.type == PCCL_GIN_SIGNAL_TYPE_VA) {
        pcclxGinTroceMemHandle *mh = (pcclxGinTroceMemHandle *)signal.vaSignal.signalWindow;
        signalRkey = loadConst(loadConst(&mh->rkeys) + peer);
        signalOff = signal.vaSignal.signalOffset;
      }
      uint64_t signalVal = (signalOp == pcclGinSignalInc) ? 1 : signalOpArg;

      pcclxGinTroceMemHandle *dstMh = (pcclxGinTroceMemHandle *)dstWin;
      __be32 dstRkey = loadConst(loadConst(&dstMh->rkeys) + peer);

      if (required == hggc::thread_scope_system && given > required)
        pcclx::gin::troce::fenceRelease<hggc::thread_scope_system>();

      const bool signalSet = signalOp == pcclxGinSignalSet;
      const uint32_t signalWqebbs = signalSet ? 1u : 2u;
      const uint32_t nWqe = 1u + (hasSignal ? signalWqebbs : 0u);
      uint64_t base = pcclx::gin::troce::reserveWqeSlots<scope>(qp, nWqe, optFlags);
      uint64_t idx = base;

      pcclx::gin::troce::troceWriteRdmaInlineWqe(qp, &srcVal, (uint32_t)sizeof(T), dstOff, dstRkey,
                                              (uint16_t)idx, /*needCqe=*/true,
                                              pcclx::gin::troce::getWqePtr(qp, idx));
      idx++;
      if (hasSignal) {
        if (signalSet) {
          pcclx::gin::troce::troceWriteFlushWqe(qp, signalVal, signalOff, signalRkey, (uint16_t)idx,
                                             /*needCqe=*/true, pcclx::gin::troce::getWqePtr(qp, idx));
        } else {
          pcclx::gin::troce::troceWriteAtomicFaWqe(qp, signalVal, 0,
                                              loadConst(&gda->sink_buffer_lkey), signalOff, signalRkey,
                                              (uint16_t)idx, /*needCqe=*/true,
                                              pcclx::gin::troce::getWqePtr(qp, idx));
        }
        idx += signalWqebbs;
      }
      pcclx::gin::troce::markWqesReady<scope>(qp, base, idx - 1);
      pcclx::gin::troce::submit<scope>(qp, idx, optFlags);
#ifdef PCCLX_DEVICE_GIN_TROCE_ENABLE_DEBUG
      pcclx::gin::troce::troceQuiet(qp, nullptr);
#endif
    }
    coop.sync();
    (void)hasDescriptor;
    (void)descriptor;
  }
};

#endif /* _PCCLX_DEVICE_GIN_TROCE_H_ */
