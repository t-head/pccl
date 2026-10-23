/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_GIN_GDAKI_TROCE_DEVICE_HOST_COMMON_H_
#define _PCCL_DEVICE_GIN_GDAKI_TROCE_DEVICE_HOST_COMMON_H_

#include <linux/types.h>
#include <stdint.h>

// Host/device ABI version; increment when the shared layout changes.
#define PCCL_GIN_TROCE_VERSION 100

// Number of uint64 bookkeeping words per QP in the device index pool:
// resv_head, ready_head, prod_idx, cons_idx, progress_lock, db_lock.
#define PCCL_GIN_TROCE_QP_IDX_WORDS 6

// SQ capacity constant expressed in WQEBBs.
#define PCCL_GIN_GDAKI_TROCE_MAX_SQ_WQES 32768

// Per-context view of device-resident signal or counter storage. Remote keys
// are indexed by peer rank; offset selects this context's slice of the table.
template <typename T>
struct pcclGinGdakiTroceGlobalGPUBufferTable {
  T *buffer;
  __be32 *rkeys;
  __be32 lkey;
  unsigned int offset;
};

// Device-visible QP state for WQE reservation, publication, submission, and
// completion progress. Shared indices reside in device global memory and count
// WQEBBs across all CTAs using the QP.
struct pcclGinGdakiTroceQp {
  uint32_t qpn;
  uint32_t nwqes;          // SQ depth in WQEBBs, power of two
  uint32_t traffic_class;  // QP traffic class encoded in the SQ doorbell.
  uint32_t cqn;
  uint32_t ncqes;         // Logical CQ capacity; one in single-slot mode.

  void *sq_wqe;  // SQ buffer base (device memory)
  void *sq_db;   // SQ doorbell (mapped MMIO)
  void *cq_cqe;  // CQ buffer base (device memory)
  void *cq_db;   // CQ doorbell (mapped MMIO)

  uint64_t *resv_head;      // last reserved WQEBB idx + 1
  uint64_t *ready_head;     // last contiguously published WQEBB idx + 1
  uint64_t *prod_idx;       // Submitted WQEBB tail; updated under db_lock.
  uint64_t *cons_idx;       // Retired WQEBB tail after completion-counter updates.
  uint64_t *progress_lock;  // serializes CQ consumption and counter updates
  uint64_t *db_lock;        // per-QP doorbell ownership: 0 free, 1 held
  uint64_t sink_addr;       // byte offset into the sink MR for this QP's atomic result
  // Local completion counters are updated by CQ progress. The pending slot
  // for the batch's completion WQE holds counterId+1; retiring that slot adds
  // one to the corresponding entry in counters_buf.
  uint32_t *pending_counter;  // nwqes entries, 0 = no counter for that WQE
  uint64_t *counters_buf;     // this context's counters_table base
};


struct pcclGinGdakiTroceGPUContext {
  struct pcclGinGdakiTroceQp *qps; // one per peer rank
  int nRanks;

  struct pcclGinGdakiTroceGlobalGPUBufferTable<uint64_t> counters_table;
  struct pcclGinGdakiTroceGlobalGPUBufferTable<uint64_t> signals_table;

  // Atomic return storage; fetched values are not consumed.
  uint64_t *sink_buffer;
  __be32 sink_buffer_lkey;
};

struct pcclGinGdakiTroceMemHandle {
  __be32 *rkeys;
  __be32 lkey;
};

#endif /* _PCCL_DEVICE_GIN_GDAKI_TROCE_DEVICE_HOST_COMMON_H_ */
