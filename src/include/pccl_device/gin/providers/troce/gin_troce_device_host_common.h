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

// Version of the host/device ABI below. Bump on any layout change.
#define PCCL_GIN_TROCE_VERSION 101

// Maximum number of WQEBBs a single SQ may hold. Mirrors the GDAKI queue-depth
// default so the two backends can be compared with identical configuration.
#define PCCL_GIN_GDAKI_TROCE_MAX_SQ_WQES 32768

// Per-context global buffer (signals / counters). Same shape as the GDAKI
// table: the buffer lives in device memory, rkeys[] is indexed by peer rank and
// `offset` is the start of this context's slice inside the global table.
template <typename T>
struct pcclGinGdakiTroceGlobalGPUBufferTable {
  T *buffer;
  __be32 *rkeys;
  __be32 lkey;
  unsigned int offset;
};

// Device-visible queue-pair view. This is the GDAKI_TROCE counterpart of
// `doca_gpu_dev_verbs_qp`: everything the GPU needs in order to reserve a WQE
// slot, fill it, ring the doorbell and poll the send CQ.
//
// Mutable indices (resv/ready/prod/cons) live in device global memory so they
// can be shared by every CTA using this QP. All of them count WQEBBs and are
// monotonically increasing, exactly like the GDAKI `sq_rsvd_index`.
struct pcclGinGdakiTroceQp {
  uint32_t qpn;
  uint32_t nwqes;         // SQ depth in WQEBBs, power of two
  uint32_t traffic_class; // troce doorbell carries the TC of the QP
  uint32_t cqn;
  uint32_t ncqes;         // CQ depth in CQEs, power of two

  void *sq_wqe;     // SQ buffer base (device memory)
  void *sq_db;      // SQ doorbell (mapped MMIO)
  void *cq_cqe;     // CQ buffer base (device memory)
  void *cq_db;      // CQ doorbell (mapped MMIO)

  uint64_t *resv_head;  // last reserved WQEBB idx + 1
  uint64_t *ready_head; // last filled WQEBB idx + 1
  uint64_t *prod_idx;   // last doorbell-ed WQEBB idx + 1
  uint64_t *cons_idx;   // last CQ-polled WQEBB idx + 1
  uint64_t *progress_lock; // serializes CQ consumption and counter updates
  uint64_t sink_addr;   // byte offset into the sink MR for this QP's atomic result
  // Local completion counters. troce has no WAIT WQE and cannot bind a completion
  // on one QP to work on another, so a counter cannot be bumped autonomously by
  // the NIC (an atomic issued on this QP would target the peer, not us). Instead
  // the CQ poll does it in software: pending_counter[wqe_idx] holds counterId+1
  // for the WQE that carries the batch's CQE, and the poll turns that into an
  // atomicAdd on counters_buf.
  uint32_t *pending_counter;  // nwqes entries, 0 = no counter for that WQE
  uint64_t *counters_buf;     // this context's counters_table base
};


struct pcclGinGdakiTroceGPUContext {
  struct pcclGinGdakiTroceQp *qps; // one per peer rank
  int nRanks;

  struct pcclGinGdakiTroceGlobalGPUBufferTable<uint64_t> counters_table;
  struct pcclGinGdakiTroceGlobalGPUBufferTable<uint64_t> signals_table;

  // Local buffer we don't consume but is required to receive the fetched value
  // of remote atomics. Mirrors the GDAKI sink buffer.
  uint64_t *sink_buffer;
  __be32 sink_buffer_lkey;
};

struct pcclGinGdakiTroceMemHandle {
  __be32 *rkeys;
  __be32 lkey;
};

#endif /* _PCCL_DEVICE_GIN_GDAKI_TROCE_DEVICE_HOST_COMMON_H_ */
