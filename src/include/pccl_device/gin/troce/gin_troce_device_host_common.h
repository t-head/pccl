/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_GIN_TROCE_DEVICE_HOST_COMMON_H_
#define _PCCL_DEVICE_GIN_TROCE_DEVICE_HOST_COMMON_H_

#include <linux/types.h>
#include <stdint.h>

// Host/device ABI version; increment when the shared layout changes.
#define PCCL_GIN_TROCE_VERSION 100

// SQ capacity constant expressed in WQEBBs.
#define PCCL_GIN_TROCE_MAX_SQ_WQES 32768
#define PCCL_GIN_TROCE_MAX_TRANSFER_SIZE (1UL << 30)

// Per-context view of device-resident signal storage. Remote keys
// are indexed by peer rank; offset selects this context's slice of the table.
template <typename T>
struct pcclGinTroceGlobalGPUBufferTable {
  T *buffer;
  __be32 *rkeys;
  __be32 lkey;
  unsigned int offset;
};

// Device-visible QP state for WQE reservation, publication, submission, and
// completion progress. Indices count WQEBBs across all CTAs using the QP.
struct pcclGinTroceQp {
  uint64_t resv_head;       // last reserved WQEBB idx + 1
  uint64_t ready_head;      // last contiguously published WQEBB idx + 1
  uint64_t prod_idx;        // submitted WQEBB tail
  uint64_t cons_idx;        // retired WQEBB tail
  int db_lock;             // per-QP doorbell ownership: 0 free, 1 held
  uint32_t qpn;
  uint32_t nwqes;          // SQ depth in WQEBBs, power of two
  uint32_t traffic_class;  // QP traffic class encoded in the SQ doorbell.
  uint32_t cqn;
  __be32 sink_rkey;        // Remote sink MR key.

  void *sq_wqe;  // SQ buffer base (device memory)
  void *sq_db;   // SQ doorbell (mapped MMIO)
  void *cq_cqe;  // CQ buffer base (device memory)
};

struct pcclGinTroceGPUContext {
  struct pcclGinTroceQp *qps;  // one per peer rank, followed by the loopback flush QP
  struct pcclGinTroceGlobalGPUBufferTable<uint64_t> signals_table;

  // Atomic return and zero-byte flush storage; values are not consumed.
  __be32 sink_buffer_lkey;
};

struct pcclGinTroceMemHandle {
  __be32 *rkeys;
  __be32 lkey;
};

#endif /* _PCCL_DEVICE_GIN_TROCE_DEVICE_HOST_COMMON_H_ */
