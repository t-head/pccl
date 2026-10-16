/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _TROCEDVWRAP_H_
#define _TROCEDVWRAP_H_

#include <stddef.h>
#include <stdint.h>

// NOTE: do not include <infiniband/verbs.h> here. PCCL's ibvwrap.h carries its
// own copy of the ibv type declarations, and pulling in the system header as
// well makes every shared type a redefinition. The troce entry points only take
// ibv objects by pointer, so forward declarations are enough.
struct ibv_pd;
struct ibv_qp;
struct ibv_cq;
struct ibv_context;
struct ibv_comp_channel;
struct ibv_qp_init_attr;

// Private troce (T-Head RoCE) queue-allocation ABI.
//
// libtroce ships no development headers, so the structures below mirror the ABI
// used by sailSHMEM's IBGDA transport (src/modules/transport/ibgda/troce_ibgda.h)
// and the entry points are resolved with dlopen/dlsym at runtime: PCCL must keep
// building and running on hosts without libtroce.
//
// The interesting part is TROCE_ALLOC_GPU: it lets the caller supply the SQ/RQ/CQ
// ring buffers (in device memory) and returns the matching tail registers, which
// is exactly what a GPU-initiated path needs in order to post WQEs and ring the
// doorbell without CPU involvement.

#ifdef __cplusplus
extern "C" {
#endif

#define TROCE_SEND_WQE_BB 64
#define TROCE_SEND_WQE_BB_FACTOR 4
#define TROCE_SQ_MIN_WQEBB 32UL
#define TROCE_SQ_ALIGN 1024UL
#define TROCE_SQ_UAR_SIZE 8

#define TROCE_RECV_WQE_BB 16
#define TROCE_MIN_RQ_SIZE 64UL
#define TROCE_RQ_ALIGN 4096UL

#define TROCE_CQ_UAR_SIZE 4
#define TROCE_CQE_SIZE 64

// Value of IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN from <infiniband/verbs.h>
// (enum ibv_create_cq_attr_flags). Declared locally because this header must
// not pull in the system verbs header. Used for the collapsed single-slot CQ:
// the NIC overwrites CQE #0 on every completion and never stalls on CQ-full.
#define TROCE_CQ_ATTR_IGNORE_OVERRUN (1u << 1)

enum troce_queue_alloc_type {
  TROCE_ALLOC_DRIVER = 0, /* normal mode: driver owns the rings */
  TROCE_ALLOC_GPU = 1,    /* gpu mode: caller supplies device-memory rings */
};

struct troce_qp_context {
  // Declared as plain int in the driver's trocedv.h, not as the enum type. Keep
  // it an int so assigning arbitrary values (e.g. for validation probes) is
  // legal and the ABI matches exactly.
  int alloc_type; /* input */
  void *sq_addr;  /* input, sq ring buffer address */
  size_t sq_len;  /* input, sq ring buffer length */

  void *rq_addr; /* input, rq ring buffer address */
  size_t rq_len; /* input, rq ring buffer length */

  // input, GPU VRAM address for the RQ doorbell record. Leaving it NULL makes
  // the provider allocate the record itself. The field must be present even
  // when unused: omitting it shifts sq_tail_reg/rq_tail_reg by one slot and
  // makes the provider write sq_tail_reg into this slot (so the caller reads
  // NULL) and rq_tail_reg past the end of this struct (memory corruption).
  // This mirrors sailSHMEM's troce_ibgda.h layout exactly.
  void *recv_db_addr;

  void *sq_tail_reg; /* output, the tail reg of sq */
  void *rq_tail_reg; /* output, the tail reg of rq */
};

struct troce_cq_context {
  int alloc_type;    /* input */
  void *addr;        /* input, cq ring buffer address */
  size_t len;        /* input, cq ring buffer length */
  void *cq_tail_reg; /* output, the tail reg of cq */
  uint32_t cqn;      /* output, return cqn */
  uint32_t flags;    /* input, e.g. IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN */
};

enum trocedv_qp_attr_mask {
  TROCEDV_QP_ATTR_TC = 1U << 0,
};

struct trocedv_qp_attr {
  uint32_t attr_mask;
  uint32_t actual_mask;
  uint8_t tc;
};

struct ncclTroceFunctionTable {
  struct ibv_qp *(*troce_create_qp_with_context)(struct ibv_pd *pd,
                                                 struct ibv_qp_init_attr *attr,
                                                 struct troce_qp_context *qp_context);
  int (*troce_destroy_qp_with_context)(struct ibv_qp *ibqp,
                                       struct troce_qp_context *qp_context);
  struct ibv_cq *(*troce_create_cq_with_context)(struct ibv_context *context, int cqe,
                                                 struct ibv_comp_channel *channel,
                                                 int comp_vector,
                                                 struct troce_cq_context *cq_context);
  int (*troce_destroy_cq_with_context)(struct ibv_cq *cq, struct troce_cq_context *cq_context);
  int (*trocedv_query_qp_attr)(struct ibv_qp *qp, struct trocedv_qp_attr *qp_attr);
};

#ifdef __cplusplus
}
#endif

// Resolve libtroce once per process. Returns ncclSuccess only when every symbol
// above was found; on failure the table is zeroed and the caller must fall back
// to a non-GDA path.
ncclResult_t ncclTroceFtableInit(struct ncclTroceFunctionTable **ftable);
void ncclTroceFtableFini();

#endif /* _TROCEDVWRAP_H_ */
