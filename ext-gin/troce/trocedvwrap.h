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

// Forward declarations avoid conflicts between PCCL's ibvwrap.h types and the
// system verbs header. Provider entry points use these types only by pointer.
struct ibv_pd;
struct ibv_qp;
struct ibv_cq;
struct ibv_context;
struct ibv_comp_channel;
struct ibv_qp_init_attr;

// Private TROCE queue-allocation ABI. Field types and order must remain
// compatible with libtroce. Resolve entry points at runtime so other providers
// remain usable without this optional library.
//
// TROCE_ALLOC_GPU accepts caller-allocated device queue buffers and returns
// the MMIO registers required for direct device submission and CQ management.

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

// IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN from enum ibv_create_cq_attr_flags.
// Declare the value locally to avoid including the system verbs header.
// Used with the provider's single-slot completion mode.
#define TROCE_CQ_ATTR_IGNORE_OVERRUN (1u << 1)

enum troce_queue_alloc_type {
  TROCE_ALLOC_DRIVER = 0, /* normal mode: driver owns the rings */
  TROCE_ALLOC_GPU = 1,    /* gpu mode: caller supplies device-memory rings */
};

struct troce_qp_context {
  // Integer field required by the provider ABI; not an enum-typed member.
  int alloc_type; /* input */
  void *sq_addr;  /* input, sq ring buffer address */
  size_t sq_len;  /* input, sq ring buffer length */

  void *rq_addr; /* input, rq ring buffer address */
  size_t rq_len; /* input, rq ring buffer length */

  // Input RQ doorbell-record address in device memory. NULL selects provider
  // allocation. Retain this member to preserve the ABI offsets of output fields.
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

// Initialize the provider function table. Return ncclSuccess only when all
// required symbols are available; cache the result until Fini resets the table.
ncclResult_t ncclTroceFtableInit(struct ncclTroceFunctionTable **ftable);
void ncclTroceFtableFini();

#endif /* _TROCEDVWRAP_H_ */
