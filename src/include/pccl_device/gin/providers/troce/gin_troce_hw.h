/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_GIN_GDAKI_TROCE_HW_H_
#define _PCCL_DEVICE_GIN_GDAKI_TROCE_HW_H_

#include <stdint.h>

// troce (T-Head RoCE) hardware descriptor layout: SQ WQE segments, SQ/CQ
// doorbells and the CQE. This is a hardware ABI, so the bitfield layout is
// reproduced verbatim from sailSHMEM's ibgda_troce_device.h; do not reorder or
// resize any field.

struct troce_sqe_ctrl_seg {
  union {
    struct {
      uint32_t q_type : 6;
      uint32_t se : 1;
      uint32_t join_hmq_en : 1;
      uint32_t qpn : 16;
      uint32_t tc : 3;
      uint32_t wqe_size : 4;
      uint32_t rsvd : 1;
    } bs;
    uint32_t value;
  } dw0;

  union {
    struct {
      uint32_t wqe_idx : 16;
      uint32_t inline_offset : 2;
      uint32_t opc : 8;
      uint32_t fm : 3;
      uint32_t in_dat : 1;
      uint32_t ce : 2;
    } bs;
    uint32_t value;
  } dw1;

  union {
    struct {
      uint32_t req_id : 16;
      uint32_t start_handle_wqe_flag : 2;
      uint32_t local_sge_num : 6;
      uint32_t wqe_sig : 8;
    } bs;
    uint32_t value;
  } dw2;

  uint32_t imm_dat_or_invld_key;
  uint32_t pld_len;
};

struct troce_sq_db {
  union {
    struct {
      uint32_t rsv1 : 8;
      uint32_t qid : 16;
      uint32_t rsv0 : 2;
      uint32_t q_type : 6;
    } bs;
    uint32_t value;
  } dw0;

  union {
    struct {
      uint32_t rsv3 : 5;
      uint32_t tc : 3;
      uint32_t rsv2 : 8;
      uint32_t pi : 16;
    } bs;
    uint32_t value;
  } dw1;
};

typedef union {
  struct troce_sq_db db;
  uint64_t value;
} troce_sq_db_u;

struct troce_sqe_data_seg {
  uint64_t addr;
  uint32_t len;
  union {
    struct {
      uint32_t lkey : 28;
      uint32_t signature : 4;
    } bs;
    uint32_t value;
  } dw3;
};

struct troce_raddr_seg {
  uint64_t raddr;
  uint32_t rkey;
  uint32_t len;
};

struct troce_atomic_seg {
  uint64_t swap_or_add_dat;
  uint64_t comp_dat;
};

typedef struct troce_sqe_ctrl_seg pcclTroceCtrlSeg;
typedef struct troce_raddr_seg pcclTroceRaddrSeg;
typedef struct troce_sqe_data_seg pcclTroceDataSeg;
typedef struct troce_atomic_seg pcclTroceAtomicSeg;
typedef troce_sq_db_u pcclTroceSqDb;

#define PCCL_TROCE_16B_PAD(_len) (((0x10 - ((_len) & 0xf))) & 0xf)
#define PCCL_TROCE_ATOMIC_SIZE 8
#define PCCL_TROCE_DIV_ROUND_UP(size, alignment) (((size) + (alignment) - 1) / (alignment))
#define PCCL_TROCE_SEND_WQE_BB 64

enum pcclTroceQType {
  PCCL_PAU_QTYPE_ROCE_SQ = 0,
  PCCL_PAU_QTYPE_ROCE_RQ = 1,
  PCCL_PAU_QTYPE_ROCE_SRQ = 2,
  PCCL_PAU_QTYPE_ROCE_CQ = 3,
};

enum pcclTroceOpcode {
  PCCL_TROCE_OPCODE_RDMA_READ = 0x00,
  PCCL_TROCE_OPCODE_RDMA_WRITE = 0x04,
  PCCL_TROCE_OPCODE_ATOMIC_CS = 0x06,
  PCCL_TROCE_OPCODE_ATOMIC_FA = 0x07,
  PCCL_TROCE_OPCODE_DUMP = 0x13,
  PCCL_TROCE_OPCODE_NOP = 0x14,
};

struct troce_be_cqe {
  uint32_t rsv_0_0;
  uint32_t rsv_0_1;
  uint32_t rsv_0_2;
  uint32_t rsv_0_3;
  uint32_t start_handle_time_hi;
  uint32_t start_handle_time_lo;
  uint32_t timestamp_hi;
  uint32_t timestamp_lo;
  uint32_t rsv_ud_0;
  uint32_t imme_invld_key;
  uint32_t rsv_ud_1;
  uint32_t smac_l;
  union {
    struct {
      uint32_t srqn : 16;
      uint32_t rsv2 : 2;
      uint32_t sig_err_info : 4;
      uint32_t ip_version : 1;
      uint32_t ud_vld : 1;
      uint32_t req_opcode : 8;
    } bs;
    uint32_t value;
  } dw12;

  uint32_t byte_cnt_mini_cqe_num;
  union {
    struct {
      uint32_t status : 8;
      uint32_t rsv3 : 5;
      uint32_t cqe_act_size : 2;
      uint32_t solicited : 1;
      uint32_t wqe_idx : 16;
    } bs;
    uint32_t value;
  } dw14;

  union {
    struct {
      uint32_t qpn : 16;
      uint32_t signature : 8;
      uint32_t owner : 1;
      uint32_t cqe_format : 2;
      uint32_t rsv4 : 1;
      uint32_t cqe_opcode : 4;
    } bs;
    uint32_t value;
  } dw15;
};

struct troce_cq_db {
  union {
    struct {
      uint32_t rsv1 : 8;
      uint32_t cqid : 16;
      uint32_t rsv0 : 2;
      uint32_t q_type : 6;
    } bs;
    uint32_t value;
  } dw0;

  union {
    struct {
      uint32_t rsv_0 : 2;
      uint32_t cmdsn : 2;
      uint32_t rsv3 : 3;
      uint32_t cmd : 1;
      uint32_t rsv2 : 8;
      uint32_t ci : 16;
    } bs;
    uint32_t value;
  } dw1;
};

typedef union {
  struct troce_cq_db db;
  uint64_t value;
} pcclTroceCqDb;

#define PCCL_TROCE_CQ_DB_REQ_NOT 0

#define PCCL_TROCE_CQE_OWNER_MASK 0x01000000u
#define PCCL_TROCE_CQE_OWNER_SHIFT 24
#define PCCL_TROCE_CQE_OPCODE_MASK 0xF0000000u
#define PCCL_TROCE_CQE_OPCODE_SHIFT 28
#define PCCL_TROCE_CQE_WQE_IDX_MASK 0xFFFF0000u
#define PCCL_TROCE_CQE_WQE_IDX_SHIFT 16

enum pcclTroceCqeOpcode {
  PCCL_TROCE_CQE_REQ = 0x0,
  PCCL_TROCE_CQE_REQ_ERR = 0x5,
  PCCL_TROCE_CQE_RESP_ERR = 0x6,
  PCCL_TROCE_CQE_INVALID = 0xf,
};

#endif /* _PCCL_DEVICE_GIN_GDAKI_TROCE_HW_H_ */
