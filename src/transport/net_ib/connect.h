/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_IB_CONNECT_H_
#define NET_IB_CONNECT_H_

#include "common.h"
#include "ibvwrap.h"

// Per-QP connection metatdata
struct ncclIbQpInfo {
  uint32_t lid;
  uint8_t ib_port;
  uint32_t qpn[NCCL_IB_MAX_QPS];

  // For RoCE
  uint64_t spn;
  uint64_t iid;
  enum ibv_mtu mtu;

  // FIFO RDMA info
  uint32_t fifoRkey;
  uint64_t fifoAddr;

  // OOO RQ capability used by the peer during data QP setup.
  // TODO: Remove this compatibility field after porting NCCL's per-device metadata
  // exchange to PCCL; OOO RQ capability should then come from that metadata.
  uint32_t oooRqSize;
};

ncclResult_t ncclIbCreateQp(uint8_t ib_port, struct ncclIbNetCommDevBase* commDevBase, int access_flags,
                            int oooRq, struct ibv_qp** qp, void* qp_context);
ncclResult_t ncclIbRtrQp(ibv_qp* qp, uint32_t qpn, struct ncclIbQpInfo* info, int gidIndex);
ncclResult_t ncclIbRtsQp(ibv_qp* qp);
void pcclIbSetSendTxAffinity(struct ncclIbSendComm* comm, int numDataQps);

// Phase-2 connection setup: complete sender/receiver QP transition.
// Must be called after ncclIbConnect/ncclIbAccept before posting WRs directly.
ncclResult_t ncclSendCheck(struct ncclIbSendComm* comm, int numDataQps);
ncclResult_t ncclRecvCheck(struct ncclIbRecvComm* comm);

#endif //NET_IB_CONNECT_H_
