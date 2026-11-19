/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "connect.h"
#include "afedwrap.h"
#include <atomic>

PCCL_PARAM(IbTxAffinityEnable, "IB_TX_AFFINITY_ENABLE", 1);

// Shared by NET and proxy GIN connections within this process.
static std::atomic<unsigned int> nextTxAffinity[MAX_IB_DEVS] = {};

void pcclIbSetSendTxAffinity(struct ncclIbSendComm* comm, int numDataQps) {
  // The connection's QP count comes from IB_QPS_PER_CONNECTION. Single-QP
  // connections retain the device's default selection without loading afed.
  if (pcclParamIbTxAffinityEnable() == 0 || comm->base.nqps < 2) return;
  if (numDataQps <= 0 || numDataQps > comm->base.nqps) return;
  const ncclIbDev* dev = &ncclIbDevs[comm->netDev];
  if (dev->ibProvider != IB_PROVIDER_MLX5 || dev->link != IBV_LINK_LAYER_ETHERNET) return;
  if (wrap_afed_symbols() != ncclSuccess) return;

  uint8_t num_lag_ports = 0;
  int ret = wrap_afed_query_num_lag_ports(dev->context, &num_lag_ports);
  if (ret != 0) {
    VERBOSE(NCCL_NET, "NET/IB: TX affinity query failed on %s: rc=%d, keeping default selection", dev->devName, ret);
    return;
  }
  if (num_lag_ports != 2 && num_lag_ports != 4) {
    VERBOSE(NCCL_NET, "NET/IB: TX affinity skipped on %s: num_lag_ports=%u", dev->devName, num_lag_ports);
    return;
  }
  // Only send data QPs reach here, after all RESET -> INIT -> RTR -> RTS
  // transitions. Receive-side FIFO and GPU flush QPs retain their defaults.
  // Reserve only QPs carrying data: proxy GIN uses qps[0] exclusively.
  // The counter allocates port slots, so it needs no cross-thread publication.
  unsigned int first = nextTxAffinity[comm->netDev].fetch_add(numDataQps, std::memory_order_relaxed);
  for (int q = 0; q < numDataQps; q++) {
    struct ibv_qp* qp = comm->base.qps[q];
    uint8_t egress = (first + q) % num_lag_ports;
    ret = wrap_afed_set_qp_tx_affinity(qp, num_lag_ports, egress);
    if (ret != 0) {
      VERBOSE(NCCL_NET, "NET/IB: TX affinity failed: dev=%s qp_num=%u q=%d num_lag_ports=%u expect_egress=%u rc=%d",
              dev->devName, qp->qp_num, q, num_lag_ports, egress, ret);
    } else {
      VERBOSE(NCCL_NET, "NET/IB: TX affinity set: dev=%s qp_num=%u q=%d num_lag_ports=%u expect_egress=%u",
              dev->devName, qp->qp_num, q, num_lag_ports, egress);
    }
  }
}
