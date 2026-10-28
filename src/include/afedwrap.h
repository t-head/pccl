/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_AFEDWRAP_H_
#define PCCL_AFEDWRAP_H_

#include "core.h"
#include <stdint.h>

// Keep afed optional on SDKs without infiniband/tx_affinity.h.
struct ibv_context;
struct ibv_qp;

ncclResult_t wrap_afed_symbols(void);
// Preserve provider return codes for the transport's best-effort diagnostics.
int wrap_afed_query_num_lag_ports(struct ibv_context* context, uint8_t* num_lag_ports);
int wrap_afed_set_qp_tx_affinity(struct ibv_qp* qp, uint8_t num_lag_ports, uint8_t expect_egress);

#endif // PCCL_AFEDWRAP_H_
