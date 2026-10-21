/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_RMA_H_
#define PCCL_RMA_H_

#include "pccl_internal.h"
#include "pccl_common.h"
#include "rma/rma_types.h"
#include "rma/rma_ce.h"
#include "rma/rma_proxy.h"

struct ncclKernelPlan;

// RMA engine state, held by ncclComm.
struct ncclRmaState {
  struct ncclRmaProxyState rmaProxyState;
  struct ncclRmaCeState rmaCeState;
};

// True when the proxy engine is needed: a single LSA team can serve every peer
// through CE, so the proxy is only brought up for ranks outside the team.
bool ncclRmaProxyEnabled(ncclComm_t comm);

// True once every engine this communicator relies on has completed its collective
// setup. pcclSignal/pcclWaitSignal carry no window and cannot drive that setup
// themselves, so they consult this before enqueueing.
bool ncclRmaInitialized(ncclComm_t comm);

// Moves RMA tasks from comm->planner into `plan`, splitting them between the CE and
// proxy queues by whether the target rank is reachable over LSA.
ncclResult_t scheduleRmaTasksToPlan(ncclComm_t comm, struct ncclKernelPlan* plan);

// Executes an RMA plan previously built by scheduleRmaTasksToPlan.
ncclResult_t ncclLaunchRma(ncclComm_t comm, struct ncclKernelPlan* plan);

// Per-engine launch entry points used by ncclLaunchRma. When a plan carries both
// proxy and CE tasks the two run in parallel through the CE stream/event pair.
ncclResult_t ncclRmaPut(ncclComm_t comm, struct ncclKernelPlan* plan, hggcStream_t stream);
ncclResult_t ncclRmaWaitSignal(ncclComm_t comm, struct ncclKernelPlan* plan, hggcStream_t stream);

#endif
