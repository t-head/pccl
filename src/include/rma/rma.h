/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_RMA_H_
#define PCCL_RMA_H_

#include "pccl_internal.h"
#include "pccl_common.h"
#include "rma/rma_ce.h"

struct ncclKernelPlan;

// Internal signal mode enum
typedef enum {
  NCCL_SIGNAL_NONE = 0,        // No signaling
  NCCL_SIGNAL = 1              // Default signal operation
} ncclSignalMode_t;

struct ncclRmaArgs{
  int ctx;
  ncclFunc_t func;
  int nRmaTasks;
  int nRmaTasksCe;
};

struct ncclRmaState {
  struct ncclRmaCeState rmaCeState;
};

bool ncclRmaInitialized(struct pcclComm* comm);

// Main RMA function declarations
ncclResult_t scheduleRmaTasksToPlan(struct pcclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t ncclLaunchRma(struct pcclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t ncclRmaWaitSignal(struct pcclComm* comm, struct ncclKernelPlan* plan, hggcStream_t stream);
ncclResult_t ncclRmaPut(struct pcclComm* comm, struct ncclKernelPlan* plan, hggcStream_t stream);

#endif
