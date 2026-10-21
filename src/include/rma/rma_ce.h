/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_RMA_CE_H_
#define PCCL_RMA_CE_H_

#include "pccl_internal.h"
#include "pccl_common.h"
#include "dev_runtime.h"

struct ncclRmaArgs;
struct ncclKernelPlan;

struct ncclRmaCeInitTask {
  struct ncclRmaCeInitTask* next;
  ncclComm_t comm;
};

struct ncclRmaCeCtx {
  ncclComm_t comm;

  // Per-rank sequence number for the signal operations
  uint64_t* signalOpSeqs;
  uint64_t* signalOpSeqsDev;
  // Host buffer to track the expected values of the signals
  uint64_t* signalsHost;

  // Signal memory layout and management
  // Each RMA context allocates a signal buffer with the following layout:
  // - Offsets [0 to nRanks*8-1]: per-rank distinct signals (8 bytes per rank)
  // - Offset [nRanks*8]: shared aggregate signal counter (8 bytes)
  // Total signal buffer size: (nRanks + 1) * 8 bytes
  struct ncclDevrWindow* signalsWin;
  uint64_t* signalsDev;       // per-rank signals
  size_t signalOffset;        // byte offset of signals
};

struct ncclRmaCeState {
  bool initialized;
  int rmaCeCtxCount;
  void** rmaCeCtxs;
  hggcStream_t ceStream;
  hggcEvent_t ceEvent;
};

// CE-specific function declarations
ncclResult_t ncclRmaCeInit(ncclComm_t comm);
ncclResult_t ncclRmaCeFinalize(ncclComm_t comm);
ncclResult_t ncclRmaCePutLaunch(ncclComm_t comm, struct ncclKernelPlan* plan, hggcStream_t stream);
ncclResult_t ncclRmaCeWaitLaunch(ncclComm_t comm, struct ncclKernelPlan* plan, hggcStream_t stream);

#endif
