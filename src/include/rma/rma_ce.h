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

  // Per-rank sequence number for non-graph signal operations
  uint64_t* signalOpSeqs;
  uint64_t* signalOpSeqsDev;
  // Host buffer to track the expected values of the non-graph signals
  uint64_t* signalsHost;

  // Single symmetric window for all signal and ack memory.
  // Layout (all uint64_t slots):
  //   [0 .. nRanks-1]              non-graph per-rank signals
  //   [nRanks]                     non-graph aggregate signal
  //   [nRanks+1 .. 2*nRanks]       graph per-rank signals
  //   [2*nRanks+1]                 graph aggregate signal
  //   [2*nRanks+2 .. 3*nRanks+1]   graph per-rank ack flags
  // Total: (3*nRanks + 2) * sizeof(uint64_t)
  struct ncclDevrWindow* signalsWin;
  uint64_t* signalsDev;       // non-graph per-rank signals
  uint64_t* graphSignalsDev;  // graph per-rank signals
  uint64_t* graphAckDev;      // graph per-rank ack flags
  size_t signalOffset;        // byte offset of non-graph signals
  size_t graphSignalOffset;   // byte offset of graph signals
  size_t graphAckOffset;      // byte offset of graph ack flags

  // Device-resident constants for graph-safe D2D signal/ack writes
  uint64_t* signalConstDev;
  uint64_t* signalConstOneDev;
  uint64_t* signalConstZeroDev;
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
