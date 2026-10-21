/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
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

struct ncclRmaCeCtx {
  ncclComm_t comm;

  // CE only targets intra-node (LSA) peers, so all signal state is indexed by
  // LSA-local rank: lsaSize * numRmaSig slots.
  // Per-(sigIdx, rank) sequence numbers for non-graph signal operations
  uint64_t* signalOpSeqs;
  // Host buffer to track the expected values of the non-graph signals
  uint64_t* signalsHost;
  // Device staging slots for non-graph signal values, with capacity comm->nRanks
  uint64_t* signalOpSeqsDev;

  // Single symmetric window for all signal and ack memory.
  // signalSlots = lsaSize * numRmaSig
  // Layout (all uint64_t slots):
  //   [0 .. signalSlots-1]                 non-graph per-(sigIdx, rank) signals
  //   [signalSlots .. 2*signalSlots-1]     graph per-(sigIdx, rank) signals
  //   [2*signalSlots .. 3*signalSlots-1]   graph per-(sigIdx, rank) ack flags
  // Total: 3 * signalSlots * sizeof(uint64_t)
  struct ncclDevrWindow* signalsWin;
  uint64_t* signalsDev;       // non-graph per-(sigIdx, rank) signals
  uint64_t* graphSignalsDev;  // graph per-(sigIdx, rank) signals
  uint64_t* graphAckDev;      // graph per-(sigIdx, rank) ack flags
  size_t signalOffset;        // byte offset of non-graph signals
  size_t graphSignalOffset;   // byte offset of graph signals
  size_t graphAckOffset;      // byte offset of graph ack flags

  // Device-resident constants for graph-safe D2D signal/ack writes
  uint64_t* signalConstDev;
  uint64_t* signalConstOneDev;
  uint64_t* signalConstZeroDev;
};

struct ncclRmaCeState {
  // initInProgress guards re-entry while ncclRmaCeInit registers its own
  // signal window through ncclDevrWindowRegisterInGroup.
  bool initialized;
  bool initInProgress;
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
