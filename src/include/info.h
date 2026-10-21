/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INFO_H_
#define NCCL_INFO_H_

#include "pccl_internal.h"
#include "devcomm.h"
#include "strongstream.h"
#include "utils.h"

typedef enum : uint8_t {
  ncclPatternRing,
  ncclPatternRingTwice,
  ncclPatternPipelineFrom,
  ncclPatternPipelineTo,
  ncclPatternTreeUp,
  ncclPatternTreeDown,
  ncclPatternTreeUpDown,
  ncclPatternCollTreeUp,
  ncclPatternCollTreeDown,
  ncclPatternCollTreeUpDown,
  ncclPatternPatUp,
  ncclPatternPatDown,
  ncclPatternSend,
  ncclPatternRecv,
  ncclPatternProfiler,
} ncclPattern_t;

// Used to pass NCCL call information between functions
struct ncclInfo {
  ncclFunc_t coll;
  const char* opName;
  // NCCL Coll Args
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t datatype;
  ncclRedOp_t op;
  int root;
  ncclComm_t comm;
  hggcStream_t stream;
  // Algorithm details
  int chunkSteps;
  int sliceSteps;
  // One-sided ops: sigIdx in [0, numRmaSig), ctx in [0, numRmaCtx). PutSignal
  // requires localbuff and peerWin in PCCL_WIN_COLL_SYMMETRIC windows.
  size_t peerWinOffset;
  ncclWindow_t peerWin;
  int sigIdx;
  int ctx;
  unsigned int flags;
  // WaitSignal descriptors; sigIdx/ctx ranges as above.
  int nDesc;
  ncclWaitSignalDesc_t* signalDescs;
  // Computed later
  struct ncclDevRedOpFull opFull;
  int algorithm;
  int protocol;
  uint16_t funcIndex;
  ncclPattern_t pattern;
  int nChannels;
  int nThreads;
  size_t nBytes;
  int nstepsPerLoop;
  int nchunksPerLoop;
  ssize_t sendbytes;
  ssize_t recvbytes;
  int recvChunkSize;
  int sendChunkSize;
  int channelId;
};

#endif
