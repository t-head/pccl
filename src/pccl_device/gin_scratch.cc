/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "core.h"
#include "../device/symmetric/gin_scratch__funcs.h"

NCCL_API(ncclResult_t, pcclGinOutboxCreateRequirement,
  int nBlocks, int size_log2,
  pcclGinOutboxHandle* outHandle, pcclDevResourceRequirements* outReq
);

ncclResult_t pcclGinOutboxCreateRequirement(
    int nBlocks, int size_log2,
    pcclGinOutboxHandle* outHandle, pcclDevResourceRequirements* outReq
  ) {
  memset(outReq, 0, sizeof(*outReq));
  size_log2 = std::max<int>(size_log2, /*log2(128)=*/7);
  outHandle->size_log2 = size_log2;
  outReq->bufferSize = nBlocks*(sizeof(ncclGinOutboxState) + alignUp(1<<size_log2, alignof(ncclGinOutboxState)));
  outReq->bufferAlign = 128;
  outReq->outBufferHandle = &outHandle->bufHandle;
  outReq->ginCounterCount = nBlocks << ncclGinScratchMaxBufs_log2;
  outReq->outGinCounterStart = &outHandle->counter0;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, pcclGinInboxA2ACreateRequirement,
  pcclTeam peers, int nBlocks, int size_log2,
  pcclGinInboxA2AHandle* outHandle, pcclDevResourceRequirements* outReq
);

ncclResult_t pcclGinInboxA2ACreateRequirement(
    pcclTeam peers, int nBlocks, int size_log2,
    pcclGinInboxA2AHandle* outHandle, pcclDevResourceRequirements* outReq
  ) {
  int nPeers = peers.nRanks - 1;
  memset(outReq, 0, sizeof(*outReq));
  outHandle->size_log2 = size_log2;
  outHandle->nPeers_rcp32 = idivRcp32(nPeers);
  outReq->bufferSize = nBlocks*(sizeof(ncclGinInboxA2AState) + alignUp(1<<size_log2, alignof(ncclGinInboxA2AState)));
  outReq->bufferAlign = 128;
  outReq->outBufferHandle = &outHandle->bufHandle;
  outReq->ginSignalCount = nBlocks*4*(nPeers + (1<<ncclGinScratchMaxBufs_log2));
  outReq->outGinSignalStart = &outHandle->signals;
  return ncclSuccess;
}
