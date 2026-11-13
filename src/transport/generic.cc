/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "transport.h"

ncclResult_t ncclTransportRingConnect(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  uint64_t devMemUsedStart = comm->devMemUsed;
  uint64_t pinnedMemUsedStart = comm->pinnedMemUsed;
  uint64_t sharedDevMemUsedStart = comm->sharedRes->devMemUsed;
  uint64_t sharedPinnedMemUsedStart = comm->sharedRes->pinnedMemUsed;

  if (comm && comm->nRanks > 1) {
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channel, 1, &channel->ring.prev, 1, &channel->ring.next, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 0), ret, fail);
  }

  VERBOSE(NCCL_ALLOC, "comm %p Ring totally used %ldB %ldB(sharedRes) device memory splitShare %d nranks %d rank %d", comm, comm->devMemUsed - devMemUsedStart, comm->sharedRes->devMemUsed - sharedDevMemUsedStart, comm->config.splitShare, comm->nRanks, comm->rank);
  VERBOSE(NCCL_ALLOC, "comm %p Ring totally used %ldB %ldB(sharedRes) pinned memory splitShare %d nranks %d rank %d", comm, comm->pinnedMemUsed - pinnedMemUsedStart, comm->sharedRes->pinnedMemUsed - sharedPinnedMemUsedStart, comm->config.splitShare, comm->nRanks, comm->rank);
  INFO(NCCL_INIT, "Connected all rings");
exit:

  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportTreeConnect(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  uint64_t devMemUsedStart = comm->devMemUsed;
  uint64_t pinnedMemUsedStart = comm->pinnedMemUsed;
  uint64_t sharedDevMemUsedStart = comm->sharedRes->devMemUsed;
  uint64_t sharedPinnedMemUsedStart = comm->sharedRes->pinnedMemUsed;

  if (comm && comm->nRanks > 1) {
    // Connect Trees
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channel, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up, 0), ret, fail);
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channel, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    VERBOSE(NCCL_ALLOC, "comm %p Tree totally used %ldB %ldB(sharedRes) device memory splitShare %d nranks %d rank %d", comm, comm->devMemUsed - devMemUsedStart, comm->sharedRes->devMemUsed - sharedDevMemUsedStart, comm->config.splitShare, comm->nRanks, comm->rank);
    VERBOSE(NCCL_ALLOC, "comm %p Tree totally used %ldB %ldB(sharedRes) pinned memory splitShare %d nranks %d rank %d", comm, comm->pinnedMemUsed - pinnedMemUsedStart, comm->sharedRes->pinnedMemUsed - sharedPinnedMemUsedStart, comm->config.splitShare, comm->nRanks, comm->rank);
    INFO(NCCL_INIT, "Connected all trees");
  }
exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportPatConnect(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  uint64_t devMemUsedStart = comm->devMemUsed;
  uint64_t pinnedMemUsedStart = comm->pinnedMemUsed;
  uint64_t sharedDevMemUsedStart = comm->sharedRes->devMemUsed;
  uint64_t sharedPinnedMemUsedStart = comm->sharedRes->pinnedMemUsed;

  if (comm && comm->nRanks > 1) {
    for (int mask = 1; mask < comm->nRanks; mask <<= 1) {
      int prevPeer = (comm->rank + mask) % comm->nRanks;
      int nextPeer = (comm->rank + comm->nRanks - mask) % comm->nRanks;
      for (int c = 0; c < comm->nChannels; c++) {
        struct ncclChannel* channel = comm->channels + c;
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channel, 1, &prevPeer, 1, &nextPeer, 0), ret, fail); // ReduceScatter
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
      for (int c = 0; c < comm->nChannels; c++) {
        struct ncclChannel* channel = comm->channels + c;
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channel, 1, &nextPeer, 1, &prevPeer, 0), ret, fail); // AllGather
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    }
  }
  VERBOSE(NCCL_ALLOC, "comm %p PAT totally used %ldB %ldB(sharedRes) device memory splitShare %d nranks %d rank %d", comm, comm->devMemUsed - devMemUsedStart, comm->sharedRes->devMemUsed - sharedDevMemUsedStart, comm->config.splitShare, comm->nRanks, comm->rank);
  VERBOSE(NCCL_ALLOC, "comm %p PAT totally used %ldB %ldB(sharedRes) pinned memory splitShare %d nranks %d rank %d", comm, comm->pinnedMemUsed - pinnedMemUsedStart, comm->sharedRes->pinnedMemUsed - sharedPinnedMemUsedStart, comm->config.splitShare, comm->nRanks, comm->rank);
  INFO(NCCL_INIT, "Connected binomial trees");
exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportFcConnect(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  uint64_t devMemUsedStart = comm->devMemUsed;
  uint64_t pinnedMemUsedStart = comm->pinnedMemUsed;
  uint64_t sharedDevMemUsedStart = comm->sharedRes->devMemUsed;
  uint64_t sharedPinnedMemUsedStart = comm->sharedRes->pinnedMemUsed;

  if (comm && comm->nRanks > 1) {
    for (int c = 0; c < comm->fcnChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channel, NCCL_MAX_FC_ARITY, channel->fc.peerRanks,
            NCCL_MAX_FC_ARITY, channel->fc.peerRanks, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_FC], 0), ret, fail);
    VERBOSE(NCCL_ALLOC, "comm %p Fc totally used %ldB %ldB(sharedRes) device memory splitShare %d nranks %d rank %d", comm, comm->devMemUsed - devMemUsedStart, comm->sharedRes->devMemUsed - sharedDevMemUsedStart, comm->config.splitShare, comm->nRanks, comm->rank);
    VERBOSE(NCCL_ALLOC, "comm %p Fc totally used %ldB %ldB(sharedRes) pinned memory splitShare %d nranks %d rank %d", comm, comm->pinnedMemUsed - pinnedMemUsedStart, comm->sharedRes->pinnedMemUsed - sharedPinnedMemUsedStart, comm->config.splitShare, comm->nRanks, comm->rank);
    INFO(NCCL_INIT, "Connected all fc graphs");
  }
exit:
  return ret;
fail:
  goto exit;
}
