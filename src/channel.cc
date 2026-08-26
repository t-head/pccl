/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "channel.h"
#include "param.h"
#include "transport.h"
#include "gdrwrap.h"

// Multiplied by SPINS_BEFORE_CHECK_ABORT as an estimated polling loops per second in kernel
// 20 spins per second is nearly 5 minutes
#define SPINS_BEFORE_REPORT_HANG_PER_SEC 20
extern int64_t ncclParamStateMonitorTimeoutMs;
ncclResult_t initChannel(ncclComm_t comm, int channelId) {
  struct ncclChannel* channel = comm->channels + channelId;
  if (channel->id != -1) return ncclSuccess;
  int nRanks = comm->nRanks;
  int nPeers = nRanks + 1 /* Collnet */;
  channel->id = channelId;

  struct ncclSharedResources* sharedRes = comm->sharedRes;

  if (channel->peers == NULL) {
    // The extra on nRanks+1 is for collnet root (i.e. network)
    // Allocate everything related to sharedRes with ncclCalloc as this can be
    // shared between communicators hence should not be tied to comm.
    if (sharedRes->peers[channelId] == NULL) {
      NCCLCHECK(ncclCalloc(sharedRes->peers + channelId, sharedRes->tpNRanks));
    }
    // channel->peers = ncclMemoryStackAlloc<struct ncclChannelPeer*>(&comm->memPermanent, nPeers);
    NCCLCHECK(ncclCalloc(&channel->peers, nPeers));
    for (int r = 0; r < nRanks; r++) {
      channel->peers[r] = comm->sharedRes->peers[channelId] + comm->topParentRanks[r];
      ncclAtomicRefCountIncrement(&channel->peers[r]->refCount);
    }
  }

  if (channel->devPeers == NULL) {
    if (sharedRes->devPeers[channelId] == NULL) {
      NCCLCHECK(ncclHggcCallocLegacy(sharedRes->devPeers + channelId, sharedRes->tpNRanks, &(sharedRes->devMemUsed)));
    }
    /* channel->devPeers is not shared, so just free it when calling commFree() */
    NCCLCHECK(ncclHggcCallocLegacy(&channel->devPeers, nPeers, &(comm->devMemUsed)));
    NCCLCHECK(ncclCalloc(&channel->devPeersHostPtr, nPeers));
    for (int r = 0; r < nRanks; r++) {
      uintptr_t addr = (uintptr_t)(comm->sharedRes->devPeers[channelId] + comm->topParentRanks[r]);
      NCCLCHECK(ncclHggcMemcpy((uintptr_t*)(channel->devPeers + r), (uintptr_t*)&addr, 1));
      channel->devPeersHostPtr[r] = (struct ncclDevChannelPeer*)addr;
    }
  }

  NCCLCHECK(ncclStrongStreamAcquireUncaptured(&comm->sharedRes->deviceStream));

  // Ring index to user rank table.
  NCCLCHECK(ncclHggcCallocAsyncLegacy(&channel->ring.devUserRanks, comm->nRanks, &(comm->devMemUsed), comm->sharedRes->deviceStream.hggcStream));
  NCCLCHECK(ncclCalloc(&channel->ring.userRanks, comm->nRanks));

  channel->nSpinsBeforeReportHang = DIVUP(ncclParamStateMonitorTimeoutMs, 1e3) * SPINS_BEFORE_REPORT_HANG_PER_SEC;
  NCCLCHECK(ncclHggcHostCalloc(&channel->hostDevIndexLevel1, 1, &(comm->pinnedMemUsed)));
  NCCLCHECK(ncclHggcHostCalloc(&channel->hostDevIndexLevel2, 1, &(comm->pinnedMemUsed)));
  NCCLCHECK(ncclHggcCallocAsyncLegacy(&channel->devIndexLevel1, 1, &(comm->devMemUsed), comm->sharedRes->deviceStream.hggcStream));
  NCCLCHECK(ncclHggcCallocAsyncLegacy(&channel->devPerKernelBatchCounter, 1, &(comm->devMemUsed), comm->sharedRes->deviceStream.hggcStream));

#ifdef NO_TSM
  NCCLCHECK(ncclHggcCallocAsyncLegacy(&channel->shmem, 1, &(comm->devMemUsed), comm->sharedRes->deviceStream.hggcStream));
  if (pcclPatEnable(comm)) {
    NCCLCHECK(ncclHggcCallocAsyncLegacy((char**)&channel->patShmem, NCCL_PAT_SHMEM_SIZE, &(comm->devMemUsed), comm->sharedRes->deviceStream.hggcStream));
  } else {
    channel->patShmem = nullptr;
  }
#endif

  /* guarantee addr has been copied into channel->devPeers */
  NCCLCHECK(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream));
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->deviceStream));
  return ncclSuccess;
}

ncclResult_t freeChannel(struct ncclChannel* channel, int nRanks) {
  int nPeers = nRanks + 1;
  if (channel->id == -1) return ncclSuccess;

  NCCLCHECK(ncclHggcHostFree(channel->hostDevIndexLevel1));
  NCCLCHECK(ncclHggcHostFree(channel->hostDevIndexLevel2));

  // Free per-channel device counters allocated unconditionally in initChannel.
  // These are not tied to channel->peers, so they must be freed before the
  // early return below to avoid an 8-byte device leak per counter per channel.
  NCCLCHECK(ncclHggcFreeLegacy(channel->devIndexLevel1));
  NCCLCHECK(ncclHggcFreeLegacy(channel->devPerKernelBatchCounter));
#ifdef NO_TSM
  NCCLCHECK(ncclHggcFreeLegacy(channel->shmem));
  if (channel->patShmem) NCCLCHECK(ncclHggcFreeLegacy(channel->patShmem));
#endif

  // Free Ring index to rank tables
  NCCLCHECK(ncclFree(channel->ring.userRanks));
  NCCLCHECK(ncclHggcFreeLegacy(channel->ring.devUserRanks));

  /* channel peers are only valid when async init thread completes commAlloc() and
   * the channel is intialized with initChannel(); if either is not done, this channel
   * should never be free. */
  if (channel->peers == NULL) return ncclSuccess;

  // Free transport proxy resources
  // Note: free all send resources first due to CollNet arrangement
  for (int r=0; r<nPeers; r++) {
    struct ncclChannelPeer* peer = channel->peers[r];
    if (peer) {
      if (ncclAtomicRefCountDecrement(&peer->refCount) == 0) {
        for (int b = 0; b < NCCL_MAX_CONNS; b++) {
          if (peer->send[b].transportComm) NCCLCHECK(peer->send[b].transportComm->free(peer->send + b));
          if (peer->recv[b].transportComm) NCCLCHECK(peer->recv[b].transportComm->free(peer->recv + b));
        }
      }
    }
  }

  // Free the peer structures.
  NCCLCHECK(ncclHggcFreeLegacy(channel->devPeers));
  NCCLCHECK(ncclFree(channel->peers));
  NCCLCHECK(ncclFree(channel->devPeersHostPtr));

  return ncclSuccess;
}
