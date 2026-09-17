/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_CHANNEL_H_
#define NCCL_CHANNEL_H_
#include "comm.h"

ncclResult_t initChannel(ncclComm_t comm, int channelid);
ncclResult_t freeChannel(struct ncclChannel* channel, int nRanks, ncclComm_t comm);
// For PAT algorithm.
int pcclPatEnable(ncclComm_t comm);

inline int pcclP2pChannelId(ncclComm_t comm, int p2pRound, int part) {
  int base;
  if (comm->nNodes > 1) {
    int localSize = comm->p2pSchedGroupSize;
    int groupDelta = p2pRound / localSize;
    int localDelta = p2pRound % localSize;
    base = groupDelta * divUp(localSize, NCCL_MAX_DEV_WORK_P2P_PER_BATCH);
    base += localDelta / NCCL_MAX_DEV_WORK_P2P_PER_BATCH;
  } else {
    base = p2pRound;
  }
  return (base + comm->p2pChannels[part % comm->p2pnChannelsPerPeer]) % comm->p2pnChannels;
}

#endif
