/**************************************************************************
* SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
*
* SPDX-License-Identifier: Apache-2.0
*
* See LICENSE.txt for more license information
**************************************************************************/

#ifndef PCCL_ANALYZER_INTERNAL_H_
#define PCCL_ANALYZER_INTERNAL_H_

#include "ras_wire.h"

// Shared rank/peer access for the legacy analyzer and Ring diagnostics.
#ifndef NCCL_DEVICE_H_
typedef enum { CONN_NORMAL=0, CONN_ICN, CONN_PCIE, CONN_SHM, CONN_GDRDMA,
               CONN_RDMA, CONN_SOCKET, CONN_NET_EXT, CONN_NUM } ncclConn_t;
#endif

struct analyzerRankEntry {
  int rank;
  struct rasDiagRankHeader* rh;
};

static inline struct rasDiagChannel* findChannelInRank(struct rasDiagRankHeader* rh, int channelId) {
  struct rasDiagChannel* dc = rasDiagFirstChannel(rh);
  for (int c = 0; c < rh->nChannels; c++) {
    if (dc->channelId == channelId) return dc;
    dc = rasDiagNextChannel(dc);
  }
  return nullptr;
}

static inline struct rasDiagPeer* findPeerInChannel(struct rasDiagChannel* dc, int peerRank, uint8_t direction, uint8_t connector) {
  struct rasDiagPeer* dp = rasDiagFirstPeer(dc);
  for (int i = 0; i < dc->nPeerEntries; i++) {
    if (dp[i].peerRank == peerRank && dp[i].direction == direction && dp[i].connector == connector) {
      return &dp[i];
    }
  }
  return nullptr;
}

#endif  // PCCL_ANALYZER_INTERNAL_H_
