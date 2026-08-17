/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "register_inline.h"
#include "transport.h"
#include "enqueue.h"
#include "param.h"
#include "cudawrap.h"

NCCL_PARAM(GraphRegister, "GRAPH_REGISTER", 1);
static ncclResult_t registerCheckP2PConnection(ncclComm_t comm, struct ncclConnector* conn, struct ncclTopoGraph* graph, int peer, bool* needReg) {
  if (conn->connected) {
    if (conn->conn.direct == NCCL_DIRECT_GPU) {
      *needReg = true;
    } else {
      // network connection
      *needReg = false;
    }
  } else {
    struct ncclPeerInfo* peerInfo = &comm->peerInfo[peer];
    struct ncclPeerInfo* myInfo = &comm->peerInfo[comm->rank];
    int canConnect = 0;
    NCCLCHECK(ncclTransports[0]->canConnect(&canConnect, comm, comm->topo, graph, myInfo, peerInfo));
    if (canConnect) {
      *needReg = true;
    } else {
      *needReg = false;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclRegisterCollBuffers(
    ncclComm_t comm, struct ncclTaskColl* info,
    void* outRegBufSend[NCCL_MAX_LOCAL_RANKS],
    void* outRegBufRecv[NCCL_MAX_LOCAL_RANKS],
    struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue
  ) {
  ncclResult_t result = ncclSuccess;

  info->regBufType = NCCL_REGULAR_BUFFER;
  if (!ncclCuMemEnable() && !ncclLegacyCudaRegister()) goto exit;
  if (!(ncclParamLocalRegister() || (comm->planner.persistent && ncclParamGraphRegister()))) goto exit;
  // Add NVLS/NET handle later
  // INFO: in nccl, do different peer registration according to algo
  //       in pccl, we do the FC registration.
    // IPC buffer registration
  if (!info->isExtKernel) {
    if (info->algorithm == NCCL_ALGO_RING && ((info->func == ncclFuncAllReduce && info->sendbuff == info->recvbuff) || info->func == ncclFuncReduce)) goto exit;
    if (info->algorithm == NCCL_ALGO_TREE && info->sendbuff == info->recvbuff) goto exit;
    // disable register for allgather and reducescatter(perf regression)
    if (info->func == ncclFuncReduceScatter) goto exit;
  }

  // TODO: Add NVLS/COLLNET later

  if (PROTO_IS_SIMPLE_CLASS(info->protocol) || info->isExtKernel) {
    int peerRanks[NCCL_MAX_LOCAL_RANKS];
    int nPeers = 0;
    size_t elementSize = ncclTypeSize(info->datatype);
    size_t sendbuffSize = elementSize*ncclFuncSendCount(info->func, comm->nRanks, info->count);
    size_t recvbuffSize = elementSize*ncclFuncRecvCount(info->func, comm->nRanks, info->count);
    int regBufFlag = 0;
    memset(peerRanks, 0xff, sizeof(int) * NCCL_MAX_LOCAL_RANKS);

    if (info->isExtKernel || info->algorithm == NCCL_ALGO_FC) {
      for (int i = 0; i < comm->nRanks; ++i) {
        if (i != comm->rank) peerRanks[nPeers++] = i;
      }
    } else if (info->algorithm == NCCL_ALGO_RING) {
      for (int c = 0; c < comm->nChannels; ++c) {
        struct ncclChannel* channel = comm->channels + c;
        for (int r = 0; r < 2; ++r) {
          int peer;
          if (r == 0) {
            peer = channel->ring.prev;
          } else {
            peer = channel->ring.next;
          }
          struct ncclConnector* recvConn = NULL;
          recvConn = &channel->peers[peer]->recv[0];
          if (recvConn->conn.direct == NCCL_DIRECT_GPU) {
            bool found = false;
            for (int p = 0; p < nPeers; ++p) {
              if (peerRanks[p] == peer) {
                found = true;
                break;
              }
            }
            if (!found) peerRanks[nPeers++] = peer;
          }
        }
      }
    } else if (info->algorithm == NCCL_ALGO_TREE) {
      for (int c = 0; c < comm->nChannels; ++c) {
        struct ncclChannel* channel = comm->channels + c;
        struct ncclTree* tree = NULL;
        int peers[NCCL_MAX_TREE_ARITY + 1];

        tree = &channel->tree;
        for (int p = 0; p < NCCL_MAX_TREE_ARITY; ++p) peers[p] = tree->down[p];
        peers[NCCL_MAX_TREE_ARITY] = tree->up;
        for (int p = 0; p < NCCL_MAX_TREE_ARITY + 1; ++p) {
          int peer = peers[p];
          bool peerNeedReg = false;
          struct ncclConnector* recvConn = NULL;
          // P2P transport
          if (peer == -1 || peer == comm->nRanks) continue;
          recvConn = &channel->peers[peer]->recv[0];
          NCCLCHECK(registerCheckP2PConnection(comm, recvConn, &comm->graphs[info->algorithm], peer, &peerNeedReg));

          if (peerNeedReg) {
            bool found = false;
            for (int pindex = 0; pindex < nPeers; ++pindex) {
              if (peerRanks[pindex] == peer) {
                found = true;
                break;
              }
            }
            if (!found) peerRanks[nPeers++] = peer;
          }
        }
      }
    } else {
      goto exit;
    }

    struct ncclReg* recvRegRecord = NULL;
    struct ncclReg* sendRegRecord = NULL;

    NCCLCHECK(ncclRegFind(comm, info->recvbuff, recvbuffSize, &recvRegRecord));
    if (recvRegRecord == NULL && !(comm->planner.persistent && ncclParamGraphRegister())) goto exit;
    NCCLCHECK(ncclRegFind(comm, info->sendbuff, sendbuffSize, &sendRegRecord));
    if (sendRegRecord == NULL && !(comm->planner.persistent && ncclParamGraphRegister())) goto exit;

    if (nPeers > 0 && comm->localP2pSupport) {
      if (comm->planner.persistent && ncclParamGraphRegister()) {
        ncclIpcGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, peerRanks, nPeers, NCCL_IPC_COLLECTIVE, &regBufFlag, &info->recvbuffOffset, &info->recvbuffRmtAddrs, cleanupQueue, &info->nCleanupQueueElts);
      }
      if (!regBufFlag && ncclParamLocalRegister()) {
        ncclIpcLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, peerRanks, nPeers, NCCL_IPC_COLLECTIVE, &regBufFlag, &info->recvbuffOffset, &info->recvbuffRmtAddrs);
      }
    }
    if (regBufFlag) {
      info->regBufType = NCCL_IPC_REG_BUFFER;
    }
  }
exit:
  return result;
}
