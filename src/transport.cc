/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <sys/time.h>
#include "comm.h"
#include "info.h"
#include "transport.h"
#include "bootstrap.h"

struct ncclTransport* ncclTransports[NTRANSPORTS + 1] = {
  &p2pTransport,
  &shmTransport,
  &netTransport,
  &collNetTransport,
  &profilerTransport
};

template <int type>
static ncclResult_t selectTransport(ncclComm_t comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connect, int channelId, int peer, int connIndex) {
  struct ncclConnector* connector = (type == 1) ? comm->channels[channelId].peers[peer]->send + connIndex :
                                    comm->channels[channelId].peers[peer]->recv + connIndex;
  for (int t=0; t<NTRANSPORTS; t++) {
    struct ncclTransport *transport = ncclTransports[t];
    struct ncclTransportComm* transportComm = type == 1 ? &transport->send : &transport->recv;
    int ret = 0;
    NCCLCHECK(transport->canConnect(&ret, comm, comm->topo, graph, myInfo, peerInfo));
    if (ret) {
      connector->transportComm = transportComm;
      NCCLCHECK(transportComm->setup(comm, graph, myInfo, peerInfo, connect, connector, channelId, connIndex));
      return ncclSuccess;
    }
  }
  ERROR("No transport found !");
  return ncclInternalError;
}

ncclResult_t ncclTransportP2pConnect(ncclComm_t comm, struct ncclChannel* channel, int nrecv, int* peerRecv, int nsend, int* peerSend, int connIndex) {
  TRACE(NCCL_INIT, "nsend %d nrecv %d", nsend, nrecv);
  uint64_t mask = 1UL << channel->id;
  for (int i=0; i<nrecv; i++) {
    int peer = peerRecv[i];
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer]->recv[connIndex].connected) continue;
    comm->connectRecv[peer] |= mask;
  }
  for (int i=0; i<nsend; i++) {
    int peer = peerSend[i];
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer]->send[connIndex].connected) continue;
    comm->connectSend[peer] |= mask;
  }
  return ncclSuccess;
}

void dumpData(struct ncclConnect* data, int ndata) {
  for (int n=0; n<ndata; n++) {
    printf("[%d] ", n);
    uint8_t* d = (uint8_t*)data;
    for (int i=0; i<sizeof(struct ncclConnect); i++) printf("%02x", d[i]);
    printf("\n");
  }
}

PCCL_PARAM(DumpConnInfo, "DUMP_CONN_INFO", 0);
static ncclResult_t dumpConnInfo(const struct ncclConnInfo& conn, int channelId,
    int rank, int peerRank, int send) {
  constexpr int HeaderSize = 64;
  constexpr int PrintBuffSize = 1024;
  char lineHeader[HeaderSize];
  sprintf(lineHeader, "channel: %d, rank: %d, peer: %d, send: %d  ---- ", channelId, rank, peerRank, send);
  char lineBuff[PrintBuffSize];
  int offset = 0;
  sprintf(lineBuff, "%s head: %p, tail: %p, step: %p\n", lineHeader, conn.head, conn.tail, &(conn.step));
  offset = strlen(lineBuff);
  sprintf(lineBuff+offset, "%s direct: %d, shared: %d, ptrExchange: %p\n", lineHeader, conn.direct, conn.shared, conn.ptrExchange);
  offset = strlen(lineBuff);
  sprintf(lineBuff+offset, "%s minPathNum: %u, signalMarker: %p\n", lineHeader, conn.minPathNum, conn.signalMarker);
  offset = strlen(lineBuff);
  sprintf(lineBuff+offset, "%s sizesFifo: %p, offsFifo: %p\n", lineHeader, conn.sizesFifo, conn.offsFifo);
  offset = strlen(lineBuff);
  sprintf(lineBuff+offset, "%s llLastCleaning: %p\n", lineHeader, &(conn.llLastCleaning));
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    offset = strlen(lineBuff);
    sprintf(lineBuff+offset, "%s %s buff addr: %p\n", lineHeader, ncclProtoStr[p], conn.buffs[p]);
  }

  INFO(NCCL_INIT, "\n%s", lineBuff);
  return ncclSuccess;
}

NCCL_PARAM(ConnectRoundMaxPeers, "CONNECT_ROUND_MAX_PEERS", 128);
NCCL_PARAM(ReportConnectProgress, "REPORT_CONNECT_PROGRESS", 0);

ncclResult_t ncclTransportCheckP2pType(ncclComm_t comm, bool* localP2pSupport, bool* directMode, bool* globalP2pSupport) {
  ncclResult_t res = ncclSuccess;
  bool localSupportFlag = true;
  bool globalSuportFlag = true;
  bool directFlag = false;
  if (comm->localRanks == 1) localSupportFlag = false;
  if (comm->nNodes > 1 && (!comm->MNNVL || comm->clique.size != comm->nRanks)) globalSuportFlag = false;

  int* canConnectFlags = NULL;
  NCCLCHECKGOTO(ncclCalloc(&canConnectFlags, comm->nRanks*comm->nRanks), res, end);
  // check p2p connectivity of all ranks when mnnvl enabled, otherwise of local ranks
  if (localSupportFlag || globalSuportFlag) {
    int nRanks = globalSuportFlag ? comm->nRanks : comm->localRanks;
    int* localRankToRank = globalSuportFlag ? NULL : comm->localRankToRank;
    for (int i = 0; i < nRanks; i++) {
      for (int j = i+1; j < nRanks; j++) {
        int ipeer = localRankToRank ? localRankToRank[i] : i;
        int jpeer = localRankToRank ? localRankToRank[j] : j;
        struct ncclPeerInfo* ipeerInfo = &comm->peerInfo[ipeer];
        struct ncclPeerInfo* jpeerInfo = &comm->peerInfo[jpeer];
        int idx = ipeer * comm->nRanks + jpeer;
        NCCLCHECKGOTO(ncclTopoCheckP2p(comm, comm->topo, ipeerInfo->rank, jpeerInfo->rank, &canConnectFlags[idx], NULL, NULL), res, end);
        if (!canConnectFlags[idx] && globalSuportFlag == true) {
          globalSuportFlag = false;
        }
      }
    }
  }
  // set local p2p connectivity flag according to canConnectFlags and check direct mode
  if (localSupportFlag) {
    for (int i = 0; i < comm->localRanks; i++) {
      for (int j = i+1; j < comm->localRanks; j++) {
        int ipeer = comm->localRankToRank[i];
        int jpeer = comm->localRankToRank[j];
        struct ncclPeerInfo* ipeerInfo = &comm->peerInfo[ipeer];
        struct ncclPeerInfo* jpeerInfo = &comm->peerInfo[jpeer];
        if (!canConnectFlags[ipeer*comm->nRanks+jpeer] && localSupportFlag == true) {
          localSupportFlag = false;
        }
        if (PCCL_SAME_PID(ipeerInfo, jpeerInfo)) directFlag = true;
        if (!localSupportFlag && directFlag) break;
      }
    }
  }

end:
  *localP2pSupport = localSupportFlag;
  *globalP2pSupport = globalSuportFlag;
  *directMode = directFlag;
  if (comm->rank == 0) INFO(NCCL_INIT, "Check P2P Type localP2pSupport %d directMode %d globalP2pSupport %d", *localP2pSupport, *directMode, *globalP2pSupport);
  NCCLCHECK(ncclFree(canConnectFlags));
  return res;
}

ncclResult_t ncclTransportP2pSetup(ncclComm_t comm, struct ncclTopoGraph* graph, int connIndex) {
  ncclResult_t ret = ncclSuccess;
  struct ncclConnect** data; // Store intermediate send/recvData structs for connect
  struct ncclConnect** recvData = NULL; // Points to entries inside data for given recv connection within a channel
  struct ncclConnect** sendData = NULL; // Points to entries inside data for given send connection within a channel
  int done = 0;
  int maxPeers = ncclParamConnectRoundMaxPeers();

  struct timeval timeStart, timeLast;
  gettimeofday(&timeStart, NULL);
  timeLast = timeStart;
  bool timeReported = false;

  NCCLCHECK(ncclCalloc(&data, maxPeers));
  NCCLCHECKGOTO(ncclCalloc(&recvData, maxPeers), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&sendData, maxPeers), ret, fail);

  NCCLCHECKGOTO(ncclStrongStreamAcquireUncaptured(&comm->sharedRes->hostStream), ret, fail);

  for (int i=1; i<comm->nRanks; i++) {
    int bootstrapTag = (i<<8) + (graph ? graph->id+1 : 0);
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;
    uint64_t recvMask = comm->connectRecv[recvPeer];
    uint64_t sendMask = comm->connectSend[sendPeer];

    // Data[i] contains all ncclConnect information for all send and receive connections with a given send and recv peer
    // This data is packed in the array based on the number of sendChannels and recvChannels connected with these peers
    // The first N entries contain recvData, connection information for recv connections
    // The next M entries contain sendData, connection information for send connections
    // It's not guaranteed that each entry of data has the same number of total or send/recv specific connections
    int p = i-(done+1);
    if (recvMask || sendMask) {
      if (data[p] == NULL) NCCLCHECKGOTO(ncclCalloc(data + p, 2 * MAXCHANNELS), ret, fail);
      else memset(data[p], 0, 2 * MAXCHANNELS * sizeof(struct ncclConnect));
    }
    recvData[p] = data[p];
    int sendChannels = 0, recvChannels = 0;
    for (int c=0; c<MAXCHANNELS; c++) {
      if (recvMask & (1UL<<c)) {
        NCCLCHECKGOTO(selectTransport<0>(comm, graph, comm->peerInfo+comm->rank, comm->peerInfo+recvPeer, recvData[p]+recvChannels++, c, recvPeer, connIndex), ret, fail);
      }
    }
    sendData[p] = recvData[p]+recvChannels;
    for (int c=0; c<MAXCHANNELS; c++) {
      if (sendMask & (1UL<<c)) {
        NCCLCHECKGOTO(selectTransport<1>(comm, graph, comm->peerInfo+comm->rank, comm->peerInfo+sendPeer, sendData[p]+sendChannels++, c, sendPeer, connIndex), ret, fail);
      }
    }

    if (sendPeer == recvPeer) {
      if (recvChannels+sendChannels) {
         NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, data[p], sizeof(struct ncclConnect)*(recvChannels+sendChannels)), ret, fail);
         NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, data[p], sizeof(struct ncclConnect)*(recvChannels+sendChannels)), ret, fail);
         sendData[p] = data[p];
         recvData[p] = data[p]+sendChannels;
      }
    } else {
      if (recvChannels) NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, recvData[p], sizeof(struct ncclConnect)*recvChannels), ret, fail);
      if (sendChannels) NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, sendData[p], sizeof(struct ncclConnect)*sendChannels), ret, fail);
      if (sendChannels) NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, sendData[p], sizeof(struct ncclConnect)*sendChannels), ret, fail);
      if (recvChannels) NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, recvData[p], sizeof(struct ncclConnect)*recvChannels), ret, fail);
    }

    if (i-done == maxPeers || i == comm->nRanks-1) {
      // Loop until all channels with all ranks have been connected
      bool allChannelsConnected;
      allChannelsConnected = false;
      while (!allChannelsConnected) {
        allChannelsConnected = true;
        for (int j=done+1; j<=i; j++) {
          int recvPeer = (comm->rank - j + comm->nRanks) % comm->nRanks;
          int sendPeer = (comm->rank + j) % comm->nRanks;
          uint64_t recvMask = comm->connectRecv[recvPeer];
          uint64_t sendMask = comm->connectSend[sendPeer];

          int p = j-(done+1);
          int sendDataOffset = 0;
          int recvDataOffset = 0;
          for (int c=0; c<MAXCHANNELS; c++) {
            if (sendMask & (1UL<<c)) {
              struct ncclConnector* conn = comm->channels[c].peers[sendPeer]->send + connIndex;
              // This connector hasn't completed connection yet
              if (conn->connected == 0) {
                NCCLCHECKGOTO(conn->transportComm->connect(comm, sendData[p] + sendDataOffset, 1, comm->rank, conn), ret, fail);
                if (ret == ncclSuccess) {
                  conn->connected = 1;
                  if (pcclParamDumpConnInfo() == 1) {
                    NCCLCHECKGOTO(dumpConnInfo(conn->conn, c, comm->rank, sendPeer, 1), ret, fail);
                  }
                  /* comm->channels[c].devPeers[sendPeer]->send is a device memory access. */
                  if (ncclPerfmodelSpeedup) {
                    ncclHggcMemcpy(&comm->channels[c].devPeersHostPtr[sendPeer]->send[connIndex], &conn->conn, 1);
                  } else {
                    HGGCCHECKGOTO(hggcMemcpyAsync(&comm->channels[c].devPeersHostPtr[sendPeer]->send[connIndex], &conn->conn, sizeof(struct ncclConnInfo), hggcMemcpyHostToDevice, comm->sharedRes->hostStream.hggcStream), ret, fail);
                  }
                } else if (ret == ncclInProgress) {
                  allChannelsConnected = false;
                }
              }
              sendDataOffset++;
            }

            if (recvMask & (1UL<<c)) {
              struct ncclConnector* conn = comm->channels[c].peers[recvPeer]->recv + connIndex;
              // This connector hasn't completed connection yet
              if (conn->connected == 0) {
                NCCLCHECKGOTO(conn->transportComm->connect(comm, recvData[p] + recvDataOffset, 1, comm->rank, conn), ret, fail);
                if (ret == ncclSuccess) {
                  conn->connected = 1;
                  if (pcclParamDumpConnInfo() == 1) {
                    NCCLCHECKGOTO(dumpConnInfo(conn->conn, c, comm->rank, recvPeer, 0), ret, fail);
                  }
                  /* comm->channels[c].devPeers[recvPeer]->recv is a device memory access. */
                  if (ncclPerfmodelSpeedup) {
                    ncclHggcMemcpy(&comm->channels[c].devPeersHostPtr[recvPeer]->recv[connIndex], &conn->conn, 1);
                  }
                  else {
                    HGGCCHECKGOTO(hggcMemcpyAsync(&comm->channels[c].devPeersHostPtr[recvPeer]->recv[connIndex], &conn->conn, sizeof(struct ncclConnInfo), hggcMemcpyHostToDevice, comm->sharedRes->hostStream.hggcStream), ret, fail);
                  }
                } else if (ret == ncclInProgress) {
                  allChannelsConnected = false;
                }
              }
              recvDataOffset++;
            }
          }
        }
        if (ncclParamReportConnectProgress() && comm->rank == 0) {
          struct timeval now;
          gettimeofday(&now, NULL);
          if (((now.tv_sec - timeLast.tv_sec)*1.0 + (now.tv_usec-timeLast.tv_usec)*1e-6) > 1) {
            float elapsed = (now.tv_sec - timeStart.tv_sec)*1.0 + (now.tv_usec-timeStart.tv_usec)*1e-6;
            float remaining = elapsed*(comm->nRanks-i)/i;
            INFO(NCCL_INIT, "P2p connect: %g%% Elapsed %d:%02d Remaining %d:%02d",
                i*100.0/comm->nRanks, ((int)elapsed)/60, ((int)elapsed)%60, ((int)remaining)/60, ((int)remaining)%60);
            timeReported = true;
            timeLast = now; // struct copy;
          }
        }
      }
      done = i;
    }
  }

  {
    struct timeval now;
    gettimeofday(&now, NULL);
    float elapsed = (now.tv_sec - timeStart.tv_sec)*1.0 + (now.tv_usec-timeStart.tv_usec)*1e-6;
    if (elapsed > 1.0) INFO(NCCL_PROFILE, "timings: rank %d nranks %d P2p connect done in %.2f", comm->rank, comm->nRanks, elapsed);
    if (timeReported) {
      INFO(NCCL_INIT, "P2p connect done in %d:%02d", ((int)elapsed)/60, ((int)elapsed)%60);
    }
  }

  /* We need to sync ranks here since some ranks might run too fast after connection setup
   * and start to destroy the connection after returning from this function; however, the
   * others might still be trying to connect and import the buffer. No sync can lead to invalid
   * shmem/cuda buffer. In addition, we also clear all connect masks and free each connectInfo array */
  for (int i = 1; i < comm->nRanks; i++) {
    int bootstrapTag = (i << 8) + (1 << 7) + (graph ? graph->id + 1 : 0);
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;

    if (recvPeer != sendPeer) {
      if (comm->connectSend[sendPeer] != 0UL) NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
      if (comm->connectRecv[recvPeer] != 0UL) NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, NULL, 0), ret, fail);
      if (comm->connectSend[sendPeer] != 0UL) NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
      if (comm->connectRecv[recvPeer] != 0UL) NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, NULL, 0), ret, fail);
    } else {
      if (comm->connectSend[sendPeer] != 0UL || comm->connectRecv[recvPeer] != 0UL) {
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
      }
    }
    comm->connectRecv[recvPeer] = comm->connectSend[sendPeer] = 0UL;
  }

exit:
  for(int i=0; i<maxPeers; ++i){
    NCCLCHECK(ncclFree(data[i]));
  }
  NCCLCHECK(ncclFree(data));
  NCCLCHECK(ncclFree(sendData));
  NCCLCHECK(ncclFree(recvData));

  NCCLCHECK(ncclStrongStreamWaitStream(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, &comm->sharedRes->hostStream));
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->hostStream));
  return ret;
fail:
  goto exit;
}

ncclResult_t GetConnType(ncclComm_t comm, struct ncclTopoGraph* graph, int channelId, struct ncclPeerInfo* myInfo,
  struct ncclPeerInfo* peerInfo, bool send, ncclConn_t* connType) {

  *connType = CONN_NORMAL;
  for (int t=0; t<NTRANSPORTS; t++) {
    struct ncclTransport *transport = ncclTransports[t];
    int ret = 0;
    NCCLCHECK(transport->canConnect(&ret, comm, comm->topo, graph, myInfo, peerInfo));
    if (ret) {
      if (t == TRANSPORT_P2P) {
        NCCLCHECK(ncclTopoGetConnType(comm->topo, myInfo->rank, peerInfo->rank, connType));
        if (usingMemcpy()) {
          *connType = CONN_NORMAL;
        }
      } else if (t == TRANSPORT_SHM) {
        *connType = CONN_SHM;
      } else if (t == TRANSPORT_NET) {
        int proxyRank, netDev;
        int64_t netId;
        int useGdr;
        NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, peerInfo->rank, &netId, &netDev, &proxyRank));
        NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->rank, netId, send, &useGdr));

        if (useGdr) {
          if (send && (proxyRank != myInfo->rank)) {
            // use gdr and has remote rank proxy
            *connType = CONN_ICN;
          } else {
            *connType = CONN_GDRDMA;
          }
        } else if (strcmp(ncclNetName(), "IB") == 0) {
          *connType = CONN_RDMA;
        } else if (strcmp(ncclNetName(), "Socket") == 0) {
          *connType = CONN_SOCKET;
        } else {
          *connType = CONN_NET_EXT;
        }
      } else {
        // TRANSPORT_COLLNET not supported yet.
        ERROR("TRANSPORT_COLLNET is not supported");
        *connType = CONN_NORMAL;
        return ncclInternalError;
      }
      return ncclSuccess;
    }
  }
  ERROR("no connect find from rank %d to %d", myInfo->rank, peerInfo->rank);
  return ncclInternalError;
}

ncclResult_t ncclConnTypeCollect(ncclComm_t comm, struct ncclTopoGraph* graph) {

  if (!graph) return ncclSuccess;

  struct ncclPeerInfo* myInfo = &comm->peerInfo[comm->rank];
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    if (graph->id == 0) {
      // ring graph
      int sendPeer = channel->ring.next;
      int recvPeer = channel->ring.prev;

      // next conn type
      ncclConn_t sendConnType = CONN_NORMAL;
      struct ncclPeerInfo* sendPeerInfo = &comm->peerInfo[sendPeer];
      NCCLCHECK(GetConnType(comm, graph, c, myInfo, sendPeerInfo, true, &sendConnType));
      comm->channels[c].ring.sendConnType = sendConnType;
      comm->maxConnType = sendConnType > comm->maxConnType ? sendConnType : comm->maxConnType;
      // prev conn type
      ncclConn_t recvConnType = CONN_NORMAL;
      struct ncclPeerInfo* recvPeerInfo = &comm->peerInfo[recvPeer];
      NCCLCHECK(GetConnType(comm, graph, c, myInfo, recvPeerInfo, false, &recvConnType));
      comm->channels[c].ring.recvConnType = recvConnType;
      comm->maxConnType = recvConnType > comm->maxConnType ? recvConnType : comm->maxConnType;

    } else if (graph->id == 1) {
      // tree graph
      // up conn type
      int upPeer = channel->tree.up;
      comm->channels[c].tree.upConnType = CONN_NORMAL;
      if (upPeer != -1 && upPeer != comm->rank) {
        ncclConn_t upPeerConnType = CONN_NORMAL;
        struct ncclPeerInfo* upPeerInfo = &comm->peerInfo[upPeer];
        NCCLCHECK(GetConnType(comm, graph, c, myInfo, upPeerInfo, true, &upPeerConnType));
        comm->channels[c].tree.upConnType = upPeerConnType;
      }
      // down conn type
      comm->channels[c].tree.downConnType = CONN_NORMAL;
      for (int d=0; d<NCCL_MAX_TREE_ARITY; d++) {
        int downPeer = channel->tree.down[d];
        if (downPeer == -1 || downPeer == comm->rank) continue;
        ncclConn_t downPeerConnType = CONN_NORMAL;
        struct ncclPeerInfo* downPeerInfo = &comm->peerInfo[downPeer];
        NCCLCHECK(GetConnType(comm, graph, c, myInfo, downPeerInfo, true, &downPeerConnType));
        comm->channels[c].tree.downConnType = downPeerConnType;
        if (downPeerConnType == CONN_NORMAL) {
          break;
        }
      }

    } else if (graph->id == 3) {
      channel->fc.connType = CONN_NORMAL;
      int peer = channel->fc.peerRanks[0];
      if (peer >= 0) {
        ncclConn_t peerConnType = CONN_NORMAL;
        struct ncclPeerInfo* peerInfo = &comm->peerInfo[peer];
        NCCLCHECK(GetConnType(comm, graph, c, myInfo, peerInfo, true, &peerConnType));
        channel->fc.connType = peerConnType;
      }
    }
  }

  return ncclSuccess;
}
