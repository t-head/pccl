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

// All ranks must participate in collNetSetup call
// type: 0 for send, 1 for recv
// return: 0 - unsupported, 1 - supported
// We do not NCCLCHECK this call because we would fall back to P2P network in case CollNet setup fails
static int collNetSetup(ncclComm_t comm, struct ncclTopoGraph* collNetGraph, struct ncclChannel* channel, int rank, int nranks,  int masterRank, int masterPeer, int nMasters, int type) {
  int rankInCollNet = -1;
  int supported = 0;
  int isMaster = (rank == masterRank) ? 1 : 0;
  struct {
    int collNetRank;
    ncclConnect connect;
  } sendrecvExchange;
  memset(&(sendrecvExchange.connect), 0, sizeof(struct ncclConnect));

  // check if we can connect to collnet, whose root is the nranks-th rank
  struct ncclPeerInfo *myInfo = comm->peerInfo+rank, *peerInfo = comm->peerInfo+nranks;
  peerInfo->rank = nranks;
  int ret = 1;
  if (isMaster) {
    NCCLCHECK(collNetTransport.canConnect(&ret, comm, comm->topo, collNetGraph, myInfo, peerInfo));
  }

  // send master receives connect info from peer recv master
  if (isMaster && type == collNetSend) {
    NCCLCHECK(bootstrapRecv(comm->bootstrap, masterPeer, collNetGraph->id, &sendrecvExchange, sizeof(sendrecvExchange)));
    rankInCollNet = sendrecvExchange.collNetRank;
    INFO(NCCL_INIT, "CollNet [send] : rank %d collNetRank %d collNetNranks %d received connect from rank %d", rank, rankInCollNet, nMasters, masterPeer);
  }

  // select
  struct ncclChannelPeer* root = channel->peers[nranks];
  struct ncclConnector* conn = (type == collNetRecv) ? root->recv + type : root->send + type;
  struct ncclTransportComm* transportComm = (type == collNetRecv) ? &(collNetTransport.recv) : &(collNetTransport.send);
  conn->transportComm = transportComm;
  // setup
  struct ncclConnect myConnect;
  memset(&myConnect, 0, sizeof(struct ncclConnect));
  if (isMaster && ret > 0) {
    NCCLCHECK(transportComm->setup(comm, collNetGraph, myInfo, peerInfo, &myConnect, conn, channel->id, type));
  }
  // prepare connect handles
  ncclResult_t res;
  struct {
    int isMaster;
    ncclConnect connect;
  } *allConnects = NULL;
  ncclConnect *masterConnects = NULL;
  NCCLCHECKGOTO(ncclCalloc(&masterConnects, nMasters), res, cleanup);
  if (type == 1) {  // recv side: AllGather
    // all ranks must participate
    NCCLCHECKGOTO(ncclCalloc(&allConnects, nranks), res, cleanup);
    allConnects[rank].isMaster = isMaster;
    memcpy(&(allConnects[rank].connect), &myConnect, sizeof(struct ncclConnect));
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allConnects, sizeof(*allConnects)), res, cleanup);
    // consolidate
    int c = 0;
    for (int r = 0; r < nranks; r++) {
      if (allConnects[r].isMaster) {
        memcpy(masterConnects+c, &(allConnects[r].connect), sizeof(struct ncclConnect));
        if (r == rank) rankInCollNet = c;
        c++;
      }
    }
  } else { // send side : copy in connect info received from peer recv master
    if (isMaster) memcpy(masterConnects+rankInCollNet, &(sendrecvExchange.connect), sizeof(struct ncclConnect));
  }
  // connect
  if (isMaster && ret > 0) {
    NCCLCHECKGOTO(transportComm->connect(comm, masterConnects, nMasters, rankInCollNet, conn), res, cleanup);
    struct ncclDevChannelPeer* devRoot = channel->devPeers[nranks];
    struct ncclConnInfo* devConn = (type == 1) ? &devRoot->recv[0] : &devRoot->send[0];
    HGGCCHECKGOTO(hggcMemcpy(devConn, &(conn->conn), sizeof(struct ncclConnInfo), hggcMemcpyHostToDevice), res, cleanup);
  }
  // recv side sends connect info to send side
  if (isMaster && type == 1) {
    sendrecvExchange.collNetRank = rankInCollNet;
    memcpy(&sendrecvExchange.connect, masterConnects+rankInCollNet, sizeof(struct ncclConnect));
    NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, masterPeer, collNetGraph->id, &sendrecvExchange, sizeof(sendrecvExchange)), res, cleanup);
    INFO(NCCL_INIT, "CollNet [recv] : rank %d collNetRank %d collNetNranks %d sent connect to rank %d", rank, rankInCollNet, nMasters, masterPeer);
  }
  if (ret > 0) {
    supported = 1;
  }
cleanup:
  NCCLCHECK(ncclFree(allConnects));
  NCCLCHECK(ncclFree(masterConnects));
  return supported;
}

static ncclResult_t checkCollNetSetup(ncclComm_t comm, int rank, int collNetSetupFail) {
  ncclResult_t res = ncclSuccess;
  int nranks = comm->nRanks;
  // AllGather collNet setup results
  int* allGatherFailures = NULL;
  NCCLCHECKGOTO(ncclCalloc(&allGatherFailures, nranks), res, exit);
  allGatherFailures[rank] = collNetSetupFail;
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allGatherFailures, sizeof(int)), res, exit);
  for (int i=0; i<nranks; i++) {
    if (allGatherFailures[i] != 0) {
      collNetSetupFail = 1;
      break;
    }
  }
  if (collNetSetupFail) {
    if (rank == 0) WARN("Cannot initialize CollNet, using %s instead", ncclNetName());
    // Free collNet resources
    for (int r=0; r<comm->nChannels; r++) {
      struct ncclChannel* channel = comm->channels+r;
      struct ncclChannelPeer* peer = channel->peers[nranks];
      for (int b=0; b<NCCL_MAX_CONNS; b++) {
        struct ncclConnector* send = peer->send + b;
        if (send->transportResources && send->transportComm) NCCLCHECK(send->transportComm->free(send));
        send->transportResources = NULL; // avoid double free
      }
      for (int b=0; b<NCCL_MAX_CONNS; b++) {
        struct ncclConnector* recv = peer->recv + b;
        if (recv->transportResources && recv->transportComm) NCCLCHECK(recv->transportComm->free(recv));
        recv->transportResources = NULL; // avoid double free
      }
    }
    // Set support to 0
    comm->collNetSupport = 0;
  } else {
    comm->collNetSupport = 1;
  }
exit:
  NCCLCHECK(ncclFree(allGatherFailures));
  return res;
}

ncclResult_t ncclTransportCollnetConnect(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  uint64_t devMemUsedStart = comm->devMemUsed;
  uint64_t pinnedMemUsedStart = comm->pinnedMemUsed;
  uint64_t sharedDevMemUsedStart = comm->sharedRes->devMemUsed;
  uint64_t sharedPinnedMemUsedStart = comm->sharedRes->pinnedMemUsed;

  if (comm && comm->nRanks > 1) {
    struct ncclTopoGraph *collNetGraph = &comm->graphs[NCCL_ALGO_COLLNET];
    int logicChannels = comm->nChannels/2;
    int collNetSetupFail = 0;
    const int recvIndex = 0;  // recv GPU index is always 0
    const int sendIndex = collNetGraph->pattern == NCCL_TOPO_PATTERN_TREE ? 0 : 1;  // send GPU index depends on topo pattern
    for (int c=0; c<logicChannels; c++) {
      struct ncclChannel* channelRecv = comm->channels+logicChannels+c;
      struct ncclChannel* channelSend = comm->channels+c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channelRecv, 1, &channelRecv->collTree.up, 1, channelRecv->collTree.down, 0), ret, fail);
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channelSend, 1, channelSend->collTree.down, 1, &channelSend->collTree.up, 0), ret, fail);
      const int recvMaster = collNetGraph->intra[c*comm->localRanks+recvIndex];
      const int sendMaster = collNetGraph->intra[c*comm->localRanks+sendIndex];
      if (collNetSetup(comm, collNetGraph, channelRecv, comm->rank, comm->nRanks, recvMaster, sendMaster, comm->nNodes, 1) != 1)
        collNetSetupFail = 1;
      else if (collNetSetup(comm, collNetGraph, channelSend, comm->rank, comm->nRanks, sendMaster, recvMaster, comm->nNodes, 0) != 1)
        collNetSetupFail = 1;
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, collNetGraph, 0), ret, fail);

    // Verify CollNet setup across ranks
    NCCLCHECKGOTO(checkCollNetSetup(comm, comm->rank, collNetSetupFail), ret, fail);
    VERBOSE(NCCL_ALLOC, "comm %p CollNet totally used %ldB %ldB(sharedRes) device memory splitShare %d nranks %d rank %d",
      comm, comm->devMemUsed - devMemUsedStart, comm->sharedRes->devMemUsed - sharedDevMemUsedStart, comm->config.splitShare, comm->nRanks, comm->rank);
    VERBOSE(NCCL_ALLOC, "comm %p CollNet totally used %ldB %ldB(sharedRes) pinned memory splitShare %d nranks %d rank %d",
      comm, comm->pinnedMemUsed - pinnedMemUsedStart, comm->sharedRes->pinnedMemUsed - sharedPinnedMemUsedStart, comm->config.splitShare, comm->nRanks, comm->rank);
  }
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
