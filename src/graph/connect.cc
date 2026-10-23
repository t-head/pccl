/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "graph.h"
#include "trees.h"
#include "rings.h"

/******************************************************************/
/********************* Internode connection ***********************/
/******************************************************************/
PCCL_PARAM(ReverseRingInterleave, "REVERSE_RING_INTERLEAVE", 0);

ncclResult_t ncclTopoPreset(ncclComm_t comm,
    struct ncclTopoGraph* treeGraph, struct ncclTopoGraph* ringGraph, struct ncclTopoGraph* collNetGraph,
    struct ncclTopoGraph* fcGraph, struct ncclTopoRanks* topoRanks) {
  int rank = comm->rank;
  int localRanks = comm->localRanks;
  int nChannels = comm->nChannels;

  for (int c=0; c<nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    channel->ring.prev = channel->ring.next = -1;
    channel->tree.up = -1;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->tree.down[i] = -1;
    channel->collTree.up = -1;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->collTree.down[i] = -1;
    if (fcGraph->nChannels > 0) {
      for (int i=0; i<NCCL_MAX_FC_ARITY; i++) channel->fc.peerRanks[i] = -1;
      channel->fc.nPeers = localRanks-1;
    }

    int* ringIntra = ringGraph->intra+c*localRanks;
    int* treeIntra = treeGraph->intra+c*localRanks;
    int* collNetIntra = collNetGraph->intra+c*localRanks;
    int* fcIntra = fcGraph->intra;
    int fcPeer = 0;
    static int reverseRingInterleave = pcclParamReverseRingInterleave();

    if (reverseRingInterleave == 1) {
      if (ringGraph->crossNic == 0) INFO(NCCL_GRAPH, "SameNic graph mode might have no effects while using reverse ring.");
      if (comm->nNodes%2 == 1) INFO(NCCL_GRAPH, "Reverse ring order interleave might have no perf gains while total node number is odd.");
    }

    for (int i=0; i<localRanks; i++) {
      if (ringIntra[i] == rank) {
        if (reverseRingInterleave == 1 && comm->node%2 == 1) {
          // For HPN7.0 reverse ring order
          // When set reverse even ring, make sure every node has the same localRanks
          topoRanks->ringRecv[c] = ringIntra[localRanks-1];
          topoRanks->ringSend[c] = ringIntra[0];
          channel->ring.prev = (i == localRanks-1) ? -1 : ringIntra[i+1];
          channel->ring.next = (i == 0) ? -1 : ringIntra[i-1];
        } else {
          topoRanks->ringRecv[c] = ringIntra[0];
          topoRanks->ringSend[c] = ringIntra[localRanks-1];
          channel->ring.prev = (i == 0) ? -1 : ringIntra[i-1];
          channel->ring.next = (i == localRanks-1) ? -1 : ringIntra[i+1];
        }
      }
      if (treeIntra[i] == rank) {
        int parentIndex = 0;
        int child0Index = treeGraph->pattern == NCCL_TOPO_PATTERN_TREE ? 0 : 1;
        int child1Index = treeGraph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE ? 1 : 0;

        topoRanks->treeToParent[c] = treeIntra[parentIndex];
        topoRanks->treeToChild0[c] = treeIntra[child0Index];
        topoRanks->treeToChild1[c] = treeIntra[child1Index];
        channel->tree.up         = i == 0 ? -1 : treeIntra[i-1];
        channel->tree.down[0]    = i == localRanks-1 ? -1 : treeIntra[i+1];
      }
      if (collNetIntra[i] == rank) {
        int prev = (i-1+localRanks)%localRanks, next = (i+1)%localRanks;

        channel->collTree.up      = collNetIntra[prev];
        channel->collTree.down[0] = collNetIntra[next];
      }
      if (fcGraph->nChannels > 0) {
        if (fcIntra[i] == rank) {
          channel->fc.rank = rank;
        } else {
          channel->fc.peerRanks[fcPeer++] = fcIntra[i];
        }
      }
    }
    topoRanks->ringPrev[c] = channel->ring.prev;
    topoRanks->ringNext[c] = channel->ring.next;
  }

  INFO(NCCL_GRAPH, "comm rank: %d", comm->rank);
  // Duplicate channels rings/trees
  struct ncclChannel* channel0 = comm->channels;
  struct ncclChannel* channel1 = channel0+nChannels;
  memcpy(channel1, channel0, nChannels*sizeof(struct ncclChannel));

  // Duplicate channels fcs
  if (fcGraph->nChannels>nChannels) {
    int fcNChannels = 2 * fcGraph->nChannels;
    for (int duplicatedNChannels = nChannels * 2; duplicatedNChannels < fcNChannels;) {
      int extraNChannels = std::min(duplicatedNChannels, fcNChannels - duplicatedNChannels);
      memcpy(comm->channels+duplicatedNChannels, comm->channels, extraNChannels*sizeof(struct ncclChannel));
      duplicatedNChannels += extraNChannels;
    }
  }
#ifndef KERNEL_LOOPBACK
  if (comm->nRanks < 2) {
    comm->nChannels = 1;
  }
#endif
  return ncclSuccess;
}

static ncclResult_t connectRings(ncclComm_t comm, int* ringRecv, int* ringSend, int* ringPrev, int* ringNext, int* firstRanks) {
  int nChannels = comm->nChannels;
  int nNodes = comm->nNodes;
  for (int c=0; c<nChannels; c++) {
    int* recv = ringRecv+c*comm->nRanks;
    int* send = ringSend+c*comm->nRanks;
    int* prev = ringPrev+c*comm->nRanks;
    int* next = ringNext+c*comm->nRanks;
    struct ncclChannel* channel0 = comm->channels+c;
    struct ncclChannel* channel1 = channel0+nChannels;
    for (int n=0; n<nNodes; n++) {
      int recvRank = recv[firstRanks[n]];
      int prevSendRank = send[firstRanks[(n-1+nNodes)%nNodes]];
      prev[recvRank] = prevSendRank;
      if (comm->rank == recvRank) {
        channel0->ring.prev = prevSendRank;
        channel1->ring.prev = prevSendRank;
      }
      int sendRank = send[firstRanks[n]];
      int nextRecvRank = recv[firstRanks[(n+1)%nNodes]];
      next[sendRank] = nextRecvRank;
      if (comm->rank == sendRank) {
        channel0->ring.next = nextRecvRank;
        channel1->ring.next = nextRecvRank;
      }
    }
#ifdef KERNEL_LOOPBACK
    if (comm->nRanks == 1) {
      channel0->ring.prev = -1;
      channel1->ring.prev = -1;
      channel0->ring.next = -1;
      channel1->ring.next = -1;
    }
#endif
    TRACE(NCCL_GRAPH, "Ring %d : %d -> %d -> %d", c, channel0->ring.prev, comm->rank, channel0->ring.next);
    TRACE(NCCL_GRAPH, "Ring %d : %d -> %d -> %d", c+nChannels, channel1->ring.prev, comm->rank, channel1->ring.next);
  }
  return ncclSuccess;
}

static ncclResult_t getIndexes(int* ranks, int* indexes, int nNodes, int* firstRanks) {
 for (int n=0; n<nNodes; n++) indexes[n] = ranks[firstRanks[n]];
 return ncclSuccess;
}

static ncclResult_t setTreeUp(struct ncclTree* tree, int* indexes, int u) {
  if (u == -1) return ncclSuccess;
  tree->up = indexes[u];
  return ncclSuccess;
}

static ncclResult_t setTreeDown(struct ncclTree* tree, int* indexes, int d) {
  if (d == -1) return ncclSuccess;
  int x = 0;
  while (x < NCCL_MAX_TREE_ARITY && tree->down[x] >= 0) x++;
  if (x == NCCL_MAX_TREE_ARITY) {
    ERROR("Internal error : tree already has %d children (%d %d %d)", x, tree->down[0], tree->down[1], tree->down[2]);
    return ncclInternalError;
  }
  tree->down[x] = indexes[d];
  return ncclSuccess;
}

static ncclResult_t connectTrees(ncclComm_t comm, int* treeToParent, int* treeToChild0, int* treeToChild1, int* firstRanks, int* treePatterns) {
  ncclResult_t res = ncclSuccess;
  const int nChannels = comm->nChannels, nNodes = comm->nNodes, node = comm->node;

  // Compute tree depth. Not an exact value but a good approximation in most
  // cases
  int depth = comm->nRanks/nNodes - 1 + log2i(nNodes);

  int* ranksToParent = NULL, *ranksToChild0 = NULL, *ranksToChild1 = NULL;
  NCCLCHECKGOTO(ncclCalloc(&ranksToParent, nNodes), res, end);
  NCCLCHECKGOTO(ncclCalloc(&ranksToChild0, nNodes), res, end);
  NCCLCHECKGOTO(ncclCalloc(&ranksToChild1, nNodes), res, end);

  int t0u, t0d0, t0d1, t0ChildType, t1u, t1d0, t1d1, t1ChildType;
  NCCLCHECKGOTO(ncclGetDtree(nNodes, node, &t0u, &t0d0, &t0d1, &t0ChildType, &t1u, &t1d0, &t1d1, &t1ChildType), res, end);
  for (int c=0; c<nChannels; c++) {
     struct ncclChannel* channel0 = comm->channels+c;
     struct ncclChannel* channel1 = channel0+nChannels;
     NCCLCHECKGOTO(getIndexes(treeToParent+c*comm->nRanks, ranksToParent, nNodes, firstRanks), res, end);
     NCCLCHECKGOTO(getIndexes(treeToChild0+c*comm->nRanks, ranksToChild0, nNodes, firstRanks), res, end);
     NCCLCHECKGOTO(getIndexes(treeToChild1+c*comm->nRanks, ranksToChild1, nNodes, firstRanks), res, end);
     if (comm->rank == ranksToParent[node]) {
       NCCLCHECKGOTO(setTreeUp(&channel0->tree, t0ChildType == 0 ? ranksToChild0 : ranksToChild1, t0u), res, end);
       NCCLCHECKGOTO(setTreeUp(&channel1->tree, t1ChildType == 0 ? ranksToChild0 : ranksToChild1, t1u), res, end);
     }
     if (comm->rank == ranksToChild0[node]) {
       NCCLCHECKGOTO(setTreeDown(&channel0->tree, ranksToParent, t0d0), res, end);
       NCCLCHECKGOTO(setTreeDown(&channel1->tree, ranksToParent, t1d0), res, end);
     }
     if (comm->rank == ranksToChild1[node]) {
       NCCLCHECKGOTO(setTreeDown(&channel0->tree, ranksToParent, t0d1), res, end);
       NCCLCHECKGOTO(setTreeDown(&channel1->tree, ranksToParent, t1d1), res, end);
     }
     if (comm->rank == ranksToParent[node] ||
         comm->rank == ranksToChild0[node] ||
         comm->rank == ranksToChild1[node]) {
       INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c,           channel0->tree.up, comm->rank, channel0->tree.down[0], channel0->tree.down[1], channel0->tree.down[2]);
       INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c+nChannels, channel1->tree.up, comm->rank, channel1->tree.down[0], channel1->tree.down[1], channel1->tree.down[2]);
     }
     channel0->tree.depth = channel1->tree.depth = depth;
  }

end:
  NCCLCHECK(ncclFree(ranksToParent));
  NCCLCHECK(ncclFree(ranksToChild0));
  NCCLCHECK(ncclFree(ranksToChild1));
  return res;
}

ncclResult_t ncclTopoConnectCollNet(ncclComm_t comm, struct ncclTopoGraph* collNetGraph, int rank) {
  int nranks = comm->nRanks;
  int depth = nranks/comm->nNodes;
  int sendIndex = collNetGraph->pattern == NCCL_TOPO_PATTERN_TREE ? 0 : 1;  // send GPU index depends on topo pattern
  int sendEndIndex = (sendIndex+comm->localRanks-1)%comm->localRanks;
  for (int c=0; c<comm->nChannels/2; c++) {
    struct ncclChannel* channel = comm->channels+c;
    // Set root of collTree to id nranks
    if (rank == collNetGraph->intra[sendIndex+c*comm->localRanks]) { // is master
      channel->collTree.up = nranks;
    }
    if (rank == collNetGraph->intra[sendEndIndex+c*comm->localRanks]) { // is bottom of intra-node chain
      channel->collTree.down[0] = -1;
    }
    channel->collTree.depth = depth;
    INFO(NCCL_GRAPH, "CollNet Channel %d rank %d up %d down %d", c, rank, channel->collTree.up, channel->collTree.down[0]);
  }
  int recvIndex = 0;  // recv GPU index is always 0
  int recvEndIndex = (recvIndex+comm->localRanks-1)%comm->localRanks;
  for (int c=0; c<comm->nChannels/2; c++) {
    struct ncclChannel* channel = comm->channels+comm->nChannels/2+c;
    // Set root of collTree to id nranks
    if (rank == collNetGraph->intra[recvIndex+c*comm->localRanks]) { // is master
      channel->collTree.up = nranks;
    }
    if (rank == collNetGraph->intra[recvEndIndex+c*comm->localRanks]) { // is bottom of intra-node chain
      channel->collTree.down[0] = -1;
    }
    channel->collTree.depth = depth;
    INFO(NCCL_GRAPH, "CollNet Channel %d rank %d up %d down %d", comm->nChannels/2+c, rank, channel->collTree.up, channel->collTree.down[0]);
  }
  return ncclSuccess;
}

// Legacy naming
NCCL_PARAM(MinNrings, "MIN_NRINGS", -2);
NCCL_PARAM(MaxNrings, "MAX_NRINGS", -2);
// New naming
NCCL_PARAM(MinNchannels, "MIN_NCHANNELS", -2);
NCCL_PARAM(MaxNchannels, "MAX_NCHANNELS", -2);

int ncclMinNchannels(ncclComm_t comm) {
  int minNchannels = 0;
  if (ncclParamMinNrings() != -2) minNchannels = ncclParamMinNrings();
  if (ncclParamMinNchannels() != -2) minNchannels = ncclParamMinNchannels();
  if (comm->tuner != NULL) {
    int tunerMaxChannels = -1;
    NCCLCHECK(comm->tuner->getChannels(&tunerMaxChannels));
    minNchannels = std::max(minNchannels, tunerMaxChannels);
  }
  if ((minNchannels > MAXCHANNELS)) {
    WARN("User asked for a minimum of %d channels, limiting to %d\n", minNchannels, MAXCHANNELS);
    minNchannels = MAXCHANNELS;
  }
  if (minNchannels < 0) minNchannels = 0;
  return minNchannels;
}

extern int64_t ncclParamWorkArgsBytes();

int ncclMaxNchannels(ncclComm_t comm) {
  int maxNchannels = (comm->compCap == 80) ? MAXCHANNELS : comm->smResConfig.defaultMaxChannelNum;
  if (ncclParamMaxNrings() != -2) maxNchannels = ncclParamMaxNrings();
  if (ncclParamMaxNchannels() != -2) maxNchannels = ncclParamMaxNchannels();
  maxNchannels = std::min(maxNchannels, ncclDevMaxChannelsForArgsBytes(ncclParamWorkArgsBytes()));
  if (maxNchannels > MAXCHANNELS) maxNchannels = MAXCHANNELS;
  if (maxNchannels < 1) {
    WARN("User asked for a maximum of %d channels, setting it to 1\n", maxNchannels);
    maxNchannels = 1;
  }
  return maxNchannels;
}

static int copyChannels(ncclComm_t comm, int start, int end, int* ringPrev, int* ringNext) {
  int nranks = comm->nRanks;
  int c;
  for (c=start; c<end; c++) {
    memcpy(ringPrev+c*nranks, ringPrev+(c-start)*nranks, nranks*sizeof(int));
    memcpy(ringNext+c*nranks, ringNext+(c-start)*nranks, nranks*sizeof(int));
    memcpy(comm->channels+c, comm->channels+c-start, sizeof(struct ncclChannel));
  }
  return c;
}

extern int ppuDiesNum;
ncclResult_t pcclShuffleChannel(ncclComm_t comm, struct ncclTopoGraph* graph, int ngpus) {
  if (!comm->channelShuffle) return ncclSuccess;
  // now only shuffle when the graph has replay channels
  if (graph->nSubChannels < graph->nChannels && graph->nSubChannels > 1) {
    int finalChannelsPerDie = std::min(std::min(graph->nChannels*2, std::max(ncclMaxNchannels(comm), ncclMinNchannels(comm))), comm->config.maxCTAs) / ppuDiesNum;
    int repeat = graph->nGraphChannels / graph->nSubChannels;
    bool isPerfectRepetition = finalChannelsPerDie % graph->nSubChannels == 0 ?  true : false;
    // check if the graph channels are perfect repetition of sub channels
    for (int c=1; isPerfectRepetition && c<graph->nGraphChannels; c++) {
      if (memcmp(graph->intra+c*ngpus, graph->subIntra+c/repeat*ngpus, sizeof(int)*ngpus) != 0) {
        isPerfectRepetition = false;
        break;
      }
    }
    if (isPerfectRepetition) {
      VERBOSE(NCCL_GRAPH, "Start to shuffle channels, num of sub channels: %d, num of graph channels %d, final channels per die %d num of dies %d",
                          graph->nSubChannels, graph->nGraphChannels, finalChannelsPerDie, ppuDiesNum);
      // per die channel shuffle, try put the different channels to different CEs when enable one batch per ce
      for (int c=0; c<finalChannelsPerDie; c++) {
        memcpy(graph->intra+c*ngpus, graph->subIntra+(c%graph->nSubChannels)*ngpus, ngpus*sizeof(int));
      }
      // duplicated for multi-die
      for (int d=1; d<ppuDiesNum; d++) {
        memcpy(graph->intra+d*finalChannelsPerDie*ngpus, graph->intra, finalChannelsPerDie*ngpus*sizeof(int));
      }
    }
  }
  return ncclSuccess;
}

PCCL_PARAM(QuadrupleChannels, "QUADRUPLE_CHANNELS", 0);
ncclResult_t ncclTopoPostset(ncclComm_t comm, int* firstRanks, int* treePatterns, struct ncclTopoRanks** allTopoRanks, int* rings) {
  ncclResult_t res = ncclSuccess;
  // Gather data from all ranks
  int *ringRecv = NULL, *ringSend = NULL, *ringPrev = NULL, *ringNext = NULL, *treeToParent = NULL, *treeToChild0 = NULL, *treeToChild1 = NULL;
  int nranks = comm->nRanks;
  int nChannels = comm->nChannels;
  int envMaxChannels = ncclMaxNchannels(comm);
  int envMinChannels = ncclMinNchannels(comm);
  NCCLCHECKGOTO(ncclCalloc(&ringRecv, nranks*MAXCHANNELS), res, end);
  NCCLCHECKGOTO(ncclCalloc(&ringSend, nranks*MAXCHANNELS), res, end);
  NCCLCHECKGOTO(ncclCalloc(&ringPrev, nranks*MAXCHANNELS), res, end);
  NCCLCHECKGOTO(ncclCalloc(&ringNext, nranks*MAXCHANNELS), res, end);
  NCCLCHECKGOTO(ncclCalloc(&treeToParent, nranks*MAXCHANNELS), res, end);
  NCCLCHECKGOTO(ncclCalloc(&treeToChild0, nranks*MAXCHANNELS), res, end);
  NCCLCHECKGOTO(ncclCalloc(&treeToChild1, nranks*MAXCHANNELS), res, end);
  for (int i=0; i<nranks; i++) {
    for (int c=0; c<nChannels; c++) {
      ringRecv[c*nranks+i] = allTopoRanks[i]->ringRecv[c];
      ringSend[c*nranks+i] = allTopoRanks[i]->ringSend[c];
      ringPrev[c*nranks+i] = allTopoRanks[i]->ringPrev[c];
      ringNext[c*nranks+i] = allTopoRanks[i]->ringNext[c];
      treeToParent[c*nranks+i] = allTopoRanks[i]->treeToParent[c];
      treeToChild0[c*nranks+i] = allTopoRanks[i]->treeToChild0[c];
      treeToChild1[c*nranks+i] = allTopoRanks[i]->treeToChild1[c];
    }
  }

  // Connect rings and trees. This should also duplicate the channels.
  NCCLCHECKGOTO(connectRings(comm, ringRecv, ringSend, ringPrev, ringNext, firstRanks), res, end);
  NCCLCHECKGOTO(connectTrees(comm, treeToParent, treeToChild0, treeToChild1, firstRanks, treePatterns), res, end);

  // Duplicate ringPrev/ringNext for ncclBuildRing
  memcpy(ringPrev+nChannels*nranks, ringPrev, nChannels*nranks*sizeof(int));
  memcpy(ringNext+nChannels*nranks, ringNext, nChannels*nranks*sizeof(int));

  // Duplication should be complete now
  nChannels = comm->nChannels = std::min(std::min(MAXCHANNELS,nChannels*2), comm->config.maxCTAs);
  if (comm->fcnChannels > 0) {
    comm->fcnChannels = std::min(MAXCHANNELS, comm->fcnChannels*2);
  }

  // Quadruple channels number for ppu performance
  if (pcclParamQuadrupleChannels() == 1) {
    if (nChannels < MAXCHANNELS) {
      comm->nChannels = std::min(MAXCHANNELS,nChannels*2);
      copyChannels(comm, nChannels, comm->nChannels, ringPrev, ringNext);
      nChannels = comm->nChannels;
    }
    if (comm->fcnChannels < MAXCHANNELS) {
      int fcnChannels = std::min(MAXCHANNELS, comm->fcnChannels*2);
      if (fcnChannels > comm->nChannels) {
        int extraChannels = fcnChannels - comm->nChannels;
        memcpy(comm->channels+comm->nChannels, comm->channels, extraChannels*sizeof(struct ncclChannel));
      }
      comm->fcnChannels = fcnChannels;
    }
  }

  // Honor NCCL_MIN_NRINGS/NCCL_MAX_NRINGS.
  // We permit combining max, then min, to only use the first channels, then duplicate them.
  if (comm->sharedRes->owner != comm) {
    /* child comm #channels cannot exceed top parent #channels. */
    nChannels = comm->nChannels = std::min(std::min(std::min(envMaxChannels, nChannels), comm->config.maxCTAs), comm->sharedRes->tpNChannels);
    if (comm->fcnChannels > 0) {
      comm->fcnChannels = std::min(std::min(std::min(envMaxChannels, comm->fcnChannels), comm->config.maxCTAs), comm->sharedRes->tpNChannels);
    }
    nChannels = comm->nChannels = copyChannels(comm, nChannels, std::min(envMinChannels, comm->sharedRes->tpNChannels), ringPrev, ringNext);
    if (comm->fcnChannels > 0) {
      comm->fcnChannels = std::min(std::max(envMinChannels, comm->fcnChannels), comm->sharedRes->tpNChannels);
    }
  } else {
    nChannels = comm->nChannels = std::min(std::min(envMaxChannels, nChannels), comm->config.maxCTAs);
    if (comm->fcnChannels > 0) {
      comm->fcnChannels = std::min(envMaxChannels, comm->fcnChannels);
    }
    nChannels = comm->nChannels = copyChannels(comm, nChannels, envMinChannels, ringPrev, ringNext);
    if (comm->fcnChannels > 0) {
      comm->fcnChannels = std::max(envMinChannels, comm->fcnChannels);
    }
  }

  // Create rings array and check all is fine
  NCCLCHECKGOTO(ncclBuildRings(nChannels, rings, comm->rank, comm->nRanks, ringPrev, ringNext), res, end);

  if (ncclHggcCompCap() == 80 && comm->channelShuffle) {
    int ratio = comm->nChannels / comm->nGraphChannels;
    for (int c = comm->nChannels-1; c>0; c--) {
      int c0 = c / ratio;
      int c1 = c;
      memcpy(comm->channels+c1, comm->channels+c0, sizeof(struct ncclChannel));
      memcpy(rings+c1*nranks, rings+c0*nranks, nranks*sizeof(int));
    }
    for (int c = 0; c < comm->nChannels; c++) {
      char prefix[40];
      sprintf(prefix, "Shuffle Channel %02d/%02d :", c, comm->nChannels);
      dumpLine(rings+c*nranks, comm->nRanks, prefix);
    }
  }

end:
  if (ringRecv) free(ringRecv);
  if (ringSend) free(ringSend);
  if (ringPrev) free(ringPrev);
  if (ringNext) free(ringNext);
  if (treeToParent) free(treeToParent);
  if (treeToChild0) free(treeToChild0);
  if (treeToChild1) free(treeToChild1);

  return res;
}
