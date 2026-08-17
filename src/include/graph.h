/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_GRAPH_H_
#define NCCL_GRAPH_H_

#include "pccl_internal.h"
#include "devcomm.h"
#include <limits.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
#include <sched.h>
#include "check_items.h"

ncclResult_t ncclTopoHggcPath(int hggcDev, char** path);

struct ncclTopoSystem;
// Build the topology
ncclResult_t ncclTopoGetSystem(ncclComm_t comm, struct ncclTopoSystem** system);
ncclResult_t ncclTopoSortSystem(struct ncclTopoSystem* system);
ncclResult_t ncclTopoPrint(ncclComm_t comm, struct ncclTopoSystem* system);
ncclResult_t ncclTopoPrintPaths(struct ncclTopoSystem* system);

ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system, ncclComm_t comm);
void ncclTopoFree(struct ncclTopoSystem* system);
ncclResult_t ncclTopoTrimSystem(struct ncclTopoSystem* system, ncclComm_t comm);
ncclResult_t ncclTopoAdjustSystemImpl(struct ncclTopoSystem* system);
ncclResult_t ncclTopoAdjustSystem(struct ncclTopoSystem* system);
ncclResult_t ncclTopoComputeP2pChannels(ncclComm_t comm);
ncclResult_t ncclTopoComputeP2pChannelsPerPeer(ncclComm_t comm);

// Query topology
ncclResult_t ncclTopoGetMinPaths(struct ncclTopoSystem* system, int id1, int id2, int* minPathNum);
ncclResult_t ncclTopoGetConnType(struct ncclTopoSystem* system, int id1, int id2, ncclConn_t* connType);
ncclResult_t pcclFillPPUTopoArray(int *ppuTopoArray, ppuDeviceInfo *ppuDevInfo);
ncclResult_t pcclFillPPUTopoArray(int *ppuTopoArray, ncclComm_t comm);
ncclResult_t ncclTopoGetNetDev(ncclComm_t comm, int rank, struct ncclTopoGraph* graph, int channelId, int peerRank, int64_t* id, int* dev, int* proxyRank);
ncclResult_t ncclTopoCheckP2p(ncclComm_t comm, struct ncclTopoSystem* system, int rank1, int rank2, int* p2p, int *read, int* intermediateRank);
ncclResult_t ncclTopoCheckGlobalIcnP2p(ncclComm_t comm, struct ncclTopoSystem* system, bool* globalIcnP2pSupport);
ncclResult_t ncclTopoCheckMNNVL(struct ncclTopoSystem* system, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2, int* ret);
ncclResult_t ncclTopoCheckGdr(struct ncclTopoSystem* topo, int rank, int64_t netId, int read, int* useGdr);
ncclResult_t ncclTopoCheckCrossNicSupport(bool* supported);
int ncclPxnDisable();
ncclResult_t ncclTopoGetPxnRanks(ncclComm_t comm, int** intermediateRanks, int* nranks);
ncclResult_t ncclTopoGetLocalRank(ncclComm_t comm, int rank, int* localRank);

ncclResult_t ncclGetUserP2pLevel(int* level);

// Find CPU affinity
ncclResult_t ncclTopoGetCpuAffinity(struct ncclTopoSystem* system, int rank, cpu_set_t* affinity);

#define NCCL_TOPO_CPU_ARCH_X86 1
#define NCCL_TOPO_CPU_ARCH_POWER 2
#define NCCL_TOPO_CPU_ARCH_ARM 3
#define NCCL_TOPO_CPU_ARCH_MIXED 4
#define NCCL_TOPO_CPU_VENDOR_INTEL 1
#define NCCL_TOPO_CPU_VENDOR_AMD 2
#define NCCL_TOPO_CPU_VENDOR_ZHAOXIN 3
#define NCCL_TOPO_CPU_VENDOR_HYGON 4
#define NCCL_TOPO_CPU_VENDOR_MIXED 5
#define NCCL_TOPO_CPU_MODEL_INTEL_BDW 1
#define NCCL_TOPO_CPU_MODEL_INTEL_SKL 2
#define NCCL_TOPO_CPU_MODEL_INTEL_SRP 3
#define NCCL_TOPO_CPU_MODEL_INTEL_ERP 4
#define NCCL_TOPO_CPU_MODEL_YONGFENG 1

ncclResult_t ncclTopoCpuType(struct ncclTopoSystem* system, int* arch, int* vendor, int* model);
ncclResult_t ncclTopoGetNetCount(struct ncclTopoSystem* system, int* count);
ncclResult_t ncclParseList(const char* str, const char* prefixElems[], int nprefixes, const char* elems[], int nelems, int* list);
ncclResult_t ncclTopoGetLocalNet(struct ncclTopoSystem* system, int rank, int64_t* id, int* dev);
ncclResult_t ncclTopoGetLocalGinDevs(ncclComm_t comm, int* localGinDevs, int* localGinCount);

#define NCCL_TOPO_MAX_NODES 256

// Init search. Needs to be done before calling ncclTopoCompute
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system);

#define NCCL_TOPO_PATTERN_BALANCED_TREE 1   // Spread NIC traffic between two GPUs (Tree parent + one child on first GPU, second child on second GPU)
#define NCCL_TOPO_PATTERN_SPLIT_TREE 2      // Spread NIC traffic between two GPUs (Tree parent on first GPU, tree children on the second GPU)
#define NCCL_TOPO_PATTERN_TREE 3            // All NIC traffic going to/from the same GPU
#define NCCL_TOPO_PATTERN_RING 4            // Ring
#define NCCL_TOPO_PATTERN_FC 5              // FC
struct ncclTopoGraph {
  // Input / output
  int id; // ring : 0, tree : 1, collnet : 2, fc : 3
  int pattern;
  int crossNic;
  int collNet;
  int minChannels;
  int maxChannels;
  // Output
  int nSubChannels;
  int nGraphChannels;
  int nChannels;
  float speedIntra;
  float speedInter;
  int typeIntra;
  int typeInter;
  int sameChannels;
  int nHops;
  int intra[MAXCHANNELS*NCCL_TOPO_MAX_NODES];
  int subIntra[MAXCHANNELS*NCCL_TOPO_MAX_NODES];
  int64_t inter[MAXCHANNELS*2];
  int proxyDevs[MAXCHANNELS]; // PXN proxy GPU dev per channel (node-local index), -1 = no PXN
};
ncclResult_t ncclTopoCompute(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);

ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);
ncclResult_t ncclTopoDumpGraphs(struct ncclTopoSystem* system, int ngraphs, struct ncclTopoGraph** graphs);
ncclResult_t ncclTopoDumpRings(int *rings, int nrings, int nranks);
ncclResult_t ncclTopoDevToRank(struct ncclTopoSystem* system, int systemId, int hgmlDev, int *rank);
ncclResult_t ncclTopoIdToNetDev(struct ncclTopoSystem* system, int64_t id, int* netDev);
ncclResult_t pcclShuffleChannel(ncclComm_t comm, struct ncclTopoGraph* graph, int ngpus);

struct ncclTopoRanks {
  int ringRecv[MAXCHANNELS];
  int ringSend[MAXCHANNELS];
  int ringPrev[MAXCHANNELS];
  int ringNext[MAXCHANNELS];
  int treeToParent[MAXCHANNELS];
  int treeToChild0[MAXCHANNELS];
  int treeToChild1[MAXCHANNELS];
};

ncclResult_t ncclTopoPreset(ncclComm_t comm,
    struct ncclTopoGraph* treeGraph, struct ncclTopoGraph* ringGraph, struct ncclTopoGraph* collNetGraph,
    struct ncclTopoGraph* fcGraph, struct ncclTopoRanks* topoRanks);

ncclResult_t ncclTopoPostset(ncclComm_t comm, int* firstRanks, int* treePatterns,
    struct ncclTopoRanks** allTopoRanks, int* rings);

ncclResult_t ncclTopoConnectCollNet(ncclComm_t comm, struct ncclTopoGraph* collNetGraph, int rank);

ncclResult_t ncclTopoTuneModel(ncclComm_t comm, int minCompCap, int maxCompCap, struct ncclTopoGraph* treeGraph, struct ncclTopoGraph* ringGraph, struct ncclTopoGraph* collNetGraph, struct ncclTopoGraph* fcGraph);
#include "info.h"
ncclResult_t ncclTopoGetAlgoTime(ncclComm_t comm, int coll, int algorithm, int protocol, size_t nBytes, int nChannels, float* time);

#endif
