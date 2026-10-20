/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_TOPO_H_
#define NCCL_TOPO_H_

#include "graph.h"
#include "core.h"
#include <sched.h>

#define LOC_WIDTH 5000.0
#define PCI_WIDTH 12.0           // PCI Gen3 x16
#define AMD_WIDTH 16.0
#define BDW_QPI_WIDTH 6.0
#define SKL_QPI_WIDTH 10.0
#define SRP_QPI_WIDTH 22.0
#define ERP_QPI_WIDTH 40.0
#define HYGON_WIDTH 6.0 // TODO: a speculative value. To be determined.
#define ZPI_BW 6.0
#define YONGFENG_ZPI_BW 9.0
#define P9_WIDTH 32.0
#define ARM_WIDTH 6.0
#define NET_WIDTH 12.0           // 100Gbit

#define ICN_WIDTH 42.0
#define ICN_MAX_WIDTH 50.0

// Intel CPU convert GPU P2P traffic into 64B PCI TLPs, so GPU
// to GPU traffic consumes more PCI bandwidth.
#define INTEL_P2P(speed) (speed*9/12)
#define INTEL_P2P_OVERHEAD(speed) (speed*12/9)

#define NCCL_TOPO_NODE_TYPES 6
#define GPU 0
#define PCI 1
#define NVS 2
#define CPU 3 // Actually NUMA domains
#define NIC 4
#define NET 5

extern const char* topoNodeTypeStr[];

// We want link types and path types to match as much as possible
#define LINK_LOC 0
#define LINK_NVL 1
// Skipping 2 for PATH_NVB (PPU)
#define LINK_PCI 3
// Skipping 4 for PATH_PXB
// Skipping 5 for PATH_PXN
// Skipping 6 for PATH_PXR
// Skipping 7 for PATH_PHBH
// Skipping 8 for PATH_PHB
#define LINK_SYS 9
#define LINK_NET 10

extern const char* topoLinkTypeStr[];

#define PATH_LOC 0
#define PATH_NVL 1  // single step device link (icnlink / nvlink)
#define PATH_NVB 2  // multi-step device link (icn route / nvswitch)
#define PATH_PIX 3
#define PATH_PXB 4
#define PATH_PXN 5
#define PATH_PXR 6 // ICR LINK + Network
#define PATH_PHBH 7 // high priority PHB when graph search
#define PATH_PHB 8
#define PATH_SYS 9

extern const char* topoPathTypeStr[];

struct ncclTopoNode;
struct ncclTopoLink {
  int type;
  int count;      // link count
  float width;    // total width
  struct ncclTopoNode* remNode;
};
// change 32 to be 64, for  local multi-ppus scaling out tests
#define NCCL_TOPO_MAX_LINKS 64
#define NCCL_TOPO_MAX_HOPS (NCCL_TOPO_MAX_NODES*NCCL_TOPO_NODE_TYPES)

struct ncclTopoLinkList {
  struct ncclTopoLink* list[NCCL_TOPO_MAX_HOPS];
  int count;
  float width;
  int type;
  int minPathNum;
};

#define NCCL_TOPO_CPU_INTEL_BDW 1
#define NCCL_TOPO_CPU_INTEL_SKL 2

#define NCCL_TOPO_UNDEF (-1)

#define NCCL_TOPO_ID_LOCAL_ID_MASK 0x00ffffffffffffff
#define NCCL_TOPO_ID_SYSTEM_ID(id) (id >> 56)
#define NCCL_TOPO_ID_LOCAL_ID(id) (id & NCCL_TOPO_ID_LOCAL_ID_MASK)
#define NCCL_TOPO_LOCAL_NIC_ID(numaid, busid) (((int64_t)numaid << 56) + busid)
#define NCCL_TOPO_ID(systemid, localid) (((int64_t)systemid << 56) + (localid & NCCL_TOPO_ID_LOCAL_ID_MASK))

struct ncclTopoNode {
  int type;
  int64_t id;
  // Type specific data
  union {
    struct {
      int dev; // NVML dev number
      int rank;
      int hostId;
      int hggcCompCap;
      int gdrSupport;
      int64_t netAffinity; // netId
    }gpu;
    struct {
      int dev; // Plugin dev number
      uint64_t asic;
      int port;
      float width;
      int gdrSupport;
      int collSupport;
      int maxChannels;
    }net;
    struct {
      int arch;
      int vendor;
      int model;
      cpu_set_t affinity;
    }cpu;
    struct {
      uint64_t device;
    }pci;
  };
  int nlinks;
  struct ncclTopoLink links[NCCL_TOPO_MAX_LINKS];
  // number of shortest paths per peer
  int* npaths[NCCL_TOPO_NODE_TYPES];
  // Pre-computed paths to GPUs and NICs
  struct ncclTopoLinkList* paths[NCCL_TOPO_NODE_TYPES];
  // Used during search
  uint64_t used;
  bool local;
};

struct ncclTopoNodeSet {
  int count;
  struct ncclTopoNode nodes[NCCL_TOPO_MAX_NODES];
};

struct ncclTopoSystem {
  bool isEicNet;
  int systemId;
  uint64_t hostHashes[NCCL_TOPO_MAX_NODES];
  int nHosts;
  struct ncclTopoNodeSet nodes[NCCL_TOPO_NODE_TYPES];
  float maxWidth;
  float totalWidth;
  float totalNetWidth;
  int icnFullyConnected; // all devices in this topo can connect through icnlink
};

struct ncclHostNeighborInfo {
  int left;
  int right;
  int up;
  int down;
  int leftDevs[2];
  int rightDevs[2];
};

ncclResult_t ncclTopoGetNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, int64_t id);
ncclResult_t ncclTopoCreateNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, int64_t id);
ncclResult_t ncclTopoRemoveNode(struct ncclTopoSystem* system, int type, int id);
ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, int count, float width);
ncclResult_t ncclTopoPrintPaths(struct ncclTopoSystem* system);
ncclResult_t ncclTopoLoadSystem(const char* xmlTopoFile, struct ncclTopoSystem* system);
ncclResult_t ncclTopoGetIntermediateRank(struct ncclTopoSystem* system, int rank, int64_t netId, int* intermediateRank);

ncclResult_t ncclTopoGetLocalNet(struct ncclTopoSystem* system, int rank, int64_t* id, int* dev);
ncclResult_t ncclTopoGetBalancedProxyRank(struct ncclTopoSystem* system, int g, int n, int* proxyRank);

#define NCCL_TOPO_XML_MAX_NODES 256
#define NCCL_GRAPH_XML_MAX_NODES 4096
ncclResult_t ncclTopoGetSystemFromXml(struct ncclXml* xml, struct ncclTopoSystem** topoSystem, const uint64_t localHostHash, bool isEicNet = false);
ncclResult_t ncclTopoGetGraphFromXml(struct ncclXmlNode *xmlGraphs, struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int* nChannels);
ncclResult_t ncclTopoGetXmlFromGraphs(int ngraphs, struct ncclTopoGraph** graphs, struct ncclTopoSystem* system, struct ncclXml *xml);
ncclResult_t pcclAffinityNetsPathAdjust(struct ncclTopoSystem* system);
ncclResult_t pcclTopoSetNetAffinityPPUs(struct ncclXml* xml, int ppuCount, int netDevCount);
int pcclGetPathTypeAlias(int type);

ncclResult_t ncclParseVisibleDevices(const int ppuCount, int* visibleOrderDevices);
ncclResult_t ncclTopoGetHggcDevByHgmlDev(const int hgmlDev, int* hggcDev);

static ncclResult_t ncclTopoIdToIndex(struct ncclTopoSystem* system, int type, int64_t id, int* index) {
  *index = -1;
  for (int i=0; i<system->nodes[type].count; i++) {
    if (system->nodes[type].nodes[i].id == id) {
      *index = i;
      return ncclSuccess;
    }
  }
  return ncclInternalError;
}

static ncclResult_t ncclTopoRankToIndex(struct ncclTopoSystem* system, int rank, int* index) {
  *index = -1;
  for (int i=0; i<system->nodes[GPU].count; i++) {
    if (system->nodes[GPU].nodes[i].gpu.rank == rank) {
      *index = i;
      return ncclSuccess;
    }
  }
  return ncclInternalError;
}

// Returns NVLink speed in GB/s
static float ncclTopoNVLinkSpeed(int hggcCompCap) {
  return ICN_WIDTH;
}

// Returns theoretical link speed in GB/s
static float ncclTopoIcnLinkTheoreticalSpeed() {
  return ICN_MAX_WIDTH;
}

static int mirrorBits(int val, int pow2) {
  int mirror = 0;
  for (int b=1, mb=(pow2>>1); b<pow2; b<<=1, mb>>=1) if (val & b) mirror |= mb;
  return mirror;
}

#endif
