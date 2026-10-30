/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "core.h"
#include "graph.h"
#include "topo.h"
#include "comm.h"
#include "net.h"
#include "channel.h"
#include "transport.h"

// Pre-compute GPU->NIC, GPU->GPU and NIC->GPU paths

struct ncclTopoNodeList {
  struct ncclTopoNode* list[NCCL_TOPO_MAX_NODES];
  int count;
};

static ncclResult_t getPathByBusId(struct ncclTopoSystem* system, struct ncclTopoNode* n, int t, int64_t id, struct ncclTopoLinkList** path, int** npaths=NULL) {
  for (int i=0; i<system->nodes[t].count; i++) {
    if (system->nodes[t].nodes[i].id == id) {
      *path = n->paths[t]+i;
      if (npaths != NULL) *npaths = n->npaths[t]+i;
      return ncclSuccess;
    }
  }
  ERROR("Could not find node of type %s id %lx", topoLinkTypeStr[t], id);
  return ncclInternalError;
}

static ncclResult_t getPath(struct ncclTopoSystem* system, struct ncclTopoNode* node1, struct ncclTopoNode* node2, struct ncclTopoLinkList** path, int** npaths=NULL) {
  return getPathByBusId(system, node1, node2->type, node2->id, path, npaths);
}

NCCL_PARAM(NvbDisable, "NVB_DISABLE", 0);

static ncclResult_t ncclTopoSetPaths(struct ncclTopoNode* baseNode, struct ncclTopoSystem* system) {
  if (baseNode->paths[baseNode->type] == NULL) {
    NCCLCHECK(ncclCalloc(baseNode->paths+baseNode->type, system->nodes[baseNode->type].count));
    NCCLCHECK(ncclCalloc(baseNode->npaths+baseNode->type, system->nodes[baseNode->type].count));
  }

  // breadth-first search to set all paths to that node in the system
  struct ncclTopoNodeList nodeList;
  struct ncclTopoNodeList nextNodeList;
  nodeList.count = 1; nodeList.list[0] = baseNode;
  nextNodeList.count = 0;
  int *baseNpaths;
  struct ncclTopoLinkList* basePath;
  NCCLCHECK(getPath(system, baseNode, baseNode, &basePath, &baseNpaths));
  basePath->count = 0;
  basePath->width = LOC_WIDTH;
  basePath->type = PATH_LOC;
  basePath->minPathNum = 0;
  *baseNpaths = 1;

  while (nodeList.count) {
    nextNodeList.count = 0;
    for (int n=0; n<nodeList.count; n++) {
      // if an node is already in the nextNodeList, its path has been
      // updated by preceding node(s) in the nodeList
      bool updated = false;
      for (int i=0; i<nextNodeList.count; i++) {
        if (nextNodeList.list[i] == nodeList.list[n]) {
          updated = true; break;
        }
      }
      if (updated) continue;

      struct ncclTopoNode* node = nodeList.list[n];
      struct ncclTopoLinkList* path;
      NCCLCHECK(getPath(system, node, baseNode, &path));
      for (int l=0; l<node->nlinks; l++) {
        struct ncclTopoLink* link = node->links+l;
        struct ncclTopoNode* remNode = link->remNode;
        if (remNode->paths[baseNode->type] == NULL) {
          NCCLCHECK(ncclCalloc(remNode->paths+baseNode->type, system->nodes[baseNode->type].count));
          NCCLCHECK(ncclCalloc(remNode->npaths+baseNode->type, system->nodes[baseNode->type].count));
        }
        int *remNpaths;
        struct ncclTopoLinkList* remPath;
        NCCLCHECK(getPath(system, remNode, baseNode, &remPath, &remNpaths));
        float width = std::min(path->width, link->width);

        if (node != baseNode && node->type == GPU &&
            (remNode->type != GPU || link->type != LINK_NVL
              /*routing through icn, or p2p connect through pci or net...*/
              || (link->type == LINK_NVL && path->type > PATH_NVB)
              || ncclParamNvbDisable() == 1)) continue;

        if ((remPath->width == 0 || remPath->count > path->count ||
             remPath->type > path->type) && remPath->width < width) {
          // Find reverse link
          for (int l=0; l<remNode->nlinks; l++) {
            if (remNode->links[l].remNode == node) {
              remPath->list[0] = remNode->links+l;
              break;
            }
          }
          if (remPath->list[0] == NULL) {
            WARN("Failed to find reverse path from remNode %d/%lx nlinks %d to node %d/%lx",
                 remNode->type, remNode->id, remNode->nlinks, node->type, node->id);
            return ncclInternalError;
          }
          // Copy the rest of the path
          for (int i=0; i<path->count; i++) remPath->list[i+1] = path->list[i];
          remPath->count = path->count + 1;
          remPath->width = width;
          *remNpaths = 1;
          if (path->count == 0) remPath->minPathNum = link->count;
          else remPath->minPathNum = std::min(path->minPathNum, link->count);

          // Start with path type = link type. PATH and LINK types are supposed to match.
          // Don't consider LINK_NET as we only care about the NIC->GPU path.
          int type = link->type == LINK_NET ? LINK_LOC : link->type;
          // Differentiate between one and multiple PCI switches
          if (node->type == PCI && remNode->type == PCI) type = PATH_PXB;
          // Consider a path going through the CPU as PATH_PHB
          if (link->type == LINK_PCI && (node->type == CPU || link->remNode->type == CPU)) type = PATH_PHB;
          // Set multi-hop icnlink or one-hop nvlink as NVB
          if (node->type == GPU && path->type == PATH_NVL && type <= PATH_NVB && remPath->count > 1) type = PATH_NVB;

          remPath->type = std::max(path->type, type);

          // Add to the list for the next iteration if not already in the list
          int i;
          for (i=0; i<nextNodeList.count; i++) if (nextNodeList.list[i] == remNode) break;
          if (i == nextNodeList.count) nextNodeList.list[nextNodeList.count++] = remNode;
        } else if (remPath->count == path->count + 1 &&
                   (remPath->type == path->type
                   || (remPath->type == PATH_NVB && path->type == PATH_NVL)
                   )) {
          // count number of shortest paths (width can be various)
          (*remNpaths)++;
          remPath->minPathNum += std::min(path->minPathNum, link->count);
        }
      }
    }
    memcpy(&nodeList, &nextNodeList, sizeof(nodeList));
  }
  return ncclSuccess;
}

static void printNodePaths(struct ncclTopoSystem* system, struct ncclTopoNode* node) {
  const int linesize = LARGE_LOG_SIZE;
  char line[linesize];

#ifdef ENABLE_TRACE
  INFO(NCCL_GRAPH, "Paths from %s/%lX :", topoNodeTypeStr[node->type], node->id);
#else
  snprintf(line, linesize, "%s/%lX :", topoNodeTypeStr[node->type], node->id);
  int offset = strlen(line);
#endif
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) {
    if (node->paths[t] == NULL) continue;
    for (int n = 0; n<system->nodes[t].count; n++) {
#ifdef ENABLE_TRACE
      line[0] = 0;
      int offset = 0;
      for (int i=0; i<node->paths[t][n].count; i++) {
        struct ncclTopoLink* link = node->paths[t][n].list[i];
        struct ncclTopoNode* remNode = link->remNode;
        int ret = snprintf(line+offset, linesize-offset, "--%s->%s/%lX", topoLinkTypeStr[link->type], topoNodeTypeStr[remNode->type], remNode->id);
        offset = strlen(line);
        if (ret <= 0) {
          WARN("line length excced %d, skip type %s, node %d's %d path info", linesize, topoNodeTypeStr[t], system->nodes[t].nodes[n].id, n);
          break;
        }
      }
      INFO(NCCL_GRAPH, "%s(%f)(npaths %d, minpaths %d)", line, node->paths[t][n].width, node->npaths[t][n], node->paths[t][n].minPathNum);
#else
      int ret = snprintf(line+offset, linesize-offset, " %4s/%lX(%d/%6.1f/%4s)(%d/%d)", topoNodeTypeStr[t], system->nodes[t].nodes[n].id, node->paths[t][n].count, node->paths[t][n].width, topoPathTypeStr[node->paths[t][n].type], node->npaths[t][n], node->paths[t][n].minPathNum);
      offset = strlen(line);
      if (ret <= 0) {
        WARN("line length excced %d, skip type %s, node %d's %d path info", linesize, topoNodeTypeStr[t], system->nodes[t].nodes[n].id, n);
        break;
      }
#endif
    }
  }

#ifndef ENABLE_TRACE
  INFO(NCCL_GRAPH, "%s", line);
#endif
}

ncclResult_t ncclTopoPrintPaths(struct ncclTopoSystem* system) {
  for (int i=0; i<system->nodes[GPU].count; i++) {
    printNodePaths(system, system->nodes[GPU].nodes+i);
  }
  for (int i=0; i<system->nodes[NET].count; i++) {
    printNodePaths(system, system->nodes[NET].nodes+i);
  }
  return ncclSuccess;
}

static ncclResult_t getLocalCpu(struct ncclTopoSystem* system, int gpu, int* retCpu) {
  // Find the closest CPU to a GPU
  int minHops = 0;
  int localCpu = -1;
  struct ncclTopoLinkList* paths = system->nodes[GPU].nodes[gpu].paths[CPU];
  for (int c=0; c<system->nodes[CPU].count; c++) {
    int hops = paths[c].count;
    if (minHops == 0 || hops < minHops) {
      localCpu = c;
      minHops = hops;
    }
  }
  if (localCpu == -1) {
    ERROR("Error : could not find CPU close to PPU %d", gpu);
    return ncclInternalError;
  }
  *retCpu = localCpu;
  return ncclSuccess;
}

static ncclResult_t addInterStep(struct ncclTopoSystem* system, int tx, int ix, int t1, int i1, int t2, int i2) {
  struct ncclTopoNode* cpuNode = system->nodes[tx].nodes+ix;
  struct ncclTopoNode* srcNode = system->nodes[t1].nodes+i1;

  int l=0;
  // Node 1 -> CPU
  for (int i=0; i<srcNode->paths[tx][ix].count; i++) srcNode->paths[t2][i2].list[l++] = srcNode->paths[tx][ix].list[i];
  // CPU -> Node 2
  for (int i=0; i<cpuNode->paths[t2][i2].count; i++) srcNode->paths[t2][i2].list[l++] = cpuNode->paths[t2][i2].list[i];

  // Update path characteristics
  srcNode->paths[t2][i2].count = l;
  srcNode->paths[t2][i2].type = std::max(srcNode->paths[tx][ix].type, cpuNode->paths[t2][i2].type);
  if (tx == GPU) {
    if (cpuNode->paths[GPU][i1].type == PATH_NVL)
      srcNode->paths[t2][i2].type = PATH_PXN;
    else
      srcNode->paths[t2][i2].type = PATH_PXR;
  }
  srcNode->paths[t2][i2].width = std::min(srcNode->paths[tx][ix].width, cpuNode->paths[t2][i2].width);
  return ncclSuccess;
}

// Remove/free paths for a given type
static void ncclTopoRemovePathType(struct ncclTopoSystem* system, int nodeType) {
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) {
    // Remove links _to_ the given type
    for (int n=0; n<system->nodes[t].count; n++) {
      struct ncclTopoNode* node = system->nodes[t].nodes+n;
      ncclFree(node->paths[nodeType]);
      ncclFree(node->npaths[nodeType]);
    }
    // Remove links _from_ the given type
    for (int n=0; n<system->nodes[nodeType].count; n++) {
      struct ncclTopoNode* node = system->nodes[nodeType].nodes+n;
      ncclFree(node->paths[t]);
      ncclFree(node->npaths[t]);
    }
  }
}

static const int levelsOldToNew[] = { PATH_LOC, PATH_PIX, PATH_PXB, PATH_PHBH, PATH_PHB, PATH_SYS, PATH_SYS };
ncclResult_t ncclGetLevel(int* level, const char* disableEnv, const char* levelEnv) {
  if (*level == -1) {
    int l = -1;
    if (disableEnv) {
      const char* str = pcclGetEnvWithDualPrefix(disableEnv);
      if (str) {
        int disable = strtol(str, NULL, 0);
        if (disable == 1) l = 0;
      }
    }
    if (l == -1) {
      const char* str = pcclGetEnvWithDualPrefix(levelEnv);
      if (str) {
        for (int i=0; i<=PATH_SYS; i++) {
          if (strcmp(str, topoPathTypeStr[i]) == 0) {
            l = i;
            break;
          }
        }
        // Old style numbering
        if (l == -1 && str[0] >= '0' && str[0] <= '9') {
          int oldLevel = strtol(str, NULL, 0);
          const int maxOldLevel = sizeof(levelsOldToNew)/sizeof(int) - 1;
          if (oldLevel > maxOldLevel) oldLevel = maxOldLevel;
          l = levelsOldToNew[oldLevel];
        }
      }
    }
    if (l >= 0) INFO(NCCL_ALL, "%s set by environment to %s", levelEnv, topoPathTypeStr[l]);
    *level = l >= 0 ? l : -2;
  }
  return ncclSuccess;
}

extern int ppuDiesNum;
PCCL_PARAM(GetMinPathNumByDriver, "GET_MIN_PATH_NUM_BY_DRIVER", -2);
ncclResult_t ncclTopoGetMinPaths(struct ncclTopoSystem* system, int rank1, int rank2, int* minPathNum) {
  int g1, g2;
  NCCLCHECK(ncclTopoRankToIndex(system, rank1, &g1));
  NCCLCHECK(ncclTopoRankToIndex(system, rank2, &g2));
  struct ncclTopoNode* gpu1 = system->nodes[GPU].nodes+g1;
  if (pcclParamGetMinPathNumByDriver() == 1) {
    int hggcDev1, hggcDev2;
    struct ncclTopoNode* gpu2 = system->nodes[GPU].nodes+g2;
    ncclTopoGetHggcDevByHgmlDev(gpu1->gpu.dev, &hggcDev1);
    ncclTopoGetHggcDevByHgmlDev(gpu2->gpu.dev, &hggcDev2);
    HGGCCHECK(hggcEitInterface::getInstance().wrapHggcDeviceGetP2PAttributeMinPath(minPathNum, hggcDev1, hggcDev2));
  } else {
    *minPathNum = gpu1->paths[GPU][g2].minPathNum;
    if (gpu1->gpu.hggcCompCap == 89) {
      *minPathNum /= ppuDiesNum;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetConnType(struct ncclTopoSystem* system, int rank1, int rank2, ncclConn_t *connType) {
  int g1, g2;
  NCCLCHECK(ncclTopoRankToIndex(system, rank1, &g1));
  NCCLCHECK(ncclTopoRankToIndex(system, rank2, &g2));
  struct ncclTopoNode* gpu1 = system->nodes[GPU].nodes+g1;
  int pathType = gpu1->paths[GPU][g2].type;
  if (pathType <= PATH_NVB) {
    *connType = CONN_ICN;
  } else if (pathType <= PATH_PHB) {
    *connType = CONN_PCIE;
  } else {
    *connType = CONN_NORMAL;
  }
  return ncclSuccess;
}

NCCL_PARAM(IgnoreDisabledP2p, "IGNORE_DISABLED_P2P", 0);

static int ncclTopoUserP2pLevel = -1; // Initially "uninitialized".  When initialized but unset, changes to -2.

// Gets the user-provided value of NCCL_P2P_LEVEL/NCCL_P2P_DISABLE.  If the user did not provide any, the value
// of the "level" argument is left unchanged.
ncclResult_t ncclGetUserP2pLevel(int* level) {
  if (ncclTopoUserP2pLevel == -1)
    NCCLCHECK(ncclGetLevel(&ncclTopoUserP2pLevel, "PCCL_P2P_DISABLE", "PCCL_P2P_LEVEL"));
  if (ncclTopoUserP2pLevel != -2)
    *level = ncclTopoUserP2pLevel;
  return ncclSuccess;
}

PCCL_PARAM(EnableIntermedRank, "ENABLE_INTERMED_RANK", 0);
ncclResult_t ncclTopoCheckP2p(ncclComm_t comm, struct ncclTopoSystem* system, int rank1, int rank2,
                              int* p2p, int *read, int* intermediateRank) {
  int mnnvl = 0;
  struct ncclPeerInfo* info1 = NULL;
  struct ncclPeerInfo* info2 = NULL;
  *p2p = 0;
  if (read) *read = 0;
  if (intermediateRank) *intermediateRank = -1;

  // Rule out different nodes / isolated containers
  if (comm && comm->peerInfo) {
    info1 = comm->peerInfo+rank1;
    info2 = comm->peerInfo+rank2;
    if (info1->hostHash != info2->hostHash) {
      if (comm->MNNVL) {
        NCCLCHECK(ncclTopoCheckMNNVL(comm->topo, info1, info2, &mnnvl));
        if (!mnnvl) return ncclSuccess;
      } else {
        return ncclSuccess;
      }
    } else if (info1->shmDev != info2->shmDev) {
      return ncclSuccess;
    }
  }

  // Get GPUs from topology
  int g1, g2;

  NCCLCHECK(ncclTopoRankToIndex(system, rank1, &g1));

  struct ncclTopoNode* gpu1 = system->nodes[GPU].nodes+g1;
  if (ncclTopoRankToIndex(system, rank2, &g2) == ncclInternalError) {
    // GPU not found, we can't use p2p.
    return ncclSuccess;
  }

  struct ncclTopoLinkList* path = gpu1->paths[GPU]+g2;

  int intermediateIndex = -1;
  if (pcclParamEnableIntermedRank()) {
    // Set intermediate GPU rank, if routing through an intermediate GPU.
    if (path->count == 2) {
      struct ncclTopoNode* intermediateNode = path->list[0]->remNode;
      if (intermediateNode->type == GPU && intermediateRank) {
        intermediateIndex = intermediateNode - system->nodes[GPU].nodes;
        *intermediateRank = intermediateNode->gpu.rank;
      }
    }
  }

  // In general, use P2P whenever we can.
  int p2pLevel = PATH_SYS;

  // Don't use P2P through ARM CPUs
  int arch, vendor, model;
  NCCLCHECK(ncclTopoCpuType(system, &arch, &vendor, &model));
  if (arch == NCCL_TOPO_CPU_ARCH_ARM) p2pLevel = PATH_PXB;
  if (arch == NCCL_TOPO_CPU_ARCH_X86 && vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
    if (model == NCCL_TOPO_CPU_MODEL_INTEL_BDW) p2pLevel = PATH_PXB;
    else p2pLevel = PATH_PHB;
  }

  // User override
  NCCLCHECK(ncclGetUserP2pLevel(&p2pLevel));

  // Compute the PCI distance and compare with the p2pLevel.
  if (path->type <= p2pLevel) *p2p = 1;

  if (*p2p == 1 && pcclHgmlEnsureInitialized() == ncclSuccess) {
    // NCCL_IGNORE_DISABLED_P2P=2 is used by unit tests that don't want to
    // validate against NVML at all since they are pretending to be on other hw.
    if (g1 != g2 && (comm == NULL || (comm->peerInfo && info1->hostHash == comm->peerInfo[comm->rank].hostHash &&
                                      info1->hostHash == info2->hostHash)) && ncclParamIgnoreDisabledP2p() != 2) {
      int indexes[3] = {-1,-1,-1};
      int verticeN = 0;

      indexes[verticeN++] = system->nodes[GPU].nodes[g1].gpu.dev;
      if (intermediateIndex != -1) indexes[verticeN++] = system->nodes[GPU].nodes[intermediateIndex].gpu.dev;
      indexes[verticeN++] = system->nodes[GPU].nodes[g2].gpu.dev;

      for (int i=1; i < verticeN; i++) {
        hgmlGpuP2PStatus_t status;
        status = pcclHgmlDevicePairs[indexes[i-1]][indexes[i-0]].p2pStatusRead;
        bool good = status == HGML_P2P_STATUS_OK;
        status = pcclHgmlDevicePairs[indexes[i-1]][indexes[i-0]].p2pStatusWrite;
        good &= status == HGML_P2P_STATUS_OK;
        if (!good) {
          if (!ncclParamIgnoreDisabledP2p()) {
            if (path->type <= PATH_NVB) {
              WARN("P2P is disabled between NVLINK connected GPUs %d and %d. This should not be the case given their connectivity, and is probably due to a hardware issue. If you still want to proceed, you can set PCCL_IGNORE_DISABLED_P2P=1.", indexes[i-1], indexes[i-0]);
              return ncclUnhandledHggcError;
            } else if (path->type < PATH_SYS) {
              INFO(NCCL_INIT, "P2P is disabled between connected GPUs %d and %d. You can repress this message with PCCL_IGNORE_DISABLED_P2P=1.", indexes[i-1], indexes[i-0]);
            }
          }
          *p2p = 0;
        }
      }
    }
  }

  return ncclSuccess;
}

PCCL_PARAM(NetDisableIntra, "NET_DISABLE_INTRA", 0);

// Check whether going through the network would be faster than going through P2P/SHM.
ncclResult_t ncclTopoCheckNet(struct ncclTopoSystem* system, int rank1, int rank2, int* net) {
  if (pcclParamNetDisableIntra() == 1) {
    *net = 0;
    return ncclSuccess;
  }
  *net = 1;
  // First check the current GPU-to-GPU speed.
  int g1, g2;
  if (ncclTopoRankToIndex(system, rank1, &g1) != ncclSuccess ||
      ncclTopoRankToIndex(system, rank2, &g2) != ncclSuccess) {
    return ncclSuccess;
  }

  struct ncclTopoNode* gpu1 = system->nodes[GPU].nodes + g1;
  struct ncclTopoNode* gpu2 = system->nodes[GPU].nodes + g2;
  float speed = gpu1->paths[GPU][g2].width;

  float netSpeed1 = 0, netSpeed2 = 0;
  for (int n = 0; n < system->nodes[NET].count; n++) {
    struct ncclTopoLinkList* path = gpu1->paths[NET] + n;
    if (pcclGetPathTypeAlias(path->type) <= PATH_PXB && path->width > netSpeed1) netSpeed1 = path->width;
    path = gpu2->paths[NET] + n;
    if (pcclGetPathTypeAlias(path->type) <= PATH_PXB && path->width > netSpeed2) netSpeed2 = path->width;
  }
  if (netSpeed1 > speed && netSpeed2 > speed) return ncclSuccess;
  *net = 0;
  return ncclSuccess;
}

ncclResult_t pcclTopoCheckGlobalIcnP2p(ncclComm_t comm, struct ncclTopoSystem* system, bool* globalIcnP2pSupport) {
  *globalIcnP2pSupport = (system->icnFullyConnected == 1 && comm->globalP2pSupport);
  return ncclSuccess;
}

// MNNVL: Check whether peers are in the same fabric cluster and clique
ncclResult_t ncclTopoCheckMNNVL(struct ncclTopoSystem* system, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2, int* ret) {
  *ret = 0;

  hgmlGpuFabricInfoV_t *fabricInfo1 = &info1->fabricInfo;
  hgmlGpuFabricInfoV_t *fabricInfo2 = &info2->fabricInfo;
  // A zero UUID means we don't have MNNVL fabric info
  if ((((long *)&fabricInfo2->clusterUuid)[0]|((long *)fabricInfo2->clusterUuid)[1]) == 0) return ncclSuccess;
  if ((memcmp(fabricInfo1->clusterUuid, fabricInfo2->clusterUuid, HGML_GPU_FABRIC_UUID_LEN) == 0) &&
      (fabricInfo1->cliqueId == fabricInfo2->cliqueId)) {
    TRACE(NCCL_NET, "MNNVL matching peer 0x%lx UUID %lx.%lx cliqueId 0x%x",
         info2->busId, ((long *)fabricInfo2->clusterUuid)[0], ((long *)fabricInfo2->clusterUuid)[1], fabricInfo2->cliqueId);
    *ret = 1;
  }
  return ncclSuccess;
}

NCCL_PARAM(NetGdrRead, "NET_GDR_READ", -2);
int ncclTopoUserGdrLevel = -1;

ncclResult_t ncclTopoCheckGdr(struct ncclTopoSystem* system, int rank, int64_t netId, int read, int* useGdr, bool send) {
  *useGdr = 0;

  // Get GPU and NET
  int n, g;
  NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &n));
  struct ncclTopoNode* net = system->nodes[NET].nodes+n;
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &g));
  struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;

  // Check that both the NIC and GPUs support it
  if (net->net.gdrSupport == 0) return ncclSuccess;
  if (gpu->gpu.gdrSupport == 0) return ncclSuccess;

  if (read) { // For reads (sends) only enable under certain conditions
    int gdrReadParam = ncclParamNetGdrRead();
    if (gdrReadParam == 0) return ncclSuccess;
    if (gdrReadParam < 0) {
      int nvlink = 0;
      // Since we don't know whether there are other communicators,
      // it's better to keep things local if we have a single GPU.
      if (system->nodes[GPU].count == 1) nvlink = 1;
      for (int i=0; i<system->nodes[GPU].count; i++) {
        if (i == g) continue;
        if (gpu->paths[GPU][i].type == PATH_NVL) {
          nvlink = 1;
          break;
        }
      }
      if (!nvlink) return ncclSuccess;
    }
  }

  // Check if we are close enough that it makes sense to enable GDR
  int netGdrLevel = PATH_PXB;
  NCCLCHECK(ncclGetLevel(&ncclTopoUserGdrLevel, NULL, "PCCL_NET_GDR_LEVEL"));
  if (ncclTopoUserGdrLevel != -2) netGdrLevel = ncclTopoUserGdrLevel;
  // PXN/PXR relays are send-only; receives use the NIC->PPU path.
  // Path computation keeps send=true with read=0 to bypass NET_GDR_READ policy.
  int distance = send ? gpu->paths[NET][n].type : net->paths[GPU][g].type;
  if (send && (distance == PATH_PXN || distance == PATH_PXR)) {
    int proxyRank, g;
    NCCLCHECK(ncclTopoGetIntermediateRank(system, gpu->gpu.rank, netId, &proxyRank));
    NCCLCHECK(ncclTopoRankToIndex(system, proxyRank, &g));
    struct ncclTopoNode* proxyGpu = system->nodes[GPU].nodes+g;
    distance = proxyGpu->paths[NET][n].type;
  }
  if (pcclGetPathTypeAlias(distance) > netGdrLevel) {
    INFO(NCCL_NET,"GPU Direct RDMA Disabled for Rank %d / HCA %lx (distance %d > %d)", rank, netId, distance, netGdrLevel);
    return ncclSuccess;
  }

  *useGdr = 1;
  INFO(NCCL_NET,"GPU Direct RDMA Enabled for Rank %d / HCA %lx (distance %d <= %d), read %d", rank, netId, distance, netGdrLevel, read);
  return ncclSuccess;
}

ncclResult_t ncclTopoGetIntermediateRank(struct ncclTopoSystem* system, int rank, int64_t netId, int* intermediateRank) {
  // Get GPU and NET
  int n, g;
  NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &n));
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &g));
  struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
  struct ncclTopoLinkList* path = gpu->paths[NET]+n;
  if (path->type == PATH_PXN || path->type == PATH_PXR) {
    struct ncclTopoNode* node = NULL;
    int type = -1;
    if (system->nodes[NVS].count == 0) {
      type = GPU;
      for (int i=0; i<path->count; i++) {
        struct ncclTopoNode* tmp = path->list[i]->remNode;
        if (tmp->type != GPU)
          break;
        node = tmp;
        type = node->type;
        VERBOSE(NCCL_NET, "path count %d: type %s, id %lx", i, topoNodeTypeStr[type], node->id);
      }
    } else {
      type = NVS;
      for (int i=0; i<path->count && type == NVS; i++) {
        node = path->list[i]->remNode;
        type = node->type;
        VERBOSE(NCCL_NET, "path count %d: type %s, id %d", i, topoNodeTypeStr[type], node->id);
      }
    }
    if (type != GPU || node == NULL) {
      WARN("Could not find intermediate GPU between GPU rank %d and NIC %lx\n", rank, netId);
      return ncclInternalError;
    }
    *intermediateRank = node->gpu.rank;
  } else {
    *intermediateRank = rank;
  }
  return ncclSuccess;
}

NCCL_PARAM(PxnDisable, "PXN_DISABLE", 0);

// Net v4 plugins don't have non-blocking connect/accept. We can't therefore use
// remote proxies without risking deadlocks
int ncclPxnDisable() {
  static int pxnDisable = -1;
  if (pxnDisable == -1) {
    if (ncclNetVersion() == 4) {
      INFO(NCCL_INIT, "PXN Disabled as plugin is v4");
      pxnDisable = 1;
    } else {
      pxnDisable = ncclParamPxnDisable();
    }
  }
  return pxnDisable;
}

ncclResult_t ncclTopoGetPxnRanks(ncclComm_t comm, int** intermediateRanks, int* nranks) {
  struct ncclTopoSystem* system = comm->topo;
  *nranks = 0;
  *intermediateRanks = NULL;
  if (system->icnFullyConnected) return ncclSuccess;

  int nr = 0;
  int* ranks = NULL;
  for (int rank=0; rank<comm->nRanks; rank++) {
    int proxyRank;
    int64_t netId;
    NCCLCHECK(ncclTopoGetNetDev(comm, comm->rank, NULL, 0, rank, &netId, NULL, &proxyRank));
    if (proxyRank == comm->rank) continue;
    int useGdr;
    NCCLCHECK(ncclTopoCheckGdr(comm->topo, comm->rank, netId, 1, &useGdr));
    if (useGdr == 0) continue;
    int found = 0;
    for (int r=0; r<nr; r++) {
      if (ranks[r] == proxyRank) found = 1;
    }
    if (!found) {
      NCCLCHECK(ncclRealloc(&ranks, nr, nr+1));
      ranks[nr++] = proxyRank;
    }
  }
  *nranks = nr;
  *intermediateRanks = ranks;
  return ncclSuccess;
}

ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system, ncclComm_t comm) {
  // Precompute paths between GPUs/NICs.

  // Remove everything in case we're re-computing
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) ncclTopoRemovePathType(system, t);

  // Set direct paths from/to CPUs. We need them in many cases.
  for (int c=0; c<system->nodes[CPU].count; c++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[CPU].nodes+c, system));
  }

  // Set direct paths to GPUs.
  for (int g=0; g<system->nodes[GPU].count; g++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[GPU].nodes+g, system));
  }

  // Set direct paths to NICs.
  for (int n=0; n<system->nodes[NET].count; n++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[NET].nodes+n, system));
  }

  struct ncclPeerInfo* peerInfos = comm ? comm->peerInfo : NULL;
  for (int g=0; g<system->nodes[GPU].count; g++) {
    // Update path when we don't want to / can't use GPU Direct P2P
    for (int p=0; p<system->nodes[GPU].count; p++) {
      int p2p;
      NCCLCHECK(ncclTopoCheckP2p(comm, system, system->nodes[GPU].nodes[p].gpu.rank,
                 system->nodes[GPU].nodes[g].gpu.rank, &p2p, NULL, NULL));
      if (p2p == 0) {
        // Divert all traffic through the CPU
        int cpu;
        NCCLCHECK(getLocalCpu(system, g, &cpu));
        NCCLCHECK(addInterStep(system, CPU, cpu, GPU, p, GPU, g));
      }
    }

    if (peerInfos == NULL) continue;
    // Remove GPUs we can't (or don't want to) communicate with through P2P or SHM
    struct ncclPeerInfo* dstInfo = peerInfos+system->nodes[GPU].nodes[g].gpu.rank;
    for (int p=0; p<system->nodes[GPU].count; p++) {
      if (p == g) continue;
      struct ncclPeerInfo* srcInfo = peerInfos+system->nodes[GPU].nodes[p].gpu.rank;
      int p2p = 0;
      NCCLCHECK(ncclTransports[TRANSPORT_P2P]->canConnect(&p2p, comm, system, NULL, srcInfo, dstInfo));
      if (p2p == 0) {
        int shm;
        NCCLCHECK(ncclTransports[TRANSPORT_SHM]->canConnect(&shm, comm, system, NULL, srcInfo, dstInfo));
        if (shm == 0) {
          // Mark this peer as inaccessible. We'll trim it later.
          system->nodes[GPU].nodes[p].paths[GPU][g].count = 0;
        }
      }
    }
  }

  // update the GPU -> NIC path in the case of C2C + PHB
  for (int n=0; n<system->nodes[NET].count; n++) {
    struct ncclTopoNode* netNode = system->nodes[NET].nodes+n;
    for (int g=0; g<system->nodes[GPU].count; g++) {
      // Check whether we can access the NIC through another NVLink-connected GPU (PXN)
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
      if (ncclPxnDisable() != 1 && gpu->paths[NET][n].type > PATH_PXB) {
        int pxnGpu = -1;
        int nHops = INT_MAX;
        for (int p=0; p<system->nodes[GPU].count; p++) {

          // PXN = PCI + NVLink.
          struct ncclTopoNode* peerNode = system->nodes[GPU].nodes+p;
          if (p == g || NCCL_TOPO_ID_SYSTEM_ID(peerNode->id) != NCCL_TOPO_ID_SYSTEM_ID(gpu->id)
              || netNode->paths[GPU][p].type > PATH_PXB || peerNode->paths[GPU][g].type > PATH_NVB) continue;

          // To ensure proper balancing, use only a local GPU which advertised that NIC as its preferred one.
          int64_t netId;
          NCCLCHECK(ncclTopoGetLocalNet(system, peerNode->gpu.rank, &netId, NULL));
          // Make sure we can allocate memory on that GPU.
          if (netId != netNode->id) continue;

          // found best PXR type with min hops or just PXN
          if (peerNode->paths[GPU][g].count < nHops) {
            pxnGpu = p;
            nHops = peerNode->paths[GPU][g].count;
            if (nHops == 1) break; // PXN break
          }
        }

        if (pxnGpu != -1) {
          // We can use that GPU as relay to communicate with that NIC.
          // Only enabling it in the GPU->NIC direction for now to favor
          // receiving locally and sending remotely (consistent with net.cc)
          NCCLCHECK(addInterStep(system, GPU, pxnGpu, GPU, g, NET, n));
        }
      }
      // Update path when we dont want to / can't use GPU Direct RDMA.
      int gdr;
      NCCLCHECK(ncclTopoCheckGdr(system, system->nodes[GPU].nodes[g].gpu.rank, netNode->id, 0, &gdr));
      if (gdr == 0) {
        // We cannot use GPU Direct RDMA, divert all traffic through the CPU local to the GPU
        int localCpu;
        NCCLCHECK(getLocalCpu(system, g, &localCpu));
        NCCLCHECK(addInterStep(system, CPU, localCpu, NET, n, GPU, g));
        NCCLCHECK(addInterStep(system, CPU, localCpu, GPU, g, NET, n));
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoTrimSystem(struct ncclTopoSystem* system, ncclComm_t comm) {
  ncclResult_t res = ncclSuccess;
  int myDomain = 0;
  const int ngpus = system->nodes[GPU].count;
  int *domains = NULL;
  int64_t *ids = NULL;
  int pathLevel = PATH_LOC;
  NCCLCHECKGOTO(ncclCalloc(&domains, system->nodes[GPU].count), res, end);
  NCCLCHECKGOTO(ncclCalloc(&ids, system->nodes[GPU].count), res, end);
  for (int g=0; g<system->nodes[GPU].count; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
    domains[g] = g;
    ids[g] = gpu->id;
    for (int p=0; p<g; p++) {
      struct ncclTopoLinkList* path = gpu->paths[GPU] + p;
      if (path->count > 0) {
        domains[g] = std::min(domains[g], domains[p]);
        pathLevel = std::max(pathLevel, path->type);
      }
    }
    if (gpu->gpu.rank == comm->rank) myDomain = domains[g];
  }

  system->icnFullyConnected = pathLevel <= PATH_NVB ? 1 : 0;
  for (int i=0; i<ngpus; i++) {
    system->icnFullyConnected = (system->icnFullyConnected == 1 && domains[i] == myDomain) ? 1 : 0;
    if (domains[i] == myDomain) continue;
    struct ncclTopoNode* gpu = NULL;
    int g;
    for (g=0; g<system->nodes[GPU].count /* This one varies over the loops */; g++) {
      gpu = system->nodes[GPU].nodes+g;
      if (gpu->id == ids[i]) break; else gpu=NULL;
    }
    if (gpu == NULL) {
      WARN("Could not find id %lx", ids[i]);
      NCCLCHECK(ncclFree(domains));
      NCCLCHECK(ncclFree(ids));
      return ncclInternalError;
    }
    NCCLCHECKGOTO(ncclTopoRemoveNode(system, GPU, g), res, end);
  }

  // A local ICN domain is globally fully connected only if it covers the whole communicator.
  system->icnFullyConnected &= (system->nodes[GPU].count == comm->nRanks);
  comm->localRanks = system->nodes[GPU].count;

end:
  NCCLCHECK(ncclFree(domains));
  NCCLCHECK(ncclFree(ids));
  return res;
}

void ncclTopoFree(struct ncclTopoSystem* system) {
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) ncclTopoRemovePathType(system, t);
  ncclFree(system);
}

PCCL_PARAM(P2pNChannelsFactorNPaths, "P2P_NCHANNELS_FACTOR_NPATHS", 0);

static ncclResult_t ncclTopoGetNchannels(struct ncclTopoSystem* system, int g /*local gpu index*/, int peerRank, int* nChannels) {
  int peer;
  struct ncclTopoLinkList* path = NULL;
  if (ncclTopoRankToIndex(system, peerRank, &peer) == ncclSuccess) {
    // Same rank
    if (g == peer) {
      *nChannels = -1;
      return ncclSuccess;
    }
    // Local rank
    path = system->nodes[GPU].nodes[peer].paths[GPU]+g;
    int* npaths = system->nodes[GPU].nodes[peer].npaths[GPU]+g;
    if (path->type == PATH_NVL || path->type == PATH_NVB) {
      float linkWidth = ncclTopoNVLinkSpeed(system->nodes[GPU].nodes[g].gpu.hggcCompCap);
      *nChannels = 2*std::max(1, (int)(path->width / linkWidth));
    } else {
      *nChannels = 2;
    }
    if (pcclParamP2pNChannelsFactorNPaths() == 1) {
      *nChannels *= (*npaths);
    }
  } else {
    // Remote rank, use network
    *nChannels = 1;
  }
  return ncclSuccess;
}

NCCL_PARAM(MinP2pNChannels, "MIN_P2P_NCHANNELS", 1);
NCCL_PARAM(MaxP2pNChannels, "MAX_P2P_NCHANNELS", MAXCHANNELS);
PCCL_PARAM(P2pNChannelsPerPeer, "P2P_NCHANNELS_PER_PEER", -2);
extern int64_t ncclParamWorkArgsBytes();

ncclResult_t ncclTopoComputeP2pChannelsPerPeer(ncclComm_t comm) {
  int g = 0;
  NCCLCHECK(ncclTopoRankToIndex(comm->topo, comm->rank, &g));
  int minChannels = MAXCHANNELS;
  for (int r = 0; r < comm->nRanks; r++) {
    int nChannels;
    NCCLCHECK(ncclTopoGetNchannels(comm->topo, g, r, &nChannels));
    if (nChannels >= 0) minChannels = std::min(minChannels, nChannels);
  }
  comm->p2pnChannelsPerPeer = minChannels;
  return ncclSuccess;
}

ncclResult_t ncclTopoComputeP2pChannels(ncclComm_t comm) {
  if (comm->sharedRes->owner != comm) {
    comm->p2pnChannels = std::min(comm->nChannels, (int)ncclParamMaxP2pNChannels());
    comm->p2pnChannels = std::min(std::max(comm->p2pnChannels, (int)ncclParamMinP2pNChannels()), comm->sharedRes->tpP2pNChannels);
  } else {
    comm->p2pnChannels = std::min(comm->nChannels, (int)ncclParamMaxP2pNChannels());
    comm->p2pnChannels = std::max(comm->p2pnChannels, (int)ncclParamMinP2pNChannels());
  }

  comm->p2pnChannels = nextPow2(comm->p2pnChannels);
  comm->p2pnChannelsPerPeer = nextPow2(comm->p2pnChannelsPerPeer);
  comm->p2pnChannelsPerPeer = std::max(prePow2(comm->p2pnChannels / comm->nRanks), comm->p2pnChannelsPerPeer);
  if (pcclEnableGlobalIcnP2pThroughSwitchOpt(comm)) {
    // default to use max channels(24) for 2/4/8 ranks, 16(power of 2) channels for greater nRanks
    comm->p2pnChannels = (MAXCHANNELS_PPU0015 % comm->nRanks == 0) ? MAXCHANNELS_PPU0015 : prePow2(MAXCHANNELS_PPU0015);
    comm->p2pnChannels = std::min(comm->p2pnChannels, (int)ncclParamMaxP2pNChannels());
    comm->p2pnChannels = std::max(comm->p2pnChannels, (int)ncclParamMinP2pNChannels());
    comm->p2pnChannelsPerPeer = comm->p2pnChannels;
  }
  comm->p2pnChannels = std::min(comm->p2pnChannels, pow2Down(ncclDevMaxChannelsForArgsBytes(ncclParamWorkArgsBytes())));
  if (pcclParamP2pNChannelsPerPeer() > 0) {
    comm->p2pnChannelsPerPeer = pcclParamP2pNChannelsPerPeer();
  }

  // Init channels that weren't used so far
  for (int c=comm->nChannels; c<comm->p2pnChannels; c++) NCCLCHECK(initChannel(comm, c));

  if ((comm->p2pnChannels & (comm->p2pnChannels-1)) == 0) {
    // We want to spread channels used when there aren't many and progressively
    // fill the whole space of nChannels. To do so we mirror the bits in the
    // nChannels space.
    for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
      int mirror = 0;
      for (int b=1, mb=(comm->p2pnChannels>>1); b<comm->p2pnChannels; b<<=1, mb>>=1) if (c & b) mirror |= mb;
      comm->p2pChannels[c] = mirror;
    }
  } else {
    // p2pnChannels may not be power of 2. we spread channels at 'nranks' intervals.
    int c = 0, start = 0;
    while (c < comm->p2pnChannelsPerPeer) {
      for (int step = start; step < comm->p2pnChannels && c < comm->p2pnChannelsPerPeer; step += comm->nRanks) {
        comm->p2pChannels[c++] = step;
      }
      start++;
    }
  }

  INFO(NCCL_INIT, "%d coll channels, %d fc channels, %d p2p channels, %d p2p channels per peer", comm->nChannels, comm->fcnChannels, comm->p2pnChannels, comm->p2pnChannelsPerPeer);
  return ncclSuccess;
}
