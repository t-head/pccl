/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "core.h"
#include "graph.h"
#include "topo.h"
#include "transport.h"
#include "xml.h"
#include <math.h>

NCCL_PARAM(CrossNic, "CROSS_NIC", 2);

ncclResult_t ncclTopoCheckCrossNicSupport(bool* supported) {
  *supported = (ncclParamCrossNic() != 0);
  return ncclSuccess;
}

const char* ncclTopoPatternStr[] = {"UNKNOWN", "BALANCED_TREE", "SPLIT_TREE", "TREE", "RING", "FC"};
PCCL_PARAM(BlocksPerFcLink, "BLOCKS_PER_FC_LINK", 2);
// Initialize system->maxWidth. This is the per-channel (i.e. per-SM)
// max speed.
static float getMaxWidth(struct ncclTopoSystem* system, struct ncclTopoNode* gpu, int type) {
  float maxWidth = 0.0;
  for (int i=0; i<system->nodes[type].count; i++) {
    struct ncclTopoLinkList* path = gpu->paths[type]+i;
    float width = path->width;
    if (path->count == 0) continue;
    maxWidth = std::max(maxWidth, width);
  }
  return maxWidth;
}
static float getTotalWidth(struct ncclTopoSystem* system, struct ncclTopoNode* gpu) {
  float nvlinkWidth = 0.0, pciWidth = 0.0;
  for (int l=0; l<gpu->nlinks; l++) {
    struct ncclTopoLink* link = gpu->links+l;
    if (link->type == LINK_NVL) nvlinkWidth += link->width;
    if (link->type == LINK_PCI) pciWidth = link->width;
  }
  return std::max(pciWidth, nvlinkWidth);
}

ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system) {
  system->maxWidth = 0.0;
  system->totalWidth = 0.0;
  system->totalNetWidth = LOC_WIDTH; // for intra-node case
  int inter = !system->icnFullyConnected;
  if (inter == 0 && system->nodes[GPU].count == 1) {
    system->maxWidth = LOC_WIDTH;
    return ncclSuccess;
  }
  for (int g=0; g<system->nodes[GPU].count; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
    system->maxWidth = std::max(system->maxWidth, getMaxWidth(system, gpu, inter ? NET : GPU));
    system->totalWidth = std::max(system->totalWidth, getTotalWidth(system, gpu));
  }

  if (inter > 0) {
    // inter-node case
    system->totalNetWidth = .0f;

    // accumulate net width
    for (int n=0; n<system->nodes[NET].count; n++) {
      struct ncclTopoNode* net = system->nodes[NET].nodes+n;
      float netWidth = .0f; // per net card
      for (int g=0; g<system->nodes[GPU].count; g++) {
        struct ncclTopoLinkList* path = net->paths[GPU]+g;
        if (path->type == PATH_PIX || path->type == PATH_PXB) {
          if ((netWidth+path->width) > net->net.width) break;
          netWidth += path->width;
        }
      }
      system->totalNetWidth += netWidth;
    }

    // in case of only machine head cards
    if (system->totalNetWidth < 0.1f) system->totalNetWidth = LOC_WIDTH;
  }
  return ncclSuccess;
}

static ncclResult_t findRevLink(struct ncclTopoNode* node1, struct ncclTopoNode* node2, struct ncclTopoLink** revLink) {
  for (int l=0; l<node2->nlinks; l++) {
    struct ncclTopoLink* link = node2->links+l;
    if (link->remNode == node1) {
      *revLink = link;
      return ncclSuccess;
    }
  }
  ERROR("Could not find rev link for %d/%lX -> %d/%lX", node1->type, node1->id, node2->type, node2->id);
  return ncclInternalError;
}

// This is unfortunately needed since manipulating floats often results in rounding errors.
#define SUB_ROUND(a, b) (a = roundf((a-b)*1000)/1000)

PCCL_PARAM(IgnoreIntelP2pOverHead, "IGNORE_INTEL_P2P_OVERHEAD", 1);
static ncclResult_t followPath(struct ncclTopoLinkList* path, struct ncclTopoNode* start, int maxSteps, float speed, int* steps, struct ncclTopoNode** proxyNode=NULL) {
  float pciSpeed = speed;
  static int ignoreIntelP2pOverHead = pcclParamIgnoreIntelP2pOverHead();
  if (ignoreIntelP2pOverHead != 1) {
    for (int step=0; step<path->count; step++) {
      struct ncclTopoNode* node = path->list[step]->remNode;
      if (node->type == CPU) {
        // Account for P2P inefficiency through Intel CPU RC
        if (path->type == PATH_PHB && start->type == GPU &&
            node->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 &&
            node->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
          pciSpeed = INTEL_P2P_OVERHEAD(speed);
        }
      }
    }
  }

  struct ncclTopoNode* node = start;
  for (int step=0; step<maxSteps; step++) {
    struct ncclTopoLink* link = path->list[step];
    struct ncclTopoLink* revLink = NULL;
    float fwSpeed = link->type == LINK_PCI ? pciSpeed : speed;
    float revSpeed = 0;
    if (link->remNode->type == GPU && link->remNode->gpu.hggcCompCap < 80 && start->type != GPU) {
      if (revLink == NULL) NCCLCHECK(findRevLink(node, link->remNode, &revLink));
      revSpeed += fwSpeed/8;
    }
    if (link->remNode->type == CPU && link->type == LINK_NVL) {
      if (revLink == NULL) NCCLCHECK(findRevLink(node, link->remNode, &revLink));
      revSpeed += fwSpeed;
    }
    if (link->width < fwSpeed || (revSpeed && revLink->width < revSpeed)) { *steps = step; return ncclSuccess; }
    SUB_ROUND(link->width, fwSpeed);
    if (revSpeed) SUB_ROUND(revLink->width, revSpeed);
    node = link->remNode;
    if (node->type == GPU && proxyNode) *proxyNode = node;
  }
  *steps = maxSteps;
  return ncclSuccess;
}

// Try to go from node type1/index1 to no type2/index2. mult indicates whether we are counting the bandwidth (1) or undoing (-1).
static ncclResult_t ncclTopoFollowPath(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int type1, int index1, int type2, int index2, int mult, struct ncclTopoNode** node, struct ncclTopoNode** proxyNode=NULL) {
  // First handle easy cases
  *node = system->nodes[type2].nodes+index2;
  if (type1 == -1) return ncclSuccess;
  struct ncclTopoNode* node1 = system->nodes[type1].nodes+index1;
  struct ncclTopoLinkList* path = node1->paths[type2]+index2;
  if (path->count == 0 ) return ncclSuccess;

  // Now check link type
  *node = NULL;
  int intra = type1 == GPU && type2 == GPU;
  float speed = intra ? graph->speedIntra : graph->speedInter;
  int type = intra ? graph->typeIntra : graph->typeInter;

  if (mult == 1 && (path->type > type)) return ncclSuccess;

  speed *= mult;

  // Check there is enough bandwidth on paths.
  int step = 0;
  NCCLCHECK(followPath(path, node1, path->count, speed, &step, proxyNode));
  if (step < path->count) goto rewind;

  // Enough bandwidth : return destination node.
  graph->nHops += mult*path->count;
  *node = system->nodes[type2].nodes+index2;
  return ncclSuccess;

rewind:
  // Not enough bandwidth : rewind and exit.
  NCCLCHECK(followPath(path, node1, step, -speed, &step));
  return ncclSuccess;
}

static int gpuPciWidth(struct ncclTopoNode* gpu) {
  for (int l=0; l<gpu->nlinks; l++) {
    struct ncclTopoLink* gpuLink = gpu->links+l;
    if (gpuLink->type != LINK_PCI) continue;
    struct ncclTopoNode* pci = gpuLink->remNode;
    for (int l=0; l<pci->nlinks; l++) {
      struct ncclTopoLink* pciLink = pci->links+l;
      if (pciLink->remNode != gpu) continue;
      return std::min(gpuLink->width, pciLink->width);
    }
  }
  return -1;
}

/* Choose the order in which we try next GPUs. This is critical for the search
   to quickly converge to the best solution even if it eventually times out. */
struct ncclGpuScore {
  int g;             // Retain the index
  int startIndex;    // Least important
  int intraNhops;
  int intraWidth;
  int interNhops;
  int interPciWidth;
  int interWidth;    // Most important
};

static int cmpScore(const void * g1, const void * g2) {
   struct ncclGpuScore *s1 = (struct ncclGpuScore*)g1;
   struct ncclGpuScore *s2 = (struct ncclGpuScore*)g2;
   int d;
   if ((d = (s2->interWidth - s1->interWidth))) return d;
   if ((d = (s2->interPciWidth - s1->interPciWidth))) return d;
   if ((d = (s1->interNhops - s2->interNhops))) return d;
   if ((d = (s2->intraWidth - s1->intraWidth))) return d;
   if ((d = (s1->intraNhops - s2->intraNhops))) return d;
   return s1->startIndex - s2->startIndex;
}

static int cmpIntraScores(struct ncclGpuScore* scores, int count) {
  int intraWidth = scores[0].intraWidth;
  int intraNhops = scores[0].intraNhops;
  for (int i=1; i<count; i++) {
    if (scores[i].intraWidth != intraWidth || scores[i].intraNhops != intraNhops) return 1;
  }
  return 0;
}

static ncclResult_t getGpuIndex(struct ncclTopoSystem* system, int rank, int* index) {
  for (int g=0; g<system->nodes[GPU].count; g++) {
    if (system->nodes[GPU].nodes[g].gpu.rank == rank) {
      *index = g;
      return ncclSuccess;
    }
  }
  ERROR("Could not find ppu rank %d", rank);
  return ncclInternalError;
}

static ncclResult_t getNetIndex(struct ncclTopoSystem* system, int64_t id, int* index) {
  for (int n=0; n<system->nodes[NET].count; n++) {
    if (system->nodes[NET].nodes[n].id == id) {
      *index = n;
      return ncclSuccess;
    }
  }
  ERROR("Could not find net id %lx", id);
  return ncclInternalError;
}

static ncclResult_t getNetPaths(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoLinkList** netPaths) {
  int64_t netId = graph->inter[graph->nChannels*2];
  int n;
  NCCLCHECK(getNetIndex(system, netId, &n));
  *netPaths=system->nodes[NET].nodes[n].paths[GPU];
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchNextGpuSort(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoNode* gpu, int* next, int* countPtr, int sortNet) {
  const uint64_t flag = 1ULL<<(graph->nChannels);
  int ngpus = system->nodes[GPU].count;
  struct ncclTopoLinkList* paths = gpu->paths[GPU];
  struct ncclTopoLinkList* netPaths = NULL;
  if (sortNet == 1) NCCLCHECK(getNetPaths(system, graph, &netPaths));

  struct ncclGpuScore scores[NCCL_TOPO_MAX_NODES];
  memset(scores, 0, ngpus*sizeof(struct ncclGpuScore));
  int start = gpu-system->nodes[GPU].nodes;
  int count = 0;
  for (int i=1; i<ngpus; i++) {
    int g = (start+i)%ngpus;
    if (paths[g].count == 0) continue; // There is no path to that GPU
    if (system->nodes[GPU].nodes[g].used & flag) continue;
    scores[count].g = g;
    scores[count].startIndex = i;
    scores[count].intraNhops = paths[g].count;
    scores[count].intraWidth = paths[g].width;
    if (netPaths) {
      scores[count].interNhops = netPaths[g].count;
      scores[count].interPciWidth = gpuPciWidth(system->nodes[GPU].nodes+g);
      scores[count].interWidth = netPaths[g].width;
    }
    count++;
  }

  // Sort GPUs
  qsort(scores, count, sizeof(struct ncclGpuScore), cmpScore);

  // Check if all have the same intra-node score in which case we go reverse for sortNet = -1
  if (sortNet == -1 && cmpIntraScores(scores, count) == 0) {
    for (int i=0; i<count; i++) next[i] = scores[count-1-i].g;
  } else {
    for (int i=0; i<count; i++) next[i] = scores[i].g;
  }
  *countPtr = count;
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRec(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int* time);

// Try to keep all searchs within one second
#define NCCL_SEARCH_GLOBAL_TIMEOUT (3ULL<<20)
#define NCCL_SEARCH_TIMEOUT (1<<18)
#define NCCL_SEARCH_TIMEOUT_TREE (1<<17)
#define NCCL_SEARCH_TIMEOUT_SAMECHANNELS (1<<10)

#define FORCED_ORDER_PCI 1
#define FORCED_ORDER_REPLAY 2

ncclResult_t ncclTopoReplayGetGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int step, int* g) {
  *g = -1;
  if (graph->nChannels == 0) return ncclInternalError;
  int ngpus = system->nodes[GPU].count;
  int nextRank = graph->intra[(graph->nChannels-1)*ngpus+step+1];
  for (int i=0; i<ngpus; i++) if (system->nodes[GPU].nodes[i].gpu.rank == nextRank) {
    *g = i;
    return ncclSuccess;
  }
  if (*g == -1) return ncclInternalError;
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRecGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, struct ncclTopoNode* gpu, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time);

ncclResult_t ncclTopoSearchTryGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time, int type, int index, int g) {
  const uint64_t flag = 1ULL<<(graph->nChannels);
  struct ncclTopoNode* gpu;
  NCCLCHECK(ncclTopoFollowPath(system, graph, type, index, GPU, g, 1, &gpu));
  if (gpu) {
    gpu->used ^= flag;
    NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, backToNet, backToFirstRank, forcedOrder, time));
    gpu->used ^= flag;
    NCCLCHECK(ncclTopoFollowPath(system, graph, type, index, GPU, g, -1, &gpu));
  }
  return ncclSuccess;
}

bool isGraphWithAllPxnChannels(ncclTopoGraph *graph) {
  bool ret = true;
  for (int c = 0; c < graph->nChannels; c++) {
    if (graph->proxyDevs[c] == -1) {
      ret = false;
      break;
    }
  }
  return ret;
}

ncclResult_t ncclTopoCompareGraphs(struct ncclTopoGraph* graph, struct ncclTopoGraph* refGraph, int* copy) {
  // 1. Constraint to get the same nChannels between Rings and Trees
  if (graph->nChannels < graph->minChannels) return ncclSuccess;

  // 2. Try to get better bandwidth
  if (graph->nChannels*graph->speedIntra < refGraph->nChannels*refGraph->speedIntra) return ncclSuccess;
  if (graph->nChannels*graph->speedIntra > refGraph->nChannels*refGraph->speedIntra) {
    *copy = 1;
    return ncclSuccess;
  }

  // 3. Less hops (but not at the price of going cross NICs)
  if (graph->pattern == refGraph->pattern &&
      graph->crossNic == refGraph->crossNic &&
      graph->typeInter != PATH_PXN &&
      refGraph->typeInter != PATH_PXN &&
      graph->nHops < refGraph->nHops) {
    *copy = 1;
    return ncclSuccess;
  }

  // 4. Pefect PXN
  if ((graph->nChannels == refGraph->nChannels) &&
      (graph->speedInter == refGraph->speedInter) &&
      (graph->typeInter == PATH_PXN && refGraph->typeInter == PATH_PXN) &&
      (isGraphWithAllPxnChannels(graph) && !isGraphWithAllPxnChannels(refGraph))) {
    *copy = 1;
    return ncclSuccess;
  }

  // 5. Better Path type
  if (graph->typeInter < refGraph->typeInter) {
    *copy = 1;
  } else if (graph->typeInter == refGraph->typeInter) {
    if (graph->typeIntra < refGraph->typeIntra) {
      *copy = 1;
    }
  }

  return ncclSuccess;
}

// Build a list of the best NETs to try.
//
// "gpu" can be set to -1 to build a list suitable for all GPUs (search start) or to a given gpu
//  index when trying to get back to the NIC.
//
// The list is built the following way:
// 1. Select NETs starting with those close to GPU(s), based on paths[n].type.
// 2. For each GPU, once that list of NICs with a given distance is prepared, shuffle the list
//    based on the GPU NVML index so that e.g. GPU 1 chooses NIC 1 first instead of NIC 0 which
//    might have been choosen by GPU 0 (case with multiple independent communicators per node)
// 3. Then add the NETs to the final list if they were not already added by another closer GPU.

ncclResult_t ncclTopoSelectNets(struct ncclTopoSystem* system, int typeInter, int gpu, int* nets, int* netCountRet) {
  int netCount = 0;
  int localNetCount;
  int* localNets;
  NCCLCHECK(ncclCalloc(&localNets, system->nodes[NET].count));

  for (int t=0; t <= typeInter; t++) {
    for (int g=0; g<system->nodes[GPU].count; g++) {
      if (gpu != -1 && gpu != g) continue;
      localNetCount = 0;
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
      struct ncclTopoLinkList* paths = gpu->paths[NET];
      for (int n=0; n<system->nodes[NET].count; n++) {
        if (paths[n].type == t) localNets[localNetCount++] = n;
      }
      if (localNetCount == 0) continue;
      // Shuffle by gpu NVML device number so that GPUs on the same PCI switch
      // with multiple NICs don't use the same one as first choice.
      for (int r=0; r<system->nodes[GPU].nodes[g].gpu.dev % localNetCount; r++) {
        int net0 = localNets[0];
        for (int i=0; i<localNetCount-1; i++) localNets[i] = localNets[i+1];
        localNets[localNetCount-1] = net0;
      }
      // Append NICs to list
      for (int i=0; i<localNetCount; i++) {
        int n = localNets[i];
        int found = 0;
        while (nets[found] != n && found<netCount) found++;
        if (found == netCount) nets[netCount++] = n;
      }
    }
  }

  *netCountRet = netCount;
  NCCLCHECK(ncclFree(localNets));

  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRecGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, struct ncclTopoNode* gpu, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time) {
  if ((*time) <= 0) return ncclSuccess;
  (*time)--;

  int ngpus = system->nodes[GPU].count;
  if (step == ngpus) {
    // record the replay info for possible channel shuffle
    if (forcedOrder != FORCED_ORDER_REPLAY) {
      graph->nSubChannels++;
    }
    // Determine whether we found a better solution or not
    int copy = 0;
    graph->nChannels++;
    NCCLCHECK(ncclTopoCompareGraphs(graph, saveGraph, &copy));
    if (copy) {
      memcpy(saveGraph, graph, sizeof(struct ncclTopoGraph));
      if (graph->nChannels == graph->maxChannels) *time = -1;
    }
#ifdef DEBUG_SEARCH
    TRACE(NCCL_GRAPH, "graph copying %d:", copy);
    ncclTopoPrintGraph(system, graph);
#endif
    if (graph->nChannels < graph->maxChannels) {
      NCCLCHECK(ncclTopoSearchRec(system, graph, saveGraph, time));
    }
    graph->nChannels--;
    if (forcedOrder != FORCED_ORDER_REPLAY) graph->nSubChannels--;
    return ncclSuccess;
  }
  graph->intra[graph->nChannels*ngpus+step] = gpu->gpu.rank;
  if (forcedOrder != FORCED_ORDER_REPLAY) graph->subIntra[graph->nSubChannels*ngpus+step] = gpu->gpu.rank;

  int g = gpu - system->nodes[GPU].nodes;
  if (step == backToNet) {
    // first get back to NIC
    if (!system->icnFullyConnected) {
      int startNetIndex;
      NCCLCHECK(getNetIndex(system, graph->inter[graph->nChannels*2], &startNetIndex));
      struct ncclTopoNode* startNet = system->nodes[NET].nodes+startNetIndex;
      int netcount;
      int* nets;
      NCCLCHECK(ncclCalloc(&nets, system->nodes[NET].count));
      NCCLCHECK(ncclTopoSelectNets(system, graph->typeInter, g, nets, &netcount));
      for (int i=0; i<netcount; i++) {
        int n = nets[i];
        struct ncclTopoNode* net = system->nodes[NET].nodes+n;
        struct ncclTopoNode* proxyGpu = NULL;
        if (graph->pattern == NCCL_TOPO_PATTERN_TREE && net->id != startNet->id) continue; // Trees are symmetric
        if (graph->crossNic != 1 && (net->net.asic != startNet->net.asic || net->net.port != startNet->net.port)) continue;

        // Balanced Tree : count half of the bandwidth on first two GPUs
        int nextBackToNet = -1;
        float speedInterSave = graph->speedInter;
        if (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {
          // Count half of the bandwidth on each of the first two GPUs
          if (step == 0) nextBackToNet = 1;
          else if (net->id != graph->inter[graph->nChannels*2+1]) continue;
          graph->speedInter /= 2;
        }

        NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, NET, n, 1, &net, &proxyGpu));
        graph->speedInter = speedInterSave;
        if (net) {
          graph->inter[graph->nChannels*2+1] = net->id;
          graph->proxyDevs[graph->nChannels] = -1;
          if (proxyGpu) graph->proxyDevs[graph->nChannels] = proxyGpu->gpu.dev;
          NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, nextBackToNet, backToFirstRank, forcedOrder, time));

          if (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) graph->speedInter /= 2;
          NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, NET, n, -1, &net));
          graph->speedInter = speedInterSave;
        }
      }
      NCCLCHECK(ncclFree(nets));
    }
  } else if (step < system->nodes[GPU].count-1) {
    // Go to next GPU
    int next[NCCL_TOPO_MAX_NODES];
    int count;
    if (forcedOrder == FORCED_ORDER_PCI) { // Try the PCI order
      next[0] = step+1;
      count = 1;
    } else if (forcedOrder == FORCED_ORDER_REPLAY) { // Try last channel order
      NCCLCHECK(ncclTopoReplayGetGpu(system, graph, step, next));
      count = 1;
    } else { // Normal search
      NCCLCHECK(ncclTopoSearchNextGpuSort(system, graph, gpu, next, &count, backToNet == -1 ? 0 : backToNet == step+1 ? 1 : -1 ));
    }
    for (int i=0; i<count; i++) {
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, step+1, backToNet, backToFirstRank, forcedOrder, time, GPU, g, next[i]));
    }
  } else if (step == backToFirstRank) {
    // Find first GPU and loop back to it
    int p;
    NCCLCHECK(getGpuIndex(system, graph->intra[graph->nChannels*ngpus], &p));
    struct ncclTopoNode* firstGpu;
    NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, p, 1, &firstGpu));
    if (firstGpu) {
      NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, firstGpu, step+1, backToNet, -1, forcedOrder, time));
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, p, -1, &firstGpu));
    }
  } else {
    // Next path
    NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, ngpus, -1, -1, forcedOrder, time));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRecNet(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int backToNet, int backToFirstRank, int* time) {
  const int speed = graph->speedInter;
  int* nets;
  NCCLCHECK(ncclCalloc(&nets, system->nodes[NET].count));
  int netcount;
  NCCLCHECK(ncclTopoSelectNets(system, graph->typeInter, -1, nets, &netcount));
  for (int i=0; i<netcount; i++) {
    int n = nets[i];
    struct ncclTopoNode* net = system->nodes[NET].nodes+n;
    struct ncclTopoNode* gpu;
    if (graph->collNet && net->net.collSupport == 0) continue;
    if (net->net.width < speed) continue;
    if (net->net.maxChannels == 0) continue;

    graph->inter[graph->nChannels*2] = net->id;
    for (int i=0; i<system->nodes[NET].count; i++) {
      if ((system->nodes[NET].nodes[i].net.asic == net->net.asic) &&
          (system->nodes[NET].nodes[i].net.port == net->net.port)) {
        system->nodes[NET].nodes[i].net.width -= speed;
      }
    }
    net->net.maxChannels--;

    // First try to replay the last channel
    if (graph->nChannels > 0) {
      int g;
      NCCLCHECK(ncclTopoReplayGetGpu(system, graph, -1, &g));
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time, NET, n, g));
    }
    if (graph->nChannels == 0 || graph->sameChannels == 0) {
      if (graph->nChannels == 0) {
        // Always try the PCI order first to set a reference, but don't count in the timeout nor let it run for long
        int t = 1 << 10;
        NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, &t, NET, n, 0));
        if (t == -1) *time = -1;
      }

      // Then try the most local GPUs
      float maxWidth = 0;
      int minHops = 0xfffffff;
      struct ncclTopoLinkList* paths = net->paths[GPU];
      for (int g=0; g<system->nodes[GPU].count; g++) {
        if (paths[g].width > maxWidth) {
          maxWidth = paths[g].width;
          minHops = paths[g].count;
        } else if (paths[g].width == maxWidth && paths[g].count < minHops) {
          minHops = paths[g].count;
        }
      }
      if (maxWidth >= speed) {
        // In the first loop, avoid using GPUs in both directions between channels (one channel
        // sending from that GPU and one channel receiving to that GPU), since that usually leads
        // to lower BW.
        for (int tryGpuBidir=0; tryGpuBidir<2; tryGpuBidir++) {
          for (int g=0; g<system->nodes[GPU].count; g++) {
            if (paths[g].width == maxWidth && paths[g].count == minHops) {
              gpu = system->nodes[GPU].nodes+g;
              int gpuUsed = gpuPciWidth(gpu) > 0 ? 0 : 1;
              if (tryGpuBidir == gpuUsed) {
                NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n, g));
              }
            }
          }
        }
      }
    }

    net->net.maxChannels++;
    for (int i=0; i<system->nodes[NET].count; i++) {
      if ((system->nodes[NET].nodes[i].net.asic == net->net.asic) &&
          (system->nodes[NET].nodes[i].net.port == net->net.port)) {
        system->nodes[NET].nodes[i].net.width += speed;
      }
    }
  }
  NCCLCHECK(ncclFree(nets));
  return ncclSuccess;
}

/* Search Patterns
 *
 *     Intra-node
 * Ring            : GPU a -> GPU b -> .. -> GPU x -> GPU a
 * (=Split Tree Loop)
 * Tree            : GPU a -> GPU b -> .. -> GPU x
 * (=Split Tree)
 *
 *     Inter-node
 * Ring            : NET n -> GPU a -> GPU b -> .. -> GPU x -> NET n (or m if crossNic)
 * Tree            : NET n -> GPU a -> GPU b -> .. -> GPU x
 *                              `--> NET n (or m if crossNic)
 * Split Tree      : NET n -> GPU a -> GPU b -> .. -> GPU x
 *                                       `--> NET n (or m if crossNic)
 * Split Tree Loop : NET n -> GPU a -> GPU b -> .. -> GPU x -> GPU a
 *                                       `--> NET n (or m if crossNic)
 */
ncclResult_t ncclTopoSearchParams(struct ncclTopoSystem* system, int pattern, int* backToNet, int* backToFirstRank) {
  if (!system->icnFullyConnected) {
    if (pattern == NCCL_TOPO_PATTERN_RING) *backToNet = system->nodes[GPU].count-1;
    else if (pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) *backToNet = 1;
    else *backToNet = 0;
    *backToFirstRank = -1;
  } else {
    *backToNet = -1;
    if (pattern == NCCL_TOPO_PATTERN_RING) *backToFirstRank = system->nodes[GPU].count-1;
    else *backToFirstRank = -1;
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRec(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int* time) {
  int backToNet, backToFirstRank;
  NCCLCHECK(ncclTopoSearchParams(system, graph->pattern, &backToNet, &backToFirstRank));
  if (!system->icnFullyConnected) {
    // Start from NET
    ncclTopoSearchRecNet(system, graph, saveGraph, backToNet, backToFirstRank, time);
  } else {
    // Intra-node only.
    if (graph->nChannels == 0) {
      // Try PCI order first
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, time, -1, -1, 0));
    } else {
      // Also try to replay previous channel
      int g;
      NCCLCHECK(ncclTopoReplayGetGpu(system, graph, -1, &g));
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time, -1, -1, g));
    }
    if (graph->sameChannels == 0 || graph->nChannels == 0) {
      // Finally, try all other possibilities unless we are forced to use the same channels
      // The graph search for ring always start from rank 0 to avoid duplicated searches
      int maxStartRank = graph->pattern == NCCL_TOPO_PATTERN_RING ? 0 : system->nodes[GPU].count-1;
      for (int g=0; g<=maxStartRank; g++) {
        NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, -1, -1, g));
      }
    }
  }
  return ncclSuccess;
}

/************************************/
/* User defined graph from XML file */
/************************************/

struct kvDict kvDictLinkType[] = {
  { "SYS", PATH_SYS },
  { "PHB", PATH_PHB },
  { "PHBH", PATH_PHBH },
  { "PXR", PATH_PXR },
  { "PXN", PATH_PXN },
  { "PXB", PATH_PXB },
  { "PIX", PATH_PIX },
  { "ICR", PATH_NVB },
  { "ICN", PATH_NVL },
  { "LOC", PATH_LOC },
  { NULL, 0 }
};

ncclResult_t ncclTopoGetChannelFromXml(struct ncclXmlNode *xmlChannel, int c, struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  int ngpus = system->nodes[GPU].count;
  int64_t* inter = graph->inter+2*c;
  int* intra = graph->intra+ngpus*c;
  int n=0, g=0;
  NCCLCHECK(xmlGetAttrInt(xmlChannel, "proxyDev", &graph->proxyDevs[c], true));
  for (int s=0; s<xmlChannel->nSubs; s++) {
    struct ncclXmlNode* sub = xmlChannel->subs[s];
    int64_t dev;
    const char* str;
    NCCLCHECK(xmlGetAttrStr(sub, "dev", &str));
    dev = strtol(str, NULL, 16);
    if (strcmp(sub->name, "net") == 0) {
      inter[n++] = dev;
    } else if (strcmp(sub->name, "gpu") == 0) {
      int rank = -1;
      for (int g=0; g<ngpus; g++) {
        int systemId = NCCL_TOPO_ID_SYSTEM_ID(system->nodes[GPU].nodes[g].id);
        if (NCCL_TOPO_ID(systemId, system->nodes[GPU].nodes[g].gpu.dev) == dev) rank = system->nodes[GPU].nodes[g].gpu.rank;
      }
      if (rank == -1) {
        ERROR("XML Import Channel : dev %ld not found.", dev);
        return ncclSystemError;
      }
      intra[g++] = rank;
    }
  }
  return ncclSuccess;
}
ncclResult_t ncclTopoGetGraphFromXmlSub(struct ncclXmlNode *xmlGraph, struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int* nChannels) {
  int id;
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "id", &id));
  if (graph->id != id) return ncclSuccess;

  int crossNic;
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "crossnic", &crossNic));
  if (graph->crossNic == 0 && crossNic == 1) return ncclSuccess;
  graph->crossNic = crossNic;

  NCCLCHECK(xmlGetAttrInt(xmlGraph, "pattern", &graph->pattern));
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "nchannels", &graph->nChannels));
  NCCLCHECK(xmlGetAttrFloat(xmlGraph, "speedintra", &graph->speedIntra));
  NCCLCHECK(xmlGetAttrFloat(xmlGraph, "speedinter", &graph->speedInter));
  const char* str;
  NCCLCHECK(xmlGetAttr(xmlGraph, "typeintra", &str));
  NCCLCHECK(kvConvertToInt(str, &graph->typeIntra, kvDictLinkType));
  NCCLCHECK(xmlGetAttr(xmlGraph, "typeinter", &str));
  NCCLCHECK(kvConvertToInt(str, &graph->typeInter, kvDictLinkType));
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "samechannels", &graph->sameChannels));
  for (int s=0; s<xmlGraph->nSubs; s++) {
    NCCLCHECK(ncclTopoGetChannelFromXml(xmlGraph->subs[s], s, system, graph));
  }
  *nChannels = xmlGraph->nSubs;
  return ncclSuccess;
}
ncclResult_t ncclTopoGetGraphFromXml(struct ncclXmlNode *xmlGraphs, struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int* nChannels) {
  for (int s=0; s<xmlGraphs->nSubs; s++) {
    NCCLCHECK(ncclTopoGetGraphFromXmlSub(xmlGraphs->subs[s], system, graph, nChannels));
  }
  return ncclSuccess;
}

/* And the reverse : graph->xml */
ncclResult_t ncclTopoGetXmlFromChannel(struct ncclTopoGraph* graph, int c, struct ncclTopoSystem* system, struct ncclXml *xml, struct ncclXmlNode* parent) {
  struct ncclXmlNode* xmlChannel;
  int ngpus = system->nodes[GPU].count;
  int64_t* inter = graph->inter+2*c;
  int* intra = graph->intra+ngpus*c;
  NCCLCHECK(xmlAddNode(xml, parent, "channel", &xmlChannel));
  NCCLCHECK(xmlSetAttrInt(xmlChannel, "proxyDev", graph->proxyDevs[c]));
  struct ncclXmlNode* node;
  if (!system->icnFullyConnected) {
    NCCLCHECK(xmlAddNode(xml, xmlChannel, "net", &node));
    NCCLCHECK(xmlSetAttrLong(node, "dev", inter[0]));
  }
  for (int g=0; g<ngpus; g++) {
    NCCLCHECK(xmlAddNode(xml, xmlChannel, "gpu", &node));
    int dev = -1;
    for (int i=0; i<ngpus; i++) {
      if (system->nodes[GPU].nodes[i].gpu.rank == intra[g]) dev = system->nodes[GPU].nodes[i].gpu.dev;
    }
    if (dev == -1) {
      ERROR("XML Export Channel : rank %d not found.", intra[g]);
      return ncclInternalError;
    }
    NCCLCHECK(xmlSetAttrInt(node, "dev", dev));
  }
  if (!system->icnFullyConnected) {
    NCCLCHECK(xmlAddNode(xml, xmlChannel, "net", &node));
    NCCLCHECK(xmlSetAttrLong(node, "dev", inter[1]));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlFromGraph(struct ncclTopoGraph* graph, struct ncclTopoSystem* system, struct ncclXml *xml, struct ncclXmlNode* parent) {
  struct ncclXmlNode* xmlGraph;
  NCCLCHECK(xmlAddNode(xml, parent, "graph", &xmlGraph));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "id", graph->id));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "pattern", graph->pattern));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "crossnic", graph->crossNic));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "nchannels", graph->nChannels));
  NCCLCHECK(xmlSetAttrFloat(xmlGraph, "speedintra", graph->speedIntra));
  NCCLCHECK(xmlSetAttrFloat(xmlGraph, "speedinter", graph->speedInter));
  const char* str;
  NCCLCHECK(kvConvertToStr(graph->typeIntra, &str, kvDictLinkType));
  NCCLCHECK(xmlSetAttr(xmlGraph, "typeintra", str));
  NCCLCHECK(kvConvertToStr(graph->typeInter, &str, kvDictLinkType));
  NCCLCHECK(xmlSetAttr(xmlGraph, "typeinter", str));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "samechannels", graph->sameChannels));
  for (int c=0; c<graph->nChannels; c++) {
    NCCLCHECK(ncclTopoGetXmlFromChannel(graph, c, system, xml, xmlGraph));
  }
  return ncclSuccess;
}
ncclResult_t ncclTopoGetXmlFromGraphs(int ngraphs, struct ncclTopoGraph** graphs, struct ncclTopoSystem* system, struct ncclXml *xml) {
  xml->maxIndex = 0;
  struct ncclXmlNode* xmlGraphs;
  NCCLCHECK(xmlAddNode(xml, NULL, "graphs", &xmlGraphs));
  NCCLCHECK(xmlSetAttrInt(xmlGraphs, "version", NCCL_GRAPH_XML_VERSION));
  for (int g=0; g<ngraphs; g++) {
    NCCLCHECK(ncclTopoGetXmlFromGraph(graphs[g], system, xml, xmlGraphs));
  }
  return ncclSuccess;
}

float speedArray[] = {84.0, 42.0, 30.0, 24.0, 21.0, 18.0, 15.0, 12.0, 10.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.4, 1.2, 0.24, 0.12 };
#define NSPEEDS (sizeof(speedArray)/sizeof(float))

PCCL_PARAM(DisableFcGraph, "DISABLE_FC_GRAPH", 0);
ncclResult_t ncclTopoSearchTryFC(ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  graph->nChannels = 0;
  int disableFcGraph = pcclParamDisableFcGraph();
  if (disableFcGraph > 0) {
    return ncclSuccess;
  }

  if (!system->icnFullyConnected) {
    return ncclSuccess;
  }

  // Altough this is ugly, but should refine later
  if (usingMemcpy()) {
    return ncclSuccess;
  }

  int ngpus = system->nodes[GPU].count;
  constexpr int FcRankNum = 4;
  if (ngpus != FcRankNum) {
    return ncclSuccess;
  }
  int nPeers = FcRankNum - 1;

  struct {
    int maxType;
    int minWidth;
    int maxHops;
  } *perRankData;
  NCCLCHECK(ncclCalloc(&perRankData, ngpus));

  int numLinks = 0;
  int blocksPerLink = pcclParamBlocksPerFcLink();
  ncclResult_t ret = ncclSuccess;

  // fill perRankData with path info between each rank pairs
  for (int srcDev = 0; srcDev < ngpus; ++srcDev) {
    struct ncclTopoNode* srcNode = system->nodes[GPU].nodes + srcDev;
    // initialize perRankData of srcDev
    perRankData[srcDev].maxType = PATH_LOC;
    perRankData[srcDev].minWidth = LOC_WIDTH;
    perRankData[srcDev].maxHops = 0;

    for (int dstDev = 0; dstDev < ngpus; ++dstDev) {
      if (srcDev == dstDev) continue;
      struct ncclTopoLinkList* path = srcNode->paths[GPU]+dstDev;
      if (path == NULL) {
        VERBOSE(NCCL_GRAPH, "FC: No paths existed between GPU %d and GPU %d.", srcDev, dstDev);
        goto cleanup;
      }
      // check path transport type between srcDev and dstDev
      if (path->type != perRankData[srcDev].maxType) {
        if (perRankData[srcDev].maxType == PATH_LOC) {
          perRankData[srcDev].maxType = path->type;
        } else {
          VERBOSE(NCCL_GRAPH, "FC: Have distinct path types between GPU %d and its peer devices, %d and %d separately.",
              srcDev, perRankData[srcDev].maxType, path->type);
          goto cleanup;
        }
      }

      // check path width between srcDev and dstDev
      if (path->width != perRankData[srcDev].minWidth) {
        if (perRankData[srcDev].minWidth == LOC_WIDTH) {
          perRankData[srcDev].minWidth = path->width;
        } else {
          VERBOSE(NCCL_GRAPH, "FC: Have distinct bandwidths between GPU %d and its peers, %d and %d separately.",
              srcDev, perRankData[srcDev].minWidth, path->width);
          goto cleanup;
        }
      }

      // check path hopps between srcDev and dstDev
      if (path->count != perRankData[srcDev].maxHops + 1) {
        if (perRankData[srcDev].maxHops == 0) {
          perRankData[srcDev].maxHops = path->count - 1;
        } else {
          VERBOSE(NCCL_GRAPH, "FC: Have distinct Hops between GPU %d and its peers, %d and %d separately.",
              srcDev, perRankData[srcDev].maxHops+1, path->count);
          goto cleanup;
        }
      }
    }
  }

  if (perRankData[0].maxType > PATH_NVB) {
    VERBOSE(NCCL_GRAPH, "FC: Path %s is not supported", topoPathTypeStr[PATH_NVB]);
    goto cleanup;
  } else {
    for (int g = 1; g < ngpus; ++g) {
      if (perRankData[g].maxHops != perRankData[0].maxHops ||
          perRankData[g].maxType != perRankData[0].maxType ||
          perRankData[g].minWidth != perRankData[0].minWidth) {
          VERBOSE(NCCL_GRAPH, "FC: GPU %d does not have qualified direct-FC paths with other GPUs", graph->id, graph->pattern);
          goto cleanup;
      }
    }
  }

  // calculate nChannels and intra
  numLinks = perRankData[0].minWidth / ncclTopoNVLinkSpeed(system->nodes[GPU].nodes[0].gpu.hggcCompCap);
  graph->nChannels = numLinks * nPeers * blocksPerLink;
  INFO(NCCL_GRAPH, "Each FC node has %d %s to each peer", numLinks, topoPathTypeStr[perRankData[0].maxType]);
  if (graph->nChannels > 0) {
    graph->nChannels = std::max(graph->nChannels, graph->minChannels);
    graph->nChannels = std::min(graph->nChannels, graph->maxChannels);
  } else {
    VERBOSE(NCCL_GRAPH, "0 channels found for FC.");
    goto cleanup;
  }

  graph->typeIntra = perRankData[0].maxType;
  graph->speedIntra = perRankData[0].minWidth * nPeers / graph->nChannels;
  graph->nHops = perRankData[0].maxHops;
  for (int i=0; i<ngpus; i++) {
    struct ncclTopoNode* node = system->nodes[GPU].nodes + i;
    graph->intra[i] = node->gpu.rank;
  }
  std::sort(graph->intra, graph->intra+ngpus);

  // release inter-mediate resource
cleanup:
  NCCLCHECK(ncclFree(perRankData));

  return ret;
}

ncclResult_t ncclTopoDoubleGraphChannels(ncclTopoGraph *graph, int ngpus, int *dupChannels) {
  *dupChannels = std::min(graph->nChannels*2, graph->maxChannels);
  memcpy(graph->intra+graph->nChannels*ngpus, graph->intra, (*dupChannels-graph->nChannels)*ngpus*sizeof(int));
  memcpy(graph->inter+graph->nChannels*2, graph->inter, (*dupChannels-graph->nChannels)*2*sizeof(int64_t));
  memcpy(graph->proxyDevs+graph->nChannels, graph->proxyDevs, graph->nChannels*sizeof(int));

  return ncclSuccess;
}

PCCL_PARAM(GraphSearchStartInterType, "GRAPH_SEARCH_START_INTER_TYPE", PATH_PXN);
ncclResult_t ncclTopoCompute(ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  int ngpus = system->nodes[GPU].count;
  int startInterType = PATH_PIX;
  graph->crossNic = ncclParamCrossNic();
  int crossNic = (system->nodes[NET].count > 1) && graph->crossNic ? 1 : 0;
  graph->speedIntra = graph->speedInter = 0;
  if (graph->crossNic == 2) graph->crossNic = 0;
  graph->typeIntra = ngpus == 1 ? PATH_LOC : PATH_NVL;
  graph->nGraphChannels = graph->nChannels = graph->nSubChannels = 0;
  graph->sameChannels = 1;
  if (!system->icnFullyConnected && graph->pattern == NCCL_TOPO_PATTERN_RING && ngpus > 1)
    startInterType = pcclParamGraphSearchStartInterType();
  graph->typeInter = startInterType;

  for (int i = 0; i < MAXCHANNELS; i++) graph->proxyDevs[i] = -1;

  const char* str = pcclGetEnvWithDualPrefix("PCCL_GRAPH_FILE");
  if (str) {
    INFO(NCCL_ENV, "PCCL_GRAPH_FILE set by environment to %s", str);
    struct ncclXml* xml;
    NCCLCHECK(ncclCalloc(&xml, 1));
    NCCLCHECK(ncclTopoGetXmlGraphFromFile(str, xml));
    int nChannels;
    NCCLCHECK(ncclTopoGetGraphFromXml(xml->nodes, system, graph, &nChannels));
    INFO(NCCL_GRAPH, "Search %d : %d channels loaded from XML graph", graph->id, nChannels);
    NCCLCHECK(ncclFree(xml));
    if (graph->nChannels > 0) return ncclSuccess;
  }

  int nSpeeds = 0;
  const char *speedStr = pcclGetEnvWithDualPrefix("PCCL_GRAPH_SPEEDS");
  if (speedStr) {
    INFO(NCCL_ENV, "PCCL_GRAPH_SPEEDS set by environment to %s", speedStr);
    char* tokStr = strdup(speedStr);
    char* tmpStr;
    char* token = strtok_r(tokStr, ",", &tmpStr);
    while (token && nSpeeds < NSPEEDS) {
        speedArray[nSpeeds++] = strtof(token, NULL);
        token = strtok_r(NULL, ",", &tmpStr);
    }
    NCCLCHECK(ncclFree(tokStr));
  } else {
    nSpeeds = NSPEEDS;
  }

  if (graph->pattern == NCCL_TOPO_PATTERN_FC) {
    // check if we could construct a fc graph
    NCCLCHECK(ncclTopoSearchTryFC(system, graph));

    if (graph->nChannels > 0) {
      VERBOSE(NCCL_GRAPH, "Search graph id(%d): %d channels built", graph->id, graph->nChannels);
    } else {
      VERBOSE(NCCL_GRAPH, "Could not build graph id(%d) with graph pattern: %s", graph->id, ncclTopoPatternStr[graph->pattern]);
    }
    return ncclSuccess;
  }

  if (ngpus == 1) if (graph->pattern != NCCL_TOPO_PATTERN_RING) graph->pattern = NCCL_TOPO_PATTERN_TREE;

  struct ncclTopoGraph tmpGraph;
  memcpy(&tmpGraph, graph, sizeof(struct ncclTopoGraph));

  // First try crossnic, then decrease speed and finally increase speedIntra.
  int pass = 1;
  int speedIndex = 0;
  while (speedArray[speedIndex] > system->maxWidth && speedIndex < nSpeeds-1) speedIndex++;
  tmpGraph.speedIntra = tmpGraph.speedInter = speedArray[speedIndex];
  int64_t globalTimeout = NCCL_SEARCH_GLOBAL_TIMEOUT;

search:
  int time = tmpGraph.sameChannels ? NCCL_SEARCH_TIMEOUT_SAMECHANNELS :
    tmpGraph.pattern == NCCL_TOPO_PATTERN_TREE ? NCCL_SEARCH_TIMEOUT_TREE : NCCL_SEARCH_TIMEOUT;
  tmpGraph.nChannels = 0;
  globalTimeout -= time;

  NCCLCHECK(ncclTopoSearchRec(system, &tmpGraph, graph, &time));
#ifdef DEBUG_SEARCH
  printf("Pattern %d, crossNic %d, Speed %g/%g, type %d/%d, channels %d-%d sameChannels %d -> nChannels %dx%g/%g %s\n", tmpGraph.pattern, tmpGraph.crossNic, tmpGraph.speedInter, tmpGraph.speedIntra, tmpGraph.typeInter, tmpGraph.typeIntra, tmpGraph.minChannels, tmpGraph.maxChannels, tmpGraph.sameChannels, graph->nChannels, graph->speedInter, graph->speedIntra, time == 0 ? "TIMEOUT" : time == -1 ? "PERFECT" : "");
  for (int c=0; c<graph->nChannels; c++) {
    printf("%2d : ", c);
    for (int g=0; g<ngpus; g++) {
      printf("%d ", graph->intra[c*ngpus+g]);
    }
    printf("[%lx %lx]", graph->inter[c*2+0], graph->inter[c*2+1]);
    printf("\n");
  }
#endif
  // Optimal solution, stop here
  if (time == -1) goto done;
  if (graph->nChannels*graph->speedInter >= system->totalWidth and graph->typeInter < PATH_PHB) goto done;
  if ((graph->nChannels*graph->speedInter >= system->totalNetWidth) &&
      (graph->speedInter == PATH_PXN && isGraphWithAllPxnChannels(graph))) {
    goto done;
  }

  if (pass == 1) {
    // First pass, we don't have a solution yet ; try other options

    // Try having different channels
    if (tmpGraph.sameChannels == 1) {
      tmpGraph.sameChannels = 0;
      goto search;
    }
    tmpGraph.sameChannels = 1;

    if (time != -1) globalTimeout += time;
    else globalTimeout = NCCL_SEARCH_GLOBAL_TIMEOUT;
    if (globalTimeout < 0 && graph->nChannels) goto done;

    int maxTypeIntra = !system->icnFullyConnected ? tmpGraph.typeInter : PATH_SYS;
    if (tmpGraph.typeIntra < maxTypeIntra && (graph->nChannels == 0 || tmpGraph.typeIntra < graph->typeIntra)) {
      tmpGraph.typeIntra += 1;
      goto search;
    }
    tmpGraph.typeIntra = ngpus == 1 ? PATH_LOC : PATH_NVL;
    if (!system->icnFullyConnected && tmpGraph.typeInter < PATH_SYS && (graph->nChannels == 0 || tmpGraph.typeInter < graph->typeInter || tmpGraph.typeInter < PATH_PXR)) {
      tmpGraph.typeInter += 1;
      goto search;
    }
    tmpGraph.typeInter = startInterType;

    if (crossNic && tmpGraph.crossNic == 0) {
      // Try again with crossNic if permitted
      tmpGraph.crossNic = crossNic;
      goto search;
    }
    tmpGraph.crossNic = 0;

    // Try a simpler tree
    if (tmpGraph.pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) {
      tmpGraph.pattern = NCCL_TOPO_PATTERN_TREE;
      goto search;
    }
    tmpGraph.pattern = graph->pattern;

    // Decrease speed until we find a solution
    if ((speedIndex < nSpeeds-1) && (graph->nChannels == 0 || (speedArray[speedIndex+1]/graph->speedInter > .49))) {
      tmpGraph.speedInter = tmpGraph.speedIntra = speedArray[++speedIndex];
      goto search;
    }
    speedIndex = 0;
    while (speedArray[speedIndex] > system->maxWidth && speedIndex < nSpeeds-1) speedIndex++;
    tmpGraph.speedIntra = tmpGraph.speedInter = speedArray[speedIndex];

  }

done:
  // We have a solution. Start from that solution and move to pass 2.
  if (pass == 1) {
    time = -1;
    memcpy(&tmpGraph, graph, sizeof(tmpGraph));
    speedIndex = 0;
    while (speedArray[speedIndex] > graph->speedInter && speedIndex < nSpeeds-1) speedIndex++;
    tmpGraph.speedIntra = tmpGraph.speedInter = speedArray[speedIndex];
    tmpGraph.minChannels = graph->nChannels;
    pass = 2;
  }

  // 3. See if we can increase speedIntra for trees (2 nodes or collnet)
  if (pass == 2) {
    if (time != 0 && graph->pattern != NCCL_TOPO_PATTERN_RING &&
        tmpGraph.speedIntra == graph->speedIntra && tmpGraph.speedIntra < tmpGraph.speedInter*2 &&
        speedIndex > 0) {
      tmpGraph.speedIntra = speedArray[--speedIndex];
      goto search;
    }
    time = -1;
    memcpy(&tmpGraph, graph, sizeof(tmpGraph));
  }

  if (graph->nChannels == 0 && graph->collNet == 0) {
    INFO(NCCL_GRAPH, "Could not find a path for pattern %s, falling back to simple order", ncclTopoPatternStr[graph->pattern]);
    for (int i=0; i<ngpus; i++) graph->intra[i] = system->nodes[GPU].nodes[i].gpu.rank;
    graph->inter[0] = graph->inter[1] = 0;
    graph->speedIntra = graph->speedInter = 0.1;
    graph->typeIntra = graph->typeInter = PATH_SYS;
    graph->nChannels = 1;
  }
  graph->nGraphChannels = graph->nChannels;
  while (graph->speedIntra >= ncclTopoNVLinkSpeed(system->nodes[GPU].nodes[0].gpu.hggcCompCap)) {
    int dupChannels = -1;
    NCCLCHECK(ncclTopoDoubleGraphChannels(graph, ngpus, &dupChannels));

    graph->speedIntra /= DIVUP(dupChannels, graph->nChannels);
    graph->speedInter /= DIVUP(dupChannels, graph->nChannels);
    graph->nChannels = dupChannels;
    if (dupChannels >= graph->maxChannels) break;
  }

  return ncclSuccess;
}

ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  if (graph->nChannels <= 0) return ncclSuccess;

  INFO(NCCL_GRAPH, "Pattern %s, crossNic %d, nChannels %d, speed %f/%f, type %s/%s, sameChannels %d, perfectPxn %d", ncclTopoPatternStr[graph->pattern], graph->crossNic, graph->nChannels, graph->speedIntra, graph->speedInter, topoPathTypeStr[graph->typeIntra], topoPathTypeStr[graph->typeInter], graph->sameChannels, isGraphWithAllPxnChannels(graph));
  if (graph->id >= 3) {
    return ncclSuccess;
  }

  int ngpus = system->nodes[GPU].count;

  char line[1024];
  for (int c=0; c<graph->nChannels; c++) {
    sprintf(line, "%2d :", c);
    int offset = strlen(line);
    if (!system->icnFullyConnected) {
      sprintf(line+offset, " %s/%lx-%lx", topoNodeTypeStr[NET], NCCL_TOPO_ID_SYSTEM_ID(graph->inter[2*c]), NCCL_TOPO_ID_LOCAL_ID(graph->inter[2*c]));
      offset = strlen(line);
    }
    // only print intra of ring/tree/collNet for now
    for (int i=0; i<ngpus; i++) {
      sprintf(line+offset, " %s/%d", topoNodeTypeStr[GPU], graph->intra[ngpus*c+i]);
      offset = strlen(line);
    }
    if (!system->icnFullyConnected) {
      sprintf(line+offset, " %s/%lx-%lx", topoNodeTypeStr[NET], NCCL_TOPO_ID_SYSTEM_ID(graph->inter[2*c+1]), NCCL_TOPO_ID_LOCAL_ID(graph->inter[2*c+1]));
      offset = strlen(line);
    }
    INFO(NCCL_GRAPH, "%s", line);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoDumpGraphs(struct ncclTopoSystem* system, int ngraphs, struct ncclTopoGraph** graphs) {
  ncclResult_t res = ncclSuccess;
  struct ncclXml* xml = nullptr;
  const char* str = pcclGetEnvWithDualPrefix("PCCL_GRAPH_DUMP_FILE");
  if (str) {
    INFO(NCCL_ENV, "PCCL_GRAPH_DUMP_FILE set by environment to %s", str);
    NCCLCHECK(xmlAlloc(&xml, NCCL_GRAPH_XML_MAX_NODES));
    NCCLCHECKGOTO(ncclTopoGetXmlFromGraphs(ngraphs, graphs, system, xml), res, end);
    NCCLCHECKGOTO(ncclTopoDumpXmlToFile(str, xml), res, end);
  }

end:
  NCCLCHECK(ncclFree(xml));
  return res;
}

ncclResult_t ncclTopoDumpRings(int *rings, int nrings, int nranks) {
  ncclResult_t res = ncclSuccess;
  const char *xmlRingFile = pcclGetEnvWithDualPrefix("PCCL_RING_DUMP_FILE");
  struct ncclXml* xml = NULL;
  if (xmlRingFile) {
    INFO(NCCL_ENV, "PCCL_RING_DUMP_FILE set by environment to %s", xmlRingFile);
    NCCLCHECK(ncclCalloc(&xml, 1));
    struct ncclXmlNode* xmlRings = NULL;
    NCCLCHECKGOTO(xmlAddNode(xml, NULL, "rings", &xmlRings), res, fail);
    for (int i=0; i<nrings; i++) {
      struct ncclXmlNode* xmlRing = NULL;
      NCCLCHECKGOTO(xmlAddNode(xml, xmlRings, "channel", &xmlRing), res, fail);
      NCCLCHECKGOTO(xmlSetAttrInt(xmlRing, "id", i), res, fail);
      for (int j=0; j<nranks; j++) {
        char key[255];
        sprintf(key, "%d", j);
        NCCLCHECKGOTO(xmlSetAttrInt(xmlRing, key, rings[i*nranks+j]), res, fail);
      }
    }
    NCCLCHECKGOTO(ncclTopoDumpXmlToFile(xmlRingFile, xml), res, fail);
  }

fail:
  NCCLCHECK(ncclFree(xml));
  return res;
}

// 0: don't use PXN for P2P, 1: use PXN if needed, 2: use PXN as much as possible to maximize aggregation
NCCL_PARAM(P2pPxnLevel, "P2P_PXN_LEVEL", 2);

#include "comm.h"

static void ncclTopoGetP2pPxnCandidates(ncclComm_t comm, int g, int n, const int* localGpus, int* cands, int* count) {
  *count = 0;
  // Exclude sources that cannot use this NIC from the balancing group.
  if (comm->topo->nodes[GPU].nodes[g].paths[NET][n].type > PATH_PXR) return;
  struct ncclTopoNode* net = comm->topo->nodes[NET].nodes+n;
  int nHops = INT_MAX;
  for (int r=0; r<comm->localRanks; r++) {
    int peerRank = comm->localRankToRank[r];
    if (comm->peerInfo[peerRank].netId != net->id) continue;
    int p = localGpus[r];
    struct ncclTopoLinkList* path = comm->topo->nodes[GPU].nodes[p].paths[GPU]+g;
    if (path->type > PATH_NVB || net->paths[GPU][p].type > PATH_PXB) continue;
    if (path->count < nHops) {
      nHops = path->count;
      *count = 0;
    }
    // Local-rank order gives identical candidate sets an identical ordering.
    if (path->count == nHops) cands[(*count)++] = peerRank;
  }
}

ncclResult_t ncclTopoGetNetDev(ncclComm_t comm, int rank, struct ncclTopoGraph* graph, int channelId, int peerRank, int64_t* id, int* dev, int* proxyRank) {
  int64_t netId = -1;
  int netDev = -1;
  if (graph) {
    // Honor the net device in the graph
    int channel = channelId%graph->nChannels;
    if (ncclHggcCompCap() == 80 && comm->channelShuffle) {
      channel = channelId / (comm->nChannels / graph->nGraphChannels);
    }
    int ngpus = comm->topo->nodes[GPU].count;
    int index = graph->intra[channel*ngpus] == rank ? 0 : 1;
    netId = graph->inter[channel*2+index];
    NCCLCHECK(ncclTopoIdToNetDev(comm->topo, netId, &netDev));
    if (dev) *dev = netDev;
    if (id) *id = netId;
    if (proxyRank) { // recv is null
      if (graph->proxyDevs[channelId] != -1) {
        NCCLCHECK(ncclTopoDevToRank(comm->topo, comm->topo->systemId, graph->proxyDevs[channelId], proxyRank));
      } else {
        NCCLCHECK(ncclTopoGetIntermediateRank(comm->topo, rank, netId, proxyRank));
      }
    }
  } else if (peerRank == -1) {
    return ncclInternalError;
  } else {
    // Start with our local NIC and local Rank
    NCCLCHECK(ncclTopoGetLocalNet(comm->topo, rank, &netId, &netDev));
    if (dev) *dev = netDev;
    if (id) *id = netId;
    if (!proxyRank) return ncclSuccess;

    *proxyRank = rank;

    int pxnLevel = ncclPxnDisable() == 1 ? 0 : ncclParamP2pPxnLevel();
    // See whether we can use the remote rank preferred device.
    if (ncclParamCrossNic() == 0 || (pxnLevel != 0)) {
      int hgmlDev = comm->peerInfo[peerRank].hgmlDev;
      int localRank;
      if (ncclTopoDevToRank(comm->topo, comm->topo->systemId, hgmlDev, &localRank) != ncclSuccess) return ncclSuccess;
      netId = comm->peerInfo[localRank].netId;
      NCCLCHECK(ncclTopoIdToNetDev(comm->topo, netId, &netDev));
      // Check that device exists on our node
      if (ncclParamCrossNic() == 0) {
        if (dev) *dev = netDev;
        if (id) *id = netId;
      }
      if (pxnLevel == -1) {
        // no PXN and no Proxy, handle it by self
        netId = comm->peerInfo[rank].netId;
        NCCLCHECK(ncclTopoIdToNetDev(comm->topo, netId, &netDev));
        if (dev) *dev = netDev;
        if (id) *id = netId;
      } else if (pxnLevel == 1) {
        int g, n;
        NCCLCHECK(ncclTopoRankToIndex(comm->topo, rank, &g));
        NCCLCHECK(ncclTopoIdToIndex(comm->topo, NET, netId, &n));
        struct ncclTopoNode* gpu = comm->topo->nodes[GPU].nodes+g;
        if (gpu->paths[NET][n].type <= PATH_PXR) {
          if (dev) *dev = netDev;
          if (id) *id = netId;
          NCCLCHECK(ncclTopoGetIntermediateRank(comm->topo, rank, netId, proxyRank));
        }
      } else if (pxnLevel == 2) {
        int g, n;
        NCCLCHECK(ncclTopoRankToIndex(comm->topo, rank, &g));
        NCCLCHECK(ncclTopoIdToIndex(comm->topo, NET, netId, &n));
        // Keep proxy selection consistent with GDR path checks.
        // Reject PHB/SYS paths to avoid PXN shared-buffer failures.
        if (comm->topo->nodes[GPU].nodes[g].paths[NET][n].type <= PATH_PXR) {
          int localGpus[NCCL_TOPO_MAX_NODES];
          // Resolve indices once; group comparisons only scan cached paths.
          for (int r=0; r<comm->localRanks; r++) {
            NCCLCHECK(ncclTopoRankToIndex(comm->topo, comm->localRankToRank[r], localGpus+r));
          }
          int cands[NCCL_TOPO_MAX_NODES];
          int count;
          ncclTopoGetP2pPxnCandidates(comm, g, n, localGpus, cands, &count);
          if (count > 0) {
            int ordinal = 0;
            if (count > 1) {
              int otherCands[NCCL_TOPO_MAX_NODES];
              // Only earlier sources with this exact candidate set advance the ordinal.
              for (int r=0; r<comm->rankToLocalRank[rank]; r++) {
                int otherCount;
                ncclTopoGetP2pPxnCandidates(comm, localGpus[r], n, localGpus, otherCands, &otherCount);
                if (otherCount == count && memcmp(cands, otherCands, count*sizeof(int)) == 0) ordinal++;
              }
            }
            *proxyRank = cands[ordinal % count];
            if (dev) *dev = netDev;
            if (id) *id = netId;
          }
        }
      } else if (pxnLevel == 3) {
        // Try to use balanced proxy for that NIC.
        int g, n;
        NCCLCHECK(ncclTopoRankToIndex(comm->topo, rank, &g));
        NCCLCHECK(ncclTopoIdToIndex(comm->topo, NET, netId, &n));
        struct ncclTopoNode* gpu = comm->topo->nodes[GPU].nodes+g;
        if (gpu->paths[NET][n].type <= PATH_PXR) {
          if (dev) *dev = netDev;
          if (id) *id = netId;
          NCCLCHECK(ncclTopoGetBalancedProxyRank(comm->topo, g, n, proxyRank));
        }
      }
    }
  }
  return ncclSuccess;
}
