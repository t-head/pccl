/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "core.h"
#include "devcomm.h"
#include "comm.h"
#include "topo.h"
#include "channel.h"

#include <unordered_map>
#include <array>

NCCL_PARAM(Nthreads, "NTHREADS", -2);
NCCL_PARAM(Ll128Nthreads, "LL128_NTHREADS", -2);
PCCL_PARAM(MultiHostsBulkMode, "MULTI_HOSTS_BULK_MODE", 1);
PCCL_PARAM(TuningFallBack, "TUNING_FALL_BACK", 1);

static int getNthreads(const char* name, int env, int min, int max, int def) {
  int nt = env;
  if (nt > 0) {
    if (nt % WARP_SIZE != 0) {
      WARN("Invalid %s %d (must be a multiple of %d)", name, nt, WARP_SIZE);
      nt = max;
    } else if (nt > max) {
      WARN("Invalid %s %d (maximum %d).", name, nt, max);
      nt = max;
    } else if (nt < min) {
      WARN("Invalid %s %d (minimum %d).", name, nt, min);
      nt = min;
     }
  } else {
    nt = def;
  }
  return nt;
}

// Parse a map of prefixes to a list of elements. The first prefix is
// optional and, if not present, the list of elements will be applied
// to all prefixes. Only the first list of elements can lack a
// prefix. Prefixes (if present) are followed by a colon. Lists of
// elements are comma delimited. Mappings of prefix to the lists of
// elements are semi-colon delimited.
//
// For example:
//
//     NCCL_ALGO="ring,collnetdirect;allreduce:tree,collnetdirect;broadcast:ring"
// Enable ring and collnetdirect for all functions, then select tree
// and collnetdirect for allreduce and ring for broadcast.
//
//     NCCL_PROTO="LL,Simple;allreduce:^LL"
// Enable LL and Simple for all functions, but everything except LL
// for allreduce.
//
//     NCCL_PROTO="^LL128;allreduce:LL128"
// Enable everything but LL128, but only LL128 for allreduce.
ncclResult_t ncclParseList(const char* str, const char* prefixElems[], int nprefixes, const char* elems[], int nelems, int* list) {
  ncclResult_t ret = ncclSuccess;
  char* fullStr = strdup(str);
  char* tmpFullStr;
  char* fullToken = strtok_r(fullStr, ";", &tmpFullStr);
  char* subToken = nullptr;
  while (fullToken) {
    subToken = strdup(fullToken);
    char* tmpSubStr;
    char* prefix = strtok_r(subToken, ":", &tmpSubStr);
    char* elemList = strtok_r(NULL, ":", &tmpSubStr);
    if (elemList == NULL) {
      if (fullToken != fullStr) {
        // It makes no sense for any entry other than the first to not have a prefix,
        // because then all the prefixes before the prefix-less entry would be
        // overwritten.
        WARN("All entries except the first must have a prefix: \"%s\"", str);
        ret = ncclInvalidUsage;
        goto exit;
      }
      elemList = prefix;
      prefix = NULL;
    }

    int unset, set;
    if (elemList[0] == '^') {
      unset = 1; set = 0; elemList++;
    } else {
      unset = 0; set = 1;
    }

    bool foundPrefix = false;
    for (int p=0; p<nprefixes; p++) {
      if (prefix && strcasecmp(prefix, prefixElems[p]) != 0) continue;
      foundPrefix = true;
      for (int e=0; e<nelems; e++) list[p*nelems+e] = unset;

      char* tokStr = strdup(elemList);
      char* tmpStr;
      char* elem = strtok_r(tokStr, ",", &tmpStr);
      while (elem) {
        int e;
        for (e=0; e<nelems; e++) {
          if (strcasecmp(elem, elems[e]) == 0) {
            list[p*nelems+e] = set;
            break;
          }
        }
        if (e==nelems) {
          WARN("Unrecognized element token \"%s\" when parsing \"%s\" for %s", elem, str, set ? "set" : "unset");
          if (set) {
            ret =  ncclInvalidUsage;
            ncclFree(tokStr);
            goto exit;
          }
        }
        elem = strtok_r(NULL, ",", &tmpStr);
      }
      ncclFree(tokStr);
    }
    if (!foundPrefix) {
      WARN("Unrecognized prefix token \"%s\" when parsing \"%s\"", prefix, str);
      ret = ncclInvalidUsage;
      goto exit;
    }
    ncclFree(subToken);

    fullToken = strtok_r(NULL, ";", &tmpFullStr);
  }

exit:
  ncclFree(subToken);
  ncclFree(fullStr);
  return ret;
}

static const int kernelEnabled[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {
  { /* Broadcast */
    /* Tree (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */    {0,  0,  0,  0,  0,  0,  0 },
    /* Ring (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */    {80, 80, 80, 80, 80, 80, 89},
    /* CollNet (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */ {0,  0,  0,  0,  0,  0,  0 },
    /* FC (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */      {80, 0,  80, 0,  80, 0,  0 },
    /* PAT (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */     {0,  0,  0,  0,  0,  0,  0 },
  },
  { /* Reduce */
    /* Tree (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */    {0,  0,  0,  0,  0,  0,  0 },
    /* Ring (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */    {80, 80, 80, 80, 80, 80, 89},
    /* CollNet (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */ {0,  0,  0,  0,  0,  0,  0 },
    /* FC (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */      {80, 0,  80, 0,  80, 0,  0 },
    /* PAT (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */     {0,  0,  0,  0,  0,  0,  0 },
  },
  { /* AllGather */
    /* Tree (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */    {0,  0,  0,  0,  0,  0,  0 },
    /* Ring (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */    {80, 80, 80, 80, 80, 80, 89},
    /* CollNet (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */ {0,  0,  0,  0,  0,  0,  0 },
    /* FC (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */      {80, 0,  80, 0,  80, 0,  0 },
    /* PAT (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */     {0,  0,  80, 0,  0,  0,  0 },
  },
  { /* ReduceScatter */
    /* Tree (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */    {0,  0,  0,  0,  0,  0,  0 },
    /* Ring (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */    {80, 80, 80, 80, 80, 80, 89},
    /* CollNet (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */ {0,  0,  0,  0,  0,  0,  0 },
    /* FC (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */      {80, 0,  80, 0,  80, 0,  0 },
    /* PAT (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */     {0,  0,  80, 0,  0,  0,  0 },
  },
  { /* AllReduce */
    /* Tree (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */    {80, 80, 80, 80, 80, 80, 0 },
    /* Ring (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */    {80, 80, 80, 80, 80, 80, 89},
    /* CollNet (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */ {80, 80, 80, 80, 80, 80, 0 },
    /* FC (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */      {80, 0,  80, 0,  80, 0,  0 },
    /* PAT (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */     {0,  0,  0,  0,  0,  0,  0 },
  }
};

/* LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR */
static const int p2pKernelEnabled[NCCL_NUM_PROTOCOLS] = {0, 0, 80, 80, 80, 80, 89};

// Latencies in us, Bandwidths in GB/s
// Tree { LL, LL128, Simple, Signal, SimpleBulk, simpleBulkAR } , Ring { LL, LL128, Simple, Signal, SimpleBulk, simpleBulkAR }
// CollNet { LL, LL128, Simple, Signal, SimpleBulk, simpleBulkAR } , FC { LL, LL128, Simple, Signal, SimpleBulk, simpleBulkAR }
// PAT { LL, LL128, Simple, Signal, SimpleBulk, simpleBulkAR }
static const float baseLat  [NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {
  { 4.4, 4.4,  0, 0, 0, 0, 0 }, { 3.6, 10.0, 8.4, 8.4, 8.4, 8.4, 8.4 },
  { 4.4, 4.4,  0, 0, 0, 0, 0 }, { 3.6, 10.0, 8.0, 8.4, 8.4, 8.4, 8.4 },
  { 4.4, 4.4,  0, 0, 0, 0, 0 } };

// NVLink, PCI, Network
#define NCCL_HW_NVLINK 0
#define NCCL_HW_PCI 1
#define NCCL_HW_NET 2
// Tree/Simple is the latency a 256kB chunk, which is ~ base lat + 256k/12GB/s (+ 256k/12GB/s for the network).
static const float hwLat [3][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] =
{ /* NVLINK */
  { /* Tree (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/simpleBulkAR)*/ { .52, 1.25, 28, 28, 28, 28, 28 },
    /* Ring (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/simpleBulkAR)*/ { .47, 1.9, 3.4, 3.4, 3.4, 3.4, 3.4 },
    /* CollNet (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/simpleBulkAR)*/ {  .5, 1.2, 4.0, 4.0, 4.0, 4.0, 4.0 },
    /* FC (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/simpleBulkAR)*/ { .47, 1.9, 3.4, 3.4, 3.4, 3.4, 3.4 },
    /* PAT (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/simpleBulkAR)*/ { .47, 1.9, 3.4, 3.4, 3.4, 3.4, 3.4 }
  },
  /* PCI */
  { /* Tree (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/simpleBulkAR)*/ { 1.0, 1.9, 28, 28, 28, 28, 28 },
    /* Ring (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/simpleBulkAR)*/ { 1.0, 2.5, 5.7, 5.7, 5.7, 5.7, 5.7 },
    /* CollNet (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/simpleBulkAR)*/ { 1.0, 1.9, 5.5, 5.5, 5.5, 5.5, 5.5 },
    /* FC (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/simpleBulkAR)*/ { 1.0, 2.5, 5.7, 5.7, 5.7, 5.7, 5.7 },
    /* PAT (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/simpleBulkAR)*/ { 1.0, 2.5, 5.7, 5.7, 5.7, 5.7, 5.7 }
  },
  /* NET */
  { /* Tree (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/simpleBulkAR)*/ { 5.0, 8.5, 28, 28, 28, 28, 28 },
    /* Ring (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/simpleBulkAR)*/ { 2.7, 4.0, 9.6, 9.6, 9.6, 9.6, 9.6 },
    /* CollNet (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/simpleBulkAR)*/ { 5.0, 5.0, 10.7, 10.7, 10.7, 10.7, 10.7 },
    /* FC (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/simpleBulkAR)*/ { 2.7, 4.0, 9.6, 9.6, 9.6, 9.6, 9.6 },
    /* PAT (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/simpleBulkAR)*/ { 2.7, 4.0, 9.6, 9.6, 9.6, 9.6, 9.6 }
  }
};

// LL128 max BW (per channel) for the different collectives
// ncclFuncBroadcast, ncclFuncReduce, ncclFuncAllGather, ncclFuncReduceScatter, ncclFuncAllReduce
static const double ll128MaxBwPerCh[NCCL_NUM_FUNCTIONS] = { 18.8, 12.0, 18.3, 15.2, 16.9 };
static const double llMaxBws[2][3] = { /* Volta-N1/Intel-N2/Intel-N4) */ {39.0, 39.0, 20.4}, /* Ampere-N1/AMD-N2/AMD-N4) */ {87.7, 22.5 /*avg of ring & tree*/, 19.0} };
static const double perChMaxTreeBws[2][3] = { /* Volta (N1/N2/N4) */ {26.5, 18.5, 10.0}, /* Ampere (N1/N2/N4) */ {24.0, 22.5, 16.0} };

PCCL_PARAM(PatEnable, "PAT_ENABLE", 2);
int pcclPatEnable(ncclComm_t comm) {
  int patEnable = pcclParamPatEnable();
  if (patEnable != 2) return patEnable;
  if (comm->nNodes != comm->nRanks) return 0; // PAT only supports 1 GPU per node
  return 1;
}

ncclResult_t ncclTopoTuneModel(ncclComm_t comm, int minCompCap, int maxCompCap, struct ncclTopoGraph* treeGraph,
    struct ncclTopoGraph* ringGraph, struct ncclTopoGraph* collNetGraph, struct ncclTopoGraph* fcGraph) {
  // on 810e single node tp16 scenario, use less warps for simplebulk/ll128 proto when use ring algo will achieve better performance.
  bool lessWarps = comm->ppuType == _PPU_ZW810E && comm->nRanks == 16 && comm->nNodes == 1;
  int simpleDefaultThreads = ((ringGraph->speedIntra*ringGraph->nChannels <= PCI_WIDTH) || lessWarps) ? NCCL_SIMPLE_MAX_NTHREADS/2 : NCCL_SIMPLE_MAX_NTHREADS;
  // SIMPLE_BULK_AR
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE_BULK_AR] =
    getNthreads("PCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, simpleDefaultThreads);
  comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE_BULK_AR] = comm->maxThreads[NCCL_ALGO_COLLNET][NCCL_PROTO_SIMPLE_BULK_AR] =
    getNthreads("PCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, NCCL_SIMPLE_MAX_NTHREADS);
  comm->maxThreads[NCCL_ALGO_FC][NCCL_PROTO_SIMPLE_BULK_AR] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE_BULK_AR];
  // SIGNAL_BULK
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIGNAL_BULK] =
    getNthreads("PCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, simpleDefaultThreads);
  comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIGNAL_BULK] = comm->maxThreads[NCCL_ALGO_COLLNET][NCCL_PROTO_SIGNAL_BULK] =
    getNthreads("PCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, NCCL_SIMPLE_MAX_NTHREADS);
  comm->maxThreads[NCCL_ALGO_FC][NCCL_PROTO_SIGNAL_BULK] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIGNAL_BULK];
  // SIMPLE_BULK
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE_BULK] =
    getNthreads("PCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, simpleDefaultThreads);
  comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE_BULK] = comm->maxThreads[NCCL_ALGO_COLLNET][NCCL_PROTO_SIMPLE_BULK] =
    getNthreads("PCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, NCCL_SIMPLE_MAX_NTHREADS);
  comm->maxThreads[NCCL_ALGO_FC][NCCL_PROTO_SIMPLE_BULK] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE_BULK];
  // SIGNAL
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIGNAL] =
    getNthreads("PCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, simpleDefaultThreads);
  comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIGNAL] = comm->maxThreads[NCCL_ALGO_COLLNET][NCCL_PROTO_SIGNAL] =
    getNthreads("PCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, NCCL_SIMPLE_MAX_NTHREADS);
  comm->maxThreads[NCCL_ALGO_FC][NCCL_PROTO_SIGNAL] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIGNAL];
  // SIMPLE
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] =
    getNthreads("PCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, simpleDefaultThreads);
  comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = comm->maxThreads[NCCL_ALGO_COLLNET][NCCL_PROTO_SIMPLE] =
    getNthreads("PCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, NCCL_SIMPLE_MAX_NTHREADS);
  // PAT needs NCCL_MAX_NTHREADS to fit NCCL_PAT_NWORKERS + the algorithm thread.
  comm->maxThreads[NCCL_ALGO_PAT][NCCL_PROTO_SIMPLE] = NCCL_MAX_NTHREADS;
  comm->maxThreads[NCCL_ALGO_FC][NCCL_PROTO_SIMPLE] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE];
  // LL
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL] =
    getNthreads("PCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_LL_MAX_NTHREADS, NCCL_LL_MAX_NTHREADS);
  comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL] = comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL];
  comm->maxThreads[NCCL_ALGO_COLLNET][NCCL_PROTO_LL] = comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL];
  comm->maxThreads[NCCL_ALGO_FC][NCCL_PROTO_LL] = comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL];
  // LL128
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL128] =
    getNthreads("PCCL_LL128_NTHREADS", ncclParamLl128Nthreads(), NCCL_LL128_MAX_NTHREADS/4, NCCL_LL128_MAX_NTHREADS, lessWarps ? NCCL_LL128_MAX_NTHREADS/2 : NCCL_LL128_MAX_NTHREADS);
  comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL128] = comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL128];
  comm->maxThreads[NCCL_ALGO_COLLNET][NCCL_PROTO_LL128] = comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL128];
  comm->maxThreads[NCCL_ALGO_FC][NCCL_PROTO_LL128] = comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL128];

  int nNodes = comm->nNodes;
  int nRanks = comm->nRanks;
#ifndef KERNEL_LOOPBACK
  if (nRanks <= 1) return ncclSuccess;
#endif

  int compCap80 = minCompCap == 80 && maxCompCap == 80 ? 1 : 0;
  int cpuArch, cpuVendor, cpuModel;
  NCCLCHECK(ncclTopoCpuType(comm->topo, &cpuArch, &cpuVendor, &cpuModel));
  int index2 = nNodes <= 2 ? nNodes-1 : 2;
  // LL: for single node, we look at GPU type; for multi-node, we look at CPU type
  int index1 = nNodes == 1 ? compCap80 : cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD ? 1 : 0;
  double llMaxBw = llMaxBws[index1][index2];
  double perChMaxTreeBw = perChMaxTreeBws[compCap80][index2];
  float ppn = (float)nRanks / nNodes; // if ppn < 2, then we are sending/receiving at the same GPU through the NIC, apply some bw discount
  int multiHostsBulkMode = pcclParamMultiHostsBulkMode();

  struct ncclTopoGraph* graphs[NCCL_NUM_ALGORITHMS] = { treeGraph, ringGraph, collNetGraph, fcGraph, treeGraph };
  int intraHw[NCCL_NUM_ALGORITHMS], hw[NCCL_NUM_ALGORITHMS];
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) intraHw[a] = graphs[a]->typeIntra == LINK_NVL ? NCCL_HW_NVLINK : NCCL_HW_PCI;
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) hw[a] = nNodes == 1 ? intraHw[a] : NCCL_HW_NET;

  for (int coll=0; coll<NCCL_NUM_FUNCTIONS; coll++) {
    int nsteps = coll == ncclFuncAllReduce ? 2*(nRanks-1) :
      coll == ncclFuncReduceScatter || coll == ncclFuncAllGather ? nRanks-1 :
      nRanks;
    int nInterSteps = coll == ncclFuncAllReduce ? 2*(nNodes-1) :
      coll == ncclFuncReduceScatter || coll == ncclFuncAllGather ? nNodes-1 :
      nNodes;
    int fcnSteps = coll == ncclFuncAllReduce ? 2 : 1;

    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        if (kernelEnabled[coll][a][p] == 0 || kernelEnabled[coll][a][p] > maxCompCap) continue;
        if (a == NCCL_ALGO_PAT && ((coll != ncclFuncAllGather && coll != ncclFuncReduceScatter) || p != NCCL_PROTO_SIMPLE || pcclPatEnable(comm) == 0)) continue;
        float speed = nNodes <= 2 || a == NCCL_ALGO_COLLNET ? graphs[a]->speedIntra : graphs[a]->speedInter;
        float busBw = graphs[a]->nChannels * speed;

        // Extra model for ppu
        // normal v4 ld/st Vs remote v4 ld/st, normal every 128B with 26B header, remote every 512B with 26B header
        // tread remote as 1, normal deacy fac is (512+26)/(512+26*4)
        if (multiHostsBulkMode != 0) {
          if (multiHostsBulkMode == 1 || nNodes == 1 || comm->MNNVL == 1 ||
              (graphs[a]->id == 0 && graphs[a]->crossNic == 1) || (ppn == 1)) {
            float normalLoadStoreDecayFac = (512.f+26.f)*1.f/(512.f+26.f*4.f);
            if (p != NCCL_PROTO_SIMPLE_BULK && p != NCCL_PROTO_SIGNAL_BULK) busBw *= normalLoadStoreDecayFac;
          }
        }

        // disable remote bulk proto for none icn connection, do not consider inter/intra use different protoEnable
        // as bottleneck is inter node.
        if ((graphs[a]->typeIntra != PATH_NVL && graphs[a]->typeIntra != PATH_NVB) || (nNodes > 1 && comm->MNNVL == 0)) {
          if (p == NCCL_PROTO_SIGNAL || p == NCCL_PROTO_SIGNAL_BULK) {
            busBw = 0.0f;
          }
        }

        // Various model refinements
        if (a == NCCL_ALGO_RING && p == NCCL_PROTO_LL) { busBw = std::min(llMaxBw, busBw * ((nNodes > 1 || coll == ncclFuncAllReduce || coll == ncclFuncReduce) ? 1.0/4.0 : 1.0/3.0)); }
        if (a == NCCL_ALGO_RING && p == NCCL_PROTO_LL128) busBw = std::min(busBw * (ppn < 2 ? 0.7 : 0.92 /*120.0/128.0*/), ll128MaxBwPerCh[coll]*graphs[a]->nChannels);
        if (a == NCCL_ALGO_TREE) busBw = std::min(busBw*.92, graphs[a]->nChannels*perChMaxTreeBw);
        if (a == NCCL_ALGO_TREE && p == NCCL_PROTO_LL) busBw = std::min(busBw*1.0/3.8, llMaxBw);
        if (a == NCCL_ALGO_TREE && p == NCCL_PROTO_LL128) busBw = std::min(busBw * (nNodes == 1 ? 7.0/9.0 : 0.915 /*120.0/128.0*/), ll128MaxBwPerCh[coll]*graphs[a]->nChannels);
        if (a == NCCL_ALGO_COLLNET) busBw *= .9;
        if (a == NCCL_ALGO_COLLNET && p == NCCL_PROTO_LL) busBw *= 1.0/6.0; // Take into account that GDR read is disabled on both sides
        if (a == NCCL_ALGO_COLLNET && p == NCCL_PROTO_LL128) busBw = 0;  // CollNet does not support LL128

        // Convert bus BW to algorithm BW
        float ratio = (a == NCCL_ALGO_RING || a == NCCL_ALGO_FC) ? (1.0 * nRanks) / nsteps : .5;
        comm->bandwidths[coll][a][p] = busBw * ratio;
        /* Ring bandwidth backup */
        if (a == NCCL_ALGO_RING)
          comm->ringbdw[coll][p] = comm->bandwidths[coll][NCCL_ALGO_RING][p];

        comm->latencies[coll][a][p] = baseLat[a][p];
        float intraLat = hwLat[intraHw[a]][a][p];
        float interLat = hwLat[NCCL_HW_NET][a][p];
        if (nNodes > 1 && p == NCCL_PROTO_LL) intraLat *= 1.8;
        if (a == NCCL_ALGO_RING) {
          float lat = hwLat[hw[a]][a][p];
          if ((coll == ncclFuncReduce || coll == ncclFuncBroadcast)) {
            if (ringGraph->sameChannels) {
              comm->latencies[coll][a][p] += lat;
            } else {
              if (p == NCCL_PROTO_SIMPLE || p == NCCL_PROTO_SIGNAL ||
                  p == NCCL_PROTO_SIMPLE_BULK || p == NCCL_PROTO_SIGNAL_BULK) {
                lat = hwLat[hw[a]][NCCL_ALGO_TREE][p]; // Add some chunk latency, waiting for proper chunk modeling
              }
              comm->latencies[coll][a][p] += nsteps*lat;
            }
          } else {
            comm->latencies[coll][a][p] += (nsteps-nInterSteps)*intraLat + nInterSteps*interLat;
          }
        } else if (a == NCCL_ALGO_TREE) {
          comm->latencies[coll][a][p] +=
            2 * ((nRanks/nNodes-1) * intraLat + log2i(nNodes) * interLat);
        } else if (a == NCCL_ALGO_COLLNET) {
          comm->latencies[coll][a][p] +=
            2 * (nRanks/nNodes-1) * intraLat + interLat;
        } else if (a == NCCL_ALGO_FC) {
          comm->latencies[coll][a][p] += fcnSteps * hwLat[intraHw[a]][a][p];
        } else if (a == NCCL_ALGO_PAT) {
          // PAT is a binomial tree; latency grows logarithmically with rank count.
          int patSteps = log2i(nRanks);
          int interPatSteps = log2i(nNodes);
          comm->latencies[coll][a][p] += 2 * ((patSteps - interPatSteps) * intraLat + interPatSteps * interLat);
        } else {
          // for now, FC only supports nNodes == 1 scene, so no interLat count
          // TODO: need to support latency theoretical evaluation of 'FC' for runtime algo selection
          comm->latencies[coll][a][p] += 2 * intraLat;
        }
      }
    }
  }

  // Protocols/Algorithms enable/disable, and user overrides.
  // All are enabled except ll128 which is enabled by default only in certain cases.
  int protoEnable[NCCL_NUM_FUNCTIONS_WITH_P2P*NCCL_NUM_PROTOCOLS];
  for (int f=0; f<NCCL_NUM_FUNCTIONS_WITH_P2P; f++) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      protoEnable[f*NCCL_NUM_PROTOCOLS+p] = (p == NCCL_PROTO_LL128 || p == NCCL_PROTO_LL || p == NCCL_PROTO_SIMPLE_BULK_AR) ? ncclTuningCustomEnable : ncclTuningEnable;
    }
  }
  int* algoEnable = comm->algoEnable;

  const char *protoStr = pcclGetEnvWithDualPrefix("PCCL_PROTO");
  if (protoStr) {
    INFO(NCCL_ENV, "PCCL_PROTO set by environment to %s", protoStr);
    NCCLCHECK(ncclParseList(protoStr, ncclFuncStr, NCCL_NUM_FUNCTIONS_WITH_P2P, ncclProtoStr, NCCL_NUM_PROTOCOLS, protoEnable));
  }

  const char *algoStr = pcclGetEnvWithDualPrefix("PCCL_ALGO");
  if (comm->rank == 0 && (algoStr||protoStr)) {
    constexpr int strLength = 1024;
    char funcAlgoProtoTuningStr[strLength];
    int offset = 0;
    offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "\n     Function | ");
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "%12s  ", ncclProtoStr[p]);
    }
    offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), " | ");
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "%8s  ", ncclAlgoStr[a]);
    }
    offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "\n");

    for (int f=0; f<NCCL_NUM_FUNCTIONS; f++) {
      offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "%13s | ", ncclFuncStr[f]);
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "%12d  ", protoEnable[f*NCCL_NUM_PROTOCOLS+p]);
      }
      offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), " | ");
      for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
        offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "%8d  ", algoEnable[f*NCCL_NUM_ALGORITHMS+a]);
      }
      offset += snprintf(funcAlgoProtoTuningStr+offset, std::max(0, strLength-offset), "\n");
    }

    INFO(NCCL_ENV, "Enabled PCCL Func/Proto/Algo Matrix:%s", funcAlgoProtoTuningStr);
  }

  int fallBack = pcclParamTuningFallBack();
  for (int f=0; f<NCCL_NUM_FUNCTIONS; f++) {
    bool hasAvailableAlgo = false;
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      int disable = 0;
      // Disable unsupported algorithms
      if (comm->collNetSupport == 0 && a == NCCL_ALGO_COLLNET) disable = 1;
      if (a == NCCL_ALGO_FC &&
          (fcGraph->nChannels == 0 ||
            (comm->icnRmtType == PCCL_ICNLINK_SWITCH && algoEnable[f*NCCL_NUM_ALGORITHMS+a] == ncclTuningCustomEnable))) disable = 1;
      if (disable) algoEnable[f*NCCL_NUM_ALGORITHMS+a] = 0;
      if (fallBack && algoEnable[f*NCCL_NUM_ALGORITHMS+a] >= 1) hasAvailableAlgo = true;
    }
    // if no available algo, fallback to default config
    if (fallBack && !hasAvailableAlgo) {
      // if fc is not available, set the default val to 0
      int algoDefaultEnable[NCCL_NUM_ALGORITHMS] =
          {ncclTuningEnable, ncclTuningEnable, ncclTuningDisable, fcGraph->nChannels == 0 ? ncclTuningDisable : ncclTuningCustomEnable};
      for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
        algoEnable[f*NCCL_NUM_ALGORITHMS+a] = algoDefaultEnable[a];
      }
    }
  }

  for (int c=0; c<NCCL_NUM_FUNCTIONS; c++) for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    int pEnable = protoEnable[c*NCCL_NUM_PROTOCOLS+p];
    if (pEnable == 1 && p == NCCL_PROTO_LL128) {
      if (graphs[a]->typeInter > PATH_PXR || graphs[a]->typeIntra > PATH_PXB) {
        WARN("Invalid LL128 force use case for algo[%d], type_inter[%d], type_intra[%d]", a, graphs[a]->typeInter, graphs[a]->typeIntra);
        pEnable = 0;
      }
    }
    if (pEnable == 2 && p == NCCL_PROTO_LL128) {
      // Enable LL128 by default only on Volta/Ampere+NVLink. Other cases are not tested and may cause silent data corruption.
      pEnable = (graphs[a]->typeInter <= PATH_PXR) && graphs[a]->typeIntra <= PATH_NVB ? 1 : 0;
    }
    if (usingMemcpy() && (p == NCCL_PROTO_SIGNAL || p == NCCL_PROTO_SIGNAL_BULK)) {
      // disable Signal and SignalBulk when use ce memcpy;
      pEnable = 0;
    }

    if ((c == ncclFuncBroadcast || c == ncclFuncReduce) && (a == NCCL_ALGO_FC && algoEnable[a] == 2)) {
      // disable fc algo of broadcast and reduce by default
      pEnable = 0;
    }
    if (pEnable == 2 && ((a == NCCL_ALGO_FC && p == NCCL_PROTO_LL) || p == NCCL_PROTO_SIMPLE_BULK_AR)) {
      // disable fc+ll and simpleBulkAR by default untill tuning strategy is ready
      pEnable = 0;
    }
    if (pEnable == 1 && p == NCCL_PROTO_SIMPLE_BULK_AR && maxCompCap < 89) {
      WARN("Invalid SimpleBulkAR force use case for sm%d", maxCompCap);
      pEnable = 0;
    }
    if (pEnable == 0) comm->bandwidths[c][a][p] = 0;
    // disable algo
    if (algoEnable[c*NCCL_NUM_ALGORITHMS+a] == 0) comm->bandwidths[c][a][p] = 0;
  }
  // set proto enable for p2p(send/recv) kernel
  int availableP2pProtos = 0;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    int pEnable = protoEnable[ncclFuncSendRecv*NCCL_NUM_PROTOCOLS+p];
    // disable protos depend on icnlink when global p2p through icnlink not supported
    if ((pEnable == ncclTuningEnable) && (comm->globalIcnP2pSupport == 0) && (PROTO_DEPEND_ON_ICN(p))) {
      pEnable = ncclTuningDisable;
    }
    // default to disable bulkar proto unless enabled by env
    if ((pEnable == ncclTuningCustomEnable) && (p == NCCL_PROTO_SIMPLE_BULK_AR)) {
      pEnable = ncclTuningDisable;
    }
    // disable protos not implemented yet
    if ((pEnable != ncclTuningDisable) &&
        (p2pKernelEnabled[p] == 0 ||  p2pKernelEnabled[p] > maxCompCap)) {
      pEnable = ncclTuningDisable;
    }
    comm->p2pProtoEnable[p] = pEnable;
    availableP2pProtos += pEnable;
  }

  if (fallBack) {
    for (int c = 0; c < NCCL_NUM_FUNCTIONS; c++) {
      bool available = false;
      for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++)
        for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++)
          if (comm->bandwidths[c][a][p] != 0) {
            available = true;
            goto check_avail;
          }
    check_avail:
      if (available == false) {
        /* at least set ring algo available */
        for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++)
          comm->bandwidths[c][NCCL_ALGO_RING][p] = comm->ringbdw[c][p];
      }
    }
    if (availableP2pProtos == 0) {
      int pFallback = comm->globalIcnP2pSupport ? NCCL_PROTO_SIMPLE_BULK : NCCL_PROTO_SIMPLE;
      INFO(NCCL_TUNING, "P2P proto set by PCCL_PROTO=%s is not available, fallback to %s for globalIcnP2pSupport %d", protoStr, ncclProtoStr[pFallback], comm->globalIcnP2pSupport);
      comm->p2pProtoEnable[pFallback] = 1;
    }
  }

  if (comm->rank == 0) {
    char line[1024];
    sprintf(line, "Latency/AlgBw/Max Nthreads |");
    for (int c=0; c<NCCL_NUM_FUNCTIONS_WITH_P2P; c++) {
      sprintf(line+strlen(line), "   %-14s |", ncclFuncStr[c]);
    }

    INFO(NCCL_TUNING, "%s", line);
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        sprintf(line, "%7s/%-18s |", ncclAlgoStr[a], ncclProtoStr[p]);
        for (int c=0; c<NCCL_NUM_FUNCTIONS; c++) {
          sprintf(line+strlen(line), " %5.1f/%5.1f/%-4d |", comm->latencies[c][a][p], comm->bandwidths[c][a][p], comm->maxThreads[a][p]);
        }
        sprintf(line+strlen(line), " %-5s %-5d %-4s |", " ", a == NCCL_ALGO_RING ? comm->p2pProtoEnable[p] : 0, " ");
        INFO(NCCL_TUNING, "%s", line);
      }
    }
  }

  // Set per-thread amount of work before we increase nThreads and nChannels
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    comm->threadThresholds[a][NCCL_PROTO_LL] = NCCL_LL_THREAD_THRESHOLD;
    comm->threadThresholds[a][NCCL_PROTO_LL128] = NCCL_LL128_THREAD_THRESHOLD;
    comm->threadThresholds[a][NCCL_PROTO_SIMPLE] = NCCL_SIMPLE_THREAD_THRESHOLD;
    comm->threadThresholds[a][NCCL_PROTO_SIGNAL] = NCCL_SIMPLE_THREAD_THRESHOLD;
    comm->threadThresholds[a][NCCL_PROTO_SIMPLE_BULK] = NCCL_SIMPLE_THREAD_THRESHOLD;
    comm->threadThresholds[a][NCCL_PROTO_SIGNAL_BULK] = NCCL_SIMPLE_THREAD_THRESHOLD;
    comm->threadThresholds[a][NCCL_PROTO_SIMPLE_BULK_AR] = NCCL_SIMPLE_THREAD_THRESHOLD;
  }
  comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL] *= nRanks;

  // Override defaults with user env
  const char* str = pcclGetEnvWithDualPrefix("PCCL_THREAD_THRESHOLDS");
  if (str) {
    INFO(NCCL_ENV, "PCCL_THREAD_THRESHOLDS set by environment to %s", str);
    ssize_t t[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {{ -2, -2, -2, -2, -2, -2 }, { -2, -2, -2, -2, -2, -2}};
    sscanf(str, "%ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld",
        t[0], t[0]+1, t[0]+2, t[0]+3, t[0]+4, t[0]+5, t[1], t[1]+1, t[1]+2, t[1]+3, t[1]+4, t[1]+5);
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        if (t[a][p] >= 0) comm->threadThresholds[a][p] = t[a][p];
      }
    }
  }

  INFO(NCCL_INIT, "threadThresholds %ld/%ld/%ld/%ld/%ld/%ld/%ld | %ld/%ld/%ld/%ld/%ld/%ld/%ld | %ld/%ld/%ld/%ld/%ld/%ld/%ld | %ld/%ld/%ld/%ld/%ld/%ld/%ld",
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL],
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL128],
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE],
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_SIGNAL],
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE_BULK],
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_SIGNAL_BULK],
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE_BULK_AR],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL128],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_SIGNAL],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE_BULK],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_SIGNAL_BULK],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE_BULK_AR],
      comm->threadThresholds[NCCL_ALGO_COLLNET][NCCL_PROTO_LL],
      comm->threadThresholds[NCCL_ALGO_COLLNET][NCCL_PROTO_LL128],
      comm->threadThresholds[NCCL_ALGO_COLLNET][NCCL_PROTO_SIMPLE],
      comm->threadThresholds[NCCL_ALGO_COLLNET][NCCL_PROTO_SIGNAL],
      comm->threadThresholds[NCCL_ALGO_COLLNET][NCCL_PROTO_SIMPLE_BULK],
      comm->threadThresholds[NCCL_ALGO_COLLNET][NCCL_PROTO_SIGNAL_BULK],
      comm->threadThresholds[NCCL_ALGO_COLLNET][NCCL_PROTO_SIMPLE_BULK_AR],
      comm->threadThresholds[NCCL_ALGO_FC][NCCL_PROTO_LL],
      comm->threadThresholds[NCCL_ALGO_FC][NCCL_PROTO_LL128],
      comm->threadThresholds[NCCL_ALGO_FC][NCCL_PROTO_SIMPLE],
      comm->threadThresholds[NCCL_ALGO_FC][NCCL_PROTO_SIGNAL],
      comm->threadThresholds[NCCL_ALGO_FC][NCCL_PROTO_SIMPLE_BULK],
      comm->threadThresholds[NCCL_ALGO_FC][NCCL_PROTO_SIGNAL_BULK],
      comm->threadThresholds[NCCL_ALGO_FC][NCCL_PROTO_SIMPLE_BULK_AR]);
  return ncclSuccess;
}

// Trees are not perfectly sticking to the model for medium sizes. Applying a static correction
// factor is not ideal but works quite well. Powers of two, 64 B to 256MB.
static float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][23] = {
  { 1.0, 1.0, 1.0, 1.0,  .9,  .8,  .7,  .7,  .7,  .7,  .6,  .5,  .4,  .4,  .5,  .6,  .7,  .8,  .9, 1.0, 1.0, 1.0, 1.0 },
  { 1.0, 1.0, 1.0, 1.0, 1.0,  .9,  .8,  .8,  .8,  .7,  .6,  .6,  .6,  .5,  .6,  .6,  .7,  .7,  .8,  .9,  .9, .92, .92 },
  {  .9,  .9,  .9,  .9,  .9,  .9,  .9,  .8,  .7,  .6,  .6,  .5,  .5,  .5,  .5,  .6,  .7,  .8,  .7,  .7,  .8,  .9,  .9 },
  {  .9,  .9,  .9,  .9,  .9,  .9,  .9,  .8,  .7,  .6,  .6,  .5,  .5,  .5,  .5,  .6,  .7,  .8,  .7,  .7,  .8,  .9,  .9 },
  {  .9,  .9,  .9,  .9,  .9,  .9,  .9,  .8,  .7,  .6,  .6,  .5,  .5,  .5,  .5,  .6,  .7,  .8,  .7,  .7,  .8,  .9,  .9 },
  {  .9,  .9,  .9,  .9,  .9,  .9,  .9,  .8,  .7,  .6,  .6,  .5,  .5,  .5,  .5,  .6,  .7,  .8,  .7,  .7,  .8,  .9,  .9 },
  {  .9,  .9,  .9,  .9,  .9,  .9,  .9,  .8,  .7,  .6,  .6,  .5,  .5,  .5,  .5,  .6,  .7,  .8,  .7,  .7,  .8,  .9,  .9 }
};

#define NUM_FACTORS 28
constexpr uint64_t packKey(int coll, int proto) {
  return (static_cast<uint64_t>(coll) << 32) | static_cast<uint32_t>(proto);
}

// When total ranks is 4 and the proto is simple bulk, the steps of ring allgather is 3 while the steps of fc allgather is 1
// In order to use ring, we need set fc factor smaller than 1/3, so we set the simplebulk factor to 0.3.
// the data array is powers of two, 64 B to 16G.
static const std::unordered_map<uint64_t, std::array<float, NUM_FACTORS>> fcCorrectionFactor = {
    //64B, 128B,512B, 1K, 2K,  4K,   8K, 16K, 32K, 64K,128K,256K,512K,  1M,  2M,  4M,  8M, 16M, 32M, 64M,128M,256M,512M,  1G,  2G, 4G,  8G, 16G
  {packKey(ncclFuncAllGather, NCCL_PROTO_SIMPLE),
    { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  .3, 1.0, 1.0,  .5,  .5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }},
  {packKey(ncclFuncReduceScatter, NCCL_PROTO_SIMPLE),
    { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  .3, 1.0, 1.0,  .5,  .5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }},
  {packKey(ncclFuncAllGather, NCCL_PROTO_SIMPLE_BULK),
    {  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3, 1.0, 1.0 }},
  {packKey(ncclFuncReduceScatter, NCCL_PROTO_SIMPLE_BULK),
    {  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3, 1.0 }},
  {packKey(ncclFuncAllReduce, NCCL_PROTO_SIMPLE_BULK),
    {  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3,  .3, 1.0, 1.0, 1.0 }},
};

ncclResult_t ncclTopoGetAlgoTime(ncclComm_t comm, int coll, int algorithm, int protocol, size_t nBytes, int nChannels, float* time) {
  float bw = comm->bandwidths[coll][algorithm][protocol];
  float lat = comm->latencies[coll][algorithm][protocol];
  if (bw == 0) {
    *time = -1.0; return ncclSuccess;
  }

  int logSize = log2i(nBytes>>6);
  if (algorithm == NCCL_ALGO_TREE && logSize < 23) bw *= treeCorrectionFactor[protocol][logSize];
  if (algorithm == NCCL_ALGO_FC && (comm->ppuType == _ZW_M890P || comm->ppuType == _ZW_M890L) && comm->nRanks == 4 && logSize < 28) {
    auto it = fcCorrectionFactor.find(packKey(coll, protocol));
    if (it != fcCorrectionFactor.end()) {
      bw *= it->second[logSize];
    }
  }
  if (nChannels != 0) {
    int nChannels = (algorithm == NCCL_ALGO_FC) ? comm->fcnChannels : comm->nChannels;
    bw = bw / nChannels * nChannels;
  }
  if (algorithm == NCCL_ALGO_RING && (protocol == NCCL_PROTO_SIMPLE || protocol == NCCL_PROTO_SIGNAL
        || protocol == NCCL_PROTO_SIMPLE_BULK || protocol == NCCL_PROTO_SIGNAL_BULK
        || protocol == NCCL_PROTO_SIMPLE_BULK_AR) && comm->nNodes > 1
      && coll == ncclFuncAllReduce && nBytes >= comm->nRanks/16.0*65536) lat *= 1.9; // Plateau effect of ring
  *time = lat + (nBytes) / (1000 * bw);
  if (comm->rank == 0) TRACE(NCCL_TUNING, "Tuning algo %d proto %d nBytes %ld lat %f bw %f time %f\n", algorithm, protocol, nBytes, lat, bw, *time);
  return ncclSuccess;
}
