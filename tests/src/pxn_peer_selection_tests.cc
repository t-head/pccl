// SPDX-FileCopyrightText: Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// Topology regression for level 2 PXN selection; no live communication is launched.
#include "pccl_internal.h"
#include "comm.h"
#include "transport.h"
#include "topo.h"
#include "xml.h"
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#define CHECK(call) do { ncclResult_t rc = (call); if (rc != ncclSuccess) { fprintf(stderr, "line %d: %s returned %d\n", __LINE__, #call, rc); exit(2); } } while (0)
#define REQUIRE(cond) do { if (!(cond)) { fprintf(stderr, "line %d: requirement failed: %s\n", __LINE__, #cond); exit(3); } } while (0)

static std::string list(const std::vector<int>& v) {
  std::string s = "[";
  for (size_t i=0; i<v.size(); i++) { if (i) s += ","; s += std::to_string(v[i]); }
  return s + "]";
}

static std::vector<int> candidates(ncclComm_t comm, int rank, int n, bool verbose) {
  int g;
  CHECK(ncclTopoRankToIndex(comm->topo, rank, &g));
  auto* net = &comm->topo->nodes[NET].nodes[n];
  std::vector<int> c;
  int hops = INT_MAX;
  for (int r=0; r<comm->localRanks; r++) {
    int peer = comm->localRankToRank[r], p;
    if (comm->peerInfo[peer].netId != net->id) continue;
    CHECK(ncclTopoRankToIndex(comm->topo, peer, &p));
    auto* path = &comm->topo->nodes[GPU].nodes[p].paths[GPU][g];
    bool eligible = path->type <= PATH_NVB && net->paths[GPU][p].type <= PATH_PXB;
    if (verbose) printf("PATH src=%d peer=%d peer_to_src=%s hops=%d nic_to_peer=%s eligible=%d\n", rank, peer, topoPathTypeStr[path->type], path->count, topoPathTypeStr[net->paths[GPU][p].type], eligible);
    if (!eligible) continue;
    if (path->count < hops) { hops = path->count; c.clear(); }
    if (path->count == hops) c.push_back(peer);
  }
  int type = comm->topo->nodes[GPU].nodes[g].paths[NET][n].type;
  if (verbose) printf("SOURCE_GATE rank=%d nic=%d path=%s accepted=%d\n", rank, net->net.dev, topoPathTypeStr[type], type <= PATH_PXR);
  if (type > PATH_PXR) c.clear();
  return c;
}

static int layouts = 0, imbalancedGroups = 0, balancedGroups = 0;
static void run(const char* file, const std::vector<int>& devices, bool interleaved, bool verbose) {
  const int size = devices.size();
  std::vector<int> local(size), remote(size), rankToLocal(size*2);
  for (int i=0; i<size; i++) {
    local[i] = interleaved ? 2*i : i;
    remote[i] = interleaved ? 2*i+1 : size+i;
    rankToLocal[local[i]] = rankToLocal[remote[i]] = i;
  }
  ncclXml* xml = nullptr;
  CHECK(xmlAlloc(&xml, NCCL_TOPO_XML_MAX_NODES));
  CHECK(ncclTopoGetXmlFromFile(file, xml));
  for (int i=0; i<xml->maxIndex; i++) {
    auto* node = &xml->nodes[i];
    if (!strcmp(node->name, "cpu")) CHECK(xmlSetAttrLong(node, "host_hash", getHostHash()));
    if (!strcmp(node->name, "net")) CHECK(xmlSetAttrInt(node, "keep", 1));
    if (!strcmp(node->name, "gpu")) {
      int dev;
      CHECK(xmlGetAttrInt(node, "dev", &dev));
      CHECK(xmlSetAttrInt(node, "net_affinity", -1));
      auto it = std::find(devices.begin(), devices.end(), dev);
      CHECK(xmlSetAttrInt(node, "keep", it != devices.end()));
      if (it != devices.end()) CHECK(xmlSetAttrInt(node, "rank", local[it-devices.begin()]));
    }
  }
  CHECK(pcclTopoSetNetAffinityPPUs(xml, 8, 4));
  CHECK(ncclTopoTrimXml(xml));
  ncclTopoSystem* system = nullptr;
  CHECK(ncclTopoGetSystemFromXml(xml, &system, getHostHash()));
  ncclComm_t comm = nullptr;
  CHECK(ncclCalloc(&comm, 1));
  comm->topo = system;
  comm->rank = local[0];
  comm->nRanks = size*2;
  comm->nNodes = 2;
  comm->localRanks = size;
  comm->rankToLocalRank = rankToLocal.data();
  comm->localRankToRank = local.data();
  CHECK(ncclTopoComputePaths(system, comm));
  CHECK(ncclTopoTrimSystem(system, comm));
  CHECK(ncclTopoComputePaths(system, comm));
  CHECK(pcclAffinityNetsPathAdjust(system));
  CHECK(ncclTopoSearchInit(system));
  std::vector<ncclPeerInfo> peers(size*2);
  comm->peerInfo = peers.data();
  for (int i=0; i<size; i++) {
    peers[local[i]].hgmlDev = peers[remote[i]].hgmlDev = devices[i];
    CHECK(ncclTopoGetLocalNet(system, local[i], &peers[local[i]].netId, nullptr));
    peers[remote[i]].netId = peers[local[i]].netId;
    if (verbose) printf("MAP local=%d global=%d device=%d preferred_net=%lld\n", i, local[i], devices[i], (long long)NCCL_TOPO_ID_LOCAL_ID(peers[local[i]].netId));
  }
  for (int targetDev : {0, 2}) {
    auto target = std::find(devices.begin(), devices.end(), targetDev);
    REQUIRE(target != devices.end());
    int targetLocal = target-devices.begin();
    int64_t netId = peers[local[targetLocal]].netId;
    int n;
    CHECK(ncclTopoIdToIndex(system, NET, netId, &n));
    std::map<std::vector<int>, std::vector<int>> groups;
    std::vector<int> actual(size), proposed(size, -1);
    for (int i=0; i<size; i++) {
      comm->rank = local[i]; comm->localRank = i;
      int64_t chosenNet;
      int chosenDev;
      CHECK(ncclTopoGetNetDev(comm, local[i], nullptr, 0, remote[targetLocal], &chosenNet, &chosenDev, &actual[i]));
      auto c = candidates(comm, local[i], n, verbose);
      if (!c.empty()) {
        REQUIRE(chosenNet == netId);
        REQUIRE(std::find(c.begin(), c.end(), actual[i]) != c.end());
        groups[c].push_back(i);
      } else {
        REQUIRE(actual[i] == local[i]);
      }
      if (verbose) printf("ACTUAL devices=%s interleaved=%d nic=%d src_local=%d src_device=%d candidates_global=%s selected_local=%d selected_device=%d\n", list(devices).c_str(), interleaved, system->nodes[NET].nodes[n].net.dev, i, devices[i], list(c).c_str(), rankToLocal[actual[i]], devices[rankToLocal[actual[i]]]);
    }
    for (const auto& group : groups) {
      const auto& c = group.first;
      const auto& sources = group.second;
      std::vector<int> oldCounts(c.size()), newCounts(c.size());
      for (size_t ordinal=0; ordinal<sources.size(); ordinal++) {
        int src = sources[ordinal];
        oldCounts[std::find(c.begin(), c.end(), actual[src])-c.begin()]++;
        // Reference assignment from the independently collected source groups.
        proposed[src] = c[ordinal % c.size()];
        REQUIRE(actual[src] == proposed[src]);
        newCounts[ordinal % c.size()]++;
      }
      int oldSpread = *std::max_element(oldCounts.begin(), oldCounts.end())-*std::min_element(oldCounts.begin(), oldCounts.end());
      int newSpread = *std::max_element(newCounts.begin(), newCounts.end())-*std::min_element(newCounts.begin(), newCounts.end());
      REQUIRE(newSpread <= 1);
      REQUIRE(oldSpread <= 1);
      if (c.size() > 1) {
        balancedGroups++;
        if (oldSpread > 1) imbalancedGroups++;
        printf("GROUP devices=%s interleaved=%d nic=%d candidates_global=%s source_locals=%s actual_counts=%s proposed_counts=%s\n", list(devices).c_str(), interleaved, system->nodes[NET].nodes[n].net.dev, list(c).c_str(), list(sources).c_str(), list(oldCounts).c_str(), list(newCounts).c_str());
      }
    }
    if (devices == std::vector<int>({0,2,1,3}) && targetDev == 0) {
      REQUIRE(actual[1] == local[0] && actual[3] == local[2]);
      REQUIRE(proposed[1] == local[0] && proposed[3] == local[2]);
      printf("COUNTEREXAMPLE_FIXED interleaved=%d actual_peer_locals=[0,2]\n", interleaved);
    }
    // Reversing call order and changing channel must not advance mutable state.
    for (int i=size-1; i>=0; i--) {
      comm->rank = local[i]; comm->localRank = i;
      int repeated;
      CHECK(ncclTopoGetNetDev(comm, local[i], nullptr, 3, remote[targetLocal], nullptr, nullptr, &repeated));
      REQUIRE(repeated == actual[i]);
    }
    if (devices == std::vector<int>({0,2,1,3}) && targetDev == 0 && !interleaved) {
      int firstSource;
      CHECK(ncclTopoRankToIndex(system, local[1], &firstSource));
      auto* sourcePath = &system->nodes[GPU].nodes[firstSource].paths[NET][n];
      int savedType = sourcePath->type;
      sourcePath->type = PATH_SYS;
      int selected;
      comm->rank = local[3]; comm->localRank = 3;
      CHECK(ncclTopoGetNetDev(comm, local[3], nullptr, 0, remote[targetLocal], nullptr, nullptr, &selected));
      REQUIRE(selected == local[0]); // Rejected earlier source must not consume ordinal 0.
      comm->rank = local[1]; comm->localRank = 1;
      CHECK(ncclTopoGetNetDev(comm, local[1], nullptr, 0, remote[targetLocal], nullptr, nullptr, &selected));
      REQUIRE(selected == local[1]);
      sourcePath->type = savedType;
      // Keep target NIC resolution intact, but make all NIC-to-peer paths ineligible.
      std::vector<int> nicTypes(system->nodes[GPU].count);
      for (int g=0; g<system->nodes[GPU].count; g++) {
        nicTypes[g] = system->nodes[NET].nodes[n].paths[GPU][g].type;
        system->nodes[NET].nodes[n].paths[GPU][g].type = PATH_SYS;
      }
      comm->rank = local[3]; comm->localRank = 3;
      CHECK(ncclTopoGetNetDev(comm, local[3], nullptr, 0, remote[targetLocal], nullptr, nullptr, &selected));
      REQUIRE(selected == local[3]);
      for (int g=0; g<system->nodes[GPU].count; g++) system->nodes[NET].nodes[n].paths[GPU][g].type = nicTypes[g];
      printf("FALLBACK_CHECKS rejected_source_excluded=pass source_gate=pass no_candidates=pass\n");
    }
  }
  ncclTopoFree(system);
  CHECK(ncclFree(xml));
  CHECK(ncclFree(comm));
  layouts++;
}

int main(int argc, char** argv) {
  REQUIRE(argc <= 2);
  const char* fixture = argc == 2 ? argv[1] : "topo_and_reference_graph/8PPU1.0_4CX7_2PIXPPUsPerNET_OAM_4Links_PXN/topo.xml";
  setenv("PCCL_IGNORE_TOPO_DETECTION", "1", 1);
  setenv("NCCL_IGNORE_DISABLED_P2P", "2", 1);
  setenv("PCCL_IGNORE_DISABLED_P2P", "2", 1);
  // All other routing parameters, including PXN level 2, use library defaults.
  std::vector<int> permutation = {0,1,2,3};
  do {
    bool verbose = permutation == std::vector<int>({0,2,1,3}) || permutation == std::vector<int>({0,1,2,3});
    run(fixture, permutation, false, verbose);
    run(fixture, permutation, true, false);
  } while (std::next_permutation(permutation.begin(), permutation.end()));
  run(fixture, {0,1,2,3,4,5,6,7}, false, false);
  run(fixture, {0,2,1,3,4,6,5,7}, false, false);
  printf("SUMMARY layouts=%d tied_groups=%d current_imbalanced_groups=%d all_group_spreads_le_1=1\n", layouts, balancedGroups, imbalancedGroups);
  REQUIRE(layouts == 50 && balancedGroups == 100 && imbalancedGroups == 0);
  return 0;
}
