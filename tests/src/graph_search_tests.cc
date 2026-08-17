//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include "include/argparse.hpp"
#include "include/file_compare.h"
#include "pccl_internal.h"
#include "topo.h"
#include "xml.h"
#include "globals.h"
#include <fcntl.h>
#include <string.h>
#include <gtest/gtest.h>

enum AlgoMode {
  ALGO_AUTO = 0,
  ALGO_SPECIFIED = 1,
};

enum NodeCount {
  ONE_NODE = 1,
  TWO_NODES = 2,
};

#define TEST_CASE(testClass, testName, topoDir, refGraph, nNodes, algoMode, ppus, ...) \
TEST(testClass, testName) { \
  const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();  \
  std::string testCaseName = std::string(test_info->name()) + "." + std::string(test_info->test_case_name()); \
  std::string topoPath = globalTopoAndRefDir + topoDir; \
  std::string topoFile = topoPath + "topo.xml"; \
  std::string graphDumpFile = "/tmp/" + testCaseName + "." + "graph.xml"; \
  std::vector ppuList = {ppus, ##__VA_ARGS__}; \
  ncclResult_t res = testGraphSearch(ppuList, topoFile.c_str(), graphDumpFile.c_str(), nNodes, algoMode); \
  EXPECT_EQ(res, ncclSuccess); \
  std::string refFile = topoPath + refGraph;  \
  bool areEqual = CompareFiles(graphDumpFile.c_str(), refFile.c_str()); \
  EXPECT_TRUE(areEqual); \
}

#define TEST_TEMP(testName, topoDir, refGraph, nNodes, algoMode, ppus, ...) TEST_CASE(graphSearchTestWithoutPXN, testName, topoDir, refGraph, nNodes, algoMode, ppus, __VA_ARGS__)

std::string globalTopoAndRefDir;

struct graphSearchTestArgs : public argparse::Args {
  std::string &root= kwarg("r", "root dir for topo file and reference graph").set_default("./topo_and_reference_graph/");
  bool &verbose = flag("v,verbose", "A flag to toggle verbose").set_default(false);
  bool &pxn = flag("p,pxn", "A flag to control pxn enable or disable").set_default(false);
};

ncclResult_t setEnvs(const char *topoFile, const char *graphFile) {
  if (topoFile != NULL) {
    setenv("NCCL_TOPO_FILE", topoFile, 1);
    INFO(NCCL_INIT, "Override NCCL_TOPO_FILE by %s", topoFile);
  }

  if (graphFile != NULL) {
    setenv("NCCL_GRAPH_DUMP_FILE", graphFile, 1);
    INFO(NCCL_INIT, "Override NCCL_GRAPH_DUMP_FILE by %s", graphFile);
  }

  setenv("PCCL_IGNORE_TOPO_DETECTION", "1", 1);

  char *xmlTopoFile = getenv("NCCL_TOPO_FILE");
  if (xmlTopoFile) {
    INFO(NCCL_INIT, "NCCL_TOPO_FILE set by environment to %s", xmlTopoFile);
  } else {
    ERROR("Error, no NCCL_TOPO_FILE set");
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

static ncclResult_t ncclTopoGetSubNodes(struct ncclXmlNode* node, const char* subName, int* index, struct ncclXmlNode** nodes) {
  const char* name = NULL;
  name = node->name;
  if (name && strcmp(name, subName) == 0) {
    nodes[*index] = node;
    *index += 1;
  } else {
    // Copy nSubs and subs as they could change as we trim recursively.
    struct ncclXmlNode* subs[MAX_SUBS];
    int nSubs = node->nSubs;
    memcpy(subs, node->subs, node->nSubs*sizeof(struct ncclXmlNode*));
    for (int s=0; s<nSubs; s++) {
      NCCLCHECK(ncclTopoGetSubNodes(subs[s], subName, index, nodes));
    }
  }
  return ncclSuccess;
}

static int ncclTopoGraphTreePattern(ncclComm_t comm, ncclTopoSystem *system) {
  int pattern = 0;
  if (pattern > NCCL_TOPO_PATTERN_TREE || pattern < NCCL_TOPO_PATTERN_BALANCED_TREE) {
    pattern = comm->nNodes <= 2 ? NCCL_TOPO_PATTERN_TREE : NCCL_TOPO_PATTERN_BALANCED_TREE;

    // for 8+ CX6, go simple tree.for now
    int netCount = 0;
    ncclTopoGetNetCount(system, &netCount);
    pattern = netCount >= 8 ? NCCL_TOPO_PATTERN_TREE : NCCL_TOPO_PATTERN_BALANCED_TREE;
  }

  return pattern;
}

ncclResult_t testGraphSearch(std::vector<int> ppuList, const char *topoFile, const char *graphFile, int nNodes, AlgoMode algoMode) {
  NCCLCHECK(setEnvs(topoFile, graphFile));

  ncclTopoSystem *system;
  struct ncclXml *xml;
  bool isAlgoSpecified = algoMode == ALGO_SPECIFIED;
  NCCLCHECK(xmlAlloc(&xml, NCCL_TOPO_XML_MAX_NODES));

  char *xmlTopoFile = getenv("NCCL_TOPO_FILE");
  NCCLCHECK(ncclTopoGetXmlFromFile(xmlTopoFile, xml));

  struct ncclXmlNode* cpuNode = NULL;
  NCCLCHECK(xmlFindTag(xml, "cpu", &cpuNode));
  while (cpuNode != nullptr) {
    NCCLCHECK(xmlSetAttrLong(cpuNode, "host_hash", getHostHash()));
    NCCLCHECK(xmlFindNextTag(xml, "cpu", cpuNode, &cpuNode));
  }

  struct ncclXmlNode** ppuNodes = NULL;
  struct ncclXmlNode** netNodes = NULL;
  NCCLCHECK(ncclCalloc(&ppuNodes, 8*sizeof(struct ncclXmlNode)));
  NCCLCHECK(ncclCalloc(&netNodes, 256*sizeof(struct ncclXmlNode)));
  int ppuCount = 0;
  int netCount = 0;
  ncclTopoGetSubNodes(xml->nodes, "gpu", &ppuCount, ppuNodes);
  ncclTopoGetSubNodes(xml->nodes, "net", &netCount, netNodes);

  int nRanks = ppuList.size()*nNodes;
  for (auto p=0; p<ppuList.size(); p++) {
    NCCLCHECK(xmlSetAttrInt(ppuNodes[ppuList[p]], "keep", 1));
    NCCLCHECK(xmlSetAttrInt(ppuNodes[ppuList[p]], "net_affinity", -1));
  }

  assert(ppuCount <= 16);

  for (int n=0; n<netCount; n++) {
    NCCLCHECK(xmlSetAttrInt(netNodes[n], "keep", 1));
  }

  NCCLCHECK(pcclTopoSetNetAffinityPPUs(xml, ppuCount, netCount));
  NCCLCHECK(ncclTopoTrimXml(xml));

  NCCLCHECK(ncclTopoGetSystemFromXml(xml, &system, getHostHash()));
  ncclComm_t comm;
  NCCLCHECK(ncclCalloc(&comm, 1));
  comm->rank = 0;
  comm->nRanks = nRanks;
  comm->nNodes = nNodes;
  comm->localRanks = (comm->nRanks > system->nodes[GPU].count) ? system->nodes[GPU].count : comm->nRanks;

  NCCLCHECK(ncclTopoComputePaths(system, comm));
  NCCLCHECK(ncclTopoTrimSystem(system, comm));
  NCCLCHECK(ncclTopoComputePaths(system, comm));
  NCCLCHECK(pcclAffinityNetsPathAdjust(system));
  NCCLCHECK(ncclTopoSearchInit(system));
  NCCLCHECK(ncclTopoPrint(comm, system));

  ncclTopoGraph ringGraph;
  ringGraph.id = 0;
  ringGraph.pattern = NCCL_TOPO_PATTERN_RING;
  ringGraph.crossNic = ncclParamCrossNic();
  ringGraph.collNet = 0;
  ringGraph.minChannels = 1;
  ringGraph.maxChannels = MAXCHANNELS/2;
  NCCLCHECK(ncclTopoCompute(system, &ringGraph));
  NCCLCHECK(ncclTopoPrintGraph(system, &ringGraph));

  ncclTopoGraph treeGraph;
  treeGraph.id = 1;
  treeGraph.pattern = isAlgoSpecified ? NCCL_TOPO_PATTERN_TREE : ncclTopoGraphTreePattern(comm, system);
  treeGraph.crossNic = ncclParamCrossNic();
  treeGraph.collNet = 0;
  treeGraph.minChannels = isAlgoSpecified ? 1 : ringGraph.nChannels;
  treeGraph.maxChannels = isAlgoSpecified ? MAXCHANNELS/2 : ringGraph.nChannels;
  NCCLCHECK(ncclTopoCompute(system, &treeGraph));
  NCCLCHECK(ncclTopoPrintGraph(system, &treeGraph));

  struct ncclTopoGraph *graphs[] = {&ringGraph, &treeGraph};
  NCCLCHECK(ncclTopoDumpGraphs(system, 2, graphs));

  ncclFree(xml);
  ncclFree(ppuNodes);
  ncclFree(netNodes);
  ncclFree(comm);
  return ncclSuccess;
}

// 8PPU 4CX7
TEST_TEMP(8PPU4CX7Tp1GraphSearchTest, "./8PPU1.0_4CX7_OAM_4Links/", "graph_tp1.xml", TWO_NODES, ALGO_SPECIFIED, 0,1,2,3,4,5,6,7)

TEST_TEMP(8PPU4CX7Tp2Sub0GraphSearchTest, "./8PPU1.0_4CX7_OAM_4Links/", "graph_tp2_0167.xml", TWO_NODES, ALGO_SPECIFIED, 0,1,6,7)
TEST_TEMP(8PPU4CX7Tp2Sub1GraphSearchTest, "./8PPU1.0_4CX7_OAM_4Links/", "graph_tp2_2345.xml", TWO_NODES, ALGO_SPECIFIED, 2,3,4,5)

TEST_TEMP(8PPU4CX7Tp4Sub0GraphSearchTest, "./8PPU1.0_4CX7_OAM_4Links/", "graph_tp4_06.xml", TWO_NODES, ALGO_SPECIFIED, 0,6)
TEST_TEMP(8PPU4CX7Tp4Sub1GraphSearchTest, "./8PPU1.0_4CX7_OAM_4Links/", "graph_tp4_17.xml", TWO_NODES, ALGO_SPECIFIED, 1,7)
TEST_TEMP(8PPU4CX7Tp4Sub2GraphSearchTest, "./8PPU1.0_4CX7_OAM_4Links/", "graph_tp4_24.xml", TWO_NODES, ALGO_SPECIFIED, 2,4)
TEST_TEMP(8PPU4CX7Tp4Sub3GraphSearchTest, "./8PPU1.0_4CX7_OAM_4Links/", "graph_tp4_35.xml", TWO_NODES, ALGO_SPECIFIED, 3,5)

TEST_TEMP(8PPU4CX7Tp8Sub0GraphSearchTest, "./8PPU1.0_4CX7_OAM_4Links/", "graph_tp8_0.xml", TWO_NODES, ALGO_SPECIFIED, 0)
TEST_TEMP(8PPU4CX7Tp8Sub1GraphSearchTest, "./8PPU1.0_4CX7_OAM_4Links/", "graph_tp8_1.xml", TWO_NODES, ALGO_SPECIFIED, 1)
TEST_TEMP(8PPU4CX7Tp8Sub2GraphSearchTest, "./8PPU1.0_4CX7_OAM_4Links/", "graph_tp8_2.xml", TWO_NODES, ALGO_SPECIFIED, 2)
TEST_TEMP(8PPU4CX7Tp8Sub3GraphSearchTest, "./8PPU1.0_4CX7_OAM_4Links/", "graph_tp8_3.xml", TWO_NODES, ALGO_SPECIFIED, 3)
TEST_TEMP(8PPU4CX7Tp8Sub4GraphSearchTest, "./8PPU1.0_4CX7_OAM_4Links/", "graph_tp8_4.xml", TWO_NODES, ALGO_SPECIFIED, 4)
TEST_TEMP(8PPU4CX7Tp8Sub5GraphSearchTest, "./8PPU1.0_4CX7_OAM_4Links/", "graph_tp8_5.xml", TWO_NODES, ALGO_SPECIFIED, 5)
TEST_TEMP(8PPU4CX7Tp8Sub6GraphSearchTest, "./8PPU1.0_4CX7_OAM_4Links/", "graph_tp8_6.xml", TWO_NODES, ALGO_SPECIFIED, 6)
TEST_TEMP(8PPU4CX7Tp8Sub7GraphSearchTest, "./8PPU1.0_4CX7_OAM_4Links/", "graph_tp8_7.xml", TWO_NODES, ALGO_SPECIFIED, 7)

// 8PPU 4CX7
TEST_TEMP(8PPU4CX6Tp1GraphSearchTest, "./8PPU1.0_4CX6_OAM_4Links_PIX/", "graph_tp1.xml", TWO_NODES, ALGO_SPECIFIED, 0,1,2,3,4,5,6,7)

TEST_TEMP(8PPU4CX6Tp2Sub0GraphSearchTest, "./8PPU1.0_4CX6_OAM_4Links_PIX/", "graph_tp2_0167.xml", TWO_NODES, ALGO_SPECIFIED, 0,1,6,7)
// TEST_TEMP(8PPU4CX6Tp2Sub1GraphSearchTest, "./8PPU1.0_4CX6_OAM_4Links_PIX/", "graph_tp2_2345.xml", 2,3,4,5)

TEST_TEMP(8PPU4CX6Tp4Sub0GraphSearchTest, "./8PPU1.0_4CX6_OAM_4Links_PIX/", "graph_tp4_07.xml", TWO_NODES, ALGO_SPECIFIED, 0,7)
// TEST_TEMP(8PPU4CX6Tp4Sub1GraphSearchTest, "./8PPU1.0_4CX6_OAM_4Links_PIX/", "graph_tp4_17.xml", 1,7)
// TEST_TEMP(8PPU4CX6Tp4Sub2GraphSearchTest, "./8PPU1.0_4CX6_OAM_4Links_PIX/", "graph_tp4_24.xml", 2,4)
// TEST_TEMP(8PPU4CX6Tp4Sub3GraphSearchTest, "./8PPU1.0_4CX6_OAM_4Links_PIX/", "graph_tp4_35.xml", 3,5)

// TEST_TEMP(8PPU4CX6Tp8Sub0GraphSearchTest, "./8PPU1.0_4CX6_OAM_4Links_PIX/", "graph_tp8_0.xml", 0)
// TEST_TEMP(8PPU4CX6Tp8Sub1GraphSearchTest, "./8PPU1.0_4CX6_OAM_4Links_PIX/", "graph_tp8_1.xml", 1)
// TEST_TEMP(8PPU4CX6Tp8Sub2GraphSearchTest, "./8PPU1.0_4CX6_OAM_4Links_PIX/", "graph_tp8_2.xml", 2)
// TEST_TEMP(8PPU4CX6Tp8Sub3GraphSearchTest, "./8PPU1.0_4CX6_OAM_4Links_PIX/", "graph_tp8_3.xml", 3)
// TEST_TEMP(8PPU4CX6Tp8Sub4GraphSearchTest, "./8PPU1.0_4CX6_OAM_4Links_PIX/", "graph_tp8_4.xml", 4)
// TEST_TEMP(8PPU4CX6Tp8Sub5GraphSearchTest, "./8PPU1.0_4CX6_OAM_4Links_PIX/", "graph_tp8_5.xml", 5)
TEST_TEMP(8PPU4CX6Tp8Sub6GraphSearchTest, "./8PPU1.0_4CX6_OAM_4Links_PIX/", "graph_tp8_6.xml", TWO_NODES, ALGO_SPECIFIED, 6)
// TEST_TEMP(8PPU4CX6Tp8Sub7GraphSearchTest, "./8PPU1.0_4CX6_OAM_4Links_PIX/", "graph_tp8_7.xml", 7)

// 8PPU 4CX6

// 8PPU 8CX6 with 2 PPUs & 2 NETs per PCIe switch
TEST_TEMP(8PPU8CX62PPUsAnd2NETsPerPCIeSwitchGraphSearchTest, "./8PPU1.0_8CX6_4PIX_OAM_4Links/", "graph.xml", TWO_NODES, ALGO_SPECIFIED, 0,1,2,3,4,5,6,7)

TEST_CASE(graphSearchTestWithPXN, PXN8PPUs8CX6, "./8PPU1.0_8CX6_OAM_4Links_PXN/", "graph.xml", TWO_NODES, ALGO_SPECIFIED, 0,1,2,3,4,5,6,7)
TEST_CASE(graphSearchTestWithPXN, PXN8PPUs4BF3, "./8PPU1.0_4BF3_OAM_4Links_PXN/", "graph.xml", TWO_NODES, ALGO_SPECIFIED, 0,1,2,3,4,5,6,7)

TEST_CASE(graphSearchTestWithPXN, PXN16PPUs4BF3, "./16PPU1.1_4BF3_7Links_PXN/", "graph.xml", TWO_NODES, ALGO_SPECIFIED, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
TEST_TEMP(16PPUs4BF3SameNICPIX, "./16PPU1.1_4BF3_7Links_PIX/", "graph.xml", TWO_NODES, ALGO_SPECIFIED, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
TEST_CASE(graphSearchTestWithPXN, PXN8PPUs2PPUsPerNET, "./8PPU1.0_4CX7_2PIXPPUsPerNET_OAM_4Links_PXN/", "graph.xml", TWO_NODES, ALGO_SPECIFIED, 0,1,2,3,4,5,6,7)

// 8PPU 890L
// Single node, distributed across eight NUMA nodes.
TEST_CASE(graphSearchTestWithPXN, 8PPUs890LSingleNode8Numa, "./8PPU1.5_890L_SingleNode_8_Links_8Numa/", "graph.xml", ONE_NODE, ALGO_AUTO, 0,1,2,3,4,5,6,7)

// 8PPU 890L 4CX7
TEST_CASE(graphSearchTestWithPXN, PXN8PPUs890L4CX7, "./8PPU1.5_4CX7_2PIXPPUsPerNET_OAM_8Links_PXN/", "graph.xml", TWO_NODES, ALGO_AUTO, 0,1,2,3,4,5,6,7)

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto args = argparse::parse<graphSearchTestArgs>(argc, argv);
  if (args.verbose) {
    setenv("NCCL_DEBUG", "TRACE", 1);
    setenv("NCCL_DEBUG_SUBSYS", "ALL", 1);
  }

  // overwrite default enable setting
  if (!args.pxn) {
    setenv("NCCL_PXN_DISABLE", "1", 1);
    setenv("PCCL_DISABLE_ADJUST_SYSTEM", "0", 1);
    setenv("PCCL_GRAPH_SEARCH_START_INTER_TYPE", std::to_string(PATH_PIX).c_str(), 1);
  }

  globalTopoAndRefDir = args.root;
  return RUN_ALL_TESTS();
}
