//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include "hggc_runtime.h"
#include "pccl.h"
#include "comm.h"
#include "symmetric.h"
#include "ce_coll.h"
#include "ce_coll_oki.h"

#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static thread_local int rankId = -1;
static int nRanks;
static pthread_barrier_t rankBarrier;

[[noreturn]] static void failTest(const char* expression, const char* file, int line) {
  fprintf(stderr, "rank %d %s:%d: %s\n", rankId, file, line, expression);
  fflush(stderr);
  // Terminate every rank thread without running concurrent process exit handlers.
  _exit(EXIT_FAILURE);
}

#define TEST_CHECK(condition) do {                                            \
  if (!(condition)) {                                                         \
    failTest(#condition, __FILE__, __LINE__);                                 \
  }                                                                           \
} while (0)

// Internal PCCL headers provide a returning HGGCCHECK. This standalone test
// must terminate the process on failure so rank threads do not remain in a collective.
#undef HGGCCHECK
#define HGGCCHECK(cmd) do {                                                   \
  hggcError_t error = (cmd);                                                  \
  if (error != hggcSuccess) {                                                 \
    fprintf(stderr, "HGGC: %s\n", hggcGetErrorString(error));                 \
    failTest(#cmd, __FILE__, __LINE__);                                       \
  }                                                                           \
} while (0)

#define PCCLCHECK(cmd) do {                                                   \
  pcclResult_t result = (cmd);                                                \
  if (result != pcclSuccess) {                                                \
    fprintf(stderr, "PCCL: %s\n", pcclGetErrorString(result));                \
    failTest(#cmd, __FILE__, __LINE__);                                       \
  }                                                                           \
} while (0)

#undef PTHREADCHECK
#define PTHREADCHECK(cmd) do {                                                \
  int result = (cmd);                                                         \
  if (result != 0) {                                                          \
    fprintf(stderr, "pthread error: %s\n", strerror(result));                 \
    failTest(#cmd, __FILE__, __LINE__);                                       \
  }                                                                           \
} while (0)

static void synchronizeRanks() {
  int result = pthread_barrier_wait(&rankBarrier);
  TEST_CHECK(result == 0 || result == PTHREAD_BARRIER_SERIAL_THREAD);
}

enum CollectiveKind {
  AllReduce,
  ReduceScatter,
  AllGather,
  CollectiveCount
};

struct TestOp {
  CollectiveKind kind;
  bool inPlace;
  bool useAvg;
};

struct TestGraph {
  hggcGraph_t graph;
  hggcGraphExec_t exec;
};

// Interpose the actual launchers to reject a passing test that silently used another backend.
static thread_local unsigned int launchCounts[CollectiveCount];
using CeLauncher = ncclResult_t (*)(ncclComm_t, ncclCeCollArgs*, hggcStream_t);

static ncclResult_t callCeLauncher(CeLauncher wrapper, CollectiveKind kind, ncclComm_t comm,
                                   ncclCeCollArgs* args, hggcStream_t stream) {
  Dl_info symbolInfo = {};
  TEST_CHECK(dladdr(reinterpret_cast<void*>(wrapper), &symbolInfo) && symbolInfo.dli_sname);
  CeLauncher launcher = reinterpret_cast<CeLauncher>(dlsym(RTLD_NEXT, symbolInfo.dli_sname));
  TEST_CHECK(launcher);
  ++launchCounts[kind];
  return launcher(comm, args, stream);
}

ncclResult_t pcclCeOkiAllReduce(ncclComm_t comm, ncclCeCollArgs* args, hggcStream_t stream) {
  return callCeLauncher(&pcclCeOkiAllReduce, AllReduce, comm, args, stream);
}

ncclResult_t pcclCeOkiReduceScatter(ncclComm_t comm, ncclCeCollArgs* args, hggcStream_t stream) {
  return callCeLauncher(&pcclCeOkiReduceScatter, ReduceScatter, comm, args, stream);
}

ncclResult_t ncclCeAllGather(ncclComm_t comm, ncclCeCollArgs* args, hggcStream_t stream) {
  return callCeLauncher(&ncclCeAllGather, AllGather, comm, args, stream);
}

// Change every replay's data and include an index-dependent pattern: a stale CQ
// or mailbox that lets a replay finish early must not pass by returning old data.
static float inputValue(int rank, size_t index, int epoch) {
  return float((epoch + 3 * rank + index % 8) % 16) * nRanks;
}

struct CeOkiMixedTest {
  pcclComm_t comm;
  hggcStream_t streams[2];
  pcclWindow_t sendWin;
  pcclWindow_t recvWin;
  float* sendBuff;
  float* recvBuff;
  size_t count;
  size_t strideElements;
  size_t numSlots = 4;
  std::vector<float> hostBuff;

  CeOkiMixedTest(size_t elementCount, const pcclUniqueId& commId) : count(elementCount) {
    // Align every op's region even for partial-count cases.
    strideElements = ((count * nRanks + 127) / 128) * 128;
    // Match the application's runtime setup: initialize the device and buffers
    // before communicator creation (the SDK OKI layer consumes this context).
    HGGCCHECK(hggcSetDevice(rankId));
    for (hggcStream_t& stream : streams) {
      HGGCCHECK(hggcStreamCreateWithFlags(&stream, hggcStreamNonBlocking));
      // Capture cannot set hints; bind the fresh stream before its first captured enqueue.
      PCCLCHECK(pcclCeHintMemcpyStream(stream, 0));
    }
    PCCLCHECK(pcclMemAlloc((void**)&sendBuff, numSlots * strideElements * sizeof(float)));
    PCCLCHECK(pcclMemAlloc((void**)&recvBuff, numSlots * strideElements * sizeof(float)));
    // Match the former rank rendezvous after each device's buffers are ready.
    synchronizeRanks();
    PCCLCHECK(pcclGroupStart());
    PCCLCHECK(pcclCommInitRank(&comm, nRanks, commId, rankId));
    PCCLCHECK(pcclGroupEnd());
    TEST_CHECK(comm->okiSupport);
    TEST_CHECK(comm->ceStreams && comm->ceEvents && comm->ceSyncEvent && comm->ceStreamsNum > 0);
    TEST_CHECK(comm->ceCollConfigs.agOneStageStrategy);
    PCCLCHECK(pcclCommWindowRegister(comm, sendBuff, numSlots * strideElements * sizeof(float),
                                    &sendWin, PCCL_WIN_COLL_SYMMETRIC));
    PCCLCHECK(pcclCommWindowRegister(comm, recvBuff, numSlots * strideElements * sizeof(float),
                                    &recvWin, PCCL_WIN_COLL_SYMMETRIC));
    hostBuff.resize(strideElements);
  }

  uint64_t readOkiEpoch() {
    uint64_t epoch;
    ncclCeGridSyncMem* gridSync = (ncclCeGridSyncMem*)ncclSymDevBase_getGridSyncBase(
        (ncclSymDevBase*)comm->ceColl.ceSyncBase, nRanks);
    HGGCCHECK(hggcMemcpy(&epoch, &gridSync->okiStep, sizeof(epoch), hggcMemcpyDeviceToHost));
    return epoch;
  }

  void prepare(const std::vector<TestOp>& ops, int epoch) {
    TEST_CHECK(ops.size() <= numSlots);
    HGGCCHECK(hggcMemset(recvBuff, 0xff, numSlots * strideElements * sizeof(float)));
    for (size_t slot = 0; slot < ops.size(); ++slot) {
      const TestOp& op = ops[slot];
      for (size_t i = 0; i < strideElements; ++i) {
        hostBuff[i] = inputValue(rankId, i, epoch);
      }
      HGGCCHECK(hggcMemcpy(sendBuff + slot * strideElements, hostBuff.data(),
                          strideElements * sizeof(float), hggcMemcpyHostToDevice));
      if (op.inPlace && op.kind != ReduceScatter) {
        size_t offset = op.kind == AllGather ? rankId * count : 0;
        HGGCCHECK(hggcMemcpy(recvBuff + slot * strideElements + offset, hostBuff.data(),
                            count * sizeof(float), hggcMemcpyHostToDevice));
      }
    }
    synchronizeRanks();
  }

  void enqueue(const std::vector<TestOp>& ops, hggcStream_t stream) {
    unsigned int previousLaunches[CollectiveCount];
    unsigned int expectedLaunches[CollectiveCount] = {};
    std::memcpy(previousLaunches, launchCounts, sizeof(previousLaunches));
    for (size_t slot = 0; slot < ops.size(); ++slot) {
      const TestOp& op = ops[slot];
      ++expectedLaunches[op.kind];
      float* src = sendBuff + slot * strideElements;
      float* dst = recvBuff + slot * strideElements;
      if (op.inPlace) {
        if (op.kind == ReduceScatter) {
          dst = src + rankId * count;
        } else {
          src = dst + (op.kind == AllGather ? rankId * count : 0);
        }
      }
      pcclRedOp_t redOp = op.useAvg ? pcclAvg : pcclSum;
      switch (op.kind) {
        case AllReduce:
          PCCLCHECK(pcclAllReduce(src, dst, count, pcclFloat32, redOp, comm, stream));
          break;
        case ReduceScatter:
          PCCLCHECK(pcclReduceScatter(src, dst, count, pcclFloat32, redOp, comm, stream));
          break;
        case AllGather:
          PCCLCHECK(pcclAllGather(src, dst, count, pcclFloat32, comm, stream));
          break;
        default:
          failTest("unknown collective kind", __FILE__, __LINE__);
      }
    }
    for (int i = 0; i < CollectiveCount; ++i) {
      TEST_CHECK(launchCounts[i] == previousLaunches[i] + expectedLaunches[i]);
    }
  }

  void verify(const std::vector<TestOp>& ops, int epoch, hggcStream_t stream) {
    HGGCCHECK(hggcStreamSynchronize(stream));
    for (size_t slot = 0; slot < ops.size(); ++slot) {
      const TestOp& op = ops[slot];
      size_t numElements = op.kind == AllGather ? count * nRanks : count;
      float* data = (op.inPlace && op.kind == ReduceScatter)
          ? sendBuff + slot * strideElements + rankId * count : recvBuff + slot * strideElements;
      HGGCCHECK(hggcMemcpy(hostBuff.data(), data, numElements * sizeof(float), hggcMemcpyDeviceToHost));
      for (size_t i = 0; i < numElements; ++i) {
        float expected = 0;
        if (op.kind == AllGather) {
          expected = inputValue(i / count, i % count, epoch);
        } else {
          size_t index = op.kind == ReduceScatter ? rankId * count + i : i;
          for (int rank = 0; rank < nRanks; ++rank) {
            expected += inputValue(rank, index, epoch);
          }
          if (op.useAvg) {
            expected /= nRanks;
          }
        }
        if (!std::isfinite(hostBuff[i]) || hostBuff[i] != expected) {
          fprintf(stderr, "rank=%d kind=%d inplace=%d avg=%d epoch=%d index=%zu got=%g expected=%g\n",
                  rankId, op.kind, op.inPlace, op.useAvg, epoch, i, hostBuff[i], expected);
          failTest("incorrect result", __FILE__, __LINE__);
        }
      }
    }
    synchronizeRanks();
  }

  TestGraph capture(const std::vector<TestOp>& ops, hggcStream_t stream) {
    uint64_t previousEpoch = readOkiEpoch();
    TestGraph result;
    HGGCCHECK(hggcStreamBeginCapture(stream, hggcStreamCaptureModeThreadLocal));
    enqueue(ops, stream);
    HGGCCHECK(hggcStreamEndCapture(stream, &result.graph));
    HGGCCHECK(hggcGraphInstantiate(&result.exec, result.graph, nullptr, nullptr, 0));
    TEST_CHECK(readOkiEpoch() == previousEpoch);  // Capture must not reserve execution epochs.
    return result;
  }

  void execute(const std::vector<TestOp>& ops, int epoch, hggcStream_t stream, TestGraph* graph = nullptr) {
    if (rankId == 0 && std::getenv("OKI_GRAPH_TEST_VERBOSE")) {
      printf("execute epoch=%d graph=%d count=%zu ops=", epoch, graph != nullptr, count);
      for (const TestOp& op : ops) {
        printf(" %d/%d/%d", op.kind, op.inPlace, op.useAvg);
      }
      printf("\n");
      fflush(stdout);
    }
    prepare(ops, epoch);
    uint64_t previousEpoch = readOkiEpoch();
    if (graph) {
      HGGCCHECK(hggcGraphLaunch(graph->exec, stream));
    } else {
      enqueue(ops, stream);
    }
    verify(ops, epoch, stream);
    uint64_t expectedDelta = 0;
    for (const TestOp& op : ops) {
      if (op.kind == AllGather) {
        continue;
      }
      size_t rankElements = count;
      if (op.kind == AllReduce) {
        size_t alignElements = 512 / sizeof(float);
        rankElements = ((count + nRanks - 1) / nRanks + alignElements - 1) / alignElements * alignElements;
        rankElements = std::min(rankElements, count);
      }
      size_t chunkElements = pcclCeReduceChunkSize / sizeof(float);
      expectedDelta += (rankElements + chunkElements - 1) / chunkElements + 1;
    }
    TEST_CHECK(readOkiEpoch() == previousEpoch + expectedDelta);
  }

  static void destroyGraph(TestGraph& graph) {
    HGGCCHECK(hggcGraphExecDestroy(graph.exec));
    HGGCCHECK(hggcGraphDestroy(graph.graph));
  }

  ~CeOkiMixedTest() {
    PCCLCHECK(pcclCommWindowDeregister(comm, sendWin));
    PCCLCHECK(pcclCommWindowDeregister(comm, recvWin));
    PCCLCHECK(pcclCommDestroy(comm));
    PCCLCHECK(pcclMemFree(sendBuff));
    PCCLCHECK(pcclMemFree(recvBuff));
    for (hggcStream_t stream : streams) {
      HGGCCHECK(hggcStreamDestroy(stream));
    }
  }
};

struct TestConfig {
  size_t count;
  const char* order;
  const char* execution;
  const char* layout;
  pcclUniqueId commId;
  std::vector<TestOp> coldOps;
};

struct RankThread {
  pthread_t thread;
  int rank;
  const TestConfig* config;
};

static void* runRank(void* arg) {
  RankThread* rankThread = static_cast<RankThread*>(arg);
  rankId = rankThread->rank;
  const TestConfig& config = *rankThread->config;
  {
    CeOkiMixedTest test(config.count, config.commId);
    int epoch = 1;
    // Preserve cold-start coverage; do not warm up collectives before this sequence.
    if (!strcmp(config.execution, "graph")) {
      TestGraph graph = test.capture(config.coldOps, test.streams[0]);
      test.execute(config.coldOps, epoch++, test.streams[0], &graph);
      CeOkiMixedTest::destroyGraph(graph);
    } else {
      test.execute(config.coldOps, epoch++, test.streams[0]);
    }
    if (rankId == 0) {
      printf("COLD_PASS order=%s execution=%s layout=%s ranks=%d count=%zu\n",
             config.order, config.execution, config.layout, nRanks, config.count);
      fflush(stdout);
    }

    for (CollectiveKind kind : {AllReduce, ReduceScatter}) {
      for (bool inPlace : {false, true}) {
        std::vector<TestOp> forwardOps = {{kind, inPlace, false}, {AllGather, inPlace, false}};
        std::vector<TestOp> reverseOps = {{AllGather, inPlace, false}, {kind, inPlace, true}};
        TestGraph forwardGraph = test.capture(forwardOps, test.streams[0]);
        TestGraph reverseGraph = test.capture(reverseOps, test.streams[1]);
        for (int replay = 0; replay < 3; ++replay) {
          test.execute(reverseOps, epoch++, test.streams[1], &reverseGraph);
          test.execute(forwardOps, epoch++, test.streams[0], &forwardGraph);
          test.execute(forwardOps, epoch++, test.streams[1]);
        }
        CeOkiMixedTest::destroyGraph(forwardGraph);
        CeOkiMixedTest::destroyGraph(reverseGraph);
      }
    }
    std::vector<TestOp> mixedOps = {
      {AllReduce, false, true},
      {AllGather, true, false},
      {ReduceScatter, true, false},
      {AllReduce, true, false}
    };
    TestGraph graph = test.capture(mixedOps, test.streams[0]);
    for (int replay = 0; replay < 3; ++replay) {
      test.execute(mixedOps, epoch++, test.streams[0], &graph);
      test.execute(mixedOps, epoch++, test.streams[1]);
    }
    CeOkiMixedTest::destroyGraph(graph);
    if (rankId == 0) {
      printf("PASS: CE AllGather + OKI AR/RS mixed; ranks=%d count=%zu "
             "dtype=%d executions=%d launchers=%u/%u/%u\n",
             nRanks, config.count, int(pcclFloat32), epoch - 1,
             launchCounts[AllReduce], launchCounts[ReduceScatter], launchCounts[AllGather]);
    }
  }
  return nullptr;
}

int main(int argc, char** argv) {
  TEST_CHECK(argc <= 6);
  nRanks = argc > 5 ? atoi(argv[5]) : 8;
  TEST_CHECK(nRanks == 2 || nRanks == 4 || nRanks == 8);
  TestConfig config = {};
  config.count = argc > 1 ? strtoull(argv[1], nullptr, 0) : 258;
  TEST_CHECK(config.count > 0 && (config.count * sizeof(float)) % 8 == 0);
  config.order = argc > 2 ? argv[2] : "ar-ag";
  config.execution = argc > 3 ? argv[3] : "graph";
  config.layout = argc > 4 ? argv[4] : "oop";
  TEST_CHECK(!strcmp(config.execution, "graph") || !strcmp(config.execution, "eager"));
  TEST_CHECK(!strcmp(config.layout, "ip") || !strcmp(config.layout, "oop"));
  bool inPlace = !strcmp(config.layout, "ip");
  if (!strcmp(config.order, "ar-ag")) {
    config.coldOps = {{AllReduce, inPlace, false}, {AllGather, inPlace, false}};
  } else if (!strcmp(config.order, "rs-ag")) {
    config.coldOps = {{ReduceScatter, inPlace, false}, {AllGather, inPlace, false}};
  } else if (!strcmp(config.order, "ag-ar")) {
    config.coldOps = {{AllGather, inPlace, false}, {AllReduce, inPlace, false}};
  } else if (!strcmp(config.order, "ag-rs")) {
    config.coldOps = {{AllGather, inPlace, false}, {ReduceScatter, inPlace, false}};
  } else if (!strcmp(config.order, "ag")) {
    config.coldOps = {{AllGather, inPlace, false}};
  } else {
    failTest("unknown operation order", __FILE__, __LINE__);
  }
  // Configure process-wide policy before starting any rank threads.
  TEST_CHECK(setenv("PCCL_CTA_POLICY", "2", 1) == 0);
  TEST_CHECK(setenv("PCCL_CE_ENABLE_AR", "1", 1) == 0);
  TEST_CHECK(setenv("PCCL_CE_ENABLE_RS", "1", 1) == 0);
  TEST_CHECK(setenv("PCCL_CE_OKI_ENABLE_AG", "0", 1) == 0);
  TEST_CHECK(setenv("PCCL_P2P_DMA_ENABLE", "0", 1) == 0);
  int deviceCount;
  HGGCCHECK(hggcGetDeviceCount(&deviceCount));
  TEST_CHECK(deviceCount >= nRanks);
  PCCLCHECK(pcclGetUniqueId(&config.commId));
  PTHREADCHECK(pthread_barrier_init(&rankBarrier, nullptr, nRanks));
  std::vector<RankThread> threads(nRanks);
  for (int rank = 0; rank < nRanks; ++rank) {
    threads[rank].rank = rank;
    threads[rank].config = &config;
    PTHREADCHECK(pthread_create(&threads[rank].thread, nullptr, runRank, &threads[rank]));
  }
  for (int rank = 0; rank < nRanks; ++rank) {
    PTHREADCHECK(pthread_join(threads[rank].thread, nullptr));
  }
  PTHREADCHECK(pthread_barrier_destroy(&rankBarrier));
  return 0;
}
