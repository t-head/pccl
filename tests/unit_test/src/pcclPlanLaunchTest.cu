// SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "hggc_runtime.h"
#include "pccl.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define CHECK_HG(call) do { \
  hggcError_t err = (call); \
  if (err != hggcSuccess) { \
    fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, hggcGetErrorString(err)); \
    std::abort(); \
  } \
} while (0)
#define CHECK_PC(call) do { \
  pcclResult_t err = (call); \
  if (err != pcclSuccess) { \
    fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, pcclGetErrorString(err)); \
    std::abort(); \
  } \
} while (0)

constexpr int kRanks = 2;
constexpr int kComms = 2;
constexpr int kElements = 32;

struct RankState {
  pcclComm_t comm;
  hggcStream_t stream;
  float* send;
  float* recv;
  hggcGraph_t graph;
  hggcGraphExec_t exec;
};

// Use independent output slots to detect omissions and stale/wrapped work.
static void enqueue(RankState state[kComms][kRanks], int firstRank, int endRank, int ops, bool uneven) {
  CHECK_PC(pcclGroupStart());
  for (int c = 0; c < kComms; c++) {
    int count = uneven && c == 1 ? 1 : ops;
    for (int op = 0; op < count; op++) {
      for (int r = firstRank; r < endRank; r++) {
        CHECK_HG(hggcSetDevice(r));
        RankState& s = state[c][r];
        CHECK_PC(pcclAllReduce(s.send + op*kElements, s.recv + op*kElements,
                              kElements, pcclFloat, pcclSum, s.comm, s.stream));
      }
    }
  }
  CHECK_PC(pcclGroupEnd());
}

int main(int argc, char** argv) {
  int ops = argc > 1 ? std::atoi(argv[1]) : 1024;
  bool capture = argc > 2 && std::strcmp(argv[2], "graph") == 0;
  bool threaded = argc > 2 && std::strcmp(argv[2], "threaded") == 0;
  if (ops < 1 || ops > 65536) return 2;
  int nDevices = 0;
  CHECK_HG(hggcGetDeviceCount(&nDevices));
  if (nDevices < kRanks) {
    fprintf(stderr, "pcclPlanLaunchTest requires two assigned devices\n");
    return 2;
  }

  RankState state[kComms][kRanks] = {};
  size_t bytes = (size_t)ops * kElements * sizeof(float);
  for (int c = 0; c < kComms; c++) {
    pcclComm_t comms[kRanks];
    CHECK_PC(pcclCommInitAll(comms, kRanks, nullptr));
    for (int r = 0; r < kRanks; r++) {
      CHECK_HG(hggcSetDevice(r));
      RankState& s = state[c][r];
      s.comm = comms[r];
      CHECK_HG(hggcStreamCreateWithFlags(&s.stream, hggcStreamNonBlocking));
      CHECK_HG(hggcMalloc((void**)&s.send, bytes));
      CHECK_HG(hggcMalloc((void**)&s.recv, bytes));
      std::vector<float> input(ops*kElements);
      for (int op = 0; op < ops; op++) {
        for (int i = 0; i < kElements; i++) input[op*kElements+i] = c*10 + r + 1 + op%17 + i;
      }
      CHECK_HG(hggcMemcpy(s.send, input.data(), bytes, hggcMemcpyHostToDevice));
    }
  }

  CHECK_PC(pcclGroupStart());
  CHECK_PC(pcclGroupEnd());
  CHECK_PC(pcclGroupStart());
  for (int c = 0; c < kComms; c++) {
    for (int r = 0; r < kRanks; r++) {
      CHECK_HG(hggcSetDevice(r));
      RankState& s = state[c][r];
      CHECK_PC(pcclAllReduce(s.send, s.recv, 0, pcclFloat, pcclSum, s.comm, s.stream));
    }
  }
  CHECK_PC(pcclGroupEnd());

  // Establish transports before capture. Both cliques use the same device set
  // but different plan counts in the measured group (second clique: one op).
  enqueue(state, 0, kRanks, 1, false);
  for (int c = 0; c < kComms; c++) {
    for (int r = 0; r < kRanks; r++) {
      CHECK_HG(hggcSetDevice(r));
      CHECK_HG(hggcStreamSynchronize(state[c][r].stream));
    }
  }

  // Repetition exercises FIFO wrap and the return to a new group's first round.
  for (int iter = 0; iter < 3; iter++) {
    for (int c = 0; c < kComms; c++) {
      for (int r = 0; r < kRanks; r++) {
        CHECK_HG(hggcSetDevice(r));
        CHECK_HG(hggcMemsetAsync(state[c][r].recv, 0, bytes, state[c][r].stream));
      }
    }
    if (capture && iter == 0) {
      for (int c = 0; c < kComms; c++) {
        for (int r = 0; r < kRanks; r++) {
          CHECK_HG(hggcSetDevice(r));
          CHECK_HG(hggcStreamBeginCapture(state[c][r].stream, hggcStreamCaptureModeThreadLocal));
        }
      }
    }
    if (!capture || iter == 0) {
      if (threaded) {
        std::thread ranks[kRanks];
        for (int r = 0; r < kRanks; r++) {
          ranks[r] = std::thread([&, r] { enqueue(state, r, r+1, ops, true); });
        }
        for (auto& rank : ranks) rank.join();
      } else {
        enqueue(state, 0, kRanks, ops, true);
      }
    }
    if (capture) {
      if (iter == 0) {
        for (int c = 0; c < kComms; c++) {
          for (int r = 0; r < kRanks; r++) {
            CHECK_HG(hggcSetDevice(r));
            RankState& s = state[c][r];
            CHECK_HG(hggcStreamEndCapture(s.stream, &s.graph));
            CHECK_HG(hggcGraphInstantiate(&s.exec, s.graph, nullptr, nullptr, 0));
          }
        }
      }
      for (int c = 0; c < kComms; c++) {
        for (int r = 0; r < kRanks; r++) {
          CHECK_HG(hggcSetDevice(r));
          CHECK_HG(hggcGraphLaunch(state[c][r].exec, state[c][r].stream));
        }
      }
    }
    for (int c = 0; c < kComms; c++) {
      for (int r = 0; r < kRanks; r++) {
        CHECK_HG(hggcSetDevice(r));
        RankState& s = state[c][r];
        CHECK_HG(hggcStreamSynchronize(s.stream));
        std::vector<float> output(ops*kElements);
        CHECK_HG(hggcMemcpy(output.data(), s.recv, bytes, hggcMemcpyDeviceToHost));
        for (int op = 0; op < ops; op++) {
          for (int i = 0; i < kElements; i++) {
            float expected = c == 1 && op > 0 ? 0 : kRanks*(c*10 + 1 + op%17 + i) + 1;
            if (output[op*kElements+i] != expected) {
              fprintf(stderr, "iter=%d comm=%d rank=%d op=%d element=%d got=%g expected=%g\n",
                      iter, c, r, op, i, output[op*kElements+i], expected);
              return 1;
            }
          }
        }
      }
    }
  }

  for (int c = 0; c < kComms; c++) {
    for (int r = 0; r < kRanks; r++) {
      CHECK_HG(hggcSetDevice(r));
      RankState& s = state[c][r];
      if (capture) {
        CHECK_HG(hggcGraphExecDestroy(s.exec));
        CHECK_HG(hggcGraphDestroy(s.graph));
      }
      CHECK_PC(pcclCommDestroy(s.comm));
      CHECK_HG(hggcFree(s.send));
      CHECK_HG(hggcFree(s.recv));
      CHECK_HG(hggcStreamDestroy(s.stream));
    }
  }
  printf("PASS plan launch: ops=%d capture=%d threaded=%d iterations=3\n", ops, capture, threaded);
  return 0;
}
