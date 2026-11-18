//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE.txt for more license information
//*************************************************************************

#include "pccl.h"
#include "hggc_runtime.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#define CHECK(call) do { if ((call) != 0) { \
  std::fprintf(stderr, "Failed line %d: %s\n", __LINE__, #call); std::exit(1); \
} } while (0)

struct Task {
  void* send;
  void* recv;
  void* sendReg;
  void* recvReg;
  size_t count;
};

template<typename T>
static void run(pcclComm_t* comms, hggcStream_t* streams, int ranks, bool registered) {
  const int tasks = 192;
  const size_t counts[] = {1, 257, 16385, 65, 2049, 513};
  const bool mixedTypes = std::is_same<T, float>::value;
  std::vector<Task> buffers(ranks*tasks);
  for (int rank = 0; rank < ranks; ++rank) {
    CHECK(hggcSetDevice(rank));
    for (int t = 0; t < tasks; ++t) {
      Task& b = buffers[rank*tasks+t];
      b.count = counts[t%6];
      size_t sendCount = b.count * (t%7 == 3 ? ranks : 1);
      size_t recvCount = b.count * (t%7 == 2 ? ranks : 1);
      CHECK(hggcMalloc(&b.send, sendCount*sizeof(T)));
      CHECK(hggcMalloc(&b.recv, recvCount*sizeof(T)));
      std::vector<T> input(sendCount);
      for (size_t i = 0; i < sendCount; ++i) {
        int value = rank+1 + t%5 + i%7;
        input[i] = T(value);
        if (mixedTypes && t%2) std::memcpy(&input[i], &value, sizeof(value));
      }
      CHECK(hggcMemcpy(b.send, input.data(), sendCount*sizeof(T), hggcMemcpyHostToDevice));
      CHECK(hggcMemset(b.recv, 0, recvCount*sizeof(T)));
      if (registered) {
        CHECK(pcclCommRegister(comms[rank], b.send, sendCount*sizeof(T), &b.sendReg));
        CHECK(pcclCommRegister(comms[rank], b.recv, recvCount*sizeof(T), &b.recvReg));
      }
    }
  }
  // All ranks submit identical metadata and order, while buffers, user-op
  // handles and scalar values differ by rank. Every operation owns its buffers.
  for (int iteration = 0; iteration < 3; ++iteration) {
    CHECK(pcclGroupStart());
    for (int t = 0; t < tasks; ++t) {
      for (int rank = 0; rank < ranks; ++rank) {
        CHECK(hggcSetDevice(rank));
        Task& b = buffers[rank*tasks+t];
        pcclDataType_t type = mixedTypes && t%2 == 0 ? pcclFloat32 : pcclInt32;
        pcclRedOp_t op = pcclSum;
        T scalar = T(rank+1+t%3);
        int intScalar = rank+1+t%3;
        if (t%7 == 6) {
          void* scalarPtr = type == pcclFloat32 ? static_cast<void*>(&scalar) : static_cast<void*>(&intScalar);
          CHECK(pcclRedOpCreatePreMulSum(&op, scalarPtr, type, pcclScalarHostImmediate, comms[rank]));
        }
        switch (t%7) {
        case 0: case 1: case 6:
          CHECK(pcclAllReduce(b.send, b.recv, b.count, type, t%7 == 1 ? pcclMax : op,
              comms[rank], streams[rank]));
          break;
        case 2:
          CHECK(pcclAllGather(b.send, b.recv, b.count, type, comms[rank], streams[rank]));
          break;
        case 3:
          CHECK(pcclReduceScatter(b.send, b.recv, b.count, type, pcclSum, comms[rank], streams[rank]));
          break;
        case 4:
          CHECK(pcclBroadcast(b.send, b.recv, b.count, type, t%ranks, comms[rank], streams[rank]));
          break;
        case 5:
          CHECK(pcclReduce(b.send, b.recv, b.count, type, pcclSum, t%ranks, comms[rank], streams[rank]));
          break;
        }
        // The queued task must own a snapshot of the reduction operation.
        if (t%7 == 6) CHECK(pcclRedOpDestroy(op, comms[rank]));
      }
    }
    CHECK(pcclGroupEnd());
    for (int rank = 0; rank < ranks; ++rank) {
      CHECK(hggcSetDevice(rank));
      CHECK(hggcStreamSynchronize(streams[rank]));
      for (int t = 0; t < tasks; ++t) {
        Task& b = buffers[rank*tasks+t];
        if (t%7 == 5 && rank != t%ranks) continue;
        size_t recvCount = b.count * (t%7 == 2 ? ranks : 1);
        std::vector<T> result(recvCount);
        CHECK(hggcMemcpy(result.data(), b.recv, recvCount*sizeof(T), hggcMemcpyDeviceToHost));
        for (size_t i = 0; i < recvCount; ++i) {
          size_t index = t%7 == 3 ? rank*b.count+i : (t%7 == 2 ? i%b.count : i);
          T expected = T(0);
          for (int source = 0; source < ranks; ++source) {
            T value = T(source+1+t%5+index%7);
            if (t%7 == 6) value *= T(source+1+t%3);
            if (t%7 == 1) expected = value;
            else if (t%7 == 2) { if (source == int(i/b.count)) expected = value; }
            else if (t%7 == 4) { if (source == t%ranks) expected = value; }
            else expected += value;
          }
          T actual = result[i];
          if (mixedTypes && t%2) {
            int value;
            std::memcpy(&value, &result[i], sizeof(value));
            actual = T(value);
          }
          if (actual != expected) {
            std::fprintf(stderr, "Mismatch rank=%d task=%d index=%zu got=%g expected=%g\n",
                rank, t, i, double(actual), double(expected));
            std::exit(1);
          }
        }
      }
    }
  }
  for (int rank = 0; rank < ranks; ++rank) {
    CHECK(hggcSetDevice(rank));
    for (int t = 0; t < tasks; ++t) {
      Task& b = buffers[rank*tasks+t];
      if (registered) {
        CHECK(pcclCommDeregister(comms[rank], b.sendReg));
        CHECK(pcclCommDeregister(comms[rank], b.recvReg));
      }
      CHECK(hggcFree(b.send));
      CHECK(hggcFree(b.recv));
    }
  }
}

int main(int argc, char** argv) {
  int ranks = argc > 1 ? std::atoi(argv[1]) : 2;
  bool registered = argc > 2 && std::atoi(argv[2]) != 0;
  int devices;
  CHECK(hggcGetDeviceCount(&devices));
  if (ranks < 2 || ranks > devices) return 1;
  std::vector<pcclComm_t> comms(ranks);
  std::vector<hggcStream_t> streams(ranks);
  CHECK(pcclCommInitAll(comms.data(), ranks, nullptr));
  for (int rank = 0; rank < ranks; ++rank) {
    CHECK(hggcSetDevice(rank));
    CHECK(hggcStreamCreate(&streams[rank]));
  }
  run<int>(comms.data(), streams.data(), ranks, registered);
  run<float>(comms.data(), streams.data(), ranks, registered);
  for (int rank = 0; rank < ranks; ++rank) {
    CHECK(hggcSetDevice(rank));
    CHECK(hggcStreamDestroy(streams[rank]));
    CHECK(pcclCommDestroy(comms[rank]));
  }
  std::printf("COLLECTIVE_SORT_PASS ranks=%d registered=%d tasks=192 repeats=3 types=int32,mixed-int32-float32\n",
      ranks, int(registered));
}
