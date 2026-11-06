//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include <hggc_runtime.h>
#include <hggc_fp16.h>
#include <hggc_bf16.h>
#include <hggc_fp8.h>
#include <pccl.h>
#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int rankId, ranks;
static bool exactFp8Inputs = false;
static bool cancellationInputs = false;
static bool uniformCounts = false;
#define HG_CHECK(call) do { auto e = (call); if (e != hggcSuccess) { \
  fprintf(stderr, "rank=%d %s: %s\n", rankId, #call, hggcGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define PC_CHECK(call) do { auto e = (call); if (e != pcclSuccess) { \
  fprintf(stderr, "rank=%d %s: %s\n", rankId, #call, pcclGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

// slot: the ReduceScatter destination or AllGather source rank of an element; -1 for AllReduce.
static float sample(int peer, int work, int slot, size_t index) {
  if (cancellationInputs) return peer < 3 ? 30000.0f : (peer < 6 ? -30000.0f : 1.0f);
  if (exactFp8Inputs) {
    // Default FP8 reduces round every partial sum; at most eight equal power-of-two terms keep them exact.
    // ReduceScatter slot d sums 1 + d % 8 nonzero terms, so destinations differ.
    size_t terms = std::min(ranks, 8);
    if (slot >= 0) terms = 1 + slot % terms;
    bool term = (size_t(peer) + ranks - index % ranks) % ranks < terms;
    return term ? float(1u << ((size_t(work) + index) % 3)) : 0.0f;
  }
  // The slot's own rank adds 2 * slot, so AllGather sources and ReduceScatter destinations differ.
  return float(1 + peer % 2 + work % 2 + index % 3 + (peer == slot ? 2 * slot : 0));
}

template<typename T>
static int run(pcclComm_t comm, pcclDataType_t type, const char* coll, int nWorks,
               size_t count, int registration, bool graphMode, int iterations, size_t offset, bool inPlace) {
  bool ag = strcmp(coll, "ag") == 0;
  bool rs = strcmp(coll, "rs") == 0;
  struct Buffer {
    T *inputBase, *outputBase, *input, *output;
    size_t n, sendElts, recvElts, inputCapacity, outputCapacity;
    pcclWindow_t inputWin, outputWin;
  };
  std::vector<Buffer> buffers(nWorks);
  hggcStream_t stream;
  HG_CHECK(hggcStreamCreate(&stream));
  for (int w = 0; w < nWorks; w++) {
    auto& b = buffers[w];
    b.n = uniformCounts ? count : count + (size_t(w) * 7919) % (count + 1);
    b.sendElts = rs ? b.n * ranks : b.n;
    b.recvElts = ag ? b.n * ranks : b.n;
    b.inputCapacity = std::max(b.sendElts, b.recvElts) + offset + 16;
    b.outputCapacity = b.recvElts + offset + 16;
    PC_CHECK(pcclMemAlloc((void**)&b.inputBase, b.inputCapacity * sizeof(T)));
    PC_CHECK(pcclMemAlloc((void**)&b.outputBase, b.outputCapacity * sizeof(T)));
    b.input = b.inputBase + offset;
    b.output = b.outputBase + offset;
    if (inPlace) {
      b.output = b.input;
      if (ag) b.input += rankId * b.n;
      if (rs) b.output += rankId * b.n;
    }
    b.inputWin = b.outputWin = nullptr;
  }
  // Each collective owns its buffers, so every grouped result is checked independently.
  PC_CHECK(pcclGroupStart());
  for (auto& b : buffers) {
    if (registration & 1)
      PC_CHECK(pcclCommWindowRegister(comm, b.inputBase, b.inputCapacity * sizeof(T),
                                     &b.inputWin, PCCL_WIN_COLL_SYMMETRIC));
    if ((registration & 2) && !(inPlace && (registration & 1))) {
      void* base = inPlace ? (void*)b.inputBase : (void*)b.outputBase;
      size_t bytes = (inPlace ? b.inputCapacity : b.outputCapacity) * sizeof(T);
      PC_CHECK(pcclCommWindowRegister(comm, base, bytes, &b.outputWin, PCCL_WIN_COLL_SYMMETRIC));
    }
  }
  PC_CHECK(pcclGroupEnd());
  for (auto& b : buffers) {
    if (((registration & 1) && !b.inputWin) ||
        ((registration & 2) && !b.outputWin && !(inPlace && b.inputWin))) {
      fprintf(stderr, "rank=%d required symmetric window unavailable; refusing fallback-only validation\n", rankId);
      MPI_Abort(MPI_COMM_WORLD, 3);
    }
  }

  auto reset = [&]() {
    for (int w = 0; w < nWorks; w++) {
      auto& b = buffers[w];
      std::vector<T> input(b.inputCapacity, T(-7.0f));
      size_t start = size_t(b.input - b.inputBase);
      for (size_t i = 0; i < b.sendElts; i++) {
        int slot = ag ? rankId : (rs ? int(i / b.n) : -1);
        input[start + i] = T(sample(rankId, w, slot, i % b.n));
      }
      HG_CHECK(hggcMemcpy(b.inputBase, input.data(), input.size() * sizeof(T), hggcMemcpyHostToDevice));
      std::vector<T> output(b.outputCapacity, T(-7.0f));
      HG_CHECK(hggcMemcpy(b.outputBase, output.data(), output.size() * sizeof(T), hggcMemcpyHostToDevice));
    }
  };
  auto enqueue = [&]() {
    PC_CHECK(pcclGroupStart());
    for (auto& b : buffers) {
      if (ag) PC_CHECK(pcclAllGather(b.input, b.output, b.n, type, comm, stream));
      else if (rs) PC_CHECK(pcclReduceScatter(b.input, b.output, b.n, type, pcclSum, comm, stream));
      else PC_CHECK(pcclAllReduce(b.input, b.output, b.n, type, pcclSum, comm, stream));
    }
    PC_CHECK(pcclGroupEnd());
  };

  reset();
  hggcGraph_t graph = nullptr;
  hggcGraphExec_t executable = nullptr;
  if (graphMode) {
    // Run once before capture so lazily created communicator resources exist, as applications do.
    enqueue();
    HG_CHECK(hggcStreamSynchronize(stream));
    HG_CHECK(hggcStreamBeginCapture(stream, hggcStreamCaptureModeThreadLocal));
    enqueue();
    HG_CHECK(hggcStreamEndCapture(stream, &graph));
    HG_CHECK(hggcGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  }
  auto launch = [&]() {
    if (graphMode) HG_CHECK(hggcGraphLaunch(executable, stream));
    else enqueue();
  };
  hggcEvent_t begin, end;
  HG_CHECK(hggcEventCreate(&begin));
  HG_CHECK(hggcEventCreate(&end));
  for (int i = 0; i < 10; i++) launch();
  HG_CHECK(hggcStreamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);
  HG_CHECK(hggcEventRecord(begin, stream));
  auto cpuBegin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; i++) launch();
  auto cpuEnd = std::chrono::steady_clock::now();
  HG_CHECK(hggcEventRecord(end, stream));
  HG_CHECK(hggcEventSynchronize(end));
  float milliseconds;
  HG_CHECK(hggcEventElapsedTime(&milliseconds, begin, end));
  double eventUs = milliseconds * 1000 / iterations;
  double cpuUs = std::chrono::duration<double, std::micro>(cpuEnd - cpuBegin).count() / iterations;
  MPI_Allreduce(MPI_IN_PLACE, &eventUs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &cpuUs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  auto check = [&]() {
    size_t wrong = 0;
    for (int w = 0; w < nWorks; w++) {
      auto& b = buffers[w];
      T* base = inPlace ? b.inputBase : b.outputBase;
      size_t capacity = inPlace ? b.inputCapacity : b.outputCapacity;
      std::vector<T> result(capacity);
      HG_CHECK(hggcMemcpy(result.data(), base, capacity * sizeof(T), hggcMemcpyDeviceToHost));
      size_t start = size_t(b.output - base);
      for (size_t i = 0; i < b.recvElts; i++) {
        float expected = 0;
        if (ag) expected = sample(int(i / b.n), w, int(i / b.n), i % b.n);
        else for (int peer = 0; peer < ranks; peer++) expected += sample(peer, w, rs ? rankId : -1, i % b.n);
        float rounded = float(T(expected));
        if (float(result[start + i]) != rounded) {
          if (wrong < 4) fprintf(stderr, "rank=%d work=%d index=%zu got=%g expected=%g\n",
                                 rankId, w, i, double(float(result[start + i])), double(rounded));
          wrong++;
        }
      }
      // Guard regions retain the sentinel; non-target in-place RS input slots are not guards.
      size_t dataStart = offset;
      size_t dataEnd = dataStart + (inPlace ? std::max(b.sendElts, b.recvElts) : b.recvElts);
      for (size_t i = 0; i < dataStart; i++) if (float(result[i]) != -7.0f) wrong++;
      for (size_t i = dataEnd; i < capacity; i++) if (float(result[i]) != -7.0f) wrong++;
    }
    return wrong;
  };

  // Reset before checking so repeated in-place accumulation cannot change the reference input.
  reset();
  launch();
  HG_CHECK(hggcStreamSynchronize(stream));
  size_t wrong = check();

  HG_CHECK(hggcEventDestroy(begin));
  HG_CHECK(hggcEventDestroy(end));
  if (graphMode) {
    HG_CHECK(hggcGraphExecDestroy(executable));
    HG_CHECK(hggcGraphDestroy(graph));
    // Submit again after graph destruction to exercise argument-pool reuse.
    reset();
    enqueue();
    HG_CHECK(hggcStreamSynchronize(stream));
    wrong += check();
  }
  unsigned long long globalWrong = wrong;
  MPI_Allreduce(MPI_IN_PLACE, &globalWrong, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (rankId == 0)
    printf("coll=%s dtype=%d works=%d count=%zu reg=%d graph=%d offset=%zu inplace=%d cancellation=%d event_us=%.3f cpu_us=%.3f wrong=%llu\n",
           coll, int(type), nWorks, count, registration, int(graphMode), offset, int(inPlace),
           int(cancellationInputs), eventUs, cpuUs, globalWrong);

  for (auto& b : buffers) {
    if (b.inputWin) PC_CHECK(pcclCommWindowDeregister(comm, b.inputWin));
    if (b.outputWin) PC_CHECK(pcclCommWindowDeregister(comm, b.outputWin));
    PC_CHECK(pcclMemFree(b.inputBase));
    PC_CHECK(pcclMemFree(b.outputBase));
  }
  HG_CHECK(hggcStreamDestroy(stream));
  return globalWrong ? 1 : 0;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (argc < 8) {
    if (rankId == 0) fprintf(stderr, "usage: %s ar|ag|rs float|half|bf16|f8e4m3|f8e5m2 works count reg graph iters [offset] [inplace] [cancellation] [uniform-counts]\n", argv[0]);
    MPI_Finalize();
    return 2;
  }
  int works = atoi(argv[3]), reg = atoi(argv[5]), graph = atoi(argv[6]), iters = atoi(argv[7]);
  size_t count = strtoull(argv[4], nullptr, 10), offset = argc > 8 ? strtoull(argv[8], nullptr, 10) : 0;
  bool inPlace = argc > 9 && atoi(argv[9]);
  cancellationInputs = argc > 10 && atoi(argv[10]);
  uniformCounts = argc > 11 && atoi(argv[11]);
  if (works < 1 || works > 128 || count > (1u << 26) || offset > 1024 ||
      reg < 0 || reg > 3 || graph < 0 || graph > 1 || iters < 1 || iters > 100000 ||
      (strcmp(argv[1], "ar") && strcmp(argv[1], "ag") && strcmp(argv[1], "rs") &&
       strcmp(argv[1], "perf-sweep") && strcmp(argv[1], "functional-sweep"))) MPI_Abort(MPI_COMM_WORLD, 2);
  exactFp8Inputs = (!strcmp(argv[2], "f8e4m3") || !strcmp(argv[2], "f8e5m2")) && strcmp(argv[1], "ag");
  int devices;
  HG_CHECK(hggcGetDeviceCount(&devices));
  if (ranks > devices) MPI_Abort(MPI_COMM_WORLD, 2);
  HG_CHECK(hggcSetDevice(rankId));
  pcclUniqueId id;
  if (rankId == 0) PC_CHECK(pcclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  pcclComm_t comm;
  PC_CHECK(pcclCommInitRank(&comm, ranks, id, rankId));
  int result = 2;
  if (!strcmp(argv[1], "functional-sweep")) {
    const char* collectives[] = {"ar", "ag", "rs"};
    int workCounts[] = {1, 4, 17};
    size_t elementCounts[] = {1, 257, 16387};
    result = 0;
    for (auto coll : collectives)
      for (int dtype = 0; dtype < 5; dtype++)
        for (auto workCount : workCounts)
          for (auto n : elementCounts) {
            exactFp8Inputs = dtype >= 3 && strcmp(coll, "ag");
            int error;
            if (dtype == 0)
              error = run<float>(comm, pcclFloat32, coll, workCount, n, reg, graph, iters, offset, inPlace);
            else if (dtype == 1)
              error = run<half>(comm, pcclFloat16, coll, workCount, n, reg, graph, iters, offset, inPlace);
            else if (dtype == 2)
              error = run<__ppu_bfloat16>(comm, pcclBfloat16, coll, workCount, n, reg, graph, iters, offset, inPlace);
            else if (dtype == 3)
              error = run<__hg_fp8_e4m3>(comm, pcclFloat8e4m3, coll, workCount, n, reg, graph, iters, offset, inPlace);
            else
              error = run<__hg_fp8_e5m2>(comm, pcclFloat8e5m2, coll, workCount, n, reg, graph, iters, offset, inPlace);
            if (error) result = error;
          }
  } else if (!strcmp(argv[1], "perf-sweep")) {
    if (strcmp(argv[2], "float")) MPI_Abort(MPI_COMM_WORLD, 2);
    const char* collectives[] = {"ar", "ag", "rs"};
    int workCounts[] = {1, 4};
    size_t elementCounts[] = {2, 256, 16384, 262144, 1048576};
    result = 0;
    for (auto coll : collectives)
      for (auto workCount : workCounts)
        for (auto n : elementCounts) {
          int error = run<float>(comm, pcclFloat32, coll, workCount, n, reg, graph,
                                 iters, offset, inPlace);
          if (error) result = error;
        }
  } else if (!strcmp(argv[2], "float")) result = run<float>(comm, pcclFloat32, argv[1], works, count, reg, graph, iters, offset, inPlace);
  else if (!strcmp(argv[2], "half")) result = run<half>(comm, pcclFloat16, argv[1], works, count, reg, graph, iters, offset, inPlace);
  else if (!strcmp(argv[2], "bf16")) result = run<__ppu_bfloat16>(comm, pcclBfloat16, argv[1], works, count, reg, graph, iters, offset, inPlace);
  else if (!strcmp(argv[2], "f8e4m3")) result = run<__hg_fp8_e4m3>(comm, pcclFloat8e4m3, argv[1], works, count, reg, graph, iters, offset, inPlace);
  else if (!strcmp(argv[2], "f8e5m2")) result = run<__hg_fp8_e5m2>(comm, pcclFloat8e5m2, argv[1], works, count, reg, graph, iters, offset, inPlace);
  else if (rankId == 0) fprintf(stderr, "unsupported dtype: %s (no fallback)\n", argv[2]);
  PC_CHECK(pcclCommDestroy(comm));
  MPI_Finalize();
  return result;
}
