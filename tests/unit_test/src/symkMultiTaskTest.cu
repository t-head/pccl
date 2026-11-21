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
#include "comm.h"
#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static int rankId, ranks;
static bool exactFp8Inputs = false;
static bool cancellationInputs = false;
static bool uniformCounts = false;
static bool verifyPath = false;
static bool epochWrap = false;
static int emptyEvery = 0;
static int expectedLaunches = 0;
static int fp8Pattern = 0;
extern int64_t pcclParamSymWarps();
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
static T sampleValue(int peer, int work, int slot, size_t index) {
  if (fp8Pattern && sizeof(T) == 1) {
    unsigned char code = (fp8Pattern - 1 + peer * 37 + work * 29 + index * 31) % 256;
    T value(0.0f);
    do {
      memcpy(&value, &code, 1);
      code++;
    } while (!std::isfinite(float(value)));
    return value;
  }
  return T(sample(peer, work, slot, index));
}

// Inspect the existing launch monitor only in correctness mode, never in timed performance runs.
static void checkPath(pcclComm_t comm, uint64_t before, const char* coll, pcclDataType_t type,
                      std::vector<size_t> expectedCounts) {
  uint64_t launches = comm->kernelNum - before;
  // Launch preparation discards empty plans; they are boundaries, not device launches.
  expectedCounts.erase(std::remove(expectedCounts.begin(), expectedCounts.end(), size_t(0)), expectedCounts.end());
  if (expectedCounts.empty() && launches == 0) {
    if (expectedLaunches != 0) MPI_Abort(MPI_COMM_WORLD, 4);
    if (rankId == 0) printf("SYMK_PATH_CHECK launches=0 batches=0 works=0 ok=1\n");
    return;
  }
  int depth = int(pcclDevKernelArgsFifoDepth());
  if (!launches || launches > uint64_t(depth) || !comm->kernelInfoMonitorPtr) {
    fprintf(stderr, "rank=%d path monitor unavailable: before=%llu after=%llu depth=%d level=%d\n",
            rankId, (unsigned long long)before, (unsigned long long)comm->kernelNum, depth,
            comm->hostDevComm.comm.stateMonitorLevel);
    MPI_Abort(MPI_COMM_WORLD, 4);
  }
  std::vector<size_t> actualCounts;
  int batches = 0;
  const char* forced = getenv("PCCL_SYM_KERNEL");
  for (uint64_t i = before; i < comm->kernelNum; i++) {
    auto& info = comm->kernelInfoMonitorPtr[i % depth];
    bool ag = !strcmp(coll, "ag"), rs = !strcmp(coll, "rs");
    int id = info.symFuncIndex;
    bool rightKernel = ag ? (id >= ncclSymkKernelId_AllGather_LL && id <= ncclSymkKernelId_AllGather_ST_2D)
                     : rs ? (id == ncclSymkKernelId_ReduceScatter_LL || id == ncclSymkKernelId_ReduceScatter_LD)
                          : (id == ncclSymkKernelId_AllReduce_AGxLL_R || id == ncclSymkKernelId_AllReduce_RSxLD_AGxST);
    if (!info.isSymColl || !rightKernel || info.kernelIdx != i + 1 || !info.channelMask ||
        (forced && strcmp(forced, ncclSymkKernelIdToString(id)))) {
      fprintf(stderr, "rank=%d path mismatch: coll=%s sym=%d id=%d seq=%llu expected=%llu mask=%llx\n",
              rankId, coll, int(info.isSymColl), id, (unsigned long long)info.kernelIdx,
              (unsigned long long)(i + 1), (unsigned long long)info.channelMask);
      MPI_Abort(MPI_COMM_WORLD, 4);
    }
    int nWorks = 1;
    int kernelIndex = ncclSymkGetKernelIndex((ncclSymkKernelId)id, ncclDevSum, type);
    void* expectedKernel = ncclSymkKernelList_fast[kernelIndex];
    bool shortEntry = false;
    if (info.symkArgsPooled) {
      expectedKernel = ncclSymkKernelList[kernelIndex];
      auto* args = reinterpret_cast<pcclSymkDevWorkArgs*>(&info.kernelArgs);
      if (args->nMaxChannels < 1 || args->nMaxChannels > MAXCHANNELS) MPI_Abort(MPI_COMM_WORLD, 4);
      size_t header = pcclSymkDevWorkArgs::calcArgsSize(args->nMaxChannels, 0);
      if (info.argsBytes < header || info.argsBytes > sizeof(pcclSymkDevWorkArgs1K) ||
          (info.argsBytes - header) % sizeof(pcclSymkDevWork)) MPI_Abort(MPI_COMM_WORLD, 4);
      nWorks = int((info.argsBytes - header) / sizeof(pcclSymkDevWork));
      if (nWorks < 2 || nWorks > 7) MPI_Abort(MPI_COMM_WORLD, 4);
      auto* works = args->getWorks(args->nMaxChannels);
      for (int w = 0; w < nWorks; w++) actualCounts.push_back(works[w].nElts);
      batches++;
    } else {
      if (info.argsBytes != sizeof(ncclSymDevArgs)) MPI_Abort(MPI_COMM_WORLD, 4);
      auto* args = reinterpret_cast<ncclSymDevArgs*>(&info.kernelArgs);
      actualCounts.push_back(args->nElts);
      shortEntry = args->nBlocks == 1 && args->nElts > 0 && args->nElts < 8 &&
                   pcclParamSymWarps() <= 0 && pcclSymkKernelList_short[kernelIndex] != nullptr;
      if (shortEntry) expectedKernel = pcclSymkKernelList_short[kernelIndex];
      if (args->nBlocks != __builtin_popcountll(info.channelMask)) MPI_Abort(MPI_COMM_WORLD, 4);
    }
    if (info.kernelFn != expectedKernel || (shortEntry && info.threadPerBlock != WARP_SIZE))
      MPI_Abort(MPI_COMM_WORLD, 4);
    if (rankId == 0)
      printf("SYMK_PATH kernel=%s entry=%s nWorks=%d argsBytes=%zu threads=%d launch=%llu\n",
             ncclSymkKernelIdToString(id), shortEntry ? "short" : info.symkArgsPooled ? "multi" : "fast",
             nWorks, info.argsBytes, info.threadPerBlock, (unsigned long long)(i - before));
  }
  std::sort(expectedCounts.begin(), expectedCounts.end());
  std::sort(actualCounts.begin(), actualCounts.end());
  if (actualCounts != expectedCounts || (expectedLaunches && launches != uint64_t(expectedLaunches))) {
    fprintf(stderr, "rank=%d path count mismatch: works=%zu/%zu launches=%llu expected-launches=%d\n",
            rankId, actualCounts.size(), expectedCounts.size(), (unsigned long long)launches, expectedLaunches);
    MPI_Abort(MPI_COMM_WORLD, 4);
  }
  if (rankId == 0)
    printf("SYMK_PATH_CHECK launches=%llu batches=%d works=%zu ok=1\n",
           (unsigned long long)launches, batches, actualCounts.size());
}

__global__ void seedEpochs(ncclSymDevBase* base, int nRanks) {
  if (threadIdx.x == 0) {
    base->llEpoch[blockIdx.x] = 0xfffffffcu;
    base->barEpochUc[blockIdx.x] = 0xfffffffcu;
  }
  // Seed inboxes consistently with the local epoch so old low epochs cannot satisfy a wrapped barrier.
  for (int r = threadIdx.x; r < nRanks; r += blockDim.x)
    base->barInboxPerPeer[blockIdx.x * nRanks + r] = 0xfffffffcu;
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
    b.n = emptyEvery && w % emptyEvery == 0 ? 0
        : uniformCounts ? count : count + (size_t(w) * 7919) % (count + 1);
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
        input[start + i] = sampleValue<T>(rankId, w, slot, i % b.n);
      }
      HG_CHECK(hggcMemcpy(b.inputBase, input.data(), input.size() * sizeof(T), hggcMemcpyHostToDevice));
      std::vector<T> output(b.outputCapacity, T(-7.0f));
      HG_CHECK(hggcMemcpy(b.outputBase, output.data(), output.size() * sizeof(T), hggcMemcpyHostToDevice));
    }
  };
  std::vector<size_t> expectedCounts;
  for (auto& b : buffers) expectedCounts.push_back(b.n * (ag ? sizeof(T) : 1));
  auto enqueue = [&]() {
    uint64_t before = verifyPath ? comm->kernelNum : 0;
    if (verifyPath && count == 0 && !comm->memPool_ncclKernelPlan.head) {
      // Make empty-plan ownership observable even before the first nonempty launch.
      auto* plan = ncclMemoryPoolAlloc<ncclKernelPlan>(&comm->memPool_ncclKernelPlan, &comm->memPermanent);
      ncclMemoryPoolFree(&comm->memPool_ncclKernelPlan, plan);
    }
    auto* emptyPlanCell = verifyPath && count == 0 ? comm->memPool_ncclKernelPlan.head : nullptr;
    PC_CHECK(pcclGroupStart());
    for (auto& b : buffers) {
      if (ag) PC_CHECK(pcclAllGather(b.input, b.output, b.n, type, comm, stream));
      else if (rs) PC_CHECK(pcclReduceScatter(b.input, b.output, b.n, type, pcclSum, comm, stream));
      else PC_CHECK(pcclAllReduce(b.input, b.output, b.n, type, pcclSum, comm, stream));
    }
    PC_CHECK(pcclGroupEnd());
    if (verifyPath) {
      checkPath(comm, before, coll, type, expectedCounts);
      if (emptyPlanCell) {
        bool returned = false;
        for (auto* cell = comm->memPool_ncclKernelPlan.head; cell; cell = cell->next)
          if (cell == emptyPlanCell) returned = true;
        if (!returned) {
          fprintf(stderr, "rank=%d empty plan was not returned to its pool\n", rankId);
          MPI_Abort(MPI_COMM_WORLD, 4);
        }
      }
    }
  };

  reset();
  if (epochWrap) {
    // Finish initialization, then force the actual LL/barrier state across uint32_t wraparound.
    enqueue();
    HG_CHECK(hggcStreamSynchronize(stream));
    // Use the same LSA-flat virtual address as the communication kernels, not its allocation alias.
    auto* base = comm->symkState.kcomm.base;
    void* params[] = {&base, &ranks};
    HG_CHECK(hggcLaunchKernel((const void*)seedEpochs, dim3(ncclSymMaxBlocks), dim3(32), params, 0, stream));
    HG_CHECK(hggcStreamSynchronize(stream));
    uint32_t epochs[ncclSymMaxBlocks];
    HG_CHECK(hggcMemcpy(epochs, base->llEpoch, sizeof(epochs), hggcMemcpyDeviceToHost));
    for (uint32_t value : epochs) if (value != 0xfffffffcu) MPI_Abort(MPI_COMM_WORLD, 4);
    fprintf(stderr, "SYMK_EPOCH_SEED rank=%d value=0xfffffffc\n", rankId);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  hggcGraph_t graph = nullptr;
  hggcGraphExec_t executable = nullptr;
  int persistentBefore = verifyPath ? comm->localPersistentRefs : 0;
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
        float rounded;
        if (fp8Pattern) {
          // FP8 LL sums round after each rank, through half, rather than rounding only the final sum.
          T accumulator = sampleValue<T>(0, w, -1, i % b.n);
          for (int peer = 1; peer < ranks; peer++) {
            float next = float(sampleValue<T>(peer, w, -1, i % b.n));
            accumulator = T(float(__float2half_rn(float(accumulator) + next)));
          }
          rounded = float(accumulator);
        } else {
          float expected = 0;
          if (ag) expected = sample(int(i / b.n), w, int(i / b.n), i % b.n);
          else for (int peer = 0; peer < ranks; peer++) expected += sample(peer, w, rs ? rankId : -1, i % b.n);
          rounded = float(T(expected));
        }
        if (float(result[start + i]) != rounded &&
            !(std::isnan(float(result[start + i])) && std::isnan(rounded))) {
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
  if (epochWrap) {
    auto& last = comm->kernelInfoMonitorPtr[(comm->kernelNum - 1) % pcclDevKernelArgsFifoDepth()];
    bool ll = last.symFuncIndex == ncclSymkKernelId_AllReduce_AGxLL_R ||
              last.symFuncIndex == ncclSymkKernelId_AllGather_LL ||
              last.symFuncIndex == ncclSymkKernelId_ReduceScatter_LL;
    auto* base = comm->symkState.kcomm.base;
    uint32_t epochs[ncclSymMaxBlocks];
    HG_CHECK(hggcMemcpy(epochs, ll ? base->llEpoch : base->barEpochUc,
                        sizeof(epochs), hggcMemcpyDeviceToHost));
    bool wrapped = false;
    for (uint32_t value : epochs) if (value < 0xfffffffcu) wrapped = true;
    if (!wrapped) MPI_Abort(MPI_COMM_WORLD, 4);
    fprintf(stderr, "SYMK_EPOCH_WRAP rank=%d protocol=%s ok=1\n", rankId, ll ? "LL" : "barrier");
  }

  HG_CHECK(hggcEventDestroy(begin));
  HG_CHECK(hggcEventDestroy(end));
  if (graphMode) {
    HG_CHECK(hggcGraphExecDestroy(executable));
    HG_CHECK(hggcGraphDestroy(graph));
    if (verifyPath) {
      // Graph destruction publishes reclaimers asynchronously; consume them before checking ownership.
      for (int retry = 0; retry < 1000 && comm->localPersistentRefs != persistentBefore; retry++) {
        PC_CHECK(ncclCommPollCallbacks(comm, false));
        if (comm->localPersistentRefs != persistentBefore)
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      if (comm->localPersistentRefs != persistentBefore) MPI_Abort(MPI_COMM_WORLD, 4);
      if (rankId == 0) printf("SYMK_GRAPH_RECLAIM refs=%d ok=1\n", persistentBefore);
    }
    // Submit again after graph destruction to exercise argument-pool reuse.
    reset();
    enqueue();
    HG_CHECK(hggcStreamSynchronize(stream));
    wrong += check();
  }
  unsigned long long globalWrong = wrong;
  MPI_Allreduce(MPI_IN_PLACE, &globalWrong, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (rankId == 0)
    printf("coll=%s dtype=%d works=%d count=%zu reg=%d graph=%d offset=%zu inplace=%d cancellation=%d fp8_pattern=%d event_us=%.3f cpu_us=%.3f wrong=%llu\n",
           coll, int(type), nWorks, count, registration, int(graphMode), offset, int(inPlace),
           int(cancellationInputs), fp8Pattern, eventUs, cpuUs, globalWrong);

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
    if (rankId == 0) fprintf(stderr, "usage: %s ar|ag|rs float|half|bf16|f8e4m3|f8e5m2 works count reg graph iters [offset] [inplace] [cancellation] [uniform-counts] [verify-path] [empty-every] [expected-launches] [epoch-wrap] [fp8-pattern]\n", argv[0]);
    MPI_Finalize();
    return 2;
  }
  int works = atoi(argv[3]), reg = atoi(argv[5]), graph = atoi(argv[6]), iters = atoi(argv[7]);
  size_t count = strtoull(argv[4], nullptr, 10), offset = argc > 8 ? strtoull(argv[8], nullptr, 10) : 0;
  bool inPlace = argc > 9 && atoi(argv[9]);
  cancellationInputs = argc > 10 && atoi(argv[10]);
  uniformCounts = argc > 11 && atoi(argv[11]);
  verifyPath = argc > 12 && atoi(argv[12]);
  emptyEvery = argc > 13 ? atoi(argv[13]) : 0;
  expectedLaunches = argc > 14 ? atoi(argv[14]) : 0;
  epochWrap = argc > 15 && atoi(argv[15]);
  fp8Pattern = argc > 16 ? atoi(argv[16]) : 0;
  if (fp8Pattern < 0 || fp8Pattern > 256 ||
      (fp8Pattern && (strcmp(argv[1], "ar") ||
                     (strcmp(argv[2], "f8e4m3") && strcmp(argv[2], "f8e5m2"))))) MPI_Abort(MPI_COMM_WORLD, 2);
  if (epochWrap && (!verifyPath || count == 0 || emptyEvery == 1)) MPI_Abort(MPI_COMM_WORLD, 2);
  if (verifyPath) {
    setenv("PCCL_STATE_MONITOR_DISABLE", "0", 1);
    setenv("PCCL_STATE_MONITOR_LEVEL", "1", 1);
  }
  if (works < 1 || works > 128 || count > (1u << 26) || offset > 1024 ||
      reg < 0 || reg > 3 || graph < 0 || graph > 1 || iters < 1 || iters > 100000 ||
      emptyEvery < 0 || emptyEvery > works || expectedLaunches < 0 || expectedLaunches > works ||
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
    size_t elementCounts[] = {0, 1, 257, 16387};
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
