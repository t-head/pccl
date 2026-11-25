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
#ifndef SYMK_TEST_PUBLIC_ONLY
#include "comm.h"
#include "pccl_device.h"
#include "scheduler.h"
#include "p2p_dma.h"
#include <dlfcn.h>
#endif
#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sstream>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

#ifndef SYMK_TEST_PUBLIC_ONLY
extern int64_t ncclParamSymCTAs();
extern int64_t ncclParamSymNoWinEnable();
extern int64_t ncclParamSymGinKernelsEnable();
extern int64_t ncclParamSymTmaEnable();
extern int64_t ncclParamWorkArgsBytes();
extern int64_t ncclParamWinStride();
extern int64_t ncclParamSingleProcMemRegEnable();
static_assert(ncclMaxKernelArgsSize() == 4096, "Regular kernel capacity must be 4KiB");
static_assert(sizeof(ncclDevKernelArgs4K) == 4096, "Regular argument storage must be 4KiB");
static_assert(ncclDevMaxChannelsForArgsBytes(4096) == MAXCHANNELS,
              "The full budget must retain all P2P channels");
#endif

static int rankId, ranks;
static bool exactFp8Inputs = false;
static bool cancellationInputs = false;
static bool uniformCounts = false;
static int verifyPath = 0;
static bool epochWrap = false;
static int emptyEvery = 0;
static int expectedLaunches = 0;
static int fp8Pattern = 0;
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
    unsigned char code = (fp8Pattern - 1 + peer * 37 + work * 29 + index * 31 +
                          (slot >= 0 ? slot * 17 : 0)) % 256;
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
#ifndef SYMK_TEST_PUBLIC_ONLY
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
    if (verifyPath == 2) {
      if (info.isSymColl || info.kernelIdx != i + 1 || !info.kernelFn ||
          info.argsBytes > sizeof(ncclDevKernelArgs4K)) MPI_Abort(MPI_COMM_WORLD, 4);
      continue;
    }
    bool ag = !strcmp(coll, "ag"), rs = !strcmp(coll, "rs") || !strcmp(coll, "rs-avg");
    int id = info.symFuncIndex;
    bool rightKernel = ag ? (id == ncclSymkKernelId_AllGather_LL || id == ncclSymkKernelId_AllGather_ST)
                     : rs ? (id == ncclSymkKernelId_ReduceScatter_LL || id == ncclSymkKernelId_ReduceScatter_LD)
                          : (id == ncclSymkKernelId_AllReduce_AGxLL_R || id == ncclSymkKernelId_AllReduce_RSxLD_AGxST);
    if (!info.isSymColl || !rightKernel || info.kernelIdx != i + 1 || !info.channelMask ||
        (forced && strcmp(forced, "^") && strcmp(forced, ncclSymkKernelIdToString(id)))) {
      fprintf(stderr, "rank=%d path mismatch: coll=%s sym=%d id=%d seq=%llu expected=%llu mask=%llx\n",
              rankId, coll, int(info.isSymColl), id, (unsigned long long)info.kernelIdx,
              (unsigned long long)(i + 1), (unsigned long long)info.channelMask);
      MPI_Abort(MPI_COMM_WORLD, 4);
    }
    int red = !strcmp(coll, "rs-avg") ? ncclDevSumPostDiv : ncclDevSum;
    int kernelIndex = ncclSymkGetKernelIndex((ncclSymkKernelId)id, red, type);
    if (kernelIndex < 0 || kernelIndex >= ncclSymkKernelCount) MPI_Abort(MPI_COMM_WORLD, 4);
    auto* args = reinterpret_cast<ncclSymkDevWorkArgs*>(&info.kernelArgs);
    if (args->nMaxChannels < 1 || args->nMaxChannels > MAXCHANNELS) MPI_Abort(MPI_COMM_WORLD, 4);
    size_t header = ncclSymkDevWorkArgs::calcArgsSize(args->nMaxChannels, 0);
    if (info.argsBytes <= header || info.argsBytes > sizeof(ncclSymkDevWorkArgs4K) ||
        (info.argsBytes - header)%sizeof(ncclSymkDevWork)) MPI_Abort(MPI_COMM_WORLD, 4);
    int nWorks = int((info.argsBytes - header)/sizeof(ncclSymkDevWork));
    int userCTAs = std::min<int>(ncclSymkMaxBlocks, ncclParamSymCTAs());
    if (userCTAs > 0 && args->nMaxChannels != userCTAs) MPI_Abort(MPI_COMM_WORLD, 4);
    if (ncclSymkDevWorkArgs::calcArgsSize(args->nMaxChannels, nWorks) != info.argsBytes ||
        args->maxDynamicSmem != ncclSymkKernelMaxDynamicSmem[kernelIndex] ||
        info.kernelFn != ncclSymkKernelList[kernelIndex] || info.threadPerBlock != ncclSymkMaxThreads)
      MPI_Abort(MPI_COMM_WORLD, 4);
    if (nWorks > 1 && ((1u << id)&ncclSymkLLKernelMask())) MPI_Abort(MPI_COMM_WORLD, 4);
    auto* works = args->getWorks(args->nMaxChannels);
    for (int w = 0; w < nWorks; w++) actualCounts.push_back(works[w].nElts);
    batches += nWorks > 1;
    if (rankId == 0)
      printf("SYMK_PATH kernel=%s entry=unified4K nWorks=%d argsBytes=%zu threads=%d launch=%llu\n",
             ncclSymkKernelIdToString(id), nWorks, info.argsBytes, info.threadPerBlock,
             (unsigned long long)(i - before));
  }
  if (verifyPath == 2) {
    if (expectedLaunches && launches != uint64_t(expectedLaunches)) MPI_Abort(MPI_COMM_WORLD, 4);
    if (rankId == 0) printf("LEGACY_PATH_CHECK launches=%llu inputWorks=%zu ok=1\n",
                           (unsigned long long)launches, expectedCounts.size());
    return;
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

__global__ void seedEpochs(pcclDevComm comm, pcclLLA2AHandle handle) {
  auto* ll = static_cast<uint4*>(pcclGetResourceBufferLocalPointer(comm, handle.bufHandle));
  auto* bar = static_cast<uint32_t*>(pcclGetResourceBufferLocalPointer(comm, comm.lsaBarrier.bufHandle));
  if (threadIdx.x == 0) {
    ll[blockIdx.x*(pcclSymkLLMetadataBytesPerBlock/sizeof(uint4))].x = 0xfffffffcu;
    bar[comm.lsaBarrier.nBarriers + blockIdx.x] = 0xfffffffcu;
  }
  for (int r = threadIdx.x; r < comm.lsaSize; r += blockDim.x)
    bar[3*comm.lsaBarrier.nBarriers + blockIdx.x*comm.lsaSize + r] = 0xfffffffcu;
}

static uint32_t readEpoch(pcclComm_t comm, bool ll, int block) {
  auto const& kcomm = comm->symkState.kcomm;
  auto const& device = kcomm.devComm;
  size_t offset = ll ? pcclGetResourceBufferOffset(kcomm.lsaLLA2A.bufHandle) +
                       block*pcclSymkLLMetadataBytesPerBlock
                     : pcclGetResourceBufferOffset(device.lsaBarrier.bufHandle) +
                       (device.lsaBarrier.nBarriers + block)*sizeof(uint32_t);
  void* resource = add4G(device.resourceWindow_inlined.lsaFlatBase,
                        device.lsaRank*device.resourceWindow_inlined.stride4G);
  uint32_t epoch;
  HG_CHECK(hggcMemcpy(&epoch, static_cast<char*>(resource) + offset, sizeof(epoch), hggcMemcpyDeviceToHost));
  return epoch;
}
#endif

template<typename T>
static int run(pcclComm_t comm, pcclDataType_t type, const char* coll, int nWorks,
               size_t count, int registration, bool graphMode, int iterations, size_t offset, bool inPlace) {
  bool ag = strcmp(coll, "ag") == 0;
  bool average = strcmp(coll, "rs-avg") == 0;
  bool rs = strcmp(coll, "rs") == 0 || average;
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
#ifndef SYMK_TEST_PUBLIC_ONLY
    uint64_t before = verifyPath ? comm->kernelNum : 0;
    auto* emptyPlanCell = verifyPath && count == 0 ? comm->memPool_ncclKernelPlan.head : nullptr;
#endif
    PC_CHECK(pcclGroupStart());
    for (auto& b : buffers) {
      if (ag) PC_CHECK(pcclAllGather(b.input, b.output, b.n, type, comm, stream));
      else if (rs) PC_CHECK(pcclReduceScatter(b.input, b.output, b.n, type, average ? pcclAvg : pcclSum, comm, stream));
      else PC_CHECK(pcclAllReduce(b.input, b.output, b.n, type, pcclSum, comm, stream));
    }
    PC_CHECK(pcclGroupEnd());
#ifndef SYMK_TEST_PUBLIC_ONLY
    if (verifyPath) {
      checkPath(comm, before, coll, type, expectedCounts);
      if (count == 0 && comm->memPool_ncclKernelPlan.head != emptyPlanCell) MPI_Abort(MPI_COMM_WORLD, 4);
    }
#endif
  };

  reset();
#ifndef SYMK_TEST_PUBLIC_ONLY
  if (epochWrap) {
    // Finish initialization, then force the actual LL/barrier state across uint32_t wraparound.
    enqueue();
    HG_CHECK(hggcStreamSynchronize(stream));
    auto device = comm->symkState.kcomm.devComm;
    auto handle = comm->symkState.kcomm.lsaLLA2A;
    void* params[] = {&device, &handle};
    HG_CHECK(hggcLaunchKernel((const void*)seedEpochs, dim3(ncclSymkMaxBlocks), dim3(32), params, 0, stream));
    HG_CHECK(hggcStreamSynchronize(stream));
    for (int block = 0; block < ncclSymkMaxBlocks; block++) {
      if (readEpoch(comm, true, block) != 0xfffffffcu ||
          readEpoch(comm, false, block) != 0xfffffffcu) MPI_Abort(MPI_COMM_WORLD, 4);
    }
    fprintf(stderr, "SYMK_EPOCH_SEED rank=%d value=0xfffffffc\n", rankId);
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif
  hggcGraph_t graph = nullptr;
  hggcGraphExec_t executable = nullptr;
#ifndef SYMK_TEST_PUBLIC_ONLY
  int persistentBefore = verifyPath ? comm->localPersistentRefs : 0;
#endif
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
          int slot = rs ? rankId : -1;
          int first = 0;
#ifndef SYMK_TEST_PUBLIC_ONLY
          if (average) {
            auto const& info = comm->kernelInfoMonitorPtr[(comm->kernelNum - 1) % pcclDevKernelArgsFifoDepth()];
            if (info.symFuncIndex == ncclSymkKernelId_ReduceScatter_LD) first = rankId;
          }
#endif
          float accumulator = float(sampleValue<T>(first, w, slot, i % b.n));
          for (int step = 1; step < ranks; step++) {
            int peer = (first + step) % ranks;
            float added = accumulator + float(sampleValue<T>(peer, w, slot, i % b.n));
            accumulator = average ? float(half(added)) : added;
          }
          if (average) accumulator = float(half(accumulator * float(half(1.0f/ranks))));
          rounded = float(T(accumulator));
        } else {
          float expected = 0;
          if (ag) expected = sample(int(i / b.n), w, int(i / b.n), i % b.n);
          else for (int peer = 0; peer < ranks; peer++) expected += sample(peer, w, rs ? rankId : -1, i % b.n);
          if (average) expected *= 1.0f/ranks;
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
#ifndef SYMK_TEST_PUBLIC_ONLY
  if (epochWrap) {
    auto& last = comm->kernelInfoMonitorPtr[(comm->kernelNum - 1) % pcclDevKernelArgsFifoDepth()];
    bool ll = last.symFuncIndex == ncclSymkKernelId_AllReduce_AGxLL_R ||
              last.symFuncIndex == ncclSymkKernelId_AllGather_LL ||
              last.symFuncIndex == ncclSymkKernelId_ReduceScatter_LL;
    bool wrapped = false;
    for (int block = 0; block < ncclSymkMaxBlocks; block++)
      if (readEpoch(comm, ll, block) < 0xfffffffcu) wrapped = true;
    if (!wrapped) MPI_Abort(MPI_COMM_WORLD, 4);
    fprintf(stderr, "SYMK_EPOCH_WRAP rank=%d protocol=%s ok=1\n", rankId, ll ? "LL" : "barrier");
  }
#endif

  HG_CHECK(hggcEventDestroy(begin));
  HG_CHECK(hggcEventDestroy(end));
  if (graphMode) {
    HG_CHECK(hggcGraphExecDestroy(executable));
    HG_CHECK(hggcGraphDestroy(graph));
#ifndef SYMK_TEST_PUBLIC_ONLY
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
#endif
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

#ifndef SYMK_TEST_PUBLIC_ONLY
struct ParamCase {
  const char* key;
  int64_t (*get)();
  int64_t defaultValue;
  bool registered;
};
static const ParamCase paramCases[] = {
  {"SYM_CTAS", ncclParamSymCTAs, 0, false},
  {"SYM_NOWIN_ENABLE", ncclParamSymNoWinEnable, 0, false},
  {"SYM_GIN_KERNELS_ENABLE", ncclParamSymGinKernelsEnable, 1, false},
  {"SYM_TMA_ENABLE", ncclParamSymTmaEnable, 0, true},
  {"WORK_ARGS_BYTES", ncclParamWorkArgsBytes, INT64_MAX, false},
  {"WIN_STRIDE", ncclParamWinStride, -1, false},
  {"SINGLE_PROC_MEM_REG_ENABLE", ncclParamSingleProcMemRegEnable, 0, true},
};

static int checkParamChild(const char* key, const char* firstText, const char* secondText) {
  for (auto const& spec : paramCases) {
    if (strcmp(key, spec.key)) continue;
    int64_t first = spec.get();
    int64_t expectedFirst = strtoll(firstText, nullptr, 10);
    int64_t expectedSecond = strtoll(secondText, nullptr, 10);
    std::string pcclKey = std::string("PCCL_") + key;
    std::string ncclKey = std::string("NCCL_") + key;
    bool ok = first == expectedFirst;
    pcclParamHandle_t* handle = nullptr;
    if (spec.registered) {
      int64_t typed = 0;
      ok &= pcclParamBind(&handle, pcclKey.c_str()) == pcclSuccess;
      ok &= pcclParamGetI64(handle, &typed) == pcclSuccess && typed == first;
      pcclParamHandle_t* alias = nullptr;
      ok &= pcclParamBind(&alias, ncclKey.c_str()) == pcclInvalidArgument;
    }
    if (setenv(pcclKey.c_str(), "7", 1)) return 2;
    int64_t second = spec.get();
    ok &= second == expectedSecond;
    if (spec.registered) {
      int64_t typed = 0;
      ok &= pcclParamGetI64(handle, &typed) == pcclSuccess && typed == second;
    }
    printf("PARAM_CHECK key=%s first=%lld expected=%lld second=%lld expected=%lld ok=%d\n",
           key, (long long)first, (long long)expectedFirst,
           (long long)second, (long long)expectedSecond, int(ok));
    return ok ? 0 : 1;
  }
  return 2;
}

static int checkParamContract(const char* executable) {
  char emptyFile[] = "/tmp/pccl-symk-empty-XXXXXX";
  char configFile[] = "/tmp/pccl-symk-config-XXXXXX";
  int empty = mkstemp(emptyFile), configured = mkstemp(configFile);
  if (empty < 0 || configured < 0) {
    if (empty >= 0) { close(empty); unlink(emptyFile); }
    if (configured >= 0) { close(configured); unlink(configFile); }
    return 2;
  }
  close(empty);
  FILE* file = fdopen(configured, "w");
  if (!file) {
    close(configured);
    unlink(emptyFile);
    unlink(configFile);
    return 2;
  }
  int written = fprintf(file, "PCCL_SYM_CTAS=11\nNCCL_SYM_CTAS=15\nPCCL_SINGLE_PROC_MEM_REG_ENABLE=1\n");
  int closed = fclose(file);
  if (written < 0 || closed) {
    unlink(emptyFile);
    unlink(configFile);
    return 2;
  }
  int tested = 0, failed = 0;
  auto runCase = [&](ParamCase const& spec, const char* pcclValue, const char* ncclValue,
                     const char* noCache, const char* ncclNoCache,
                     const char* conf, int64_t first, int64_t second) {
    fflush(stdout);
    pid_t child = fork();
    if (child == 0) {
      for (auto const& param : paramCases) {
        unsetenv((std::string("PCCL_") + param.key).c_str());
        unsetenv((std::string("NCCL_") + param.key).c_str());
      }
      unsetenv("PCCL_NO_CACHE");
      unsetenv("NCCL_NO_CACHE");
      setenv("PCCL_CONF_FILE", conf, 1);
      setenv("NCCL_CONF_FILE", configFile, 1);
      if (pcclValue) setenv((std::string("PCCL_") + spec.key).c_str(), pcclValue, 1);
      if (ncclValue) setenv((std::string("NCCL_") + spec.key).c_str(), ncclValue, 1);
      if (noCache) setenv("PCCL_NO_CACHE", noCache, 1);
      if (ncclNoCache) setenv("NCCL_NO_CACHE", ncclNoCache, 1);
      std::string firstText = std::to_string(first), secondText = std::to_string(second);
      execl(executable, executable, "param-check-child", spec.key,
            firstText.c_str(), secondText.c_str(), (char*)nullptr);
      _exit(127);
    }
    int status = 0;
    pid_t joined = -1;
    if (child >= 0) {
      do { joined = waitpid(child, &status, 0); } while (joined < 0 && errno == EINTR);
    }
    if (child < 0 || joined != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0) failed++;
    tested++;
  };
  for (auto const& spec : paramCases) {
    std::string pcclKey = std::string("PCCL_") + spec.key;
    runCase(spec, nullptr, nullptr, nullptr, nullptr, emptyFile, spec.defaultValue, spec.defaultValue);
    runCase(spec, "3", nullptr, nullptr, nullptr, emptyFile, 3, 3);
    runCase(spec, nullptr, "5", nullptr, nullptr, emptyFile, spec.defaultValue, spec.defaultValue);
    runCase(spec, "3", "5", nullptr, nullptr, emptyFile, 3, 3);
    runCase(spec, "3", "5", pcclKey.c_str(), nullptr, emptyFile, 3, 7);
    runCase(spec, "3", "5", "ALL", nullptr, emptyFile, 3, 7);
    runCase(spec, "3", "5", nullptr, "ALL", emptyFile, 3, 3);
  }
  runCase(paramCases[0], nullptr, "15", nullptr, nullptr, configFile, 11, 11);
  runCase(paramCases[0], "4", "15", nullptr, nullptr, configFile, 4, 4);
  runCase(paramCases[6], nullptr, "0", nullptr, nullptr, configFile, 1, 1);
  unlink(emptyFile);
  unlink(configFile);
  printf("PARAM_CONTRACT cases=%d failed=%d\n", tested, failed);
  return failed ? 1 : 0;
}

static int checkAlignmentPolicies() {
  HG_CHECK(hggcSetDevice(rankId));
  unsetenv("PCCL_SYM_KERNEL");
  setenv("PCCL_P2P_DMA_ENABLE", "0", 1);
  setenv("PCCL_SYM_NOWIN_ENABLE", "1", 1);
  setenv("PCCL_SYM_CTAS", "1", 1);
  setenv("PCCL_SINGLE_PROC_MEM_REG_ENABLE", "0", 1);
  setenv("PCCL_STATE_MONITOR_DISABLE", "0", 1);
  setenv("PCCL_STATE_MONITOR_LEVEL", "1", 1);
  pcclUniqueId id;
  if (!rankId) PC_CHECK(pcclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  pcclComm_t comm;
  PC_CHECK(pcclCommInitRank(&comm, ranks, id, rankId));
  float* buffers[5];
  pcclWindow_t windows[4];
  for (auto& pointer : buffers) PC_CHECK(pcclMemAlloc((void**)&pointer, 4096));
  PC_CHECK(pcclGroupStart());
  for (int i = 0; i < 4; i++)
    PC_CHECK(pcclCommWindowRegister(comm, buffers[i], 4096, &windows[i], PCCL_WIN_COLL_SYMMETRIC));
  PC_CHECK(pcclGroupEnd());
  ncclDevrWindow *first, *last;
  PC_CHECK(ncclDevrFindWindow(comm, buffers[0], &first));
  PC_CHECK(ncclDevrFindWindow(comm, buffers[1], &last));
  if (!first || !last || first == last) MPI_Abort(MPI_COMM_WORLD, 3);
  auto sysmem = reinterpret_cast<void (*)(ncclDevrWindow*)>(dlsym(RTLD_DEFAULT, "sysmemPolicyFaultSet"));
  if (!sysmem) MPI_Abort(MPI_COMM_WORLD, 5);
  // 仅注入host判定，不改变实际物理内存或解引用私有Memory布局。
  sysmem(last);
  ncclTaskColl tasks[2] = {};
  for (int i = 0; i < 2; i++) {
    tasks[i].func = ncclFuncAllReduce;
    tasks[i].opHost = ncclSum;
    tasks[i].opDev.op = ncclDevSum;
    tasks[i].datatype = ncclFloat32;
    tasks[i].count = 1024;
    tasks[i].algorithm = NCCL_ALGO_UNDEF;
    tasks[i].sendbuff = buffers[i];
    tasks[i].recvbuff = buffers[i + 2];
  }
  tasks[0].next = &tasks[1];
  ncclIntruQueue<ncclTaskColl, &ncclTaskColl::next> symmetric;
  ncclIntruQueueConstruct(&symmetric);
  ncclTaskColl* remaining = nullptr;
  int savedTasks = comm->planner.nTasksColl;
  comm->planner.nTasksColl = 2;
  PC_CHECK(ncclMakeSymmetricTaskList(comm, &tasks[0], &symmetric, &remaining));
  bool ok = ncclIntruQueueEmpty(&symmetric) && remaining == &tasks[1] &&
            remaining->next == &tasks[0] && !tasks[1].isSymLast && tasks[0].isSymLast &&
            comm->planner.nTasksColl == 2;
  sysmem(nullptr);
  comm->planner.nTasksColl = savedTasks;
  if (!ok) MPI_Abort(MPI_COMM_WORLD, 4);
  auto arm = reinterpret_cast<void (*)(pcclComm_t)>(dlsym(RTLD_DEFAULT, "algoCompareFaultArm"));
  auto hits = reinterpret_cast<int (*)()>(dlsym(RTLD_DEFAULT, "algoCompareFaultHits"));
  if (!arm || !hits) MPI_Abort(MPI_COMM_WORLD, 5);
  std::vector<float> values(1024, float(rankId + 1));
  HG_CHECK(hggcMemcpy(buffers[0], values.data(), 4096, hggcMemcpyHostToDevice));
  hggcStream_t stream;
  HG_CHECK(hggcStreamCreate(&stream));
  uint64_t before = comm->kernelNum;
  arm(comm);
  PC_CHECK(pcclAllReduce(buffers[0], buffers[4], 1024, pcclFloat32, pcclSum, comm, stream));
  HG_CHECK(hggcStreamSynchronize(stream));
  HG_CHECK(hggcMemcpy(values.data(), buffers[4], 4096, hggcMemcpyDeviceToHost));
  ok = hits() == 1 && comm->kernelNum == before + 1 &&
       comm->kernelInfoMonitorPtr[before%pcclDevKernelArgsFifoDepth()].isSymColl;
  for (float value : values) ok &= value == float(ranks*(ranks + 1)/2);
  int failed = !ok, wrong;
  MPI_Allreduce(&failed, &wrong, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  HG_CHECK(hggcStreamDestroy(stream));
  for (auto window : windows) PC_CHECK(pcclCommWindowDeregister(comm, window));
  PC_CHECK(pcclCommDestroy(comm));
  for (auto pointer : buffers) PC_CHECK(pcclMemFree(pointer));
  if (!rankId) printf("ALIGNMENT_POLICIES sysmem_metadata_only=1 batch_fallback=2 optional_query_fault=1 wrong=%d\n", wrong);
  return wrong ? 1 : 0;
}

static int checkFirstP2pDma(bool failInitialization = false) {
  if (ranks != 2 && ranks != 4) return 2;
  HG_CHECK(hggcSetDevice(rankId));
  setenv("PCCL_P2P_DMA_ENABLE", "1", 1);
  setenv("PCCL_P2P_DMA_MIN_BYTES", "0", 1);
  pcclUniqueId id;
  if (!rankId) PC_CHECK(pcclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  pcclConfig_t config = PCCL_CONFIG_INITIALIZER;
  config.CTAPolicy = PCCL_CTA_POLICY_ZERO;
  pcclComm_t comm;
  PC_CHECK(pcclCommInitRankConfig(&comm, ranks, id, rankId, &config));
  if (comm->ceColl.initialized || !pcclP2pDmaEnabled(comm)) {
    fprintf(stderr, "DMA_FIRST admission failed rank=%d policy=%d oki=%d qps=%d fifo=%u\n",
            rankId, comm->config.CTAPolicy, comm->okiSupport, comm->okiQpCount, comm->workFifoBytes);
    MPI_Abort(MPI_COMM_WORLD, 3);
  }
  constexpr size_t count = 65536, guard = 16, total = count + 2*guard;
  float *input, *output;
  PC_CHECK(pcclMemAlloc((void**)&input, total*sizeof(float)));
  PC_CHECK(pcclMemAlloc((void**)&output, total*sizeof(float)));
  pcclWindow_t sendWin, recvWin;
  auto faultArm = reinterpret_cast<void (*)(pcclComm_t)>(dlsym(RTLD_DEFAULT, "ceInitFaultArm"));
  auto faultHits = reinterpret_cast<int (*)()>(dlsym(RTLD_DEFAULT, "ceInitFaultHits"));
  if (failInitialization) {
    if (!faultArm || !faultHits) MPI_Abort(MPI_COMM_WORLD, 5);
    faultArm(comm);
  }
  PC_CHECK(pcclGroupStart());
  PC_CHECK(pcclCommWindowRegister(comm, input, total*sizeof(float), &sendWin, PCCL_WIN_COLL_SYMMETRIC));
  PC_CHECK(pcclCommWindowRegister(comm, output, total*sizeof(float), &recvWin, PCCL_WIN_COLL_SYMMETRIC));
  auto registered = pcclGroupEnd();
  if (failInitialization) {
    // CE初始化错误须沿failure链传播且不留下半初始化状态；OKI comm销毁后同进程重建被SDK
    // 设备缓存缺陷阻断（okiDeviceFinish不清g_deviceCache，纯SDK复现见oki-cycle诊断），后续通信由新进程用例覆盖。
    bool ok = registered == pcclUnhandledHggcError && faultHits() == 1 &&
              !comm->ceColl.initialized && !comm->ceColl.ceSyncBase && !comm->p2pDmaState.initialized &&
              ncclIntruQueueEmpty(&comm->devrState.regTaskQueue) &&
              ncclIntruQueueEmpty(&comm->planner.planQueue);
    if (!ok) MPI_Abort(MPI_COMM_WORLD, 6);
    PC_CHECK(pcclCommAbort(comm));
    PC_CHECK(pcclMemFree(input));
    PC_CHECK(pcclMemFree(output));
    if (!rankId) printf("DMA_INIT_FAULT propagated=1 queues_empty=1 abort=1 wrong=0\n");
    return 0;
  }
  PC_CHECK(registered);
  if (!comm->ceColl.initialized || !comm->ceColl.ceSyncBase || !comm->p2pDmaState.initialized)
    MPI_Abort(MPI_COMM_WORLD, 4);
  void* syncBase = comm->ceColl.ceSyncBase;
  void* mailbox = comm->p2pDmaState.mboxBase;
  hggcStream_t stream;
  HG_CHECK(hggcStreamCreate(&stream));
  int destination = (rankId + 1)%ranks, source = (rankId + ranks - 1)%ranks;
  bool ok = true;
  for (int repeat = 0; repeat < 3; repeat++) {
    std::vector<float> expected(total, -7.0f), actual(total, -7.0f);
    for (size_t i = guard; i < guard + count; i++) expected[i] = float(rankId + repeat + 1);
    HG_CHECK(hggcMemcpy(input, expected.data(), total*sizeof(float), hggcMemcpyHostToDevice));
    HG_CHECK(hggcMemcpy(output, actual.data(), total*sizeof(float), hggcMemcpyHostToDevice));
    uint64_t kernels = comm->kernelNum;
    PC_CHECK(pcclGroupStart());
    PC_CHECK(pcclSend(input + guard, count, pcclFloat32, destination, comm, stream));
    PC_CHECK(pcclRecv(output + guard, count, pcclFloat32, source, comm, stream));
    PC_CHECK(pcclGroupEnd());
    HG_CHECK(hggcStreamSynchronize(stream));
    HG_CHECK(hggcMemcpy(actual.data(), output, total*sizeof(float), hggcMemcpyDeviceToHost));
    ok &= comm->kernelNum == kernels && comm->p2pDmaState.peerSeq[source] == unsigned(repeat + 1) &&
          comm->p2pDmaState.peerSeq[destination] == unsigned(repeat + 1) &&
          comm->ceColl.ceSyncBase == syncBase && comm->p2pDmaState.mboxBase == mailbox;
    for (size_t i = 0; i < total; i++)
      ok &= actual[i] == (i >= guard && i < guard + count ? float(source + repeat + 1) : -7.0f);
  }
  int failed = !ok, wrong;
  MPI_Allreduce(&failed, &wrong, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  HG_CHECK(hggcStreamDestroy(stream));
  PC_CHECK(pcclCommWindowDeregister(comm, sendWin));
  PC_CHECK(pcclCommWindowDeregister(comm, recvWin));
  PC_CHECK(pcclCommDestroy(comm));
  PC_CHECK(pcclMemFree(input));
  PC_CHECK(pcclMemFree(output));
  if (!rankId) printf("DMA_FIRST ranks=%d repeats=3 ce_before_collective=1 legacy_launches=0 guards=1 wrong=%d\n", ranks, wrong);
  return wrong ? 1 : 0;
}

static int checkBudgetContract(bool graphMode) {
  HG_CHECK(hggcSetDevice(rankId));
  size_t budget = 128;
  std::string budgetText = std::to_string(budget);
  setenv("PCCL_WORK_ARGS_BYTES", budgetText.c_str(), 1);
  setenv("PCCL_MIN_NCHANNELS", "8", 1);
  setenv("PCCL_MAX_NCHANNELS", "8", 1);
  setenv("PCCL_SYM_KERNEL", "AllReduce_RSxLD_AGxST", 1);
  setenv("PCCL_STATE_MONITOR_DISABLE", "0", 1);
  setenv("PCCL_STATE_MONITOR_LEVEL", "1", 1);
  pcclUniqueId id;
  if (rankId == 0) PC_CHECK(pcclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  pcclComm_t comm;
  PC_CHECK(pcclCommInitRank(&comm, ranks, id, rankId));
  float *input, *output;
  PC_CHECK(pcclMemAlloc((void**)&input, 4096));
  PC_CHECK(pcclMemAlloc((void**)&output, 4096));
  pcclWindow_t inputWin, outputWin;
  PC_CHECK(pcclGroupStart());
  PC_CHECK(pcclCommWindowRegister(comm, input, 4096, &inputWin, PCCL_WIN_COLL_SYMMETRIC));
  PC_CHECK(pcclCommWindowRegister(comm, output, 4096, &outputWin, PCCL_WIN_COLL_SYMMETRIC));
  PC_CHECK(pcclGroupEnd());
  if (!inputWin || !outputWin) MPI_Abort(MPI_COMM_WORLD, 3);
  std::vector<float> values(1024, float(rankId + 1));
  HG_CHECK(hggcMemcpy(input, values.data(), 4096, hggcMemcpyHostToDevice));
  hggcStream_t stream;
  HG_CHECK(hggcStreamCreate(&stream));
  uint64_t before = comm->kernelNum;
  hggcGraph_t graph = nullptr;
  hggcGraphExec_t executable = nullptr;
  if (graphMode) HG_CHECK(hggcStreamBeginCapture(stream, hggcStreamCaptureModeThreadLocal));
  PC_CHECK(pcclGroupStart());
  PC_CHECK(pcclAllReduce(input, output, 1024, pcclFloat32, pcclSum, comm, stream));
  pcclResult_t result = pcclGroupEnd();
  PC_CHECK(result);
  if (graphMode) {
    HG_CHECK(hggcStreamEndCapture(stream, &graph));
    HG_CHECK(hggcGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    HG_CHECK(hggcGraphLaunch(executable, stream));
  }
  HG_CHECK(hggcStreamSynchronize(stream));
  if (graphMode) {
    HG_CHECK(hggcGraphExecDestroy(executable));
    HG_CHECK(hggcGraphDestroy(graph));
  }
  HG_CHECK(hggcMemcpy(values.data(), output, 4096, hggcMemcpyDeviceToHost));
  bool ok = comm->kernelNum > before && ncclIntruQueueEmpty(&comm->planner.planQueue) &&
            ncclIntruQueueEmpty(&comm->planner.collTaskQueue) &&
            ncclIntruQueueEmpty(&comm->planner.collSymTaskQueue) && comm->planner.nTasksColl == 0;
  float sum = float(ranks*(ranks + 1)/2);
  for (float value : values) ok &= value == sum;
  auto const& sumPlan = comm->kernelInfoMonitorPtr[(comm->kernelNum - 1)%pcclDevKernelArgsFifoDepth()];
  ok &= !sumPlan.isSymColl && sumPlan.argsBytes <= budget;
  if (!ok) MPI_Abort(MPI_COMM_WORLD, 4);
  uint64_t productBefore = comm->kernelNum;
  PC_CHECK(pcclAllReduce(input, output, 1024, pcclFloat32, pcclProd, comm, stream));
  HG_CHECK(hggcStreamSynchronize(stream));
  HG_CHECK(hggcMemcpy(values.data(), output, 4096, hggcMemcpyDeviceToHost));
  float expected = 1;
  for (int peer = 1; peer <= ranks; peer++) expected *= peer;
  for (float value : values) ok &= value == expected;
  auto const& last = comm->kernelInfoMonitorPtr[(comm->kernelNum - 1)%pcclDevKernelArgsFifoDepth()];
  ok &= comm->kernelNum > productBefore && !last.isSymColl && last.argsBytes <= budget &&
        ncclIntruQueueEmpty(&comm->planner.planQueue);
  int failed = !ok, wrong = 0;
  MPI_Allreduce(&failed, &wrong, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  HG_CHECK(hggcStreamDestroy(stream));
  PC_CHECK(pcclCommWindowDeregister(comm, inputWin));
  PC_CHECK(pcclCommWindowDeregister(comm, outputWin));
  PC_CHECK(pcclMemFree(input));
  PC_CHECK(pcclMemFree(output));
  PC_CHECK(pcclCommDestroy(comm));
  if (rankId == 0) printf("SYMK_BUDGET_LEGACY graph=%d budget=%zu status=%d wrong=%d\n",
                           int(graphMode), budget, int(result), wrong);
  return wrong ? 1 : 0;
}

static int checkEmptyP2P(bool graphMode, size_t argsBudget = ncclMaxKernelArgsSize()) {
  if (ranks < 2 || (argsBudget != 1024 && argsBudget != 4096)) return 2;
  HG_CHECK(hggcSetDevice(rankId));
  std::string budgetText = std::to_string(argsBudget);
  setenv("PCCL_WORK_ARGS_BYTES", budgetText.c_str(), 1);
  setenv("PCCL_MIN_P2P_NCHANNELS", "64", 1);
  setenv("PCCL_MAX_P2P_NCHANNELS", "64", 1);
  setenv("PCCL_P2P_DMA_ENABLE", "0", 1);
  setenv("PCCL_STATE_MONITOR_DISABLE", "0", 1);
  setenv("PCCL_STATE_MONITOR_LEVEL", "1", 1);
  pcclUniqueId id;
  if (rankId == 0) PC_CHECK(pcclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  pcclComm_t comm;
  PC_CHECK(pcclCommInitRank(&comm, ranks, id, rankId));
  hggcStream_t stream;
  HG_CHECK(hggcStreamCreate(&stream));
  int *input, *output;
  HG_CHECK(hggcMalloc(&input, sizeof(int)));
  HG_CHECK(hggcMalloc(&output, sizeof(int)));
  int expectedChannels = pow2Down(ncclDevMaxChannelsForArgsBytes(argsBudget));
  bool channelsValid = comm->workArgsBytes == argsBudget && comm->p2pnChannels == expectedChannels;
  int sent = rankId + 1, sentinel = -7, received = 0;
  HG_CHECK(hggcMemcpy(input, &sent, sizeof(sent), hggcMemcpyHostToDevice));
  HG_CHECK(hggcMemcpy(output, &sentinel, sizeof(sentinel), hggcMemcpyHostToDevice));
  auto enqueue = [&](size_t count) {
    PC_CHECK(pcclGroupStart());
    PC_CHECK(pcclSend(input, count, pcclInt32, (rankId + 1)%ranks, comm, stream));
    PC_CHECK(pcclRecv(output, count, pcclInt32, (rankId + ranks - 1)%ranks, comm, stream));
    PC_CHECK(pcclGroupEnd());
  };
  uint64_t before = comm->kernelNum;
  enqueue(0);
  HG_CHECK(hggcStreamSynchronize(stream));
  bool retainedLaunch = comm->kernelNum > before;
  int failed = !retainedLaunch || !channelsValid;
  HG_CHECK(hggcMemcpy(&received, output, sizeof(received), hggcMemcpyDeviceToHost));
  failed += received != sentinel;
  hggcGraph_t graph = nullptr;
  hggcGraphExec_t executable = nullptr;
  int refs = comm->localPersistentRefs;
  if (graphMode) {
    HG_CHECK(hggcStreamBeginCapture(stream, hggcStreamCaptureModeThreadLocal));
    enqueue(0);
    HG_CHECK(hggcStreamEndCapture(stream, &graph));
    HG_CHECK(hggcGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int i = 0; i < 4; i++) HG_CHECK(hggcGraphLaunch(executable, stream));
    HG_CHECK(hggcStreamSynchronize(stream));
    HG_CHECK(hggcMemcpy(&received, output, sizeof(received), hggcMemcpyDeviceToHost));
    failed += received != sentinel;
    HG_CHECK(hggcGraphExecDestroy(executable));
    HG_CHECK(hggcGraphDestroy(graph));
    for (int retry = 0; retry < 1000 && comm->localPersistentRefs != refs; retry++) {
      PC_CHECK(ncclCommPollCallbacks(comm, false));
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    failed += comm->localPersistentRefs != refs;
  }
  enqueue(1);
  HG_CHECK(hggcStreamSynchronize(stream));
  HG_CHECK(hggcMemcpy(&received, output, sizeof(received), hggcMemcpyDeviceToHost));
  failed += received != (rankId + ranks - 1)%ranks + 1;
  for (uint64_t index = before; index < comm->kernelNum; index++) {
    auto const& record = comm->kernelInfoMonitorPtr[index%pcclDevKernelArgsFifoDepth()];
    failed += record.isSymColl || !record.kernelFn || record.kernelIdx != index + 1 ||
              record.argsBytes > sizeof(ncclDevKernelArgs4K);
  }
  int wrong = 0;
  MPI_Allreduce(&failed, &wrong, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  HG_CHECK(hggcFree(input));
  HG_CHECK(hggcFree(output));
  HG_CHECK(hggcStreamDestroy(stream));
  PC_CHECK(pcclCommDestroy(comm));
  if (rankId == 0) printf("P2P_EMPTY graph=%d ranks=%d retained_launch=%d argsBudget=%zu p2pChannels=%d wrong=%d\n",
                           int(graphMode), ranks, int(retainedLaunch), argsBudget, expectedChannels, wrong);
  return wrong ? 1 : 0;
}
#endif

int main(int argc, char** argv) {
#ifndef SYMK_TEST_PUBLIC_ONLY
  if (argc == 2 && !strcmp(argv[1], "param-contract")) return checkParamContract(argv[0]);
  if (argc == 5 && !strcmp(argv[1], "param-check-child")) return checkParamChild(argv[2], argv[3], argv[4]);
#endif
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
#ifndef SYMK_TEST_PUBLIC_ONLY
  if (argc == 2 && (!strcmp(argv[1], "alignment-policies") || !strcmp(argv[1], "p2p-dma-first") ||
                    !strcmp(argv[1], "p2p-dma-init-fault"))) {
    int result = !strcmp(argv[1], "alignment-policies") ? checkAlignmentPolicies() :
                 checkFirstP2pDma(!strcmp(argv[1], "p2p-dma-init-fault"));
    MPI_Finalize();
    return result;
  }
  if (argc == 3 && !strcmp(argv[1], "p2p-empty")) {
    int result = checkEmptyP2P(atoi(argv[2]) != 0);
    MPI_Finalize();
    return result;
  }
  if (argc == 4 && !strcmp(argv[1], "p2p-budget")) {
    int result = checkEmptyP2P(atoi(argv[3]) != 0, strtoull(argv[2], nullptr, 10));
    MPI_Finalize();
    return result;
  }
  if (argc == 3 && !strcmp(argv[1], "budget-contract")) {
    int result = checkBudgetContract(atoi(argv[2]) != 0);
    MPI_Finalize();
    return result;
  }
  if (argc == 2 && !strcmp(argv[1], "layout")) {
    size_t header = ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 0);
    if (rankId == 0) printf("%zu %zu %zu\n", header, sizeof(ncclSymkDevWork),
                            (sizeof(ncclSymkDevWorkArgs4K) - header)/sizeof(ncclSymkDevWork));
    MPI_Finalize();
    return 0;
  }
#endif
  if (argc < 8) {
    if (rankId == 0) fprintf(stderr, "usage: %s ar|ag|rs|rs-avg float|half|bf16|f8e4m3|f8e5m2 works count reg graph iters [offset] [inplace] [cancellation] [uniform-counts] [verify-path] [empty-every] [expected-launches] [epoch-wrap] [fp8-pattern]\n", argv[0]);
    MPI_Finalize();
    return 2;
  }
  int works = atoi(argv[3]), reg = atoi(argv[5]), graph = atoi(argv[6]), iters = atoi(argv[7]);
  size_t count = strtoull(argv[4], nullptr, 10), offset = argc > 8 ? strtoull(argv[8], nullptr, 10) : 0;
  bool inPlace = argc > 9 && atoi(argv[9]);
  cancellationInputs = argc > 10 && atoi(argv[10]);
  uniformCounts = argc > 11 && atoi(argv[11]);
  verifyPath = argc > 12 ? atoi(argv[12]) : 0;
  if (verifyPath < 0 || verifyPath > 2) MPI_Abort(MPI_COMM_WORLD, 2);
#ifdef SYMK_TEST_PUBLIC_ONLY
  if (verifyPath != 0 || !strcmp(argv[1], "functional-sweep") ||
      !strcmp(argv[1], "kernel-cache-contract")) MPI_Abort(MPI_COMM_WORLD, 2);
#endif
  emptyEvery = argc > 13 ? atoi(argv[13]) : 0;
  expectedLaunches = argc > 14 ? atoi(argv[14]) : 0;
  epochWrap = argc > 15 && atoi(argv[15]);
  fp8Pattern = argc > 16 ? atoi(argv[16]) : 0;
  if (fp8Pattern < 0 || fp8Pattern > 256 ||
      (fp8Pattern && ((strcmp(argv[1], "ar") && strcmp(argv[1], "rs-avg")) || verifyPath != 1 ||
                     (strcmp(argv[2], "f8e4m3") && strcmp(argv[2], "f8e5m2"))))) MPI_Abort(MPI_COMM_WORLD, 2);
  if (epochWrap && (verifyPath != 1 || works == 0 || count == 0 || emptyEvery == 1)) MPI_Abort(MPI_COMM_WORLD, 2);
#ifdef SYMK_TEST_PUBLIC_ONLY
  if (epochWrap || fp8Pattern) MPI_Abort(MPI_COMM_WORLD, 2);
#endif
  if (verifyPath) {
    setenv("PCCL_STATE_MONITOR_DISABLE", "0", 1);
    setenv("PCCL_STATE_MONITOR_LEVEL", "1", 1);
  }
#ifndef SYMK_TEST_PUBLIC_ONLY
  if (!strcmp(argv[1], "kernel-cache-contract")) {
    if (works != 1 || reg != 3 || verifyPath != 1 || strcmp(argv[2], "float")) MPI_Abort(MPI_COMM_WORLD, 2);
    setenv("PCCL_SYM_KERNEL", "AllReduce_RSxLD_AGxST", 1);
    setenv("NCCL_SYM_KERNEL", "unknown_symk_kernel", 1);
  }
#endif
  if (works < 0 || works > 128 || count > (1u << 26) || offset > 1024 ||
      reg < 0 || reg > 3 || graph < 0 || graph > 1 || iters < 1 || iters > 100000 ||
      emptyEvery < 0 || emptyEvery > works || expectedLaunches < 0 || expectedLaunches > works ||
      (strcmp(argv[1], "ar") && strcmp(argv[1], "ag") && strcmp(argv[1], "rs") && strcmp(argv[1], "rs-avg") &&
       strcmp(argv[1], "perf-sweep") && strcmp(argv[1], "perf-inputs") &&
       strcmp(argv[1], "functional-sweep") && strcmp(argv[1], "kernel-cache-contract"))) MPI_Abort(MPI_COMM_WORLD, 2);
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
  if (!strcmp(argv[1], "perf-inputs")) {
    std::istringstream inputs(argv[2]);
    std::string entry;
    result = 0;
    int cases = 0;
    while (std::getline(inputs, entry, ';')) {
      std::replace(entry.begin(), entry.end(), ',', ' ');
      std::istringstream fields(entry);
      std::string coll, dtype, extra;
      int nWorks;
      size_t nElts;
      if (!(fields >> coll >> dtype >> nWorks >> nElts) || (fields >> extra) ||
          nWorks < 1 || nWorks > 128 || nElts > (1u << 26) ||
          (coll != "ar" && coll != "ag" && coll != "rs" && coll != "rs-avg")) MPI_Abort(MPI_COMM_WORLD, 2);
      exactFp8Inputs = (dtype == "f8e4m3" || dtype == "f8e5m2") && coll != "ag";
      int error = 2;
      if (dtype == "float")
        error = run<float>(comm, pcclFloat32, coll.c_str(), nWorks, nElts, reg, graph, iters, offset, inPlace);
      else if (dtype == "half")
        error = run<half>(comm, pcclFloat16, coll.c_str(), nWorks, nElts, reg, graph, iters, offset, inPlace);
      else if (dtype == "bf16")
        error = run<__ppu_bfloat16>(comm, pcclBfloat16, coll.c_str(), nWorks, nElts, reg, graph, iters, offset, inPlace);
      else if (dtype == "f8e4m3")
        error = run<__hg_fp8_e4m3>(comm, pcclFloat8e4m3, coll.c_str(), nWorks, nElts, reg, graph, iters, offset, inPlace);
      else if (dtype == "f8e5m2")
        error = run<__hg_fp8_e5m2>(comm, pcclFloat8e5m2, coll.c_str(), nWorks, nElts, reg, graph, iters, offset, inPlace);
      else MPI_Abort(MPI_COMM_WORLD, 2);
      if (error) result = error;
      cases++;
    }
    if (!cases) MPI_Abort(MPI_COMM_WORLD, 2);
  }
#ifndef SYMK_TEST_PUBLIC_ONLY
  else if (!strcmp(argv[1], "kernel-cache-contract")) {
    result = run<float>(comm, pcclFloat32, "ar", works, count, reg, graph, iters, offset, inPlace);
    setenv("PCCL_SYM_KERNEL", "unknown_symk_kernel", 1);
    uint64_t before = comm->kernelNum;
    verifyPath = 0;
    int error = run<float>(comm, pcclFloat32, "ar", works, count, reg, graph, iters, offset, inPlace);
    bool ok = comm->kernelNum > before && comm->kernelNum - before <= pcclDevKernelArgsFifoDepth();
    int kernelIndex = ncclSymkGetKernelIndex(ncclSymkKernelId_AllReduce_RSxLD_AGxST, ncclDevSum, pcclFloat32);
    for (uint64_t index = before; index < comm->kernelNum; index++) {
      auto const& record = comm->kernelInfoMonitorPtr[index%pcclDevKernelArgsFifoDepth()];
      ok &= record.isSymColl && record.symFuncIndex == ncclSymkKernelId_AllReduce_RSxLD_AGxST &&
            record.kernelIdx == index + 1 && record.kernelFn == ncclSymkKernelList[kernelIndex] &&
            record.argsBytes <= sizeof(ncclSymkDevWorkArgs4K);
    }
    int failed = !ok, wrong = 0;
    MPI_Allreduce(&failed, &wrong, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (error || wrong) result = 1;
    if (rankId == 0) printf("SYMK_KERNEL_CACHE graph=%d changed_env=unknown retained=AllReduce_RSxLD_AGxST wrong=%d\n",
                             graph, wrong);
  }
  else if (!strcmp(argv[1], "functional-sweep")) {
    const char* collectives[] = {"ar", "ag", "rs", "rs-avg"};
    int capacity = (sizeof(ncclSymkDevWorkArgs4K) - ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 0)) /
                   sizeof(ncclSymkDevWork);
    int workCounts[] = {1, 4, 7, 17, capacity, capacity + 1, 2*capacity + 1};
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
  }
#endif
  else if (!strcmp(argv[1], "perf-sweep")) {
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
