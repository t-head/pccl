#include <stdio.h>
#include <iostream>
#include "hggc_runtime.h"
#include "pccl.h"
#include "logger.h"
#include "cxxopts.hpp"
#include <pthread.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <atomic>
#include <vector>

/*
 * exit() from several rank threads at once races on the exit handlers and
 * crashes the process, so only the first failing thread reports the error and
 * terminates the process with _exit(); later failing threads wait for it.
 */
static std::atomic<bool> testFailed(false);

static void beginTestFailure() {
  if (testFailed.exchange(true)) {
    for (;;) pause();
  }
}

[[noreturn]] static void endTestFailure(int code) {
  fflush(stdout);
  fflush(stderr);
  _exit(code);
}

#define FAIL_TEST(code, ...)                                                   \
  do {                                                                         \
    beginTestFailure();                                                        \
    ERROR(__VA_ARGS__);                                                        \
    endTestFailure(code);                                                      \
  } while (0)

#undef HGGCCHECK
#define HGGCCHECK(cmd)                                                         \
  do {                                                                         \
    hggcError_t err = cmd;                                                     \
    if (err != hggcSuccess) {                                                  \
      FAIL_TEST(testHggcError, "Test HGGC failure %s:%d '%s'", __FILE__,       \
                __LINE__, hggcGetErrorString(err));                            \
    }                                                                          \
  } while (0)

#undef PCCLCHECK
#define PCCLCHECK(cmd)                                                         \
  do {                                                                         \
    pcclResult_t res = cmd;                                                    \
    if (res != pcclSuccess && res != pcclInProgress) {                         \
      FAIL_TEST(testPcclError, "Test PCCL failure %s:%d '%s'", __FILE__,       \
                __LINE__, pcclGetErrorString(res));                            \
    }                                                                          \
  } while (0)

/*
 * Test for the PCCL memory manager: pcclCommMemStats / pcclCommSuspend /
 * pcclCommResume. One process drives every GPU with one thread per rank.
 * Each rank runs several suspend/resume cycles and checks the counters, the
 * free device memory reported by the runtime, the rejected state transitions,
 * and a real AllReduce after every resume.
 */

struct runArgs {
  size_t count;
  int nGpus;
  int iters;
  int printCycles;
};

/* Counters from pcclCommMemStats plus the runtime's free device memory. */
typedef struct {
  uint64_t trackedTotalBytes;
  uint64_t trackedPersistBytes;
  uint64_t trackedSuspendableBytes;
  uint64_t isSuspended;
  size_t deviceFreeBytes;
} MemSnapshot;

/* One table row of a printed cycle (bytes). */
typedef struct {
  long long trackedTotalBytes;
  long long trackedSuspendableBytes;
  long long trackedPersistBytes;
  long long suspendFreedBytes;  /* free after suspend - free before suspend */
  long long resumeUsedBytes;    /* free while suspended - free after resume */
} CycleRecord;

/* Filled by the thread that owns the rank. */
typedef struct {
  std::vector<CycleRecord> cycles;
} RankSummary;

struct rankThread {
  int rank;
  int nRanks;
  pcclComm_t comm;
  const struct runArgs *args;
  RankSummary *summary;
  pthread_t thread;
};

static pthread_barrier_t barrier;

void parseAndSetupArguments(struct runArgs *args, int argc, char **argv) {
  cxxopts::Options options("pcclCommMemManager", "Test program for PCCL communicator memory manager suspend/resume.");

  int deviceCount = 0;
  HGGCCHECK(hggcGetDeviceCount(&deviceCount));

  options.add_options()
      ("n,count", "Number of float elements per AllReduce buffer.", cxxopts::value<size_t>()->default_value("1073741824"))
      ("g,ngpus", "Number of GPUs (one thread and one rank per GPU).", cxxopts::value<int>()->default_value("2"))
      ("i,iters", "Number of suspend/resume cycles.", cxxopts::value<int>()->default_value("10"))
      ("p,print-cycles", "Print the memory tables of the first N cycles (0 disables, capped at --iters).",
       cxxopts::value<int>()->default_value("1"))
      ("h,help", "Print usage");

  auto params = options.parse(argc, argv);

  if (params.count("help")) {
    std::cout << options.help() << std::endl;
    exit(testSuccess);
  }

  args->count = params["count"].as<size_t>();
  args->nGpus = params["ngpus"].as<int>();
  args->iters = params["iters"].as<int>();
  args->printCycles = params["print-cycles"].as<int>();
  if (args->count == 0) {
    FAIL_TEST(testInternalError, "--count must be greater than 0");
  }
  if (args->nGpus < 1 || args->nGpus > deviceCount) {
    FAIL_TEST(testInternalError, "--ngpus must be in [1, %d], got %d", deviceCount, args->nGpus);
  }
  if (args->iters < 1) {
    FAIL_TEST(testInternalError, "--iters must be greater than 0");
  }
  if (args->printCycles < 0) {
    FAIL_TEST(testInternalError, "--print-cycles must not be negative");
  }
  if (args->printCycles > args->iters) args->printCycles = args->iters;

  INFO("Starting pccl comm mem manager test:");
  INFO("  - GPUs (threads): %d", args->nGpus);
  INFO("  - Suspend/resume cycles: %d", args->iters);
  INFO("  - Printed cycles: %d", args->printCycles);
  INFO("  - Elements per buffer: %zu (%.2f MB)", args->count,
       (double)(args->count * sizeof(float)) / (1024.0 * 1024.0));
}

static MemSnapshot takeMemSnapshot(pcclComm_t comm, int myRank) {
  MemSnapshot snapshot;
  size_t deviceTotalBytes;
  PCCLCHECK(pcclCommMemStats(comm, pcclStatGpuMemTotal, &snapshot.trackedTotalBytes));
  PCCLCHECK(pcclCommMemStats(comm, pcclStatGpuMemPersist, &snapshot.trackedPersistBytes));
  PCCLCHECK(pcclCommMemStats(comm, pcclStatGpuMemSuspend, &snapshot.trackedSuspendableBytes));
  PCCLCHECK(pcclCommMemStats(comm, pcclStatGpuMemSuspended, &snapshot.isSuspended));
  HGGCCHECK(hggcMemGetInfo(&snapshot.deviceFreeBytes, &deviceTotalBytes));

  if (snapshot.trackedTotalBytes != snapshot.trackedPersistBytes + snapshot.trackedSuspendableBytes) {
    FAIL_TEST(testInternalError, "Rank %d: total != persist + suspendable", myRank);
  }
  return snapshot;
}

/* Byte counters describe what the communicator owns, so they must never change. */
static void checkTrackedBytesUnchanged(const MemSnapshot *baseline, const MemSnapshot *current,
                                       int myRank, int iter, const char *when) {
  if (current->trackedTotalBytes != baseline->trackedTotalBytes ||
      current->trackedPersistBytes != baseline->trackedPersistBytes ||
      current->trackedSuspendableBytes != baseline->trackedSuspendableBytes) {
    FAIL_TEST(testInternalError, "Rank %d cycle %d: tracked byte counters changed %s", myRank, iter + 1, when);
  }
}

static long long toKib(long long bytes) { return bytes / 1024; }

static void printTrackedMemoryTable(int nRanks, int cycle, int iters, const RankSummary *rows) {
  const char *separator = "----------------------------------------"
                          "------------------------------------";
  printf("\nCycle %d/%d: tracked memory per rank (KiB, sampled before suspend):\n", cycle + 1, iters);
  printf("%s\n", separator);
  printf("| %-6s | %13s | %29s | %15s |\n", "rank #", "total KiB",
         "suspendable KiB (offload+scr)", "persist KiB");
  printf("%s\n", separator);
  for (int rank = 0; rank < nRanks; rank++) {
    const CycleRecord *row = &rows[rank].cycles[cycle];
    printf("| %6d | %13lld | %29lld | %15lld |\n", rank,
           toKib(row->trackedTotalBytes),
           toKib(row->trackedSuspendableBytes),
           toKib(row->trackedPersistBytes));
  }
  printf("%s\n", separator);
  fflush(stdout);
}

static void printSuspendResumeTable(int nRanks, int cycle, int iters, const RankSummary *rows) {
  const char *separator = "----------------------------------------"
                          "----------------------------------------"
                          "-------------------";
  printf("\nCycle %d/%d: suspendable memory, manager (pcclCommMemStats) vs runtime (hggcMemGetInfo) (KiB):\n",
         cycle + 1, iters);
  printf("  freed = free after suspend - free before suspend, used = free while suspended - free after resume,\n");
  printf("  diff = runtime - manager\n");
  printf("%s\n", separator);
  printf("| %-6s | %11s | %17s | %16s | %15s | %15s |\n", "rank #", "manager KiB",
         "suspend freed KiB", "suspend diff KiB", "resume used KiB", "resume diff KiB");
  printf("%s\n", separator);
  for (int rank = 0; rank < nRanks; rank++) {
    const CycleRecord *row = &rows[rank].cycles[cycle];
    long long manager = toKib(row->trackedSuspendableBytes);
    long long freed = toKib(row->suspendFreedBytes);
    long long used = toKib(row->resumeUsedBytes);
    printf("| %6d | %11lld | %17lld | %+16lld | %15lld | %+15lld |\n", rank,
           manager, freed, freed - manager, used, used - manager);
  }
  printf("%s\n", separator);
  fflush(stdout);
}

/*
 * Every rank contributes (myRank + round), so the expected sum depends on the
 * round and a stale result from an earlier round cannot pass.
 */
static void runAllReduce(pcclComm_t comm, hggcStream_t stream, float *sendbuff, float *recvbuff,
                         float *hostbuff, size_t count, int myRank, int nRanks, int round) {
  size_t sizeBytes = count * sizeof(float);

  for (size_t i = 0; i < count; i++) {
    hostbuff[i] = (float)(myRank + round);
  }
  HGGCCHECK(hggcMemcpy(sendbuff, hostbuff, sizeBytes, hggcMemcpyHostToDevice));
  HGGCCHECK(hggcMemset(recvbuff, 0, sizeBytes));

  PCCLCHECK(pcclAllReduce(sendbuff, recvbuff, count, pcclFloat, pcclSum, comm, stream));
  HGGCCHECK(hggcStreamSynchronize(stream));

  HGGCCHECK(hggcMemcpy(hostbuff, recvbuff, sizeBytes, hggcMemcpyDeviceToHost));
  float expected = (float)(nRanks * (nRanks - 1) / 2 + nRanks * round);
  for (size_t i = 0; i < count; i++) {
    if (hostbuff[i] != expected) {
      FAIL_TEST(testInternalError, "Rank %d: AllReduce round %d index %zu got %.1f expected %.1f",
            myRank, round, i, hostbuff[i], expected);
    }
  }
  if (myRank == 0) {
    INFO("  AllReduce round %d verified (expected %.1f)", round, expected);
  }
}

static void *runRank(void *arg) {
  struct rankThread *t = (struct rankThread *)arg;
  int myRank = t->rank;
  int nRanks = t->nRanks;
  pcclComm_t comm = t->comm;
  RankSummary *summary = t->summary;

  HGGCCHECK(hggcSetDevice(myRank));

  // user buffers; hggcMalloc memory is not tracked by the memory manager
  size_t count = t->args->count;
  size_t sizeBytes = count * sizeof(float);
  float *sendbuff, *recvbuff;
  hggcStream_t s;
  HGGCCHECK(hggcMalloc((void **)&sendbuff, sizeBytes));
  HGGCCHECK(hggcMalloc((void **)&recvbuff, sizeBytes));
  HGGCCHECK(hggcStreamCreate(&s));
  float *hostbuff = (float *)malloc(sizeBytes);
  if (hostbuff == NULL) {
    FAIL_TEST(testInternalError, "Rank %d: failed to allocate host buffer", myRank);
  }

  // warm-up AllReduce so PCCL allocates its internal buffers
  if (myRank == 0) INFO("Warm-up AllReduce:");
  runAllReduce(comm, s, sendbuff, recvbuff, hostbuff, count, myRank, nRanks, /*round=*/1);

  MemSnapshot baseline = takeMemSnapshot(comm, myRank);
  if (baseline.isSuspended != 0) {
    FAIL_TEST(testInternalError, "Rank %d: new communicator reports suspended=1", myRank);
  }

  for (int iter = 0; iter < t->args->iters; iter++) {
    if (myRank == 0) INFO("Suspend/resume cycle %d/%d:", iter + 1, t->args->iters);

    MemSnapshot active = takeMemSnapshot(comm, myRank);
    if (active.isSuspended != 0) {
      FAIL_TEST(testInternalError, "Rank %d cycle %d: communicator reports suspended=1 before suspend", myRank, iter + 1);
    }
    checkTrackedBytesUnchanged(&baseline, &active, myRank, iter, "before suspend");

    // every rank must suspend with no collective in flight
    PCCLCHECK(pcclCommSuspend(comm, PCCL_SUSPEND_MEM));
    // peers in this process map our buffers, so sample only after all of them unmapped
    pthread_barrier_wait(&barrier);

    MemSnapshot suspended = takeMemSnapshot(comm, myRank);
    if (suspended.isSuspended != 1) {
      FAIL_TEST(testInternalError, "Rank %d cycle %d: suspended flag not set after pcclCommSuspend", myRank, iter + 1);
    }
    checkTrackedBytesUnchanged(&baseline, &suspended, myRank, iter, "during suspend");
    // Allocator granularity moves the number, so only require half.
    long long suspendRuntimeDelta = (long long)suspended.deviceFreeBytes - (long long)active.deviceFreeBytes;
    if (suspended.trackedSuspendableBytes > 0 &&
        suspendRuntimeDelta < (long long)(suspended.trackedSuspendableBytes / 2)) {
      FAIL_TEST(testInternalError, "Rank %d cycle %d: device free memory grew by %lld bytes, manager claims %llu suspendable",
            myRank, iter + 1, suspendRuntimeDelta, (unsigned long long)suspended.trackedSuspendableBytes);
    }

    if (pcclCommSuspend(comm, PCCL_SUSPEND_MEM) != pcclInvalidUsage) {
      FAIL_TEST(testInternalError, "Rank %d cycle %d: second pcclCommSuspend was not rejected", myRank, iter + 1);
    }
    if (myRank == 0) INFO("  Suspend verified, duplicate suspend rejected with pcclInvalidUsage");

    // all ranks must reach resume before any collective is issued
    pthread_barrier_wait(&barrier);
    PCCLCHECK(pcclCommResume(comm));
    pthread_barrier_wait(&barrier);

    MemSnapshot resumed = takeMemSnapshot(comm, myRank);
    if (resumed.isSuspended != 0) {
      FAIL_TEST(testInternalError, "Rank %d cycle %d: suspended flag still set after pcclCommResume", myRank, iter + 1);
    }
    checkTrackedBytesUnchanged(&baseline, &resumed, myRank, iter, "after resume");
    long long resumeRuntimeDelta = (long long)resumed.deviceFreeBytes - (long long)suspended.deviceFreeBytes;
    if (resumed.trackedSuspendableBytes > 0 && resumeRuntimeDelta >= 0) {
      FAIL_TEST(testInternalError, "Rank %d cycle %d: device free memory did not shrink after resume", myRank, iter + 1);
    }

    if (pcclCommResume(comm) != pcclInvalidUsage) {
      FAIL_TEST(testInternalError, "Rank %d cycle %d: pcclCommResume on an active communicator was not rejected", myRank, iter + 1);
    }
    if (myRank == 0) INFO("  Resume verified, resume on active communicator rejected with pcclInvalidUsage");

    if (iter < t->args->printCycles) {
      summary->cycles.push_back({(long long)active.trackedTotalBytes,
                                 (long long)active.trackedSuspendableBytes,
                                 (long long)active.trackedPersistBytes,
                                 suspendRuntimeDelta, -resumeRuntimeDelta});
    }

    // input differs every cycle so a stale result cannot pass
    runAllReduce(comm, s, sendbuff, recvbuff, hostbuff, count, myRank, nRanks, /*round=*/2 + iter);
  }

  // cleanup; the memory manager releases what it still tracks on destroy
  free(hostbuff);
  HGGCCHECK(hggcFree(sendbuff));
  HGGCCHECK(hggcFree(recvbuff));
  HGGCCHECK(hggcStreamDestroy(s));
  PCCLCHECK(pcclCommFinalize(comm));
  PCCLCHECK(pcclCommDestroy(comm));
  return NULL;
}

int main(int argc, char* argv[])
{
  struct runArgs args;
  parseAndSetupArguments(&args, argc, argv);
  int nRanks = args.nGpus;

  // one communicator per GPU; rank i uses device i
  pcclUniqueId id;
  std::vector<pcclComm_t> comms(nRanks);
  PCCLCHECK(pcclGetUniqueId(&id));
  PCCLCHECK(pcclGroupStart());
  for (int i = 0; i < nRanks; i++) {
    HGGCCHECK(hggcSetDevice(i));
    PCCLCHECK(pcclCommInitRank(&comms[i], nRanks, id, i));
  }
  PCCLCHECK(pcclGroupEnd());

  std::vector<RankSummary> summaries(nRanks);
  std::vector<struct rankThread> threads(nRanks);
  pthread_barrier_init(&barrier, NULL, nRanks);
  for (int i = 0; i < nRanks; i++) {
    threads[i].rank = i;
    threads[i].nRanks = nRanks;
    threads[i].comm = comms[i];
    threads[i].args = &args;
    threads[i].summary = &summaries[i];
    if (pthread_create(&threads[i].thread, NULL, runRank, &threads[i]) != 0) {
      FAIL_TEST(testInternalError, "Failed to create thread for rank %d", i);
    }
  }
  // Any failing rank exits the process, so joining every thread means every rank succeeded
  for (int i = 0; i < nRanks; i++) {
    pthread_join(threads[i].thread, NULL);
  }
  pthread_barrier_destroy(&barrier);

  for (int cycle = 0; cycle < args.printCycles; cycle++) {
    printTrackedMemoryTable(nRanks, cycle, args.iters, summaries.data());
    printSuspendResumeTable(nRanks, cycle, args.iters, summaries.data());
  }
  printf("\n");
  INFO("All %d ranks Success", nRanks);
  return 0;
}
