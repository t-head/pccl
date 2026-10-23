//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

// ltOki dev0 -> dev1 ICN benchmark; see --help for parameters.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <climits>

#include "hggc.h"
#include "hggc_runtime.h"

#include "lt_oki.h"
#include "lt_oki_device.cuh"

#define HGCHECK(cmd) do {                                                     \
    HGresult e_ = (cmd);                                                      \
    if (e_ != HGGC_SUCCESS) {                                                 \
      const char* s_ = hggcGetErrorString((hggcError_t)e_);                    \
      fprintf(stderr, "HG error %s:%d '%s' returned %d (%s)\n",               \
              __FILE__, __LINE__, #cmd, (int)e_, s_ ? s_ : "?");              \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
} while (0)

#define HGGCCHECK(cmd) do {                                                      \
    hggcError_t e_ = (cmd);                                                   \
    if (e_ != hggcSuccess) {                                                  \
      fprintf(stderr, "hggc error %s:%d '%s' returned %s\n",                  \
              __FILE__, __LINE__, #cmd, hggcGetErrorString(e_));              \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
} while (0)

#define OKICHECK(cmd) do {                                                    \
    ltOkiStatus_t s_ = (cmd);                                                 \
    if (s_ != LT_OKI_SUCCESS) {                                               \
      fprintf(stderr, "ltOki error %s:%d '%s' returned %d\n",                 \
              __FILE__, __LINE__, #cmd, (int)s_);                             \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
} while (0)

namespace {

constexpr size_t kWindowBytes = 256ull << 20;
constexpr int kMaxQueueId = 16;
constexpr size_t kRemoteFenceBytes = kMaxQueueId * sizeof(uint64_t);

constexpr double kPhysicalLimitGbps = 400.0;

hggcStream_t g_stream;

struct StreamTimer {
  hggcEvent_t start, stop;
  StreamTimer() {
    HGGCCHECK(hggcEventCreate(&start));
    HGGCCHECK(hggcEventCreate(&stop));
  }
  ~StreamTimer() {
    HGGCCHECK(hggcEventDestroy(start));
    HGGCCHECK(hggcEventDestroy(stop));
  }
  void begin() { HGGCCHECK(hggcEventRecord(start, g_stream)); }
  double end() {
    HGGCCHECK(hggcEventRecord(stop, g_stream));
    HGGCCHECK(hggcEventSynchronize(stop));
    float ms = 0;
    HGGCCHECK(hggcEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

double run_memcpy_baseline(HGdeviceptr src, HGdeviceptr dst, int reps) {
  printf("\n=== baseline: hggcMemcpy (runtime copy engine) dev0 -> dev1 ===\n");
  printf("  bytes(MiB)  time(ms)  GB/s\n");
  printf("  ----------  --------  ---------\n");
  StreamTimer timer;
  double best = 0;
  for (size_t bytes = 1ull << 20; bytes <= kWindowBytes; bytes <<= 2) {

    HGGCCHECK(hggcMemcpyAsync((void*)(uintptr_t)dst, (void*)(uintptr_t)src, bytes,
                           hggcMemcpyDeviceToDevice, g_stream));
    HGGCCHECK(hggcStreamSynchronize(g_stream));

    double bestMs = 1e30;
    for (int rep = 0; rep < reps; rep++) {
      timer.begin();
      HGGCCHECK(hggcMemcpyAsync((void*)(uintptr_t)dst, (void*)(uintptr_t)src,
                             bytes, hggcMemcpyDeviceToDevice, g_stream));
      const double ms = timer.end();
      if (!std::isfinite(ms) || ms <= 0) { fprintf(stderr, "invalid event duration\n"); exit(EXIT_FAILURE); }
      if (ms < bestMs) bestMs = ms;
    }
    const double gbps = bytes / bestMs / 1e6;
    printf("  %10zu  %8.3f  %9.2f\n", bytes >> 20, bestMs, gbps);
    if (gbps > best) best = gbps;
  }
  printf("  baseline peak: %.2f GB/s\n", best);
  return best;
}

uint64_t covered_bytes(int rounds, int wqesPerRound, uint32_t wqeBytes,
                       uint64_t segBytes) {
  const uint64_t written = (uint64_t)rounds * wqesPerRound * wqeBytes;
  return written < segBytes ? written : segBytes;
}

bool verify_slices(const uint8_t* dstBase, const std::vector<uint32_t>& srcHost,
                   int workers, uint64_t segBytes, uint64_t bytes,
                   uint32_t wqeBytes, const char* phase) {
  if (bytes == 0) return true;

  constexpr size_t kChunk = 4ull << 20;
  std::vector<uint8_t> buf(kChunk);
  const uint8_t* srcBytes = (const uint8_t*)srcHost.data();
  for (int w = 0; w < workers; w++) {
    const uint64_t sliceOff = (uint64_t)w * segBytes;
    for (uint64_t done = 0; done < bytes; ) {
      const size_t n =
          (size_t)((bytes - done < kChunk) ? (bytes - done) : kChunk);
      HGGCCHECK(hggcMemcpy(buf.data(),
                        (void*)(uintptr_t)(dstBase + sliceOff + done), n,
                        hggcMemcpyDeviceToHost));
      const uint8_t* want = srcBytes + sliceOff + done;
      if (memcmp(buf.data(), want, n) != 0) {
        size_t i = 0;
        while (i < n && buf[i] == want[i]) i++;
        const uint64_t off = done + i;
        fprintf(stderr,
                "  %s VERIFY FAIL: worker %d (slice @ %llu MiB), offset %llu "
                "= WQE #%llu + %llu B, got %02x want %02x\n",
                phase, w, (unsigned long long)(sliceOff >> 20),
                (unsigned long long)off,
                (unsigned long long)(off / wqeBytes),
                (unsigned long long)(off % wqeBytes),
                (unsigned)buf[i], (unsigned)want[i]);
        return false;
      }
      done += n;
    }
  }
  return true;
}

}

// Preserve die-interleaved slice indices when grouping warps by die.
__host__ __device__ int bench_slice_slot(int block, int warp,
                                        int warpsPerBlock,
                                        int dieNum, int diePerBlock) {
  return diePerBlock ? warp * dieNum + block :
      block * warpsPerBlock + warp;
}

__global__ void bench_kernel(ltOkiQP_t* qps, const uint64_t* stepBase,
                             const uint8_t* src,
                             uint8_t* dst, uint64_t segBytes,
                             uint32_t wqeBytes, int wqesPerRound, int rounds,
                             int roundsInFlight,
                             int ringsPerDie, int dieNum,
                             int warpsPerBlock, int diePerBlock = 0,
                             int phaseMode = 0, uint32_t* prepared = nullptr,
                             int roundOffset = 0) {
  const int tid = threadIdx.x;

  // OKI requires one QP per warp.
  const int warpId = tid / 32;
  const int laneInWarp = tid % 32;

  const bool worker = laneInWarp == 0;

  const int slotIdx = bench_slice_slot(blockIdx.x, warpId, warpsPerBlock,
                                      dieNum, diePerBlock);
  const int qpSlot = slotIdx;

  const int die = qpSlot % dieNum;
  const int ring = qpSlot / dieNum;

  const int qpIdx = die * ringsPerDie + ring;

  ltOkiQP_t qp = qps[qpIdx];
  volatile uint64_t* cq = ltOkiQPGetCQ(qp);

  // Fence targets must increase across launches to avoid stale-CQ completion.
  const uint64_t base = stepBase[qpIdx];
  const uint64_t srcBase = (uint64_t)src + (uint64_t)slotIdx * segBytes;
  const uint64_t dstBase = (uint64_t)dst + (uint64_t)slotIdx * segBytes;


  // phaseMode: 0 = normal, 1 = prepare, 2 = submit the prepared batch.
  if (phaseMode == 2) {
    if (worker) {
      ltOkiPostSend(qp, prepared[slotIdx*2], prepared[slotIdx*2+1]);
      ltOkiRingDB(qp);
      ltOkiPollCQ((uint64_t*)cq, base + 1);
    }
    __syncwarp();
    return;
  }
  for (int r = 0; r < rounds; r++) {
    if (worker) {
      uint32_t skip = 0;
      uint32_t want = (uint32_t)wqesPerRound + 1 + (wqesPerRound > 0);
      const uint32_t slot = ltOkiReserveWqeSlots(qp, want, skip);
      uint64_t off = (((uint64_t)r + roundOffset) * wqesPerRound * wqeBytes) % segBytes;
      for (int i = 0; i < wqesPerRound; i++) {
        ltOkiQPWriteIcnDmaCopyWqe(qp, srcBase + off, dstBase + off,
                                  wqeBytes, /*Local2Remote=*/true);
        off += wqeBytes;
        if (off >= segBytes) off -= segBytes;
      }

      if (wqesPerRound > 0) {
        const uint64_t remoteFence = (uint64_t)dst + kWindowBytes + qpIdx * sizeof(uint64_t);
        ltOkiQPWriteIcnFenceWqe(qp, remoteFence, base + (uint64_t)(r + 1), false);
      }

      // The SDK fence upload argument must use the QP die index.
      ltOkiQPWriteFenceWqe(qp, (uint64_t)cq, base + (uint64_t)(r + 1),
                           /*dieIdx=*/die, /*fence64=*/true);

      if (phaseMode == 1) {

        // Reserve may update want; submit the exact returned slot/count.
        prepared[slotIdx*2] = slot;
        prepared[slotIdx*2+1] = want;
      } else {
        ltOkiPostSend(qp, slot, want);
        ltOkiRingDB(qp);

        if ((r + 1) % roundsInFlight == 0) {
          ltOkiPollCQ((uint64_t*)cq, base + (uint64_t)(r + 1));
        }
      }
    }

    // Keep QPs independent: a block barrier would couple their CQ waits.
    __syncwarp();
  }

  if (worker && phaseMode != 1) ltOkiPollCQ((uint64_t*)cq, base + (uint64_t)rounds);
  __syncwarp();
}

__global__ void report_cq_kernel(ltOkiQP_t* qps, uint64_t* out, int numQps) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  for (int q = 0; q < numQps; q++) out[q] = (uint64_t)ltOkiQPGetCQ(qps[q]);
}

__global__ void report_cqval_kernel(ltOkiQP_t* qps, uint64_t* out, int numQps) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  for (int q = 0; q < numQps; q++) {
    out[q] = *(volatile uint64_t*)ltOkiQPGetCQ(qps[q]);
  }
}

struct RunResult {
  int blocks;
  int workers;
  double gbps;
};

enum class MeasurementMode {
  BandwidthWithAssembly,
  BandwidthWithoutAssembly,
  LatencyWithAssembly,
  LatencyWithoutAssembly
};

const char* mode_name(MeasurementMode mode) {
  switch (mode) {
    case MeasurementMode::BandwidthWithAssembly: return "bandwidth-with-assembly";
    case MeasurementMode::BandwidthWithoutAssembly: return "bandwidth-without-assembly";
    case MeasurementMode::LatencyWithAssembly: return "latency-with-assembly";
    case MeasurementMode::LatencyWithoutAssembly: return "latency-without-assembly";
  }
  return "unknown";
}

struct Options {
  MeasurementMode mode = MeasurementMode::BandwidthWithAssembly;
  uint32_t cmdBytes = 4096;
  int copies = 64, rounds = 1024, reps = 3, inFlight = 0;
  int blocks = 0, warps = 1;
  bool perDie = false;
};

bool argument_error(const std::string& message) {
  fprintf(stderr, "%s (see --help)\n", message.c_str());
  return false;
}

bool parse_unsigned(const std::string& value, uint64_t limit, uint64_t& result,
                    bool byteUnits = false) {
  if (value.empty()) return false;
  size_t pos = 0;
  uint64_t n = 0;
  while (pos < value.size() && value[pos] >= '0' && value[pos] <= '9') {
    const unsigned digit = value[pos++] - '0';
    if (digit > limit || n > (limit - digit) / 10) return false;
    n = n * 10 + digit;
  }
  if (pos == 0) return false;
  uint64_t scale = 1;
  const std::string suffix = value.substr(pos);
  if (!suffix.empty()) {
    if (!byteUnits) return false;
    if (suffix == "K" || suffix == "k" || suffix == "KiB") scale = 1024;
    else if (suffix == "M" || suffix == "m" || suffix == "MiB") scale = 1024 * 1024;
    else return false;
  }
  if (n > limit / scale) return false;
  result = n * scale;
  return true;
}

void print_help(const char* program) {
  printf("Usage: %s [named options]\n\n", program);
  puts(R"HELP(Modes (--mode NAME; default bandwidth-with-assembly):
  bandwidth-with-assembly
    Aggregate GB/s using hggcEvent timestamps on the execution stream.
    Includes reserve, WQE assembly, PostSend, RingDB and CQ completion.
  bandwidth-without-assembly
    Aggregate GB/s using hggcEvent timestamps on the execution stream.
    A separate prepare kernel assembles a fresh batch before each repetition;
    prepare and verification are outside timing. Requires rounds=1 and
    copies-per-round=1..2048. Includes submission and CQ completion; excludes host synchronization time.
  latency-with-assembly
    Stream event latency for one kernel running one round across all QPs,
    including reserve, WQE assembly, PostSend, RingDB and PollCQ; reports us.
  latency-without-assembly
    Prepare each round outside timing, then measure its submit/completion
    kernel with stream events. Requires copies-per-round=0..2048.

  Without-assembly modes still execute assembly, outside their timing window.
  Latency modes launch and synchronize each round; samples cover all QPs,
  not individual QPs. Bandwidth modes time all rounds in one kernel.
  All durations use hggcEvent; verification is outside the event interval.

Workload:
  --cmd-bytes N          Copy size; default 4K. Decimal bytes or K/KiB, M/MiB
                        suffixes (K=1024). Must fit the per-QP address slice.
                        Tested sizes: 4K, 8K, 16K, 32K, 64K, 128K.
  --copies-per-round N   Copy WQEs per QP per round; default 64. Copies are
                        followed by an ICN fence (relax=false), then a CQ fence.
                        0 means CQ fence only and
                        is allowed only in latency modes.
  --rounds N            Rounds per QP per repetition; positive, default 1024.
                        Default 1 for bandwidth-without-assembly; an explicit
                        value other than 1 is rejected in that mode.
  --reps N              Repetitions per configuration; positive, default 3.
                        Each repetition is verified. Bandwidth modes report
                        each repetition and min/median/max GB/s and stream event time.
  --inflight-rounds N|auto
                        Rounds submitted between CQ waits; default auto.
                        Auto = max(1, QP_DEPTH/nextPow2(copies+2)/8) for copies>0;
                        a zero-copy round reserves one CQ fence WQE.
                        Explicit values must be positive and no greater than
                        that conservative queue budget. Latency modes and
                        bandwidth-without-assembly use 1; explicit N!=1 fails.

Layout:
  --blocks N|sweep       Positive block count, or sweep (also 0); default sweep.
                        Sweep runs 1,2,4,8,... within available QPs.
                        Positive N runs exactly N blocks, including 3 or 5.
  --warps-per-block N   Positive warps/block; default 1. One QP per warp,
                        lane 0 submits. Threads/block = 32*N.
                        Active QPs = blocks * warps-per-block; must fit the
                        available QPs and device threads/block limit.
  --mapping interleaved|per-die
                        Default interleaved: logical slots alternate dies,
                        then rings. QP selection always starts at slot 0.
                        per-die: block0 -> die0, block1 -> die1, warp -> ring.
                        Defaults to 2 blocks and 8 warps/block; conflicting
                        explicit values fail. Requires two dies of 8 rings.
                        It does not pin execution to a die's SMs.

Other:
  --help                Print help without initializing the device.

Syntax and results:
  Use --option value or --option=value. Duplicate/unknown options, positional
  arguments, malformed numbers, negative values and overflow are rejected.
  All performance modes wait for CQ completion. Old --preassemble is replaced
  by --mode bandwidth-without-assembly.
  Per-repetition bytes = QPs * copies-per-round * rounds * cmd-bytes.
  GB/s means decimal bytes/sec, not Gbit/s. Buffers wrap within 256 MiB.
  Exit: 0 success, 1 configuration/API error, 2 data/CQ verification failure.
)HELP");
  printf("copies-per-round range: 1..%d, or 0 for latency modes.\n\n"
         "Examples:\n"
         "  %s --cmd-bytes 4K --mapping per-die\n"
         "  %s --mode bandwidth-without-assembly --cmd-bytes 4K "
         "--mapping per-die --copies-per-round 2048\n"
         "  %s --mode latency-without-assembly --mapping per-die --rounds 128\n"
         "  %s --blocks 3 --warps-per-block 2 --reps 5\n",
         (int)(LT_OKI_QP_DEPTH / 8 - 2), program, program, program, program);
}

bool parse_options(int argc, char** argv, Options& o, bool& help) {
  std::vector<std::string> seen;
  bool roundsSet = false, blocksSet = false, warpsSet = false;
  bool flightSet = false;
  for (int a = 1; a < argc; ++a) {
    const std::string arg = argv[a];
    const size_t eq = arg.find('=');
    const std::string key = arg.substr(0, eq);
    if (std::find(seen.begin(), seen.end(), key) != seen.end())
      return argument_error("duplicate option: " + key);
    seen.push_back(key);
    if (key == "--help") {
      if (eq != std::string::npos) return argument_error(key + " takes no value");
      help = true;
      continue;
    }
    if (key != "--mode" && key != "--mapping" && key != "--cmd-bytes" &&
        key != "--copies-per-round" && key != "--rounds" && key != "--reps" &&
        key != "--inflight-rounds" && key != "--blocks" && key != "--warps-per-block")
      return argument_error("unknown option or positional argument: " + key);
    if (eq == std::string::npos && a + 1 == argc)
      return argument_error("missing value for " + key);
    const std::string value = eq == std::string::npos ? argv[++a] : arg.substr(eq + 1);
    if (key == "--mode") {
      if (value == "bandwidth-with-assembly") o.mode = MeasurementMode::BandwidthWithAssembly;
      else if (value == "bandwidth-without-assembly") o.mode = MeasurementMode::BandwidthWithoutAssembly;
      else if (value == "latency-with-assembly") o.mode = MeasurementMode::LatencyWithAssembly;
      else if (value == "latency-without-assembly") o.mode = MeasurementMode::LatencyWithoutAssembly;
      else return argument_error("unknown mode: " + value);
    } else if (key == "--mapping") {
      if (value != "interleaved" && value != "per-die")
        return argument_error("unknown mapping: " + value);
      o.perDie = value == "per-die";
    } else {
      uint64_t n = 0;
      const bool autoFlight = key == "--inflight-rounds" && value == "auto";
      const bool sweep = key == "--blocks" && value == "sweep";
      const uint64_t limit = key == "--cmd-bytes" ? kWindowBytes : INT_MAX;
      if (!autoFlight && !sweep && !parse_unsigned(value, limit, n, key == "--cmd-bytes"))
        return argument_error("invalid value for " + key + ": " + value);
      if (key == "--cmd-bytes") o.cmdBytes = (uint32_t)n;
      else if (key == "--blocks") { o.blocks = (int)n; blocksSet = true; }
      else if (key == "--warps-per-block") { o.warps = (int)n; warpsSet = true; }
      else {
        if (key == "--copies-per-round") o.copies = (int)n;
        else if (key == "--rounds") { o.rounds = (int)n; roundsSet = true; }
        else if (key == "--reps") o.reps = (int)n;
        else {
          if (!autoFlight && n == 0) return argument_error("use --inflight-rounds auto or a positive integer");
          o.inFlight = (int)n;
          flightSet = !autoFlight;
        }
      }
    }
  }
  if (help) return true;
  const bool prepared = o.mode == MeasurementMode::BandwidthWithoutAssembly;
  const bool withoutAssembly = prepared || o.mode == MeasurementMode::LatencyWithoutAssembly;
  const bool latency = o.mode == MeasurementMode::LatencyWithAssembly ||
                       o.mode == MeasurementMode::LatencyWithoutAssembly;
  if (prepared && !roundsSet) o.rounds = 1;
  if (o.perDie) {
    if ((blocksSet && o.blocks != 2) || (warpsSet && o.warps != 8))
      return argument_error("--mapping per-die requires --blocks 2 and --warps-per-block 8");
    o.blocks = 2; o.warps = 8;
  }
  if (!o.cmdBytes || o.rounds < 1 || o.reps < 1 || o.warps < 1 ||
      o.copies > LT_OKI_QP_DEPTH / 8 - 2 || (!latency && o.copies == 0))
    return argument_error("invalid workload: positive bytes/rounds/reps/warps required; check copies range");
  if (prepared && o.rounds != 1)
    return argument_error("bandwidth-without-assembly requires rounds=1");
  if (withoutAssembly && o.copies > 2048)
    return argument_error("without-assembly modes require copies-per-round <= 2048");
  uint32_t slotCost = 1;
  while (slotCost < (uint32_t)o.copies + 1 + (o.copies > 0)) slotCost <<= 1;
  const int budget = std::max(1, (int)(LT_OKI_QP_DEPTH / slotCost / 8));
  if (prepared || latency) {
    if (flightSet && o.inFlight != 1)
      return argument_error("selected mode requires --inflight-rounds 1 (or auto)");
    o.inFlight = 1;
  } else {
    if (o.inFlight > budget)
      return argument_error("--inflight-rounds exceeds conservative queue budget " + std::to_string(budget));
    if (!o.inFlight) o.inFlight = budget;
  }
  return true;
}

int main(int argc, char** argv) {
  Options options;
  bool help = false;
  if (!parse_options(argc, argv, options, help)) return 1;
  if (help) { print_help(argv[0]); return 0; }
  const char* modeName = mode_name(options.mode);
  const bool preassemble = options.mode == MeasurementMode::BandwidthWithoutAssembly ||
                           options.mode == MeasurementMode::LatencyWithoutAssembly;
  const int timingMode = options.mode == MeasurementMode::LatencyWithoutAssembly ? 1 :
                         options.mode == MeasurementMode::LatencyWithAssembly ? 2 : 0;
  const uint32_t wqeBytes = options.cmdBytes;
  const int wqesPerRound = options.copies, rounds = options.rounds, reps = options.reps;
  const int roundsInFlight = options.inFlight, requestedBlocks = options.blocks;
  const int warpsPerBlock = options.warps, diePerBlock = options.perDie ? 1 : 0;
  setvbuf(stdout, nullptr, _IOLBF, 0);
  printf("mode=%s\n", modeName);

  HGGCCHECK(hggcSetDevice(0));
  int maxThreadsPerBlock = 0;
  HGGCCHECK(hggcDeviceGetAttribute(&maxThreadsPerBlock,
                                hggcDevAttrMaxThreadsPerBlock, 0));
  if (warpsPerBlock > maxThreadsPerBlock / 32) {
    fprintf(stderr, "warpsPerBlock=%d exceeds device limit %d\n",
            warpsPerBlock, maxThreadsPerBlock / 32);
    return 1;
  }
  HGGCCHECK(hggcStreamCreate(&g_stream));

  int canPeer = 0;
  HGGCCHECK(hggcDeviceCanAccessPeer(&canPeer, 0, 1));
  if (canPeer) HGGCCHECK(hggcDeviceEnablePeerAccess(1, 0));
  printf("peer access 0->1: %s\n", canPeer ? "yes" : "no");

  // Map [src on dev0 | dst on dev1] into dev0 VA.
  size_t gran = 0;
  {
    HGmemAllocationProp prop = {};
    prop.type = HG_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    prop.requestedHandleTypes = HG_MEM_HANDLE_TYPE_NONE;
    HGCHECK(hgMemGetAllocationGranularity(
        &gran, &prop, HG_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  }
  size_t alloc = (kWindowBytes + gran - 1) / gran * gran;
  const size_t dstAlloc = (kWindowBytes + kRemoteFenceBytes + gran - 1) / gran * gran;
  size_t spanSize = alloc + dstAlloc;
  HGdeviceptr span = 0;
  HGCHECK(hgMemAddressReserve(&span, spanSize, 0, 0, 0));
  for (int loc = 0; loc < 2; loc++) {
    HGmemAllocationProp prop = {};
    prop.type = HG_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = loc;
    prop.requestedHandleTypes = HG_MEM_HANDLE_TYPE_NONE;
    HGmemGenericAllocationHandle h;
    const size_t bytes = loc == 0 ? alloc : dstAlloc;
    HGCHECK(hgMemCreate(&h, bytes, &prop, 0));
    HGCHECK(hgMemMap(span + loc * alloc, bytes, 0, h, 0));
    HGCHECK(hgMemRelease(h));
  }
  HGmemAccessDesc acc = {};
  acc.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
  acc.location.id = 0;
  acc.flags = HG_MEM_ACCESS_FLAGS_PROT_READWRITE;
  HGCHECK(hgMemSetAccess(span, spanSize, &acc, 1));

  const uint8_t* srcPtr = (const uint8_t*)(uintptr_t)span;
  uint8_t* dstPtr = (uint8_t*)(uintptr_t)(span + alloc);
  HGGCCHECK(hggcMemset(dstPtr + kWindowBytes, 0, kRemoteFenceBytes));

  std::vector<uint32_t> srcHost(kWindowBytes / 4);
  for (size_t i = 0; i < srcHost.size(); i++) srcHost[i] = (uint32_t)i;
  HGGCCHECK(hggcMemcpy((void*)(uintptr_t)span, srcHost.data(), kWindowBytes,
                    hggcMemcpyHostToDevice));

  if (!timingMode) run_memcpy_baseline(span, span + alloc, reps);

  // Scrub with a value absent from the source pattern to expose missing copies.
  HGGCCHECK(hggcMemset((void*)(uintptr_t)(span + alloc), 0xFF, kWindowBytes));

  ltOkiDevice_t dev;
  OKICHECK(ltOkiOpenDevice(&dev, 0));
  ltOkiQP_t qpArr[kMaxQueueId] = {};
  int numQps = 0;
  for (int q = 0; q < kMaxQueueId; q++) {
    ltOkiStatus_t s = ltOkiCreateQP(&qpArr[numQps], dev, /*isRemote=*/true,
                                    /*queueId=*/q, LT_OKI_QP_DEPTH);
    if (s != LT_OKI_SUCCESS) {
      printf("queueId=%d unavailable (status=%d), stopping probe\n", q, (int)s);
      break;
    }
    numQps++;
  }
  printf("remote ICN QPs: %d (queueId 0..%d)\n", numQps, numQps - 1);
  if (numQps == 0) {
    fprintf(stderr, "no remote QP available\n");
    return 1;
  }

  ltOkiQP_t* qps = nullptr;
  HGGCCHECK(hggcHostAlloc((void**)&qps, kMaxQueueId * sizeof(ltOkiQP_t),
                       hggcHostAllocDefault));
  for (int q = 0; q < numQps; q++) qps[q] = qpArr[q];

  uint64_t* stepBase = nullptr;
  HGGCCHECK(hggcHostAlloc((void**)&stepBase, kMaxQueueId * sizeof(uint64_t),
                       hggcHostAllocDefault));
  for (int q = 0; q < numQps; q++) stepBase[q] = 0;

  uint64_t* cqAddr = nullptr;
  HGGCCHECK(hggcHostAlloc((void**)&cqAddr, kMaxQueueId * sizeof(uint64_t),
                       hggcHostAllocDefault));

  {
    report_cq_kernel<<<1, 32, 0, g_stream>>>(qps, cqAddr, numQps);
    HGGCCHECK(hggcDeviceSynchronize());
    int aliased = 0;
    for (int q = 0; q < numQps; q++) {
      for (int p = 0; p < q; p++) if (cqAddr[p] == cqAddr[q]) aliased++;
    }
    printf("CQ addresses:");
    for (int q = 0; q < numQps; q++) printf(" %d:%p", q, (void*)cqAddr[q]);
    printf("\n%s\n", aliased ? "WARNING: some QPs SHARE a CQ" : "all CQs distinct");
  }

  {
    report_cqval_kernel<<<1, 32, 0, g_stream>>>(qps, cqAddr, numQps);
    HGGCCHECK(hggcDeviceSynchronize());
    printf("CQ values at startup:");
    for (int q = 0; q < numQps; q++) printf(" %d:%llu", q,
                                            (unsigned long long)cqAddr[q]);
    printf("\n");

    {
      uint64_t initial = 0;
      for (int q = 0; q < numQps; ++q) initial = std::max(initial, cqAddr[q]);
      for (int q = 0; q < numQps; ++q) stepBase[q] = initial;
    }
  }

  printf("\nwqe=%u B  wqesPerRound=%d  rounds=%d  reps=%d  inFlight=%d  "
         "warps/blk=%d  window=%zu MB\n",
         wqeBytes, wqesPerRound, rounds, reps, roundsInFlight, warpsPerBlock,
         kWindowBytes >> 20);

  std::vector<RunResult> results;

  const int ringsPerDie = LT_OKI_DMA_RING_PER_DIE;
  const int dieNum = numQps / ringsPerDie;
  if (numQps % ringsPerDie || dieNum == 0) {
    fprintf(stderr, "need complete dies\n");
    return 1;
  }
  if (diePerBlock && (dieNum != 2 || ringsPerDie != 8 || numQps != 16)) {
    fprintf(stderr, "per-die mapping requires two dies with eight rings each\n");
    return 1;
  }
  const int qpsPerBlock = warpsPerBlock;
  const int maxBlocks = numQps / qpsPerBlock;
  if (maxBlocks < 1 || requestedBlocks > maxBlocks) {
    fprintf(stderr, "invalid block/QP selection: blocks=%d, "
                    "QPs/block=%lld, available QPs=%d; max blocks=%d\n",
            requestedBlocks, (long long)qpsPerBlock, numQps, maxBlocks);
    return 1;
  }
  printf("QP mapping: %s\n", diePerBlock ? "one block per die" : "die-interleaved warps");
  if (diePerBlock) {
    for (int b = 0; b < 2; ++b) {
      for (int w = 0; w < 8; ++w) {
        const int slot = bench_slice_slot(b, w, 8, 2, diePerBlock);
        printf("MAP block=%d warp=%d die=%d ring=%d queueId=%d slice=%d\n",
               b, w, slot%2, slot/2, (slot%2)*8+slot/2, slot);
      }
    }
  }
  uint32_t* prepared = nullptr;
  if (preassemble) {
    HGGCCHECK(hggcMalloc((void**)&prepared,kMaxQueueId*2*sizeof(uint32_t)));
    printf("PREASSEMBLED: one batch/QP; assembly excluded; stream event timing; %d copies/batch, fresh fence each repetition\n",wqesPerRound);
  }
  StreamTimer timer;
  bool allVerified = true;
  for (int blocks = requestedBlocks ? requestedBlocks : 1;
       blocks <= maxBlocks;
       blocks = requestedBlocks ? maxBlocks + 1 : blocks * 2) {
    const int workers = blocks * warpsPerBlock;
    const int threadsPerBlock = 32 * warpsPerBlock;
    const uint64_t segBytes = (uint64_t)(kWindowBytes / workers / wqeBytes) * wqeBytes;
    if (!segBytes) { fprintf(stderr, "cmd-bytes exceeds per-QP slice\n"); return 1; }
    uint64_t dataBytes = workers;
    for (uint64_t factor : {(uint64_t)rounds, (uint64_t)wqesPerRound, (uint64_t)wqeBytes}) {
      if (factor && dataBytes > UINT64_MAX / factor) {
        fprintf(stderr, "per-repetition byte count overflows uint64\n"); return 1;
      }
      dataBytes *= factor;
    }
    auto advance = [&](int count) {
      for (int q = 0; q < numQps; ++q) stepBase[q] += count;
    };
    auto verifyCQ = [&]() {
      report_cqval_kernel<<<1, 32, 0, g_stream>>>(qps, cqAddr, numQps);
      HGGCCHECK(hggcDeviceSynchronize());
      bool ok = true;
      for (int warp = 0; warp < workers; ++warp) {
        const int slot = bench_slice_slot(warp/warpsPerBlock, warp%warpsPerBlock,
                                         warpsPerBlock, dieNum, diePerBlock);
        const int q = (slot%dieNum)*ringsPerDie + slot/dieNum;
        if (cqAddr[q] != stepBase[q]) {
          fprintf(stderr, "CQ mismatch q=%d got=%llu expected=%llu\n", q,
                  (unsigned long long)cqAddr[q], (unsigned long long)stepBase[q]);
          ok = false;
        }
      }
      return ok;
    };
    fprintf(stderr, "[blocks=%d QPs=%d] warmup\n", blocks, workers);
    HGGCCHECK(hggcMemset((void*)(uintptr_t)dstPtr, 0xFF, kWindowBytes));
    bench_kernel<<<blocks, threadsPerBlock, 0, g_stream>>>(
        qps, stepBase, srcPtr, dstPtr, segBytes, wqeBytes, wqesPerRound, 1,
        1, ringsPerDie, dieNum, warpsPerBlock, diePerBlock);
    HGGCCHECK(hggcDeviceSynchronize());
    advance(1);
    const bool warmupData = verify_slices(dstPtr, srcHost, workers, segBytes,
        covered_bytes(1, wqesPerRound, wqeBytes, segBytes), wqeBytes, "warmup");
    const bool warmupCQ = verifyCQ();
    if (!warmupData || !warmupCQ) {
      allVerified = false;
      fprintf(stderr, "WARMUP VERIFICATION FAILED, skipping configuration\n");
      continue;
    }
    std::vector<double> eventTimes;
    std::vector<double> latencySamples;
    bool configVerified = true;
    for (int rep = 0; rep < reps; ++rep) {
      // Scrub each repetition so previous writes cannot mask missing transfers.
      HGGCCHECK(hggcMemset((void*)(uintptr_t)dstPtr, 0xFF, kWindowBytes));
      HGGCCHECK(hggcDeviceSynchronize());
      double elapsedMs = 0;
      const int launches = timingMode ? rounds : 1;
      for (int r = 0; r < launches; ++r) {
        if (preassemble) {
          bench_kernel<<<blocks, threadsPerBlock, 0, g_stream>>>(
              qps, stepBase, srcPtr, dstPtr, segBytes, wqeBytes, wqesPerRound,
              1, 1, ringsPerDie, dieNum, warpsPerBlock, diePerBlock, 1, prepared, r);
        }
        timer.begin();
        bench_kernel<<<blocks, threadsPerBlock, 0, g_stream>>>(
            qps, stepBase, srcPtr, dstPtr, segBytes, wqeBytes, wqesPerRound,
            timingMode ? 1 : rounds, roundsInFlight, ringsPerDie, dieNum,
            warpsPerBlock, diePerBlock, preassemble ? 2 : 0, prepared, r);
        const double ms = timer.end();
        if (!std::isfinite(ms) || ms <= 0) {
          fprintf(stderr, "invalid event duration\n"); return 2;
        }
        elapsedMs += ms;
        if (timingMode) latencySamples.push_back(ms * 1000.0);
        advance(timingMode ? 1 : rounds);
      }
      const bool dataOk = verify_slices(dstPtr, srcHost, workers, segBytes,
          covered_bytes(rounds, wqesPerRound, wqeBytes, segBytes), wqeBytes, "timed rep");
      const bool cqOk = verifyCQ();
      if (!dataOk || !cqOk || elapsedMs <= 0) {
        allVerified = configVerified = false;
        fprintf(stderr, "mode=%s blocks=%d rep=%d VERIFICATION FAILED\n", modeName, blocks, rep);
        break;
      }
      if (timingMode) {
        printf("LATENCY_REP mode=%s blocks=%d QPs=%d rep=%d samples=%d "
               "mean_us=%.4f timing=stream-event scope=all-QPs verified=yes\n",
               modeName, blocks, workers, rep, rounds, elapsedMs * 1000.0 / rounds);
      } else {
        eventTimes.push_back(elapsedMs);
        printf("BANDWIDTH_REP mode=%s blocks=%d QPs=%d rep=%d bytes=%llu "
               "event_ms=%.6f GB/s=%.6f timing=stream-event verified=yes\n", modeName, blocks, workers, rep,
               (unsigned long long)dataBytes, elapsedMs, dataBytes/elapsedMs/1e6);
      }
    }
    if (!configVerified) continue;
    if (timingMode) {
      std::sort(latencySamples.begin(), latencySamples.end());
      long double sum = 0;
      for (double us : latencySamples) sum += us;
      const size_t n = latencySamples.size();
      const double mean = (double)(sum/n);
      const double median = n % 2 ? latencySamples[n/2] :
          latencySamples[n/2-1]/2.0 + latencySamples[n/2]/2.0;
      printf("LATENCY mode=%s blocks=%d QPs=%d copies=%d cmd_bytes=%u samples=%zu "
             "mean_us=%.4f median_us=%.4f p95_us=%.4f "
             "min_us=%.4f max_us=%.4f timing=stream-event scope=all-QPs verified=yes\n",
             modeName, blocks, workers, wqesPerRound, wqeBytes, n, mean,
             median, latencySamples[(n-1)/100*95 + (n-1)%100*95/100],
             latencySamples.front(), latencySamples.back());
    } else {
      std::sort(eventTimes.begin(), eventTimes.end());
      std::vector<double> bandwidths;
      for (double ms : eventTimes) bandwidths.push_back(dataBytes/ms/1e6);
      std::sort(bandwidths.begin(), bandwidths.end());
      auto median = [](const std::vector<double>& values) {
        const size_t n = values.size();
        return n % 2 ? values[n/2] : (values[n/2-1] + values[n/2])/2;
      };
      printf("BANDWIDTH mode=%s blocks=%d warps_per_block=%d QPs=%d reps=%d "
             "min_GB/s=%.6f median_GB/s=%.6f max_GB/s=%.6f "
             "min_ms=%.6f median_ms=%.6f max_ms=%.6f timing=stream-event verified=yes%s\n",
             modeName, blocks, warpsPerBlock, workers, reps,
             bandwidths.front(), median(bandwidths), bandwidths.back(),
             eventTimes.front(), median(eventTimes), eventTimes.back(),
             bandwidths.back() > kPhysicalLimitGbps ? " ABOVE LINK LIMIT" : "");
      results.push_back({blocks, workers, bandwidths.back()});
    }
  }
  if (!results.empty()) {
    double peak = 0;
    for (const auto& result : results) peak = std::max(peak, result.gbps);
    printf("mode=%s peak_GB/s=%.6f\n", modeName, peak);
  }

  if (prepared) HGGCCHECK(hggcFree(prepared));

  for (int q = 0; q < numQps; q++) OKICHECK(ltOkiDestroyQP(dev, qpArr[q]));
  OKICHECK(ltOkiCloseDevice(dev));
  HGGCCHECK(hggcFreeHost(qps));
  HGGCCHECK(hggcFreeHost(stepBase));
  HGGCCHECK(hggcFreeHost(cqAddr));
  HGGCCHECK(hggcStreamDestroy(g_stream));
  HGCHECK(hgMemUnmap(span, spanSize));
  HGCHECK(hgMemAddressFree(span, spanSize));
  printf("benchmark done\n");
  return allVerified ? 0 : 2;
}
