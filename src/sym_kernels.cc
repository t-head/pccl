/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
#include "sym_kernels.h"
#include "comm.h"
#include "compiler.h"
#include "devcomm.h"
#include "pccl_device/core.h"
#include "transport.h"
#include "symmetric.h"
#include "graph/topo.h"
#include "ce_coll.h"     // ncclCeInit
#include "check_items.h" // pcclGetPpuDevInfo, PPU_TYPE
#include <cmath>
#include <cfloat>

constexpr char const* kernelName[] = {
  // Must align with enum ncclSymkKernelId definition in src/include/sym_kernels.h
  "AllReduce_AGxLL_R",
  "AllReduce_RSxLD_AGxST",
  "AllGather_LL",
  "AllGather_ST",
  "AllGather_ST_Ring",
  "AllGather_ST_2D",
  "ReduceScatter_LL",
  "ReduceScatter_LD",
};

constexpr uint32_t kernelMask_LL = 1<<ncclSymkKernelId_AllReduce_AGxLL_R |
                                   1<<ncclSymkKernelId_AllGather_LL |
                                   1<<ncclSymkKernelId_ReduceScatter_LL;

constexpr uint32_t kernelMask_AG = 1<<ncclSymkKernelId_AllGather_LL |
                                   1<<ncclSymkKernelId_AllGather_ST |
                                   1<<ncclSymkKernelId_AllGather_ST_Ring |
                                   1<<ncclSymkKernelId_AllGather_ST_2D;

constexpr uint32_t kernelMask_AR = 1<<ncclSymkKernelId_AllReduce_AGxLL_R |
                                   1<<ncclSymkKernelId_AllReduce_RSxLD_AGxST;

constexpr uint32_t kernelMask_RS = 1<<ncclSymkKernelId_ReduceScatter_LL |
                                   1<<ncclSymkKernelId_ReduceScatter_LD;

int ncclSymkLLKernelMask() {
  return kernelMask_LL;
}

static uint32_t kernelMask_coll(ncclFunc_t coll) {
  switch (coll) {
  case ncclFuncAllGather: return kernelMask_AG;
  case ncclFuncAllReduce: return kernelMask_AR;
  case ncclFuncReduceScatter: return kernelMask_RS;
  default: return 0;
  }
}

PCCL_PARAM(SymKernelId, "SYM_KERNEL_ID", -1);
static uint32_t kernelMask_user() {
  static uint32_t cache = -1u;
  uint32_t got = COMPILER_ATOMIC_LOAD(&cache, std::memory_order_relaxed);
  if (got == -1u) {
    char const* name = pcclGetEnvWithDualPrefix("PCCL_SYM_KERNEL");
    if (name == nullptr || strcmp(name, "^") == 0) {
      static_assert((int)ncclSymkKernelId_Count < 32, "Use more than 32 bits");
      got = (1<<(int)ncclSymkKernelId_Count)-1;
    } else {
      got = 0;
      for (int k=0; k < (int)ncclSymkKernelId_Count; k++) {
        if (strcmp(kernelName[k], name) == 0) {
          COMPILER_ATOMIC_STORE(&cache, uint32_t(1<<k), std::memory_order_relaxed);
          got = 1<<k;
          break;
        }
      }
    }
    COMPILER_ATOMIC_STORE(&cache, got, std::memory_order_relaxed);
  }
  got = pcclParamSymKernelId() >= 0 ? pcclParamSymKernelId() : got;
  return got;
}

NCCL_PARAM(SymCTAs, "SYM_CTAS", 0)
NCCL_PARAM(SymGinKernelsEnable, "SYM_GIN_KERNELS_ENABLE", 0)
PCCL_PARAM(SymWarps, "SYM_WARPS", 0)

#define DEFAULT_SYM_WARPS (ncclSymkMaxThreads/WARP_SIZE)
// In a hypercube-connected topology, each rank is connected to at most 4 peers.
#define MAX_DIRECT_CONNS 4
// Recommended max num of blocks
#define DEFAULT_SYM_MAX_CTAS 24

static double softplus(double x, double softness) {
  double z = x/softness;
  return 100.0 <= z ? x : softness*std::log1p(std::exp(z));
}

static double model(double busBytes, double baseLat, int nSMs, double smBw, double busMultiplier, double peakBw) {
  double totalSmBw = nSMs*smBw*busMultiplier;
  double bw = totalSmBw > peakBw ? peakBw : totalSmBw;
  return baseLat + softplus(busBytes/bw - 1, 1);
}

// PPU-tuned per-(topology, coll, proto) constants; measured by experiments.
namespace {
enum COLL_TYPE { ALL_REDUCE, ALL_GATHER, REDUCE_SCATTER, NUM_COLL };
enum PROTO_TYPE { LDST, LL, NUM_PROTO };
enum TOPO_TYPE { PPU_0010_DIRECT, PPU_0015_DIRECT, PPU_0015_SHM, NUM_TOPO };

constexpr double BASELAT[NUM_TOPO][NUM_PROTO] = {
  { 8.0, 1.0}, // ppu1.0
  {10.0, 1.5}, // ppu1.5
  { 7.0, 3.5}, // ppu1.5+SHM
};
constexpr double SMBW[NUM_TOPO][NUM_COLL][NUM_PROTO] = {
  { {16.0, 5.0},  {33.0, 5.0},  {16.0, 5.0} }, // ppu1.0    AR / AG / RS
  { {30.0, 5.3},  {30.0, 5.3},  {10.0, 5.3} }, // ppu1.5
  { {40.0, 12.5}, {40.0, 12.5}, { 7.5, 12.5} }, // ppu1.5+SHM
};
constexpr double PEAKBW[NUM_TOPO][NUM_PROTO] = {
  { 42.0,  14.0}, // ppu1.0
  { 80.0,  13.0}, // ppu1.5
  {340.0, 200.0}, // ppu1.5+SHM
};
} // namespace

// Given the kernel and bytes, return the minimum number of blocks to run on such that
// perf is 99% of running at max blocks, and return the estimate runtime for that
// block count.
static void queryModel(struct pcclComm* comm, ncclSymkKernelId k, size_t nBytes, float* timeUs, int* nBlocks, bool isICNSTopo) {
  constexpr double LL_BusFactor = 2; // 2X the bytes (1x for data, 1x for flags)

  int nRanks = comm->nRanks;
  int nMaxBlocks = DEFAULT_SYM_MAX_CTAS;
  size_t busBytes;
  double busMultiplier = 1;

  switch (k) {
  default:                                       busBytes = size_t(1)<<50; break;
  case ncclSymkKernelId_AllReduce_AGxLL_R:       busBytes = nRanks*nBytes*LL_BusFactor; break;
  case ncclSymkKernelId_AllReduce_RSxLD_AGxST:   busBytes = 2*nBytes*(nRanks-1)/nRanks; break;
  case ncclSymkKernelId_AllGather_LL:            busBytes = nRanks*nBytes*LL_BusFactor; break;
  case ncclSymkKernelId_AllGather_ST:
  case ncclSymkKernelId_AllGather_ST_Ring:
  case ncclSymkKernelId_AllGather_ST_2D:      busBytes = (nRanks-1)*nBytes; break;
  case ncclSymkKernelId_ReduceScatter_LL:        busBytes = nRanks*nBytes*LL_BusFactor; break;
  case ncclSymkKernelId_ReduceScatter_LD:        busBytes = (nRanks-1)*nBytes; break;
  }

  nMaxBlocks = std::min<int>(nMaxBlocks, comm->config.maxCTAs);
  int nMinBlocks = comm->config.minCTAs;

  int nUserCTAs = std::min<int>(ncclSymkMaxBlocks, ncclParamSymCTAs());
  if (nUserCTAs > 0) nMinBlocks = nMaxBlocks = nUserCTAs;

  bool isLL = kernelMask_LL>>k & 1;
  bool isAR = kernelMask_AR>>k & 1;
  bool isAG = kernelMask_AG>>k & 1;
  constexpr double GBps = (1<<30)/1.e6;

  int collIdx  = isAR ? ALL_REDUCE : (isAG ? ALL_GATHER : REDUCE_SCATTER);
  int protoIdx = isLL ? LL : LDST;
  int topoIdx  = comm->compCap < 89 ? PPU_0010_DIRECT
               : isICNSTopo         ? PPU_0015_SHM
                                    : PPU_0015_DIRECT;

  double baseLat = BASELAT[topoIdx][protoIdx];
  double smBw    = SMBW[topoIdx][collIdx][protoIdx] * GBps;
  double peakBw  = isICNSTopo ? PEAKBW[topoIdx][protoIdx] * GBps
                              : std::min<int>(nRanks-1, MAX_DIRECT_CONNS) * PEAKBW[topoIdx][protoIdx] * GBps;

  if (k == ncclSymkKernelId_AllGather_ST_Ring && isICNSTopo && nRanks > 2) peakBw *= 1.1; // to beat AllGather_ST
  // to beat AllGather_ST_Ring, whose write-target contention degrades at 64 ranks
  if (k == ncclSymkKernelId_AllGather_ST_2D && isICNSTopo && nRanks >= 64) peakBw *= 1.15;

  *nBlocks = nMaxBlocks;
  *timeUs = model(busBytes, baseLat, nMaxBlocks, smBw, busMultiplier, peakBw);
  nMinBlocks = pow2Up(nMinBlocks);
  for (int bn = nMinBlocks; bn < nMaxBlocks; bn *= 2) {
    double time = model(busBytes, baseLat, bn, smBw, busMultiplier, peakBw);
    if (time <= 1.025*(*timeUs)) {
      *nBlocks = bn;
      *timeUs = time;
      break;
    }
  }
}

ncclResult_t ncclSymkInitOnce(struct pcclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  // ncclTeamLsa() below calls this internally but drops the error code so we do it here.
  NCCLCHECK(ncclDevrInitOnce(comm));

  struct ncclSymkState* symk = &comm->symkState;
  if (symk->initialized || symk->initInProgress) return ncclSuccess;
  symk->initInProgress = true;
  {
    struct ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = ncclSymkMaxBlocks;

    struct ncclDevResourceRequirements lla2aReq;
    ncclLLA2ACreateRequirement(
      ncclSymkMaxBlocks, ncclLLA2ACalcSlots(ncclTeamLsa(comm).nRanks*ncclSymkMaxThreads, ncclSymkLLMaxEltSize),
      &symk->kcomm.lsaLLA2A, &lla2aReq
    );
    lla2aReq.next = reqs.resourceRequirementsList;
    reqs.resourceRequirementsList = &lla2aReq;

    struct ncclDevResourceRequirements ginInboxRailReq = {};
    struct ncclDevResourceRequirements ginOutboxReq = {};
    struct ncclDevResourceRequirements railSignalReq = {};
    if (ncclParamSymGinKernelsEnable() && ncclTeamLsa(comm).nRanks < comm->nRanks) {
      int maxBlocks = ncclSymkMaxBlocks;
      size_t bufSize = 1 << 16;

      maxBlocks = std::max(maxBlocks, comm->config.minCTAs);
      maxBlocks = std::min(maxBlocks, comm->config.maxCTAs);
      if (ncclParamSymCTAs() >= 1) maxBlocks = ncclParamSymCTAs();
      maxBlocks = std::min(maxBlocks, ncclSymkMaxBlocks);
      symk->maxGinInboxBlocks = maxBlocks;

      ncclGinInboxA2ACreateRequirement(
        ncclTeamRail(comm), maxBlocks, log2Up(bufSize),
        &symk->kcomm.ginInboxRail, &ginInboxRailReq
      );
      ginInboxRailReq.next = reqs.resourceRequirementsList;
      reqs.resourceRequirementsList = &ginInboxRailReq;

      ncclGinOutboxCreateRequirement(
        maxBlocks, log2Up(bufSize),
        &symk->kcomm.ginOutbox, &ginOutboxReq
      );
      ginOutboxReq.next = reqs.resourceRequirementsList;
      reqs.resourceRequirementsList = &ginOutboxReq;

      uint32_t railSignalCount = ncclTeamRail(comm).nRanks * ncclSymkMaxBlocks;

      railSignalReq.bufferSize = 0;
      railSignalReq.bufferAlign = 0;
      railSignalReq.outBufferHandle = nullptr;
      railSignalReq.ginSignalCount = railSignalCount;
      railSignalReq.outGinSignalStart = &symk->kcomm.ginSyncHandle.railSignals;
      railSignalReq.ginCounterCount = ncclSymkMaxBlocks;
      railSignalReq.outGinCounterStart = &symk->kcomm.ginCounterPerBlock;
      railSignalReq.next = reqs.resourceRequirementsList;
      reqs.resourceRequirementsList = &railSignalReq;
      reqs.barrierCount = ncclSymkMaxBlocks;
      reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
    }

    NCCLCHECKGOTO(ncclDevrCommCreateInternal(comm, &reqs, &symk->kcomm.devComm, true), ret, exit);

    // symk's ncclSymDevBase prefix (barInbox/llEpoch/LL buffer) in a devr
    // window. kcomm.base = lsa-flat self slot so ncclSymPrims::peerPtr lands
    // on peers' barInbox via the window's IPC mapping.
    {
      uint8_t* symkDevBase = nullptr;
      size_t symkDevBaseSize = ncclSymDevBase::size_symk(ncclTeamLsa(comm).nRanks);
      ncclWindow_vidmem* symkWinDev = nullptr;
      ncclWindow_vidmem* symkWinDevHost = nullptr;
      NCCLCHECKGOTO(pcclMemAlloc((void**)&symkDevBase, symkDevBaseSize), ret, fail_symkdev);
      NCCLCHECKGOTO(ncclDevrWindowRegisterInGroup(
          comm, symkDevBase, symkDevBaseSize, NCCL_WIN_COLL_SYMMETRIC, &symkWinDev), ret, fail_symkdev);
      NCCLCHECKGOTO(ncclShadowPoolToHost(&comm->devrState.shadows, symkWinDev, &symkWinDevHost), ret, fail_symkdev);
      symk->symkWin = (struct ncclDevrWindow*)symkWinDevHost->winHost;
      HGGCCHECKGOTO(hggcMemset(symk->symkWin->userPtr, 0, symkDevBaseSize), ret, exit);
      {
        char* selfSlot = (char*)symkWinDevHost->lsaFlatBase
                         + (size_t)symkWinDevHost->lsaRank * comm->devrState.bigSize;
        symk->kcomm.base = (struct ncclSymDevBase*)selfSlot;
      }
      goto symkdev_ok;
    fail_symkdev:
      if (symkDevBase) pcclMemFree(symkDevBase);
      goto exit;
    symkdev_ok:;
    }
    NCCLCHECKGOTO(ncclCeInit(comm), ret, exit);
    symk->initialized = true;
  }
exit:
  symk->initInProgress = false;
  return ret;
}

ncclResult_t ncclSymkFinalize(struct pcclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSymkState* symk = &comm->symkState;
  if (symk->initialized) {
    NCCLCHECKIGNORE(ncclDevCommDestroy(comm, &symk->kcomm.devComm), ret);
    if (symk->symkWin) {
      void* userPtr = symk->symkWin->userPtr;
      NCCLCHECKIGNORE(ncclCommWindowDeregister(comm, symk->symkWin->vidmem), ret);
      NCCLCHECKIGNORE(pcclMemFree(userPtr), ret);
      symk->symkWin = nullptr;
    }
  }
  return ret;
}

static bool ncclSymkImplemented(ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty) {
  bool isFloat;
  switch (ty) {
  case ncclFloat64:
  case ncclFloat32:
  case ncclFloat16:
  case ncclBfloat16:
  case ncclFloat8e4m3:
  case ncclFloat8e5m2:
    isFloat = true;
    break;
  default:
    isFloat = false;
    break;
  }

  switch (coll) {
  case ncclFuncAllGather:
    return true;
  case ncclFuncAllReduce:
  case ncclFuncReduceScatter:
    if (red == ncclDevSum) {
      return isFloat && ty != ncclFloat64;
    }
  default:
    return false;
  }
}

constexpr int PCCL_SYM_NUM_COLL = 3;
constexpr int PCCL_SYM_NUM_NRANKS = 4;
static const size_t _810ESymmetricThreshold[PCCL_SYM_NUM_COLL][PCCL_SYM_NUM_NRANKS] = {
    {  4_MB, 512_KB, 512_KB, 8_KB}, // AR, nranks=2,4,8,16
    { 64_MB,   8_MB,   2_MB, 2_MB}, // AG, nranks=2,4,8,16
    {256_MB,   8_MB,   2_MB, 2_MB}  // RS, nranks=2,4,8,16
};

static bool ncclSymkBeneficial(int nRanks, ncclFunc_t coll, bool isICNSTopo, size_t nBusBytes) {
  if (isICNSTopo) return true;
  ppuDeviceInfo *ppuDevInfo = pcclGetPpuDevInfo();
  if (ppuDevInfo == nullptr) return true;
  int collIdx = (coll == ncclFuncAllReduce) ? 0 : (coll == ncclFuncAllGather ? 1 : 2);
  switch (ppuDevInfo->ppuType) {
    case _PPU_ZW810:
    case _ZW_M890P:
    case _ZW_M890L:
      if (nRanks <= 4 || nBusBytes <= 2_MB) return true;
      break;
    case _PPU_ZW810E:
      if (nRanks >= 2 && nRanks <= MAX_GPUS_PER_NODE &&
          nBusBytes <= _810ESymmetricThreshold[collIdx][log2Up(nRanks)-1]) return true;
      break;
    default:
      return false;
  }
  return false;
}

static void ncclSymkApplySpecialCases(int nRanks, ncclFunc_t coll, bool isICNSTopo,
                                      size_t nBusBytes, uint32_t* kmask) {
  char const* name = pcclGetEnvWithDualPrefix("PCCL_SYM_KERNEL");
  bool isUserSetKernel = (name != nullptr || pcclParamSymKernelId() >= 0);
  if (isUserSetKernel || !isICNSTopo) return;

  switch (coll) {
    case ncclFuncAllReduce:
      if (nBusBytes <= 64_KB) *kmask &= kernelMask_LL;
      break;
    case ncclFuncAllGather:
      if (nRanks > 2 && nBusBytes <= 512_KB) *kmask &= kernelMask_LL;
      break;
    case ncclFuncReduceScatter:
      if (nBusBytes <= (size_t)nRanks * 128_KB) {
        *kmask &= kernelMask_LL;
      } else {
        *kmask &= ~kernelMask_LL;
      }
      break;
    default:
      break;
  }
}

static uint32_t ncclSymkMask(struct pcclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty, size_t nElts) {
  uint32_t kmask = kernelMask_coll(coll);
  kmask &= kernelMask_user();
  // A forced kernel skips the auto-selection policy below; only the hard
  // correctness limits still apply.
  bool isUserSetKernel = kernelMask_user() != (1u<<(int)ncclSymkKernelId_Count)-1;

  size_t nBytes = nElts*ncclTypeSize(ty);
  size_t nBusBytes = (coll == ncclFuncAllReduce ? 1 : comm->nRanks)*nBytes;

  // Correctness limits, even under force: kernels track element counts and
  // unrolled-chunk indices with 32-bit ints, which overflow past these sizes.
  if (nBusBytes >= (size_t(2)<<30)) kmask &= ~kernelMask_LL;
  if (nBusBytes >= 32*(size_t(2)<<30)) kmask = 0;

  bool isICNSTopo = (comm->topo->nodes[NVS].count > 0);

  // Hard eligibility for the 2D kernel. The kernel itself derives nGroups as
  // nRanks/8, so only keep it when the rank count can form full 8-rank groups.
  if (comm->nRanks % ncclSymkAg2DGroupSize != 0)
    kmask &= ~(1u<<ncclSymkKernelId_AllGather_ST_2D);

  // Auto-selection policy from here on: whether symmetric is worth it and which
  // kernel is fastest.
  if (!isUserSetKernel) {
    if (!ncclSymkBeneficial(comm->nRanks, coll, isICNSTopo, nBusBytes)) kmask = 0;
    ncclSymkApplySpecialCases(comm->nRanks, coll, isICNSTopo, nBusBytes, &kmask);
    // Below ~1MB the 2D kernel is latency-bound (3 barriers vs the flat ring's 2),
    // yet the cost model's peakBw bonus would still pick it in the 512KB..1MB
    // window that LL does not cover.
    if (coll == ncclFuncAllGather && nBusBytes < 1_MB)
      kmask &= ~(1u<<ncclSymkKernelId_AllGather_ST_2D);
  }
  return kmask;
}

bool ncclSymkAvailable(struct pcclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red,
                       ncclDataType_t ty, size_t nElts) {
  if (!comm->symmetricSupport)
    return false;
  // Multi-node symk is disabled: surviving kernels are LSA-only.
  if (comm->nNodes > 1)
    return false;
  if (!ncclSymkImplemented(coll, red, ty))
    return false;
  return (ncclSymkMask(comm, coll, red, ty, nElts) != 0);
}

ncclResult_t ncclSymkPickKernel(
    struct pcclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty,
    size_t nEltsTotal, size_t nEltsMax, int nWorks, ncclSymRegType_t winRegType,
    float* estTimeUs, ncclSymkKernelId* kernelId, int* nBlocks, int* nWarps, bool* forced
  ) {
  uint32_t kmask = ncclSymkMask(comm, coll, red, ty, nEltsMax);

  *forced = !(kernelMask_user() == (1<<(int)ncclSymkKernelId_Count)-1);
  // LL bodies don't unpack per-work; multi-task batches require non-LL.
  if (nWorks > 1) kmask &= ~kernelMask_LL;

  // LD/ST require registered windows; otherwise fall back to LL.
  if (coll == ncclFuncAllReduce) {
    if (winRegType != ncclSymSendRegRecvReg) kmask &= kernelMask_LL;
  } else if (coll == ncclFuncAllGather) {
    if (winRegType != ncclSymSendRegRecvReg && winRegType != ncclSymSendNonregRecvReg) kmask &= kernelMask_LL;
  } else if (coll == ncclFuncReduceScatter) {
    if (winRegType != ncclSymSendRegRecvReg && winRegType != ncclSymSendRegRecvNonreg) kmask &= kernelMask_LL;
  }

  bool isICNSTopo = (comm->topo->nodes[NVS].count > 0);
  ncclSymkKernelId bestKernel = ncclSymkKernelId_Count;
  float bestTime = 1.e30f;
  int bestBlocks = 999;
  size_t nBytes = nEltsTotal*ncclTypeSize(ty);

  constexpr float smPenalty = .025f; // 2.5% percent increase in time per SM
  uint32_t kmaskRemain = kmask;
  while (kmaskRemain != 0) {
    ncclSymkKernelId k = (ncclSymkKernelId)popFirstOneBit(&kmaskRemain);
    float kTime;
    int kBlocks;
    queryModel(comm, k, nBytes, &kTime, &kBlocks, isICNSTopo);
    if (kTime*(1.0f + smPenalty*kBlocks) < bestTime*(1.0f + smPenalty*bestBlocks)) {
      bestKernel = k;
      bestTime = kTime;
      bestBlocks = kBlocks;
    }
  }

  *kernelId = bestKernel;
  *estTimeUs = kmask==0 || kernelMask_user() == (1<<ncclSymkKernelId_Count)-1 ? bestTime : 0.0f;
  *nBlocks = bestBlocks;
  *nWarps = (pcclParamSymWarps() > 0 && pcclParamSymWarps() <= DEFAULT_SYM_WARPS) ? pcclParamSymWarps() : DEFAULT_SYM_WARPS;
  return ncclSuccess;
}

const char* ncclSymkKernelIdToString(int kernelId) {
  if (kernelId < 0 || kernelId >= ncclSymkKernelId_Count) {
    return "Unknown";
  }
  return kernelName[kernelId];
}

ncclResult_t ncclSymkMakeDevWork(struct pcclComm* comm, struct ncclTaskColl* task, struct pcclSymkDevWork* outDevWork) {
  outDevWork->rootRank = task->root;
  outDevWork->redOpArg = task->opDev.scalarArg;
  outDevWork->nElts = task->count;
  outDevWork->inputWin = task->sendWin ? task->sendWin->vidmem : nullptr;
  outDevWork->inputOff = task->sendWin ? (uint8_t*)task->sendbuff - (uint8_t*)task->sendWin->userPtr : (size_t)task->sendbuff;
  outDevWork->outputWin = task->recvWin ? task->recvWin->vidmem : nullptr;
  outDevWork->outputOff = task->recvWin ? (uint8_t*)task->recvbuff - (uint8_t*)task->recvWin->userPtr : (size_t)task->recvbuff;
  outDevWork->sChannelId = 0xffff;
  outDevWork->nChannels = 0;
  return ncclSuccess;
}

ncclResult_t ncclGetSymRegType(struct ncclDevrWindow* sendWin, struct ncclDevrWindow* recvWin, ncclSymRegType_t* winRegType) {
  bool isSendSymmReg = false;
  bool isRecvSymmReg = false;
  if (sendWin && (sendWin->winFlags & NCCL_WIN_COLL_SYMMETRIC)) isSendSymmReg = true;
  if (recvWin && (recvWin->winFlags & NCCL_WIN_COLL_SYMMETRIC)) isRecvSymmReg = true;
  // determine the registration type
  if (!isSendSymmReg && !isRecvSymmReg) {
    *winRegType = ncclSymSendNonregRecvNonreg;
  } else if (isSendSymmReg && !isRecvSymmReg) {
    *winRegType = ncclSymSendRegRecvNonreg;
  } else if (!isSendSymmReg && isRecvSymmReg) {
    *winRegType = ncclSymSendNonregRecvReg;
  } else if (isSendSymmReg && isRecvSymmReg) {
    *winRegType = ncclSymSendRegRecvReg;
  }
  return ncclSuccess;
}

// ncclSymkKernelCount / List / Requirements / GetKernelIndex are generated
// by collectives/device/symmetric/generate.py.
