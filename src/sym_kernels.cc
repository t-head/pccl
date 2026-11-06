/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
#include "sym_kernels.h"
#include "dev_runtime.h"
#include "param/param.h"
#include "comm.h"
#include "compiler.h"
#include "devcomm.h"
#include "pccl_device/core.h"
#include "transport.h"
#include "graph/topo.h"
#include "check_items.h" // pcclGetPpuDevInfo, PPU_TYPE
#include <cmath>
#include <cfloat>

constexpr char const* kernelName[] = {
  // Must align with enum ncclSymkKernelId definition in src/include/sym_kernels.h
  "AllReduce_AGxLL_R",
  "AllReduce_AGxLLMC_R",
  "AllReduce_RSxTmaLD_AGxTmaST",
  "AllReduce_RSxLD_AGxST",
  "AllReduce_RSxLDMC_AGxSTMC",
  "AllGather_LL",
  "AllGather_LLMC",
  "AllGather_TmaST",
  "AllGather_ST",
  "AllGather_TmaSTMC",
  "AllGather_STMC",
  "AllGather_RailRing_LsaSTMC",
  "ReduceScatter_LL",
  "ReduceScatter_TmaLD",
  "ReduceScatter_LD",
  "ReduceScatter_LDMC",
  "ReduceScatter_RailA2A_LsaLD",
  "ReduceScatter_RailA2A_LsaLDMC",
};

constexpr uint32_t kernelMask_LL = 1u<<ncclSymkKernelId_AllReduce_AGxLL_R |
                                  1u<<ncclSymkKernelId_AllReduce_AGxLLMC_R |
                                  1u<<ncclSymkKernelId_AllGather_LL |
                                  1u<<ncclSymkKernelId_AllGather_LLMC |
                                  1u<<ncclSymkKernelId_ReduceScatter_LL;
constexpr uint32_t kernelMask_AR = 1u<<ncclSymkKernelId_AllReduce_AGxLL_R |
                                  1u<<ncclSymkKernelId_AllReduce_AGxLLMC_R |
                                  1u<<ncclSymkKernelId_AllReduce_RSxLD_AGxST |
                                  1u<<ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC |
                                  1u<<ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST;
constexpr uint32_t kernelMask_AG = 1u<<ncclSymkKernelId_AllGather_LL |
                                  1u<<ncclSymkKernelId_AllGather_LLMC |
                                  1u<<ncclSymkKernelId_AllGather_ST |
                                  1u<<ncclSymkKernelId_AllGather_STMC |
                                  1u<<ncclSymkKernelId_AllGather_TmaST |
                                  1u<<ncclSymkKernelId_AllGather_TmaSTMC |
                                  1u<<ncclSymkKernelId_AllGather_RailRing_LsaSTMC;
constexpr uint32_t kernelMask_RS = 1u<<ncclSymkKernelId_ReduceScatter_LL |
                                  1u<<ncclSymkKernelId_ReduceScatter_TmaLD |
                                  1u<<ncclSymkKernelId_ReduceScatter_LD |
                                  1u<<ncclSymkKernelId_ReduceScatter_LDMC |
                                  1u<<ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD |
                                  1u<<ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC;
constexpr uint32_t kernelMask_STMC = 1u<<ncclSymkKernelId_AllGather_LLMC |
                                    1u<<ncclSymkKernelId_AllGather_STMC |
                                    1u<<ncclSymkKernelId_AllGather_TmaSTMC |
                                    1u<<ncclSymkKernelId_AllReduce_AGxLLMC_R |
                                    1u<<ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC |
                                    1u<<ncclSymkKernelId_ReduceScatter_LDMC |
                                    1u<<ncclSymkKernelId_AllGather_RailRing_LsaSTMC;
constexpr uint32_t kernelMask_LDMC = 1u<<ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC |
                                    1u<<ncclSymkKernelId_ReduceScatter_LDMC |
                                    1u<<ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC;
constexpr uint32_t kernelMask_Gin = 1u<<ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD |
                                   1u<<ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC |
                                   1u<<ncclSymkKernelId_AllGather_RailRing_LsaSTMC;
constexpr uint32_t kernelMask_Tma = 1u<<ncclSymkKernelId_AllGather_TmaST |
                                   1u<<ncclSymkKernelId_AllGather_TmaSTMC |
                                   1u<<ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST |
                                   1u<<ncclSymkKernelId_ReduceScatter_TmaLD;

int ncclSymkLLKernelMask() {
  return kernelMask_LL;
}

int ncclSymkDynamicSmemKernelMask() {
  return (kernelMask_Gin & kernelMask_RS) | kernelMask_Tma;
}

static uint32_t kernelMask_coll(ncclFunc_t coll) {
  switch (coll) {
  case ncclFuncAllGather: return kernelMask_AG;
  case ncclFuncAllReduce: return kernelMask_AR;
  case ncclFuncReduceScatter: return kernelMask_RS;
  default: return 0;
  }
}

static uint32_t kernelMask_user() {
  static uint32_t cache = -1u;
  uint32_t got = COMPILER_ATOMIC_LOAD(&cache, std::memory_order_relaxed);
  if (got == -1u) {
    char const* name = ncclGetEnv("PCCL_SYM_KERNEL");
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
  return got;
}

NCCL_PARAM(SymCTAs, "SYM_CTAS", 0)
NCCL_PARAM(SymGinKernelsEnable, "SYM_GIN_KERNELS_ENABLE", 1)
DEFINE_PCCL_PARAM(pcclParamSymTmaEnable, int64_t, PCCL_SYM_TMA_ENABLE, 0,
                  PCCL_PARAM_FLAG_CACHED, PCCL_PARAM_DEFAULT, "TMA symmetric kernels");
int64_t ncclParamSymTmaEnable() { return pcclParamSymTmaEnable(); }
// In a hypercube-connected topology, each rank is connected to at most 4 peers.
#define MAX_DIRECT_CONNS 4

static double softplus(double x, double softness) {
  double z = x/softness;
  return 100.0 <= z ? x : softness*std::log1p(std::exp(z));
}

static double model(double busBytes, double baseLat, int nSMs, double smBw, double busMultiplier, double peakBw) {
  double bw = std::min(nSMs*smBw*busMultiplier, peakBw);
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
static void queryModel(struct pcclComm* comm, ncclSymkKernelId k, ncclDataType_t ty, size_t nBytes,
                       float* timeUs, int* nBlocks, bool isICNSTopo) {
  constexpr double LL_BusFactor = 2; // 2X the bytes (1x for data, 1x for flags)

  int nRanks = comm->nRanks;
  int nMaxBlocks = k == ncclSymkKernelId_AllReduce_RSxLD_AGxST ? 32
                 : k == ncclSymkKernelId_ReduceScatter_LD ? 48 : 24;
  nMaxBlocks = std::min(ncclSymkMaxBlocks, std::max(nMaxBlocks, comm->config.minCTAs));
  size_t busBytes;
  double busMultiplier = 1;

  switch (k) {
  default:                                       busBytes = size_t(1)<<50; break;
  case ncclSymkKernelId_AllReduce_AGxLL_R:       busBytes = nRanks*nBytes*LL_BusFactor; break;
  case ncclSymkKernelId_AllReduce_RSxLD_AGxST:   busBytes = 2*nBytes*(nRanks-1)/nRanks; break;
  case ncclSymkKernelId_AllGather_LL:            busBytes = nRanks*nBytes*LL_BusFactor; break;
  case ncclSymkKernelId_AllGather_ST:           busBytes = (nRanks-1)*nBytes; break;
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
  int topoIdx  = comm->minCompCap < 89 ? PPU_0010_DIRECT
               : isICNSTopo            ? PPU_0015_SHM
                                       : PPU_0015_DIRECT;

  double baseLat = BASELAT[topoIdx][protoIdx];
  double smBw    = SMBW[topoIdx][collIdx][protoIdx] * GBps;
  if (k == ncclSymkKernelId_AllReduce_RSxLD_AGxST && ncclTypeSize(ty) < 4)
    smBw *= ncclTypeSize(ty)/4.0;
  double peakBw  = isICNSTopo ? PEAKBW[topoIdx][protoIdx] * GBps
                              : std::min<int>(nRanks-1, MAX_DIRECT_CONNS) * PEAKBW[topoIdx][protoIdx] * GBps;

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
  if (symk->initialized) return ncclSuccess;
  // 本轮内部symk仅支持单LSA；窗口注册仍可成功，公共GIN资源由用户Device API独立申请。
  if (ncclTeamLsa(comm).nRanks < comm->nRanks) return ncclSuccess;
  {
    struct ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    symk->hasLsaMultimem = false;
    reqs.lsaMultimem = symk->hasLsaMultimem;
    reqs.lsaBarrierCount = ncclSymkMaxBlocks;

    struct ncclDevResourceRequirements lla2aReq;
    ncclLLA2ACreateRequirement(
      ncclSymkMaxBlocks, ncclLLA2ACalcSlots(ncclTeamLsa(comm).nRanks*ncclSymkMaxThreads, ncclSymkLLMaxEltSize),
      &symk->kcomm.lsaLLA2A, &lla2aReq
    );
    lla2aReq.bufferAlign = pcclSymkLLMetadataBytesPerBlock;
    lla2aReq.bufferSize = ncclSymkMaxBlocks*(pcclSymkLLMetadataBytesPerBlock +
                                          size_t(2)*symk->kcomm.lsaLLA2A.nSlots*16);
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

    symk->initialized = true;
  }
exit:
  return ret;
}

ncclResult_t ncclSymkFinalize(struct pcclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSymkState* symk = &comm->symkState;
  if (symk->initialized) {
    NCCLCHECKIGNORE(ncclDevCommDestroy(comm, &symk->kcomm.devComm), ret);
    symk->initialized = false;
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
    if (red == ncclDevSum || (coll == ncclFuncReduceScatter && red == ncclDevSumPostDiv)) {
      return isFloat && ty != ncclFloat64;
    }
  default:
    return false;
  }
}

static uint32_t ncclSymkMask(struct pcclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty, size_t nElts) {
  uint32_t kmask = kernelMask_coll(coll);
  kmask &= kernelMask_user();
  kmask &= ~(kernelMask_STMC | kernelMask_LDMC | kernelMask_Tma | kernelMask_Gin);
  if ((ty == ncclFloat8e4m3 || ty == ncclFloat8e5m2) && comm->minCompCap < 89) kmask = 0;

  size_t nBytes = nElts*ncclTypeSize(ty);
  size_t nBusBytes = (coll == ncclFuncAllReduce ? 1 : comm->nRanks)*nBytes;

  // Correctness limits, even under force: kernels track element counts and
  // unrolled-chunk indices with 32-bit ints, which overflow past these sizes.
  if (nBusBytes >= (size_t(2)<<30)) kmask &= ~kernelMask_LL;
  if (nBusBytes >= 32*(size_t(2)<<30)) kmask = 0;

  if (ncclTeamLsa(comm).nRanks < comm->nRanks) kmask &= kernelMask_Gin;
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
    size_t nEltsTotal, size_t nEltsMax, size_t nEltsActualMax, int nWorks, ncclSymRegType_t winRegType,
    float* estTimeUs, ncclSymkKernelId* kernelId, int* nBlocks, int* nWarps, bool* forced
  ) {
  uint32_t kmask = ncclSymkMask(comm, coll, red, ty, nEltsMax);

  *forced = !(kernelMask_user() == (1<<(int)ncclSymkKernelId_Count)-1);
  if (nWorks > 1) kmask &= ~kernelMask_LL;

  // LD/ST require registered windows; otherwise fall back to LL.
  if (coll == ncclFuncAllReduce) {
    if (winRegType != ncclSymSendRegRecvReg) kmask &= kernelMask_LL;
  } else if (coll == ncclFuncAllGather) {
    if (winRegType != ncclSymSendRegRecvReg && winRegType != ncclSymSendNonregRecvReg) kmask &= kernelMask_LL;
    if (winRegType != ncclSymSendRegRecvReg && comm->nNodes > 1) kmask &= ~kernelMask_Gin;
  } else if (coll == ncclFuncReduceScatter) {
    if (winRegType != ncclSymSendRegRecvReg && winRegType != ncclSymSendRegRecvNonreg) kmask &= kernelMask_LL;
  }

  bool isICNSTopo = (comm->topo->nodes[NVS].count > 0);
  size_t nBytes = nEltsTotal*ncclTypeSize(ty);
  if (!*forced && isICNSTopo) {
    size_t nActualBytes = nEltsActualMax*ncclTypeSize(ty);
    if (nWorks == 1) {
      size_t nBusBytes = coll == ncclFuncAllReduce ? nActualBytes : comm->nRanks*nActualBytes;
      switch (coll) {
      case ncclFuncAllReduce:
        if (nBusBytes <= (size_t(1)<<16)) kmask &= kernelMask_LL;
        else if (comm->nRanks == 2 && nBusBytes >= (size_t(1)<<18) && (kmask & ~kernelMask_LL))
          kmask &= ~kernelMask_LL;
        break;
      case ncclFuncAllGather:
        if (comm->nRanks > 2 && nBusBytes <= (size_t(1)<<19)) kmask &= kernelMask_LL;
        break;
      case ncclFuncReduceScatter:
        if (nActualBytes <= (size_t(1)<<17)) kmask &= kernelMask_LL;
        else if (kmask & ~kernelMask_LL) kmask &= ~kernelMask_LL;
        break;
      default: break;
      }
    }
  }
  ncclSymkKernelId bestKernel = ncclSymkKernelId_Count;
  float bestTime = 1.e30f;
  int bestBlocks = 999;

  constexpr float smPenalty = .025f; // 2.5% percent increase in time per SM
  uint32_t kmaskRemain = kmask;
  while (kmaskRemain != 0) {
    ncclSymkKernelId k = (ncclSymkKernelId)popFirstOneBit(&kmaskRemain);
    int index = ncclSymkGetKernelIndex(k, red, ty);
    if (index < 0 || index >= ncclSymkKernelCount || ncclSymkKernelList[index] == nullptr ||
        ncclSymkKernelRequirements[index] > comm->minCompCap) continue;
    float kTime;
    int kBlocks;
    queryModel(comm, k, ty, nBytes, &kTime, &kBlocks, isICNSTopo);
    if (kTime*(1.0f + smPenalty*kBlocks) < bestTime*(1.0f + smPenalty*bestBlocks)) {
      bestKernel = k;
      bestTime = kTime;
      bestBlocks = kBlocks;
    }
  }

  *kernelId = bestKernel;
  *estTimeUs = bestKernel == ncclSymkKernelId_Count ||
               kernelMask_user() == (1<<ncclSymkKernelId_Count)-1 ? bestTime : 0.0f;
  *nBlocks = bestBlocks;
  *nWarps = 16;
  return ncclSuccess;
}

const char* ncclSymkKernelIdToString(int kernelId) {
  if (kernelId < 0 || kernelId >= ncclSymkKernelId_Count) {
    return "Unknown";
  }
  return kernelName[kernelId];
}

int ncclSymkMaxChunkElts(struct pcclComm* comm, ncclSymkKernelId kernelId, int red, ncclDataType_t ty) {
  bool isReduce = 1u & ((kernelMask_AR | kernelMask_RS) >> kernelId);
  int eltSize = ncclTypeSize(ty);
  int accMult = !isReduce ? 1 : eltSize < 4 ? 2 : 1;
  int index = ncclSymkGetKernelIndex(kernelId, red, ty);
  if (index < 0 || index >= ncclSymkKernelCount) return 0;
  return ncclSymkKernelMaxDynamicSmem[index]/(eltSize*accMult);
}

ncclResult_t ncclSymkMakeDevWork(struct pcclComm* comm, struct ncclTaskColl* task, struct ncclSymkDevWork* outDevWork) {
  outDevWork->rootRank = task->root;
  outDevWork->redOpArg = task->opDev.scalarArg;
  outDevWork->nElts = task->count;
  outDevWork->inputWin = task->sendWin ? task->sendWin->vidmem : nullptr;
  outDevWork->inputOff = task->sendWin ? (uint8_t*)task->sendbuff - (uint8_t*)task->sendWin->userPtr : (size_t)task->sendbuff;
  outDevWork->outputWin = task->recvWin ? task->recvWin->vidmem : nullptr;
  outDevWork->outputOff = task->recvWin ? (uint8_t*)task->recvbuff - (uint8_t*)task->recvWin->userPtr : (size_t)task->recvbuff;
  bool localOnly = task->devFuncId == ncclSymkKernelId_AllReduce_AGxLL_R ||
                   task->devFuncId == ncclSymkKernelId_AllGather_LL ||
                   task->devFuncId == ncclSymkKernelId_ReduceScatter_LL;
  if (localOnly && !comm->planner.persistent) {
    if (task->sendWin) {
      ncclWindow_vidmem* hostWindow;
      NCCLCHECK(ncclShadowPoolToHost(&comm->devrState.shadows,
                                   task->sendWin->vidmem, &hostWindow));
      outDevWork->inputOff += (size_t)add4G(hostWindow->lsaFlatBase, hostWindow->lsaRank*hostWindow->stride4G);
      outDevWork->inputWin = nullptr;
    }
    if (task->recvWin) {
      ncclWindow_vidmem* hostWindow;
      NCCLCHECK(ncclShadowPoolToHost(&comm->devrState.shadows,
                                   task->recvWin->vidmem, &hostWindow));
      outDevWork->outputOff += (size_t)add4G(hostWindow->lsaFlatBase, hostWindow->lsaRank*hostWindow->stride4G);
      outDevWork->outputWin = nullptr;
    }
  }
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
