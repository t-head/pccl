//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include <map>
#include <mutex>
#include <utility>

#include "core.h"
#include "comm.h"
#include "drv_extension.h"

// dispatch strategy profile
PCCL_PARAM(DispatchTunerEn, "DISPATCH_TUNER_EN", 1);

// both ppu1.0 and ppu1.5
PCCL_PARAM(ForceDefaultCeMask,    "FORCE_DEFAULT_CE_MASK",    0);
PCCL_PARAM(ForceDefaultDisp,      "FORCE_DEFAULT_DISP",       0);

PCCL_PARAM(CuMask,                "CU_MASK",                  -2);
PCCL_PARAM(DispMode,              "DISP_MODE",                -2);
PCCL_PARAM(TbPerCu,               "TB_PER_CU",                -2);
PCCL_PARAM(OneBatchPerCe,         "ONE_BATCH_PER_CE",         -2);
PCCL_PARAM(TraverseYStep,         "TRAVERSE_Y_STEP",          -2);

// ppu1.0
PCCL_PARAM(CeMask,                "CE_MASK",                  -2);

// ppu1.5
PCCL_PARAM(SubDispatchEn, "SUB_DISPATCH_EN", -2);

PCCL_PARAM(JumpDim,               "JUMP_DIM",                 -2);
PCCL_PARAM(SkewEn,                "SKEW_EN",                  -2);
PCCL_PARAM(WaitEn,                "WAIT_EN",                  -2);
PCCL_PARAM(D2dDispPingPongEn,     "D2D_DISP_PING_PONG_EN",    -2);
PCCL_PARAM(StartCeEn,             "START_CE_EN",              -2);
PCCL_PARAM(StartCeId,             "START_CE_ID",              -2);

PCCL_PARAM(TraverseXStep,         "TRAVERSE_X_STEP",          -2);
PCCL_PARAM(PreAllocEn,            "PRE_ALLOC_EN",             -2);
PCCL_PARAM(LastRoundEn,           "LAST_ROUND_EN",            -2);
PCCL_PARAM(LastRoundOnetbPerCe,   "LAST_ROUND_ONETB_PER_CE",  -2);
PCCL_PARAM(LastRoundTbPerCu,      "LAST_ROUND_TB_PER_CU",     -2);

PCCL_PARAM(LastRoundBeginNumber,  "LAST_ROUND_BEGIN_NUMBER",  -2);

PCCL_PARAM(FbStallStEn,           "FB_STALL_ST_EN",           -2);
PCCL_PARAM(FbStallStType,         "FB_STALL_ST_TYPE",         -2);
PCCL_PARAM(BlockAgeEn,            "BLOCK_AGE_EN",             -2);

PCCL_PARAM(CeNumPerDie,           "CE_NUM_PER_DIE",           -2);
PCCL_PARAM(CuNumPerCe,            "CU_NUM_PER_CE",            -2);

// The dispatch mask the device reports. The layout is fixed for the life of the
// device, so ask the driver once, but only keep an answer we actually got.
static ncclResult_t deviceAutoMask(int device, int compCap, uint64_t* outMask) {
  static std::mutex lock;
  static uint64_t cached[NCCL_MAX_LOCAL_RANKS] = {0};
  if (device < 0 || device >= NCCL_MAX_LOCAL_RANKS) return ncclInvalidArgument;
  std::lock_guard<std::mutex> lk(lock);
  if (cached[device] == 0) {
    hggcEitInterface& eit = hggcEitInterface::getInstance();
    if (compCap == 89) {
      uint8_t ceCuMasks[MAX_PPU_CE_NUM] = {0};
      HGGCCHECK(eit.wrapHggcDeviceGetAttributeCuMasks(ceCuMasks, device));
      uint64_t mask = 0;
      for (int d = 0; d < DIE_NUM_PPU0015; d++) {
        for (int i = 0; i < CE_NUM_PER_DIE_PPU0015; i++) {
          int ce = d * MAX_CE_NUM_PER_DIE + i;
          mask |= ((uint64_t)ceCuMasks[ce] << (ce * CU_NUM_PER_CE));
        }
      }
      cached[device] = mask;
    }
  }
  if (cached[device] == 0) {
    WARN("Dispatch: driver reported no dispatch resource on device %d", device);
    return ncclSystemError;
  }
  *outMask = cached[device];
  return ncclSuccess;
}

static bool eitDispatchUsable() {
  if (hggcEitInterface::getInstance().checkDispatchExtTable()) return true;
  static std::once_flag reported;
  std::call_once(reported, []() { ERROR("Dispatch: EIT table unavailable, dispatch config ignored"); });
  return false;
}

static void applyCommonDispatchEnv0010(ppuDispatchStrategy_0010* schedule,
                                 ppuDispatchMask_0010* dispMask = nullptr) {
  CEHCK_ENV_SETTING(schedule->bits.disp_mode, pcclParamDispMode());
  CEHCK_ENV_SETTING(schedule->bits.tb_per_cu, pcclParamTbPerCu());
  CEHCK_ENV_SETTING(schedule->bits.onebatch_per_ce, pcclParamOneBatchPerCe());
  CEHCK_ENV_SETTING(schedule->bits.traverse_y_step, pcclParamTraverseYStep());
  if (dispMask == nullptr) return;
  CEHCK_ENV_SETTING(dispMask->bits.cu_mask, pcclParamCuMask());
  CEHCK_ENV_SETTING(dispMask->bits.ce_mask, pcclParamCeMask());
}

static void applyCommonDispatchEnv0015(ppuDispatchStrategy_0015* schedule, ppukernelControl_0015* control,
                                 uint64_t* cuMask = nullptr) {
  CEHCK_ENV_SETTING(schedule->strategy.bits.dispMode, pcclParamDispMode());
  CEHCK_ENV_SETTING(schedule->strategy.bits.jumpDim, pcclParamJumpDim());
  CEHCK_ENV_SETTING(schedule->strategy.bits.oneBatchPerCe, pcclParamOneBatchPerCe());
  CEHCK_ENV_SETTING(schedule->strategy.bits.skewEn, pcclParamSkewEn());
  CEHCK_ENV_SETTING(schedule->strategy.bits.tbPerCu, pcclParamTbPerCu());
  CEHCK_ENV_SETTING(schedule->strategy.bits.waitEn, pcclParamWaitEn());
  CEHCK_ENV_SETTING(schedule->strategy.bits.d2dDispPingPongEn, pcclParamD2dDispPingPongEn());
  CEHCK_ENV_SETTING(schedule->strategy.bits.startCeEn, pcclParamStartCeEn());
  CEHCK_ENV_SETTING(schedule->strategy.bits.startCeId, pcclParamStartCeId());

  CEHCK_ENV_SETTING(schedule->strategy1.bits.traverseYStep, pcclParamTraverseYStep());
  CEHCK_ENV_SETTING(schedule->strategy1.bits.traverseXStep, pcclParamTraverseXStep());
  CEHCK_ENV_SETTING(schedule->strategy1.bits.preAllocateEn, pcclParamPreAllocEn());
  CEHCK_ENV_SETTING(schedule->strategy1.bits.lastRoundEn, pcclParamLastRoundEn());
  CEHCK_ENV_SETTING(schedule->strategy1.bits.lastRoundOnetbPerCe, pcclParamLastRoundOnetbPerCe());
  CEHCK_ENV_SETTING(schedule->strategy1.bits.lastRoundTbPerCu, pcclParamLastRoundTbPerCu());

  CEHCK_ENV_SETTING(schedule->strategy2.bits.lastRoundBeginNumber, pcclParamLastRoundBeginNumber());

  CEHCK_ENV_SETTING(control->bits.fbStallStEn, pcclParamFbStallStEn());
  CEHCK_ENV_SETTING(control->bits.fbStallStType, pcclParamFbStallStType());
  CEHCK_ENV_SETTING(control->bits.blockAgeEn, pcclParamBlockAgeEn());
  if (cuMask == nullptr) return;
  int64_t envCuMask = pcclParamCuMask();
  if (envCuMask > 0) *cuMask = (uint64_t)envCuMask;
}

#define NUM_DISPATCH_STRATEGIES 4
#define MAX_BYTES 96_GB
static dispatchTuner g_dispatchStrategyList[NUM_DISPATCH_STRATEGIES] = {
  // 810 4ppu
  { .matcher = {.coll = ncclFuncBroadcast, .nNodes = 1, .nChannels = 12, .minBytes = 4_GB, .maxBytes = MAX_BYTES},
    .dispType = dispatchTypeSchedule_0010,
    .modifier = {{"onebatch_per_ce", 0}, {"tb_per_cu", 2}},
    .finalValue = 0
  },
  // 810E 4ppu, 810 8ppu
  { .matcher = {.coll = ncclFuncBroadcast, .nNodes = 1, .nChannels = 16, .minBytes = 512_KB, .maxBytes = MAX_BYTES},
    .dispType = dispatchTypeSchedule_0010,
    .modifier = {{"onebatch_per_ce", 0}, {"tb_per_cu", 2}},
    .finalValue = 0
  },
  // 810E 16ppu
  { .matcher = {.coll = ncclFuncBroadcast, .nNodes = 1, .nChannels = 28, .minBytes = 128, .maxBytes = MAX_BYTES},
    .dispType = dispatchTypeSchedule_0010,
    .modifier = {{"onebatch_per_ce", 0}, {"tb_per_cu", 1}},
    .finalValue = 0
  },
  // 810E 16ppu
  { .matcher = {.coll = ncclFuncReduce, .nNodes = 1, .nChannels = 28, .minBytes = 8_MB, .maxBytes = 4_GB},
    .dispType = dispatchTypeSchedule_0010,
    .modifier = {{"onebatch_per_ce", 0}, {"tb_per_cu", 1}},
    .finalValue = 0
  }
};

#define SET_BITFIELD(config, MEMBER_NAME, CONFIG_NAME, CONFIG_VAL) \
  do {                                                             \
    if (strcmp(bitFieldName.c_str(), #MEMBER_NAME) == 0) {         \
      config.bits.MEMBER_NAME = configVal;                         \
    }                                                              \
  } while (0)

// NOTE: First-match priority. The table is searched linearly from top to bottom.
// Place specialized rules (more constraints) at the top, and generic rules (wildcards) at the bottom.
static const SmArchConfig g_smResConfigList[] = {
    //                 Condition                         |                        Policy
    // compCap |  ppuType     | nRansk |      rmtType    |          Level         | CU/CE | CE/Die| MaxCh |
    { 89,       _ZW_M890P,      4,      PCCL_ICNLINK_PPU, {SM_POLICY_LAYOUT_FIXED,  3,      4,    24, }},
    { 89,       _ZW_M890P,     -1,      PCCL_ICNLINK_ANY, {SM_POLICY_LAYOUT_FIXED,  3,      4,    16, }},
    { 89,       _PPU_TYPE_ANY, -1,      PCCL_ICNLINK_ANY, {SM_POLICY_RES_LIMITED,  -1,     -1,    16, }}
};

template<typename T>
inline bool isMatch(T configVal, T runtimeVal) {
    return (configVal == (T)-1) || (configVal == runtimeVal);
}

ncclResult_t initSmResourcesConfig(ncclComm_t comm) {
  // Currently, this resource config is only valid to Compute Capability 89
  if (comm->compCap < 89) return ncclSuccess;
  for (const auto& entry : g_smResConfigList) {
    if (isMatch<int>(entry.compCap, comm->compCap) &&
        isMatch<PPU_TYPE>(entry.ppuType, comm->ppuType) &&
        isMatch<int>(entry.nRanks, comm->nRanks) &&
        isMatch<pcclICNRemoteDeviceType_t>(entry.icnRmtType, comm->icnRmtType)) {
      comm->smResConfig.defaultMaxChannelNum = entry.config.defaultMaxChannelNum;
      comm->smResConfig.smResourcePolicy = entry.config.smResourcePolicy;
      if (comm->smResConfig.smResourcePolicy == SM_POLICY_LAYOUT_FIXED) {
        comm->smResConfig.cuNumUsePerCe = entry.config.cuNumUsePerCe;
        comm->smResConfig.ceNumUsePerDie = entry.config.ceNumUsePerDie;
      }
      break;
    }
  }
  // check env setting
  int envCesPerDie = pcclParamCeNumPerDie();
  int envCusPerCe = pcclParamCuNumPerCe();
  if (envCesPerDie > 0 && envCusPerCe > 0) {
    comm->smResConfig.smResourcePolicy = SM_POLICY_LAYOUT_FIXED;
    comm->smResConfig.ceNumUsePerDie = envCesPerDie;
    comm->smResConfig.cuNumUsePerCe = envCusPerCe;
  }
  return ncclSuccess;
}

ncclResult_t getPpuDeviceDispatchCuInfo(uint64_t* cuMask, int* totalCuNum, int device, int compCap, int nDies, PPU_TYPE ppuType, struct smResourceConfig_t* smResConfig, int nChannels) {
  ncclResult_t res;
  uint8_t* ppuCuMasks = NULL;
  uint64_t cuMaskPerCe = 0;
  *cuMask = 0;
  if (compCap < 89) return ncclSuccess;
  // when cu mask is 0, totalCuNum will be assigned the value of all available CUs later.
  int64_t cuMaskEnvValue = pcclParamCuMask();
  if (nDies == 1 && ((cuMaskEnvValue >> 32) > 0 || cuMaskEnvValue == 0)) {
    WARN("The number of dies is 1, but cu Mask 0x%lx would use the resources of two dies", cuMaskEnvValue);
  }
  if (cuMaskEnvValue > 0) {
    *totalCuNum = __builtin_popcountll((uint64_t)cuMaskEnvValue);
    *cuMask = (uint64_t)cuMaskEnvValue;
    return ncclSuccess;
  }
  if (nChannels > MAXCHANNELS_PPU0015 || cuMaskEnvValue == 0) {
    VERBOSE(NCCL_TUNING, "Special user configuration detected: bypassing resource limits (SM_POLICY_RES_FULL).");
    smResConfig->smResourcePolicy = SM_POLICY_RES_FULL;
  }

  NCCLCHECKGOTO(ncclCalloc(&ppuCuMasks, sizeof(uint8_t) * MAX_PPU_CE_NUM), res, cleanup);
  HGGCCHECKGOTO(hggcEitInterface::getInstance().wrapHggcDeviceGetAttributeCuMasks(ppuCuMasks, device), res, cleanup);

  // compute cu mask
  for (int d = 0; d < nDies; d++) {
    int cnt = 0;
    for (int i = 0 ; i < MAX_CE_NUM_PER_DIE; i++) {
      uint64_t ceIdx = d * MAX_CE_NUM_PER_DIE + i;
      *totalCuNum += countOneBits(ppuCuMasks[ceIdx]);
      // Consider the scene that pccl kernels run in parallel with other kernels,
      // won't use all CUs in one CE
      if (smResConfig->smResourcePolicy == SM_POLICY_LAYOUT_FIXED) {
        if (cnt < smResConfig->ceNumUsePerDie && countOneBits(ppuCuMasks[ceIdx]) > smResConfig->cuNumUsePerCe) {
          selectTopNOneBitsFromPos<uint64_t>(ppuCuMasks[ceIdx], CU_NUM_PER_CE - 1, smResConfig->cuNumUsePerCe, &cuMaskPerCe);
          *cuMask |= (cuMaskPerCe << (ceIdx * CU_NUM_PER_CE));
          cnt++;
        }
      } else if (smResConfig->smResourcePolicy == SM_POLICY_RES_LIMITED){
        // masked the first CU per CE
        cuMaskPerCe = ppuCuMasks[ceIdx] & (ppuCuMasks[ceIdx] - 1);
        *cuMask |= cuMaskPerCe << (ceIdx * CU_NUM_PER_CE);
      }
    }
  }

cleanup:
  NCCLCHECK(ncclFree(ppuCuMasks));
  return res;
}

uint32_t selectSetBits(const uint32_t input, uint32_t defaultCeMask, const int ceNum) {
  if (countOneBits(input) < ceNum) return defaultCeMask;
  uint32_t output = 0;
  bool res = selectTopNOneBitsFromPos<uint32_t>(input, MAX_SELECT_CE_IDX, ceNum, &output);
  if (!res) {
    ERROR("input %x select ceNum %d from pos %d failed", input, ceNum, MAX_SELECT_CE_IDX);
    assert(false);
  }
  return output;
}

bool checkIcnDispatchMask(int icnDispatchMask, PPU_TYPE ppuType) {
  ppuIcnCeInfo ceInfo = {.val = icnDispatchMask};
  if (ppuType == _PPU_ZW810 || ppuType == _PPU_ZW810E || ppuType == _PPU_ZW810_H) {
    return (ceInfo.bits.full_ce_left || ceInfo.bits.full_ce_right) ? true : false;
  }
  return true;
}

bool checkFullCeLeft(int icnDispatchMask, PPU_TYPE ppuType) {
  if (ppuType == _PPU_ZW810 || ppuType == _PPU_ZW810E || ppuType == _PPU_ZW810_H) {
    ppuIcnCeInfo ceInfo = {.val = icnDispatchMask};
    return ceInfo.bits.full_ce_left ? true : false;
  }
  return true;
}

bool checkFullCeRight(int icnDispatchMask, PPU_TYPE ppuType) {
  if (ppuType == _PPU_ZW810 || ppuType == _PPU_ZW810E || ppuType == _PPU_ZW810_H) {
    ppuIcnCeInfo ceInfo = {.val = icnDispatchMask};
    return ceInfo.bits.full_ce_right ? true : false;
  }
  return true;
}

ncclResult_t initDefaultDispatchStrategy(ncclComm_t comm, int nDies) {
  if (comm->compCap == 80) {
    ppuDispatchStrategy_0010 schedule = {0};
    schedule.bits.tb_per_cu = 1;
    schedule.bits.onebatch_per_ce = 1;
    schedule.bits.traverse_y_step = 1;
    // TODO: fine grain strategy policies adjustment may need be done in the future
    /*
    schedule.bits.override_thread_block_per_cu = 1;
    schedule.bits.jump_dim = (matrix_a_size > matrix_b_size) ? 0 : 1;
    */

    int icnDispatchMask = 0;
    HGGCCHECK(hggcEitInterface::getInstance().wrapHggcDeviceGetAttributeIcnDispatchMask(&icnDispatchMask, comm->hggcDev));
    if (icnDispatchMask && (!checkIcnDispatchMask(icnDispatchMask, comm->ppuType))) {
      ERROR("invalid icnDispatchMask %x", icnDispatchMask);
      return ncclInternalError;
    }
    // get ce mask
    ppuDispatchMask_0010 dispMask = {0};
    ppuDispatchMask_0010 dispMask7Ce = {0};
    ppuDispatchMask_0010 dispMask6Ce = {0};
    if (icnDispatchMask != 0) {
      if (checkFullCeRight(icnDispatchMask, comm->ppuType)) {
        // dispatch to bit 8 ~ 15, 17
        dispMask.bits.ce_mask = icnDispatchMask & CE_DISPATCH_MASK_H;

        dispMask7Ce.bits.ce_mask = selectSetBits(icnDispatchMask & CE_DISPATCH_MASK_8_15, dispMask.bits.ce_mask, 7);
        dispMask6Ce.bits.ce_mask = selectSetBits(icnDispatchMask & CE_DISPATCH_MASK_8_15, dispMask.bits.ce_mask, 6);
      } else {
        // dispatch to bit 0 ~ 7, 16
        dispMask.bits.ce_mask = icnDispatchMask & CE_DISPATCH_MASK_L;

        dispMask7Ce.bits.ce_mask = selectSetBits(icnDispatchMask & CE_DISPATCH_MASK_0_7, dispMask.bits.ce_mask, 7);
        dispMask6Ce.bits.ce_mask = selectSetBits(icnDispatchMask & CE_DISPATCH_MASK_0_7, dispMask.bits.ce_mask, 6);
      }
    } else {
      WARN("No specified or ICN dispatch CE mask, use 0xff as default");
      dispMask.bits.ce_mask = 0xff;
      dispMask7Ce.bits.ce_mask = 0xfe;
      dispMask6Ce.bits.ce_mask = 0xfc;
    }

    CEHCK_ENV_SETTING(dispMask.bits.ce_mask, pcclParamCeMask());
    CEHCK_ENV_SETTING(dispMask7Ce.bits.ce_mask, pcclParamCeMask());
    CEHCK_ENV_SETTING(dispMask6Ce.bits.ce_mask, pcclParamCeMask());

    // get cu mask
    if (comm->ppuType == _PPU_ZW810E && comm->config.CTAPolicy == PCCL_CTA_POLICY_EFFICIENCY) {
      // use the last 2 CU without tensor core to avoid competing resources with compute kernel
      dispMask.bits.cu_mask = 0xc;
    } else {
      dispMask.bits.cu_mask = 0xf;
    }

    CEHCK_ENV_SETTING(dispMask.bits.cu_mask, pcclParamCuMask());
    dispMask7Ce.bits.cu_mask = dispMask.bits.cu_mask;
    dispMask6Ce.bits.cu_mask = dispMask.bits.cu_mask;

    // check env setting
    applyCommonDispatchEnv0010(&schedule);

    // record init value
    comm->defaultStrategy.dispStrategy = schedule.val;
    comm->defaultStrategy.dispMask[CeNumIdxDefault_t] = dispMask.val;
    comm->defaultStrategy.dispMask[CeNumIdx7Ce_t] = dispMask7Ce.val;
    comm->defaultStrategy.dispMask[CeNumIdx6Ce_t] = dispMask6Ce.val;

    // init value of strategy profile list
    initProfileValue(dispatchTypeSchedule_0010, schedule.val);

  } else if (comm->compCap == 89) {
    // init value
    int totalCuNum = 0;
    ppuDispatchMask_0015 cuMask = {0};
    NCCLCHECK(getPpuDeviceDispatchCuInfo(&cuMask.cu_mask, &totalCuNum, comm->hggcDev, comm->compCap, nDies, comm->ppuType, &(comm->smResConfig), comm->nChannels));

    ppuDispatchStrategy_0015 schedule = {0};
    schedule.strategy.bits.oneBatchPerCe = (comm->smResConfig.smResourcePolicy != SM_POLICY_LAYOUT_FIXED) ? 0 : 1;
    schedule.strategy.bits.tbPerCu = DIVUP(comm->nChannels, totalCuNum);
    schedule.strategy.bits.d2dDispPingPongEn = 0;
    schedule.strategy1.bits.traverseYStep = 1;
    schedule.strategy1.bits.traverseXStep = 1;

    ppukernelControl_0015 control = {0};

    // check env setting
    applyCommonDispatchEnv0015(&schedule, &control);

    // record init value
    comm->defaultStrategy.dispStrategy = schedule.strategy.val;
    comm->defaultStrategy.dispStrategy1 = schedule.strategy1.val;
    comm->defaultStrategy.dispStrategy2 = schedule.strategy2.val;
    comm->defaultStrategy.kernelControl = control.val;
    comm->defaultStrategy.dispMask[CeNumIdxDefault_t] = cuMask.cu_mask;
    VERBOSE(NCCL_TUNING, "Set default cumask 0x%010lx", cuMask.cu_mask);
    comm->defaultStrategy.launchConfig.subDispatch = pcclParamSubDispatchEn();
  }
  return ncclSuccess;
}

namespace {
struct dispatchLogRow {
  const char* state;
  const char* reg;
  const char* fields;
};

const char kDispDashes[] = "--------------------------------------------------------------------------------";

void boxSep(char* b, int cap, int* off) {
  *off += snprintf(b + *off, std::max(0, cap - *off), "  +%.*s+%.*s+%.*s+\n",
                   9, kDispDashes, 10, kDispDashes, 62, kDispDashes);
}

// Word-wrap the fields into the box so a long field list stays inside the frame.
void boxRow(char* b, int cap, int* off, const char* state, const char* reg, const char* fields) {
  const int W = 60;
  int n = (int)strlen(fields), start = 0, first = 1;
  do {
    int end = (n - start <= W) ? n : start + W;
    if (end < n) { int brk = end; while (brk > start && fields[brk] != ' ') brk--; if (brk > start) end = brk; }
    *off += snprintf(b + *off, std::max(0, cap - *off), "  | %-7s | %-8s | %-*.*s |\n",
                     first ? state : "", first ? reg : "", W, end - start, fields + start);
    first = 0;
    for (start = end; start < n && fields[start] == ' '; start++) {}
  } while (start < n);
}

void logCommonDispatchTable(const char* path, int device, const void* kernel, int compCap, const char* title,
                           const void* schedule, const char* strategyState,
                           const void* control, const char* controlState,
                           const void* mask, const char* maskState) {
  struct dispatchLogRow rows[3];
  char f0[256], f1[64], f2[64];
  int nRows = 0;
  if (compCap == 80) {
    const ppuDispatchStrategy_0010* s = (const ppuDispatchStrategy_0010*)schedule;
    const ppuDispatchMask_0010* m = (const ppuDispatchMask_0010*)mask;
    snprintf(f0, sizeof(f0), "dispMode=%u oneBatchPerCe=%u tbPerCu=%u traverseYStep=%u",
             s->bits.disp_mode, s->bits.onebatch_per_ce,
             s->bits.tb_per_cu, s->bits.traverse_y_step);
    snprintf(f1, sizeof(f1), "ceMask=0x%05x cuMask=0x%x", m->bits.ce_mask, m->bits.cu_mask);
    rows[0] = {strategyState, "strategy", f0};
    rows[1] = {maskState, "dispMask", f1};
    nRows = 2;
  } else {
    const ppuDispatchStrategy_0015* s = (const ppuDispatchStrategy_0015*)schedule;
    const ppukernelControl_0015* c = (const ppukernelControl_0015*)control;
    const ppuDispatchMask_0015* m = (const ppuDispatchMask_0015*)mask;
    snprintf(f0, sizeof(f0),
             "dispMode=%u jumpDim=%u oneBatchPerCe=%u skewEn=%u tbPerCu=%u waitEn=%u "
             "d2dDispPingPongEn=%u startCeEn=%u startCeId=%u traverseYStep=%u traverseXStep=%u "
             "preAllocateEn=%u lastRoundEn=%u lastRoundOnetbPerCe=%u lastRoundTbPerCu=%u lastRoundBeginNumber=%u",
             s->strategy.bits.dispMode, s->strategy.bits.jumpDim,
             s->strategy.bits.oneBatchPerCe, s->strategy.bits.skewEn,
             s->strategy.bits.tbPerCu, s->strategy.bits.waitEn,
             s->strategy.bits.d2dDispPingPongEn, s->strategy.bits.startCeEn,
             s->strategy.bits.startCeId, s->strategy1.bits.traverseYStep,
             s->strategy1.bits.traverseXStep, s->strategy1.bits.preAllocateEn,
             s->strategy1.bits.lastRoundEn, s->strategy1.bits.lastRoundOnetbPerCe,
             s->strategy1.bits.lastRoundTbPerCu, s->strategy2.bits.lastRoundBeginNumber);
    snprintf(f1, sizeof(f1), "fbStallStEn=%u fbStallStType=%u blockAgeEn=%u",
             c->bits.fbStallStEn, c->bits.fbStallStType, c->bits.blockAgeEn);
    snprintf(f2, sizeof(f2), "cuMask=0x%013lx", m->cu_mask);
    rows[0] = {strategyState, "strategy", f0};
    rows[1] = {controlState, "control", f1};
    rows[2] = {maskState, "cuMask", f2};
    nRows = 3;
  }
  char tbl[2048];
  int off = snprintf(tbl, sizeof(tbl), "\n");
  boxSep(tbl, sizeof(tbl), &off);
  off += snprintf(tbl + off, std::max(0, (int)sizeof(tbl) - off), "  | %-7s | %-8s | %-60s |\n",
                  "state", "register", "fields");
  boxSep(tbl, sizeof(tbl), &off);
  for (int i = 0; i < nRows; i++) boxRow(tbl, sizeof(tbl), &off, rows[i].state, rows[i].reg, rows[i].fields);
  boxSep(tbl, sizeof(tbl), &off);
  VERBOSE(NCCL_TUNING, "Dispatch config: %s dev %d kernel %p sm%d %s%s", path, device, kernel, compCap, title, tbl);
}
}  // namespace

static ncclResult_t setCommonDispatchStrategy(const char* path, const struct pcclDispatchFunc* fn, const char* title, int compCap,
                                        const struct ncclFuncDispatchStrategy* newStrategy,
                                        struct ncclFuncDispatchStrategy* oldStrategy) {
  const void* key = fn->symbol ? fn->symbol : (const void*)fn->handle;
  if (ncclDebugLevel == -1) ncclDebugInit();
  const bool dispLogOn = (ncclDebugLevel >= PCCL_LOG_VERBOSE) && (ncclDebugMask & NCCL_TUNING);

  if (compCap == 80) {
    ppuDispatchStrategy_0010 schedule = {.val = newStrategy->dispStrategy};
    ppuDispatchMask_0010 dispMask = {.val = (unsigned int)newStrategy->dispMask};
    bool setStrategy = (oldStrategy->dispStrategy != schedule.val);
    bool setMask = ((oldStrategy->dispMask & 0xffffffff) != dispMask.val);

    auto set80Attr = [&](int attr, unsigned val) -> hggcError_t {
      if (fn->symbol != NULL)
        return hggcFuncSetAttribute(fn->symbol, static_cast<hggcFuncAttribute>(attr), val);
      return (hggcError_t)hgFuncSetAttribute(fn->handle, static_cast<HGfunction_attribute>(attr), (int)val);
    };
    if (setStrategy) {
      HGGCCHECK(set80Attr(ppuFuncAttributeDispatchStrategy, schedule.val));
      oldStrategy->dispStrategy = schedule.val;
      oldStrategy->launchConfig.blocksPerMultiprocessor = schedule.bits.tb_per_cu;
      oldStrategy->launchConfig.gridStepY = schedule.bits.traverse_y_step;
      oldStrategy->launchConfig.gridStepX = 1;
    }
    if (setMask) {
      HGGCCHECK(set80Attr(ppuFuncAttributeDispatchMask, dispMask.val));
      oldStrategy->dispMask = dispMask.val & 0xffffffff;
    }

    if (dispLogOn && (setStrategy || setMask)) {
      int device = 0;
      if (hggcGetDevice(&device) == hggcSuccess && device == 0) {
        logCommonDispatchTable(path, device, key, compCap, title,
                               &schedule, setStrategy ? "applied" : "cached",
                               NULL, NULL,
                               &dispMask, setMask ? "applied" : "cached");
      }
    }
  } else if (compCap == 89) {
    if (!eitDispatchUsable()) return ncclSuccess;
    ppuDispatchStrategy_0015 schedule = {
      .strategy = {.val = newStrategy->dispStrategy},
      .strategy1 = {.val = newStrategy->dispStrategy1},
      .strategy2 = {.val = newStrategy->dispStrategy2}
    };
    ppukernelControl_0015 control = {.val = newStrategy->kernelControl};
    ppuDispatchMask_0015 cuMask = {.cu_mask = newStrategy->dispMask};
    oldStrategy->launchConfig.subDispatch = newStrategy->launchConfig.subDispatch;

    bool setStrategy = ((oldStrategy->dispStrategy ^ schedule.strategy.val) |
                        (oldStrategy->dispStrategy1 ^ schedule.strategy1.val) |
                        (oldStrategy->dispStrategy2 ^ schedule.strategy2.val)) > 0;
    bool setControl = (control.bits.fbStallStEn == 1 && control.val != oldStrategy->kernelControl);
    bool setMask = (oldStrategy->dispMask != cuMask.cu_mask);
    bool keepDisp = pcclParamForceDefaultDisp();
    bool keepMask = pcclParamForceDefaultCeMask();

    hggcEitInterface& eit = hggcEitInterface::getInstance();
    auto set89Attr = [&](HGeitFuncAttribute attr, void* data, uint32_t size) -> hggcError_t {
      return eit.wrapHggcEitFuncSetAttribute(fn, attr, data, size);
    };
    if (setStrategy) {
      if (!keepDisp) HGGCCHECK(set89Attr((HGeitFuncAttribute)ppuEitFuncAttributeDispatchStrategy, &schedule, sizeof(schedule)));
      oldStrategy->dispStrategy = schedule.strategy.val;
      oldStrategy->dispStrategy1 = schedule.strategy1.val;
      oldStrategy->dispStrategy2 = schedule.strategy2.val;
      oldStrategy->launchConfig.blocksPerMultiprocessor = schedule.strategy.bits.tbPerCu;
      oldStrategy->launchConfig.gridStepX = schedule.strategy1.bits.traverseXStep;
      oldStrategy->launchConfig.gridStepY = schedule.strategy1.bits.traverseYStep;
    }
    if (setControl) {
      HGGCCHECK(set89Attr((HGeitFuncAttribute)ppuEitFuncAttributeKernelControl, &control, sizeof(control)));
      oldStrategy->kernelControl = control.val;
    }
    if (setMask) {
      if (!keepMask) HGGCCHECK(set89Attr((HGeitFuncAttribute)ppuEitFuncAttributeDispatchCuMask, &cuMask, sizeof(cuMask)));
      oldStrategy->dispMask = cuMask.cu_mask;
    }

    if (dispLogOn && (setStrategy || setControl || setMask)) {
      int device = 0;
      if (hggcGetDevice(&device) == hggcSuccess && device == 0) {
        logCommonDispatchTable(path, device, key, compCap, title,
                               &schedule, setStrategy ? (keepDisp ? "ignored" : "applied") : "cached",
                               &control, setControl ? "applied" : "cached",
                               &cuMask, setMask ? (keepMask ? "ignored" : "applied") : "cached");
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t setNativeDispatchStrategy(const void* func, struct ncclFuncDispatchStrategy* oldStrategy, struct ncclDispatchStrategy defaultStrategy, const char* funcName,
                                int compCap, dispatchInfo dInfo) {
  struct ncclFuncDispatchStrategy newStrategy = {};
  newStrategy.launchConfig = defaultStrategy.launchConfig;

  if (compCap == 80) {
    uint32_t schedule = defaultStrategy.dispStrategy;
    uint32_t dispMask = (uint32_t)defaultStrategy.dispMask[CeNumIdxDefault_t];
    // check profiles
    matchDispatchInfoPpu0010(dInfo, defaultStrategy.dispMask, &schedule, &dispMask);
    newStrategy.dispStrategy = schedule;
    newStrategy.dispMask = dispMask;
  } else if (compCap == 89) {
    newStrategy.dispStrategy = defaultStrategy.dispStrategy;
    newStrategy.dispStrategy1 = defaultStrategy.dispStrategy1;
    newStrategy.dispStrategy2 = defaultStrategy.dispStrategy2;
    newStrategy.kernelControl = defaultStrategy.kernelControl;
    newStrategy.dispMask = defaultStrategy.dispMask[CeNumIdxDefault_t];
  } else {
    return ncclSuccess;
  }
  struct pcclDispatchFunc fn = {func, NULL};
  return setCommonDispatchStrategy("native", &fn, funcName, compCap, &newStrategy, oldStrategy);
}

ncclResult_t setUserDispatchStrategy(const struct pcclDispatchFunc* fn, const char* policyName, int compCap,
                                     int oneBatchPerCe, int tbPerCu,
                                     int cuMaskPolicy, uint64_t cuMask, uint32_t ceMask) {
  static std::mutex lock;
  static std::map<std::pair<int, const void*>, struct ncclFuncDispatchStrategy> oldStrategies;

  int device = 0;
  HGGCCHECK(hggcGetDevice(&device));

  const void* key = fn->symbol ? fn->symbol : (const void*)fn->handle;

  std::lock_guard<std::mutex> lk(lock);
  struct ncclFuncDispatchStrategy* oldStrategy = &oldStrategies[std::make_pair(device, key)];

  struct ncclFuncDispatchStrategy newStrategy = {};

  if (compCap == 80) {
    ppuDispatchStrategy_0010 schedule = {0};
    schedule.bits.tb_per_cu = (tbPerCu >= 0) ? tbPerCu : 1;
    schedule.bits.onebatch_per_ce = (oneBatchPerCe >= 0) ? oneBatchPerCe : 1;
    schedule.bits.traverse_y_step = 1;
    ppuDispatchMask_0010 dispMask = {0};
    switch (cuMaskPolicy) {
      case PCCLX_KERNEL_DISPATCH_AUTO:
      case PCCLX_KERNEL_DISPATCH_DRV_DEFAULT: break;
      default:
        dispMask.bits.cu_mask = (unsigned int)cuMask;
        dispMask.bits.ce_mask = ceMask;
        break;
    }
    applyCommonDispatchEnv0010(&schedule, &dispMask);
    newStrategy.dispStrategy = schedule.val;
    newStrategy.dispMask = dispMask.val;
  } else if (compCap == 89) {
    if (!eitDispatchUsable()) return ncclSuccess;

    switch (cuMaskPolicy) {
      case PCCLX_KERNEL_DISPATCH_AUTO:        NCCLCHECK(deviceAutoMask(device, compCap, &cuMask)); break;
      case PCCLX_KERNEL_DISPATCH_DRV_DEFAULT: cuMask = 0; break;
      default: break;
    }
    ppuDispatchStrategy_0015 schedule = {0};
    schedule.strategy.bits.tbPerCu = (tbPerCu >= 0) ? tbPerCu : 1;
    schedule.strategy.bits.oneBatchPerCe = (oneBatchPerCe >= 0) ? oneBatchPerCe : 1;
    schedule.strategy.bits.d2dDispPingPongEn = 0;
    schedule.strategy1.bits.traverseYStep = 1;
    schedule.strategy1.bits.traverseXStep = 1;
    ppukernelControl_0015 control = {0};
    applyCommonDispatchEnv0015(&schedule, &control, &cuMask);
    newStrategy.dispStrategy = schedule.strategy.val;
    newStrategy.dispStrategy1 = schedule.strategy1.val;
    newStrategy.dispStrategy2 = schedule.strategy2.val;
    newStrategy.kernelControl = control.val;
    newStrategy.dispMask = cuMask;
  } else {
    return ncclSuccess;
  }

  return setCommonDispatchStrategy("user", fn, policyName, compCap, &newStrategy, oldStrategy);
}

ncclResult_t initProfileValue(pcclDispatchType_t dispType, uint32_t lastVal) {
  for (int i = 0; i < NUM_DISPATCH_STRATEGIES; i++) {
    if (g_dispatchStrategyList[i].dispType != dispType) {
      continue;
    }
    for (auto config :  g_dispatchStrategyList[i].modifier) {
      setProfileVal(dispType, lastVal, config.first, config.second);
    }
    g_dispatchStrategyList[i].finalValue = lastVal;
  }
  return ncclSuccess;
}

ncclResult_t setProfileVal (pcclDispatchType_t dispType, uint32_t& lastVal, std::string bitFieldName, int configVal) {
  switch (dispType) {
    case dispatchTypeSchedule_0010: {
      ppuDispatchStrategy_0010 config = {.val = lastVal};
      SET_BITFIELD(config, onebatch_per_ce, bitFieldName, configVal);
      SET_BITFIELD(config, tb_per_cu, bitFieldName, configVal);
      lastVal = config.val;
      break;
    }
    case dispatchTypeSchedule_0015:
      break;
    case dispatchTypeSchedule1_0015:
      break;
    case dispatchTypeSchedule2_0015:
      break;
    case dispatchTypeMask:
      break;
    case dispatchTypeKernelControl:
      break;
    default:
      ERROR("Unknown pattern for dispatch type %d", dispType);
      break;
  }
  return ncclSuccess;
}

ncclResult_t checkProfileMatcher(pcclDispatchType_t dispType, dispatchInfo dInfo, uint32_t* defaultVal) {
  if (pcclParamDispatchTunerEn()) {
    for (int i = 0; i < NUM_DISPATCH_STRATEGIES; i++) {
      if (dispType == g_dispatchStrategyList[i].dispType &&
          dInfo.coll == g_dispatchStrategyList[i].matcher.coll &&
          dInfo.nBytes >= g_dispatchStrategyList[i].matcher.minBytes &&
          dInfo.nBytes < g_dispatchStrategyList[i].matcher.maxBytes &&
          dInfo.nChannels == g_dispatchStrategyList[i].matcher.nChannels
          ) {
        *defaultVal = g_dispatchStrategyList[i].finalValue;
        break;
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t matchDispatchInfoPpu0010(dispatchInfo dInfo, uint64_t* defaultDispMask, uint32_t* schedule, uint32_t* dispMask) {
  if ((dInfo.nChannels > 0) && (dInfo.nChannels % 8 != 0)) {
    if (dInfo.nChannels % 7 == 0) {
      *dispMask = (unsigned int)(defaultDispMask[CeNumIdx7Ce_t]);
    } else if (dInfo.nChannels % 6 == 0) {
      *dispMask = (unsigned int)(defaultDispMask[CeNumIdx6Ce_t]);
    }
  }

  checkProfileMatcher(dispatchTypeSchedule_0010, dInfo, schedule);
  return ncclSuccess;
}
