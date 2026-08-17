//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

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
#define CE_DISPATCH_MASK_L 0x100FF
#define CE_DISPATCH_MASK_H 0x2FF00
#define CE_DISPATCH_MASK_0_7 0xFF
#define CE_DISPATCH_MASK_8_15 0xFF00
#define MAX_SELECT_CE_IDX 15
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
    CEHCK_ENV_SETTING(schedule.bits.disp_mode, pcclParamDispMode());
    CEHCK_ENV_SETTING(schedule.bits.tb_per_cu, pcclParamTbPerCu());
    CEHCK_ENV_SETTING(schedule.bits.onebatch_per_ce, pcclParamOneBatchPerCe());
    CEHCK_ENV_SETTING(schedule.bits.traverse_y_step, pcclParamTraverseYStep());

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
    CEHCK_ENV_SETTING(schedule.strategy.bits.dispMode, pcclParamDispMode());
    CEHCK_ENV_SETTING(schedule.strategy.bits.jumpDim, pcclParamJumpDim());
    CEHCK_ENV_SETTING(schedule.strategy.bits.oneBatchPerCe, pcclParamOneBatchPerCe());
    CEHCK_ENV_SETTING(schedule.strategy.bits.skewEn, pcclParamSkewEn());
    CEHCK_ENV_SETTING(schedule.strategy.bits.tbPerCu, pcclParamTbPerCu());
    CEHCK_ENV_SETTING(schedule.strategy.bits.waitEn, pcclParamWaitEn());
    CEHCK_ENV_SETTING(schedule.strategy.bits.d2dDispPingPongEn, pcclParamD2dDispPingPongEn());
    CEHCK_ENV_SETTING(schedule.strategy.bits.startCeEn, pcclParamStartCeEn());
    CEHCK_ENV_SETTING(schedule.strategy.bits.startCeId, pcclParamStartCeId());

    CEHCK_ENV_SETTING(schedule.strategy1.bits.traverseYStep, pcclParamTraverseYStep());
    CEHCK_ENV_SETTING(schedule.strategy1.bits.traverseXStep, pcclParamTraverseXStep());
    CEHCK_ENV_SETTING(schedule.strategy1.bits.preAllocateEn, pcclParamPreAllocEn());
    CEHCK_ENV_SETTING(schedule.strategy1.bits.lastRoundEn, pcclParamLastRoundEn());
    CEHCK_ENV_SETTING(schedule.strategy1.bits.lastRoundOnetbPerCe, pcclParamLastRoundOnetbPerCe());
    CEHCK_ENV_SETTING(schedule.strategy1.bits.lastRoundTbPerCu, pcclParamLastRoundTbPerCu());

    CEHCK_ENV_SETTING(schedule.strategy2.bits.lastRoundBeginNumber, pcclParamLastRoundBeginNumber());

    CEHCK_ENV_SETTING(control.bits.fbStallStEn, pcclParamFbStallStEn());
    CEHCK_ENV_SETTING(control.bits.fbStallStType, pcclParamFbStallStType());
    CEHCK_ENV_SETTING(control.bits.blockAgeEn, pcclParamBlockAgeEn());

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

ncclResult_t setDispatchStrategy(const void* func, struct ncclFuncDispatchStrategy* oldStrategy, struct ncclDispatchStrategy defaultStrategy, const char* funcName,
                                int compCap, dispatchInfo dInfo) {
  if (compCap == 80) {
    ppuDispatchStrategy_0010 schedule = {.val = defaultStrategy.dispStrategy};
    ppuDispatchMask_0010 dispMask = {.val = (unsigned int)defaultStrategy.dispMask[CeNumIdxDefault_t]};

    // check profiles
    matchDispatchInfoPpu0010(dInfo, defaultStrategy.dispMask, &schedule.val, &dispMask.val);

    if (oldStrategy->dispStrategy != schedule.val) {
      VERBOSE(NCCL_TUNING, "Func %s set dispatch strategy 0x%08x", funcName, schedule.val);
      HGGCCHECK(hggcFuncSetAttribute(func, static_cast<hggcFuncAttribute>(ppuFuncAttributeDispatchStrategy), schedule.val));
      oldStrategy->dispStrategy = schedule.val;

      // export config to driver api
      oldStrategy->launchConfig.blocksPerMultiprocessor = schedule.bits.tb_per_cu;
      oldStrategy->launchConfig.gridStepY = schedule.bits.traverse_y_step;
      oldStrategy->launchConfig.gridStepX = 1; // default value
    }

    if ((oldStrategy->dispMask & 0xffffffff) != dispMask.val) {
      VERBOSE(NCCL_TUNING, "Func %s set dispatch mask 0x%08x", funcName, dispMask.val);
      HGGCCHECK(hggcFuncSetAttribute(func, static_cast<hggcFuncAttribute>(ppuFuncAttributeDispatchMask), dispMask.val));
      oldStrategy->dispMask = dispMask.val & 0xffffffff;
    }
  } else if (compCap == 89) {
    ppuDispatchStrategy_0015 schedule = {
      .strategy = {.val = defaultStrategy.dispStrategy},
      .strategy1 = {.val = defaultStrategy.dispStrategy1},
      .strategy2 = {.val = defaultStrategy.dispStrategy2}
    };

    // export config to driver api
    oldStrategy->launchConfig.subDispatch = defaultStrategy.launchConfig.subDispatch;

    if (((oldStrategy->dispStrategy ^ schedule.strategy.val) |
        (oldStrategy->dispStrategy1 ^ schedule.strategy1.val) |
        (oldStrategy->dispStrategy2 ^ schedule.strategy2.val)) > 0) {
      if (pcclParamForceDefaultDisp()) {
        VERBOSE(NCCL_TUNING, "Func %s update dispatch strategy 0x%08x 0x%08x 0x%08x -> 0x%08x 0x%08x 0x%08x, ignored", funcName,
                oldStrategy->dispStrategy, oldStrategy->dispStrategy1, oldStrategy->dispStrategy2,
                schedule.strategy.val, schedule.strategy1.val, schedule.strategy2.val);
      } else {
        VERBOSE(NCCL_TUNING, "Func %s set dispatch strategy 0x%08x 0x%08x 0x%08x", funcName,
                schedule.strategy.val, schedule.strategy1.val, schedule.strategy2.val);
        HGGCCHECK(hggcEitInterface::getInstance().wrapHggcFuncSetDispatchStrategy(func, (void*)&schedule, sizeof(schedule)));
      }
      oldStrategy->dispStrategy = schedule.strategy.val;
      oldStrategy->dispStrategy1 = schedule.strategy1.val;
      oldStrategy->dispStrategy2 = schedule.strategy2.val;

      // export config to driver api
      oldStrategy->launchConfig.blocksPerMultiprocessor = schedule.strategy.bits.tbPerCu;
      oldStrategy->launchConfig.gridStepX = schedule.strategy1.bits.traverseXStep;
      oldStrategy->launchConfig.gridStepY = schedule.strategy1.bits.traverseYStep;
    }

    ppukernelControl_0015 control = {.val = defaultStrategy.kernelControl};
    if (control.bits.fbStallStEn == 1 && control.val != oldStrategy->kernelControl) {
      VERBOSE(NCCL_TUNING, "Func %s set kernel control 0x%08x", funcName, control.val);
      HGGCCHECK(hggcEitInterface::getInstance().wrapHggcFuncSetKernelControl(func, (void*)&control, sizeof(control)));
      oldStrategy->kernelControl = control.val;
    }

    ppuDispatchMask_0015 cuMask = {.cu_mask = defaultStrategy.dispMask[CeNumIdxDefault_t]};
    if (oldStrategy->dispMask != cuMask.cu_mask) {
      if (pcclParamForceDefaultCeMask()) {
        VERBOSE(NCCL_TUNING, "Func %s update cumask 0x%010lx -> 0x%010lx, ignored", funcName, oldStrategy->dispMask, cuMask.cu_mask);
      } else {
        VERBOSE(NCCL_TUNING, "Func %s set cumask 0x%010lx", funcName, cuMask.cu_mask);
        HGGCCHECK(hggcEitInterface::getInstance().wrapHggcFuncSetDispatchCuMask(func, (void*)&cuMask, sizeof(cuMask)));
      }
      oldStrategy->dispMask = cuMask.cu_mask;
    }
  }
  return ncclSuccess;
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
