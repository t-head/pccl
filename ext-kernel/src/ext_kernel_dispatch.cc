/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/* ext_kernel.h must be included first so that ncclResult_t is the real enum
 * type, not the fallback typedef int in ext_kernel_dispatch.h. */
#include "./pccl/ext_kernel_utils.h"
#include "./pccl/ext_kernel_dispatch.h"
#include <hggc.h>

/*
 * PPU dispatch strategy / mask types.
 * Byte-level layout must match src/include/drv_extension.h.
 */
union ppuDispatchStrategy_0010 {
  struct {
    unsigned int        disp_mode             : 1;
    unsigned int        jump_dim              : 1;
    unsigned int        onebatch_per_ce       : 1;
    unsigned int        skew_en               : 1;
    unsigned int        tb_per_cu             : 4;
    unsigned int        traverse_y_step       : 7;
    unsigned int        rsvd_15               : 1;
    unsigned int        traverse_stride_width : 7;
    unsigned int        rsvd_23               : 1;
    unsigned int        wait_en               : 1;
    unsigned int        rsvd_31_25            : 7;
  } bits;
  unsigned int          val;
};

union ppuDispatchMask_0010 {
  struct {
    unsigned int        ce_mask              : 18;
    unsigned int        rsvd_27_18           : 10;
    unsigned int        cu_mask              : 4;
  } bits;
  unsigned int          val;
};

struct ppuDispatchStrategy_0015 {
  union {
    struct {
      uint32_t dispMode          : 1;
      uint32_t jumpDim           : 1;
      uint32_t oneBatchPerCe     : 1;
      uint32_t skewEn            : 1;
      uint32_t tbPerCu           : 4;
      uint32_t rsvd23_8          : 16;
      uint32_t waitEn            : 1;
      uint32_t d2dDispPingPongEn : 1;
      uint32_t startCeEn         : 1;
      uint32_t startCeId         : 4;
      uint32_t reserved          : 1;
    } bits;
    uint32_t val;
  } strategy;

  union {
    struct {
      uint32_t traverseYStep       : 7;
      uint32_t traverseXStep       : 6;
      uint32_t preAllocateEn       : 1;
      uint32_t lastRoundEn         : 1;
      uint32_t lastRoundOnetbPerCe : 1;
      uint32_t lastRoundTbPerCu    : 4;
      uint32_t reserved            : 12;
    } bits;
    uint32_t val;
  } strategy1;

  union {
    struct {
      uint32_t lastRoundBeginNumber : 32;
    } bits;
    uint32_t val;
  } strategy2;
};

union ppukernelControl_0015 {
  struct {
    uint32_t rsvd_13_0       : 14;
    uint32_t fbStallStEn     : 1;
    uint32_t fbStallStType   : 1;
    uint32_t rsvd_24_16      : 9;
    uint32_t blockAgeEn      : 1;
    uint32_t rsvd_31_26      : 6;
  } bits;
  uint32_t val;
};

/*
 * Attribute values for hggcFuncSetAttribute on PPU 1.5 (compCap 89).
 * Matches ppuFuncAttribute in src/include/drv_extension.h.
 */
#define PPU_ATTR_DISPATCH_STRATEGY    128
#define PPU_ATTR_BLOCK_AGE_EN         129
#define PPU_ATTR_DISPATCH_MASK        130
#define PPU_ATTR_KERNEL_CONTROL       131

static ncclResult_t matchDispatchInfoPpu0010(dispatchInfo dInfo, uint64_t* defaultDispMask,
                                             uint32_t* schedule, uint32_t* dispMask);

ncclResult_t setDispatchStrategy(const void* func, struct ncclFuncDispatchStrategy* oldStrategy,
                                 struct ncclDispatchStrategy defaultStrategy, const char* funcName,
                                 int compCap, dispatchInfo dInfo) {
  if (compCap == 80) {
    uint32_t scheduleVal = defaultStrategy.dispStrategy;
    uint32_t dispMaskVal = (unsigned int)defaultStrategy.dispMask[CeNumIdxDefault_t];
    ppuDispatchStrategy_0010 schedule = {.val = scheduleVal};
    ppuDispatchMask_0010 dispMask = {.val = dispMaskVal};

    /* Adjust mask based on channel count. */
    matchDispatchInfoPpu0010(dInfo, defaultStrategy.dispMask, &schedule.val, &dispMask.val);

    if (oldStrategy->dispStrategy != schedule.val) {
      VERBOSE(NCCL_TUNING, "Func %s set dispatch strategy 0x%08x", funcName, schedule.val);
      HGGCCHECK(hggcFuncSetAttribute(func, (hggcFuncAttribute)PPU_ATTR_DISPATCH_STRATEGY, schedule.val));
      oldStrategy->dispStrategy = schedule.val;

      /* Export config to driver API. */
      oldStrategy->launchConfig.blocksPerMultiprocessor = schedule.bits.tb_per_cu;
      oldStrategy->launchConfig.gridStepY = schedule.bits.traverse_y_step;
      oldStrategy->launchConfig.gridStepX = 1;
    }

    if ((oldStrategy->dispMask & 0xffffffff) != dispMask.val) {
      VERBOSE(NCCL_TUNING, "Func %s set dispatch mask 0x%08x", funcName, dispMask.val);
      HGGCCHECK(hggcFuncSetAttribute(func, (hggcFuncAttribute)PPU_ATTR_DISPATCH_MASK, dispMask.val));
      oldStrategy->dispMask = dispMask.val & 0xffffffff;
    }
  } else if (compCap == 89) {
    /* Dispatch strategy (3 x uint32_t). */
    if ((oldStrategy->dispStrategy ^ defaultStrategy.dispStrategy) |
        (oldStrategy->dispStrategy1 ^ defaultStrategy.dispStrategy1) |
        (oldStrategy->dispStrategy2 ^ defaultStrategy.dispStrategy2)) {
      VERBOSE(NCCL_TUNING, "Func %s set dispatch strategy 0x%08x 0x%08x 0x%08x",
              funcName, defaultStrategy.dispStrategy, defaultStrategy.dispStrategy1,
              defaultStrategy.dispStrategy2);
      HGGCCHECK(hggcFuncSetAttribute(func, (hggcFuncAttribute)PPU_ATTR_DISPATCH_STRATEGY,
                                     defaultStrategy.dispStrategy));
      HGGCCHECK(hggcFuncSetAttribute(func, (hggcFuncAttribute)(PPU_ATTR_DISPATCH_STRATEGY + 1),
                                     defaultStrategy.dispStrategy1));
      HGGCCHECK(hggcFuncSetAttribute(func, (hggcFuncAttribute)(PPU_ATTR_DISPATCH_STRATEGY + 2),
                                     defaultStrategy.dispStrategy2));

      oldStrategy->dispStrategy = defaultStrategy.dispStrategy;
      oldStrategy->dispStrategy1 = defaultStrategy.dispStrategy1;
      oldStrategy->dispStrategy2 = defaultStrategy.dispStrategy2;

      /* Export launch config. */
      ppuDispatchStrategy_0015 sched = {
        .strategy = {.val = defaultStrategy.dispStrategy},
        .strategy1 = {.val = defaultStrategy.dispStrategy1},
        .strategy2 = {.val = defaultStrategy.dispStrategy2}
      };
      oldStrategy->launchConfig.blocksPerMultiprocessor = sched.strategy.bits.tbPerCu;
      oldStrategy->launchConfig.gridStepX = sched.strategy1.bits.traverseXStep;
      oldStrategy->launchConfig.gridStepY = sched.strategy1.bits.traverseYStep;
      oldStrategy->launchConfig.subDispatch = defaultStrategy.launchConfig.subDispatch;
    }

    /* Kernel control. */
    ppukernelControl_0015 control = {.val = defaultStrategy.kernelControl};
    if (control.bits.fbStallStEn == 1 && control.val != oldStrategy->kernelControl) {
      VERBOSE(NCCL_TUNING, "Func %s set kernel control 0x%08x", funcName, control.val);
      HGGCCHECK(hggcFuncSetAttribute(func, (hggcFuncAttribute)PPU_ATTR_KERNEL_CONTROL, control.val));
      oldStrategy->kernelControl = control.val;
    }

    /* CU mask. */
    uint64_t cuMask = defaultStrategy.dispMask[CeNumIdxDefault_t];
    if (oldStrategy->dispMask != cuMask) {
      VERBOSE(NCCL_TUNING, "Func %s set cumask 0x%010lx", funcName, cuMask);
      HGGCCHECK(hggcFuncSetAttribute(func, (hggcFuncAttribute)PPU_ATTR_DISPATCH_MASK, (uint32_t)cuMask));
      oldStrategy->dispMask = cuMask;
    }
  }
  return ncclSuccess;
}

static ncclResult_t matchDispatchInfoPpu0010(dispatchInfo dInfo, uint64_t* defaultDispMask,
                                             uint32_t* schedule, uint32_t* dispMask) {
  (void)schedule;
  if ((dInfo.nChannels > 0) && (dInfo.nChannels % 8 != 0)) {
    if (dInfo.nChannels % 7 == 0) {
      *dispMask = (unsigned int)(defaultDispMask[CeNumIdx7Ce_t]);
    } else if (dInfo.nChannels % 6 == 0) {
      *dispMask = (unsigned int)(defaultDispMask[CeNumIdx6Ce_t]);
    }
  }
  return ncclSuccess;
}
