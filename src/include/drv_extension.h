/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef DRV_EXTENSION_H_
#define DRV_EXTENSION_H_

#include "debug.h"
#include <hggc_runtime.h>
#include <hggc.h>

#include "drv_eit/drv_eit_common_types.h"
#include "drv_eit/drv_eit_pccl.h"
#include "check_items.h"
#include <vector>
#include <string>

// both ppu1.0 and ppu1.5. MAX_CE_NUM_PER_DIE is the CE slot stride of the
// driver cuMasks[] array and of cu_mask, not the populated CE count of a die.
#define MAX_PPU_CE_NUM 32
#define MAX_CE_NUM_PER_DIE 8
#define CU_NUM_PER_CE 4

// ppu1.0: 18-bit CE dispatch mask, the two halves are bits 0~7 plus 16 and
// bits 8~15 plus 17; the 0_7 / 8_15 variants drop the odd high bit.
#define CE_DISPATCH_MASK_L 0x100FF
#define CE_DISPATCH_MASK_H 0x2FF00
#define CE_DISPATCH_MASK_0_7 0xFF
#define CE_DISPATCH_MASK_8_15 0xFF00
#define MAX_SELECT_CE_IDX 15

// ppu1.5: 64-bit CU dispatch mask, CE_NUM_PER_DIE_PPU0015 CEs populated per die
#define DIE_NUM_PPU0015 2
#define CE_NUM_PER_DIE_PPU0015 5

#define HG_DMA_RING_PER_DIE 8

#define HG_MEMHOSTALLOC_64K_PAGE 0x80000

// Ranges the dispatch hardware accepts. Programming a value outside them leaves
// the kernel undispatched, so an out-of-range request keeps the default.
#define MIN_TB_PER_CU 1
#define MAX_TB_PER_CU 8
#define MAX_ONE_BATCH_PER_CE 1
#define MAX_D2D_DISP_PING_PONG_EN 1

#define CEHCK_ENV_SETTING(RES, ENV)              \
  do {                                           \
      int64_t temp_val = (ENV);                  \
      if (temp_val >= 0) {                       \
          (RES) = temp_val;                      \
      }                                          \
  } while (0)

#define CEHCK_ENV_SETTING_RANGE(RES, ENV, LO, HI, NAME)                           \
  do {                                                                            \
      int64_t temp_val = (ENV);                                                   \
      if (temp_val >= 0) {                                                        \
          if (temp_val < (LO) || temp_val > (HI)) {                               \
              WARN("Dispatch: env %s=%ld out of range [%d, %d], keep default %u", \
                   (NAME), (long)temp_val, (int)(LO), (int)(HI), (unsigned)(RES));\
          } else {                                                                \
              (RES) = temp_val;                                                   \
          }                                                                       \
      }                                                                           \
  } while (0)

union ppuIcnCeInfo {
  struct {
    int ce_mask              : 18;
    int rsvd_29_18           : 12;
    int full_ce_left         : 1;
    int full_ce_right        : 1;
  } bits;
  int val;
};

union  ppuDispatchStrategy_0010 {
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

struct ppuDispatchMask_0015 {
  uint64_t cu_mask;
  uint64_t cu_mask1;
  uint64_t cu_mask2;
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

// The kernel to configure: either the host symbol a <<<>>> launch uses, or a
// driver handle from module loading. Exactly one member is set.
struct pcclDispatchFunc {
  const void* symbol;
  HGfunction handle;
};

enum ppuFuncAttribute {
  ppuFuncAttributeMaxDynamicSharedMemorySize        = 8, /**< Maximum dynamic shared memory size */
  ppuFuncAttributePreferredSharedMemoryCarveout     = 9, /**< Preferred shared memory-L1 cache split */
  ppuFuncAttributeDispatchStrategy                  = 128, /**< Dispatch Strategy */
  ppuFuncAttributeBlockAgeEn                        = 129, /* block age priority when scheduling warps*/
  ppuFuncAttributeDispatchMask                      = 130, /* Dispatch mask*/
  ppuFuncAttributeMax
};

enum ppuDeviceAttr {
  ppuDevAttrPpuId                             = 201, /**< Device's PPU ID */
  ppuDevAttrDevDispatchMask                   = 202, /**< Device's dispatch mask (CE and CU mask) */
  ppuDevAttrIcnDispatchMask                   = 203, /**< Device's dispatch CE mask that is icn performance friendly */
  ppuDevAttrPrivateEnd
};

enum ppuDeviceP2PAttr {
    ppuDevP2PAttrMinPath                      = 256, /**< Path count between two devices */
    ppuDevP2PAttrBandWidth                    = 257, /**< Band width in MB */
    ppuDevP2PAttrIsNeighbor                   = 258, /**< Whether two devices are direct neighbor or not */
};

enum ppuEitFuncAttribute {
    ppuEitFuncAttributeKernelControl     = 0, /**< Kernel control setting */
    ppuEitFuncAttributeDispatchCuMask    = 1, /**< Kernel dispatch ce/cumask setting */
    ppuEitFuncAttributeDispatchStrategy  = 200, /**< Kernel dispatch strategy setting */
};

class hggcEitInterface {
public:
  hggcEitInterface(hggcEitInterface const&) = delete;
  void operator=(hggcEitInterface const&) = delete;
  static hggcEitInterface& getInstance() {
    static hggcEitInterface instance;
    return instance;
  }

  bool checkExtTable() {
    if (pFuncTable == NULL) return false;
    else return true;
  }

  bool checkDispatchExtTable() {
    if (pFuncTable == NULL || pFuncTable->hgEitFuncSetAttribute == NULL) return false;
    else return true;
  }

  hggcError_t wrapHggcDeviceGetP2PAttributeMinPath(int* value, int srcDevice, int dstDevice) {
    return hggcDeviceGetP2PAttribute(value, (hggcDeviceP2PAttr)(ppuDevP2PAttrMinPath), srcDevice, dstDevice);
  }

  hggcError_t wrapHggcDeviceGetAttributePpuId(int* value, int device) {
    return hggcDeviceGetAttribute(value, (hggcDeviceAttr)(ppuDevAttrPpuId), device);
  }

  hggcError_t wrapHggcDeviceGetAttributeDevDispatchMask(int* value, int device) {
    return hggcDeviceGetAttribute(value, (hggcDeviceAttr)(ppuDevAttrDevDispatchMask), device);
  }

  hggcError_t wrapHggcDeviceGetAttributeIcnDispatchMask(int* value, int device) {
    return hggcDeviceGetAttribute(value, (hggcDeviceAttr)(ppuDevAttrIcnDispatchMask), device);
  }

  hggcError_t wrapHggcDeviceGetAttributeCuMasks(uint8_t* value, int device) {
    hggcError_t res = hggcSuccess;
    if (pFuncTable != NULL && pFuncTable->hgGetExtDeviceInfo != NULL) {
      HGextDeviceInfo_v2 devInfo;
      res = (hggcError_t)pFuncTable->hgGetExtDeviceInfo(device, &devInfo);

      if (res == hggcSuccess) {
        for (int i = 0 ; i < MAX_PPU_CE_NUM; ++i) {
          value[i] = devInfo.cuMasks[i];
        }
      }
    } else {
      WARN("EIT: hgGetExtDeviceInfo not available.");
    }
    return res;
  }

  // force cast from HGresult(eit api return value) to hggcError_t.
  hggcError_t wrapHggcPointerGetIpcAttribute(HGdeviceptr devPtr, HGpointerIpc_attribute* ptrIpcAttr) {
    if (pFuncTable != NULL && pFuncTable->hgPointerGetIpcAttribute != NULL) {
      return (hggcError_t)pFuncTable->hgPointerGetIpcAttribute(devPtr, ptrIpcAttr);
    } else {
      WARN("EIT: hgPointerGetIpcAttribute not available");
      return hggcSuccess;
    }
  }

  hggcError_t wrapHggcMemDump(void* cpuPtr, HGdeviceptr devPtr, ssize_t size) {
    if (pFuncTable != NULL && pFuncTable->hgMemDump != NULL) {
      return (hggcError_t)pFuncTable->hgMemDump(cpuPtr, devPtr, size);
    } else {
      WARN("EIT: hgMemDump not available");
      return hggcSuccess;
    }
  }

  hggcError_t wrapHggcMemLock(HGdeviceptr devPtr, void** ppCpuPtr) {
    if (pFuncTable != NULL && pFuncTable->hgMemLock != NULL) {
      return (hggcError_t)pFuncTable->hgMemLock(devPtr, ppCpuPtr);
    } else {
      WARN("EIT: hgMemLock not available");
      return hggcSuccess;
    }
  }

  hggcError_t wrapHggcMemUnlock(HGdeviceptr dptr) {
    if (pFuncTable != NULL && pFuncTable->hgMemUnlock != NULL) {
      return (hggcError_t)pFuncTable->hgMemUnlock(dptr);
    } else {
      WARN("EIT: hgMemUnlock not available");
      return hggcSuccess;
    }
  }

  hggcError_t wrapHggcCoreDumpGenerate(int devId, const char* fileName, HGCoredumpGenerationFlags flags) {
    HGdevice device;
    // force convert devId to hgDevice for compile reason.
    device = (HGdevice)devId;

    if (pFuncTable != NULL && pFuncTable->hgCoredumpGenerate != NULL) {
      return (hggcError_t)pFuncTable->hgCoredumpGenerate(device, fileName, flags);
    } else {
      WARN("EIT: hgCoredumpGenerate not available");
      return hggcSuccess;
    }
  }

  hggcError_t wrapHggcStreamCreateForCopyEngine(hggcStream_t* stream) {
    if (pFuncTable != NULL && pFuncTable->hgStreamCreate!= NULL) {
      return (hggcError_t)pFuncTable->hgStreamCreate((HGstream*)stream, HG_EIT_STREAM_NON_BLOCKING | HG_EIT_STREAM_COPY_ENGINE);
    } else {
      ERROR("EIT: hgStreamCreate not available");
      return hggcSuccess;
    }
  }

  hggcError_t wrapHggcEitFuncSetAttribute(const struct pcclDispatchFunc* fn, HGeitFuncAttribute attr, void* pData, uint32_t size) {
    if (pFuncTable != NULL && pFuncTable->hgEitFuncSetAttribute != NULL) {
      HGfunction pFunc = fn->handle;
      // A driver handle is already what the EIT table wants; a host symbol is not.
      if (pFunc == NULL) {
        hggcError_t res = hggcGetFuncBySymbol(reinterpret_cast<hggcFunction_t*>(&pFunc), fn->symbol);
        if (res != hggcSuccess) return res;
      }
      return (hggcError_t)pFuncTable->hgEitFuncSetAttribute(pFunc, attr, pData, size);
    } else {
      WARN("EIT: hgEitFuncSetAttribute not available");
      return hggcSuccess;
    }
  }

  hggcError_t wrapHggcFuncSetDispatchStrategy(const struct pcclDispatchFunc* fn, void* pData, uint32_t size) {
    return wrapHggcEitFuncSetAttribute(fn, (HGeitFuncAttribute)(ppuEitFuncAttributeDispatchStrategy), pData, size);
  }

  hggcError_t wrapHggcFuncSetKernelControl(const struct pcclDispatchFunc* fn, void* pData, uint32_t size) {
    return wrapHggcEitFuncSetAttribute(fn, (HGeitFuncAttribute)(ppuEitFuncAttributeKernelControl), pData, size);
  }

  hggcError_t wrapHggcFuncSetDispatchCuMask(const struct pcclDispatchFunc* fn, void* pData, uint32_t size) {
    return wrapHggcEitFuncSetAttribute(fn, (HGeitFuncAttribute)(ppuEitFuncAttributeDispatchCuMask), pData, size);
  }

  hggcError_t wrapHggcStreamHintChannel(hggcStream_t stream, HGstreamChannelType type, uint32_t ringId) {
    if (pFuncTable != NULL && pFuncTable->hgStreamHintChannel != NULL) {
      return (hggcError_t)pFuncTable->hgStreamHintChannel((HGstream)stream, type, ringId);
    } else {
      WARN("EIT: hgStreamHintChannel not available");
      return hggcSuccess;
    }
  }

  HGresult wrapHgMemGetAllocationGranularityEx(size_t *granularity, const HGmemAllocationProp *prop, HGmemAllocationGranularity_flags option) {
    if (pFuncTable != NULL && pFuncTable->hgMemGetAllocationGranularityEx!= NULL) {
      return (HGresult)pFuncTable->hgMemGetAllocationGranularityEx(granularity, prop, option);
    } else {
      ERROR("EIT: hgMemGetAllocationGranularityEx not available");
      return HGGC_SUCCESS;
    }
  }

  HGresult wrapHgMemCreateEx(HGmemGenericAllocationHandle *handle, size_t size, const HGmemAllocationProp *prop, unsigned long long flags) {
    if (pFuncTable != NULL && pFuncTable->hgMemCreateEx!= NULL) {
      return (HGresult)pFuncTable->hgMemCreateEx(handle, size, prop, flags);
    } else {
      ERROR("EIT: hgMemCreateEx not available");
      return HGGC_SUCCESS;
    }
  }

  HGresult wrapHggcDriverGetVersion(HGdriverVersionInfo* driverVersion) {
    if (pFuncTable != NULL && pFuncTable->struct_size >= offsetof(HGeitPccl, hgDriverGetVersion) + sizeof(pFuncTable->hgDriverGetVersion) && pFuncTable->hgDriverGetVersion != NULL) {
      return (HGresult)pFuncTable->hgDriverGetVersion(driverVersion);
    } else {
      ERROR("EIT: hgDriverGetVersion not available");
      return HGGC_ERROR_UNKNOWN;
    }
  }

private:
  hggcEitInterface() {
    hggcError_t getEitRes = hggcSuccess;
    pFuncTable = NULL;
    getEitRes = hggcGetExportTable(reinterpret_cast<const void**>(&pFuncTable), reinterpret_cast<const hggcUUID_t *>(&HG_EIT_ID_PCCL));
    if (getEitRes != hggcSuccess || pFuncTable == NULL) {
      WARN("EIT: Fail to get hggc pccl eit table, hggcPcclEit: %p", pFuncTable);
      pFuncTable = NULL;
    }
  }

  const HGeitPccl *pFuncTable;
};

/*
 * =============================================================================
 * HARDWARE RESOURCE POLICIES
 * -----------------------------------------------------------------------------
 * These structures define hardware-level constraints, resource allocation
 * policies (e.g., CUs, CEs, and Channel limits).
 * =============================================================================
 */

/* --- Data Structures --- */
typedef enum : int {
  SM_POLICY_RES_FULL = 0,     // Use all available SMs without any limits on amount or layout
  SM_POLICY_RES_LIMITED = 1,  // Limits the total amount of SMs used, but the physical layout is flexible
  SM_POLICY_LAYOUT_FIXED = 2  // Strictly limits both the amount and layout of SMs
} smPolicyLevel_t;

struct smResourceConfig_t {
  smPolicyLevel_t smResourcePolicy;
  int cuNumUsePerCe;
  int ceNumUsePerDie;
  int defaultMaxChannelNum;
};

struct SmArchConfig {
  // --- Conditions ---
  int compCap;                            // Compute Capability (e.g., 80, 89)
  PPU_TYPE ppuType;                       // Specific PPU Type (-1 for any)
  int nRanks;                             // Specific num of ranks (-1 for any)
  pcclICNRemoteDeviceType_t icnRmtType;   // Remote connection type (-1 for any)
  // --- Policy Results ---
  struct smResourceConfig_t config;
};

/* --- Functions --- */
ncclResult_t initSmResourcesConfig(ncclComm_t comm);
ncclResult_t getPpuDeviceDispatchCuMasks(uint64_t* value, int* totalCuNum, int device, int compCap, int nDie, PPU_TYPE ppuType, int nChannels);

/*
 * =============================================================================
 * KERNEL DISPATCH STRATEGIES
 * -----------------------------------------------------------------------------
 * These structures define how kernels are scheduled and launched. They control
 * the dispatching masks and strategy.
 * =============================================================================
 */

/* --- Data Structures --- */
struct dispatchStrategyExAD {
  unsigned int blocksPerMultiprocessor;
  unsigned int gridStepX;
  unsigned int gridStepY;
  int subDispatch;
  int cooperative;
};

struct ncclFuncDispatchStrategy {
  uint32_t dispStrategy;
  uint32_t dispStrategy1;
  uint32_t dispStrategy2;
  uint64_t dispMask;
  uint32_t kernelControl;
  struct dispatchStrategyExAD launchConfig;
};

typedef enum {
  CeNumIdxDefault_t,
  CeNumIdx7Ce_t,
  CeNumIdx6Ce_t,
  CeNumIdxNum_t
} CeNumIdx_t;

struct ncclDispatchStrategy {
  uint32_t dispStrategy;
  uint32_t dispStrategy1;
  uint32_t dispStrategy2;
  uint64_t dispMask[CeNumIdxNum_t];
  uint32_t kernelControl;
  struct dispatchStrategyExAD launchConfig;
};

typedef enum {
  dispatchTypeSchedule_0010,
  dispatchTypeSchedule_0015,
  dispatchTypeSchedule1_0015,
  dispatchTypeSchedule2_0015,
  dispatchTypeMask,
  dispatchTypeKernelControl
} pcclDispatchType_t;

typedef struct {
    ncclFunc_t coll;
    int nNodes;
    int nChannels;
    size_t nBytes;
} dispatchInfo;

typedef struct {
    ncclFunc_t coll;
    int nNodes;
    int nChannels;
    size_t minBytes;
    size_t maxBytes;
} dispatchMatcher;

typedef struct {
    dispatchMatcher matcher;
    pcclDispatchType_t dispType;
    std::vector<std::pair<std::string, int>> modifier;
    uint32_t finalValue;
} dispatchTuner;

/* --- Functions --- */
bool checkIcnDispatchMask(int icnDispatchMask, PPU_TYPE ppuType);
bool checkFullCeLeft(int icnDispatchMask, PPU_TYPE ppuType);
bool checkFullCeRight(int icnDispatchMask, PPU_TYPE ppuType);
ncclResult_t initDefaultDispatchStrategy(ncclComm_t comm, int nDies);
ncclResult_t setNativeDispatchStrategy(const void* func, struct ncclFuncDispatchStrategy* oldStrategy, struct ncclDispatchStrategy defaultStrategy, const char* funcName,
                                int compCap, dispatchInfo dInfo);
ncclResult_t setUserDispatchStrategy(const struct pcclDispatchFunc* fn, const char* policyName, int compCap,
                                     int oneBatchPerCe, int tbPerCu, int d2dDispPingPongEn,
                                     int cuMaskPolicy, uint64_t cuMask, uint32_t ceMask);
// for dispatch strategy tuning
ncclResult_t setProfileVal (pcclDispatchType_t dispType, uint32_t& lastVal, std::string bitFieldName, int configVal);
ncclResult_t initProfileValue(pcclDispatchType_t dispType, uint32_t defaultVal);
ncclResult_t checkProfileMatcher (pcclDispatchType_t dispType, size_t nBytes, dispatchMatcher mather, uint32_t &defaultVal);
ncclResult_t matchDispatchInfoPpu0010(dispatchInfo info, uint64_t* defaultDispMask, uint32_t* schedule, uint32_t* dispMask);
#endif
