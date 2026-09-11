/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "core.h"
#include "drv_extension.h"
#include "pccl_device/kernel_dispatch.h"

static const char* policyNameUser(int policy) {
  switch (policy) {
    case PCCLX_KERNEL_DISPATCH_AUTO:         return "policy AUTO";
    case PCCLX_KERNEL_DISPATCH_DRV_DEFAULT:  return "policy DRV_DEFAULT";
    case PCCLX_KERNEL_DISPATCH_CUSTOM:       return "policy CUSTOM";
    default:                                 return "policy UNKNOWN";
  }
}

// Translate the public policy into the mask decision the host layer expects,
// then hand the whole job over to it.
static pcclResult_t setUserDispatchConfig(const struct pcclDispatchFunc* fn,
                                      const pcclxKernelDispatchConfig_t* config) {
  int compCap = ncclHggcCompCap();
  if (compCap != 80 && compCap != 89) return ncclSuccess;

  pcclxKernelDispatchConfig_t cfg = PCCLX_KERNEL_DISPATCH_CONFIG_INITIALIZER;
  if (config != NULL) {
    // The struct has a fixed size, so a mismatch means the caller never ran it
    // through PCCLX_KERNEL_DISPATCH_CONFIG_INITIALIZER.
    if (config->size != sizeof(*config)) {
      WARN("DevApi: bad config size %zu", config->size);
      return ncclInvalidArgument;
    }
    if (config->policy < 0 || config->policy >= PCCLX_KERNEL_DISPATCH_POLICY_COUNT) {
      WARN("DevApi: bad config policy %d", config->policy);
      return ncclInvalidArgument;
    }
    cfg.policy = config->policy;
    cfg.oneBatchPerCe = config->oneBatchPerCe;
    cfg.tbPerCu = config->tbPerCu;
    cfg.d2dDispPingPongEn = config->d2dDispPingPongEn;

    if (cfg.policy == PCCLX_KERNEL_DISPATCH_CUSTOM) {
      cfg.cuMask = config->cuMask;
      cfg.ceMask = config->ceMask;
    }
  }

  return setUserDispatchStrategy(fn, policyNameUser(cfg.policy), compCap, cfg.oneBatchPerCe,
                                 cfg.tbPerCu, cfg.d2dDispPingPongEn, (int)cfg.policy,
                                 cfg.cuMask, cfg.ceMask);
}

NCCL_API(pcclResult_t, pcclxKernelSetDispatchConfig, const void* func, const pcclxKernelDispatchConfig_t* config);
pcclResult_t pcclxKernelSetDispatchConfig(const void* func, const pcclxKernelDispatchConfig_t* config) {
  if (func == NULL) { WARN("DevApi: NULL kernel"); return ncclInvalidArgument; }
  struct pcclDispatchFunc fn = {.symbol = func, .handle = NULL};
  return setUserDispatchConfig(&fn, config);
}

NCCL_API(pcclResult_t, pcclxKernelSetDispatchConfigByHandle, HGfunction func, const pcclxKernelDispatchConfig_t* config);
pcclResult_t pcclxKernelSetDispatchConfigByHandle(HGfunction func, const pcclxKernelDispatchConfig_t* config) {
  if (func == NULL) { WARN("DevApi: NULL kernel handle"); return ncclInvalidArgument; }
  struct pcclDispatchFunc fn = {.symbol = NULL, .handle = func};
  return setUserDispatchConfig(&fn, config);
}
