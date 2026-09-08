/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_DEVICE_KERNEL_DISPATCH_H_
#define PCCL_DEVICE_KERNEL_DISPATCH_H_

#include "pccl.h"
#include "utility.h"

/* Per-kernel dispatch config for user-launched kernels (Device API).
 * Precedence: PCCL_* env > config > default. */
typedef enum {
  PCCLX_KERNEL_DISPATCH_AUTO = 0,         // let pccl pick the config
  PCCLX_KERNEL_DISPATCH_DRV_DEFAULT = 1,  // use the driver default config
  PCCLX_KERNEL_DISPATCH_CUSTOM = 2,       // use config cuMask/ceMask
  PCCLX_KERNEL_DISPATCH_POLICY_COUNT
} pcclxKernelCuMaskPolicy_t;

/* Init with PCCLX_KERNEL_DISPATCH_CONFIG_INITIALIZER. -1 = "use default";
 * cuMask/ceMask are read only under PCCLX_KERNEL_DISPATCH_CUSTOM. */
typedef struct pcclxKernelDispatchConfig {
  size_t size;
  pcclxKernelCuMaskPolicy_t policy;
  int oneBatchPerCe;                      // valid values: 0, 1
  int tbPerCu;                            // valid values: 0 ~ 8
  uint64_t cuMask;
  uint32_t ceMask;
  int reserved[23];
} pcclxKernelDispatchConfig_t;

#define PCCLX_KERNEL_DISPATCH_CONFIG_INITIALIZER \
  {                                              \
      sizeof(pcclxKernelDispatchConfig_t), PCCLX_KERNEL_DISPATCH_DRV_DEFAULT, 1, 1, 0, 0, {0}}

static_assert(sizeof(pcclxKernelDispatchConfig_t) == 128, "config must stay 128 bytes.");

#ifndef __clang_llvm_bitcode_lib__

#include <hggc.h>

/* Configure a kernel by host symbol. NULL config = the AUTO policy. */
PCCL_EXTERN_C __host__ pcclResult_t pcclxKernelSetDispatchConfig(
    const void* func, const pcclxKernelDispatchConfig_t* config);

/* Same, for a kernel known only by driver handle. */
PCCL_EXTERN_C __host__ pcclResult_t pcclxKernelSetDispatchConfigByHandle(
    HGfunction func, const pcclxKernelDispatchConfig_t* config);

#endif // __clang_llvm_bitcode_lib__

#endif // PCCL_DEVICE_KERNEL_DISPATCH_H_
