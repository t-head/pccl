/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*************************************************************************
 * Kernel launch macro for ext-kernel plugin.
 * Replaces the LAUNCH_KERNEL from src/include/utils.h
 ************************************************************************/

#ifndef EXT_KERNEL_PCCL_LAUNCH_H_
#define EXT_KERNEL_PCCL_LAUNCH_H_

#include <hggc_ad.h>
#include "ext_kernel_utils.h"

#ifndef LAUNCH_KERNEL
#define LAUNCH_KERNEL(kernel, gridDim, blockDim, args, sharedMem, stream, subDisaptch, launchConfig)                                                                  \
  do                                                                                                                                                                  \
  {                                                                                                                                                                   \
    HGlaunchConfigAD cfg = {gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, (unsigned int)sharedMem, stream, nullptr, 0};                        \
    HGlaunchAttributeAD attr[1];                                                                                                                                      \
    attr[0].id = HGAD_LAUNCH_ATTRIBUTE_SCHED_PREFERENCE;                                                                                                              \
    attr[0].value.schedPreference.blocksPerMultiprocessor = launchConfig.blocksPerMultiprocessor;                                                                     \
    attr[0].value.schedPreference.gridStepX = launchConfig.gridStepX;                                                                                                 \
    attr[0].value.schedPreference.gridStepY = launchConfig.gridStepY;                                                                                                 \
    attr[0].value.schedPreference.flags = (launchConfig.subDispatch >= 0 ? launchConfig.subDispatch : subDisaptch) ? HGAD_SCHED_PREFER_SYMM : HGAD_SCHED_PREFER_NONE; \
    cfg.attrs = attr;                                                                                                                                                 \
    cfg.numAttrs = 1;                                                                                                                                                 \
    HGfunction hfunc;                                                                                                                                                 \
    HGGCCHECK(hggcGetFuncBySymbol(&hfunc, (const void *)kernel));                                                                                                     \
    HGGCCHECK((hggcError_t)hgLaunchKernelExAD(&cfg, hfunc, args, nullptr));                                                                                           \
  } while (0)
#endif

#endif /* EXT_KERNEL_PCCL_LAUNCH_H_ */
