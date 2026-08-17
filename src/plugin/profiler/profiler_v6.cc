/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "pccl_profiler.h"
#include "checks.h"
#include <dlfcn.h>

static ncclProfiler_v6_t* ncclProfiler_v6;

ncclProfiler_t* getNcclProfiler_v6(void* lib) {
  ncclProfiler_v6 = (ncclProfiler_v6_t*)dlsym(lib, "pcclProfiler_v6");
  if (ncclProfiler_v6) {
    INFO(NCCL_INIT, "PROFILER/Plugin: Loaded %s (v6)", ncclProfiler_v6->name);
    return ncclProfiler_v6;
  }
  return NULL;
}

