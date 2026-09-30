/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "pccl_gin.h"
#include "comm.h"
#include "proxy.h"
#include <dlfcn.h>

static ncclGin_v13_t* ncclGin_v13;

ncclGin_t* getNcclGin_v13(void* lib) {
  ncclGin_v13 = (ncclGin_v13_t*)dlsym(lib, "pcclGinPlugin_v13");
  if (ncclGin_v13) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded gin plugin %s (v13)", ncclGin_v13->name);
    return ncclGin_v13;
  }
  return nullptr;
}


