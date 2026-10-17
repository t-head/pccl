/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _GIN_HOST_GDAKI_MLX_H_
#define _GIN_HOST_GDAKI_MLX_H_

#ifndef DOCA_VERBS_USE_HGGC_WRAPPER
#define DOCA_VERBS_USE_HGGC_WRAPPER
#endif

#ifndef DOCA_VERBS_USE_NET_WRAPPER
#define DOCA_VERBS_USE_NET_WRAPPER
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <linux/types.h>
#include <hggc.h>
#include <hggc_runtime_api.h>
#include "pccl_internal.h"
#include "gin/gin_host.h"

ncclResult_t ncclGinGdakiMlxCreateContext(void *collComm, int nSignals, int nCounters, int nContexts, int queueDepth,
                                       int trafficClass, void **outGinCtx, ncclNetDeviceHandle_t **outDevHandle);
ncclResult_t ncclGinGdakiMlxDestroyContext(void *ginCtx);
ncclResult_t ncclGinGdakiMlxRegMrSym(void *collComm, void *data, size_t size, int type, uint64_t mr_flags, void **mhandle,
                                  void **ginHandle);
ncclResult_t ncclGinGdakiMlxDeregMrSym(void *collComm, void *mhandle);
ncclResult_t ncclGinGdakiMlxProgress(void *ginCtx);
ncclResult_t ncclGinGdakiMlxQueryLastError(void *ginCtx, bool *hasError);

#endif
