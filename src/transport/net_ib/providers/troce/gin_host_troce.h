/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _GIN_HOST_GDAKI_TROCE_H_
#define _GIN_HOST_GDAKI_TROCE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <linux/types.h>
#include <hggc.h>
#include <hggc_runtime_api.h>
#include "pccl_internal.h"
#include "gin/gin_host.h"

// Host side of the GDAKI_TROCE GIN backend. Mirrors the GDAKI entry points 1:1 so the
// two backends stay structurally interchangeable; the difference is that the
// queue pairs are built with plain verbs (and, from step 2 on, mapped so the GPU
// can post WQEs itself) instead of going through DOCA-GPUNetIO.
ncclResult_t ncclGinGdakiTroceCreateContext(void *collComm, int nSignals, int nCounters, int nContexts,
                                       int queueDepth, int trafficClass, void **outGinCtx,
                                       ncclNetDeviceHandle_t **outDevHandle);
ncclResult_t ncclGinGdakiTroceDestroyContext(void *ginCtx);
ncclResult_t ncclGinGdakiTroceRegMrSym(void *collComm, void *data, size_t size, int type,
                                  uint64_t mr_flags, void **mhandle, void **ginHandle);
ncclResult_t ncclGinGdakiTroceDeregMrSym(void *collComm, void *mhandle);
ncclResult_t ncclGinGdakiTroceProgress(void *ginCtx);
ncclResult_t ncclGinGdakiTroceQueryLastError(void *ginCtx, bool *hasError);

#endif /* _GIN_HOST_GDAKI_TROCE_H_ */
