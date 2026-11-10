/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _GIN_HOST_TROCE_H_
#define _GIN_HOST_TROCE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <linux/types.h>
#include <hggc.h>
#include <hggc_runtime_api.h>
#include "pccl_internal.h"
#include "gin/gin_host.h"

// Host lifecycle and memory-registration interfaces for the TROCE GDA provider.
ncclResult_t ncclGinTroceCreateContext(void *collComm, int nSignals, int nCounters, int nContexts,
                                       int queueDepth, int trafficClass, void **outGinCtx,
                                       ncclNetDeviceHandle_t **outDevHandle);
ncclResult_t ncclGinTroceDestroyContext(void *ginCtx);
ncclResult_t ncclGinTroceRegMrSym(void *collComm, void *data, size_t size, int type,
                                  uint64_t mr_flags, void **mhandle, void **ginHandle);
ncclResult_t ncclGinTroceDeregMrSym(void *collComm, void *mhandle);
ncclResult_t ncclGinTroceProgress(void *ginCtx);
ncclResult_t ncclGinTroceQueryLastError(void *ginCtx, bool *hasError);

#endif /* _GIN_HOST_TROCE_H_ */
