/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef __OKI_CE_COLL_H__
#define __OKI_CE_COLL_H__
#include "pccl_internal.h"
#include "ce_coll.h"
#include "cudawrap.h"

#define PCCL_MIN_KMD_VERSION_FOR_OKI PCCL_SYSTEM_DRIVER_VERSION(2, 1, 0)
#define PCCL_LT_OKI_MAX_DMA_BYTE_CNT ((1<<30) - 256)
#define PCCL_LT_OKI_QP_DEPTH (32*1024)
// hggc allows at most this many threads per block.
#define PCCL_OKI_MAX_THREADS 1024

ncclResult_t pcclCeOkiAllReduce(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream);
ncclResult_t pcclCeOkiReduceScatter(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream);
ncclResult_t pcclCeOkiAllGather(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream);

#endif // end include guard
