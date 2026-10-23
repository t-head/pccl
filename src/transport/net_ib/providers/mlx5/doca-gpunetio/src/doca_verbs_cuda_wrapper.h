/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef DOCA_VERBS_HGGC_WRAPPER_H
#define DOCA_VERBS_HGGC_WRAPPER_H

/* PCCL: always use the dlopen wrapper to avoid hard dependency on libhggc.so at
 * link time. Mirrors upstream NCCL which enables DOCA_VERBS_USE_CUDA_WRAPPER
 * unconditionally for the DOCA GPUNetIO sources. */
#ifndef DOCA_VERBS_USE_HGGC_WRAPPER
#define DOCA_VERBS_USE_HGGC_WRAPPER
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef DOCA_VERBS_USE_HGGC_WRAPPER

/* HGGC type declarations for builds without hggc.h */
typedef enum hggcDriverError_enum {
    HGGC_SUCCESS = 0,
    HGGC_ERROR_NOT_INITIALIZED = 3,
} HGresult;
typedef int HGdevice;
typedef unsigned long long HGdeviceptr;
typedef enum HGmemRangeHandleType_enum {
    HG_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = 0x1,
    HG_MEM_RANGE_HANDLE_TYPE_MAX = 0x7FFFFFFF
} HGmemRangeHandleType;

typedef enum HGpointer_attribute_enum {
    HG_POINTER_ATTRIBUTE_SYNC_MEMOPS =
        6, /**< Synchronize every synchronous memory operation initiated on this region */
} HGpointer_attribute;

typedef enum HGdevice_attribute_enum {
    HG_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING =
        118, /**< GPUDirect RDMA writes to the device do not need to be flushed for consumers within
                the scope indicated by the returned attribute. See ::HGGPUDirectRDMAWritesOrdering
                for the numerical values returned here. */
    HG_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED =
        124, /**< Device supports buffer sharing with dma_buf mechanism. */
} HGdevice_attribute;

typedef enum HGflushGPUDirectRDMAWritesScope_enum {
    HG_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER = 100, /**< Blocks until remote writes are visible to
                                                       the HGGC device context owning the data. */
    HG_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES =
        200 /**< Blocks until remote writes are visible to all HGGC device contexts. */
} HGflushGPUDirectRDMAWritesScope;

typedef void *HGcontext;

/* Wrapper function declarations */
HGresult doca_verbs_wrapper_hgDeviceGetAttribute(int *pi, HGdevice_attribute attrib, HGdevice dev);
HGresult doca_verbs_wrapper_hgPointerSetAttribute(const void *value, HGpointer_attribute attribute,
                                                  HGdeviceptr ptr);
HGresult doca_verbs_wrapper_hgMemGetHandleForAddressRange(int *pHandle, HGdeviceptr dptr,
                                                          size_t size,
                                                          HGmemRangeHandleType handleType,
                                                          unsigned long long flags);
HGresult doca_verbs_wrapper_hgCtxGetCurrent(HGcontext *pctx);

/* Initialization function */
int doca_hggc_wrapper_init(void);

#else

#include <hggc.h>

/* Direct API calls when wrapper is not enabled */
#define doca_verbs_wrapper_hgDeviceGetAttribute hgDeviceGetAttribute
#define doca_verbs_wrapper_hgPointerSetAttribute hgPointerSetAttribute
#define doca_verbs_wrapper_hgMemGetHandleForAddressRange hgMemGetHandleForAddressRange
#define doca_verbs_wrapper_hgCtxGetCurrent hgCtxGetCurrent

/* No initialization needed when wrapper is not enabled */
#define doca_hggc_wrapper_init() 0

#endif

#ifdef __cplusplus
}
#endif

#endif /* DOCA_VERBS_HGGC_WRAPPER_H */
