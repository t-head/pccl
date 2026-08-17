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

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syslog.h>
#include <mutex>

#include "doca_verbs_cuda_wrapper.h"
#include "doca_gpunetio_log.hpp"

/* Function pointer types for HGGC device APIs */
typedef HGresult (*hgDeviceGetAttribute_t)(int *pi, HGdevice_attribute attrib, HGdevice dev);
typedef HGresult (*hgPointerSetAttribute_t)(const void *value, HGpointer_attribute attribute,
                                            HGdeviceptr ptr);
typedef HGresult (*hgMemGetHandleForAddressRange_t)(int *pHandle, HGdeviceptr dptr, size_t size,
                                                    HGmemRangeHandleType handleType,
                                                    unsigned long long flags);
typedef HGresult (*hgCtxGetCurrent_t)(HGcontext *pctx);

/* Global function pointers */
hgDeviceGetAttribute_t p_hgDeviceGetAttribute = nullptr;
hgPointerSetAttribute_t p_hgPointerSetAttribute = nullptr;
hgMemGetHandleForAddressRange_t p_hgMemGetHandleForAddressRange = nullptr;
hgCtxGetCurrent_t p_hgCtxGetCurrent = nullptr;

static void *hggc_handle = nullptr;

/* Helper function to get function pointer from libhggc */
static void *get_hggc_symbol(const char *symbol_name) {
    void *symbol = dlsym(hggc_handle, symbol_name);
    if (!symbol) {
        DOCA_LOG(LOG_ERR, "Failed to get symbol %s: %s\n", symbol_name, dlerror());
        return nullptr;
    }
    return symbol;
}

static void doca_verbs_wrapper_init_once(int *ret) {
    /* Open libhggc.so */
    hggc_handle = dlopen("libhggc.so.1", RTLD_NOW);
    if (!hggc_handle) {
        hggc_handle = dlopen("libhggc.so", RTLD_NOW);
    }
    if (!hggc_handle) {
        DOCA_LOG(LOG_ERR, "Failed to open libhggc: %s\n", dlerror());
        *ret = -1;
        return;
    }

    /* Get function pointers */
    p_hgDeviceGetAttribute = (hgDeviceGetAttribute_t)get_hggc_symbol("hgDeviceGetAttribute");
    p_hgPointerSetAttribute = (hgPointerSetAttribute_t)get_hggc_symbol("hgPointerSetAttribute");
    p_hgMemGetHandleForAddressRange =
        (hgMemGetHandleForAddressRange_t)get_hggc_symbol("hgMemGetHandleForAddressRange");
    p_hgCtxGetCurrent = (hgCtxGetCurrent_t)get_hggc_symbol("hgCtxGetCurrent");

    /* Check if all symbols were found */
    if (!p_hgDeviceGetAttribute || !p_hgPointerSetAttribute || !p_hgMemGetHandleForAddressRange ||
        !p_hgCtxGetCurrent) {
        DOCA_LOG(LOG_ERR, "Failed to get all required HGGC symbols\n");
        dlclose(hggc_handle);
        hggc_handle = nullptr;
        *ret = -1;
        return;
    }

    *ret = 0;
}

static int init_hggc_wrapper(void) {
    static int ret = 0;
    static std::once_flag once;
    std::call_once(once, doca_verbs_wrapper_init_once, &ret);
    return ret;
}

/* Wrapper function implementations */
HGresult doca_verbs_wrapper_hgDeviceGetAttribute(int *pi, HGdevice_attribute attrib, HGdevice dev) {
    if (init_hggc_wrapper() != 0) return HGGC_ERROR_NOT_INITIALIZED;
    return p_hgDeviceGetAttribute(pi, attrib, dev);
}

HGresult doca_verbs_wrapper_hgPointerSetAttribute(const void *value, HGpointer_attribute attribute,
                                                  HGdeviceptr ptr) {
    if (init_hggc_wrapper() != 0) return HGGC_ERROR_NOT_INITIALIZED;
    return p_hgPointerSetAttribute(value, attribute, ptr);
}

HGresult doca_verbs_wrapper_hgMemGetHandleForAddressRange(int *pHandle, HGdeviceptr dptr,
                                                          size_t size,
                                                          HGmemRangeHandleType handleType,
                                                          unsigned long long flags) {
    if (init_hggc_wrapper() != 0) return HGGC_ERROR_NOT_INITIALIZED;
    return p_hgMemGetHandleForAddressRange(pHandle, dptr, size, handleType, flags);
}

HGresult doca_verbs_wrapper_hgCtxGetCurrent(HGcontext *pctx) {
    if (init_hggc_wrapper() != 0) return HGGC_ERROR_NOT_INITIALIZED;
    return p_hgCtxGetCurrent(pctx);
}
