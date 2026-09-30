/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions.
 * See https://nvidia.github.io/NVTX/LICENSE.txt for license information.
 */

/*
 * Modified by T-Head Semiconductor.
 * Changes: Replaced all nvtx* symbols with hgtx* (function names, types,
 * macros, headers) while preserving full API compatibility.
 * Original source: https://github.com/NVIDIA/NVTX
 */

#ifndef HGTX_IMPL_GUARD
#error Never include this file directly -- it is automatically included by hgToolsExt.h (except when HGTX_NO_IMPL is defined).
#endif

#if defined(HGTX_AS_SYSTEM_HEADER)
#if defined(__clang__)
#pragma clang system_header
#elif defined(__GNUC__) || defined(__HGCOMPILER)
#pragma GCC system_header
#elif defined(_MSC_VER)
#pragma system_header
#endif
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>

/* ---- Include required platform headers ---- */

#if defined(_WIN32)

#include <windows.h>

#else
#include <unistd.h>

#if defined(__ANDROID__)
#include <android/api-level.h>
#endif

#if defined(__linux__) || defined(__CYGWIN__)
#include <sched.h>
#endif

#include <sys/types.h>
#include <limits.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>

#endif

/* ---- Define macros used in this file ---- */

#define HGTX_INIT_STATE_FRESH 0
#define HGTX_INIT_STATE_STARTED 1
#define HGTX_INIT_STATE_COMPLETE 2

#ifdef HGTX_DEBUG_PRINT
#ifdef __ANDROID__
#include <android/log.h>
#define HGTX_ERR(...) __android_log_print(ANDROID_LOG_ERROR, "HGTOOLSEXT", __VA_ARGS__);
#define HGTX_INFO(...) __android_log_print(ANDROID_LOG_INFO, "HGTOOLSEXT", __VA_ARGS__);
#else
#include <stdio.h>
#define HGTX_ERR(...) fprintf(stderr, "HGTX_ERROR: " __VA_ARGS__)
#define HGTX_INFO(...) fprintf(stderr, "HGTX_INFO: " __VA_ARGS__)
#endif
#else /* !defined(HGTX_DEBUG_PRINT) */
#define HGTX_ERR(...)
#define HGTX_INFO(...)
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

/* ---- Forward declare all functions referenced in globals ---- */

HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)(void);
HGTX_LINKONCE_FWDDECL_FUNCTION int HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxEtiGetModuleFunctionTable)(
    HgtxCallbackModule module,
    HgtxFunctionTable* out_table,
    unsigned int* out_size);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxEtiSetInjectionHgtxVersion)(
    uint32_t version);
HGTX_LINKONCE_FWDDECL_FUNCTION const void* HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxGetExportTable)(
    uint32_t exportTableId);

#include "hgtxInitDecls.h"

/* ---- Define all globals ---- */

typedef struct hgtxGlobals_t
{
    volatile unsigned int initState;
    HgtxExportTableCallbacks etblCallbacks;
    HgtxExportTableVersionInfo etblVersionInfo;

    /* Implementation function pointers */
    hgtxMarkEx_impl_fntype hgtxMarkEx_impl_fnptr;
    hgtxMarkA_impl_fntype hgtxMarkA_impl_fnptr;
    hgtxMarkW_impl_fntype hgtxMarkW_impl_fnptr;
    hgtxRangeStartEx_impl_fntype hgtxRangeStartEx_impl_fnptr;
    hgtxRangeStartA_impl_fntype hgtxRangeStartA_impl_fnptr;
    hgtxRangeStartW_impl_fntype hgtxRangeStartW_impl_fnptr;
    hgtxRangeEnd_impl_fntype hgtxRangeEnd_impl_fnptr;
    hgtxRangePushEx_impl_fntype hgtxRangePushEx_impl_fnptr;
    hgtxRangePushA_impl_fntype hgtxRangePushA_impl_fnptr;
    hgtxRangePushW_impl_fntype hgtxRangePushW_impl_fnptr;
    hgtxRangePop_impl_fntype hgtxRangePop_impl_fnptr;
    hgtxNameCategoryA_impl_fntype hgtxNameCategoryA_impl_fnptr;
    hgtxNameCategoryW_impl_fntype hgtxNameCategoryW_impl_fnptr;
    hgtxNameOsThreadA_impl_fntype hgtxNameOsThreadA_impl_fnptr;
    hgtxNameOsThreadW_impl_fntype hgtxNameOsThreadW_impl_fnptr;

    hgtxNameHgDeviceA_fakeimpl_fntype hgtxNameHgDeviceA_impl_fnptr;
    hgtxNameHgDeviceW_fakeimpl_fntype hgtxNameHgDeviceW_impl_fnptr;
    hgtxNameHgContextA_fakeimpl_fntype hgtxNameHgContextA_impl_fnptr;
    hgtxNameHgContextW_fakeimpl_fntype hgtxNameHgContextW_impl_fnptr;
    hgtxNameHgStreamA_fakeimpl_fntype hgtxNameHgStreamA_impl_fnptr;
    hgtxNameHgStreamW_fakeimpl_fntype hgtxNameHgStreamW_impl_fnptr;
    hgtxNameHgEventA_fakeimpl_fntype hgtxNameHgEventA_impl_fnptr;
    hgtxNameHgEventW_fakeimpl_fntype hgtxNameHgEventW_impl_fnptr;

    hgtxNameClDeviceA_fakeimpl_fntype hgtxNameClDeviceA_impl_fnptr;
    hgtxNameClDeviceW_fakeimpl_fntype hgtxNameClDeviceW_impl_fnptr;
    hgtxNameClContextA_fakeimpl_fntype hgtxNameClContextA_impl_fnptr;
    hgtxNameClContextW_fakeimpl_fntype hgtxNameClContextW_impl_fnptr;
    hgtxNameClCommandQueueA_fakeimpl_fntype hgtxNameClCommandQueueA_impl_fnptr;
    hgtxNameClCommandQueueW_fakeimpl_fntype hgtxNameClCommandQueueW_impl_fnptr;
    hgtxNameClMemObjectA_fakeimpl_fntype hgtxNameClMemObjectA_impl_fnptr;
    hgtxNameClMemObjectW_fakeimpl_fntype hgtxNameClMemObjectW_impl_fnptr;
    hgtxNameClSamplerA_fakeimpl_fntype hgtxNameClSamplerA_impl_fnptr;
    hgtxNameClSamplerW_fakeimpl_fntype hgtxNameClSamplerW_impl_fnptr;
    hgtxNameClProgramA_fakeimpl_fntype hgtxNameClProgramA_impl_fnptr;
    hgtxNameClProgramW_fakeimpl_fntype hgtxNameClProgramW_impl_fnptr;
    hgtxNameClEventA_fakeimpl_fntype hgtxNameClEventA_impl_fnptr;
    hgtxNameClEventW_fakeimpl_fntype hgtxNameClEventW_impl_fnptr;

    hgtxNameHggcDeviceA_fakeimpl_fntype hgtxNameHggcDeviceA_impl_fnptr;
    hgtxNameHggcDeviceW_fakeimpl_fntype hgtxNameHggcDeviceW_impl_fnptr;
    hgtxNameHggcStreamA_fakeimpl_fntype hgtxNameHggcStreamA_impl_fnptr;
    hgtxNameHggcStreamW_fakeimpl_fntype hgtxNameHggcStreamW_impl_fnptr;
    hgtxNameHggcEventA_fakeimpl_fntype hgtxNameHggcEventA_impl_fnptr;
    hgtxNameHggcEventW_fakeimpl_fntype hgtxNameHggcEventW_impl_fnptr;

    hgtxDomainMarkEx_impl_fntype hgtxDomainMarkEx_impl_fnptr;
    hgtxDomainRangeStartEx_impl_fntype hgtxDomainRangeStartEx_impl_fnptr;
    hgtxDomainRangeEnd_impl_fntype hgtxDomainRangeEnd_impl_fnptr;
    hgtxDomainRangePushEx_impl_fntype hgtxDomainRangePushEx_impl_fnptr;
    hgtxDomainRangePop_impl_fntype hgtxDomainRangePop_impl_fnptr;
    hgtxDomainResourceCreate_impl_fntype hgtxDomainResourceCreate_impl_fnptr;
    hgtxDomainResourceDestroy_impl_fntype hgtxDomainResourceDestroy_impl_fnptr;
    hgtxDomainNameCategoryA_impl_fntype hgtxDomainNameCategoryA_impl_fnptr;
    hgtxDomainNameCategoryW_impl_fntype hgtxDomainNameCategoryW_impl_fnptr;
    hgtxDomainRegisterStringA_impl_fntype hgtxDomainRegisterStringA_impl_fnptr;
    hgtxDomainRegisterStringW_impl_fntype hgtxDomainRegisterStringW_impl_fnptr;
    hgtxDomainCreateA_impl_fntype hgtxDomainCreateA_impl_fnptr;
    hgtxDomainCreateW_impl_fntype hgtxDomainCreateW_impl_fnptr;
    hgtxDomainDestroy_impl_fntype hgtxDomainDestroy_impl_fnptr;
    hgtxInitialize_impl_fntype hgtxInitialize_impl_fnptr;

    hgtxDomainSyncUserCreate_fakeimpl_fntype hgtxDomainSyncUserCreate_impl_fnptr;
    hgtxDomainSyncUserDestroy_fakeimpl_fntype hgtxDomainSyncUserDestroy_impl_fnptr;
    hgtxDomainSyncUserAcquireStart_fakeimpl_fntype hgtxDomainSyncUserAcquireStart_impl_fnptr;
    hgtxDomainSyncUserAcquireFailed_fakeimpl_fntype hgtxDomainSyncUserAcquireFailed_impl_fnptr;
    hgtxDomainSyncUserAcquireSuccess_fakeimpl_fntype hgtxDomainSyncUserAcquireSuccess_impl_fnptr;
    hgtxDomainSyncUserReleasing_fakeimpl_fntype hgtxDomainSyncUserReleasing_impl_fnptr;

    /* Tables of function pointers -- Extra null added to the end to ensure
    *  a crash instead of silent corruption if a tool reads off the end. */
    HgtxFunctionPointer* functionTable_CORE  [HGTX_CBID_CORE_SIZE   + 1];
    HgtxFunctionPointer* functionTable_HGGC  [HGTX_CBID_HGGC_SIZE   + 1];
    HgtxFunctionPointer* functionTable_OPENCL[HGTX_CBID_OPENCL_SIZE + 1];
    HgtxFunctionPointer* functionTable_HGGCRT[HGTX_CBID_HGGCRT_SIZE + 1];
    HgtxFunctionPointer* functionTable_CORE2 [HGTX_CBID_CORE2_SIZE  + 1];
    HgtxFunctionPointer* functionTable_SYNC  [HGTX_CBID_SYNC_SIZE   + 1];
} hgtxGlobals_t;

#define HGTX_GLOBAL_TABLE_ENTRY(name) ( HGTX_REINTERPRET_CAST(HgtxFunctionPointer*, &HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).name ## _impl_fnptr ) )

HGTX_LINKONCE_DEFINE_GLOBAL hgtxGlobals_t HGTX_VERSIONED_IDENTIFIER(hgtxGlobals) =
{
    HGTX_INIT_STATE_FRESH,

    {
        sizeof(HgtxExportTableCallbacks),
        HGTX_VERSIONED_IDENTIFIER(hgtxEtiGetModuleFunctionTable)
    },
    {
        sizeof(HgtxExportTableVersionInfo),
        HGTX_VERSION,
        0,
        HGTX_VERSIONED_IDENTIFIER(hgtxEtiSetInjectionHgtxVersion)
    },

    /* Implementation function pointers */
    HGTX_VERSIONED_IDENTIFIER(hgtxMarkEx_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxMarkA_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxMarkW_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxRangeStartEx_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxRangeStartA_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxRangeStartW_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxRangeEnd_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxRangePushEx_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxRangePushA_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxRangePushW_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxRangePop_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameCategoryA_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameCategoryW_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameOsThreadA_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameOsThreadW_impl_init),

    HGTX_VERSIONED_IDENTIFIER(hgtxNameHgDeviceA_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameHgDeviceW_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameHgContextA_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameHgContextW_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameHgStreamA_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameHgStreamW_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameHgEventA_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameHgEventW_impl_init),

    HGTX_VERSIONED_IDENTIFIER(hgtxNameClDeviceA_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameClDeviceW_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameClContextA_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameClContextW_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameClCommandQueueA_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameClCommandQueueW_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameClMemObjectA_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameClMemObjectW_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameClSamplerA_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameClSamplerW_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameClProgramA_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameClProgramW_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameClEventA_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameClEventW_impl_init),

    HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcDeviceA_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcDeviceW_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcStreamA_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcStreamW_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcEventA_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcEventW_impl_init),

    HGTX_VERSIONED_IDENTIFIER(hgtxDomainMarkEx_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxDomainRangeStartEx_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxDomainRangeEnd_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxDomainRangePushEx_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxDomainRangePop_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxDomainResourceCreate_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxDomainResourceDestroy_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxDomainNameCategoryA_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxDomainNameCategoryW_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxDomainRegisterStringA_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxDomainRegisterStringW_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxDomainCreateA_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxDomainCreateW_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxDomainDestroy_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxInitialize_impl_init),

    HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserCreate_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserDestroy_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserAcquireStart_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserAcquireFailed_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserAcquireSuccess_impl_init),
    HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserReleasing_impl_init),

    /* Tables of function pointers */
    {
        HGTX_NULLPTR,
        HGTX_GLOBAL_TABLE_ENTRY(hgtxMarkEx),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxMarkA),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxMarkW),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxRangeStartEx),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxRangeStartA),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxRangeStartW),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxRangeEnd),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxRangePushEx),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxRangePushA),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxRangePushW),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxRangePop),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameCategoryA),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameCategoryW),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameOsThreadA),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameOsThreadW),
        HGTX_NULLPTR
    },
    {
        HGTX_NULLPTR,
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameHgDeviceA),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameHgDeviceW),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameHgContextA),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameHgContextW),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameHgStreamA),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameHgStreamW),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameHgEventA),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameHgEventW),
        HGTX_NULLPTR
    },
    {
        HGTX_NULLPTR,
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameClDeviceA),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameClDeviceW),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameClContextA),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameClContextW),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameClCommandQueueA),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameClCommandQueueW),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameClMemObjectA),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameClMemObjectW),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameClSamplerA),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameClSamplerW),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameClProgramA),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameClProgramW),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameClEventA),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameClEventW),
        HGTX_NULLPTR
    },
    {
        HGTX_NULLPTR,
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameHggcDeviceA),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameHggcDeviceW),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameHggcStreamA),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameHggcStreamW),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameHggcEventA),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxNameHggcEventW),
        HGTX_NULLPTR
    },
    {
        HGTX_NULLPTR,
        HGTX_GLOBAL_TABLE_ENTRY(hgtxDomainMarkEx),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxDomainRangeStartEx),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxDomainRangeEnd),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxDomainRangePushEx),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxDomainRangePop),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxDomainResourceCreate),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxDomainResourceDestroy),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxDomainNameCategoryA),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxDomainNameCategoryW),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxDomainRegisterStringA),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxDomainRegisterStringW),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxDomainCreateA),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxDomainCreateW),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxDomainDestroy),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxInitialize),
        HGTX_NULLPTR
    },
    {
        HGTX_NULLPTR,
        HGTX_GLOBAL_TABLE_ENTRY(hgtxDomainSyncUserCreate),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxDomainSyncUserDestroy),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxDomainSyncUserAcquireStart),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxDomainSyncUserAcquireFailed),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxDomainSyncUserAcquireSuccess),
        HGTX_GLOBAL_TABLE_ENTRY(hgtxDomainSyncUserReleasing),
        HGTX_NULLPTR
    }
};

#undef HGTX_GLOBAL_TABLE_ENTRY

/* ---- Define static inline implementations of core API functions ---- */

#include "hgtxImplCore.h"

/* ---- Define implementations of export table functions ---- */

HGTX_LINKONCE_DEFINE_FUNCTION int HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxEtiGetModuleFunctionTable)(
    HgtxCallbackModule module,
    HgtxFunctionTable* out_table,
    unsigned int* out_size)
{
    unsigned int bytes = 0;
    HgtxFunctionTable table = HGTX_NULLPTR;

    switch (module)
    {
    case HGTX_CB_MODULE_CORE:
        table = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).functionTable_CORE;
        bytes = HGTX_STATIC_CAST(unsigned int, sizeof(HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).functionTable_CORE));
        break;
    case HGTX_CB_MODULE_HGGC:
        table = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).functionTable_HGGC;
        bytes = HGTX_STATIC_CAST(unsigned int, sizeof(HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).functionTable_HGGC));
        break;
    case HGTX_CB_MODULE_OPENCL:
        table = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).functionTable_OPENCL;
        bytes = HGTX_STATIC_CAST(unsigned int, sizeof(HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).functionTable_OPENCL));
        break;
    case HGTX_CB_MODULE_HGGCRT:
        table = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).functionTable_HGGCRT;
        bytes = HGTX_STATIC_CAST(unsigned int, sizeof(HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).functionTable_HGGCRT));
        break;
    case HGTX_CB_MODULE_CORE2:
        table = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).functionTable_CORE2;
        bytes = HGTX_STATIC_CAST(unsigned int, sizeof(HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).functionTable_CORE2));
        break;
    case HGTX_CB_MODULE_SYNC:
        table = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).functionTable_SYNC;
        bytes = HGTX_STATIC_CAST(unsigned int, sizeof(HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).functionTable_SYNC));
        break;
    case HGTX_CB_MODULE_INVALID:
    case HGTX_CB_MODULE_SIZE:
    case HGTX_CB_MODULE_FORCE_INT:
    default: return 0;
    }

    if (out_size)
        *out_size = (bytes / HGTX_STATIC_CAST(unsigned int, sizeof(HgtxFunctionPointer*))) - 1;

    if (out_table)
        *out_table = table;

    return 1;
}

HGTX_LINKONCE_DEFINE_FUNCTION const void* HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxGetExportTable)(uint32_t exportTableId)
{
    switch (exportTableId)
    {
    case HGTX_ETID_CALLBACKS:       return &HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).etblCallbacks;
    case HGTX_ETID_VERSIONINFO:     return &HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).etblVersionInfo;
    default:                        return HGTX_NULLPTR;
    }
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxEtiSetInjectionHgtxVersion)(uint32_t version)
{
    /* Reserved for custom implementations to resolve problems with tools */
    (void)version;
}

/* ---- Define implementations of init versions of all API functions ---- */

#include "hgtxInitDefs.h"

/* ---- Define implementations of initialization functions ---- */

#include "hgtxInit.h"

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */
