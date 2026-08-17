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

/* This header defines types which are used by the internal implementation
*  of HGTX and callback subscribers.  API clients do not use these types,
*  so they are defined here instead of in hgToolsExt.h to clarify they are
*  not part of the HGTX client API. */

#ifndef HGTX_IMPL_GUARD
#error Never include this file directly -- it is automatically included by hgToolsExt.h.
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

/* ------ Dependency-free types binary-compatible with real types ------- */

/* In order to avoid having the HGTX core API headers depend on non-HGTX
*  headers like hggc.h, HGTX defines binary-compatible types to use for
*  safely making the initialization versions of all HGTX functions without
*  needing to have definitions for the real types. */

typedef int   hgtx_HGdevice;
typedef void* hgtx_HGcontext;
typedef void* hgtx_HGstream;
typedef void* hgtx_HGevent;

typedef void* hgtx_hggcStream_t;
typedef void* hgtx_hggcEvent_t;

typedef void* hgtx_cl_platform_id;
typedef void* hgtx_cl_device_id;
typedef void* hgtx_cl_context;
typedef void* hgtx_cl_command_queue;
typedef void* hgtx_cl_mem;
typedef void* hgtx_cl_program;
typedef void* hgtx_cl_kernel;
typedef void* hgtx_cl_event;
typedef void* hgtx_cl_sampler;

typedef void* hgtx_hgtxSyncUser_t;
typedef void hgtx_hgtxSyncUserAttributes_t;

/* --------- Types for function pointers (with fake API types) ---------- */

typedef void (HGTX_API * hgtxMarkEx_impl_fntype)(const hgtxEventAttributes_t* eventAttrib);
typedef void (HGTX_API * hgtxMarkA_impl_fntype)(const char* message);
typedef void (HGTX_API * hgtxMarkW_impl_fntype)(const wchar_t* message);
typedef hgtxRangeId_t (HGTX_API * hgtxRangeStartEx_impl_fntype)(const hgtxEventAttributes_t* eventAttrib);
typedef hgtxRangeId_t (HGTX_API * hgtxRangeStartA_impl_fntype)(const char* message);
typedef hgtxRangeId_t (HGTX_API * hgtxRangeStartW_impl_fntype)(const wchar_t* message);
typedef void (HGTX_API * hgtxRangeEnd_impl_fntype)(hgtxRangeId_t id);
typedef int (HGTX_API * hgtxRangePushEx_impl_fntype)(const hgtxEventAttributes_t* eventAttrib);
typedef int (HGTX_API * hgtxRangePushA_impl_fntype)(const char* message);
typedef int (HGTX_API * hgtxRangePushW_impl_fntype)(const wchar_t* message);
typedef int (HGTX_API * hgtxRangePop_impl_fntype)(void);
typedef void (HGTX_API * hgtxNameCategoryA_impl_fntype)(uint32_t category, const char* name);
typedef void (HGTX_API * hgtxNameCategoryW_impl_fntype)(uint32_t category, const wchar_t* name);
typedef void (HGTX_API * hgtxNameOsThreadA_impl_fntype)(uint32_t threadId, const char* name);
typedef void (HGTX_API * hgtxNameOsThreadW_impl_fntype)(uint32_t threadId, const wchar_t* name);

/* Real impl types are defined in hgtxImplHggc_v3.h, where HGGC headers are included */
typedef void (HGTX_API * hgtxNameHgDeviceA_fakeimpl_fntype)(hgtx_HGdevice device, const char* name);
typedef void (HGTX_API * hgtxNameHgDeviceW_fakeimpl_fntype)(hgtx_HGdevice device, const wchar_t* name);
typedef void (HGTX_API * hgtxNameHgContextA_fakeimpl_fntype)(hgtx_HGcontext context, const char* name);
typedef void (HGTX_API * hgtxNameHgContextW_fakeimpl_fntype)(hgtx_HGcontext context, const wchar_t* name);
typedef void (HGTX_API * hgtxNameHgStreamA_fakeimpl_fntype)(hgtx_HGstream stream, const char* name);
typedef void (HGTX_API * hgtxNameHgStreamW_fakeimpl_fntype)(hgtx_HGstream stream, const wchar_t* name);
typedef void (HGTX_API * hgtxNameHgEventA_fakeimpl_fntype)(hgtx_HGevent event, const char* name);
typedef void (HGTX_API * hgtxNameHgEventW_fakeimpl_fntype)(hgtx_HGevent event, const wchar_t* name);

/* Real impl types are defined in hgtxImplOpenCL_v3.h, where OPENCL headers are included */
typedef void (HGTX_API * hgtxNameClDeviceA_fakeimpl_fntype)(hgtx_cl_device_id device, const char* name);
typedef void (HGTX_API * hgtxNameClDeviceW_fakeimpl_fntype)(hgtx_cl_device_id device, const wchar_t* name);
typedef void (HGTX_API * hgtxNameClContextA_fakeimpl_fntype)(hgtx_cl_context context, const char* name);
typedef void (HGTX_API * hgtxNameClContextW_fakeimpl_fntype)(hgtx_cl_context context, const wchar_t* name);
typedef void (HGTX_API * hgtxNameClCommandQueueA_fakeimpl_fntype)(hgtx_cl_command_queue command_queue, const char* name);
typedef void (HGTX_API * hgtxNameClCommandQueueW_fakeimpl_fntype)(hgtx_cl_command_queue command_queue, const wchar_t* name);
typedef void (HGTX_API * hgtxNameClMemObjectA_fakeimpl_fntype)(hgtx_cl_mem memobj, const char* name);
typedef void (HGTX_API * hgtxNameClMemObjectW_fakeimpl_fntype)(hgtx_cl_mem memobj, const wchar_t* name);
typedef void (HGTX_API * hgtxNameClSamplerA_fakeimpl_fntype)(hgtx_cl_sampler sampler, const char* name);
typedef void (HGTX_API * hgtxNameClSamplerW_fakeimpl_fntype)(hgtx_cl_sampler sampler, const wchar_t* name);
typedef void (HGTX_API * hgtxNameClProgramA_fakeimpl_fntype)(hgtx_cl_program program, const char* name);
typedef void (HGTX_API * hgtxNameClProgramW_fakeimpl_fntype)(hgtx_cl_program program, const wchar_t* name);
typedef void (HGTX_API * hgtxNameClEventA_fakeimpl_fntype)(hgtx_cl_event evnt, const char* name);
typedef void (HGTX_API * hgtxNameClEventW_fakeimpl_fntype)(hgtx_cl_event evnt, const wchar_t* name);

/* Real impl types are defined in hgtxImplHggcRt_v3.h, where HGGCRT headers are included */
typedef void (HGTX_API * hgtxNameHggcDeviceA_fakeimpl_fntype)(int device, const char* name);
typedef void (HGTX_API * hgtxNameHggcDeviceW_fakeimpl_fntype)(int device, const wchar_t* name);
typedef void (HGTX_API * hgtxNameHggcStreamA_fakeimpl_fntype)(hgtx_hggcStream_t stream, const char* name);
typedef void (HGTX_API * hgtxNameHggcStreamW_fakeimpl_fntype)(hgtx_hggcStream_t stream, const wchar_t* name);
typedef void (HGTX_API * hgtxNameHggcEventA_fakeimpl_fntype)(hgtx_hggcEvent_t event, const char* name);
typedef void (HGTX_API * hgtxNameHggcEventW_fakeimpl_fntype)(hgtx_hggcEvent_t event, const wchar_t* name);

typedef void (HGTX_API * hgtxDomainMarkEx_impl_fntype)(hgtxDomainHandle_t domain, const hgtxEventAttributes_t* eventAttrib);
typedef hgtxRangeId_t (HGTX_API * hgtxDomainRangeStartEx_impl_fntype)(hgtxDomainHandle_t domain, const hgtxEventAttributes_t* eventAttrib);
typedef void (HGTX_API * hgtxDomainRangeEnd_impl_fntype)(hgtxDomainHandle_t domain, hgtxRangeId_t id);
typedef int (HGTX_API * hgtxDomainRangePushEx_impl_fntype)(hgtxDomainHandle_t domain, const hgtxEventAttributes_t* eventAttrib);
typedef int (HGTX_API * hgtxDomainRangePop_impl_fntype)(hgtxDomainHandle_t domain);
typedef hgtxResourceHandle_t (HGTX_API * hgtxDomainResourceCreate_impl_fntype)(hgtxDomainHandle_t domain, hgtxResourceAttributes_t* attribs);
typedef void (HGTX_API * hgtxDomainResourceDestroy_impl_fntype)(hgtxResourceHandle_t resource);
typedef void (HGTX_API * hgtxDomainNameCategoryA_impl_fntype)(hgtxDomainHandle_t domain, uint32_t category, const char* name);
typedef void (HGTX_API * hgtxDomainNameCategoryW_impl_fntype)(hgtxDomainHandle_t domain, uint32_t category, const wchar_t* name);
typedef hgtxStringHandle_t (HGTX_API * hgtxDomainRegisterStringA_impl_fntype)(hgtxDomainHandle_t domain, const char* string);
typedef hgtxStringHandle_t (HGTX_API * hgtxDomainRegisterStringW_impl_fntype)(hgtxDomainHandle_t domain, const wchar_t* string);
typedef hgtxDomainHandle_t (HGTX_API * hgtxDomainCreateA_impl_fntype)(const char* message);
typedef hgtxDomainHandle_t (HGTX_API * hgtxDomainCreateW_impl_fntype)(const wchar_t* message);
typedef void (HGTX_API * hgtxDomainDestroy_impl_fntype)(hgtxDomainHandle_t domain);
typedef void (HGTX_API * hgtxInitialize_impl_fntype)(const void* reserved);

typedef hgtx_hgtxSyncUser_t (HGTX_API * hgtxDomainSyncUserCreate_fakeimpl_fntype)(hgtxDomainHandle_t domain, const hgtx_hgtxSyncUserAttributes_t* attribs);
typedef void (HGTX_API * hgtxDomainSyncUserDestroy_fakeimpl_fntype)(hgtx_hgtxSyncUser_t handle);
typedef void (HGTX_API * hgtxDomainSyncUserAcquireStart_fakeimpl_fntype)(hgtx_hgtxSyncUser_t handle);
typedef void (HGTX_API * hgtxDomainSyncUserAcquireFailed_fakeimpl_fntype)(hgtx_hgtxSyncUser_t handle);
typedef void (HGTX_API * hgtxDomainSyncUserAcquireSuccess_fakeimpl_fntype)(hgtx_hgtxSyncUser_t handle);
typedef void (HGTX_API * hgtxDomainSyncUserReleasing_fakeimpl_fntype)(hgtx_hgtxSyncUser_t handle);

/* ---------------- Types for callback subscription --------------------- */

typedef const void *(HGTX_API * HgtxGetExportTableFunc_t)(uint32_t exportTableId);
typedef int (HGTX_API * HgtxInitializeInjectionHgtxFunc_t)(HgtxGetExportTableFunc_t exportTable);

typedef enum HgtxCallbackModule
{
    HGTX_CB_MODULE_INVALID                 = 0,
    HGTX_CB_MODULE_CORE                    = 1,
    HGTX_CB_MODULE_HGGC                    = 2,
    HGTX_CB_MODULE_OPENCL                  = 3,
    HGTX_CB_MODULE_HGGCRT                  = 4,
    HGTX_CB_MODULE_CORE2                   = 5,
    HGTX_CB_MODULE_SYNC                    = 6,
    /* --- New constants must only be added directly above this line --- */
    HGTX_CB_MODULE_SIZE,
    HGTX_CB_MODULE_FORCE_INT               = 0x7fffffff
} HgtxCallbackModule;

typedef enum HgtxCallbackIdCore
{
    HGTX_CBID_CORE_INVALID                 =  0,
    HGTX_CBID_CORE_MarkEx                  =  1,
    HGTX_CBID_CORE_MarkA                   =  2,
    HGTX_CBID_CORE_MarkW                   =  3,
    HGTX_CBID_CORE_RangeStartEx            =  4,
    HGTX_CBID_CORE_RangeStartA             =  5,
    HGTX_CBID_CORE_RangeStartW             =  6,
    HGTX_CBID_CORE_RangeEnd                =  7,
    HGTX_CBID_CORE_RangePushEx             =  8,
    HGTX_CBID_CORE_RangePushA              =  9,
    HGTX_CBID_CORE_RangePushW              = 10,
    HGTX_CBID_CORE_RangePop                = 11,
    HGTX_CBID_CORE_NameCategoryA           = 12,
    HGTX_CBID_CORE_NameCategoryW           = 13,
    HGTX_CBID_CORE_NameOsThreadA           = 14,
    HGTX_CBID_CORE_NameOsThreadW           = 15,
    /* --- New constants must only be added directly above this line --- */
    HGTX_CBID_CORE_SIZE,
    HGTX_CBID_CORE_FORCE_INT = 0x7fffffff
} HgtxCallbackIdCore;

typedef enum HgtxCallbackIdCore2
{
    HGTX_CBID_CORE2_INVALID                 = 0,
    HGTX_CBID_CORE2_DomainMarkEx            = 1,
    HGTX_CBID_CORE2_DomainRangeStartEx      = 2,
    HGTX_CBID_CORE2_DomainRangeEnd          = 3,
    HGTX_CBID_CORE2_DomainRangePushEx       = 4,
    HGTX_CBID_CORE2_DomainRangePop          = 5,
    HGTX_CBID_CORE2_DomainResourceCreate    = 6,
    HGTX_CBID_CORE2_DomainResourceDestroy   = 7,
    HGTX_CBID_CORE2_DomainNameCategoryA     = 8,
    HGTX_CBID_CORE2_DomainNameCategoryW     = 9,
    HGTX_CBID_CORE2_DomainRegisterStringA   = 10,
    HGTX_CBID_CORE2_DomainRegisterStringW   = 11,
    HGTX_CBID_CORE2_DomainCreateA           = 12,
    HGTX_CBID_CORE2_DomainCreateW           = 13,
    HGTX_CBID_CORE2_DomainDestroy           = 14,
    HGTX_CBID_CORE2_Initialize              = 15,
    /* --- New constants must only be added directly above this line --- */
    HGTX_CBID_CORE2_SIZE,
    HGTX_CBID_CORE2_FORCE_INT               = 0x7fffffff
} HgtxCallbackIdCore2;

typedef enum HgtxCallbackIdHggc
{
    HGTX_CBID_HGGC_INVALID                 =  0,
    HGTX_CBID_HGGC_NameHgDeviceA           =  1,
    HGTX_CBID_HGGC_NameHgDeviceW           =  2,
    HGTX_CBID_HGGC_NameHgContextA          =  3,
    HGTX_CBID_HGGC_NameHgContextW          =  4,
    HGTX_CBID_HGGC_NameHgStreamA           =  5,
    HGTX_CBID_HGGC_NameHgStreamW           =  6,
    HGTX_CBID_HGGC_NameHgEventA            =  7,
    HGTX_CBID_HGGC_NameHgEventW            =  8,
    /* --- New constants must only be added directly above this line --- */
    HGTX_CBID_HGGC_SIZE,
    HGTX_CBID_HGGC_FORCE_INT               = 0x7fffffff
} HgtxCallbackIdHggc;

typedef enum HgtxCallbackIdHggcRt
{
    HGTX_CBID_HGGCRT_INVALID               =  0,
    HGTX_CBID_HGGCRT_NameHggcDeviceA       =  1,
    HGTX_CBID_HGGCRT_NameHggcDeviceW       =  2,
    HGTX_CBID_HGGCRT_NameHggcStreamA       =  3,
    HGTX_CBID_HGGCRT_NameHggcStreamW       =  4,
    HGTX_CBID_HGGCRT_NameHggcEventA        =  5,
    HGTX_CBID_HGGCRT_NameHggcEventW        =  6,
    /* --- New constants must only be added directly above this line --- */
    HGTX_CBID_HGGCRT_SIZE,
    HGTX_CBID_HGGCRT_FORCE_INT             = 0x7fffffff
} HgtxCallbackIdHggcRt;

typedef enum HgtxCallbackIdOpenCL
{
    HGTX_CBID_OPENCL_INVALID               =  0,
    HGTX_CBID_OPENCL_NameClDeviceA         =  1,
    HGTX_CBID_OPENCL_NameClDeviceW         =  2,
    HGTX_CBID_OPENCL_NameClContextA        =  3,
    HGTX_CBID_OPENCL_NameClContextW        =  4,
    HGTX_CBID_OPENCL_NameClCommandQueueA   =  5,
    HGTX_CBID_OPENCL_NameClCommandQueueW   =  6,
    HGTX_CBID_OPENCL_NameClMemObjectA      =  7,
    HGTX_CBID_OPENCL_NameClMemObjectW      =  8,
    HGTX_CBID_OPENCL_NameClSamplerA        =  9,
    HGTX_CBID_OPENCL_NameClSamplerW        = 10,
    HGTX_CBID_OPENCL_NameClProgramA        = 11,
    HGTX_CBID_OPENCL_NameClProgramW        = 12,
    HGTX_CBID_OPENCL_NameClEventA          = 13,
    HGTX_CBID_OPENCL_NameClEventW          = 14,
    /* --- New constants must only be added directly above this line --- */
    HGTX_CBID_OPENCL_SIZE,
    HGTX_CBID_OPENCL_FORCE_INT             = 0x7fffffff
} HgtxCallbackIdOpenCL;

typedef enum HgtxCallbackIdSync
{
    HGTX_CBID_SYNC_INVALID                      = 0,
    HGTX_CBID_SYNC_DomainSyncUserCreate         = 1,
    HGTX_CBID_SYNC_DomainSyncUserDestroy        = 2,
    HGTX_CBID_SYNC_DomainSyncUserAcquireStart   = 3,
    HGTX_CBID_SYNC_DomainSyncUserAcquireFailed  = 4,
    HGTX_CBID_SYNC_DomainSyncUserAcquireSuccess = 5,
    HGTX_CBID_SYNC_DomainSyncUserReleasing      = 6,
    /* --- New constants must only be added directly above this line --- */
    HGTX_CBID_SYNC_SIZE,
    HGTX_CBID_SYNC_FORCE_INT                    = 0x7fffffff
} HgtxCallbackIdSync;

/* IDs for HGTX Export Tables */
typedef enum HgtxExportTableID
{
    HGTX_ETID_INVALID                      = 0,
    HGTX_ETID_CALLBACKS                    = 1,
    HGTX_ETID_RESERVED0                    = 2,
    HGTX_ETID_VERSIONINFO                  = 3,
    /* --- New constants must only be added directly above this line --- */
    HGTX_ETID_SIZE,
    HGTX_ETID_FORCE_INT                    = 0x7fffffff
} HgtxExportTableID;

typedef void (* HgtxFunctionPointer)(void); /* generic uncallable function pointer, must be cast to appropriate function type */
typedef HgtxFunctionPointer** HgtxFunctionTable; /* double pointer because array(1) of pointers(2) to function pointers */

typedef struct HgtxExportTableCallbacks
{
    size_t struct_size;

    /* returns an array of pointer to function pointers*/
    int (HGTX_API *GetModuleFunctionTable)(
        HgtxCallbackModule module,
        HgtxFunctionTable* out_table,
        unsigned int* out_size);
} HgtxExportTableCallbacks;

typedef struct HgtxExportTableVersionInfo
{
    /* sizeof(HgtxExportTableVersionInfo) */
    size_t struct_size;

    /* The API version comes from the HGTX library linked to the app.  The
    * injection library is can use this info to make some assumptions */
    uint32_t version;

    /* Reserved for alignment, do not use */
    uint32_t reserved0;

    /* This must be set by tools when attaching to provide applications
    *  the ability to, in emergency situations, detect problematic tools
    *  versions and modify the HGTX source to prevent attaching anything
    *  that causes trouble in the app.  Currently, this value is ignored. */
    void (HGTX_API *SetInjectionHgtxVersion)(
        uint32_t version);
} HgtxExportTableVersionInfo;







