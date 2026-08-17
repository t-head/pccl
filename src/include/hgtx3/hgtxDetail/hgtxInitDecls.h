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

HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxMarkEx_impl_init)(const hgtxEventAttributes_t* eventAttrib);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxMarkA_impl_init)(const char* message);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxMarkW_impl_init)(const wchar_t* message);
HGTX_LINKONCE_FWDDECL_FUNCTION hgtxRangeId_t HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxRangeStartEx_impl_init)(const hgtxEventAttributes_t* eventAttrib);
HGTX_LINKONCE_FWDDECL_FUNCTION hgtxRangeId_t HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxRangeStartA_impl_init)(const char* message);
HGTX_LINKONCE_FWDDECL_FUNCTION hgtxRangeId_t HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxRangeStartW_impl_init)(const wchar_t* message);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxRangeEnd_impl_init)(hgtxRangeId_t id);
HGTX_LINKONCE_FWDDECL_FUNCTION int HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxRangePushEx_impl_init)(const hgtxEventAttributes_t* eventAttrib);
HGTX_LINKONCE_FWDDECL_FUNCTION int HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxRangePushA_impl_init)(const char* message);
HGTX_LINKONCE_FWDDECL_FUNCTION int HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxRangePushW_impl_init)(const wchar_t* message);
HGTX_LINKONCE_FWDDECL_FUNCTION int HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxRangePop_impl_init)(void);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameCategoryA_impl_init)(uint32_t category, const char* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameCategoryW_impl_init)(uint32_t category, const wchar_t* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameOsThreadA_impl_init)(uint32_t threadId, const char* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameOsThreadW_impl_init)(uint32_t threadId, const wchar_t* name);

HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHgDeviceA_impl_init)(hgtx_HGdevice device, const char* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHgDeviceW_impl_init)(hgtx_HGdevice device, const wchar_t* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHgContextA_impl_init)(hgtx_HGcontext context, const char* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHgContextW_impl_init)(hgtx_HGcontext context, const wchar_t* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHgStreamA_impl_init)(hgtx_HGstream stream, const char* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHgStreamW_impl_init)(hgtx_HGstream stream, const wchar_t* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHgEventA_impl_init)(hgtx_HGevent event, const char* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHgEventW_impl_init)(hgtx_HGevent event, const wchar_t* name);

HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClDeviceA_impl_init)(hgtx_cl_device_id device, const char* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClDeviceW_impl_init)(hgtx_cl_device_id device, const wchar_t* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClContextA_impl_init)(hgtx_cl_context context, const char* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClContextW_impl_init)(hgtx_cl_context context, const wchar_t* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClCommandQueueA_impl_init)(hgtx_cl_command_queue command_queue, const char* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClCommandQueueW_impl_init)(hgtx_cl_command_queue command_queue, const wchar_t* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClMemObjectA_impl_init)(hgtx_cl_mem memobj, const char* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClMemObjectW_impl_init)(hgtx_cl_mem memobj, const wchar_t* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClSamplerA_impl_init)(hgtx_cl_sampler sampler, const char* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClSamplerW_impl_init)(hgtx_cl_sampler sampler, const wchar_t* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClProgramA_impl_init)(hgtx_cl_program program, const char* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClProgramW_impl_init)(hgtx_cl_program program, const wchar_t* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClEventA_impl_init)(hgtx_cl_event evnt, const char* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClEventW_impl_init)(hgtx_cl_event evnt, const wchar_t* name);

HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcDeviceA_impl_init)(int device, const char* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcDeviceW_impl_init)(int device, const wchar_t* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcStreamA_impl_init)(hgtx_hggcStream_t stream, const char* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcStreamW_impl_init)(hgtx_hggcStream_t stream, const wchar_t* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcEventA_impl_init)(hgtx_hggcEvent_t event, const char* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcEventW_impl_init)(hgtx_hggcEvent_t event, const wchar_t* name);

HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainMarkEx_impl_init)(hgtxDomainHandle_t domain, const hgtxEventAttributes_t* eventAttrib);
HGTX_LINKONCE_FWDDECL_FUNCTION hgtxRangeId_t HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainRangeStartEx_impl_init)(hgtxDomainHandle_t domain, const hgtxEventAttributes_t* eventAttrib);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainRangeEnd_impl_init)(hgtxDomainHandle_t domain, hgtxRangeId_t id);
HGTX_LINKONCE_FWDDECL_FUNCTION int HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainRangePushEx_impl_init)(hgtxDomainHandle_t domain, const hgtxEventAttributes_t* eventAttrib);
HGTX_LINKONCE_FWDDECL_FUNCTION int HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainRangePop_impl_init)(hgtxDomainHandle_t domain);
HGTX_LINKONCE_FWDDECL_FUNCTION hgtxResourceHandle_t HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainResourceCreate_impl_init)(hgtxDomainHandle_t domain, hgtxResourceAttributes_t* attribs);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainResourceDestroy_impl_init)(hgtxResourceHandle_t resource);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainNameCategoryA_impl_init)(hgtxDomainHandle_t domain, uint32_t category, const char* name);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainNameCategoryW_impl_init)(hgtxDomainHandle_t domain, uint32_t category, const wchar_t* name);
HGTX_LINKONCE_FWDDECL_FUNCTION hgtxStringHandle_t HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainRegisterStringA_impl_init)(hgtxDomainHandle_t domain, const char* string);
HGTX_LINKONCE_FWDDECL_FUNCTION hgtxStringHandle_t HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainRegisterStringW_impl_init)(hgtxDomainHandle_t domain, const wchar_t* string);
HGTX_LINKONCE_FWDDECL_FUNCTION hgtxDomainHandle_t HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainCreateA_impl_init)(const char* message);
HGTX_LINKONCE_FWDDECL_FUNCTION hgtxDomainHandle_t HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainCreateW_impl_init)(const wchar_t* message);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainDestroy_impl_init)(hgtxDomainHandle_t domain);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxInitialize_impl_init)(const void* reserved);

HGTX_LINKONCE_FWDDECL_FUNCTION hgtx_hgtxSyncUser_t HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserCreate_impl_init)(hgtxDomainHandle_t domain, const hgtx_hgtxSyncUserAttributes_t* attribs);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserDestroy_impl_init)(hgtx_hgtxSyncUser_t handle);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserAcquireStart_impl_init)(hgtx_hgtxSyncUser_t handle);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserAcquireFailed_impl_init)(hgtx_hgtxSyncUser_t handle);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserAcquireSuccess_impl_init)(hgtx_hgtxSyncUser_t handle);
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserReleasing_impl_init)(hgtx_hgtxSyncUser_t handle);
