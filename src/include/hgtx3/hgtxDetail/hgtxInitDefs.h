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

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxMarkEx_impl_init)(const hgtxEventAttributes_t* eventAttrib){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    hgtxMarkEx(eventAttrib);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxMarkA_impl_init)(const char* message){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    hgtxMarkA(message);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxMarkW_impl_init)(const wchar_t* message){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    hgtxMarkW(message);
}

HGTX_LINKONCE_DEFINE_FUNCTION hgtxRangeId_t HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxRangeStartEx_impl_init)(const hgtxEventAttributes_t* eventAttrib){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    return hgtxRangeStartEx(eventAttrib);
}

HGTX_LINKONCE_DEFINE_FUNCTION hgtxRangeId_t HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxRangeStartA_impl_init)(const char* message){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    return hgtxRangeStartA(message);
}

HGTX_LINKONCE_DEFINE_FUNCTION hgtxRangeId_t HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxRangeStartW_impl_init)(const wchar_t* message){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    return hgtxRangeStartW(message);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxRangeEnd_impl_init)(hgtxRangeId_t id){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    hgtxRangeEnd(id);
}

HGTX_LINKONCE_DEFINE_FUNCTION int HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxRangePushEx_impl_init)(const hgtxEventAttributes_t* eventAttrib){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    return hgtxRangePushEx(eventAttrib);
}

HGTX_LINKONCE_DEFINE_FUNCTION int HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxRangePushA_impl_init)(const char* message){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    return hgtxRangePushA(message);
}

HGTX_LINKONCE_DEFINE_FUNCTION int HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxRangePushW_impl_init)(const wchar_t* message){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    return hgtxRangePushW(message);
}

HGTX_LINKONCE_DEFINE_FUNCTION int HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxRangePop_impl_init)(void){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    return hgtxRangePop();
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameCategoryA_impl_init)(uint32_t category, const char* name){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    hgtxNameCategoryA(category, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameCategoryW_impl_init)(uint32_t category, const wchar_t* name){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    hgtxNameCategoryW(category, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameOsThreadA_impl_init)(uint32_t threadId, const char* name){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    hgtxNameOsThreadA(threadId, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameOsThreadW_impl_init)(uint32_t threadId, const wchar_t* name){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    hgtxNameOsThreadW(threadId, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainMarkEx_impl_init)(hgtxDomainHandle_t domain, const hgtxEventAttributes_t* eventAttrib){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    hgtxDomainMarkEx(domain, eventAttrib);
}

HGTX_LINKONCE_DEFINE_FUNCTION hgtxRangeId_t HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainRangeStartEx_impl_init)(hgtxDomainHandle_t domain, const hgtxEventAttributes_t* eventAttrib){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    return hgtxDomainRangeStartEx(domain, eventAttrib);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainRangeEnd_impl_init)(hgtxDomainHandle_t domain, hgtxRangeId_t id){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    hgtxDomainRangeEnd(domain, id);
}

HGTX_LINKONCE_DEFINE_FUNCTION int HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainRangePushEx_impl_init)(hgtxDomainHandle_t domain, const hgtxEventAttributes_t* eventAttrib){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    return hgtxDomainRangePushEx(domain, eventAttrib);
}

HGTX_LINKONCE_DEFINE_FUNCTION int HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainRangePop_impl_init)(hgtxDomainHandle_t domain){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    return hgtxDomainRangePop(domain);
}

HGTX_LINKONCE_DEFINE_FUNCTION hgtxResourceHandle_t HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainResourceCreate_impl_init)(hgtxDomainHandle_t domain, hgtxResourceAttributes_t* attribs){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    return hgtxDomainResourceCreate(domain, attribs);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainResourceDestroy_impl_init)(hgtxResourceHandle_t resource){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    hgtxDomainResourceDestroy(resource);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainNameCategoryA_impl_init)(hgtxDomainHandle_t domain, uint32_t category, const char* name){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    hgtxDomainNameCategoryA(domain, category, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainNameCategoryW_impl_init)(hgtxDomainHandle_t domain, uint32_t category, const wchar_t* name){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    hgtxDomainNameCategoryW(domain, category, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION hgtxStringHandle_t HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainRegisterStringA_impl_init)(hgtxDomainHandle_t domain, const char* string){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    return hgtxDomainRegisterStringA(domain, string);
}

HGTX_LINKONCE_DEFINE_FUNCTION hgtxStringHandle_t HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainRegisterStringW_impl_init)(hgtxDomainHandle_t domain, const wchar_t* string){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    return hgtxDomainRegisterStringW(domain, string);
}

HGTX_LINKONCE_DEFINE_FUNCTION hgtxDomainHandle_t HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainCreateA_impl_init)(const char* message){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    return hgtxDomainCreateA(message);
}

HGTX_LINKONCE_DEFINE_FUNCTION hgtxDomainHandle_t HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainCreateW_impl_init)(const wchar_t* message){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    return hgtxDomainCreateW(message);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainDestroy_impl_init)(hgtxDomainHandle_t domain){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    hgtxDomainDestroy(domain);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxInitialize_impl_init)(const void* reserved){
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    hgtxInitialize(reserved);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHgDeviceA_impl_init)(hgtx_HGdevice device, const char* name){
    hgtxNameHgDeviceA_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgDeviceA_impl_fnptr;
    if (local)
        local(device, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHgDeviceW_impl_init)(hgtx_HGdevice device, const wchar_t* name){
    hgtxNameHgDeviceW_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgDeviceW_impl_fnptr;
    if (local)
        local(device, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHgContextA_impl_init)(hgtx_HGcontext context, const char* name){
    hgtxNameHgContextA_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgContextA_impl_fnptr;
    if (local)
        local(context, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHgContextW_impl_init)(hgtx_HGcontext context, const wchar_t* name){
    hgtxNameHgContextW_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgContextW_impl_fnptr;
    if (local)
        local(context, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHgStreamA_impl_init)(hgtx_HGstream stream, const char* name){
    hgtxNameHgStreamA_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgStreamA_impl_fnptr;
    if (local)
        local(stream, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHgStreamW_impl_init)(hgtx_HGstream stream, const wchar_t* name){
    hgtxNameHgStreamW_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgStreamW_impl_fnptr;
    if (local)
        local(stream, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHgEventA_impl_init)(hgtx_HGevent event, const char* name){
    hgtxNameHgEventA_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgEventA_impl_fnptr;
    if (local)
        local(event, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHgEventW_impl_init)(hgtx_HGevent event, const wchar_t* name){
    hgtxNameHgEventW_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgEventW_impl_fnptr;
    if (local)
        local(event, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcDeviceA_impl_init)(int device, const char* name){
    hgtxNameHggcDeviceA_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcDeviceA_impl_fnptr;
    if (local)
        local(device, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcDeviceW_impl_init)(int device, const wchar_t* name){
    hgtxNameHggcDeviceW_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcDeviceW_impl_fnptr;
    if (local)
        local(device, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcStreamA_impl_init)(hgtx_hggcStream_t stream, const char* name){
    hgtxNameHggcStreamA_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcStreamA_impl_fnptr;
    if (local)
        local(stream, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcStreamW_impl_init)(hgtx_hggcStream_t stream, const wchar_t* name){
    hgtxNameHggcStreamW_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcStreamW_impl_fnptr;
    if (local)
        local(stream, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcEventA_impl_init)(hgtx_hggcEvent_t event, const char* name){
    hgtxNameHggcEventA_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcEventA_impl_fnptr;
    if (local)
        local(event, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcEventW_impl_init)(hgtx_hggcEvent_t event, const wchar_t* name){
    hgtxNameHggcEventW_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcEventW_impl_fnptr;
    if (local)
        local(event, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClDeviceA_impl_init)(hgtx_cl_device_id device, const char* name){
    hgtxNameClDeviceA_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClDeviceA_impl_fnptr;
    if (local)
        local(device, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClDeviceW_impl_init)(hgtx_cl_device_id device, const wchar_t* name){
    hgtxNameClDeviceW_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClDeviceW_impl_fnptr;
    if (local)
        local(device, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClContextA_impl_init)(hgtx_cl_context context, const char* name){
    hgtxNameClContextA_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClContextA_impl_fnptr;
    if (local)
        local(context, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClContextW_impl_init)(hgtx_cl_context context, const wchar_t* name){
    hgtxNameClContextW_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClContextW_impl_fnptr;
    if (local)
        local(context, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClCommandQueueA_impl_init)(hgtx_cl_command_queue command_queue, const char* name){
    hgtxNameClCommandQueueA_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClCommandQueueA_impl_fnptr;
    if (local)
        local(command_queue, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClCommandQueueW_impl_init)(hgtx_cl_command_queue command_queue, const wchar_t* name){
    hgtxNameClCommandQueueW_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClCommandQueueW_impl_fnptr;
    if (local)
        local(command_queue, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClMemObjectA_impl_init)(hgtx_cl_mem memobj, const char* name){
    hgtxNameClMemObjectA_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClMemObjectA_impl_fnptr;
    if (local)
        local(memobj, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClMemObjectW_impl_init)(hgtx_cl_mem memobj, const wchar_t* name){
    hgtxNameClMemObjectW_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClMemObjectW_impl_fnptr;
    if (local)
        local(memobj, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClSamplerA_impl_init)(hgtx_cl_sampler sampler, const char* name){
    hgtxNameClSamplerA_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClSamplerA_impl_fnptr;
    if (local)
        local(sampler, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClSamplerW_impl_init)(hgtx_cl_sampler sampler, const wchar_t* name){
    hgtxNameClSamplerW_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClSamplerW_impl_fnptr;
    if (local)
        local(sampler, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClProgramA_impl_init)(hgtx_cl_program program, const char* name){
    hgtxNameClProgramA_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClProgramA_impl_fnptr;
    if (local)
        local(program, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClProgramW_impl_init)(hgtx_cl_program program, const wchar_t* name){
    hgtxNameClProgramW_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClProgramW_impl_fnptr;
    if (local)
        local(program, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClEventA_impl_init)(hgtx_cl_event evnt, const char* name){
    hgtxNameClEventA_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClEventA_impl_fnptr;
    if (local)
        local(evnt, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxNameClEventW_impl_init)(hgtx_cl_event evnt, const wchar_t* name){
    hgtxNameClEventW_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClEventW_impl_fnptr;
    if (local)
        local(evnt, name);
}

HGTX_LINKONCE_DEFINE_FUNCTION hgtx_hgtxSyncUser_t HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserCreate_impl_init)(hgtxDomainHandle_t domain, const hgtx_hgtxSyncUserAttributes_t* attribs){
    hgtxDomainSyncUserCreate_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserCreate_impl_fnptr;
    if (local) {
        return local(domain, attribs);
    }
    return HGTX_NULLPTR;
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserDestroy_impl_init)(hgtx_hgtxSyncUser_t handle){
    hgtxDomainSyncUserDestroy_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserDestroy_impl_fnptr;
    if (local)
        local(handle);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserAcquireStart_impl_init)(hgtx_hgtxSyncUser_t handle){
    hgtxDomainSyncUserAcquireStart_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserAcquireStart_impl_fnptr;
    if (local)
        local(handle);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserAcquireFailed_impl_init)(hgtx_hgtxSyncUser_t handle){
    hgtxDomainSyncUserAcquireFailed_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserAcquireFailed_impl_fnptr;
    if (local)
        local(handle);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserAcquireSuccess_impl_init)(hgtx_hgtxSyncUser_t handle){
    hgtxDomainSyncUserAcquireSuccess_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserAcquireSuccess_impl_fnptr;
    if (local)
        local(handle);
}

HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_API HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserReleasing_impl_init)(hgtx_hgtxSyncUser_t handle){
    hgtxDomainSyncUserReleasing_fakeimpl_fntype local;
    HGTX_VERSIONED_IDENTIFIER(hgtxInitOnce)();
    local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserReleasing_impl_fnptr;
    if (local)
        local(handle);
}

HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_VERSIONED_IDENTIFIER(hgtxSetInitFunctionsToNoops)(int forceAllToNoops);
HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_VERSIONED_IDENTIFIER(hgtxSetInitFunctionsToNoops)(int forceAllToNoops)
{
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxMarkEx_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxMarkEx_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxMarkEx_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxMarkA_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxMarkA_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxMarkA_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxMarkW_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxMarkW_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxMarkW_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangeStartEx_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxRangeStartEx_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangeStartEx_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangeStartA_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxRangeStartA_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangeStartA_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangeStartW_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxRangeStartW_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangeStartW_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangeEnd_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxRangeEnd_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangeEnd_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangePushEx_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxRangePushEx_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangePushEx_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangePushA_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxRangePushA_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangePushA_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangePushW_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxRangePushW_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangePushW_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangePop_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxRangePop_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangePop_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameCategoryA_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameCategoryA_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameCategoryA_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameCategoryW_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameCategoryW_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameCategoryW_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameOsThreadA_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameOsThreadA_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameOsThreadA_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameOsThreadW_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameOsThreadW_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameOsThreadW_impl_fnptr = HGTX_NULLPTR;

    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgDeviceA_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameHgDeviceA_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgDeviceA_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgDeviceW_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameHgDeviceW_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgDeviceW_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgContextA_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameHgContextA_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgContextA_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgContextW_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameHgContextW_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgContextW_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgStreamA_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameHgStreamA_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgStreamA_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgStreamW_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameHgStreamW_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgStreamW_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgEventA_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameHgEventA_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgEventA_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgEventW_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameHgEventW_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgEventW_impl_fnptr = HGTX_NULLPTR;

    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClDeviceA_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameClDeviceA_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClDeviceA_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClDeviceW_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameClDeviceW_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClDeviceW_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClContextA_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameClContextA_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClContextA_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClContextW_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameClContextW_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClContextW_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClCommandQueueA_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameClCommandQueueA_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClCommandQueueA_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClCommandQueueW_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameClCommandQueueW_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClCommandQueueW_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClMemObjectA_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameClMemObjectA_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClMemObjectA_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClMemObjectW_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameClMemObjectW_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClMemObjectW_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClSamplerA_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameClSamplerA_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClSamplerA_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClSamplerW_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameClSamplerW_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClSamplerW_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClProgramA_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameClProgramA_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClProgramA_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClProgramW_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameClProgramW_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClProgramW_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClEventA_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameClEventA_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClEventA_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClEventW_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameClEventW_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClEventW_impl_fnptr = HGTX_NULLPTR;

    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcDeviceA_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcDeviceA_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcDeviceA_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcDeviceW_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcDeviceW_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcDeviceW_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcStreamA_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcStreamA_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcStreamA_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcStreamW_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcStreamW_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcStreamW_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcEventA_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcEventA_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcEventA_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcEventW_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxNameHggcEventW_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcEventW_impl_fnptr = HGTX_NULLPTR;

    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainMarkEx_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxDomainMarkEx_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainMarkEx_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainRangeStartEx_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxDomainRangeStartEx_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainRangeStartEx_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainRangeEnd_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxDomainRangeEnd_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainRangeEnd_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainRangePushEx_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxDomainRangePushEx_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainRangePushEx_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainRangePop_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxDomainRangePop_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainRangePop_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainResourceCreate_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxDomainResourceCreate_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainResourceCreate_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainResourceDestroy_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxDomainResourceDestroy_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainResourceDestroy_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainNameCategoryA_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxDomainNameCategoryA_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainNameCategoryA_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainNameCategoryW_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxDomainNameCategoryW_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainNameCategoryW_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainRegisterStringA_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxDomainRegisterStringA_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainRegisterStringA_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainRegisterStringW_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxDomainRegisterStringW_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainRegisterStringW_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainCreateA_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxDomainCreateA_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainCreateA_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainCreateW_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxDomainCreateW_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainCreateW_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainDestroy_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxDomainDestroy_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainDestroy_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxInitialize_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxInitialize_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxInitialize_impl_fnptr = HGTX_NULLPTR;

    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserCreate_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserCreate_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserCreate_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserDestroy_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserDestroy_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserDestroy_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserAcquireStart_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserAcquireStart_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserAcquireStart_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserAcquireFailed_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserAcquireFailed_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserAcquireFailed_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserAcquireSuccess_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserAcquireSuccess_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserAcquireSuccess_impl_fnptr = HGTX_NULLPTR;
    if (HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserReleasing_impl_fnptr == HGTX_VERSIONED_IDENTIFIER(hgtxDomainSyncUserReleasing_impl_init) || forceAllToNoops)
        HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserReleasing_impl_fnptr = HGTX_NULLPTR;
}
