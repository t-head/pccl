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

#ifndef HGTX_IMPL_GUARD_OPENCL
#error Never include this file directly -- it is automatically included by hgToolsExtHggc.h (except when HGTX_NO_IMPL is defined).
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


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef void (HGTX_API * hgtxNameClDeviceA_impl_fntype)(cl_device_id device, const char* name);
typedef void (HGTX_API * hgtxNameClDeviceW_impl_fntype)(cl_device_id device, const wchar_t* name);
typedef void (HGTX_API * hgtxNameClContextA_impl_fntype)(cl_context context, const char* name);
typedef void (HGTX_API * hgtxNameClContextW_impl_fntype)(cl_context context, const wchar_t* name);
typedef void (HGTX_API * hgtxNameClCommandQueueA_impl_fntype)(cl_command_queue command_queue, const char* name);
typedef void (HGTX_API * hgtxNameClCommandQueueW_impl_fntype)(cl_command_queue command_queue, const wchar_t* name);
typedef void (HGTX_API * hgtxNameClMemObjectA_impl_fntype)(cl_mem memobj, const char* name);
typedef void (HGTX_API * hgtxNameClMemObjectW_impl_fntype)(cl_mem memobj, const wchar_t* name);
typedef void (HGTX_API * hgtxNameClSamplerA_impl_fntype)(cl_sampler sampler, const char* name);
typedef void (HGTX_API * hgtxNameClSamplerW_impl_fntype)(cl_sampler sampler, const wchar_t* name);
typedef void (HGTX_API * hgtxNameClProgramA_impl_fntype)(cl_program program, const char* name);
typedef void (HGTX_API * hgtxNameClProgramW_impl_fntype)(cl_program program, const wchar_t* name);
typedef void (HGTX_API * hgtxNameClEventA_impl_fntype)(cl_event evnt, const char* name);
typedef void (HGTX_API * hgtxNameClEventW_impl_fntype)(cl_event evnt, const wchar_t* name);

HGTX_DECLSPEC void HGTX_API hgtxNameClDeviceA(cl_device_id device, const char* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)device;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameClDeviceA_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameClDeviceA_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClDeviceA_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(device, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameClDeviceW(cl_device_id device, const wchar_t* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)device;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameClDeviceW_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameClDeviceW_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClDeviceW_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(device, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameClContextA(cl_context context, const char* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)context;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameClContextA_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameClContextA_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClContextA_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(context, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameClContextW(cl_context context, const wchar_t* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)context;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameClContextW_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameClContextW_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClContextW_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(context, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameClCommandQueueA(cl_command_queue command_queue, const char* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)command_queue;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameClCommandQueueA_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameClCommandQueueA_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClCommandQueueA_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(command_queue, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameClCommandQueueW(cl_command_queue command_queue, const wchar_t* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)command_queue;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameClCommandQueueW_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameClCommandQueueW_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClCommandQueueW_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(command_queue, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameClMemObjectA(cl_mem memobj, const char* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)memobj;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameClMemObjectA_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameClMemObjectA_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClMemObjectA_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(memobj, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameClMemObjectW(cl_mem memobj, const wchar_t* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)memobj;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameClMemObjectW_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameClMemObjectW_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClMemObjectW_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(memobj, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameClSamplerA(cl_sampler sampler, const char* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)sampler;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameClSamplerA_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameClSamplerA_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClSamplerA_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(sampler, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameClSamplerW(cl_sampler sampler, const wchar_t* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)sampler;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameClSamplerW_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameClSamplerW_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClSamplerW_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(sampler, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameClProgramA(cl_program program, const char* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)program;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameClProgramA_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameClProgramA_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClProgramA_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(program, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameClProgramW(cl_program program, const wchar_t* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)program;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameClProgramW_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameClProgramW_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClProgramW_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(program, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameClEventA(cl_event evnt, const char* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)evnt;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameClEventA_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameClEventA_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClEventA_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(evnt, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameClEventW(cl_event evnt, const wchar_t* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)evnt;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameClEventW_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameClEventW_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameClEventW_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(evnt, name);
#endif /* HGTX_DISABLE */
}

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */
