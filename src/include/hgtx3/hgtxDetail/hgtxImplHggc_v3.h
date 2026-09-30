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

#ifndef HGTX_IMPL_GUARD_HGGC
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

typedef void (HGTX_API * hgtxNameHgDeviceA_impl_fntype)(HGdevice device, const char* name);
typedef void (HGTX_API * hgtxNameHgDeviceW_impl_fntype)(HGdevice device, const wchar_t* name);
typedef void (HGTX_API * hgtxNameHgContextA_impl_fntype)(HGcontext context, const char* name);
typedef void (HGTX_API * hgtxNameHgContextW_impl_fntype)(HGcontext context, const wchar_t* name);
typedef void (HGTX_API * hgtxNameHgStreamA_impl_fntype)(HGstream stream, const char* name);
typedef void (HGTX_API * hgtxNameHgStreamW_impl_fntype)(HGstream stream, const wchar_t* name);
typedef void (HGTX_API * hgtxNameHgEventA_impl_fntype)(HGevent event, const char* name);
typedef void (HGTX_API * hgtxNameHgEventW_impl_fntype)(HGevent event, const wchar_t* name);

HGTX_DECLSPEC void HGTX_API hgtxNameHgDeviceA(HGdevice device, const char* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)device;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameHgDeviceA_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgDeviceA_impl_fnptr;
    if (local != HGTX_NULLPTR)
        (*local)(device, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameHgDeviceW(HGdevice device, const wchar_t* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)device;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameHgDeviceW_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgDeviceW_impl_fnptr;
    if (local != HGTX_NULLPTR)
        (*local)(device, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameHgContextA(HGcontext context, const char* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)context;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameHgContextA_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameHgContextA_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgContextA_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(context, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameHgContextW(HGcontext context, const wchar_t* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)context;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameHgContextW_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameHgContextW_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgContextW_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(context, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameHgStreamA(HGstream stream, const char* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)stream;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameHgStreamA_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameHgStreamA_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgStreamA_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(stream, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameHgStreamW(HGstream stream, const wchar_t* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)stream;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameHgStreamW_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameHgStreamW_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgStreamW_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(stream, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameHgEventA(HGevent event, const char* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)event;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameHgEventA_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameHgEventA_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgEventA_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(event, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameHgEventW(HGevent event, const wchar_t* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)event;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameHgEventW_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameHgEventW_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHgEventW_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(event, name);
#endif /* HGTX_DISABLE */
}

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

