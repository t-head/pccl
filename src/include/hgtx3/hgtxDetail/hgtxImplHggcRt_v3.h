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

#ifndef HGTX_IMPL_GUARD_HGGCRT
#error Never include this file directly -- it is automatically included by hgToolsExtHggcRt.h (except when HGTX_NO_IMPL is defined).
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

typedef void (HGTX_API * hgtxNameHggcDeviceA_impl_fntype)(int device, const char* name);
typedef void (HGTX_API * hgtxNameHggcDeviceW_impl_fntype)(int device, const wchar_t* name);
typedef void (HGTX_API * hgtxNameHggcStreamA_impl_fntype)(hggcStream_t stream, const char* name);
typedef void (HGTX_API * hgtxNameHggcStreamW_impl_fntype)(hggcStream_t stream, const wchar_t* name);
typedef void (HGTX_API * hgtxNameHggcEventA_impl_fntype)(hggcEvent_t event, const char* name);
typedef void (HGTX_API * hgtxNameHggcEventW_impl_fntype)(hggcEvent_t event, const wchar_t* name);

HGTX_DECLSPEC void HGTX_API hgtxNameHggcDeviceA(int device, const char* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)device;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameHggcDeviceA_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcDeviceA_impl_fnptr;
    if (local != HGTX_NULLPTR)
        (*local)(device, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameHggcDeviceW(int device, const wchar_t* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)device;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameHggcDeviceW_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcDeviceW_impl_fnptr;
    if (local != HGTX_NULLPTR)
        (*local)(device, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameHggcStreamA(hggcStream_t stream, const char* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)stream;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameHggcStreamA_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameHggcStreamA_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcStreamA_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(stream, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameHggcStreamW(hggcStream_t stream, const wchar_t* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)stream;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameHggcStreamW_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameHggcStreamW_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcStreamW_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(stream, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameHggcEventA(hggcEvent_t event, const char* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)event;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameHggcEventA_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameHggcEventA_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcEventA_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(event, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameHggcEventW(hggcEvent_t event, const wchar_t* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)event;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameHggcEventW_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxNameHggcEventW_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameHggcEventW_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(event, name);
#endif /* HGTX_DISABLE */
}

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

