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

#if defined(HGTX_AS_SYSTEM_HEADER)
#if defined(__clang__)
#pragma clang system_header
#elif defined(__GNUC__) || defined(__HGCOMPILER)
#pragma GCC system_header
#elif defined(_MSC_VER)
#pragma system_header
#endif
#endif

HGTX_DECLSPEC void HGTX_API hgtxMarkEx(const hgtxEventAttributes_t* eventAttrib)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)eventAttrib;
#else /* HGTX_DISABLE */
    hgtxMarkEx_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxMarkEx_impl_fnptr;
    if (local != HGTX_NULLPTR)
        (*local)(eventAttrib);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxMarkA(const char* message)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)message;
#else /* HGTX_DISABLE */
    hgtxMarkA_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxMarkA_impl_fnptr;
    if (local != HGTX_NULLPTR)
        (*local)(message);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxMarkW(const wchar_t* message)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)message;
#else /* HGTX_DISABLE */
    hgtxMarkW_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxMarkW_impl_fnptr;
    if (local != HGTX_NULLPTR)
        (*local)(message);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC hgtxRangeId_t HGTX_API hgtxRangeStartEx(const hgtxEventAttributes_t* eventAttrib)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)eventAttrib;
#else /* HGTX_DISABLE */
    hgtxRangeStartEx_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangeStartEx_impl_fnptr;
    if (local != HGTX_NULLPTR)
        return (*local)(eventAttrib);
    else
#endif /* HGTX_DISABLE */
        return HGTX_STATIC_CAST(hgtxRangeId_t, 0);
}

HGTX_DECLSPEC hgtxRangeId_t HGTX_API hgtxRangeStartA(const char* message)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)message;
#else /* HGTX_DISABLE */
    hgtxRangeStartA_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangeStartA_impl_fnptr;
    if (local != HGTX_NULLPTR)
        return (*local)(message);
    else
#endif /* HGTX_DISABLE */
        return HGTX_STATIC_CAST(hgtxRangeId_t, 0);
}

HGTX_DECLSPEC hgtxRangeId_t HGTX_API hgtxRangeStartW(const wchar_t* message)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)message;
#else /* HGTX_DISABLE */
    hgtxRangeStartW_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangeStartW_impl_fnptr;
    if (local != HGTX_NULLPTR)
        return (*local)(message);
    else
#endif /* HGTX_DISABLE */
        return HGTX_STATIC_CAST(hgtxRangeId_t, 0);
}

HGTX_DECLSPEC void HGTX_API hgtxRangeEnd(hgtxRangeId_t id)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)id;
#else /* HGTX_DISABLE */
    hgtxRangeEnd_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangeEnd_impl_fnptr;
    if (local != HGTX_NULLPTR)
        (*local)(id);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC int HGTX_API hgtxRangePushEx(const hgtxEventAttributes_t* eventAttrib)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)eventAttrib;
#else /* HGTX_DISABLE */
    hgtxRangePushEx_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangePushEx_impl_fnptr;
    if (local != HGTX_NULLPTR)
        return (*local)(eventAttrib);
    else
#endif /* HGTX_DISABLE */
        return HGTX_STATIC_CAST(int, HGTX_NO_PUSH_POP_TRACKING);
}

HGTX_DECLSPEC int HGTX_API hgtxRangePushA(const char* message)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)message;
#else /* HGTX_DISABLE */
    hgtxRangePushA_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangePushA_impl_fnptr;
    if (local != HGTX_NULLPTR)
        return (*local)(message);
    else
#endif /* HGTX_DISABLE */
        return HGTX_STATIC_CAST(int, HGTX_NO_PUSH_POP_TRACKING);
}

HGTX_DECLSPEC int HGTX_API hgtxRangePushW(const wchar_t* message)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)message;
#else /* HGTX_DISABLE */
    hgtxRangePushW_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangePushW_impl_fnptr;
    if (local != HGTX_NULLPTR)
        return (*local)(message);
    else
#endif /* HGTX_DISABLE */
        return HGTX_STATIC_CAST(int, HGTX_NO_PUSH_POP_TRACKING);
}

HGTX_DECLSPEC int HGTX_API hgtxRangePop(void)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifndef HGTX_DISABLE
    hgtxRangePop_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxRangePop_impl_fnptr;
    if (local != HGTX_NULLPTR)
        return (*local)();
    else
#endif /* HGTX_DISABLE */
        return HGTX_STATIC_CAST(int, HGTX_NO_PUSH_POP_TRACKING);
}

HGTX_DECLSPEC void HGTX_API hgtxNameCategoryA(uint32_t category, const char* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)category;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameCategoryA_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameCategoryA_impl_fnptr;
    if (local != HGTX_NULLPTR)
        (*local)(category, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameCategoryW(uint32_t category, const wchar_t* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)category;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameCategoryW_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameCategoryW_impl_fnptr;
    if (local != HGTX_NULLPTR)
        (*local)(category, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameOsThreadA(uint32_t threadId, const char* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)threadId;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameOsThreadA_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameOsThreadA_impl_fnptr;
    if (local != HGTX_NULLPTR)
        (*local)(threadId, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxNameOsThreadW(uint32_t threadId, const wchar_t* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)threadId;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxNameOsThreadW_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxNameOsThreadW_impl_fnptr;
    if (local != HGTX_NULLPTR)
        (*local)(threadId, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxDomainMarkEx(hgtxDomainHandle_t domain, const hgtxEventAttributes_t* eventAttrib)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)domain;
    (void)eventAttrib;
#else /* HGTX_DISABLE */
    hgtxDomainMarkEx_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainMarkEx_impl_fnptr;
    if (local != HGTX_NULLPTR)
        (*local)(domain, eventAttrib);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC hgtxRangeId_t HGTX_API hgtxDomainRangeStartEx(hgtxDomainHandle_t domain, const hgtxEventAttributes_t* eventAttrib)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)domain;
    (void)eventAttrib;
#else /* HGTX_DISABLE */
    hgtxDomainRangeStartEx_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainRangeStartEx_impl_fnptr;
    if (local != HGTX_NULLPTR)
        return (*local)(domain, eventAttrib);
    else
#endif /* HGTX_DISABLE */
        return HGTX_STATIC_CAST(hgtxRangeId_t, 0);
}

HGTX_DECLSPEC void HGTX_API hgtxDomainRangeEnd(hgtxDomainHandle_t domain, hgtxRangeId_t id)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)domain;
    (void)id;
#else /* HGTX_DISABLE */
    hgtxDomainRangeEnd_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainRangeEnd_impl_fnptr;
    if (local != HGTX_NULLPTR)
        (*local)(domain, id);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC int HGTX_API hgtxDomainRangePushEx(hgtxDomainHandle_t domain, const hgtxEventAttributes_t* eventAttrib)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)domain;
    (void)eventAttrib;
#else /* HGTX_DISABLE */
    hgtxDomainRangePushEx_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainRangePushEx_impl_fnptr;
    if (local != HGTX_NULLPTR)
        return (*local)(domain, eventAttrib);
    else
#endif /* HGTX_DISABLE */
        return HGTX_STATIC_CAST(int, HGTX_NO_PUSH_POP_TRACKING);
}

HGTX_DECLSPEC int HGTX_API hgtxDomainRangePop(hgtxDomainHandle_t domain)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)domain;
#else /* HGTX_DISABLE */
    hgtxDomainRangePop_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainRangePop_impl_fnptr;
    if (local != HGTX_NULLPTR)
        return (*local)(domain);
    else
#endif /* HGTX_DISABLE */
        return HGTX_STATIC_CAST(int, HGTX_NO_PUSH_POP_TRACKING);
}

HGTX_DECLSPEC hgtxResourceHandle_t HGTX_API hgtxDomainResourceCreate(hgtxDomainHandle_t domain, hgtxResourceAttributes_t* attribs)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)domain;
    (void)attribs;
#else /* HGTX_DISABLE */
    hgtxDomainResourceCreate_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainResourceCreate_impl_fnptr;
    if (local != HGTX_NULLPTR)
        return (*local)(domain, attribs);
    else
#endif /* HGTX_DISABLE */
        return HGTX_NULLPTR;
}

HGTX_DECLSPEC void HGTX_API hgtxDomainResourceDestroy(hgtxResourceHandle_t resource)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)resource;
#else /* HGTX_DISABLE */
    hgtxDomainResourceDestroy_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainResourceDestroy_impl_fnptr;
    if (local != HGTX_NULLPTR)
        (*local)(resource);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxDomainNameCategoryA(hgtxDomainHandle_t domain, uint32_t category, const char* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)domain;
    (void)category;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxDomainNameCategoryA_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainNameCategoryA_impl_fnptr;
    if (local != HGTX_NULLPTR)
        (*local)(domain, category, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxDomainNameCategoryW(hgtxDomainHandle_t domain, uint32_t category, const wchar_t* name)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)domain;
    (void)category;
    (void)name;
#else /* HGTX_DISABLE */
    hgtxDomainNameCategoryW_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainNameCategoryW_impl_fnptr;
    if (local != HGTX_NULLPTR)
        (*local)(domain, category, name);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC hgtxStringHandle_t HGTX_API hgtxDomainRegisterStringA(hgtxDomainHandle_t domain, const char* string)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)domain;
    (void)string;
#else /* HGTX_DISABLE */
    hgtxDomainRegisterStringA_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainRegisterStringA_impl_fnptr;
    if (local != HGTX_NULLPTR)
        return (*local)(domain, string);
    else
#endif /* HGTX_DISABLE */
        return HGTX_NULLPTR;
}

HGTX_DECLSPEC hgtxStringHandle_t HGTX_API hgtxDomainRegisterStringW(hgtxDomainHandle_t domain, const wchar_t* string)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)domain;
    (void)string;
#else /* HGTX_DISABLE */
    hgtxDomainRegisterStringW_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainRegisterStringW_impl_fnptr;
    if (local != HGTX_NULLPTR)
        return (*local)(domain, string);
    else
#endif /* HGTX_DISABLE */
        return HGTX_NULLPTR;
}

HGTX_DECLSPEC hgtxDomainHandle_t HGTX_API hgtxDomainCreateA(const char* message)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)message;
#else /* HGTX_DISABLE */
    hgtxDomainCreateA_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainCreateA_impl_fnptr;
    if (local != HGTX_NULLPTR)
        return (*local)(message);
    else
#endif /* HGTX_DISABLE */
        return HGTX_NULLPTR;
}

HGTX_DECLSPEC hgtxDomainHandle_t HGTX_API hgtxDomainCreateW(const wchar_t* message)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)message;
#else /* HGTX_DISABLE */
    hgtxDomainCreateW_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainCreateW_impl_fnptr;
    if (local != HGTX_NULLPTR)
        return (*local)(message);
    else
#endif /* HGTX_DISABLE */
        return HGTX_NULLPTR;
}

HGTX_DECLSPEC void HGTX_API hgtxDomainDestroy(hgtxDomainHandle_t domain)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)domain;
#else /* HGTX_DISABLE */
    hgtxDomainDestroy_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainDestroy_impl_fnptr;
    if (local != HGTX_NULLPTR)
        (*local)(domain);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxInitialize(const void* reserved)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)reserved;
#else /* HGTX_DISABLE */
    hgtxInitialize_impl_fntype local = HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxInitialize_impl_fnptr;
    if (local != HGTX_NULLPTR)
        (*local)(reserved);
#endif /* HGTX_DISABLE */
}
