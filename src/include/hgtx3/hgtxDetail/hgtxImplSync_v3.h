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

#ifndef HGTX_IMPL_GUARD_SYNC
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

typedef hgtxSyncUser_t (HGTX_API * hgtxDomainSyncUserCreate_impl_fntype)(hgtxDomainHandle_t domain, const hgtxSyncUserAttributes_t* attribs);
typedef void (HGTX_API * hgtxDomainSyncUserDestroy_impl_fntype)(hgtxSyncUser_t handle);
typedef void (HGTX_API * hgtxDomainSyncUserAcquireStart_impl_fntype)(hgtxSyncUser_t handle);
typedef void (HGTX_API * hgtxDomainSyncUserAcquireFailed_impl_fntype)(hgtxSyncUser_t handle);
typedef void (HGTX_API * hgtxDomainSyncUserAcquireSuccess_impl_fntype)(hgtxSyncUser_t handle);
typedef void (HGTX_API * hgtxDomainSyncUserReleasing_impl_fntype)(hgtxSyncUser_t handle);

HGTX_DECLSPEC hgtxSyncUser_t HGTX_API hgtxDomainSyncUserCreate(hgtxDomainHandle_t domain, const hgtxSyncUserAttributes_t* attribs)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)domain;
    (void)attribs;
#else /* HGTX_DISABLE */
    hgtxDomainSyncUserCreate_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxDomainSyncUserCreate_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserCreate_impl_fnptr);
    if (local != HGTX_NULLPTR)
        return (*local)(domain, attribs);
    else
#endif /* HGTX_DISABLE */
        return HGTX_NULLPTR;
}

HGTX_DECLSPEC void HGTX_API hgtxDomainSyncUserDestroy(hgtxSyncUser_t handle)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)handle;
#else /* HGTX_DISABLE */
    hgtxDomainSyncUserDestroy_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxDomainSyncUserDestroy_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserDestroy_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(handle);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxDomainSyncUserAcquireStart(hgtxSyncUser_t handle)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)handle;
#else /* HGTX_DISABLE */
    hgtxDomainSyncUserAcquireStart_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxDomainSyncUserAcquireStart_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserAcquireStart_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(handle);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxDomainSyncUserAcquireFailed(hgtxSyncUser_t handle)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)handle;
#else /* HGTX_DISABLE */
    hgtxDomainSyncUserAcquireFailed_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxDomainSyncUserAcquireFailed_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserAcquireFailed_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(handle);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxDomainSyncUserAcquireSuccess(hgtxSyncUser_t handle)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)handle;
#else /* HGTX_DISABLE */
    hgtxDomainSyncUserAcquireSuccess_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxDomainSyncUserAcquireSuccess_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserAcquireSuccess_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(handle);
#endif /* HGTX_DISABLE */
}

HGTX_DECLSPEC void HGTX_API hgtxDomainSyncUserReleasing(hgtxSyncUser_t handle)
{
    HGTX_SET_NAME_MANGLING_OPTIONS
#ifdef HGTX_DISABLE
    (void)handle;
#else /* HGTX_DISABLE */
    hgtxDomainSyncUserReleasing_impl_fntype local = HGTX_REINTERPRET_CAST(hgtxDomainSyncUserReleasing_impl_fntype, HGTX_VERSIONED_IDENTIFIER(hgtxGlobals).hgtxDomainSyncUserReleasing_impl_fnptr);
    if (local != HGTX_NULLPTR)
        (*local)(handle);
#endif /* HGTX_DISABLE */
}

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */
