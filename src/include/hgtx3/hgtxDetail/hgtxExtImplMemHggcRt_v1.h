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

#ifndef HGTX_EXT_IMPL_MEM_HGGCRT_GUARD
#error Never include this file directly -- it is automatically included by hgToolsExtMemHggcRt.h (except when HGTX_NO_IMPL is defined).
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

#ifndef HGTX_EXT_IMPL_MEM_HGGCRT_V1
#define HGTX_EXT_IMPL_MEM_HGGCRT_V1

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* HGTX_EXT_MEM_IMPL_FN_V1 defined in hgtxExtImplMem_v1.h */

/* Non-void functions. */
#define HGTX_EXT_FN_RETURN return
#define HGTX_EXT_FN_RETURN_INVALID(rtype) return HGTX_NULLPTR;

HGTX_EXT_MEM_IMPL_FN_V1(hgtxMemPermissionsHandle_t, hgtxMemHggcGetProcessWidePermissions, (hgtxDomainHandle_t domain), (domain))

HGTX_EXT_MEM_IMPL_FN_V1(hgtxMemPermissionsHandle_t, hgtxMemHggcGetDeviceWidePermissions, (hgtxDomainHandle_t domain, int device), (domain, device))

#undef HGTX_EXT_FN_RETURN
#undef HGTX_EXT_FN_RETURN_INVALID
/* END: Non-void functions. */

/* void functions. */
#define HGTX_EXT_FN_RETURN
#define HGTX_EXT_FN_RETURN_INVALID(rtype)

HGTX_EXT_MEM_IMPL_FN_V1(void, hgtxMemHggcSetPeerAccess, (hgtxDomainHandle_t domain, hgtxMemPermissionsHandle_t permissions, int devicePeer, uint32_t flags), (domain, permissions, devicePeer, flags))

HGTX_EXT_MEM_IMPL_FN_V1(void, hgtxMemHggcMarkInitialized, (hgtxDomainHandle_t domain, hggcStream_t stream, uint8_t isPerThreadStream, hgtxMemMarkInitializedBatch_t const* desc), (domain, stream, isPerThreadStream, desc))

#undef HGTX_EXT_FN_RETURN
#undef HGTX_EXT_FN_RETURN_INVALID
/* END: void functions. */

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* HGTX_EXT_IMPL_MEM_HGGCRT_V1 */
