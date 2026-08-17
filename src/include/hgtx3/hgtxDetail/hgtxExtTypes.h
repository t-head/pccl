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

#ifndef HGTX_EXT_TYPES_GUARD
#error Never include this file directly -- it is automatically included by hgToolsExt[EXTENSION].h.
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

/* This header defines types which are used by the internal implementation
*  of HGTX and callback subscribers.  API clients do not use these types,
*  so they are defined here instead of in hgToolsExt.h to clarify they are
*  not part of the HGTX client API. */

#ifndef HGTXEXTTYPES_H
#define HGTXEXTTYPES_H

typedef intptr_t (HGTX_API * HgtxExtGetExportFunction_t)(uint32_t exportFunctionId);

typedef struct hgtxExtModuleSegment_t
{
    size_t segmentId;
    size_t slotCount;
    intptr_t* functionSlots;
} hgtxExtModuleSegment_t;

typedef struct hgtxExtModuleInfo_t
{
    uint16_t hgtxVer;
    uint16_t structSize;
    uint16_t moduleId;
    uint16_t compatId;
    size_t segmentsCount;
    hgtxExtModuleSegment_t* segments;
    HgtxExtGetExportFunction_t getExportFunction;
    const void* extInfo;
} hgtxExtModuleInfo_t;

typedef int (HGTX_API * HgtxExtInitializeInjectionFunc_t)(hgtxExtModuleInfo_t* moduleInfo);

#endif /* HGTXEXTTYPES_H */
