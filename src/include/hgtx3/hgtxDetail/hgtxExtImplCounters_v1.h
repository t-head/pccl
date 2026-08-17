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

#ifndef HGTX_EXT_IMPL_COUNTERS_GUARD
#error Never include this file directly -- it is automatically included by hgToolsExtCounters.h (except when HGTX_NO_IMPL is defined).
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

#define HGTX_EXT_IMPL_GUARD
#include "hgtxExtImpl.h"
#undef HGTX_EXT_IMPL_GUARD

#ifndef HGTX_EXT_IMPL_COUNTERS_V1
#define HGTX_EXT_IMPL_COUNTERS_V1

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifdef HGTX_DISABLE

#include "hgtxExtHelperMacros.h"

#define HGTX_EXT_COUNTERS_IMPL_FN_V1(ret_type, fn_name, signature, arg_names) \
HGTX_DECLSPEC ret_type HGTX_API fn_name signature { \
    HGTX_SET_NAME_MANGLING_OPTIONS \
    HGTX_EXT_HELPER_UNUSED_ARGS arg_names \
    HGTX_EXT_FN_RETURN_INVALID(ret_type) \
}

#else /* HGTX_DISABLE */

/*
 * Function slots for the counters extension. First entry is the module state,
 * initialized to `0` (`HGTX_EXTENSION_FRESH`).
 */
#define HGTX_EXT_COUNTERS_SLOT_COUNT 63

HGTX_LINKONCE_DEFINE_GLOBAL intptr_t
HGTX_EXT_COUNTERS_VERSIONED_ID(hgtxExtCountersSlots)[HGTX_EXT_COUNTERS_SLOT_COUNT + 1]
    = {0};

/* Avoid warnings about missing prototype. */
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_EXT_COUNTERS_VERSIONED_ID(hgtxExtCountersInitOnce)(void);
HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_EXT_COUNTERS_VERSIONED_ID(hgtxExtCountersInitOnce)(void)
{
    intptr_t* fnSlots = HGTX_EXT_COUNTERS_VERSIONED_ID(hgtxExtCountersSlots) + 1;
    hgtxExtModuleSegment_t segment = {
        0, /* unused (only one segment) */
        HGTX_EXT_COUNTERS_SLOT_COUNT,
        HGTX_NULLPTR /* function slots */
    };

    hgtxExtModuleInfo_t module = {
        HGTX_VERSION, sizeof(hgtxExtModuleInfo_t),
        HGTX_EXT_COUNTERS_MODULEID, HGTX_EXT_COUNTERS_COMPATID,
        1, HGTX_NULLPTR, /* number of segments, segments */
        HGTX_NULLPTR, /* no export function needed */
        HGTX_NULLPTR /* no extension private info */
    };

    segment.functionSlots = fnSlots;
    module.segments = &segment;

    HGTX_INFO( "%s\n", __FUNCTION__  );

    HGTX_VERSIONED_IDENTIFIER(hgtxExtInitOnce)(&module,
        HGTX_EXT_COUNTERS_VERSIONED_ID(hgtxExtCountersSlots));
}

#define HGTX_EXT_COUNTERS_IMPL_FN_V1(ret_type, fn_name, signature, arg_names) \
typedef ret_type (*fn_name##_impl_fntype)signature; \
HGTX_DECLSPEC ret_type HGTX_API fn_name signature { \
    HGTX_SET_NAME_MANGLING_OPTIONS \
    intptr_t* pSlot = &HGTX_EXT_COUNTERS_VERSIONED_ID(hgtxExtCountersSlots)[HGTX3EXT_CBID_##fn_name + 1]; \
    intptr_t slot = *pSlot; \
    if (slot != HGTX_EXTENSION_DISABLED) { \
        if (slot != HGTX_EXTENSION_FRESH) { \
            HGTX_EXT_FN_RETURN (*HGTX_REINTERPRET_CAST(fn_name##_impl_fntype, slot)) arg_names; \
        } else { \
            HGTX_EXT_COUNTERS_VERSIONED_ID(hgtxExtCountersInitOnce)(); \
            /* Re-read function slot after extension initialization. */ \
            slot = *pSlot; \
            if (slot != HGTX_EXTENSION_DISABLED && slot != HGTX_EXTENSION_FRESH) { \
                HGTX_EXT_FN_RETURN (*HGTX_REINTERPRET_CAST(fn_name##_impl_fntype, slot)) arg_names; \
            } \
        } \
    } \
    HGTX_EXT_FN_RETURN_INVALID(ret_type) /* No tool attached. */ \
}

#endif /* HGTX_DISABLE */

/* Non-void functions. */
#define HGTX_EXT_FN_RETURN return
#define HGTX_EXT_FN_RETURN_INVALID(rtype) return HGTX_STATIC_CAST(rtype, 0);

HGTX_EXT_COUNTERS_IMPL_FN_V1(uint64_t, hgtxCounterRegister,
    (hgtxDomainHandle_t domain, const hgtxCounterAttr_t* attr),
    (domain, attr))

#undef HGTX_EXT_FN_RETURN
#undef HGTX_EXT_FN_RETURN_INVALID
/* END: Non-void functions. */

/* void functions. */
#define HGTX_EXT_FN_RETURN
#define HGTX_EXT_FN_RETURN_INVALID(rtype)

HGTX_EXT_COUNTERS_IMPL_FN_V1(void, hgtxCounterSampleInt64,
    (hgtxDomainHandle_t domain, uint64_t counterId, int64_t value),
    (domain, counterId, value))

HGTX_EXT_COUNTERS_IMPL_FN_V1(void, hgtxCounterSampleFloat64,
    (hgtxDomainHandle_t domain, uint64_t counterId, double value),
    (domain, counterId, value))

HGTX_EXT_COUNTERS_IMPL_FN_V1(void, hgtxCounterSample,
    (hgtxDomainHandle_t domain, uint64_t counterId, const void* values, size_t size),
    (domain, counterId, values, size))

HGTX_EXT_COUNTERS_IMPL_FN_V1(void, hgtxCounterSampleNoValue,
    (hgtxDomainHandle_t domain, uint64_t counterId, uint8_t reason),
    (domain, counterId, reason))

HGTX_EXT_COUNTERS_IMPL_FN_V1(void, hgtxCounterBatchSubmit,
    (hgtxDomainHandle_t domain, const hgtxCounterBatch_t* counterData),
    (domain, counterData))

#undef HGTX_EXT_FN_RETURN
#undef HGTX_EXT_FN_RETURN_INVALID
/* END: void functions. */

/* Keep HGTX_EXT_COUNTERS_IMPL_FN_V1 defined for a future version of this extension. */

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* HGTX_EXT_IMPL_COUNTERS_V1 */
