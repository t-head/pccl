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

#ifndef HGTX_EXT_IMPL_PAYLOAD_GUARD
#error Never include this file directly -- it is automatically included by hgToolsExtPayload.h (except when HGTX_NO_IMPL is defined).
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

#ifndef HGTX_EXT_IMPL_PAYLOAD_V1
#define HGTX_EXT_IMPL_PAYLOAD_V1

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifdef HGTX_DISABLE

#include "hgtxExtHelperMacros.h"

#define HGTX_EXT_PAYLOAD_IMPL_FN_V1(ret_type, fn_name, signature, arg_names) \
HGTX_DECLSPEC ret_type HGTX_API fn_name signature { \
    HGTX_SET_NAME_MANGLING_OPTIONS \
    HGTX_EXT_HELPER_UNUSED_ARGS arg_names \
    HGTX_EXT_FN_RETURN_INVALID(ret_type) \
}

#define HGTX_EXT_PAYLOAD_IMPL_FN_NOARGS_V1(ret_type, fn_name) \
HGTX_DECLSPEC ret_type HGTX_API fn_name (void) { \
    HGTX_SET_NAME_MANGLING_OPTIONS \
    HGTX_EXT_FN_RETURN_INVALID(ret_type) \
}

#else /* HGTX_DISABLE */

#include "hgtxExtPayloadTypeInfo.h"

/*
 * Function slots for the payload extension. First entry is the module state,
 * initialized to `0` (`HGTX_EXTENSION_FRESH`).
 */
#define HGTX_EXT_PAYLOAD_SLOT_COUNT 63

HGTX_LINKONCE_DEFINE_GLOBAL intptr_t
HGTX_EXT_PAYLOAD_VERSIONED_ID(hgtxExtPayloadSlots)[HGTX_EXT_PAYLOAD_SLOT_COUNT + 1]
    = {0};

/* Avoid warnings about missing prototype. */
HGTX_LINKONCE_FWDDECL_FUNCTION void HGTX_EXT_PAYLOAD_VERSIONED_ID(hgtxExtPayloadInitOnce)(void);
HGTX_LINKONCE_DEFINE_FUNCTION void HGTX_EXT_PAYLOAD_VERSIONED_ID(hgtxExtPayloadInitOnce)(void)
{
    intptr_t* fnSlots = HGTX_EXT_PAYLOAD_VERSIONED_ID(hgtxExtPayloadSlots) + 1;
    hgtxExtModuleSegment_t segment = {
        0, /* unused (only one segment) */
        HGTX_EXT_PAYLOAD_SLOT_COUNT,
        HGTX_NULLPTR /* function slots */
    };

    hgtxExtModuleInfo_t module = {
        HGTX_VERSION, sizeof(hgtxExtModuleInfo_t),
        HGTX_EXT_PAYLOAD_MODULEID, HGTX_EXT_PAYLOAD_COMPATID,
        1, HGTX_NULLPTR, /* number of segments, segments */
        HGTX_NULLPTR, /* no export function needed */
        /* bake type sizes and alignment information into program binary */
        &(HGTX_EXT_PAYLOAD_VERSIONED_ID(hgtxExtPayloadTypeInfo))
    };

    segment.functionSlots = fnSlots;
    module.segments = &segment;

    HGTX_INFO( "%s\n", __FUNCTION__  );

    HGTX_VERSIONED_IDENTIFIER(hgtxExtInitOnce)(&module,
        HGTX_EXT_PAYLOAD_VERSIONED_ID(hgtxExtPayloadSlots));
}

#define HGTX_EXT_PAYLOAD_IMPL_FN_V1(ret_type, fn_name, signature, arg_names) \
typedef ret_type (*fn_name##_impl_fntype)signature; \
HGTX_DECLSPEC ret_type HGTX_API fn_name signature { \
    HGTX_SET_NAME_MANGLING_OPTIONS \
    intptr_t* pSlot = &HGTX_EXT_PAYLOAD_VERSIONED_ID(hgtxExtPayloadSlots)[HGTX3EXT_CBID_##fn_name + 1]; \
    intptr_t slot = *pSlot; \
    if (slot != HGTX_EXTENSION_DISABLED) { \
        if (slot != HGTX_EXTENSION_FRESH) { \
            HGTX_EXT_FN_RETURN (*HGTX_REINTERPRET_CAST(fn_name##_impl_fntype, slot)) arg_names; \
        } else { \
            HGTX_EXT_PAYLOAD_VERSIONED_ID(hgtxExtPayloadInitOnce)(); \
            /* Re-read function slot after extension initialization. */ \
            slot = *pSlot; \
            if (slot != HGTX_EXTENSION_DISABLED && slot != HGTX_EXTENSION_FRESH) { \
                HGTX_EXT_FN_RETURN (*HGTX_REINTERPRET_CAST(fn_name##_impl_fntype, slot)) arg_names; \
            } \
        } \
    } \
    HGTX_EXT_FN_RETURN_INVALID(ret_type) /* No tool attached. */ \
}

#define HGTX_EXT_PAYLOAD_IMPL_FN_NOARGS_V1(ret_type, fn_name) \
    HGTX_EXT_PAYLOAD_IMPL_FN_V1(ret_type, fn_name, (void), ())

#endif /* HGTX_DISABLE */

/* Push/pop functions return `HGTX_NO_PUSH_POP_TRACKING` if no tool is attached. */
#define HGTX_EXT_FN_RETURN return
#define HGTX_EXT_FN_RETURN_INVALID(rtype) return HGTX_NO_PUSH_POP_TRACKING;

HGTX_EXT_PAYLOAD_IMPL_FN_V1(int, hgtxRangePushPayload,
    (hgtxDomainHandle_t domain, const hgtxPayloadData_t* payloadData, size_t count),
    (domain, payloadData, count))

HGTX_EXT_PAYLOAD_IMPL_FN_V1(int, hgtxRangePopPayload,
    (hgtxDomainHandle_t domain, const hgtxPayloadData_t* payloadData, size_t count),
    (domain, payloadData, count))

#undef HGTX_EXT_FN_RETURN
#undef HGTX_EXT_FN_RETURN_INVALID

/* Non-void functions. */
#define HGTX_EXT_FN_RETURN return
#define HGTX_EXT_FN_RETURN_INVALID(rtype) return HGTX_STATIC_CAST(rtype, 0);

HGTX_EXT_PAYLOAD_IMPL_FN_V1(uint64_t, hgtxPayloadSchemaRegister,
    (hgtxDomainHandle_t domain, const hgtxPayloadSchemaAttr_t* attr),
    (domain, attr))

HGTX_EXT_PAYLOAD_IMPL_FN_V1(uint64_t, hgtxPayloadEnumRegister,
    (hgtxDomainHandle_t domain, const hgtxPayloadEnumAttr_t* attr),
    (domain, attr))

HGTX_EXT_PAYLOAD_IMPL_FN_V1(hgtxRangeId_t, hgtxRangeStartPayload,
    (hgtxDomainHandle_t domain, const hgtxPayloadData_t* payloadData, size_t count),
    (domain, payloadData, count))

HGTX_EXT_PAYLOAD_IMPL_FN_V1(uint8_t, hgtxDomainIsEnabled, (hgtxDomainHandle_t domain), (domain))

HGTX_EXT_PAYLOAD_IMPL_FN_V1(uint64_t, hgtxScopeRegister, (hgtxDomainHandle_t domain,
    const hgtxScopeAttr_t* attr), (domain, attr))

HGTX_EXT_PAYLOAD_IMPL_FN_NOARGS_V1(int64_t, hgtxTimestampGet)

HGTX_EXT_PAYLOAD_IMPL_FN_V1(uint64_t, hgtxTimeDomainRegister,
    (hgtxDomainHandle_t domain, const hgtxTimeDomainAttr_t* attr),
    (domain, attr))

#undef HGTX_EXT_FN_RETURN
#undef HGTX_EXT_FN_RETURN_INVALID
/* END: Non-void functions. */

/* void functions. */
#define HGTX_EXT_FN_RETURN
#define HGTX_EXT_FN_RETURN_INVALID(rtype)

HGTX_EXT_PAYLOAD_IMPL_FN_V1(void, hgtxMarkPayload, (hgtxDomainHandle_t domain,
    const hgtxPayloadData_t* payloadData, size_t count), (domain, payloadData, count))

HGTX_EXT_PAYLOAD_IMPL_FN_V1(void, hgtxRangeEndPayload, (hgtxDomainHandle_t domain,
    hgtxRangeId_t id, const hgtxPayloadData_t* payloadData, size_t count),
    (domain, id, payloadData, count))

HGTX_EXT_PAYLOAD_IMPL_FN_V1(void, hgtxTimerSource,
    (hgtxDomainHandle_t domain, uint64_t timeDomainId, uint64_t flags, int64_t (*timestampProviderFn)(void)),
    (domain, timeDomainId, flags, timestampProviderFn))

HGTX_EXT_PAYLOAD_IMPL_FN_V1(void, hgtxTimerSourceWithData,
    (hgtxDomainHandle_t domain, uint64_t timeDomainId, uint64_t flags, int64_t (*timestampProviderFn)(void* data), void* data),
    (domain, timeDomainId, flags, timestampProviderFn, data))

HGTX_EXT_PAYLOAD_IMPL_FN_V1(void, hgtxTimeSyncPoint,
    (hgtxDomainHandle_t domain, uint64_t timeDomainId1, uint64_t timeDomainId2,
        int64_t timestamp1, int64_t timestamp2),
    (domain, timeDomainId1, timeDomainId2, timestamp1, timestamp2))

HGTX_EXT_PAYLOAD_IMPL_FN_V1(void, hgtxTimeSyncPointTable,
    (hgtxDomainHandle_t domain, uint64_t timeDomainIdSrc, uint64_t timeDomainIdDst,
    const hgtxSyncPoint_t* syncPoints, size_t count),
    (domain, timeDomainIdSrc, timeDomainIdDst, syncPoints, count))

HGTX_EXT_PAYLOAD_IMPL_FN_V1(void, hgtxTimestampConversionFactor,
    (hgtxDomainHandle_t domain, uint64_t timeDomainIdSrc, uint64_t timeDomainIdDst,
        double slope, int64_t timestampSrc, int64_t timestampDst),
    (domain, timeDomainIdSrc, timeDomainIdDst, slope, timestampSrc, timestampDst))

HGTX_EXT_PAYLOAD_IMPL_FN_V1(void, hgtxEventSubmit,
    (hgtxDomainHandle_t domain, const hgtxPayloadData_t* payloadData, size_t numPayloads),
    (domain, payloadData, numPayloads))

HGTX_EXT_PAYLOAD_IMPL_FN_V1(void, hgtxEventBatchSubmit, (hgtxDomainHandle_t domain,
    const hgtxEventBatch_t* eventBatch), (domain, eventBatch))

#undef HGTX_EXT_FN_RETURN
#undef HGTX_EXT_FN_RETURN_INVALID
/* END: void functions. */

/* Keep HGTX_EXT_PAYLOAD_IMPL_FN_V1 defined for a future version of this extension. */

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* HGTX_EXT_IMPL_PAYLOAD_V1 */

