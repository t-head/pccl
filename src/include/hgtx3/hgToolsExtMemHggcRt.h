/*
 * SPDX-FileCopyrightText: Copyright (c) 2009-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "hgToolsExtMem.h"

#include "hggc.h"
#include "hggc_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifndef HGTX_MEM_HGGCRT_CONTENTS_V1
#define HGTX_MEM_HGGCRT_CONTENTS_V1

/** \defgroup MEMORY_HGGCRT Memory HGGC Runtime
 * See page \ref PAGE_MEMORY_HGGCRT.
 * @{
 */

/** \brief The memory is from a HGGC runtime array.
 *
 * Relevant functions: hggcMallocArray,  hggcMalloc3DArray
 * Also hggcArray_t from other types such as hggcMipmappedArray_t
 *
 * HGTX_MEM_HEAP_HANDLE_PROCESS_WIDE is not supported
 *
 * hgtxMemHeapRegister receives a heapDesc of type hggcArray_t because the description can be retrieved by tools through hggcArrayGetInfo()
 * hgtxMemRegionRegisterEx receives a regionDesc of type hgtxMemHggcArrayRangeDesc_t
 */
#define HGTX_MEM_TYPE_HGGC_ARRAY 0x11

/** \brief structure to describe memory in a HGGC array object
 */
typedef struct hgtxMemHggcArrayRangeDesc_v1
{
    uint16_t extCompatID; /* Set to HGTX_EXT_COMPATID_MEM */
    uint16_t structSize; /* Size of the structure. */
    uint32_t reserved0;
    hggcArray_t  src;
    size_t offset[3];
    size_t extent[3];
} hgtxMemHggcArrayRangeDesc_v1;
typedef hgtxMemHggcArrayRangeDesc_v1 hgtxMemHggcArrayRangeDesc_t;


/** \brief The memory is from a HGGC device array.
 *
 * Relevant functions: hgArrayCreate,  hgArray3DCreate
 * Also HGarray from other types such as HGmipmappedArray
 *
 * HGTX_MEM_HEAP_HANDLE_PROCESS_WIDE is not supported
 *
 * hgtxMemHeapRegister receives a heapDesc of type hggcArray_t because the description can be retrieved by tools through hggcArrayGetInfo()
 * hgtxMemRegionRegisterEx receives a regionDesc of type hgtxMemHgArrayRangeDesc_t
 */
#define HGTX_MEM_TYPE_HG_ARRAY 0x12

/** \brief structure to describe memory in a HGGC array object
 */
typedef struct hgtxMemHgArrayRangeDesc_v1
{
    uint16_t extCompatID; /* Set to HGTX_EXT_COMPATID_MEM */
    uint16_t structSize; /* Size of the structure. */
    uint32_t reserved0;
    HGarray  src;
    size_t offset[3];
    size_t extent[3];
} hgtxMemHgArrayRangeDesc_v1;
typedef hgtxMemHgArrayRangeDesc_v1 hgtxMemHgArrayRangeDesc_t;

/* Reserving 0x2-0xF for more common types */

#define HGTX_MEM_HGGC_PEER_ALL_DEVICES -1

/** \brief Get the permission object that represent the HGGC runtime device
 * or hggc driver context
 *
 * This object will allow developers to adjust permissions applied to work executed
 * on the GPU.  It may be inherited or overridden by permissions object bound
 * with HGTX_MEM_PERMISSIONS_BIND_SCOPE_HGGC_STREAM, depending on the binding flags.
 *
 * Ex. change the peer to peer access permissions between devices in entirety
 * or punch through special holes
 *
 * By default, all memory is accessible that naturally would be to a HGGC kernel until
 * modified otherwise by hgtxMemHggcSetPeerAccess or changing regions.
 *
 * This object should also represent the HGGC driver API level context.
*/
HGTX_DECLSPEC hgtxMemPermissionsHandle_t HGTX_API hgtxMemHggcGetProcessWidePermissions(
    hgtxDomainHandle_t domain);

/** \brief Get the permission object that represent the HGGC runtime device
 * or hggc driver context
 *
 * This object will allow developers to adjust permissions applied to work executed
 * on the GPU.  It may be inherited or overridden by permissions object bound
 * with HGTX_MEM_PERMISSIONS_BIND_SCOPE_HGGC_STREAM, depending on the binding flags.
 *
 * Ex. change the peer to peer access permissions between devices in entirety
 * or punch through special holes
 *
 * By default, all memory is accessible that naturally would be to a HGGC kernel until
 * modified otherwise by hgtxMemHggcSetPeerAccess or changing regions.
 *
 * This object should also represent the HGGC driver API level context.
*/
HGTX_DECLSPEC hgtxMemPermissionsHandle_t HGTX_API hgtxMemHggcGetDeviceWidePermissions(
    hgtxDomainHandle_t domain,
    int device);

/** \brief Change the default behavior for all memory mapped in from a particular device.
 *
 * While typically all memory defaults to readable and writable, users may desire to limit
 * access to reduced default permissions such as read-only and a per-device basis.
 *
 * Regions can used to further override smaller windows of memory.
 *
 * devicePeer can be HGTX_MEM_HGGC_PEER_ALL_DEVICES
 *
*/
HGTX_DECLSPEC void HGTX_API hgtxMemHggcSetPeerAccess(
    hgtxDomainHandle_t domain,
    hgtxMemPermissionsHandle_t permissions,
    int devicePeer, /* device number such as from hggcGetDevice() or HGTX_MEM_HGGC_PEER_ALL_DEVICES */
    uint32_t flags); /* HGTX_MEM_PERMISSIONS_REGION_FLAGS_* */

/** \brief Mark memory ranges as initialized.
*
* The heap refers the the heap within which the region resides.
* This can be from hgtxMemHeapRegister, HGTX_MEM_HEAP_HANDLE_PROCESS_WIDE, or one provided from other extension API.
*
* The regionType arg will define which type is used in regionDescArray.
* The most commonly used type is HGTX_MEM_TYPE_VIRTUAL_ADDRESS.
*
* The regionCount arg is how many element are in regionDescArray and regionHandleArrayOut.
*
* The regionHandleArrayOut arg points to an array where the tool will provide region handles.
* If a pointer if provided, it is expected to have regionCount elements.
* This pointer can be NULL if regionType is HGTX_MEM_TYPE_VIRTUAL_ADDRESS.  In this case,
* the user can use the pointer to the virtual memory to reference the region in other
* related functions which accept a hgtxMemRegionRef_t.
*/
typedef struct hgtxMemMarkInitializedBatch_v1
{
    uint16_t extCompatID; /* Set to HGTX_EXT_COMPATID_MEM */
    uint16_t structSize; /* Size of the structure. */

    uint32_t regionType; /* HGTX_MEM_TYPE_* */

    size_t regionDescCount;
    size_t regionDescElementSize;
    void const* regionDescElements; /* this will also become the handle for this region */

} hgtxMemMarkInitializedBatch_v1;
typedef hgtxMemMarkInitializedBatch_v1 hgtxMemMarkInitializedBatch_t;

/** \brief Register a region of memory inside of a heap of linear process virtual memory
*
* stream is the HGGC stream where the range was accessed and initialized.
*/
HGTX_DECLSPEC void HGTX_API hgtxMemHggcMarkInitialized(
    hgtxDomainHandle_t domain,
    hggcStream_t stream,
    uint8_t isPerThreadStream, /* 0 for false, otherwise true */
    hgtxMemMarkInitializedBatch_t const* desc);

/** @} */

#endif /* HGTX_MEM_HGGCRT_CONTENTS_V1 */

#ifdef __GNUC__
#pragma GCC visibility push(internal)
#endif

#ifndef HGTX_NO_IMPL
#define HGTX_EXT_IMPL_MEM_HGGCRT_GUARD /* Ensure other headers cannot be included directly */
#include "hgtxDetail/hgtxExtImplMemHggcRt_v1.h"
#undef HGTX_EXT_IMPL_MEM_HGGCRT_GUARD
#endif /*HGTX_NO_IMPL*/

#ifdef __GNUC__
#pragma GCC visibility pop
#endif


#ifdef __cplusplus
}
#endif /* __cplusplus */
