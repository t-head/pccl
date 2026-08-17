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

#include "hgToolsExt.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifndef HGTX_EXT_MODULEID_MEM
#define HGTX_EXT_MODULEID_MEM 1
#endif

/* \cond SHOW_HIDDEN
 * \brief A compatibility ID value used in structures and initialization to
 * identify version differences.
 */
#ifndef HGTX_EXT_COMPATID_MEM
#define HGTX_EXT_COMPATID_MEM 0x0102
#endif
/* \endcond
 */

#ifndef HGTX_MEM_CONTENTS_V1
#define HGTX_MEM_CONTENTS_V1

/* \cond SHOW_HIDDEN
 * \brief This value is returned by functions that return `hgtxMemHeapHandle_t`,
 * if a tool is not attached.
 */
#define HGTX_MEM_HEAP_HANDLE_NO_TOOL (HGTX_STATIC_CAST(hgtxMemHeapHandle_t, HGTX_STATIC_CAST(intptr_t, -1)))
/* \endcond
 */

/* \cond SHOW_HIDDEN
 * \brief This value is returned by functions that return `hgtxMemRegionHandle_t`
 * if a tool is not attached.
 */
#define HGTX_MEM_REGION_HANDLE_NO_TOOL (HGTX_STATIC_CAST(hgtxMemRegionHandle_t, HGTX_STATIC_CAST(intptr_t, -1)))
/* \endcond
 */

/* \cond SHOW_HIDDEN
 * \brief This value is returned by functions that return `hgtxMemPermissionsHandle_t`
 * if a tool is not attached.
 */
#define HGTX_MEM_PERMISSIONS_HANDLE_NO_TOOL (HGTX_STATIC_CAST(hgtxMemPermissionsHandle_t, -1))
/* \endcond
 */


/* \cond SHOW_HIDDEN
 * \brief This should not be used and is considered an error but defined to
 * detect an accidental use of zero or NULL.
 */
#define HGTX_MEM_HEAP_USAGE_UNKNOWN 0x0
/* \endcond
 */


/* \cond SHOW_HIDDEN
 * \brief This should not be used and is considered an error but defined to
 * detect an accidental use of zero or NULL.
 */
#define HGTX_MEM_TYPE_UNKNOWN 0x0
/* \endcond
 */


/*  ------------------------------------------------------------------------- */
/** \defgroup MEMORY Memory
 * See page \ref PAGE_MEMORY.
 * @{
 */

/**
 * \brief To indicate the full process virtual address space as a heap for
 * functions where a hgtxMemHeapHandle_t is accepted.
 *
 * The heap by default is always read-write-execute permissions without creating regions.
 * Regions created in this heap have read-write access by default but not execute.
 */
#define HGTX_MEM_HEAP_HANDLE_PROCESS_WIDE (HGTX_STATIC_CAST(hgtxMemHeapHandle_t, 0))

/** \brief This heap is a sub-allocator.
 *
 * Heap created with this usage should not be accessed by the user until regions are registered.
 * Regions from a heap with this usage have read-write access by default but not execute.
 */
#define HGTX_MEM_HEAP_USAGE_TYPE_SUB_ALLOCATOR 0x1

/**
 * \brief This is a heap of memory that has an explicit layout.
 *
 * The layout could be static or dynamic (calculated). This often represents an algorithm's
 * structures that are packed together. By default this heap is assumed to be accessible for
 * scopes where the memory is naturally accessible by hardware. Regions may be use to further
 * annotate or restrict access. A tool may have an option to be more strict, but special
 * consideration must be made for `HGTX_MEM_HEAP_HANDLE_PROCESS_WIDE`.
 *
 * The behavior of this usage is similar to HGTX_MEM_HEAP_HANDLE_PROCESS_WIDE but
 * a tool can use it to track special behaviors and reservation.
 *
 * Memory in a heap with this usage has read-write permissions by default but not execute without
 * creating regions. Regions created in this heap have the same default permission access.
 */
#define HGTX_MEM_HEAP_USAGE_TYPE_LAYOUT 0x2


/**
 * \brief Standard process userspace virtual addresses for linear allocations.
 *
 * APIs that map into this space, such as HGGC UVA should use this type.
 *
 * Relevant functions: hggcMalloc, hggcMallocManaged, hggcHostAlloc, hggcMallocHost
 * HGTX_MEM_HEAP_HANDLE_PROCESS_WIDE is supported
 *
 * hgtxMemHeapRegister receives a heapDesc of type hgtxMemVirtualRangeDesc_t
 */
#define HGTX_MEM_TYPE_VIRTUAL_ADDRESS 0x1


/**
 * \brief To indicate you are modifying permissions to the process-wide
 * full virtual address space.
 *
 * This is a companion object to `HGTX_MEM_HEAP_HANDLE_PROCESS_WIDE`.
 */
#define HGTX_MEM_PERMISSIONS_HANDLE_PROCESS_WIDE (HGTX_STATIC_CAST(hgtxMemPermissionsHandle_t, 0))

#define HGTX_MEM_PERMISSIONS_CREATE_FLAGS_NONE 0x0
#define HGTX_MEM_PERMISSIONS_CREATE_FLAGS_EXCLUDE_GLOBAL_READ 0x1
#define HGTX_MEM_PERMISSIONS_CREATE_FLAGS_EXCLUDE_GLOBAL_WRITE 0x2
#define HGTX_MEM_PERMISSIONS_CREATE_FLAGS_EXCLUDE_GLOBAL_ATOMIC 0x4


/* \cond SHOW_HIDDEN
 * \brief Forward declaration of opaque memory heap structure.
 */
struct hgtxMemHeap_v1;
typedef struct hgtxMemHeap_v1 hgtxMemHeap_t;
/* \endcond
 */

/** \brief A handle returned by a tool to represent a memory heap. */
typedef hgtxMemHeap_t* hgtxMemHeapHandle_t;

/* \cond SHOW_HIDDEN
 * \brief Forward declaration of opaque memory heap structure.
 */
struct hgtxMemRegion_v1;
typedef struct hgtxMemRegion_v1 hgtxMemRegion_t;
/* \endcond
 */

/** \brief A handle returned by a tool to represent a memory region. */
typedef hgtxMemRegion_t* hgtxMemRegionHandle_t;

/** \brief A reference to a memory region (by pointer or handle).
 * Which member of the union will be determined by a type or flag field outside.
 */
typedef union hgtxMemRegionRef_t
{
    void const* pointer;
    hgtxMemRegionHandle_t handle;
} hgtxMemRegionRef_t;

/* \cond SHOW_HIDDEN
 * \brief Forward declaration of opaque memory permissions structure
 */
struct hgtxMemPermissions_v1;
typedef struct hgtxMemPermissions_v1 hgtxMemPermissions_t;
/* \endcond
 */

/** \brief A handle returned by a tool to represent a memory permissions mask. */
typedef hgtxMemPermissions_t* hgtxMemPermissionsHandle_t;


typedef struct hgtxMemVirtualRangeDesc_v1
{
    size_t  size;
    void const*  ptr;
} hgtxMemVirtualRangeDesc_v1 ;
typedef hgtxMemVirtualRangeDesc_v1 hgtxMemVirtualRangeDesc_t;


/** \brief structure to describe a heap in process virtual memory. */
typedef struct hgtxMemHeapDesc_v1
{
    uint16_t extCompatID; /* Set to HGTX_EXT_COMPATID_MEM */
    uint16_t structSize; /* Size of the structure. */
    uint32_t reserved0;

    /** \brief Usage characteristics of the heap
     *
     * Usage characteristics help tools like memcheckers, sanitizers,
     * as well as other debugging and profiling tools to determine some
     * special behaviors they should apply to the heap and its regions.
     * The value follows the convention HGTX_MEM_HEAP_USAGE_*
     *
     * Default Value is 0, which is invalid.
     */
    uint32_t usage;

    /** \brief Memory type characteristics of the heap
     *
     * The 'type' indicates how to interpret the ptr field of the heapDesc.
     * This is intended to support many additional types of memory, beyond
     * standard process virtual memory, such as API specific memory only
     * addressed by handles or multi-dimensional memory requiring more complex
     * descriptions to handle features like strides, tiling, or interlace.
     *
     * The values conforms to HGTX_MEM_TYPE_*
     *
     * The value in the field 'type' identifies the descriptor type that will
     * be in the field 'typeSpecificDesc'.  'typeSpecificDesc' is void* because
     * it is extensible.  Example usage is if type is HGTX_MEM_TYPE_VIRTUAL_ADDRESS,
     * then typeSpecificDesc points to a hgtxMemVirtualRangeDesc_t.
     *
     * Default Value is 0, which is invalid.
     */
    uint32_t type;

    /** \brief size of the heap memory descriptor pointed to by typeSpecificDesc
     *
     * Default Value is 0 which is invalid.
     */
    size_t typeSpecificDescSize;

    /** \brief Pointer to the heap memory descriptor
     *
     * The value in the field 'type' identifies the descriptor type that will
     * be in the field 'typeSpecificDesc'.  'typeSpecificDesc' is void* because
     * it is extensible.  Example usage is if type is HGTX_MEM_TYPE_VIRTUAL_ADDRESS,
     * then typeSpecificDesc points to a hgtxMemVirtualRangeDesc_t.
     *
     * Default Value is 0, which is invalid.
     */
    void const* typeSpecificDesc;

    /** \brief ID of the category the event is assigned to.
     *
     * A category is a user-controlled ID that can be used to group
     * events.  The tool may use category IDs to improve filtering or
     * enable grouping of events in the same category. The functions
     * \ref ::hgtxNameCategoryA or \ref ::hgtxNameCategoryW can be used
     * to name a category.
     *
     * Default Value is 0.
     */
    uint32_t category;

    /** \brief Message type specified in this attribute structure.
     *
     * Defines the message format of the attribute structure's \ref MEM_MESSAGE_FIELD
     * "message" field.
     *
     * Default Value is `HGTX_MESSAGE_UNKNOWN`.
     */
    uint32_t messageType;            /* hgtxMessageType_t */

    /** \brief Message assigned to this attribute structure. \anchor MEM_MESSAGE_FIELD
     *
     * The text message that is attached to an event.
     */
    hgtxMessageValue_t message;

} hgtxMemHeapDesc_v1 ;
typedef hgtxMemHeapDesc_v1 hgtxMemHeapDesc_t;

/**
 * \brief Create a memory heap to represent a object or range of memory that will be further
 * sub-divided into regions.
 *
 * The handle used to address the heap will depend on the heap's type.  Where the heap is virtual
 * memory accessible, the address of the heap's memory itself is its handle. This will likewise
 * be returned from the function.
 *
 * For more advanced types, where the heap is not virtual memory accessible the tools may be
 * responsible for returning a void const * that that uniquely identifies the object. Please see
 * the description of each heap type for more details on whether this is expected to be a uniquely
 * generated by the tool or otherwise.
 */
HGTX_DECLSPEC hgtxMemHeapHandle_t HGTX_API hgtxMemHeapRegister(
    hgtxDomainHandle_t domain,
    hgtxMemHeapDesc_t const* desc);

 /** \brief Destroy a memory heap. */
HGTX_DECLSPEC void HGTX_API hgtxMemHeapUnregister(
    hgtxDomainHandle_t domain,
    hgtxMemHeapHandle_t heap);/* HGTX_MEM_HEAP_HANDLE_PROCESS_WIDE is not supported */

/**
 * \brief Reset the memory heap wipes out any changes, as if it were a fresh heap.
 *
 * This includes invalidating all regions and their handles.
 */
HGTX_DECLSPEC void HGTX_API hgtxMemHeapReset(
    hgtxDomainHandle_t domain,
    hgtxMemHeapHandle_t heap); /* HGTX_MEM_HEAP_HANDLE_PROCESS_WIDE is supported */

/**
 * \brief Register a region of memory inside of a heap.
 *
 * The heap refers the the heap within which the region resides. This can be from
 * `hgtxMemHeapRegister`, `HGTX_MEM_HEAP_HANDLE_PROCESS_WIDE`, or one provided
 * from other extension API.
 *
 * The regionType arg will define which type is used in regionDescArray.
 * The most commonly used type is `HGTX_MEM_TYPE_VIRTUAL_ADDRESS`.
 * In this case regionDescElements is an array of `hgtxMemVirtualRangeDesc_t`.
 *
 * The regionCount arg is how many element are in regionDescArray and regionHandleArrayOut.
 *
 * The regionHandleArrayOut arg points to an array where the tool will provide region handles. If
 * a pointer is provided, it is expected to have regionCount elements. This pointer can be NULL if
 * regionType is HGTX_MEM_TYPE_VIRTUAL_ADDRESS. In this case, the user can use the pointer to the
 * virtual memory to reference the region in other related functions which accept hgtxMemRegionRef_t.
 */
typedef struct hgtxMemRegionsRegisterBatch_v1
{
    uint16_t extCompatID; /* Set to HGTX_EXT_COMPATID_MEM */
    uint16_t structSize; /* Size of the structure. */

    uint32_t regionType; /* HGTX_MEM_TYPE_* */

    hgtxMemHeapHandle_t heap;

    size_t regionCount;
    size_t regionDescElementSize;
    void const* regionDescElements; /* This will also become the handle for this region. */
    hgtxMemRegionHandle_t* regionHandleElementsOut; /* This will also become the handle for this region. */

} hgtxMemRegionsRegisterBatch_v1;
typedef hgtxMemRegionsRegisterBatch_v1 hgtxMemRegionsRegisterBatch_t;

 /** \brief Register a region of memory inside of a heap of linear process virtual memory
 */
HGTX_DECLSPEC void HGTX_API hgtxMemRegionsRegister(
    hgtxDomainHandle_t domain,
    hgtxMemRegionsRegisterBatch_t const* desc);



/**
 * \brief Register a region of memory inside of a heap.
 *
 * The heap refers the the heap within which the region resides.
 * This can be from hgtxMemHeapRegister, HGTX_MEM_HEAP_HANDLE_PROCESS_WIDE, or
 * one provided from other extension API.
 *
 * The regionType arg will define which type is used in regionDescArray.
 * The most commonly used type is HGTX_MEM_TYPE_VIRTUAL_ADDRESS.
 *
 * The regionCount arg is how many element are in regionDescArray and regionHandleArrayOut.
 *
 * The regionHandleArrayOut arg points to an array where the tool will provide region handles. If
 * a pointer if provided, it is expected to have regionCount elements. This pointer can be NULL if
 * regionType is HGTX_MEM_TYPE_VIRTUAL_ADDRESS. In this case, the user can use the pointer to the
 * virtual memory to reference the region in other related functions which accept hgtxMemRegionRef_t.
 */
typedef struct hgtxMemRegionsResizeBatch_v1
{
    uint16_t extCompatID; /* Set to HGTX_EXT_COMPATID_MEM */
    uint16_t structSize; /* Size of the structure. */

    uint32_t regionType; /* HGTX_MEM_TYPE_* */

    size_t regionDescCount;
    size_t regionDescElementSize;
    void const* regionDescElements; /* This will also become the handle for this region. */

} hgtxMemRegionsResizeBatch_v1;
typedef hgtxMemRegionsResizeBatch_v1 hgtxMemRegionsResizeBatch_t;

 /** \brief Register a region of memory inside of a heap of linear process virtual memory
 */
HGTX_DECLSPEC void HGTX_API hgtxMemRegionsResize(
    hgtxDomainHandle_t domain,
    hgtxMemRegionsResizeBatch_t const* desc);


#define HGTX_MEM_REGION_REF_TYPE_UNKNOWN 0x0
#define HGTX_MEM_REGION_REF_TYPE_POINTER 0x1
#define HGTX_MEM_REGION_REF_TYPE_HANDLE 0x2

/**
 * \brief Register a region of memory inside of a heap.
 *
 * The heap refers the the heap within which the region resides.
 * This can be from hgtxMemHeapRegister, `HGTX_MEM_HEAP_HANDLE_PROCESS_WIDE`, or
 * one provided from other extension API.
 *
 * The regionType arg will define which type is used in `regionDescArray`.
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
typedef struct hgtxMemRegionsUnregisterBatch_v1
{
    uint16_t extCompatID; /* Set to HGTX_EXT_COMPATID_MEM */
    uint16_t structSize; /* Size of the structure. */

    uint32_t refType; /* HGTX_MEM_REGION_REF_TYPE_* */

    size_t refCount; /* count of elements in refArray */
    size_t refElementSize;
    hgtxMemRegionRef_t const* refElements; /* This will also become the handle for this region. */

} hgtxMemRegionsUnregisterBatch_v1;
typedef hgtxMemRegionsUnregisterBatch_v1 hgtxMemRegionsUnregisterBatch_t;

/**
 * \brief Unregistration for regions of process virtual memory
 *
 * This is not necessary if the hgtx heap destroy function has been called that
 * contains this object.
 */
HGTX_DECLSPEC void HGTX_API hgtxMemRegionsUnregister(
    hgtxDomainHandle_t domain,
    hgtxMemRegionsUnregisterBatch_t const* desc);

typedef struct hgtxMemRegionNameDesc_v1
{
    uint32_t regionRefType; /* HGTX_MEM_REGION_REF_TYPE_* */
    uint32_t nameType; /* hgtxMessageType_t */

    hgtxMemRegionRef_t region;
    hgtxMessageValue_t name;

    uint32_t category;
    uint32_t reserved0;
} hgtxMemRegionNameDesc_v1;
typedef hgtxMemRegionNameDesc_v1 hgtxMemRegionNameDesc_t;


typedef struct hgtxMemRegionsNameBatch_v1
{
    uint16_t extCompatID; /* Set to HGTX_EXT_COMPATID_MEM */
    uint16_t structSize; /* Size of the structure. */

    uint32_t reserved0;

    size_t regionCount;
    size_t regionElementSize;
    hgtxMemRegionNameDesc_t const* regionElements;
    size_t reserved1;
} hgtxMemRegionsNameBatch_v1 ;
typedef hgtxMemRegionsNameBatch_v1 hgtxMemRegionsNameBatch_t;


 /** \brief Name or rename a region. */
HGTX_DECLSPEC void HGTX_API hgtxMemRegionsName(
    hgtxDomainHandle_t domain,
    hgtxMemRegionsNameBatch_t const* desc);

/** \brief There are no permissions for this memory. */
#define HGTX_MEM_PERMISSIONS_REGION_FLAGS_NONE 0x0

/** \brief The memory is readable. */
#define HGTX_MEM_PERMISSIONS_REGION_FLAGS_READ 0x1

/** \brief The memory is writable. */
#define HGTX_MEM_PERMISSIONS_REGION_FLAGS_WRITE 0x2

/** \brief The memory is for atomic RW. */
#define HGTX_MEM_PERMISSIONS_REGION_FLAGS_ATOMIC 0x4

/**
 * \brief The memory access permissions are reset for a region.
 *
 * This is as if never set, rather than documented defaults.  As as result any flags
 * indicating how unspecified regions are handle will affect this area.
 *
 * This should not be used with READ, WRITE, nor ATOMIC, as those flags would have no effect.
 */
#define HGTX_MEM_PERMISSIONS_REGION_FLAGS_RESET 0x8


typedef struct hgtxMemPermissionsAssignRegionDesc_v1
{
    uint32_t flags; /* HGTX_MEM_PERMISSIONS_REGION_FLAGS_* */
    uint32_t regionRefType; /* HGTX_MEM_REGION_REF_TYPE_* */
    hgtxMemRegionRef_t region;

} hgtxMemPermissionsAssignRegionDesc_v1 ;
typedef hgtxMemPermissionsAssignRegionDesc_v1 hgtxMemPermissionsAssignRegionDesc_t;


typedef struct hgtxMemPermissionsAssignBatch_v1
{
    uint16_t extCompatID; /* Set to HGTX_EXT_COMPATID_MEM */
    uint16_t structSize; /* Size of the structure. */

    uint32_t reserved0;

    hgtxMemPermissionsHandle_t permissions;

    size_t regionCount;
    size_t regionElementSize;
    hgtxMemPermissionsAssignRegionDesc_t const* regionElements;

    size_t reserved1;
} hgtxMemPermissionsAssignBatch_v1 ;
typedef hgtxMemPermissionsAssignBatch_v1 hgtxMemPermissionsAssignBatch_t;


 /** \brief Change the permissions of a region of process virtual memory. */
HGTX_DECLSPEC void HGTX_API hgtxMemPermissionsAssign(
    hgtxDomainHandle_t domain,
    hgtxMemPermissionsAssignBatch_t const* desc);


/**
 * \brief Create a permissions object for fine grain thread-local control in
 * multi-threading scenarios
 *
 * Unlike the global permissions object (HGTX_MEM_PERMISSIONS_HANDLE_PROCESS_WIDE), a new
 * permissions object is empty. There are no regions registered to it, so more memory is accessible
 * if bound(bind) without calls to hgtxMemPermissionsSetAccess* first. The permissions are not
 * active until hgtxMemPermissionsBind. See `hgtxMemPermissionsBind` for more details.
 *
 * Use the flags HGTX_MEM_PERMISSIONS_CREATE_FLAGS_EXCLUDE_GLOBAL_* to control  how the regions in
 * this permission object will interact with global permissions when bound. You may choose to
 * either replace global memory regions setting or overlay on top of them. The most common uses are
 * as follows:
 *     * To limit tools to validate writing exclusively specified in this object but inherit all
 *       global read access regions use `HGTX_MEM_PERMISSIONS_CREATE_FLAGS_EXCLUDE_GLOBAL_WRITE`
 *     * To limit tools to validate both read & write permissions exclusively specified in this
 *        object use HGTX_MEM_PERMISSIONS_CREATE_FLAGS_EXCLUDE_GLOBAL_READ
 *                   & HGTX_MEM_PERMISSIONS_CREATE_FLAGS_EXCLUDE_GLOBAL_WRITE
 *
 * Also see `hgtxMemPermissionsBind` & `hgtxMemPermissionsSetAccess*`.
 */
HGTX_DECLSPEC hgtxMemPermissionsHandle_t HGTX_API hgtxMemPermissionsCreate(
    hgtxDomainHandle_t domain,
    int32_t creationflags); /* HGTX_MEM_PERMISSIONS_CREATE_FLAGS_* */

/**
 * \brief Destroy the permissions object.
 *
 * If bound(bind), destroy will also unbind it.
 */
HGTX_DECLSPEC void HGTX_API hgtxMemPermissionsDestroy(
    hgtxDomainHandle_t domain,
    hgtxMemPermissionsHandle_t permissionsHandle); /* only supported on objects from hgtxMemPermissionsCreate */

/** \brief Reset the permissions object back to its created state. */
HGTX_DECLSPEC void HGTX_API hgtxMemPermissionsReset(
    hgtxDomainHandle_t domain,
    hgtxMemPermissionsHandle_t permissionsHandle);
/* HGTX_MEM_PERMISSIONS_HANDLE_PROCESS_WIDE and other special handles are supported */


#define HGTX_MEM_PERMISSIONS_BIND_FLAGS_NONE 0x0

 /** \brief Upon binding, with the thread, exclude parent scope write regions instead of overlaying on top of them.
  *
   * EX A developer may chose to first prevent all writes except the ones specified to avoid
  * OOB writes, since there are typically fewer regions written to than read from.
 **/
#define HGTX_MEM_PERMISSIONS_BIND_FLAGS_STRICT_WRITE 0x2

 /** \brief Upon binding, with the thread, exclude parent scope read regions instead of overlaying on top of them.
  *
  * EX After eliminating any errors when applying strict writes, a developer may then choose to
  * annotate and enforce strict reads behaviors in segments of code.
 **/
#define HGTX_MEM_PERMISSIONS_BIND_FLAGS_STRICT_READ 0x1

 /** \brief Upon binding, with the thread, exclude parent scope atomic RW regions instead of overlaying on top of them.
  *
  * EX After eliminating any errors from read and write, a developer may chose to ensure
  * that atomics are in their own region, removing standard read/write, and replacing with
  * this strict atomic only access.  This way they know that conventional reads or writes
  * will not cause unexpected issues.
 **/
#define HGTX_MEM_PERMISSIONS_BIND_FLAGS_STRICT_ATOMIC 0x4


#define HGTX_MEM_PERMISSIONS_BIND_SCOPE_UNKNOWN 0x0

 /** \brief Bind to thread scope.  In this case, tools should validate that local thread's
  * execution is honoring the permissions as well as the state of HGTX_MEM_PERMISSIONS_HANDLE_PROCESS_WIDE
  * at the time of binding.  If this is not bound then HGTX_MEM_PERMISSIONS_HANDLE_PROCESS_WIDE should be
  * used to validate the memory.
  *
  * Not all tools will support every scope, such a GPU sanitizer.
 **/
#define HGTX_MEM_PERMISSIONS_BIND_SCOPE_CPU_THREAD 0x1

/**
 * \brief Bind to HGGC stream scope.
 *
 * In this case, work enqueued to a HGGC stream should be validated by the tool,
 * when it executes, that it respect the permission of the permission at the point
 * of binding, as well as the appropriate hgtxMemHggcGetDevicePermissions at the
 * time of binding. If this is not bound then hgtxMemHggcGetDevicePermissions at
 * the time of stream enqueue should be used to validate the memory.
 *
 * This could apply to work done either on the GPU like a kernel launch or to
 * CPU based callbacks like hggcStreamAddCallback if the tools supports it.
 *
 * Binding is applies locally to a CPU thread so that if N CPU threads are enqueuing
 * work to the same stream (like the default stream) that there cannot be a race
 * condition between thread binding vs launching their work. IE users should
 * expect the permissions bound in the thread to be honored by the proceeding
 * work (launches, copies, etc) invoked from in the CPU thread until unbound.
 */
#define HGTX_MEM_PERMISSIONS_BIND_SCOPE_HGGC_STREAM 0x2


/**
 * \brief Bind the permissions object into a particular scope on the caller thread
 *
 * Permissions do not take affect until binding. Binding permissions is a thread local
 * activity that overrides global behaviors.  This is to avoid multi-threaded race conditions,
 *
 * The scope dictates what type of processing it applies to, and when in some cases.
 * EX1: HGTX_MEM_PERMISSIONS_BIND_SCOPE_CPU_THREAD applies to CPU code accessing memory while bound.
 * EX2: HGTX_MEM_PERMISSIONS_BIND_SCOPE_HGGC_STREAM applies to HGGC streams, and the permissions
 * must be recorded and applied when the work in the stream dequeues to executes.  In this case
 * it could be GPU or CPU, if the tool support both.
 *
 * Bind can be called again on the same object and thread to take any updates to the
 * specified permission object or the inherited properties.
 *
 * Bind flags support changing how the binding process inherits region access control.
 * In the case of thread scope this is HGTX_MEM_PERMISSIONS_HANDLE_PROCESS_WIDE and from HGGC_STREAM
 * this is hgtxMemHggcGetDevicePermissions.  Choosing stricter modes allows the user to
 * further reduce the access with less work, since memory by default, behaves as natural
 * until the HGTX annotations instructs a tool to treat it anther way.  See strict flags
 * for more details.
 *
 * Also see hgtxMemPermissionsUnbind
 */
HGTX_DECLSPEC void HGTX_API hgtxMemPermissionsBind(
    hgtxDomainHandle_t domain,
    hgtxMemPermissionsHandle_t permissions, /* special object like HGTX_MEM_PERMISSIONS_HANDLE_PROCESS_WIDE are not supported */
    uint32_t bindScope, /* HGTX_MEM_PERMISSIONS_BIND_SCOPE_* */
    uint32_t bindFlags); /* HGTX_MEM_PERMISSIONS_BIND_FLAGS_* */

/**
 * \brief Unbind the permissions object bound to the caller thread.
 *
 * Upon unbind, the thread local permissions for a scope are restored to the default
 * behavior defined by the scope.
 */
HGTX_DECLSPEC void HGTX_API hgtxMemPermissionsUnbind(
    hgtxDomainHandle_t domain,
    uint32_t bindScope);

/** @} */

#endif /* HGTX_MEM_CONTENTS_V1 */

#ifndef HGTX_MEM_CALLBACK_ID_V1
#define HGTX_MEM_CALLBACK_ID_V1

#define HGTX3EXT_CBID_hgtxMemHeapRegister                  0
#define HGTX3EXT_CBID_hgtxMemHeapUnregister                1
#define HGTX3EXT_CBID_hgtxMemHeapReset                     2
#define HGTX3EXT_CBID_hgtxMemRegionsRegister               3
#define HGTX3EXT_CBID_hgtxMemRegionsResize                 4
#define HGTX3EXT_CBID_hgtxMemRegionsUnregister             5
#define HGTX3EXT_CBID_hgtxMemRegionsName                   6
#define HGTX3EXT_CBID_hgtxMemPermissionsAssign             7
#define HGTX3EXT_CBID_hgtxMemPermissionsCreate             8
#define HGTX3EXT_CBID_hgtxMemPermissionsDestroy            9
#define HGTX3EXT_CBID_hgtxMemPermissionsReset              10
#define HGTX3EXT_CBID_hgtxMemPermissionsBind               11
#define HGTX3EXT_CBID_hgtxMemPermissionsUnbind             12

/* 13-16 in hgtxExtImplMemHggcRt_v1.h */
#define HGTX3EXT_CBID_hgtxMemHggcGetProcessWidePermissions 13
#define HGTX3EXT_CBID_hgtxMemHggcGetDeviceWidePermissions  14
#define HGTX3EXT_CBID_hgtxMemHggcSetPeerAccess             15
#define HGTX3EXT_CBID_hgtxMemHggcMarkInitialized           16

#endif /* HGTX_MEM_CALLBACK_ID_V1 */

/* Macros to create versioned symbols. */
#ifndef HGTX_EXT_MEM_VERSIONED_IDENTIFIERS_V1
#define HGTX_EXT_MEM_VERSIONED_IDENTIFIERS_V1
#define HGTX_EXT_MEM_VERSIONED_IDENTIFIER_L3(NAME, VERSION, COMPATID) \
    NAME##_v##VERSION##_mem##COMPATID
#define HGTX_EXT_MEM_VERSIONED_IDENTIFIER_L2(NAME, VERSION, COMPATID) \
    HGTX_EXT_MEM_VERSIONED_IDENTIFIER_L3(NAME, VERSION, COMPATID)
#define HGTX_EXT_MEM_VERSIONED_ID(NAME) \
    HGTX_EXT_MEM_VERSIONED_IDENTIFIER_L2(NAME, HGTX_VERSION, HGTX_EXT_COMPATID_MEM)
#endif /* HGTX_EXT_MEM_VERSIONED_IDENTIFIERS_V1 */

#ifdef __GNUC__
#pragma GCC visibility push(internal)
#endif

/* Extension types are required for the implementation and the HGTX handler. */
#define HGTX_EXT_TYPES_GUARD /* Ensure other headers cannot be included directly */
#include "hgtxDetail/hgtxExtTypes.h"
#undef HGTX_EXT_TYPES_GUARD

#ifndef HGTX_NO_IMPL
/* Ensure other headers cannot be included directly */
#define HGTX_EXT_IMPL_MEM_GUARD
#include "hgtxDetail/hgtxExtImplMem_v1.h"
#undef HGTX_EXT_IMPL_MEM_GUARD
#endif /*HGTX_NO_IMPL*/

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */
