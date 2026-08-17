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

#ifndef HGTOOLSEXT_SYNC_V3
#define HGTOOLSEXT_SYNC_V3

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* \cond SHOW_HIDDEN
* \version HGTX_VERSION_2
*/
#define HGTX_SYNCUSER_ATTRIB_STRUCT_SIZE (HGTX_STATIC_CAST(uint16_t, sizeof(hgtxSyncUserAttributes_v0)))
/** \endcond */


/**
* \page PAGE_SYNCHRONIZATION Synchronization
*
* This section covers a subset of the API that allow users to track additional
* synchronization details of their application.   Naming OS synchronization primitives
* may allow users to better understand the data collected by traced synchronization
* APIs.  Additionally, a user defined synchronization object can allow the users to
* to tell the tools when the user is building their own synchronization system
* that do not rely on the OS to provide behaviors and instead use techniques like
* atomic operations and spinlocks.
*
* See module \ref SYNCHRONIZATION for details.
*
* \par Example
* Instrument a mutex class:
* \code
* class MyMutex
* {
*     volatile long bLocked;
*     hgtxSyncUser_t hSync;
* public:
*     MyMutex(const char* name, hgtxDomainHandle_t d) {
*          bLocked = 0;
*
*          hgtxSyncUserAttributes_t attribs = { 0 };
*          attribs.version = HGTX_VERSION;
*          attribs.size = HGTX_SYNCUSER_ATTRIB_STRUCT_SIZE;
*          attribs.messageType = HGTX_MESSAGE_TYPE_ASCII;
*          attribs.message.ascii = name;
*          hSync = hgtxDomainSyncUserCreate(d, &attribs);
*     }
*
*     ~MyMutex() {
*          hgtxDomainSyncUserDestroy(hSync);
*     }
*
*     bool Lock() {
*          hgtxDomainSyncUserAcquireStart(hSync);
*          bool acquired = __sync_bool_compare_and_swap(&bLocked, 0, 1); // atomic compiler intrinsic
*
*          if (acquired) {
*              hgtxDomainSyncUserAcquireSuccess(hSync);
*          }
*          else {
*              hgtxDomainSyncUserAcquireFailed(hSync);
*          }
*          return acquired;
*     }
*
*     void Unlock() {
*          hgtxDomainSyncUserReleasing(hSync);
*          bLocked = false;
*     }
* };
* \endcode
*
* \version HGTX_VERSION_2
*/

/*  ------------------------------------------------------------------------- */
/* \cond SHOW_HIDDEN
* \brief Used to build a non-colliding value for resource types separated class
* \version HGTX_VERSION_2
*/
#define HGTX_RESOURCE_CLASS_SYNC_OS 2 /**< Synchronization objects that are OS specific. */
#define HGTX_RESOURCE_CLASS_SYNC_PTHREAD 3 /**< Synchronization objects that are from the POSIX Threads API (pthread)*/
/** \endcond */


/*  ------------------------------------------------------------------------- */
/** \defgroup SYNCHRONIZATION Synchronization
* See page \ref PAGE_SYNCHRONIZATION.
* @{
*/

/** \brief Resource type values for OSs with POSIX Thread API support
 */
typedef enum hgtxResourceSyncPosixThreadType_t
{
    HGTX_RESOURCE_TYPE_SYNC_PTHREAD_MUTEX = HGTX_RESOURCE_MAKE_TYPE(SYNC_PTHREAD, 1), /* pthread_mutex_t  */
    HGTX_RESOURCE_TYPE_SYNC_PTHREAD_CONDITION = HGTX_RESOURCE_MAKE_TYPE(SYNC_PTHREAD, 2), /* pthread_cond_t  */
    HGTX_RESOURCE_TYPE_SYNC_PTHREAD_RWLOCK = HGTX_RESOURCE_MAKE_TYPE(SYNC_PTHREAD, 3), /* pthread_rwlock_t  */
    HGTX_RESOURCE_TYPE_SYNC_PTHREAD_BARRIER = HGTX_RESOURCE_MAKE_TYPE(SYNC_PTHREAD, 4), /* pthread_barrier_t  */
    HGTX_RESOURCE_TYPE_SYNC_PTHREAD_SPINLOCK = HGTX_RESOURCE_MAKE_TYPE(SYNC_PTHREAD, 5), /* pthread_spinlock_t  */
    HGTX_RESOURCE_TYPE_SYNC_PTHREAD_ONCE = HGTX_RESOURCE_MAKE_TYPE(SYNC_PTHREAD, 6) /* pthread_once_t  */
} hgtxResourceSyncPosixThreadType_t;

/** \brief Resource type values for Windows OSs
*/
typedef enum hgtxResourceSyncWindowsType_t
{
    HGTX_RESOURCE_TYPE_SYNC_WINDOWS_MUTEX = HGTX_RESOURCE_MAKE_TYPE(SYNC_OS, 1),
    HGTX_RESOURCE_TYPE_SYNC_WINDOWS_SEMAPHORE = HGTX_RESOURCE_MAKE_TYPE(SYNC_OS, 2),
    HGTX_RESOURCE_TYPE_SYNC_WINDOWS_EVENT = HGTX_RESOURCE_MAKE_TYPE(SYNC_OS, 3),
    HGTX_RESOURCE_TYPE_SYNC_WINDOWS_CRITICAL_SECTION = HGTX_RESOURCE_MAKE_TYPE(SYNC_OS, 4),
    HGTX_RESOURCE_TYPE_SYNC_WINDOWS_SRWLOCK = HGTX_RESOURCE_MAKE_TYPE(SYNC_OS, 5)
} hgtxResourceSyncWindowsType_t;

/** \brief Resource type values for Linux and Linux derived OSs such as Android
* \sa
* ::hgtxResourceSyncPosixThreadType_t
*/
typedef enum hgtxResourceSyncLinuxType_t
{
    HGTX_RESOURCE_TYPE_SYNC_LINUX_MUTEX = HGTX_RESOURCE_MAKE_TYPE(SYNC_OS, 1),
    HGTX_RESOURCE_TYPE_SYNC_LINUX_FUTEX = HGTX_RESOURCE_MAKE_TYPE(SYNC_OS, 2),
    HGTX_RESOURCE_TYPE_SYNC_LINUX_SEMAPHORE = HGTX_RESOURCE_MAKE_TYPE(SYNC_OS, 3),
    HGTX_RESOURCE_TYPE_SYNC_LINUX_COMPLETION = HGTX_RESOURCE_MAKE_TYPE(SYNC_OS, 4),
    HGTX_RESOURCE_TYPE_SYNC_LINUX_SPINLOCK = HGTX_RESOURCE_MAKE_TYPE(SYNC_OS, 5),
    HGTX_RESOURCE_TYPE_SYNC_LINUX_SEQLOCK = HGTX_RESOURCE_MAKE_TYPE(SYNC_OS, 6),
    HGTX_RESOURCE_TYPE_SYNC_LINUX_RCU = HGTX_RESOURCE_MAKE_TYPE(SYNC_OS, 7)
} hgtxResourceSyncLinuxType_t;

/** \brief Resource type values for Android come from Linux.
* \sa
* ::hgtxResourceSyncLinuxType_t
* ::hgtxResourceSyncPosixThreadType_t
*/
typedef enum hgtxResourceSyncLinuxType_t hgtxResourceSyncAndroidType_t;

/** \brief User Defined Synchronization Object Handle .
* \anchor SYNCUSER_HANDLE_STRUCTURE
*
* This structure is opaque to the user and is used as a handle to reference
* a user defined synchronization object.  The tools will return a pointer through the API for the application
* to hold on its behalf to reference the string in the future.
*
*/
typedef struct hgtxSyncUser* hgtxSyncUser_t;

/** \brief User Defined Synchronization Object Attributes Structure.
* \anchor USERDEF_SYNC_ATTRIBUTES_STRUCTURE
*
* This structure is used to describe the attributes of a user defined synchronization
* object.  The layout of the structure is defined by a specific version of the tools
* extension library and can change between different versions of the Tools Extension
* library.
*
* \par Guidelines
* The caller should always perform the following three tasks when using
* attributes:
* <ul>
*    <li>Zero the structure
*    <li>Set the version field
*    <li>Set the size field
* </ul>
*
* Zeroing the structure sets all the event attributes types and values
* to the default value.
*
* The version and size field are used by the Tools Extension
* implementation to handle multiple versions of the attributes structure.
*
* It is recommended that the caller use one of the following to methods
* to initialize the event attributes structure:
*
* \par Method 1
* Initializing hgtxEventAttributes for future compatibility:
* \code
* hgtxSyncUserAttributes_t attribs = {0};
* attribs.version = HGTX_VERSION;
* attribs.size = HGTX_SYNCUSER_ATTRIB_STRUCT_SIZE;
* \endcode
*
* \par Method 2
* Initializing hgtxSyncUserAttributes_t for a specific version:
* \code
* hgtxSyncUserAttributes_t attribs = {0};
* attribs.version = 1;
* attribs.size = (uint16_t)(sizeof(hgtxSyncUserAttributes_t));
* \endcode
*
* If the caller uses Method 1 it is critical that the entire binary
* layout of the structure be configured to 0 so that all fields
* are initialized to the default value.
*
* The caller should either use both HGTX_VERSION and
* HGTX_SYNCUSER_ATTRIB_STRUCT_SIZE (Method 1) or use explicit values
* and a versioned type (Method 2).  Using a mix of the two methods
* will likely cause either source level incompatibility or binary
* incompatibility in the future.
*
* \par Example
* Populate a sync attributes structure:
* \code
* // Initialize
* hgtxSyncUserAttributes_t attribs = {0};
* attribs.version = HGTX_VERSION;
* attribs.size = HGTX_SYNCUSER_ATTRIB_STRUCT_SIZE;
*
* // Configure the Attributes
* attribs.messageType = HGTX_MESSAGE_TYPE_ASCII;
* attribs.message.ascii = "Example";
* \endcode
*
* \sa
* ::hgtxDomainSyncUserCreate
*/
typedef struct hgtxSyncUserAttributes_v0
{
    /**
    * \brief Version flag of the structure.
    *
    * Needs to be set to HGTX_VERSION to indicate the version of HGTX APIs
    * supported in this header file. This can optionally be overridden to
    * another version of the tools extension library.
    */
    uint16_t version;

    /**
    * \brief Size of the structure.
    *
    * Needs to be set to the size in bytes of the event attribute
    * structure used to specify the event.
    */
    uint16_t size;

    /** \brief Message type specified in this attribute structure.
    *
    * Defines the message format of the attribute structure's \ref hgtxSyncUserAttributes_v0::message
    * "message" field.
    *
    * Default Value is HGTX_MESSAGE_UNKNOWN
    */
    int32_t messageType;            /* hgtxMessageType_t */

    /** \brief Message assigned to this attribute structure.
    *
    * The text message that is attached to an event.
    */
    hgtxMessageValue_t message;

} hgtxSyncUserAttributes_v0;

typedef struct hgtxSyncUserAttributes_v0 hgtxSyncUserAttributes_t;

/* ------------------------------------------------------------------------- */
/** \brief Create a user defined synchronization object
* This is used to track non-OS synchronization working with spinlocks and atomics
*
* \param domain - Domain to own the resource
* \param attribs - A structure to assign multiple attributes to the object.
*
* \return A handle that represents the newly created user defined synchronization object.
*
* \sa
* ::hgtxDomainSyncUserCreate
* ::hgtxDomainSyncUserDestroy
* ::hgtxDomainSyncUserAcquireStart
* ::hgtxDomainSyncUserAcquireFailed
* ::hgtxDomainSyncUserAcquireSuccess
* ::hgtxDomainSyncUserReleasing
*
* \version HGTX_VERSION_2
*/
HGTX_DECLSPEC hgtxSyncUser_t HGTX_API hgtxDomainSyncUserCreate(hgtxDomainHandle_t domain, const hgtxSyncUserAttributes_t* attribs);

/* ------------------------------------------------------------------------- */
/** \brief Destroy a user defined synchronization object
* This is used to track non-OS synchronization working with spinlocks and atomics
*
* \param handle - A handle to the object to operate on.
*
* \sa
* ::hgtxDomainSyncUserCreate
* ::hgtxDomainSyncUserDestroy
* ::hgtxDomainSyncUserAcquireStart
* ::hgtxDomainSyncUserAcquireFailed
* ::hgtxDomainSyncUserAcquireSuccess
* ::hgtxDomainSyncUserReleasing
*
* \version HGTX_VERSION_2
*/
HGTX_DECLSPEC void HGTX_API hgtxDomainSyncUserDestroy(hgtxSyncUser_t handle);

/* ------------------------------------------------------------------------- */
/** \brief Signal to tools that an attempt to acquire a user defined synchronization object
*
* \param handle - A handle to the object to operate on.
*
* \sa
* ::hgtxDomainSyncUserCreate
* ::hgtxDomainSyncUserDestroy
* ::hgtxDomainSyncUserAcquireStart
* ::hgtxDomainSyncUserAcquireFailed
* ::hgtxDomainSyncUserAcquireSuccess
* ::hgtxDomainSyncUserReleasing
*
* \version HGTX_VERSION_2
*/
HGTX_DECLSPEC void HGTX_API hgtxDomainSyncUserAcquireStart(hgtxSyncUser_t handle);

/* ------------------------------------------------------------------------- */
/** \brief Signal to tools of failure in acquiring a user defined synchronization object
* This should be called after \ref hgtxDomainSyncUserAcquireStart
*
* \param handle - A handle to the object to operate on.
*
* \sa
* ::hgtxDomainSyncUserCreate
* ::hgtxDomainSyncUserDestroy
* ::hgtxDomainSyncUserAcquireStart
* ::hgtxDomainSyncUserAcquireFailed
* ::hgtxDomainSyncUserAcquireSuccess
* ::hgtxDomainSyncUserReleasing
*
* \version HGTX_VERSION_2
*/
HGTX_DECLSPEC void HGTX_API hgtxDomainSyncUserAcquireFailed(hgtxSyncUser_t handle);

/* ------------------------------------------------------------------------- */
/** \brief Signal to tools of success in acquiring a user defined synchronization object
* This should be called after \ref hgtxDomainSyncUserAcquireStart.
*
* \param handle - A handle to the object to operate on.
*
* \sa
* ::hgtxDomainSyncUserCreate
* ::hgtxDomainSyncUserDestroy
* ::hgtxDomainSyncUserAcquireStart
* ::hgtxDomainSyncUserAcquireFailed
* ::hgtxDomainSyncUserAcquireSuccess
* ::hgtxDomainSyncUserReleasing
*
* \version HGTX_VERSION_2
*/
HGTX_DECLSPEC void HGTX_API hgtxDomainSyncUserAcquireSuccess(hgtxSyncUser_t handle);

/* ------------------------------------------------------------------------- */
/** \brief Signal to tools of releasing a reservation on user defined synchronization object
* This should be called after \ref hgtxDomainSyncUserAcquireSuccess.
*
* \param handle - A handle to the object to operate on.
*
* \sa
* ::hgtxDomainSyncUserCreate
* ::hgtxDomainSyncUserDestroy
* ::hgtxDomainSyncUserAcquireStart
* ::hgtxDomainSyncUserAcquireFailed
* ::hgtxDomainSyncUserAcquireSuccess
* ::hgtxDomainSyncUserReleasing
*
* \version HGTX_VERSION_2
*/
HGTX_DECLSPEC void HGTX_API hgtxDomainSyncUserReleasing(hgtxSyncUser_t handle);


/** @} */ /*END defgroup*/

#ifdef __cplusplus
}
#endif /* __cplusplus */

#ifndef HGTX_NO_IMPL
#define HGTX_IMPL_GUARD_SYNC /* Ensure other headers cannot be included directly */
#include "hgtxDetail/hgtxImplSync_v3.h"
#undef HGTX_IMPL_GUARD_SYNC
#endif /*HGTX_NO_IMPL*/

#endif /* HGTOOLSEXT_SYNC_V3 */
