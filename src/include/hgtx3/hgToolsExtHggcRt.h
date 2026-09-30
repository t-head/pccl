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

#include "hggc.h"
#include "driver_types.h"

#ifndef HGTOOLSEXT_HGGCRT_V3
#define HGTOOLSEXT_HGGCRT_V3

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* ========================================================================= */
/** \name Functions for HGGC Resource Naming
*/
/** \addtogroup RESOURCE_NAMING
 * \section RESOURCE_NAMING_HGGCRT HGGC Runtime Resource Naming
 *
 * This section covers the API functions that allow to annotate HGGC resources
 * with user-provided names.
 *
 * @{
 */

/*  ------------------------------------------------------------------------- */
/* \cond SHOW_HIDDEN
* \brief Used to build a non-colliding value for resource types separated class
* \version HGTX_VERSION_2
*/
#define HGTX_RESOURCE_CLASS_HGGCRT 5
/** \endcond */

/*  ------------------------------------------------------------------------- */
/** \brief Resource types for HGGCRT
*/
typedef enum hgtxResourceHGGCRTType_t
{
    HGTX_RESOURCE_TYPE_HGGCRT_DEVICE = HGTX_RESOURCE_MAKE_TYPE(HGGCRT, 0), /* int device */
    HGTX_RESOURCE_TYPE_HGGCRT_STREAM = HGTX_RESOURCE_MAKE_TYPE(HGGCRT, 1), /* hggcStream_t */
    HGTX_RESOURCE_TYPE_HGGCRT_EVENT = HGTX_RESOURCE_MAKE_TYPE(HGGCRT, 2) /* hggcEvent_t */
} hgtxResourceHGGCRTType_t;


/* ------------------------------------------------------------------------- */
/** \brief Annotates a HGGC device.
 *
 * Allows the user to associate a HGGC device with a user-provided name.
 *
 * \param device - The id of the HGGC device to name.
 * \param name   - The name of the HGGC device.
 *
 * \version HGTX_VERSION_1
 * @{ */
HGTX_DECLSPEC void HGTX_API hgtxNameHggcDeviceA(int device, const char* name);
HGTX_DECLSPEC void HGTX_API hgtxNameHggcDeviceW(int device, const wchar_t* name);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Annotates a HGGC stream.
 *
 * Allows the user to associate a HGGC stream with a user-provided name.
 *
 * \param stream - The handle of the HGGC stream to name.
 * \param name   - The name of the HGGC stream.
 *
 * \version HGTX_VERSION_1
 * @{ */
HGTX_DECLSPEC void HGTX_API hgtxNameHggcStreamA(hggcStream_t stream, const char* name);
HGTX_DECLSPEC void HGTX_API hgtxNameHggcStreamW(hggcStream_t stream, const wchar_t* name);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Annotates a HGGC event.
 *
 * Allows the user to associate a HGGC event with a user-provided name.
 *
 * \param event - The handle of the HGGC event to name.
 * \param name  - The name of the HGGC event.
 *
 * \version HGTX_VERSION_1
 * @{ */
HGTX_DECLSPEC void HGTX_API hgtxNameHggcEventA(hggcEvent_t event, const char* name);
HGTX_DECLSPEC void HGTX_API hgtxNameHggcEventW(hggcEvent_t event, const wchar_t* name);
/** @} */

/** @} */ /* END RESOURCE_NAMING */

/* ========================================================================= */
#ifdef UNICODE
  #define hgtxNameHggcDevice hgtxNameHggcDeviceW
  #define hgtxNameHggcStream hgtxNameHggcStreamW
  #define hgtxNameHggcEvent  hgtxNameHggcEventW
#else
  #define hgtxNameHggcDevice hgtxNameHggcDeviceA
  #define hgtxNameHggcStream hgtxNameHggcStreamA
  #define hgtxNameHggcEvent  hgtxNameHggcEventA
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */

#ifndef HGTX_NO_IMPL
#define HGTX_IMPL_GUARD_HGGCRT /* Ensure other headers cannot be included directly */
#include "hgtxDetail/hgtxImplHggcRt_v3.h"
#undef HGTX_IMPL_GUARD_HGGCRT
#endif /*HGTX_NO_IMPL*/

#endif /* HGTOOLSEXT_HGGCRT_V3 */
