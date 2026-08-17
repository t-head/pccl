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

#include <CL/cl.h>

#ifndef HGTOOLSEXT_OPENCL_V3
#define HGTOOLSEXT_OPENCL_V3

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* ========================================================================= */
/** \name Functions for OpenCL Resource Naming
 */
/** \addtogroup RESOURCE_NAMING
 * \section RESOURCE_NAMING_OPENCL OpenCL Resource Naming
 *
 * This section covers the API functions that allow to annotate OpenCL resources
 * with user-provided names.
 *
 * @{
 */

/*  ------------------------------------------------------------------------- */
/* \cond SHOW_HIDDEN
* \brief Used to build a non-colliding value for resource types separated class
* \version HGTX_VERSION_2
*/
#define HGTX_RESOURCE_CLASS_OPENCL 6
/** \endcond */

/*  ------------------------------------------------------------------------- */
/** \brief Resource types for OpenCL
*/
typedef enum hgtxResourceOpenCLType_t
{
    HGTX_RESOURCE_TYPE_OPENCL_DEVICE = HGTX_RESOURCE_MAKE_TYPE(OPENCL, 1),
    HGTX_RESOURCE_TYPE_OPENCL_CONTEXT = HGTX_RESOURCE_MAKE_TYPE(OPENCL, 2),
    HGTX_RESOURCE_TYPE_OPENCL_COMMANDQUEUE = HGTX_RESOURCE_MAKE_TYPE(OPENCL, 3),
    HGTX_RESOURCE_TYPE_OPENCL_MEMOBJECT = HGTX_RESOURCE_MAKE_TYPE(OPENCL, 4),
    HGTX_RESOURCE_TYPE_OPENCL_SAMPLER = HGTX_RESOURCE_MAKE_TYPE(OPENCL, 5),
    HGTX_RESOURCE_TYPE_OPENCL_PROGRAM = HGTX_RESOURCE_MAKE_TYPE(OPENCL, 6),
    HGTX_RESOURCE_TYPE_OPENCL_EVENT = HGTX_RESOURCE_MAKE_TYPE(OPENCL, 7)
} hgtxResourceOpenCLType_t;


/* ------------------------------------------------------------------------- */
/** \brief Annotates an OpenCL device.
 *
 * Allows to associate an OpenCL device with a user-provided name.
 *
 * \param device - The handle of the OpenCL device to name.
 * \param name   - The name of the OpenCL device.
 *
 * \version HGTX_VERSION_1
 * @{ */
HGTX_DECLSPEC void HGTX_API hgtxNameClDeviceA(cl_device_id device, const char* name);
HGTX_DECLSPEC void HGTX_API hgtxNameClDeviceW(cl_device_id device, const wchar_t* name);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Annotates an OpenCL context.
 *
 * Allows to associate an OpenCL context with a user-provided name.
 *
 * \param context - The handle of the OpenCL context to name.
 * \param name    - The name of the OpenCL context.
 *
 * \version HGTX_VERSION_1
 * @{ */
HGTX_DECLSPEC void HGTX_API hgtxNameClContextA(cl_context context, const char* name);
HGTX_DECLSPEC void HGTX_API hgtxNameClContextW(cl_context context, const wchar_t* name);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Annotates an OpenCL command queue.
 *
 * Allows to associate an OpenCL command queue with a user-provided name.
 *
 * \param command_queue - The handle of the OpenCL command queue to name.
 * \param name          - The name of the OpenCL command queue.
 *
 * \version HGTX_VERSION_1
 * @{ */
HGTX_DECLSPEC void HGTX_API hgtxNameClCommandQueueA(cl_command_queue command_queue, const char* name);
HGTX_DECLSPEC void HGTX_API hgtxNameClCommandQueueW(cl_command_queue command_queue, const wchar_t* name);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Annotates an OpenCL memory object.
 *
 * Allows to associate an OpenCL memory object with a user-provided name.
 *
 * \param memobj - The handle of the OpenCL memory object to name.
 * \param name   - The name of the OpenCL memory object.
 *
 * \version HGTX_VERSION_1
 * @{ */
HGTX_DECLSPEC void HGTX_API hgtxNameClMemObjectA(cl_mem memobj, const char* name);
HGTX_DECLSPEC void HGTX_API hgtxNameClMemObjectW(cl_mem memobj, const wchar_t* name);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Annotates an OpenCL sampler.
 *
 * Allows to associate an OpenCL sampler with a user-provided name.
 *
 * \param sampler - The handle of the OpenCL sampler to name.
 * \param name    - The name of the OpenCL sampler.
 *
 * \version HGTX_VERSION_1
 * @{ */
HGTX_DECLSPEC void HGTX_API hgtxNameClSamplerA(cl_sampler sampler, const char* name);
HGTX_DECLSPEC void HGTX_API hgtxNameClSamplerW(cl_sampler sampler, const wchar_t* name);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Annotates an OpenCL program.
 *
 * Allows to associate an OpenCL program with a user-provided name.
 *
 * \param program - The handle of the OpenCL program to name.
 * \param name    - The name of the OpenCL program.
 *
 * \code
 * cpProgram = clCreateProgramWithSource(cxGPUContext, 1,
 *     (const char **) &cSourceCL, &program_length, &ciErrNum);
 * shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
 * hgtxNameClProgram(cpProgram, L"PROGRAM_NAME");
 * \endcode
 *
 * \version HGTX_VERSION_1
 * @{ */
HGTX_DECLSPEC void HGTX_API hgtxNameClProgramA(cl_program program, const char* name);
HGTX_DECLSPEC void HGTX_API hgtxNameClProgramW(cl_program program, const wchar_t* name);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Annotates an OpenCL event.
 *
 * Allows to associate an OpenCL event with a user-provided name.
 *
 * \param evnt - The handle of the OpenCL event to name.
 * \param name - The name of the OpenCL event.
 *
 * \version HGTX_VERSION_1
 * @{ */
HGTX_DECLSPEC void HGTX_API hgtxNameClEventA(cl_event evnt, const char* name);
HGTX_DECLSPEC void HGTX_API hgtxNameClEventW(cl_event evnt, const wchar_t* name);
/** @} */

/** @} */ /* END RESOURCE_NAMING */

/* ========================================================================= */
#ifdef UNICODE
  #define hgtxNameClDevice        hgtxNameClDeviceW
  #define hgtxNameClContext       hgtxNameClContextW
  #define hgtxNameClCommandQueue  hgtxNameClCommandQueueW
  #define hgtxNameClMemObject     hgtxNameClMemObjectW
  #define hgtxNameClSampler       hgtxNameClSamplerW
  #define hgtxNameClProgram       hgtxNameClProgramW
  #define hgtxNameClEvent         hgtxNameClEventW
#else
  #define hgtxNameClDevice        hgtxNameClDeviceA
  #define hgtxNameClContext       hgtxNameClContextA
  #define hgtxNameClCommandQueue  hgtxNameClCommandQueueA
  #define hgtxNameClMemObject     hgtxNameClMemObjectA
  #define hgtxNameClSampler       hgtxNameClSamplerA
  #define hgtxNameClProgram       hgtxNameClProgramA
  #define hgtxNameClEvent         hgtxNameClEventA
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */

#ifndef HGTX_NO_IMPL
#define HGTX_IMPL_GUARD_OPENCL /* Ensure other headers cannot be included directly */
#include "hgtxDetail/hgtxImplOpenCL_v3.h"
#undef HGTX_IMPL_GUARD_OPENCL
#endif /*HGTX_NO_IMPL*/

#endif /* HGTOOLSEXT_OPENCL_V3 */
