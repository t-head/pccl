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

#ifndef HGTX_EXT_IMPL_GUARD
#error Never include this file directly -- it is automatically included by hgToolsExt.h (except when HGTX_NO_IMPL is defined).
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

#ifndef HGTX_EXT_IMPL_H
#define HGTX_EXT_IMPL_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>

/* ---- Include required platform headers ---- */

#if defined(_WIN32)

#include <windows.h>

#else
#include <unistd.h>

#if defined(__ANDROID__)
#include <android/api-level.h>
#endif

#if defined(__linux__) || defined(__CYGWIN__)
#include <sched.h>
#endif

#include <sys/types.h>
#include <limits.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>

#endif

/* ---- Define macros used in this file ---- */

#ifdef HGTX_DEBUG_PRINT
#ifdef __ANDROID__
#include <android/log.h>
#define HGTX_ERR(...) __android_log_print(ANDROID_LOG_ERROR, "HGTOOLSEXT", __VA_ARGS__);
#define HGTX_INFO(...) __android_log_print(ANDROID_LOG_INFO, "HGTOOLSEXT", __VA_ARGS__);
#else
#include <stdio.h>
#define HGTX_ERR(...) fprintf(stderr, "HGTX_ERROR: " __VA_ARGS__)
#define HGTX_INFO(...) fprintf(stderr, "HGTX_INFO: " __VA_ARGS__)
#endif
#else /* !defined(HGTX_DEBUG_PRINT) */
#define HGTX_ERR(...)
#define HGTX_INFO(...)
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

#define HGTX_EXTENSION_FRESH 0
#define HGTX_EXTENSION_DISABLED 1
#define HGTX_EXTENSION_STARTING 2
#define HGTX_EXTENSION_LOADED 3

/* Function slots are local to each extension */
typedef struct hgtxExtGlobals1_t
{
    HgtxExtInitializeInjectionFunc_t injectionFnPtr;
} hgtxExtGlobals1_t;

HGTX_LINKONCE_DEFINE_GLOBAL hgtxExtGlobals1_t HGTX_VERSIONED_IDENTIFIER(hgtxExtGlobals1) =
{
    HGTX_NULLPTR
};

#define HGTX_EXT_INIT_GUARD
#include "hgtxExtInit.h"
#undef HGTX_EXT_INIT_GUARD

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* HGTX_EXT_IMPL_H */
