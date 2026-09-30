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

typedef void* hgtx_payload_pointer_type;

#if (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
#ifndef __APPLE__
#include <uchar.h>
#endif
#include <stdalign.h>
#endif

/* `char8_t` is available as of C++20 or C23 */
#if ((defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L) || (defined(__cplusplus) && __cplusplus >= 201811L)) && !defined(__APPLE__)
#define HGTX_HAVE_CHAR8 1
#else
#define HGTX_HAVE_CHAR8 0
#endif

/* `char16_t` and `char32_t` are available as of C++11 or C11 */
#if ((defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L) || (defined(__cplusplus) && __cplusplus >= 200704L)) && !defined(__APPLE__)
#define HGTX_HAVE_CHAR16_CHAR32 1
#else
#define HGTX_HAVE_CHAR16_CHAR32 0
#endif

/* `alignof` is available as of C11 or C++11. */
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)) || (defined(__cplusplus) && __cplusplus >= 201103L)

#define hgtx_alignof(type) alignof(type)
#define hgtx_alignof2(type,tname) alignof(type)

#else /* (__STDC_VERSION__ >= 201112L) || (__cplusplus >= 201103L) */

/* Create helper structs to determine type alignment. */
#define MKTYPEDEF(type) typedef struct {char c; type d;} _hgtx_##type
#define MKTYPEDEF2(type,tname) typedef struct {char c; type d;} _hgtx_##tname

MKTYPEDEF(char);
MKTYPEDEF2(unsigned char, uchar);
MKTYPEDEF(short);
MKTYPEDEF2(unsigned short, ushort);
MKTYPEDEF(int);
MKTYPEDEF2(unsigned int, uint);
MKTYPEDEF(long);
MKTYPEDEF2(unsigned long, ulong);
MKTYPEDEF2(long long, longlong);
MKTYPEDEF2(unsigned long long, ulonglong);

MKTYPEDEF(int8_t);
MKTYPEDEF(uint8_t);
MKTYPEDEF(int16_t);
MKTYPEDEF(uint16_t);
MKTYPEDEF(int32_t);
MKTYPEDEF(uint32_t);
MKTYPEDEF(int64_t);
MKTYPEDEF(uint64_t);

MKTYPEDEF(float);
MKTYPEDEF(double);
MKTYPEDEF2(long double, longdouble);

MKTYPEDEF(size_t);
MKTYPEDEF(hgtx_payload_pointer_type);

MKTYPEDEF(wchar_t);

#if HGTX_HAVE_CHAR8
    MKTYPEDEF(char8_t);
#endif

#if HGTX_HAVE_CHAR16_CHAR32
    MKTYPEDEF(char16_t);
    MKTYPEDEF(char32_t);
#endif

/* C requires to include stddef.h to use `offsetof` */
#ifndef __cplusplus
#include <stddef.h>
#endif

#define hgtx_alignof(tname) offsetof(_hgtx_##tname, d)
#define hgtx_alignof2(type, tname) offsetof(_hgtx_##tname, d)

#endif /*  __STDC_VERSION__ >= 201112L */

#undef MKTYPEDEF
#undef MKTYPEDEF2

/*
 * Helper array to get the alignment for each predefined C/C++ language type.
 * The order of entries must match the values in`enum hgtxPayloadSchemaEntryType`.
 *
 * In C++, `const` variables use internal linkage by default, but we need it to
 * be public (extern) since weak declarations must be public.
 */
HGTX_LINKONCE_DEFINE_GLOBAL
#ifdef __cplusplus
extern
#endif
const hgtxPayloadEntryTypeInfo_t
HGTX_EXT_PAYLOAD_VERSIONED_ID(hgtxExtPayloadTypeInfo)[HGTX_PAYLOAD_ENTRY_TYPE_INFO_ARRAY_SIZE] =
{
    /* The first entry contains this array's length and the size of each entry in this array. */
    {HGTX_PAYLOAD_ENTRY_TYPE_INFO_ARRAY_SIZE, sizeof(hgtxPayloadEntryTypeInfo_t)},

    /*** C integer types ***/
    /* HGTX_PAYLOAD_ENTRY_TYPE_CHAR */   {sizeof(char), hgtx_alignof(char)},
    /* HGTX_PAYLOAD_ENTRY_TYPE_UCHAR */  {sizeof(unsigned char), hgtx_alignof2(unsigned char, uchar)},
    /* HGTX_PAYLOAD_ENTRY_TYPE_SHORT */  {sizeof(short), hgtx_alignof(short)},
    /* HGTX_PAYLOAD_ENTRY_TYPE_USHORT */ {sizeof(unsigned short), hgtx_alignof2(unsigned short, ushort)},
    /* HGTX_PAYLOAD_ENTRY_TYPE_INT */    {sizeof(int), hgtx_alignof(int)},
    /* HGTX_PAYLOAD_ENTRY_TYPE_UINT */   {sizeof(unsigned int), hgtx_alignof2(unsigned int, uint)},
    /* HGTX_PAYLOAD_ENTRY_TYPE_LONG */   {sizeof(long), hgtx_alignof(long)},
    /* HGTX_PAYLOAD_ENTRY_TYPE_ULONG */  {sizeof(unsigned long), hgtx_alignof2(unsigned long, ulong)},
    /* HGTX_PAYLOAD_ENTRY_TYPE_LONGLONG */  {sizeof(long long), hgtx_alignof2(long long, longlong)},
    /* HGTX_PAYLOAD_ENTRY_TYPE_ULONGLONG */ {sizeof(unsigned long long), hgtx_alignof2(unsigned long long,ulonglong)},

    /*** Integer types with explicit size ***/
    /* HGTX_PAYLOAD_ENTRY_TYPE_INT8 */   {sizeof(int8_t),   hgtx_alignof(int8_t)},
    /* HGTX_PAYLOAD_ENTRY_TYPE_UINT8 */  {sizeof(uint8_t),  hgtx_alignof(uint8_t)},
    /* HGTX_PAYLOAD_ENTRY_TYPE_INT16 */  {sizeof(int16_t),  hgtx_alignof(int16_t)},
    /* HGTX_PAYLOAD_ENTRY_TYPE_UINT16 */ {sizeof(uint16_t), hgtx_alignof(uint16_t)},
    /* HGTX_PAYLOAD_ENTRY_TYPE_INT32 */  {sizeof(int32_t),  hgtx_alignof(int32_t)},
    /* HGTX_PAYLOAD_ENTRY_TYPE_UINT32 */ {sizeof(uint32_t), hgtx_alignof(uint32_t)},
    /* HGTX_PAYLOAD_ENTRY_TYPE_INT64 */  {sizeof(int64_t),  hgtx_alignof(int64_t)},
    /* HGTX_PAYLOAD_ENTRY_TYPE_UINT64 */ {sizeof(uint64_t), hgtx_alignof(uint64_t)},

    /*** C floating point types ***/
    /* HGTX_PAYLOAD_ENTRY_TYPE_FLOAT */      {sizeof(float),       hgtx_alignof(float)},
    /* HGTX_PAYLOAD_ENTRY_TYPE_DOUBLE */     {sizeof(double),      hgtx_alignof(double)},
    /* HGTX_PAYLOAD_ENTRY_TYPE_LONGDOUBLE */ {sizeof(long double), hgtx_alignof2(long double, longdouble)},

    /* HGTX_PAYLOAD_ENTRY_TYPE_SIZE */    {sizeof(size_t),       hgtx_alignof(size_t)},
    /* HGTX_PAYLOAD_ENTRY_TYPE_ADDRESS */ {sizeof(hgtx_payload_pointer_type), hgtx_alignof(hgtx_payload_pointer_type)},

    /*** Special character types ***/
    /* HGTX_PAYLOAD_ENTRY_TYPE_WCHAR */ {sizeof(wchar_t), hgtx_alignof(wchar_t)},

#if HGTX_HAVE_CHAR8
    /* HGTX_PAYLOAD_ENTRY_TYPE_CHAR8 */ {sizeof(char8_t), hgtx_alignof(char8_t)},
#else
    /* HGTX_PAYLOAD_ENTRY_TYPE_CHAR8 */ {0, 0},
#endif

#if HGTX_HAVE_CHAR16_CHAR32
    /* HGTX_PAYLOAD_ENTRY_TYPE_CHAR16 */ {sizeof(char16_t), hgtx_alignof(char16_t)},
    /* HGTX_PAYLOAD_ENTRY_TYPE_CHAR32 */ {sizeof(char32_t), hgtx_alignof(char32_t)}
#else
    /* HGTX_PAYLOAD_ENTRY_TYPE_CHAR16 */ {0, 0},
    /* HGTX_PAYLOAD_ENTRY_TYPE_CHAR32 */ {0, 0}
#endif
};

#undef hgtx_alignof
#undef hgtx_alignof2
#undef HGTX_HAVE_CHAR8
#undef HGTX_HAVE_CHAR16_CHAR32
