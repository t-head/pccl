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

#if defined(HGTX_AS_SYSTEM_HEADER)
#if defined(__clang__)
#pragma clang system_header
#elif defined(__GNUC__) || defined(__HGCOMPILER)
#pragma GCC system_header
#elif defined(_MSC_VER)
#pragma system_header
#endif
#endif

#ifndef HGTX_EXT_PAYLOAD_HELPER_INTERNAL_H
#define HGTX_EXT_PAYLOAD_HELPER_INTERNAL_H

/* General helper macros */
#include "hgtxExtHelperMacros.h"

/* Get variable name with line number (almost unique per file). */
#define _HGTX_PAYLOAD_DATA_VAR HGTX_EXT_CONCAT(hgtxDFDB,__LINE__)

/* Create real arguments from just pasting tokens next to each other. */
#define _HGTX_PAYLOAD_PASS_THROUGH(...) __VA_ARGS__

/* Avoid prefixing `HGTX_PAYLOAD_ENTRY_` for nested payloads. */
#define HGTX_PAYLOAD_ENTRY_THROWAWAY
#define _HGTX_PAYLOAD_NESTED(id) THROWAWAY id

/*
 * Create the HGTX binary payloads schema attributes.
 *
 * @param struct_id The name of the struct.
 * @param schema_name The name of the schema.
 * @param schema_flags Additional schema flags
 * @param mask_add Fields to be added to the mask.
 * @param num_entries The number schema entries.
 */
#define HGTX_PAYLOAD_SCHEMA_ATTR(struct_id, schema_name, schema_flags, schema_id, mask_add, num_entries) \
    hgtxPayloadSchemaAttr_t struct_id##Attr = { \
        /*.fieldMask = */HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_TYPE | mask_add \
            HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_ENTRIES | \
            HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_NUM_ENTRIES | \
            HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_STATIC_SIZE, \
        /*.name = */schema_name, \
        /*.type = */HGTX_PAYLOAD_SCHEMA_TYPE_STATIC, \
        /*.flags = */schema_flags, \
        /*.entries = */struct_id##Schema, /*.numEntries = */num_entries, \
        /*.payloadStaticSize = */sizeof(struct_id), \
        /*.packAlign = */0, /*.schemaId = */schema_id};


/*****************************************************************/
/*** Helper for `HGTX_DEFINE_SCHEMA_FOR_STRUCT[_AND_REGISTER]` ***/

/* First part of schema entry for different number of arguments. */
#define _HGTX_PAYLOAD_SCHEMA_EF2(member, etype) \
    0, HGTX_PAYLOAD_ENTRY_##etype, HGTX_NULLPTR, HGTX_NULLPTR, 0,
#define _HGTX_PAYLOAD_SCHEMA_EF3(member, etype, name) \
    0, HGTX_PAYLOAD_ENTRY_##etype, name, HGTX_NULLPTR, 0,
#define _HGTX_PAYLOAD_SCHEMA_EF4(member, etype, name, desc) \
    0, HGTX_PAYLOAD_ENTRY_##etype, name, desc, 0,
#define _HGTX_PAYLOAD_SCHEMA_EF5(member, etype, name, desc, arraylen) \
    0, HGTX_PAYLOAD_ENTRY_##etype, name, desc, arraylen,
#define _HGTX_PAYLOAD_SCHEMA_EF6(member, etype, name, desc, arraylen, flags) \
    HGTX_PAYLOAD_ENTRY_FLAG_##flags, HGTX_PAYLOAD_ENTRY_##etype, name, desc, arraylen,

#define _HGTX_PAYLOAD_SCHEMA_ENTRY_FRONT(...) \
    HGTX_EXT_CONCAT(_HGTX_PAYLOAD_SCHEMA_EF, HGTX_EXT_NUM_ARGS(__VA_ARGS__))(__VA_ARGS__)

/* Second part of schema entry (append struct member).
   (At least two arguments are passed (`member` and `etype`). */
#define _HGTX_PAYLOAD_SCHEMA_ENTRY_END(member, ...) member

/* Resolve to schema entry. `entry` is `(ctype, name, ...)`. */
#define _HGTX_PAYLOAD_SCHEMA_ENTRY(struct_id, entry) \
    {_HGTX_PAYLOAD_SCHEMA_ENTRY_FRONT entry \
    offsetof(struct_id, _HGTX_PAYLOAD_SCHEMA_ENTRY_END entry)},

/* Handle up to 16 schema entries. */
#define _HGTX_PAYLOAD_SME1(s,e1,...)  _HGTX_PAYLOAD_SCHEMA_ENTRY(s,e1)
#define _HGTX_PAYLOAD_SME2(s,e1,...)  _HGTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _HGTX_PAYLOAD_SME1(s,__VA_ARGS__)
#define _HGTX_PAYLOAD_SME3(s,e1,...)  _HGTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _HGTX_PAYLOAD_SME2(s,__VA_ARGS__)
#define _HGTX_PAYLOAD_SME4(s,e1,...)  _HGTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _HGTX_PAYLOAD_SME3(s,__VA_ARGS__)
#define _HGTX_PAYLOAD_SME5(s,e1,...)  _HGTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _HGTX_PAYLOAD_SME4(s,__VA_ARGS__)
#define _HGTX_PAYLOAD_SME6(s,e1,...)  _HGTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _HGTX_PAYLOAD_SME5(s,__VA_ARGS__)
#define _HGTX_PAYLOAD_SME7(s,e1,...)  _HGTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _HGTX_PAYLOAD_SME6(s,__VA_ARGS__)
#define _HGTX_PAYLOAD_SME8(s,e1,...)  _HGTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _HGTX_PAYLOAD_SME7(s,__VA_ARGS__)
#define _HGTX_PAYLOAD_SME9(s,e1,...)  _HGTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _HGTX_PAYLOAD_SME8(s,__VA_ARGS__)
#define _HGTX_PAYLOAD_SME10(s,e1,...) _HGTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _HGTX_PAYLOAD_SME9(s,__VA_ARGS__)
#define _HGTX_PAYLOAD_SME11(s,e1,...) _HGTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _HGTX_PAYLOAD_SME10(s,__VA_ARGS__)
#define _HGTX_PAYLOAD_SME12(s,e1,...) _HGTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _HGTX_PAYLOAD_SME11(s,__VA_ARGS__)
#define _HGTX_PAYLOAD_SME13(s,e1,...) _HGTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _HGTX_PAYLOAD_SME12(s,__VA_ARGS__)
#define _HGTX_PAYLOAD_SME14(s,e1,...) _HGTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _HGTX_PAYLOAD_SME13(s,__VA_ARGS__)
#define _HGTX_PAYLOAD_SME15(s,e1,...) _HGTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _HGTX_PAYLOAD_SME14(s,__VA_ARGS__)
#define _HGTX_PAYLOAD_SME16(s,e1,...) _HGTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _HGTX_PAYLOAD_SME15(s,__VA_ARGS__)

#define _HGTX_PAYLOAD_SCHEMA_ENTRIES(struct_id, ...) \
  hgtxPayloadSchemaEntry_t struct_id##Schema[] = { \
    HGTX_EXT_CONCAT(_HGTX_PAYLOAD_SME, HGTX_EXT_NUM_ARGS(__VA_ARGS__))(struct_id, __VA_ARGS__) \
    {0, 0} \
  };

/*
 * Handle optional parameters for `HGTX_DEFINE_SCHEMA_FOR_STRUCT[_AND_REGISTER]`.
 */
#define _HGTX_DEFINE_S4S_6(struct_id, schema_name, prefix, schema_flags, schema_id, entries) \
    prefix _HGTX_PAYLOAD_SCHEMA_ENTRIES(struct_id, _HGTX_PAYLOAD_PASS_THROUGH entries) \
    prefix HGTX_PAYLOAD_SCHEMA_ATTR(struct_id, schema_name, schema_flags, schema_id, \
        HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_NAME | HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_FLAGS | HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_SCHEMA_ID |,\
        HGTX_EXT_NUM_ARGS(_HGTX_PAYLOAD_PASS_THROUGH entries))
#define _HGTX_DEFINE_S4S_5(struct_id, schema_name, prefix, schema_flags, entries) \
    prefix _HGTX_PAYLOAD_SCHEMA_ENTRIES(struct_id, _HGTX_PAYLOAD_PASS_THROUGH entries) \
    prefix HGTX_PAYLOAD_SCHEMA_ATTR(struct_id, schema_name, schema_flags, 0, \
        HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_NAME | HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_FLAGS |, \
        HGTX_EXT_NUM_ARGS(_HGTX_PAYLOAD_PASS_THROUGH entries))
#define _HGTX_DEFINE_S4S_4(struct_id, schema_name, prefix, entries) \
    prefix _HGTX_PAYLOAD_SCHEMA_ENTRIES(struct_id, _HGTX_PAYLOAD_PASS_THROUGH entries) \
    prefix HGTX_PAYLOAD_SCHEMA_ATTR(struct_id, schema_name, HGTX_PAYLOAD_SCHEMA_FLAG_NONE, 0, \
        HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_NAME |, \
        HGTX_EXT_NUM_ARGS(_HGTX_PAYLOAD_PASS_THROUGH entries))
#define _HGTX_DEFINE_S4S_3(struct_id, schema_name, entries) \
    _HGTX_DEFINE_S4S_4(struct_id, schema_name, /*prefix*/, entries)
#define _HGTX_DEFINE_S4S_2(struct_id, entries) \
    _HGTX_PAYLOAD_SCHEMA_ENTRIES(struct_id, _HGTX_PAYLOAD_PASS_THROUGH entries) \
    HGTX_PAYLOAD_SCHEMA_ATTR(struct_id, HGTX_NULLPTR, HGTX_PAYLOAD_SCHEMA_FLAG_NONE, 0, ,\
        HGTX_EXT_NUM_ARGS(_HGTX_PAYLOAD_PASS_THROUGH entries))

#define _HGTX_DEFINE_SCHEMA_FOR_STRUCT(struct_id, ...) \
    HGTX_EXT_CONCAT(_HGTX_DEFINE_S4S_, \
        HGTX_EXT_NUM_ARGS(struct_id, __VA_ARGS__))(struct_id, __VA_ARGS__)

/*** END: Helper for `HGTX_PAYLOAD_STATIC_SCHEMA_{DEFINE,SETUP}` ***/


/******************************************************************/
/*** Helper for `HGTX_DEFINE_STRUCT_WITH_SCHEMA[_AND_REGISTER]` ***/

/* Extract struct member for fixed-size arrays. */
#define _HGTX_PAYLOAD_STRUCT_ARR_MEM1(name) name
#define _HGTX_PAYLOAD_STRUCT_ARR_MEM2(name, count) name[count]

/* Extract type and member name and handle special case of fixed-size array. */
#define _HGTX_PAYLOAD_STRUCT_E2(type, member) type member;
#define _HGTX_PAYLOAD_STRUCT_E3(type, member, etype) type member;
#define _HGTX_PAYLOAD_STRUCT_E4(type, member, etype, name) type member;
#define _HGTX_PAYLOAD_STRUCT_E5(type, member, etype, name, desc) type member;
#define _HGTX_PAYLOAD_STRUCT_E6(type, member, etype, name, desc, arraylen) \
    type HGTX_EXT_CONCAT(_HGTX_PAYLOAD_STRUCT_ARR_MEM, HGTX_EXT_NUM_ARGS member) member;
#define _HGTX_PAYLOAD_STRUCT_E7(type, member, etype, name, desc, arraylen, flags) \
    _HGTX_PAYLOAD_STRUCT_E6(type, member, etype, name, desc, arraylen)

/* Handle different number of arguments per struct entry. */
#define _HGTX_PAYLOAD_STRUCT_ENTRY_(...) \
    HGTX_EXT_CONCAT(_HGTX_PAYLOAD_STRUCT_E, HGTX_EXT_NUM_ARGS(__VA_ARGS__))(__VA_ARGS__)

/* Handle up to 16 struct members. */
#define _HGTX_PAYLOAD_STRUCT_ENTRY(entry) _HGTX_PAYLOAD_STRUCT_ENTRY_ entry
#define _HGTX_PAYLOAD_STRUCT1(e1, ...)  _HGTX_PAYLOAD_STRUCT_ENTRY(e1)
#define _HGTX_PAYLOAD_STRUCT2(e1, ...)  _HGTX_PAYLOAD_STRUCT_ENTRY(e1) _HGTX_PAYLOAD_STRUCT1(__VA_ARGS__)
#define _HGTX_PAYLOAD_STRUCT3(e1, ...)  _HGTX_PAYLOAD_STRUCT_ENTRY(e1) _HGTX_PAYLOAD_STRUCT2(__VA_ARGS__)
#define _HGTX_PAYLOAD_STRUCT4(e1, ...)  _HGTX_PAYLOAD_STRUCT_ENTRY(e1) _HGTX_PAYLOAD_STRUCT3(__VA_ARGS__)
#define _HGTX_PAYLOAD_STRUCT5(e1, ...)  _HGTX_PAYLOAD_STRUCT_ENTRY(e1) _HGTX_PAYLOAD_STRUCT4(__VA_ARGS__)
#define _HGTX_PAYLOAD_STRUCT6(e1, ...)  _HGTX_PAYLOAD_STRUCT_ENTRY(e1) _HGTX_PAYLOAD_STRUCT5(__VA_ARGS__)
#define _HGTX_PAYLOAD_STRUCT7(e1, ...)  _HGTX_PAYLOAD_STRUCT_ENTRY(e1) _HGTX_PAYLOAD_STRUCT6(__VA_ARGS__)
#define _HGTX_PAYLOAD_STRUCT8(e1, ...)  _HGTX_PAYLOAD_STRUCT_ENTRY(e1) _HGTX_PAYLOAD_STRUCT7(__VA_ARGS__)
#define _HGTX_PAYLOAD_STRUCT9(e1, ...)  _HGTX_PAYLOAD_STRUCT_ENTRY(e1) _HGTX_PAYLOAD_STRUCT8(__VA_ARGS__)
#define _HGTX_PAYLOAD_STRUCT10(e1, ...) _HGTX_PAYLOAD_STRUCT_ENTRY(e1) _HGTX_PAYLOAD_STRUCT9(__VA_ARGS__)
#define _HGTX_PAYLOAD_STRUCT11(e1, ...) _HGTX_PAYLOAD_STRUCT_ENTRY(e1) _HGTX_PAYLOAD_STRUCT10(__VA_ARGS__)
#define _HGTX_PAYLOAD_STRUCT12(e1, ...) _HGTX_PAYLOAD_STRUCT_ENTRY(e1) _HGTX_PAYLOAD_STRUCT11(__VA_ARGS__)
#define _HGTX_PAYLOAD_STRUCT13(e1, ...) _HGTX_PAYLOAD_STRUCT_ENTRY(e1) _HGTX_PAYLOAD_STRUCT12(__VA_ARGS__)
#define _HGTX_PAYLOAD_STRUCT14(e1, ...) _HGTX_PAYLOAD_STRUCT_ENTRY(e1) _HGTX_PAYLOAD_STRUCT13(__VA_ARGS__)
#define _HGTX_PAYLOAD_STRUCT15(e1, ...) _HGTX_PAYLOAD_STRUCT_ENTRY(e1) _HGTX_PAYLOAD_STRUCT14(__VA_ARGS__)
#define _HGTX_PAYLOAD_STRUCT16(e1, ...) _HGTX_PAYLOAD_STRUCT_ENTRY(e1) _HGTX_PAYLOAD_STRUCT15(__VA_ARGS__)

/* Generate the typedef. */
#define _HGTX_PAYLOAD_TYPEDEF_STRUCT(struct_id, ...) \
  typedef struct { \
      HGTX_EXT_CONCAT(_HGTX_PAYLOAD_STRUCT, HGTX_EXT_NUM_ARGS(__VA_ARGS__))(__VA_ARGS__) \
  } struct_id;

/* Generate first part of the schema entry. */
#define _HGTX_PAYLOAD_INIT_SCHEMA_N3(type, memberId, etype) \
    0, HGTX_PAYLOAD_ENTRY_##etype, HGTX_NULLPTR, HGTX_NULLPTR, 0,
#define _HGTX_PAYLOAD_INIT_SCHEMA_N4(type, memberId, etype, name) \
    0, HGTX_PAYLOAD_ENTRY_##etype, name, HGTX_NULLPTR, 0,
#define _HGTX_PAYLOAD_INIT_SCHEMA_N5(type, memberId, etype, name, desc) \
    0, HGTX_PAYLOAD_ENTRY_##etype, name, desc, 0,
#define _HGTX_PAYLOAD_INIT_SCHEMA_N6(type, memberId, etype, name, desc, arraylen) \
    0, HGTX_PAYLOAD_ENTRY_##etype, name, desc, arraylen,
#define _HGTX_PAYLOAD_INIT_SCHEMA_N7(type, memberId, etype, name, desc, arraylen, flags) \
    HGTX_PAYLOAD_ENTRY_FLAG_##flags, HGTX_PAYLOAD_ENTRY_##etype, name, desc, arraylen,

#define _HGTX_PAYLOAD_SCHEMA_INIT_ENTRY_FRONT(...) \
    HGTX_EXT_CONCAT(_HGTX_PAYLOAD_INIT_SCHEMA_N, HGTX_EXT_NUM_ARGS(__VA_ARGS__))(__VA_ARGS__)

#define _HGTX_PAYLOAD_ARRAY_MEMBER1(name) name
#define _HGTX_PAYLOAD_ARRAY_MEMBER2(name, count) name

/* Resolve to last part of schema entry (append struct member). */
#define _HGTX_PAYLOAD_INIT_SCHEMA_NX3(type, memberId, ...) memberId
#define _HGTX_PAYLOAD_INIT_SCHEMA_NX4(type, memberId, ...) memberId
#define _HGTX_PAYLOAD_INIT_SCHEMA_NX5(type, memberId, ...) memberId
#define _HGTX_PAYLOAD_INIT_SCHEMA_NX6(type, memberId, ...) \
    HGTX_EXT_CONCAT(_HGTX_PAYLOAD_ARRAY_MEMBER, HGTX_EXT_NUM_ARGS memberId) memberId
#define _HGTX_PAYLOAD_INIT_SCHEMA_NX7(type, memberId, ...) \
    _HGTX_PAYLOAD_INIT_SCHEMA_NX6(type, memberId, __VA_ARGS__)

#define _HGTX_PAYLOAD_SCHEMA_INIT_ENTRY_END(...) \
    HGTX_EXT_CONCAT(_HGTX_PAYLOAD_INIT_SCHEMA_NX, HGTX_EXT_NUM_ARGS(__VA_ARGS__))(__VA_ARGS__)

/* Resolve to schema entry. `entry` is `(ctype, name, ...)`. */
#define _HGTX_PAYLOAD_SCHEMA_INIT_ENTRY(struct_id, entry) \
    {_HGTX_PAYLOAD_SCHEMA_INIT_ENTRY_FRONT entry \
    offsetof(struct_id, _HGTX_PAYLOAD_SCHEMA_INIT_ENTRY_END entry)},

/* Handle up to 16 schema entries. */
#define _HGTX_PAYLOAD_INIT_SME1(s, e1, ...)  _HGTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1)
#define _HGTX_PAYLOAD_INIT_SME2(s, e1, ...)  _HGTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _HGTX_PAYLOAD_INIT_SME1(s, __VA_ARGS__)
#define _HGTX_PAYLOAD_INIT_SME3(s, e1, ...)  _HGTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _HGTX_PAYLOAD_INIT_SME2(s, __VA_ARGS__)
#define _HGTX_PAYLOAD_INIT_SME4(s, e1, ...)  _HGTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _HGTX_PAYLOAD_INIT_SME3(s, __VA_ARGS__)
#define _HGTX_PAYLOAD_INIT_SME5(s, e1, ...)  _HGTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _HGTX_PAYLOAD_INIT_SME4(s, __VA_ARGS__)
#define _HGTX_PAYLOAD_INIT_SME6(s, e1, ...)  _HGTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _HGTX_PAYLOAD_INIT_SME5(s, __VA_ARGS__)
#define _HGTX_PAYLOAD_INIT_SME7(s, e1, ...)  _HGTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _HGTX_PAYLOAD_INIT_SME6(s, __VA_ARGS__)
#define _HGTX_PAYLOAD_INIT_SME8(s, e1, ...)  _HGTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _HGTX_PAYLOAD_INIT_SME7(s, __VA_ARGS__)
#define _HGTX_PAYLOAD_INIT_SME9(s, e1, ...)  _HGTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _HGTX_PAYLOAD_INIT_SME8(s, __VA_ARGS__)
#define _HGTX_PAYLOAD_INIT_SME10(s, e1, ...) _HGTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _HGTX_PAYLOAD_INIT_SME9(s, __VA_ARGS__)
#define _HGTX_PAYLOAD_INIT_SME11(s, e1, ...) _HGTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _HGTX_PAYLOAD_INIT_SME10(s, __VA_ARGS__)
#define _HGTX_PAYLOAD_INIT_SME12(s, e1, ...) _HGTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _HGTX_PAYLOAD_INIT_SME11(s, __VA_ARGS__)
#define _HGTX_PAYLOAD_INIT_SME13(s, e1, ...) _HGTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _HGTX_PAYLOAD_INIT_SME12(s, __VA_ARGS__)
#define _HGTX_PAYLOAD_INIT_SME14(s, e1, ...) _HGTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _HGTX_PAYLOAD_INIT_SME13(s, __VA_ARGS__)
#define _HGTX_PAYLOAD_INIT_SME15(s, e1, ...) _HGTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _HGTX_PAYLOAD_INIT_SME14(s, __VA_ARGS__)
#define _HGTX_PAYLOAD_INIT_SME16(s, e1, ...) _HGTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _HGTX_PAYLOAD_INIT_SME15(s, __VA_ARGS__)

#define _HGTX_PAYLOAD_SCHEMA_INIT_ENTRIES(struct_id, ...) \
  hgtxPayloadSchemaEntry_t struct_id##Schema[] = { \
    HGTX_EXT_CONCAT(_HGTX_PAYLOAD_INIT_SME, HGTX_EXT_NUM_ARGS(__VA_ARGS__))(struct_id, __VA_ARGS__) \
    {0, 0} \
  };

/*
 * Handle optional parameters for `HGTX_DEFINE_STRUCT_WITH_SCHEMA[_AND_REGISTER]`.
 */
#define _HGTX_DEFINE_SWS_6(struct_id, schema_name, prefix, schema_flags, schema_id, entries) \
  _HGTX_PAYLOAD_TYPEDEF_STRUCT(struct_id, _HGTX_PAYLOAD_PASS_THROUGH entries) \
  prefix _HGTX_PAYLOAD_SCHEMA_INIT_ENTRIES(struct_id, _HGTX_PAYLOAD_PASS_THROUGH entries) \
  prefix HGTX_PAYLOAD_SCHEMA_ATTR(struct_id, schema_name, schema_flags, schema_id, \
      HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_NAME | HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_FLAGS | \
      HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_SCHEMA_ID |, \
      HGTX_EXT_NUM_ARGS(_HGTX_PAYLOAD_PASS_THROUGH entries))
#define _HGTX_DEFINE_SWS_5(struct_id, schema_name, prefix, schema_flags, entries) \
  _HGTX_PAYLOAD_TYPEDEF_STRUCT(struct_id, _HGTX_PAYLOAD_PASS_THROUGH entries) \
  prefix _HGTX_PAYLOAD_SCHEMA_INIT_ENTRIES(struct_id, _HGTX_PAYLOAD_PASS_THROUGH entries) \
  prefix HGTX_PAYLOAD_SCHEMA_ATTR(struct_id, schema_name, schema_flags, 0, \
      HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_NAME | HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_FLAGS |, \
      HGTX_EXT_NUM_ARGS(_HGTX_PAYLOAD_PASS_THROUGH entries))
#define _HGTX_DEFINE_SWS_4(struct_id, schema_name, prefix, entries) \
  _HGTX_PAYLOAD_TYPEDEF_STRUCT(struct_id, _HGTX_PAYLOAD_PASS_THROUGH entries) \
  prefix _HGTX_PAYLOAD_SCHEMA_INIT_ENTRIES(struct_id, _HGTX_PAYLOAD_PASS_THROUGH entries) \
  prefix HGTX_PAYLOAD_SCHEMA_ATTR(struct_id, schema_name, HGTX_PAYLOAD_SCHEMA_FLAG_NONE, 0, \
      HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_NAME |, \
      HGTX_EXT_NUM_ARGS(_HGTX_PAYLOAD_PASS_THROUGH entries))
#define _HGTX_DEFINE_SWS_3(struct_id, schema_name, entries) \
  _HGTX_DEFINE_SWS_4(struct_id, schema_name, /* no prefix */, entries)
#define _HGTX_DEFINE_SWS_2(struct_id, entries) \
  _HGTX_PAYLOAD_TYPEDEF_STRUCT(struct_id, _HGTX_PAYLOAD_PASS_THROUGH entries) \
  _HGTX_PAYLOAD_SCHEMA_INIT_ENTRIES(struct_id, _HGTX_PAYLOAD_PASS_THROUGH entries) \
  HGTX_PAYLOAD_SCHEMA_ATTR(struct_id, HGTX_NULLPTR, HGTX_PAYLOAD_SCHEMA_FLAG_NONE, 0, , \
      HGTX_EXT_NUM_ARGS(_HGTX_PAYLOAD_PASS_THROUGH entries))

#define _HGTX_DEFINE_STRUCT_WITH_SCHEMA(struct_id, ...) \
    HGTX_EXT_CONCAT(_HGTX_DEFINE_SWS_, \
        HGTX_EXT_NUM_ARGS(struct_id, __VA_ARGS__))(struct_id, __VA_ARGS__)

/*** END: Helper for `HGTX_PAYLOAD_STATIC_SCHEMA_{INIT,CREATE}` */

#endif /* HGTX_EXT_PAYLOAD_HELPER_INTERNAL_H */
