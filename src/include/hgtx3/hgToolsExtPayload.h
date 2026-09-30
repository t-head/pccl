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

/* Optionally include helper macros. */
/* #include "hgToolsExtPayloadHelper.h" */

/**
 * If needed, semantic extension headers can be included after this header.
 */

/**
 * \brief The compatibility ID is used for versioning of this extension.
 */
#ifndef HGTX_EXT_PAYLOAD_COMPATID
#define HGTX_EXT_PAYLOAD_COMPATID 0x0104
#endif

/**
 * \brief The module ID identifies the payload extension. It has to be unique
 * among the extension modules.
 */
#ifndef HGTX_EXT_PAYLOAD_MODULEID
#define HGTX_EXT_PAYLOAD_MODULEID 2
#endif

/**
 * \brief Additional value for the enum @ref hgtxPayloadType_t
 */
#ifndef HGTX_PAYLOAD_TYPE_EXT
#define HGTX_PAYLOAD_TYPE_EXT (HGTX_STATIC_CAST(int32_t, 0xDFBD0009))
#endif

/** ---------------------------------------------------------------------------
 * Payload schema entry flags. Used for @ref hgtxPayloadSchemaEntry_t::flags.
 * ------------------------------------------------------------------------- */
#ifndef HGTX_PAYLOAD_ENTRY_FLAGS_V1
#define HGTX_PAYLOAD_ENTRY_FLAGS_V1

#define HGTX_PAYLOAD_ENTRY_FLAG_UNUSED 0

/**
 * Absolute pointer into a payload (entry) of the same event.
 */
#define HGTX_PAYLOAD_ENTRY_FLAG_POINTER          (1 << 1)

/**
 * Offset from base address of the payload.
 */
#define HGTX_PAYLOAD_ENTRY_FLAG_OFFSET_FROM_BASE (1 << 2)

/**
 * Offset from the end of this payload entry.
 */
#define HGTX_PAYLOAD_ENTRY_FLAG_OFFSET_FROM_HERE (1 << 3)

/**
 * The value is an array with fixed length, set with the field `arrayLength`.
 */
#define HGTX_PAYLOAD_ENTRY_FLAG_ARRAY_FIXED_SIZE           (1 << 4)

/**
 * The value is a zero-/null-terminated array.
 */
#define HGTX_PAYLOAD_ENTRY_FLAG_ARRAY_ZERO_TERMINATED      (2 << 4)

/**
 * \brief A single or multi-dimensional array of variable length.
 *
 * The field `arrayOrUnionDetail` contains the index of the schema entry that
 * holds the length(s). If the length entry is a scalar, then this entry is a 1D
 * array. If the length entry is a fixed-size array, then the number of
 * dimensions is defined with the registration of the schema. If the length
 * entry is a zero-terminated array, then the array of the dimensions can be
 * determined at runtime.
 * For multidimensional arrays, values are stored in row-major order, with rows
 * being stored consecutively in contiguous memory. The size of the entry (in
 * bytes) is the product of the dimensions multiplied with size of the array
 * element.
 */
#define HGTX_PAYLOAD_ENTRY_FLAG_ARRAY_LENGTH_INDEX         (3 << 4)

/**
 * \brief A single or multi-dimensional array of variable length, where the
 * dimensions are stored in a different payload (index) of the same event.
 *
 * This enables an existing address to an array to be directly passed, while the
 * dimensions are defined in a separate payload (with only one payload entry).
 */
#define HGTX_PAYLOAD_ENTRY_FLAG_ARRAY_LENGTH_PAYLOAD_INDEX (4 << 4)

/**
 * \brief The value or data that is pointed to by this payload entry value shall
 * be copied by the HGTX handler.
 *
 * A tool may not support deep copy and just ignore this flag.
 * See @ref HGTX_PAYLOAD_SCHEMA_FLAG_DEEP_COPY for more details.
 */
#define HGTX_PAYLOAD_ENTRY_FLAG_DEEP_COPY          (1 << 8)

/**
 * Notifies the HGTX handler to hide this entry in case of visualization.
 */
#define HGTX_PAYLOAD_ENTRY_FLAG_HIDE               (1 << 9)

/**
 * The entry specifies the event message. Any string type can be used.
 */
#define HGTX_PAYLOAD_ENTRY_FLAG_EVENT_MESSAGE      (1 << 10)

/**
 * \brief The entry contains a timestamp.
 *
 * The time source might be provided via the entry semantics field. In most
 * cases, the timestamp (entry) type is @ref HGTX_PAYLOAD_ENTRY_TYPE_INT64.
 */
#define HGTX_PAYLOAD_ENTRY_FLAG_TIMESTAMP          (2 << 10)

/**
 * These flags specify the HGTX event type to which an entry refers.
 */
#define HGTX_PAYLOAD_ENTRY_FLAG_RANGE_BEGIN        (1 << 12)
#define HGTX_PAYLOAD_ENTRY_FLAG_RANGE_END          (2 << 12)
#define HGTX_PAYLOAD_ENTRY_FLAG_MARK               (3 << 12)
#define HGTX_PAYLOAD_ENTRY_FLAG_COUNTER            (4 << 12)

#endif /* HGTX_PAYLOAD_ENTRY_FLAGS_V1 */
/** ---------------------------------------------------------------------------
 * END: Payload schema entry flags.
 * ------------------------------------------------------------------------- */

/**
 * @note The 'array' flags assume that the array is embedded. Otherwise,
 * @ref HGTX_PAYLOAD_ENTRY_FLAG_POINTER has to be additionally specified. Some
 * combinations may be invalid based on the `HGTX_PAYLOAD_SCHEMA_TYPE_*` this
 * entry is enclosed. For instance, variable length embedded arrays are valid
 * within @ref HGTX_PAYLOAD_SCHEMA_TYPE_DYNAMIC but invalid with
 * @ref HGTX_PAYLOAD_SCHEMA_TYPE_STATIC. See `HGTX_PAYLOAD_SCHEMA_TYPE_*` for
 * additional details.
 */

/* Helper macro to check if an entry represents an array. */
#define HGTX_PAYLOAD_ENTRY_FLAG_IS_ARRAY (\
    HGTX_PAYLOAD_ENTRY_FLAG_ARRAY_FIXED_SIZE | \
    HGTX_PAYLOAD_ENTRY_FLAG_ARRAY_ZERO_TERMINATED | \
    HGTX_PAYLOAD_ENTRY_FLAG_ARRAY_LENGTH_INDEX)

#define HGTX_PAYLOAD_ENTRY_FLAG_ARRAY_TYPE(F) \
    ((F) & HGTX_PAYLOAD_ENTRY_FLAG_IS_ARRAY)


/** ---------------------------------------------------------------------------
 * Types of entries in a payload schema.
 *
 * @note Several of the predefined types contain the size (in bits) in their
 * names. For some data types the size (in bytes) is not fixed and may differ
 * for different platforms/operating systems/compilers. To provide portability,
 * an array of sizes (in bytes) for type 1 to 28 ( @ref
 * HGTX_PAYLOAD_ENTRY_TYPE_CHAR to @ref HGTX_PAYLOAD_ENTRY_TYPE_INFO_ARRAY_SIZE)
 * is passed to the HGTX extension initialization function
 * @ref InitializeInjectionHgtxExtension via the `extInfo` field of
 * @ref hgtxExtModuleInfo_t.
 * ------------------------------------------------------------------------- */
#ifndef HGTX_PAYLOAD_ENTRY_TYPES_V1
#define HGTX_PAYLOAD_ENTRY_TYPES_V1

#define HGTX_PAYLOAD_ENTRY_TYPE_INVALID     0

/**
 * Basic integer types.
 */
#define HGTX_PAYLOAD_ENTRY_TYPE_CHAR        1
#define HGTX_PAYLOAD_ENTRY_TYPE_UCHAR       2
#define HGTX_PAYLOAD_ENTRY_TYPE_SHORT       3
#define HGTX_PAYLOAD_ENTRY_TYPE_USHORT      4
#define HGTX_PAYLOAD_ENTRY_TYPE_INT         5
#define HGTX_PAYLOAD_ENTRY_TYPE_UINT        6
#define HGTX_PAYLOAD_ENTRY_TYPE_LONG        7
#define HGTX_PAYLOAD_ENTRY_TYPE_ULONG       8
#define HGTX_PAYLOAD_ENTRY_TYPE_LONGLONG    9
#define HGTX_PAYLOAD_ENTRY_TYPE_ULONGLONG  10

/**
 * Integer types with explicit size.
 */
#define HGTX_PAYLOAD_ENTRY_TYPE_INT8       11
#define HGTX_PAYLOAD_ENTRY_TYPE_UINT8      12
#define HGTX_PAYLOAD_ENTRY_TYPE_INT16      13
#define HGTX_PAYLOAD_ENTRY_TYPE_UINT16     14
#define HGTX_PAYLOAD_ENTRY_TYPE_INT32      15
#define HGTX_PAYLOAD_ENTRY_TYPE_UINT32     16
#define HGTX_PAYLOAD_ENTRY_TYPE_INT64      17
#define HGTX_PAYLOAD_ENTRY_TYPE_UINT64     18

/**
 * Floating point types
 */
#define HGTX_PAYLOAD_ENTRY_TYPE_FLOAT      19
#define HGTX_PAYLOAD_ENTRY_TYPE_DOUBLE     20
#define HGTX_PAYLOAD_ENTRY_TYPE_LONGDOUBLE 21

/**
 * Size type (`size_t` in C).
 */
#define HGTX_PAYLOAD_ENTRY_TYPE_SIZE       22

/**
 * Any address, e.g. `void*`. If the pointer type matters, use the flag @ref
 * HGTX_PAYLOAD_ENTRY_FLAG_POINTER and the respective type instead.
 */
#define HGTX_PAYLOAD_ENTRY_TYPE_ADDRESS    23

/**
 * Special character types.
 */
#define HGTX_PAYLOAD_ENTRY_TYPE_WCHAR      24 /* wide character (since C90) */
#define HGTX_PAYLOAD_ENTRY_TYPE_CHAR8      25 /* since C2x and C++20 */
#define HGTX_PAYLOAD_ENTRY_TYPE_CHAR16     26
#define HGTX_PAYLOAD_ENTRY_TYPE_CHAR32     27

/**
 * There is type size and alignment information for all previous types.
 */
#define HGTX_PAYLOAD_ENTRY_TYPE_INFO_ARRAY_SIZE (HGTX_PAYLOAD_ENTRY_TYPE_CHAR32 + 1)

/**
 * Store raw 8-bit binary data. As with `char`, 1-byte alignment is assumed.
 * Typically, a tool will display this as hex or binary.
 */
#define HGTX_PAYLOAD_ENTRY_TYPE_BYTE       32

/**
 * These types do not have standardized equivalents. It is assumed that the
 * number at the end corresponds to the bits used to store the value and that
 * the alignment corresponds to standardized types of the same size.
 * A tool may not support these types.
 */
#define HGTX_PAYLOAD_ENTRY_TYPE_INT128     33
#define HGTX_PAYLOAD_ENTRY_TYPE_UINT128    34

#define HGTX_PAYLOAD_ENTRY_TYPE_FLOAT16    42
#define HGTX_PAYLOAD_ENTRY_TYPE_FLOAT32    43
#define HGTX_PAYLOAD_ENTRY_TYPE_FLOAT64    44
#define HGTX_PAYLOAD_ENTRY_TYPE_FLOAT128   45

#define HGTX_PAYLOAD_ENTRY_TYPE_BF16       50
#define HGTX_PAYLOAD_ENTRY_TYPE_TF32       52

/**
 * Data types are as defined by HGTXv3 core.
 */
#define HGTX_PAYLOAD_ENTRY_TYPE_CATEGORY   68 /* uint32_t */
#define HGTX_PAYLOAD_ENTRY_TYPE_COLOR_ARGB 69 /* uint32_t */

/**
 * The scope of events or counters (see `hgtxScopeRegister`).
 */
#define HGTX_PAYLOAD_ENTRY_TYPE_SCOPE_ID   70 /* uint64_t */

/**
 * Process ID as scope.
 */
#define HGTX_PAYLOAD_ENTRY_TYPE_PID_UINT32 71
#define HGTX_PAYLOAD_ENTRY_TYPE_PID_UINT64 72

/**
 * Thread ID as scope.
 */
#define HGTX_PAYLOAD_ENTRY_TYPE_TID_UINT32 73
#define HGTX_PAYLOAD_ENTRY_TYPE_TID_UINT64 74

/**
 * \brief String types.
 *
 * If no flags are set for the entry and `arrayOrUnionDetail > 0`, the entry is
 * assumed to be a fixed-size string with the given length, embedded in the payload.
 * `HGTX_PAYLOAD_ENTRY_FLAG_ARRAY_FIXED_SIZE` is redundant for fixed-size strings.
 *
 * Setting the flag `HGTX_PAYLOAD_ENTRY_FLAG_ARRAY_ZERO_TERMINATED` specifies a
 * zero-terminated string. If `arrayOrUnionDetail > 0`, the entry is handled as
 * a zero-terminated array of fixed-size strings.
 *
 * Setting the flag `HGTX_PAYLOAD_ENTRY_FLAG_ARRAY_LENGTH_INDEX` specifies a
 * variable-length string with the length given in the entry specified by the
 * field `arrayOrUnionDetail`.
 */
#define HGTX_PAYLOAD_ENTRY_TYPE_CSTRING       75 /* `char*`, system LOCALE */
#define HGTX_PAYLOAD_ENTRY_TYPE_CSTRING_UTF8  76
#define HGTX_PAYLOAD_ENTRY_TYPE_CSTRING_UTF16 77
#define HGTX_PAYLOAD_ENTRY_TYPE_CSTRING_UTF32 78

/**
 * The entry value is of type @ref hgtxStringHandle_t returned by
 * @ref hgtxDomainRegisterString.
 */
#define HGTX_PAYLOAD_ENTRY_TYPE_HGTX_REGISTERED_STRING_HANDLE 80

/**
 * This type marks the union selector member (entry index) in schemas used by
 * a union with internal selector.
 * See @ref HGTX_PAYLOAD_SCHEMA_TYPE_UNION_WITH_INTERNAL_SELECTOR.
 */
#define HGTX_PAYLOAD_ENTRY_TYPE_UNION_SELECTOR 100

/**
 * \brief Predefined schema ID for payload data that is referenced in another payload.
 *
 * This schema ID can be used in @ref hgtxPayloadData_t::schema_id to indicate that the
 * payload is a blob of memory which other payload entries may point into.
 * A tool will not expose this payload directly.
 *
 * This schema ID cannot be used as schema entry type!
 */
#define HGTX_TYPE_PAYLOAD_SCHEMA_REFERENCED 1022

/**
 * \brief Predefined schema ID for raw payload data.
 *
 * This schema ID can be used in @ref hgtxPayloadData_t::schema_id to indicate
 * that the payload is a blob, which can be shown with an arbitrary data viewer.
 * This schema ID cannot be used as schema entry type!
 */
#define HGTX_TYPE_PAYLOAD_SCHEMA_RAW        1023

/* Custom (static) schema IDs. */
#define HGTX_PAYLOAD_SCHEMA_ID_STATIC_START  (1 << 24)

/* Dynamic schema IDs (generated by the tool) start here. */
#define HGTX_PAYLOAD_SCHEMA_ID_DYNAMIC_START (HGTX_STATIC_CAST(uint64_t, 1) << 32)

#endif /* HGTX_PAYLOAD_ENTRY_TYPES_V1 */
/** ---------------------------------------------------------------------------
 * END: Payload schema entry types.
 * ------------------------------------------------------------------------- */


#ifndef HGTX_PAYLOAD_SCHEMA_TYPES_V1
#define HGTX_PAYLOAD_SCHEMA_TYPES_V1

/**
 * \brief The payload schema type.
 *
 * A schema can be either of the following types. It is set with
 * @ref hgtxPayloadSchemaAttr_t::type.
 */
#define HGTX_PAYLOAD_SCHEMA_TYPE_INVALID                      0
#define HGTX_PAYLOAD_SCHEMA_TYPE_STATIC                       1
#define HGTX_PAYLOAD_SCHEMA_TYPE_DYNAMIC                      2
#define HGTX_PAYLOAD_SCHEMA_TYPE_UNION                        3
#define HGTX_PAYLOAD_SCHEMA_TYPE_UNION_WITH_INTERNAL_SELECTOR 4

#endif /* HGTX_PAYLOAD_SCHEMA_TYPES_V1 */


#ifndef HGTX_PAYLOAD_SCHEMA_FLAGS_V1
#define HGTX_PAYLOAD_SCHEMA_FLAGS_V1

/**
 * \brief Flags for static and dynamic schemas.
 *
 * The schema flags are used with @ref hgtxPayloadSchemaAttr_t::flags.
 */
#define HGTX_PAYLOAD_SCHEMA_FLAG_NONE           0

/**
 * This flag indicates that a schema and the corresponding payloads can
 * contain fields which require a deep copy.
 */
#define HGTX_PAYLOAD_SCHEMA_FLAG_DEEP_COPY      (1 << 1)

/**
 * This flag indicates that a schema and the corresponding payload can be
 * referenced by another payload of the same event. If the schema is not
 * intended to be visualized directly, it is possible use
 * @ref HGTX_TYPE_PAYLOAD_SCHEMA_REFERENCED instead.
 */
#define HGTX_PAYLOAD_SCHEMA_FLAG_REFERENCED     (1 << 2)

/**
 * The schema defines a counter group. An HGTX handler can expect that the schema
 * contains entries with counter semantics.
 */
#define HGTX_PAYLOAD_SCHEMA_FLAG_COUNTER_GROUP  (1 << 3)

/**
 * The schema defines a range or marker. An HGTX handler can expect that the
 * schema contains a message and timestamp(s).
 */
#define HGTX_PAYLOAD_SCHEMA_FLAG_RANGE_PUSHPOP  (2 << 3)
#define HGTX_PAYLOAD_SCHEMA_FLAG_RANGE_STARTEND (3 << 3)
#define HGTX_PAYLOAD_SCHEMA_FLAG_MARK           (4 << 3)

#endif /* HGTX_PAYLOAD_SCHEMA_FLAGS_V1 */


#ifndef HGTX_PAYLOAD_SCHEMA_ATTR_FIELDS_V1
#define HGTX_PAYLOAD_SCHEMA_ATTR_FIELDS_V1

/**
 * The values allow the valid fields in @ref hgtxPayloadSchemaAttr_t to be
 * specified via setting the field `fieldMask`.
 */
#define HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_NAME        (1 << 1)
#define HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_TYPE        (1 << 2)
#define HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_FLAGS       (1 << 3)
#define HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_ENTRIES     (1 << 4)
#define HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_NUM_ENTRIES (1 << 5)
#define HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_STATIC_SIZE (1 << 6)
#define HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_ALIGNMENT   (1 << 7)
#define HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_SCHEMA_ID   (1 << 8)
#define HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_EXTENSION   (1 << 9)

#endif /* HGTX_PAYLOAD_SCHEMA_ATTR_FIELDS_V1 */


#ifndef HGTX_PAYLOAD_ENUM_ATTR_FIELDS_V1
#define HGTX_PAYLOAD_ENUM_ATTR_FIELDS_V1

/**
 * The values are used to set the field `fieldMask` and specify which fields in
 * @ref hgtxPayloadEnumAttr_t are set.
 */
#define HGTX_PAYLOAD_ENUM_ATTR_FIELD_NAME        (1 << 1)
#define HGTX_PAYLOAD_ENUM_ATTR_FIELD_ENTRIES     (1 << 2)
#define HGTX_PAYLOAD_ENUM_ATTR_FIELD_NUM_ENTRIES (1 << 3)
#define HGTX_PAYLOAD_ENUM_ATTR_FIELD_SIZE        (1 << 4)
#define HGTX_PAYLOAD_ENUM_ATTR_FIELD_SCHEMA_ID   (1 << 5)
#define HGTX_PAYLOAD_ENUM_ATTR_FIELD_EXTENSION   (1 << 6)

#endif /* HGTX_PAYLOAD_ENUM_ATTR_FIELDS_V1 */

/**
 * An HGTX scope specifies the execution scope or source of events or counters.
 * A tool determines the value for a predefined scope when the sample is taken.
 */
#ifndef HGTX_SCOPES_V1
#define HGTX_SCOPES_V1

#define HGTX_SCOPE_NONE                    0 /* No scope specified. */
#define HGTX_SCOPE_ROOT                    1 /* The root in a hierarchy. */

/* Hardware events */
#define HGTX_SCOPE_CURRENT_HW_MACHINE      2 /* Node/machine name */
#define HGTX_SCOPE_CURRENT_HW_SOCKET       3
#define HGTX_SCOPE_CURRENT_HW_CPU_PHYSICAL 4 /* Physical CPU core */
#define HGTX_SCOPE_CURRENT_HW_CPU_LOGICAL  5 /* Logical CPU core */
/* Innermost HW execution context */
#define HGTX_SCOPE_CURRENT_HW_INNERMOST   15

/* Virtualized hardware, virtual machines */
#define HGTX_SCOPE_CURRENT_HYPERVISOR     16
#define HGTX_SCOPE_CURRENT_VM             17
#define HGTX_SCOPE_CURRENT_KERNEL         18
#define HGTX_SCOPE_CURRENT_CONTAINER      19
#define HGTX_SCOPE_CURRENT_OS             20

/* Software scopes */
#define HGTX_SCOPE_CURRENT_SW_PROCESS     21 /* Process scope */
#define HGTX_SCOPE_CURRENT_SW_THREAD      22 /* Thread scope */
/* Innermost SW execution context */
#define HGTX_SCOPE_CURRENT_SW_INNERMOST   31

/** Static (user-provided) scope IDs (feed forward) */
#define HGTX_SCOPE_ID_STATIC_START  (1 << 24)

/* Dynamically (tool) generated scope IDs */
#define HGTX_SCOPE_ID_DYNAMIC_START (HGTX_STATIC_CAST(uint64_t, 1) << 32)

#endif /* HGTX_SCOPES_V1 */

#ifndef HGTX_TIME_V1
#define HGTX_TIME_V1

/**
 * Timestamp source is not known, e.g. NIC or switch. The HGTX handler can
 * assume that at least two synchronization points are created with HGTX
 * instrumentation.
 */
#define HGTX_TIMESTAMP_TYPE_NONE  0

/** The timestamp was provided by the HGTX handler via `hgtxTimestampGet()`. */
#define HGTX_TIMESTAMP_TYPE_TOOL_PROVIDED  1

/** CPU timestamp sources */
#define HGTX_TIMESTAMP_TYPE_CPU_TSC  /* RDTSC on x86, CNTVCT on ARM */ 10
#define HGTX_TIMESTAMP_TYPE_CPU_TSC_NONVIRTUALIZED /* CNTPCT on ARM */ 11
#define HGTX_TIMESTAMP_TYPE_CPU_CLOCK_GETTIME_REALTIME                 12
#define HGTX_TIMESTAMP_TYPE_CPU_CLOCK_GETTIME_REALTIME_COARSE          13
#define HGTX_TIMESTAMP_TYPE_CPU_CLOCK_GETTIME_MONOTONIC                14
#define HGTX_TIMESTAMP_TYPE_CPU_CLOCK_GETTIME_MONOTONIC_RAW            15
#define HGTX_TIMESTAMP_TYPE_CPU_CLOCK_GETTIME_MONOTONIC_COARSE         16
#define HGTX_TIMESTAMP_TYPE_CPU_CLOCK_GETTIME_BOOTTIME                 17
#define HGTX_TIMESTAMP_TYPE_CPU_CLOCK_GETTIME_PROCESS_CPUTIME_ID       18
#define HGTX_TIMESTAMP_TYPE_CPU_CLOCK_GETTIME_THREAD_CPUTIME_ID        19

#define HGTX_TIMESTAMP_TYPE_WIN_QPC      30
#define HGTX_TIMESTAMP_TYPE_WIN_GSTAFT   31
#define HGTX_TIMESTAMP_TYPE_WIN_GSTAFTP  32

#define HGTX_TIMESTAMP_TYPE_C_TIME          40
#define HGTX_TIMESTAMP_TYPE_C_CLOCK         41
#define HGTX_TIMESTAMP_TYPE_C_TIMESPEC_GET  42

#define HGTX_TIMESTAMP_TYPE_CPP_STEADY_CLOCK           50
#define HGTX_TIMESTAMP_TYPE_CPP_HIGH_RESOLUTION_CLOCK  51
#define HGTX_TIMESTAMP_TYPE_CPP_SYSTEM_CLOCK           52
#define HGTX_TIMESTAMP_TYPE_CPP_UTC_CLOCK              53
#define HGTX_TIMESTAMP_TYPE_CPP_TAI_CLOCK              54
#define HGTX_TIMESTAMP_TYPE_CPP_GPS_CLOCK              55
#define HGTX_TIMESTAMP_TYPE_CPP_FILE_CLOCK             56

/** GPU timestamp sources */
#define HGTX_TIMESTAMP_TYPE_GPU_GLOBALTIMER  80 /* e.g. PTIMER */

/** Returned by `hgtxTimeDomainRegister` if time domain registration failed. */
#define HGTX_TIME_DOMAIN_ID_NONE 0

/** Static (user-provided) time domain IDs (feed forward) */
#define HGTX_TIME_DOMAIN_ID_STATIC_START  (1 << 24)

/* Dynamically (tool) generated time domain IDs */
#define HGTX_TIME_DOMAIN_ID_DYNAMIC_START (HGTX_STATIC_CAST(uint64_t, 1) << 32)

/** Timer properties */
#define HGTX_TIMER_FLAG_NONE             0
#define HGTX_TIMER_FLAG_CLOCK_MONOTONIC  (1 << 1)
#define HGTX_TIMER_FLAG_CLOCK_STEADY     (1 << 2)

/** Point in time when the timer starts (its value is 0). */
#define HGTX_TIMER_START_UNKNOWN         0
#define HGTX_TIMER_START_SYSTEM_BOOT     1
#define HGTX_TIMER_START_VM_BOOT         2
#define HGTX_TIMER_START_UNIX_EPOCH      3 /* 1 January 1970 */
#define HGTX_TIMER_START_WIN_FILETIME    4 /* 1 January 1601 */

/**
 * Flags specifying whether it is safe or unsafe to call the timestamp
 * provider after process teardown.
 */
#define HGTX_TIMER_SOURCE_SAFE_CALL_AFTER_PROCESS_TEARDOWN   0
#define HGTX_TIMER_SOURCE_UNSAFE_CALL_AFTER_PROCESS_TEARDOWN 1

#endif /* HGTX_TIME_V1 */

#ifndef HGTX_BATCH_FLAGS_V1
#define HGTX_BATCH_FLAGS_V1

/**
 * Timestamp ordering flags for a batch of deferred events or counters.
 * By default, chronological order by the first timestamp of the event or
 * counter is assumed.
 */
#define HGTX_BATCH_FLAG_TIME_SORTED            0
#define HGTX_BATCH_FLAG_TIME_SORTED_PARTIALLY  (1 << 1)
#define HGTX_BATCH_FLAG_TIME_SORTED_PER_SCOPE  (2 << 1)
#define HGTX_BATCH_FLAG_UNSORTED               (3 << 1)

#endif /* HGTX_BATCH_FLAGS_V1 */

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifndef HGTX_PAYLOAD_TYPEDEFS_V1
#define HGTX_PAYLOAD_TYPEDEFS_V1

/**
 * \brief Size and alignment information for predefined payload entry types.
 *
 * The struct contains the size and the alignment size in bytes. A respective
 * array for the predefined types is passed via hgtxExtModuleInfo_t to the HGTX
 * client/handler. The type (ID) is used as index into this array.
 */
typedef struct hgtxPayloadEntryTypeInfo_v1
{
    uint16_t size;
    uint16_t align;
} hgtxPayloadEntryTypeInfo_t;

/**
 * \brief Binary payload data, size and decoding information.
 *
 * An array of type `hgtxPayloadData_t` is passed to the HGTX event attached to
 * an HGTX event via the `payload.ullvalue` field of HGTX event attributes.
 *
 * The `schemaId` be a predefined schema entry type (`HGTX_PAYLOAD_ENTRY_TYPE*`),
 * a schema ID (statically specified or dynamically created) or one of
 * `HGTX_PAYLOAD_TYPE_REFERENCED` or `HGTX_PAYLOAD_TYPE_RAW`.
 *
 * Setting the size of a payload to `MAX_SIZE` can be useful to reduce the
 * overhead of HGTX instrumentation, when no HGTX handler is attached. However,
 * a tool might not be able to detect the size of a payload and thus skip it.
 * A reasonable use case is a payload that represents a null-terminated
 * C string, where the HGTX handler can call `strlen()`.
 */
typedef struct hgtxPayloadData_v1
{
    /**
     * The schema ID, which defines the layout of the binary data.
     */
    uint64_t    schemaId;

    /**
     * Size of the payload (blob) in bytes. `SIZE_MAX` (`-1`) indicates the tool
     * that it should figure out the size, which might not be possible.
     */
    size_t      size;

    /**
     * Pointer to the binary payload data.
     */
    const void* payload;
} hgtxPayloadData_t;


/**
 * \brief Header of the payload entry's semantic field.
 *
 * If the semantic field of the payload schema entry is set, the first four
 * fields (header) are defined with this type. A tool can iterate through the
 * extensions and check, if it supports (can handle) it.
 */
typedef struct hgtxSemanticsHeader_v1
{
    uint32_t structSize; /** Size of semantic extension struct. */
    uint16_t semanticId;
    uint16_t version;
    const struct hgtxSemanticsHeader_v1* next; /** linked list */
    /* Additional fields are defined by the specific semantic extension. */
} hgtxSemanticsHeader_t;

/**
 * \brief Entry in a schema.
 *
 * A payload schema consists of an array of payload schema entries. It is
 * registered with @ref hgtxPayloadSchemaRegister. `flag` can be set to `0` for
 * simple values, 'type' is the only "required" field. If not set explicitly,
 * all other fields are zero-initialized, which means that the entry has no name
 * and the offset is determined based on self-alignment rules.
 *
 * Example schema:
 *  hgtxPayloadSchemaEntry_t schema[] = {
 *      {0, HGTX_EXT_PAYLOAD_TYPE_UINT8, "one byte"},
 *      {0, HGTX_EXT_PAYLOAD_TYPE_INT32, "four bytes"}
 *  };
 */
typedef struct hgtxPayloadSchemaEntry_v1
{
    /**
     * \brief Flags to augment the basic type.
     *
     * This field allows additional properties of the payload entry to be
     * specified. Valid values are `HGTX_PAYLOAD_ENTRY_FLAG_*`.
     */
    uint64_t       flags;

    /**
     * \brief Predefined payload schema entry type or custom schema ID.
     *
     * Predefined types are `HGTX_PAYLOAD_ENTRY_TYPE_*`. Passing a schema ID
     * enables nesting of schemas.
     */
    uint64_t       type;

    /**
     * \brief Name or label of the payload entry. (Optional)
     *
     * A meaningful name or label can help organizing and interpreting the data.
     */
    const char*    name;

    /**
     * \brief Description of the payload entry. (Optional)
     *
     * A more detail description of the data that is stored with this entry.
     */
    const char*    description;

    /**
     * \brief String length, array length or member selector for union types.
     *
     * If @ref type is a C string type, this field specifies the string length.
     *
     * If @ref flags specify that the entry is an array, this field specifies
     * the array length. See `HGTX_PAYLOAD_ENTRY_FLAG_ARRAY_*` for more details.
     *
     * If @ref type is a union with schema type @ref HGTX_PAYLOAD_SCHEMA_TYPE_UNION
     * (external selection of the union member), this field contains the index
     * (starting with 0) to an entry of integral type in the same schema. The
     * associated field value specifies the selected union member.
     *
     * @note An array of schema type @ref HGTX_PAYLOAD_SCHEMA_TYPE_UNION is not
     * supported. @ref HGTX_PAYLOAD_SCHEMA_TYPE_UNION_WITH_INTERNAL_SELECTOR can
     * be used instead.
     */
    uint64_t       arrayOrUnionDetail;

    /**
     * \brief Offset in the binary payload data (in bytes).
     *
     * This field specifies the byte offset from the base address of the actual
     * binary data (blob) to the start address of the data of this entry.
     *
     * It is recommended (but not required) to provide the offset it. Otherwise,
     * the HGTX handler will determine the offset from natural alignment rules.
     * In some cases, e.g. dynamic schema layouts, the offset cannot be set and
     * has to be determined based on the data of prior entries.
     *
     * Setting the offset can also be used to skip entries during payload parsing.
     */
    uint64_t       offset;

    /**
     * \brief Additional semantics of the payload entry.
     *
     * The field points to the first element in a linked list, which enables
     * multiple semantic extensions.
     */
    const hgtxSemanticsHeader_t* semantics;

    /**
     * \brief Reserved for future use. Do not use it!
     */
    const void*    reserved;
} hgtxPayloadSchemaEntry_t;

/**
 * \brief HGTX payload schema attributes.
 */
typedef struct hgtxPayloadSchemaAttr_v1
{
    /**
     * \brief Mask of valid fields in this struct.
     *
     * Use the `HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_*` defines.
     */
    uint64_t                        fieldMask;

    /**
     * \brief Name of the payload schema. (Optional)
     */
    const char*                     name;

    /**
     * \brief Payload schema type. (Mandatory) \anchor PAYLOAD_TYPE_FIELD
     *
     * Use the `HGTX_PAYLOAD_SCHEMA_TYPE_*` defines.
     */
    uint64_t                        type;

    /**
     * \brief Payload schema flags. (Optional)
     *
     * Flags defined by `HGTX_PAYLOAD_SCHEMA_FLAG_*` can be used to set
     * additional properties of the schema.
     */
    uint64_t                        flags;

    /**
     * \brief Entries of a payload schema. (Mandatory) \anchor ENTRIES_FIELD
     *
     * This field is a pointer to an array of schema entries, each describing a
     * field in a data structure, e.g. in a C struct or union.
     */
    const hgtxPayloadSchemaEntry_t* entries;

    /**
     * \brief Number of entries in the payload schema. (Mandatory)
     *
     * Number of entries in the array of payload entries \ref ENTRIES_FIELD.
     */
    size_t                          numEntries;

    /**
     * \brief The binary payload size in bytes for static payload schemas.
     *
     * If \ref PAYLOAD_TYPE_FIELD is @ref HGTX_PAYLOAD_SCHEMA_TYPE_DYNAMIC this
     * value is ignored. If this field is not specified for a schema of type
     * @ref HGTX_PAYLOAD_SCHEMA_TYPE_STATIC, the size can be automatically
     * determined by a tool.
     */
    size_t                          payloadStaticSize;

    /**
     * \brief The byte alignment for packed structures.
     *
     * If not specified, this field defaults to `0`, which means that the fields
     * in the data structure are not packed and natural alignment rules can be
     * applied.
     */
    size_t                          packAlign;

    /**
     * A static payload schema ID must be unique within the domain,
     * >= HGTX_PAYLOAD_SCHEMA_ID_STATIC_START and
     * < HGTX_PAYLOAD_SCHEMA_ID_DYNAMIC_START
     */
    uint64_t                        schemaId;

    /**
     * Flexible extension for schema attributes.
     * (Do not use. Reserved for future use.)
     */
    void*                           extension;
} hgtxPayloadSchemaAttr_t;

/**
 * \brief This type is used to describe an enumeration.
 *
 * Since the value of an enum entry might not be meaningful for the analysis
 * and/or visualization, a tool can show the name of enum entry instead.
 *
 * An array of this struct is passed to @ref hgtxPayloadEnumAttr_t::entries to be
 * finally registered via @ref hgtxPayloadEnumRegister with the HGTX handler.
 *
 * @note EXPERIMENTAL
 */
typedef struct hgtxPayloadEnum_v1
{
    /**
     * Name of the enum value.
     */
    const char* name;

    /**
     * Value of the enum entry.
     */
    uint64_t    value;

    /**
     * Indicates that this entry sets a specific set of bits, which can be used
     * to define bitsets.
     */
    int8_t      isFlag;
} hgtxPayloadEnum_t;

/**
 * \brief HGTX payload enumeration type attributes.
 *
 * A pointer to this struct is passed to @ref hgtxPayloadEnumRegister.
 */
typedef struct hgtxPayloadEnumAttr_v1
{
    /**
     * Mask of valid fields in this struct. See `HGTX_PAYLOAD_ENUM_ATTR_FIELD_*`.
     */
    uint64_t                 fieldMask;

    /**
     * Name of the enum. (Optional)
     */
    const char*              name;

    /**
     * Entries of the enum. (Mandatory)
     */
    const hgtxPayloadEnum_t* entries;

    /**
     * Number of entries in the enum. (Mandatory)
     */
    size_t                   numEntries;

    /**
     * Size of enumeration type in bytes
     */
    size_t                   sizeOfEnum;

    /**
     * A static payload schema ID must be unique within the domain,
     * >= HGTX_PAYLOAD_SCHEMA_ID_STATIC_START and
     * < HGTX_PAYLOAD_SCHEMA_ID_DYNAMIC_START
     */
    uint64_t                 schemaId;

    /**
     * Flexible extension for enumeration attributes.
     * (Do not use. Reserved for future use.)
     */
    void*                    extension;
} hgtxPayloadEnumAttr_t;

typedef struct hgtxScopeAttr_v1
{
    size_t      structSize;

    /**
     * Path delimited by '/' characters, relative to parentScope. Leading
     * slashes are ignored. Nodes in the path may use name[key] syntax to
     * indicate an array of sibling nodes, which may be combined with other
     * non-array nodes or different arrays at the same scope. Node names should
     * be UTF8 printable characters. '\' has to be used to escape '/', '[', and
     * ']' characters in node names. An empty C string "" and `NULL` are valid
     * inputs and treated equivalently.
     *
     * A GPU can be specified using its
     * - unique identifier (UUID) with "GPU[UUID:#]",
     * - HGGC device ID (sensitive to HGGC_VISIBLE_DEVICES) with "GPU[HGGCID:#]",
     * - HGML (ppu-smi) device ID with "GPU[PPUSMI:#]"
     * (replace `#` with the actual device ID).
     * For display purposes, a tool is recommended to show a pretty name.
     */
    const char* path;

    /** Identifier of the parent scope, to which `path` is appended. */
    uint64_t    parentScope;

    /**
     * Static scope ID. Must be unique within the domain,
     * >= HGTX_SCOPE_ID_STATIC_START, and < HGTX_SCOPE_ID_DYNAMIC_START.
     * Use HGTX_SCOPE_NONE to let the tool create a (dynamic) scope ID.
     */
    uint64_t    scopeId;
} hgtxScopeAttr_t;

#endif /* HGTX_PAYLOAD_TYPEDEFS_V1 */

#ifndef HGTX_PAYLOAD_TYPEDEFS_DEFERRED_V1
#define HGTX_PAYLOAD_TYPEDEFS_DEFERRED_V1

/** Attributes of an HGTX time domain. */
typedef struct hgtxTimeDomainAttr_v1
{
    /** Identifyer of the HGTX scope the time domain is associated with. */
    uint64_t scopeId;

    /** Predefined `HGTX_TIMESTAMP_TYPE_*`. */
    uint64_t timestampTypeId;

    /**
     * Static (feed-forward) time domain ID. `0` makes the tool generate the ID.
     * The static schema ID must be >= HGTX_TIME_DOMAIN_ID_STATIC_START and
     * < HGTX_TIME_DOMAIN_ID_DYNAMIC_START
     */
    uint64_t timeDomainId;

    /** Properties of the timer (use HGTX_TIMER_FLAG_*). */
    uint64_t timerFlags;

    /** Ticks per second (0 means unknown). */
    int64_t  timerResolution;

    /** Point in time when the timer starts (use HGTX_TIMER_START_*). */
    uint64_t timerStart;
} hgtxTimeDomainAttr_t;

/** Synchronization point between two time domains. */
typedef struct hgtxSyncPoint_v1
{
    int64_t src;
    int64_t dst;
} hgtxSyncPoint_t;

/**
 * \brief Helper struct to submit a batch of events (marks or ranges).
 *
 * By default, events are assumed to be chronologically sorted by the first
 * timestamp in the event (start time in a range). If the events are not sorted,
 * the `flags` field must be set accordingly (see `HGTX_BATCH_FLAG_*`).
 */
typedef struct hgtxEventBatch_v1
{
    /**
     * Identifier of the data layout of a deferred event in the array of events.
     * Only layouts with static payload size are allowed. The size of an event
     * in the array is specified by the static payload size during the schema
     * registration. The time domain of event timestamps is provided via time
     * semantics in the schema registration.
     */
    uint64_t    eventSchemaId;

    /** Size of the array of deferred events (in bytes). */
    size_t      size;

    /** Pointer to the array of deferred events. */
    const void* events;

    /** Scope of all events or counters in the batch. */
    uint64_t    scope;

    /** Timestamp ordering (sorted, partially sorted, unsorted), etc. */
    uint64_t    flags;

    /** Flexible data which can be referenced by events in the batch. */
    const void* flexData;

    /** Size of the flexible data memory blob. */
    size_t      flexDataSize;

    /**
     * Offset from the `flexData` pointer to the begin of the flexible data
     * in bytes.
     */
    size_t      flexDataOffset;
} hgtxEventBatch_t;

#endif /* HGTX_PAYLOAD_TYPEDEFS_DEFERRED_V1 */

#ifndef HGTX_PAYLOAD_API_FUNCTIONS_V1
#define HGTX_PAYLOAD_API_FUNCTIONS_V1

/**
 * \brief Register a payload schema.
 *
 * @param domain HGTX domain handle.
 * @param attr HGTX payload schema attributes.
 */
HGTX_DECLSPEC uint64_t HGTX_API hgtxPayloadSchemaRegister(
    hgtxDomainHandle_t domain,
    const hgtxPayloadSchemaAttr_t* attr);

/**
 * \brief Register an enumeration type with the payload extension.
 *
 * @param domain HGTX domain handle
 * @param attr HGTX payload enumeration type attributes.
 */
HGTX_DECLSPEC uint64_t HGTX_API hgtxPayloadEnumRegister(
    hgtxDomainHandle_t domain,
    const hgtxPayloadEnumAttr_t* attr);

/**
 * \brief Register a scope.
 *
 * @param domain HGTX domain handle
 * @param attr Scope attributes.
 *
 * @return an identifier for the scope. If the operation was not successful,
 * `HGTX_SCOPE_NONE` is returned.
 */
HGTX_DECLSPEC uint64_t HGTX_API hgtxScopeRegister(
    hgtxDomainHandle_t domain,
    const hgtxScopeAttr_t* attr);

/**
 * \brief Marks an instantaneous event in the application with the attributes
 * being passed via the extended payload.
 *
 * An HGTX handler can assume that the payload contains the event message.
 * Otherwise, it might ignore the event.
 *
 * @param domain HGTX domain handle
 * @param payloadData pointer to an array of structured payloads.
 * @param count number of payload BLOBs.
 */
HGTX_DECLSPEC void HGTX_API hgtxMarkPayload(
    hgtxDomainHandle_t domain,
    const hgtxPayloadData_t* payloadData,
    size_t count);

/**
 * \brief Begin a nested thread range with the attributes being passed via the
 * payload.
 *
 * @param domain HGTX domain handle
 * @param payloadData Pointer to an array of extended payloads.
 * @param count Number of payloads.
 *
 * @return The level of the range being ended. If an error occurs a negative
 * value is returned on the current thread.
 */
HGTX_DECLSPEC int HGTX_API hgtxRangePushPayload(
    hgtxDomainHandle_t domain,
    const hgtxPayloadData_t* payloadData,
    size_t count);

/**
 * \brief End a nested thread range with an additional custom payload.
 *
 * HGTX event attributes passed to this function (via the payloads) overwrite
 * event attributes (message and color) that have been set in the push event.
 * Other payload entries extend the data of the range.
 *
 * @param domain HGTX domain handle
 * @param payloadData pointer to an array of structured payloads.
 * @param count number of payload BLOBs.
 *
 * @return The level of the range being ended. If an error occurs a negative
 * value is returned on the current thread.
 */
HGTX_DECLSPEC int HGTX_API hgtxRangePopPayload(
    hgtxDomainHandle_t domain,
    const hgtxPayloadData_t* payloadData,
    size_t count);

/**
 * \brief Start a thread range with attributes passed via the extended payload.
 *
 * @param domain HGTX domain handle
 * @param payloadData pointer to an array of structured payloads.
 * @param count number of payload BLOBs.
 *
 * @return The level of the range being ended. If an error occurs a negative
 * value is returned on the current thread.
 */
HGTX_DECLSPEC hgtxRangeId_t HGTX_API hgtxRangeStartPayload(
    hgtxDomainHandle_t domain,
    const hgtxPayloadData_t* payloadData,
    size_t count);

/**
 * \brief End a thread range and pass a custom payload.
 *
 * HGTX event attributes passed to this function (via the payloads) overwrite
 * event attributes (message and color) that have been set in the start event.
 * Other payload entries extend the data of the range.
 *
 * @param domain HGTX domain handle
 * @param id The correlation ID returned from a HGTX range start call.
 * @param payloadData pointer to an array of structured payloads.
 * @param count number of payload BLOBs.
 */
HGTX_DECLSPEC void HGTX_API hgtxRangeEndPayload(
    hgtxDomainHandle_t domain,
    hgtxRangeId_t id,
    const hgtxPayloadData_t* payloadData,
    size_t count);

/**
 * @brief Checks if an HGTX domain is enabled.
 *
 * @param domain HGTX domain handle
 * @return 0 if the domain is not enabled.
 */
HGTX_DECLSPEC uint8_t HGTX_API hgtxDomainIsEnabled(
    hgtxDomainHandle_t domain);

#endif /* HGTX_PAYLOAD_API_FUNCTIONS_V1 */

#ifndef HGTX_PAYLOAD_API_FUNCTIONS_DEFERRED_V1
#define HGTX_PAYLOAD_API_FUNCTIONS_DEFERRED_V1

/**
 * Get a timestamp from the HGTX handler or tool. If no tool is attached, the
 * CPU TSC might be returned. No guarantees are made.
 * The returned timestamp is just meant to be used in deferred events/counters.
 */
HGTX_DECLSPEC int64_t HGTX_API hgtxTimestampGet(void);

/**
 * Register a time domain. Associates an HGTX scope with the time domain.
 * Timestamps of HGTX events or counters in the scope are interpreted according
 * to the time domain definitions.
 *
 * @param domain HGTX domain handle.
 * @param timeAttr Time domain attributes (timestamp type, scope, flags, etc.).
 * @return time domain ID.
 */
HGTX_DECLSPEC uint64_t HGTX_API hgtxTimeDomainRegister(
    hgtxDomainHandle_t domain,
    const hgtxTimeDomainAttr_t* timeAttr);

/**
 * Provide the pointer to a function that returns a timestamp.
 * This enables the tool to create time synchronization points.
 *
 * @param domain HGTX domain handle.
 * @param timeDomainId time domain identifier or timestamp type ID, if it is
 *                     unambiguous.
 * @param flags indicates if it is safe to call the timestamp provider after
 *             process teardown.
 * @param timestampProviderFn Pointer to a function that returns a timestamp.
 */
HGTX_DECLSPEC void HGTX_API hgtxTimerSource(
    hgtxDomainHandle_t domain,
    uint64_t timeDomainId,
    uint64_t flags,
    int64_t (*timestampProviderFn)(void));

/**
 * Same as `hgtxTimerSource`, but with an additional data pointer argument.
 *
 * @param domain HGTX domain handle.
 * @param timeDomainId time domain identifier or timestamp type ID, if it is
 *                     unambiguous.
 * @param flags indicates if it is safe to call the timestamp provider after
 *             process teardown.
 * @param timestampProviderFn Pointer to a function that returns a timestamp.
 * @param data Pointer to data that is passed to the timestamp provider function.
 */
HGTX_DECLSPEC void HGTX_API hgtxTimerSourceWithData(
    hgtxDomainHandle_t domain,
    uint64_t timeDomainId,
    uint64_t flags,
    int64_t (*timestampProviderFn)(void* data),
    void* data);

/**
 * Provides a synchronization point between two time domains.
 * Two synchronization points are required to enable a timestamp conversion.
 * The tool must know one of the time domains or it least must be able to chain
 * conversions to enable the conversion between the given timestamps.
 *
 * @param domain HGTX domain handle.
 * @param timeDomainId1 time domain 1 ID or timestamp type ID, if it is
 *                      unambiguous.
 * @param timeDomainId2 time domain 2 ID or timestamp type ID, if it is
 *                      unambiguous.
 * @param timestamp1 Timestamp in the first time domain.
 * @param timestamp2 Timestamp in the second time domain.
 */
HGTX_DECLSPEC void HGTX_API hgtxTimeSyncPoint(
    hgtxDomainHandle_t domain,
    uint64_t timeDomainId1,
    uint64_t timeDomainId2,
    int64_t timestamp1,
    int64_t timestamp2);

/**
 * The same as `hgtxTimeSyncPoint` but with multiple synchronization points.
 *
 * @param domain HGTX domain handle.
 * @param timeDomainIdSrc source time domain ID or timestamp type ID, if it is
 *                        unambiguous.
 * @param timeDomainIdDst destination time domain ID or timestamp type ID, if it
 *                        is unambiguous.
 * @param syncPoints Pointer to an array of synchronization points.
 * @param count Number of synchronization points.
 */
HGTX_DECLSPEC void HGTX_API hgtxTimeSyncPointTable(
    hgtxDomainHandle_t domain,
    uint64_t timeDomainIdSrc,
    uint64_t timeDomainIdDst,
    const hgtxSyncPoint_t* syncPoints,
    size_t count);

/**
 * @brief Pass a conversion factor between two time domains to the HGTX handler.
 *
 * @param domain HGTX domain handle.
 * @param timeDomainIdSrc source time domain ID or timestamp type ID, if it is
 *                        unambiguous.
 * @param timeDomainIdDst destination time domain ID or timestamp type ID, if it
 *                        is unambiguous.
 * @param slope Conversion factor between the two time domains.
 * @param timestampSrc Timestamp in the source time domain.
 * @param timestampDst Timestamp in the destination time domain.
 */
HGTX_DECLSPEC void HGTX_API hgtxTimestampConversionFactor(
    hgtxDomainHandle_t domain,
    uint64_t timeDomainIdSrc,
    uint64_t timeDomainIdDst,
    double slope,
    int64_t timestampSrc,
    int64_t timestampDst);

/**
 * @brief Submit one deferred event.
 *
 * @param domain HGTX domain handle.
 * @param payloadData Pointer to an array of structured payloads.
 * @param numPayloads Number of payloads of the event.
 */
HGTX_DECLSPEC void HGTX_API hgtxEventSubmit(
    hgtxDomainHandle_t domain,
    const hgtxPayloadData_t* payloadData,
    size_t numPayloads);

/**
 * \brief Submit a batch of deferred events in the given domain.
 *
 * @param domain HGTX domain handle.
 * @param eventBatch Pointer to deferred events batch details.
 */
HGTX_DECLSPEC void HGTX_API hgtxEventBatchSubmit(
    hgtxDomainHandle_t domain,
    const hgtxEventBatch_t* eventBatch);

#endif /* HGTX_PAYLOAD_API_FUNCTIONS_DEFERRED_V1 */

/**
 * \brief Callback IDs of API functions in the payload extension.
 *
 * The HGTX handler can use these values to register a handler function. When
 * `InitializeInjectionHgtxExtension(hgtxExtModuleInfo_t* moduleInfo)` is
 * executed, a handler routine can be registered as follows:
 * \code{.c}
 *      moduleInfo->segments->slots[HGTX3EXT_CBID_hgtxPayloadSchemaRegister] =
 *          (intptr_t)PayloadSchemaRegisterHandlerFn;
 * \endcode
 */
#ifndef HGTX_PAYLOAD_CALLBACK_ID_V1
#define HGTX_PAYLOAD_CALLBACK_ID_V1

#define HGTX3EXT_CBID_hgtxPayloadSchemaRegister      0
#define HGTX3EXT_CBID_hgtxPayloadEnumRegister        1
#define HGTX3EXT_CBID_hgtxMarkPayload                2
#define HGTX3EXT_CBID_hgtxRangePushPayload           3
#define HGTX3EXT_CBID_hgtxRangePopPayload            4
#define HGTX3EXT_CBID_hgtxRangeStartPayload          5
#define HGTX3EXT_CBID_hgtxRangeEndPayload            6
#define HGTX3EXT_CBID_hgtxDomainIsEnabled            7
#define HGTX3EXT_CBID_hgtxScopeRegister             12

#endif /* HGTX_PAYLOAD_CALLBACK_ID_V1 */

#ifndef HGTX_PAYLOAD_CALLBACK_ID_DEFERRED_V1
#define HGTX_PAYLOAD_CALLBACK_ID_DEFERRED_V1

#define HGTX3EXT_CBID_hgtxTimestampGet               8
#define HGTX3EXT_CBID_hgtxTimeDomainRegister         9
#define HGTX3EXT_CBID_hgtxTimerSource               10
#define HGTX3EXT_CBID_hgtxTimerSourceWithData       11
#define HGTX3EXT_CBID_hgtxTimeSyncPoint             13
#define HGTX3EXT_CBID_hgtxTimeSyncPointTable        14
#define HGTX3EXT_CBID_hgtxTimestampConversionFactor 15
#define HGTX3EXT_CBID_hgtxEventSubmit               16
#define HGTX3EXT_CBID_hgtxEventBatchSubmit          17

#endif /* HGTX_PAYLOAD_CALLBACK_ID_DEFERRED_V1 */

/*** Helper utilities ***/

/** \brief  Helper macro for safe double-cast of pointer to uint64_t value. */
#ifndef HGTX_POINTER_AS_PAYLOAD_ULLVALUE
# ifdef __cplusplus
# define HGTX_POINTER_AS_PAYLOAD_ULLVALUE(p) \
    static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p))
# else
#define HGTX_POINTER_AS_PAYLOAD_ULLVALUE(p) (HGTX_STATIC_CAST(uint64_t, HGTX_STATIC_CAST(uintptr_t, p))
# endif
#endif

#ifndef HGTX_PAYLOAD_EVTATTR_SET_DATA
/**
 * \brief Helper macro to attach a single payload to an HGTX event attribute.
 *
 * @param evtAttr HGTX event attribute (variable name)
 * @param pldata_addr Address of `hgtxPayloadData_t` variable.
 * @param schema_id HGTX binary payload schema ID.
 * @param pl_addr Address of the (actual) payload.
 * @param sz size of the (actual) payload.
 */
#define HGTX_PAYLOAD_EVTATTR_SET_DATA(evtAttr, pldata_addr, schema_id, pl_addr, sz) \
    (pldata_addr)->schemaId = schema_id; \
    (pldata_addr)->size = sz; \
    (pldata_addr)->payload = pl_addr; \
    (evtAttr).payload.ullValue = HGTX_POINTER_AS_PAYLOAD_ULLVALUE(pldata_addr); \
    (evtAttr).payloadType = HGTX_PAYLOAD_TYPE_EXT; \
    (evtAttr).reserved0 = 1;
#endif /* HGTX_PAYLOAD_EVTATTR_SET_DATA */

#ifndef HGTX_PAYLOAD_EVTATTR_SET_MULTIPLE
/**
 * \brief Helper macro to attach multiple payloads to an HGTX event attribute.
 *
 * @param evtAttr HGTX event attribute (variable name)
 * @param pldata Payload data array (of type `hgtxPayloadData_t`)
 */
#define HGTX_PAYLOAD_EVTATTR_SET_MULTIPLE(evtAttr, pldata) \
    (evtAttr).payloadType = HGTX_PAYLOAD_TYPE_EXT; \
    (evtAttr).reserved0 = sizeof(pldata)/sizeof(hgtxPayloadData_t); \
    (evtAttr).payload.ullValue = HGTX_POINTER_AS_PAYLOAD_ULLVALUE(pldata);
#endif /* HGTX_PAYLOAD_EVTATTR_SET_MULTIPLE */

#ifndef HGTX_PAYLOAD_EVTATTR_SET
/*
 * Do not use this macro directly! It is a helper to attach a single payload to
 * an HGTX event attribute.
 * @warning The HGTX push, start or mark operation must not be in an outer scope.
 */
#define HGTX_PAYLOAD_EVTATTR_SET(evtAttr, schema_id, pl_addr, sz) \
    hgtxPayloadData_t _HGTX_PAYLOAD_DATA_VAR[] = \
        {{schema_id, sz, pl_addr}}; \
    (evtAttr)->payload.ullValue = \
        HGTX_POINTER_AS_PAYLOAD_ULLVALUE(_HGTX_PAYLOAD_DATA_VAR); \
    (evtAttr)->payloadType = HGTX_PAYLOAD_TYPE_EXT; \
    (evtAttr)->reserved0 = 1;
#endif /* HGTX_PAYLOAD_EVTATTR_SET */

#ifndef hgtxPayloadRangePush
/**
 * \brief Helper macro to push a range with extended payload.
 *
 * @param domain HGTX domain handle
 * @param evtAttr pointer to HGTX event attribute.
 * @param schemaId HGTX payload schema ID
 * @param plAddr Pointer to the binary data (actual payload)
 * @param size Size of the binary payload data in bytes.
 */
#define hgtxPayloadRangePush(domain, evtAttr, schemaId, plAddr, size) \
do { \
    HGTX_PAYLOAD_EVTATTR_SET(evtAttr, schemaId, plAddr, size) \
    hgtxDomainRangePushEx(domain, evtAttr); \
} while (0)
#endif /* hgtxPayloadRangePush */

#ifndef hgtxPayloadMark
/**
 * \brief Helper macro to set a marker with extended payload.
 *
 * @param domain HGTX domain handle
 * @param evtAttr pointer to HGTX event attribute.
 * @param schemaId HGTX payload schema ID
 * @param plAddr Pointer to the binary data (actual payload)
 * @param size Size of the binary payload data in bytes.
 */
#define hgtxPayloadMark(domain, evtAttr, schemaId, plAddr, size) \
do { \
    HGTX_PAYLOAD_EVTATTR_SET(evtAttr, schemaId, plAddr, size) \
    hgtxDomainMarkEx(domain, evtAttr); \
} while (0)
#endif /* hgtxPayloadMark */

/* Macros to create versioned symbols. */
#ifndef HGTX_EXT_PAYLOAD_VERSIONED_IDENTIFIERS_V1
#define HGTX_EXT_PAYLOAD_VERSIONED_IDENTIFIERS_V1
#define HGTX_EXT_PAYLOAD_VERSIONED_IDENTIFIER_L3(NAME, VERSION, COMPATID) \
    NAME##_v##VERSION##_bpl##COMPATID
#define HGTX_EXT_PAYLOAD_VERSIONED_IDENTIFIER_L2(NAME, VERSION, COMPATID) \
    HGTX_EXT_PAYLOAD_VERSIONED_IDENTIFIER_L3(NAME, VERSION, COMPATID)
#define HGTX_EXT_PAYLOAD_VERSIONED_ID(NAME) \
    HGTX_EXT_PAYLOAD_VERSIONED_IDENTIFIER_L2(NAME, HGTX_VERSION, HGTX_EXT_PAYLOAD_COMPATID)
#endif /* HGTX_EXT_PAYLOAD_VERSIONED_IDENTIFIERS_V1 */

#ifdef __GNUC__
#pragma GCC visibility push(internal)
#endif

/* Extension types are required for the implementation and the HGTX handler. */
#define HGTX_EXT_TYPES_GUARD
#include "hgtxDetail/hgtxExtTypes.h"
#undef HGTX_EXT_TYPES_GUARD

#ifndef HGTX_NO_IMPL
#define HGTX_EXT_IMPL_PAYLOAD_GUARD
#include "hgtxDetail/hgtxExtImplPayload_v1.h"
#undef HGTX_EXT_IMPL_PAYLOAD_GUARD
#endif /* HGTX_NO_IMPL */

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */
