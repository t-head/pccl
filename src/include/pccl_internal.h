/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_INTERNAL_H_
#define PCCL_INTERNAL_H_

#include "pccl.h"

#include "pccl_common.h"

#include "pccl_compat.h"

// Internal-only nccl* type aliases used by .cc translation units that keep
// the nccl* spelling internally.
typedef pcclComm_t   ncclComm_t;
typedef pcclResult_t ncclResult_t;
typedef pcclWindow_t ncclWindow_t;
typedef pcclConfig_t ncclConfig_t;
typedef pcclUniqueId ncclUniqueId;
#define NCCL_CONFIG_INITIALIZER PCCL_CONFIG_INITIALIZER
#define NCCL_VERSION(X,Y,Z) (((X) <= 2 && (Y) <= 8) ? (X) * 1000 + (Y) * 100 + (Z) : (X) * 10000 + (Y) * 100 + (Z))
#define NCCL_UNIQUE_ID_BYTES 128
#define NCCL_API_MAGIC 0xcafebeef

constexpr auto ncclSuccess = pcclSuccess;
constexpr auto ncclUnhandledHggcError = pcclUnhandledHggcError;
constexpr auto ncclUnhandledLtOkiError = pcclUnhandledLtOkiError;
constexpr auto ncclSystemError = pcclSystemError;
constexpr auto ncclInternalError = pcclInternalError;
constexpr auto ncclInvalidArgument = pcclInvalidArgument;
constexpr auto ncclInvalidUsage = pcclInvalidUsage;
constexpr auto ncclRemoteError = pcclRemoteError;
constexpr auto ncclInProgress = pcclInProgress;
constexpr auto ncclNumResults = pcclNumResults;

/* Reduction operation selector */
typedef pcclRedOp_dummy_t ncclRedOp_dummy_t;
constexpr auto ncclNumOps_dummy = pcclNumOps_dummy;

typedef pcclRedOp_t ncclRedOp_t;
constexpr auto ncclSum = pcclSum;
constexpr auto ncclProd = pcclProd;
constexpr auto ncclMax = pcclMax;
constexpr auto ncclMin = pcclMin;
constexpr auto ncclAvg = pcclAvg;
constexpr auto ncclNumOps = pcclNumOps;
constexpr auto ncclMaxRedOp = pcclMaxRedOp;

/* Data types */
typedef pcclDataType_t ncclDataType_t;
constexpr auto ncclInt8 = pcclInt8;
constexpr auto ncclChar = pcclChar;
constexpr auto ncclUint8 = pcclUint8;
constexpr auto ncclInt32 = pcclInt32;
constexpr auto ncclInt = pcclInt;
constexpr auto ncclUint32 = pcclUint32;
constexpr auto ncclInt64 = pcclInt64;
constexpr auto ncclUint64 = pcclUint64;
constexpr auto ncclFloat16 = pcclFloat16;
constexpr auto ncclHalf = pcclHalf;
constexpr auto ncclFloat32 = pcclFloat32;
constexpr auto ncclFloat = pcclFloat;
constexpr auto ncclFloat64 = pcclFloat64;
constexpr auto ncclDouble = pcclDouble;
constexpr auto ncclBfloat16 = pcclBfloat16;
constexpr auto ncclFloat8e4m3 = pcclFloat8e4m3;
constexpr auto ncclFloat8e5m2 = pcclFloat8e5m2;
constexpr auto ncclNumTypes = pcclNumTypes;

typedef pcclScalarResidence_t ncclScalarResidence_t;
constexpr auto ncclScalarDevice = pcclScalarDevice;
constexpr auto ncclScalarHostImmediate = pcclScalarHostImmediate;

typedef pcclDebugLogLevel ncclDebugLogLevel;
constexpr auto NCCL_LOG_NONE = PCCL_LOG_ERROR;
constexpr auto NCCL_LOG_VERSION = PCCL_LOG_VERSION;
constexpr auto NCCL_LOG_WARN = PCCL_LOG_WARN;
constexpr auto NCCL_LOG_INFO = PCCL_LOG_INFO;
constexpr auto NCCL_LOG_ABORT = PCCL_LOG_ABORT;
constexpr auto NCCL_LOG_TRACE = PCCL_LOG_TRACE;

typedef pcclDebugLogSubSys ncclDebugLogSubSys;
constexpr auto NCCL_INIT = PCCL_INIT;
constexpr auto NCCL_COLL = PCCL_COLL;
constexpr auto NCCL_P2P = PCCL_P2P;
constexpr auto NCCL_SHM = PCCL_SHM;
constexpr auto NCCL_NET = PCCL_NET;
constexpr auto NCCL_GRAPH = PCCL_GRAPH;
constexpr auto NCCL_TUNING = PCCL_TUNING;
constexpr auto NCCL_EXT = PCCL_EXT;
constexpr auto NCCL_ENV = PCCL_ENV;
constexpr auto NCCL_ALLOC = PCCL_ALLOC;
constexpr auto NCCL_CALL = PCCL_CALL;
constexpr auto NCCL_DUMP = PCCL_DUMP;
constexpr auto NCCL_PROFILE = PCCL_PROFILE;
constexpr auto NCCL_REG = PCCL_REG;
constexpr auto NCCL_RAS = PCCL_RAS;
constexpr auto NCCL_ALL = PCCL_ALL;
constexpr auto NCCL_PROXY = PCCL_PROXY;
constexpr auto NCCL_BOOTSTRAP = PCCL_BOOTSTRAP;
constexpr auto NCCL_INSPECTOR = PCCL_INSPECTOR;

typedef pcclDebugLogger_t ncclDebugLogger_t;

typedef pcclSimInfo_t ncclSimInfo_t;

#define NCCL_WIN_DEFAULT PCCL_WIN_DEFAULT
#define NCCL_WIN_COLL_SYMMETRIC PCCL_WIN_COLL_SYMMETRIC
#define NCCL_WIN_REQUIRED_ALIGNMENT 4096

// Note: pcclDevCommRequirements_t and PCCL_COMM_PROPERTIES_INITIALIZER come from
// pccl_device/core.h. The previous *_pub_t / *_ptr_t aliases against unbuilt
// pccl* types are dropped — they had no callers.

#define NCCL_VERSION_CODE PCCL_VERSION_CODE
#define NCCL_CONFIG_UNDEF_INT PCCL_CONFIG_UNDEF_INT
#define NCCL_WIN_STRICT_ORDERING PCCL_WIN_STRICT_ORDERING
constexpr auto ncclTimeout = pcclTimeout;
constexpr auto NCCL_DESTROY = PCCL_DESTROY;
#define ncclCommRegister pcclCommRegister
#define ncclCommDeregister pcclCommDeregister
#define ncclCommWindowRegister pcclCommWindowRegister
#define ncclCommWindowDeregister pcclCommWindowDeregister

#endif // end include guard
