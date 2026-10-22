/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*************************************************************************
 * Self-contained plugin SDK header for ext-kernel.
 * Defines all types needed by the ext-kernel plugin without depending
 * on pccl internal headers.
 ************************************************************************/

#ifndef EXT_KERNEL_PCCL_SDK_H_
#define EXT_KERNEL_PCCL_SDK_H_

#include <stdint.h>
#include <stdbool.h>
#include <sys/types.h>
#include <stddef.h>
#include <hggc.h>

#ifdef __cplusplus
extern "C" {
#endif

/*========================================================================
 * Result type
 *========================================================================*/
typedef enum {
  pcclSuccess                 = 0,
  pcclUnhandledHggcError      = 1,
  pcclSystemError             = 2,
  pcclInternalError           = 3,
  pcclInvalidArgument         = 4,
  pcclInvalidUsage            = 5,
  pcclRemoteError             = 6,
  pcclInProgress              = 7,
  pcclNumResults              = 8
} pcclResult_t;

typedef pcclResult_t ncclResult_t;
#define ncclSuccess             pcclSuccess
#define ncclUnhandledHggcError  pcclUnhandledHggcError
#define ncclSystemError         pcclSystemError
#define ncclInternalError       pcclInternalError
#define ncclInvalidArgument     pcclInvalidArgument
#define ncclInvalidUsage        pcclInvalidUsage
#define ncclInProgress          pcclInProgress

/*========================================================================
 * Data types
 *========================================================================*/
typedef enum {
  pcclInt8       = 0, pcclChar       = 0,
  pcclUint8      = 1,
  pcclInt32      = 2, pcclInt        = 2,
  pcclUint32     = 3,
  pcclInt64      = 4,
  pcclUint64     = 5,
  pcclFloat16    = 6, pcclHalf      = 6,
  pcclFloat32    = 7, pcclFloat     = 7,
  pcclFloat64    = 8, pcclDouble    = 8,
  pcclBfloat16   = 9,
  pcclFloat8e4m3 = 10,
  pcclFloat8e5m2 = 11,
  pcclNumTypes   = 12
} pcclDataType_t;

typedef pcclDataType_t ncclDataType_t;
#define ncclHalf     pcclHalf
#define ncclFloat    pcclFloat
#define ncclBfloat16 pcclBfloat16

/*========================================================================
 * Reduction operations
 *========================================================================*/
typedef enum {
  ncclDevSum, ncclDevProd, ncclDevMax, ncclDevMin,
  ncclDevPreMulSum, ncclDevSumPostDiv,
  ncclNumDevRedOps
} pcclDevRedOp_t ;

typedef pcclDevRedOp_t ncclDevRedOp_t;

/*========================================================================
 * Collective function types
 *========================================================================*/
typedef enum {
  ncclFuncBroadcast = 0,
  ncclFuncReduce = 1,
  ncclFuncAllGather = 2,
  ncclFuncReduceScatter = 3,
  ncclFuncAllReduce = 4,
  ncclFuncSendRecv = 5
} ncclFunc_t;

/*========================================================================
 * PPU hardware type
 *========================================================================*/
typedef enum : int {
  _PPU_TYPE_ANY = -1,
  _PPU_UNKNOWN = 0,
  _PPU_ZW610 = 1,
  _PPU_ZW810 = 2,
  _PPU_ZW805 = 3,
  _PPU_ZW610E = 4,
  _PPU_ZW810E = 5,
  _PPU_ZW810_H = 6,
  _PPU_ZW805_H = 7,
  _ZW_M890P = 8,
  _ZW_M890L = 9,
  _ZW_M530 = 10,
  _ZW_M530P = 11,
  _PPU_TYPE_COUNT
} PPU_TYPE;

/*========================================================================
 * ICN remote device type
 *========================================================================*/
typedef enum : int {
  PCCL_ICNLINK_UNKNOWN = 0,
  PCCL_ICNLINK_PPU = 1,
  PCCL_ICNLINK_SWITCH = 2,
} pcclICNRemoteDeviceType_t;

/*========================================================================
 * Debug log levels and subsystems
 *========================================================================*/
typedef enum {
  PCCL_LOG_ERROR = 0,
  PCCL_LOG_VERSION = 1,
  PCCL_LOG_WARN = 2,
  PCCL_LOG_INFO = 3,
  PCCL_LOG_VERBOSE = 4,
  PCCL_LOG_ABORT = 5,
  PCCL_LOG_TRACE = 6
} pcclDebugLogLevel;

typedef pcclDebugLogLevel ncclDebugLogLevel;
#define NCCL_LOG_NONE  PCCL_LOG_ERROR
#define NCCL_LOG_WARN  PCCL_LOG_WARN
#define NCCL_LOG_INFO  PCCL_LOG_INFO

typedef enum {
  PCCL_INIT    = 0x1,
  PCCL_COLL    = 0x2,
  PCCL_P2P     = 0x4,
  PCCL_SHM     = 0x8,
  PCCL_NET     = 0x10,
  PCCL_GRAPH   = 0x20,
  PCCL_TUNING  = 0x40,
  PCCL_ENV     = 0x80,
  PCCL_DUMP    = 0x100,
  PCCL_PROFILE = 0x200,
  PCCL_ALLOC   = 0x400,
  PCCL_CALL    = 0x800,
  PCCL_EXT     = 0x1000,
  PCCL_PROXY   = 0x2000,
  PCCL_REG     = 0x4000,
  PCCL_RAS     = 0x8000,
  PCCL_ALL     = ~0
} pcclDebugLogSubSys;

/* NCCL-compat aliases */
#define NCCL_INIT   PCCL_INIT
#define NCCL_COLL   PCCL_COLL
#define NCCL_P2P    PCCL_P2P
#define NCCL_SHM    PCCL_SHM
#define NCCL_NET    PCCL_NET
#define NCCL_GRAPH  PCCL_GRAPH
#define NCCL_TUNING PCCL_TUNING
#define NCCL_ENV    PCCL_ENV
#define NCCL_EXT    PCCL_EXT
#define NCCL_ALLOC  PCCL_ALLOC
#define NCCL_CALL   PCCL_CALL
#define NCCL_ALL    PCCL_ALL

typedef void (*ncclDebugLogger_t)(pcclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...);

/*========================================================================
 * Constants
 *========================================================================*/
#define NCCL_MAX_LOCAL_RANKS 64

/*========================================================================
 * pcclExtTaskColl — lightweight task descriptor for ext-kernel plugin.
 * Contains only the fields the plugin actually needs.
 * The pccl core converts from its internal ncclTaskColl before calling
 * the plugin interface.
 *========================================================================*/
struct pcclExtTaskColl {
  ncclFunc_t func;
  ncclDevRedOp_t op;
  ncclDataType_t datatype;
  size_t count;
  void const* sendbuff;
  void* recvbuff;
  int regUsed;
  uintptr_t sendbuffOffset;
  uintptr_t recvbuffOffset;
  uintptr_t* sendbuffRmtAddrs;
  uintptr_t* recvbuffRmtAddrs;
};

#ifdef __cplusplus
} // extern "C"
#endif

/*========================================================================
 * Dispatch strategy types (C++ section)
 * Must match src/include/drv_extension.h layout
 *========================================================================*/
#include "ext_kernel_dispatch.h"

/*========================================================================
 * Plugin init info struct — replaces ncclComm_t in plugin interface
 *========================================================================*/
struct pcclExtKernelInitInfo {
  int rank;
  int nRanks;
  int localRanks;
  int compCap;
  PPU_TYPE ppuType;
  pcclICNRemoteDeviceType_t icnRmtType;
  struct ncclDispatchStrategy defaultStrategy;
  const int* topoArray;  // localRanks x localRanks, pre-filled by core
  // Device-visible comm abort flag, set by pcclCommAbort
  uint32_t* abortFlagDev;
};

/*========================================================================
 * Plugin interface
 *========================================================================*/
typedef struct {
  const char* name;

  pcclResult_t (*kernelPluginInit)(const struct pcclExtKernelInitInfo* info, void** params, ssize_t* workspaceSize, ncclDebugLogger_t logfn);
  pcclResult_t (*kernelPluginConfigWorkspace)(void** ptrs, int nGpus, void* params);
  pcclResult_t (*kernelPluginAvailCheck)(const struct pcclExtTaskColl* task, const void* params, bool* usingExtKernel);
  pcclResult_t (*kernelPluginBuildParams)(const struct pcclExtTaskColl* task, void* params);
  pcclResult_t (*kernelPluginExecute)(void* params, hggcStream_t stream);
  pcclResult_t (*kernelPluginDump)(void* params, const char* filename);
  pcclResult_t (*kernelPluginDestroy)(void* params);
} pcclExtKernel_v1_t;

typedef pcclExtKernel_v1_t pcclExtKernel_t;

#define PCCL_EXT_KERNEL_PLUGIN_SYMBOL "pcclExtKernelPlugin_v1"

#endif /* EXT_KERNEL_PCCL_SDK_H_ */
