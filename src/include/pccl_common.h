/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2023-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_COMMON_H_
#define PCCL_COMMON_H_

#include <cstdint>

// NCCL core profiler callback for network defined events instrumentation
enum {
  ncclProfilerNetEventStart = 0,
  ncclProfilerNetEventStop,
  ncclProfilerNetEventUpdate,
  ncclProfilerNetEventUpdateAndStop,
};

#define NCCL_NUM_FUNCTIONS 5 // Send/Recv not included for now
#define NCCL_NUM_FUNCTIONS_WITH_P2P (NCCL_NUM_FUNCTIONS+1) // Send/Recv included
typedef enum {
  ncclFuncBroadcast = 0,
  ncclFuncReduce = 1,
  ncclFuncAllGather = 2,
  ncclFuncReduceScatter = 3,
  ncclFuncAllReduce = 4,
  ncclFuncSendRecv = 5,
  ncclFuncSend = 6,
  ncclFuncRecv = 7,
  ncclFuncAlltoAll = 8,
  ncclFuncScatter = 9,
  ncclFuncGather = 10,
  ncclNumFuncs = 11
} ncclFunc_t;

#define NCCL_NUM_ALGORITHMS 4 // Tree/Ring/CollNet/FC
#define NCCL_ALGO_UNDEF -1
#define NCCL_ALGO_TREE 0
#define NCCL_ALGO_RING 1
#define NCCL_ALGO_COLLNET 2
#define NCCL_ALGO_FC 3

#define NCCL_NUM_PROTOCOLS 7 // Simple/LL/LL128/Signal/SimpleBulk/SignalBulk/SimpleBulkAR
#define NCCL_PROTO_UNDEF -1
#define NCCL_PROTO_LL 0
#define NCCL_PROTO_LL128 1
#define NCCL_PROTO_SIMPLE 2
#define NCCL_PROTO_SIGNAL 3
#define NCCL_PROTO_SIMPLE_BULK 4
#define NCCL_PROTO_SIGNAL_BULK 5
#define NCCL_PROTO_SIMPLE_BULK_AR 6
typedef enum { ncclTuningDisable, ncclTuningEnable, ncclTuningCustomEnable} ncclTuningStatus_t;

#define PROTO_IS_SIMPLE_CLASS(p) \
  (p == NCCL_PROTO_SIMPLE || p == NCCL_PROTO_SIMPLE_BULK || p == NCCL_PROTO_SIGNAL || p == NCCL_PROTO_SIGNAL_BULK || p == NCCL_PROTO_SIMPLE_BULK_AR)

#define PROTO_DEPEND_ON_ICN(p) \
  (p == NCCL_PROTO_SIMPLE_BULK || p == NCCL_PROTO_SIGNAL || p == NCCL_PROTO_SIGNAL_BULK || p == NCCL_PROTO_SIMPLE_BULK_AR)

typedef enum {
  PCCL_LOG_ERROR = 0,
  PCCL_LOG_VERSION = 1,
  PCCL_LOG_WARN = 2,
  PCCL_LOG_INFO = 3,
  PCCL_LOG_VERBOSE = 4,
  PCCL_LOG_ABORT = 5,
  PCCL_LOG_TRACE = 6
} pcclDebugLogLevel;

typedef enum {
  PCCL_INIT = 0x1,
  PCCL_COLL = 0x2,
  PCCL_P2P = 0x4,
  PCCL_SHM = 0x8,
  PCCL_NET = 0x10,
  PCCL_GRAPH = 0x20,
  PCCL_TUNING = 0x40,
  PCCL_ENV = 0x80,
  PCCL_DUMP = 0x100,
  PCCL_PROFILE = 0x200,
  PCCL_ALLOC = 0x400,
  PCCL_CALL = 0x800,
  PCCL_EXT = 0x1000,
  PCCL_PROXY = 0x2000,
  PCCL_REG = 0x4000,
  PCCL_RAS = 0x8000,
  PCCL_DESTROY = 0x10000,
  PCCL_INSPECTOR = 0x100000, // big number to avoid short-term conflicts
  PCCL_ALL = ~0
} pcclDebugLogSubSys;

typedef void (*pcclDebugLogger_t)(pcclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...);

#endif
