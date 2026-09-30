/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TUNER_H_
#define NCCL_TUNER_H_

#include "pccl.h"

typedef enum {NCCL_LOG_ERROR=0, NCCL_LOG_VERSION=1, NCCL_LOG_WARN=2, NCCL_LOG_INFO=3, NCCL_LOG_ABORT=4, NCCL_LOG_TRACE=5} pcclDebugLogLevel;
typedef enum {NCCL_INIT=1, NCCL_COLL=2, NCCL_P2P=4, NCCL_SHM=8, NCCL_NET=16, NCCL_GRAPH=32, NCCL_TUNING=64, NCCL_ENV=128, NCCL_ALL=~0} pcclDebugLogSubSys;

typedef void (*pcclDebugLogger_t)(pcclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...);

#define NCCL_NUM_FUNCTIONS 5 // Send/Recv not included for now
typedef enum { pcclFuncBroadcast, pcclFuncReduce, pcclFuncAllGather, pcclFuncReduceScatter, pcclFuncAllReduce, pcclFuncSendRecv} pcclFunc_t;

#define NCCL_NUM_ALGORITHMS 4 // Tree/Ring/CollNet/FC
#define NCCL_ALGO_UNDEF -1
#define NCCL_ALGO_TREE 0
#define NCCL_ALGO_RING 1
#define NCCL_ALGO_COLLNET 2
#define NCCL_ALGO_FC 3

#define NCCL_NUM_PROTOCOLS 6 // Simple/LL/LL128
#define NCCL_PROTO_UNDEF -1
#define NCCL_PROTO_LL 0
#define NCCL_PROTO_LL128 1
#define NCCL_PROTO_SIMPLE 2
#define NCCL_PROTO_SIGNAL 3
#define NCCL_PROTO_SIMPLE_BULK 4
#define NCCL_PROTO_SIGNAL_BULK 5

typedef enum {
    _PPU_UNKNOWN = 0, // default
    _PPU_ZW610 = 1,
    _PPU_ZW810 = 2,
    _PPU_ZW805 = 3,
    _PPU_ZW610E = 4,
    _PPU_ZW810E = 5,
    _PPU_ZW810_H = 6,
    _PPU_ZW805_H = 7
} PPU_TYPE;

// API to be implemented by external tuner
typedef struct {
  // Name of the tuner
  const char* name;

  // Initializes tuner states.
  // nRanks: number of ranks in current communicator. Each communicator initialize its own tuner.
  // nNodes: number of nodes in current communicator.
  // logFunction: a logFunction can be useful to integrate logging together with NCCL core.
  pcclResult_t (*init)(PPU_TYPE ppuType, size_t nRanks, size_t nNodes, pcclDebugLogger_t logFunction);

  // Gets info (algo, protocol, number of ctas and threads) for a given collective.
  // Inputs:
  //   - collType: collective type , e.g., allreduce, allgather…
  //   - nBytes: collective size in bytes
  //   - collNetSupport: whether collnet supports this type
  //   - nvlsSupport: whether nvlink sharp supports this time
  //   - numPipeOps: number of operations in the group
  //
  // Outputs:
  //   - algorithm: selected algorithm to be used for the given collective
  //   - protocol: selected protocol to be used for the given collective
  //   - nChannels: number of channels (hence SMs) to be used.
  //
  // If getCollInfo() does not return pcclSuccess, NCCL will fall back to the
  // default tuning for the given collective.
  // Also, the plugin is allowed to not set any output, or set only the
  // algorithm and protocol, but not only the algorithm or only the protocol.
  // Unset fields will be set automatically by NCCL.
  pcclResult_t (*getCollInfo)(pcclFunc_t collType, int nRanks, size_t nBytes,
                              int collNetSupport, int nvlsSupport, int numPipeOps,
                              int *algorithm, int *protocol, int* nChannels, int* nThreads);

  pcclResult_t (*getChannels)(int* channels);

  // Terminates the plugin and cleans up any resources that the plugin allocated.
  pcclResult_t (*destroy)();
} pcclTuner_v1_t;

typedef pcclTuner_v1_t pcclTuner_t;

#define NCCL_TUNER_PLUGIN_SYMBOL "pcclTunerPlugin_v1"

#define PCCL_PARAM(name, env, default_value) \
pthread_mutex_t pcclParamMutex##name = PTHREAD_MUTEX_INITIALIZER; \
int64_t pcclParam##name() { \
  static_assert(default_value != -1LL, "default value cannot be -1"); \
  static int64_t value = -1LL; \
  pthread_mutex_lock(&pcclParamMutex##name); \
  if (value == -1LL) { \
    value = default_value; \
    char* str = getenv("PCCL_" env); \
    if (str && strlen(str) > 0) { \
      errno = 0; \
      int64_t v = strtoll(str, NULL, 0); \
      if (errno) { \
        INFO(PCCL_ALL,"Invalid value %s for %s, using default %lu.", str, "PCCL_" env, value); \
      } else { \
        value = v; \
        INFO(PCCL_ALL,"%s set by environment to %lu.", "PCCL_" env, value);  \
      } \
    } \
  } \
  pthread_mutex_unlock(&pcclParamMutex##name); \
  return value; \
}

#endif
