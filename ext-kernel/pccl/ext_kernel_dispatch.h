/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*************************************************************************
 * Dispatch strategy types for ext-kernel plugin.
 * Must match layout of structs in src/include/drv_extension.h
 ************************************************************************/

#ifndef EXT_KERNEL_PCCL_DISPATCH_H_
#define EXT_KERNEL_PCCL_DISPATCH_H_

#include <stdint.h>

/* Forward-declare ncclFunc_t if not already defined */
#ifndef EXT_KERNEL_PCCL_SDK_H_
typedef enum {
  ncclFuncBroadcast = 0,
  ncclFuncReduce = 1,
  ncclFuncAllGather = 2,
  ncclFuncReduceScatter = 3,
  ncclFuncAllReduce = 4,
  ncclFuncSendRecv = 5
} ncclFunc_t;
#endif

typedef enum {
  CeNumIdxDefault_t,
  CeNumIdx7Ce_t,
  CeNumIdx6Ce_t,
  CeNumIdxNum_t
} CeNumIdx_t;

struct dispatchStrategyExAD {
  unsigned int blocksPerMultiprocessor;
  unsigned int gridStepX;
  unsigned int gridStepY;
  int subDispatch;
};

struct ncclFuncDispatchStrategy {
  uint32_t dispStrategy;
  uint32_t dispStrategy1;
  uint32_t dispStrategy2;
  uint64_t dispMask;
  uint32_t kernelControl;
  struct dispatchStrategyExAD launchConfig;
};

struct ncclDispatchStrategy {
  uint32_t dispStrategy;
  uint32_t dispStrategy1;
  uint32_t dispStrategy2;
  uint64_t dispMask[CeNumIdxNum_t];
  uint32_t kernelControl;
  struct dispatchStrategyExAD launchConfig;
};

typedef struct {
  ncclFunc_t coll;
  int nNodes;
  int nChannels;
  size_t nBytes;
} dispatchInfo;

/* Resolved at link time from libpccl.so */
#ifndef EXT_KERNEL_PCCL_SDK_H_
typedef int ncclResult_t;
#endif

ncclResult_t setDispatchStrategy(const void* func, struct ncclFuncDispatchStrategy* oldStrategy,
                                 struct ncclDispatchStrategy defaultStrategy, const char* funcName,
                                 int compCap, dispatchInfo dInfo);

#endif /* EXT_KERNEL_PCCL_DISPATCH_H_ */
