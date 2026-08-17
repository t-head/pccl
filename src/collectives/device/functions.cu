/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "devcomm.h"
#include "collectives.h"
#include "common.h"


#define NCCL_FUNC5(func, algo, devredop, type, nullify) \
  MACRO_IF(nullify, nullptr, NCCL_FUNC_NAME(func, algo, LL,     devredop, type)), \
  MACRO_IF(nullify, nullptr, NCCL_FUNC_NAME(func, algo, LL128,  devredop, type)), \
  MACRO_IF(nullify, nullptr, NCCL_FUNC_NAME(func, algo, SIMPLE, devredop, type)), \
  MACRO_IF(nullify, nullptr, NCCL_FUNC_NAME(func, algo, SIGNAL, devredop, type)), \
  MACRO_IF(nullify, nullptr, NCCL_FUNC_NAME(func, algo, SIMPLE_BULK, devredop, type)), \
  MACRO_IF(nullify, nullptr, NCCL_FUNC_NAME(func, algo, SIGNAL_BULK, devredop, type)), \
  MACRO_IF(nullify, nullptr, NCCL_FUNC_NAME(func, algo, SIMPLE_BULK_AR, devredop, type))

#define NCCL_FUNC4(func, devredop, type, nullify) \
  NCCL_FUNC5(func, TREE,    devredop, type, nullify), \
  NCCL_FUNC5(func, RING,    devredop, type, nullify), \
  NCCL_FUNC5(func, COLLNET, devredop, type, nullify), \
  NCCL_FUNC5(func, FC, devredop, type, nullify)

// Must be consistent with ncclDataType_t
#define NCCL_FUNCS3A(func, devredop, nullForFloat) \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, uint8_t, 0), \
  NCCL_FUNC4(func, devredop, int32_t, 0), \
  NCCL_FUNC4(func, devredop, uint32_t, 0), \
  NCCL_FUNC4(func, devredop, int64_t, 0), \
  NCCL_FUNC4(func, devredop, uint64_t, 0), \
  NCCL_FUNC4(func, devredop, half, nullForFloat), \
  NCCL_FUNC4(func, devredop, float, nullForFloat), \
  NCCL_FUNC4(func, devredop, double, nullForFloat), \
  NCCL_FUNC4(func, devredop, __ppu_bfloat16, nullForFloat), \
  NCCL_FUNC4(func, devredop, __hg_fp8_e4m3, nullForFloat), \
  NCCL_FUNC4(func, devredop, __hg_fp8_e5m2, nullForFloat)

#define NCCL_FUNCS3B(func, devredop) \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0), \
  NCCL_FUNC4(func, devredop, int8_t, 0)

// Must be consistent with ncclRedOp_t
#define NCCL_FUNCS2A(func) \
  NCCL_FUNCS3A(func, Sum,        /*nullForFloat=*/0), \
  NCCL_FUNCS3A(func, Prod,       /*nullForFloat=*/0), \
  NCCL_FUNCS3A(func, Max,        /*nullForFloat=*/0), \
  NCCL_FUNCS3A(func, Min,        /*nullForFloat=*/0), \
  NCCL_FUNCS3A(func, PreMulSum,  /*nullForFloat=*/0), \
  NCCL_FUNCS3A(func, SumPostDiv, /*nullForFloat=*/1)

#define NCCL_FUNCS2B(func) \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum), \
  NCCL_FUNCS3B(func, Sum)

// Must be consistent with ncclFunc_t
#define NCCL_FUNCS() { \
  NCCL_FUNC_NAME(SendRecv, RING, SIMPLE, Sum, int8_t),\
  NCCL_FUNC_NAME(SendRecv, RING, SIMPLE_BULK, Sum, int8_t),\
  NCCL_FUNC_NAME(SendRecv, RING, SIGNAL, Sum, int8_t),\
  NCCL_FUNC_NAME(SendRecv, RING, SIGNAL_BULK, Sum, int8_t),\
  NCCL_FUNC_NAME(SendRecv, RING, SIMPLE_BULK_AR, Sum, int8_t),\
  NCCL_FUNCS2B(Broadcast), \
  NCCL_FUNCS2A(Reduce), \
  NCCL_FUNCS2B(AllGather), \
  NCCL_FUNCS2A(ReduceScatter), \
  NCCL_FUNCS2A(AllReduce) }

// Must be consistent with the ncclFuncSet enum
__device__ ncclKern_t pcclFuncs[NUM_P2P_FUNC+NCCL_NUM_FUNCTIONS*ncclNumDevRedOps*ncclNumTypes*NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS] = {
// Don't try to initialize the host shadow copy of this device-side global
// variable. There is no host pointer to a device-side function, which
// confuses clang. This will be fixed in the next clang release.
#if defined(__HGGC_ARCH__)
  NCCL_FUNC_NAME(SendRecv, RING, SIMPLE, Sum, int8_t),
  NCCL_FUNC_NAME(SendRecv, RING, SIMPLE_BULK, Sum, int8_t),
  NCCL_FUNC_NAME(SendRecv, RING, SIGNAL, Sum, int8_t),
  NCCL_FUNC_NAME(SendRecv, RING, SIGNAL_BULK, Sum, int8_t),
  NCCL_FUNC_NAME(SendRecv, RING, SIMPLE_BULK_AR, Sum, int8_t),
  NCCL_FUNCS2B(Broadcast),
  NCCL_FUNCS2A(Reduce),
  NCCL_FUNCS2B(AllGather),
  NCCL_FUNCS2A(ReduceScatter),
  NCCL_FUNCS2A(AllReduce)
#endif
};

// Workaround for https://reviews.llvm.org/D55580
__device__ void pcclWorkaroundClangD55580() {}
