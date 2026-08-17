/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_COLLECTIVES_H_
#define NCCL_COLLECTIVES_H_

#define NUM_P2P_FUNC 5
#define FUNC_INDEX_P2P_SIMPLE 0
#define FUNC_INDEX_P2P_SIMPLE_BULK 1
#define FUNC_INDEX_P2P_SIGNAL 2
#define FUNC_INDEX_P2P_SIGNAL_BULK 3
#define FUNC_INDEX_P2P_SIMPLE_BULK_AR 4
#define FUNC_INDEX(func, devredop, ncclType, al, pr) (NUM_P2P_FUNC+(((((func)*ncclNumDevRedOps + (devredop))*ncclNumTypes) + (ncclType))*NCCL_NUM_ALGORITHMS+(al))*NCCL_NUM_PROTOCOLS+(pr))

#define TOTAL_FUNCS_NUM (NUM_P2P_FUNC+NCCL_NUM_FUNCTIONS*ncclNumDevRedOps*ncclNumTypes*NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS)

#define NCCL_FUNC_NAME(func, algo, proto, devredop, type) \
  pcclFunction_##func##_##algo##_##proto##_##devredop##_##type

#define NCCL_KERN_NAME(func, algo, proto, devredop, type) \
  pcclKernel_##func##_##algo##_##proto##_##devredop##_##type

#define NCCL_IMPL_NAME(func, algo, proto) \
  pccl##func##algo##proto

/* Declare all collective operations */
#define DECL5(func, algo, proto, devredop, type) \
  extern __device__ void NCCL_FUNC_NAME(func, algo, proto, devredop, type)(void* args); \
  extern __global__ void NCCL_KERN_NAME(func, algo, proto, devredop, type)(ncclDevKernelArgs1K const args4K); \

#define CONCAT(a,b) a##b
#define MACRO_IF(cond, t, f) CONCAT(MACRO_IF_, cond)(t, f)
#define MACRO_IF_0(t, f) f
#define MACRO_IF_1(t, f) t

#define DECL_P2P(func, algo, devredop, type) \
  DECL5(func, algo, SIMPLE, devredop, type) \
  DECL5(func, algo, SIMPLE_BULK, devredop, type) \
  DECL5(func, algo, SIGNAL, devredop, type) \
  DECL5(func, algo, SIGNAL_BULK, devredop, type) \
  DECL5(func, algo, SIMPLE_BULK_AR, devredop, type)

#define DECL4(func, algo, devredop, type, undef) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, SIMPLE_BULK_AR, devredop, type)) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, SIGNAL_BULK, devredop, type)) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, SIMPLE_BULK, devredop, type)) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, SIGNAL, devredop, type)) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, SIMPLE, devredop, type)) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, LL,     devredop, type)) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, LL128,  devredop, type))

#define DECL3(func, devredop, type, undef) \
  DECL4(func, RING,    devredop, type, undef) \
  DECL4(func, TREE,    devredop, type, undef) \
  DECL4(func, COLLNET, devredop, type, undef) \
  DECL4(func, FC, devredop, type, undef)

#define DECL2(func, devredop, undefForFloat) \
  DECL3(func, devredop, int8_t, /*undef=*/0) \
  DECL3(func, devredop, uint8_t, /*undef=*/0) \
  DECL3(func, devredop, int32_t, /*undef=*/0) \
  DECL3(func, devredop, uint32_t, /*undef=*/0) \
  DECL3(func, devredop, int64_t, /*undef=*/0) \
  DECL3(func, devredop, uint64_t, /*undef=*/0) \
  DECL3(func, devredop, half, /*undef=*/undefForFloat) \
  DECL3(func, devredop, float, /*undef=*/undefForFloat) \
  DECL3(func, devredop, double, /*undef=*/undefForFloat) \
  DECL3(func, devredop, __ppu_bfloat16, /*undef=*/undefForFloat) \
  DECL3(func, devredop, __hg_fp8_e4m3, /*undef=*/undefForFloat) \
  DECL3(func, devredop, __hg_fp8_e5m2, /*undef=*/undefForFloat)


#define DECL(func) \
  DECL2(func, Sum, /*undefForFloat=*/0) \
  DECL2(func, Prod, /*undefForFloat=*/0) \
  DECL2(func, Max, /*undefForFloat=*/0) \
  DECL2(func, Min, /*undefForFloat=*/0) \
  DECL2(func, PreMulSum, /*undefForFloat=*/0) \
  DECL2(func, SumPostDiv, /*undefForFloat=*/1)

#define DECL_ALL \
  DECL2(Broadcast, Sum, /*undefForFloat=*/0) \
  DECL(Reduce) \
  DECL2(AllGather, Sum, /*undefForFloat=*/0) \
  DECL(ReduceScatter) \
  DECL(AllReduce) \
  DECL_P2P(SendRecv, RING, Sum, int8_t)

DECL_ALL

// CHUNKSIZE must be a multiple of SLICESIZE
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4)
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)
#define ALLGATHER_SLICESTEPS (NCCL_STEPS/4)
#define ALLGATHER_CHUNKSTEPS (NCCL_STEPS/2)
#define ALLTOALL_SLICESTEPS 1
#define ALLTOALL_CHUNKSTEPS 1
#define REDUCESCATTER_SLICESTEPS (NCCL_STEPS/4)
#define REDUCESCATTER_CHUNKSTEPS (NCCL_STEPS/2)
#define BROADCAST_SLICESTEPS 1
#define BROADCAST_CHUNKSTEPS 1
#define GATHER_SLICESTEPS 1
#define GATHER_CHUNKSTEPS 1
#define SCATTER_SLICESTEPS 1
#define SCATTER_CHUNKSTEPS 1
#define REDUCE_SLICESTEPS 1
#define REDUCE_CHUNKSTEPS 1
#define SENDRECV_SLICEFACTOR 4
#define NCCL_MAX_SLICE_PER_CHUNK 2

#endif
