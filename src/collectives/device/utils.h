/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2024-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef UTILS_H_
#define UTILS_H_

#define as_uint64_t(X) ({                                                      \
  uint64_t __as_uint64_t_val = (*(uint64_t*)(&X));                             \
  __as_uint64_t_val;                                                           \
})

#define as_uint2(X) ({                                                         \
  uint2 __as_uint2_val = (*(uint2*)(&X));                                      \
  __as_uint2_val;                                                              \
})

__inline__ __device__ uint64_t __ppu_global_load_volatile_b64(volatile void* ptr) {
  uint2 val = __ppu_global_load_volatile_b32x2(ptr);
  return as_uint64_t(val);
}

__inline__ __device__ void __ppu_global_store_volatile_b64(volatile void* ptr, uint64_t val) {
  __ppu_global_store_volatile_b32x2(ptr, as_uint2(val));
}

inline __device__ uint64_t volatileLoadU64(volatile uint64_t* ptr) {
  return __ppu_global_load_volatile_b64(ptr);
}

inline __device__ void volatileStoreU64(volatile uint64_t* ptr, uint64_t val) {
  __ppu_global_store_volatile_b64(ptr, val);
}

inline __device__ uint64_t patConnStepLoadVolatile(volatile uint64_t* ptr) {
  return volatileLoadU64(ptr);
}

inline __device__ void patConnStepStoreVolatile(volatile uint64_t* ptr, uint64_t val) {
  volatileStoreU64(ptr, val);
}

inline __device__ void printU64Value(int rank, int bid, int tid,
    const char* u64ValName, const uint64_t u64Val) {
  printf("rank: %d; bid: %d; tid: %d; %s: h32 0x%lx, l32 0x%lx\n",
      rank, bid, tid, u64ValName, u64Val >> 32, u64Val);
}

#endif