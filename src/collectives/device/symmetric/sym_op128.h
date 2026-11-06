/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef OP128_H_
#define OP128_H_

#include <type_traits>

inline __device__ void load128(const uint64_t* ptr, uint64_t &v0, uint64_t &v1) {
  asm volatile("ppu.ld.volatile.global.v2.u64 {%0,%1}, [%2];"
      : "=l"(v0), "=l"(v1) : "l"(ptr) : "memory");
}

inline __device__ void store128(uint64_t* ptr, uint64_t v0, uint64_t v1) {
  asm volatile("ppu.st.volatile.global.v2.u64 [%2], {%0,%1};"
      :: "l"(v0), "l"(v1), "l"(ptr) : "memory");
}

inline __device__ uint64_t* shmemCvtPtr(volatile uint64_t* shmemGenericPtr) {
  uint64_t* shmemAsmPtr;
  asm volatile("ppu.cvta.to.shared.u64 %0, %1;" : "=l"(shmemAsmPtr) : "l"(shmemGenericPtr) : "memory");
  return shmemAsmPtr;
}

inline __device__ void loadShmem128(uint64_t* shmemAsmPtr, uint64_t &v0, uint64_t &v1) {
  asm volatile("ppu.ld.volatile.shared.v2.u64 {%0,%1}, [%2];"
      : "=l"(v0), "=l"(v1) : "l"(shmemAsmPtr) : "memory");
}

inline __device__ void storeShmem128(uint64_t* shmemAsmPtr, uint64_t v0, uint64_t v1) {
  asm volatile("ppu.st.volatile.shared.v2.u64 [%2], {%0,%1};"
      :: "l"(v0), "l"(v1), "l"(shmemAsmPtr) : "memory");
}

template<typename T>
inline __device__ void loadShmemMisaligned128(T *ptr, uint64_t &v0, uint64_t &v1) {
  union {
    uint32_t tmp4[4];
    uint64_t tmp8[2];
  };
  if(sizeof(T) < 4) {
    uint32_t *ptr4 = reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(ptr) & -uintptr_t(4));
    #pragma unroll
    for(int e=0; e < 4; e++) {
      // Produce 4 bytes of sub-register type by reading 2 4-byte
      // aligned values and shifting.
      uint32_t lo, hi;
      asm volatile("ppu.ld.shared.b32 %0,[%1];" : "=r"(lo) : "l"(ptr4+e+0) : "memory");
      asm volatile("ppu.ld.shared.b32 %0,[%1];" : "=r"(hi) : "l"(ptr4+e+1) : "memory");
      tmp4[e] = __funnelshift_r(lo, hi, 8*(int(reinterpret_cast<uintptr_t>(ptr))%4));
    }
  }
  else if(sizeof(T) == 4) {
    #pragma unroll
    for(int e=0; e < 4; e++)
      asm volatile("ppu.ld.shared.b32 %0,[%1];" : "=r"(tmp4[e]) : "l"(ptr+e) : "memory");
  }
  else /*sizeof(T)==8*/ {
    #pragma unroll
    for(int e=0; e < 2; e++)
      asm volatile("ppu.ld.shared.b64 %0,[%1];" : "=l"(tmp8[e]) : "l"(ptr+e) : "memory");
  }
  v0 = tmp8[0];
  v1 = tmp8[1];
}

#include "op128.h"

#endif
