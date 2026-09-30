/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef OP128_H_
#define OP128_H_

  inline __device__ void load128(const uint64_t* ptr, uint64_t &v0, uint64_t &v1) {
#ifndef LOAD_BYPASS
    union ncclLLFifoLine data;
    // data.i4 = __ppu_global_load_bulk_volatile_b32x4(ptr);
    data.i4 = __ppu_global_load_volatile_b32x4(ptr);
    v0 = data.v[0];
    v1 = data.v[1];
#else
    v0 = v1 = 0UL;
#endif
  }

  inline __device__ void store128(uint64_t* ptr, uint64_t v0, uint64_t v1) {
#ifndef STORE_BYPASS
    union ncclLLFifoLine data;
    data.v[0] = v0;
    data.v[1] = v1;
    // __ppu_global_store_bulk_volatile_b32x4((volatile void*)(ptr), data.i4);
    __ppu_global_store_volatile_b32x4((volatile void*)(ptr), data.i4);
#endif
  }

  inline __device__ void loadShmem128(volatile uint64_t* shmemAsmPtr, uint64_t &v0, uint64_t &v1) {
#ifndef LOAD_BYPASS
    v0 = shmemAsmPtr[0];
    v1 = shmemAsmPtr[1];
    // union ncclLLFifoLine data;
    // data.i4 = __ppu_shared_load_bulk_b32x4(shmemAsmPtr);
    // v0 = data.v[0];
    // v1 = data.v[1];
#else
    v0 = v1 = 0UL;
#endif
  }

  inline __device__ void storeShmem128(volatile uint64_t* shmemAsmPtr, uint64_t v0, uint64_t v1) {
#ifndef STORE_BYPASS
    shmemAsmPtr[0] = v0;
    shmemAsmPtr[1] = v1;
    // union ncclLLFifoLine data;
    // data.v[0] = v0;
    // data.v[1] = v1;
    // __ppu_shared_store_bulk_b32x4((void*)(shmemAsmPtr), data.i4);
#endif
  }

template<typename T>
inline __device__ void loadShmemMisaligned128(T *ptr, uint64_t &v0, uint64_t &v1) {
#ifndef LOAD_BYPASS
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
      asm("ppu.ld.shared.b32 %0,[%1];" : "=r"(lo) : "l"(ptr4+e+0));
      asm("ppu.ld.shared.b32 %0,[%1];" : "=r"(hi) : "l"(ptr4+e+1));
      tmp4[e] = __funnelshift_r(lo, hi, 8*(int(reinterpret_cast<uintptr_t>(ptr))%4));
    }
  } else if(sizeof(T) == 4) {
    #pragma unroll
    for(int e=0; e < 4; e++)
      asm("ppu.ld.shared.b32 %0,[%1];" : "=r"(tmp4[e]) : "l"(ptr+e));
  } else /*sizeof(T)==8*/ {
    #pragma unroll
    for(int e=0; e < 2; e++)
      asm("ppu.ld.shared.b64 %0,[%1];" : "=l"(tmp8[e]) : "l"(ptr+e));
  }
  v0 = tmp8[0];
  v1 = tmp8[1];
#else
  v0 = v1 = 0UL;
#endif
}

// todo: Implement in a better performing way
template<typename T>
inline __device__ void loadMisaligned128(T *ptr, uint64_t &v0, uint64_t &v1) {
#ifndef LOAD_BYPASS
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
      asm("ppu.ld.global.b32 %0,[%1];" : "=r"(lo) : "l"(ptr4+e+0));
      asm("ppu.ld.global.b32 %0,[%1];" : "=r"(hi) : "l"(ptr4+e+1));
      tmp4[e] = __funnelshift_r(lo, hi, 8*(int(reinterpret_cast<uintptr_t>(ptr))%4));
    }
  } else if(sizeof(T) == 4) {
    #pragma unroll
    for(int e=0; e < 4; e++)
      asm("ppu.ld.global.b32 %0,[%1];" : "=r"(tmp4[e]) : "l"(ptr+e));
  } else /*sizeof(T)==8*/ {
    #pragma unroll
    for(int e=0; e < 2; e++)
      asm("ppu.ld.global.b64 %0,[%1];" : "=l"(tmp8[e]) : "l"(ptr+e));
  }
  v0 = tmp8[0];
  v1 = tmp8[1];
#else
  v0 = v1 = 0UL;
#endif
}
#endif
