/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_BYTEPACK_INCLUDED
#define PCCL_BYTEPACK_INCLUDED

#include <stdint.h>
#include <type_traits>

#include "devcomm.h"

template<typename T>
__device__ __forceinline__ uint32_t cvta_to_shared(T* ptr) {
  return (uint32_t)__cvta_generic_to_shared(ptr);
}
template<typename T>
__device__ __forceinline__ uintptr_t cvta_to_global(T* ptr) {
  return (uintptr_t)__cvta_generic_to_global(ptr);
}

template<typename T>
__device__ __forceinline__ T* cvta_from_shared(uint32_t shptr) {
  T* ans;
  asm("ppu.cvta.shared.u64 %0, %1;" : "=l"(ans) : "l"(uint64_t(shptr)));
  return ans;
}
template<typename T>
__device__ __forceinline__ T* cvta_from_global(uintptr_t gptr) {
  T* ans;
  asm("ppu.cvta.global.u64 %0, %1;" : "=l"(ans) : "l"(gptr));
  return ans;
}

////////////////////////////////////////////////////////////////////////////////
// BytePack<Size>: struct of bytes.

template<int Size>
union BytePack;
template<>
union BytePack<0> {};
template<>
union BytePack<1> {
  uint8_t u8[1], native;
};
template<>
union BytePack<2> {
  BytePack<1> half[2];
  BytePack<1> b1[2];
  uint8_t u8[2];
  uint16_t u16[1], native;
};
template<>
union BytePack<4> {
  BytePack<2> half[2];
  BytePack<1> b1[4];
  BytePack<2> b2[2];
  uint8_t u8[4];
  uint16_t u16[2];
  uint32_t u32[1], native;
};
template<>
union BytePack<8> {
  BytePack<4> half[2];
  BytePack<1> b1[8];
  BytePack<2> b2[4];
  BytePack<4> b4[2];
  uint8_t u8[8];
  uint16_t u16[4];
  uint32_t u32[2];
  uint64_t u64[1], native;
};
template<>
union alignas(16) BytePack<16> {
  BytePack<8> half[2];
  BytePack<1> b1[16];
  BytePack<2> b2[8];
  BytePack<4> b4[4];
  BytePack<8> b8[2];
  uint8_t u8[16];
  uint16_t u16[8];
  uint32_t u32[4];
  uint64_t u64[2];
  uint4 u4;
  ulong2 ul2[1], native;
};
template<int Size>
union BytePack {
  BytePack<Size/2> half[2];
  BytePack<1> b1[Size];
  BytePack<2> b2[Size/2];
  BytePack<4> b4[Size/4];
  BytePack<8> b8[Size/8];
  BytePack<16> b16[Size/16];
  uint8_t u8[Size];
  uint16_t u16[Size/2];
  uint32_t u32[Size/4];
  uint64_t u64[Size/8];
};

template<typename T>
struct BytePackOf {
  static constexpr int Size = sizeof(T);
  using Pack = BytePack<Size>;
};
template<>
struct BytePackOf<BytePack<0>> {
  static constexpr int Size = 0;
  using Pack = BytePack<0>;
};

template<typename T>
__device__ __forceinline__ typename BytePackOf<T>::Pack toPack(T value)  {
  union { typename BytePackOf<T>::Pack p; T v; };
  // Coverity recommends the use of std::move here but, given that T is a POD
  // scalar, a plain copy will be just as efficient.
  // coverity[copy_assignment_call]
  v = value;
  return p;
}

template<typename T>
__device__ __forceinline__ T fromPack(typename BytePackOf<T>::Pack pack)  {
  union { typename BytePackOf<T>::Pack p; T v; };
  p = pack;
  return v;
}

////////////////////////////////////////////////////////////////////////////////
// Load/store of BytePack<?> using integral addresses.

template<int Size> __device__ BytePack<Size> ld_global(uintptr_t addr);
template<int Size> __device__ BytePack<Size> ld_shared(uint32_t addr);
template<int Size> __device__ BytePack<Size> ld_volatile_global(uintptr_t addr);
template<int Size> __device__ BytePack<Size> ld_volatile_shared(uint32_t addr);
template<int Size> __device__ BytePack<Size> ld_relaxed_gpu_global(uintptr_t addr);
template<int Size> __device__ void st_global(uintptr_t addr, BytePack<Size> value);
template<int Size> __device__ void st_shared(uint32_t addr, BytePack<Size> value);
template<int Size> __device__ void st_volatile_global(uintptr_t addr, BytePack<Size> value);
template<int Size> __device__ void st_volatile_shared(uint32_t addr, BytePack<Size> value);
template<int Size> __device__ void st_relaxed_gpu_global(uintptr_t addr, BytePack<Size> value);

template<> __device__ __forceinline__ BytePack<0> ld_global<0>(uintptr_t addr) { return {}; }
template<> __device__ __forceinline__ BytePack<0> ld_shared<0>(uint32_t addr) { return {}; }
template<> __device__ __forceinline__ BytePack<0> ld_volatile_global<0>(uintptr_t addr) { return {}; }
template<> __device__ __forceinline__ BytePack<0> ld_volatile_shared<0>(uint32_t addr) { return {}; }
template<> __device__ __forceinline__ BytePack<0> ld_relaxed_gpu_global<0>(uintptr_t addr) { return {}; }
template<> __device__ __forceinline__ void st_global<0>(uintptr_t addr, BytePack<0> value) {}
template<> __device__ __forceinline__ void st_shared<0>(uint32_t addr, BytePack<0> value) {}
template<> __device__ __forceinline__ void st_volatile_global<0>(uintptr_t addr, BytePack<0> value) {}
template<> __device__ __forceinline__ void st_volatile_shared<0>(uint32_t addr, BytePack<0> value) {}
template<> __device__ __forceinline__ void st_relaxed_gpu_global<0>(uintptr_t addr, BytePack<0> value) {}

// Used to define implementations for above prototypes.
#define DEFINE_ld_st__size_space(bytes, data_cxx_ty, data_ptx_ty, data_reg_ty, space, addr_cxx_ty, addr_reg_ty) \
  template<> \
  __device__ __forceinline__ BytePack<bytes> ld_##space<bytes>(addr_cxx_ty addr) { \
    data_cxx_ty tmp; \
    asm volatile("ppu.ld." #space "." #data_ptx_ty " %0, [%1];" : "="#data_reg_ty(tmp) : #addr_reg_ty(addr) : "memory"); \
    BytePack<bytes> ans; \
    ans.native = tmp; \
    return ans; \
  } \
  template<> \
  __device__ __forceinline__ BytePack<bytes> ld_volatile_##space<bytes>(addr_cxx_ty addr) { \
    data_cxx_ty tmp; \
    asm volatile("ppu.ld.volatile." #space "." #data_ptx_ty " %0, [%1];" : "="#data_reg_ty(tmp) : #addr_reg_ty(addr) : "memory"); \
    BytePack<bytes> ans; \
    ans.native = tmp; \
    return ans; \
  } \
  template<> \
  __device__ __forceinline__ void st_##space<bytes>(addr_cxx_ty addr, BytePack<bytes> value) { \
    data_cxx_ty tmp = value.native; \
    asm volatile("ppu.st." #space "." #data_ptx_ty " [%0], %1;" :: #addr_reg_ty(addr), #data_reg_ty(tmp) : "memory"); \
  } \
  template<> \
  __device__ __forceinline__ void st_volatile_##space<bytes>(addr_cxx_ty addr, BytePack<bytes> value) { \
    data_cxx_ty tmp = value.native; \
    asm volatile("ppu.st.volatile." #space "." #data_ptx_ty " [%0], %1;" :: #addr_reg_ty(addr), #data_reg_ty(tmp) : "memory"); \
  }

#define PTX_relaxed_gpu "volatile"

#define DEFINE_ld_st_gpu_relaxed__size(bytes, data_cxx_ty, data_ptx_ty, data_reg_ty) \
  template<> \
  __device__ __forceinline__ BytePack<bytes> ld_relaxed_gpu_global<bytes>(uintptr_t addr) { \
    data_cxx_ty tmp; \
    asm volatile("ppu.ld." PTX_relaxed_gpu ".global." #data_ptx_ty " %0, [%1];" : "="#data_reg_ty(tmp) : "l"(addr) : "memory"); \
    BytePack<bytes> ans; \
    ans.native = tmp; \
    return ans; \
  } \
  template<> \
  __device__ __forceinline__ void st_relaxed_gpu_global<bytes>(uintptr_t addr, BytePack<bytes> value) { \
    data_cxx_ty tmp = value.native; \
    asm volatile("ppu.st." PTX_relaxed_gpu ".global." #data_ptx_ty " [%0], %1;" :: "l"(addr), #data_reg_ty(tmp) : "memory"); \
  }

#define DEFINE_ld_st__size(bytes, data_cxx_ty, data_ptx_ty, data_reg_ty) \
  DEFINE_ld_st__size_space(bytes, data_cxx_ty, data_ptx_ty, data_reg_ty, global, uintptr_t, l) \
  DEFINE_ld_st__size_space(bytes, data_cxx_ty, data_ptx_ty, data_reg_ty, shared, uint32_t, r) \
  DEFINE_ld_st_gpu_relaxed__size(bytes, data_cxx_ty, data_ptx_ty, data_reg_ty)

// Single-byte types use 4-byte registers since there is no 1-byte register
DEFINE_ld_st__size(1, uint32_t, b8, r)
DEFINE_ld_st__size(2, uint16_t, b16, h)
DEFINE_ld_st__size(4, uint32_t, b32, r)
DEFINE_ld_st__size(8, uint64_t, b64, l)

#undef DEFINE_ld_st__size_space
#undef DEFINE_ld_st__size

#define DEFINE_ld_st_16__space(space, addr_cxx_ty, addr_reg_ty) \
  template<> \
  __device__ __forceinline__ BytePack<16> ld_##space<16>(addr_cxx_ty addr) { \
    BytePack<16> ans; \
    asm volatile("ppu.ld." #space ".v2.b64 {%0,%1}, [%2];" : "=l"(ans.u64[0]), "=l"(ans.u64[1]) : #addr_reg_ty(addr) : "memory"); \
    return ans; \
  } \
  template<> \
  __device__ __forceinline__ BytePack<16> ld_volatile_##space<16>(addr_cxx_ty addr) { \
    BytePack<16> ans; \
    asm volatile("ppu.ld.volatile." #space ".v2.b64 {%0,%1}, [%2];" : "=l"(ans.u64[0]), "=l"(ans.u64[1]) : #addr_reg_ty(addr) : "memory"); \
    return ans; \
  } \
  template<> \
  __device__ __forceinline__ void st_##space<16>(addr_cxx_ty addr, BytePack<16> value) { \
    asm volatile("ppu.st." #space ".v2.b64 [%0], {%1,%2};" :: #addr_reg_ty(addr), "l"(value.u64[0]), "l"(value.u64[1]) : "memory"); \
  } \
  template<> \
  __device__ __forceinline__ void st_volatile_##space<16>(addr_cxx_ty addr, BytePack<16> value) { \
    asm volatile("ppu.st.volatile." #space ".v2.b64 [%0], {%1,%2};" :: #addr_reg_ty(addr), "l"(value.u64[0]), "l"(value.u64[1]) : "memory"); \
  }
DEFINE_ld_st_16__space(global, uintptr_t, l)
DEFINE_ld_st_16__space(shared, uint32_t, r)
#undef DEFINE_ld_st_16__space
#undef PTX_relaxed_gpu

////////////////////////////////////////////////////////////////////////////////
// Atomic load/store using c++ pointers.

__device__ __forceinline__ uint64_t ld_volatile_global(uint64_t* ptr) {
  uint64_t ans;
  asm volatile("ppu.ld.volatile.global.u64 %0, [%1];" : "=l"(ans) : "l"(cvta_to_global(ptr)) : "memory");
  return ans;
}
__device__ __forceinline__ uint64_t ld_relaxed_sys_global(uint64_t* ptr) {
  uint64_t ans;
  asm volatile("ppu.ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(ans) : "l"(cvta_to_global(ptr)) : "memory");
  return ans;
}
__device__ __forceinline__ uint64_t ld_relaxed_gpu_global(uint64_t* ptr) {
  uint64_t ans;
  asm volatile("ppu.ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(ans) : "l"(cvta_to_global(ptr)) : "memory");
  return ans;
}
__device__ __forceinline__ uint64_t ld_acquire_sys_global(uint64_t* ptr) {
  uint64_t ans;
  asm volatile("ppu.ld.acquire.sys.global.u64 %0, [%1];" : "=l"(ans) : "l"(cvta_to_global(ptr)) : "memory");
  return ans;
}

__device__ __forceinline__ void st_volatile_global(uint64_t* ptr, uint64_t val) {
  asm volatile("ppu.st.volatile.global.u64 [%0], %1;" :: "l"(cvta_to_global(ptr)), "l"(val) : "memory");
}
__device__ __forceinline__ void st_relaxed_sys_global(uint64_t* ptr, uint64_t val) {
  asm volatile("ppu.st.relaxed.sys.global.u64 [%0], %1;" :: "l"(cvta_to_global(ptr)), "l"(val) : "memory");
}
__device__ __forceinline__ void st_release_sys_global(uint64_t* ptr, uint64_t val) {
  asm volatile("ppu.st.release.sys.global.u64 [%0], %1;" :: "l"(cvta_to_global(ptr)), "l"(val) : "memory");
}

__device__ __forceinline__ void fence_acq_rel_sys() {
  asm volatile("ppu.fence.acq_rel.sys;" ::: "memory");
}
__device__ __forceinline__ void fence_acq_rel_gpu() {
  asm volatile("ppu.fence.acq_rel.gpu;" ::: "memory");
}

////////////////////////////////////////////////////////////////////////////////
// Multimem stores of BytePack<?>

template<int Size>
__device__ void multimem_st_global(uintptr_t addr, BytePack<Size> val);

// Load pack starting at index in array. Ignore elements past end (length of array).
template<typename Pack, typename T>
__device__ __forceinline__ Pack loadPack(T* ptr, int ix, int end) {
  constexpr int Size = sizeof(Pack);
  ptr += ix;
  int n = end - ix;
  if (alignof(T) == Size && sizeof(T) == Size) {
    return *(Pack*)ptr;
  } else if ((Size+3)/4 + 1 < Size/sizeof(T)) {
    union { Pack ans; uint32_t part[Size/4 + 1]; };
    int misalign = reinterpret_cast<uintptr_t>(ptr) % 4;
    uint32_t* down = reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(ptr) & -uintptr_t(4));
    // ndiv holds the number of aligned 4-byte loads needed to cover the entire input
    int ndiv = (n*sizeof(T) + misalign + 3)/4;
    int imax = min(Size/4 + (misalign > 0), ndiv);
    int i;
    #pragma unroll
    for (i=0; i < Size/4 + 1; i++) {
      if (i < imax) part[i] = down[i];
    }
    if (misalign > 0) {
      #pragma unroll
      for (i=0; i < Size/4; i++) {
        part[i] = __funnelshift_r(part[i], part[i+1], 8*misalign);
      }
    }
    return ans;
  } else {
    union { Pack ans; BytePack<sizeof(T)> part[Size/sizeof(T)]; };
    #pragma unroll
    for (int i=0; i < Size/sizeof(T); i++) {
      if (i < 1 || i < n) part[i] = ((BytePack<sizeof(T)>*)ptr)[i];
    }
    return ans;
  }
}

// Store pack starting at index in array. Ignore elements past end (length of array).
template<typename Pack, typename T>
__device__ __forceinline__ void storePack(T* ptr, int ix, int end, Pack val) {
  constexpr int Size = sizeof(Pack);
  union { Pack tmp; BytePack<sizeof(T)> part[Size/sizeof(T)]; };
  tmp = val;
  ptr += ix;
  int n = end - ix;
  #pragma unroll
  for (int i=0; i < Size/sizeof(T); i++) {
    if (i < 1 || i < n) ((BytePack<sizeof(T)>*)ptr)[i] = part[i];
  }
}

// Warp-uniform memory copy from shared address (not generic) to global memory.
// The number of bytes copied is `min(MaxBytes, nBytesAhead)`, a negative value
// is interpreted as zero. EltSize is the guaranteed alignment of the addresses and sizes.
template <int EltSize, int MaxBytes, bool Multimem, typename IntBytes>
__device__ __forceinline__ void copyGlobalShared_WarpUnrolled(int lane, uintptr_t dstAddr, uint32_t srcAddr,
                                                              IntBytes nBytesAhead) {
  static_assert(std::is_signed<IntBytes>::value, "`IntBytes` must be a signed integral type.");
  int nBytes = min(nBytesAhead, (IntBytes)MaxBytes);
  int nFrontBytes = min(nBytes, (16 - int(dstAddr % 16)) % 16);
  int nMiddleBytes = (nBytes - nFrontBytes) & -16;
  int nBackBytes = (nBytes - nFrontBytes) % 16;

  {
    int backLane = WARP_SIZE - 1 - lane;
    bool hasFront = lane * EltSize < nFrontBytes;
    bool hasBack = backLane * EltSize < nBackBytes;
    int offset = hasFront ? lane * EltSize : (nBytes - (backLane + 1) * EltSize);
    if (hasFront | hasBack) {
      BytePack<EltSize> tmp = ld_shared<EltSize>(srcAddr + offset);
      // Can't use multimem_st since it doesn't support EltSize==2
      st_volatile_global<EltSize>(dstAddr + offset, tmp);
    }
  }

  srcAddr += nFrontBytes;
  int srcMisalign = EltSize < 4 ? (srcAddr % 4) : 0;
  srcAddr += -srcMisalign + lane * 16;
  dstAddr += nFrontBytes + lane * 16;
  nMiddleBytes -= lane * 16;
  #pragma unroll
  for (int u = 0; u < divUp(MaxBytes, WARP_SIZE * 16); u++) {
    if (nMiddleBytes <= 0) break;
    union {
      BytePack<4> b4[4];
      BytePack<16> b16;
    };
    b4[0] = ld_shared<4>(srcAddr + 0 * 4);
    b4[1] = ld_shared<4>(srcAddr + 1 * 4);
    b4[2] = ld_shared<4>(srcAddr + 2 * 4);
    b4[3] = ld_shared<4>(srcAddr + 3 * 4);
    if (srcMisalign != 0) {
      BytePack<4> b4_4 = ld_shared<4>(srcAddr + 4 * 4);
      b4[0].native = __funnelshift_r(b4[0].native, b4[1].native, srcMisalign * 8);
      b4[1].native = __funnelshift_r(b4[1].native, b4[2].native, srcMisalign * 8);
      b4[2].native = __funnelshift_r(b4[2].native, b4[3].native, srcMisalign * 8);
      b4[3].native = __funnelshift_r(b4[3].native, b4_4.native, srcMisalign * 8);
    }
    if (Multimem) multimem_st_global<16>(dstAddr, b16);
    else st_volatile_global<16>(dstAddr, b16);

    srcAddr += WARP_SIZE * 16;
    dstAddr += WARP_SIZE * 16;
    nMiddleBytes -= WARP_SIZE * 16;
  }
}

#endif // PCCL_BYTEPACK_INCLUDED

// LL/LL128 FIFO 128-bit primitives. Mutually exclusive with
// symmetric/sym_op128.h, which defines OP128_H_ for the symmetric variants.
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

  inline __device__ uint64_t* shmemCvtPtr(volatile uint64_t* shmemGenericPtr) {
    uint64_t* shmemAsmPtr;
    asm volatile("ppu.cvta.to.shared.u64 %0, %1;" : "=l"(shmemAsmPtr) : "l"(shmemGenericPtr) : "memory");
    return shmemAsmPtr;
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

#endif // OP128_H_
