/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_REDUCE_KERNEL_H_
#define NCCL_REDUCE_KERNEL_H_

#include "sym_op128.h"
#include <limits>
#include <type_traits>

template<typename T>
struct IsFloatingPoint: std::false_type {};
template<>
struct IsFloatingPoint<half>: std::true_type {};
template<>
struct IsFloatingPoint<__ppu_bfloat16>: std::true_type {};
template<>
struct IsFloatingPoint<float>: std::true_type {};
template<>
struct IsFloatingPoint<double>: std::true_type {};

////////////////////////////////////////////////////////////////////////////////
// The reduction function classes. All classes must:
//  1. Expose the `EltType` typedef.
//  2. Have constructor taking no arguments (default constructible).
//  3. Have constructor taking `uint64_t opArg`.

template<typename T>
struct FuncCopy { using EltType = T; __device__ __forceinline__ FuncCopy(uint64_t opArg=0) {}; };
template<typename T>
struct FuncSum  { using EltType = T; __device__ __forceinline__ FuncSum(uint64_t opArg=0) {}; };
template<typename T>
struct FuncProd { using EltType = T; __device__ __forceinline__ FuncProd(uint64_t opArg=0) {}; };

template<typename T>
struct FuncMinMax {
  using EltType = T;
  BytePack<sizeof(T)> xormask; // only used by integers
  bool isMinNotMax; // only used by floats
  __device__ __forceinline__ FuncMinMax(uint64_t opArg=0) {
    xormask.native = opArg;
    isMinNotMax = (opArg&1)==0;
  }
};

template<typename T> struct FuncPreMulSum;
template<typename T> struct FuncSumPostDiv;

////////////////////////////////////////////////////////////////////////////////
// Trait class for handling the reduction argument.

template<typename Fn>
struct RedOpArg { // default case: no argument
  static constexpr bool ArgUsed = false;
  __device__ __forceinline__ static uint64_t loadArg(void *ptr) { return 0; }
};

template<typename T>
struct RedOpArg<FuncMinMax<T>> {
  static constexpr bool ArgUsed = true;
  __device__ __forceinline__ static uint64_t loadArg(void *ptr) {
    union { uint64_t u64; T val; };
    u64 = 0;
    val = *(T*)ptr;
    return u64;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Trait classes for reduction functions. Given a function (FuncSum, etc.)
// and a number of elements in a pack, will reduce, preOp, or postOp a pack
// of elements. These classes are intended to be specialized for specific
// combinations of reduction function and pack size.

template<typename A, typename B, int EltPerPackA>
struct Apply_Cast/*{
  static BytePack<EltPerPackA*sizeof(B)/sizeof(A)> cast(BytePack<EltPerPackA*sizeof(A)> a);
}*/;

template<typename Fn, int EltPerPack>
struct Apply_Reduce /*{
  static BytePack<EltPerPack*sizeof(T)> reduce(
    Fn fn, BytePack<EltPerPack*sizeof(T)> a, BytePack<EltPerPack*sizeof(T)> b
  );
}*/;
template<typename Fn, int EltPerPack>
struct Apply_PreOp/*{
  static constexpr bool IsIdentity;
  static BytePack<EltPerPack*sizeof(T)> preOp(Fn fn, BytePack<EltPerPack*sizeof(T)> a);
}*/;
template<typename Fn, int EltPerPack>
struct Apply_PostOp/*{
  static constexpr bool IsIdentity;
  static BytePack<EltPerPack*sizeof(T)> postOp(Fn fn, BytePack<EltPerPack*sizeof(T)> a);
}*/;

template<int BytePerPack, bool IsLocal, int CachePolicy, int Rtmd>
struct Apply_Fetch/*{
  static BytePack<BytePerPack> load(Fn fn, uintptr_t addr);
}*/;

template<int BytePerPack, bool IsLocal, int CachePolicy, int Rtmd>
struct Apply_Store/*{
  static BytePack<BytePerPack> store(Fn fn, uintptr_t addr);
}*/;

// Helpers for dealing with BytePack<0>'s
template<typename A, typename B, int EltPerPack>
struct Apply_Cast_MaybeEmpty: Apply_Cast<A, B, EltPerPack> {};
template<typename A, typename B>
struct Apply_Cast_MaybeEmpty<A, B, /*EltPerPack=*/0> {
  __device__ constexpr static BytePack<0> cast(BytePack<0> a) { return {}; }
};

template<typename Fn, int EltPerPack>
struct Apply_Reduce_MaybeEmpty: Apply_Reduce<Fn, EltPerPack> {};
template<typename Fn>
struct Apply_Reduce_MaybeEmpty<Fn, 0> {
  __device__ constexpr static BytePack<0> reduce(Fn fn, BytePack<0> a, BytePack<0> b) { return {}; }
};

template<typename Fn, int EltPerPack>
struct Apply_PreOp_MaybeEmpty: Apply_PreOp<Fn, EltPerPack> {};
template<typename Fn>
struct Apply_PreOp_MaybeEmpty<Fn, 0> {
  static constexpr bool IsIdentity = true;
  __device__ constexpr static BytePack<0> preOp(Fn fn, BytePack<0> a) { return {}; }
};

template<typename Fn, int EltPerPack>
struct Apply_PostOp_MaybeEmpty: Apply_PostOp<Fn, EltPerPack> {};
template<typename Fn>
struct Apply_PostOp_MaybeEmpty<Fn, 0> {
  static constexpr bool IsIdentity = true;
  __device__ constexpr static BytePack<0> postOp(Fn fn, BytePack<0> a) { return {}; }
};

template<int BytePerPack, bool IsLocal, int CachePolicy, int Rtmd>
struct Apply_Fetch_MaybeEmpty: Apply_Fetch<BytePerPack, IsLocal, CachePolicy, Rtmd> {};
template<bool IsLocal, int CachePolicy, int Rtmd>
struct Apply_Fetch_MaybeEmpty<0, IsLocal, CachePolicy, Rtmd> {
  __device__ static void load(uintptr_t addr, BytePack<0>& val) {}
};

template<int BytePerPack, bool IsLocal, int CachePolicy, int Rtmd>
struct Apply_Store_MaybeEmpty: Apply_Store<BytePerPack, IsLocal, CachePolicy, Rtmd> {};
template<bool IsLocal, int CachePolicy, int Rtmd>
struct Apply_Store_MaybeEmpty<0, IsLocal, CachePolicy, Rtmd> {
  __device__ static void store(uintptr_t addr, const BytePack<0>& val) {}
};

////////////////////////////////////////////////////////////////////////////////
// Public API for calling the trait classes. These take the data elements as a
// pack of any type, which could be a BytePack<?> or any integral type (uint64_t,
// uint32_t, etc.), and will return a new pack where each element has been
// transformed appropriately.

template<typename A, typename B, typename PackA>
__device__ __forceinline__ BytePack<BytePackOf<PackA>::Size*sizeof(B)/sizeof(A)> applyCast(PackA a) {
  return Apply_Cast_MaybeEmpty<A, B, BytePackOf<PackA>::Size/sizeof(A)>::cast(toPack(a));
}

template<typename Fn, typename Pack>
__device__ __forceinline__ Pack applyReduce(Fn fn, Pack a, Pack b) {
  return fromPack<Pack>(
    Apply_Reduce_MaybeEmpty<Fn, BytePackOf<Pack>::Size/sizeof(typename Fn::EltType)>
      ::reduce(fn, toPack(a), toPack(b))
  );
}

template<typename Fn, typename Pack>
__device__ __forceinline__ Pack applyPreOp(Fn fn, Pack a) {
  return fromPack<Pack>(
    Apply_PreOp_MaybeEmpty<Fn, BytePackOf<Pack>::Size/sizeof(typename Fn::EltType)>
      ::preOp(fn, toPack(a))
  );
}

template<typename Fn, typename Pack>
__device__ __forceinline__ Pack applyPostOp(Fn fn, Pack a) {
  return fromPack<Pack>(
    Apply_PostOp_MaybeEmpty<Fn, BytePackOf<Pack>::Size/sizeof(typename Fn::EltType)>
      ::postOp(fn, toPack(a))
  );
}

template<int BytePerPack, bool IsLocal, int CachePolicy, int Rtmd>
__device__ __forceinline__ void applyFetch(uintptr_t addr, BytePack<BytePerPack>& val) {
  Apply_Fetch_MaybeEmpty<BytePerPack, IsLocal, CachePolicy, Rtmd>::load(addr, val);
}

template<int BytePerPack, bool IsLocal, int CachePolicy, int Rtmd>
__device__ __forceinline__ void applyStore(uintptr_t addr, const BytePack<BytePerPack>& val) {
  Apply_Store_MaybeEmpty<BytePerPack, IsLocal, CachePolicy, Rtmd>::store(addr, val);
}

////////////////////////////////////////////////////////////////////////////////
// Apply_Cast

template<typename A, typename B, int EltPerPack>
struct Apply_Cast {
  __device__ __forceinline__ static BytePack<EltPerPack*sizeof(B)> cast(BytePack<EltPerPack*sizeof(A)> a) {
    BytePack<EltPerPack*sizeof(B)> b;
    b.half[0] = Apply_Cast<A, B, EltPerPack/2>::cast(a.half[0]);
    b.half[1] = Apply_Cast<A, B, EltPerPack/2>::cast(a.half[1]);
    return b;
  }
};

template<typename A, typename B>
struct Apply_Cast<A, B, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(B)> cast(BytePack<sizeof(A)> a) {
    return toPack(B(fromPack<A>(a)));
  }
};

template<>
struct Apply_Cast<__half, float, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(float)> cast(BytePack<sizeof(__half)> a) {
    return toPack(__half2float(fromPack<__half>(a)));
  }
};
template<>
struct Apply_Cast<float, __half, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(__half)> cast(BytePack<sizeof(float)> a) {
    return toPack(__float2half_rn(fromPack<float>(a)));
  }
};

template<>
struct Apply_Cast<__half, float, /*EltPerPack=*/2> {
  __device__ __forceinline__ static BytePack<4*2> cast(BytePack<2*2> a) {
    return toPack(__half22float2(fromPack<__half2>(a)));
  }
};
template<>
struct Apply_Cast<float, __half, /*EltPerPack=*/2> {
  __device__ __forceinline__ static BytePack<2*2> cast(BytePack<4*2> a) {
    return toPack(__float22half2_rn(fromPack<float2>(a)));
  }
};

template<>
struct Apply_Cast<__ppu_bfloat16, float, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(float)> cast(BytePack<sizeof(__ppu_bfloat16)> a) {
    return toPack(__bfloat162float(fromPack<__ppu_bfloat16>(a)));
  }
};
template<>
struct Apply_Cast<float, __ppu_bfloat16, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(__ppu_bfloat16)> cast(BytePack<sizeof(float)> a) {
    return toPack(__float2bfloat16(fromPack<float>(a)));
  }
};

template<>
struct Apply_Cast<__ppu_bfloat16, float, /*EltPerPack=*/2> {
  __device__ __forceinline__ static BytePack<4*2> cast(BytePack<2*2> a) {
    return toPack(__bfloat1622float2(fromPack<__ppu_bfloat162>(a)));
  }
};
template<>
struct Apply_Cast<float, __ppu_bfloat16, /*EltPerPack=*/2> {
  __device__ __forceinline__ static BytePack<2*2> cast(BytePack<4*2> a) {
    return toPack(__float22bfloat162_rn(fromPack<float2>(a)));
  }
};

#define EASY_CAST(A, B, EltPerPack, VecA, VecB) \
  template<> \
  struct Apply_Cast<A, B, EltPerPack> { \
    __device__ __forceinline__ static BytePack<sizeof(B)*EltPerPack> cast(BytePack<sizeof(A)*EltPerPack> a) { \
      return toPack(VecB(fromPack<VecA>(a))); \
    } \
  }; \
  template<> \
  struct Apply_Cast<B, A, EltPerPack> { \
    __device__ __forceinline__ static BytePack<sizeof(A)*EltPerPack> cast(BytePack<sizeof(B)*EltPerPack> b) { \
      return toPack(VecA(fromPack<VecB>(b))); \
    } \
  };

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150) && defined(__HGGC_FP8_TYPES_EXIST__)
EASY_CAST(__hg_fp8_e5m2, float, 2, __hg_fp8x2_e5m2, float2)
EASY_CAST(__hg_fp8_e5m2, float, 4, __hg_fp8x4_e5m2, float4)

EASY_CAST(__hg_fp8_e4m3, float, 2, __hg_fp8x2_e4m3, float2)
EASY_CAST(__hg_fp8_e4m3, float, 4, __hg_fp8x4_e4m3, float4)
#endif
#undef EASY_CAST

////////////////////////////////////////////////////////////////////////////////
// Apply_Reduce

// Nonsensical base case
template<typename Fn>
struct Apply_Reduce<Fn, /*EltPerPack=*/0> {
  __device__ __forceinline__ static BytePack<0> reduce(Fn fn, BytePack<0> a, BytePack<0> b) {
    return  {};
  }
};

// General recursive definition (EltPerPack > 1). This is how we iterate over
// all elements in a pack of any size, by breaking it into halves. Eventually
// we'll hit a base case (a more specific template specialization which takes
// precedence).
template<typename Fn, int EltPerPack>
struct Apply_Reduce {
  template<int Size>
  __device__ __forceinline__ static BytePack<Size> reduce(Fn fn, BytePack<Size> a, BytePack<Size> b) {
    a.half[0] = Apply_Reduce<Fn, EltPerPack/2>::reduce(fn, a.half[0], b.half[0]);
    a.half[1] = Apply_Reduce<Fn, EltPerPack/2>::reduce(fn, a.half[1], b.half[1]);
    return a;
  }
};

// Base case definitions (EltPerPack == 1)
template<typename T>
struct Apply_Reduce<FuncCopy<T>, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(T)> reduce(FuncCopy<T> fn, BytePack<sizeof(T)> a, BytePack<sizeof(T)> b) {
    return a;
  }
};
template<typename T>
struct Apply_Reduce<FuncSum<T>, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(T)> reduce(FuncSum<T> fn, BytePack<sizeof(T)> a, BytePack<sizeof(T)> b) {
    return toPack<T>(fromPack<T>(a) + fromPack<T>(b));
  }
};
template<typename T>
struct Apply_Reduce<FuncProd<T>, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(T)> reduce(FuncProd<T> fn, BytePack<sizeof(T)> a, BytePack<sizeof(T)> b) {
    return toPack<T>(fromPack<T>(a) * fromPack<T>(b));
  }
};
template<typename T>
struct Apply_Reduce<FuncMinMax<T>, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(T)> reduce(FuncMinMax<T> fn, BytePack<sizeof(T)> a, BytePack<sizeof(T)> b) {
    return (a.native ^ fn.xormask.native) < (b.native ^ fn.xormask.native) ? a : b;
  }
};

// Optimizations for specific types and element count combinations:
template<>
struct Apply_Reduce<FuncSum<uint8_t>, /*EltPerPack=*/4> {
  __device__ __forceinline__ static BytePack<4> reduce(FuncSum<uint8_t> fn, BytePack<4> a, BytePack<4> b) {
    constexpr uint32_t even = 0x00ff00ffu;
    uint32_t x = (a.native &  even) + (b.native &  even);
    uint32_t y = (a.native & ~even) + (b.native & ~even);
    //a.native = (x & even) | (y & ~even);
    a.native = __byte_perm(x, y, 0x7250);
    return a;
  }
};

template<>
struct Apply_Reduce<FuncMinMax<uint8_t>, /*EltPerPack=*/4> {
  __device__ static BytePack<4> reduce(FuncMinMax<uint8_t> fn, BytePack<4> a, BytePack<4> b) {
    constexpr uint32_t ones = 0x01010101u;
    constexpr uint32_t even = 0x00ff00ffu; // even byte mask
    // Replicate xormask to all bytes
    uint32_t x = fn.xormask.native * ones;
    // Transform inputs by xormask
    uint32_t ax = a.native ^ x;
    uint32_t bx = b.native ^ x;
    // Use 9-bit arithmetic to compute d=a-b
    uint32_t d0 = (ax    & even) + (~bx      & even) + ones;
    uint32_t d1 = (ax>>8 & even) + (~(bx>>8) & even) + ones;
    // Move sign bit of each 9-bit delta into the least bit of origin byte
    //uint32_t s = (d0>>8 & ones & even) | (d1 & ones & ~even);
    uint32_t s = __byte_perm(d0, d1, 0x7351) & ones;
    // Broadcast least bit across whole byte
    s *= 0xffu;
    // Compose result by selecting bytes via: signbit(a-b)==1 ? a : b
    a.native = (a.native & s) | (b.native & ~s);
    return a;
  }
};

template<>
struct Apply_Reduce<FuncProd<uint8_t>, /*EltPerPack=*/4> {
  __device__ __forceinline__ static BytePack<4> reduce(FuncProd<uint8_t> fn, BytePack<4> apack, BytePack<4> bpack) {
    uint32_t a = apack.native;
    uint32_t b = bpack.native;
    uint32_t ab0 = (a*b) & 0xffu;
    asm volatile("ppu.mad.lo.u32 %0, %1, %2, %0;" : "+r"(ab0) : "r"(a&0xff00u), "r"(b&0xff00u));
    uint32_t ab1;
    asm volatile("ppu.mul.hi.u32 %0, %1, %2;"     : "=r"(ab1) : "r"(a&0xff0000), "r"(b&0xff0000));
    asm volatile("ppu.mad.hi.u32 %0, %1, %2, %0;" : "+r"(ab1) : "r"(a&0xff000000u), "r"(b&0xff000000u));
    apack.native = __byte_perm(ab0, ab1, 0x6420);
    return apack;
  }
};

#define SPECIALIZE_REDUCE(Fn, T, EltPerPack, Vec, expr_of_fn_x_y) \
  template<> \
  struct Apply_Reduce<Fn<T>, EltPerPack> { \
    __device__ __forceinline__ static BytePack<sizeof(Vec)> reduce( \
        Fn<T> fn, BytePack<sizeof(Vec)> a, BytePack<sizeof(Vec)> b \
      ) { \
      Vec x = fromPack<Vec>(a); \
      Vec y = fromPack<Vec>(b); \
      return toPack<Vec>(expr_of_fn_x_y); \
    } \
  };

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150) && defined(__HGGC_FP8_TYPES_EXIST__)
  SPECIALIZE_REDUCE(FuncSum, __hg_fp8_e4m3, 1, __hg_fp8_e4m3, __hg_fp8_e4m3(__hadd(__half(x),__half(y))))
  SPECIALIZE_REDUCE(FuncSum, __hg_fp8_e4m3, 2, __hg_fp8x2_e4m3, __hg_fp8x2_e4m3(__hadd2(__half2(x),__half2(y))))

  SPECIALIZE_REDUCE(FuncSum, __hg_fp8_e5m2, 1, __hg_fp8_e5m2, __hg_fp8_e5m2(__hadd(__half(x),__half(y))))
  SPECIALIZE_REDUCE(FuncSum, __hg_fp8_e5m2, 2, __hg_fp8x2_e5m2, __hg_fp8x2_e5m2(__hadd2(__half2(x),__half2(y))))
#endif
SPECIALIZE_REDUCE(FuncSum, half, 1, half, __hadd(x, y))
SPECIALIZE_REDUCE(FuncSum, half, 2, half2, __hadd2(x, y))

SPECIALIZE_REDUCE(FuncSum, __ppu_bfloat16, 1, __ppu_bfloat16, __hadd(x, y))
SPECIALIZE_REDUCE(FuncSum, __ppu_bfloat16, 2, __ppu_bfloat162, __hadd2(x, y))

#undef SPECIALIZE_REDUCE

////////////////////////////////////////////////////////////////////////////////
// Apply_PreOp

// General recursive definition (EltPerPack > 1)
template<typename Fn, int EltPerPack>
struct Apply_PreOp {
  static constexpr bool IsIdentity = Apply_PreOp<Fn, EltPerPack/2>::IsIdentity;
  template<int Size>
  __device__ __forceinline__ static BytePack<Size> preOp(Fn fn, BytePack<Size> a) {
    #if __cpp_if_constexpr
    if constexpr(!IsIdentity) {
    #else
    if (!IsIdentity) {
    #endif
      // The `if (!IsIdentity)` condition is not strictly necessary, but it may help
      // compiler in that it won't have to tear a register apart for no reason
      // just to put it back together again.
      a.half[0] = Apply_PreOp<Fn, EltPerPack/2>::preOp(fn, a.half[0]);
      a.half[1] = Apply_PreOp<Fn, EltPerPack/2>::preOp(fn, a.half[1]);
    }
    return a;
  }
};
// Base case definition (EltPerPack == 1), by default is identity function.
template<typename Fn>
struct Apply_PreOp<Fn, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = true;
  template<int Size>
  __device__ __forceinline__ static BytePack<Size> preOp(Fn fn, BytePack<Size> a) {
    return a;
  }
};
// Base case definition (EltPerPack == 0), is nonsense!
template<typename Fn>
struct Apply_PreOp<Fn, /*EltPerPack=*/0> {
  static constexpr bool IsIdentity = true;
  __device__ __forceinline__ static BytePack<0> preOp(Fn fn, BytePack<0> a) {
    return {};
  }
};

////////////////////////////////////////////////////////////////////////////////
// Apply_PostOp

// General recursive definition (EltPerPack > 1)
template<typename Fn, int EltPerPack>
struct Apply_PostOp {
  static constexpr bool IsIdentity = Apply_PostOp<Fn, EltPerPack/2>::IsIdentity;
  template<int Size>
  __device__ __forceinline__ static BytePack<Size> postOp(Fn fn, BytePack<Size> a) {
    #if __cpp_if_constexpr
    if constexpr(!IsIdentity) {
    #else
    if (!IsIdentity) {
    #endif
      // The `if (!IsIdentity)` condition is not strictly necessary, but it may help
      // compiler in that it won't have to tear a register apart for no reason
      // just to put it back together again.
      a.half[0] = Apply_PostOp<Fn, EltPerPack/2>::postOp(fn, a.half[0]);
      a.half[1] = Apply_PostOp<Fn, EltPerPack/2>::postOp(fn, a.half[1]);
    }
    return a;
  }
};
// Base case definition (EltPerPack == 1), by default is identity function.
template<typename Fn>
struct Apply_PostOp<Fn, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = true;
  template<int Size>
  __device__ __forceinline__ static BytePack<Size> postOp(Fn fn, BytePack<Size> a) {
    return a;
  }
};
// Base case definition (EltPerPack == 0), is nonsense!
template<typename Fn>
struct Apply_PostOp<Fn, /*EltPerPack=*/0> {
  static constexpr bool IsIdentity = true;
  __device__ __forceinline__ static BytePack<0> postOp(Fn fn, BytePack<0> a) {
    return {};
  }
};


////////////////////////////////////////////////////////////////////////////////
// FuncPreMulSum

template<typename T>
struct RedOpArg<FuncPreMulSum<T>> {
  static constexpr bool ArgUsed = true;
  __device__ __forceinline__ static uint64_t loadArg(void *ptr) {
    union { uint64_t u64; T val; };
    u64 = 0;
    val = *(T*)ptr;
    return u64;
  }
};

// General definition for all integral types, float, and double.
template<typename T>
struct FuncPreMulSum {
  using EltType = T;
  T scalar;
  __device__ __forceinline__ FuncPreMulSum(uint64_t opArg=0) {
    union { uint64_t u64; T val; };
    u64 = opArg;
    scalar = val;
  }
};

template<>
// Coverity recommends the users of this type to use std::move in certain cases but,
// given that half is a scalar, a plain copy will be just as efficient.
// coverity[moveable_type]
struct FuncPreMulSum<half> {
  using EltType = half;
  float scalar;
  __device__ __forceinline__ FuncPreMulSum(uint64_t opArg=0) {
    union { uint64_t u64; __half val; };
    u64 = opArg;
    scalar = (float)val;
  }
};

template<typename T, int EltPerPack>
struct Apply_Reduce<FuncPreMulSum<T>, EltPerPack> {
  __device__ __forceinline__ static BytePack<EltPerPack*sizeof(T)> reduce(FuncPreMulSum<T> fn, BytePack<EltPerPack*sizeof(T)> a, BytePack<EltPerPack*sizeof(T)> b) {
    // FuncPreMulSum reduce dispatches to FuncSum.
    return Apply_Reduce<FuncSum<T>, EltPerPack>::reduce(FuncSum<T>(), a, b);
  }
};

// PreOp of FuncPreMulSum for integral types, float, and double.
template<typename T>
struct Apply_PreOp<FuncPreMulSum<T>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(T)> preOp(FuncPreMulSum<T> fn, BytePack<sizeof(T)> a) {
    return toPack<T>(fromPack<T>(a) * fn.scalar);
  }
};

////////////////////////////////////////////////////////////////////////////////
// Apply_PreOp of FuncPreMulSum for float16.

template<>
struct Apply_PreOp<FuncPreMulSum<half>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(half)> preOp(FuncPreMulSum<half> fn, BytePack<sizeof(half)> a) {
    return toPack<half>(__float2half(__half2float(fromPack<half>(a)) * fn.scalar));
  }
};

////////////////////////////////////////////////////////////////////////////////
// Apply_PreOp of FuncPreMulSum for bfloat16.

// TODO

////////////////////////////////////////////////////////////////////////////////
// FuncSumPostDiv

// TODO

#define LOC_BULK_CP 0
#define RMT_BULK_CP_C 0
#define RMT_BULK_CP_VOLATILE 1
#define RMT_BULK_CP_NCP 2
#define RMT_BULK_CP_BYPASS 3

#define RTMD_NORMAL 0
#define RTMD_PA 1
#define RTMD_AR 2

////////////////////////////////////////////////////////////////////////////////
// Apply_Fetch
template<int BytePerPack, bool IsLocal, int CachePolicy, int Rtmd>
struct Apply_Fetch {
  __device__ __forceinline__ static void load(uintptr_t addr, BytePack<BytePerPack>& val) {
    __trap();
  }
};

template<int CachePolicy, int Rtmd>
struct Apply_Fetch<16, true, CachePolicy, Rtmd> {
  __device__ __forceinline__ static void load(uintptr_t addr, BytePack<16>& val) {
    val.u4 = __ppu_global_ldcg_bulk_b32x4((const void*)(addr));
  }
};

template<int CachePolicy, int Rtmd>
struct Apply_Fetch<16, false, CachePolicy, Rtmd> {
  __device__ __forceinline__ static void load(uintptr_t addr, BytePack<16>& val) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ < 150
    switch (CachePolicy) {
      case RMT_BULK_CP_C: val.u4 = __ppu_remote_load_bulk_b32x4((const void*)addr); break;
      case RMT_BULK_CP_VOLATILE: val.u4 = __ppu_remote_load_bulk_volatile_b32x4((volatile const void *)addr); break;
      case RMT_BULK_CP_NCP: val.u4 = __ppu_remote_load_bulk_ncp_b32x4((const void*)addr); break;
      case RMT_BULK_CP_BYPASS: val.u4 = __ppu_remote_load_bulk_bypass_b32x4((const void*)addr); break;
    }
#else
    switch (CachePolicy) {
      case RMT_BULK_CP_C: val.u4 = __ppu_remote_load_bulk_b32x4((const void*)addr, Rtmd); break;
      case RMT_BULK_CP_VOLATILE: val.u4 = __ppu_remote_load_bulk_volatile_b32x4((volatile const void *)addr, Rtmd); break;
      case RMT_BULK_CP_NCP: val.u4 = __ppu_remote_load_bulk_ncp_b32x4((const void*)addr, Rtmd); break;
      case RMT_BULK_CP_BYPASS: val.u4 = __ppu_remote_load_bulk_bypass_b32x4((const void*)addr, Rtmd); break;
    }
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
// Apply_Store
template<int BytePerPack, bool IsLocal, int CachePolicy, int Rtmd>
struct Apply_Store {
  __device__ __forceinline__ static void store(uintptr_t addr, const BytePack<BytePerPack>& val) {
    __trap();
  }
};

template<int CachePolicy, int Rtmd>
struct Apply_Store<16, true, CachePolicy, Rtmd> {
  __device__ __forceinline__ static void store(uintptr_t addr, const BytePack<16>& val) {
    __ppu_global_store_bulk_b32x4((void*)(addr), val.u4);
  }
};

template<int CachePolicy, int Rtmd>
struct Apply_Store<16, false, CachePolicy, Rtmd> {
  __device__ __forceinline__ static void store(uintptr_t addr, const BytePack<16>& val) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ < 150
    switch (CachePolicy) {
      case RMT_BULK_CP_C: __ppu_remote_store_bulk_b32x4((void*)addr, val.u4); break;
      case RMT_BULK_CP_VOLATILE: __ppu_remote_store_bulk_volatile_b32x4((volatile void *)addr, val.u4); break;
      case RMT_BULK_CP_NCP: __ppu_remote_store_bulk_ncp_b32x4((void*)addr, val.u4); break;
      case RMT_BULK_CP_BYPASS: __ppu_remote_store_bulk_bypass_b32x4((void*)addr, val.u4); break;
    }
#else
    switch (CachePolicy) {
      case RMT_BULK_CP_C: __ppu_remote_store_bulk_b32x4((void*)addr, val.u4, Rtmd); break;
      case RMT_BULK_CP_VOLATILE: __ppu_remote_store_bulk_volatile_b32x4((volatile void *)addr, val.u4, Rtmd); break;
      case RMT_BULK_CP_NCP: __ppu_remote_store_bulk_ncp_b32x4((void*)addr, val.u4, Rtmd); break;
      case RMT_BULK_CP_BYPASS: __ppu_remote_store_bulk_bypass_b32x4((void*)addr, val.u4, Rtmd); break;
    }
#endif
  }
};

#endif // REDUCE_KERNEL_H_
