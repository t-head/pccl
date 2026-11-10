/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifdef PCCL_SYM_REDUCE_KERNEL_H_
#error "collectives/device/reduce_kernel.h and symmetric/sym_reduce_kernel.h define the same names with incompatible functor contracts; include at most one per translation unit"
#endif

#ifndef NCCL_REDUCE_KERNEL_H_
#define NCCL_REDUCE_KERNEL_H_

#include "common_kernel.h"
#include <limits>
#include <type_traits>
#include "op128.h"

template<typename T>
struct IsFloatingPoint: std::false_type {};
template<>
struct IsFloatingPoint<half>: std::true_type {};
template<>
struct IsFloatingPoint<__hg_fp8_e4m3>: std::true_type {};
template<>
struct IsFloatingPoint<__hg_fp8_e5m2>: std::true_type {};
template<>
struct IsFloatingPoint<__ppu_bfloat16>: std::true_type {};
template<>
struct IsFloatingPoint<float>: std::true_type {};
template<>
struct IsFloatingPoint<double>: std::true_type {};

template<typename T>
struct FuncCopy {
  using EltType = T;
  __device__ __forceinline__ FuncCopy(uint64_t opArg = 0) {};
};

template<typename T>
struct FuncSum {
  __device__ FuncSum(uint64_t opArg=0) {}
  __device__ T operator()(const T x, const T y) const {
    return x + y;
  }
};

template<typename T>
struct FuncProd {
  __device__ FuncProd(uint64_t opArg=0) {}
  __device__ T operator()(const T x, const T y) const {
    return x * y;
  }
};

template<typename T, bool IsFloating = IsFloatingPoint<T>::value>
struct MinMaxOverloads;

template<typename T>
struct FuncMinMax {
  using EltType = T;
  BytePack<sizeof(T)> xormask; // only used by integers
  bool isMinNotMax;            // only used by floats
  __device__ __forceinline__ FuncMinMax(uint64_t opArg = 0) {
    xormask.native = opArg;
    isMinNotMax = (opArg & 1) == 0;
  }

  template<typename... Args>
  __device__ __forceinline__ auto operator()(Args... args) const
      -> decltype(MinMaxOverloads<T>::apply(BytePack<sizeof(T)>(), bool(), args...)) {
    return MinMaxOverloads<T>::apply(xormask, isMinNotMax, args...);
  }
};

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

template<typename Fn>
struct FuncTraits { // generic identity preOp/postOp; the scaling operators specialize below
  static constexpr bool IsPreOpIdentity = true;
  static constexpr bool IsPostOpIdentity = true;

  template<typename T>
  __device__ static T preOp(Fn, T x) { return x; }
  template<typename T>
  __device__ static T postOp(Fn, T x) { return x; }
};


#define MASK0 0x00ff00ff
#define MASK1 0xff00ff00
static __device__ uint32_t addChar4(const uint32_t x, const uint32_t y) {
  /* This can be used both for signed and unsigned 8-bit addition */
  const uint32_t x0 = x & MASK0;
  const uint32_t x1 = x & MASK1;
  const uint32_t y0 = y & MASK0;
  const uint32_t y1 = y & MASK1;
  const uint32_t r0 = (x0+y0);
  const uint32_t r1 = (x1+y1);
  return (r0 & MASK0) | (r1 & MASK1);
}

template<>
struct FuncSum<int8_t> {
  __device__ FuncSum(uint64_t opArg=0) {}
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if 0
    int32_t rv, z=0;
    asm("vadd4.s32.s32.s32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    return rv;
#else
    return addChar4(x, y);
#endif
  }
  __device__ int8_t operator()(const int8_t x, const int8_t y) const {
    return x+y;
  }
};
template<>
struct FuncSum<uint8_t> {
  __device__ FuncSum(uint64_t opArg=0) {}
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if 0
    int32_t rv, z=0;
    asm("vadd4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    return rv;
#else
    return addChar4(x, y);
#endif
  }
  __device__ uint8_t operator()(const uint8_t x, const uint8_t y) const {
    return x+y;
  }
};

static __device__ uint32_t mulChar4(const uint32_t x, const uint32_t y) {
  /* This can be used both for signed and unsigned 8-bit multiplication */
  union converter { uint32_t storage; char4 a; };
  converter cx, cy, cr;
  cx.storage = x;
  cy.storage = y;
  cr.a.x = cx.a.x * cy.a.x;
  cr.a.y = cx.a.y * cy.a.y;
  cr.a.z = cx.a.z * cy.a.z;
  cr.a.w = cx.a.w * cy.a.w;
  return cr.storage;
}

template<>
struct FuncProd<int8_t> {
  __device__ FuncProd(uint64_t opArg=0) {}
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
    return mulChar4(x, y);
  }
  __device__ int8_t operator()(const int8_t x, const int8_t y) const {
    return x*y;
  }
};
template<>
struct FuncProd<uint8_t> {
  __device__ FuncProd(uint64_t opArg=0) {}
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
    return mulChar4(x, y);
  }
  __device__ uint8_t operator()(const uint8_t x, const uint8_t y) const {
    return x*y;
  }
};

template<>
struct FuncSum<half> {
  __device__ FuncSum(uint64_t opArg=0) {}
  __device__ half2 operator()(const half2 x, const half2 y) const {
#if 1
    return __hadd2(x, y);
#else
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = fx.x + fy.x;
    fr.y = fx.y + fy.y;
    return __float22half2_rn(fr);
#endif
  }
  __device__ half operator()(const half x, const half y) const {
#if 1
    return __hadd(x, y);
#else
    return __float2half( __half2float(x) + __half2float(y) );
#endif
  }
};

template<>
struct FuncSum<__ppu_bfloat16> {
  __device__ FuncSum(uint64_t opArg=0) {}
  __device__ __ppu_bfloat162 operator()(const __ppu_bfloat162 x, const __ppu_bfloat162 y) const {
#if 1
    return __hadd2(x, y);
#else
    float fxl, fxh, fyl, fyh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    fyl = __low2float(y);
    fyh = __high2float(y);
    return __floats2bfloat162_rn(fxl + fyl, fxh + fyh);
#endif
   }
  __device__ __ppu_bfloat16 operator()(const __ppu_bfloat16 x, const __ppu_bfloat16 y) const {
#if 1
    return __hadd(x, y);
#else
    return __float2bfloat16( __bfloat162float(x) + __bfloat162float(y) );
#endif
  }
};

template<>
struct FuncSum<__hg_fp8_e4m3> {
  __device__ FuncSum(uint64_t opArg=0) {}
  __device__ __hg_fp8_e4m3 operator()(const __hg_fp8_e4m3 x, const __hg_fp8_e4m3 y) const {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
    const __half xCvtHalf = __half(x);
    const __half yCvtHalf = __half(y);
    return __hg_fp8_e4m3(__hadd(xCvtHalf, yCvtHalf));
#else
    return __hg_fp8_e4m3(0);
#endif
  }
  __device__ __hg_fp8x2_e4m3 operator()(const __hg_fp8x2_e4m3 x, const __hg_fp8x2_e4m3 y) const {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
    const __half2 xCvtHalf2 = __half2(x);
    const __half2 yCvtHalf2 = __half2(y);
    return __hg_fp8x2_e4m3(__hadd2(xCvtHalf2, yCvtHalf2));
#else
    return __hg_fp8x2_e4m3(make_double2(0.0, 0.0));
#endif
  }
};

template<>
struct FuncSum<__hg_fp8_e5m2> {
  __device__ FuncSum(uint64_t opArg=0) {}
  __device__ __hg_fp8_e5m2 operator()(const __hg_fp8_e5m2 x, const __hg_fp8_e5m2 y) const {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
    const __half xCvtHalf = __half(x);
    const __half yCvtHalf = __half(y);
    return __hg_fp8_e5m2(__hadd(xCvtHalf, yCvtHalf));
#else
    return __hg_fp8_e5m2(0);
#endif
  }
  __device__ __hg_fp8x2_e5m2 operator()(const __hg_fp8x2_e5m2 x, const __hg_fp8x2_e5m2 y) const {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
    const __half2 xCvtHalf2 = __half2(x);
    const __half2 yCvtHalf2 = __half2(y);
    return __hg_fp8x2_e5m2(__hadd2(xCvtHalf2, yCvtHalf2));
#else
    return __hg_fp8x2_e5m2(make_double2(0.0, 0.0));
#endif
  }
};

template<>
struct FuncProd<half> {
  __device__ FuncProd(uint64_t opArg=0) {}
  __device__ half2 operator()(const half2 x, const half2 y) const {
#if 1
    return __hmul2(x, y);
#else
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = fx.x * fy.x;
    fr.y = fx.y * fy.y;
    return __float22half2_rn(fr);
#endif
  }
  __device__ half operator()(const half x, const half y) const {
#if 1
    return __hmul(x, y);
#else
    return __float2half( __half2float(x) * __half2float(y) );
#endif
  }
};

template<>
struct FuncProd<__ppu_bfloat16> {
  __device__ FuncProd(uint64_t opArg=0) {}
  __device__ __ppu_bfloat162 operator()(const __ppu_bfloat162 x, const __ppu_bfloat162 y) const {
#if 1
    return __hmul2(x, y);
#else
    float fxl, fxh, fyl, fyh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    fyl = __low2float(y);
    fyh = __high2float(y);
    return __floats2bfloat162_rn(fxl * fyl, fxh * fyh);
#endif
  }
  __device__ __ppu_bfloat16 operator()(const __ppu_bfloat16 x, const __ppu_bfloat16 y) const {
#if 1
    return __hmul(x, y);
#else
    return __float2bfloat16( __bfloat162float(x) * __bfloat162float(y) );
#endif
  }
};

template<>
struct FuncProd<__hg_fp8_e4m3> {
  __device__ FuncProd(uint64_t opArg=0) {}
  __device__ __hg_fp8_e4m3 operator()(const __hg_fp8_e4m3 x, const __hg_fp8_e4m3 y) const {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
    const __half xCvtHalf = __half(x);
    const __half yCvtHalf = __half(y);
    return __hg_fp8_e4m3(__hmul(xCvtHalf, yCvtHalf));
#else
    return __hg_fp8_e4m3(0);
#endif
  }
  __device__ __hg_fp8x2_e4m3 operator()(const __hg_fp8x2_e4m3 x, const __hg_fp8x2_e4m3 y) const {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
    const __half2 xCvtHalf2 = __half2(x);
    const __half2 yCvtHalf2 = __half2(y);
    return __hg_fp8x2_e4m3(__hmul2(xCvtHalf2, yCvtHalf2));
#else
    return __hg_fp8x2_e4m3(make_double2(0.0, 0.0));
#endif
  }
};

template<>
struct FuncProd<__hg_fp8_e5m2> {
  __device__ FuncProd(uint64_t opArg=0) {}
  __device__ __hg_fp8_e5m2 operator()(const __hg_fp8_e5m2 x, const __hg_fp8_e5m2 y) const {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
    const __half xCvtHalf = __half(x);
    const __half yCvtHalf = __half(y);
    return __hg_fp8_e5m2(__hmul(xCvtHalf, yCvtHalf));
#else
    return __hg_fp8_e5m2(0);
#endif
  }
  __device__ __hg_fp8x2_e5m2 operator()(const __hg_fp8x2_e5m2 x, const __hg_fp8x2_e5m2 y) const {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
    const __half2 xCvtHalf2 = __half2(x);
    const __half2 yCvtHalf2 = __half2(y);
    return __hg_fp8x2_e5m2(__hmul2(xCvtHalf2, yCvtHalf2));
#else
    return __hg_fp8x2_e5m2(make_double2(0.0, 0.0));
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
// MinMaxOverloads: the min/max arithmetic behind the legacy MULTI<> path.

template<typename T>
struct MinMaxOverloads<T, /*IsFloating=*/true> {
  __device__ __forceinline__ static T apply(
      BytePack<sizeof(T)>, bool isMinNotMax, T x, T y) {
    return isMinNotMax ? ((x < y) ? x : y) : ((x < y) ? y : x);
  }
};

template<>
struct MinMaxOverloads<half, /*IsFloating=*/true> {
  __device__ __forceinline__ static half2 apply(
      BytePack<sizeof(half)>, bool isMinNotMax, half2 x, half2 y) {
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = isMinNotMax ? fminf(fx.x, fy.x) : fmaxf(fx.x, fy.x);
    fr.y = isMinNotMax ? fminf(fx.y, fy.y) : fmaxf(fx.y, fy.y);
    return __float22half2_rn(fr);
  }
  __device__ __forceinline__ static half apply(
      BytePack<sizeof(half)>, bool isMinNotMax, half x, half y) {
    float fx, fy, fm;
    fx = __half2float(x);
    fy = __half2float(y);
    fm = isMinNotMax ? fminf(fx, fy) : fmaxf(fx, fy);
    return __float2half(fm);
  }
};

template<>
struct MinMaxOverloads<__ppu_bfloat16, /*IsFloating=*/true> {
  __device__ __forceinline__ static __ppu_bfloat162 apply(
      BytePack<sizeof(__ppu_bfloat16)>, bool isMinNotMax,
      __ppu_bfloat162 x, __ppu_bfloat162 y) {
#if 1
    return isMinNotMax ? __hmin2(x, y) : __hmax2(x, y);
#else
    float fxl, fxh, fyl, fyh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    fyl = __low2float(y);
    fyh = __high2float(y);
    return __floats2bfloat162_rn(
        isMinNotMax ? fminf(fxl, fyl) : fmaxf(fxl, fyl),
        isMinNotMax ? fminf(fxh, fyh) : fmaxf(fxh, fyh));
#endif
  }
  __device__ __forceinline__ static __ppu_bfloat16 apply(
      BytePack<sizeof(__ppu_bfloat16)>, bool isMinNotMax,
      __ppu_bfloat16 x, __ppu_bfloat16 y) {
#if 1
    return isMinNotMax ? __hmin(x, y) : __hmax(x, y);
#else
    float fx, fy;
    fx = __bfloat162float(x);
    fy = __bfloat162float(y);
    return __float2bfloat16(isMinNotMax ? fminf(fx, fy) : fmaxf(fx, fy));
#endif
  }
};

template<>
struct MinMaxOverloads<__hg_fp8_e4m3, /*IsFloating=*/true> {
  __device__ __forceinline__ static __hg_fp8_e4m3 apply(
      BytePack<sizeof(__hg_fp8_e4m3)>, bool isMinNotMax,
      __hg_fp8_e4m3 x, __hg_fp8_e4m3 y) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
    const __half xCvtHalf = __half(x);
    const __half yCvtHalf = __half(y);
    return __hg_fp8_e4m3(isMinNotMax ? __hmin(xCvtHalf, yCvtHalf) : __hmax(xCvtHalf, yCvtHalf));
#else
    return __hg_fp8_e4m3(0);
#endif
  }
  __device__ __forceinline__ static __hg_fp8x2_e4m3 apply(
      BytePack<sizeof(__hg_fp8_e4m3)>, bool isMinNotMax,
      __hg_fp8x2_e4m3 x, __hg_fp8x2_e4m3 y) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
    const __half2 xCvtHalf2 = __half2(x);
    const __half2 yCvtHalf2 = __half2(y);
    return __hg_fp8x2_e4m3(isMinNotMax ? __hmin2(xCvtHalf2, yCvtHalf2) : __hmax2(xCvtHalf2, yCvtHalf2));
#else
    return __hg_fp8x2_e4m3(make_double2(0.0, 0.0));
#endif
  }
};

template<>
struct MinMaxOverloads<__hg_fp8_e5m2, /*IsFloating=*/true> {
  __device__ __forceinline__ static __hg_fp8_e5m2 apply(
      BytePack<sizeof(__hg_fp8_e5m2)>, bool isMinNotMax,
      __hg_fp8_e5m2 x, __hg_fp8_e5m2 y) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
    const __half xCvtHalf = __half(x);
    const __half yCvtHalf = __half(y);
    return __hg_fp8_e5m2(isMinNotMax ? __hmin(xCvtHalf, yCvtHalf) : __hmax(xCvtHalf, yCvtHalf));
#else
    return __hg_fp8_e5m2(0);
#endif
  }
  __device__ __forceinline__ static __hg_fp8x2_e5m2 apply(
      BytePack<sizeof(__hg_fp8_e5m2)>, bool isMinNotMax,
      __hg_fp8x2_e5m2 x, __hg_fp8x2_e5m2 y) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
    const __half2 xCvtHalf2 = __half2(x);
    const __half2 yCvtHalf2 = __half2(y);
    return __hg_fp8x2_e5m2(isMinNotMax ? __hmin2(xCvtHalf2, yCvtHalf2) : __hmax2(xCvtHalf2, yCvtHalf2));
#else
    return __hg_fp8x2_e5m2(make_double2(0.0, 0.0));
#endif
  }
};

template<>
struct MinMaxOverloads<float, /*IsFloating=*/true> {
  __device__ __forceinline__ static float apply(
      BytePack<sizeof(float)>, bool isMinNotMax, float x, float y) {
    return isMinNotMax ? fminf(x, y) : fmaxf(x, y);
  }
};

template<>
struct MinMaxOverloads<double, /*IsFloating=*/true> {
  __device__ __forceinline__ static double apply(
      BytePack<sizeof(double)>, bool isMinNotMax, double x, double y) {
    return isMinNotMax ? fmin(x, y) : fmax(x, y);
  }
};

// Integers wider than a byte: every caller hands over one element at a time.
template<typename T>
struct MinMaxOverloads<T, /*IsFloating=*/false> {
  __device__ __forceinline__ static T apply(
      BytePack<sizeof(T)> xormask, bool /*isMinNotMax*/, T a, T b) {
    return (toPack<T>(a).native ^ xormask.native) < (toPack<T>(b).native ^ xormask.native) ? a : b;
  }
};

template<typename T>
struct MinMaxOverloadsByte {
  // One element, as ReduceCopyMulti's by-type loop and the scalar base case ask.
  __device__ __forceinline__ static T apply(
      BytePack<1> xormask, bool /*isMinNotMax*/, T a, T b) {
    return (uint8_t(a) ^ xormask.native) < (uint8_t(b) ^ xormask.native) ? a : b;
  }
  __device__ __forceinline__ static uint32_t apply(
      BytePack<1> xormask, bool /*isMinNotMax*/, uint32_t a, uint32_t b) {
    constexpr uint32_t ones = 0x01010101u;
    constexpr uint32_t even = 0x00ff00ffu; // even byte mask
    // Replicate the mask to all four lanes, then transform both inputs by it.
    uint32_t m = xormask.native * ones;
    uint32_t ax = a ^ m;
    uint32_t bx = b ^ m;
    // ax-bx as (ax + ~bx + 1) in 9-bit lanes, so no borrow crosses a lane. The
    // +ones also drops a 1 into the byte above each lane, which is what makes
    // bit 0 of that byte end up set exactly when its lane borrowed, i.e. when
    // ax < bx -- the same condition the scalar overload above selects `a` on.
    uint32_t d0 = (ax      & even) + (~bx        & even) + ones;
    uint32_t d1 = (ax >> 8 & even) + (~(bx >> 8) & even) + ones;
    // Gather those four bits into the low bit of their own byte, then broadcast
    // each across its byte to make a select mask.
    uint32_t s = __byte_perm(d0, d1, 0x7351) & ones;
    s *= 0xffu;
    return (a & s) | (b & ~s);
  }
};

template<>
struct MinMaxOverloads<int8_t, /*IsFloating=*/false>: MinMaxOverloadsByte<int8_t> {};
template<>
struct MinMaxOverloads<uint8_t, /*IsFloating=*/false>: MinMaxOverloadsByte<uint8_t> {};

template<typename T>
struct FuncPreMulSum: FuncSum<T> { // integral T since all floats are specialized below
  static constexpr bool IsPreOpIdentity = false;
  static constexpr bool IsPostOpIdentity = true;
  T scale;
  __device__ FuncPreMulSum(uint64_t opArg) { scale = *(T*)&opArg; }
  // inherits FuncSum::operator()
  __device__ T preOp(T x) const { return x*scale; }
  __device__ T postOp(T x) const { return x; }
};

template<>
struct FuncPreMulSum<double>: FuncSum<double> {
  static constexpr bool IsPreOpIdentity = false;
  static constexpr bool IsPostOpIdentity = true;
  double scale;
  __device__ FuncPreMulSum(uint64_t opArg) {
    scale = *(double*)&opArg;
  }
  // inherits FuncSum::operator()
  __device__ double preOp(double x) const {
    return IsPreOpIdentity ? x : x*scale;
  }
  __device__ double postOp(double x) const {
    return IsPostOpIdentity ? x : x*scale;
  }
};

template<>
struct FuncPreMulSum<float>: FuncSum<float> {
  static constexpr bool IsPreOpIdentity = false;
  static constexpr bool IsPostOpIdentity = true;
  float scale;
  __device__ FuncPreMulSum(uint64_t opArg) {
    scale = *(float*)&opArg;
  }
  // inherits FuncSum::operator()
  __device__ float preOp(float x) const {
    return IsPreOpIdentity ? x : x*scale;
  }
  __device__ float postOp(float x) const {
    return IsPostOpIdentity ? x : x*scale;
  }
};

template<>
struct FuncPreMulSum<half>: FuncSum<half> {
  // Change these to switch between all prescale, all postscale, or both by sqrt(N).
  // Obviously, the only invalid combination is both true. An improvement would be
  // make this parameterized as a build time setting and passed here through
  // preprocessor definitions.
  static constexpr bool IsPreOpIdentity = false;
  static constexpr bool IsPostOpIdentity = true;

  half2 scale;
  __device__ FuncPreMulSum(uint64_t opArg) {
    scale.x = *(half*)&opArg;
    scale.y = scale.x;
  }
  // inherits FuncSum::operator()
  __device__ half preOp(half x) const {
    return IsPreOpIdentity ? x : __hmul(x, scale.x);
  }
  __device__ half2 preOp(half2 x) const {
    return IsPreOpIdentity ? x : __hmul2(x, scale);
  }
  __device__ half postOp(half x) const {
    return IsPostOpIdentity ? x : __hmul(x, scale.x);
  }
  __device__ half2 postOp(half2 x) const {
    return IsPostOpIdentity ? x : __hmul2(x, scale);
  }
};

template<>
struct FuncPreMulSum<__ppu_bfloat16>: FuncSum<__ppu_bfloat16> {
  // Change these to switch between all prescale, all postscale, or both by sqrt(N).
  // Obviously, the only invalid combination is both true. An improvement would be
  // make this parameterized as a build time setting and passed here through
  // preprocessor definitions.
  static constexpr bool IsPreOpIdentity = false;
  static constexpr bool IsPostOpIdentity = true;

  __ppu_bfloat162 scale;
  __device__ FuncPreMulSum(uint64_t opArg) {
    scale.x = *(__ppu_bfloat16*)&opArg;
    scale.y = scale.x;
  }
  // inherits FuncSum::operator()
  __device__ __ppu_bfloat16 preOp(__ppu_bfloat16 x) const {
    return IsPreOpIdentity ? x : __hmul(x, scale.x);
  }
  __device__ __ppu_bfloat162 preOp(__ppu_bfloat162 x) const {
    return IsPreOpIdentity ? x : __hmul2(x, scale);
  }
  __device__ __ppu_bfloat16 postOp(__ppu_bfloat16 x) const {
    return IsPostOpIdentity ? x : __hmul(x, scale.x);
  }
  __device__ __ppu_bfloat162 postOp(__ppu_bfloat162 x) const {
    return IsPostOpIdentity ? x : __hmul2(x, scale);
  }
};

template<>
struct FuncPreMulSum<__hg_fp8_e4m3>: FuncSum<__hg_fp8_e4m3> {
  static constexpr bool IsPreOpIdentity = false;
  static constexpr bool IsPostOpIdentity = true;

  __half2 scale;
  __device__ FuncPreMulSum(uint64_t opArg) {
      union { uint64_t u64; __hg_fp8_storage_t val; };
      u64 = opArg;
      scale.x = __half(__hg_cvt_fp8_to_halfraw(val, __HG_E4M3));
      scale.y = scale.x;
  }

  __device__ __hg_fp8_e4m3 preOp(const __hg_fp8_e4m3 x) const {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
    return IsPreOpIdentity ? x : __hg_fp8_e4m3(__hmul(__half(x), scale.x));
#else
    return __hg_fp8_e4m3(0);
#endif
  }

  __device__ __hg_fp8x2_e4m3 preOp(const __hg_fp8x2_e4m3 x) const {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
    return IsPreOpIdentity ? x : __hg_fp8x2_e4m3(__hmul2(__half2(x), scale));
#else
    return __hg_fp8x2_e4m3(make_double2(0.0, 0.0));
#endif
  }

  __device__ __hg_fp8_e4m3 postOp(const __hg_fp8_e4m3 x) const {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
    return IsPostOpIdentity ? x : __hg_fp8_e4m3(__hmul(__half(x), scale.x));
#else
    return __hg_fp8_e4m3(0);
#endif
  }

  __device__ __hg_fp8x2_e4m3 postOp(const __hg_fp8x2_e4m3 x) const {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
    return IsPostOpIdentity ? x : __hg_fp8x2_e4m3(__hmul2(__half2(x), scale));
#else
    return __hg_fp8x2_e4m3(make_double2(0.0, 0.0));
#endif
  }
};

template<>
struct FuncPreMulSum<__hg_fp8_e5m2>: FuncSum<__hg_fp8_e5m2> {
  static constexpr bool IsPreOpIdentity = false;
  static constexpr bool IsPostOpIdentity = true;

  __half2 scale;
  __device__ FuncPreMulSum(uint64_t opArg) {
      union { uint64_t u64; __hg_fp8_storage_t val; };
      u64 = opArg;
      scale.x = __half(__hg_cvt_fp8_to_halfraw(val, __HG_E5M2));
      scale.y = scale.x;
  }

  __device__ __hg_fp8_e5m2 preOp(const __hg_fp8_e5m2 x) const {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
    return IsPreOpIdentity ? x : __hg_fp8_e5m2(__hmul(__half(x), scale.x));
#else
    return __hg_fp8_e5m2(0);
#endif
  }

  __device__ __hg_fp8x2_e5m2 preOp(const __hg_fp8x2_e5m2 x) const {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
    return IsPreOpIdentity ? x : __hg_fp8x2_e5m2(__hmul2(__half2(x), scale));
#else
    return __hg_fp8x2_e5m2(make_double2(0.0, 0.0));
#endif
  }

  __device__ __hg_fp8_e5m2 postOp(const __hg_fp8_e5m2 x) const {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
    return IsPostOpIdentity ? x : __hg_fp8_e5m2(__hmul(__half(x), scale.x));
#else
    return __hg_fp8_e5m2(0);
#endif
  }

  __device__ __hg_fp8x2_e5m2 postOp(const __hg_fp8x2_e5m2 x) const {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
    return IsPostOpIdentity ? x : __hg_fp8x2_e5m2(__hmul2(__half2(x), scale));
#else
    return __hg_fp8x2_e5m2(make_double2(0.0, 0.0));
#endif
  }
};

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

template<typename T>
struct FuncTraits<FuncPreMulSum<T>> {
  static constexpr bool IsPreOpIdentity = FuncPreMulSum<T>::IsPreOpIdentity;
  static constexpr bool IsPostOpIdentity = FuncPreMulSum<T>::IsPostOpIdentity;

  template<typename U>
  __device__ static U preOp(FuncPreMulSum<T> fn, U x) {
    return fn.preOp(x);
  }
  template<typename U>
  __device__ static U postOp(FuncPreMulSum<T> fn, U x) {
    return fn.postOp(x);
  }
};

template<class FUNC>
struct RedOpIdentityArg {
  static __device__ uint64_t value() { return 0; }
};

template<typename T>
struct RedOpIdentityArg<FuncPreMulSum<T>> {
  static __device__ uint64_t value() {
    uint64_t arg = 0;
    T one = T(1);
    *(T*)&arg = one;
    return arg;
  }
};

template<>
struct RedOpIdentityArg<FuncPreMulSum<__hg_fp8_e4m3>> {
  static __device__ uint64_t value() {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
    return 0x38ULL;
#else
    return 0ULL;
#endif
  }
};

template<>
struct RedOpIdentityArg<FuncPreMulSum<__hg_fp8_e5m2>> {
  static __device__ uint64_t value() {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
    return 0x3CULL;
#else
    return 0ULL;
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
// Trait classes for reduction functions. Given a function (FuncSum, etc.)
// and a number of elements in a pack, will reduce, preOp, or postOp a pack
// of elements. These classes are intended to be specialized for specific
// combinations of reduction function and pack size.

template<typename Fn>
struct FuncEltType { using type = typename Fn::EltType; };
template<typename T> struct FuncEltType<FuncSum<T>> { using type = T; };
template<typename T> struct FuncEltType<FuncProd<T>> { using type = T; };
template<typename T> struct FuncEltType<FuncPreMulSum<T>> { using type = T; };

template<typename A, typename B, int EltPerPackA>
struct Apply_Cast;
template<typename Fn, int EltPerPack> struct Apply_Reduce;
template<typename Fn, int EltPerPack> struct Apply_PreOp;
template<typename Fn, int EltPerPack> struct Apply_PostOp;

template<typename Fn>
struct LoadMultimem_BigPackSize {
  static constexpr int BigPackSize = 0;
};

template<typename A, typename B, int EltPerPack>
struct Apply_Cast_MaybeEmpty: Apply_Cast<A, B, EltPerPack> {};
template<typename A, typename B>
struct Apply_Cast_MaybeEmpty<A, B, /*EltPerPack=*/0> {
  __device__ constexpr static BytePack<0> cast(BytePack<0> a) { return {}; }
};

template<typename Fn, int EltPerPack>
struct Apply_Reduce_MaybeEmpty: Apply_Reduce<Fn, EltPerPack> {};
template<typename Fn>
struct Apply_Reduce_MaybeEmpty<Fn, /*EltPerPack=*/0> {
  __device__ constexpr static BytePack<0> reduce(Fn fn, BytePack<0> a, BytePack<0> b) { return {}; }
};

template<typename Fn, int EltPerPack>
struct Apply_PreOp_MaybeEmpty: Apply_PreOp<Fn, EltPerPack> {};
template<typename Fn>
struct Apply_PreOp_MaybeEmpty<Fn, /*EltPerPack=*/0> {
  static constexpr bool IsIdentity = true;
  __device__ constexpr static BytePack<0> preOp(Fn fn, BytePack<0> a) { return {}; }
};

template<typename Fn, int EltPerPack>
struct Apply_PostOp_MaybeEmpty: Apply_PostOp<Fn, EltPerPack> {};
template<typename Fn>
struct Apply_PostOp_MaybeEmpty<Fn, /*EltPerPack=*/0> {
  static constexpr bool IsIdentity = true;
  __device__ constexpr static BytePack<0> postOp(Fn fn, BytePack<0> a) { return {}; }
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
    Apply_Reduce_MaybeEmpty<Fn, BytePackOf<Pack>::Size/sizeof(typename FuncEltType<Fn>::type)>
      ::reduce(fn, toPack(a), toPack(b)));
}

template<typename Fn, typename Pack>
__device__ __forceinline__ Pack applyPreOp(Fn fn, Pack a) {
  return fromPack<Pack>(
    Apply_PreOp_MaybeEmpty<Fn, BytePackOf<Pack>::Size/sizeof(typename FuncEltType<Fn>::type)>
      ::preOp(fn, toPack(a)));
}

template<typename Fn, typename Pack>
__device__ __forceinline__ Pack applyPostOp(Fn fn, Pack a) {
  return fromPack<Pack>(
    Apply_PostOp_MaybeEmpty<Fn, BytePackOf<Pack>::Size/sizeof(typename FuncEltType<Fn>::type)>
      ::postOp(fn, toPack(a)));
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
struct Apply_Cast<half, float, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(float)> cast(BytePack<sizeof(half)> a) {
    return toPack(__half2float(fromPack<half>(a)));
  }
};
template<>
struct Apply_Cast<float, half, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(half)> cast(BytePack<sizeof(float)> a) {
    return toPack(__float2half_rn(fromPack<float>(a)));
  }
};

template<>
struct Apply_Cast<half, float, /*EltPerPack=*/2> {
  __device__ __forceinline__ static BytePack<4*2> cast(BytePack<2*2> a) {
    return toPack(__half22float2(fromPack<half2>(a)));
  }
};
template<>
struct Apply_Cast<float, half, /*EltPerPack=*/2> {
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

// Nonsensical base case.
template<typename Fn>
struct Apply_Reduce<Fn, /*EltPerPack=*/0> {
  __device__ __forceinline__ static BytePack<0> reduce(Fn fn, BytePack<0> a, BytePack<0> b) { return {}; }
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
  __device__ __forceinline__ static BytePack<sizeof(T)> reduce(
      FuncMinMax<T> fn, BytePack<sizeof(T)> a, BytePack<sizeof(T)> b) {
    return (a.native ^ fn.xormask.native) < (b.native ^ fn.xormask.native) ? a : b;
  }
};

template<typename T, int EltPerPack>
struct Apply_Reduce<FuncPreMulSum<T>, EltPerPack> {
  __device__ __forceinline__ static BytePack<EltPerPack * sizeof(T)> reduce(
      FuncPreMulSum<T> fn, BytePack<EltPerPack * sizeof(T)> a, BytePack<EltPerPack * sizeof(T)> b) {
    return Apply_Reduce<FuncSum<T>, EltPerPack>::reduce(FuncSum<T>(), a, b);
  }
};

// Optimizations for specific types and element count combinations:
template<>
struct Apply_Reduce<FuncSum<uint8_t>, /*EltPerPack=*/4> {
  __device__ __forceinline__ static BytePack<4> reduce(FuncSum<uint8_t> fn, BytePack<4> a, BytePack<4> b) {
    constexpr uint32_t even = 0x00ff00ffu;
    uint32_t x = (a.native & even) + (b.native & even);
    uint32_t y = (a.native & ~even) + (b.native & ~even);
    // a.native = (x & even) | (y & ~even);
    a.native = __byte_perm(x, y, 0x7250);
    return a;
  }
};
template<>
struct Apply_Reduce<FuncSum<int8_t>, /*EltPerPack=*/4> {
  __device__ __forceinline__ static BytePack<4> reduce(FuncSum<int8_t> fn, BytePack<4> a, BytePack<4> b) {
    constexpr uint32_t even = 0x00ff00ffu;
    uint32_t x = (a.native & even) + (b.native & even);
    uint32_t y = (a.native & ~even) + (b.native & ~even);
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
    uint32_t d0 = (ax & even) + (~bx & even) + ones;
    uint32_t d1 = (ax >> 8 & even) + (~(bx >> 8) & even) + ones;
    // Move sign bit of each 9-bit delta into the least bit of origin byte
    uint32_t s = __byte_perm(d0, d1, 0x7351) & ones;
    // Broadcast least bit across whole byte
    s *= 0xffu;
    // Compose result by selecting bytes via: signbit(a-b)==1 ? a : b
    a.native = (a.native & s) | (b.native & ~s);
    return a;
  }
};
template<>
struct Apply_Reduce<FuncMinMax<int8_t>, /*EltPerPack=*/4> {
  __device__ static BytePack<4> reduce(FuncMinMax<int8_t> fn, BytePack<4> a, BytePack<4> b) {
    constexpr uint32_t ones = 0x01010101u;
    constexpr uint32_t even = 0x00ff00ffu; // even byte mask
    // Replicate xormask to all bytes
    uint32_t x = fn.xormask.native * ones;
    // Transform inputs by xormask
    uint32_t ax = a.native ^ x;
    uint32_t bx = b.native ^ x;
    // Use 9-bit arithmetic to compute d=a-b
    uint32_t d0 = (ax & even) + (~bx & even) + ones;
    uint32_t d1 = (ax >> 8 & even) + (~(bx >> 8) & even) + ones;
    // Move sign bit of each 9-bit delta into the least bit of origin byte
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
    uint32_t ab0 = (a * b) & 0xffu;
    asm volatile("ppu.mad.lo.u32 %0, %1, %2, %0;" : "+r"(ab0) : "r"(a & 0xff00u), "r"(b & 0xff00u));
    uint32_t ab1;
    asm volatile("ppu.mul.hi.u32 %0, %1, %2;" : "=r"(ab1) : "r"(a & 0xff0000), "r"(b & 0xff0000));
    asm volatile("ppu.mad.hi.u32 %0, %1, %2, %0;" : "+r"(ab1) : "r"(a & 0xff000000u), "r"(b & 0xff000000u));
    apack.native = __byte_perm(ab0, ab1, 0x6420);
    return apack;
  }
};
template<>
struct Apply_Reduce<FuncProd<int8_t>, /*EltPerPack=*/4> {
  __device__ __forceinline__ static BytePack<4> reduce(FuncProd<int8_t> fn, BytePack<4> apack, BytePack<4> bpack) {
    uint32_t a = apack.native;
    uint32_t b = bpack.native;
    uint32_t ab0 = (a * b) & 0xffu;
    asm volatile("ppu.mad.lo.u32 %0, %1, %2, %0;" : "+r"(ab0) : "r"(a & 0xff00u), "r"(b & 0xff00u));
    uint32_t ab1;
    asm volatile("ppu.mul.hi.u32 %0, %1, %2;" : "=r"(ab1) : "r"(a & 0xff0000), "r"(b & 0xff0000));
    asm volatile("ppu.mad.hi.u32 %0, %1, %2, %0;" : "+r"(ab1) : "r"(a & 0xff000000u), "r"(b & 0xff000000u));
    apack.native = __byte_perm(ab0, ab1, 0x6420);
    return apack;
  }
};

#define SPECIALIZE_REDUCE(Fn, T, EltPerPack, Vec, expr_of_fn_x_y) \
  template<> \
  struct Apply_Reduce<Fn<T>, EltPerPack> { \
    __device__ __forceinline__ static BytePack<sizeof(Vec)> reduce( \
        Fn<T> fn, BytePack<sizeof(Vec)> a, BytePack<sizeof(Vec)> b) { \
      Vec x = fromPack<Vec>(a); \
      Vec y = fromPack<Vec>(b); \
      return toPack<Vec>(expr_of_fn_x_y); \
    } \
  };

SPECIALIZE_REDUCE(FuncMinMax, float, 1, float, fn.isMinNotMax ? fminf(x, y) : fmaxf(x, y))
SPECIALIZE_REDUCE(FuncMinMax, double, 1, double, fn.isMinNotMax ? fmin(x, y) : fmax(x, y))

SPECIALIZE_REDUCE(FuncSum, float, 2, float2, make_float2(x.x + y.x, x.y + y.y))
SPECIALIZE_REDUCE(FuncProd, float, 2, float2, make_float2(x.x * y.x, x.y * y.y))

SPECIALIZE_REDUCE(FuncSum, half, 1, half, __hadd(x, y))
SPECIALIZE_REDUCE(FuncSum, half, 2, half2, __hadd2(x, y))
SPECIALIZE_REDUCE(FuncProd, half, 1, half, __hmul(x, y))
SPECIALIZE_REDUCE(FuncProd, half, 2, half2, __hmul2(x, y))
SPECIALIZE_REDUCE(FuncMinMax, half, 1, half, fn.isMinNotMax ? __hmin(x, y) : __hmax(x, y))
SPECIALIZE_REDUCE(FuncMinMax, half, 2, half2, fn.isMinNotMax ? __hmin2(x, y) : __hmax2(x, y))

SPECIALIZE_REDUCE(FuncSum, __ppu_bfloat16, 1, __ppu_bfloat16, __hadd(x, y))
SPECIALIZE_REDUCE(FuncSum, __ppu_bfloat16, 2, __ppu_bfloat162, __hadd2(x, y))
SPECIALIZE_REDUCE(FuncProd, __ppu_bfloat16, 1, __ppu_bfloat16, __hmul(x, y))
SPECIALIZE_REDUCE(FuncProd, __ppu_bfloat16, 2, __ppu_bfloat162, __hmul2(x, y))
SPECIALIZE_REDUCE(FuncMinMax, __ppu_bfloat16, 1, __ppu_bfloat16, fn.isMinNotMax ? __hmin(x, y) : __hmax(x, y))
SPECIALIZE_REDUCE(FuncMinMax, __ppu_bfloat16, 2, __ppu_bfloat162, fn.isMinNotMax ? __hmin2(x, y) : __hmax2(x, y))

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
SPECIALIZE_REDUCE(FuncSum, __hg_fp8_e4m3, 1, __hg_fp8_e4m3, __hg_fp8_e4m3(__hadd(__half(x), __half(y))))
SPECIALIZE_REDUCE(FuncSum, __hg_fp8_e4m3, 2, __hg_fp8x2_e4m3, __hg_fp8x2_e4m3(__hadd2(__half2(x), __half2(y))))
SPECIALIZE_REDUCE(FuncProd, __hg_fp8_e4m3, 1, __hg_fp8_e4m3, __hg_fp8_e4m3(__hmul(__half(x), __half(y))))
SPECIALIZE_REDUCE(FuncProd, __hg_fp8_e4m3, 2, __hg_fp8x2_e4m3, __hg_fp8x2_e4m3(__hmul2(__half2(x), __half2(y))))
SPECIALIZE_REDUCE(FuncMinMax, __hg_fp8_e4m3, 1, __hg_fp8_e4m3, __hg_fp8_e4m3(fn.isMinNotMax ? __hmin(__half(x), __half(y)) : __hmax(__half(x), __half(y))))
SPECIALIZE_REDUCE(FuncMinMax, __hg_fp8_e4m3, 2, __hg_fp8x2_e4m3, __hg_fp8x2_e4m3(fn.isMinNotMax ? __hmin2(__half2(x), __half2(y)) : __hmax2(__half2(x), __half2(y))))

SPECIALIZE_REDUCE(FuncSum, __hg_fp8_e5m2, 1, __hg_fp8_e5m2, __hg_fp8_e5m2(__hadd(__half(x), __half(y))))
SPECIALIZE_REDUCE(FuncSum, __hg_fp8_e5m2, 2, __hg_fp8x2_e5m2, __hg_fp8x2_e5m2(__hadd2(__half2(x), __half2(y))))
SPECIALIZE_REDUCE(FuncProd, __hg_fp8_e5m2, 1, __hg_fp8_e5m2, __hg_fp8_e5m2(__hmul(__half(x), __half(y))))
SPECIALIZE_REDUCE(FuncProd, __hg_fp8_e5m2, 2, __hg_fp8x2_e5m2, __hg_fp8x2_e5m2(__hmul2(__half2(x), __half2(y))))
SPECIALIZE_REDUCE(FuncMinMax, __hg_fp8_e5m2, 1, __hg_fp8_e5m2, __hg_fp8_e5m2(fn.isMinNotMax ? __hmin(__half(x), __half(y)) : __hmax(__half(x), __half(y))))
SPECIALIZE_REDUCE(FuncMinMax, __hg_fp8_e5m2, 2, __hg_fp8x2_e5m2, __hg_fp8x2_e5m2(fn.isMinNotMax ? __hmin2(__half2(x), __half2(y)) : __hmax2(__half2(x), __half2(y))))
#else
SPECIALIZE_REDUCE(FuncSum, __hg_fp8_e4m3, 1, __hg_fp8_e4m3, __hg_fp8_e4m3(0))
SPECIALIZE_REDUCE(FuncSum, __hg_fp8_e4m3, 2, __hg_fp8x2_e4m3, __hg_fp8x2_e4m3(make_double2(0.0, 0.0)))
SPECIALIZE_REDUCE(FuncProd, __hg_fp8_e4m3, 1, __hg_fp8_e4m3, __hg_fp8_e4m3(0))
SPECIALIZE_REDUCE(FuncProd, __hg_fp8_e4m3, 2, __hg_fp8x2_e4m3, __hg_fp8x2_e4m3(make_double2(0.0, 0.0)))
SPECIALIZE_REDUCE(FuncMinMax, __hg_fp8_e4m3, 1, __hg_fp8_e4m3, __hg_fp8_e4m3(0))
SPECIALIZE_REDUCE(FuncMinMax, __hg_fp8_e4m3, 2, __hg_fp8x2_e4m3, __hg_fp8x2_e4m3(make_double2(0.0, 0.0)))

SPECIALIZE_REDUCE(FuncSum, __hg_fp8_e5m2, 1, __hg_fp8_e5m2, __hg_fp8_e5m2(0))
SPECIALIZE_REDUCE(FuncSum, __hg_fp8_e5m2, 2, __hg_fp8x2_e5m2, __hg_fp8x2_e5m2(make_double2(0.0, 0.0)))
SPECIALIZE_REDUCE(FuncProd, __hg_fp8_e5m2, 1, __hg_fp8_e5m2, __hg_fp8_e5m2(0))
SPECIALIZE_REDUCE(FuncProd, __hg_fp8_e5m2, 2, __hg_fp8x2_e5m2, __hg_fp8x2_e5m2(make_double2(0.0, 0.0)))
SPECIALIZE_REDUCE(FuncMinMax, __hg_fp8_e5m2, 1, __hg_fp8_e5m2, __hg_fp8_e5m2(0))
SPECIALIZE_REDUCE(FuncMinMax, __hg_fp8_e5m2, 2, __hg_fp8x2_e5m2, __hg_fp8x2_e5m2(make_double2(0.0, 0.0)))
#endif

#undef SPECIALIZE_REDUCE

////////////////////////////////////////////////////////////////////////////////
// Apply_PreOp

// General recursive definition (EltPerPack > 1).
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
  __device__ __forceinline__ static BytePack<0> preOp(Fn fn, BytePack<0> a) { return {}; }
};

////////////////////////////////////////////////////////////////////////////////
// Apply_PostOp

// General recursive definition (EltPerPack > 1).
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
  __device__ __forceinline__ static BytePack<0> postOp(Fn fn, BytePack<0> a) { return {}; }
};

////////////////////////////////////////////////////////////////////////////////
// FuncPreMulSum

// PreOp of FuncPreMulSum for integral types, float, and double.
template<typename T>
struct Apply_PreOp<FuncPreMulSum<T>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(T)> preOp(FuncPreMulSum<T> fn, BytePack<sizeof(T)> a) {
    return toPack<T>(fromPack<T>(a) * fn.scale);
  }
};

////////////////////////////////////////////////////////////////////////////////
// Apply_PreOp of FuncPreMulSum for half.

template<>
struct Apply_PreOp<FuncPreMulSum<half>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(half)> preOp(FuncPreMulSum<half> fn, BytePack<sizeof(half)> a) {
    return toPack<half>(__hmul(fromPack<half>(a), fn.scale.x));
  }
};
template<>
struct Apply_PreOp<FuncPreMulSum<half>, /*EltPerPack=*/2> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(half2)> preOp(FuncPreMulSum<half> fn, BytePack<sizeof(half2)> a) {
    return toPack<half2>(__hmul2(fromPack<half2>(a), fn.scale));
  }
};

////////////////////////////////////////////////////////////////////////////////
// Apply_PreOp of FuncPreMulSum for bfloat16.

template<>
struct Apply_PreOp<FuncPreMulSum<__ppu_bfloat16>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(__ppu_bfloat16)> preOp(FuncPreMulSum<__ppu_bfloat16> fn, BytePack<sizeof(__ppu_bfloat16)> a) {
    return toPack<__ppu_bfloat16>(__hmul(fromPack<__ppu_bfloat16>(a), fn.scale.x));
  }
};
template<>
struct Apply_PreOp<FuncPreMulSum<__ppu_bfloat16>, /*EltPerPack=*/2> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(__ppu_bfloat162)> preOp(FuncPreMulSum<__ppu_bfloat16> fn, BytePack<sizeof(__ppu_bfloat162)> a) {
    return toPack<__ppu_bfloat162>(__hmul2(fromPack<__ppu_bfloat162>(a), fn.scale));
  }
};

////////////////////////////////////////////////////////////////////////////////
// Apply_PreOp of FuncPreMulSum for fp8.

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
template<>
struct Apply_PreOp<FuncPreMulSum<__hg_fp8_e4m3>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(__hg_fp8_e4m3)> preOp(FuncPreMulSum<__hg_fp8_e4m3> fn, BytePack<sizeof(__hg_fp8_e4m3)> a) {
    return toPack<__hg_fp8_e4m3>(__hg_fp8_e4m3(__hmul(__half(fromPack<__hg_fp8_e4m3>(a)), fn.scale.x)));
  }
};
template<>
struct Apply_PreOp<FuncPreMulSum<__hg_fp8_e4m3>, /*EltPerPack=*/2> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(__hg_fp8x2_e4m3)> preOp(FuncPreMulSum<__hg_fp8_e4m3> fn, BytePack<sizeof(__hg_fp8x2_e4m3)> a) {
    return toPack<__hg_fp8x2_e4m3>(__hg_fp8x2_e4m3(__hmul2(__half2(fromPack<__hg_fp8x2_e4m3>(a)), fn.scale)));
  }
};

template<>
struct Apply_PreOp<FuncPreMulSum<__hg_fp8_e5m2>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(__hg_fp8_e5m2)> preOp(FuncPreMulSum<__hg_fp8_e5m2> fn, BytePack<sizeof(__hg_fp8_e5m2)> a) {
    return toPack<__hg_fp8_e5m2>(__hg_fp8_e5m2(__hmul(__half(fromPack<__hg_fp8_e5m2>(a)), fn.scale.x)));
  }
};
template<>
struct Apply_PreOp<FuncPreMulSum<__hg_fp8_e5m2>, /*EltPerPack=*/2> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(__hg_fp8x2_e5m2)> preOp(FuncPreMulSum<__hg_fp8_e5m2> fn, BytePack<sizeof(__hg_fp8x2_e5m2)> a) {
    return toPack<__hg_fp8x2_e5m2>(__hg_fp8x2_e5m2(__hmul2(__half2(fromPack<__hg_fp8x2_e5m2>(a)), fn.scale)));
  }
};
#else
template<>
struct Apply_PreOp<FuncPreMulSum<__hg_fp8_e4m3>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(__hg_fp8_e4m3)> preOp(FuncPreMulSum<__hg_fp8_e4m3> fn, BytePack<sizeof(__hg_fp8_e4m3)> a) {
    return toPack<__hg_fp8_e4m3>(__hg_fp8_e4m3(0));
  }
};
template<>
struct Apply_PreOp<FuncPreMulSum<__hg_fp8_e4m3>, /*EltPerPack=*/2> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(__hg_fp8x2_e4m3)> preOp(FuncPreMulSum<__hg_fp8_e4m3> fn, BytePack<sizeof(__hg_fp8x2_e4m3)> a) {
    return toPack<__hg_fp8x2_e4m3>(__hg_fp8x2_e4m3(make_double2(0.0, 0.0)));
  }
};

template<>
struct Apply_PreOp<FuncPreMulSum<__hg_fp8_e5m2>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(__hg_fp8_e5m2)> preOp(FuncPreMulSum<__hg_fp8_e5m2> fn, BytePack<sizeof(__hg_fp8_e5m2)> a) {
    return toPack<__hg_fp8_e5m2>(__hg_fp8_e5m2(0));
  }
};
template<>
struct Apply_PreOp<FuncPreMulSum<__hg_fp8_e5m2>, /*EltPerPack=*/2> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(__hg_fp8x2_e5m2)> preOp(FuncPreMulSum<__hg_fp8_e5m2> fn, BytePack<sizeof(__hg_fp8x2_e5m2)> a) {
    return toPack<__hg_fp8x2_e5m2>(__hg_fp8x2_e5m2(make_double2(0.0, 0.0)));
  }
};
#endif

////////////////////////////////////////////////////////////////////////////////
// FuncSumPostDiv

template<typename T, bool IsFloating=IsFloatingPoint<T>::value>
struct FuncSumPostDiv;

template<typename T>
struct RedOpArg<FuncSumPostDiv<T>> {
  static constexpr bool ArgUsed = true;
  __device__ __forceinline__ static uint64_t loadArg(void *ptr) {
    return *(uint64_t*)ptr;
  }
};

template<typename T>
struct FuncSumPostDiv<T, /*IsFloating=*/false>: FuncSum<T> {
  static_assert(T(0) < T(-1), "FuncSumPostDiv is only for implementing ncclAvg on uint types.");
  using UintType = typename std::conditional<sizeof(T) == 8, uint64_t, uint32_t>::type;
  static constexpr bool IsPreOpIdentity = true;
  static constexpr bool IsPostOpIdentity = false;
  uint32_t divisor:31, isSigned:1;
  UintType recip;
  __device__ __forceinline__ FuncSumPostDiv(uint64_t opArg) {
    isSigned = opArg & 1;
    divisor = opArg >> 1;
    recip = UintType(-1) / divisor;
  }
  __device__ T preOp(T x) const { return x; }
  __device__ __forceinline__ T divide(T x) const {
    // x is negative iff we are in signed mode and the top bit is set.
    bool xneg = isSigned && (x & ~(T(-1)>>1));
    // Compute abs(x):
    // T(-x), not -T(x): the latter negates after the implicit promotion to int,
    // so for T=uint8_t and x=0xff (i.e. -1) it yields 0xffffff01 instead of the
    // 0x1 we want.
    UintType xabs = xneg ? T(-x) : x;
    // Compute quotient by multiplying by reciprocal.
    UintType q = sizeof(T) == 8 ? __umul64hi(xabs, recip) : __umulhi(xabs, recip);
    // Quotient may be off by one so do a fixup.
    if (xabs - q * divisor >= divisor) q += 1;
    // If original x was negative then we have to negate it back since we were
    // working with its abs val.
    return xneg ? -T(q) : T(q);
  }
  __device__ T postOp(T x) const { return divide(x); }
};

template<typename T>
struct FuncSumPostDiv<T, /*IsFloating=*/true> {
  static_assert(sizeof(T)!=sizeof(T), "FuncSumPostDiv is only for implementing ncclAvg on integral types.");
};

template<typename T>
struct FuncTraits<FuncSumPostDiv<T>> {
  static constexpr bool IsPreOpIdentity = FuncSumPostDiv<T>::IsPreOpIdentity;
  static constexpr bool IsPostOpIdentity = FuncSumPostDiv<T>::IsPostOpIdentity;

  template<typename U>
  __device__ static U preOp(FuncSumPostDiv<T> fn, U x) {
    return fn.preOp(x);
  }
  template<typename U>
  __device__ static U postOp(FuncSumPostDiv<T> fn, U x) {
    return fn.postOp(x);
  }
};

template<typename T>
struct RedOpIdentityArg<FuncSumPostDiv<T, /*IsFloating=*/false>> {
  static __device__ uint64_t value() { return 1u<<1; }
};

template<typename T, bool IsFloating>
struct FuncEltType<FuncSumPostDiv<T, IsFloating>> { using type = T; };

template<typename T, int EltPerPack>
struct Apply_Reduce<FuncSumPostDiv<T, /*IsFloating=*/false>, EltPerPack> {
  __device__ __forceinline__ static BytePack<EltPerPack * sizeof(T)> reduce(
      FuncSumPostDiv<T, false> fn, BytePack<EltPerPack * sizeof(T)> a, BytePack<EltPerPack * sizeof(T)> b) {
    // FuncSumPostDiv reduce dispatches to FuncSum; the division is a postOp.
    return Apply_Reduce<FuncSum<T>, EltPerPack>::reduce(FuncSum<T>(), a, b);
  }
};

template<typename T>
struct Apply_PostOp<FuncSumPostDiv<T, /*IsFloating=*/false>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(T)> postOp(FuncSumPostDiv<T, false> fn, BytePack<sizeof(T)> a) {
    return toPack<T>(fn.divide(fromPack<T>(a)));
  }
};

////////////////////////////////////////////////////////////////////////////////
// reduceCopy: pack-wise reduce+copy, shared by the protocol implementations.

template<typename RedFn, typename T, int Unroll, int BytePerPack,
         int MultimemSrcs, int MinSrcs, int MaxSrcs,
         int MultimemDsts, int MinDsts, int MaxDsts,
         int PreOpSrcs, typename IntBytes, typename SrcPtrFn, typename DstPtrFn>
__device__ __forceinline__ void reduceCopyPacks(
    int nThreads, int &thread, uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, SrcPtrFn const &srcPtrFn, int nDsts, DstPtrFn const &dstPtrFn,
    IntBytes &nBytesBehind, IntBytes &nBytesAhead
  ) {
  static_assert(std::is_signed<IntBytes>::value, "IntBytes must be a signed integral type.");
  static_assert(BytePerPack != 0, "BytePerPack must be non-zero.");
  static_assert(MultimemSrcs == 0 && MultimemDsts == 0,
                "PPU has no multimem.* instruction; multimem reduce/copy is not implemented.");
  static_assert(MinSrcs >= 1, "reduceCopyPacks needs at least one guaranteed source.");

  constexpr int BytePerHunk = Unroll*WARP_SIZE*BytePerPack;
  int nWarps = nThreads/WARP_SIZE;
  int warp = thread/WARP_SIZE;
  int lane = thread%WARP_SIZE;

  // This thread's initial position.
  IntBytes threadBytesBehind = nBytesBehind + (warp*BytePerHunk + lane*BytePerPack);
  IntBytes threadBytesAhead = nBytesAhead - (warp*BytePerHunk + lane*BytePerPack);
  // Number of hunks to be consumed over all warps.
  IntBytes nHunksAhead = nBytesAhead/(BytePerHunk + !BytePerHunk);
  // Advance collective position.
  nBytesBehind += nHunksAhead*BytePerHunk;
  nBytesAhead -= nHunksAhead*BytePerHunk;
  if (Unroll==1 && BytePerPack <= nBytesAhead) {
    // Only Unroll=1 can do partial hunks (where not all threads partake).
    nHunksAhead += 1;
    nBytesBehind += nBytesAhead - (nBytesAhead%(BytePerPack + !BytePerPack));
    nBytesAhead = nBytesAhead%(BytePerPack + !BytePerPack);
  }
  nHunksAhead -= warp;

  RedFn redFn(redArg);
  uintptr_t minSrcs[MinSrcs + !MinSrcs];
  uintptr_t minDsts[MinDsts + !MinDsts];
  #pragma unroll
  for (int s=0; s < MinSrcs; s++)
    minSrcs[s] = cvta_to_global(srcPtrFn(s)) + threadBytesBehind;
  #pragma unroll
  for (int d=0; d < MinDsts; d++)
    minDsts[d] = cvta_to_global(dstPtrFn(d)) + threadBytesBehind;

  // The loop termination condition depends on whether partial hunks can be
  // handled or not.
  while (Unroll==1 ? (BytePerPack <= threadBytesAhead) : (0 < nHunksAhead)) {
    BytePack<BytePerPack> acc[Unroll];

    { // minSrcs[0] cannot be nullptr, so it is always processed.
      RedFn preFn(0 < PreOpSrcs ? preOpArgs[0] : 0);
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        // Volatile loads, in case credits are polled for with volatile rather
        // than acquire.
        acc[u] = ld_volatile_global<BytePerPack>(minSrcs[0]);
        if (0 < PreOpSrcs) acc[u] = applyPreOp(preFn, acc[u]);
        minSrcs[0] += WARP_SIZE*BytePerPack;
      }
    }

    #pragma unroll (MinSrcs - 1 + !(MinSrcs - 1))
    for (int s=1; s < MinSrcs; s++) {
      BytePack<BytePerPack> tmp[Unroll];
      RedFn preFn(s < PreOpSrcs ? preOpArgs[s] : 0);
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        tmp[u] = ld_volatile_global<BytePerPack>(minSrcs[s]);
        minSrcs[s] += WARP_SIZE*BytePerPack;
      }
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        if (s < PreOpSrcs) tmp[u] = applyPreOp(preFn, tmp[u]);
        acc[u] = applyReduce(redFn, acc[u], tmp[u]);
      }
    }

    for (int s=MinSrcs; (MinSrcs < MaxSrcs) && (s < MaxSrcs) && (s < nSrcs); s++) {
      uintptr_t src = cvta_to_global(srcPtrFn(s)) + threadBytesBehind;
      BytePack<BytePerPack> tmp[Unroll];
      RedFn preFn(s < PreOpSrcs ? preOpArgs[s] : 0);
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        tmp[u] = ld_volatile_global<BytePerPack>(src);
        src += WARP_SIZE*BytePerPack;
      }
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        if (s < PreOpSrcs) tmp[u] = applyPreOp(preFn, tmp[u]);
        acc[u] = applyReduce(redFn, acc[u], tmp[u]);
      }
    }

    if (postOp) {
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++)
        acc[u] = applyPostOp(redFn, acc[u]);
    }

    #pragma unroll (MinDsts + !MinDsts)
    for (int d=0; d < MinDsts; d++) {
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        st_volatile_global<BytePerPack>(minDsts[d], acc[u]);
        minDsts[d] += WARP_SIZE*BytePerPack;
      }
    }
    for (int d=MinDsts; (MinDsts < MaxDsts) && (d < MaxDsts) && (d < nDsts); d++) {
      uintptr_t dst = cvta_to_global(dstPtrFn(d)) + threadBytesBehind;
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        st_volatile_global<BytePerPack>(dst, acc[u]);
        dst += WARP_SIZE*BytePerPack;
      }
    }

    nWarps = nThreads/WARP_SIZE;
    #pragma unroll
    for (int s=0; s < MinSrcs; s++) minSrcs[s] += (nWarps-1)*BytePerHunk;
    #pragma unroll
    for (int d=0; d < MinDsts; d++) minDsts[d] += (nWarps-1)*BytePerHunk;
    threadBytesBehind += nWarps*BytePerHunk;
    threadBytesAhead -= nWarps*BytePerHunk;
    nHunksAhead -= nWarps;
  }

  nWarps = nThreads/WARP_SIZE;
  warp = thread/WARP_SIZE;
  lane = thread%WARP_SIZE;
  if (Unroll==1 && nHunksAhead > 0) nHunksAhead -= nWarps;
  warp = -nHunksAhead;
  thread = warp*WARP_SIZE + lane;
}

template<int Unroll, typename RedFn, typename T,
         int MultimemSrcs, int MinSrcs, int MaxSrcs,
         int MultimemDsts, int MinDsts, int MaxDsts,
         int PreOpSrcs, typename IntBytes, typename SrcPtrFn, typename DstPtrFn>
__device__ __forceinline__ void reduceCopy(
    int thread, int nThreads, uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, SrcPtrFn const &srcPtrFn, int nDsts, DstPtrFn const &dstPtrFn,
    IntBytes nElts
  ) {
  static_assert(MultimemSrcs <= MinSrcs && MultimemDsts <= MinDsts,
                "Multimem pointers cannot exceed respective Min values.");
  static_assert(MultimemSrcs == 0 && MultimemDsts == 0,
                "PPU has no multimem.* instruction; multimem reduce/copy is not implemented.");

  constexpr int BigPackSize = (MultimemSrcs == 0) ? 16 : LoadMultimem_BigPackSize<RedFn>::BigPackSize;

  if (MaxDsts==0) return;
  if (MinDsts==0 && nDsts==0) return;

  IntBytes nBytesBehind = 0;
  IntBytes nBytesAhead = nElts*sizeof(T);

  #if __cpp_if_constexpr
  if constexpr (BigPackSize > sizeof(T)) {
  #else
  if (BigPackSize > sizeof(T)) {
  #endif
    // Take the wide path only if every pointer this warp holds is BigPackSize
    // aligned.
    int lane = thread%WARP_SIZE;
    bool aligned = true;
    if (lane < nSrcs) aligned &= 0 == cvta_to_global(srcPtrFn(lane))%(BigPackSize + !BigPackSize);
    if (lane < nDsts) aligned &= 0 == cvta_to_global(dstPtrFn(lane))%(BigPackSize + !BigPackSize);
    aligned = __all_sync(~0u, aligned);
    if (aligned) {
      reduceCopyPacks<RedFn, T, Unroll, BigPackSize,
        MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
        (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
         nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead);
      if (nBytesAhead == 0) return;

      reduceCopyPacks<RedFn, T, /*Unroll=*/1, BigPackSize,
        MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
        (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
         nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead);
      if (nBytesAhead == 0) return;
    }
  }

  reduceCopyPacks<RedFn, T, Unroll*(16/sizeof(T))/2, /*BytePerPack=*/sizeof(T),
    MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
    (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
     nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead);
  if (nBytesAhead == 0) return;

  reduceCopyPacks<RedFn, T, /*Unroll=*/1, /*BytePerPack=*/sizeof(T),
    MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
    (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
     nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead);
}

struct ReduceCopyPtrArray {
  void **ptrs;
  __device__ __forceinline__ ReduceCopyPtrArray(void **ptrs): ptrs(ptrs) {}
  __device__ __forceinline__ void* operator()(int i) const { return ptrs[i]; }
};

template<int Unroll, typename RedFn, typename T,
         int MultimemSrcs, int MinSrcs, int MaxSrcs,
         int MultimemDsts, int MinDsts, int MaxDsts,
         int PreOpSrcs, typename IntBytes>
__device__ __forceinline__ void reduceCopy(
    int thread, int nThreads, uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, void **srcPtrs, int nDsts, void **dstPtrs, IntBytes nElts
  ) {
  reduceCopy<Unroll, RedFn, T, MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
    (thread, nThreads, redArg, preOpArgs, postOp,
     nSrcs, ReduceCopyPtrArray(srcPtrs), nDsts, ReduceCopyPtrArray(dstPtrs), nElts);
}

#endif // REDUCE_KERNEL_H_
