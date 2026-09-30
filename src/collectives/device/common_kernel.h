/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_COMMON_KERNEL_H_
#define NCCL_COMMON_KERNEL_H_

#include "devcomm.h"
#include "utils.h"
#include <cstdio>
#include <cstdint>

#include <hggc_runtime.h>

#ifndef NO_TSM
extern __shared__ struct ncclShmemData pcclShmem;
#endif

__inline__ __device__ struct ncclShmemData* getShmemPtr(struct ncclKernelComm* comm) {
#ifdef NO_TSM
  return comm->channels[blockIdx.x].shmem;
#else
  return &pcclShmem;
#endif
}

__inline__ __device__ struct ncclKernelComm* getShmemComm(struct ncclDevWorkP2p* work) {
#ifdef NO_TSM
  return work->comm;
#else
  return &pcclShmem.comm;
#endif
}

__inline__ __device__ struct ncclKernelComm* getShmemComm(struct ncclDevWorkColl* work) {
#ifdef NO_TSM
  return work->comm;
#else
  return &pcclShmem.comm;
#endif
}

// Define min for ssize_t
static __device__ int min(int a, ssize_t b) { return (a < b) ? a : b; }

inline __device__ int loadInt(int* ptr) {
  int v;
  asm volatile("ppu.ld.volatile.global.u32 %0, [%1];"
      : "=r"(v) : "l"(ptr));
  return v;
}

template <typename T>
inline __device__ void loadPtr(void** ptr, T* &v) {
  void* volatile* tmp = ptr;
  union {
    T* ptrPeer;
    uint2 ptrLoadValue;
  };
  ptrLoadValue = __ppu_global_load_volatile_b32x2(tmp);
  v = ptrPeer;
}

template <typename T>
inline __device__ void storePtr(void** ptr, T* v) {
  volatile void* tmp = ptr;
  union {
    T* ptrPeer;
    uint2 ptrLoadValue;
  };
  ptrPeer = v;
  __ppu_global_store_volatile_b32x2(tmp, ptrLoadValue);
}

inline __device__ void printPtrValue(int rank, int bid, int tid,
    const char* ptrName, const void* ptrValue) {
  printU64Value(rank, bid, tid, ptrName, uint64_t(ptrValue));
}

typedef uint64_t PackType;

template<typename Fn>
struct FuncTraits /*{
  __device__ static Fn make();
  __device__ static T preOp(Fn, T);
  __device__ static T postOp(Fn, T);
}*/;

// unpack x and y to elements of type T and apply FUNC to each element
template<class FUNC, typename T>
struct MULTI {
  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const;
  __device__ PackType preOp(FUNC fn, PackType x) const;
  __device__ PackType postOp(FUNC fn, PackType x) const;
};

template<class FUNC>
struct MULTI<FUNC, int8_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(uint32_t),
      "PackType must be twice the size of uint32_t.");
  union converter {
    PackType storage;
    struct {
      uint32_t a, b;
    };
  };

  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    // for char, we do these as vector ops
    cr.a = fn(cx.a, cy.a);
    cr.b = fn(cx.b, cy.b);

    return cr.storage;
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      int8_t elt[8];
    } u;
    u.pack = x;
    #pragma unroll
    for (int i=0; i < 8; i++)
      u.elt[i] = FuncTraits<FUNC>().preOp(fn, u.elt[i]);
    return u.pack;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      int8_t elt[8];
    } u;
    u.pack = x;
    #pragma unroll
    for (int i=0; i < 8; i++)
      u.elt[i] = FuncTraits<FUNC>().postOp(fn, u.elt[i]);
    return u.pack;
  }
};

template<class FUNC>
struct MULTI<FUNC, uint8_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(uint32_t),
      "PackType must be twice the size of uint32_t.");
  union converter {
    PackType storage;
    struct {
      uint32_t a, b;
    };
  };

  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    // for char, we do these as vector ops
    cr.a = fn(cx.a, cy.a);
    cr.b = fn(cx.b, cy.b);

    return cr.storage;
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      uint8_t elt[8];
    } u;
    u.pack = x;
    #pragma unroll
    for (int i=0; i < 8; i++)
      u.elt[i] = FuncTraits<FUNC>().preOp(fn, u.elt[i]);
    return u.pack;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      uint8_t elt[8];
    } u;
    u.pack = x;
    #pragma unroll
    for (int i=0; i < 8; i++)
      u.elt[i] = FuncTraits<FUNC>().postOp(fn, u.elt[i]);
    return u.pack;
  }
};

template<class FUNC>
struct MULTI<FUNC, int32_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(int32_t),
      "PackType must be twice the size of int.");
  union converter {
    PackType storage;
    struct {
      int32_t a, b;
    };
  };

  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = fn(cx.a, cy.a);
    cr.b = fn(cx.b, cy.b);

    return cr.storage;
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      int32_t elt[2];
    } u;
    u.pack = x;
    u.elt[0] = FuncTraits<FUNC>().preOp(fn, u.elt[0]);
    u.elt[1] = FuncTraits<FUNC>().preOp(fn, u.elt[1]);
    return u.pack;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      int32_t elt[2];
    } u;
    u.pack = x;
    u.elt[0] = FuncTraits<FUNC>().postOp(fn, u.elt[0]);
    u.elt[1] = FuncTraits<FUNC>().postOp(fn, u.elt[1]);
    return u.pack;
  }
};

template<class FUNC>
struct MULTI<FUNC, uint32_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(uint32_t),
      "PackType must be twice the size of int.");
  union converter {
    PackType storage;
    struct {
      uint32_t a, b;
    };
  };

  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = fn(cx.a, cy.a);
    cr.b = fn(cx.b, cy.b);

    return cr.storage;
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      uint32_t elt[2];
    } u;
    u.pack = x;
    u.elt[0] = FuncTraits<FUNC>().preOp(fn, u.elt[0]);
    u.elt[1] = FuncTraits<FUNC>().preOp(fn, u.elt[1]);
    return u.pack;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      uint32_t elt[2];
    } u;
    u.pack = x;
    u.elt[0] = FuncTraits<FUNC>().postOp(fn, u.elt[0]);
    u.elt[1] = FuncTraits<FUNC>().postOp(fn, u.elt[1]);
    return u.pack;
  }
};

template<class FUNC>
struct MULTI<FUNC, half> {
  static_assert(sizeof(PackType) == 4 * sizeof(half),
      "PackType must be four times the size of half.");

  union Converter {
    PackType pack;
    half2 h2[2];
  };
  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    Converter cx, cy, cr;
    cx.pack = x;
    cy.pack = y;
    cr.h2[0] = fn(cx.h2[0], cy.h2[0]);
    cr.h2[1] = fn(cx.h2[1], cy.h2[1]);
    return cr.pack;
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    Converter c;
    c.pack = x;
    c.h2[0] = FuncTraits<FUNC>().preOp(fn, c.h2[0]);
    c.h2[1] = FuncTraits<FUNC>().preOp(fn, c.h2[1]);
    return c.pack;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    Converter c;
    c.pack = x;
    c.h2[0] = FuncTraits<FUNC>().postOp(fn, c.h2[0]);
    c.h2[1] = FuncTraits<FUNC>().postOp(fn, c.h2[1]);
    return c.pack;
  }
};

template<class FUNC>
struct MULTI<FUNC, __ppu_bfloat16> {
  static_assert(sizeof(PackType) == 4 * sizeof(__ppu_bfloat16),
      "PackType must be four times the size of __ppu_bfloat16.");

  union Converter {
    PackType pack;
    __ppu_bfloat162 h2[2];
  };
  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    Converter cx, cy, cr;
    cx.pack = x;
    cy.pack = y;
    cr.h2[0] = fn(cx.h2[0], cy.h2[0]);
    cr.h2[1] = fn(cx.h2[1], cy.h2[1]);
    return cr.pack;
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    Converter c;
    c.pack = x;
    c.h2[0] = FuncTraits<FUNC>().preOp(fn, c.h2[0]);
    c.h2[1] = FuncTraits<FUNC>().preOp(fn, c.h2[1]);
    return c.pack;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    Converter c;
    c.pack = x;
    c.h2[0] = FuncTraits<FUNC>().postOp(fn, c.h2[0]);
    c.h2[1] = FuncTraits<FUNC>().postOp(fn, c.h2[1]);
    return c.pack;
  }
};

template<class FUNC>
struct MULTI<FUNC, __hg_fp8_e4m3> {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
  static_assert(sizeof(PackType) == 4 * sizeof(__hg_fp8x2_e4m3),
    "PackType must be four times the size of __hg_fp8x2_e4m3.");

  union Converter {
    PackType pack;
    __hg_fp8x2_e4m3 f2[4];
  };
  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    Converter cx, cy, cr;
    cx.pack = x;
    cy.pack = y;
    for (int i = 0; i < 4; i++) {
      cr.f2[i] = fn(cx.f2[i], cy.f2[i]);
    }
    return cr.pack;
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    Converter c;
    c.pack = x;
    for (int i = 0; i < 4; i++) {
      c.f2[i] = FuncTraits<FUNC>().preOp(fn, c.f2[i]);
    }
    return c.pack;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    Converter c;
    c.pack = x;
    for (int i = 0; i < 4; i++) {
      c.f2[i] = FuncTraits<FUNC>().postOp(fn, c.f2[i]);
    }
    return c.pack;
  }
#else
  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    return 0;
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    return 0;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    return 0;
  }
#endif
};

template<class FUNC>
struct MULTI<FUNC, __hg_fp8_e5m2> {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= 150)
  static_assert(sizeof(PackType) == 4 * sizeof(__hg_fp8x2_e5m2),
    "PackType must be four times the size of __hg_fp8x2_e5m2.");

  union Converter {
    PackType pack;
    __hg_fp8x2_e5m2 f2[4];
  };
  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    Converter cx, cy, cr;
    cx.pack = x;
    cy.pack = y;
    for (int i = 0; i < 4; i++) {
      cr.f2[i] = fn(cx.f2[i], cy.f2[i]);
    }
    return cr.pack;
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    Converter c;
    c.pack = x;
    for (int i = 0; i < 4; i++) {
      c.f2[i] = FuncTraits<FUNC>().preOp(fn, c.f2[i]);
    }
    return c.pack;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    Converter c;
    c.pack = x;
    for (int i = 0; i < 4; i++) {
      c.f2[i] = FuncTraits<FUNC>().postOp(fn, c.f2[i]);
    }
    return c.pack;
  }
#else
  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    return 0;
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    return 0;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    return 0;
  }
#endif
};

template<class FUNC>
struct MULTI<FUNC, float> {
  static_assert(sizeof(PackType) == 2 * sizeof(float),
      "PackType must be twice the size of float.");
  union converter {
    PackType storage;
    struct {
      float a, b;
    };
  };

  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = fn(cx.a, cy.a);
    cr.b = fn(cx.b, cy.b);

    return cr.storage;
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      float elt[2];
    } u;
    u.pack = x;
    u.elt[0] = FuncTraits<FUNC>().preOp(fn, u.elt[0]);
    u.elt[1] = FuncTraits<FUNC>().preOp(fn, u.elt[1]);
    return u.pack;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      float elt[2];
    } u;
    u.pack = x;
    u.elt[0] = FuncTraits<FUNC>().postOp(fn, u.elt[0]);
    u.elt[1] = FuncTraits<FUNC>().postOp(fn, u.elt[1]);
    return u.pack;
  }
};

template<class FUNC>
struct MULTI<FUNC, double> {
  static_assert(sizeof(PackType) == sizeof(double),
      "PackType must be the same size as double.");
  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    double rv = fn(__longlong_as_double(x), __longlong_as_double(y));
    return __double_as_longlong(rv);
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      double elt;
    } u;
    u.pack = x;
    u.elt = FuncTraits<FUNC>().preOp(fn, u.elt);
    return u.pack;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      double elt;
    } u;
    u.pack = x;
    u.elt = FuncTraits<FUNC>().postOp(fn, u.elt);
    return u.pack;
  }
};

template<class FUNC>
struct MULTI<FUNC, uint64_t> {
  static_assert(sizeof(PackType) == sizeof(uint64_t),
      "PackType must be the same size as uint64_t.");
  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    uint64_t rv = fn(x, y);
    return rv;
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      uint64_t elt;
    } u;
    u.pack = x;
    u.elt = FuncTraits<FUNC>().preOp(fn, u.elt);
    return u.pack;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      uint64_t elt;
    } u;
    u.pack = x;
    u.elt = FuncTraits<FUNC>().postOp(fn, u.elt);
    return u.pack;
  }
};

template<class FUNC>
struct MULTI<FUNC, int64_t> {
  static_assert(sizeof(PackType) == sizeof(int64_t),
      "PackType must be the same size as int64_t.");
  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    int64_t rv = fn((int64_t)x, (int64_t)y);
    return rv;
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      int64_t elt;
    } u;
    u.pack = x;
    u.elt = FuncTraits<FUNC>().preOp(fn, u.elt);
    return u.pack;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      int64_t elt;
    } u;
    u.pack = x;
    u.elt = FuncTraits<FUNC>().postOp(fn, u.elt);
    return u.pack;
  }
};

template<typename T> inline __device__
T vFetch(const volatile T* ptr) {
#ifndef LOAD_BYPASS
  return *ptr;
#else
  return (T)0;
#endif
}

template<typename T> inline __device__
void vStore(volatile T* ptr, const T val) {
#ifndef STORE_BYPASS
  *ptr = val;
#endif
}

template<> inline __device__
half vFetch<half>(const volatile half* ptr) {
  half r;
#ifndef LOAD_BYPASS
  r = ptr[0];
#else
  r = (half)0;
#endif
  return r;
}

template<> inline __device__
void vStore<half>(volatile half* ptr, const half val) {
#ifndef STORE_BYPASS
  ptr[0] = val;
#endif
}

template<> inline __device__
__ppu_bfloat16 vFetch<__ppu_bfloat16>(const volatile __ppu_bfloat16* ptr) {
  __ppu_bfloat16 r;
#ifndef LOAD_BYPASS
  r = ptr[0];
#else
  r = (__ppu_bfloat16)0;
#endif
  return r;
}

template<> inline __device__
void vStore<__ppu_bfloat16>(volatile __ppu_bfloat16* ptr, const __ppu_bfloat16 val) {
#ifndef STORE_BYPASS
  ptr[0] = val;
#endif
}

template<> inline __device__
__hg_fp8_e4m3 vFetch<__hg_fp8_e4m3>(const volatile __hg_fp8_e4m3* ptr) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
#ifndef LOAD_BYPASS
  const volatile __hg_fp8_storage_t* storagePtr = reinterpret_cast<const volatile __hg_fp8_storage_t*>(ptr);
  const __hg_fp8_storage_t rawVal = storagePtr[0];
  union {
    __hg_fp8_e4m3 val;
    __hg_fp8_storage_t storageVal;
  } converter;
  converter.storageVal = rawVal;
  return converter.val;
#else
  union { __hg_fp8_e4m3 val; __hg_fp8_storage_t storageVal; } zeroConverter;
  zeroConverter.storageVal = 0;
  return zeroConverter.val;
#endif
#else
  return __hg_fp8_e4m3(0);
#endif
}

template<> inline __device__
void vStore<__hg_fp8_e4m3>(volatile __hg_fp8_e4m3* ptr, const __hg_fp8_e4m3 val) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
#ifndef STORE_BYPASS
  union {
    __hg_fp8_e4m3       val;
    __hg_fp8_storage_t  storageVal;
  } converter;
  converter.val = val;
  const __hg_fp8_storage_t rawVal = converter.storageVal;
  volatile __hg_fp8_storage_t* storagePtr = reinterpret_cast<volatile __hg_fp8_storage_t*>(ptr);
  storagePtr[0] = rawVal;
#endif
#endif
}

template<> inline __device__
__hg_fp8_e5m2 vFetch<__hg_fp8_e5m2>(const volatile __hg_fp8_e5m2* ptr) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
#ifndef LOAD_BYPASS
  const volatile __hg_fp8_storage_t* storagePtr = reinterpret_cast<const volatile __hg_fp8_storage_t*>(ptr);
  const __hg_fp8_storage_t rawVal = storagePtr[0];
  union {
    __hg_fp8_e5m2 val;
    __hg_fp8_storage_t storageVal;
  } converter;
  converter.storageVal = rawVal;
  return converter.val;
#else
  union { __hg_fp8_e5m2 val; __hg_fp8_storage_t storageVal; } zeroConverter;
  zeroConverter.storageVal = 0;
  return zeroConverter.val;
#endif
#else
  return __hg_fp8_e5m2(0);
#endif
}

template<> inline __device__
void vStore<__hg_fp8_e5m2>(volatile __hg_fp8_e5m2* ptr, const __hg_fp8_e5m2 val) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
#ifndef STORE_BYPASS
  union {
    __hg_fp8_e5m2       val;
    __hg_fp8_storage_t  storageVal;
  } converter;
  converter.val = val;
  const __hg_fp8_storage_t rawVal = converter.storageVal;
  volatile __hg_fp8_storage_t* storagePtr = reinterpret_cast<volatile __hg_fp8_storage_t*>(ptr);
  storagePtr[0] = rawVal;
#endif
#endif
}
// #endif
// #endif

union Pack128 {
  uint4  i4;
  ulong2 l2;
};

#define PACKELEMS (sizeof(Pack128) / sizeof(T))

template<class FUNC, typename T>
struct MULTI128 {
  __device__ void operator()(FUNC fn, Pack128& x, Pack128 const& y) const {
    x.l2.x = MULTI<FUNC, T>()(fn, x.l2.x, y.l2.x);
    x.l2.y = MULTI<FUNC, T>()(fn, x.l2.y, y.l2.y);
  }
  __device__ void preOp(FUNC fn, Pack128 &x) const {
    x.l2.x = MULTI<FUNC, T>().preOp(fn, x.l2.x);
    x.l2.y = MULTI<FUNC, T>().preOp(fn, x.l2.y);
  }
  __device__ void postOp(FUNC fn, Pack128 &x) const {
    x.l2.x = MULTI<FUNC, T>().postOp(fn, x.l2.x);
    x.l2.y = MULTI<FUNC, T>().postOp(fn, x.l2.y);
   }
};

template<class FUNC, typename T>
struct BULK_MULTI128 {
  __device__ void operator()(FUNC fn, Pack128& x, Pack128& y) const {
    x.l2.x = MULTI<FUNC, T>()(fn, x.l2.x, y.l2.x);
    x.l2.y = MULTI<FUNC, T>()(fn, x.l2.y, y.l2.y);
  }
  __device__ void preOp(FUNC fn, Pack128 &x) const {
    x.l2.x = MULTI<FUNC, T>().preOp(fn, x.l2.x);
    x.l2.y = MULTI<FUNC, T>().preOp(fn, x.l2.y);
  }
  __device__ void postOp(FUNC fn, Pack128 &x) const {
    x.l2.x = MULTI<FUNC, T>().postOp(fn, x.l2.x);
    x.l2.y = MULTI<FUNC, T>().postOp(fn, x.l2.y);
   }
};

template<class FUNC>
struct BULK_MULTI128<FUNC, int64_t> {
  __device__ void operator()(FUNC fn, Pack128& x, Pack128& y) const {
    x.i4 = __ppu_swizzle_bulk_b32x4(x.i4);
    y.i4 = __ppu_swizzle_bulk_b32x4(y.i4);
    x.l2.x = MULTI<FUNC, int64_t>()(fn, x.l2.x, y.l2.x);
    x.l2.y = MULTI<FUNC, int64_t>()(fn, x.l2.y, y.l2.y);
    x.i4 = __ppu_swizzle_bulk_b32x4(x.i4);
    y.i4 = __ppu_swizzle_bulk_b32x4(y.i4);
  }
  __device__ void preOp(FUNC fn, Pack128 &x) const {
    x.i4 = __ppu_swizzle_bulk_b32x4(x.i4);
    x.l2.x = MULTI<FUNC, int64_t>().preOp(fn, x.l2.x);
    x.l2.y = MULTI<FUNC, int64_t>().preOp(fn, x.l2.y);
    x.i4 = __ppu_swizzle_bulk_b32x4(x.i4);
  }
  __device__ void postOp(FUNC fn, Pack128 &x) const {
    x.i4 = __ppu_swizzle_bulk_b32x4(x.i4);
    x.l2.x = MULTI<FUNC, int64_t>().postOp(fn, x.l2.x);
    x.l2.y = MULTI<FUNC, int64_t>().postOp(fn, x.l2.y);
    x.i4 = __ppu_swizzle_bulk_b32x4(x.i4);
   }
};

template<class FUNC>
struct BULK_MULTI128<FUNC, uint64_t> {
  __device__ void operator()(FUNC fn, Pack128& x, Pack128& y) const {
    x.i4 = __ppu_swizzle_bulk_b32x4(x.i4);
    y.i4 = __ppu_swizzle_bulk_b32x4(y.i4);
    x.l2.x = MULTI<FUNC, uint64_t>()(fn, x.l2.x, y.l2.x);
    x.l2.y = MULTI<FUNC, uint64_t>()(fn, x.l2.y, y.l2.y);
    x.i4 = __ppu_swizzle_bulk_b32x4(x.i4);
    y.i4 = __ppu_swizzle_bulk_b32x4(y.i4);
  }
  __device__ void preOp(FUNC fn, Pack128 &x) const {
    x.i4 = __ppu_swizzle_bulk_b32x4(x.i4);
    x.l2.x = MULTI<FUNC, uint64_t>().preOp(fn, x.l2.x);
    x.l2.y = MULTI<FUNC, uint64_t>().preOp(fn, x.l2.y);
    x.i4 = __ppu_swizzle_bulk_b32x4(x.i4);
  }
  __device__ void postOp(FUNC fn, Pack128 &x) const {
    x.i4 = __ppu_swizzle_bulk_b32x4(x.i4);
    x.l2.x = MULTI<FUNC, uint64_t>().postOp(fn, x.l2.x);
    x.l2.y = MULTI<FUNC, uint64_t>().postOp(fn, x.l2.y);
    x.i4 = __ppu_swizzle_bulk_b32x4(x.i4);
   }
};

template<class FUNC>
struct BULK_MULTI128<FUNC, double> {
  __device__ void operator()(FUNC fn, Pack128& x, Pack128& y) const {
    x.i4 = __ppu_swizzle_bulk_b32x4(x.i4);
    y.i4 = __ppu_swizzle_bulk_b32x4(y.i4);
    x.l2.x = MULTI<FUNC, double>()(fn, x.l2.x, y.l2.x);
    x.l2.y = MULTI<FUNC, double>()(fn, x.l2.y, y.l2.y);
    x.i4 = __ppu_swizzle_bulk_b32x4(x.i4);
    y.i4 = __ppu_swizzle_bulk_b32x4(y.i4);
  }
  __device__ void preOp(FUNC fn, Pack128 &x) const {
    x.i4 = __ppu_swizzle_bulk_b32x4(x.i4);
    x.l2.x = MULTI<FUNC, double>().preOp(fn, x.l2.x);
    x.l2.y = MULTI<FUNC, double>().preOp(fn, x.l2.y);
    x.i4 = __ppu_swizzle_bulk_b32x4(x.i4);
  }
  __device__ void postOp(FUNC fn, Pack128 &x) const {
    x.i4 = __ppu_swizzle_bulk_b32x4(x.i4);
    x.l2.x = MULTI<FUNC, double>().postOp(fn, x.l2.x);
    x.l2.y = MULTI<FUNC, double>().postOp(fn, x.l2.y);
    x.i4 = __ppu_swizzle_bulk_b32x4(x.i4);
   }
};

inline __device__ void Fetch128(Pack128& v, const Pack128* p) {
#ifndef LOAD_BYPASS
  v.i4 = __ppu_global_load_volatile_b32x4((volatile const void*)(p));
#else
  v.i4 = make_uint4(0U, 0U, 0U, 0U);
#endif
}

inline __device__ void Store128(Pack128* p, Pack128& v) {
#ifndef STORE_BYPASS
  __ppu_global_store_volatile_b32x4((volatile void*)(p), v.i4);
#endif
}

template<class FUNC, typename T, int UNROLL, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS>
__device__ __forceinline__ void ReduceCopyMulti(const int w, const int nw, const int t,
    FUNC fn, bool preOpSrc0, bool postOp, int nsrcs, const T** s, int ndsts, T** d, const int elemOffset, const int Nelem) {
  const int inc = nw * UNROLL * WARP_SIZE;
  int offset = w * UNROLL * WARP_SIZE + t;

  const T* srcs[MAXSRCS];
  for (int i=0; i<MAXSRCS; i++) srcs[i] = s[i]+elemOffset+offset;
  T* dsts[MAXDSTS];
  for (int i=0; i<MAXDSTS; i++) dsts[i] = d[i]+elemOffset+offset;

  while (offset < Nelem) {
    T vals[UNROLL];
    // Load and reduce
    for (int u = 0; u < UNROLL; ++u) vals[u] = vFetch(srcs[0]+u*WARP_SIZE);
    if (preOpSrc0) {
      for (int u = 0; u < UNROLL; ++u) vals[u] = FuncTraits<FUNC>().preOp(fn, vals[u]);
    }

    #pragma unroll
    for (int i=1; i<MINSRCS; i++) {
      T vals2[UNROLL];
      for (int u = 0; u < UNROLL; ++u) vals2[u] = vFetch(srcs[i]+u*WARP_SIZE);
      for (int u = 0; u < UNROLL; ++u) vals[u] = fn(vals[u], vals2[u]);
    }
    #pragma unroll
    for (int i=MINSRCS; i<MAXSRCS; i++) {
      if (i<nsrcs) {
        T vals2[UNROLL];
        for (int u = 0; u < UNROLL; ++u) vals2[u] = vFetch(srcs[i]+u*WARP_SIZE);
        for (int u = 0; u < UNROLL; ++u) vals[u] = fn(vals[u], vals2[u]);
      }
    }

    if (postOp) {
      #pragma unroll
      for (int u = 0; u < UNROLL; ++u) vals[u] = FuncTraits<FUNC>().postOp(fn, vals[u]);
    }

    // Store
    #pragma unroll
    for (int i = 0; i < MINDSTS; i++) {
      for (int u = 0; u < UNROLL; ++u) vStore(dsts[i]+u*WARP_SIZE, vals[u]);
    }
    #pragma unroll
    for (int i=MINDSTS; i<MAXDSTS; i++) {
      if (i<ndsts) {
        for (int u = 0; u < UNROLL; ++u) vStore(dsts[i]+u*WARP_SIZE, vals[u]);
      }
    }
    for (int i=0; i<MAXSRCS; i++) srcs[i] += inc;
    for (int i=0; i<MAXDSTS; i++) dsts[i] += inc;
    offset += inc;
  }
}

template<class FUNC, typename T, int UNROLL, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS>
__device__ __forceinline__ void ReduceCopy128bMulti(const int w, const int nw, const int t,
    FUNC fn, bool preOpSrc0, bool postOp, int nsrcs, const T** s, int ndsts, T** d, const int elemOffset, const int Npack) {
  const int inc = nw * UNROLL * WARP_SIZE;
  int offset = w * UNROLL * WARP_SIZE + t;

  const Pack128* srcs[MAXSRCS];
  for (int i=0; i<MAXSRCS; i++) srcs[i] = ((const Pack128*)(s[i]+elemOffset))+offset;
  Pack128* dsts[MAXDSTS];
  for (int i=0; i<MAXDSTS; i++) dsts[i] = ((Pack128*)(d[i]+elemOffset))+offset;

  while (offset < Npack) {
    Pack128 vals[UNROLL];
    // Load and reduce
    for (int u = 0; u < UNROLL; ++u) Fetch128(vals[u], srcs[0]+u*WARP_SIZE);
    if (preOpSrc0) {
      for (int u = 0; u < UNROLL; ++u) MULTI128<FUNC, T>().preOp(fn, vals[u]);
    }

    #pragma unroll
    for (int i=1; i<MINSRCS; i++) {
      Pack128 vals2[UNROLL];
      for (int u = 0; u < UNROLL; ++u) Fetch128(vals2[u], srcs[i]+u*WARP_SIZE);
      for (int u = 0; u < UNROLL; ++u) MULTI128<FUNC, T>()(fn, vals[u], vals2[u]);
    }
    #pragma unroll
    for (int i=MINSRCS; i<MAXSRCS; i++) {
      if (i<nsrcs) {
        Pack128 vals2[UNROLL];
        for (int u = 0; u < UNROLL; ++u) Fetch128(vals2[u], srcs[i]+u*WARP_SIZE);
        for (int u = 0; u < UNROLL; ++u) MULTI128<FUNC, T>()(fn, vals[u], vals2[u]);
      }
    }

    if (postOp) {
      #pragma unroll
      for (int u = 0; u < UNROLL; ++u) MULTI128<FUNC, T>().postOp(fn, vals[u]);
    }

    // Store
    #pragma unroll
    for (int i = 0; i < MINDSTS; i++) {
      for (int u = 0; u < UNROLL; ++u) Store128(dsts[i]+u*WARP_SIZE, vals[u]);
    }
    #pragma unroll
    for (int i=MINDSTS; i<MAXDSTS; i++) {
      if (i<ndsts) {
        for (int u = 0; u < UNROLL; ++u) Store128(dsts[i]+u*WARP_SIZE, vals[u]);
      }
    }
    for (int i=0; i<MAXSRCS; i++) srcs[i] += inc;
    for (int i=0; i<MAXDSTS; i++) dsts[i] += inc;
    offset += inc;
  }
}

template <typename T>
__device__ int ptrAlign128(T* ptr) { return (uint64_t)ptr % alignof(Pack128); }

template<int UNROLL, class FUNC, typename T, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS>
__device__ __forceinline__ void ReduceOrCopyMulti(const int tid, const int nthreads,
    FUNC fn, bool preOpSrc0, bool postOp, int nsrcs, const T** srcs, int ndsts, T** dsts,
    int N) {
  int Nrem = N;
  if (Nrem <= 0) return;

  int w = tid / WARP_SIZE;       // Warp number
  int nw = nthreads / WARP_SIZE; // Number of warps
  int t = tid % WARP_SIZE;       // Thread (inside the warp)

  // Check that all is 16B aligned. If not don't use 16B load/stores.
  int align = 0;
  #pragma unroll
  for (int i=0; i<MINSRCS; i++) align |= ptrAlign128(srcs[i]);
  for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) align |= ptrAlign128(srcs[i]);
  #pragma unroll
  for (int i=0; i<MINDSTS; i++) align |= ptrAlign128(dsts[i]);
  for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) align |= ptrAlign128(dsts[i]);

  int offset = 0;
  if (align == 0) {
    // fast path: use 128b loads/stores to do the bulk of the work,
    // assuming the pointers we have are all 128-bit aligned.

    // main loop
    int Npack = (Nrem / (PACKELEMS*UNROLL*WARP_SIZE)) * (UNROLL*WARP_SIZE); // round down
    int Nelem = Npack * PACKELEMS;

    ReduceCopy128bMulti<FUNC, T, UNROLL, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(w, nw, t, fn, preOpSrc0, postOp, nsrcs, srcs, ndsts, dsts, offset, Npack);

    Nrem -= Nelem;
    if (Nrem == 0) return;
    offset += Nelem;

    // slightly less optimized for section when we don't have full unrolling
    Npack = Nrem / PACKELEMS;
    Nelem = Npack * PACKELEMS;

    ReduceCopy128bMulti<FUNC, T, 1, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(w, nw, t, fn, preOpSrc0, postOp, nsrcs, srcs, ndsts, dsts, offset, Npack);

    Nrem -= Nelem;
    if (Nrem == 0) return;
    offset += Nelem;
  }

  // unrolled, by-type (mostly for unaligned buffers)
  int Nelem = (Nrem / (UNROLL*PACKELEMS/2*WARP_SIZE)) * (UNROLL*PACKELEMS/2*WARP_SIZE); // round down

  ReduceCopyMulti<FUNC, T, UNROLL*PACKELEMS/2, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(w, nw, t, fn, preOpSrc0, postOp, nsrcs, srcs, ndsts, dsts, offset, Nelem);

  Nrem -= Nelem;
  if (Nrem == 0) return;
  offset += Nelem;

  // no unroll, by type. Should finish what's remaining.
  ReduceCopyMulti<FUNC, T, 1, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(w, nw, t, fn, preOpSrc0, postOp, nsrcs, srcs, ndsts, dsts, offset, Nrem);
}

#define BULK_LOAD_CP_VOLATILE 0
#define BULK_LOAD_CP_CA 1
#define BULK_LOAD_CP_G 2
#define BULK_LOAD_CP_CG 3
#define BULK_LOAD_CP_CV 4
#define BULK_LOAD_CP_CS 5
#define BULK_LOAD_CP_CE 6
#define BULK_LOAD_CP_BL 7
#define BULK_LOAD_CP_BA 8

template <int CACHE_POLICY>
inline __device__ void BulkFetch128(Pack128& v, const void* p) {
#ifndef LOAD_BYPASS
  switch(CACHE_POLICY) {
    case BULK_LOAD_CP_VOLATILE: v.i4 = __ppu_global_load_bulk_volatile_b32x4(p); break;
    case BULK_LOAD_CP_CA: v.i4 = __ppu_global_ldca_bulk_b32x4(p); break;
    case BULK_LOAD_CP_G: v.i4 = __ppu_global_ldg_bulk_b32x4(p); break;
    case BULK_LOAD_CP_CG: v.i4 = __ppu_global_ldcg_bulk_b32x4(p); break;
    case BULK_LOAD_CP_CV: v.i4 = __ppu_global_ldcv_bulk_b32x4(p); break;
    case BULK_LOAD_CP_CS: v.i4 = __ppu_global_ldcs_bulk_b32x4(p); break;
    case BULK_LOAD_CP_CE: v.i4 = __ppu_global_ldce_bulk_b32x4(p); break;
    case BULK_LOAD_CP_BL: v.i4 = __ppu_global_ldbl_bulk_b32x4(p); break;
    case BULK_LOAD_CP_BA: v.i4 = __ppu_global_ldba_bulk_b32x4(p); break;
  }
#else
  v.i4 = make_uint4(0U, 0U, 0U, 0U);
#endif
}

#define BULK_STORE_CP_VOLATILE 0
#define BULK_STORE_CP_WB 1
#define BULK_STORE_CP_CG 2
#define BULK_STORE_CP_WT 3
#define BULK_STORE_CP_CS 4
#define BULK_STORE_CP_BL 5
#define BULK_STORE_CP_BA 6

template<int CACHE_POLICY>
inline __device__ void BulkStore128(void* p, Pack128& v) {
#ifndef STORE_BYPASS
  switch(CACHE_POLICY) {
    case BULK_STORE_CP_VOLATILE: __ppu_global_store_bulk_volatile_b32x4(p, v.i4); break;
    case BULK_STORE_CP_WB: __ppu_global_stwb_bulk_b32x4(p, v.i4); break;
    case BULK_STORE_CP_CG: __ppu_global_stcg_bulk_b32x4(p, v.i4); break;
    case BULK_STORE_CP_WT: __ppu_global_stwt_bulk_b32x4(p, v.i4); break;
    case BULK_STORE_CP_CS: __ppu_global_stcs_bulk_b32x4(p, v.i4); break;
    case BULK_STORE_CP_BL: __ppu_global_stbl_bulk_b32x4(p, v.i4); break;
    case BULK_STORE_CP_BA: __ppu_global_stba_bulk_b32x4(p, v.i4); break;
  }
#endif
}

#define RMT_BULK_CP_C 0
#define RMT_BULK_CP_VOLATILE 1
#define RMT_BULK_CP_NCP 2
#define RMT_BULK_CP_BYPASS 3

#define RTMD_NORMAL 0
#define RTMD_PA 1
#define RTMD_AR 2

template <int CACHE_POLICY, int RTMD>
inline __device__ void RemoteBulkFetch128(Pack128& v, const void* p) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ < 150
  switch (CACHE_POLICY) {
    case RMT_BULK_CP_C: v.i4 = __ppu_remote_load_bulk_b32x4(p); break;
    case RMT_BULK_CP_VOLATILE: v.i4 = __ppu_remote_load_bulk_volatile_b32x4((volatile const void *)p); break;
    case RMT_BULK_CP_NCP: v.i4 = __ppu_remote_load_bulk_ncp_b32x4(p); break;
    case RMT_BULK_CP_BYPASS: v.i4 = __ppu_remote_load_bulk_bypass_b32x4(p); break;
  }
#else
  switch (CACHE_POLICY) {
    case RMT_BULK_CP_C: v.i4 = __ppu_remote_load_bulk_b32x4(p, RTMD); break;
    case RMT_BULK_CP_VOLATILE: v.i4 = __ppu_remote_load_bulk_volatile_b32x4((volatile const void *)p, RTMD); break;
    case RMT_BULK_CP_NCP: v.i4 = __ppu_remote_load_bulk_ncp_b32x4(p, RTMD); break;
    case RMT_BULK_CP_BYPASS: v.i4 = __ppu_remote_load_bulk_bypass_b32x4(p, RTMD); break;
  }
#endif
}

template <int CACHE_POLICY, int RTMD>
inline __device__ void RemoteBulkStore128(void* p, Pack128& v) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ < 150
  switch (CACHE_POLICY) {
    case RMT_BULK_CP_C: __ppu_remote_store_bulk_b32x4(p, v.i4); break;
    case RMT_BULK_CP_VOLATILE: __ppu_remote_store_bulk_volatile_b32x4((volatile void *)p, v.i4); break;
    case RMT_BULK_CP_NCP: __ppu_remote_store_bulk_ncp_b32x4(p, v.i4); break;
    case RMT_BULK_CP_BYPASS: __ppu_remote_store_bulk_bypass_b32x4(p, v.i4); break;
  }
#else
  switch (CACHE_POLICY) {
    case RMT_BULK_CP_C: __ppu_remote_store_bulk_b32x4(p, v.i4, RTMD); break;
    case RMT_BULK_CP_VOLATILE: __ppu_remote_store_bulk_volatile_b32x4((volatile void *)p, v.i4, RTMD); break;
    case RMT_BULK_CP_NCP: __ppu_remote_store_bulk_ncp_b32x4(p, v.i4, RTMD); break;
    case RMT_BULK_CP_BYPASS: __ppu_remote_store_bulk_bypass_b32x4(p, v.i4, RTMD); break;
  }
#endif
}

#define RMT_BULK_V4_ELEMS (512 / sizeof(T))
#define RMT_BULK_V4_ALIGN 128
#define RMT_BULK_V4_ALIGN_4B 4
#define REG_PHASE_NUM 2

template <typename T>
__device__ int ptrAlign4B(T* ptr) { return (uint64_t)ptr % RMT_BULK_V4_ALIGN_4B; }

template<class FUNC, typename T, int UNROLL, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS,
  int SRC, int DST, int RTMD, ncclConn_t SENDCONNTYPE, ncclConn_t RECVCONNTYPE>
__device__ __forceinline__ void ReduceCopy512BMultiBulk(int w, const int nw,
    FUNC fn, bool preOpSrc0, bool postOp, int nsrcs, const T** s, int ndsts, T** d, const int elemOffset, const int Npack) {
  const int inc = nw * UNROLL * RMT_BULK_V4_ELEMS;
  int offset = w * UNROLL * RMT_BULK_V4_ELEMS;
  int nMainLoops = Npack - nw;
  int nOptLoops = nMainLoops/(2*nw)*(2*nw);

  const T* srcs[MAXSRCS];
  for (int i=0; i<MAXSRCS; i++) srcs[i] = s[i] + elemOffset + offset;
  T* dsts[MAXDSTS];
  for (int i=0; i<MAXDSTS; i++) dsts[i] = d[i] + elemOffset + offset;

  // Start phase: preload current round of data
  Pack128 vals[REG_PHASE_NUM][MAXSRCS][UNROLL];
  if (w < Npack) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      if (SRC) {
        BulkFetch128<BULK_LOAD_CP_CG>(vals[0][0][u], srcs[0]+u*RMT_BULK_V4_ELEMS);
      } else {
        BulkFetch128<BULK_LOAD_CP_CV>(vals[0][0][u], srcs[0]+u*RMT_BULK_V4_ELEMS);
      }
      for (int i=1; i<MINSRCS; i++) {
        BulkFetch128<BULK_LOAD_CP_CV>(vals[0][i][u], srcs[i]+u*RMT_BULK_V4_ELEMS);
      }

      for (int i=MINSRCS; i<MAXSRCS; i++) {
        if (i<nsrcs) {
          BulkFetch128<BULK_LOAD_CP_CV>(vals[0][i][u], srcs[i]+u*RMT_BULK_V4_ELEMS);
        }
      }
    }
    if (preOpSrc0) {
      for (int u = 0; u < UNROLL; ++u) BULK_MULTI128<FUNC, T>().preOp(fn, vals[0][0][u]);
    }
  }
  // Main loop: next round of Load and current round of Load->Reduce->Store
  while (w<nOptLoops) {

    // Double registers main opt loop
    #pragma unroll
    for (int phase0=0; phase0<REG_PHASE_NUM; ++phase0) {
      int phase1 = (phase0+1) & 0x1;
      // next round of Load
      #pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        if (SRC) {
          BulkFetch128<BULK_LOAD_CP_CG>(vals[phase1][0][u], srcs[0]+inc+u*RMT_BULK_V4_ELEMS);
        } else {
          BulkFetch128<BULK_LOAD_CP_CV>(vals[phase1][0][u], srcs[0]+inc+u*RMT_BULK_V4_ELEMS);
        }

        for (int i=1; i<MINSRCS; i++) {
          BulkFetch128<BULK_LOAD_CP_CV>(vals[phase1][i][u], srcs[i]+inc+u*RMT_BULK_V4_ELEMS);
        }

        for (int i=MINSRCS; i<MAXSRCS; i++) {
          if (i<nsrcs) {
            BulkFetch128<BULK_LOAD_CP_CV>(vals[phase1][i][u], srcs[i]+inc+u*RMT_BULK_V4_ELEMS);
          }
        }
      }
      if (preOpSrc0) {
        for (int u = 0; u < UNROLL; ++u) BULK_MULTI128<FUNC, T>().preOp(fn, vals[phase1][0][u]);
      }

      // Reduce phase
      #pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        for (int i=1; i<MINSRCS; i++) {
          BULK_MULTI128<FUNC, T>()(fn, vals[phase0][0][u], vals[phase0][i][u]);
        }

        for (int i=MINSRCS; i<MAXSRCS; i++) {
          if (i<nsrcs) {
            BULK_MULTI128<FUNC, T>()(fn, vals[phase0][0][u], vals[phase0][i][u]);
          }
        }
      }

      if (postOp) {
        for (int u = 0; u < UNROLL; ++u) BULK_MULTI128<FUNC, T>().postOp(fn, vals[phase0][0][u]);
      }

      // Local Store and remote store
      #pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        if (DST == 1) {
          BulkStore128<BULK_STORE_CP_CG>(dsts[0]+u*RMT_BULK_V4_ELEMS, vals[phase0][0][u]);
        }

        for (int i = DST; i < MINDSTS; i++) {
          if (SENDCONNTYPE == CONN_ICN) {
            RemoteBulkStore128<RMT_BULK_CP_NCP, RTMD>(dsts[i]+u*RMT_BULK_V4_ELEMS, vals[phase0][0][u]);
          } else {
            BulkStore128<BULK_STORE_CP_WT>(dsts[i]+u*RMT_BULK_V4_ELEMS, vals[phase0][0][u]);
          }
        }


        for (int i=MINDSTS; i<MAXDSTS; i++) {
          if (i<ndsts) {
            if (SENDCONNTYPE == CONN_ICN) {
              RemoteBulkStore128<RMT_BULK_CP_NCP, RTMD>(dsts[i]+u*RMT_BULK_V4_ELEMS, vals[phase0][0][u]);
            } else {
              BulkStore128<BULK_STORE_CP_WT>(dsts[i]+u*RMT_BULK_V4_ELEMS, vals[phase0][0][u]);
            }
          }
        }
      }

      // calculate next of next round load
      for (int i=0; i<MAXSRCS; i++) srcs[i] += inc;
      for (int i=0; i<MAXDSTS; i++) dsts[i] += inc;
      w += nw;
    }
  }

  // End phase: last border loop we only do store to avoid redundant loads
  while  (w < Npack) {
    // Reduce phase
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      for (int i=1; i<MINSRCS; i++) {
        BULK_MULTI128<FUNC, T>()(fn, vals[0][0][u], vals[0][i][u]);
      }

      for (int i=MINSRCS; i<MAXSRCS; i++) {
        if (i<nsrcs) {
          BULK_MULTI128<FUNC, T>()(fn, vals[0][0][u], vals[0][i][u]);
        }
      }
    }

    if (postOp) {
      for (int u = 0; u < UNROLL; ++u) BULK_MULTI128<FUNC, T>().postOp(fn, vals[0][0][u]);
    }

    #pragma unroll
    for (int u=0; u<UNROLL; ++u) {
      if (DST == 1) {
        BulkStore128<BULK_STORE_CP_CG>(dsts[0]+u*RMT_BULK_V4_ELEMS, vals[0][0][u]);
      }

      #pragma unroll
      for (int i = DST; i < MINDSTS; i++) {
        if (SENDCONNTYPE == CONN_ICN) {
          RemoteBulkStore128<RMT_BULK_CP_NCP, RTMD>(dsts[i]+u*RMT_BULK_V4_ELEMS, vals[0][0][u]);
        } else {
          BulkStore128<BULK_STORE_CP_WT>(dsts[i]+u*RMT_BULK_V4_ELEMS, vals[0][0][u]);
        }
      }

      #pragma unroll
      for (int i=MINDSTS; i<MAXDSTS; i++) {
        if (i<ndsts) {
          if (SENDCONNTYPE == CONN_ICN) {
            RemoteBulkStore128<RMT_BULK_CP_NCP, RTMD>(dsts[i]+u*RMT_BULK_V4_ELEMS, vals[0][0][u]);
          } else {
            BulkStore128<BULK_STORE_CP_WT>(dsts[i]+u*RMT_BULK_V4_ELEMS, vals[0][0][u]);
          }
        }
      }
    }

    w += nw;
    if (w < Npack) {
      // calculate last round of load
      for (int i=0; i<MAXSRCS; i++) srcs[i] += inc;
      for (int i=0; i<MAXDSTS; i++) dsts[i] += inc;
      #pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        if (SRC) {
          BulkFetch128<BULK_LOAD_CP_CG>(vals[0][0][u], srcs[0]+u*RMT_BULK_V4_ELEMS);
        } else {
          BulkFetch128<BULK_LOAD_CP_CV>(vals[0][0][u], srcs[0]+u*RMT_BULK_V4_ELEMS);
        }
        for (int i=1; i<MINSRCS; i++) {
          BulkFetch128<BULK_LOAD_CP_CV>(vals[0][i][u], srcs[i]+u*RMT_BULK_V4_ELEMS);
        }

        for (int i=MINSRCS; i<MAXSRCS; i++) {
          if (i<nsrcs) {
            BulkFetch128<BULK_LOAD_CP_CV>(vals[0][i][u], srcs[i]+u*RMT_BULK_V4_ELEMS);
          }
        }
      }

      if (preOpSrc0) {
        for (int u = 0; u < UNROLL; ++u) BULK_MULTI128<FUNC, T>().preOp(fn, vals[0][0][u]);
      }
    }
  }
}

template<int UNROLL, class FUNC, typename T, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS,
  int SRC, int DST, int RTMD, ncclConn_t SENDCONNTYPE, ncclConn_t RECVCONNTYPE>
__device__ __forceinline__ void ReduceOrCopyMultiBulk(const int tid, const int nthreads,
    FUNC fn, bool preOpSrc0, bool postOp, int nsrcs, const T** srcs, int ndsts, T** dsts,
    int N) {
  int Nrem = N;
  if (Nrem <= 0) return;

  int w = tid / WARP_SIZE;       // Warp number
  int nw = nthreads / WARP_SIZE; // Number of warps
  int t = tid % WARP_SIZE;       // Thread (inside the warp)

  // Check that all is 4B aligned. If not don't use 512B load/stores.
  int align4B = 0;
  #pragma unroll
  for (int i=0; i<MINSRCS; i++) align4B |= ptrAlign4B(srcs[i]);
  for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) align4B |= ptrAlign4B(srcs[i]);
  #pragma unroll
  for (int i=0; i<MINDSTS; i++) align4B |= ptrAlign4B(dsts[i]);
  for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) align4B |= ptrAlign4B(dsts[i]);

  int offset = 0;
  if (align4B == 0) {
    // fast path: use 512B loads/stores to do the bulk of the work,
    // assuming the pointers we have are all 512-byte aligned.

    // main loop
    int Npack = Nrem / (RMT_BULK_V4_ELEMS*UNROLL); // round down
    int Nelem = Npack * (RMT_BULK_V4_ELEMS*UNROLL);

    if (Npack>0) {
      ReduceCopy512BMultiBulk<FUNC, T, UNROLL, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS,
        SRC, DST, RTMD, SENDCONNTYPE, RECVCONNTYPE>(w, nw, fn, preOpSrc0, postOp, nsrcs, srcs, ndsts, dsts, offset, Npack);
    }

    Nrem -= Nelem;
    if (Nrem == 0) return;
    offset += Nelem;

    // slightly less optimized for section when we don't have full unrolling
    Npack = Nrem / (RMT_BULK_V4_ELEMS);
    Nelem = Npack * RMT_BULK_V4_ELEMS;

    if (Npack>0) {
      ReduceCopy512BMultiBulk<FUNC, T, 1, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS,
        SRC, DST, RTMD, SENDCONNTYPE, RECVCONNTYPE>(w, nw, fn, preOpSrc0, postOp, nsrcs, srcs, ndsts, dsts, offset, Npack);
    }

    Nrem -= Nelem;
    if (Nrem == 0) return;
    offset += Nelem;
  }

  // unrolled, by-type (mostly for unaligned buffers)
  int Nelem = (Nrem / (UNROLL*PACKELEMS/2*WARP_SIZE)) * (UNROLL*PACKELEMS/2*WARP_SIZE); // round down

  ReduceCopyMulti<FUNC, T, UNROLL*PACKELEMS/2, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(w, nw, t, fn, preOpSrc0, postOp, nsrcs, srcs, ndsts, dsts, offset, Nelem);

  Nrem -= Nelem;
  if (Nrem == 0) return;
  offset += Nelem;

  // no unroll, by type. Should finish what's remaining.
  ReduceCopyMulti<FUNC, T, 1, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(w, nw, t, fn, preOpSrc0, postOp, nsrcs, srcs, ndsts, dsts, offset, Nrem);
}

template<typename T, int UNROLL, int NPEERS, class FUNC, int SRC>
__device__ __forceinline__ void OneToOneCopyMulti(const int w, const int nw, const int t, FUNC fn,
    int npeers, const T** s, T** d, const int elemOffset, const int Nelem) {
  if (Nelem == 0) return;

  const int inc = nw * UNROLL * WARP_SIZE;
  int offset = w * UNROLL * WARP_SIZE + t;

  const T* srcs[NPEERS];
  T* dsts[NPEERS];
  for (int i=0; i<NPEERS; i++) {
    srcs[i] = s[i]+elemOffset+offset;
    dsts[i] = d[i]+elemOffset+offset;
  }

  while (offset < Nelem) {
    T vals[NPEERS][UNROLL];
    // Load
    #pragma unroll
    for (int i=0; i<NPEERS; i++) {
      for (int u = 0; u < UNROLL; ++u) {
        if (i < npeers) {
          vals[i][u] = vFetch(srcs[i]+u*WARP_SIZE);
          if (SRC) vals[i][u] = FuncTraits<FUNC>().preOp(fn, vals[i][u]);
          vStore(dsts[i]+u*WARP_SIZE, vals[i][u]);
        }
      }
    }

    for (int i=0; i<NPEERS; i++) {
      srcs[i] += inc;
      dsts[i] += inc;
    }
    offset += inc;
  }
}

template<typename T, int UNROLL, int NPEERS, int SEND, class FUNC, int SRC, int RTMD, ncclConn_t CONNTYPE>
__device__ __forceinline__ void OneToOneCopy128bMultiBulk(int w, const int nw, FUNC fn,
    int npeers, const T** s, T** d, const int elemOffset, const int Npack) {
  if (Npack == 0) return;
  const int inc = nw * UNROLL * RMT_BULK_V4_ELEMS;
  int offset = w * UNROLL * RMT_BULK_V4_ELEMS;
  int nMainLoop = Npack - nw;

  const T* srcs[NPEERS];
  T* dsts[NPEERS];
  #pragma unroll
  for (int i=0; i<NPEERS; i++) {
    srcs[i] = s[i]+elemOffset+offset;
    dsts[i] = d[i]+elemOffset+offset;
  }

  // Start phase: preload current round of data
  Pack128 currVals[NPEERS][UNROLL];
  Pack128 nextVals[NPEERS][UNROLL];
  #pragma unroll
  for (int i=0; i<NPEERS; i++) {
    if (i < npeers) {
      for (int u = 0; u < UNROLL; ++u) {
        if (SRC) {
          BulkFetch128<BULK_LOAD_CP_CG>(currVals[i][u], srcs[i]+u*RMT_BULK_V4_ELEMS);
        } else {
          BulkFetch128<BULK_LOAD_CP_CV>(currVals[i][u], srcs[i]+u*RMT_BULK_V4_ELEMS);
        }
        if (SRC) BULK_MULTI128<FUNC, T>().preOp(fn, currVals[i][u]);
      }
    }
  }

  // Main loop: Load and Store
  while (w < nMainLoop) {

    if (SEND) {
      #pragma unroll
      for (int i=0; i<NPEERS; i++) {
        if (i < npeers) {
          for (int u = 0; u < UNROLL; ++u) BulkFetch128<BULK_LOAD_CP_CG>(nextVals[i][u], srcs[i]+inc+u*RMT_BULK_V4_ELEMS);
          for (int u = 0; u < UNROLL; ++u) {
            if (CONNTYPE == CONN_ICN) {
              RemoteBulkStore128<RMT_BULK_CP_NCP, RTMD>(dsts[i]+u*RMT_BULK_V4_ELEMS, currVals[i][u]);
            } else {
              BulkStore128<BULK_STORE_CP_WT>(dsts[i]+u*RMT_BULK_V4_ELEMS, currVals[i][u]);
            }
          }
        }
      }
    } else {
      #pragma unroll
      for (int i=0; i<NPEERS; i++) {
        if (i < npeers) {
          for (int u = 0; u < UNROLL; ++u) BulkFetch128<BULK_LOAD_CP_CV>(nextVals[i][u], srcs[i]+inc+u*RMT_BULK_V4_ELEMS);
          for (int u = 0; u < UNROLL; ++u) BulkStore128<BULK_STORE_CP_CG>(dsts[i]+u*RMT_BULK_V4_ELEMS, currVals[i][u]);
        }
      }
    }

    #pragma unroll
    for (int i=0; i<NPEERS; i++) {
      if (i < npeers) {
        for (int u = 0; u < UNROLL; ++u) {
          currVals[i][u] = nextVals[i][u];
          if (SRC) BULK_MULTI128<FUNC, T>().preOp(fn, currVals[i][u]);
        }
      }
    }

    #pragma unroll
    for (int i=0; i<NPEERS; i++) {
      if (i < npeers) {
        srcs[i] += inc;
        dsts[i] += inc;
      }
    }
    w += nw;
  }

  if (w < Npack) {
    // End phase: last border loop we only do store to avoid redundant loads
    if (SEND) {
      #pragma unroll
      for (int i=0; i<NPEERS; i++) {
        if (i < npeers) {
          for (int u = 0; u < UNROLL; ++u) {
            if (CONNTYPE == CONN_ICN) {
              RemoteBulkStore128<RMT_BULK_CP_NCP, RTMD>(dsts[i]+u*RMT_BULK_V4_ELEMS, currVals[i][u]);
            } else {
              BulkStore128<BULK_STORE_CP_WT>(dsts[i]+u*RMT_BULK_V4_ELEMS, currVals[i][u]);
            }
          }
        }
      }
    } else {
      #pragma unroll
      for (int i=0; i<NPEERS; i++) {
        if (i < npeers) {
          for (int u = 0; u < UNROLL; ++u) BulkStore128<BULK_STORE_CP_CG>(dsts[i]+u*RMT_BULK_V4_ELEMS, currVals[i][u]);
        }
      }
    }
  }
}

template<typename T, int UNROLL, int NPEERS, class FUNC, int SRC>
__device__ __forceinline__ void OneToOneCopy128bMulti(const int w, const int nw, const int t, FUNC fn,
    int npeers, const T** s, T** d, const int elemOffset, const int Npack) {
  if (Npack == 0) return;
  const int inc = nw * UNROLL * WARP_SIZE;
  int offset = w * UNROLL * WARP_SIZE + t;

  const Pack128* srcs[NPEERS];
  Pack128* dsts[NPEERS];
  for (int i=0; i<NPEERS; i++) {
    srcs[i] = ((const Pack128*)(s[i]+elemOffset))+offset;
    dsts[i] = ((Pack128*)(d[i]+elemOffset))+offset;
  }

  while (offset < Npack) {
    Pack128 vals[NPEERS][UNROLL];
    #pragma unroll
    for (int i=0; i<NPEERS; i++) {
      for (int u = 0; u < UNROLL; ++u) {
        if (i < npeers) {
          Fetch128(vals[i][u], srcs[i]+u*WARP_SIZE);
          if (SRC) MULTI128<FUNC, T>().preOp(fn, vals[i][u]);
          Store128(dsts[i]+u*WARP_SIZE, vals[i][u]);
        }
      }
    }

    for (int i=0; i<NPEERS; i++) {
      srcs[i] += inc;
      dsts[i] += inc;
    }
    offset += inc;
  }
}

template<int UNROLL, typename T, int NPEERS, class FUNC, int SRC>
__device__ __forceinline__ void ScatterGatherMulti(const int tid, const int nthreads, FUNC fn,
    const T** srcs, T** dsts, int* peerSizes, int npeers) {
  // peerSizes[0] should be the largest size
  while (npeers>0 && peerSizes[npeers-1] <= 0) npeers--;
  if (npeers <= 0) return;
  int Nmax = peerSizes[0];
  int Nmin = peerSizes[npeers-1];
  int Nrem = Nmin;
  if (Nrem <= 0) return;

  int w = tid / WARP_SIZE;       // Warp number
  int nw = nthreads / WARP_SIZE; // Number of warps
  int t = tid % WARP_SIZE;       // Thread (inside the warp)

  // Check that all is 16B aligned. If not don't use 16B load/stores.
  int align = 0;
  for (int i=0; i<NPEERS && i<npeers; i++) {
    align |= ptrAlign128(srcs[i]);
    align |= ptrAlign128(dsts[i]);
  }

  int offset = 0;
  if (align == 0) {
    // fast path: use 128b loads/stores to do the bulk of the work,
    // assuming the pointers we have are all 128-bit aligned.

    // main loop
    int Npack = (Nrem / (PACKELEMS*UNROLL*WARP_SIZE)) * (UNROLL*WARP_SIZE); // round down
    int Nelem = Npack * PACKELEMS;

    OneToOneCopy128bMulti<T, UNROLL, NPEERS, FUNC, SRC>(w, nw, t, fn, npeers, srcs, dsts, offset, Npack);

    Nrem -= Nelem;
    offset += Nelem;

    // slightly less optimized for section when we don't have full unrolling
    Npack = Nrem / PACKELEMS;
    Nelem = Npack * PACKELEMS;

    OneToOneCopy128bMulti<T, 1, NPEERS, FUNC, SRC>(w, nw, t, fn, npeers, srcs, dsts, offset, Npack);

    Nrem -= Nelem;
    offset += Nelem;
  }

  // unrolled, by-type (mostly for unaligned buffers)
  int Nelem = (Nrem / (UNROLL*PACKELEMS/2*WARP_SIZE)) * (UNROLL*PACKELEMS/2*WARP_SIZE); // round down

  OneToOneCopyMulti<T, UNROLL*PACKELEMS/2, NPEERS, FUNC, SRC>(w, nw, t, fn, npeers, srcs, dsts, offset, Nelem);

  Nrem -= Nelem;
  offset += Nelem;

  // no unroll, by type. Should finish what's remaining.
  OneToOneCopyMulti<T, 1, NPEERS, FUNC, SRC>(w, nw, t, fn, npeers, srcs, dsts, offset, Nrem);

  // deal with border elems due to last peer rank's inequality with others
  if (Nmax > Nmin) {
    offset += Nrem;
    Nrem = Nmax - Nmin;
    OneToOneCopyMulti<T, 1, NPEERS, FUNC, SRC>(w, nw, t, fn, npeers-1, srcs, dsts, offset, Nrem);
  }
}

template<int UNROLL, typename T, int NPEERS, int SEND, class FUNC, int SRC, int RTMD, ncclConn_t CONNTYPE>
__device__ __forceinline__ void ScatterGatherMultiBulk(const int tid, const int nthreads, FUNC fn,
    const T** srcs, T** dsts, int* peerSizes, int npeers) {
  // peerSizes[0] should be the largest size
  while (npeers>0 && peerSizes[npeers-1] <= 0) npeers--;
  if (npeers <= 0) return;
  int Nmax = peerSizes[0];
  int Nmin = peerSizes[npeers-1];
  int Nrem = Nmin;
  if (Nrem <= 0) return;

  int w = tid / WARP_SIZE;       // Warp number
  int nw = nthreads / WARP_SIZE; // Number of warps
  int t = tid % WARP_SIZE;       // Thread (inside the warp)

  // Check that all is 512B aligned. If not don't use 512B load/stores.
  int align4B = 0;
  for (int i=0; i<NPEERS && i<npeers; i++) {
    align4B |= ptrAlign4B(srcs[i]);
    align4B |= ptrAlign4B(dsts[i]);
  }

  int offset = 0;
  if (align4B == 0 && npeers == NPEERS) {
    // fast path: use 512B loads/stores to do the bulk of the work,
    // assuming the pointers we have are all 512-byte aligned.

    // main loop
    int Npack = Nrem / (nw*RMT_BULK_V4_ELEMS*UNROLL); // round down
    Npack *= nw;
    int Nelem = Npack * (RMT_BULK_V4_ELEMS*UNROLL);

    OneToOneCopy128bMultiBulk<T, UNROLL, NPEERS, SEND, FUNC, SRC, RTMD, CONNTYPE>(w, nw, fn, npeers, srcs, dsts, offset, Npack);

    Nrem -= Nelem;
    offset += Nelem;

    // slightly less optimized for section when we don't have full unrolling
    Npack = Nrem / (nw*RMT_BULK_V4_ELEMS);
    Npack *= nw;
    Nelem = Npack * RMT_BULK_V4_ELEMS;

    OneToOneCopy128bMultiBulk<T, 1, NPEERS, SEND, FUNC, SRC, RTMD, CONNTYPE>(w, nw, fn, npeers, srcs, dsts, offset, Npack);

    Nrem -= Nelem;
    offset += Nelem;
  }

  // unrolled, by-type (mostly for unaligned buffers)
  int Nelem = (Nrem / (UNROLL*PACKELEMS/2*WARP_SIZE)) * (UNROLL*PACKELEMS/2*WARP_SIZE); // round down

  OneToOneCopyMulti<T, UNROLL*PACKELEMS/2, NPEERS, FUNC, SRC>(w, nw, t, fn, npeers, srcs, dsts, offset, Nelem);

  Nrem -= Nelem;
  offset += Nelem;

  // no unroll, by type. Should finish what's remaining.
  OneToOneCopyMulti<T, 1, NPEERS, FUNC, SRC>(w, nw, t, fn, npeers, srcs, dsts, offset, Nrem);

  // deal with border elems due to last peer rank's inequality with others
  if (Nmax > Nmin) {
    offset += Nrem;
    Nrem = Nmax - Nmin;
    OneToOneCopyMulti<T, 1, NPEERS, FUNC, SRC>(w, nw, t, fn, npeers-1, srcs, dsts, offset, Nrem);
  }
}

#endif // COMMON_KERNEL_H_
