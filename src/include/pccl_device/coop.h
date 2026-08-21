/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_COOP_H_
#define _PCCL_DEVICE_COOP_H_
#include "utility.h"

#if PCCL_CHECK_HGGCCC
// pcclCoop[Foo]: PCCL's versions of CUDA's Cooperative Groups. They conform
// to just this subset of the CUDA API:
//   int Coop::thread_rank();
//   int Coop::size();
//   int Coop::num_threads(); // same as size()
//   void Coop::sync();

struct pcclCoopAny {
  struct Storage { alignas(alignof(void*)) char space[16]; };
  struct VTable {
    int(*thread_rank)(void const*);
    int(*size)(void const*);
    void(*sync)(void*);
  };

  template<typename Impl>
  __device__ static int thread_rank(void const* o) {
    return static_cast<Impl const*>(o)->thread_rank();
  }
  template<typename Impl>
  __device__ static int size(void const* o) {
    return static_cast<Impl const*>(o)->size();
  }
  template<typename Impl>
  __device__ static void sync(void* o) {
    static_cast<Impl*>(o)->sync();
  }

  template<typename Impl>
  __device__ static VTable const* get_vtable() {
    static_assert(sizeof(Impl) <= sizeof(Storage), "Incompatible coop type size");
    static_assert(alignof(Impl) <= alignof(Storage), "Incompatible coop type alignment");
    static constexpr VTable v = {
      &thread_rank<Impl>,
      &size<Impl>,
      &sync<Impl>
    };
    return &v;
  }

  Storage storage;
  VTable const* vtable;

  pcclCoopAny(pcclCoopAny const&) = default;
  pcclCoopAny(pcclCoopAny&&) = default;
  pcclCoopAny() = default;

  template<typename Impl>
  __device__ pcclCoopAny(Impl impl) {
    ::new (&this->storage) Impl(impl);
    this->vtable = get_vtable<Impl>();
  }

  __device__ int thread_rank() const { return vtable->thread_rank(&storage); }
  __device__ int size() const { return vtable->size(&storage); }
  __device__ int num_threads() const { return vtable->size(&storage); }
  __device__ void sync() { vtable->sync(&storage); }
};

template<int nThreadsPow2>
struct pcclCoopTile { // An aligned pow2 set of threads within the warp.
  static_assert(pccl::utility::isPow2(nThreadsPow2) && nThreadsPow2 <= 32, "Condition required");

  PCCL_DEVICE_INLINE int thread_rank() const {
    return pccl::utility::lane() % nThreadsPow2;
  }
  PCCL_DEVICE_INLINE constexpr int size() const { return nThreadsPow2; }
  PCCL_DEVICE_INLINE constexpr int num_threads() const { return nThreadsPow2; }

  PCCL_DEVICE_INLINE uint32_t laneMask() const {
    return (-1u>>(32-nThreadsPow2))<<(pccl::utility::lane() & -nThreadsPow2);
  }
  PCCL_DEVICE_INLINE void sync() {
    if (nThreadsPow2 > 1) __syncwarp(laneMask());
  }
};

typedef pcclCoopTile<1> pcclCoopThread;
typedef pcclCoopTile<32> pcclCoopWarp;

struct pcclCoopLanes { // Some lanes of this warp.
  uint32_t lmask;

  PCCL_DEVICE_INLINE constexpr pcclCoopLanes(uint32_t lmask=~0u): lmask(lmask) {}

  PCCL_DEVICE_INLINE int thread_rank() const {
    return __popc(lmask & pccl::utility::lanemask_lt());
  }
  PCCL_DEVICE_INLINE int size() const {
    return __popc(lmask);
  }
  PCCL_DEVICE_INLINE int num_threads() const {
    return __popc(lmask);
  }
  PCCL_DEVICE_INLINE void sync() {
    __syncwarp(lmask);
  }
};

// A set of consecutive warps that the user has also supplied with a unique
// id from [0..15]. It is an error for two different warp spans with the same
// id to be in a collective concurrently.
struct pcclCoopWarpSpan {
  uint32_t warp0:8, nWarps:8, id:8;

  PCCL_DEVICE_INLINE constexpr pcclCoopWarpSpan(int warp0, int nWarps, int id):
    warp0(warp0), nWarps(nWarps), id(id) {
  }

  PCCL_DEVICE_INLINE int thread_rank() const {
    return threadIdx.x - 32*warp0;
  }
  PCCL_DEVICE_INLINE int size() const {
    return 32*nWarps;
  }
  PCCL_DEVICE_INLINE int num_threads() const {
    return 32*nWarps;
  }

  PCCL_DEVICE_INLINE void sync() {
    //asm volatile("barrier.sync %0, %1;" :: "r"(1+id), "r"(32*nWarps) : "memory");
    __barrier_sync_count(1+id, 32*nWarps);
  }
};

struct pcclCoopCta {
  PCCL_DEVICE_INLINE int thread_rank() const { return threadIdx.x; }
  PCCL_DEVICE_INLINE int size() const { return blockDim.x; }
  PCCL_DEVICE_INLINE int num_threads() const { return blockDim.x; }
  PCCL_DEVICE_INLINE void sync() { __syncthreads(); }
};

template<int nThreadsPow2>
PCCL_DEVICE_INLINE uint32_t pcclCoopGetLaneMask(pcclCoopTile<nThreadsPow2> coop) {
  return coop.laneMask();
}
PCCL_DEVICE_INLINE uint32_t pcclCoopGetLaneMask(pcclCoopLanes coop) {
  return coop.lmask;
}
PCCL_DEVICE_INLINE uint32_t pcclCoopGetLaneMask(pcclCoopWarpSpan coop) {
  return -1u;
}
PCCL_DEVICE_INLINE uint32_t pcclCoopGetLaneMask(pcclCoopCta coop) {
  return -1u;
}

// pcclCoopIsThread:
// At compile time do we know the given coop is a single thread only.
template<int nThreads>
PCCL_DEVICE_INLINE constexpr bool pcclCoopIsThread(pcclCoopTile<nThreads>) {
  return nThreads == 1;
}
PCCL_DEVICE_INLINE constexpr bool pcclCoopIsThread(pcclCoopLanes) { return false; }
PCCL_DEVICE_INLINE constexpr bool pcclCoopIsThread(pcclCoopWarpSpan) { return false; }
PCCL_DEVICE_INLINE constexpr bool pcclCoopIsThread(pcclCoopCta) { return false; }

template<int nThreads>
PCCL_DEVICE_INLINE constexpr bool pcclCoopWithinWarp(pcclCoopTile<nThreads>) { return true; }
PCCL_DEVICE_INLINE constexpr bool pcclCoopWithinWarp(pcclCoopLanes) { return true; }
PCCL_DEVICE_INLINE constexpr bool pcclCoopWithinWarp(pcclCoopWarpSpan) { return false; }
PCCL_DEVICE_INLINE constexpr bool pcclCoopWithinWarp(pcclCoopCta) { return false; }

// Pick threads of our warp that are safe to use collectively.
PCCL_DEVICE_INLINE pcclCoopLanes pcclCoopCoalesced() {
  return pcclCoopLanes{__activemask()};
}

// Pick threads of our warp that are safe to use collectively given that this
// is a collective on the provided cooperative group.
template<typename Coop>
PCCL_DEVICE_INLINE pcclCoopTile<32> pcclCoopCoalesced(Coop) {
  return pcclCoopTile<32>();
}
PCCL_DEVICE_INLINE pcclCoopLanes pcclCoopCoalesced(pcclCoopLanes coop) {
  return coop;
}
template<int nThreads>
PCCL_DEVICE_INLINE pcclCoopTile<nThreads> pcclCoopCoalesced(pcclCoopTile<nThreads> coop) {
  return coop;
}

template<int nThreads, typename T>
PCCL_DEVICE_INLINE T pcclCoopBcast(pcclCoopTile<nThreads>, T value, int root, bool entrySync=true) {
  constexpr int n = (sizeof(T)+4-1)/4;
  union { uint32_t u[n]; T v; };
  v = value;
  #pragma unroll
  for (int i=0; i < n; i++) u[i] = __shfl_sync(-1u, u[i], root, nThreads);
  return v;
}
template<typename T>
PCCL_DEVICE_INLINE T pcclCoopBcast(pcclCoopLanes coop, T value, int root, bool entrySync=true) {
  uint32_t m = coop.lmask;
  uint32_t r = root == 0 ? __ffs(m)-1 : __fns(m, 0, 1+root);
  constexpr int n = (sizeof(T)+4-1)/4;
  union { uint32_t u[n]; T v; };
  v = value;
  #pragma unroll
  for (int i=0; i < n; i++) u[i] = __shfl_sync(m, u[i], r);
  return v;
}

PCCL_DEVICE_INLINE ulong2* pcclCoopBcast_WarpSpan_stash() {
#ifdef __HGGC_ARCH__
  __shared__ ulong2 stash[15];
  return stash;
#else
  return nullptr;
#endif
}

template<typename T>
PCCL_DEVICE_INLINE T pcclCoopBcast(pcclCoopWarpSpan coop, T value, int root, bool entrySync=true) {
  static_assert(sizeof(T) <= sizeof(pcclCoopBcast_WarpSpan_stash()[0]), "Required");
  if (entrySync) coop.sync();
  if (coop.thread_rank() == root) *(T*)&pcclCoopBcast_WarpSpan_stash()[coop.id] = value;
  coop.sync();
  return *(T*)&pcclCoopBcast_WarpSpan_stash()[coop.id];
}

PCCL_DEVICE_INLINE ulong2* pcclCoopBcast_Cta_stash() {
#ifdef __HGGC_ARCH__
  __shared__ ulong2 stash;
  return &stash;
#else
  return nullptr;
#endif
}

template<typename T>
PCCL_DEVICE_INLINE T pcclCoopBcast(pcclCoopCta coop, T value, int root, bool entrySync=true) {
  static_assert(sizeof(T) <= sizeof(*pcclCoopBcast_Cta_stash()), "Required");
  if (entrySync) coop.sync();
  if (coop.thread_rank() == root) *(T*)pcclCoopBcast_Cta_stash() = value;
  coop.sync();
  return *(T*)pcclCoopBcast_Cta_stash();
}
#endif // PCCL_CHECK_HGGCCC

#endif
