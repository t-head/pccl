/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef CUSTOM_DEVICE_UTILS_H_
#define CUSTOM_DEVICE_UTILS_H_
#pragma once

#include "customDeviceUtils.h"
#include <hggc_bf16.h>
#include <hggc_fp16.h>
#include <sys/types.h>

// Forward declarations — full definitions in customAllReduceKernels.h
struct extKernelDevParams;
struct extKernelHostParams;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Device-side utility functions shared across AllReduce and AllGather kernels
////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void stFlagRelease(uint32_t const& flag, volatile uint32_t* flagAddr) {
  asm volatile("ppu.st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(flagAddr));
}

static inline __device__ uint32_t ldFlagAcquire(volatile uint32_t* flagAddr) {
  uint32_t flag;
  asm volatile("ppu.ld.global.acquire.sys.b32 %0, [%1];" : "=r"(flag) : "l"(flagAddr));
  return flag;
}

__forceinline__ __device__ int getLaneId() {
  int lane_id;
  asm("ppu.mov.s32 %0, %laneid;" : "=r"(lane_id));
  return lane_id;
}

using PackedUintptr_t = union {
  int4 packed;
  uint4 upacked;
  uintptr_t unpacked[2];
};

// Type Converter that packs data format to 128 bits data type
using PackedFloat = union {
  int4 packed;
  uint4 upacked;
  float unpacked[4];
};

using PackedHalf = union {
  int4 packed;
  uint4 upacked;
  half2 unpacked[4];
};

using PackedBFloat16 = union {
  int4 packed;
  uint4 upacked;
  __ppu_bfloat162 unpacked[4];
};

template <typename T>
struct PackedOn16Bytes {
};

template <>
struct PackedOn16Bytes<float> {
  using Type = PackedFloat;
};

template <>
struct PackedOn16Bytes<half> {
  using Type = PackedHalf;
};

template <>
struct PackedOn16Bytes<__ppu_bfloat16> {
  using Type = PackedBFloat16;
};

// 8-byte packed types for LL protocol (each LL slot carries 8 bytes of data)
using PackedLLFloat = union {
  uint32_t u32[2];
  float unpacked[2];
  float elts[2];
  uint2 packed;
};

using PackedLLHalf = union {
  uint32_t u32[2];
  half2 unpacked[2];
  half elts[4];
  uint2 packed;
};

using PackedLLBFloat16 = union {
  uint32_t u32[2];
  __ppu_bfloat162 unpacked[2];
  __ppu_bfloat16 elts[4];
  uint2 packed;
};

using PackedLLInt8 = union {
  uint32_t u32[2];
  int8_t unpacked[8];
  int8_t elts[8];
  uint2 packed;
};

template <typename T>
struct PackedLLOn8Bytes {};

template <>
struct PackedLLOn8Bytes<float> {
  using Type = PackedLLFloat;
};

template <>
struct PackedLLOn8Bytes<half> {
  using Type = PackedLLHalf;
};

template <>
struct PackedLLOn8Bytes<__ppu_bfloat16> {
  using Type = PackedLLBFloat16;
};

template <>
struct PackedLLOn8Bytes<int8_t> {
  using Type = PackedLLInt8;
};

template <typename T>
inline __device__ void addLLPack(T& a, T& b) {
  a.unpacked[0] = a.unpacked[0] + b.unpacked[0];
  a.unpacked[1] = a.unpacked[1] + b.unpacked[1];
}

// add two 128b data
template <typename T>
inline __device__ int4 add128b(T& a, T& b) {
  T c;
  c.unpacked[0] = a.unpacked[0] + b.unpacked[0];
  c.unpacked[1] = a.unpacked[1] + b.unpacked[1];
  c.unpacked[2] = a.unpacked[2] + b.unpacked[2];
  c.unpacked[3] = a.unpacked[3] + b.unpacked[3];
  return c.packed;
}

__inline__ __device__ void multiGpuBarrier(volatile uint32_t** signals, const uint32_t flag, const size_t rank, const size_t worldSize, const int tidx, const int bidx) {
  // At the end of the function, we now that has least block 0 from all others GPUs have reached that point.
  if (tidx < worldSize) {
    // we can think of signals having the shape [world_size, world_size]
    // Dimension 0 is the "listening" dimension, dimension 1 is "emitting" dimension
    // TODO: how to understand listening and emitting ?
    if (bidx == 0) {
      stFlagRelease(flag, signals[tidx] + rank);
    }
    // Busy-wait until all ranks are ready.
    volatile uint32_t* peerBarrierD = signals[rank] + tidx;
    while (ldFlagAcquire(peerBarrierD) != flag) {}

    // this extra sync may introduce 4-5us overhead, comment it out before we find the necessity
#if 0
    if (bidx == 0) {
      stFlagRelease(flag, signals[tidx] + worldSize + rank);
    }
    // Busy-wait until all ranks are ready.
    peerBarrierD = signals[rank] + worldSize + tidx;
    while (ldFlagAcquire(peerBarrierD) != flag) {}
#endif
  }

  // Make sure we can move on...
  __syncthreads();
}

template<int NRANKS>
__inline__ __device__ void exchangePtrs(volatile void** ptrs, uintptr_t sendbuffOffset, uintptr_t recvbuffOffset,
                                       uintptr_t* sendbuffRmtPtrs, uintptr_t* recvbuffRmtPtrs,
                                       int32_t const myRank, int const tidx, int const bidx) {
  if (tidx < NRANKS && tidx != myRank) {
    // INFO: all block share the same sendbuffRmtPtr and recvbuffRmtPtr
    //       how can we share them by send only once?
    PackedUintptr_t peerRmtPtrs;
    uint32_t blockOffset = bidx * NRANKS;
    // peerRmtPtrs.unpacked[0] = sendbuffRmtPtrs[tidx] + sendbuffOffset;
    peerRmtPtrs.unpacked[1] = recvbuffRmtPtrs[tidx] + recvbuffOffset;
    void* peerAddr = (void*)((uintptr_t)ptrs[tidx] + (blockOffset + myRank) * 2 * sizeof(uintptr_t));
    __ppu_global_store_volatile_b32x4(peerAddr, peerRmtPtrs.upacked);
  }
  __syncthreads();
}

template<int NRANKS>
__inline__ __device__ void blockBarrier(volatile uint32_t** signals, uint32_t const flag, int32_t const myRank, int const tidx, int const bidx) {
  // After this function, the block of id == bidx of each GPU has reached the barrier
  if (tidx < NRANKS) {
    // we can think of signals having the shape [world_size, 2, num_blocks, world_size]
    // (+ an offset on dim 2 to account for flags used in multi_gpu_barrier)
    // Dimension 0 is the "listening" dimension, dimension 3 is "emitting" dimension

    // Block broadcast its flag (local_rank on emitting dimension) to all receivers
    uint32_t flagBlockOffset = NRANKS + bidx * NRANKS;
    stFlagRelease(flag, signals[tidx] + flagBlockOffset + myRank);
    // Blocks check that corresponding blocks on other GPUs have also set the flag
    volatile uint32_t* peerBarrierD = signals[myRank] + flagBlockOffset + tidx;
    while (ldFlagAcquire(peerBarrierD) != flag) {}

    // this extra sync may introduce 4-5us overhead, comment it out before we find the necessity
#if 0
    flagBlockOffset += (MAX_ALL_REDUCE_BLOCKS + 1) * worldSize;
    stFlagRelease(flag, signals[tidx] + flagBlockOffset + myRank);
    // Blocks check that corresponding blocks on other GPUs have also set the flag
    peerBarrierD = signals[myRank] + flagBlockOffset + tidx;
    while (ldFlagAcquire(peerBarrierD) != flag) {}
#endif
  }
  __syncthreads();
}

template <int NRANKS>
__inline__ __device__ uint32_t getBarrierFlag(volatile uint32_t** signals, int32_t const myRank, int const tidx, int const bidx) {
  uint32_t flagBlockOffset = NRANKS + bidx * NRANKS;
  volatile uint32_t* peerBarrierD = signals[myRank] + flagBlockOffset + myRank;
  uint32_t flag = *peerBarrierD;
  return flag;
}

// Attention: it is little tricky to do non-full mask bulk ld/st,
//            especially in simt mod, threads in one warp may diverging execute.
//            which may lead more non-full bulk operation.
//            to make bulk always execute with full mask, use warp level size/offset control.

inline __device__ void Load128(uint4& v, const uint4* p) {
  v = __ppu_global_load_volatile_b32x4((const void*)(p));
}

inline __device__ void Load64(uint2& v, const uint2* p) {
  v = __ppu_global_load_volatile_b32x2((const void*)(p));
}

inline __device__ void Store128(uint4& v, uint4* p) {
  __ppu_global_store_volatile_b32x4((volatile void*)(p), v);
}

// Device-side epoch size computation (mirrors host-side extKernelLLEpochSize but with __device__ qualifier)
template <int NRANKS>
inline __device__ int devExtKernelLLEpochSize() {
  constexpr int a = EXT_KERNEL_LL_MAX_THREADS * NRANKS * EXT_KERNEL_LL_PACK_SIZE;
  constexpr int b = EXT_KERNEL_LL_MAX_THREADS * EXT_KERNEL_LL_MAX_ELT_SIZE;  // simplifies: maxSlots(maxEltSize) * maxEltSize = maxThreads * maxEltSize
  return 2 * (a > b ? a : b);
}

// LL buffer addressing: get the LL buffer base for a given peer/block/epoch
template <int NRANKS>
inline __device__ uint4* extGetLLBuf(void** peerLLBufferPtrs, int peer, int block, uint32_t epoch) {
  int epochSize = devExtKernelLLEpochSize<NRANKS>();
  char* base = (char*)peerLLBufferPtrs[peer];
  base += block * 2 * epochSize;
  base += (epoch & 1) * epochSize;
  return (uint4*)base;
}

// LL store: write {data_lo, epoch, data_hi, epoch} to LL slot
inline __device__ void LLStore(uint4* p, uint32_t data_lo, uint32_t data_hi, uint32_t epoch) {
  uint4 v = make_uint4(data_lo, epoch, data_hi, epoch);
  Store128(v, p);
}

// LL load: volatile read of LL slot
inline __device__ void LLLoad(uint4& v, const uint4* p) {
  Load128(v, p);
}

template <typename T, int INPUT_ALIGN, int ELT_PER_PACK>
__forceinline__ __device__ void llLoadPack(typename PackedLLOn8Bytes<T>::Type& out, const T* src, int tidx, int validBound) {
  if (INPUT_ALIGN) {
    uint2 tmp;
    Load64(tmp, (const uint2*)(src + tidx * ELT_PER_PACK));
    out.u32[0] = tmp.x;
    out.u32[1] = tmp.y;
  } else {
    T* p = const_cast<T*>(src) + tidx * ELT_PER_PACK;
    #pragma unroll
    for (int e = 0; e < ELT_PER_PACK; e++) {
      if (tidx * ELT_PER_PACK + e < validBound)
        out.elts[e] = p[e];
      else
        out.elts[e] = (T)0;
    }
  }
}

inline __device__ void BulkLoad128(uint4& v, const uint4* p) {
  v = __ppu_global_load_bulk_volatile_b32x4((const void*)p);
}

inline __device__ void BulkStore128(uint4& v, uint4* p) {
  __ppu_global_store_bulk_volatile_b32x4((volatile void*)p, v);
}

inline __device__ void RemoteBulkLoad128(uint4& v, const uint4* p) {
  v = __ppu_remote_load_bulk_volatile_b32x4((volatile void*)p);
}

inline __device__ void RemoteBulkStore128(uint4& v, uint4* p) {
  __ppu_remote_store_bulk_volatile_b32x4((volatile void*)p, v);
}

template<typename T, int SRC_AT_LOCAL, int DST_AT_LOCAL, int ELT_WISE_TAIL = 0>
__inline__ __device__ void memcpyPackWise(volatile T* dstPtr, volatile T* srcPtr, int32_t Npack, int32_t exElts, int32_t warpGroupSize, int32_t wid, int32_t laneId) {
  static constexpr int PACKED_ELTS_PER_THREAD = PACKED_ELEMS_SIZE / sizeof(T);
  static constexpr int PACKED_ELTS_PER_WARP = PACKED_ELTS_PER_THREAD * WARP_SIZE;
  const int32_t warpEltsOffset = wid * PACKED_ELTS_PER_WARP;
  const int32_t groupElts = warpGroupSize * PACKED_ELTS_PER_WARP;

  volatile T* src = srcPtr + warpEltsOffset;
  volatile T* dst = dstPtr + warpEltsOffset;
  while (wid < Npack) {
    uint4 data;
    if (SRC_AT_LOCAL) BulkLoad128(data, (const uint4 *)(src));
    else RemoteBulkLoad128(data, (const uint4 *)(src));

    if (DST_AT_LOCAL) BulkStore128(data, (uint4 *)(dst));
    else RemoteBulkStore128(data, (uint4 *)(dst));

    src += groupElts;
    dst += groupElts;
    wid += warpGroupSize;
  }

  if (exElts > 0 && wid == Npack) {
    int32_t alignedEx = ELT_WISE_TAIL ? (exElts - (exElts % PACKED_ELTS_PER_THREAD)) : exElts;
    int32_t laneOffset = laneId * PACKED_ELTS_PER_THREAD;
    // Caller only enters the pack path with a 16B-aligned base, and laneOffset*sizeof(T)
    // is a 16B multiple, so each lane's 128-bit access stays aligned in the tail too.
    if (laneOffset < alignedEx) {
      uint4 data;
      Load128(data, (const uint4*)(src + laneOffset));
      Store128(data, (uint4*)(dst + laneOffset));
    }

    // Trailing elements that don't fill a full 16B pack.
    if (ELT_WISE_TAIL) {
      int32_t tailCount = exElts - alignedEx;
      if (laneId < tailCount) {
        dst[alignedEx + laneId] = src[alignedEx + laneId];
      }
    }
  }
}

template<typename T, int SRC_AT_LOCAL, int DST_AT_LOCAL>
__inline__ __device__ void memcpyElementWise(volatile T* dstPtr, volatile T* srcPtr, int32_t totalElts, int32_t warpGroupSize, int32_t wid, int32_t laneId) {
  // elment wise data load/store.
  // do not fuse with bulk load/store for now. handle the address later.
  int32_t offset = laneId + wid * WARP_SIZE;
  while(offset < totalElts) {
    dstPtr[offset] = srcPtr[offset];
    offset += warpGroupSize * WARP_SIZE;
  }
}


template<typename T, int NRANKS>
__forceinline__ __device__ void broadcastToAllPeers(volatile T* dstPtrs[NRANKS], volatile T* src, int32_t Npack, int32_t exElts, int32_t warpGroupSize, int32_t wid, int32_t laneId) {
  static constexpr int PACKED_ELTS_PER_THREAD = PACKED_ELEMS_SIZE / sizeof(T);
  static constexpr int PACKED_ELTS_PER_WARP = PACKED_ELTS_PER_THREAD * WARP_SIZE;
  const int32_t groupElts = warpGroupSize * PACKED_ELTS_PER_WARP;
  int32_t offset = wid * PACKED_ELTS_PER_WARP;

  while (wid < Npack) {
    uint4 data;
    BulkLoad128(data, (const uint4*)(src + offset));

    #pragma unroll
    for (int r = 0; r < NRANKS; r++) {
      if (r == 0)
        BulkStore128(data, (uint4*)(dstPtrs[0] + offset));
      else
        RemoteBulkStore128(data, (uint4*)(dstPtrs[r] + offset));
    }

    offset += groupElts;
    wid += warpGroupSize;
  }

  if (exElts > 0 && wid == Npack) {
    int32_t laneOffset = laneId * PACKED_ELTS_PER_THREAD;
    if (laneOffset < exElts) {
      uint4 data;
      Load128(data, (const uint4*)(src + offset + laneOffset));

      #pragma unroll
      for (int r = 0; r < NRANKS; r++) {
        if (r == 0)
          Store128(data, (uint4*)(dstPtrs[0] + offset + laneOffset));
        else
          Store128(data, (uint4*)(dstPtrs[r] + offset + laneOffset));
      }
    }
  }
}

template<typename T, int NRANKS>
__forceinline__ __device__ void broadcastToAllPeersElemWise(volatile T* dstPtrs[NRANKS], volatile T* src, int32_t totalElts, int32_t warpGroupSize, int32_t wid, int32_t laneId) {
  int32_t offset = laneId + wid * WARP_SIZE;
  while (offset < totalElts) {
    T val = src[offset];
    #pragma unroll
    for (int r = 0; r < NRANKS; r++) {
      dstPtrs[r][offset] = val;
    }
    offset += warpGroupSize * WARP_SIZE;
  }
}

// Keep a bounded window of remote bulk requests in flight. The PPU scheduling boundary is
// required: without it, the backend sinks each load next to its dependent add.
// This is a compiler scheduling boundary, not a memory fence or warp barrier.
template<typename T, int NRANKS>
__forceinline__ __device__ uint4 reduceBulkWithPrefetch(volatile T* srcs[NRANKS], int32_t offset) {
  using PackedStruct = typename PackedOn16Bytes<T>::Type;
  constexpr int window = NRANKS > 8 ? 8 : NRANKS - 1;
  PackedStruct sum;
  PackedStruct pending[window];
  BulkLoad128(sum.upacked, (const uint4*)(srcs[0] + offset));
  #pragma unroll
  for (int slot = 0; slot < window; ++slot) {
    RemoteBulkLoad128(pending[slot].upacked, (const uint4*)(srcs[slot + 1] + offset));
  }
  asm volatile("ppu.sched.bound;" ::: "memory");

  #pragma unroll
  for (int rank = 1; rank < NRANKS; ++rank) {
    const int slot = (rank - 1) % window;
    // Preserve rank order, including for fp16/bf16, and consume before reuse.
    sum.packed = add128b(sum, pending[slot]);
    if (rank + window < NRANKS) {
      asm volatile("ppu.sched.bound;" ::: "memory");
      RemoteBulkLoad128(pending[slot].upacked, (const uint4*)(srcs[rank + window] + offset));
      // The next add consumes an older slot, not the request just issued.
      asm volatile("ppu.sched.bound;" ::: "memory");
    }
  }
  // Returning the bits also avoids copying the bf16 packed union.
  return sum.upacked;
}

template<typename T, int NRANKS, int DSTS, int ELT_WISE_TAIL = 0>
__inline__ __device__ void reduceFromMultiSource(volatile T* srcs[NRANKS], volatile T** dsts, int32_t Npack, int32_t exElts, int32_t warpGroupSize, int32_t wid, int32_t laneId) {
  static constexpr int PACKED_ELTS_PER_THREAD = PACKED_ELEMS_SIZE / sizeof(T);
  static constexpr int PACKED_ELTS_PER_WARP = PACKED_ELTS_PER_THREAD * WARP_SIZE;
  using PackedStruct = typename PackedOn16Bytes<T>::Type;

  const int32_t eltsPerGroup = warpGroupSize * PACKED_ELTS_PER_WARP;
  int32_t offset = wid * PACKED_ELTS_PER_WARP;

  while(wid < Npack) {
    PackedStruct sum;
    if (NRANKS > 8) {
      sum.upacked = reduceBulkWithPrefetch<T, NRANKS>(srcs, offset);
    } else {
      PackedStruct vals[NRANKS];
      BulkLoad128(vals[0].upacked, (const uint4*)(srcs[0] + offset));

      #pragma unroll
      for (int ii = 1; ii < NRANKS; ++ii) {
        RemoteBulkLoad128(vals[ii].upacked, (const uint4*)(srcs[ii] + offset));
      }
      // Sum the values from the different ranks.
      #pragma unroll
      for (int ii = 1; ii < NRANKS; ++ii) {
        vals[0].packed = add128b(vals[0], vals[ii]);
      }
      sum.upacked = vals[0].upacked;
    }
    BulkStore128(sum.upacked, (uint4*)(dsts[0] + offset));
    if (DSTS > 1) {
      #pragma unroll
      for (int ii = 1; ii < NRANKS; ++ii) {
        RemoteBulkStore128(sum.upacked, (uint4*)(dsts[ii] + offset));
      }
    }

    offset += eltsPerGroup;
    wid += warpGroupSize;
  }

  if (exElts > 0 && wid == Npack) {
    int32_t baseOffset = offset;
    int32_t alignedEx = ELT_WISE_TAIL ? (exElts - (exElts % PACKED_ELTS_PER_THREAD)) : exElts;
    int32_t laneOffset = laneId * PACKED_ELTS_PER_THREAD;
    // srcs/dsts are 512B-aligned and laneOffset*sizeof(T) is a 16B multiple,
    // so the pack tail always stays 128-bit aligned.
    if (laneOffset < alignedEx) {
      PackedStruct vals[NRANKS];
      int32_t elemOffset = baseOffset + laneOffset;
      Load128(vals[0].upacked, (const uint4*)(srcs[0] + elemOffset));

      #pragma unroll
      for (int ii = 1; ii < NRANKS; ++ii) {
        Load128(vals[ii].upacked, (const uint4*)(srcs[ii] + elemOffset));
      }
      #pragma unroll
      for (int ii = 1; ii < NRANKS; ++ii) {
        vals[0].packed = add128b(vals[0], vals[ii]);
      }
      Store128(vals[0].upacked, (uint4*)(dsts[0] + elemOffset));
      if (DSTS > 1) {
        #pragma unroll
        for (int ii = 1; ii < NRANKS; ++ii) {
          Store128(vals[0].upacked, (uint4*)(dsts[ii] + elemOffset));
        }
      }
    }

    // Trailing elements that don't fill a full 16B pack.
    if (ELT_WISE_TAIL) {
      int32_t tailCount = exElts - alignedEx;
      if (laneId < tailCount) {
        int32_t tailOffset = baseOffset + alignedEx + laneId;
        T sum = (T)srcs[0][tailOffset];
        #pragma unroll
        for (int ii = 1; ii < NRANKS; ++ii) {
          T val = (T)srcs[ii][tailOffset];
          sum = sum + val;
        }
        dsts[0][tailOffset] = sum;
        if (DSTS > 1) {
          #pragma unroll
          for (int ii = 1; ii < NRANKS; ++ii) {
            dsts[ii][tailOffset] = sum;
          }
        }
      }
    }
  }
}

template<typename T, int NRANKS, int DSTS>
__inline__ __device__ void reduceFromMultiSourceElemWise(volatile T* srcs[NRANKS], volatile T** dsts, int32_t totalElts, int32_t warpGroupSize, int32_t wid, int32_t laneId) {
  // elment wise data load/store.
  // do not fuse with bulk load/store for now. handle the address later.
  int32_t offset = laneId + wid * WARP_SIZE;
  while(offset < totalElts) {
    T vals[NRANKS];
    #pragma unroll
    for (int ii = 0; ii < NRANKS; ++ii) {
      vals[ii] = srcs[ii][offset];
    }
    #pragma unroll
    for (int ii = 1; ii < NRANKS; ++ii) {
      vals[0] += vals[ii];
    }
    dsts[0][offset] = vals[0];

    if (DSTS > 1) {
      #pragma unroll
      for (int ii = 1; ii < NRANKS; ++ii) {
        dsts[ii][offset] = vals[0];
      }
    }

    offset += warpGroupSize * WARP_SIZE;
  }
}

#endif  // CUSTOM_DEVICE_UTILS_H_
