/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "symmetric.h"
#include "devcomm.h"
#include "symmetric/all_reduce.cuh"
#include "symmetric/kernel.cuh"
#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
#include "symmetric/ce_kernel_oki.cuh"
#include "symmetric/p2p_dma_kernel.cuh"
#endif

#ifdef DEBUG
#define PRINTF(format, ...) printf("" format "\n", ##__VA_ARGS__)

#else
#define PRINTF(format, ...) do {} while(0)
#endif

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

using PackedInt64_t = union {
  int4 packed;
  uint4 upacked;
  int64_t i64[2];
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

// pcclAvg support: every source value is scaled by 1/NRANKS before being summed.
// Pre-scaling rather than dividing the sum keeps a half accumulator from overflowing,
// and since CE reduce only runs on NRANKS in {2,4,8} the factor is a power of two, so
// the scaling itself is exact.
template <typename T, int NRANKS>
struct AvgPreScale;

template <int NRANKS>
struct AvgPreScale<float, NRANKS> {
  static inline __device__ void apply(PackedFloat& a) {
    constexpr float s = 1.0f / NRANKS;
    #pragma unroll
    for (int i = 0; i < 4; ++i) a.unpacked[i] = a.unpacked[i] * s;
  }
  static inline __device__ void applyElt(float& v) {
    v = v * (1.0f / NRANKS);
  }
};

template <int NRANKS>
struct AvgPreScale<half, NRANKS> {
  static inline __device__ void apply(PackedHalf& a) {
    half2 s;
    s.x = __float2half(1.0f / NRANKS);
    s.y = s.x;
    #pragma unroll
    for (int i = 0; i < 4; ++i) a.unpacked[i] = __hmul2(a.unpacked[i], s);
  }
  static inline __device__ void applyElt(half& v) {
    v = __hmul(v, __float2half(1.0f / NRANKS));
  }
};

template <int NRANKS>
struct AvgPreScale<__ppu_bfloat16, NRANKS> {
  static inline __device__ void apply(PackedBFloat16& a) {
    __ppu_bfloat162 s;
    s.x = __float2bfloat16(1.0f / NRANKS);
    s.y = s.x;
    #pragma unroll
    for (int i = 0; i < 4; ++i) a.unpacked[i] = __hmul2(a.unpacked[i], s);
  }
  static inline __device__ void applyElt(__ppu_bfloat16& v) {
    v = __hmul(v, __float2bfloat16(1.0f / NRANKS));
  }
};

__forceinline__ __device__ int getLaneId() {
  int lane_id;
  asm("ppu.mov.s32 %0, %laneid;" : "=r"(lane_id));
  return lane_id;
}

__forceinline__ __device__ void sleepClock(uint32_t count) {
  asm volatile("ppu.nanosleep.u32 %0;" :: "r"(count));
}

__global__ void gpuGEQ(volatile uint64_t* flagPtr, uint64_t targetValue) {
  uint64_t start = clock64();
  while(*flagPtr < targetValue) {
    uint64_t end = clock64();
    if (end - start > 100000000L) {
      printf("timeout here: wait :%lx, is %lx\n", targetValue, *flagPtr);
      start = end;
      break;
    }
  }
}

__global__ void gpuBarrier(ncclSymDevArgs const args) {
  ncclSymPrims prim(args.comm, 1, ncclSymPrims_UseBarrier);
  prim.barrierArrive(ncclCoopCta(), /*release=*/true);
  prim.barrierWait(ncclCoopCta(), /*acquire=*/false);
}

#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
__global__ void gpuBarrierWithTrigger(ncclSymDevArgs const args, uint64_t* addr, uint64_t value) {
  ncclSymPrims prim(args.comm, 1, ncclSymPrims_UseBarrier);
  prim.barrierArrive(ncclCoopCta(), /*release=*/true);
  prim.barrierWait(ncclCoopCta(), /*acquire=*/false);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    if (addr != NULL) {
      __ppu_global_store_volatile_b64((volatile void*)addr, value);
    }
  }
}
#endif

__device__ void fastDivModOfPow2(int32_t dividend, uint32_t divisor, int32_t* quotient, int32_t* remainder) {
  if ((divisor & (divisor - 1)) == 0) {
    int log2 = __ffs(divisor) - 1;
    *quotient = dividend >> log2;
    *remainder = dividend & (divisor - 1);
  } else {
    *quotient = dividend / (int32_t)divisor;
    *remainder = dividend % (int32_t)divisor;
  }
}

template<typename T>
__device__ T fastRoundDown(T value, uint32_t multiple) {
  if ((multiple & (multiple - 1)) == 0) {
    int log2 = __ffs(multiple) - 1;
    return ((value >> log2) << log2);
  }
  return (value / (T)multiple) * (T)multiple;
}

inline __device__ void BulkStore128(uint4& v, uint4* p) {
  __ppu_global_store_bulk_volatile_b32x4((volatile void*)p, v);
}

inline __device__ void BulkLoad128(uint4& v, const uint4* p) {
  v = __ppu_global_load_bulk_volatile_b32x4((const void*)p);
}

inline __device__ void Store128(uint4& v, uint4* p) {
  __ppu_global_store_volatile_b32x4((volatile void*)(p), v);
}

inline __device__ void Load128(uint4& v, const uint4* p) {
  v = __ppu_global_load_volatile_b32x4((const void*)(p));
}

__device__ void grid_sync(
    volatile uint32_t* gridSyncCounter,         // Atomic counter for blocks reaching sync in current generation
    volatile uint32_t* gridSyncGeneration,   // Tracks different sync rounds (generations)
    uint32_t blockNum) {
  uint32_t current_generation_value = *gridSyncGeneration; // Read current generation

  uint32_t current_count = atomicAdd((uint32_t*)gridSyncCounter, 1);
  if (current_count == blockNum - 1) {
    *gridSyncCounter = 0;
    atomicAdd((uint32_t*)gridSyncGeneration, 1);
  } else {
#ifdef DEBUG_KERN_TIMEOUT
    uint64_t start = clock();
    while (*gridSyncGeneration == current_generation_value) {
      uint64_t end = clock();
      if (end - start > 100000000L) {
        printf("grid_sync timeout: %u vs %u, currentCount:%u\n", *gridSyncGeneration, current_generation_value, *gridSyncCounter);
        break;
      }
    }
#else
    while (*gridSyncGeneration == current_generation_value) {}
#endif
  }
}

// TODO: refact the reduce like symmetric implement.
template<typename T, int NRANKS, bool AVG, int UNROLL=2>
__inline__ __device__ void reduceFromMultiSource(volatile T* srcs[NRANKS], volatile T* dst, int32_t Packets512B, int32_t Packets16B, int32_t PacketsElt, int32_t warpGroupSize, int32_t wid, int32_t laneId, int myRank) {
  static constexpr int PACKED_ELTS_PER_THREAD = 16 / sizeof(T);
  static constexpr int PACKED_ELTS_PER_WARP = PACKED_ELTS_PER_THREAD * WARP_SIZE;
  using PackedStruct = typename PackedOn16Bytes<T>::Type;
  using AvgScale = AvgPreScale<T, NRANKS>;

  const int32_t eltsPerGroup = warpGroupSize * PACKED_ELTS_PER_WARP;
  int32_t offset = wid * PACKED_ELTS_PER_WARP;
  PackedStruct vals[UNROLL][NRANKS];

  uint32_t unrollSize = warpGroupSize * UNROLL;
  int32_t unrollPacks512B = fastRoundDown(Packets512B, unrollSize);

  while(wid < unrollPacks512B) {
    #pragma unroll UNROLL
    for (int u = 0; u < UNROLL; u++) {
      BulkLoad128(vals[u][0].upacked, (const uint4*)(srcs[0] + offset + u * eltsPerGroup));

      #pragma unroll
      for (int ii = 1; ii < NRANKS; ++ii) {
        BulkLoad128(vals[u][ii].upacked, (const uint4*)(srcs[ii] + offset + u * eltsPerGroup));
      }
    }
    #pragma unroll UNROLL
    for (int u = 0; u < UNROLL; u++) {
      if (AVG) {
        #pragma unroll
        for (int ii = 0; ii < NRANKS; ++ii) AvgScale::apply(vals[u][ii]);
      }
      // Sum the values from the different ranks.
      #pragma unroll
      for (int ii = 1; ii < NRANKS; ++ii) {
        vals[u][0].packed = add128b(vals[u][0], vals[u][ii]);
      }
      BulkStore128(vals[u][0].upacked, (uint4*)(dst + offset + u * eltsPerGroup));
    }

    offset += UNROLL * eltsPerGroup;
    wid += UNROLL * warpGroupSize;
  }

  while(wid < Packets512B) {
    BulkLoad128(vals[0][0].upacked, (const uint4*)(srcs[0] + offset));
    // PRINTF("rank:%d <%d %d>, 0 %x %x %x %x\n", myRank, blockIdx.x, threadIdx.x, vals[0][0].upacked.x, vals[0][0].upacked.y, vals[0][0].upacked.z, vals[0][0].upacked.w);

    #pragma unroll
    for (int ii = 1; ii < NRANKS; ++ii) {
      BulkLoad128(vals[0][ii].upacked, (const uint4*)(srcs[ii] + offset));
      // PRINTF("rank:%d <%d %d>, %d %x %x %x %x\n", myRank, blockIdx.x, threadIdx.x, ii, vals[0][ii].upacked.x, vals[0][ii].upacked.y, vals[0][ii].upacked.z, vals[0][ii].upacked.w);
    }
    // Sum the values from the different ranks.
    if (AVG) {
      #pragma unroll
      for (int ii = 0; ii < NRANKS; ++ii) AvgScale::apply(vals[0][ii]);
    }
    #pragma unroll
    for (int ii = 1; ii < NRANKS; ++ii) {
      vals[0][0].packed = add128b(vals[0][0], vals[0][ii]);
    }
    BulkStore128(vals[0][0].upacked, (uint4*)(dst + offset));

    offset += eltsPerGroup;
    wid += warpGroupSize;
  }

  if ((Packets16B > 0 || PacketsElt > 0) && wid == Packets512B) {
    offset += laneId * PACKED_ELTS_PER_THREAD;
    if (laneId < Packets16B) {
      Load128(vals[0][0].upacked, (const uint4*)(srcs[0] + offset));

      #pragma unroll
      for (int ii = 1; ii < NRANKS; ++ii) {
        Load128(vals[0][ii].upacked, (const uint4*)(srcs[ii] + offset));
      }
      // Sum the values from the different ranks.
      if (AVG) {
        #pragma unroll
        for (int ii = 0; ii < NRANKS; ++ii) AvgScale::apply(vals[0][ii]);
      }
      #pragma unroll
      for (int ii = 1; ii < NRANKS; ++ii) {
        vals[0][0].packed = add128b(vals[0][0], vals[0][ii]);
      }
      Store128(vals[0][0].upacked, (uint4*)(dst + offset));
    }

    if (PacketsElt > 0 && laneId == Packets16B) {
      for (int elt = 0; elt < PacketsElt; ++elt) {
        T vals[NRANKS];
        #pragma unroll
        for (int ii = 0; ii < NRANKS; ++ii) vals[ii] = srcs[ii][offset + elt];

        if (AVG) {
          #pragma unroll
          for (int ii = 0; ii < NRANKS; ++ii) AvgScale::applyElt(vals[ii]);
        }
        #pragma unroll
        for (int ii = 1; ii < NRANKS; ++ii) vals[0] += vals[ii];

        dst[offset + elt] = vals[0];
      }
    }
  }
}

#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
// Consume station chunks and publish tails for the cmd blocks.
template<typename T, int NRANKS, bool AVG>
__device__ void ceLocalReduce(const struct pcclCeOkiReduceArgs& args) {
  T* src = (T*)args.myStaBuff;
  T* dst = (T*)args.myRecvBuff;
  ssize_t eltsTotal = args.myElts;
  uint32_t myRank = args.myRank;
  uint32_t nDies = args.nDies;
  uint64_t* tailBase = args.tailOnRank0;
  uint64_t* headBase = args.headOnRank0;
  uint32_t iters = args.loopCnt;
  uint32_t stride4G = args.stride4G;
  void* gridSyncPtr = args.gridSyncPtr;
  int64_t baseStep = args.reduceStepBase;
  constexpr uint32_t blockNum = pcclCeReduceMaxBlocks;

  static constexpr int PACKED_ELTS_PER_THREAD = 16 / sizeof(T);
  static constexpr int PACKED_ELTS_PER_WARP = PACKED_ELTS_PER_THREAD * WARP_SIZE;
  constexpr uint32_t splitElts = pcclCeReduceChunkSize / sizeof(T);

  const uint32_t nWarps = blockDim.x / WARP_SIZE;
  const uint32_t wid = __ppu_read_firstlane(threadIdx.x / WARP_SIZE);
  const uint32_t laneId = getLaneId();

  volatile T* baseSrcs[NRANKS];
  #pragma unroll
  for (int i = 0; i < NRANKS; i++) {
    baseSrcs[i] = src + (i * pcclSymMidBufferPerRank / sizeof(T));
  }

  int64_t remainElts = eltsTotal;

  uint64_t* tailPtr = tailBase;
  uint64_t* headPtr = headBase;
  if (threadIdx.x < NRANKS) {
    // two consequent u64 represent heads/tails for die0 and die1. only support max two dies for now.
    headPtr = (uint64_t*)add4G((uint8_t*)headPtr, threadIdx.x * stride4G);
    tailPtr = (uint64_t*)add4G((uint8_t*)tailPtr, threadIdx.x * stride4G);
  }

  for (int iter = 0; iter < iters; ++iter) { // can't combine baseStep into for loop calculation, as C2CDMA always use sect0 first for now.
    int64_t currentStep = baseStep + iter;
    uint64_t barrierFlag = baseStep + iter + 1;
    int32_t chunkElts = 0;
    if (remainElts > splitElts) chunkElts = splitElts;
    else chunkElts = remainElts;
    remainElts -= splitElts;


    int32_t blockOffset;
    int32_t currentBlockElts;
    int32_t extElts;
    fastDivModOfPow2(chunkElts, blockNum, &blockOffset, &extElts);
    // Keep block strides 16-byte aligned for packed loads and stores, including 2-byte types.
    blockOffset = fastRoundDown(blockOffset, PACKED_ELTS_PER_THREAD);
    extElts = chunkElts - blockOffset * blockNum;
    currentBlockElts = blockOffset;
    if (blockIdx.x == blockNum - 1) currentBlockElts += extElts;

    volatile T* srcs[NRANKS];
    volatile T* currentDst;
    uint32_t sect = iter % pcclCeReduceMaxSect;
    #pragma unroll
    for (int i = 0; i < NRANKS; i++) {
      srcs[i] = baseSrcs[i] + sect * splitElts + blockIdx.x * blockOffset;
    }
    currentDst = dst + blockIdx.x * blockOffset;

    int32_t Packets512B = currentBlockElts / PACKED_ELTS_PER_WARP;
    int32_t Packets16B = (currentBlockElts - Packets512B * PACKED_ELTS_PER_WARP) / PACKED_ELTS_PER_THREAD;
    int32_t PacketsElt = currentBlockElts % PACKED_ELTS_PER_THREAD;

    if (threadIdx.x < NRANKS) {
      PackedInt64_t head;
      Load128(head.upacked, (uint4*)headPtr);
#ifdef DEBUG_KERN_TIMEOUT
      uint64_t start = clock64();
      while(head.i64[0] < barrierFlag || (nDies > 1 && head.i64[1] < barrierFlag)) {
        // limit the remote flag load freq: 1 us
        sleepClock(800L);
        Load128(head.upacked, (uint4*)headPtr);
        uint64_t end = clock64();
        if (end - start > 100000000L) {
          printf("reduceWaitHead, rank:%d, <%d, %d>, timeout here: barrierFlag:%p:%lx, 0:%lx, 1:%lx\n", myRank, blockIdx.x, threadIdx.x, headPtr, barrierFlag, head.i64[0], head.i64[1]);
          break;
        }
      }
#else
      while(head.i64[0] < barrierFlag || (nDies > 1 && head.i64[1] < barrierFlag)) {
        // limit the remote flag load freq: 1 us
        sleepClock(800L);
        Load128(head.upacked, (uint4*)headPtr);
      }
#endif
    }
    __syncthreads();

    if (chunkElts > 0) {
      reduceFromMultiSource<T, NRANKS, AVG>(srcs, currentDst, Packets512B, Packets16B, PacketsElt, nWarps, wid, laneId, myRank);
    }

    __syncthreads();
    if (threadIdx.x == 0) {
      struct ncclCeGridSyncMem* gridSync = (struct ncclCeGridSyncMem*)gridSyncPtr;
      grid_sync(&gridSync->gridSyncCounter, &gridSync->gridSyncGeneration, blockNum);
    }
    __syncthreads();

    if (threadIdx.x < NRANKS && blockIdx.x == 0) {
      PackedInt64_t tail;
      tail.i64[0] = barrierFlag;
      tail.i64[1] = barrierFlag;

      Store128(tail.upacked, (uint4 *)tailPtr);
    }

    dst = dst + chunkElts;
  }
}

template<typename T, int NRANKS, bool AVG>
__global__ void pcclCeOkiReduceFused(struct pcclCeOkiReduceArgs args) {
  if (blockIdx.x < pcclCeReduceMaxBlocks) {
    ceLocalReduce<T, NRANKS, AVG>(args);
    return;
  }

  // The cmd block. The barriers around the DMA are what used to be gpuBarrierWithTrigger and
  // the trailing gpuBarrier: the first one keeps any rank from writing into a peer's station
  // buffer before that peer is in this collective, the second one -- placed after the cmd path
  // has polled its own queues -- is what makes every peer's remote writes visible in recvBuff
  // by the time the kernel retires.
  ncclSymPrims prim(args.comm, 1, ncclSymPrims_UseBarrier);
  prim.barrierArrive(ncclCoopCta(), /*release=*/true);
  prim.barrierWait(ncclCoopCta(), /*acquire=*/false);

  if (args.nDies == 1) okiReduceDmaCmd<1>(args, NRANKS);
  else okiReduceDmaCmd<2>(args, NRANKS);

  prim.barrierArrive(ncclCoopCta(), /*release=*/true);
  prim.barrierWait(ncclCoopCta(), /*acquire=*/false);
}

#define CE_OKI_REDUCE_IMPL(T, nRanks, avg) \
  template __global__ void pcclCeOkiReduceFused<T, nRanks, avg>(struct pcclCeOkiReduceArgs args);

#define CE_OKI_REDUCE_T_IMPL(nRanks, avg) \
  CE_OKI_REDUCE_IMPL(half, nRanks, avg) \
  CE_OKI_REDUCE_IMPL(__ppu_bfloat16, nRanks, avg) \
  CE_OKI_REDUCE_IMPL(float, nRanks, avg)

#define CE_OKI_REDUCE_R_IMPL(avg) \
  CE_OKI_REDUCE_T_IMPL(2, avg) \
  CE_OKI_REDUCE_T_IMPL(4, avg) \
  CE_OKI_REDUCE_T_IMPL(8, avg)

CE_OKI_REDUCE_R_IMPL(false)
CE_OKI_REDUCE_R_IMPL(true)

#define CE_OKI_REDUCE_FUNCNAME(T, nRanks, avg) \
  (void*)pcclCeOkiReduceFused<T, nRanks, avg>,

#define CE_OKI_REDUCE_T_FUNCNAME(nRanks, avg) \
  CE_OKI_REDUCE_FUNCNAME(half, nRanks, avg) \
  CE_OKI_REDUCE_FUNCNAME(__ppu_bfloat16, nRanks, avg) \
  CE_OKI_REDUCE_FUNCNAME(float, nRanks, avg)

#define CE_OKI_REDUCE_R_FUNCNAME(avg) \
  CE_OKI_REDUCE_T_FUNCNAME(2, avg) \
  CE_OKI_REDUCE_T_FUNCNAME(4, avg) \
  CE_OKI_REDUCE_T_FUNCNAME(8, avg)

// Match localReduceFuncIndex: op, rank count, then dtype; shared by AllReduce and ReduceScatter.
void* pcclCeOkiReduceFuncs[] = {
  CE_OKI_REDUCE_R_FUNCNAME(false)
  CE_OKI_REDUCE_R_FUNCNAME(true)
};
#endif
