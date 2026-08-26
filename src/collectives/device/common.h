/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

#include "common_kernel.h"
#include "collectives.h"
#include "devcomm.h"
#include "utils.h"
#include "reduce_kernel.h"
#include <iostream>

// Exit If Abort Barrier across CTA: make sure all threads exit consistently
// Each thread sets a predicate to true if abort == 1
// all CTA's threads enter the barrier and do a popc on their predicates being True
// If any of the thread's predicate was True, all the threads call exit()
static inline __device__ void exitIfAbortBarrier(int abort) {
  uint32_t popc;
  popc = __ppu_barrier_sync_count(0, 0, abort);
  if (popc) { __ppu_exit(); }
}

static inline __device__ void exitIfAbort(int abort, volatile int* pAbort, int tid) {
  if (tid == 0) {
    *pAbort = abort;
  }
  __syncthreads();
  if (*pAbort) {
    __ppu_exit();
  }
}

typedef void(*ncclKern_t)(void* args);
extern __device__ ncclKern_t pcclFuncs[];

static __device__ void init_parallel(void* dst, size_t size, int tid) {
  int* d = (int*)dst;
  for (int o = tid; o < (size/sizeof(int)); o += blockDim.x) d[o] = 0;
}

// Copy 16-byte aligned data. You must call with at least `(bytes+15)/16` threads.
inline __device__ void copyToShmem16(int tid, void* dst, void const* src, int bytes) {
  int offset = 16*tid;
  if (offset < bytes) {
    uint64_t a=0, b=0;
    asm volatile("ppu.ld.global.v2.u64 {%0,%1},[%2];" : "=l"(a),"=l"(b) : "l"((char const*)src + offset) : "memory");
#ifdef NO_TSM
    asm volatile("ppu.st.v2.u64 [%0],{%1,%2};" :: "r"(dst + offset), "l"(a), "l"(b) : "memory");
#else
    uint32_t udst = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("ppu.st.shared.v2.u64 [%0],{%1,%2};" :: "r"(udst + offset), "l"(a), "l"(b) : "memory");
#endif
  }
}

extern __shared__ ncclShmemData pcclShmem;
extern __shared__ ulong2 pcclShmemPerWarp[/*ncclShmemDynamicSize()/sizeof(ulong2)*/];

__device__ inline void* ncclScratchForWarp(int warp) {
  return (char*)pcclShmemPerWarp + warp*ncclShmemScratchWarpSize();
}

__device__ inline struct ncclPatShmem* ncclPatShmemPtr(struct ncclKernelComm* comm) {
#ifdef NO_TSM
  return (struct ncclPatShmem*)comm->channels[blockIdx.x].patShmem;
#else
  static_assert(sizeof(struct ncclPatShmem) <= ncclShmemScratchWarpSize()*(NCCL_MAX_NTHREADS/WARP_SIZE),
                "PAT scratch too small");
  return (struct ncclPatShmem*)ncclScratchForWarp(0);
#endif
}

#ifdef NO_TSM
__device__ inline int find_nth_set_bit(uint32_t bitmask, int n) {
    int result;
    asm volatile(
        "{\n"
        "  .reg .u32 mask;\n"
        "  .reg .u32 index;\n"
        "  mov.b32 mask, %1;\n"
        "  fns.b32 index, mask, 0, %2;\n"
        "  mov.s32 %0, index;\n"
        "}\n"
        : "=r"(result)
        : "r"(bitmask), "r"(n)
        : "memory"
    );
    return result;
}
__device__ inline int find_nth_set_bit_64(uint64_t bitmask, int n, int nLow32) {
    uint32_t low32bit = uint32_t(bitmask);
    if (n <= nLow32) {
      return find_nth_set_bit(low32bit, n);
    }
    uint32_t hig32hbit = uint32_t(bitmask>>32);
    return find_nth_set_bit(hig32hbit, n - nLow32) + 32;
}
#endif

// Must run with at least 64 threads
__device__ __forceinline__ void loadWorkBatchToShmem(
    struct ncclShmemData* pShmem, int tid, int tn, struct ncclDevKernelArgs const* args, int batchIx
  ) {
  int lane = tid%WARP_SIZE;
  int workCursor = 0; // num works written in previous loop iterations.
  while (true) {
    struct ncclDevWorkBatch batch = ((struct ncclDevWorkBatch*)(args+1))[batchIx];


#ifndef NO_TSM
    // fnsOfBitset[n] = index of n'th set bit in batch.offsetBitset.
    // PTX has instruction "fns" (find n-th set) but it expands to a lot of SASS,
    // since we know all lanes will be querying the same bitmask we can compute
    // much faster using shared memory.
    uint8_t* fnsOfBitset = (uint8_t*)ncclScratchForWarp(threadIdx.x/WARP_SIZE);
    __syncwarp();
    if (uint32_t(batch.offsetBitset) & (1u<<lane)) {
      int nWorksBelow = __popc(uint32_t(batch.offsetBitset) & ((1u<<lane)-1));
      fnsOfBitset[nWorksBelow] = lane;
    }
    int nWorksLow32 = __popc(uint32_t(batch.offsetBitset)); // just of low 32 bits
    if (uint32_t(batch.offsetBitset>>32) & (1u<<lane)) {
      int nWorksBelow = nWorksLow32;
      nWorksBelow += __popc(uint32_t(batch.offsetBitset>>32) & ((1u<<lane)-1));
      fnsOfBitset[nWorksBelow] = 32 + lane;
    }
    int nWorks = nWorksLow32 + __popc(uint32_t(batch.offsetBitset>>32)); // add high 32 bits
#else
    int nWorksLow32 = __popc(uint32_t(batch.offsetBitset));
    int nWorks = nWorksLow32 + __popc(uint32_t(batch.offsetBitset>>32)); // add high 32 bits
#endif
    __syncwarp();

    int workSize;
    int nPacks; // total number of packs loaded, each pack is 16 bytes
    int packInWork; // my pack index within work struct
    int dstWork; // my work index in contiguous destination shmem
    switch (batch.workType) {
    case (int)ncclDevWorkTypeP2p:
      workSize = sizeof(struct ncclDevWorkP2p);
      nPacks = nWorks*(workSize/16);
      packInWork = tid%(workSize/16);
      dstWork = tid/(workSize/16);
      break;
    case (int)ncclDevWorkTypeColl:
      workSize = sizeof(struct ncclDevWorkColl);
      nPacks = nWorks*(workSize/16);
      packInWork = tid%(workSize/16);
      dstWork = tid/(workSize/16);
      break;
    case (int)ncclDevWorkTypeCollReg:
    default:
      workSize = sizeof(struct ncclDevWorkCollReg);
      nPacks = nWorks*(workSize/16);
      packInWork = tid%(workSize/16);
      dstWork = tid/(workSize/16);
      break;
    }
    if (tid == 0) {
      pShmem->workSize = workSize;
    }
    // We deliberately replicate these div and mod calculations into the case
    // blocks above so that they get constant divisor optimizations by the compiler.
    //   packInWork = tid%(workSize/16);
    //   dstWork = tid/(workSize/16);

    // We can only assume we have 64 threads, which means we can read at most 1024 bytes
    // here which is the per batch maximum.
    if (tid < nPacks) {
#ifdef NO_TSM
      int srcWork = find_nth_set_bit_64(batch.offsetBitset, dstWork + 1, nWorksLow32);
#else
      int srcWork = fnsOfBitset[dstWork]; // find n'th set bit in batch.offsetBitset
#endif
      ulong2 tmp;
      // The loads done in these two cases must be kept separate since we are
      // relying on the compiler to use "ld.param" in the first one. The parameter
      // space is not generically addressable, so any attempt to load through
      // a pointer that *might* be parameter space backed will cause the
      // compiler to spill the parameter struct (4K!) to each thread's local space
      // before creating a pointer (to the spill) and decimate perf.
      //
      // An example of what not to do would be the following:
      //
      // if (condition) {
      //   // The compiler could spill parameter_variable to local space and take
      //   // the address of that, since when src is loaded below it could also
      //   // be global space.
      //   src = &parameter_variable;
      // } else {
      //   src = &global_variable;
      // }
      // memcpy(dst, src, n);
      if (pShmem->args.workStorageType == ncclDevWorkStorageTypeArgs) {
        char* src = (char*)args + (batch.offsetBase + srcWork*workSize + packInWork*16);
        tmp = *(ulong2*)src; // becomes ld.param.v2.u64
      } else {
        char* src = (char*)pShmem->args.workBuf + ((batch.offsetBase + srcWork*workSize + packInWork*16) & (pShmem->args.workMask));
        tmp = *(ulong2*)src; // becomes ld.v2.u64
      }
      char* dst = pShmem->workStorage;
      dst += (workCursor + dstWork)*workSize + packInWork*16;
      *(ulong2*)dst = tmp;
    }
    workCursor += nWorks;

    if (batch.nextExtends) {
      batchIx += batch.nextJump;
      tid -= 64; // Rotate threads so we use the next two warps for next batch struct.
      if (tid < 0) tid += tn;
    } else {
      if (tid == 0) {
        pShmem->batchIx = batchIx;
        pShmem->nextBatchIx = (batch.nextJump == 0) ? -1 : batchIx + batch.nextJump;
        pShmem->workType = (enum ncclDevWorkType)batch.workType;
        pShmem->nWorks = workCursor;
        pShmem->funcId = batch.funcId;
      }
      break;
    }
  }
}

template <ncclFunc_t FUNCTION, int ALGO, int PROTO, class REDOP, typename T, int UNROLL>
class RunWorkColl {
  public:
  __device__ void run(struct ncclDevWorkColl* work) {}
};

template <ncclFunc_t FUNCTION, int ALGO, int PROTO, class REDOP, typename T, int UNROLL>
class RunWorkP2p {
  public:
  __device__ void run(struct ncclDevWorkP2p* work) {}
};

template <ncclFunc_t FUNCTION, int ALGO, int PROTO, class REDOP, typename T, int UNROLL>
class RunWorkBatch {
  public:
  __device__ __forceinline__ void run(void* args) {
  #ifdef NO_TSM
    struct ncclShmemData* pShmem = (struct ncclShmemData*)args;
  #else
    ncclShmemData* pShmem = &pcclShmem;
  #endif

    if (pShmem->funcId > NUM_P2P_FUNC) {
      int tid = threadIdx.x;
      int tn = blockDim.x;

      if (RedOpArg<REDOP>::ArgUsed) {
        int nWorks = pShmem->nWorks;
        for (int w=tid; w < nWorks; w += tn) {
          struct ncclDevWorkColl* work = (ncclDevWorkColl*)(pShmem->workStorage + w*pShmem->workSize);
          if (work->redOpArgIsPtr) {
            work->redOpArg = RedOpArg<REDOP>::loadArg(reinterpret_cast<void*>(work->redOpArg));
          }
        }
        __syncthreads();
      }

      for (int w=0; w < pShmem->nWorks; w++) {
        struct ncclDevWorkColl* work = (struct ncclDevWorkColl*)(pShmem->workStorage + w*pShmem->workSize);
        if (w != 0) {
          struct ncclDevWorkColl* workPrev = (struct ncclDevWorkColl*)(pShmem->workStorage + (w-1)*pShmem->workSize);
          if (work->nThreads != workPrev->nThreads) __syncthreads();
        }
        int subtn = work->nThreads;
        if (tid < subtn) {
          RunWorkColl<FUNCTION, ALGO, PROTO, REDOP, T, UNROLL>().run(work);
        }
      }
    } else {
      ncclDevWorkP2p* work = (ncclDevWorkP2p*)pShmem->workStorage;
      RunWorkP2p<FUNCTION, ALGO, PROTO, REDOP, T, UNROLL>().run(work);
    }
  }
};

#define START 0
#define STOP  1
#define FINI  2

__device__ __forceinline__ unsigned long long int globaltimer() {
  unsigned long long int timer;
  // PPU1.x shader does not support %globaltimer. Below is equivalent to clock64().
  asm volatile("ppu.mov.u64 %0, %%globaltimer;" : "=l"(timer));
  return timer;
}

__device__ __forceinline__ bool profilerEnabled(ncclShmemData* pShmem, int workItemIdx) {
  return (pShmem->workType == ncclDevWorkTypeP2p)
    ? ((struct ncclDevWorkP2p*)pShmem->workStorage)[workItemIdx].profilerEnabled
    : ((struct ncclDevWorkColl*)pShmem->workStorage)[workItemIdx].profilerEnabled;
}

__device__ __forceinline__ void profiler(ncclShmemData* pShmem, int action) {
  // Early exit if profiling is globally disabled to avoid expensive loops
  if (__builtin_expect(!pShmem->comm.profilerGlobalEnabled, 1)) {
    if (threadIdx.x == 0) {
      if (action == STOP) {
        pShmem->channel.workCounter += pShmem->nWorks;
      } else if (action == FINI) {
        pShmem->channel.workCounter += pShmem->nWorks;
        ((ncclKernelCommAndChannels*)pShmem->args.comm)->channels[pShmem->channelId].workCounter = pShmem->channel.workCounter;
      }
    }
    return;
  }

  // Original profiling logic when profiler is enabled
  if (threadIdx.x == 0) {
    int idx = 0;
    uint64_t wc = pShmem->channel.workCounter + 1;
    if (action == START) {
      for (; wc <= pShmem->channel.workCounter + pShmem->nWorks; wc++) {
        if (!profilerEnabled(pShmem, idx++)) continue;
        pShmem->comm.workStarted[pShmem->channelId].data[wc%MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp = globaltimer();
        pShmem->comm.workStarted[pShmem->channelId].data[wc%MAX_PROFILER_EVENTS_PER_CHANNEL].counter   = wc;
      }
    } else {
      for (; wc <= pShmem->channel.workCounter + pShmem->nWorks; wc++) {
        if (!profilerEnabled(pShmem, idx++)) continue;
        pShmem->comm.workCompleted[pShmem->channelId].data[wc%MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp = globaltimer();
        pShmem->comm.workCompleted[pShmem->channelId].data[wc%MAX_PROFILER_EVENTS_PER_CHANNEL].counter   = wc;
      }
      pShmem->channel.workCounter += pShmem->nWorks;
      if (action == FINI) ((ncclKernelCommAndChannels*)pShmem->args.comm)->channels[pShmem->channelId].workCounter = pShmem->channel.workCounter;
    }
  }
}

__device__ __forceinline__ void kernelExecStateMonitor(ncclShmemData* pShmem, int action) {
  // graph hang monitor not supported
  if (pShmem->args.persistent) {
    return;
  }
  if (threadIdx.x == 0) {
    if (action == START) {
      pShmem->channel.perKernelBatchCounter = 0;
      volatileStoreU64(pShmem->args.comm->channels[pShmem->channelId].devPerKernelBatchCounter, 0);
    } else if (action == STOP) {
      pShmem->channel.perKernelBatchCounter += 1;

      volatileStoreU64(pShmem->args.comm->channels[pShmem->channelId].devPerKernelBatchCounter, pShmem->channel.perKernelBatchCounter);
    } else if (action == FINI) {
      pShmem->channel.perKernelBatchCounter += 1;

      uint8_t stateMonitorLevel = pShmem->comm.stateMonitorLevel;
      if (stateMonitorLevel == 2) {
        volatileStoreU64(pShmem->args.comm->channels[pShmem->channelId].hostDevIndexLevel2, pShmem->args.kernelIdx);
      }
      if (stateMonitorLevel == 1) {
        volatileStoreU64(pShmem->args.comm->channels[pShmem->channelId].devIndexLevel1, pShmem->args.kernelIdx);
      }

      volatileStoreU64(pShmem->args.comm->channels[pShmem->channelId].devPerKernelBatchCounter, pShmem->channel.perKernelBatchCounter);
    }
  }
}

#ifndef NO_TSM
extern __shared__ struct ncclShmemData pcclShmem;
#endif
template <ncclFunc_t FUNCTION, int ALGO, int PROTO, class REDOP, typename T, int UNROLL, int FINDEX>
__device__ void ncclKernelMain(struct ncclDevKernelArgs const* args)  {
  int tid = threadIdx.x;
  int tn = blockDim.x;
  auto f = RunWorkBatch<FUNCTION, ALGO, PROTO, REDOP, T, UNROLL>();

#ifdef NO_TSM
  ncclShmemData* pShmem = getShmemPtr(args->comm);
#else
  ncclShmemData* pShmem = &pcclShmem;
#endif

  // Copy kernel args to shmem and then only read those. Otherwise the compiler
  // will end up putting the args into thread local stack which is very wasteful.
  if (tid < sizeof(ncclDevKernelArgs)/sizeof(uint32_t)) {
    ((uint32_t*)&(pShmem->args))[tid] = ((uint32_t*)args)[tid];
  }

  // To map blockId to channelId, we need the n'th set bit of channelMask which
  // is the inverse of counting the number of set bits among the the first n.
  // PTX has the fns instruction which does this but is extremely slow. We can
  // do better when we know all threads are querying the same bitmask.
  if (tid < MAXCHANNELS && (args->channelMask & (1ull<<tid))) {
    int n = __popcll(args->channelMask & ((1ull<<tid)-1));
    if (blockIdx.x == n) pShmem->channelId = tid;
  }
  __syncthreads(); // publish ncclShmem.{args, channelId}
  /* set abort flag to 0 */
  if (tid == 0) {
    pShmem->aborted = 0;
    pShmem->channel.workCounter = ((ncclKernelCommAndChannels*)pShmem->args.comm)->channels[pShmem->channelId].workCounter;
  }

  // Use first 2 warps to load comm and channel, and remaining load work batch.
  switch (tid/WARP_SIZE) {
  case 0:
    { void* dst = &(pShmem->comm);
      void* src = pShmem->args.comm;
      int bytes = sizeof(ncclKernelComm);
      static_assert(sizeof(ncclKernelComm) <= 16*WARP_SIZE, "ncclKernelComm cannot be loaded by a single warp in one insn.");
      copyToShmem16(tid, dst, src, bytes);
    } break;
  case 1:
    { // Get address of channel without incurring indirect load from ncclKernelComm::channels
      void* dst = &(pShmem->channel);
      void* src = &((ncclKernelCommAndChannels*)pShmem->args.comm)->channels[pShmem->channelId];
      int bytes = sizeof(ncclDevChannel);
      static_assert(sizeof(ncclDevChannel) <= 16*WARP_SIZE, "ncclDevChannel cannot be loaded by a single warp in one insn.");
      copyToShmem16(tid-WARP_SIZE, dst, src, bytes);
    } break;
  default:
    { int subtid = tid - 2*WARP_SIZE;
      int subtn = tn - 2*WARP_SIZE;
      // Coverity reports a possible thread divergence due to not all threads participating in the collective.
      // However, the code ensures that the participation is on a per-warp basis.
      // coverity[device_thread_diverged:FALSE]
      loadWorkBatchToShmem(pShmem, subtid, subtn, args, /*batchIx=*/blockIdx.x);
    } break;
  }
  __syncthreads(); // publish pcclShmem

  while (pShmem->aborted == 0) {
    kernelExecStateMonitor(pShmem, START);
    profiler(pShmem, START);
    if (pShmem->funcId == (unsigned)FINDEX) {
      f.run(pShmem);
    } else {
      pcclFuncs[pShmem->funcId](pShmem);
    }

    if (pShmem->nextBatchIx == -1) break;
    int batchIx = pShmem->nextBatchIx;
    __syncthreads();
    kernelExecStateMonitor(pShmem, STOP);
    profiler(pShmem, STOP);
    loadWorkBatchToShmem(pShmem, tid, tn, args, batchIx);
    __syncthreads();
  }

  __syncthreads();
  kernelExecStateMonitor(pShmem, FINI);
  profiler(pShmem, FINI);
}

// Only generate kernels for SUM
#if NCCL_OP == 0
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 150
// relax max vreg limit for ppu1.5. (640, 2) -> 96, (640, 1) -> 168, (896, 1) -> 128
#define NCCL_KERN_LAUNCH_BOUNDS_MAX_NTHREADS 896
#define NCCL_KERN_LAUNCH_BOUNDS_MIN_NBLOCKS 1
#define IMPL_COLL_KERN(func, algo, proto, devredop, type, fIndex) \
__global__ void __launch_bounds__(NCCL_KERN_LAUNCH_BOUNDS_MAX_NTHREADS, NCCL_KERN_LAUNCH_BOUNDS_MIN_NBLOCKS) \
  NCCL_KERN_NAME(func, algo, proto, devredop, type)(ncclDevKernelArgs1K const args4K) { \
    ncclKernelMain<ncclFunc##func, NCCL_ALGO_##algo, NCCL_PROTO_##proto, Func##devredop<type>, type, ncclCollUnroll(), fIndex>(&args4K.args); \
}
#else
#define IMPL_COLL_KERN(func, algo, proto, devredop, type, fIndex) \
__global__ void NCCL_KERN_NAME(func, algo, proto, devredop, type)(ncclDevKernelArgs1K const args4K) { \
  ncclKernelMain<ncclFunc##func, NCCL_ALGO_##algo, NCCL_PROTO_##proto, Func##devredop<type>, type, ncclCollUnroll(), fIndex>(&args4K.args); \
}
#endif
#else
#define IMPL_COLL_KERN(func, algo, proto, devredop, type, fInded)
#endif

// Examples :     AllReduce, RING, LL,    Sum,   uint8
#if FAKE_KERNEL == 0
#define IMPL_COLL_FUNC(func, algo, proto, devredop, type) \
__device__ void NCCL_FUNC_NAME(func, algo, proto, devredop, type)(void* args) { \
  auto f = RunWorkBatch<ncclFunc##func, NCCL_ALGO_##algo, NCCL_PROTO_##proto, Func##devredop<type>, type, ncclCollUnroll()>(); \
  f.run(args); \
}
#else
#define IMPL_COLL_FUNC(func, algo, proto, devredop, type) \
__device__ void NCCL_FUNC_NAME(func, algo, proto, devredop, type)(void* args) { \
  int tid = threadIdx.x; \
  if (tid == 0) { \
    printf("Fake kernel not built into pccl lib yet."); \
  } \
  __ppu_exit(); \
}
#endif

// Only generate inline kernels for LL
#define IMPL_COLL4(func, algo, devredop, type, ncclType) \
  IMPL_COLL_FUNC(func, algo, LL,     devredop, type) \
  IMPL_COLL_FUNC(func, algo, LL128,  devredop, type) \
  IMPL_COLL_FUNC(func, algo, SIMPLE, devredop, type) \
  IMPL_COLL_FUNC(func, algo, SIGNAL, devredop, type) \
  IMPL_COLL_FUNC(func, algo, SIMPLE_BULK, devredop, type) \
  IMPL_COLL_FUNC(func, algo, SIGNAL_BULK, devredop, type) \
  IMPL_COLL_FUNC(func, algo, SIMPLE_BULK_AR, devredop, type) \
  IMPL_COLL_KERN(func, algo, LL,     devredop, type, FUNC_INDEX(ncclFunc##func, nccl##devredop, ncclType, NCCL_ALGO_##algo, NCCL_PROTO_LL)) \

#define IMPL_COLL3(func, devredop, type, ncclType) \
  IMPL_COLL4(func, TREE,    devredop, type, ncclType) \
  IMPL_COLL4(func, RING,    devredop, type, ncclType) \
  IMPL_COLL4(func, COLLNET, devredop, type, ncclType) \
  IMPL_COLL4(func, FC,      devredop, type, ncclType) \
  IMPL_COLL4(func, PAT,     devredop, type, ncclType)

#if NCCL_TYPE == 0
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, int8_t,   ncclInt8)
#elif NCCL_TYPE == 1
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, uint8_t,  ncclUint8)
#elif NCCL_TYPE == 2
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, int32_t,  ncclInt32)
#elif NCCL_TYPE == 3
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, uint32_t, ncclUint32)
#elif NCCL_TYPE == 4
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, int64_t,  ncclInt64)
#elif NCCL_TYPE == 5
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, uint64_t, ncclUint64)
#elif NCCL_TYPE == 6
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, half,     ncclFloat16)
#elif NCCL_TYPE == 7
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, float,    ncclFloat32)
#elif NCCL_TYPE == 8
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, double,   ncclFloat64)
#elif NCCL_TYPE == 9
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, __ppu_bfloat16,   ncclBfloat16)
#elif NCCL_TYPE == 10
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, __hg_fp8_e4m3,   ncclFloat8e4m3)
#elif NCCL_TYPE == 11
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, __hg_fp8_e5m2,   ncclFloat8e5m2)
#endif

// Reduction define all functions
#if NCCL_OP == 0
#define IMPL_COLL_R(func) IMPL_COLL2(func, Sum);
#elif NCCL_OP == 1
#define IMPL_COLL_R(func) IMPL_COLL2(func, Prod);
#elif NCCL_OP == 2
#define IMPL_COLL_R(func) IMPL_COLL2(func, Max);
#elif NCCL_OP == 3
#define IMPL_COLL_R(func) IMPL_COLL2(func, Min);
#elif NCCL_OP == 4
#define IMPL_COLL_R(func) IMPL_COLL2(func, PreMulSum);
#elif NCCL_OP == 5
  #if NCCL_TYPE < 6
    #define IMPL_COLL_R(func) IMPL_COLL2(func, SumPostDiv);
  #else
    #define IMPL_COLL_R(func) // skip SumPostDiv for floating point
  #endif
#endif

#if NCCL_OP == 0 && NCCL_TYPE == 0
// Copy primitives only define one function for copy
#define IMPL_COLL_C(func) IMPL_COLL3(func, Sum, int8_t, ncclInt8);

// Point-to-point primitives only have one function/kernel.
#define IMPL_COLL_P(func) \
  IMPL_COLL_FUNC(func, RING, SIMPLE, Sum, int8_t); \
  IMPL_COLL_KERN(func, RING, SIMPLE, Sum, int8_t, 0); \
  IMPL_COLL_FUNC(func, RING, SIMPLE_BULK, Sum, int8_t); \
  IMPL_COLL_KERN(func, RING, SIMPLE_BULK, Sum, int8_t, 0); \
  IMPL_COLL_FUNC(func, RING, SIGNAL, Sum, int8_t); \
  IMPL_COLL_KERN(func, RING, SIGNAL, Sum, int8_t, 0); \
  IMPL_COLL_FUNC(func, RING, SIGNAL_BULK, Sum, int8_t); \
  IMPL_COLL_KERN(func, RING, SIGNAL_BULK, Sum, int8_t, 0); \
  IMPL_COLL_FUNC(func, RING, SIMPLE_BULK_AR, Sum, int8_t); \
  IMPL_COLL_KERN(func, RING, SIMPLE_BULK_AR, Sum, int8_t, 0);
#else
#define IMPL_COLL_C(func)
#define IMPL_COLL_P(func)
#endif

#endif
