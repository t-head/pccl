/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_COLLECTIVES_H_
#define NCCL_COLLECTIVES_H_

#include "devcomm.h"

#define NUM_P2P_FUNC 5
#define FUNC_INDEX_P2P_SIMPLE 0
#define FUNC_INDEX_P2P_SIMPLE_BULK 1
#define FUNC_INDEX_P2P_SIGNAL 2
#define FUNC_INDEX_P2P_SIGNAL_BULK 3
#define FUNC_INDEX_P2P_SIMPLE_BULK_AR 4
#define FUNC_INDEX(func, devredop, ncclType, al, pr) (NUM_P2P_FUNC+(((((func)*ncclNumDevRedOps + (devredop))*ncclNumTypes) + (ncclType))*NCCL_NUM_ALGORITHMS+(al))*NCCL_NUM_PROTOCOLS+(pr))

#define TOTAL_FUNCS_NUM (NUM_P2P_FUNC+NCCL_NUM_FUNCTIONS*ncclNumDevRedOps*ncclNumTypes*NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS)

#define NCCL_FUNC_NAME(func, algo, proto, devredop, type) \
  pcclFunction_##func##_##algo##_##proto##_##devredop##_##type

#define NCCL_KERN_NAME(func, algo, proto, devredop, type) \
  pcclKernel_##func##_##algo##_##proto##_##devredop##_##type

#define NCCL_IMPL_NAME(func, algo, proto) \
  pccl##func##algo##proto

/* Declare all collective operations */
#define DECL5(func, algo, proto, devredop, type) \
  extern __device__ void NCCL_FUNC_NAME(func, algo, proto, devredop, type)(void* args); \
  extern __global__ void NCCL_KERN_NAME(func, algo, proto, devredop, type)(ncclDevKernelArgs1K const args4K); \

#define CONCAT(a,b) a##b
#define MACRO_IF(cond, t, f) CONCAT(MACRO_IF_, cond)(t, f)
#define MACRO_IF_0(t, f) f
#define MACRO_IF_1(t, f) t

#define DECL_P2P(func, algo, devredop, type) \
  DECL5(func, algo, SIMPLE, devredop, type) \
  DECL5(func, algo, SIMPLE_BULK, devredop, type) \
  DECL5(func, algo, SIGNAL, devredop, type) \
  DECL5(func, algo, SIGNAL_BULK, devredop, type) \
  DECL5(func, algo, SIMPLE_BULK_AR, devredop, type)

#define DECL4(func, algo, devredop, type, undef) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, SIMPLE_BULK_AR, devredop, type)) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, SIGNAL_BULK, devredop, type)) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, SIMPLE_BULK, devredop, type)) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, SIGNAL, devredop, type)) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, SIMPLE, devredop, type)) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, LL,     devredop, type)) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, LL128,  devredop, type))

#define DECL3(func, devredop, type, undef) \
  DECL4(func, RING,    devredop, type, undef) \
  DECL4(func, TREE,    devredop, type, undef) \
  DECL4(func, COLLNET, devredop, type, undef) \
  DECL4(func, FC,      devredop, type, undef) \
  DECL4(func, PAT,     devredop, type, undef)

#define DECL2(func, devredop, undefForFloat) \
  DECL3(func, devredop, int8_t, /*undef=*/0) \
  DECL3(func, devredop, uint8_t, /*undef=*/0) \
  DECL3(func, devredop, int32_t, /*undef=*/0) \
  DECL3(func, devredop, uint32_t, /*undef=*/0) \
  DECL3(func, devredop, int64_t, /*undef=*/0) \
  DECL3(func, devredop, uint64_t, /*undef=*/0) \
  DECL3(func, devredop, half, /*undef=*/undefForFloat) \
  DECL3(func, devredop, float, /*undef=*/undefForFloat) \
  DECL3(func, devredop, double, /*undef=*/undefForFloat) \
  DECL3(func, devredop, __ppu_bfloat16, /*undef=*/undefForFloat) \
  DECL3(func, devredop, __hg_fp8_e4m3, /*undef=*/undefForFloat) \
  DECL3(func, devredop, __hg_fp8_e5m2, /*undef=*/undefForFloat)


#define DECL(func) \
  DECL2(func, Sum, /*undefForFloat=*/0) \
  DECL2(func, Prod, /*undefForFloat=*/0) \
  DECL2(func, Max, /*undefForFloat=*/0) \
  DECL2(func, Min, /*undefForFloat=*/0) \
  DECL2(func, PreMulSum, /*undefForFloat=*/0) \
  DECL2(func, SumPostDiv, /*undefForFloat=*/1)

#define DECL_ALL \
  DECL2(Broadcast, Sum, /*undefForFloat=*/0) \
  DECL(Reduce) \
  DECL2(AllGather, Sum, /*undefForFloat=*/0) \
  DECL(ReduceScatter) \
  DECL(AllReduce) \
  DECL_P2P(SendRecv, RING, Sum, int8_t)

DECL_ALL
DECL5(SendRecvNetPoll, RING, SIMPLE, Sum, int8_t)

// Need a power of two to ensure it divides by parallelFactor (which is also a power of two)
#define NCCL_PAT_NWORKERS 512

static constexpr int PatUsed = 0x1, PatSkipped = 0x2;

struct ncclPatStep {
  int recvDim, sendDim, recvOffset, sendOffset, stepOffset, postRecv, postSend, nelem, last, flags;
  size_t inpIx, outIx;
};

struct ncclPatPeer {
  uint64_t step;
  struct ncclConnInfo* conn;
  void* buff;
  int* sizesFifo;      // GPU -> proxy size notification for NET send (PCCL uses sizesFifo)
  uint64_t* headPtr;
  uint64_t* tailPtr;
  uint64_t stepCache;
  long long int accSize;
  int connStepSize;
};

#define NCCL_SHMEM_PAT_STEPS 32
struct ncclPatShmem {
  struct ncclPatStep patSteps[NCCL_SHMEM_PAT_STEPS];
  int parallelFactor;
  long long int localAccSize;
  struct ncclPatPeer sendDims[32]; // Should cover 2^32 ranks
  struct ncclPatPeer recvDims[32];
};

#if defined(__HGGC_ARCH__)
inline __device__ void patStepFlagsStoreRelease(volatile int* ptr, int val) {
  asm volatile("ppu.st.release.blk.b32 [%0], %1;" :: "l"(ptr), "r"(val) : "memory");
}
inline __device__ int patStepFlagsLoadAcquire(volatile int* ptr) {
  int val;
  asm volatile("ppu.ld.acquire.blk.b32 %0,[%1];" : "=r"(val) : "l"(ptr) : "memory");
  return val;
}
#endif

template <typename T>
class PatRSAlgorithm {
  size_t offset;
  size_t end;
  size_t count;
  int chunkCount;
  int nelem;
  int rank;
  int nranks;
  int nrPow2;
  int postFreq;
  int lastA;
  int parallelFactor;
  int aggFactor;
  int as; // aggregated steps
  int a; // step inside aggregated step
  int sendSkipped; // number of skipped steps during aggregation
  int stepOffset;
  int aggDelta;
  int scale;
  int phase;

  __device__ __host__ ssize_t min(ssize_t a, ssize_t b) {
    return (a < b) ? a : b;
  }

  __device__ __host__ int getNelem() {
    return min(chunkCount, end - offset);
  }

  __device__ __host__ int mirrorInvert(int i, int max) {
    int ret = 0;
    for (int mask = 1, imask = max / 2; mask < max; mask <<= 1, imask >>= 1) {
      if ((i & mask) == 0) ret += imask;
    }
    return ret;
  }

  __device__ __host__ int firstBitSet(int i, int max) {
    int ffs = __builtin_ffs(i);
    return ffs ? ffs - 1 : max;
  }

  __device__ __host__ void resetA() {
    a = 0;
    sendSkipped = stepOffset = 0;
    lastA = aggFactor;
    if (phase >= 2) lastA /= 2 * scale;
    if (phase == 4) lastA = 1;
  }

  __device__ __host__ void reset() {
    nelem = getNelem();
    phase = 0;
    scale = 1;
    as = aggDelta - 1;
    resetA();
  }

  __device__ __host__ int nBitsSet(int i) {
    return __builtin_popcount(i);
  }

  // Return 1 when only upper bits are set. For example, if nrpow2==16 we'll return 1 for 8, 12, 14, 15.
  // A number being in the form of 1111000 implies that the complementary is 0000111 meaning it's a power of 2 minus 1.
  __device__ __host__ int newPeer(int i, int pow2) {
    return nBitsSet((i ^ (pow2 - 1)) + 1) == 1 ? 1 : 0;
  }

public:
  __device__ __host__ PatRSAlgorithm(int stepSize, int stepDepth, int maxParallelFactor, size_t offset, size_t end,
                                     size_t count, int chunkCount, int rank, int nranks)
    : offset(offset), end(end), count(count), chunkCount(chunkCount), rank(rank), nranks(nranks) {
    parallelFactor = maxParallelFactor;
    aggDelta = nrPow2 = (1 << log2Up(nranks));

    aggFactor = 1;
    size_t channelSize = end - offset;
    while (stepSize / (channelSize * sizeof(T) * aggFactor) >= 2 && aggFactor < nranks / 2) {
      aggFactor *= 2;
      aggDelta /= 2;
    }
    postFreq = aggFactor;
    if (postFreq < parallelFactor) parallelFactor = postFreq;
    int d = stepDepth;
    while (d > 1 && aggFactor < nranks / 2) {
      d /= 2;
      aggFactor *= 2;
      aggDelta /= 2;
    }

    reset();
  }

  __device__ __host__ int getParallelFactor() {
    return parallelFactor;
  }

  __device__ __host__ void getNextOp(struct ncclPatStep* ps) {
    ps->last = 0;
    ps->nelem = nelem;
    ps->outIx = offset;
    ps->stepOffset = stepOffset;
    int skip = 0;
    if (a >= lastA) {
      skip = 1;
    } else if (phase == 0) {
      int s = mirrorInvert(a, lastA) * aggDelta + as;
      if (s >= nranks) skip = 1;
      int sendDataRank = (rank + s) % nranks;
      ps->inpIx = sendDataRank * count + offset;
      ps->recvDim = -1;
      ps->sendDim = 0;
      ps->outIx = 0;
      ps->recvOffset = -1;
      ps->sendOffset = (a % postFreq) * nelem;
      if (((a % postFreq) + 1 >= postFreq) || (a == lastA - 1)) {
        ps->postSend = 1;
      } else {
        ps->postSend = 0;
      }
      ps->postRecv = 0;
    } else if (phase == 1) {
      int s = mirrorInvert(a, lastA) * aggDelta + as;
      if (s >= nranks) skip = 1;
      ps->recvDim = firstBitSet(s, nrPow2);
      ps->sendOffset = (a % postFreq) * nelem;
      ps->recvOffset = (a % postFreq) * nelem;
      ps->postSend = 0;
      if (ps->recvDim == 0 && (((a % postFreq) + 1 >= postFreq) || (a == lastA - 1))) ps->postSend = 1;
      if (((a % postFreq) + 1 >= postFreq) || (a == lastA - 1)) {
        ps->postRecv = 1;
      } else {
        ps->postRecv = 0;
      }
      s -= (1 << ps->recvDim);
      int recvDataRank = (rank + nranks + s) % nranks;
      ps->inpIx = recvDataRank * count + offset;
      ps->sendDim = s ? firstBitSet(s, nrPow2) : -1;
      if (ps->sendDim == -1) {
        ps->sendOffset = -1;
      } else if (as - (1 << ps->recvDim) == 0) {
        if (newPeer(a, aggFactor)) {
          sendSkipped = a;
          ps->stepOffset = stepOffset = 0;
        }
        int foffset = a - sendSkipped;
        ps->sendOffset = (foffset % postFreq) * nelem;
      }
      int recvDim = ps->recvDim;
      if (s < nranks && skip) {
        ps->recvDim = -1;
        ps->recvOffset = -1;
        ps->postRecv = 0;
        skip = 0;
      }
      if (recvDim > 0 && (((a - sendSkipped) % postFreq) + 1 >= postFreq) && skip == 0) stepOffset++;
    } else if (phase == 2) {
      int s = (2 * mirrorInvert(a, lastA) + 1) * scale * aggDelta + 1;
      ps->postRecv = 0;
      if (s >= nranks) skip = 1;
      ps->recvDim = 0;
      ps->postSend = a == lastA - 1 ? 1 : 0;
      s -= 1;
      if (s < nranks && skip) {
        ps->recvDim = -1;
        ps->recvOffset = -1;
        skip = 0;
      } else if (!skip) {
        int foffset = a + aggFactor - aggFactor / scale;
        ps->postRecv |= ((foffset + 1) % postFreq) == 0 ? 1 : 0;
        ps->recvOffset = (foffset % postFreq) * nelem;
      }
      int recvDataRank = (rank + nranks + s) % nranks;
      ps->inpIx = recvDataRank * count + offset;
      ps->sendDim = s ? firstBitSet(s, nrPow2) : -1;
      int foffset = a;
      ps->postSend |= ((foffset + 1) % postFreq) == 0 ? 1 : 0;
      ps->sendOffset = (foffset % postFreq) * nelem;
    } else if (phase == 3) {
      int s = (2 * mirrorInvert(a, lastA) + 1) * scale * aggDelta;
      ps->postRecv = a == lastA - 1 ? 1 : 0;
      if (s >= nranks) skip = 1;
      ps->recvDim = firstBitSet(s, nrPow2);
      ps->postSend = 0;
      s -= (1 << ps->recvDim);
      int foffset = a;
      ps->postRecv |= (foffset + 1) % postFreq == 0 ? 1 : 0;
      ps->recvOffset = (foffset % postFreq) * nelem;
      int recvDataRank = (rank + nranks + s) % nranks;
      ps->inpIx = recvDataRank * count + offset;
      ps->sendDim = s ? firstBitSet(s, nrPow2) : -1;
      if (s < nranks && skip) {
        ps->recvDim = -1;
        ps->recvOffset = -1;
        ps->postRecv = 0;
        skip = 0;
      }
      if (newPeer(a, aggFactor / (2 * scale))) {
        sendSkipped = a;
        ps->stepOffset = stepOffset = 0;
      }
      foffset = a - sendSkipped;
      if ((foffset % postFreq) + 1 >= postFreq && skip == 0) stepOffset++;
      ps->sendOffset = ps->sendDim >= 0 ? (foffset % postFreq) * nelem : -1;
    } else if (phase == 4) {
      ps->recvDim = 0;
      ps->sendDim = -1;
      ps->inpIx = rank * count + offset;
      ps->recvOffset = ((aggFactor - 1) % postFreq) * nelem;
      ps->sendOffset = -1;
      ps->postRecv = 1;
      ps->postSend = 0;
      offset += chunkCount;
    }
    a++;
    if (a >= lastA && a >= parallelFactor) {
      int p = phase;
      if (p == 1) as--;
      if (p == 3) scale *= 2;
      phase = p == 0 ? as == 1 ? (aggFactor > 1 ? 2 : 4) : 1 :
              p == 1 ? as % 2 == 1 ? 0 : 1 :
              p == 2 ? 3 :
              p == 3 ? scale < aggFactor ? 2 : 4 :
                       5;
      if (p == 4) {
        if (offset >= end) {
          ps->last = 2;
        } else {
          reset();
        }
      } else {
        resetA();
      }
    } else if (phase == 4 && offset >= end) {
      ps->last = 1;
    }
    int flags = PatUsed | (skip ? PatSkipped : 0);
#if defined(__HGGC_ARCH__)
    patStepFlagsStoreRelease(&ps->flags, flags);
#else
    ps->flags = flags;
#endif
  }
};

template <typename T>
class PatAGAlgorithm {
  size_t offset;
  size_t end;
  size_t count;
  int chunkCount;
  int nelem;
  int rank;
  int nranks;
  int nrPow2;
  int postFreq;
  int lastA;
  int parallelFactor;
  int aggFactor;
  int as; // aggregated steps
  int a; // step inside aggregated step
  int aggDelta;
  int scale;
  int phase;

  // AS computation
  int asDim;
  int v;
  int bitCount[32];
  int bitZeroStep[32];

  __device__ __host__ ssize_t min(ssize_t a, ssize_t b) {
    return (a < b) ? a : b;
  }

  __device__ __host__ int getNelem() {
    return min(chunkCount, end - offset);
  }

  __device__ __host__ int mirror(int i, int max) {
    int ret = 0;
    for (int mask = 1, imask = max / 2; mask < max; mask <<= 1, imask >>= 1) {
      if ((i & mask)) ret += imask;
    }
    return ret;
  }

  __device__ __host__ int firstBitSet(int i, int max) {
    int ffs = __builtin_ffs(i);
    return ffs ? ffs - 1 : max;
  }

  __device__ __host__ void resetA() {
    a = 0;
    lastA = aggFactor;
    if (phase >= 2) lastA /= 2 * scale;
  }

  __device__ __host__ void reset() {
    nelem = getNelem();
    scale = aggFactor / 2;
    phase = scale ? 2 : 1;
    v = 0;
    for (int i = 0; i < asDim; i++) {
      bitCount[i] = asDim - i;
      bitZeroStep[i] = 1;
    }
    as = nextAs();
    resetA();
  }

  __device__ __host__ int nextAs() {
    for (int d = 0; d < asDim; d++) {
      int p = 1 << d;
      bitCount[d]--;
      if (bitCount[d] == 0) {
        v ^= p;
        bitCount[d] = p;
        if ((v & p) == 0) {
          bitCount[d] += firstBitSet(bitZeroStep[d], asDim) - 1;
          if (bitCount[d] == 0) {
            v ^= p;
            bitCount[d] = p;
          }
          bitZeroStep[d]++;
        }
      }
    }
    return v;
  }

public:
  __device__ __host__ PatAGAlgorithm(int stepSize, int stepDepth, int maxParallelFactor, size_t offset, size_t end,
                                     size_t count, int chunkCount, int rank, int nranks)
    : offset(offset), end(end), count(count), chunkCount(chunkCount), rank(rank), nranks(nranks) {
    parallelFactor = maxParallelFactor;
    aggDelta = nrPow2 = (1 << log2Up(nranks));

    aggFactor = 1;
    size_t channelSize = end - offset;
    while (stepSize / (channelSize * sizeof(T) * aggFactor) >= 2 && aggFactor < nranks / 2) {
      aggFactor *= 2;
      aggDelta /= 2;
    }
    postFreq = aggFactor;
    if (postFreq < parallelFactor) parallelFactor = postFreq;
    int d = stepDepth;
    while (d > 1 && aggFactor < nranks / 2) {
      d /= 2;
      aggFactor *= 2;
      aggDelta /= 2;
    }

    asDim = log2Up(aggDelta);
    reset();
  }

  __device__ __host__ int getParallelFactor() {
    return parallelFactor;
  }

  __device__ __host__ void getNextOp(struct ncclPatStep* ps) {
    ps->last = 0;
    ps->nelem = nelem;
    ps->inpIx = offset;
    int skip = 0;
    if (a >= lastA) {
      skip = 1;
    } else if (phase == 0) {
      int s = a * aggDelta + as;
      if (s >= nranks) skip = 1;
      int recvDataRank = (rank + s) % nranks;
      ps->outIx = recvDataRank * count + offset;
      ps->sendDim = -1;
      ps->recvDim = 0;
      ps->inpIx = 0;
      ps->sendOffset = -1;
      ps->recvOffset = (a % postFreq) * nelem;
      ps->stepOffset = 0;
      ps->postRecv = (a % postFreq == postFreq - 1) || ((a + 1) * aggDelta + as >= nranks) ? 1 : 0;
      ps->postSend = 0;
    } else if (phase == 1) {
      int s = a * aggDelta + as;
      if (s >= nranks) skip = 1;
      ps->sendDim = firstBitSet(s, nrPow2);
      s -= (1 << ps->sendDim);
      int sendDataRank = (rank + nranks + s) % nranks;
      ps->outIx = sendDataRank * count + offset;
      ps->recvDim = s ? firstBitSet(s, nrPow2) : -1;
      ps->sendOffset = ps->recvOffset = (a % postFreq) * nelem;
      ps->postSend = (a % postFreq == postFreq - 1) || ((a + 1) * aggDelta + as >= nranks) ? 1 : 0;
      ps->postRecv =
        (ps->sendDim == 0) && ((a % postFreq == postFreq - 1) || ((a + 1) * aggDelta + as - 1 >= nranks)) ? 1 : 0;
      ps->stepOffset = (ps->sendDim == 0) ? 0 : a / postFreq;
      if (ps->recvDim == -1) {
        ps->recvOffset = -1;
        ps->postRecv = 0;
      } else if (as - (1 << ps->sendDim) == 0) {
        int foffset = (a * aggDelta) >> (ps->recvDim + 1);
        ps->recvOffset = (foffset % postFreq) * nelem;
        ps->postRecv = (ps->sendDim == 0) && ((foffset % postFreq == postFreq - 1) ||
                                              ((((foffset + 1) * 2) + 1) << ps->recvDim) >= nranks) ?
                         1 :
                         0;
        ps->stepOffset = (ps->sendDim == 0) ? 0 : foffset / postFreq;
      }
      if (s < nranks && ps->sendDim == 0 && skip) {
        // Don't forget to receive at least once even if we don't send afterwards
        ps->sendDim = -1;
        ps->sendOffset = -1;
        ps->postSend = 0;
        skip = 0;
      }
    } else if (phase == 2) {
      int s = (2 * a + 1) * scale * aggDelta;
      ps->postSend = (a % postFreq == postFreq - 1) || ((2 * (a + 1) + 1) * scale * aggDelta >= nranks) ? 1 : 0;
      ps->postRecv = 0;
      if (s >= nranks) skip = 1;
      ps->sendDim = firstBitSet(s, nrPow2);
      s -= (1 << ps->sendDim);
      ps->sendOffset = (a % postFreq) * nelem;
      ps->stepOffset = a / postFreq;
      int sendDataRank = (rank + nranks + s) % nranks;
      ps->outIx = sendDataRank * count + offset;
      ps->recvDim = s ? firstBitSet(s, nrPow2) : -1;
      if (ps->recvDim == -1) {
        ps->recvOffset = -1;
      } else {
        s -= (1 << ps->recvDim);
        int foffset = (a * 2 * scale * aggDelta) >> (ps->recvDim + 1);
        ps->recvOffset = (foffset % postFreq) * nelem;
        ps->stepOffset = foffset / postFreq;
      }
    }
    a++;
    if (a >= lastA && a >= parallelFactor) {
      int p = phase;
      if (p == 2) scale /= 2;
      phase = p == 2 ? scale ? 2 : 1 : p == 1 ? as % 2 == 1 ? 0 : 1 : 1;
      if (p == 0 || (p == 1 && as % 2 == 0)) as = nextAs();
      if (p == 0 && as == aggDelta / 2) {
        offset += chunkCount;
        if (offset >= end) {
          ps->last = 2;
        } else {
          reset();
        }
      } else {
        resetA();
      }
    } else if (phase == 0 && as == 1 && offset + chunkCount >= end &&
               a - 1 >= ((lastA - 1) / parallelFactor) * parallelFactor) {
      ps->last = 1;
    }
    int flags = PatUsed | (skip ? PatSkipped : 0);
#if defined(__HGGC_ARCH__)
    patStepFlagsStoreRelease(&ps->flags, flags);
#else
    ps->flags = flags;
#endif
  }
};

// CHUNKSIZE must be a multiple of SLICESIZE
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4)
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)
#define ALLGATHER_SLICESTEPS (NCCL_STEPS/4)
#define ALLGATHER_CHUNKSTEPS (NCCL_STEPS/2)
#define ALLTOALL_SLICESTEPS 1
#define ALLTOALL_CHUNKSTEPS 1
#define REDUCESCATTER_SLICESTEPS (NCCL_STEPS/4)
#define REDUCESCATTER_CHUNKSTEPS (NCCL_STEPS/2)
#define BROADCAST_SLICESTEPS 1
#define BROADCAST_CHUNKSTEPS 1
#define GATHER_SLICESTEPS 1
#define GATHER_CHUNKSTEPS 1
#define SCATTER_SLICESTEPS 1
#define SCATTER_CHUNKSTEPS 1
#define REDUCE_SLICESTEPS 1
#define REDUCE_CHUNKSTEPS 1
#define SENDRECV_SLICEFACTOR 4
#define NCCL_MAX_SLICE_PER_CHUNK 2

#endif
