/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "sym_kernels.h"
#include "symmetric/kernel.cuh"
#include "symmetric/primitives.cuh"

template<int BytePerPack, int UnrollPacks, int UnrollPeers>
static __device__ void bcastDeep(
    pcclSymkPrims& prim, int tn, int t, bool waitNeeded,
    char* inputHere, char* outputRank0, bool inPlace, int nIters
  ) {
  using Pack = BytePack<BytePerPack>;
  const int wn = __ppu_read_firstlane(tn/WARP_SIZE);
  const int w = __ppu_read_firstlane(t/WARP_SIZE);
  int const& rank = prim.rank;
  int const& nRanks = prim.nRanks;
  uint32_t const& stride4G = prim.stride4G;
  uint64_t inpHerePerWarp = read_firstlane_u64((uint64_t)inputHere);
  uint64_t outRank0PerWarp = read_firstlane_u64((uint64_t)outputRank0);
  Pack* inpHere = (Pack*)inpHerePerWarp + intptr_t(w)*UnrollPacks*WARP_SIZE;
  Pack* outRank0 = (Pack*)outRank0PerWarp + intptr_t(w)*UnrollPacks*WARP_SIZE;
  Pack tmp[UnrollPacks];

  nIters -= w;
  if (nIters > 0){
    #pragma unroll
    for (int u=0; u < UnrollPacks; u++) {
      applyFetch<BytePerPack, true, LOC_BULK_CP, RTMD_PA>(reinterpret_cast<uintptr_t>(inpHere+u*WARP_SIZE), tmp[u]);
    }
  }
  if (waitNeeded) prim.barrierWait(ncclCoopCta(), /*acquire=*/false);
  if (nIters <= 0) return;

  while (true) {
    int dr = inPlace ? 1 : 0;
    int r = rank + dr;
    if (r == nRanks) r = 0;
    #pragma unroll 2
    for (int partial=0; partial <= 1; partial++) {
      #pragma unroll 1
      for (int i = 0;
            partial ? i < 1 : (dr + UnrollPeers <= nRanks);
            partial ? i++ : (dr += UnrollPeers)) {
        if (partial && dr == nRanks) break; // in case UnrollPeers=4 and nRanks=4
        #pragma unroll
        for (int ur=0; ur < UnrollPeers-partial; ur++) {
          if (partial && ur!=0 && dr+ur == nRanks) break;
          #pragma unroll UnrollPacks
          for (int u=0; u < UnrollPacks; u++) {
            if (r == rank) {
              applyStore<BytePerPack, true, LOC_BULK_CP, RTMD_PA>(reinterpret_cast<uintptr_t>(add4G(outRank0, r*stride4G)+u*WARP_SIZE), tmp[u]);
            } else {
              applyStore<BytePerPack, false, RMT_BULK_CP_C, RTMD_PA>(reinterpret_cast<uintptr_t>(add4G(outRank0, r*stride4G)+u*WARP_SIZE), tmp[u]);
            }
          }
          if (++r == nRanks) r = 0;
        }
      }
    }
    inpHere += intptr_t(wn)*UnrollPacks*WARP_SIZE;
    outRank0 += intptr_t(wn)*UnrollPacks*WARP_SIZE;
    nIters -= wn;
    if (nIters <= 0) break;

    // Load data for next iteration.
    #pragma unroll
    for (int u=0; u < UnrollPacks; u++) {
      applyFetch<BytePerPack, true, LOC_BULK_CP, RTMD_PA>(reinterpret_cast<uintptr_t>(inpHere+u*WARP_SIZE), tmp[u]);
    }
  }
}

template<int BytePerPack, int UnrollPacks>
static __device__ void bcastDeepRing(
    pcclSymkPrims& prim, int tn, int t, bool waitNeeded,
    char* inputHere, char* outputRank0, bool inPlace, int nIters
  ) {
  using Pack = BytePack<BytePerPack>;
  const int wn = __ppu_read_firstlane(tn/WARP_SIZE);
  const int w = __ppu_read_firstlane(t/WARP_SIZE);
  int const& rank = prim.rank;
  int const& nRanks = prim.nRanks;
  uint32_t const& stride4G = prim.stride4G;
  Pack tmp[UnrollPacks];

  if (waitNeeded) prim.barrierWait(ncclCoopCta(), /*acquire=*/false);

  for (int dr = inPlace ? 1 : 0; dr < nRanks; dr++) {
    int dstRank = (rank+dr) % nRanks;
    uint64_t inpHerePerWarp = read_firstlane_u64((uint64_t)inputHere);
    uint64_t outRank0PerWarp = read_firstlane_u64((uint64_t)outputRank0);
    Pack* inpHere = (Pack*)inpHerePerWarp + intptr_t(w)*UnrollPacks*WARP_SIZE;
    Pack* outRank0 = (Pack*)outRank0PerWarp + intptr_t(w)*UnrollPacks*WARP_SIZE;
    int nItersPerRank = nIters - w;
    int extraRank = -1;
    if (dstRank == rank && dr+1 < nRanks) {
      dr++;
      extraRank = (rank+dr) % nRanks;
    }
    while (0 < nItersPerRank) {
      #pragma unroll
      for (int u=0; u < UnrollPacks; u++) {
        applyFetch<BytePerPack, true, LOC_BULK_CP, RTMD_PA>(reinterpret_cast<uintptr_t>(inpHere+u*WARP_SIZE), tmp[u]);
      }
      #pragma unroll UnrollPacks
      for (int u=0; u < UnrollPacks; u++) {
        if (dstRank == rank) {
          // do local and remote store
          applyStore<BytePerPack, true, LOC_BULK_CP, RTMD_PA>(reinterpret_cast<uintptr_t>(add4G(outRank0, dstRank*stride4G)+u*WARP_SIZE), tmp[u]);
          if (extraRank >= 0) applyStore<BytePerPack, false, RMT_BULK_CP_C, RTMD_PA>(reinterpret_cast<uintptr_t>(add4G(outRank0, extraRank*stride4G)+u*WARP_SIZE), tmp[u]);
        } else {
          applyStore<BytePerPack, false, RMT_BULK_CP_C, RTMD_PA>(reinterpret_cast<uintptr_t>(add4G(outRank0, dstRank*stride4G)+u*WARP_SIZE), tmp[u]);
        }
      }
      inpHere += intptr_t(wn)*UnrollPacks*WARP_SIZE;
      outRank0 += intptr_t(wn)*UnrollPacks*WARP_SIZE;
      nItersPerRank -= wn;
    }
  }
}

template<int UnrollPeers, typename T>
static __device__ void bcastEnds(
    pcclSymkPrims& prim, int tn, int t,
    T* inputHere, T* outputRank0, bool inPlace, size_t nElts, uint32_t nPreElts, size_t nSufElts
  ) {
  int const& rank = prim.rank;
  int const& nRanks = prim.nRanks;
  uint32_t const& stride4G = prim.stride4G;
  BytePack<sizeof(T)>* inpHere = (BytePack<sizeof(T)>*)inputHere;
  BytePack<sizeof(T)>* outRank0 = (BytePack<sizeof(T)>*)outputRank0;
  #pragma unroll 1
  for (size_t i = t; i < nPreElts+nSufElts; i += tn) {
    size_t elt = i < nPreElts ? i : nElts-nPreElts-nSufElts+i;
    BytePack<sizeof(T)> tmp = inpHere[elt];
    int dr = inPlace ? 1 : 0;
    int r = rank + dr;
    if (r == nRanks) r = 0;
    #pragma unroll 1
    for (; dr + UnrollPeers <= nRanks; dr += UnrollPeers) {
      #pragma unroll UnrollPeers
      for (int u=0; u < UnrollPeers; u++) {
        *add4G(outRank0+elt, r*stride4G) = tmp;
        if (++r == nRanks) r = 0;
      }
    }
    #pragma unroll UnrollPeers
    for (int u=0; u < UnrollPeers; u++) {
      if (dr+u == nRanks) break;
      *add4G(outRank0+elt, r*stride4G) = tmp;
      if (++r == nRanks) r = 0;
    }
  }
}

template<typename T, bool RingTraverseOrder>
static __device__ void bcast(
    pcclSymkPrims& prim, int tn, int t, bool waitNeeded,
    ncclSymPtr<T> inputPtr, ncclSymPtr<T> outputPtr, size_t nElts
  ) {
  T* input = inputPtr.localPtr();
  bool inPlace = (input == outputPtr.localPtr());
  // Move to rank=0
  T* output = outputPtr.lsaPtr(0);

  uintptr_t inputUptr = reinterpret_cast<uintptr_t>(input);
  uintptr_t outputUptr = reinterpret_cast<uintptr_t>(output);
  size_t nBytes = nElts*sizeof(T);

  uint32_t nPreBytes = (BULK_ALIGN_BYTES - inputUptr)%BULK_ALIGN_BYTES;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t cursor = nPreBytes;

  constexpr int MinWarpPerBlock = 4;

  if (sizeof(T) >= 4 || (sizeof(T) < 4 && (inputUptr-outputUptr)%4 == 0)) {
    {
      // for UnrollPacks*512B
      constexpr int BytePerPack = 16, UnrollPacks = 4, UnrollPeers = 4;
      constexpr int BytePerWarp = WARP_SIZE*BytePerPack*UnrollPacks;
      uint32_t chunks = (nBytes-cursor)/BytePerWarp;
      if (chunks != 0) {
        uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerWarp;
        if (!RingTraverseOrder) {
          bcastDeep<(sizeof(T) <= BytePerPack ? BytePerPack : 0), UnrollPacks, UnrollPeers>(
            prim, tn, t, waitNeeded, (char*)input + cursor, (char*)output + cursor, inPlace,
            chunks
          );
        } else {
          bcastDeepRing<(sizeof(T) <= BytePerPack ? BytePerPack : 0), UnrollPacks>(
            prim, tn, t, waitNeeded, (char*)input + cursor, (char*)output + cursor, inPlace,
            chunks
          );
        }
        cursor = cursorAfter;
        waitNeeded = false;
      }
    }
    {
      // for 1*512B
      constexpr int BytePerPack = 16, UnrollPacks = 1, UnrollPeers = 8;
      constexpr int BytePerWarp = WARP_SIZE*BytePerPack*UnrollPacks;
      uint32_t chunks = (nBytes-cursor)/BytePerWarp;
      if (chunks != 0) {
        uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerWarp;
        bcastDeep<(sizeof(T) <= BytePerPack ? BytePerPack : 0), UnrollPacks, UnrollPeers>(
          prim, tn, t, waitNeeded, (char*)input + cursor, (char*)output + cursor, inPlace,
          chunks
        );
        cursor = cursorAfter;
        waitNeeded = false;
      }
    }
  }

  if (waitNeeded) prim.barrierWait(ncclCoopCta(), /*acquire=*/false);

  constexpr int UnrollPeers = 8;
  size_t nSufElts = (nBytes-cursor)/sizeof(T);
  bcastEnds<UnrollPeers>(prim, tn, t, input, output, inPlace, nElts, nPreBytes/sizeof(T), nSufElts);
}

template<bool RingTraverseOrder>
__device__ __forceinline__ void ncclSymkRun_AllGather_ST_impl(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x};
  pcclSymkPrims prim(handler.comm, bar);
  int rank = handler.comm.rank;
  bar.arrive(ncclCoopCta(), hggc::memory_order_relaxed);
  bool waitNeeded = true;
  handler.forEachWork<char>([&](int block, int nBlocks, size_t nElts, size_t nAllElts,
                               ncclSymPtr<char> input, ncclSymPtr<char> output) {
    // Threads numbered over rank.
    int bt = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                       block, nBlocks, threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
    int btn = nBlocks*blockDim.x;
    bcast<char, RingTraverseOrder>(prim, btn, bt, waitNeeded, input, output + rank*nAllElts, nElts);
    waitNeeded = false;
  });
  bar.sync(ncclCoopCta(), hggc::memory_order_release);
}

__device__ __forceinline__ void ncclSymkRun_AllGather_ST(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllGather_ST_impl<false>(args);
}

__device__ __forceinline__ void ncclSymkRun_AllGather_ST_Ring(ncclSymkDevWorkArgs const* args) {
  ncclSymkRun_AllGather_ST_impl<true>(args);
}

template<typename EltType>
static __device__ void allgather_LL_body(
    ncclSymkArgsHandler& handler, pcclSymkLLa2ASession& lla2a,
    EltType* input, EltType* output, int nElts, int nPacks, int nStrideElts
  ) {
  using Pack = BytePack<8>;
  constexpr int EltPerPack = 8/sizeof(EltType);

  ncclCoopCta cta;
  int rank = handler.comm.rank;
  int nRanks = handler.comm.nRanks;
  constexpr int tn = ncclSymkMaxThreads;
  int t = threadIdx.x;

  #pragma unroll 1
  while (0 < nElts) {
    int nIterPacks = min(nPacks, tn);
    if (t < nIterPacks) {
      Pack x = loadPack<Pack>(input, t*EltPerPack, nElts);
      lla2a.bcast(nIterPacks*rank + t, x);
    }

    int tn_div_nPacks = tn/nIterPacks;
    int tn_mod_nPacks = tn%nIterPacks;
    int peer = t/nIterPacks;
    int pack = t%nIterPacks;
    #if 1
      // NOTE: Unrolling speedup on eos nranks=8 size=64K: 5.7us vs 6.7us
      constexpr int Unroll = 4;
      #pragma unroll 1
      for (int i = t; i < (nRanks*nIterPacks & -(Unroll*tn)); i += Unroll*tn) {
        Pack got[Unroll];
        lla2a.template recvUnrolled<Unroll, Unroll>(i, Unroll, tn, got);
        #pragma unroll
        for (int u=0; u < Unroll; u++) {
          storePack<Pack>(output + peer*nStrideElts, pack*EltPerPack, nElts, got[u]);
          peer += tn_div_nPacks;
          pack += tn_mod_nPacks;
          if (nIterPacks <= pack) { peer += 1; pack -= nIterPacks; }
        }
      }

      int i = (nRanks*nIterPacks & -(Unroll*tn)) + t;
      int n = (nRanks*nIterPacks)/tn % Unroll;
      if (i + n*tn < nRanks*nIterPacks) n += 1;
      if (n != 0) {
        Pack got[Unroll];
        lla2a.template recvUnrolled<1, Unroll>(i, n, tn, got);
        #pragma unroll
        for (int u=0; u < Unroll; u++) {
          if (u != 0 && u == n) break;
          storePack(output + peer*nStrideElts, pack*EltPerPack, nElts, got[u]);
          peer += tn_div_nPacks;
          pack += tn_mod_nPacks;
          if (nIterPacks <= pack) { peer += 1; pack -= nIterPacks; }
        }
      }
    #else
      // The non-unrolled but "obviously correct" implementation for reference.
      #pragma unroll 1
      for (int i = t; i < nRanks*nIterPacks; i += tn) {
        Pack got = lla2a.template recv<Pack>(i);
        storePack(output + peer*nStrideElts, pack*EltPerPack, nElts, got);
        peer += tn_div_nPacks;
        pack += tn_mod_nPacks;
        if (nIterPacks <= pack) { peer += 1; pack -= nIterPacks; }
      }
    #endif

    lla2a.endEpoch(cta);

    input += tn*EltPerPack;
    output += tn*EltPerPack;
    nElts -= tn*EltPerPack;
    nPacks -= tn;
  }
}

__device__ __forceinline__ void ncclSymkRun_AllGather_LL(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  pcclSymkLLa2ASession lla2a(handler.comm, handler.lsaLLA2A, blockIdx.x);
  handler.singleWork<char>([&](size_t nElts, size_t nAllElts,
                               ncclSymPtr<char> inputPtr, ncclSymPtr<char> outputPtr) {
    char* input = inputPtr.localPtr();
    char* output = outputPtr.localPtr();
    int nPacks = divUp(nElts, 8);
    uintptr_t lowBits = nAllElts | reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(output);
    if (__builtin_expect(lowBits%8 == 0, true)) {
      allgather_LL_body(handler, lla2a, (BytePack<8>*)input, (BytePack<8>*)output,
                        nPacks, nPacks, int(nAllElts/8));
    } else {
      allgather_LL_body(handler, lla2a, input, output, int(nElts), nPacks, int(nAllElts));
    }
  });
}

// 2D (hierarchical) AllGather: view the ranks as a grid (g = rank/groupSize,
// lr = rank%groupSize) and allgather along each dimension. Phase 1 broadcasts a
// rank's own block to its group mates, ring-staggered; phase 2 exchanges whole
// group regions, with a partner that stays fixed for all of a group's sends.
// Bytes moved per rank match the flat ring, but the ring's nRanks-way rotating
// write contention becomes two phases with at most groupSize writers per target.

// Warp-collective bulk copy to a peer, grid-strided over chunks. With AlsoLocal
// the fetch also lands in `locBase`, which lets the out-of-place self copy ride
// along with the first remote send instead of costing a whole extra pass;
// `locBase` must be a valid pointer either way. Every lane of a warp must present
// the same base address, hence read_firstlane_u64.
template<int BytePerPack, int UnrollPacks, bool AlsoLocal>
static __device__ void copyDeep(
    int tn, int t, char* srcBase, char* locBase, char* dstBase, size_t nBytes
  ) {
  using Pack = BytePack<BytePerPack>;
  const int wn = __ppu_read_firstlane(tn/WARP_SIZE);
  const int w  = __ppu_read_firstlane(t/WARP_SIZE);
  constexpr size_t BytePerWarp = (size_t)WARP_SIZE*BytePerPack*UnrollPacks;
  size_t chunks = nBytes / BytePerWarp;
  if (chunks == 0) return;
  uint64_t srcPW = read_firstlane_u64((uint64_t)srcBase);
  uint64_t locPW = read_firstlane_u64((uint64_t)locBase);
  uint64_t dstPW = read_firstlane_u64((uint64_t)dstBase);
  Pack* s = (Pack*)srcPW + (intptr_t)w*UnrollPacks*WARP_SIZE;
  Pack* l = (Pack*)locPW + (intptr_t)w*UnrollPacks*WARP_SIZE;
  Pack* d = (Pack*)dstPW + (intptr_t)w*UnrollPacks*WARP_SIZE;
  for (size_t it = w; it < chunks; it += wn) {
    Pack tmp[UnrollPacks];
    #pragma unroll
    for (int u=0; u < UnrollPacks; u++)
      applyFetch<BytePerPack, true, LOC_BULK_CP, RTMD_PA>(reinterpret_cast<uintptr_t>(s+u*WARP_SIZE), tmp[u]);
    #pragma unroll
    for (int u=0; u < UnrollPacks; u++) {
      if (AlsoLocal)
        applyStore<BytePerPack, true, LOC_BULK_CP, RTMD_PA>(reinterpret_cast<uintptr_t>(l+u*WARP_SIZE), tmp[u]);
      applyStore<BytePerPack, false, RMT_BULK_CP_C, RTMD_PA>(reinterpret_cast<uintptr_t>(d+u*WARP_SIZE), tmp[u]);
    }
    s += (intptr_t)wn*UnrollPacks*WARP_SIZE;
    d += (intptr_t)wn*UnrollPacks*WARP_SIZE;
    if (AlsoLocal) l += (intptr_t)wn*UnrollPacks*WARP_SIZE;
  }
}

// End of the 2KB-per-warp region, then of the 512B one.
static __device__ void slotChunkEnds(size_t nBytes, size_t* bigEnd, size_t* smallEnd) {
  constexpr size_t Big = (size_t)WARP_SIZE*16*4;
  constexpr size_t Small = (size_t)WARP_SIZE*16;
  *bigEnd = (nBytes/Big)*Big;
  *smallEnd = *bigEnd + ((nBytes - *bigEnd)/Small)*Small;
}

// Copy one slot to a peer: 2KB per warp, then 512B, then the sub-512B tail per lane.
template<bool AlsoLocal>
static __device__ void bulkCopy(
    int tn, int t, char* src, char* loc, char* dst, size_t nBytes
  ) {
  size_t bigEnd, smallEnd;
  slotChunkEnds(nBytes, &bigEnd, &smallEnd);
  copyDeep<16, 4, AlsoLocal>(tn, t, src, loc, dst, bigEnd);
  copyDeep<16, 1, AlsoLocal>(tn, t, src+bigEnd, loc+bigEnd, dst+bigEnd, smallEnd-bigEnd);
  for (size_t i = smallEnd + t; i < nBytes; i += tn) {
    char v = src[i];
    if (AlsoLocal) loc[i] = v;
    dst[i] = v;
  }
}

__device__ __forceinline__ void ncclSymkRun_AllGather_ST_2D(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x};
  pcclSymkPrims prim(handler.comm, bar);
  int const& rank = prim.rank;
  int const& nRanks = prim.nRanks;

  handler.forEachWork<char>([&](int block, int nBlocks, size_t nElts, size_t nAllElts,
                               ncclSymPtr<char> inputPtr, ncclSymPtr<char> outputPtr) {
    int bt = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                       block, nBlocks,
                       threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
    int btn = nBlocks*blockDim.x;

    size_t nBytes = nElts; // per-rank slice size in bytes
    char* input = inputPtr.localPtr();
    char* output = outputPtr.localPtr();

    int groupSize = ncclSymkAg2DGroupSize;
    int nGroups = nRanks / groupSize;
    int groupId = rank / groupSize;
    int groupBase = groupId * groupSize;
    int lr = rank - groupBase; // local rank within group

    // Slot bases use the full rank stride. Registered windows keep these low bits
    // identical on every rank; unregistered input must use the flat protocol.
    bool bulkAligned = inputPtr.window != nullptr &&
                       (((reinterpret_cast<uintptr_t>(output) | reinterpret_cast<uintptr_t>(input)
                          | nAllElts) % BULK_ALIGN_BYTES) == 0);
    char* selfSlot = output + (size_t)rank*nAllElts;

    // Make the previous kernel's remote writes visible before overwriting output.
    bar.arrive(ncclCoopCta(), hggc::memory_order_relaxed);
    bar.wait(ncclCoopCta(), hggc::memory_order_acquire);

    if (!bulkAligned) {
      bcast<char, false>(prim, btn, bt, /*waitNeeded=*/false, inputPtr, outputPtr + rank*nAllElts, nBytes);
      bar.arrive(ncclCoopCta(), hggc::memory_order_release);
      bar.wait(ncclCoopCta(), hggc::memory_order_relaxed);
      return;
    }

    // Phase 1: intra-group allgather of this rank's own block. Out of place, the
    // copy into our own slot is fused into the first remote send (one fetch, two
    // stores) so it costs no extra pass -- which relies on there being a remote send
    // to ride along with.
    static_assert(ncclSymkAg2DGroupSize >= 2, "phase 1 needs at least one group mate");
    bool selfPending = (input != selfSlot);
    for (int dr = 1; dr < groupSize; dr++) {
      int dstRank = groupBase + (lr + dr) % groupSize;
      char* rmtSlot = outputPtr.lsaPtr(dstRank) + (size_t)rank*nAllElts;
      if (selfPending) {
        bulkCopy<true>(btn, bt, input, selfSlot, rmtSlot, nBytes);
        selfPending = false;
      } else {
        bulkCopy<false>(btn, bt, input, selfSlot, rmtSlot, nBytes);
      }
    }

    // Phase 2 only needs my group mates' phase-1 writes into MY output, so an
    // in-group barrier suffices. My phase-2 writes land in the partner's output at my
    // group's block range, disjoint from what the partner itself touches, so the
    // groups may run out of step.
    prim.groupBarrierArrive(ncclCoopCta(), groupBase, groupSize, /*release=*/true);
    prim.groupBarrierWait(ncclCoopCta(), groupBase, groupSize, /*acquire=*/true);

    // Phase 2: one outer step per other group, target staggered by groupId so
    // every group sends to a distinct group instead of piling onto the lowest one.
    // The partner is fixed for all groupSize sends, which keeps the pacing flat.
    for (int gi = 1; gi < nGroups; gi++) {
      int g = (groupId + gi) % nGroups;
      int partner = g * groupSize + lr;
      for (int k = 0; k < groupSize; k++) {
        size_t slotOff = (size_t)(groupBase + k)*nAllElts;
        char* slot = output + slotOff;
        bulkCopy<false>(btn, bt, slot, slot, outputPtr.lsaPtr(partner) + slotOff, nBytes);
      }
    }

    bar.arrive(ncclCoopCta(), hggc::memory_order_release);
    bar.wait(ncclCoopCta(), hggc::memory_order_relaxed);
  });
}
