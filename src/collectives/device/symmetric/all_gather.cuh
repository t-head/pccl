/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "symmetric.h"
#include "symmetric/kernel.cuh"
#include "symmetric/primitives.cuh"

template<int BytePerPack, int UnrollPacks, int UnrollPeers>
static __device__ void bcastDeep(
    ncclSymPrims& prim, int tn, int t, bool waitNeeded,
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
    ncclSymPrims& prim, int tn, int t, bool waitNeeded,
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
    if (dstRank == rank) {
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
          if (extraRank >=0) applyStore<BytePerPack, false, RMT_BULK_CP_C, RTMD_PA>(reinterpret_cast<uintptr_t>(add4G(outRank0, extraRank*stride4G)+u*WARP_SIZE), tmp[u]);
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
    ncclSymPrims& prim, int tn, int t,
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
    ncclSymPrims& prim, int tn, int t, bool waitNeeded, T* input, T* output, size_t nElts
  ) {
  bool inPlace = (input == output);
  // Move to rank=0
  output = prim.peerPtr(0, output);

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
__device__ __forceinline__ void pccl_symRun_AllGather_ST_impl(ncclSymDevArgs const* args) {
  ncclSymPrims prim(args->comm, args->nBlocks, ncclSymPrims_UseBarrier);
  int const& rank = prim.rank;

  // Threads numbered over rank.
  int bt = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                     prim.block, prim.nBlocks,
                     threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
  int btn = prim.nBlocks*blockDim.x;

  prim.barrierArrive(ncclCoopCta(), /*release=*/false);
  //prim.barrierWait(ncclCoopCta(), /*acquire=*/false);

  bcast<char, RingTraverseOrder>(prim, btn, bt, /*waitNeeded=*/true, (char*)args->input, (char*)args->output + rank*args->nElts, args->nElts);

  prim.barrierArrive(ncclCoopCta(), /*release=*/true);
  prim.barrierWait(ncclCoopCta(), /*acquire=*/false);
}

template<typename EltType>
static __device__ void allgather_LL_body(
    ncclSymPrims &prim, EltType* input, EltType* output, int nElts, int nPacks, int nStrideElts
  ) {
  using Pack = BytePack<8>;
  constexpr int EltPerPack = 8/sizeof(EltType);

  ncclCoopCta cta;
  int rank = prim.rank;
  int nRanks = prim.nRanks;
  constexpr int tn = ncclSymMaxThreads;
  int t = threadIdx.x;

  #pragma unroll 1
  while (0 < nElts) {
    int nIterPacks = min(nPacks, tn);
    if (t < nIterPacks) {
      Pack x = loadPack<Pack>(input, t*EltPerPack, nElts);
      prim.bcastLL(/*slot=*/nIterPacks*rank + t, x);
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
        prim.template recvLL<Unroll, Unroll>(i, Unroll, tn, /*&*/got);
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
        prim.template recvLL<1, Unroll>(i, n, tn, /*&*/got);
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
        Pack got = prim.template recvLL<Pack>(i);
        storePack(output + peer*nStrideElts, pack*EltPerPack, nElts, got);
        peer += tn_div_nPacks;
        pack += tn_mod_nPacks;
        if (nIterPacks <= pack) { peer += 1; pack -= nIterPacks; }
      }
    #endif

    prim.endLL(cta);

    input += tn*EltPerPack;
    output += tn*EltPerPack;
    nElts -= tn*EltPerPack;
    nPacks -= tn;
  }
}

static __device__ void pccl_symRun_AllGather_LL_impl(ncclSymDevArgs const* args) {
  ncclSymPrims prim(args->comm, args->nBlocks, ncclSymPrims_UseLL);
  using Pack = BytePack<8>;
  constexpr int BytePerPack = 8;
  int nElts = args->nElts;
  int nPacks = divUp(nElts, BytePerPack);

  uint32_t nPackPerBlock, nPackModBlock;
  idivmodFast32(&nPackPerBlock, &nPackModBlock, nPacks, prim.nBlocks, prim.nBlocks_rcp32);
  int blockPackBegin = prim.block*nPackPerBlock + minval<int>(prim.block, nPackModBlock);
  int blockPackEnd = blockPackBegin + nPackPerBlock + (prim.block < nPackModBlock ? 1 : 0);
  int nBlockPacks = blockPackEnd - blockPackBegin;
  int nBlockElts = nElts - blockPackBegin*BytePerPack;
  nBlockElts = min(nBlockElts, nBlockPacks*BytePerPack);
  char* blockInput = args->input + blockPackBegin*BytePerPack;
  char* blockOutput = args->output + blockPackBegin*BytePerPack;

  uint32_t lowBits = args->nElts;
  lowBits |= (uint32_t)reinterpret_cast<uintptr_t>(args->input);
  lowBits |= (uint32_t)reinterpret_cast<uintptr_t>(args->output);
  if (__builtin_expect(lowBits%8 == 0, true)) {
    // NOTE: Specializing for 8-byte alignment in one case help at size=65K: 8.9us vs 5.6us
    allgather_LL_body(prim, (BytePack<8>*)blockInput, (BytePack<8>*)blockOutput, nBlockElts/8, nBlockPacks, nElts/8);
  } else {
    allgather_LL_body(prim, blockInput, blockOutput, nBlockElts, nBlockPacks, nElts);
  }
}

__device__ __forceinline__ void pccl_symRun_AllGather_ST(ncclSymDevArgs const* args) {
  pccl_symRun_AllGather_ST_impl<false>(args);
}

__device__ __forceinline__ void pccl_symRun_AllGather_ST_Ring(ncclSymDevArgs const* args) {
  pccl_symRun_AllGather_ST_impl<true>(args);
}

__device__ __forceinline__ void pccl_symRun_AllGather_LL(ncclSymDevArgs const* args) {
  pccl_symRun_AllGather_LL_impl(args);
}
