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

template<int BytePerPack, int UnrollPacks, int UnrollPeers, typename T, typename Red>
static __device__ void reduceDeep(
    ncclSymPrims& prim, int tn, int t, bool waitNeeded,
    Red red, char* inputRank0, char* outputHere, int32_t nIters
  ) {
  using Pack = BytePack<BytePerPack>;
  using Acc = typename Red::EltType;
  using AccPack = BytePack<BytePerPack*sizeof(Acc)/sizeof(T)>;

  const int wn = __ppu_read_firstlane(tn/WARP_SIZE);
  const int w = __ppu_read_firstlane(t/WARP_SIZE);
  int const& rank = prim.rank;
  int const& nRanks = prim.nRanks;
  uint32_t const& stride4G = prim.stride4G;

  uint64_t inputRank0PerWarp = read_firstlane_u64((uint64_t)inputRank0);
  uint64_t outHerePerWarp = read_firstlane_u64((uint64_t)outputHere);
  Pack* inpRank0 = (Pack*)inputRank0PerWarp + intptr_t(w)*UnrollPacks*WARP_SIZE;
  Pack* outHere = (Pack*)outHerePerWarp + intptr_t(w)*UnrollPacks*WARP_SIZE;
  Pack acc0[UnrollPacks];


  nIters -= w;

  if (nIters > 0) {
    #pragma unroll
    for (int u=0; u < UnrollPacks; u++) {
      applyFetch<BytePerPack, true, LOC_BULK_CP, RTMD_PA>(reinterpret_cast<uintptr_t>(add4G(inpRank0, rank*stride4G)+u*WARP_SIZE), acc0[u]);
    }
  }
  if (waitNeeded) prim.barrierWait(ncclCoopCta(), /*acquire=*/false);
  if (nIters <= 0) return;

  while (true) {
    AccPack acc1[UnrollPacks];
    int r = rank+1;
    if (r == nRanks) r = 0;
    {
      Pack tmp1[UnrollPacks];
      #pragma unroll
      for (int u=0; u < UnrollPacks; u++) {
        applyFetch<BytePerPack, false, RMT_BULK_CP_C, RTMD_PA>(reinterpret_cast<uintptr_t>(add4G(inpRank0, r*stride4G)+u*WARP_SIZE), tmp1[u]);
      }
      #pragma unroll
      for (int u=0; u < UnrollPacks; u++) {
        acc1[u] = applyReduce(red, applyCast<T, Acc>(acc0[u]), applyCast<T, Acc>(tmp1[u]));
      }
    }

    r += 1;
    if (r == nRanks) r = 0;

    int dr = 2;
    #pragma unroll 2
    for (int partial=0; partial <= 1; partial++) {
      #pragma unroll 1
      for (int i = 0;
            partial ? i < 1 : (dr + UnrollPeers <= nRanks);
            partial ? i++ : (dr += UnrollPeers)) {
        if (partial && dr == nRanks) break;

        Pack tmp1[UnrollPeers][UnrollPacks];
        #pragma unroll
        for (int ur=0; ur < UnrollPeers-partial; ur++) {
          if (partial && ur!=0 && dr+ur == nRanks) break;
          #pragma unroll UnrollPacks
          for (int u=0; u < UnrollPacks; u++) {
            applyFetch<BytePerPack, false, RMT_BULK_CP_C, RTMD_PA>(reinterpret_cast<uintptr_t>(add4G(inpRank0, r*stride4G)+u*WARP_SIZE), tmp1[ur][u]);
          }
          r += 1;
          if (r == nRanks) r = 0;
        }
        #pragma unroll
        for (int ur=0; ur < UnrollPeers-partial; ur++) {
          if (partial && ur!=0 && dr+ur == nRanks) break;
          #pragma unroll UnrollPacks
          for (int u=0; u < UnrollPacks; u++) {
            acc1[u] = applyReduce(red, acc1[u], applyCast<T, Acc>(tmp1[ur][u]));
          }
        }
      }
    }

    #pragma unroll
    for (int u=0; u < UnrollPacks; u++) acc0[u] = applyCast<Acc, T>(acc1[u]);

    #pragma unroll UnrollPacks
    for (int u=0; u < UnrollPacks; u++) {
      applyStore<BytePerPack, true, LOC_BULK_CP, RTMD_PA>(reinterpret_cast<uintptr_t>(outHere+u*WARP_SIZE), acc0[u]);
    }

    inpRank0 += intptr_t(wn)*UnrollPacks*WARP_SIZE;
    outHere += intptr_t(wn)*UnrollPacks*WARP_SIZE;
    nIters -= wn;
    if (nIters <= 0) break;

    // Load data for next iteration.
    #pragma unroll
    for (int u=0; u < UnrollPacks; u++) {
      applyFetch<BytePerPack, true, LOC_BULK_CP, RTMD_PA>(reinterpret_cast<uintptr_t>(add4G(inpRank0, rank*stride4G)+u*WARP_SIZE), acc0[u]);
    }
  }
}

template<int UnrollPeers, typename Red, typename T>
static __device__ void reduceEnds(
    ncclSymPrims& prim, int tn, int t, Red red,
    T* inputRank0, T* outputHere, size_t nElts, uint32_t nPreElts, size_t nSufElts
  ) {
  using Acc = typename Red::EltType;

  int const& rank = prim.rank;
  int const& nRanks = prim.nRanks;
  uint32_t const& stride4G = prim.stride4G;
  BytePack<sizeof(T)>* inpRank0 = (BytePack<sizeof(T)>*)inputRank0;
  BytePack<sizeof(T)>* outHere = (BytePack<sizeof(T)>*)outputHere;
  #pragma unroll 1
  for (size_t i = t; i < nPreElts+nSufElts; i += tn) {
    size_t elt = i < nPreElts ? i : nElts-nSufElts-nPreElts+i;
    BytePack<sizeof(T)> acc0 = *add4G(inpRank0+elt, rank*stride4G);
    BytePack<sizeof(Acc)> acc1;
    BytePack<sizeof(T)> tmp[UnrollPeers];
    int dr = 1;
    int r = rank+1;
    if (nRanks == r) r = 0;
    bool first = true;

    #pragma unroll 2
    for (int partial=0; partial <= 1; partial++) {
      #pragma unroll 1
      for (int j = 0;
           partial ? j < 1 : (dr + UnrollPeers <= nRanks);
           partial ? j++ : (dr += UnrollPeers)) {
        if (partial && dr == nRanks) break;

        #pragma unroll
        for (int u=0; u < UnrollPeers-partial; u++) {
          if (partial && u!=0 && dr+u == nRanks) break;
          tmp[u] = *add4G(inpRank0+elt, r*stride4G);
          r += 1;
          if (r == nRanks) r = 0;
        }
        if (first) {
          first = false;
          acc1 = applyCast<T, Acc>(acc0);
        }
        #pragma unroll
        for (int u=0; u < UnrollPeers-partial; u++) {
          if (partial && u!=0 && dr+u == nRanks) break;
          acc1 = applyReduce(red, acc1, applyCast<T, Acc>(tmp[u]));
        }
      }
    }

    acc0 = applyCast<Acc, T>(acc1);
    outHere[elt] = acc0;
  }
}

template<typename Red, typename T, bool incPrecision>
static __device__ void reduce(
    ncclSymPrims& prim, int tn, int t, bool waitNeeded,
    Red red, T* input, T* output, size_t nElts
  ) {
  int nRanks = prim.nRanks;
  int nBlocks = prim.nBlocks;
  // Move input to rank=0
  input = prim.peerPtr(0, input);

  uintptr_t inputUptr = reinterpret_cast<uintptr_t>(input);
  uintptr_t outputUptr = reinterpret_cast<uintptr_t>(output);
  uint32_t alignment = uint32_t(inputUptr - outputUptr);
  size_t nBytes = nElts*sizeof(T);

  uint32_t nPreBytes = (BULK_ALIGN_BYTES - inputUptr)%BULK_ALIGN_BYTES;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t cursor = nPreBytes;

  if (sizeof(T) >= 4 || (sizeof(T) < 4 && alignment%4 == 0)) {
    {
      // for UnrollPacks*512B
      constexpr int BytePerPack = 16, UnrollPacks = (sizeof(T) < 2 || (incPrecision && sizeof(T) < 4)) ? 4 : 8;
      constexpr int UnrollPeers = (sizeof(T) < 2 || (incPrecision && sizeof(T) < 4)) ? 4 : 2;
      constexpr int BytePerWarp = WARP_SIZE*BytePerPack*UnrollPacks;
      uint32_t chunks = (nBytes-cursor)/BytePerWarp;
      if (chunks != 0) {
        uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerWarp;
        reduceDeep<(sizeof(T) <= BytePerPack ? BytePerPack : 0), UnrollPacks, UnrollPeers, T>(
          prim, tn, t, waitNeeded, red,
          (char*)input + cursor, (char*)output + cursor,
          chunks
        );
        cursor = cursorAfter;
        waitNeeded = false;
      }
    }
    {
      // for 512B
      constexpr int BytePerPack = 16, UnrollPacks = 1, UnrollPeers = 8;
      constexpr int BytePerWarp = WARP_SIZE*BytePerPack*UnrollPacks;
      uint32_t chunks = (nBytes-cursor)/BytePerWarp;
      if (chunks != 0) {
        uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerWarp;
        reduceDeep<(sizeof(T) <= BytePerPack ? BytePerPack : 0), UnrollPacks, UnrollPeers, T>(
          prim, tn, t, waitNeeded, red,
          (char*)input + cursor, (char*)output + cursor,
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
  reduceEnds<UnrollPeers>(prim, tn, t, red, input, output, nElts, nPreBytes/sizeof(T), nSufElts);
}


template<template<typename> class Red, typename T>
__device__ __forceinline__ void pccl_symRun_ReduceScatter_LD(ncclSymDevArgs const* args) {
  ncclSymPrims prim(args->comm, args->nBlocks, ncclSymPrims_UseBarrier);

  // Round robin warps over blocks.
  int t = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                    prim.block, prim.nBlocks,
                    threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
  int tn = prim.nBlocks*blockDim.x;

  prim.barrierArrive(ncclCoopCta(), /*release=*/false);
  //prim.barrierWait(ncclCoopCta(), /*acquire=*/false);
  if (args->incPrecision) {
    using Acc = typename ncclSymAccumType<Red, T, true>::Type;
    Red<Acc> red(args->redOpArg);
    reduce<Red<Acc>, T, true>(prim, tn, t, /*waitNeeded=*/true, red, (T*)args->input + prim.rank*args->nElts, (T*)args->output, args->nElts);
  } else {
    using Acc = typename ncclSymAccumType<Red, T, false>::Type;
    Red<Acc> red(args->redOpArg);
    reduce<Red<Acc>, T, false>(prim, tn, t, /*waitNeeded=*/true, red, (T*)args->input + prim.rank*args->nElts, (T*)args->output, args->nElts);
  }

  prim.barrierArrive(ncclCoopCta(), /*release=*/false);
  prim.barrierWait(ncclCoopCta(), /*acquire=*/false);
}

// T is user type, EltType is the most aligned type
template<typename T, typename Red, typename EltType>
__device__ __forceinline__ void pccl_symRun_ReduceScatter_LL_body(
    ncclSymPrims &prim, Red red, EltType* input, EltType* output, int nElts, int nPacks, int nStrideElts) {
  using Pack = BytePack<8>;
  constexpr int EltPerPack = 8/sizeof(EltType);

  int nRanks = prim.nRanks;
  int rank = prim.rank;
  int t = threadIdx.x;
  int tn = ncclSymMaxThreads;
  ncclCoopCta cta;

  #pragma unroll 1
  while (0 < nElts) {
    int nIterPacks = min(nPacks, tn);
    int tn_div_nPacks = tn/nIterPacks;
    int tn_mod_nPacks = tn%nIterPacks;
    int peer = t/nIterPacks;
    int pack = t%nIterPacks;

    #pragma unroll 1
    for (int i = t; i < nRanks*nIterPacks; i += tn) {
      Pack got = loadPack<Pack>(input + peer*nStrideElts, pack*EltPerPack, nElts);
      prim.sendLL(peer, rank*nIterPacks + pack, got);
      peer += tn_div_nPacks;
      pack += tn_mod_nPacks;
      if (nIterPacks <= pack) { peer += 1; pack -= nIterPacks; }
    }

    if (t < nIterPacks) {
      Pack got = prim.template recvReduceLL<Pack, T>(t, nIterPacks, red);
      storePack(output, t*EltPerPack, nElts, got);
    }
    prim.endLL(cta);

    input += tn*EltPerPack;
    output += tn*EltPerPack;
    nElts -= tn*EltPerPack;
    nPacks -= tn;
  }
}
template<template<typename> class Red, typename T>
__device__ __forceinline__ void pccl_symRun_ReduceScatter_LL(ncclSymDevArgs const* args) {
  ncclSymPrims prim(args->comm, args->nBlocks, ncclSymPrims_UseLL);
  Red<typename ncclSymAccumType<Red, T>::Type> red(args->redOpArg);

  using Pack = BytePack<8>;
  constexpr int EltPerPack = 8/sizeof(T);
  int nAllElts = args->nElts;
  int nAllPacks = divUp(nAllElts, EltPerPack);
  uint32_t nPackPerBlock, nPackModBlock;
  idivmodFast32(&nPackPerBlock, &nPackModBlock, nAllPacks, prim.nBlocks, prim.nBlocks_rcp32);
  int blockPackBegin = prim.block*nPackPerBlock + minval<int>(prim.block, nPackModBlock);
  int blockPackEnd = blockPackBegin + nPackPerBlock + (prim.block < nPackModBlock ? 1 : 0);
  int nPacks = blockPackEnd - blockPackBegin;
  int nElts = nAllElts - blockPackBegin*EltPerPack;
  nElts = min(nElts, nPacks*EltPerPack);
  T* input = (T*)args->input + blockPackBegin*EltPerPack;
  T* output = (T*)args->output + blockPackBegin*EltPerPack;

  uint32_t lowBits = args->nElts*sizeof(T);
  lowBits |= (uint32_t)reinterpret_cast<uintptr_t>(args->input);
  lowBits |= (uint32_t)reinterpret_cast<uintptr_t>(args->output);
  if (__builtin_expect(lowBits%8 == 0, true)) {
    pccl_symRun_ReduceScatter_LL_body<T>(prim, red, (Pack*)input, (Pack*)output, nPacks, nPacks, nAllElts/EltPerPack);
  } else {
    pccl_symRun_ReduceScatter_LL_body<T>(prim, red, input, output, nElts, nPacks, nAllElts);
  }
}
