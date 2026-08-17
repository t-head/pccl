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

template<int BytePerPack, int UnrollPacks, int UnrollPeers, typename T, class Red>
static __device__ __forceinline__ void allreduceDeep(
    ncclSymPrims& prim, int tn, int t, bool waitNeeded,
    Red red, char* inputRank0, char* outputRank0, int32_t nIters
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
  uint64_t outputRank0PerWarp = read_firstlane_u64((uint64_t)outputRank0);
  Pack* inpRank0 = (Pack*)inputRank0PerWarp + intptr_t(w)*UnrollPacks*WARP_SIZE;
  Pack* outRank0 = (Pack*)outputRank0PerWarp + intptr_t(w)*UnrollPacks*WARP_SIZE;
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
    int r = rank;
    if (++r == nRanks) r = 0;
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

    if (++r == nRanks) r = 0;

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
          if (++r == nRanks) r = 0;
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

    dr = 0;
    r = rank;
    #pragma unroll 2
    for (int partial=0; partial <= 1; partial++) {
      #pragma unroll 1
      for (int i = 0;
            partial ? i < 1 : (dr + UnrollPeers <= nRanks);
            partial ? i++ : (dr += UnrollPeers)) {
        if (partial && dr == nRanks) break;

        #pragma unroll
        for (int ur=0; ur < UnrollPeers-partial; ur++) {
          if (partial && ur!=0 && dr+ur == nRanks) break;
          #pragma unroll UnrollPacks
          for (int u=0; u < UnrollPacks; u++) {
            if (r == rank) {
              applyStore<BytePerPack, true, LOC_BULK_CP, RTMD_PA>(reinterpret_cast<uintptr_t>(add4G(outRank0, r*stride4G)+u*WARP_SIZE), acc0[u]);
            } else {
              applyStore<BytePerPack, false, RMT_BULK_CP_C, RTMD_PA>(reinterpret_cast<uintptr_t>(add4G(outRank0, r*stride4G)+u*WARP_SIZE), acc0[u]);
            }
          }
          if (++r == nRanks) r = 0;
        }
      }
    }

    inpRank0 += intptr_t(wn)*UnrollPacks*WARP_SIZE;
    outRank0 += intptr_t(wn)*UnrollPacks*WARP_SIZE;
    nIters -= wn;
    if (nIters <= 0) break;

    // Load data for next iteration.
    #pragma unroll
    for (int u=0; u < UnrollPacks; u++) {
      applyFetch<BytePerPack, true, LOC_BULK_CP, RTMD_PA>(reinterpret_cast<uintptr_t>(add4G(inpRank0, rank*stride4G)+u*WARP_SIZE), acc0[u]);
    }
  }
}

template<int UnrollPeers, class Red, typename T>
static __device__ __forceinline__ void allreduceEnds(
    ncclSymPrims& prim, int tn, int t, Red red,
    T* inputRank0, T* outputRank0, size_t nElts, uint32_t nPreElts, size_t nSufElts
  ) {
  using Acc = typename Red::EltType;
  int const& rank = prim.rank;
  int const& nRanks = prim.nRanks;
  uint32_t const& stride4G = prim.stride4G;
  BytePack<sizeof(T)>* inpRank0 = (BytePack<sizeof(T)>*)inputRank0;
  BytePack<sizeof(T)>* outRank0 = (BytePack<sizeof(T)>*)outputRank0;

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
    dr = 0;
    r = rank;
    #pragma unroll 2
    for (int partial=0; partial <= 1; partial++) {
      #pragma unroll 1
      for (int j=0;
           partial ? j < 1 : (dr + UnrollPeers <= nRanks);
           partial ? j++ : (dr += UnrollPeers)) {
        #pragma unroll
        for (int u=0; u < UnrollPeers-partial; u++) {
          if (partial && dr+u == nRanks) break;
          *add4G(outRank0+elt, r*stride4G) = acc0;
          r += 1;
          if (r == nRanks) r = 0;
        }
      }
    }
  }
}

template<class Red, typename T, bool incPrecision>
static __device__ void allreduce(
    ncclSymPrims& prim, int tn, int t, bool waitNeeded,
    Red red, T* input, T* output, size_t nElts
  ) {
  int nRanks = prim.nRanks;
  int nBlocks = prim.nBlocks;
  // Move to rank=0
  input = prim.peerPtr(0, input);
  output = prim.peerPtr(0, output);

  uintptr_t inputUptr = reinterpret_cast<uintptr_t>(input);
  uintptr_t outputUptr = reinterpret_cast<uintptr_t>(output);
  size_t nBytes = nElts*sizeof(T);

  uint32_t nPreBytes = (BULK_ALIGN_BYTES - inputUptr)%BULK_ALIGN_BYTES;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t cursor = nPreBytes;

  if (sizeof(T) >= 4 || (sizeof(T) < 4 && (inputUptr-outputUptr)%4 == 0)) {
    {
      // for UnrollPacks*512B
      constexpr int BytePerPack = 16, UnrollPacks = 4, UnrollPeers = 4;
      constexpr int BytePerWarp = WARP_SIZE*BytePerPack*UnrollPacks;
      uint32_t chunks = (nBytes-cursor)/BytePerWarp;
      if (chunks != 0) {
        uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerWarp;
        allreduceDeep<(sizeof(T) <= BytePerPack ? BytePerPack : 0), UnrollPacks, UnrollPeers, T>(
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
        allreduceDeep<(sizeof(T) <= BytePerPack ? BytePerPack : 0), UnrollPacks, UnrollPeers, T>(
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
  allreduceEnds<UnrollPeers>(prim, tn, t, red, input, output, nElts, nPreBytes/sizeof(T), nSufElts);
}

template<template<typename> class Red, typename T>
__device__ __forceinline__ void pccl_symRun_AllReduce_RSxLD_AGxST(ncclSymDevArgs const* args) {
  ncclSymPrims prim(args->comm, args->nBlocks, ncclSymPrims_UseBarrier);
  int /*const&*/ rank = prim.rank;
  int /*const&*/ nRanks = prim.nRanks;

  // Threads numbered globally such that we round robin warps by rank then block.
  int gt = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                     rank, nRanks,
                     prim.block, prim.nBlocks,
                     threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
  int gtn = nRanks*prim.nBlocks*blockDim.x;

  prim.barrierArrive(ncclCoopCta(), /*release=*/false);
  //prim.barrierWait(ncclCoopCta(), /*acquire=*/false);
  if (args->incPrecision) {
    using Acc = typename ncclSymAccumType<Red, T, true>::Type;
    Red<Acc> red(args->redOpArg);
    allreduce<Red<Acc>, T, true>(prim, gtn, gt, /*waitNeeded=*/true, red, (T*)args->input, (T*)args->output, args->nElts);
  } else {
    using Acc = typename ncclSymAccumType<Red, T, false>::Type;
    Red<Acc> red(args->redOpArg);
    allreduce<Red<Acc>, T, false>(prim, gtn, gt, /*waitNeeded=*/true, red, (T*)args->input, (T*)args->output, args->nElts);
  }

  prim.barrierArrive(ncclCoopCta(), /*release=*/true);
  prim.barrierWait(ncclCoopCta(), /*acquire=*/false);
}

template<template<typename> class Red, typename T>
__device__ __forceinline__ void pccl_symRun_AllReduce_AGxLL_R_impl(ncclSymDevArgs const* args) {
  ncclSymPrims prim(args->comm, args->nBlocks, ncclSymPrims_UseLL);
  int /*const&*/ rank = prim.rank;
  using Acc = typename ncclSymAccumType<Red, T>::Type;
  Red<Acc> red(args->redOpArg);

  using Pack = BytePack<8>;
  using AccPack = BytePack<8*sizeof(Acc)/sizeof(T)>;
  constexpr int EltPerPack = 8/sizeof(T);
  int nElts = args->nElts;
  int nPacks = divUp(nElts, EltPerPack);

  bool packAligned = 8 <= alignof(T) || (
      args->nElts*sizeof(T) |
      (uint32_t)reinterpret_cast<uintptr_t>(args->input) |
      (uint32_t)reinterpret_cast<uintptr_t>(args->output)
    )%8 == 0;

  uint32_t nPackPerBlock, nPackModBlock;
  idivmodFast32(&nPackPerBlock, &nPackModBlock, nPacks, prim.nBlocks, prim.nBlocks_rcp32);
  int begin = prim.block*nPackPerBlock + minval<int>(prim.block, nPackModBlock);
  int end = begin + nPackPerBlock + (prim.block < nPackModBlock ? 1 : 0);

  nPacks = end - begin;
  nElts -= begin*EltPerPack;
  nElts = min(nElts, nPacks*EltPerPack);
  T* input = (T*)args->input + begin*EltPerPack;
  T* output = (T*)args->output + begin*EltPerPack;

  ncclCoopCta cta;
  int t = threadIdx.x;
  int tn = ncclSymMaxThreads;

  if (__builtin_expect(packAligned, true)) {
    #pragma unroll 1
    while (0 < nPacks) {
      if (t < nPacks) {
        int nIterPacks = min(nPacks, tn);
        Pack inp = loadPack<Pack>((Pack*)input, t, nPacks);
        prim.bcastLL(/*slot=*/nIterPacks*rank + t, inp);
        Pack out = prim.template recvReduceLL<Pack, T>(t, nIterPacks, red);
        storePack((Pack*)output, t, nPacks, out);
      }
      prim.endLL(cta);

      input += tn*EltPerPack;
      output += tn*EltPerPack;
      nPacks -= tn;
    }
  } else {
    #pragma unroll 1
    while (0 < nElts) {
      if (t*EltPerPack < nElts) {
        int nIterPacks = min(nPacks, tn);
        Pack inp = loadPack<Pack>(input, t*EltPerPack, nElts);
        prim.bcastLL(/*slot=*/nIterPacks*rank + t, inp);
        Pack out = prim.template recvReduceLL<Pack, T>(t, nIterPacks, red);
        storePack(output, t*EltPerPack, nElts, out);
      }
      prim.endLL(cta);

      input += tn*EltPerPack;
      output += tn*EltPerPack;
      nElts -= tn*EltPerPack;
      nPacks -= tn;
    }
  }
}

template<template<typename> class Red, typename T>
__device__ __forceinline__ void pccl_symRun_AllReduce_AGxLL_R(ncclSymDevArgs const* args) {
  pccl_symRun_AllReduce_AGxLL_R_impl<Red, T>(args);
}

