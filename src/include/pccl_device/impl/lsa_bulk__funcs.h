/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_DEVICE_LSA_BULK__FUNCS_H_
#define PCCL_DEVICE_LSA_BULK__FUNCS_H_

#include "../lsa_bulk.h"
#include "core__types.h"

#if PCCL_CHECK_HGGCCC

namespace pccl {
namespace lsa {
namespace internal {
constexpr unsigned RtmdPa = 1;

template<typename T>
PCCL_DEVICE_INLINE T* toWarpPtr(T* ptr) {
  uint64_t bits = reinterpret_cast<uint64_t>(ptr);
  uint32_t lo = __ppu_read_firstlane(static_cast<uint32_t>(bits));
  uint32_t hi = __ppu_read_firstlane(static_cast<uint32_t>(bits >> 32));
  return reinterpret_cast<T*>((static_cast<uint64_t>(hi) << 32) | lo);
}

template<typename T>
PCCL_DEVICE_INLINE T* toPeerPtr(pcclWindow_t window, T* localPtr, int peer) {
  int lsaRank = pccl::utility::loadConst(&window->lsaRank);
  uint32_t stride4G = pccl::utility::loadConst(&window->stride4G);
  return toWarpPtr(pccl::utility::add4G(localPtr, (peer - lsaRank) * (int)stride4G));
}
} // namespace internal
} // namespace lsa
} // namespace pccl

PCCL_DEVICE_INLINE uint4 pcclxLsaBulkWarpLoad(const void* src) {
  return __ppu_global_ldcg_bulk_b32x4(pccl::lsa::internal::toWarpPtr(src));
}

PCCL_DEVICE_INLINE void pcclxLsaBulkWarpStore(void* dst, uint4 frag) {
  __ppu_global_store_bulk_b32x4(pccl::lsa::internal::toWarpPtr(dst), frag);
}

PCCL_DEVICE_INLINE uint4 pcclxLsaBulkWarpPeerLoad(pcclWindow_t window, const void* localPtr, int peer) {
  const void* remotePtr = pccl::lsa::internal::toPeerPtr(window, localPtr, peer);
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ < 150
  return __ppu_remote_load_bulk_b32x4(remotePtr);
#else
  return __ppu_remote_load_bulk_b32x4(remotePtr, pccl::lsa::internal::RtmdPa);
#endif
}

PCCL_DEVICE_INLINE void pcclxLsaBulkWarpPeerStore(pcclWindow_t window, void* localPtr, int peer, uint4 frag) {
  void* remotePtr = pccl::lsa::internal::toPeerPtr(window, localPtr, peer);
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ < 150
  __ppu_remote_store_bulk_b32x4(remotePtr, frag);
#else
  __ppu_remote_store_bulk_b32x4(remotePtr, frag, pccl::lsa::internal::RtmdPa);
#endif
}

#endif

#endif
