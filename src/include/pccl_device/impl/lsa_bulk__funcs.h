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

#if PCCL_CHECK_HGGCCC

template <int CACHE_POLICY>
PCCL_DEVICE_INLINE uint4 pcclxLsaBulkWarpLoad(const void* src) {
  static_assert(CACHE_POLICY >= PCCLX_LSA_BULK_LD_DEFAULT && CACHE_POLICY <= PCCLX_LSA_BULK_LD_BA,
                "invalid local load cache policy");
  switch (CACHE_POLICY) {
    case PCCLX_LSA_BULK_LD_DEFAULT: return __ppu_global_load_bulk_b32x4(src);
    case PCCLX_LSA_BULK_LD_VOLATILE: return __ppu_global_load_bulk_volatile_b32x4((volatile const void*)src);
    case PCCLX_LSA_BULK_LD_CA: return __ppu_global_ldca_bulk_b32x4(src);
    case PCCLX_LSA_BULK_LD_G: return __ppu_global_ldg_bulk_b32x4(src);
    case PCCLX_LSA_BULK_LD_CG: return __ppu_global_ldcg_bulk_b32x4(src);
    case PCCLX_LSA_BULK_LD_CV: return __ppu_global_ldcv_bulk_b32x4(src);
    case PCCLX_LSA_BULK_LD_CS: return __ppu_global_ldcs_bulk_b32x4(src);
    case PCCLX_LSA_BULK_LD_CE: return __ppu_global_ldce_bulk_b32x4(src);
    case PCCLX_LSA_BULK_LD_BL: return __ppu_global_ldbl_bulk_b32x4(src);
    case PCCLX_LSA_BULK_LD_BA: return __ppu_global_ldba_bulk_b32x4(src);
    default: return make_uint4(0, 0, 0, 0);
  }
}

template <int CACHE_POLICY>
PCCL_DEVICE_INLINE void pcclxLsaBulkWarpStore(void* dst, uint4 frag) {
  static_assert(CACHE_POLICY >= PCCLX_LSA_BULK_ST_DEFAULT && CACHE_POLICY <= PCCLX_LSA_BULK_ST_BA,
                "invalid local store cache policy");
  switch (CACHE_POLICY) {
    case PCCLX_LSA_BULK_ST_DEFAULT: __ppu_global_store_bulk_b32x4(dst, frag); break;
    case PCCLX_LSA_BULK_ST_VOLATILE: __ppu_global_store_bulk_volatile_b32x4((volatile void*)dst, frag); break;
    case PCCLX_LSA_BULK_ST_WB: __ppu_global_stwb_bulk_b32x4(dst, frag); break;
    case PCCLX_LSA_BULK_ST_CG: __ppu_global_stcg_bulk_b32x4(dst, frag); break;
    case PCCLX_LSA_BULK_ST_WT: __ppu_global_stwt_bulk_b32x4(dst, frag); break;
    case PCCLX_LSA_BULK_ST_CS: __ppu_global_stcs_bulk_b32x4(dst, frag); break;
    case PCCLX_LSA_BULK_ST_BL: __ppu_global_stbl_bulk_b32x4(dst, frag); break;
    case PCCLX_LSA_BULK_ST_BA: __ppu_global_stba_bulk_b32x4(dst, frag); break;
    default: return;
  }
}

template <int CACHE_POLICY, int RTMD>
PCCL_DEVICE_INLINE uint4 pcclxLsaBulkWarpPeerLoad(const void* src) {
  static_assert(CACHE_POLICY >= PCCLX_LSA_BULK_RMT_C && CACHE_POLICY <= PCCLX_LSA_BULK_RMT_BYPASS,
                "invalid remote load cache policy");
  static_assert(RTMD >= PCCLX_LSA_BULK_RTMD_NORMAL && RTMD <= PCCLX_LSA_BULK_RTMD_AR,
                "invalid remote route mode");
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ < 150
  switch (CACHE_POLICY) {
    case PCCLX_LSA_BULK_RMT_C: return __ppu_remote_load_bulk_b32x4(src);
    case PCCLX_LSA_BULK_RMT_VOLATILE: return __ppu_remote_load_bulk_volatile_b32x4((volatile const void*)src);
    case PCCLX_LSA_BULK_RMT_NCP: return __ppu_remote_load_bulk_ncp_b32x4(src);
    case PCCLX_LSA_BULK_RMT_BYPASS: return __ppu_remote_load_bulk_bypass_b32x4(src);
    default: return make_uint4(0, 0, 0, 0);
  }
#else
  switch (CACHE_POLICY) {
    case PCCLX_LSA_BULK_RMT_C: return __ppu_remote_load_bulk_b32x4(src, RTMD);
    case PCCLX_LSA_BULK_RMT_VOLATILE: return __ppu_remote_load_bulk_volatile_b32x4((volatile const void*)src, RTMD);
    case PCCLX_LSA_BULK_RMT_NCP: return __ppu_remote_load_bulk_ncp_b32x4(src, RTMD);
    case PCCLX_LSA_BULK_RMT_BYPASS: return __ppu_remote_load_bulk_bypass_b32x4(src, RTMD);
    default: return make_uint4(0, 0, 0, 0);
  }
#endif
}

template <int CACHE_POLICY, int RTMD>
PCCL_DEVICE_INLINE void pcclxLsaBulkWarpPeerStore(void* dst, uint4 frag) {
  static_assert(CACHE_POLICY >= PCCLX_LSA_BULK_RMT_C && CACHE_POLICY <= PCCLX_LSA_BULK_RMT_BYPASS,
                "invalid remote store cache policy");
  static_assert(RTMD >= PCCLX_LSA_BULK_RTMD_NORMAL && RTMD <= PCCLX_LSA_BULK_RTMD_AR,
                "invalid remote route mode");
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ < 150
  switch (CACHE_POLICY) {
    case PCCLX_LSA_BULK_RMT_C: __ppu_remote_store_bulk_b32x4(dst, frag); break;
    case PCCLX_LSA_BULK_RMT_VOLATILE: __ppu_remote_store_bulk_volatile_b32x4((volatile void*)dst, frag); break;
    case PCCLX_LSA_BULK_RMT_NCP: __ppu_remote_store_bulk_ncp_b32x4(dst, frag); break;
    case PCCLX_LSA_BULK_RMT_BYPASS: __ppu_remote_store_bulk_bypass_b32x4(dst, frag); break;
    default: return;
  }
#else
  switch (CACHE_POLICY) {
    case PCCLX_LSA_BULK_RMT_C: __ppu_remote_store_bulk_b32x4(dst, frag, RTMD); break;
    case PCCLX_LSA_BULK_RMT_VOLATILE: __ppu_remote_store_bulk_volatile_b32x4((volatile void*)dst, frag, RTMD); break;
    case PCCLX_LSA_BULK_RMT_NCP: __ppu_remote_store_bulk_ncp_b32x4(dst, frag, RTMD); break;
    case PCCLX_LSA_BULK_RMT_BYPASS: __ppu_remote_store_bulk_bypass_b32x4(dst, frag, RTMD); break;
    default: return;
  }
#endif
}

#endif

#endif
