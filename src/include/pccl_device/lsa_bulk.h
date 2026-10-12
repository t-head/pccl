/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_DEVICE_LSA_BULK_H_
#define PCCL_DEVICE_LSA_BULK_H_

#include "core.h"

#define PCCLX_HAS_LSA_BULK 1

#define PCCLX_LSA_BULK_WARP_BYTES 512

// Local load cache policies.
#define PCCLX_LSA_BULK_LD_DEFAULT 0
#define PCCLX_LSA_BULK_LD_VOLATILE 1
#define PCCLX_LSA_BULK_LD_CA 2 // cache in all levels
#define PCCLX_LSA_BULK_LD_G 3 // read-only, cache in L2/LLC
#define PCCLX_LSA_BULK_LD_CG 4 // cache in LLC only
#define PCCLX_LSA_BULK_LD_CV 5 // no cache
#define PCCLX_LSA_BULK_LD_CS 6 // streaming
#define PCCLX_LSA_BULK_LD_CE 7 // evict first
#define PCCLX_LSA_BULK_LD_BL 8 // bypass L1/L2
#define PCCLX_LSA_BULK_LD_BA 9 // bypass all levels

// Local store cache policies.
#define PCCLX_LSA_BULK_ST_DEFAULT 0
#define PCCLX_LSA_BULK_ST_VOLATILE 1
#define PCCLX_LSA_BULK_ST_WB 2 // write back
#define PCCLX_LSA_BULK_ST_CG 3 // write through, cache in LLC
#define PCCLX_LSA_BULK_ST_WT 4 // write through to memory
#define PCCLX_LSA_BULK_ST_CS 5 // streaming
#define PCCLX_LSA_BULK_ST_BL 6 // bypass L1/L2
#define PCCLX_LSA_BULK_ST_BA 7 // bypass all levels

// Remote load/store cache policies.
#define PCCLX_LSA_BULK_RMT_C 0 // cache in LLC
#define PCCLX_LSA_BULK_RMT_VOLATILE 1
#define PCCLX_LSA_BULK_RMT_NCP 2 // no cache
#define PCCLX_LSA_BULK_RMT_BYPASS 3 // bypass LLC

// Remote route modes.
#define PCCLX_LSA_BULK_RTMD_NORMAL 0
#define PCCLX_LSA_BULK_RTMD_PA 1
#define PCCLX_LSA_BULK_RTMD_AR 2

#if PCCL_CHECK_HGGCCC

// Warp-wide bulk access of PCCLX_LSA_BULK_WARP_BYTES bytes. All 32 lanes
// must participate with the same address of a complete block. The caller
// is responsible for bounds and 4B alignment (128B recommended); no address
// checks or conversions are performed. Use pcclCoopBcast(pcclCoopWarp(),
// ptr, 0) to prepare a warp-uniform base. Lane l holds the l-th dword of
// each 128B line.
template <int CACHE_POLICY = PCCLX_LSA_BULK_LD_CG>
PCCL_DEVICE_INLINE uint4 pcclxLsaBulkWarpLoad(const void* src);
template <int CACHE_POLICY = PCCLX_LSA_BULK_ST_CG>
PCCL_DEVICE_INLINE void pcclxLsaBulkWarpStore(void* dst, uint4 frag);

// Peer variants take an already translated remote LSA pointer, obtained
// with pcclGetLsaPointer (LSA-team rank) or pcclGetPeerPointer (world rank).
// The caller must select a remote peer and satisfy the warp/block contract
// above. Remote instructions must not target local memory; use the local
// variants for the caller's own rank. No peer-rank check is performed.
template <int CACHE_POLICY = PCCLX_LSA_BULK_RMT_C, int RTMD = PCCLX_LSA_BULK_RTMD_PA>
PCCL_DEVICE_INLINE uint4 pcclxLsaBulkWarpPeerLoad(const void* src);
template <int CACHE_POLICY = PCCLX_LSA_BULK_RMT_C, int RTMD = PCCLX_LSA_BULK_RTMD_PA>
PCCL_DEVICE_INLINE void pcclxLsaBulkWarpPeerStore(void* dst, uint4 frag);

#endif

#endif
