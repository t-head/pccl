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

#if PCCL_CHECK_HGGCCC

PCCL_DEVICE_INLINE uint4 pcclxLsaBulkWarpLoad(const void* src);
PCCL_DEVICE_INLINE void pcclxLsaBulkWarpStore(void* dst, uint4 frag);

PCCL_DEVICE_INLINE uint4 pcclxLsaBulkWarpPeerLoad(pcclWindow_t window, const void* localPtr, int peer);
PCCL_DEVICE_INLINE void pcclxLsaBulkWarpPeerStore(pcclWindow_t window, void* localPtr, int peer, uint4 frag);

#endif

#endif
