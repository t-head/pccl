/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_RMA_TYPES_H_
#define PCCL_RMA_TYPES_H_

#include <stddef.h>
#include <stdint.h>

// Types shared between the RMA engine and the host scheduling path. Kept free
// of any comm/plan dependency so the leaf headers can include it.

// Signal mode for put-signal operations.
typedef enum {
  NCCL_SIGNAL_NONE = 0,        // No signaling
  NCCL_SIGNAL = 1              // Default signal operation
} ncclSignalMode_t;

// Per-context signal buffer layout: numRmaSig blocks of nRanks counters each.
static inline size_t ncclRmaSignalSlot(int nRanks, int sigIdx, int rank) {
  return (size_t)sigIdx * nRanks + rank;
}

static inline size_t ncclRmaSignalOffset(int nRanks, int sigIdx, int rank) {
  return ncclRmaSignalSlot(nRanks, sigIdx, rank) * sizeof(uint64_t);
}

#endif
