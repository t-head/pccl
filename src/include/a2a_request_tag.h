/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#pragma once
// Transport tags are opaque integers. Keep the rank in the low bits and carry
// the originating API per request, never on a reusable connection.
static constexpr int PCCL_A2A_REQUEST_TAG = 1 << 30;
static inline bool pcclIsA2aRequestTag(int tag) {
  return (tag & PCCL_A2A_REQUEST_TAG) != 0;
}

// Private receive-side policy bit. CTS carries it to the sender; it is not
// part of the peer identity used for matching. No CTS layout change is needed.
static constexpr int PCCL_A2A_SINGLE_QP_TAG = 1 << 29;
static inline bool pcclA2aSingleQpTag(int tag) {
  return (tag & (PCCL_A2A_REQUEST_TAG | PCCL_A2A_SINGLE_QP_TAG)) ==
      (PCCL_A2A_REQUEST_TAG | PCCL_A2A_SINGLE_QP_TAG);
}
static inline int pcclA2aMatchTag(int tag) {
  // Per-call connection counts can differ across ranks. A receiver owns the
  // QP decision in CTS; neither policy bit changes main's peer identity.
  return pcclIsA2aRequestTag(tag)
      ? tag & ~(PCCL_A2A_REQUEST_TAG | PCCL_A2A_SINGLE_QP_TAG) : tag;
}
