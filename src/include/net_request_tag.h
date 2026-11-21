/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#pragma once
// Mark built-in IB requests that support the receiver-selected QP count.
// Keep the rank in the low bits; no collective-specific state is needed.
static constexpr int PCCL_NET_REQUEST_TAG = 1 << 30;
static inline bool pcclIsNetRequestTag(int tag) {
  return (tag & PCCL_NET_REQUEST_TAG) != 0;
}

// Private receive-side policy bit. CTS carries it to the sender; it is not
// part of the peer identity used for matching. No CTS layout change is needed.
static constexpr int PCCL_NET_SINGLE_QP_TAG = 1 << 29;
static inline bool pcclNetSingleQpTag(int tag) {
  return (tag & (PCCL_NET_REQUEST_TAG | PCCL_NET_SINGLE_QP_TAG)) ==
      (PCCL_NET_REQUEST_TAG | PCCL_NET_SINGLE_QP_TAG);
}
static inline int pcclNetMatchTag(int tag) {
  // The receiver owns the QP decision; policy bits do not change peer identity.
  return pcclIsNetRequestTag(tag)
      ? tag & ~(PCCL_NET_REQUEST_TAG | PCCL_NET_SINGLE_QP_TAG) : tag;
}
