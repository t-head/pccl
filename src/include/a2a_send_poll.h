/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#pragma once
#include <stdint.h>

// Shared encoding in the CPU call policy and ncclDevWorkP2p::sendPollFlags.
// Keep these values stable when passing work from host to device.
static constexpr uint8_t PCCL_A2A_POLL_TRACK_ACTIVE = 1u << 0;
static constexpr uint8_t PCCL_A2A_POLL_TRACE_ACTIVE = 1u << 1;

#ifndef PCCL_POLL_HD
#define PCCL_POLL_HD
#define PCCL_POLL_UNDEF_HD
#endif
// active comes from executing send primitives across this communicator's channels.
// reference is explicitly calibrated, not inferred from rank/peer/QP counts.
// Round up before applying the cap. A zero input disables the wait; a zero
// reference can still be used for active-count diagnostics by the caller.
PCCL_POLL_HD static inline uint32_t pcclSendPollTicks(uint32_t active, uint32_t base,
                                                    uint32_t reference, uint32_t limit) {
  if (!active || !base || !reference || !limit) return 0;
  const uint64_t scaled = (uint64_t(base) * active + reference - 1) / reference;
  return uint32_t(scaled > limit ? limit : scaled);
}
#ifdef PCCL_POLL_UNDEF_HD
#undef PCCL_POLL_HD
#undef PCCL_POLL_UNDEF_HD
#endif
