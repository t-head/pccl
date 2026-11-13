/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_RAS_DIAGNOSTICS_RUNTIME_H_
#define PCCL_RAS_DIAGNOSTICS_RUNTIME_H_

#include "ras_wire.h"

#define RAS_DIAG_RUNTIME_WIRE_VERSION 0x00020001u
#define RAS_DIAG_RUNTIME_MAX_CONNS 2

enum rasDiagnosticsRuntimeRecordType {
  RAS_DIAG_RUNTIME_RANK,
  RAS_DIAG_RUNTIME_CHANNEL,
  RAS_DIAG_RUNTIME_CONNECTION,
};

enum rasDiagnosticsRuntimeAvailability {
  RAS_DIAG_RUNTIME_AVAILABLE = 0,
  RAS_DIAG_RUNTIME_ABORTED = 2,
  RAS_DIAG_RUNTIME_UNAVAILABLE = 3,
};

// Shared by the live collector, diagnostics transport and offline analyzer.
// A rank contributes one RANK plus its CHANNEL and CONNECTION records.
struct rasDiagnosticsRuntimeRecord {
  struct rasDiagnosticsRankHeader rank;
  uint32_t type;
  int channelId;
  union {
    struct {
      uint32_t version;
      uint32_t availability;
      int nChannels;
      int hggcDev;
      int hgmlDev;
    } rankData;
    struct rasDiagChannel channel;
    struct rasDiagPeer connection;
  } value;
};

static_assert(offsetof(rasDiagnosticsRuntimeRecord, rank) == 0, "diagnostics rank header must be first");
static_assert(sizeof(rasDiagnosticsRuntimeRecord) % alignof(int64_t) == 0, "aligned runtime record stride");

#endif // PCCL_RAS_DIAGNOSTICS_RUNTIME_H_
