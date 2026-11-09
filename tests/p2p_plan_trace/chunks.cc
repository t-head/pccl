// SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "devcomm.h"
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <limits>

static_assert(sizeof(ncclDevWorkP2p) == 64, "PCCL work budget must not grow");
static_assert(offsetof(ncclDevWorkP2p, sendAddr) == 0, "NCCL address prefix");
static_assert(offsetof(ncclDevWorkP2p, sendBytes) == 16, "NCCL byte prefix");
static_assert(offsetof(ncclDevWorkP2p, nP2pChannels) == 40, "NCCL channel prefix");
static_assert(offsetof(ncclDevWorkP2p, sendChunkSize_u32fp8) == 44, "NCCL chunk prefix");

static void check(uint32_t bytes) {
  ncclDevWorkP2p work = {};
  work.setSendChunkSize(bytes);
  work.setRecvChunkSize(bytes ^ 0x12345u);
  assert(work.sendChunkSize() == bytes);
  assert(work.recvChunkSize() == (bytes ^ 0x12345u));
  bool exact = u32fp8Decode(u32fp8Encode(bytes)) == bytes;
  assert(work.sendChunkSizeRaw == (exact ? 0u : bytes));
  work.setSendChunkSize(1u << 20);
  assert(work.sendChunkSizeRaw == 0 && work.sendChunkSize() == (1u << 20));
  assert(work.recvChunkSize() == (bytes ^ 0x12345u));
}

int main() {
  for (uint32_t bytes = 0; bytes != (1u << 20); bytes++) check(bytes);
  for (unsigned code = 0; code != 256; code++) {
    uint32_t bytes = u32fp8Decode(code);
    check(bytes);
    if (bytes) check(bytes - 1);
    if (bytes != std::numeric_limits<uint32_t>::max()) check(bytes + 1);
  }
  check(std::numeric_limits<uint32_t>::max());
  std::puts("PASS exact P2P chunk round trips and 64-byte NCCL-prefix layout");
}
