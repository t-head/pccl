/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PCCL_GIN_GDAKI_DEVICE_HOST_COMMON_H_
#define PCCL_GIN_GDAKI_DEVICE_HOST_COMMON_H_

#include <stdint.h>

// Version 300 introduces a provider-tagged handle instead of a raw MLX5
// context array. Host and application device code must use matching headers.
#define PCCL_GIN_GDAKI_VERSION 300

enum pcclGinGdakiProvider : uint32_t {
  PCCL_GIN_GDAKI_PROVIDER_NONE = 0,
  PCCL_GIN_GDAKI_PROVIDER_MLX5 = 1,
  PCCL_GIN_GDAKI_PROVIDER_TROCE = 2,
};

// Device-resident, immutable descriptor for one GDAKI connection. contexts
// points to the provider's private context array, indexed by contextId.
struct pcclGinGdakiHandle {
  pcclGinGdakiProvider provider;
  uint32_t providerVersion;
  void* contexts;
};

static_assert(sizeof(pcclGinGdakiHandle) == 16, "GDAKI handle ABI");

#endif
