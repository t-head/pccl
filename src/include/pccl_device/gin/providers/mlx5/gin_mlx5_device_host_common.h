/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_GIN_GDAKI_MLX_DEVICE_HOST_COMMON_H_
#define _PCCL_DEVICE_GIN_GDAKI_MLX_DEVICE_HOST_COMMON_H_

#include <linux/types.h>

// Compat with doca-gpunetio device code v2.0.0.
#define PCCL_GIN_MLX5_VERSION 200

template <typename T>
struct pcclGinGdakiMlxGlobalGPUBufferTable {
  T *buffer;
  __be32 *rkeys;
  __be32 lkey;
  unsigned int offset;
};

struct pcclGinGdakiMlxGPUContext {
  struct doca_gpu_dev_verbs_qp *gdqp;
  struct doca_gpu_dev_verbs_qp *companion_gdqp;
  struct pcclGinGdakiMlxGlobalGPUBufferTable<uint64_t> counters_table;
  struct pcclGinGdakiMlxGlobalGPUBufferTable<uint64_t> signals_table;

  // Local buffer we don't consume but is required for some operations.
  __be32 sink_buffer_lkey;
};

struct pcclGinGdakiMlxMemHandle {
  __be32 *rkeys;
  __be32 lkey;
};

#endif /* _PCCL_DEVICE_GIN_GDAKI_MLX_DEVICE_HOST_COMMON_H_ */
