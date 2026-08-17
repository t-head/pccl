/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_CORE__TYPES_H_
#define _NCCL_DEVICE_CORE__TYPES_H_
#include "../core.h"
#if defined(NCCL_OS_WINDOWS)
#include <hggc.h>
/* Minimal types instead of nccl_device/gin/gin_device_host_common.h (GIN is Linux-only) */
#define NCCL_GIN_MAX_CONNECTIONS 4
typedef void *ncclGinWindow_t;
#else
#include "nccl_device/gin/gin_device_host_common.h"
#endif
#include <hggc.h>  // HGmemLocationType

struct ncclSegmentWindow {
  ncclGinWindow_t ginWins[NCCL_GIN_MAX_CONNECTIONS];
  size_t segmentSize;
  HGmemLocationType memType;
};

// pccl_internal.h has: typedef pcclWindow_t = struct ncclWindow_vidmem*
struct ncclWindow_vidmem {
  void* winHost;
  char* lsaFlatBase; // pointer to first byte for rank 0 of lsa team
  int lsaRank;
  int worldRank;
  uint32_t stride4G;
  uint32_t mcOffset4K;
  uint32_t ginOffset4K;
  ncclGinWindow_t ginWins[NCCL_GIN_MAX_CONNECTIONS];
  struct ncclSegmentWindow* ginMultiSegmentWins; // multi-segment: pointer to accommodate variable num segments
  int numSegments;
};

// Inlined resource-window. Subset of ncclWindow_vidmem for resource-buffer addressing.
struct ncclResourceWindow_vidmem {
  char reserved1[8];
  char* lsaFlatBase;
  char reserved2[8];
  uint32_t stride4G;
  uint32_t mcOffset4K;
  char reserved3[40];
};

struct ncclMultimemHandle {
  void* mcBasePtr;
};

#endif
