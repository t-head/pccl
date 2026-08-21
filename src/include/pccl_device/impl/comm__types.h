/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_COMM__TYPES_H_
#define _PCCL_DEVICE_COMM__TYPES_H_
#include "../comm.h"
#include "core__types.h"
#include "ll_a2a__types.h"
#include "lsa_barrier__types.h"
#include "gin_barrier__types.h"

#if __cplusplus
struct pcclDevCommWindowTable {
  struct Entry {
    uintptr_t base, size;
    pcclWindow_t window;
  } entries[32];
  struct pcclDevCommWindowTable* next;
};
#endif
typedef struct pcclDevCommWindowTable* pcclDevCommWindowTable_t;

struct pcclDevComm {
  // Internal PCCL structure versioning metadata.  Do not modify.
  unsigned int magic;
  unsigned int version;

  int rank, nRanks;
  uint32_t nRanks_rcp32;
  int lsaRank, lsaSize;
  uint32_t lsaSize_rcp32;

  pcclDevCommWindowTable_t windowTable;

  pcclWindow_t resourceWindow;
  struct pcclResourceWindow_vidmem resourceWindow_inlined;

  pcclMultimemHandle_t lsaMultimem;
  pcclLsaBarrierHandle_t lsaBarrier;
  pcclGinBarrierHandle_t railGinBarrier;

  uint8_t ginConnectionCount;
  uint8_t ginNetDeviceTypes[PCCL_GIN_MAX_CONNECTIONS];
  void* ginHandles[PCCL_GIN_MAX_CONNECTIONS];
  int ginSignalCount;
  int ginCounterCount;
  uint64_t* ginSignalShadows;
  uint32_t ginContextCount;
  bool ginIsRailed; // Whether the GIN connections are railed

  // FT related
  uint32_t* abortFlag;

  pcclLsaBarrierHandle_t hybridLsaBarrier;
  pcclGinBarrierHandle_t hybridRailGinBarrier;

  pcclGinBarrierHandle_t worldGinBarrier;
};

#endif // _PCCL_DEVICE_COMM__TYPES_H_
