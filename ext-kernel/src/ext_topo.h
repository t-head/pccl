/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef EXT_KERNEL_EXT_TOPO_H_
#define EXT_KERNEL_EXT_TOPO_H_

#include "pccl/ext_kernel.h"

typedef enum {
  TWO_RANKS_810_TOPO,
  TWO_RANKS_810E_TOPO,
  TWO_RANKS_M890P_TOPO,
  TWO_RANKS_M890P_SHM_TOPO,
  FOUR_RANKS_810_FC_TOPO, // 810: 0,1,2,3
  FOUR_RANKS_810_SQUARE_TOPO, // 810: 0,1,6,7
  FOUR_RANKS_810E_FC_TOPO, // 810E: 0,1,2,3
  FOUR_RANKS_810E_SQUARE1_TOPO, // 810E:
  FOUR_RANKS_810E_SQUARE2_TOPO, // 810E:
  FOUR_RANKS_M890P_FC_TOPO,
  FOUR_RANKS_M890P_SHM_TOPO,
  EIGHT_RANKS_810_TOPO, // 810:
  EIGHT_RANKS_810E_CROSS_REAR_TOPO, // 810E:
  EIGHT_RANKS_810E_ONE_REAR_TOPO, // 810E:
  EIGHT_RANKS_M890P_TOPO,
  EIGHT_RANKS_M890P_SHM_TOPO,
  SIXTEEN_RANKS_810E,
  SIXTEEN_RANKS_M890P_SHM_TOPO,
  THIRTY_TWO_RANKS_M890P_SHM_TOPO,
  UNSUPPORTED_TOPO = 0xFF,
} extTopo;

typedef struct {
  int topoId;           // should exactly match ExtTopo enum index
  int boundLow;         // lower size bound (bytes)
  int boundHigh;        // upper size bound for AllReduce (bytes)
  int oneShotLLBoundHigh; // upper size limit for ONESHOT_LL strategy (bytes)
  int twoShotSwitchPoint; // -1 means use boundHigh
  int maxBlockNum;
  int agBoundHigh;      // max total bytes for AllGather ext-kernel
  int rsBoundHigh;      // max total bytes for ReduceScatter ext-kernel
} extTopoConfig;

extern const extTopoConfig topoConfigs[];

ncclResult_t parseTopo(PPU_TYPE ppuType, pcclICNRemoteDeviceType_t icnRmtType, int localRanks, const int* topoArray, extTopo* topo);

#endif /* EXT_KERNEL_EXT_TOPO_H_ */
