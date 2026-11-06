/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *************************************************************************/
#ifndef PCCL_DEVCOMM_COMPAT_H_
#define PCCL_DEVCOMM_COMPAT_H_

#include "pccl_device.h"
#include <stddef.h>

// Frozen append-only PCCL 2.2 device API layouts. Never extend these records.
struct pcclDevCommRequirementsLegacy {
  /* attributes that users should never touch. */
  size_t size;
  unsigned int magic;
  unsigned int version;

  /* attributes that users are able to customize. */
  pcclDevResourceRequirements_t* resourceRequirementsList;
  pcclTeamRequirements_t* teamRequirementsList;

  bool lsaMultimem; // Enable multimem on lsa team

  int barrierCount;
  int lsaBarrierCount;
  int railGinBarrierCount;

  int lsaLLA2ABlockCount, lsaLLA2ASlotCount;

  bool ginForceEnable;

  int ginContextCount; // This is a hint, the actual context count in the devcomm may not match.
  int ginSignalCount; // Guaranteed to start at id=0
  int ginCounterCount; // Guaranteed to start at id=0
  pcclGinConnectionType_t ginConnectionType;
  bool ginExclusiveContexts;
  int ginQueueDepth;
  int ginTrafficClass;

  int worldGinBarrierCount;
};

struct pcclCommPropertiesLegacy {
  /* internal use only */
  size_t size;
  unsigned int magic;
  unsigned int version;

  /* attributes for users. */
  int rank;
  int nRanks;
  int hggcDev;
  int hgmlDev;
  bool deviceApiSupport;
  bool multimemSupport;
  pcclGinType_t ginType;
  int nLsaTeams;
  bool hostRmaSupport;
  pcclGinType_t railedGinType;
};

struct pcclDevCommLegacy {
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

static_assert(sizeof(pcclDevCommRequirementsLegacy) == offsetof(pcclDevCommRequirements, ginCustomStride),
              "Requirements extension must start after the complete legacy ABI");
static_assert(sizeof(pcclCommPropertiesLegacy) == offsetof(pcclCommProperties, commHash),
              "Properties extension must start after the complete legacy ABI");
static_assert(sizeof(pcclDevCommLegacy) == offsetof(pcclDevComm, ginConnectionStride),
              "Device-communicator extension must start after the complete legacy ABI");
static_assert(offsetof(pcclDevCommLegacy, rank) == offsetof(pcclDevComm, rank), "Legacy rank offset");
static_assert(offsetof(pcclDevCommLegacy, nRanks) == offsetof(pcclDevComm, nRanks), "Legacy nRanks offset");
static_assert(offsetof(pcclDevCommLegacy, lsaRank) == offsetof(pcclDevComm, lsaRank), "Legacy lsaRank offset");
static_assert(offsetof(pcclDevCommLegacy, lsaSize) == offsetof(pcclDevComm, lsaSize), "Legacy lsaSize offset");
static_assert(offsetof(pcclDevCommLegacy, windowTable) == offsetof(pcclDevComm, windowTable), "Legacy windowTable offset");
static_assert(offsetof(pcclDevCommLegacy, resourceWindow) == offsetof(pcclDevComm, resourceWindow), "Legacy resourceWindow offset");
static_assert(offsetof(pcclDevCommLegacy, resourceWindow_inlined) == offsetof(pcclDevComm, resourceWindow_inlined), "Legacy resourceWindow_inlined offset");
static_assert(offsetof(pcclDevCommLegacy, lsaMultimem) == offsetof(pcclDevComm, lsaMultimem), "Legacy lsaMultimem offset");
static_assert(offsetof(pcclDevCommLegacy, lsaBarrier) == offsetof(pcclDevComm, lsaBarrier), "Legacy lsaBarrier offset");
static_assert(offsetof(pcclDevCommLegacy, railGinBarrier) == offsetof(pcclDevComm, railGinBarrier), "Legacy railGinBarrier offset");
static_assert(offsetof(pcclDevCommLegacy, ginConnectionCount) == offsetof(pcclDevComm, ginConnectionCount), "Legacy ginConnectionCount offset");
static_assert(offsetof(pcclDevCommLegacy, ginNetDeviceTypes) == offsetof(pcclDevComm, ginNetDeviceTypes), "Legacy ginNetDeviceTypes offset");
static_assert(offsetof(pcclDevCommLegacy, ginHandles) == offsetof(pcclDevComm, ginHandles), "Legacy ginHandles offset");
static_assert(offsetof(pcclDevCommLegacy, ginSignalCount) == offsetof(pcclDevComm, ginSignalCount), "Legacy ginSignalCount offset");
static_assert(offsetof(pcclDevCommLegacy, ginCounterCount) == offsetof(pcclDevComm, ginCounterCount), "Legacy ginCounterCount offset");
static_assert(offsetof(pcclDevCommLegacy, ginSignalShadows) == offsetof(pcclDevComm, ginSignalShadows), "Legacy ginSignalShadows offset");
static_assert(offsetof(pcclDevCommLegacy, ginContextCount) == offsetof(pcclDevComm, ginContextCount), "Legacy ginContextCount offset");
static_assert(offsetof(pcclDevCommLegacy, ginIsRailed) == offsetof(pcclDevComm, ginIsRailed), "Legacy ginIsRailed offset");
static_assert(offsetof(pcclDevCommLegacy, abortFlag) == offsetof(pcclDevComm, abortFlag), "Legacy abortFlag offset");
static_assert(offsetof(pcclDevCommLegacy, hybridLsaBarrier) == offsetof(pcclDevComm, hybridLsaBarrier), "Legacy hybridLsaBarrier offset");
static_assert(offsetof(pcclDevCommLegacy, hybridRailGinBarrier) == offsetof(pcclDevComm, hybridRailGinBarrier), "Legacy hybridRailGinBarrier offset");
static_assert(offsetof(pcclDevCommLegacy, worldGinBarrier) == offsetof(pcclDevComm, worldGinBarrier), "Legacy worldGinBarrier offset");

#endif
