//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include "check_items.h"
#include "pccl_internal.h"
#include "nvmlwrap.h"
#include "comm.h"
#include "graph.h"


const char *ppuTypeStr[] = {"PPU", "PPU-ZW610", "PPU-ZW810", "PPU-ZW805",
                            "PPU-ZW610E", "PPU-ZW810E", "PPU-ZW810-H", "PPU-ZW805-H",
                            "ZW-M890P", "ZW-M890L", "ZW-M530", "ZW-M530P"
                            };
const char* icnRmtTypeStr[] = {"UNKNOWN", "PPU", "SWITCH"};
static constexpr int NTYPES = _PPU_TYPE_COUNT;
static_assert(sizeof(ppuTypeStr) / sizeof(ppuTypeStr[0]) == NTYPES,
              "ppuTypeStr array size must match PPU_TYPE enum count");

static pthread_mutex_t checkItemsLock = PTHREAD_MUTEX_INITIALIZER;

ppuDeviceInfo* pcclGetPpuDevInfo() {
  static ppuDeviceInfo ppuDevInfo;
  static bool isDevInfoInitialized = false;
  if (isDevInfoInitialized || pcclHgmlEnsureInitialized() != ncclSuccess) {
    return &ppuDevInfo;
  }

  hgmlDevice_t hgmlDev = NULL;
  ncclResult_t tmpRes = ncclSuccess;

  int ndevs = -1;
  tmpRes = pcclHgmlDeviceGetCount(&ndevs);

  pthread_mutex_lock(&checkItemsLock);
  // double check locking pattern
  if (isDevInfoInitialized) {
    goto END;
  }
  if (tmpRes == ncclSuccess) {
    if (ndevs <= 0 || ndevs > MAX_GPUS_PER_NODE) {
      WARN("Abnormal device count: %d", ndevs);
      goto END;
    }
    ppuDevInfo.ppuCount = ndevs;

    tmpRes = pcclHgmlDeviceGetHandleByIndex(0, &hgmlDev);
    if (tmpRes == ncclSuccess) {
      char device_name[PPU_DEVICE_NAME_LEN];

      tmpRes = pcclHgmlDeviceGetName(hgmlDev, device_name, PPU_DEVICE_NAME_LEN);
      if (tmpRes == ncclSuccess) {
        int idx;
        for (idx = 0; idx < NTYPES; idx++) {
          if (strncmp(device_name, ppuTypeStr[idx], PPU_DEVICE_NAME_LEN) == 0) {
            ppuDevInfo.ppuType = (PPU_TYPE)idx;
            break;
          }
        }
        if (idx == NTYPES) {
          WARN("Failed to get device name");
          ppuDevInfo.ppuType = _PPU_UNKNOWN;
        }
      }
      int hggcMajor, hggcMinor;
      if (pcclHgmlDeviceGetHggcComputeCapability(hgmlDev, &hggcMajor, &hggcMinor) == ncclSuccess) {
        ppuDevInfo.sm = hggcMajor * 10 + hggcMinor;
      } else {
        WARN("Failed to get compute capability");
      }
      // find out the first active icnlink then determine remote device type
      ppuDevInfo.icnRmtType = PCCL_ICNLINK_UNKNOWN;
      for (int l = 0; l < HGML_ICNLINK_MAX_LINKS; l++) {
        hgmlIntIcnLinkDeviceType_t icnRmtType = HGML_ICNLINK_DEVICE_TYPE_UNKNOWN;
        if (pcclHgmlDeviceGetIcnLinkRemoteDeviceType(hgmlDev, l, &icnRmtType) == ncclSuccess) {
          if (icnRmtType == HGML_ICNLINK_DEVICE_TYPE_SWITCH) ppuDevInfo.icnRmtType = PCCL_ICNLINK_SWITCH;
          else if (icnRmtType == HGML_ICNLINK_DEVICE_TYPE_PPU) ppuDevInfo.icnRmtType = PCCL_ICNLINK_PPU;
          break;
        }
      }
      if (ppuDevInfo.icnRmtType == PCCL_ICNLINK_UNKNOWN) {
        WARN("Failed to get remote device type");
      }
    }
  }

END:
  isDevInfoInitialized = true;
  pthread_mutex_unlock(&checkItemsLock);
  return &ppuDevInfo;
}

PCCL_PARAM(CheckItemsEnable, "CHECK_ITEMS_ENABLE", 1);
ncclResult_t pcclCheckICNLinkSpeed(int devId, int linkId, unsigned int bandwidth) {
  if (pcclParamCheckItemsEnable() == 0) return ncclSuccess;

  ppuDeviceInfo *ppuDevInfo = pcclGetPpuDevInfo();
  if (ppuDevInfo->ppuType == _PPU_UNKNOWN || ppuDevInfo->ppuCount == 0) {
    return ncclSuccess;
  }

  bool isLinkSpeedNormal = false;
  const char* normalSpeedStr = "";
  static int ppuLinkSpeed = 0;

  // for 810 & 810E & 805 & 890P & 890L & 530, all enabled links have 50/53 GB/s and must be the same
  // for 610 & 610E, 0/6 link has 25 GB/s, 3/4/5 link has 50/53 GB/s and must be the same
  if (ppuDevInfo->ppuType == _PPU_ZW810 || ppuDevInfo->ppuType == _PPU_ZW805 || ppuDevInfo->ppuType == _PPU_ZW810E ||
      ppuDevInfo->ppuType == _ZW_M890P || ppuDevInfo->ppuType == _ZW_M890L || ppuDevInfo->ppuType == _ZW_M530 || ppuDevInfo->ppuType == _ZW_M530P) {
    normalSpeedStr = "50/53";
    if (ppuLinkSpeed == 0) {
      ppuLinkSpeed = bandwidth;
    }
    if ((bandwidth == ICN_LINK_SPEED_50 || bandwidth == ICN_LINK_SPEED_53) && bandwidth == ppuLinkSpeed) {
      isLinkSpeedNormal = true;
    }
  } else if (ppuDevInfo->ppuType == _PPU_ZW610 || ppuDevInfo->ppuType == _PPU_ZW610E) {
    if (linkId == 0 || linkId == 6) {
      normalSpeedStr = "25";
      if (bandwidth == ICN_LINK_SPEED_25) {
        isLinkSpeedNormal = true;
      }
    } else if (linkId == 3 || linkId == 4 || linkId == 5) {
      normalSpeedStr = "50/53";
      if (ppuLinkSpeed == 0) {
        ppuLinkSpeed = bandwidth;
      }
      if ((bandwidth == ICN_LINK_SPEED_50 || bandwidth == ICN_LINK_SPEED_53) && bandwidth == ppuLinkSpeed) {
        isLinkSpeedNormal = true;
      }
    } else {
      WARN("Error link id: %d for 610/610E", linkId);
      isLinkSpeedNormal = true;
    }
  } else {
    WARN("Skip to do link speed check when ppu type is %s", ppuTypeStr[ppuDevInfo->ppuType]);
    isLinkSpeedNormal = true;
  }

  if (!isLinkSpeedNormal) {
    WARN("The link speed of devId: %d linkId: %d should be %s GB/s but get %u GB/s", devId, linkId, normalSpeedStr, bandwidth);
  }

  return ncclSuccess;
}

bool samePcieDomain(int ppu_id, int peer_ppu_id) {
  if ((ppu_id < PER_PCI_DOMAIN_DEV_NUM && peer_ppu_id < PER_PCI_DOMAIN_DEV_NUM)
      || (ppu_id >= PER_PCI_DOMAIN_DEV_NUM && peer_ppu_id >= PER_PCI_DOMAIN_DEV_NUM))
    return true;
  else return false;
}

ncclResult_t pcclCheckIntraServer810E(int *ppuTopoArray, ppuDeviceInfo *ppuDevInfo, int serverId) {
  int ppuCount = ppuDevInfo->ppuCount;
  // numbers of PPU which has two ICN1/ICN2 links with other PPUs intra domain
  int intraDomainTwoICN1 = 0;
  int intraDomainTwoICN2 = 0;
  // numbers of ICN1/ICN2 link of current PPU which connect with other PPUs intra domain
  int intraDomainICN1 = 0;
  int intraDomainICN2 = 0;

  // numbers of ICN1/ICN2 link of current PPU which connect with other PPUs inter domain
  int interDomainICN1 = 0;
  int interDomainICN2 = 0;
  for (int cur = serverId * PER_OAM_TAIL_NODE_DEV_NUM;
            cur < (serverId + 1) * PER_OAM_TAIL_NODE_DEV_NUM; cur++) {
    intraDomainICN1 = 0;
    intraDomainICN2 = 0;
    for (int rem = serverId * PER_OAM_TAIL_NODE_DEV_NUM;
             rem < (serverId + 1) * PER_OAM_TAIL_NODE_DEV_NUM; rem++) {
      if (samePcieDomain(cur % PER_OAM_TAIL_NODE_DEV_NUM, rem % PER_OAM_TAIL_NODE_DEV_NUM)) {
        if (cur == rem) continue;
        if (ppuTopoArray[cur * ppuCount + rem] == ICN1) {
          intraDomainICN1++;
        } else if (ppuTopoArray[cur * ppuCount + rem] == ICN2) {
          intraDomainICN2++;
        } else if (ppuTopoArray[cur * ppuCount + rem] != SYS) {
          WARN("[%s topo check] Abnormal ICN link type between PPU%d and PPU%d: ICN%d",
                ppuTypeStr[ppuDevInfo->ppuType], cur, rem, ppuTopoArray[cur * ppuCount + rem]);
          return ncclInternalError;
        }
      } else {
        if (ppuTopoArray[cur * ppuCount + rem] == ICN1) {
          interDomainICN1++;
        } else if (ppuTopoArray[cur * ppuCount + rem] == ICN2) {
          interDomainICN2++;
        } else if (ppuTopoArray[cur * ppuCount + rem] != SYS) {
          WARN("[%s topo check] Abnormal ICN link type between PPU%d and PPU%d: ICN%d",
                ppuTypeStr[ppuDevInfo->ppuType], cur, rem, ppuTopoArray[cur * ppuCount + rem]);
          return ncclInternalError;
        }
      }
    }

    // two ICN1 + one ICN2 or one ICN1 + two ICN2 per PPU intra domain is normal
    if (intraDomainICN1 == 2 && intraDomainICN2 == 1) {
      intraDomainTwoICN1++;
    } else if (intraDomainICN1 == 1 && intraDomainICN2 == 2) {
      intraDomainTwoICN2++;
    }

    if (cur % PER_OAM_TAIL_NODE_DEV_NUM == (PER_OAM_TAIL_NODE_DEV_NUM / 2 - 1)
        || cur % PER_OAM_TAIL_NODE_DEV_NUM == (PER_OAM_TAIL_NODE_DEV_NUM - 1)) {
      // two ICN1 + two ICN2 inter domain is normal
      if (interDomainICN1 == 2 && interDomainICN2 == 2) {
        interDomainICN1 = 0;
        interDomainICN2 = 0;
      } else {
        WARN("[%s topo check] Abnormal ICN link count between domains in server %d: %d 'ICN1' and %d 'ICN2'",
              ppuTypeStr[ppuDevInfo->ppuType], serverId, interDomainICN1, interDomainICN2);
        return ncclInternalError;
      }

      // two PPU is 221 and two PPU is 211 intra domain is normal
      if (intraDomainTwoICN1 == 2 && intraDomainTwoICN2 == 2) {
        intraDomainTwoICN1 = 0;
        intraDomainTwoICN2 = 0;
      } else {
        WARN("[%s topo check] Abnormal ICN link count intra domains in server %d: %d PPUs have 2 'ICN1' and %d PPUs have 2 'ICN2'",
              ppuTypeStr[ppuDevInfo->ppuType], serverId, intraDomainTwoICN1, intraDomainTwoICN2);
        return ncclInternalError;
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t pcclCheckInterServer810E(int *ppuTopoArray, ppuDeviceInfo *ppuDevInfo) {
  int ppuCount = ppuDevInfo->ppuCount;
  for (int cur = 0; cur < ppuCount; cur++) {
    int serverId = 0;
    if (cur < PER_OAM_TAIL_NODE_DEV_NUM) serverId = 0;
    else serverId = 1;

    int interServerICN1 = 0;
    for (int rem = (1 - serverId) * PER_OAM_TAIL_NODE_DEV_NUM;
             rem < (2 - serverId) * PER_OAM_TAIL_NODE_DEV_NUM; rem++) {
      if (ppuTopoArray[cur * ppuCount + rem] == ICN1) {
        interServerICN1++;
      } else if (ppuTopoArray[cur * ppuCount + rem] != SYS) {
        WARN("[%s topo check] Abnormal ICN link type between PPU%d and PPU%d: ICN%d",
              ppuTypeStr[ppuDevInfo->ppuType], cur, rem, ppuTopoArray[cur * ppuCount + rem]);
        return ncclInternalError;
      }
    }

    if (interServerICN1 != 1) {
      WARN("[%s topo check] Abnormal 'ICN1' link count between PPU%d and other server: %d (must be 1)",
            ppuTypeStr[ppuDevInfo->ppuType], cur, interServerICN1);
      return ncclInternalError;
    }
  }
  return ncclSuccess;
}

ncclResult_t pcclCheckTopo610(int *ppuTopoArray, ppuDeviceInfo *ppuDevInfo) {
  int ppuCount = ppuDevInfo->ppuCount;
  for (int cur = 0; cur < ppuCount; cur++) {
    int interDomainICN2 = 0;
    for (int rem = 0; rem < ppuCount; rem++) {
      if (samePcieDomain(cur, rem)) {
        if (cur == rem) continue;
        if (ppuTopoArray[cur * ppuCount + rem] != ICN1) {
          WARN("[%s topo check] No 'ICN1' link between PPU%d and PPU%d", ppuTypeStr[ppuDevInfo->ppuType], cur, rem);
          return ncclInternalError;
        }
      } else {
        if (ppuTopoArray[cur * ppuCount + rem] == ICN2) {
          interDomainICN2++;
        } else if (ppuTopoArray[cur * ppuCount + rem] != SYS) {
          WARN("[%s topo check] Abnormal ICN link type between PPU%d and PPU%d: ICN%d",
                ppuTypeStr[ppuDevInfo->ppuType], cur, rem, ppuTopoArray[cur * ppuCount + rem]);
          return ncclInternalError;
        }
      }
    }

    if (interDomainICN2 != 1) {
      WARN("[%s topo check] Abnormal 'ICN2' link count between PPU%d and other domain: %d (must be 1)",
            ppuTypeStr[ppuDevInfo->ppuType], cur, interDomainICN2);
      return ncclInternalError;
    }
  }
  return ncclSuccess;
}

ncclResult_t pcclCheckTopo810(int *ppuTopoArray, ppuDeviceInfo *ppuDevInfo) {
  int ppuCount = ppuDevInfo->ppuCount;
  for (int cur = 0; cur < ppuCount; cur++) {
    int interDomainICN1 = 0;
    for (int rem = 0; rem < ppuCount; rem++) {
      if (samePcieDomain(cur, rem)) {
        if (cur == rem) continue;
        if (ppuTopoArray[cur * ppuCount + rem] != ICN1) {
          WARN("[%s topo check] No 'ICN1' link between PPU%d and PPU%d", ppuTypeStr[ppuDevInfo->ppuType], cur, rem);
          return ncclInternalError;
        }
      } else {
        if (ppuTopoArray[cur * ppuCount + rem] == ICN1) {
          interDomainICN1++;
        } else if (ppuTopoArray[cur * ppuCount + rem] != SYS) {
          WARN("[%s topo check] Abnormal ICN link type between PPU%d and PPU%d: ICN%d",
                ppuTypeStr[ppuDevInfo->ppuType], cur, rem, ppuTopoArray[cur * ppuCount + rem]);
          return ncclInternalError;
        }
      }
    }

    if (interDomainICN1 != 1) {
      WARN("[%s topo check] Abnormal 'ICN1' link count between PPU%d and other domain: %d (must be 1)",
            ppuTypeStr[ppuDevInfo->ppuType], cur, interDomainICN1);
      return ncclInternalError;
    }
  }
  return ncclSuccess;
}

ncclResult_t pcclCheckTopo890(int *ppuTopoArray, ppuDeviceInfo *ppuDevInfo) {
  int ppuCount = ppuDevInfo->ppuCount;
  for (int cur = 0; cur < ppuCount; cur++) {
    int interDomainICN2 = 0;
    for (int rem = 0; rem < ppuCount; rem++) {
      if (samePcieDomain(cur, rem)) {
        if (cur == rem) continue;
        if (ppuTopoArray[cur * ppuCount + rem] != ICN2) {
          WARN("[%s topo check] No 'ICN2' link between PPU%d and PPU%d", ppuTypeStr[ppuDevInfo->ppuType], cur, rem);
          return ncclInternalError;
        }
      } else {
        if (ppuTopoArray[cur * ppuCount + rem] == ICN2) {
          interDomainICN2++;
        } else if (ppuTopoArray[cur * ppuCount + rem] != SYS) {
          WARN("[%s topo check] Abnormal ICN link type between PPU%d and PPU%d: ICN%d",
                ppuTypeStr[ppuDevInfo->ppuType], cur, rem, ppuTopoArray[cur * ppuCount + rem]);
          return ncclInternalError;
        }
      }
    }

    if (interDomainICN2 != 1) {
      WARN("[%s topo check] Abnormal 'ICN2' link count between PPU%d and other domain: %d (must be 1)",
            ppuTypeStr[ppuDevInfo->ppuType], cur, interDomainICN2);
      return ncclInternalError;
    }
  }
  return ncclSuccess;
}

ncclResult_t pcclCheckSHMTopo(ppuDeviceInfo *ppuDevInfo, int linkNum) {
  int ppuCount = ppuDevInfo->ppuCount;
  for (unsigned int devId = 0; devId < ppuCount; devId++) {
    hgmlDevice_t hgmlDev = NULL;
    NCCLCHECK(pcclHgmlDeviceGetHandleByIndex(devId, &hgmlDev));

    for (int lindId = 0; lindId < linkNum; lindId++) {
      hgmlEnableState_t isActive;
      if ((pcclHgmlDeviceGetIcnLinkState(hgmlDev, lindId, &isActive) == ncclSuccess) && isActive == HGML_FEATURE_ENABLED) {
        hgmlIntIcnLinkDeviceType_t icnRmtType = HGML_ICNLINK_DEVICE_TYPE_UNKNOWN;

        if (pcclHgmlDeviceGetIcnLinkRemoteDeviceType(hgmlDev, lindId, &icnRmtType) != ncclSuccess ||
            icnRmtType != HGML_ICNLINK_DEVICE_TYPE_SWITCH) {
            WARN("[%s shm topo check] ICN links on PPU%d must connect to switch, but remote endpoint is not a switch",
                  ppuTypeStr[ppuDevInfo->ppuType], devId);
            return ncclInternalError;
        }
      } else {
        WARN("[%s shm topo check] Disabled ICN link(s) detected on PPU%d", ppuTypeStr[ppuDevInfo->ppuType], devId);
        return ncclInternalError;
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t pcclCheckTopo810E(int *ppuTopoArray, ppuDeviceInfo *ppuDevInfo) {
  ncclResult_t tmpRes = ncclSuccess;
  if (ppuDevInfo->ppuCount == PER_OAM_TAIL_NODE_DEV_NUM) {
    tmpRes = pcclCheckIntraServer810E(ppuTopoArray, ppuDevInfo, 0);
    if (tmpRes != ncclSuccess) {
      // for special 8*810E that topo similar to 8*810
      tmpRes = pcclCheckTopo810(ppuTopoArray, ppuDevInfo);
    }
  } else {
    tmpRes = pcclCheckIntraServer810E(ppuTopoArray, ppuDevInfo, 0);
    tmpRes = (tmpRes == ncclSuccess) ? pcclCheckIntraServer810E(ppuTopoArray, ppuDevInfo, 1) : tmpRes;
    tmpRes = (tmpRes == ncclSuccess) ? pcclCheckInterServer810E(ppuTopoArray, ppuDevInfo) : tmpRes;
  }
 return tmpRes;
}

ncclResult_t pcclCheckPPUTopo() {
  if (pcclParamCheckItemsEnable() == 0) return ncclSuccess;

  ppuDeviceInfo *ppuDevInfo = pcclGetPpuDevInfo();
  if (ppuDevInfo->ppuType == _PPU_UNKNOWN || ppuDevInfo->ppuCount == 0) {
    return ncclSuccess;
  }

  ncclResult_t tmpRes = ncclSuccess;
  int ppuCount = ppuDevInfo->ppuCount;
  int *ppuTopoArray = NULL;

  // Topo check is based on 8/16-PPU systems; non-standard configurations (e.g., isolation in Docker) are not supported for now.
  if (ppuCount != PER_OAM_TAIL_NODE_DEV_NUM && ppuCount != PER_OAM_TAIL_NODE_DEV_NUM * 2) {
    WARN("Abnormal device count: %d in %s, skip to check topo", ppuCount, ppuTypeStr[ppuDevInfo->ppuType]);
    goto clear;
  }

  // PPU-ZW810-H & PPU-ZW805-H will not be released for the time being
  // The topo of ZW-M530 has not been determined yet
  if (ppuDevInfo->icnRmtType == PCCL_ICNLINK_SWITCH) {
    if (ppuDevInfo->ppuType == _ZW_M890P || ppuDevInfo->ppuType == _ZW_M890L) {
      tmpRes = pcclCheckSHMTopo(ppuDevInfo, ICN8);
    } else {
      WARN("Skip to do shm topo check when ppu type is %s", ppuTypeStr[ppuDevInfo->ppuType]);
      goto clear;
    }
  } else if (ppuDevInfo->icnRmtType == PCCL_ICNLINK_PPU) {
    NCCLCHECK(ncclCalloc(&ppuTopoArray, ppuCount*ppuCount));
    tmpRes = pcclFillPPUTopoArray(ppuTopoArray, ppuDevInfo);

    if (ppuDevInfo->ppuType == _PPU_ZW810 || ppuDevInfo->ppuType == _PPU_ZW805) {
      tmpRes = (tmpRes == ncclSuccess) ? pcclCheckTopo810(ppuTopoArray, ppuDevInfo) : tmpRes;
    } else if (ppuDevInfo->ppuType == _PPU_ZW810E) {
      tmpRes = (tmpRes == ncclSuccess) ? pcclCheckTopo810E(ppuTopoArray, ppuDevInfo) : tmpRes;
    } else if (ppuDevInfo->ppuType == _PPU_ZW610 || ppuDevInfo->ppuType == _PPU_ZW610E) {
      tmpRes = (tmpRes == ncclSuccess) ? pcclCheckTopo610(ppuTopoArray, ppuDevInfo) : tmpRes;
    } else if (ppuDevInfo->ppuType == _ZW_M890P || ppuDevInfo->ppuType == _ZW_M890L) {
      tmpRes = (tmpRes == ncclSuccess) ? pcclCheckTopo890(ppuTopoArray, ppuDevInfo) : tmpRes;
    } else {
      WARN("Skip to do topo check when ppu type is %s", ppuTypeStr[ppuDevInfo->ppuType]);
      goto clear;
    }
  } else {
    WARN("Skip to do topo check when remote device type is unknown");
    goto clear;
  }

  if (tmpRes == ncclSuccess) {
    INFO(NCCL_INIT, "%s %stopo check success", ppuTypeStr[ppuDevInfo->ppuType], ppuDevInfo->icnRmtType == PCCL_ICNLINK_SWITCH ? "shm " : "");
  } else {
    WARN("%s topo check error, please check it", ppuTypeStr[ppuDevInfo->ppuType]);
  }

clear:
  if (ppuTopoArray) NCCLCHECK(ncclFree(ppuTopoArray));
  return ncclSuccess;
}
