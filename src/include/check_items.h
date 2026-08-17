/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_CHECK_ITEMS_H_
#define NCCL_CHECK_ITEMS_H_

#include "pccl_internal.h"

#define SYS 0
#define ICN1 1
#define ICN2 2
#define ICN8 8

#define ICN_LINK_SPEED_25 25
#define ICN_LINK_SPEED_50 50
#define ICN_LINK_SPEED_53 53

#define PPU_DEVICE_NAME_LEN 16
#define PER_OAM_TAIL_NODE_DEV_NUM 8
#define PER_PCI_DOMAIN_DEV_NUM (PER_OAM_TAIL_NODE_DEV_NUM / 2)

typedef enum : int{
    _PPU_TYPE_ANY = -1,
    _PPU_UNKNOWN = 0, // default
    _PPU_ZW610 = 1,
    _PPU_ZW810 = 2,
    _PPU_ZW805 = 3,
    _PPU_ZW610E = 4,
    _PPU_ZW810E = 5,
    _PPU_ZW810_H = 6,
    _PPU_ZW805_H = 7,
    _ZW_M890P = 8,
    _ZW_M890L = 9,
    _ZW_M530 = 10,
    _ZW_M530P = 11,
    _PPU_TYPE_COUNT
} PPU_TYPE;

typedef enum : int {
    PCCL_ICNLINK_ANY = -1,
    PCCL_ICNLINK_UNKNOWN = 0,
    PCCL_ICNLINK_PPU = 1,
    PCCL_ICNLINK_SWITCH = 2
} pcclICNRemoteDeviceType_t;

struct ppuDeviceInfo {
    PPU_TYPE ppuType;
    pcclICNRemoteDeviceType_t icnRmtType;
    int sm;
    int ppuCount;
};

extern const char* ppuTypeStr[];
extern const char* icnRmtTypeStr[];

ncclResult_t pcclCheckICNLinkSpeed(int devId, int linkId, unsigned int bandwidth);
ncclResult_t pcclCheckPPUTopo();
struct ppuDeviceInfo* pcclGetPpuDevInfo();

#endif
