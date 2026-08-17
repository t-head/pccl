/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_COMM_DUMPER_H_
#define PCCL_COMM_DUMPER_H_
#include "pccl_internal.h"

#define COMM_DUMP_MASK_BITS 4
typedef enum {
  DUMP_NONE = 0x0,
  DUMP_ROUGH = 0x1,
  DUMP_RAW = 0x2,
  DUMP_BUFFER = 0x4,
  DUMP_HWINFO = 0x8
} pcclCommDumpMaskType;

ncclResult_t setCommDumpMaskByStr(const char* usrMaskStr, pcclCommDumpMaskType* dumpMask);
ncclResult_t setCommDumpMaskByVal(const uint32_t usrMaskStr, pcclCommDumpMaskType* dumpMask);

#endif  // PCCL_COMM_DUMPER_H_
