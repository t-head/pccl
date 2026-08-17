/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef __HGGC_UUID_H__
#define __HGGC_UUID_H__

#include "hggc.h"

#define HG_UUID_CHAR(x) (char)((unsigned char)((x) & 0xff))

#define HG_DEFINE_UUID(var, timeLow, timeMid, timeHigh, varSeq, node) \
  static const HGuuid var =                                           \
  {                                                                   \
    {                                                                 \
      HG_UUID_CHAR((timeLow)  >> 24), HG_UUID_CHAR((timeLow)  >> 16), \
      HG_UUID_CHAR((timeLow)  >> 8 ), HG_UUID_CHAR((timeLow)  >> 0 ), \
      HG_UUID_CHAR((timeMid)  >> 8 ), HG_UUID_CHAR((timeMid)  >> 0 ), \
      HG_UUID_CHAR((timeHigh) >> 8 ), HG_UUID_CHAR((timeHigh) >> 0 ), \
      HG_UUID_CHAR((varSeq)   >> 8 ), HG_UUID_CHAR((varSeq)   >> 0 ), \
      HG_UUID_CHAR((node)     >> 40), HG_UUID_CHAR((node)     >> 32), \
      HG_UUID_CHAR((node)     >> 24), HG_UUID_CHAR((node)     >> 16), \
      HG_UUID_CHAR((node)     >> 8 ), HG_UUID_CHAR((node)     >> 0 ), \
    }                                                                 \
  };

#endif // __HGGC_UUID_H__