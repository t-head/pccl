/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-HEAD (SHANGHAI) SEMICONDUCTOR CO., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_KERNEL_ENABLE_H_
#define PCCL_KERNEL_ENABLE_H_

#include "pccl_common.h"

static_assert(NCCL_NUM_ALGORITHMS == 5,
  "kernelEnabled has exactly 5 algorithm rows (TREE/RING/COLLNET/FC/PAT); "
  "add the new rows and update this guard when the algorithm set changes");
static_assert(NCCL_NUM_PROTOCOLS == 7,
  "kernelEnabled has exactly 7 protocol columns; "
  "add the new columns and update this guard when the protocol set changes");

static constexpr int kernelEnabled[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {
  { /* Broadcast */
    /* Tree (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */    {0,  0,  0,  0,  0,  0,  0 },
    /* Ring (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */    {80, 80, 80, 80, 80, 80, 89},
    /* CollNet (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */ {0,  0,  0,  0,  0,  0,  0 },
    /* FC (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */      {80, 0,  80, 0,  80, 0,  0 },
    /* PAT (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */     {0,  0,  0,  0,  0,  0,  0 },
  },
  { /* Reduce */
    /* Tree (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */    {0,  0,  0,  0,  0,  0,  0 },
    /* Ring (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */    {80, 80, 80, 80, 80, 80, 89},
    /* CollNet (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */ {0,  0,  0,  0,  0,  0,  0 },
    /* FC (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */      {80, 0,  80, 0,  80, 0,  0 },
    /* PAT (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */     {0,  0,  0,  0,  0,  0,  0 },
  },
  { /* AllGather */
    /* Tree (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */    {0,  0,  0,  0,  0,  0,  0 },
    /* Ring (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */    {80, 80, 80, 80, 80, 80, 89},
    /* CollNet (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */ {0,  0,  0,  0,  0,  0,  0 },
    /* FC (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */      {80, 0,  80, 0,  80, 0,  0 },
    /* PAT (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */     {0,  0,  80, 0,  0,  0,  0 },
  },
  { /* ReduceScatter */
    /* Tree (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */    {0,  0,  0,  0,  0,  0,  0 },
    /* Ring (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */    {80, 80, 80, 80, 80, 80, 89},
    /* CollNet (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */ {0,  0,  0,  0,  0,  0,  0 },
    /* FC (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */      {80, 0,  80, 0,  80, 0,  0 },
    /* PAT (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */     {0,  0,  80, 0,  0,  0,  0 },
  },
  { /* AllReduce */
    /* Tree (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */    {80, 80, 80, 80, 80, 80, 0 },
    /* Ring (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */    {80, 80, 80, 80, 80, 80, 89},
    /* CollNet (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */ {80, 80, 80, 80, 80, 80, 0 },
    /* FC (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */      {80, 0,  80, 0,  80, 0,  0 },
    /* PAT (LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR) */     {0,  0,  0,  0,  0,  0,  0 },
  }
};

/* LL/LL128/Simple/Signal/SimpleBulk/SignalBulk/SimpleBulkAR */
static constexpr int p2pKernelEnabled[NCCL_NUM_PROTOCOLS] = {0, 0, 80, 80, 80, 80, 89};

#endif
