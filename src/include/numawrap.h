/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_NUMAWRAP_H_
#define NCCL_NUMAWRAP_H_

#include "pccl_internal.h"

struct bitmask {
  unsigned long size; /* number of bits in the map */
  unsigned long *maskp;
};

ncclResult_t wrap_numa_symbols(void);
ncclResult_t wrap_numa_set_membind(struct bitmask *nodemask);
ncclResult_t wrap_numa_get_membind(struct bitmask **nodemask);
ncclResult_t wrap_numa_allocate_nodemask(struct bitmask **nodemask);
ncclResult_t wrap_numa_bitmask_setbit(struct bitmask *nodemask, unsigned int n, struct bitmask ** oldnodemask);
ncclResult_t wrap_numa_bitmask_free(struct bitmask *nodemask);
ncclResult_t wrap_numa_free_nodemask(struct bitmask *nodemask);

#endif
