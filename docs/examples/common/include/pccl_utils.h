/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_UTILS_H_
#define PCCL_UTILS_H_

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include "hggc_runtime.h"
#include "pccl.h"

// Error checking
#define PCCLCHECK(cmd)                                                         \
  do {                                                                         \
    pcclResult_t res = cmd;                                                    \
    if (res != pcclSuccess) {                                                  \
      fprintf(stderr, "Failed, PCCL error %s:%d '%s'\n", __FILE__, __LINE__,   \
              pcclGetErrorString(res));                                        \
      fprintf(stderr, "Failed PCCL operation: %s\n", #cmd);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define HGGCCHECK(cmd)                                                         \
  do {                                                                         \
    hggcError_t err = cmd;                                                     \
    if (err != hggcSuccess) {                                                  \
      fprintf(stderr, "Failed: HGGC error %s:%d '%s'\n", __FILE__, __LINE__,   \
              hggcGetErrorString(err));                                        \
      fprintf(stderr, "Failed HGGC operation: %s\n", #cmd);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#endif
