/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_RAS_H_
#define NCCL_RAS_H_

#include "socket.h"

// Structure used to communicate data about NCCL ranks from NCCL threads to RAS.
struct rasRankInit {
  union ncclSocketAddress addr;
  pid_t pid;
  int hggcDev;
  int hgmlDev;
  uint64_t hostHash;
  uint64_t pidHash;
};

ncclResult_t ncclRasCommInit(ncclComm_t comm, struct rasRankInit* myRank);
ncclResult_t ncclRasCommFini(const ncclComm_t comm);
ncclResult_t ncclRasAddRanks(struct rasRankInit* ranks, int nranks);

#endif // !NCCL_RAS_H_
