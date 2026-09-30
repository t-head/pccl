/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "core.h"
#include "pccl_device/impl/gin_barrier__funcs.h"

NCCL_API(ncclResult_t, pcclGinBarrierCreateRequirement, ncclComm_t comm, pcclTeam_t team, int nBarriers, pcclGinBarrierHandle_t* outHandle, pcclDevResourceRequirements_t* outReq);
ncclResult_t pcclGinBarrierCreateRequirement(
    ncclComm_t comm, pcclTeam_t team, int nBarriers,
    pcclGinBarrierHandle_t* outHandle, pcclDevResourceRequirements_t* outReq
  ) {
  memset(outReq, 0, sizeof(*outReq));
  outReq->ginSignalCount = nBarriers * team.nRanks;
  outReq->outGinSignalStart = &outHandle->signal0;
  return ncclSuccess;
}
