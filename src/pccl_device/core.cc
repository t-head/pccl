/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "core.h"
#include "comm.h"
#include "pccl_device/impl/core__funcs.h"

NCCL_API(pcclTeam_t, pcclTeamWorld, ncclComm_t comm);
pcclTeam_t pcclTeamWorld(ncclComm_t comm) {
  pcclTeam_t ans;
  ans.nRanks = comm->nRanks;
  ans.rank = comm->rank;
  ans.stride = 1;
  return ans;
}

NCCL_API(pcclTeam_t, pcclTeamLsa, ncclComm_t comm);
pcclTeam_t pcclTeamLsa(ncclComm_t comm) {
  // Ignoring errors since if it fails ncclDevrInitOnce will try again.
  // The returned team will be junk and the next "interesting" API call that
  // needs ncclDevrInitOnce will report the error.
  if (ncclSuccess != ncclDevrInitOnce(comm)) return pcclTeam_t{};

  pcclTeam_t ans;
  ans.nRanks = comm->devrState.lsaSize;
  ans.rank = comm->devrState.lsaSelf;
  ans.stride = 1;
  return ans;
}

NCCL_API(pcclTeam_t, pcclTeamRail, ncclComm_t comm);
pcclTeam_t pcclTeamRail(ncclComm_t comm) {
  // Ignoring errors as above.
  if (ncclSuccess != ncclDevrInitOnce(comm)) return pcclTeam_t{};

  pcclTeam_t ans;
  ans.nRanks = comm->nRanks/comm->devrState.lsaSize;
  ans.rank = comm->rank/comm->devrState.lsaSize;
  ans.stride = comm->devrState.lsaSize;
  return ans;
}

NCCL_API(int, pcclTeamRankToWorld, ncclComm_t comm, pcclTeam_t team, int rank);
int pcclTeamRankToWorld(ncclComm_t comm, pcclTeam_t team, int rank) {
  return comm->rank + (rank - team.rank)*team.stride;
}

NCCL_API(int, pcclTeamRankToLsa, ncclComm_t comm, pcclTeam_t team, int rank);
int pcclTeamRankToLsa(ncclComm_t comm, pcclTeam_t team, int rank) {
  // Ignoring errors as above.
  if (ncclSuccess != ncclDevrInitOnce(comm)) return -1;

  return comm->devrState.lsaSelf + (rank - team.rank)*team.stride;
}
