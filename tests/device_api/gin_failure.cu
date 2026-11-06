/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *************************************************************************/
#include "gin_test_common.h"
#include <cstring>

int main(int argc, char** argv) {
  CHECK(MPI_Init(&argc, &argv));
  int rank, ranks, localRank;
  CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
  MPI_Comm local;
  CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
  CHECK(MPI_Comm_rank(local, &localRank));
  CHECK(hggcSetDevice(localRank));
  pcclUniqueId id;
  if (rank == 0) CHECK(pcclGetUniqueId(&id));
  CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  pcclComm_t comm;
  CHECK(pcclCommInitRank(&comm, ranks, id, rank));
  pcclDevCommRequirements req = PCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  req.ginConnectionType = PCCL_GIN_CONNECTION_FULL;
  req.ginContextCount = 1;
  pcclDevComm deviceComm;
  const char* mode = getenv("PCCL_GIN_TEST_MODE");
  REQUIRE(mode);
  if (strcmp(mode, "bad-version") == 0 || strcmp(mode, "partial-progress") == 0) {
    REQUIRE(pcclDevCommCreate(comm, &req, &deviceComm) == pcclInvalidUsage);
  } else {
    if (strcmp(mode, "retry") == 0) {
      REQUIRE(pcclDevCommCreate(comm, &req, &deviceComm) == pcclSystemError);
    }
    CHECK(pcclDevCommCreate(comm, &req, &deviceComm));
    CHECK(pcclDevCommDestroy(comm, &deviceComm));
  }
  CHECK(pcclCommDestroy(comm));
  printf("GIN_FAILURE_PASS rank=%d mode=%s\n", rank, mode);
  CHECK(MPI_Comm_free(&local));
  CHECK(MPI_Finalize());
}
