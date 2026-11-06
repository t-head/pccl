/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *************************************************************************/
#include "../../src/include/devcomm/compat.h"
#include <mpi.h>
#include <hggc_runtime.h>
#include <cstdio>
#include <cstring>

#define CHECK(call) do { int status = (int)(call); if (status) { \
  fprintf(stderr, "%d: %s returned %d\n", __LINE__, #call, status); MPI_Abort(MPI_COMM_WORLD, 1); \
} } while (0)
#define REQUIRE(test) do { if (!(test)) { \
  fprintf(stderr, "%d: %s failed\n", __LINE__, #test); MPI_Abort(MPI_COMM_WORLD, 2); \
} } while (0)

template<class T> struct Guarded {
  T value;
  unsigned char canary[64];
  Guarded() { memset(this, 0xa5, sizeof(*this)); }
  void verify() { for (auto byte : canary) REQUIRE(byte == 0xa5); }
};

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
  Guarded<pcclCommProperties> props;
  props.value = PCCL_COMM_PROPERTIES_INITIALIZER;
  CHECK(pcclCommQueryProperties(comm, &props.value));
  props.verify();
  REQUIRE(props.value.rank == rank && props.value.nRanks == ranks);
  REQUIRE(props.value.ginMinStride > 0 && ranks % props.value.ginMinStride == 0);
  REQUIRE(props.value.ginSupport[props.value.railedGinType]);
  REQUIRE(props.value.devCommRuntimeVersionSize == sizeof(pcclDevComm));
  Guarded<pcclCommPropertiesLegacy> oldProps;
  memcpy(&oldProps.value, &props.value, sizeof(oldProps.value));
  oldProps.value.size = sizeof(oldProps.value);
  CHECK(pcclCommQueryProperties(comm, reinterpret_cast<pcclCommProperties*>(&oldProps.value)));
  oldProps.verify();
  props.value.size--;
  REQUIRE(pcclCommQueryProperties(comm, &props.value) == pcclInvalidUsage);

  pcclDevCommRequirements req = PCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  req.ginConnectionType = PCCL_GIN_CONNECTION_FULL;
  req.ginType = props.value.railedGinType;
  Guarded<pcclDevComm> current;
  CHECK(pcclDevCommCreate(comm, &req, &current.value));
  current.verify();
  CHECK(pcclDevCommDestroy(comm, &current.value));
  Guarded<pcclDevCommLegacy> old;
  pcclDevCommRequirementsLegacy oldReq;
  memcpy(&oldReq, &req, sizeof(oldReq));
  oldReq.size = sizeof(oldReq);
  CHECK(pcclDevCommCreate(comm, reinterpret_cast<pcclDevCommRequirements*>(&oldReq),
      reinterpret_cast<pcclDevComm*>(&old.value)));
  old.verify();
  CHECK(pcclDevCommDestroy(comm, reinterpret_cast<pcclDevComm*>(&old.value)));
  old.verify();
  req.ginType = props.value.railedGinType == PCCL_GIN_TYPE_GDAKI ? PCCL_GIN_TYPE_PROXY : PCCL_GIN_TYPE_GDAKI;
  REQUIRE(pcclDevCommCreate(comm, &req, &current.value) == pcclInvalidUsage);
  if (ranks > 1) {
    req.ginType = rank == 0 ? PCCL_GIN_TYPE_NONE : props.value.railedGinType;
    REQUIRE(pcclDevCommCreate(comm, &req, &current.value) == pcclInvalidUsage);
  }
  req = PCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  CHECK(pcclDevCommCreate(comm, &req, &current.value));
  REQUIRE(current.value.ginContextCount == 0);
  CHECK(pcclDevCommDestroy(comm, &current.value));
  CHECK(pcclCommDestroy(comm));
  printf("GIN_CONFIG_PASS rank=%d\n", rank);
  CHECK(MPI_Comm_free(&local));
  CHECK(MPI_Finalize());
}
