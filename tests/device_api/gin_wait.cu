/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *************************************************************************/
#include <mpi.h>
#include <hggc.h>
#include <hggc_runtime.h>
#include <pccl.h>
#include <pccl_device.h>
#include <cstdio>

#define CHECK(call) do { int status = (int)(call); if (status) { \
  fprintf(stderr, "%d: %s returned %d\n", __LINE__, #call, status); MPI_Abort(MPI_COMM_WORLD, 1); \
} } while (0)

template<class Coop>
__device__ void checkTimeouts(Coop coop, pcclDevComm comm, pcclWindow_t window, int* statuses) {
  pcclGin gin(comm, 0);
  statuses[threadIdx.x * 6] = gin.waitSignal(coop, 0, 0, 64, hggc::memory_order_acquire, 0);
  statuses[threadIdx.x * 6 + 1] = gin.waitSignal(coop, 0, 1, 64, hggc::memory_order_acquire, 10000);
  statuses[threadIdx.x * 6 + 2] = gin.waitCounter(coop, 0, 0, 56, hggc::memory_order_acquire, 0);
  statuses[threadIdx.x * 6 + 3] = gin.waitCounter(coop, 0, 1, 56, hggc::memory_order_acquire, 10000);
  statuses[threadIdx.x * 6 + 4] = gin.waitSignal(coop, window, 128, 1, 64, hggc::memory_order_acquire, 10000);
  statuses[threadIdx.x * 6 + 5] = gin.waitSignal(coop, 0, 0, 0, hggc::memory_order_acquire, 0);
}

__global__ void timeoutChecks(pcclDevComm comm, pcclWindow_t window, int mode, int* statuses) {
  if (mode == 0) checkTimeouts(pcclCoopThread{}, comm, window, statuses);
  else if (mode == 1) checkTimeouts(pcclCoopWarp{}, comm, window, statuses);
  else if (mode == 2) checkTimeouts(pcclCoopCta{}, comm, window, statuses);
  else if (mode == 3) checkTimeouts(pcclCoopTile<8>{}, comm, window, statuses);
  else if (mode == 4) checkTimeouts(pcclCoopLanes{~0u}, comm, window, statuses);
  else checkTimeouts(pcclCoopWarpSpan{0, 2, 0}, comm, window, statuses);
}

__global__ void delayedSignal(pcclDevComm comm, pcclWindow_t window, int* statuses) {
  if (threadIdx.x != 0) return;
  pcclGin gin(comm, 0);
  pcclCoopThread coop;
  int peer = (comm.rank + comm.nRanks / 2) % comm.nRanks;
  if (comm.rank >= comm.nRanks / 2) {
    uint64_t start = clock64();
    while (clock64() - start < 50000) {}
    gin.put(pcclTeamWorld(comm), peer, window, 64, window, 0, 8, pcclGin_SignalInc{0}, pcclGin_CounterInc{0});
  }
  statuses[0] = gin.waitSignal(coop, 0, 1, 64, hggc::memory_order_acquire, 1000000000ULL);
  if (comm.rank < comm.nRanks / 2) {
    gin.put(pcclTeamWorld(comm), peer, window, 64, window, 0, 8, pcclGin_SignalInc{0}, pcclGin_CounterInc{0});
  }
  statuses[1] = gin.waitCounter(coop, 0, 1, 56, hggc::memory_order_acquire, 1000000000ULL);
  gin.signal(pcclTeamWorld(comm), peer, pcclGin_VASignalAdd{window, 128, 1});
  statuses[2] = gin.waitSignal(coop, window, 128, 1, 64, hggc::memory_order_acquire, 1000000000ULL);
  gin.flush(coop);
}

int main(int argc, char** argv) {
  CHECK(MPI_Init(&argc, &argv));
  int rank, ranks, localRank;
  CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
  MPI_Comm local;
  CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
  CHECK(MPI_Comm_rank(local, &localRank));
  CHECK(hggcSetDevice(localRank));
  if (ranks < 2 || ranks % 2) MPI_Abort(MPI_COMM_WORLD, 2);
  pcclUniqueId id;
  if (rank == 0) CHECK(pcclGetUniqueId(&id));
  CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  pcclComm_t comm;
  CHECK(pcclCommInitRank(&comm, ranks, id, rank));
  pcclDevCommRequirements req = PCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  req.ginConnectionType = PCCL_GIN_CONNECTION_FULL;
  req.ginContextCount = 1;
  req.ginSignalCount = 1;
  req.ginCounterCount = 1;
  pcclDevComm deviceComm;
  CHECK(pcclDevCommCreate(comm, &req, &deviceComm));
  void* memory;
  CHECK(pcclMemAlloc(&memory, 65536));
  CHECK(hggcMemset(memory, 0, 65536));
  pcclWindow_t window;
  CHECK(pcclCommWindowRegister(comm, memory, 65536, &window, PCCL_WIN_GIN_ONLY));
  int* deviceStatuses;
  CHECK(hggcMalloc(&deviceStatuses, 64 * 6 * sizeof(int)));
  int statuses[64 * 6];
  int expected[] = {pcclSuccess, pcclTimeout, pcclSuccess, pcclTimeout, pcclTimeout, pcclInvalidArgument};
  for (int mode = 0; mode < 6; mode++) {
    timeoutChecks<<<1, 64>>>(deviceComm, window, mode, deviceStatuses);
    CHECK(hggcGetLastError());
    CHECK(hggcDeviceSynchronize());
    CHECK(hggcMemcpy(statuses, deviceStatuses, sizeof(statuses), hggcMemcpyDeviceToHost));
    for (int i = 0; i < 64 * 6; i++) {
      if (statuses[i] != expected[i % 6]) {
        fprintf(stderr, "mode=%d entry=%d got=%d expected=%d\n", mode, i, statuses[i], expected[i % 6]);
        MPI_Abort(MPI_COMM_WORLD, 3);
      }
    }
  }
  CHECK(MPI_Barrier(MPI_COMM_WORLD));
  delayedSignal<<<1, 32>>>(deviceComm, window, deviceStatuses);
  CHECK(hggcGetLastError());
  CHECK(hggcDeviceSynchronize());
  CHECK(hggcMemcpy(statuses, deviceStatuses, 3 * sizeof(int), hggcMemcpyDeviceToHost));
  for (int i = 0; i < 3; i++) if (statuses[i] != pcclSuccess) MPI_Abort(MPI_COMM_WORLD, 4);
  CHECK(hggcFree(deviceStatuses));
  CHECK(pcclCommWindowDeregister(comm, window));
  CHECK(pcclMemFree(memory));
  CHECK(pcclDevCommDestroy(comm, &deviceComm));
  CHECK(pcclCommDestroy(comm));
  printf("GIN_WAIT_PASS rank=%d\n", rank);
  CHECK(MPI_Comm_free(&local));
  CHECK(MPI_Finalize());
}
