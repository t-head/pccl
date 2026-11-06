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
#include <cstdlib>
#include <vector>

#define CHECK(call) do { \
  int status = static_cast<int>(call); \
  if (status != 0) { \
    fprintf(stderr, "%s:%d %s returned %d\n", __FILE__, __LINE__, #call, status); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while (0)

#define REQUIRE(condition) do { \
  if (!(condition)) { \
    fprintf(stderr, "%s:%d requirement failed: %s\n", __FILE__, __LINE__, #condition); \
    MPI_Abort(MPI_COMM_WORLD, 2); \
  } \
} while (0)

constexpr size_t chunkBytes = 4096;

struct Allocation {
  HGmemGenericAllocationHandle handle = 0;
  HGdeviceptr address = 0;
  size_t size = 0;

  Allocation(bool host, size_t requested) {
    int device = 0;
    CHECK(hggcGetDevice(&device));
    HGmemAllocationProp prop = {};
    prop.type = HG_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = host ? HG_MEM_LOCATION_TYPE_HOST_NUMA : HG_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleTypes = HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    if (host) {
      int numa = 0;
      CHECK(hgDeviceGetAttribute(&numa, HG_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device));
      prop.location.id = numa < 0 ? 0 : numa;
    }
    prop.allocFlags.gpuDirectRDMACapable = host ? 0 : 1;
    size_t granularity = 0;
    CHECK(hgMemGetAllocationGranularity(&granularity, &prop, HG_MEM_ALLOC_GRANULARITY_MINIMUM));
    size = ((requested + granularity - 1) / granularity) * granularity;
    CHECK(hgMemCreate(&handle, size, &prop, 0));
    CHECK(hgMemAddressReserve(&address, size, granularity, 0, 0));
    CHECK(hgMemMap(address, size, 0, handle, 0));
    HGmemAccessDesc access = {};
    access.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = HG_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK(hgMemSetAccess(address, size, &access, 1));
    if (host) {
      access.location = prop.location;
      CHECK(hgMemSetAccess(address, size, &access, 1));
      hggcPointerAttributes attributes = {};
      CHECK(hggcPointerGetAttributes(&attributes, ptr()));
      REQUIRE(attributes.type == hggcMemoryTypeHost);
    }
  }

  void* ptr() const { return reinterpret_cast<void*>(address); }

  ~Allocation() {
    CHECK(hgMemUnmap(address, size));
    CHECK(hgMemAddressFree(address, size));
    CHECK(hgMemRelease(handle));
  }
};

template<bool Host>
__global__ void exchange(pcclDevComm comm, pcclWindow_t source, pcclWindow_t destination, bool get) {
  if (threadIdx.x != 0) return;
  pcclCoopThread coop;
  int peer = (comm.rank + comm.nRanks / 2) % comm.nRanks;
  for (unsigned context = 0; context < comm.ginContextCount; context++) {
    pcclGin gin(comm, context);
    size_t offset = context * chunkBytes;
    if (get) {
      if constexpr (Host) {
        gin.get(pcclTeamWorld(comm), peer, source, offset, destination, offset, chunkBytes,
            coop, pcclGin_None{}, pcclGinOptFlagsDefault, pcclGin_SegmentHostNuma{});
      } else {
        gin.get(pcclTeamWorld(comm), peer, source, offset, destination, offset, chunkBytes);
      }
    } else {
      if constexpr (Host) {
        gin.put(pcclTeamWorld(comm), peer, destination, offset, source, offset, chunkBytes,
            pcclGin_None{}, pcclGin_None{}, coop, pcclGin_None{}, hggc::thread_scope_thread,
            hggc::thread_scope_system, pcclGinOptFlagsDefault, pcclGin_SegmentHostNuma{});
      } else {
        gin.put(pcclTeamWorld(comm), peer, destination, offset, source, offset, chunkBytes);
      }
    }
    gin.flush(coop);
    pcclGinBarrierSession<pcclCoopThread> barrier(coop, gin, pcclTeamTagWorld{}, 0);
    barrier.sync(coop, hggc::memory_order_acq_rel, pcclGinFenceLevel::Relaxed);
  }
}

int main(int argc, char** argv) {
  CHECK(MPI_Init(&argc, &argv));
  int rank = 0, ranks = 0;
  CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
  REQUIRE(ranks >= 2 && ranks % 2 == 0);
  MPI_Comm local;
  CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
  int localRank = 0;
  CHECK(MPI_Comm_rank(local, &localRank));
  CHECK(hggcSetDevice(localRank));
  bool host = argc > 1 && atoi(argv[1]) != 0;
  pcclUniqueId id;
  if (rank == 0) CHECK(pcclGetUniqueId(&id));
  CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  pcclComm_t comm;
  CHECK(pcclCommInitRank(&comm, ranks, id, rank));
  {
    Allocation source(host, 64 * chunkBytes), destination(host, 64 * chunkBytes);
    pcclWindow_t sourceWindow, destinationWindow, alias;
    CHECK(pcclCommWindowRegister(comm, source.ptr(), source.size, &sourceWindow, PCCL_WIN_GIN_ONLY));
    CHECK(pcclCommWindowRegister(comm, destination.ptr(), destination.size, &destinationWindow, PCCL_WIN_GIN_ONLY));
    CHECK(pcclCommWindowRegister(comm, source.ptr(), source.size, &alias, PCCL_WIN_GIN_ONLY));
    void* peer = nullptr;
    REQUIRE(pcclGetLsaDevicePointer(sourceWindow, 0, 0, &peer) == pcclInvalidUsage);
    pcclWindow_t invalid = nullptr;
    REQUIRE(pcclCommWindowRegister(comm, source.ptr(), source.size, &invalid,
        PCCL_WIN_GIN_ONLY | PCCL_WIN_COLL_SYMMETRIC) == pcclInvalidArgument);
    pcclDevCommRequirements req = PCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    req.ginConnectionType = PCCL_GIN_CONNECTION_FULL;
    req.ginContextCount = 3;
    req.worldGinBarrierCount = 1;
    pcclDevComm deviceComm;
    CHECK(pcclDevCommCreate(comm, &req, &deviceComm));
    if (host) {
      pcclWindow_vidmem window;
      CHECK(hggcMemcpy(&window, sourceWindow, sizeof(window), hggcMemcpyDeviceToHost));
      pcclSegmentWindow segment;
      CHECK(hggcMemcpy(&segment, window.ginMultiSegmentWins, sizeof(segment), hggcMemcpyDeviceToHost));
      REQUIRE(segment.memType == HG_MEM_LOCATION_TYPE_HOST_NUMA);
      REQUIRE(window.ginWins[0] != nullptr);
    }
    REQUIRE(deviceComm.ginContextCount * chunkBytes <= source.size);
    std::vector<int> input(deviceComm.ginContextCount * chunkBytes / sizeof(int));
    std::vector<int> output(input.size());
    for (size_t i = 0; i < input.size(); i++) input[i] = rank * 100000 + i + 1;
    CHECK(hggcMemcpy(source.ptr(), input.data(), input.size() * sizeof(int), hggcMemcpyHostToDevice));
    for (int get = 0; get < 2; get++) {
      CHECK(hggcMemset(destination.ptr(), 0, output.size() * sizeof(int)));
      CHECK(MPI_Barrier(MPI_COMM_WORLD));
      if (host) exchange<true><<<1, 32>>>(deviceComm, sourceWindow, destinationWindow, get);
      else exchange<false><<<1, 32>>>(deviceComm, sourceWindow, destinationWindow, get);
      CHECK(hggcGetLastError());
      CHECK(hggcDeviceSynchronize());
      CHECK(hggcMemcpy(output.data(), destination.ptr(), output.size() * sizeof(int), hggcMemcpyDeviceToHost));
      int sourceRank = (rank + ranks / 2) % ranks;
      for (size_t i = 0; i < output.size(); i++) REQUIRE(output[i] == sourceRank * 100000 + i + 1);
    }
    CHECK(pcclCommWindowDeregister(comm, alias));
    CHECK(pcclCommWindowDeregister(comm, sourceWindow));
    CHECK(pcclCommWindowDeregister(comm, destinationWindow));
    CHECK(pcclDevCommDestroy(comm, &deviceComm));
  }
  CHECK(pcclCommDestroy(comm));
  printf("GIN_WINDOW_PASS rank=%d host=%d\n", rank, host);
  CHECK(MPI_Comm_free(&local));
  CHECK(MPI_Finalize());
}
