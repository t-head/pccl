/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *************************************************************************/
#include "gin_test_common.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>

// One device segment followed by the local group's shared storage shards.
// SCM_RIGHTS provides real descriptor transfer without pidfd_getfd privileges.
struct SharedStorage {
  Allocation gpu;
  Allocation own;
  HGdeviceptr base = 0;
  std::vector<HGmemGenericAllocationHandle> handles;

  SharedStorage(bool host, int stride, int localRank, MPI_Comm group)
      : gpu(false, 8 * 1024 * 1024), own(host, 8 * 1024 * 1024), handles(stride) {
    REQUIRE(gpu.size == own.size);
    int fd;
    CHECK(hgMemExportToShareableHandle(&fd, own.handle, HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    int sock = socket(AF_UNIX, SOCK_DGRAM, 0);
    REQUIRE(sock >= 0);
    sockaddr_un address = {};
    address.sun_family = AF_UNIX;
    snprintf(address.sun_path, sizeof(address.sun_path), "/tmp/pccl-gin-storage-%d", getpid());
    REQUIRE(bind(sock, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);
    std::vector<int> pids(stride);
    int pid = getpid();
    CHECK(MPI_Allgather(&pid, 1, MPI_INT, pids.data(), 1, MPI_INT, group));
    for (int peer = 0; peer < stride; peer++) {
      sockaddr_un target = {};
      target.sun_family = AF_UNIX;
      snprintf(target.sun_path, sizeof(target.sun_path), "/tmp/pccl-gin-storage-%d", pids[peer]);
      iovec io = {&localRank, sizeof(localRank)};
      alignas(cmsghdr) char control[CMSG_SPACE(sizeof(fd))] = {};
      msghdr message = {};
      message.msg_name = &target;
      message.msg_namelen = sizeof(target);
      message.msg_iov = &io;
      message.msg_iovlen = 1;
      message.msg_control = control;
      message.msg_controllen = sizeof(control);
      cmsghdr* cmsg = CMSG_FIRSTHDR(&message);
      cmsg->cmsg_level = SOL_SOCKET;
      cmsg->cmsg_type = SCM_RIGHTS;
      cmsg->cmsg_len = CMSG_LEN(sizeof(fd));
      memcpy(CMSG_DATA(cmsg), &fd, sizeof(fd));
      REQUIRE(sendmsg(sock, &message, 0) == sizeof(localRank));
    }
    CHECK(hgMemAddressReserve(&base, gpu.size + stride * own.size, gpu.size, 0, 0));
    CHECK(hgMemMap(base, gpu.size, 0, gpu.handle, 0));
    for (int peer = 0; peer < stride; peer++) {
      int slot = -1;
      iovec io = {&slot, sizeof(slot)};
      alignas(cmsghdr) char control[CMSG_SPACE(sizeof(fd))] = {};
      msghdr message = {};
      message.msg_iov = &io;
      message.msg_iovlen = 1;
      message.msg_control = control;
      message.msg_controllen = sizeof(control);
      REQUIRE(recvmsg(sock, &message, 0) == sizeof(slot));
      REQUIRE(slot >= 0 && slot < stride && handles[slot] == 0);
      cmsghdr* cmsg = CMSG_FIRSTHDR(&message);
      REQUIRE(cmsg && cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS);
      int importedFd;
      memcpy(&importedFd, CMSG_DATA(cmsg), sizeof(importedFd));
      CHECK(hgMemImportFromShareableHandle(&handles[slot], reinterpret_cast<void*>((uintptr_t)importedFd),
          HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
      REQUIRE(close(importedFd) == 0);
      CHECK(hgMemMap(reinterpret_cast<HGdeviceptr>(shard(slot)), own.size, 0, handles[slot], 0));
    }
    REQUIRE(close(fd) == 0 && close(sock) == 0);
    REQUIRE(unlink(address.sun_path) == 0);
    int device;
    CHECK(hggcGetDevice(&device));
    HGmemAccessDesc access = {};
    access.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = HG_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK(hgMemSetAccess(base, gpu.size + stride * own.size, &access, 1));
    if (host) {
      int numa;
      CHECK(hgDeviceGetAttribute(&numa, HG_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device));
      access.location.type = HG_MEM_LOCATION_TYPE_HOST_NUMA;
      access.location.id = numa < 0 ? 0 : numa;
      CHECK(hgMemSetAccess(base + gpu.size, stride * own.size, &access, 1));
    }
  }

  void* shard(int slot) const { return reinterpret_cast<void*>(base + gpu.size + slot * own.size); }

  ~SharedStorage() {
    CHECK(hgMemUnmap(base, gpu.size));
    for (size_t slot = 0; slot < handles.size(); slot++) {
      CHECK(hgMemUnmap(reinterpret_cast<HGdeviceptr>(shard(slot)), own.size));
      CHECK(hgMemRelease(handles[slot]));
    }
    CHECK(hgMemAddressFree(base, gpu.size + handles.size() * own.size));
  }
};

__global__ void transfer(pcclDevComm comm, pcclWindow_t main, pcclWindow_t* windows, int stride, bool get) {
  if (threadIdx.x != 0) return;
  pcclCoopThread coop;
  int peer = (comm.rank + comm.nRanks / 2) % comm.nRanks;
  for (unsigned context = 0; context < comm.ginContextCount; context++) {
    pcclGin gin(comm, context);
    for (int shard = 0; shard < stride; shard++) {
      if (get) {
        gin.get(pcclTeamWorld(comm), peer, windows[shard], context * chunkBytes,
            main, (context * stride + shard) * chunkBytes, chunkBytes, coop,
            pcclGin_None{}, pcclGinOptFlagsDefault, pcclGin_SegmentMixed{});
      } else {
        size_t offset = ((comm.rank % stride) * comm.ginContextCount + context) * chunkBytes;
        gin.put(pcclTeamWorld(comm), peer, windows[shard], offset, main, context * chunkBytes, chunkBytes,
            pcclGin_None{}, pcclGin_None{}, coop, pcclGin_None{}, hggc::thread_scope_thread,
            hggc::thread_scope_system, pcclGinOptFlagsDefault, pcclGin_SegmentMixed{});
      }
    }
    gin.flush(coop);
  }
  pcclGinBarrierSession<pcclCoopThread> barrier(coop, pcclGinAllContexts(comm), pcclTeamTagWorld{}, 0);
  barrier.sync(coop, hggc::memory_order_acq_rel, pcclGinFenceLevel::Put | pcclGinFenceLevel::Get);
}

int main(int argc, char** argv) {
  CHECK(MPI_Init(&argc, &argv));
  int rank, ranks, localRank, localSize;
  CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
  MPI_Comm local, group;
  CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
  CHECK(MPI_Comm_rank(local, &localRank));
  CHECK(MPI_Comm_size(local, &localSize));
  bool host = argc > 1 && atoi(argv[1]) != 0;
  int stride = argc > 2 ? atoi(argv[2]) : 1;
  REQUIRE(stride > 0 && stride <= 4 && localSize % stride == 0 && (ranks / 2) % stride == 0);
  CHECK(MPI_Comm_split(local, localRank / stride, localRank % stride, &group));
  CHECK(hggcSetDevice(localRank));
  pcclUniqueId id;
  if (rank == 0) CHECK(pcclGetUniqueId(&id));
  CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  pcclComm_t comm;
  CHECK(pcclCommInitRank(&comm, ranks, id, rank));
  pcclDevCommRequirements req = PCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  req.ginConnectionType = PCCL_GIN_CONNECTION_CUSTOM_STRIDE;
  req.ginCustomStride = stride;
  req.ginContextCount = 3;
  req.worldGinBarrierCount = 1;
  pcclDevComm deviceComm;
  CHECK(pcclDevCommCreate(comm, &req, &deviceComm));
  {
    SharedStorage memory(host, stride, localRank % stride, group);
    size_t intsPerChunk = chunkBytes / sizeof(int);
    size_t count = stride * deviceComm.ginContextCount * intsPerChunk;
    REQUIRE(count * sizeof(int) <= memory.own.size && count * sizeof(int) <= memory.gpu.size);
    std::vector<int> input(count), output(count);
    for (size_t i = 0; i < count; i++) input[i] = rank * 100000 + i;
    CHECK(hggcMemcpy(memory.own.ptr(), input.data(), count * sizeof(int), hggcMemcpyHostToDevice));
    CHECK(MPI_Barrier(MPI_COMM_WORLD));
    pcclWindow_t main;
    CHECK(pcclCommWindowRegister(comm, reinterpret_cast<void*>(memory.base), memory.gpu.size,
        &main, PCCL_WIN_STRICT_ORDERING));
    std::vector<pcclWindow_t> windows(stride);
    for (int shard = 0; shard < stride; shard++) {
      CHECK(pcclCommWindowRegister(comm, memory.shard(shard), memory.own.size,
          &windows[shard], PCCL_WIN_GIN_ONLY));
    }
    pcclWindow_t alias;
    CHECK(pcclCommWindowRegister(comm, memory.shard(0), memory.own.size, &alias, PCCL_WIN_GIN_ONLY));
    CHECK(pcclCommWindowDeregister(comm, windows[0]));
    windows[0] = alias;
    pcclWindow_t* deviceWindows;
    CHECK(hggcMalloc(&deviceWindows, stride * sizeof(pcclWindow_t)));
    CHECK(hggcMemcpy(deviceWindows, windows.data(), stride * sizeof(pcclWindow_t), hggcMemcpyHostToDevice));
    int remoteBase = ((rank + ranks / 2) % ranks) / stride * stride;
    transfer<<<1, 32>>>(deviceComm, main, deviceWindows, stride, true);
    CHECK(hggcGetLastError());
    CHECK(hggcDeviceSynchronize());
    CHECK(hggcMemcpy(output.data(), reinterpret_cast<void*>(memory.base), count * sizeof(int), hggcMemcpyDeviceToHost));
    for (unsigned context = 0; context < deviceComm.ginContextCount; context++) {
      for (int shard = 0; shard < stride; shard++) {
        for (size_t i = 0; i < intsPerChunk; i++) {
          REQUIRE(output[(context * stride + shard) * intsPerChunk + i] ==
              (remoteBase + shard) * 100000 + context * intsPerChunk + i);
        }
      }
    }
    CHECK(MPI_Barrier(MPI_COMM_WORLD));
    CHECK(hggcMemcpy(reinterpret_cast<void*>(memory.base), input.data(), count * sizeof(int), hggcMemcpyHostToDevice));
    transfer<<<1, 32>>>(deviceComm, main, deviceWindows, stride, false);
    CHECK(hggcGetLastError());
    CHECK(hggcDeviceSynchronize());
    for (int shard = 0; shard < stride; shard++) {
      CHECK(hggcMemcpy(output.data(), memory.shard(shard), count * sizeof(int), hggcMemcpyDeviceToHost));
      for (int local = 0; local < stride; local++) {
        for (unsigned context = 0; context < deviceComm.ginContextCount; context++) {
          for (size_t i = 0; i < intsPerChunk; i++) {
            REQUIRE(output[(local * deviceComm.ginContextCount + context) * intsPerChunk + i] ==
                (remoteBase + local) * 100000 + context * intsPerChunk + i);
          }
        }
      }
    }
    CHECK(hggcFree(deviceWindows));
    for (auto window : windows) CHECK(pcclCommWindowDeregister(comm, window));
    CHECK(pcclCommWindowDeregister(comm, main));
  }
  CHECK(pcclDevCommDestroy(comm, &deviceComm));
  CHECK(pcclCommDestroy(comm));
  printf("GIN_STORAGE_PASS rank=%d host=%d stride=%d\n", rank, host, stride);
  CHECK(MPI_Comm_free(&group));
  CHECK(MPI_Comm_free(&local));
  CHECK(MPI_Finalize());
}
