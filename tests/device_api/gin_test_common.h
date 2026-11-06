/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *************************************************************************/
#ifndef PCCL_GIN_TEST_COMMON_H_
#define PCCL_GIN_TEST_COMMON_H_
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


#endif
