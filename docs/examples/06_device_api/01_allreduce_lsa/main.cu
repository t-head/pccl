/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "hggc_runtime.h"
#include "pccl.h"
#include "pccl_device.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

/*
 * PCCL Device API AllReduce Example
 *
 * This example demonstrates PCCL's Device API, which enables GPU kernels to
 * directly interact with PCCL without CPU intervention. This is particularly
 * powerful for applications that need to perform communication
 * from within HGGC kernels.
 *
 * Learning Objectives:
 * - Understand PCCL Device API vs Host API differences
 * - Learn how to register memory windows for device-side access
 * - See how GPU kernels can perform collective operations directly
 * - Practice LSA (Load Store Access) barrier synchronization
 *
 * Key Device API Concepts:
 * - pcclDevComm: Device-side communicator for kernel use (PCCL keeps the
 *   pcclDevComm name internally)
 * - pcclWindow_t: Memory windows that enable direct peer access
 * - LSA barriers: Synchronization primitives for device-side coordination
 * - pcclGetLsaPointer: Direct access to peer memory from device code
 *
 * When to Use Device API:
 * - Compute kernels that need immediate communication results
 * - Fusion of computation and communication in a single kernel
 * - Reduced host-device synchronization overhead
 * - Custom collective operations not available in standard PCCL
 *
 * Performance Considerations:
 * - Lower latency than host API for small operations
 * - Enables computation-communication overlap within kernels
 * - Requires careful synchronization and memory ordering
 * - LSA barriers add coordination overhead but enable correctness
 */

// Device API kernel launch configuration
// CTA count must match lsaBarrierCount for proper barrier synchronization
#define PCCL_DEVICE_CTA_COUNT 16
#define PCCL_DEVICE_THREADS_PER_CTA 512

// ==========================================================================
// Device Kernel Implementation
// ==========================================================================

// Device kernel that performs AllReduce sum operation
// This kernel demonstrates direct PCCL communication from GPU threads
__global__ void simpleAllReduceKernel(pcclWindow_t sendwin, size_t sendoffset,
                                      pcclWindow_t recvwin, size_t recvoffset,
                                      size_t count, struct pcclDevComm devComm) {
  // LSA barriers enable coordination between GPU threads across different ranks
  // Barrier scope: CTA (all threads in this block participate)
  // Barrier index: blockIdx.x selects this CTA's dedicated barrier (one barrier per CTA)
  pcclLsaBarrierSession<pcclCoopCta> bar { pcclCoopCta(), devComm, pcclTeamLsa(devComm),
                                           devComm.lsaBarrier, blockIdx.x };
  bar.sync(pcclCoopCta(), hggc::memory_order_acquire);

  const int rank = devComm.rank, nRanks = devComm.nRanks;

  // We are going to spread the workload across all GPU ranks.
  // So calculate the global thread ID across all ranks.
  // This maps global threads to data elements in the data to be reduced
  const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
  const int globalNthreads = blockDim.x * gridDim.x * nRanks;

  // Grid stride loop over all elements with the globalThreads
  for (size_t offset = globalTid; offset < count; offset += globalNthreads) {
    float v = 0.0f;
    // Access remote (and local [peer==rank]) memory and reduce locally
    for (int peer=0; peer<nRanks; peer++) {
      // Access peer memory directly using LSA (Load/Store Accessible) pointers
      float* sendPtr = (float*)pcclGetLsaPointer(sendwin, sendoffset, peer);
      v += sendPtr[offset];
    }
    // Write the result back to remote and local memory
    for (int peer=0; peer<nRanks; peer++) {
      float* recvPtr = (float*)pcclGetLsaPointer(recvwin, recvoffset, peer);
      recvPtr[offset] = v;
    }
  }
  // Release barrier ensures that we received data from everyone before we unblock the stream
  // and allow the next kernel(s) to process the data.
  // Critical for correctness in device-side collective operations
  bar.sync(pcclCoopCta(), hggc::memory_order_release);
}

// ==========================================================================
// Host-Side Setup and Device API Initialization
// ==========================================================================

// This function can be called inside an MPI rank or pthread thread. The
// initialization and broadcast are implemented in common/src/utils.cc for
// easier readability. For fully integrated examples using pthreads or MPI see
// examples. in 01_communicators.

void* allReduce(int my_rank, int total_ranks, int local_device, int devices_per_rank) {
  pcclComm_t comm;
  pcclUniqueId pccl_unique_id;

  if (my_rank == 0) {
    printf("Starting Device API AllReduce initialization\n");
  }

  // Standard PCCL communicator initialization (same as Host API)
  if (my_rank == 0) {
    PCCLCHECK(pcclGetUniqueId(&pccl_unique_id));
  }

  // Distribute unique ID in case of MPI.
  util_broadcast(0, my_rank, &pccl_unique_id);

  // Set device context for this rank
  HGGCCHECK(hggcSetDevice(local_device));
  printf("  Rank %d using GPU device %d\n", my_rank, local_device);

  // ==========================================================================
  // STEP 2: Initialize PCCL Communicator and Allocate Memory
  // ==========================================================================

  // Initialize PCCL communicator (same as Host API)
  PCCLCHECK(pcclCommInitRank(&comm, total_ranks, pccl_unique_id, my_rank));
  printf("  Rank %d initialized PCCL communicator for %d total ranks\n", my_rank, total_ranks);

  // Check for Device API support
  pcclCommProperties_t props = PCCL_COMM_PROPERTIES_INITIALIZER;
  PCCLCHECK(pcclCommQueryProperties(comm, &props));
  if (!props.deviceApiSupport) {
    printf("ERROR: rank %d communicator does not support Device API!\n", my_rank);
    PCCLCHECK(pcclCommFinalize(comm));
    PCCLCHECK(pcclCommDestroy(comm));
    return NULL;
  }
  // Pure LSA example requires a single team where all ranks can directly access each other
  if (props.nLsaTeams != 1) {
    printf("ERROR: rank %d communicator has %d LSA teams, expected 1 for pure LSA example!\n",
           my_rank, props.nLsaTeams);
    PCCLCHECK(pcclCommFinalize(comm));
    PCCLCHECK(pcclCommDestroy(comm));
    return NULL;
  }

  // Allocate memory for AllReduce operation
  size_t count = 1024 * 1024; // 1M elements
  size_t size_bytes = count * sizeof(float);

  float *h_data = (float*)malloc(size_bytes);
  void* d_sendbuff;
  void* d_recvbuff;
  pcclWindow_t send_win;
  pcclWindow_t recv_win;

  // Device API requires allocation compatible with symmetric memory allocation
  // This ensures memory can be accessed directly by device kernels from all ranks
  PCCLCHECK(pcclMemAlloc(&d_sendbuff, size_bytes));
  PCCLCHECK(pcclMemAlloc(&d_recvbuff, size_bytes));

  // ==========================================================================
  // STEP 4: Register Memory Windows for Device-Side Access
  // ==========================================================================

  // Register symmetric windows for LSA access
  // Windows enable direct peer-to-peer access from device kernels
  PCCLCHECK(pcclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win, PCCL_WIN_COLL_SYMMETRIC));
  PCCLCHECK(pcclCommWindowRegister(comm, d_recvbuff, size_bytes, &recv_win, PCCL_WIN_COLL_SYMMETRIC));

  // Initialize data with rank-specific values for verification
  for (size_t i = 0; i < count; i++) {
    h_data[i] = (float)my_rank;
  }
  HGGCCHECK(hggcMemcpy(d_sendbuff, h_data, size_bytes, hggcMemcpyHostToDevice));
  printf("  Rank %d initialized data with value %d\n", my_rank, my_rank);

  // ==========================================================================
  // STEP 5: Create Device Communicator and Configure LSA Barriers
  // ==========================================================================

  // Create stream for kernel execution
  hggcStream_t stream;
  HGGCCHECK(hggcStreamCreate(&stream));

  // Create device communicator - this is the key Device API component
  // Requirements specify resources to allocate (e.g., one barrier per CTA)
  pcclDevComm devComm;
  pcclDevCommRequirements reqs = PCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = PCCL_DEVICE_CTA_COUNT; // Must match kernel launch config
  PCCLCHECK(pcclDevCommCreate(comm, &reqs, &devComm));
  printf("  Rank %d created device communicator with %d LSA barriers\n", my_rank, PCCL_DEVICE_CTA_COUNT);

  if (my_rank == 0) {
    printf("Starting AllReduce with %zu elements (%zu MB) using Device API\n",
           count, size_bytes / (1024 * 1024));
    printf("Expected result: sum of ranks 0 to %d = %d per element\n",
           total_ranks - 1, (total_ranks * (total_ranks - 1)) / 2);
  }

  // ==========================================================================
  // STEP 6: Configure Kernel Dispatch and Launch
  // ==========================================================================

  pcclxKernelDispatchConfig_t dispatch_config = PCCLX_KERNEL_DISPATCH_CONFIG_INITIALIZER;
  PCCLCHECK(PCCLX_LAUNCH_CONFIG(simpleAllReduceKernel, &dispatch_config));

  // Launch device kernel to perform AllReduce
  // This kernel will directly access peer memory and perform collective operation
  simpleAllReduceKernel<<<PCCL_DEVICE_CTA_COUNT, PCCL_DEVICE_THREADS_PER_CTA, 0, stream>>>(send_win, 0, recv_win, 0, count, devComm);

  // Wait for completion - kernel performs AllReduce.
  HGGCCHECK(hggcStreamSynchronize(stream));
  printf("  Rank %d completed AllReduce kernel execution\n", my_rank);

  // ==========================================================================
  // STEP 7: Verify Results and Cleanup Resources
  // ==========================================================================

  // Verify results by copying back and checking
  HGGCCHECK(hggcMemcpy(h_data, d_recvbuff, size_bytes, hggcMemcpyDeviceToHost));
  float expected = (float)((total_ranks * (total_ranks - 1)) / 2);
  bool success = true;
  for (int i = 0; i < count; i++) {
    if (h_data[i] != expected) {
      success = false;
      break;
    }
  }

  if (my_rank == 0) {
    printf("AllReduce completed. Result verification: %s\n",
           success ? "PASSED" : "FAILED");
    if (success) {
      printf("All elements correctly sum to %.0f (ranks 0-%d)\n",
             expected, total_ranks - 1);
    }
  }

  // Cleanup resources in proper order
  free(h_data);

  // Device API specific cleanup
  PCCLCHECK(pcclDevCommDestroy(comm, &devComm));
  PCCLCHECK(pcclCommWindowDeregister(comm, send_win));
  PCCLCHECK(pcclCommWindowDeregister(comm, recv_win));
  PCCLCHECK(pcclMemFree(d_sendbuff));
  PCCLCHECK(pcclMemFree(d_recvbuff));

  // Standard PCCL cleanup
  PCCLCHECK(pcclCommFinalize(comm));
  PCCLCHECK(pcclCommDestroy(comm));
  HGGCCHECK(hggcStreamDestroy(stream));

  return NULL;
}

int main(int argc, char* argv[]) {
  // Run example using the provided utility framework
  return run_example(argc, argv, allReduce);
}
