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
 * PCCL Device API Pure GIN AlltoAll Example
 *
 * This example demonstrates PCCL's GPU-Initiated Networking (GIN) capabilities
 * for performing AlltoAll collective operations directly from GPU kernels using
 * only network-based communication.
 * GIN enables GPU kernels to initiate network communication without CPU
 * intervention, providing low-latency communication for distributed applications.
 *
 * Learning Objectives:
 * - Understand pure GIN (GPU-Initiated Networking) communication
 * - Learn how to use pcclGin for device-initiated network communication
 * - See pure GIN AlltoAll implementation for network-based communication
 * - Practice GIN barriers and signal-based synchronization
 *
 * Key GIN Concepts:
 * - pcclGin: Device-side networking object for kernel-initiated communication
 * - GIN contexts: Network communication channels for parallel operations
 * - GIN signals: Completion notifications for asynchronous operations
 * - GIN barriers: Network-based synchronization across ranks
 * - One-sided put operations: Direct remote memory writes over network
 *
 * When to Use Pure GIN:
 * - Communication between ranks that cannot use LSA (different nodes)
 * - Network-based collective operations in multi-node environments
 * - Scenarios where all communication must go through the network
 * - Testing network performance without local optimizations
 *
 * Performance Considerations:
 * - GIN provides network communication from GPU kernels
 * - All communication goes through the network (no local optimizations)
 * - Signal-based completion detection enables asynchronous operation
 * - Multiple GIN contexts can improve parallel communication performance
 *
 * Device setup (this example):
 * - worldGinBarrierCount and ginSignalCount: one per CTA (grid dimension x).
 * - ginConnectionType PCCL_GIN_CONNECTION_FULL: full GIN connectivity (each rank to all peers).
 * - pcclGinBarrierSession for cross-rank barriers before the puts.
 * - A single GIN context (index 0) is used for simplicity; production code may
 *   use multiple contexts for throughput.
 *
 * Kernel notes:
 * - signalIndex is blockIdx.x; all puts targeting a given rank use the same
 *   signal index, so only the CTA receivingCta must waitSignal for nRanks.
 * - After flush, a release pcclGinBarrierSession::sync pairs with the acquire
 *   at entry so ranks do not exit the kernel before the collective completes.
 */

// Grid width (CTAs). Must match reqs.worldGinBarrierCount and reqs.ginSignalCount.
#define PCCL_DEVICE_CTA_COUNT 16
#define PCCL_DEVICE_THREADS_PER_CTA 512

// ==========================================================================
// Device Kernel Implementations
// ==========================================================================

// Pure GIN alltoall: one put per (src rank, peer) partitioned across threads.
template <typename T>
__global__ void PureGinAlltoAllKernel(pcclWindow_t sendwin, size_t sendoffset,
                                      pcclWindow_t recvwin, size_t recvoffset,
                                      size_t count, struct pcclDevComm devComm) {
  int ginContext = 0; // single context for simplicity
  unsigned int signalIndex = blockIdx.x;
  pcclGin gin { devComm, ginContext };
  uint64_t signalValue = gin.readSignal(signalIndex);

  pcclGinBarrierSession<pcclCoopCta> bar { pcclCoopCta(), gin, pcclTeamTagWorld(), blockIdx.x };
  bar.sync(pcclCoopCta(), hggc::memory_order_acquire, pcclGinFenceLevel::Relaxed);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  const size_t size = count * sizeof(T);
  for (int r = tid; r < devComm.nRanks; r += nthreads) {
    gin.put(pcclTeamWorld(devComm), r,
        recvwin, recvoffset + devComm.rank * size,
        sendwin, sendoffset + r * size,
        size, pcclGin_SignalInc{signalIndex});
  }

  // Wait only on the CTA whose blockIdx.x (signalIndex) accumulates all puts to this rank.
  int receivingCta = (devComm.rank % nthreads) / blockDim.x;
  if (blockIdx.x == receivingCta)
    gin.waitSignal(pcclCoopCta(), signalIndex, signalValue + devComm.nRanks);

  gin.flush(pcclCoopCta());
  bar.sync(pcclCoopCta(), hggc::memory_order_release, pcclGinFenceLevel::Relaxed);
}

// ==========================================================================
// Host-Side Setup and Device API Initialization
// ==========================================================================

void* pureGinAlltoAll(int my_rank, int total_ranks, int local_device, int devices_per_rank) {
  pcclComm_t comm;
  pcclUniqueId pccl_unique_id;

  if (my_rank == 0) {
    printf("Starting Pure GIN AlltoAll initialization\n");
  }

  // Standard PCCL communicator initialization
  if (my_rank == 0) {
    PCCLCHECK(pcclGetUniqueId(&pccl_unique_id));
  }

  // Distribute unique ID
  util_broadcast(0, my_rank, &pccl_unique_id);

  // Set device context for this rank
  HGGCCHECK(hggcSetDevice(local_device));
  printf("  Rank %d using GPU device %d\n", my_rank, local_device);

  // ==========================================================================
  // STEP 2: Initialize PCCL Communicator and Allocate Memory
  // ==========================================================================

  // Initialize PCCL communicator
  PCCLCHECK(pcclCommInitRank(&comm, total_ranks, pccl_unique_id, my_rank));
  printf("  Rank %d initialized PCCL communicator for %d total ranks\n", my_rank, total_ranks);

  // Check for Device API and GIN support
  pcclCommProperties_t props = PCCL_COMM_PROPERTIES_INITIALIZER;
  PCCLCHECK(pcclCommQueryProperties(comm, &props));
  if (!props.deviceApiSupport) {
    printf("ERROR: rank %d communicator does not support Device API!\n", my_rank);
    PCCLCHECK(pcclCommFinalize(comm));
    PCCLCHECK(pcclCommDestroy(comm));
    return NULL;
  }
  if (props.ginType == PCCL_GIN_TYPE_NONE) {
    printf("ERROR: rank %d communicator does not support GIN!\n", my_rank);
    PCCLCHECK(pcclCommFinalize(comm));
    PCCLCHECK(pcclCommDestroy(comm));
    return NULL;
  }

  // Allocate memory for AlltoAll operation
  size_t count = 1024; // Elements per rank
  size_t total_elements = count * total_ranks;
  size_t size_bytes = total_elements * sizeof(float);

  float *h_sendbuff = (float*)malloc(size_bytes);
  float *h_recvbuff = (float*)malloc(size_bytes);
  void* d_sendbuff;
  void* d_recvbuff;
  pcclWindow_t send_win;
  pcclWindow_t recv_win;

  // Device API requires symmetric memory allocation
  PCCLCHECK(pcclMemAlloc(&d_sendbuff, size_bytes));
  PCCLCHECK(pcclMemAlloc(&d_recvbuff, size_bytes));

  // ==========================================================================
  // STEP 3: Register Memory Windows for Device-Side Access
  // ==========================================================================

  // Register symmetric windows for GIN access
  PCCLCHECK(pcclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win, PCCL_WIN_COLL_SYMMETRIC));
  PCCLCHECK(pcclCommWindowRegister(comm, d_recvbuff, size_bytes, &recv_win, PCCL_WIN_COLL_SYMMETRIC));

  // Initialize data: each rank sends unique values to each destination
  for (size_t i = 0; i < total_elements; i++) {
    int dest_rank = i / count;
    int element_idx = i % count;
    h_sendbuff[i] = (float)(my_rank * 1000 + dest_rank * 100 + element_idx);
  }
  HGGCCHECK(hggcMemcpy(d_sendbuff, h_sendbuff, size_bytes, hggcMemcpyHostToDevice));
  printf("  Rank %d initialized send data\n", my_rank);

  // ==========================================================================
  // STEP 4: Create Device Communicator with GIN Support
  // ==========================================================================

  // Create stream for kernel execution
  hggcStream_t stream;
  HGGCCHECK(hggcStreamCreate(&stream));

  // Create device communicator with GIN support
  pcclDevComm devComm;
  pcclDevCommRequirements reqs = PCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.worldGinBarrierCount = PCCL_DEVICE_CTA_COUNT;  // pcclGinBarrierSession (world team)
  reqs.ginSignalCount = PCCL_DEVICE_CTA_COUNT; // one signal index per CTA (blockIdx.x)
  reqs.ginConnectionType = PCCL_GIN_CONNECTION_FULL;  // full GIN connectivity: each rank to all peers
  PCCLCHECK(pcclDevCommCreate(comm, &reqs, &devComm));
  printf("  Rank %d created device communicator with GIN support\n", my_rank);

  if (my_rank == 0) {
    printf("Starting Pure GIN AlltoAll with %zu elements per rank (%zu total elements, %zu MB)\n",
            count, total_elements, size_bytes / (1024 * 1024));
  }

  // ==========================================================================
  // STEP 5: Execute Pure GIN AlltoAll Kernel
  // ==========================================================================

  if (my_rank == 0) {
    printf("\n=== Executing Pure GIN AlltoAll ===\n");
  }

  // Clear receive buffer
  HGGCCHECK(hggcMemset(d_recvbuff, 0, size_bytes));

  pcclxKernelDispatchConfig_t dispatch_config = PCCLX_KERNEL_DISPATCH_CONFIG_INITIALIZER;
  PCCLCHECK(PCCLX_LAUNCH_CONFIG(PureGinAlltoAllKernel<float>, &dispatch_config));

  // Launch pure GIN AlltoAll kernel
  PureGinAlltoAllKernel<float><<<PCCL_DEVICE_CTA_COUNT, PCCL_DEVICE_THREADS_PER_CTA, 0, stream>>>(
      send_win, 0, recv_win, 0, count, devComm);

  // Wait for completion
  HGGCCHECK(hggcStreamSynchronize(stream));
  printf("  Rank %d completed pure GIN AlltoAll kernel\n", my_rank);

  // ==========================================================================
  // STEP 6: Verify Results
  // ==========================================================================

  // Verify pure GIN results
  HGGCCHECK(hggcMemcpy(h_recvbuff, d_recvbuff, size_bytes, hggcMemcpyDeviceToHost));

#if 0
  // Debug: print first few elements at every rank's slot in recvbuf
  printf("  Rank %d recvbuff dump:\n", my_rank);
  for (int sr = 0; sr < total_ranks; sr++) {
    printf("    [src=%d] recvbuff[%zu..%zu]:", sr, sr*count, sr*count+3);
    for (int j = 0; j < 4; j++) printf(" %.0f", h_recvbuff[sr*count+j]);
    printf("\n");
  }
  // Also: did sendbuff get overwritten?
  float h_sendcheck[4];
  HGGCCHECK(hggcMemcpy(h_sendcheck, d_sendbuff, sizeof(h_sendcheck), hggcMemcpyDeviceToHost));
  printf("  Rank %d sendbuff[0..3]:", my_rank);
  for (int j = 0; j < 4; j++) printf(" %.0f", h_sendcheck[j]);
  printf("\n");
#endif

  bool gin_success = true;
  for (int src_rank = 0; src_rank < total_ranks; src_rank++) {
    for (size_t i = 0; i < count; i++) {
      size_t recv_idx = src_rank * count + i;
      float expected = (float)(src_rank * 1000 + my_rank * 100 + i);
      if (h_recvbuff[recv_idx] != expected) {
        gin_success = false;
        printf("  Rank %d: Pure GIN mismatch at [%d][%zu]: got %.0f, expected %.0f\n",
                my_rank, src_rank, i, h_recvbuff[recv_idx], expected);
        break;
      }
    }
    if (!gin_success) break;
  }

  if (my_rank == 0) {
    printf("Pure GIN AlltoAll result: %s\n", gin_success ? "PASSED" : "FAILED");
  }

  // ==========================================================================
  // STEP 7: Cleanup Resources
  // ==========================================================================

  // Cleanup host memory
  free(h_sendbuff);
  free(h_recvbuff);

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
  return run_example(argc, argv, pureGinAlltoAll);
}
