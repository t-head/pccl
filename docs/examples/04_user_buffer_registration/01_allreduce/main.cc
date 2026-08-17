/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "hggc_runtime.h"
#include "pccl.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

/*
 * PCCL User Buffer Registration AllReduce Example
 *
 * This example demonstrates how to use PCCL's user buffer registration feature
 * to optimize performance for repeated collective operations on the same
 * buffers.
 *
 * Learning Objectives:
 * - Learn how to register and deregister buffers with PCCL communicators
 * - See the proper lifecycle management of registered buffers
 *
 */

/*
 * This function can be called inside an MPI rank or pthread thread. The
 * initialization and broadcast are implemented in common/src/utils.cc for
 * easier readability. For fully integrated examples using pthreads or MPI see
 * examples in 01_communicators.
 */
void *allReduce(int my_rank, int total_ranks, int local_device,
                int devices_per_rank) {

  // ========================================================================
  // STEP 1: Initialize PCCL Communicator and Setup
  // ========================================================================

  pcclUniqueId pccl_unique_id;
  if (my_rank == 0) {
    printf("Starting AllReduce example with %d ranks\n", total_ranks);
    PCCLCHECK(pcclGetUniqueId(&pccl_unique_id));
  }

  // Distribute unique ID.
  // This step ensures all ranks have the same unique ID for communicator
  // creation
  util_broadcast(0, my_rank, &pccl_unique_id);

  // Set device context for this rank
  // Each rank manages its assigned GPU device
  HGGCCHECK(hggcSetDevice(local_device));

  // Initialize PCCL communicator
  // This creates the communication context for collective operations
  pcclComm_t comm;
  PCCLCHECK(pcclCommInitRank(&comm, total_ranks, pccl_unique_id, my_rank));
  printf("  Rank %d communicator initialized using device %d\n", my_rank,
         local_device);

  // ========================================================================
  // STEP 2: Allocate Memory Using PCCL Allocator
  // ========================================================================

  if (my_rank == 0) {
    printf("User Buffer allocation:\n");
  }
  // Allocate memory - using larger buffers to demonstrate registration
  // benefits
  size_t count = 1024 * 1024; // 1M elements
  size_t size_bytes = count * sizeof(float);

  printf("  Rank %d allocating %.2f MB per buffer\n", my_rank,
         (float)size_bytes / (1024 * 1024));

  // Allocate buffers using PCCL allocator
  // PCCL's allocator can provide optimized memory for communication
  void *d_sendbuff;
  void *d_recvbuff;
  PCCLCHECK(pcclMemAlloc(&d_sendbuff, size_bytes));
  PCCLCHECK(pcclMemAlloc(&d_recvbuff, size_bytes));

  // ========================================================================
  // STEP 3: Register Buffers with PCCL Communicator
  // ========================================================================

  // Register the buffers with PCCL
  // This is the key optimization - buffers are pre-registered for efficiency
  // The handles returned can be used to identify registered buffers
  void *send_handle;
  void *recv_handle;
  PCCLCHECK(pcclCommRegister(comm, d_sendbuff, size_bytes, &send_handle));
  PCCLCHECK(pcclCommRegister(comm, d_recvbuff, size_bytes, &recv_handle));

  // ========================================================================
  // STEP 4: Initialize Data and Prepare for Communication
  // ========================================================================

  // Initialize data - each rank contributes its rank value
  // This creates a simple test pattern for verification
  float *h_data = (float *)malloc(size_bytes);
  for (size_t i = 0; i < count; i++) {
    h_data[i] = (float)my_rank;
  }
  HGGCCHECK(hggcMemcpy(d_sendbuff, h_data, size_bytes, hggcMemcpyHostToDevice));
  printf("  Rank %d data initialized (value: %d)\n", my_rank, my_rank);

  // Create stream for asynchronous operations
  // Streams allow overlapping computation and communication
  hggcStream_t stream;
  HGGCCHECK(hggcStreamCreate(&stream));

  // ========================================================================
  // STEP 5: Perform AllReduce Operation
  // ========================================================================

  if (my_rank == 0) {
    printf("Starting AllReduce with %zu elements (%zu MB)\n", count,
           size_bytes / (1024 * 1024));
  }

  // Perform AllReduce operation
  // Since buffers are registered, this should have optimized performance
  PCCLCHECK(pcclAllReduce(d_sendbuff, d_recvbuff, count, pcclFloat, pcclSum,
                          comm, stream));

  if (my_rank == 0) {
    printf("AllReduce completed successfully\n");
  }

  // ========================================================================
  // STEP 6: Verify Results and Validate Correctness
  // ========================================================================

  // Synchronize to ensure completion
  HGGCCHECK(hggcStreamSynchronize(stream));

  // Verify results (optional - copy back and check a few elements)
  float *h_result = (float *)malloc(sizeof(float) * count);
  HGGCCHECK(hggcMemcpy(h_result, d_recvbuff, sizeof(float) * count,
                       hggcMemcpyDeviceToHost));

  // Each element should be the sum of all ranks
  float expected_sum = (float)(total_ranks * (total_ranks - 1)) / 2;
  bool all_ok = true;
  if (my_rank == 0) {
    printf("Verification - Expected: %.1f, Got: %.1f\n", expected_sum,
           h_result[0]);

    for (size_t i = 1; i < count; i++) {
      if (fabsf(h_result[i] - expected_sum) > 0.001) {
        printf(" Results verification failed at index %zu: Expected %.1f, Got "
               "%.1f\n",
               i, expected_sum, h_result[i]);
        all_ok = false;
        break;
      }
    }

    if (all_ok) {
      printf("Results verified correctly\n");
    } else {
      printf("Results verification failed\n");
    }
  }

  // ========================================================================
  // STEP 7: Cleanup and Resource Management
  // ========================================================================

  // Important: Cleanup must happen in the correct order
  // 1. Free host memory
  // 2. Deregister buffers from communicator
  // 3. Free device memory
  // 4. Destroy HGGC resources
  // 5. Finalize and destroy PCCL communicator

  free(h_data);
  free(h_result);

  // Deregister buffers from communicator
  // This must happen before freeing the buffers or destroying the
  // communicator
  PCCLCHECK(pcclCommDeregister(comm, send_handle));
  PCCLCHECK(pcclCommDeregister(comm, recv_handle));
  printf("  Rank %d buffers deregistered\n", my_rank);

  // Free device memory allocated by PCCL
  PCCLCHECK(pcclMemFree(d_sendbuff));
  PCCLCHECK(pcclMemFree(d_recvbuff));

  // Finalize and destroy PCCL communicator
  PCCLCHECK(pcclCommFinalize(comm));
  PCCLCHECK(pcclCommDestroy(comm));

  // Destroy HGGC stream
  HGGCCHECK(hggcStreamDestroy(stream));

  if (my_rank == 0) {
    printf("All resources cleaned up successfully\n");
  }

  return NULL;
}

int main(int argc, char *argv[]) {
  // Run example using the standard test framework
  // This handles MPI/pthread initialization, device assignment, and cleanup
  return run_example(argc, argv, allReduce);
}
