/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "hggc_runtime.h"
#include "pccl.h"
#include "pccl_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/*
 * PCCL AllReduce Example - Collective Communication
 *
 * This example demonstrates the fundamental AllReduce collective operation
 * using PCCL's single-process, multi-GPU approach. AllReduce is one of the most
 * important collective operations in distributed and parallel computing.
 *
 * Learning Objectives:
 * - Understand AllReduce collective communication pattern
 * - Learn PCCL single-process multi-GPU programming model
 * - See how data reduction works across multiple devices
 * - Practice verification and validation of collective results
 *
 */

int main(int argc, char *argv[]) {
  // ========================================================================
  // STEP 1: Initialize Variables and Detect Available GPUs
  // ========================================================================

  int num_gpus = 0;
  pcclComm_t *comms;
  hggcStream_t *streams;
  int *devices;
  float **sendbuff;
  float **recvbuff;

  // Get number of HGGC devices
  HGGCCHECK(hggcGetDeviceCount(&num_gpus));
  if (num_gpus < 1) {
    printf("No HGGC devices found\n");
    return EXIT_FAILURE;
  }

  printf("Using %d devices for collective communication\n", num_gpus);

  // ========================================================================
  // STEP 2: Allocate Memory for Communicators, Streams, and Data Buffers
  // ========================================================================

  // Allocate arrays for per-device resources, and array of pointers for buffers
  comms = (pcclComm_t *)malloc(num_gpus * sizeof(pcclComm_t));
  streams = (hggcStream_t *)malloc(num_gpus * sizeof(hggcStream_t));
  devices = (int *)malloc(num_gpus * sizeof(int));
  sendbuff = (float **)malloc(num_gpus * sizeof(float *));
  recvbuff = (float **)malloc(num_gpus * sizeof(float *));
  for (int i = 0; i < num_gpus; i++) devices[i] = i;

  printf("Memory allocated for %d communicators and streams\n", num_gpus);

  // ========================================================================
  // STEP 3: Initialize PCCL Communicators for All Devices
  // ========================================================================

  // pcclCommInitAll creates communicators for all devices in one call
  // This is the simplest way to set up PCCL for single-process applications
  PCCLCHECK(pcclCommInitAll(comms, num_gpus, devices));
  printf("PCCL communicators initialized for all devices\n");

  // ========================================================================
  // STEP 4: Create HGGC Streams and Allocate Device Memory
  // ========================================================================

  const size_t size = 32 * 1024 * 1024; // 32M floats for demonstration

  for (int i = 0; i < num_gpus; i++) {
    // Set device context for each GPU
    HGGCCHECK(hggcSetDevice(i));

    // Create stream for asynchronous operations
    HGGCCHECK(hggcStreamCreate(&streams[i]));

    // Allocate device memory for send and receive buffers
    HGGCCHECK(hggcMalloc((void **)&sendbuff[i], size * sizeof(float)));
    HGGCCHECK(hggcMalloc((void **)&recvbuff[i], size * sizeof(float)));

    // Initialize send buffer: zero the entire buffer, then set first element to
    // rank
    HGGCCHECK(hggcMemset(sendbuff[i], 0, size * sizeof(float)));
    float rank_value = (float)i;
    HGGCCHECK(hggcMemcpy(sendbuff[i], &rank_value, sizeof(float),
                         hggcMemcpyHostToDevice));

    printf("  Device %d initialized with data value %d\n", i, i);
  }

  // ========================================================================
  // STEP 5: Perform AllReduce Sum Operation
  // ========================================================================

  printf("Starting collective sum operation across all devices\n");

  // NOTE: pcclGroupStart and pcclGroupEnd are essential to avoid
  // deadlock when using pcclCommInitAll and multiple communication calls.
  PCCLCHECK(pcclGroupStart());
  for (int i = 0; i < num_gpus; i++) {
    // Each device performs combines all contributions and distributes result
    PCCLCHECK(pcclAllReduce(sendbuff[i], recvbuff[i], size, pcclFloat, pcclSum,
                            comms[i], streams[i]));
  }
  PCCLCHECK(pcclGroupEnd());

  // Synchronize all streams to ensure completion
  for (int i = 0; i < num_gpus; i++) {
    HGGCCHECK(hggcSetDevice(i));
    HGGCCHECK(hggcStreamSynchronize(streams[i]));
  }

  printf("Collective operation completed\n");

  // ========================================================================
  // STEP 6: Verify Results and Validate Correctness
  // ========================================================================

  // Expected result: sum of all ranks = 0 + 1 + 2 + ... + (num_gpus-1)
  // Note: We only check the first element since that's all we initialized
  float expected = (float)(num_gpus * (num_gpus - 1) / 2);
  printf("Verifying results (expected sum: %.0f)\n", expected);

  bool success = true;
  for (int i = 0; i < num_gpus; i++) {
    float result;
    HGGCCHECK(hggcSetDevice(i));
    HGGCCHECK(hggcMemcpy(&result, recvbuff[i], sizeof(float),
                         hggcMemcpyDeviceToHost));

    if (result != expected) {
      printf("  Device %d received incorrect result: %.0f (expected %.0f)\n", i,
             result, expected);
      success = false;
    } else {
      printf("  Device %d correctly received sum: %.0f\n", i, result);
    }
  }

  // ========================================================================
  // STEP 7: Cleanup Resources and Report Results
  // ========================================================================

  // Destroy PCCL communicators
  for (int i = 0; i < num_gpus; i++) {
    PCCLCHECK(pcclCommFinalize(comms[i]));
    PCCLCHECK(pcclCommDestroy(comms[i]));
  }

  // Free device memory and destroy streams
  for (int i = 0; i < num_gpus; i++) {
    HGGCCHECK(hggcSetDevice(i));
    HGGCCHECK(hggcFree(sendbuff[i]));
    HGGCCHECK(hggcFree(recvbuff[i]));
    HGGCCHECK(hggcStreamDestroy(streams[i]));
  }

  // Free host memory
  free(comms);
  free(streams);
  free(devices);
  free(sendbuff);
  free(recvbuff);

  if (success) {
    printf("Example completed successfully!\n");
  } else {
    printf("Example failed - incorrect results detected\n");
    return EXIT_FAILURE;
  }

  return 0;
}
