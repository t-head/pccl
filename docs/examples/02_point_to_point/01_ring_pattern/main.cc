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
 * PCCL Ring Pattern Example - Educational Version
 *
 * This example demonstrates the fundamental ring communication pattern using
 * PCCL's point-to-point operations. Understanding ring patterns is essential
 * for PCCL programming as they form the basis of many collective algorithms.
 *
 * Learning Objectives:
 * - Understand ring topology and neighbor communication
 * - Learn PCCL point-to-point send/recv operations
 * - See how data flows in a ring pattern
 * - Practice deadlock avoidance with pcclGroup operations
 * - Understand single-process multi-GPU patterns
 *
 */

int main(int argc, char *argv[]) {
  // ========================================================================
  // STEP 1: Initialize Environment and Detect GPUs
  // ========================================================================

  int num_gpus = 0;
  pcclComm_t *comms = NULL;
  hggcStream_t *streams = NULL;
  float **h_sendbuff = NULL;
  float **h_recvbuff = NULL;
  float **d_sendbuff = NULL;
  float **d_recvbuff = NULL;

  printf("Starting PCCL ring communication example\n");

  // Get number of available HGGC devices
  HGGCCHECK(hggcGetDeviceCount(&num_gpus));

  if (num_gpus == 0) {
    fprintf(stderr, "No HGGC devices found\n");
    return 1;
  }

  if (num_gpus < 2) {
    printf("At least 2 GPU are necessary to create inter-GPU traffic\n");
    printf("Found only %d GPU(s) - pattern will be limited\n", num_gpus);
  }

  printf("Using %d GPUs for ring communication\n", num_gpus);

  // ========================================================================
  // STEP 2: Prepare Data Structures and Device List
  // ========================================================================

  printf("Preparing data structures\n");

  // Create device list (use all available devices)
  int *devices = (int *)malloc(num_gpus * sizeof(int));
  for (int i = 0; i < num_gpus; i++) {
    devices[i] = i;
  }

  // Allocate communicators, streams, and buffer pointers
  comms = (pcclComm_t *)malloc(num_gpus * sizeof(pcclComm_t));
  streams = (hggcStream_t *)malloc(num_gpus * sizeof(hggcStream_t));
  h_sendbuff = (float **)malloc(num_gpus * sizeof(float *));
  h_recvbuff = (float **)malloc(num_gpus * sizeof(float *));
  d_sendbuff = (float **)malloc(num_gpus * sizeof(float *));
  d_recvbuff = (float **)malloc(num_gpus * sizeof(float *));

  // ========================================================================
  // STEP 3: Initialize PCCL Communicators
  // ========================================================================

  /*
   * pcclCommInitAll is the simplest way to initialize PCCL communicators
   * for single-process, multi-GPU scenarios. It automatically:
   * - Creates one communicator per GPU
   * - Assigns ranks sequentially (GPU 0 = rank 0, GPU 1 = rank 1, etc.)
   */
  printf("Initializing PCCL communicators\n");
  PCCLCHECK(pcclCommInitAll(comms, num_gpus, devices));
  printf("All communicators initialized successfully\n");

  // ========================================================================
  // STEP 4: Create Streams and Verify Communicator Setup
  // ========================================================================

  printf("Creating HGGC streams and verifying setup\n");

  // Create streams and verify communicator info
  for (int i = 0; i < num_gpus; i++) {
    HGGCCHECK(hggcSetDevice(devices[i]));
    HGGCCHECK(hggcStreamCreate(&streams[i]));

    // Query communicator information for verification
    int rank, size, device;
    PCCLCHECK(pcclCommUserRank(comms[i], &rank));
    PCCLCHECK(pcclCommCount(comms[i], &size));
    PCCLCHECK(pcclCommCuDevice(comms[i], &device));

    printf("  GPU %d -> PCCL rank %d/%d on HGGC device %d\n", i, rank, size,
           device);
  }

  // ========================================================================
  // STEP 5: Set Up Ring Topology and Allocate Buffers
  // ========================================================================

  printf("Setting up ring topology\n");
  printf("Data flow -> GPU 0 -> ... -> GPU %d -> GPU 0\n", num_gpus - 1);

  // Test with 1GB of data
  const size_t count = 256 * 1024 * 1024; // 256M floats = 1GB
  const size_t size_bytes = count * sizeof(float);

  printf("Ring transfer with %zu elements (%.2f GB per GPU)\n", count,
         size_bytes / (1024.0 * 1024.0 * 1024.0));

  // Allocate buffers for each GPU
  printf("Allocating and initializing buffers\n");
  for (int i = 0; i < num_gpus; i++) {
    HGGCCHECK(hggcSetDevice(devices[i]));

    h_sendbuff[i] = (float *)malloc(size_bytes);
    h_recvbuff[i] = (float *)malloc(size_bytes);
    HGGCCHECK(hggcMalloc((void **)&d_sendbuff[i], size_bytes));
    HGGCCHECK(hggcMalloc((void **)&d_recvbuff[i], size_bytes));

    // Initialize data with GPU-specific pattern for verification
    for (size_t j = 0; j < count; j++) {
      h_sendbuff[i][j] = (float)(i * 1000 + j % 1000);
    }
    HGGCCHECK(hggcMemcpy(d_sendbuff[i], h_sendbuff[i], size_bytes,
                         hggcMemcpyHostToDevice));
  }

  // ========================================================================
  // STEP 6: Execute Ring Communication Pattern
  // ========================================================================

  /*
   * The ring communication uses pcclGroup operations to avoid deadlock.
   * Without grouping, if all GPUs tried to send first, they would deadlock
   * waiting for receivers. Grouping allows PCCL to execute operations
   * in the optimal order.
   */
  printf("Executing ring communication\n");

  // NOTE: pcclGroupStart and pcclGroupEnd are essential to avoid deadlock
  // when using pcclCommInitAll!
  PCCLCHECK(pcclGroupStart());
  for (int i = 0; i < num_gpus; i++) {
    int next = (i + 1) % num_gpus;
    int prev = (i - 1 + num_gpus) % num_gpus;
    printf("  GPU %d sends to GPU %d, receives from GPU %d\n", i, next, prev);

    // Each GPU simultaneously sends to next and receives from previous
    PCCLCHECK(
        pcclSend(d_sendbuff[i], count, pcclFloat, next, comms[i], streams[i]));
    PCCLCHECK(
        pcclRecv(d_recvbuff[i], count, pcclFloat, prev, comms[i], streams[i]));
  }
  PCCLCHECK(pcclGroupEnd());

  // Synchronize all streams to ensure communication completes
  for (int i = 0; i < num_gpus; i++) {
    HGGCCHECK(hggcSetDevice(devices[i]));
    HGGCCHECK(hggcStreamSynchronize(streams[i]));
  }

  printf("Ring communication completed successfully\n");

  // ========================================================================
  // STEP 7: Verify Data Correctness and Report Results
  // ========================================================================

  printf("Verifying data correctness\n");
  bool all_correct = true;

  for (int i = 0; i < num_gpus; i++) {
    HGGCCHECK(hggcSetDevice(devices[i]));
    HGGCCHECK(hggcMemcpy(h_recvbuff[i], d_recvbuff[i], size_bytes,
                         hggcMemcpyDeviceToHost));

    int prev = (i - 1 + num_gpus) % num_gpus;
    // Verify that GPU i received data from GPU prev
    float expected = (float)(prev * 1000);
    bool correct = (h_recvbuff[i][0] == expected);

    printf("  GPU %d received data from GPU %d: %s\n", i, prev,
           correct ? "CORRECT" : "ERROR");

    if (!correct) {
      all_correct = false;
      printf("  Expected %.0f, got %.0f\n", expected, h_recvbuff[i][0]);
    }
  }

  if (all_correct) {
    printf("SUCCESS - All GPUs received correct data\n");
  } else {
    printf("FAILURE - Data verification failed\n");
  }

  // ========================================================================
  // STEP 8: Cleanup Resources
  // ========================================================================

  printf("Cleaning up resources\n");

  // Free buffers
  for (int i = 0; i < num_gpus; i++) {
    HGGCCHECK(hggcSetDevice(devices[i]));
    free(h_sendbuff[i]);
    free(h_recvbuff[i]);
    HGGCCHECK(hggcFree(d_sendbuff[i]));
    HGGCCHECK(hggcFree(d_recvbuff[i]));
  }

  // Destroy communicators and streams
  for (int i = 0; i < num_gpus; i++) {
    PCCLCHECK(pcclCommFinalize(comms[i]));
    PCCLCHECK(pcclCommDestroy(comms[i]));
    HGGCCHECK(hggcSetDevice(devices[i]));
    HGGCCHECK(hggcStreamDestroy(streams[i]));
  }

  // Free allocated memory
  free(devices);
  free(comms);
  free(streams);
  free(h_sendbuff);
  free(h_recvbuff);
  free(d_sendbuff);
  free(d_recvbuff);

  printf("Example completed successfully!\n");

  return 0;
}
