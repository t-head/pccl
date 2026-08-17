/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "hggc_runtime.h"
#include "pccl.h"
#include "nccl_utils.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * PCCL Pthread Example - One Device Per Thread (Simple Version)
 *
 * This example demonstrates the basic lifecycle of PCCL communicators in a
 * multi-threaded environment. Each pthread manages one GPU device and shows
 * how to properly create and destroy PCCL communicators.
 *
 * Key Learning Points:
 * - PCCL communicator creation and destruction within threads
 * - HGGC stream management per thread
 * - Proper resource cleanup order
 *
 * This is a minimal example focusing purely on communicator lifecycle
 * management without performing actual collective operations.
 */

// Thread data structure to pass parameters
typedef struct {
  int thread_id;
  int num_gpus;
  pcclUniqueId commId;
  pcclComm_t *comms;
} threadData_t;

void *thread_worker(void *arg) {
  threadData_t *data = (threadData_t *)arg;
  int thread_id = data->thread_id;
  hggcStream_t stream;

  // =========================================================================
  // Set Device Context and Create Stream
  // =========================================================================
  // Each thread must set its device context before any HGGC operations
  HGGCCHECK(hggcSetDevice(thread_id));
  HGGCCHECK(hggcStreamCreate(&stream));

  printf("  Thread %d: Set device %d and created stream\n", thread_id,
         thread_id);

  // =========================================================================
  // Initialize PCCL Communicator
  // =========================================================================
  // Each thread creates its own communicator using the shared unique ID
  PCCLCHECK(pcclCommInitRank(&data->comms[thread_id], data->num_gpus, data->commId,
                             thread_id));

  printf("  Thread %d: PCCL communicator initialized\n", thread_id);

  if (thread_id == 0) {
    printf("All threads initialized - communicators ready\n");
  }

  // =========================================================================
  // Query Communicator Properties
  // =========================================================================
  // Verify the communicator was created correctly
  int comm_thread_id, comm_size;
  PCCLCHECK(pcclCommUserRank(data->comms[thread_id], &comm_thread_id));
  PCCLCHECK(pcclCommCount(data->comms[thread_id], &comm_size));

  printf("  Thread %d: Communicator thread_id %d of %d\n", thread_id,
         comm_thread_id, comm_size);

  // Synchronize HGGC stream to ensure all GPU work is complete
  if (stream != NULL) {
    HGGCCHECK(hggcStreamSynchronize(stream));
  }

  // =========================================================================
  // Cleanup Resources (Proper Order)
  // =========================================================================
  // Destroy PCCL communicator FIRST (before HGGC resources)
  // This is important - PCCL cleanup should happen before HGGC cleanup
  if (data->comms[thread_id] != NULL) {
    PCCLCHECK(pcclCommFinalize(data->comms[thread_id]));
    PCCLCHECK(pcclCommDestroy(data->comms[thread_id]));
    printf("  Thread %d: Destroyed PCCL communicator\n", comm_thread_id);
  }

  // Now destroy HGGC stream
  if (stream != NULL) {
    HGGCCHECK(hggcStreamDestroy(stream));
  }

  printf("  Thread %d: Resources cleaned up\n", thread_id);

  return NULL;
}

int main(int argc, char *argv[]) {
  int num_gpus;
  pthread_t *threads;
  threadData_t *threadData;
  pcclComm_t *comms;
  pcclUniqueId commId;

  // =========================================================================
  // STEP 1: Initialize Variables and Check GPU Availability
  // =========================================================================

  HGGCCHECK(hggcGetDeviceCount(&num_gpus));
  const char *nThreadsEnv = getenv("NTHREADS");
  if (nThreadsEnv) {
    num_gpus = atoi(nThreadsEnv);
  }

  if (num_gpus < 1) {
    printf("No HGGC devices found\n");
    return EXIT_FAILURE;
  }

  printf("Using %d devices with pthreads\n", num_gpus);

  // =========================================================================
  // STEP 2: Allocate Memory and Prepare Data Structures
  // =========================================================================

  threads = (pthread_t *)malloc(num_gpus * sizeof(pthread_t));
  threadData = (threadData_t *)malloc(num_gpus * sizeof(threadData_t));
  comms = (pcclComm_t *)malloc(num_gpus * sizeof(pcclComm_t));

  // Generate unique ID for PCCL communicator initialization
  PCCLCHECK(pcclGetUniqueId(&commId));

  // =========================================================================
  // STEP 3: Create and Launch Pthread Threads
  // =========================================================================

  printf("Creating %d threads for PCCL communicators\n", num_gpus);

  for (int i = 0; i < num_gpus; i++) {
    threadData[i].thread_id = i;
    threadData[i].num_gpus = num_gpus;
    threadData[i].commId = commId;
    threadData[i].comms = comms;

    pthread_create(&threads[i], NULL, thread_worker, &threadData[i]);
  }

  // =========================================================================
  // STEP 4: Wait for Thread Completion
  // =========================================================================

  for (int i = 0; i < num_gpus; i++) {
    pthread_join(threads[i], NULL);
  }

  printf("All threads completed\n");

  // =========================================================================
  // STEP 5: Cleanup Resources
  // =========================================================================

  free(threads);
  free(threadData);
  free(comms);

  printf("Success\n");
  return 0;
}
