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
#include <unistd.h>

/*
 * PCCL Example: Multiple Devices Single Process
 * =============================================
 *
 * PURPOSE:
 * This example demonstrates how to initialize PCCL communicators for multiple
 * GPUs within a single process. This is the simplest PCCL setup and is ideal
 * for learning PCCL basics or for applications that want to use multiple GPUs
 * without the complexity of multi-process coordination.
 *
 * LEARNING OBJECTIVES:
 * - Learn how to use pcclCommInitAll() for simple multi-GPU setups
 * - See proper PCCL communicator lifecycle management
 * - Understand GPU device management in PCCL applications
 * - Learn proper resource cleanup patterns
 *
 * HOW IT WORKS:
 * 1. Detect all available HGGC devices
 * 2. Create communicators for all devices using pcclCommInitAll()
 * 3. Verify communicator properties (rank, size, device assignment)
 * 4. Clean up all resources properly
 *
 * KEY CONCEPTS:
 * - pcclCommInitAll(): Creates multiple communicators in a single call
 * - Single-process topology: All GPUs managed by one process
 * - Device management: Setting active HGGC device for operations
 * - Stream management: Each GPU gets its own HGGC stream
 *
 * WHEN TO USE THIS PATTERN:
 * - Learning PCCL fundamentals
 * - Single-node, multi-GPU applications
 * - Applications that don't need multi-node scaling
 * - Prototyping and testing PCCL functionality
 *
 * USAGE EXAMPLES:
 * ./multiple_devices_single_process               # Use all available GPUs
 *
 * EXPECTED OUTPUT:
 * - Detection of all available GPUs
 * - Successful communicator initialization
 * - Display of rank/size information for each GPU
 * - Clean resource cleanup confirmation
 */

// =============================================================================
// MAIN FUNCTION - PCCL Communicator Lifecycle Example
// =============================================================================

int main(int argc, char *argv[]) {
  // Variables for managing multiple GPU communicators
  int num_gpus;                 // Number of available HGGC devices
  pcclComm_t *comms = NULL;     // Array of PCCL communicators (one per GPU)
  hggcStream_t *streams = NULL; // Array of HGGC streams (one per GPU)
  int *devices = NULL;          // Array of device IDs to use

  // Discover how many HGGC devices are available
  // This determines how many PCCL communicators we'll create
  HGGCCHECK(hggcGetDeviceCount(&num_gpus));

  if (num_gpus == 0) {
    fprintf(stderr, "ERROR: No HGGC devices found on this system\n");
    fprintf(
        stderr,
        "Please ensure HGGC is properly installed and GPUs are available\n");
    return 1;
  }

  printf("Found %d HGGC device(s) available\n\n", num_gpus);

  // =========================================================================
  // STEP 1: Prepare Device Information and Memory Allocation
  // =========================================================================

  // Allocate arrays to hold our per-device resources
  // We need one communicator, stream, and device ID per GPU
  devices = (int *)malloc(num_gpus * sizeof(int));
  comms = (pcclComm_t *)malloc(num_gpus * sizeof(pcclComm_t));
  streams = (hggcStream_t *)malloc(num_gpus * sizeof(hggcStream_t));

  if (!devices || !comms || !streams) {
    fprintf(stderr, "ERROR: Failed to allocate memory for device arrays\n");
    return 1;
  }

  // Create device list and display device information
  // By default, we use all available devices (0, 1, 2, ...)
  printf("Available GPU devices:\n");
  for (int i = 0; i < num_gpus; i++) {
    devices[i] = i; // Use device i for communicator i

    // Query device properties for informational display
    hggcDeviceProp prop;
    HGGCCHECK(hggcGetDeviceProperties(&prop, devices[i]));
    printf("  GPU %d: %s (HGGC Device %d)\n", i, prop.name, devices[i]);
    printf("    Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("    Memory: %.1f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  }

  // Create an HGGC stream for each GPU
  // Each GPU needs its own stream for optimal performance
  for (int i = 0; i < num_gpus; i++) {
    // Set the active HGGC device before creating resources on it
    // This ensures the stream is created on the correct GPU
    HGGCCHECK(hggcSetDevice(devices[i]));
    HGGCCHECK(hggcStreamCreate(&streams[i]));
  }

  // =========================================================================
  // STEP 2 : Initialize PCCL Communicators
  // =========================================================================

  printf("Using pcclCommInitAll() to create all communicators "
         "simultaneously\n");

  // pcclCommInitAll() creates all communicators at once and handles the
  // coordination internally
  //
  // Parameters:
  // - comms: Array to store the created communicators
  // - num_gpus: Number of communicators to create
  // - devices: Array of HGGC device IDs to use
  //
  // After this call:
  // - comms[0] will be the communicator for devices[0] with rank 0
  // - comms[1] will be the communicator for devices[1] with rank 1
  // - ... and so on
  //
  // All communicators will have the same 'size' (total number of
  // participants)
  PCCLCHECK(pcclCommInitAll(comms, num_gpus, devices));
  printf("All %d PCCL communicators initialized successfully\n\n", num_gpus);

  // =========================================================================
  // STEP 3: Create HGGC Streams and Verify Communicator Properties
  // =========================================================================

  printf("Communicator Details:\n");

  bool sizes_match = true;
  for (int i = 0; i < num_gpus; i++) {

    // Query the communicator to verify it was set up correctly
    // These calls validate that PCCL properly assigned ranks and devices
    int rank, size, device;
    // Get this communicator's rank
    PCCLCHECK(pcclCommUserRank(comms[i], &rank));
    // Get total number of participants
    PCCLCHECK(pcclCommCount(comms[i], &size));
    // Get assigned HGGC device
    PCCLCHECK(pcclCommCuDevice(comms[i], &device));

    printf("  Communicator %d: Rank %d/%d on HGGC device %d", i, rank, size,
           device);

    // Verify the assignment is correct
    if (rank != i) {
      printf(" [WARNING: Expected rank %d]", i);
    }
    if (device != devices[i]) {
      printf(" [WARNING: Expected device %d]", devices[i]);
    }
    printf("\n");

    // Verify that all communicators have the expected size
    if (size != num_gpus) {
      printf("WARNING: Communicator %d has size %d, expected %d\n", i, size, num_gpus);
      sizes_match = false;
    }
  }
  if (sizes_match)
    printf("All communicators have the expected size of %d\n", num_gpus);

  printf("\n");

  // =========================================================================
  // STEP 4: Cleanup and Resource Management
  // =========================================================================

  // IMPORTANT: Proper cleanup is critical for PCCL applications
  // Resources must be cleaned up in the correct order to avoid issues

  // First, synchronize all streams to ensure no operations are in flight
  // This prevents destroying resources while they're still being used
  printf("Synchronizing all HGGC streams...\n");
  for (int i = 0; i < num_gpus; i++) {
    HGGCCHECK(hggcSetDevice(devices[i]));
    HGGCCHECK(hggcStreamSynchronize(streams[i]));
  }
  printf("All streams synchronized\n");

  // Next, destroy PCCL communicators first
  // This must be done before destroying HGGC resources they depend on
  printf("Destroying PCCL communicators...\n");
  for (int i = 0; i < num_gpus; i++) {
    PCCLCHECK(pcclCommFinalize(comms[i]));
    PCCLCHECK(pcclCommDestroy(comms[i]));
  }
  printf("All PCCL communicators destroyed\n");

  // Finally, destroy HGGC streams
  // This is safe now that the communicators are gone
  printf("Destroying HGGC streams...\n");
  for (int i = 0; i < num_gpus; i++) {
    HGGCCHECK(hggcSetDevice(devices[i]));
    HGGCCHECK(hggcStreamDestroy(streams[i]));
  }
  printf("All HGGC streams destroyed\n");

  // Free host memory allocations
  free(devices);
  free(comms);
  free(streams);

  printf("\n=============================================================\n");
  printf("SUCCESS: Multiple devices single process example completed!\n");
  printf("=============================================================\n\n");

  return 0;
}
