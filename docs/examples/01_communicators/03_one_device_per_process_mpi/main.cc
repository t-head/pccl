/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "hggc_runtime.h"
#include "mpi.h"
#include "pccl.h"
#include "pccl_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/**
 * PCCL Example: One Device per Process with MPI
 * =============================================
 *
 * LEARNING OBJECTIVE:
 * This example teaches the fundamental PCCL pattern: one GPU device per MPI
 * process. This is the most common deployment pattern for multi-GPU distributed
 * training.
 *
 * WHAT THIS CODE DEMONSTRATES:
 * - How to initialize PCCL communicators across multiple processes
 * - Proper GPU assignment in both single-node and multi-node environments
 * - Complete PCCL communicator lifecycle management
 * - Error handling best practices for production code
 *
 * STEP-BY-STEP PROCESS:
 * 1. MPI Setup: Initialize MPI and determine process layout
 * 2. GPU Assignment: Map each process to a local GPU device
 * 3. PCCL ID Sharing: Rank 0 creates unique ID, broadcasts to all processes
 * 4. Communicator Creation: Each process joins the PCCL communicator
 * 5. Verification: Query and verify communicator properties
 * 6. Clean Shutdown: Properly destroy all resources in correct order
 *
 * MULTI-NODE INTELLIGENCE:
 * - Automatically detects which processes are on the same physical node
 * - Assigns local GPU indices (0, 1, 2, 3...) to processes on each node
 * - Uses MPI_Comm_split_type with MPI_COMM_TYPE_SHARED for robust node
 * identification
 * - Leverages MPI's native shared memory detection for optimal performance
 *
 * USAGE EXAMPLES:
 *   Single node (4 GPUs): mpirun -np 4 ./one_device_per_process_mpi
 *
 * EXPECTED OUTPUT:
 *   Each process will report: MPI rank → PCCL rank → GPU device assignment
 *   Success message confirms all communicators were created properly
 */

// =============================================================================
// LOCAL RANK UTILITY FUNCTION - For Multi-Node GPU Assignment
// =============================================================================

/**
 * Determine the local rank of this process on its physical node
 *
 * Algorithm:
 * 1. Split the communicator based on shared memory (i.e., nodes)
 * 2. Get the rank within the node communicator
 * 3. This rank becomes the local rank for GPU assignment
 *
 * @param comm The MPI communicator to use for determining local rank
 * @return Local rank (0, 1, 2...) for GPU assignment, or -1 on error
 */
int getLocalRank(MPI_Comm comm) {

  int world_size;
  MPI_Comm_size(comm, &world_size);

  int world_rank;
  MPI_Comm_rank(comm, &world_rank);

  // Split the communicator based on shared memory (i.e., nodes)
  MPI_Comm node_comm;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL,
                      &node_comm);

  // Get the rank and size within the node communicator
  int node_rank, node_size;
  MPI_Comm_rank(node_comm, &node_rank);
  MPI_Comm_size(node_comm, &node_size);

  // Clean up the node communicator
  MPI_Comm_free(&node_comm);

  return node_rank;
}

// =============================================================================
// MAIN FUNCTION - PCCL Communicator Lifecycle Example
// =============================================================================

int main(int argc, char *argv[]) {
  // Variables for MPI, HGGC, and PCCL components
  int mpi_rank, mpi_size, local_rank;
  int num_gpus = 0;
  pcclComm_t comm = NULL;
  hggcStream_t stream = NULL;
  pcclUniqueId pccl_id;

  // =========================================================================
  // STEP 1: Initialize MPI and determine process layout
  // =========================================================================

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);


  if (mpi_rank == 0) {
    printf("Starting PCCL communicator lifecycle example with %d processes\n",
           mpi_size);
  }
  // Determine which local GPU this process should use
  local_rank = getLocalRank(MPI_COMM_WORLD);

  printf("  MPI initialized - Process %d of %d total processes\n", mpi_rank,
         mpi_size);

  // =========================================================================
  // STEP 2: Setup HGGC device for this process
  // =========================================================================

  // Check how many HGGC devices are available on this node
  HGGCCHECK(hggcGetDeviceCount(&num_gpus));
  printf("  Found %d HGGC devices on this node\n", num_gpus);

  if (num_gpus == 0) {
    fprintf(stderr, "ERROR: No HGGC devices found on this node!\n");
    exit(EXIT_FAILURE);
  }

  if (local_rank >= num_gpus) {
    fprintf(stderr,
            "ERROR: Process %d needs GPU %d but only %d devices available\n",
            mpi_rank, local_rank, num_gpus);
    exit(EXIT_FAILURE);
  }

  // Assign this process to its designated GPU device
  HGGCCHECK(hggcSetDevice(local_rank));

  // Create HGGC stream for GPU operations
  HGGCCHECK(hggcStreamCreate(&stream));

  printf("  MPI rank %d assigned to HGGC device %d\n", mpi_rank,
         local_rank);

  // =========================================================================
  // STEP 3: Initialize PCCL communicator
  // =========================================================================

  // Generate PCCL unique ID (only rank 0 needs to do this)
  if (mpi_rank == 0) {
    PCCLCHECK(pcclGetUniqueId(&pccl_id));
    printf("Rank 0 generated PCCL unique ID for all processes\n");
  }

  // Share the unique ID with all processes using MPI broadcast
  MPI_Bcast(&pccl_id, PCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD);
  printf("INFO: Rank %d received PCCL unique ID\n", mpi_rank);

  // Create PCCL communicator for this process
  // This is where each process joins the distributed PCCL communicator
  PCCLCHECK(pcclCommInitRank(&comm, mpi_size, pccl_id, mpi_rank));
  printf("  Rank %d created PCCL communicator\n", mpi_rank);

  // =========================================================================
  // STEP 4: Verify communicator setup
  // =========================================================================

  // Query communicator properties to verify everything is set up correctly
  int comm_rank, comm_size, comm_device;
  PCCLCHECK(pcclCommUserRank(comm, &comm_rank));
  PCCLCHECK(pcclCommCount(comm, &comm_size));
  PCCLCHECK(pcclCommCuDevice(comm, &comm_device));

  printf("  MPI rank %d → PCCL rank %d/%d on GPU device %d\n", mpi_rank,
         comm_rank, comm_size, comm_device);

  // Give all processes a chance to finish their printf
  MPI_Barrier(MPI_COMM_WORLD);

  // =========================================================================
  // STEP 5: Clean shutdown and resource cleanup
  // =========================================================================

  if (mpi_rank == 0) {
    printf(
        "\nAll communicators initialized successfully! Beginning cleanup...\n");
  }

  // Synchronize HGGC stream to ensure all GPU work is complete
  if (stream != NULL) {
    HGGCCHECK(hggcStreamSynchronize(stream));
  }

  // Destroy PCCL communicator FIRST (before HGGC resources)
  // This is important - PCCL cleanup should happen before HGGC cleanup
  if (comm != NULL) {
    PCCLCHECK(pcclCommFinalize(comm));
    PCCLCHECK(pcclCommDestroy(comm));
    printf("  Rank %d destroyed PCCL communicator\n", mpi_rank);
  }

  // Now destroy HGGC stream
  if (stream != NULL) {
    HGGCCHECK(hggcStreamDestroy(stream));
  }

  if (mpi_rank == 0) {
    printf(
        "\nAll PCCL communicators created and cleaned up properly!\n");
    printf("This example demonstrated the complete PCCL communicator "
           "lifecycle.\n");
    printf("Next steps: Try running PCCL collective operations (AllReduce, "
           "etc.)\n");
  }

  MPI_Finalize();
  return 0;
}
