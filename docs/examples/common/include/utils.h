/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef UTILS_H_
#define UTILS_H_

#include "hggc_runtime.h"
#include "pccl.h"
#include "pccl_utils.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef MPI_SUPPORT
#include "mpi.h"
#include "mpi_utils.h"
#else
#include <pthread.h>
#include <unistd.h>
#endif

/**
 * Broadcast PCCL unique ID
 *
 * Broadcasts the PCCL unique ID from the root rank to all other ranks.
 * Uses MPI_Bcast in MPI mode and pthread barrier in pthread mode.
 *
 * @param root      Root rank that holds the PCCL unique ID
 * @param my_rank   Current rank or thread id
 * @param arg       Pointer to PCCL unique ID to broadcast
 *
 * @return 0 on success, non-zero on error
 */
int util_broadcast(int root, int my_rank, pcclUniqueId *arg);

/**
 * Run the given PCCL example in parallel
 *
 * This function performs the complete PCCL example lifecycle:
 * 1. Initialize backend (MPI or pthread)
 * 2. Execute PCCL communicator setup function
 * 3. Cleanup of all resources
 *
 * @param argc              Command line argument count
 * @param argv              Command line arguments
 * @param pcclExample       Function pointer to example-specific PCCL setup
 *
 * @return 0 on success, non-zero on error
 */
int run_example(int argc, char *argv[],
                void *(*pcclExample)(int, int, int, int));

#endif // UTILS_H_
