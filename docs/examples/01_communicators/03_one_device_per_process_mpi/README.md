<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# PCCL Example: One Device per Process (MPI)

This example demonstrates PCCL communicator lifecycle management using MPI, with
one GPU per MPI process.

## Overview

This example shows one of the most common PCCL deployment pattern: one GPU
device per process. This approach is ideal for distributed training across
multiple nodes and provides the foundation for scalable multi-GPU applications.
MPI is used as it provides a parallel launcher and broadcast functions. It is,
however, not a requirement for multi-node PCCL applications.

Other approaches use server-client models or spawn parallel processes using
sockets. PCCL only requires that the unique ID is distributed among each
thread/process taking part in collective communication and all threads/processes
call some PCCL initialization function.

## What This Example Does

1. **Multi-node Support**:
   - Determines local rank on each node automatically
   - Maps MPI processes to GPUs on each node
   - Uses `MPI_Comm_split_type` with `MPI_COMM_TYPE_SHARED` to assign each local
     rank a GPU.

2. **Communicator Creation**:
   - Uses `pcclCommInitRank` with MPI-coordinated unique ID
   - Rank `0` generates and broadcasts PCCL unique ID
   - Each process joins the distributed communicator

3. **Verification**:
   - Displays MPI rank -> PCCL rank -> GPU device mapping
   - Confirms successful initialization across all processes

4. **Cleanup**:
   - Proper resource cleanup order
   - MPI synchronization for clean shutdown

## Building and Running

### Build
```shell
make MPI=1 [MPI_HOME=<path-to-mpi>] [PCCL_HOME=<path-to-pccl>] [PPU_SDK=<path-to-ppu-sdk>]
```

### Run example
```shell
mpirun -np <num_processes> ./one_device_per_process_mpi
```

### Run with PCCL debug output
```shell
NCCL_DEBUG=INFO mpirun -np <num_processes> ./one_device_per_process_mpi
```

## Code Walk-through

This approach:
- Automatically handles multi-node GPU assignment
- Uses MPI for coordination and PCCL for GPU communication
- Supports both single-node and multi-node deployments

### Unique ID Distribution
The PCCL unique ID must be shared with all process which call `pcclCommInitRank`. We use MPI for that:
```c
// Rank 0 generates unique ID
if (mpi_rank == 0) {
    NCCLCHECK(pcclGetUniqueId(&pccl_id));
}

// Broadcast to all processes
MPI_Bcast(&pccl_id, sizeof(pcclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
```

### Key Function: Multi-node GPU assignment
```c
// Separate function to determine the node local rank via `MPI_Comm_split_type`
int local_rank = getLocalRank(MPI_COMM_WORLD);

// Use the local rank as the GPU device number. This assumes you only start as many processes as available GPUs
HGGCCHECK(hggcSetDevice(local_rank));

pcclComm_t comm;
int mpi_rank, mpi_size; // mpi_rank & mpi_size are set during MPI initialization
pcclUniqueId pccl_id; // pccl_id is generated and broadcasted as above

// Initialize PCCL communicator across all processes
NCCLCHECK(pcclCommInitRank(&comm, mpi_size, pccl_id, mpi_rank));
```

## When to Use MPI Approach

### Ideal Use Cases
- **Multi-node clusters**: Scales across multiple machines
- **Production deployments**: Industry standard for distributed training,
  inference, and most HPC codes
- **Process isolation**: Each GPU in separate process for robustness
- **Large scale**: Supports thousands of GPUs

### When NOT to Use
- **Single-node testing**: Simpler approaches available
- **No MPI available**: Some environments don't support MPI
- **Shared memory needs**: Single-process approaches may be simpler

## Performance Considerations

- **Advantages**:
  - MPI has been optimized for large parallel startup.
  - Industry standard deployment pattern
  - Optimal for large-scale training

- **Disadvantages**:
  - MPI setup complexity
  - Inter-process communication overhead
  - Requires MPI runtime environment

## Common Issues and Solutions

1. **More MPI processes than GPUs on a node**:
   - The example reports an error if local rank exceeds available devices
   - Use fewer processes per node or more GPUs

2. **MPI broadcast hangs**:
   - Ensure all ranks participate in collective operations
   - Check MPI installation and network connectivity

3. **Multi-node communication fails**:
   - Check firewall settings and network configuration
   - Set `NCCL_SOCKET_IFNAME` to specify network interface

## Error Handling

The example uses simplified error handling with CHECK macros:
- **HGGCCHECK**: Exits immediately on HGGC errors
- **NCCLCHECK**: Exits immediately on PCCL errors
- **No async error checking**: Simplified for clarity
- **No global error coordination**: Each process exits on its own errors

## Next Steps

After understanding this example:
1. Try running collective operations (AllReduce, AllGather, etc.)
2. Experiment with multi-node deployments
