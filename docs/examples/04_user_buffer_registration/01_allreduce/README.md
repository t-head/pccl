<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# PCCL User Buffer Registration AllReduce Example

This example demonstrates how to use PCCL's user buffer registration feature to
optimize performance for repeated collective operations on the same buffers.
User Buffer Registration is a feature that allows PCCL to directly
send/receive/operate data through the user buffer without extra internal copy
(zero-copy).

## Overview

User buffer registration allows PCCL to pre-register memory buffers with
communicators, eliminating registration overhead on each operation. This is
particularly beneficial for applications that repeatedly perform collective
operations on the same memory regions, such as iterative training loops.

## What This Example Does

1. **Allocates memory using PCCL allocator** (`pcclMemAlloc`) which is provided
   by PCCL as convenience function
2. **Registers buffers with communicator** using `pcclCommRegister` for
   optimized performance
3. **Performs AllReduce sum operation** using the registered buffers for
   efficient communication

## Building and Running

The advanced examples can be built using either pthread or MPI for
parallelization. pthread is the default choice. To use MPI the user needs to
provide a valid MPI installation under `MPI_HOME`.

### Build
```shell
make [MPI=1] [MPI_HOME=<path-to-mpi>] [PCCL_HOME=<path-to-pccl>] [PPU_SDK=<path-to-ppu-sdk>]
```

### Run when compiled for pthreads (default)
```shell
[NTHREADS=N] ./allreduce_ub
```

### Run when compiled for MPI
```shell
mpirun -np <num_processes> ./allreduce_ub
```

## Code Structure

### Key Components

1. **Buffer Allocation and Registration**:
```c
size_t size_bytes; // Is set to the size of the send/receive buffers
void *d_sendbuff;
void *d_recvbuff;

// Allocate buffers using pcclMemAlloc (or another qualified allocator) on the device
PCCLCHECK(pcclMemAlloc(&d_sendbuff, size_bytes));
PCCLCHECK(pcclMemAlloc(&d_recvbuff, size_bytes));

pcclComm_t comm;   // comms is set during pcclCommInitRank
void *send_handle;
void *recv_handle;

// Register buffers with PCCL, handle is returned for De-registration
PCCLCHECK(pcclCommRegister(comm, d_sendbuff, size_bytes, &send_handle));
PCCLCHECK(pcclCommRegister(comm, d_recvbuff, size_bytes, &recv_handle));
```

2. **AllReduce with Group Operations**:
```c
size_t count;  // set to number of floats to exchange
hggcStream_t stream; // stream is set in hggcStreamCreate

PCCLCHECK(pcclAllReduce(d_sendbuff, d_recvbuff, count, pcclFloat, pcclSum,
                        comm, stream));
```

3. **Buffer Deregistration and Cleanup**:
```c
// Deregister buffers using handle from pcclCommRegister
PCCLCHECK(pcclCommDeregister(comm, send_handle));
PCCLCHECK(pcclCommDeregister(comm, recv_handle));

// Free buffers allocated with pcclMemAlloc
PCCLCHECK(pcclMemFree(d_sendbuff));
PCCLCHECK(pcclMemFree(d_recvbuff));

```

## Performance Benefits of User Buffer Registration

User buffer registration provides several performance advantages:

1. **Reduced Overhead**: Pre-registration eliminates the need to
   register/deregister buffers for each operation
2. **Better Memory Pinning**: Registered buffers are pinned in memory,
   preventing page faults
3. **Lower Latency**: Especially beneficial for repeated operations on the same
   buffers

**Important**: Buffers must be allocated with `pcclMemAlloc` or a compatible
allocator for registration to work.

**Important**: If any rank in a communicator passes registered buffers to a PCCL
communication function, all other ranks in the same communicator must pass their
registered buffers; otherwise, mixing registered and non-registered buffers can
result in undefined behavior.

## Key Insights

- **User Buffer Registration** is most beneficial for:
  - Large data transfers
  - Repeated operations on the same buffers
  - Performance-critical applications
- **Memory management** is critical - always deregister buffers before freeing

## Common Issues and Solutions

1. **Registration Failure**: Buffers MUST be allocated with `pcclMemAlloc` or
   another qualified allocator (not `hggcMalloc`) for registration.
2. **Allocation Error**: If `pcclMemAlloc` fails, check PCCL version and
   available memory
3. **Deregistration Order**: Always deregister before freeing memory
4. **Handle Management**: Keep track of registration handles for proper cleanup
5. **Memory Leaks**: Always use `pcclMemFree` for buffers allocated with
   `pcclMemAlloc`
