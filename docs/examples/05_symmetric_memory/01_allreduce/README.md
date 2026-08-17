<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# PCCL Symmetric Memory AllReduce Example

This example demonstrates how to use PCCL's symmetric memory feature for
optimized collective operations. Symmetric memory provides optimized performance
by leveraging consistent memory layouts across all participating ranks, enabling
advanced communication algorithms.

## Overview

Symmetric memory windows provide a way to register memory buffers that benefit
from optimized collective operations. When using `PCCL_WIN_COLL_SYMMETRIC`, all
ranks must provide symmetric buffers, enabling optimized communication patterns
and better performance for large-scale multi-GPU operations.

## What This Example Does

1. **Allocates memory using PCCL allocator** (`pcclMemAlloc`) which provides
   memory compatible with symmetric windows
2. **Registers buffers as symmetric windows** using `pcclCommWindowRegister`
   with `PCCL_WIN_COLL_SYMMETRIC` flag
3. **Performs AllReduce sum operation** using the symmetric memory for optimized
   communication performance

## Building and Running

The advanced examples can be built using either pthread or MPI for
parallelization. pthread is the default choice. To use MPI the user needs to set
`MPI=1` at build time and can optionally provide a valid MPI installation under
`MPI_HOME`.

### Build
```shell
make [MPI=1] [MPI_HOME=<path-to-mpi>] [PCCL_HOME=<path-to-pccl>] [PPU_SDK=<path-to-ppu-sdk>]
```

### Run when compiled for pthreads (default)
```shell
[NTHREADS=N] ./allreduce_sm
```

### Run when compiled for MPI
```shell
mpirun -np <num_processes> ./allreduce_sm
```

## Code Structure

### Key Components

1. **Buffer Allocation and Window Registration**:
```c
size_t size_bytes; // Is set to the size of the send/receive buffers
void *d_sendbuff;
void *d_recvbuff;

// Allocate buffers using pcclMemAlloc (compatible with symmetric memory)
NCCLCHECK(pcclMemAlloc(&d_sendbuff, size_bytes));
NCCLCHECK(pcclMemAlloc(&d_recvbuff, size_bytes));

pcclComm_t comm;
pcclWindow_t send_win;
pcclWindow_t recv_win;

// Register buffers as symmetric windows
NCCLCHECK(pcclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win, PCCL_WIN_COLL_SYMMETRIC));
NCCLCHECK(pcclCommWindowRegister(comm, d_recvbuff, size_bytes, &recv_win, PCCL_WIN_COLL_SYMMETRIC));
```

2. **AllReduce Operation**:
```c
size_t count;  // set to number of floats to exchange
hggcStream_t stream; // stream is set in hggcStreamCreate

// Perform AllReduce with symmetric memory optimization
NCCLCHECK(pcclAllReduce(d_sendbuff, d_recvbuff, count, pcclFloat, pcclSum,
                        comm, stream));
```

3. **Window Deregistration and Cleanup**:
```c
// Deregister symmetric memory windows
NCCLCHECK(pcclCommWindowDeregister(comm, send_win));
NCCLCHECK(pcclCommWindowDeregister(comm, recv_win));

// Free buffers allocated with pcclMemAlloc
NCCLCHECK(pcclMemFree(d_sendbuff));
NCCLCHECK(pcclMemFree(d_recvbuff));
```

## Performance Benefits of Symmetric Memory

Symmetric memory registration provides several performance advantages:

- **Optimized Communication Algorithms**: PCCL can apply advanced optimizations
  when all ranks have symmetric layouts
- **Better Memory Access Patterns**: Consistent layouts enable better caching
  and memory access optimization

**Important**: Buffers must be allocated using the Virtual Memory Management
(VMM) API. PCCL provides the `pcclMemAlloc` convenience function for symmetric
memory registration. The `PCCL_WIN_COLL_SYMMETRIC` flag requires all ranks to
provide symmetric buffers consistently.

## Key Insights

- **Symmetric Memory Windows** are most beneficial for:
  - Large-scale collective operations with consistent memory patterns
  - Latency-sensitive kernels
  - Applications with predictable allocation patterns
- **pcclCommInitRank** can be used for pthread or MPI parallel case
- **Window registration** must happen on all ranks for collective operations
- **Memory management** is critical - always deregister windows before freeing
  memory

## Common Issues and Solutions

1. **Window Registration Failure**: Buffers MUST be allocated with VMM API,
   e.g. `pcclMemAlloc` (not `hggcMalloc`) for symmetric memory.
2. **Allocation Error**: If `pcclMemAlloc` fails, check PCCL version and
   available memory
3. **Deregistration Order**: Always deregister windows before freeing memory or
   destroying communicators
4. **Symmetric Requirement**: All ranks must use `PCCL_WIN_COLL_SYMMETRIC`
   consistently in collective operations
5. **Memory Leaks**: Always use `pcclMemFree` for buffers allocated with
   `pcclMemAlloc`
