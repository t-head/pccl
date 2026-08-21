<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# PCCL Example: One Device per Thread (pthread)

This example demonstrates PCCL communicator lifecycle management using pthreads, with one GPU per
thread.

## Overview

This example shows how to use PCCL in a multi-threaded environment where each pthread manages one
GPU device. It demonstrates the proper initialization and cleanup sequence for PCCL communicators
within threads.

## What This Example Does

1. **Thread Creation**:
   - Creates one pthread per available GPU or `NTHREADS` if set
   - Each thread manages its own HGGC device context

2. **Communicator Creation**:
   - Uses `pcclCommInitRank` with unique ID across threads
   - Each thread initializes its own communicator
   - Demonstrates thread-safe PCCL initialization

3. **Verification**:
   - Queries communicator properties (rank, size, device)
   - Confirms successful initialization across all threads

4. **Cleanup**:
   - Proper resource cleanup order within each thread
   - Demonstrates correct PCCL and HGGC resource management

## Building and Running

### Build
```shell
make  [PCCL_HOME=<path-to-pccl>] [PPU_SDK=<path-to-ppu-sdk>]
```

### Run with specific thread count (number of GPUs)
```shell
[NTHREADS=n] ./one_device_per_pthread
```

### Run with PCCL debug output
```shell
PCCL_DEBUG=INFO ./one_device_per_pthread
```

## Code Walk-through

### Key Function: pcclCommInitRank in threads
```c
// Each thread creates it's own copy of struct `threadData_t`.
typedef struct {
  int thread_id; // thread_id is set when thread is created
  int num_gpus; // num_gpus is set by querying the PPU devices
  pcclUniqueId commId; // commId is set by pcclGetUniqueId
  pcclComm_t* comms;
} threadData_t;
threadData_t* data;

// Each thread initializes its own communicator
PCCLCHECK(pcclCommInitRank(&data->comms[thread_id], data->num_gpus, data->commId, data->thread_id));
```

In this approach:
- Each thread gets its own PCCL rank (0, 1, 2...)
- Does not need explicit distribution of `uniqueId` since it uses a global variable.

## When to Use pthread Approach

### Ideal Use Cases
- **Thread-based applications**: When your application is already threaded
- **Single-node workloads**: All GPUs on one machine
- **Shared memory**: Need to share data structures between GPU contexts

### When NOT to Use
- **Multi-node clusters**: Cannot scale beyond one node
- **Process isolation**: When GPU contexts should be isolated
- **Complex applications**: Multi-process approach may be cleaner

## Performance Considerations

- **Advantages**:
  - Shared address space between threads
  - Easier data sharing between GPU contexts
  - No MPI overhead

- **Disadvantages**:
  - Thread synchronization complexity
  - Limited to single node

## Common Issues and Solutions

1. **Thread synchronization errors**:
   - Ensure all threads use the same PCCL unique ID
   - Proper pthread synchronization (barriers, joins)

2. **HGGC context conflicts**:
   - Each thread must call `hggcSetDevice()` before HGGC operations
   - Don't share HGGC streams between threads

3. **Resource cleanup order**:
   - Always destroy PCCL communicators before HGGC resources
   - Synchronize streams before destroying communicators

## Error Handling

The example uses simplified error handling with CHECK macros:
- **HGGCCHECK**: Exits immediately on HGGC errors
- **PCCLCHECK**: Exits immediately on PCCL errors
- **No async error checking**: Simplified for clarity
- **Thread safety**: Each thread handles its own errors

## Highlighted Environment Variables

- `NTHREADS`: Number of threads to create (defaults to number of GPUs)

See examples/README.md for the full list.

## Next Steps

After understanding this example:
1. Try using the collective examples and add the pthread approach
2. Compare with MPI-based multi-process approach
