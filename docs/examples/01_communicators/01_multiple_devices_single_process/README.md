<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# PCCL Example: Multiple Devices Single Process

This example demonstrates how to use `pcclCommInitAll` to create PCCL
communicators for multiple GPUs within a single process, without requiring MPI
or threading.

## Overview

The `pcclCommInitAll` function provides a simplified way to initialize PCCL
communicators when:
- All GPUs are managed by a single process
- Running on a single node
- No multi-process coordination is needed

This approach is ideal for single-node multi-GPU applications where simplicity
is preferred over the flexibility of multi-process setups.

## What This Example Does

1. **Device Detection**:
   - Queries available PPU devices
   - Lists device properties for each GPU

2. **Communicator Creation**:
   - Uses `pcclCommInitAll` to create all communicators in one call
   - Automatically assigns PCCL ranks 0 through n-1
   - No PCCL unique ID distribution needed

3. **Verification**:
   - Displays communicator information for each GPU
   - Shows rank assignments and device mappings
   - Confirms successful initialization

4. **Cleanup**:
   - Properly destroys communicators and streams
   - Demonstrates correct resource management

## Building and Running

### Build
```shell
make [PCCL_HOME=<path-to-pccl>] [PPU_SDK=<path-to-ppu-sdk>]
```

### Run with all available GPUs
```shell
./multiple_devices_single_process
```

### Run with specific GPUs
```shell
# Use only GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 ./multiple_devices_single_process
```

### Run with PCCL debug output
```shell
PCCL_DEBUG=INFO ./multiple_devices_single_process
```

## Code Walk-through

### Key Function: pcclCommInitAll
For single-node collective examples we use `pcclCommInitAll` as it creates a clique of communicators in one call.
```c
int num_gpus; // num_gpus is set by querying the PPU devices
pcclComm_t* comms;
int* devices; // devices needs to be populated with PPU devices used

// Create communicators for all devices in one call
PCCLCHECK(pcclCommInitAll(comms, num_gpus, devices));
```

This single function call:
- Creates `num_gpus` communicators
- Assigns ranks 0 to (num_gpus-1)
- Sets up internal communication paths
- No unique ID needed

### Comparison with pcclCommInitRank
`pcclCommInitAll` is a convenience function and has the same functionality as:
```c
pcclUniqueId id;

pcclGetUniqueId(&id);

pcclGroupStart();
for(int i = 0; i < num_gpus; i++) {
  hggcSetDevice(i);
  pcclCommInitRank(comms[i], num_gpus, id, devices[i]);
}
pcclGroupEnd();
```

## When to Use pcclCommInitAll

### Ideal Use Cases
- **Single-node workloads**: All GPUs on one machine
- **Simple applications**: No multi-process complexity needed
- **Testing/Development**: Quick setup for experiments

### When NOT to Use
- **Multi-node clusters**: Need MPI for cross-node communication
- **Process isolation**: When GPUs should be in separate processes

## Performance Considerations

- **Advantages**:
  - Lower overhead (no inter process communication)
  - Simpler memory management
  - Direct access to all GPUs

- **Disadvantages**:
  - Limited by single process resources
  - Cannot scale beyond one node

## Common Issues and Solutions

1. **Not all GPUs visible**:
   - Check `CUDA_VISIBLE_DEVICES`
   - Ensure user has permissions for all GPUs
   - Verify no other process is using GPUs exclusively

2. **Out of memory**:
   - Single process must handle memory for all GPUs
   - Consider using multiple processes if memory limited

## Next Steps

After understanding this example:
1. Try the collective operation examples using `pcclCommInitAll`
2. Compare performance with MPI-based multi-process approach
3. Experiment with different GPU combinations
