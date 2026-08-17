<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# PCCL Ring Communication Pattern Example

This example demonstrates a ring communication pattern using PCCL P2P
operations. It runs on a single node where a single process manages all GPUs and
data flows in a circular pattern.

## Overview

The ring communication pattern creates a circular data flow where each GPU sends
data to its "next" neighbor and receives from its "previous" neighbor in the
ring. This example uses `pcclCommInitAll` for simplified single-threaded,
single-process multi-GPU setup.

## What This Example Does

1. **Detects and initializes all available GPUs** using `pcclCommInitAll` for
   simplified single-process setup
2. **Creates ring topology** where each GPU calculates its next and previous
   neighbors using modulo
3. **Executes simultaneous point-to-point communication** with each GPU sending
   to next and receiving from previous
4. **Verifies data correctness** by checking that each GPU received the expected
   data from its predecessor

## Building and Running

### Build the Example
```bash
cd examples/02_point_to_point/01_ring_pattern
make [PCCL_HOME=<path-to-pccl>] [PPU_SDK=<path-to-ppu-sdk>]
```

### Run with All Available GPUs
```bash
./ring_pattern
```

## Code Walk-through

### Ring Topology Setup

The example calculates ring neighbors using modulo arithmetic:

```cpp
for (int i = 0; i < num_gpus; i++) {
    int next = (i + 1) % num_gpus;        // Next neighbor in ring
    int prev = (i - 1 + num_gpus) % num_gpus;  // Previous neighbor in ring
}
```

### Simultaneous Communication

Uses `pcclGroupStart/End` to prevent deadlocks when scheduling all send and
receive operations:

```cpp
float **d_sendbuff; // device side send and receive buffer are allocated through hggcMalloc
float **d_recvbuff;
size_t count;       // count is set to the number of floats to be sent (usually the size of the buffers)
pcclComm_t *comms;  // comms are set during pcclCommInitAll
hggcStream_t *streams; // streams are set in hggcStreamCreate

// Each GPU simultaneously sends to next and receives from previous
NCCLCHECK(pcclGroupStart());
for (int i = 0; i < num_gpus; i++) {
    int next = (i + 1) % num_gpus;
    int prev = (i - 1 + num_gpus) % num_gpus;

    NCCLCHECK(pcclSend(d_sendbuff[i], count, pcclFloat, next, comms[i], streams[i]));
    NCCLCHECK(pcclRecv(d_recvbuff[i], count, pcclFloat, prev, comms[i], streams[i]));
}
NCCLCHECK(pcclGroupEnd());
```

## When to Use

- **Learning PCCL fundamentals**: Understanding point-to-point communication
  patterns
- **Algorithm development**: Building custom collective operations based on
  point to point communications
- **Single-node applications**: Pipeline parallelism or custom data distribution
  patterns

## Key Insights
- `pcclCommInitAll` simplifies single-node multi-GPU setup
- No MPI or pthreads needed for single-node patterns
- Ring pattern enables circular data flow among all GPUs
- `pcclGroupStart/End` prevents deadlock in simultaneous operations
- Each GPU both sends and receives in parallel

## Common Issues and Solutions

### Issue: Deadlock without group operations
**Solution:** Always use `pcclGroupStart()` and `pcclGroupEnd()` when performing
simultaneous send/recv operations.

### Issue: Verification failures
**Solution:** Check ring topology calculations and data initialization patterns.
Ensure correct neighbor calculations.

## Error Handling

This example uses comprehensive error checking with `NCCLCHECK` and `HGGCCHECK`
macros that immediately exit on any failure. In production code, consider more
graceful error handling and recovery mechanisms.

## Next Steps

After this example, try:
- **Collective operations**: Examples in `03_collectives/`
- **Multi-node approach**: Use the MPI implementation from `01_communicators` to
  send data across nodes.
