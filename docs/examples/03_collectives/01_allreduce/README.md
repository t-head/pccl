<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# PCCL AllReduce Collective Operation Example

This example demonstrates the fundamental AllReduce collective operation using
PCCL's single-process, multi-GPU approach in which a single process manages all
GPUs to perform a sum reduction.

## Overview

AllReduce combines data from all participants using a reduction operation (sum,
max, min, etc.) and distributes the result to all participants. This example
shows how each GPU contributes its rank value and all GPUs receive the combined
sum using `pcclCommInitAll` for simplified setup.

## What This Example Does

1. **Detects available GPUs** and initializes PCCL communicators for all devices
   using `pcclCommInitAll`
2. **Initializes data** with each GPU contributing its rank value (GPU 0->0, GPU
   1->1, etc.)
3. **Performs AllReduce sum operation** where all GPU values are summed and
   distributed to all participants
4. **Verifies correctness** by checking that all GPUs received the expected sum:
   0+1+2+...+(n-1)

## Building and Running

### Build the Example
```bash
cd examples/03_collectives/01_allreduce
make [PCCL_HOME=<path-to-pccl>] [PPU_SDK=<path-to-ppu-sdk>]
```

### Run with All Available GPUs
```bash
./allreduce
```

### Run with Specific GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./allreduce
```

## Code Walk-through

### Data Initialization
Each GPU sets a send buffer allocated on the GPU to its rank value:
```cpp
float** sendbuff;
float rank_value = (float)i;
size_t size; // size is the number of float to be sent

// Allocate device memory for send buffers
HGGCCHECK(hggcMalloc((void **)&sendbuff[i], size * sizeof(float)));

// Each GPU contributes its rank (GPU i contributes value i)
// Zero the entire buffer, then set first element to rank
HGGCCHECK(hggcMemset(sendbuff[i], 0, size * sizeof(float)));
HGGCCHECK(hggcMemcpy(sendbuff[i], &rank_value, sizeof(float), hggcMemcpyHostToDevice));
```

### AllReduce Operation
All GPUs participate in the sum reduction. The operations are evaluated in parallel within a PCCL group to avoid any deadlocks.
```cpp
float** recvbuff;
pcclComm_t *comms;  // comms are set during pcclCommInitAll
hggcStream_t *streams; // streams are set in hggcStreamCreate

// Allocate device memory for receive buffers
HGGCCHECK(hggcMalloc((void **)&recvbuff[i], size * sizeof(float)));

PCCLCHECK(pcclGroupStart());
for (int i = 0; i < num_gpus; i++) {
    PCCLCHECK(pcclAllReduce(sendbuff[i], recvbuff[i], size, pcclFloat,
                            pcclSum, comms[i], streams[i]));
}
PCCLCHECK(pcclGroupEnd());
```

## When to Use

- **Deep learning**: Gradient averaging in data-parallel training
- **Scientific computing**: Global reductions in parallel algorithms
- **Statistics**: Computing global sums, averages, or other reductions
- **Distributed algorithms**: Any scenario requiring collective reduction
  operations

## Key Insights
- `pcclCommInitAll` simplifies single-node multi-GPU setup
- No MPI or pthreads needed for single-node patterns
- Allocate device buffer via `hggcMalloc` and initialize via `hggcMemset`.
- Best practices to wrap all collective calls in pcclGroupStart/End
- All communication happens in parallel

## Common Issues and Solutions

### Issue: Verification failures
**Solution:** Ensure each GPU initializes its buffer correctly with its rank
value.

### Issue: Out of memory errors
**Solution:** Reduce the buffer size in the code or use fewer GPUs.

## Error Handling

This example uses comprehensive error checking with `PCCLCHECK` and `HGGCCHECK`
macros that immediately exit on any failure. In production code, consider more
graceful error handling and recovery mechanisms.

## Next Steps

After understanding AllReduce, explore:
- **Point-to-point communication**: Examples in `02_point_to_point/`
- **Other collectives**: Implement Broadcast, Reduce, AllGather operations using
  this example
- **Multi-node approach**: Use the MPI implementation from `01_communicators` to
  send data across nodes.
