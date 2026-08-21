<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# PCCL User Buffer Registration Examples

## Overview
This directory contains minimal examples that demonstrate PCCL user buffer
registration for improving performance by allowing PCCL to operate directly on
user-allocated buffers.

## Examples

### [01_allreduce](01_allreduce/)
**AllReduce with User Buffer Registration**
- **Pattern**: Register communication buffers once and reuse across operations
- **API**: `pcclCommRegister`, `pcclCommDeregister`, `pcclMemAlloc`,
  `pcclAllReduce`
- **Use case**: Repeated collectives on the same buffers; performance-critical
  workloads
- **Key features**:
  - Buffers allocated via `pcclMemAlloc` for registration compatibility
  - Registration handles managed explicitly (register -> use -> deregister)
  - Collective operations executed on registered buffers
  - Correct cleanup and verification

## Choosing the Right Pattern

*Scenario* : Optimize performance for repeated collectives on same buffers
*Addresses* : Throughput-sensitive training loops
*Dependencies* : pthread or MPI

### Why Buffer Registration?
Pre-registering buffers eliminates per-call registration overhead and enables
direct access. It can accelerate collectives and greatly reduce the resource
usage (e.g. #channel usage). Also, this is a prerequisite for advanced features
such as symmetric memory or device API calls.

```c
// Allocate using PCCL convenience function and register buffers
PCCLCHECK(pcclMemAlloc((void**)&d_send, size_bytes));
PCCLCHECK(pcclCommRegister(comm, d_send, size_bytes, &send_handle));

// Use in collectives
PCCLCHECK(pcclAllReduce(d_send, d_recv, count, pcclFloat, pcclSum, comm, stream));

// Deregister and free
PCCLCHECK(pcclCommDeregister(comm, send_handle));
PCCLCHECK(pcclMemFree(d_send));
```

## Building

### **Quick Start**
```shell
# Build example by directory name
make 01_allreduce
```

### **Individual Examples**
```shell
# Build and run AllReduce with user buffer registration
cd 01_allreduce && make
./allreduce_ub
```
