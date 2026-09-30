<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# PCCL Symmetric Memory Examples

## Overview
This directory contains minimal examples that demonstrate PCCL symmetric memory
windows for improving performance of collective operations when all ranks use
consistent memory layouts.

## Examples

### [01_allreduce](01_allreduce/)
**AllReduce with Symmetric Memory Windows**
- **Pattern**: Register symmetric windows per rank and use them for collectives
- **API**: `pcclCommWindowRegister`, `pcclCommWindowDeregister`, `pcclMemAlloc`,
  `pcclAllReduce`
- **Use case**: Large-scale collectives with consistent buffer layouts across
  ranks
- **Key features**:
  - Buffers allocated via `pcclMemAlloc` for symmetric compatibility
  - Windows registered as `PCCL_WIN_COLL_SYMMETRIC`
  - Collective operations executed on symmetric windows
  - Correct deregistration and cleanup

## Choosing the Right Pattern

*Scenario* : Large-scale training with consistent memory patterns
*Addresses* : Low-latency, high-bandwidth collectives on supported systems
*Dependencies* : pthread or MPI

### Why Symmetric Windows?
Symmetric windows enable PCCL to apply optimized collective protocols when all
ranks use consistent layouts. The memory needs to be allocated through the
Virtual Memory Management (VMM) API and registered with PCCL.

```c
// Allocate using PCCL provided convenience function and register symmetric windows
PCCLCHECK(pcclMemAlloc(&buffer, size_bytes));
PCCLCHECK(pcclCommWindowRegister(comm, buffer, size_bytes, &win, PCCL_WIN_COLL_SYMMETRIC));

// Collective using symmetric windows
PCCLCHECK(pcclAllReduce(buffer, buffer, count, pcclFloat, pcclSum, comm, stream));

// Deregister and free
PCCLCHECK(pcclCommWindowDeregister(comm, win));
PCCLCHECK(pcclMemFree(buffer));
```

## Building

### **Quick Start**
```shell
# Build example by directory name
make 01_allreduce
```

### **Individual Examples**
```shell
# Build and run AllReduce with symmetric windows
cd 01_allreduce && make
./allreduce_sm
```
