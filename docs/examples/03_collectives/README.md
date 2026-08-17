<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# PCCL Collective Communication Examples

## Overview
This directory contains minimal examples that demonstrate PCCL collective
communication operations on a single node using a single process managing all
GPUs. The focus is clarity, correct resource management, and result
verification.

## Examples

### [01_allreduce](01_allreduce/)
**AllReduce Collective Operation**
- **Pattern**: All participants reduce and distribute the result
- **API**: `pcclCommInitAll`, `pcclAllReduce`
- **Use case**: Global reductions in ML and HPC (e.g., gradient averaging)
- **Key features**:
  - Initializes all GPUs in a single process
  - Each GPU contributes its rank value
  - Executes AllReduce sum across all GPUs
  - Verifies the expected global sum

## Choosing the Right Pattern

*Scenario* : Parallel training needs efficient global communication
*Addresses* : Most commonly used collective algorithms
*Dependencies* : A functional PCCL library and its dependencies

### Why `pcclCommInitAll` here?
For single-node collective examples we use `pcclCommInitAll` as it creates a
clique of communicators in one call.

```c
// Initialize all GPUs in one call
pcclComm_t* comms;
int num_gpus;
NCCLCHECK(pcclCommInitAll(comms, num_gpus, NULL));
```

A more advanced setup using MPI to initialize communicators across multiple
nodes is shown in
[01_communicators/03_one_device_per_process_mpi](../01_communicators/03_one_device_per_process_mpi)

## Building

### **Quick Start**
```shell
# Build example by directory name
make 01_allreduce
```

### **Individual Examples**
```shell
# Build and run AllReduce
cd 01_allreduce && make
./allreduce
```
