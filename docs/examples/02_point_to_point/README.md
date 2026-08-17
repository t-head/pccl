<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# PCCL Point-to-Point Communication Examples

## Overview
This directory contains minimal examples that demonstrate PCCL point-to-point
(P2P) communication patterns on a single node. These examples focus on clarity
and correct communicator usage, resource management, and verification.

## Examples

### [01_ring_pattern](01_ring_pattern/)
**Ring Communication Pattern**
- **Pattern**: Circular data flow among all GPUs
- **API**: `pcclCommInitAll` with P2P operations (`pcclSend`/`pcclRecv`)
- **Use case**: Learning P2P communication; pipeline/data movement patterns on a
  single node
- **Key features**:
  - Initializes all GPUs in a single process
  - Computes ring neighbors with modulo arithmetic
  - Uses `pcclGroupStart/End` to prevent deadlocks
  - Verifies data correctness after transfers

## Choosing the Right Pattern

*Scenario* : Pipeline parallel training needs to send data from one GPU to
another
*Addresses* : Individual transfers between two ranks
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

## Building

### **Quick Start**
```shell
# Build example by directory name
make 01_ring_pattern
```

### **Individual Examples**
```shell
# Build and run the ring pattern
cd 01_ring_pattern && make
./ring_pattern
```
