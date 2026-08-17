<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# PCCL Device API Examples

## Overview
This directory contains minimal examples that demonstrate PCCL's device API,
enabling users to perform inter-GPU communication within their own kernels.

## Examples

### [01_allreduce_lsa](01_allreduce_lsa/)
**AllReduce with Device Kernel Implementation**
- **Pattern**: GPU kernel performs collectives using device communicators
- **API**: `pcclDevCommCreate`, `pcclCommWindowRegister`, device-side LSA
  barriers, `pcclGetLsaPointer` (manual reduction in-kernel)
- **Use case**: Allreduce operations with custom operations, fusing allreduce
  operation with previous/next compute operation.
- **Key features**:
  - Device communicator creation with LSA barrier support
  - Symmetric memory windows for peer memory access
  - Device kernels coordinating via LSA barriers
  - Host launches kernel; kernel performs AllReduce on-device

### [02_alltoall_gin](02_alltoall_gin/)
**Pure GIN AlltoAll - Network-Only Communication**
- **Pattern**: GPU kernel performs AlltoAll using only GIN for all peers
- **API**: `pcclDevCommCreate` with GIN support, `pcclGin`, GIN barriers and signals
- **Use case**: Multi-node AlltoAll with consistent network-based communication
- **Key features**:
  - Pure GIN implementation (no LSA optimizations)
  - Network barriers for cross-node synchronization (acquire before puts, release after flush)
  - Signal-based completion detection
  - Baseline network performance measurements

### [03_alltoall_hybrid](03_alltoall_hybrid/)
**Hybrid AlltoAll - Optimized Communication**
- **Pattern**: GPU kernel performs AlltoAll using LSA for local peers, GIN for remote
- **API**: `pcclDevCommCreate` with both LSA and GIN support, peer classification
- **Use case**: Multi-node AlltoAll with optimal performance across topologies
- **Key features**:
  - Hybrid implementation for optimal performance
  - Intelligent peer classification (local vs remote)
  - Combined LSA and GIN synchronization
  - Production-ready optimized communication patterns

## Choosing the Right Pattern

*Scenario* : Custom kernels fusing computation and communication.
*Addresses* : Schedule communication from inside a PPU kernel.
*Dependencies* : pthread or MPI

### Why the Device API?
The device API allows PCCL communication within PPU kernels, fusing communication and computation steps:
```cpp
// Host:
// 1) Create device communicator + requirements
// 2) Register symmetric memory windows for peer access (send + recv)
pcclDevComm devComm;
pcclDevCommRequirements reqs = PCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
reqs.lsaBarrierCount = PCCL_DEVICE_CTA_COUNT;
NCCLCHECK(pcclDevCommCreate(comm, &reqs, &devComm));
NCCLCHECK(pcclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win, PCCL_WIN_COLL_SYMMETRIC));
NCCLCHECK(pcclCommWindowRegister(comm, d_recvbuff, size_bytes, &recv_win, PCCL_WIN_COLL_SYMMETRIC));

// Device:
// - Use LSA barriers for cross-GPU synchronization
// - Access peers via symmetric windows (pcclGetLsaPointer); reduce in-kernel
simpleAllReduceKernel<<<PCCL_DEVICE_CTA_COUNT, PCCL_DEVICE_THREADS_PER_CTA, 0, stream>>>(
    send_win, 0, recv_win, 0, count, devComm);
```

### Checking for Device API and GIN Support

To check for the required Device API and GIN support, the communicator properties are queried via `pcclCommQueryProperties`:

```cpp
pcclCommProperties_t props = PCCL_COMM_PROPERTIES_INITIALIZER;
NCCLCHECK(pcclCommQueryProperties(comm, &props));

// Check for Device API support
if (!props.deviceApiSupport) {
    printf("ERROR: communicator does not support Device API!\n");
    // Exit gracefully...
}

// Check for GIN support (ginType should not be PCCL_GIN_TYPE_NONE)
if (props.ginType == PCCL_GIN_TYPE_NONE) {
    printf("ERROR: communicator does not support GIN!\n");
    // Exit gracefully...
}

// For pure LSA examples, ensure a single team where all ranks can access each other
if (props.nLsaTeams != 1) {
    printf("ERROR: expected 1 LSA team for pure LSA example!\n");
    // Exit gracefully...
}
```

## Building

### **Quick Start**
```shell
# Build example by directory name
make 01_allreduce_lsa
make 02_alltoall_gin
make 03_alltoall_hybrid
```

### **Individual Examples**
```shell
# Build and run the device API AllReduce
cd 01_allreduce_lsa && make
./allreduce_lsa

# Build and run the Pure GIN AlltoAll example
cd 02_alltoall_gin && make
./allreduce_gin

# Build and run the Hybrid AlltoAll example
cd 03_alltoall_hybrid && make
./alltoall_hybrid
```
