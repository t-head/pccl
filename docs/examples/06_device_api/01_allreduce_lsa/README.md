<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# PCCL Device API AllReduce Example

This example shows how to implement AllReduce sum operation directly in a kernel
using the PCCL device API. We first create a device communicator with
`pcclDevCommCreate` to enable kernel-initiated communication. After that,
device-side synchronization is performed with barriers and symmetric memory
windows are used to enable Load Store Accessible (LSA) memory access of peers.

## Overview

This example shows how to implement AllReduce sum operation using a GPU kernel
that directly performs the collective operations. The device communicators are
created with `pcclDevCommCreate` and device-side synchronization is ensured with
Load Store Accessible (LSA) barriers. LSA windows are used for peer memory
access.

## What This Example Does

1. **Creates device communicators** using `pcclDevCommCreate` for GPU kernel
   access to PCCL operations
2. **Registers symmetric memory windows** with `pcclCommWindowRegister` for
   direct peer-to-peer access
3. **Launches GPU kernel** that performs AllReduce sum operation entirely on
   device using LSA barriers

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
[NTHREADS=N] ./allreduce_lsa
```

### Run when compiled for MPI
```shell
mpirun -np <num_processes> ./allreduce_lsa
```

## Code Walk-through

### Device Communicator Creation (Host-side)
The `pcclDevComm` is the core component of the device API, enabling GPU kernels
to perform inter-GPU communication and fuse computation with communication. The
`pcclDevCommRequirements` specifies what resources the device communicator
should allocate. In this example, we set `lsaBarrierCount` to match our thread
block count, giving each block its own barrier for independent cross-GPU
synchronization.

```cpp
pcclDevComm devComm;
pcclDevCommRequirements reqs = PCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
// Allocate one barrier per CTA we intend to launch
reqs.lsaBarrierCount = PCCL_DEVICE_CTA_COUNT;

// Create device communicator with LSA barrier support
NCCLCHECK(pcclDevCommCreate(comm, &reqs, &devComm));
```

### Memory Window Registration (Host-side)
The device API requires symmetric memory windows registered using
`PCCL_WIN_COLL_SYMMETRIC`. See the [symmetric memory
example](../../05_symmetric_memory/) for allocation and requirements details.

```cpp
pcclComm_t comm;
void* d_sendbuff;
void* d_recvbuff;
pcclWindow_t send_win;
pcclWindow_t recv_win;

// Register symmetric windows for device-side peer access
NCCLCHECK(pcclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win, PCCL_WIN_COLL_SYMMETRIC));
NCCLCHECK(pcclCommWindowRegister(comm, d_recvbuff, size_bytes, &recv_win, PCCL_WIN_COLL_SYMMETRIC));
```

### LSA Barriers (Device-side)
LSA barriers enable cross-GPU synchronization from device code. Each thread
block uses `blockIdx.x` to select its dedicated barrier, allowing blocks to
progress independently while coordinating with corresponding blocks on other
GPUs.

```cpp
// LSA barriers enable coordination between GPU threads across different ranks
// This ensures all ranks reach the same synchronization point before proceeding
pcclLsaBarrierSession<pcclCoopCta> bar {
    pcclCoopCta(),           // Barrier scope: entire CTA (thread block)
    devComm, pcclTeamLsa(devComm), devComm.lsaBarrier,
    blockIdx.x               // Barrier index: matches our CTA index (0 to lsaBarrierCount-1)
};
bar.sync(pcclCoopCta(), hggc::memory_order_acquire);

// ...

// Release barrier ensures that we received data from everyone before we unblock the stream and allow the next kernel(s) to process the data.
bar.sync(pcclCoopCta(), hggc::memory_order_release);
```
### Memory Access (Device-side)
`pcclGetLsaPointer` allows PPU kernels to directly access other GPUs' memory
within the LSA team.

```cpp
// Access peer memory directly using LSA (Load/Store Accessible) pointers
float* peerPtr = (float*)pcclGetLsaPointer(sendwin, sendoffset, peer);
```

## When to Use

- **Kernel-level communication**: When compute kernels need immediate access to
  communication results
- **Low-latency scenarios**: Reduced host-device synchronization overhead
- **Custom collectives**: Implementing specialized reduction or communication
  patterns
- **Iterative algorithms**: Repeated communication with minimal CPU involvement

## Performance Considerations

**Advantages:**
- Lower latency for small to medium message sizes
- Eliminates host-device synchronization bottlenecks
- Enables computation-communication fusion within kernels
- Direct peer memory access without CPU copying

**Disadvantages:**
- More complex programming model requiring LSA barriers
- Requires careful memory ordering and synchronization
- Higher development complexity compared to host API
- GPUs with P2P support (e.g., ICNLink or PCI) required.

## Common Issues and Solutions

### Issue: PCCL warning communicator does not support symmetric memory
PCCL selects support for symmetric memory operations based on GPU connectivity.
If the GPUs on a node are only connected through e.g. the inter-CPU link,
symmetric memory will not be supported. **Solution:** Use `ppu-smi` to
identify and select a subset of GPUs (e.g. via `CUDA_VISIBLE_DEVICES`) connected
through ICNLink or PCIe.

### Issue: LSA barrier synchronization failures
**Solution:** Ensure `lsaBarrierCount` matches the number of thread blocks in
kernel launch configuration.

### Issue: Memory access violations in device kernel
**Solution:** Verify memory windows are registered as `PCCL_WIN_COLL_SYMMETRIC`
and all ranks use identical buffer sizes.

### Issue: Incomplete results or race conditions
**Solution:** Use proper memory ordering in LSA barriers
(`hggc::memory_order_acquire` vs `hggc::memory_order_release`).

## Performance Notes

- These are educational examples, not optimized for performance
- Real implementations should use vectorization, loop unrolling, and memory
  coalescing
- Consider PCCL's optimized device kernels for best practices related to
  performance

## Error Handling

The example uses comprehensive error checking for both HGGC and PCCL operations.
Device kernels should implement proper error handling for LSA operations and
memory access patterns.

## Next Steps

After understanding this example, explore:
- **Custom reduction operations**: Implement non-standard reduction patterns
- **Mixed host-device patterns**: Combine host and device API for complex
  workflows
- **Performance optimization**: Fine-tune LSA barrier usage and memory access
  patterns
