<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# PCCL Device API Pure GIN AlltoAll Example

This example demonstrates PCCL's GPU-Initiated Networking (GIN) capabilities for performing AlltoAll collective operations directly from GPU kernels using only network-based communication.

## Overview

This example showcases **pure GIN communication** where all data exchange happens through the network, without any Load Store Access (LSA) optimizations. This is particularly useful for:

- Multi-node environments where ranks cannot use LSA
- Testing network performance without local optimizations
- Understanding the baseline GIN communication patterns
- Scenarios where all communication must go through the network

## What This Example Does

1. **Creates device communicators** using `pcclDevCommCreate` for GPU kernel access to PCCL operations
2. **Registers symmetric memory windows** with `pcclCommWindowRegister` for direct peer-to-peer access
3. **Launches GPU kernel** that performs AlltoAll operations using pure GIN for all peer communication

## Building and Running

The advanced examples can be built using either pthread or MPI for parallelization. pthread is the default choice. To use MPI the user needs to set `MPI=1` at build time and can optionally provide a valid MPI installation under `MPI_HOME`.

### Build
```bash
make [MPI=1] [MPI_HOME=<path-to-mpi>] [PCCL_HOME=<path-to-pccl>] [PPU_SDK=<path-to-ppu-sdk>]
```

### Run when compiled for pthreads (default)
```bash
[NTHREADS=N] ./alltoall_gin
```

### Run when compiled for MPI
```bash
mpirun -np <num_processes> ./alltoall_gin
```

## Code Walk-through

### Device Communicator Creation (Host-side)
The `pcclDevComm` is the core component enabling GPU kernels to perform network communication directly. For pure GIN communication, we configure the device communicator with GIN-specific resources. The `pcclDevCommRequirements` specifies GIN barriers for network synchronization and signals for completion detection. Unlike LSA-based examples, we don't need LSA barriers since all communication goes through the network.

```cpp
pcclDevComm devComm;
pcclDevCommRequirements reqs = PCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
// GIN barriers enable cross-node synchronization over the network
reqs.worldGinBarrierCount = PCCL_DEVICE_CTA_COUNT;
// GIN signals provide completion notifications for asynchronous operations
reqs.ginSignalCount = PCCL_DEVICE_CTA_COUNT;
// Enable full GIN connectivity, i.e., connect each rank to all other ranks
reqs.ginConnectionType = PCCL_GIN_CONNECTION_FULL;

// Create device communicator with pure GIN support
PCCLCHECK(pcclDevCommCreate(comm, &reqs, &devComm));
```

### Memory Window Registration (Host-side)
The device API requires symmetric memory windows registered using `PCCL_WIN_COLL_SYMMETRIC`. These windows enable GPU kernels to access remote memory through GIN operations. Unlike LSA which provides direct memory access, GIN windows are accessed through network put/get operations.

```cpp
pcclWindow_t send_win;
pcclWindow_t recv_win;

// Register symmetric windows for GIN network access
PCCLCHECK(pcclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win, PCCL_WIN_COLL_SYMMETRIC));
PCCLCHECK(pcclCommWindowRegister(comm, d_recvbuff, size_bytes, &recv_win, PCCL_WIN_COLL_SYMMETRIC));
```

### GIN Barriers (Device-side)
GIN barriers enable cross-node synchronization from device code over the network. Each thread block uses `blockIdx.x` to select its dedicated barrier, allowing blocks to progress independently while coordinating with corresponding blocks on other nodes. An **acquire** sync at kernel entry ensures all ranks are ready before the AlltoAll exchange; a **release** sync after `flush` (see below) pairs with that acquire so ranks do not leave the kernel until the collective is complete.

```cpp
  pcclGinBarrierSession<pcclCoopCta> bar { pcclCoopCta(), gin, pcclTeamTagWorld(), blockIdx.x };
  bar.sync(pcclCoopCta(), hggc::memory_order_acquire, pcclGinFenceLevel::Relaxed);
```

### GIN Put Operations (Device-side)
GIN provides one-sided put operations for direct remote memory writes over the network. Each thread handles a subset of destination ranks, writing its rank's data to the appropriate location in each peer's receive buffer. The `pcclGin_SignalInc` parameter increments a signal counter, enabling asynchronous completion detection.

```cpp
  const size_t size = count * sizeof(T);
  for (int r = tid; r < devComm.nRanks; r += nthreads) {
    gin.put(pcclTeamWorld(devComm), r,
        recvwin, recvoffset + devComm.rank * size,
        sendwin, sendoffset + r * size,
        size, pcclGin_SignalInc{signalIndex});
  }
```

### Signal-based Completion (Device-side)
GIN uses signals for asynchronous completion detection of network operations. The kernel waits for the signal value to reach the expected count (initial value + number of ranks), indicating all put operations have completed. The `gin.flush()` ensures all pending operations are committed. A final `bar.sync` with **release** ordering runs after `flush` so every rank completes the GIN work before any rank proceeds past the kernel.

```cpp
  // Wait only on the CTA whose blockIdx.x (signalIndex) accumulates all puts to this rank.
  int receivingCta = (devComm.rank % nthreads) / blockDim.x;
  if (blockIdx.x == receivingCta)
    gin.waitSignal(pcclCoopCta(), signalIndex, signalValue + devComm.nRanks);

  gin.flush(pcclCoopCta());
  bar.sync(pcclCoopCta(), hggc::memory_order_release, pcclGinFenceLevel::Relaxed);
```

### Receiving CTA (Device-side)

`signalIndex` is `blockIdx.x`, so each `gin.put` increments the destination rank's signal for that sender CTA index. Only the CTA with `blockIdx.x == receivingCta` calls `waitSignal`; `receivingCta = (devComm.rank % nthreads) / blockDim.x` picks one waiting CTA per destination rank. The wait uses `signalValue + devComm.nRanks` because every peer sends one GIN put to this rank (pure GIN: all peers are remote).

## When to Use

- **Multi-node environments**: When ranks cannot use LSA
- **Testing network performance**: Without local optimizations

## Multiple NICs

Set `PCCL_GIN_MULTINIC=1` to enumerate devices from the selected GIN plugin.
GIN device indices do not have to match the collective NET plugin's indices.
The default remains the existing topology-affinity selection.

`PCCL_GIN_NCONNECTIONS` optionally selects 1–4 logical connections. Without an
explicit count, GIN v13 uses the enumerated device count, capped at four. All
ranks agree on the minimum connection count. Connections cycle through the
available local GIN devices if more connections than devices are requested.

Device contexts are distributed across connections: context `i` uses connection
`i % ginConnectionCount` and provider-local context `i / ginConnectionCount`.
The requested context count is rounded up to a multiple of the connection count;
use the returned `devComm.ginContextCount` when assigning contexts. Each connection
has its own device handle and window registration.

An application must complete operations on every context it uses, using the
existing signal/counter wait and flush APIs. The example uses context 0; it does
not automatically distribute transfers across NICs.

One selected GIN plugin supplies the connections of a communicator. Additional
NIC backends can use the existing GIN plugin callbacks; their
device implementations and matching handle versions must be supplied separately.
Multi-NIC enumeration does not make different device backends interchangeable.

## GIN windows and device configuration

`PCCL_WIN_GIN_ONLY` registers a window for GIN without importing peer LSA
allocations. Local address resolution remains available through
`pcclGetLocalPointer`; peer LSA pointer queries are unsupported for this window.
Do not combine the flag with `PCCL_WIN_COLL_SYMMETRIC`. Registration is still
collective, with consistent flags and segment sizes across ranks.

VMM HOST_NUMA allocations and mixed device/host ranges are registered using
their segment memory types. This does not extend window registration to
arbitrary `malloc` or ordinary pinned allocations without VMM handles. Host
windows may use GIN-only registration to avoid peer LSA IPC requirements.
Use `pcclGin_SegmentHostNuma` or `pcclGin_SegmentMixed` for transfers involving
host-backed segments and complete the transfers before CPU access or release.
The `hostRmaSupport` property describes a separate host RMA API capability.

`pcclCommQueryProperties` reports `ginSupport[]`, `ginMinStride`, and
`devCommRuntimeVersionSize`. `reqs.ginType` may explicitly request the backend
already selected for the communicator; it does not load or combine backends.
Only that selected backend is advertised as supported.

`PCCL_GIN_CONNECTION_CUSTOM_STRIDE` uses `reqs.ginCustomStride` to describe a
logical peer subset. The stride must divide the LSA team size and be a multiple
of the minimum supported stride. Provider peer tables stay dense; this version
does not reduce QP allocation with sparse contexts. Existing FULL/RAIL requests
remain supported. All ranks must agree on connection type, backend and stride.

Released device API structure prefixes remain unchanged. New fields are appended
after the legacy layout, and old callers receive only their declared layout.
Use matching headers and runtime libraries; runtime-size reporting is not a
general translation layer between arbitrary releases.

## Waiting and completion

The existing `waitSignal` and `waitCounter` calls remain available. Overloads
with a final `timeoutCycles` argument return `pcclResult_t`; the timeout uses
the device clock. A zero timeout still checks the condition once. Every thread
in the cooperative group receives the result. Counter providers may specialize
the optional progress hook when completions require software progress.

`pcclGinFenceLevel::Put` and `Get` extend the existing `Relaxed` barrier mode.
Use `pcclGinAllContexts(devComm)` to fence every context; a signal on context 0
alone must not substitute for completion on other contexts. Weak completion
action names use the existing ordered provider path, retaining its guarantees.

Focused MPI checks for these interfaces, shared storage layouts and failure
paths are documented in [tests/device_api](../../../../tests/device_api/README.md).

## Performance Considerations

- **Network overhead**: All communication goes through the network stack
- **Signal-based completion**: Enables asynchronous operation patterns
- **Barrier synchronization**: Ensures proper ordering of network operations
- **Multiple GIN contexts**: Can improve parallel communication performance

## Common Issues and Solutions

### Issue: Deadlock at util_broadcast
**Solution:** Ensure you're running with multiple GPUs/processes
```bash
NTHREADS=2 ./alltoall_gin  # For 2 GPUs
```

### Issue: Out of memory
**Solution:** Reduce the data size in the example

### Issue: Network errors
**Solution:** Ensure proper network configuration for multi-node setups

## Performance Notes

- These are educational examples, not optimized for performance
- Real implementations should consider:
  - Optimal GIN context usage for parallel operations
  - Signal pool management for high-throughput scenarios
  - Memory coalescing patterns for network operations
  - Network topology-aware communication strategies

## Error Handling

The example uses comprehensive error checking for HGGC, PCCL, and GIN operations. Device kernels should implement proper error handling for network operations and signal management.

## Next Steps

After understanding this example, explore:
- **Performance optimization**: Fine-tune GIN context usage and signal management
- **Hybrid approaches**: Combine GIN with LSA for topology-aware optimizations
- **Integration with compute**: Fuse network communication with computation kernels
