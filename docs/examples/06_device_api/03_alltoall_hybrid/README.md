<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# PCCL Device API Hybrid AlltoAll Example

This example shows how to implement AlltoAll operations using a hybrid approach that combines Load Store Access (LSA) for local peers with GPU-Initiated Networking (GIN) for remote peers. We create a device communicator with `pcclDevCommCreate` supporting both LSA and GIN capabilities, enabling optimal communication performance across different peer types. **Local** means ranks in the LSA team (`pcclTeamLsa`, typically same node or same ICNLink domain); **remote** means other world ranks, which use `gin.put`.

## Overview

This example showcases **hybrid communication** that intelligently selects the optimal communication method for each peer:

- **LSA (Load Store Access)** for local peers (same node/memory space)
- **GIN (GPU-Initiated Networking)** for remote peers (different nodes)

## What This Example Does

1. **Creates hybrid device communicators** using `pcclDevCommCreate` with both LSA and GIN support for optimal peer communication
2. **Registers symmetric memory windows** with `pcclCommWindowRegister` for both LSA direct access and GIN network operations
3. **Launches GPU kernel** that performs AlltoAll operations using LSA for local peers and GIN for remote peers
4. **Demonstrates hybrid synchronization** coordinating both LSA barriers and GIN signals for correctness

## Building and Running

The advanced examples can be built using either pthread or MPI for parallelization. pthread is the default choice. To use MPI the user needs to set `MPI=1` at build time and can optionally provide a valid MPI installation under `MPI_HOME`.

### Build
```bash
make [MPI=1] [MPI_HOME=<path-to-mpi>] [PCCL_HOME=<path-to-pccl>] [PPU_SDK=<path-to-ppu-sdk>]
```

### Run when compiled for pthreads (default)
```bash
[NTHREADS=N] ./alltoall_hybrid
```

### Run when compiled for MPI
```bash
mpirun -np <num_processes> ./alltoall_hybrid
```

## Code Walk-through

### Device Communicator Creation (Host-side)
The `pcclDevComm` is the core component enabling GPU kernels to perform both local and remote communication. For hybrid communication, we configure the device communicator with both LSA and GIN resources. The `pcclDevCommRequirements` specifies LSA barriers for local synchronization, GIN barriers for network synchronization, and GIN signals for completion detection. This dual setup enables optimal communication for each peer type.

```cpp
pcclDevComm devComm;
pcclDevCommRequirements reqs = PCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
// world-team barrier slots (pcclBarrierSession with GIN)
reqs.barrierCount = PCCL_DEVICE_CTA_COUNT;
// GIN signals provide completion notifications for asynchronous network operations
reqs.ginSignalCount = PCCL_DEVICE_CTA_COUNT;
// Enable full GIN connectivity, i.e., connect each rank to all other ranks
reqs.ginConnectionType = PCCL_GIN_CONNECTION_FULL;

// Create device communicator with hybrid LSA+GIN support
PCCLCHECK(pcclDevCommCreate(comm, &reqs, &devComm));
```

### Memory Window Registration (Host-side)
The device API requires symmetric memory windows registered using `PCCL_WIN_COLL_SYMMETRIC`. These windows enable both LSA direct access for local peers and GIN network operations for remote peers. The same memory windows support both communication methods, with the kernel automatically selecting the appropriate access pattern based on peer locality.

```cpp
pcclWindow_t send_win;
pcclWindow_t recv_win;

// Register symmetric windows for both LSA and GIN access
PCCLCHECK(pcclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win, PCCL_WIN_COLL_SYMMETRIC));
PCCLCHECK(pcclCommWindowRegister(comm, d_recvbuff, size_bytes, &recv_win, PCCL_WIN_COLL_SYMMETRIC));
```

### Hybrid Barriers (Device-side)
Hybrid barriers coordinate both local LSA operations and remote GIN operations. The barrier session uses the world team and GIN context to ensure synchronization across all ranks, regardless of their communication method. This unified barrier approach ensures all peers reach the same synchronization point before proceeding with data exchange.

```cpp
// Hybrid barriers coordinate both LSA and GIN operations across all ranks
pcclBarrierSession<pcclCoopCta> bar {
    pcclCoopCta(),              // Barrier scope: entire CTA (thread block)
    pcclTeamTagWorld(),         // Team spanning all ranks (local + remote)
    gin,                        // GIN context for network coordination
    blockIdx.x                  // Barrier index: matches our CTA index
};
bar.sync(pcclCoopCta(), hggc::memory_order_acquire, pcclGinFenceLevel::Relaxed);
```

### Peer Classification (Device-side)
The hybrid kernel intelligently classifies peers into local (LSA-accessible) and remote (GIN-only) categories. This classification determines the optimal communication method for each peer. Local peers benefit from direct memory access, while remote peers use network communication.

```cpp
// Classify peers into local (LSA) and remote (GIN) categories
pcclTeam world = pcclTeamWorld(devComm);  // All ranks
pcclTeam lsa = pcclTeamLsa(devComm);      // Local ranks only
const int startLsa = world.rank - lsa.rank;  // First local rank in world
const int lsaSize = lsa.nRanks;              // Number of local peers
```

### Memory Access (Device-side)
`pcclGetLsaPointer` allows PPU kernels to directly access other GPUs' memory within the LSA team, while `gin.put` handles remote communication over the network. The hybrid approach uses the most efficient method for each peer type.

```cpp
  const size_t size = count * sizeof(T);
  for (int r = tid; r < startLsa; r += nthreads) {
    gin.put(world, r,
        recvwin, recvoffset + world.rank * size,
        sendwin, sendoffset + r * size,
        size, pcclGin_SignalInc{signalIndex});
  }
  for (int r = startLsa + lsaSize + tid; r < world.nRanks; r += nthreads) {
    gin.put(world, r,
        recvwin, recvoffset + world.rank * size,
        sendwin, sendoffset + r * size,
        size, pcclGin_SignalInc{signalIndex});
  }
  T* sendLocal = (T*)pcclGetLocalPointer(sendwin, sendoffset);
  for (size_t offset = tid; offset < count; offset += nthreads) {
    for (int lp = 0; lp < lsa.nRanks; lp++) {
      int wr = startLsa + lp;
      T* recvPtr = (T*)pcclGetLsaPointer(recvwin, recvoffset, lp);
      recvPtr[world.rank * count + offset] = sendLocal[wr * count + offset];
    }
  }
  int numRemotePeers = world.nRanks - lsa.nRanks;
  // Wait only on the CTA whose signalIndex sees all GIN puts targeting this rank.
  int receivingCta = (world.rank % nthreads) / blockDim.x;
  if (blockIdx.x == receivingCta)
    gin.waitSignal(pcclCoopCta(), signalIndex, signalValue + numRemotePeers);

  gin.flush(pcclCoopCta());

  bar.sync(pcclCoopCta(), hggc::memory_order_release, pcclGinFenceLevel::Relaxed);
```

### Receiving CTA (Device-side)

`signalIndex` is `blockIdx.x`, so each `gin.put` increments the destination rank's signal for that sender CTA index. Only the CTA with `blockIdx.x == receivingCta` calls `waitSignal`; `receivingCta = (world.rank % nthreads) / blockDim.x` picks one waiting CTA per destination rank (same idea as the pure GIN AlltoAll example). The wait uses `signalValue + numRemotePeers` because only **remote** peers contribute GIN puts toward this rank; LSA peers do not.

## When to Use

- **Multi-node usage**: Mixed local/remote communication patterns
- **Production workloads**: Where performance is critical
- **Heterogeneous clusters**: Different node configurations

## Performance Considerations

**Advantages:**
- **Reduced Latency**: LSA provides low latency for local communication
- **Optimal Bandwidth**: GIN efficiently handles remote communication
- **Reduced Network Load**: Local traffic stays off the network
- **Scalable Design**: Efficient across different node configurations

**Disadvantages:**
- More complex programming model requiring coordination of both LSA and GIN
- Requires careful synchronization between different communication methods
- Higher development complexity compared to pure approaches

## Common Issues and Solutions

### Issue: LSA barriers not supported
**Cause:** GPUs not connected through ICNLink or PCIe for direct memory access
**Solution:** Verify GPU topology with `ppu-smi topo -m` and ensure proper LSA-capable connections

### Issue: Hybrid synchronization failures
**Solution:** Ensure `barrierCount` and `ginSignalCount` match `PCCL_DEVICE_CTA_COUNT` (kernel grid x dimension).

## Performance Notes

- These are educational examples, not optimized for performance
- Real implementations should consider:
  - Optimal balance between LSA and GIN operations based on topology
  - Memory coalescing patterns for both LSA and GIN operations
  - Barrier synchronization overhead minimization
  - Signal pool management for high-throughput GIN scenarios

## Error Handling

The example uses comprehensive error checking for HGGC, PCCL, LSA, and GIN operations. Device kernels should implement proper error handling for both direct memory access patterns and network operations.

## Next Steps

After understanding this example, explore:
- **Topology-aware optimization**: Fine-tune LSA/GIN balance based on hardware topology
- **Custom hybrid patterns**: Implement specialized communication strategies
- **Performance profiling**: Analyze LSA vs GIN performance characteristics
- **Advanced synchronization**: Optimize barrier usage for complex communication patterns
