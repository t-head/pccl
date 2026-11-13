# PCCL

PCCL (pronounced "Pickle") is the PPU Collective Communication Library, providing high-performance collective and point-to-point communication for PPU accelerators. It supports Zhenwu PPU hardware through the T-Head SAIL SDK and HGGC runtime. The library supports all-reduce, all-gather, reduce-scatter, reduce, broadcast, all-to-all, gather, scatter, and send/receive operations for distributed deep learning and HPC workloads.

PCCL supports PCIe, ICNLink, and ICNSwitch for PPU interconnects, and RDMA networks or TCP/IP sockets for network communication. It works with single-process and multi-process applications, including those using MPI.

Additionally, PCCL provides PPU-specific optimizations. The optional ext-kernel plugin offers low-latency implementations for small messages. With LTOKI, PPU kernels can directly initiate DMA transfers for selected operations. PCCL also provides platform-specific features, such as communication algorithms, protocols, and enhanced debugging support.

For more information, see the [PCCL Documentation](https://developer.t-head.cn/docs_center/doc_detail/index.html?projectId=35).

## Build

### Requirements

- T-Head SAIL SDK installed on Linux (x86_64 or aarch64), including the HGGC compiler and runtime.

### Quick Start

Build PCCL with T-Head SAIL SDK installed. Run these commands from the repository root:

```shell
export PPU_SDK=/usr/local/PPU_SDK     # or your SDK path
source "$PPU_SDK/envsetup.sh"
./build.sh
```

If your SDK is installed at a non-default path, replace `/usr/local/PPU_SDK` with the actual path before sourcing `envsetup.sh`.

### Build Options

For the full list of build options, run:

```shell
./build.sh --help
```

Common options:

```text
--rebuild          Clean and rebuild from scratch
--verbose          Show detailed compilation output
--build_type TYPE  Build mode: rel_sym (default), rel, or dbg
--ppu_archs ARCHS  Target PPU architectures (default: ppu_10,ppu_15)
--cpu_num N        Number of parallel build jobs (default: auto)
--examples         Also build examples after building the library
--enable_tests     Also build test suites (graph_search_test, unit_test)
```

The build places `libpccl.so` in `build/lib/` and public headers in `build/include/`.

## Compatibility after CollNet removal

CollNet is no longer supported. `PCCL_COLLNET_ENABLE` / `NCCL_COLLNET_ENABLE` no longer enable it,
and `CollNet` is disabled even when included in `PCCL_ALGO` / `NCCL_ALGO`.
Remove these obsolete settings and use the default algorithm selection or a supported algorithm.

`pcclConfig_t.collnetEnable` retains its original name, type, position, and initializer value.
It remains ignored, matching its behavior before this removal. Existing assignments still compile.

The `pcclTunerPlugin_v1` callback layout and the bundled tuner's configuration format are unchanged.
The core library preserves algorithm IDs: Tree=0, Ring=1, CollNet=2 (reserved and unsupported),
FC=3, PAT=4. No algorithm ID remapping is required. When a tuner callback succeeds but selects
CollNet or an out-of-range algorithm/protocol ID, PCCL falls back to internal tuning.
Preserving these IDs does not make every algorithm/protocol combination valid for every collective
or topology.

Ordinary `pcclNetPlugin_v4/v5/v6` interfaces are unchanged. Legacy CollNet plugin declarations
remain available for source compatibility, but PCCL no longer loads CollNet plugins.

All ranks in a communicator must use the same PCCL build. Rebuild components that depend on
internal headers, as internal channel layouts have changed.

Topology XML graph IDs remain Ring=0, Tree=1, and FC=3; ID 2 is reserved for the removed CollNet
graph. These are separate from algorithm IDs.

## Testing

For performance benchmarking and functional validation, use [pccl-tests](https://github.com/t-head/pccl-tests). With the T-Head SAIL SDK and PCCL environments configured, run:

```shell
git clone https://github.com/t-head/pccl-tests.git
cd pccl-tests
export PCCL_HOME=/path/to/pccl/build  # or your PCCL installation directory
./build.sh
./build/all_reduce_perf -b 8 -e 256M -f 2 -g <ngpus>
```

## Copyright

PCCL is derived from NVIDIA NCCL.

Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.

See [LICENSE.txt](LICENSE.txt) and [ThirdPartyNotices.txt](ThirdPartyNotices.txt) for license details.
