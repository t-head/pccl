# PCCL

PCCL (PPU Collective Communication Library) is a high-performance collective communication library for PPU (Parallel Processing Unit) accelerators. It is derived from [NCCL](https://github.com/NVIDIA/nccl) and has been adapted to run on PPU hardware via the PPU SDK (HGGC runtime).

PCCL provides optimized implementations of standard collective communication primitives used in distributed deep learning and HPC workloads, achieving high bandwidth over ICN and other inter/intra connect topologies.

For detailed documentation on PCCL internals and implementation details, please refer to the [PCCL Documentation](https://test-thead-developer.alibaba.net/docs_center/doc_detail/index.html?projectId=9).

## Supported Collectives

PCCL implements the following collective operations:

- All-Reduce
- All-Gather
- Reduce-Scatter
- Reduce
- Broadcast
- Send / Recv (point-to-point)
- AllToAll
- Gather / Scatter

These operations use ring-based, tree-based, and other algorithms, optimized for throughput and latency on PPU devices. Protocol paths include Simple, SimpleBulk, LL, and others, selected automatically based on message size and topology. SimpleBulk is a PPU-optimized protocol and the default for most throughput-oriented workloads.

## Requirements

- PPU SDK (v2.2 or later)
- Linux (x86_64, aarch64)

## Build

### Quick Start

```shell
source /usr/local/PPU_SDK/envsetup.sh   # or your SDK path
./build.sh
```

If your SDK is installed at a non-default path, replace `/usr/local/PPU_SDK` with the actual path before sourcing `envsetup.sh`.

### Build Options

`build.sh` supports a number of flags to control the build:

| Flag | Default | Description |
|------|---------|-------------|
| `--rebuild` | off | Clean and rebuild from scratch |
| `--verbose` | off | Show detailed compilation output |
| `--test_dt` | all types | Build kernels for specific data type (`i8`, `u8`, `i32`, `u32`, `i64`, `u64`, `f16`, `f32`, `f64`) |
| `--test_op` | all ops | Build specific kernels (`all_reduce`, `all_gather`, `broadcast`, `reduce`, `reduce_scatter`, `sendrecv`) |
| `--ppu_archs` | `ppu_10,ppu_15` | Target PPU architectures |
| `--no_tsm` | off | Do not use TSM to implement kernels |
| `--no_ext_kernel` | off | Skip building ext-kernel plugin |
| `--cpu_num` | auto | Number of parallel build threads |

Examples:

```shell
# Verbose build
./build.sh --verbose

# Build only AllReduce kernels for f32
./build.sh --test_op all_reduce --test_dt f32

# Build without TSM kernel implementation
./build.sh --no_tsm
```

Build output is placed in `build/`. The library (`libpccl.so`) and headers are installed under `build/lib/` and `build/include/`.

## Environment Setup

After building from source, source `envsetup.sh` to set up your environment:

```shell
source envsetup.sh
```

## Project Structure

```
pccl/
├── src/                  # Core library source
│   ├── collectives/      # Collective operation implementations
│   │   └── device/       # Device-side (kernel) collective code
│   ├── graph/            # Execution graph and search
│   ├── transport/        # Transport layer (SHM, P2P, NET, CollNet)
│   ├── plugin/           # Plugin loading infrastructure
│   ├── include/          # Internal headers
│   └── misc/             # Utilities
├── ext-kernel/           # External kernel plugin (custom PPU kernels)
├── ext-net/              # External network plugin interface
├── ext-tuner/            # Tuner plugin (algorithm/protocol selection)
├── ext-profiler/         # Profiler plugin interface
├── tests/                # Unit and integration tests
├── makefiles/            # Build configuration (common.mk, version.mk)
├── pkg/                  # Packaging scripts (deb, rpm, txz)
├── build.sh              # Build entry point
├── envsetup.sh           # Environment setup script
├── Makefile              # Top-level Makefile
└── LICENSE.txt           # Apache 2.0 license
```

## Plugin System

PCCL supports a modular plugin architecture for extending functionality:

**ext-kernel** — Custom PPU device kernels for accelerated collectives. Ext-kernel provides optimized implementations (e.g., one-shot AllGather) that can outperform the core ring-based path at small to medium message sizes. The plugin is loaded at runtime and activated based on message size and topology configuration.

**ext-tuner** — Algorithm and protocol selection tuner. The tuner plugin decides which algorithm (ring, tree, etc.) and protocol (Simple, SimpleBulk, LL, etc.) to use for a given collective, message size, and topology.

**ext-profiler** — Performance profiling interface. Profiler plugins capture event-level timing data from collectives, proxy operations, and kernel execution for debugging and analysis.

**ext-net** — Network plugin interface for integrating custom network transports.

## Testing

For performance benchmarking and functional validation, use [pccl-tests](https://code.alibaba-inc.com/PPU-comm/pccl-tests):

```shell
git clone git@code.alibaba-inc.com:PPU-comm/pccl-tests.git
cd pccl-tests
make
./build/all_reduce_perf -b 8 -e 256M -f 2 -t 4 -g <ngpus>
```

pccl-tests provides a comprehensive set of performance benchmarks covering all collective operations, with configurable message sizes, data types, and rank counts.

## Version

The current version is defined in `makefiles/version.mk`.

## Acknowledgments

PCCL is derived from NVIDIA NCCL.

All source code and accompanying documentation is copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.

All modifications are copyright (c) 2019-2026, T-Head Semiconductor. All rights reserved.

See `LICENSE.txt` for the full Apache 2.0 license terms.
