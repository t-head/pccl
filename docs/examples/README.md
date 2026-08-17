<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# PCCL Library Examples

Welcome to the PCCL examples directory. This collection of PCCL (PPU Collective
Communications Library) examples is designed to teach developers how to
effectively use PCCL in their applications. The examples progress from basic
concepts to advanced usage patterns, with each example featuring a detailed
README file. The APIs and features covered here are far from the complete set of
what PCCL provides.

These examples showcase individual features but are not intended to maximize the
performance for an individual communication pattern.

## Basic Examples
We start with the most basic PCCL operations. All examples in this section are
self-contained, meaning you can copy-paste one file and compile it on its own.
The only dependencies are the PCCL library itself and, in the MPI case, an MPI
implementation. These templates tend to new users coming up to speed with PCCL
for GPU communication.

### [Communicators](01_communicators/)

This section teaches you how to create, test, and destroy a communicator. We
have provided 3 examples using a single thread, multiple threads, and multiple
processes. This section shows the different options of launching a PCCL
application.

### [Point 2 Point](02_point_to_point/)

This sample send/recv implementation uses point to point communication to
communicate in a simple ring pattern.

### [Collectives](03_collectives/)

This sample implementation shows the most basic PCCL collective communication
call.

## Advanced Features

These examples are intended for experienced users looking for best practices to
use a specific feature. For complete end-to-end templates please use the basic
examples.

Since PCCL does not include its own launcher, we have provided two popular
bootstrap mechanisms. By default these examples will be launched as separate
threads, one thread per GPU. Users can set `MPI=1` to build an MPI parallel
version which can run across multiple compute nodes. Users can optionally
provide a valid MPI installation under `MPI_HOME`.

Each example can be run individually. By default you can run each executable via
```
[NTHREADS=<number of threads>] ./<example_name>
```

If `NTHREADS` is unset, the examples will use the number of visible GPUs as
number threads. If the applications are built with `MPI` support, you can run
each executable as

```
mpirun -np <number of MPI ranks> ./<example_name>
```

To ease the readability of these examples we have moved the bootstrap and
broadcast part to the [common](common/) directory. Completely self-contained
examples are provided in the sections above.

### [User Buffer Registration](04_user_buffer_registration/)

User Buffer Registration eliminates the overhead of copying data between user
buffers and internal PCCL buffers. This folder provides sample implementation
using User Buffer Registration with common collectives.

### [Symmetric Memory Registration](05_symmetric_memory/)

PCCL supports window registration, which allows users to register local buffers
into PCCL window and enables extreme low latency and high bandwidth
communication. This folder provides sample implementation using Symmetric Memory
Registration with common collectives.

### [Device APIs](06_device_api/)

Device API enables GPU kernels to directly perform inter-GPU communication. This
enables applications to perform communication from within PPU kernels and fuse
computation and communication, and fine-grained control over collective
implementation. This folder demonstrates how to implement collectives using
device-side kernels.


## Prerequisites

- The same prerequisites as building PCCL from source.
- Users can optionally add `MPI_HOME` for an MPI library in a non-standard
  location.

## Build Steps
The examples can be built while building the PCCL library from source. Users can
choose to build the examples with MPI support (`MPI=1`).

```
cd pccl
./build.sh --use_sdk --examples [MPI=1]
```

or, if PCCL has already been built, the user can optionally add a non-standard
PCCL installation location:

```
cd docs/examples
make PCCL_HOME=<path-to-pccl> [MPI=1]
```
## Environment Variables

### Build Stage
Users can use these optional variables to choose which libraries are used to
build these examples:
- `PCCL_HOME=<path>`: Local base directory of a PCCL installation.
- `MPI` : [0,1] Build the examples with MPI support.
- `MPI_HOME=<path>` : Local base directory of a MPI installation.
- `PPU_SDK=<path>` : Local base directory of a PPU SDK installation.

### Run Stage
- `NTHREADS=<n>`: Number of threads to create for the threaded examples.
  Defaults to number of visible GPUs.
- `CUDA_VISIBLE_DEVICES`: Comma delimited list of GPUs visible to the
  application.
- All other PCCL environment variables (e.g. `NCCL_DEBUG`) apply.

## Supported OS
Linux

## Troubleshooting
Each example includes a /Common Issues and Solutions/ section for the individual
tests. For general runtime issues use the debug output enabled by setting
`NCCL_DEBUG=INFO` for detailed logging.
