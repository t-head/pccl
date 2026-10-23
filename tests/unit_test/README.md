# Unit Test

## Overview

This directory contains unit tests for PCCL (PPU Collective Communication Library). The test suite includes:

- **Collective Operations Tests** (`collPcclUnitTest`): Verify PCCL collective primitives with various test modes.
- **Send/Recv Tests** (`sendRecvPcclUnitTest`): Test point-to-point communication with pcclSend/pcclRecv, supporting both single-thread and multi-threaded execution modes.
- **HGGC Graph Tests** (`hggcGraphUnitTest`): Validate HGGC graph integration with PCCL collective operations, including graph capture, task composition, and event-based synchronization.
- **Communicator Abort Tests** (`pcclCommAbortTest`): Test communicator lifecycle with abort functionality at different execution stages and memory leak detection.
- **Communicator Shrink Tests** (`pcclCommShrinkTest`): Test selective rank exclusion and communicator resizing with configurable resource sharing.
- **Group Simulation Tests** (`pcclGroupSimulateEndTest`): Estimate collective operation performance without actual execution across multiple processes.
- **Memory Management Tests** (`pcclCommMemManagerTest`): Verify communicator memory suspension/resumption with `pcclCommSuspend()`, `pcclCommResume()`, and `pcclCommMemStats()`.

## Build

```shell
source /usr/local/PPU_SDK/envsetup.sh     # or your SDK path

# Option 1: Compile pccl and all unit tests
cd pccl && ./build.sh --use_sdk --enable_tests

# Option 2: Use pccl release artifact to build unit_test standalone
source your_path_to_pccl/envsetup.sh
cd pccl/tests/unit_test && ./build.sh
```

### Build Options

- `--use_mpi` Enable multi-process MPI support for distributed testing (optional)
- `--build_type dbg` Compile debug version with detailed logging (optional)

## Usage

### Single Process PCCL Collective Kernel Test

#### Examples

Run on 2 devices (num_gpus=2), launch `AllReduce` kernel on these devices to calculate all reduce result of 32 (num_elems=32) float (dtype=float) numbers.

```Shell
./build/collPcclUnitTest --coll_type=AllReduce --num_gpus=2 --num_elems=32 --dtype=float
```

#### Arguments

Type `./build/collPcclUnitTest --help` to query all help information.

- kernel type
  - `coll_type=AllReduce`
  - `coll_type=BroadCast`
  - `coll_type=AllGather`
  - `coll_type=Reduce`
  - `coll_type=ReduceScatter`
  - `coll_type=AlltoAll`
  - `coll_type=AlltoAllArray`
- data type
  - `dtype=float`
  - `dtype=half`
  - `dtype=int8`
  - `dtype=double`
- number of devices
  - `num_gpus=<INTEGER>`
### Single Process PCCL SendRecv Kernel Test

#### Examples

Device 0 send 32 (num_elems=32) float numbers to device 1 by `pcclSend`, device 1 receive 32 float numbers from device 0 by `pcclRecv`. `pcclSend` and `pcclRecv` are wrapped in PCCL group.

```Shell
./build/sendRecvPcclUnitTest --num_elems=32
```

#### Arguments

Type `./build/sendRecvPcclUnitTest --help` to query all help information.

- execution mode
  - `mode=single` single-thread Group API mode (default)
  - `mode=thread` multi-threaded mode with barrier synchronization
- number of float elements
  - `num_elems=32` fixed element count (default)
  - `min_num_elems=<INT>` minimum for random range testing
  - `max_num_elems=<INT>` maximum for random range testing
- iteration count for stress testing
  - `num_iterations=<INT>`
- starting device ID
  - `start_dev_id=<INT>` (default 0)

### PCCL Group Simulation Test

Estimate the time a group of collectives would take by `pcclGroupSimulateEnd()`
instead of really running them. All ops given by `--op_list` are enqueued into a
single `pcclGroupStart()`/`pcclGroupSimulateEnd()` scope. Note that the current
implementation only returns the estimated time of the **last** collective op in
the group.

#### Examples

Simulate a single `AllReduce` on 4 devices (num_gpus=4) for 32B ~ 1KB, growing by
a factor of 4 each step.

```Shell
./build/pcclGroupSimulateEndTest -g 4 --op_list all_reduce -b 32 -e 1024 -f 4
```

Simulate three collectives sharing one group scope. Only the estimated time of
the last op (`reduce_scatter`) in the group is returned per size.

```Shell
./build/pcclGroupSimulateEndTest -g 4 --op_list all_reduce,all_gather,reduce_scatter -b 32 -e 1024 -f 4
```

Launch 2 processes, each managing 4 devices (8 ranks in total), to simulate the
same group across processes/nodes. Requires building with `--use_mpi`.

```Shell
mpirun -np 2 ./build/pcclGroupSimulateEndTest -g 4 --op_list all_reduce,all_gather -b 32 -e 1024 -f 4
```

Launch one process on each of 2 hosts, each managing 8 devices (16 ranks in
total), to simulate the group across nodes. Before executing this command, you
need to modify `hosts.cfg` to add hosts' ip addresses and slots of each host,
and make sure the binary and libraries are visible at the same path on every
host (or exported to remote processes via `-x LD_LIBRARY_PATH`).

```Shell
mpirun -np 2 -npernode 1 --hostfile ./hosts.cfg -x LD_LIBRARY_PATH ./build/pcclGroupSimulateEndTest -g 8 --op_list all_reduce,all_gather -b 32 -e 1024 -f 4
```

Alternatively launch one process per device (2 hosts * 8 processes * 1 device,
also 16 ranks in total).

```Shell
mpirun -np 16 -npernode 8 --hostfile ./hosts.cfg -x LD_LIBRARY_PATH ./build/pcclGroupSimulateEndTest -g 1 --op_list all_reduce,all_gather -b 32 -e 1024 -f 4
```

In both cases nRanks equals `np * num_gpus`. Processes on the same host select
disjoint devices based on a host hash, so `npernode * num_gpus` must not exceed
the number of devices visible on each host.

#### Arguments

Type `./build/pcclGroupSimulateEndTest --help` to query all help information.

- collective ops placed into one group scope, comma separated
  - `--op_list all_reduce,all_gather,reduce_scatter,broadcast,reduce`
- number of local devices per process
  - `-g,--num_gpus=<INTEGER>`
- data type
  - `-d,--data_type=float` also accepts int8/uint8/int32/uint32/int64/uint64/half/double/bfloat16/fp8_e4m3/fp8_e5m2
- data size range in bytes and the multiplication factor between sizes.
  `-b` and `-e` accept optional K/M/G suffixes (powers of 1024), e.g. `128K`
  means 128*1024 bytes
  - `-b,--begin_size=<SIZE>` `-e,--end_size=<SIZE>` `-f,--step_factor=<INTEGER>`
- root rank used by `broadcast`/`reduce`
  - `-r,--root=<INTEGER>`

### PCCL Communicator Memory Manager Test

Verify `pcclCommSuspend()` / `pcclCommResume()` / `pcclCommMemStats()` with
one process and one thread per device; thread `i` owns rank `i` on device `i`.
Each rank runs a warm-up `AllReduce`, then repeats a suspend/resume cycle
`--iters` times. Every cycle suspends the communicator with `PCCL_SUSPEND_MEM`,
checks that free device memory grows by about the suspendable bytes, checks
that a duplicate suspend and a resume on an active communicator both return
`pcclInvalidUsage`, resumes, and runs an `AllReduce` with new input to prove
the communicator still works. The tracked byte counters must stay unchanged
across all cycles.

#### Examples

Run on 2 devices (ngpus=2), 10 suspend/resume cycles (iters=10), each
`AllReduce` operates on 1 GiB of floats (count=268435456).

```Shell
./build/pcclCommMemManagerTest --ngpus=2 --iters=10 --count=268435456
```

Run on 4 devices, 5 cycles, 4 GiB buffers, and print detailed memory tables for
the first 2 cycles.

```Shell
./build/pcclCommMemManagerTest -g 4 -i 5 -n 1073741824 -p 2
```

#### Output

The test prints two tables per cycle, with cycle numbers starting at 1. Each table has one row per rank.

**Tracked memory table** shows the byte counters returned by `pcclCommMemStats()` (sampled before suspend):

```
Cycle 1/2: tracked memory per rank (KiB, sampled before suspend):
----------------------------------------------------------------------------
| rank # |     total KiB | suspendable KiB (offload+scr) |     persist KiB |
----------------------------------------------------------------------------
|      0 |        413718 |                        344064 |           69654 |
|      1 |        413718 |                        344064 |           69654 |
----------------------------------------------------------------------------
```

**Suspend/resume table** compares manager-reported suspendable bytes to the runtime free-memory changes:

```
Cycle 1/2: suspendable memory, manager (pcclCommMemStats) vs runtime (hggcMemGetInfo) (KiB):
  freed = free after suspend - free before suspend, used = free while suspended - free after resume,
  diff = runtime - manager
---------------------------------------------------------------------------------------------------
| rank # | manager KiB | suspend freed KiB | suspend diff KiB | resume used KiB | resume diff KiB |
---------------------------------------------------------------------------------------------------
|      0 |      344064 |            342608 |            -1456 |          342720 |           -1344 |
|      1 |      344064 |            342604 |            -1460 |          342720 |           -1344 |
---------------------------------------------------------------------------------------------------
```

Where:
- `manager KiB` suspendable bytes reported by `pcclCommMemStats()`
- `suspend freed KiB` free memory after suspend - free memory before suspend
- `suspend diff KiB` `suspend freed KiB` - `manager KiB`
- `resume used KiB` free memory while suspended - free memory after resume
- `resume diff KiB` `resume used KiB` - `manager KiB`

A diff close to `0` means the runtime released and mapped again what the
manager reports; it is usually slightly negative because of allocator
granularity. The test requires `suspend freed KiB` to be at least half of
`manager KiB`.

#### Arguments

Type `./build/pcclCommMemManagerTest --help` to query all help information.

- number of float elements per `AllReduce` buffer
  - `-n,--count=<INTEGER>` default `1073741824` (4 GiB per buffer)
- number of devices, one thread and one rank per device
  - `-g,--ngpus=<INTEGER>` default `2`
- number of suspend/resume cycles
  - `-i,--iters=<INTEGER>` default `10`
- number of leading cycles whose tables are printed; `0` disables them, values above `--iters` are capped
  - `-p,--print-cycles=<INTEGER>` default `1`

## bench_oki ICN benchmark

See [bench_oki documentation](docs/bench_oki.md) for build instructions,
parameters, timing modes, validation, and automated bandwidth comparisons.
