# P2P plan comparison

This diagnostic compares uploaded regular plans and the work executed by each
channel. It is intended for the staged P2P refactor: first device preprocessing,
then pair-oriented enqueue structure, while retaining PCCL partition and channel
selection policies.

Use identical binaries, topology, protocol, environment, and benchmark arguments
for each comparison except for the library under test. Reserve the devices and
check active controllers, device processes, and kernel health before launching.
Run one communicator per rank and retain complete, separate logs for every rank.

## Collect structural logs

For a two-node, eight-devices-per-node AlltoAll diagnostic, set these variables
in every rank's environment, with a separate output directory for each build:

```sh
export PCCL_NET=IB PCCL_IB_DISABLE=0 PCCL_MNNVL_ENABLE=0
export PCCL_PROTO=Simple PCCL_P2P_LL_THRESHOLD=0
export PCCL_ENABLE_EXT_KERNEL=0 PCCL_WIN_ENABLE=0
export PCCL_MIN_P2P_NCHANNELS=8 PCCL_MAX_P2P_NCHANNELS=8
export PCCL_P2P_NCHANNELS_PER_PEER=4
export PCCL_DEBUG=INFO PCCL_DEBUG_SUBSYS=INIT,NET,P2P,COLL
export PCCL_DUMP_WORK_ELEMS=1
export PCCL_DEBUG_FILE="$TRACE_DIR/rank-%h-%p.log"
```

Launch one MPI rank per device using the site's normal hostfile, network
interface and CPU-affinity settings. Each rank runs:

```sh
alltoall_perf -g 1 -w 0 -n 1 -N 1 -c 1 -d int8 \
  -b 8M -e 8M -f 2 -a 3 -k 1 -r 0 -R 0 --elapsed_type 0 --api_time
```

Here `-n 1` and `-w 0` only reduce the structural trace. These runs are not
performance measurements. Gather the per-rank log files from both nodes, then:

```sh
python3 tests/p2p_plan_trace/compare.py before/ after/ --ranks 16 > comparison.json
```

The tool requires every rank. It rejects duplicate batches, incomplete chains,
and inconsistent work totals. For each channel it follows `nextJump`, joins
`nextExtends` batches, and sums the population count of `offsetBitset` to recover
the device's `nWorks`. It compares channel masks, work/function IDs, per-plan
channel work totals, and the sequence of execution batches. Raw descriptor
offsets are excluded because descriptor sharing can legitimately change them.

The reported plan count is an **upload count**, not an independently measured
driver launch count. In the non-graph, non-external regular path, a nonempty plan
is passed to `ncclLaunchKernel`; use a separate runtime trace if launch-level
confirmation is required. This tool does not compare payload contents, partition
offsets, proxy steps, or timing. Keep benchmark correctness checks enabled.

## Nonregistered boundary checker

`boundaries.cc` validates every received byte and surrounding guard bytes for
one-way, asymmetric duplex, same-peer grouped messages, zero-byte operations,
self-copy, short LL messages and partition/network tails. It uses ordinary
`hggcMalloc` buffers and does not register them. It requires Open MPI and either
2 or 4 ranks; `OMPI_COMM_WORLD_LOCAL_RANK` selects the visible device ordinal.
With 4 ranks on two hosts, place 2 ranks on each host so the paired peers cross
hosts. Reserve and check the selected devices before any run.

Build on the authorized PPU environment, with the SDK environment sourced and
`PCCL_HOME` pointing to the library build being tested:

```sh
mpicxx -std=c++14 -O2 \
  -I "$PCCL_HOME/include" -I "$PPU_SDK_HOME/include" \
  -I "$PPU_SDK_HOME/CUDA_SDK/include" boundaries.cc \
  -L "$PCCL_HOME/lib" -L "$PPU_SDK_HOME/lib" \
  -L "$PPU_SDK_HOME/CUDA_SDK/lib" -lpccl -lhggc_wrapper -o boundaries
export LD_LIBRARY_PATH="$PCCL_HOME/lib:$LD_LIBRARY_PATH"
export PCCL_ENABLE_EXT_KERNEL=0 PCCL_WIN_ENABLE=0 PCCL_GRAPH_REGISTER=0
export PCCL_PROTO=Simple PCCL_P2P_LL_THRESHOLD=32768
export PCCL_ALLOC_P2P_NET_LL_BUFFERS=1
export PCCL_MIN_P2P_NCHANNELS=8 PCCL_MAX_P2P_NCHANNELS=8
export PCCL_P2P_NCHANNELS_PER_PEER=4
```

Launch with the site's authorized host, transport and affinity configuration.
Seven `PASS` lines and a zero exit status are required. Confirm actual LL
selection in P2P logs; the case name alone does not prove protocol coverage.
Repeat with these independent configurations:

- Four channels and `PCCL_WORK_ARGS_BYTES=160` to force FIFO storage.
- The same FIFO setting plus `P2P_CHECK_GRAPH=1` to capture the asymmetric case
  and replay it twice with different payloads. Require `GRAPH_OK`; keep graph
  registration disabled.
- 24 total/per-peer channels, first with Simple and then SimpleBulk, to cover
  PCCL's non-power-of-two channel permutation and partial final partitions.

Use `PCCL_DEBUG_SUBSYS=INIT,NET,P2P,COLL` and `PCCL_DUMP_WORK_ELEMS=1` for path
verification. These checker runs are correctness diagnostics, not performance
measurements. The launch settings, actual library hash, device state, protocol
logs and complete output should be retained together.

## Performance acceptance

For the existing non-registered AlltoAll, Gather and Scatter matrix, retain
`-b 128 -e 8G -f 2` and the default iteration count (omit `-n`), on 2/4/8 devices
within one node and 16 across two nodes. Disable work dumps for timed runs.
Keep the same timing metric and benchmark warmup implementation as the baseline.

Compare both the previous accepted step and the fixed starting baseline. A
confirmed time increase above 5% at any tested size blocks acceptance; gains at
other sizes do not cancel it. Use interleaved independent A/B runs to assess
variance. An inconclusive result remains pending, not a pass. Device errors or
ownership changes stop the sequence; do not automatically retry them.

## Recorded baseline diagnostic (2026-11-09)

The original full-range scan used three independent launches per variant, with
default `n=20`, `w=10`, `N=1`, int8, Simple, LL threshold zero, register disabled,
and maximum-rank CPU completion time. At command-line size 8 MiB (512 KiB per
peer for 16 ranks), the recorded times were:

| Variant | Source commit | Three CPU times (us) | Median (us) |
|---|---|---|---|
| A, before shared work | `ed532b3a4a3f2b543689688bbcd71972a3a40a57` | 535.61, 222.91, 300.97 | 300.97 |
| B, shared work | `d1dcb7ce6e51bb5d30b2a85938ed68e3fc99d435` | 584.21, 589.20, 590.42 | 589.20 |

The +95.8% median difference remains unresolved. A has substantial variability;
the data do not establish whether the difference is caused by the patch.
The separate IPC-registration fix is absent from both A and B.

The fixed-size structural run on 2026-11-09 reused these original libraries on
two 8-device ZW-M890P hosts. All 16 ranks were present, both runs completed with
zero incorrect values, and the decoded execution structures matched:

| Observation | A | B |
|---|---:|---:|
| Uploaded plans per rank over the diagnostic program | 3 | 3 |
| Active channels in each plan | 8 (mask `0xff`) | 8 (mask `0xff`) |
| Execution batches per channel per plan | 1 | 1 |
| `nWorks` per execution batch | 2 | 2 |
| `channelWorks` per plan/channel | 2 | 2 |

Library SHA-256:

- A: `93ade23e8fb9c52d06e54f4eba3a4b12cca1d8557beced4bf46234d505f1d5d8`
- B: `b80db72a3b6d2977a5bfa240ca25023e2d4d3092b5cce89d5f2b5e1eb42f2298`
- Benchmark: `776df660ded724e577017ad38463aa1c13b825b7e9373730629269fd1817e76a`

The benchmark derives from pccl-tests `394556512e87259e9ffa25225632c46b0a0bbcd5`
with the previously used per-size warmup change. No timing claim is made from
the work-dump runs. Baseline stability and the complete 5% acceptance gate are
still pending.

## Enqueue refactor comparison (2026-11-09)

Compared the device-preprocessing build (`aaaab9b3e134c413efeaa174142f0467af3d51e4`)
with the same tree plus the pair-oriented enqueue refactor, using the structural
command above at 8 MiB and 128 MiB. All four 16-rank launches completed with zero
incorrect values and passed the host/device health checks. Both builds retain
the same IPC fix, and registration is disabled in these tests.

At both sizes the comparator found identical structures across all 16 ranks:
three uploaded plans per rank and eight active channels per plan. At 8 MiB each
channel executes one batch with `nWorks=2`; at 128 MiB it executes two batches,
each with `nWorks=2`. The 128 MiB dumps contain shared-partition metadata, so this
case exercises shared work in addition to the private-work case at 8 MiB.

Both versions built remotely for PPU 1.5 with SendRecv int8 kernels. Library
SHA-256 values were checked on both hosts before execution:

- Device preprocessing: `a186a8dc72f115050842309335ecfaf016b23a5754610887162ed570d2cda7d1`
- Enqueue refactor: `cf941a91b50f83a8166395fdf807ad114f93820801d07d6582ad071b1cc865cb`

This is limited structural and correctness coverage. Full boundary/protocol
coverage and the 5% performance acceptance matrix remain pending.
