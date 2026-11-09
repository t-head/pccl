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
- `PCCL_BUFFSIZE=5275648` with 8 channels, including two-host and graph cases,
  to exercise exact non-FP8 chunk sizes and the proxy/device chunk bound.
- 24 total/per-peer channels, first with Simple and then SimpleBulk, to cover
  PCCL's non-power-of-two channel permutation and partial final partitions.

Use `PCCL_DEBUG_SUBSYS=INIT,NET,P2P,COLL` and `PCCL_DUMP_WORK_ELEMS=1` for path
verification. These checker runs are correctness diagnostics, not performance
measurements. The launch settings, actual library hash, device state, protocol
logs and complete output should be retained together.

`chunks.cc` is a CPU-only internal layout/encoding check. Build it in the same
authorized SDK environment, adding `-I src/include` and `-I build/include` to
the include paths above. It checks every size below 1 MiB, all FP8 code points
and their neighbors, the largest uint32 value, and both directions independently.
It requires exact round trips, fallback reset, a 64-byte work and the NCCL prefix
offsets. Run it without `NDEBUG`; it uses assertions. This check does not replace
on-device nonstandard-buffer and communication tests.

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

At the time of this initial scan, the +95.8% median difference was unresolved. A has substantial variability;
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
the work-dump runs. Baseline stability and the complete 5% acceptance gate were
still pending at this point; later linear-barrier measurements are recorded below.

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
coverage and the 5% performance acceptance matrix were pending at this point.

## Further NCCL work alignment (2026-11-10)

The reference is NCCL `v2.27.3-1`
(`72d2432094d6ae36abd6e511c3a16a2d052dbf94`). The device work now follows its
address/byte, rank, channel-count, FP8-chunk and flag prefix. The communicator
comes from the explicit shared context, including the `NO_TSM` path. Zero
channels denotes an absent direction; zero bytes with one channel still denotes
synchronization. Preprocessing only changes channel-local copies, retains active
channel counts, and clears directions whose part is out of range.

PCCL's inverse channel permutation and buffer-step partition strides remain.
Exact raw chunk fallbacks handle sizes not representable by FP8, so the complete
work stays 64 bytes rather than NCCL's 48. NET registration flags remain zero;
this does not add NET registration support. Existing IPC behavior is unchanged.
Private-only plans skip partition preprocessing and its barrier. Shared plans
use an out-of-line helper to reduce live state in the execution path.

Each send/recv primitive now receives `min(chunkSize, remainingBytes)`. The old
upward thread alignment could exceed the proxy receive bound. With
`PCCL_BUFFSIZE=5275648`, the pre-encoding implementation sent 165888 bytes against
a 164864-byte IB receive. The exact bound fixes this independently of FP8.

The existing PCCL thread assignment remains. A separate NCCL-style assignment
experiment increased the Simple stack from 412 to 440 bytes and was stopped at
the resource gate, without device or performance execution. It is not part of
this implementation and is not a measured runtime regression.

The final source was built remotely for PPU 1.5 in normal and `NO_TSM` modes.
`NO_TSM` has compile coverage only. All 63 nonregistered boundary cases passed:
six standard configurations and three custom-buffer configurations, seven cases
each. Actual LL selection, FIFO storage, 24 channels, SimpleBulk and two graph
replays with changed payloads were verified. The Simple kernel stack is 412
bytes, versus 580 before these three alignment steps; register and shared-memory
allocation did not increase.

The nonregistered performance scan covers AlltoAll, Gather and Scatter at all
27 sizes from 128 B through 8 GiB, using the default 20 iterations and linear
MPI barrier, on one-node 2/4/8 devices and two-node 16 devices over ICN and IB.
The fixed baseline is `ed532b3a4a3f2b543689688bbcd71972a3a40a57`. There are 405
size/configuration comparisons. Most use three interleaved launches per version;
Gather on local 2/4 devices and both two-node transports retains eight launches
per version, including the initial samples and independent five-round rechecks.

**The overall 5% performance gate is not yet passed.** 404 comparisons are within
5%; IB Gather on 16 devices at 512 KiB has a combined median of 24.225 -> 25.485 us
(+5.20%). Its independent five-round recheck is 24.26 -> 24.95 us (+2.84%). Slow
samples remain in both the raw results and combined statistics. The cause is
not established. Follow-up instrumentation was stopped before the candidate
launch when another container acquired the devices. No performance result from
that incomplete diagnostic is used for acceptance.

All AlltoAll and Scatter configurations are within the measured median gate.
For AlltoAll at 8 MiB, ICN medians are 74.11 -> 72.72 us and IB medians are
223.84 -> 222.62 us. These small differences do not establish a broad throughput
gain. Earlier Gather/Scatter outliers did not reproduce consistently; their
recovery cannot be attributed to the final metadata-store cleanup.

Final structural dumps match the baseline at every observed rank/channel:
AlltoAll on ICN at 8 MiB and 128 MiB has three uploaded plans per rank and eight
channels per plan, with one execution batch containing respectively 2 and 4
works per channel. At 128 MiB, 32 references use 16 unique 64-byte descriptors
instead of 32, reducing descriptor storage from 2048 to 1024 bytes. Gather on
four local devices matches all 81 uploaded plans per rank over the full scan.
These are upload/work-dump observations, not a separate driver-launch trace.

Final library SHA-256:
`6eca9e087eeabe3cabb30d5c562cc014659870e544b7572ede636fa3ca024389`.
Source archive SHA-256:
`a2274181700968d53d6bf9a48b9763b00472157374be5502724d662569606fec`.
The source archive contains the same tracked `src` content as this implementation;
validation/documentation changes are maintained separately.

An independent five-round full-range IB Gather reproduction on another idle
pair of the same device/driver type passed all 27 median comparisons: the largest
increase was 0.24% at 2 GiB, and 512 KiB was 24.95 -> 23.95 us (-4.01%). All ten
launches passed correctness, ownership and driver-health checks. These samples
are kept separate from the original pair. They do not reproduce a stable
regression, but do not waive the original +5.20% result or establish its cause.
The branch is available for review; overall performance acceptance stays pending.
