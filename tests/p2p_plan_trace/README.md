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
