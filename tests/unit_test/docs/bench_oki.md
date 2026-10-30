# bench_oki Test Methodology

Source: `tests/unit_test/src/bench_oki.cu`. Executable: `tests/unit_test/build/bench_oki`.
Currently accepts named arguments only. Use `--help` to see all defaults, ranges, and constraints.

## Build and Environment

Run all commands from the repository root. Requires a PPU SDK with ltOki support, matching PCCL runtime libraries, and devices capable of P2P communication. Replace example paths with actual installation paths.
The unit-test build compiles this SIMT RDC target when ltOki is enabled; it's skipped when `USE_OKI=0`.
This target is not affected by other unit-test `--warp_exec_mode` settings.

```bash
export BENCH_SDK_DIR=/path/to/PPU_SDK
export BENCH_PCCL_LIB_DIR=/path/to/pccl/lib
source "$BENCH_SDK_DIR/envsetup.sh"
export LD_LIBRARY_PATH="$BENCH_PCCL_LIB_DIR:$BENCH_SDK_DIR/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
make -C tests/unit_test/src ../build/bench_oki PPU_ARCHS=ppu_15
./tests/unit_test/build/bench_oki --help
```

## Four Timing Modes

| `--mode` | What It Measures |
|---|---|
| `bandwidth-with-assembly` (default) | Uses hggcEvent on the execution stream to time the entire benchmark kernel, including reserve, WQE assembly, submission, and completion wait; reports GB/s. |
| `bandwidth-without-assembly` | Each repetition first runs a separate assembly kernel, then uses hggcEvent on the same stream to time submission and completion; excludes prepare, verification, and host sync overhead. |
| `latency-with-assembly` | Each round launches the kernel independently and uses hggcEvent to measure the total time for all participating QPs in that round, including reserve, address calculation, assembly, PostSend, RingDB, and PollCQ. |
| `latency-without-assembly` | Each round first launches a prepare kernel, then uses hggcEvent to time submission and completion for all participating QPs; up to 2048 copies per QP per round, zero allowed. |

Assembly is performed in all modes; the without-assembly variants simply exclude it from the timing window.
All timing (including memcpy baseline) consistently uses `hggcEventRecord`, `hggcEventSynchronize`, and `hggcEventElapsedTime`, with start/stop events and measured operations on the same stream.
Bandwidth modes record one event interval per repetition, covering all rounds within the kernel.
Latency modes launch and wait for the kernel independently per round, with QPs running in parallel within that round. Each repetition produces `rounds` samples, where each sample is the total time across all participating QPs. Per-QP cycles or nominal clock conversions are no longer reported.
The two latency modes synchronize between rounds, which differs from the original single-kernel multi-round execution pattern.
Event measurement captures the interval on the stream timeline and may include scheduling or queuing gaps during that period—it doesn't equal pure instruction time.

## Bandwidth Comparison with Same Workload

```bash
./tests/unit_test/build/bench_oki --mode bandwidth-with-assembly \
  --cmd-bytes 4K --mapping per-die --copies-per-round 2048 \
  --rounds 1 --inflight-rounds 1 --reps 3

./tests/unit_test/build/bench_oki --mode bandwidth-without-assembly \
  --cmd-bytes 4K --mapping per-die --copies-per-round 2048 \
  --rounds 1 --inflight-rounds 1 --reps 3
```

Both commands use the same data volume, batch count, and QP layout; however, prepare and cache states still differ, so the time difference shouldn't be interpreted as isolated assembly instruction overhead.
Change `--cmd-bytes` to `8K, 16K, 32K, 64K, 128K` to vary command size, where K=1024.
Total transfer per repetition = QP count × copies-per-round × rounds × cmd-bytes; bandwidth is reported in decimal GB/s.

## Parameter Selection

- `--copies-per-round`: Number of copies per QP per round, default 64. When copies are present, each round also adds an ICN fence and a CQ fence. Latency modes allow 0, meaning CQ fence only.
- `--rounds`: Number of rounds per repetition, default 1024. The bandwidth mode that excludes assembly defaults to and only allows 1.
- `--reps`: Default 3. Each run independently verifies data/CQ. Bandwidth reports results for each run plus min/median/max.
- `--inflight-rounds auto`: Default, uses conservative queue budget based on rounded reservation size. Explicitly exceeding budget is rejected. Both latency modes and bandwidth-without-assembly only allow auto or 1, actually using 1.
- `--blocks sweep`: Scans 1, 2, 4, 8, ...; positive number specifies any valid block count, e.g., 3.
- `--warps-per-block`: Default 1, fixed to one QP per lane 0 of each warp.
- `--mapping interleaved`: Default mapping, starting from the first QP, interleaving dies first then incrementing rings.
- `--mapping per-die`: Automatically uses 2 blocks × 8 warps, with each block submitting to one of two dies. Reports error on explicit conflict; doesn't guarantee SM die affinity.

## Exit Status

Exit codes: 0 success, 1 parameter/API error, 2 data/CQ verification failure.
Invalid numbers, negative values, overflows, unknown or duplicate options are rejected. Supports both `--key value` and `--key=value` formats.

## Automated Comparison Script

The script is located at `tests/unit_test/compare_bench_oki.py`. Run all commands from the repository root.
Default binary and output directory are in `tests/unit_test/build/`, independent of current working directory.

`tests/unit_test/compare_bench_oki.py` compares five groups by default: 16×1 interleaved pipeline, 2×8 interleaved pipeline, 2×8 per-die pipeline, 2×8 per-die single-batch with assembly, and 2×8 per-die single-batch without assembly.
Covers 4K–128K, with 3 independent processes per group per size and 3 repetitions per process, totaling 90 processes.
Uses the currently compiled binary without automatic rebuilds. Requires Python 3.6 or later with no third-party Python dependencies.

After completing the environment setup above, run from the repository root:

```bash
python3 tests/unit_test/compare_bench_oki.py --output tests/unit_test/build/oki_compare_run1
# Resume after interruption with the same parameters:
python3 tests/unit_test/compare_bench_oki.py --output tests/unit_test/build/oki_compare_run1 --resume
```

Continuous pipeline defaults to 64 copies × 1024 rounds per QP with inflight=auto; the two single-batch groups use 2048 copies × 1 round with inflight=1. First compares 16×1 and 2×8 interleaved layouts, then compares mapping variations within the same 2×8 layout. Single-batch only compares the two assembly timing ranges to avoid misinterpreting batch changes as assembly gains.

The primary statistic is the median of repetitions across the median of trials. The script outputs HTML, CSV, JSON, raw logs, and complete reproduction commands, preserving all samples and cross-process variation. Device busy, non-zero exit, timeout, verification failure, missing results, or statistical anomalies will halt execution. Failed data is excluded from aggregation. The script checks binary, dependency, and self checksums; configuration or build cannot be changed on resume. After timeout, check device status before deciding whether to resume.

Use `--dry-run` to preview commands. Customize with `--sizes 4K,8K`, `--trials`, `--reps`, etc. See `python3 tests/unit_test/compare_bench_oki.py --help` for details.

### Scanning Command Sizes

```bash
python3 tests/unit_test/compare_bench_oki.py \
  --sizes 4K,8K,16K,32K,64K,128K \
  --batch-copies 2048 --trials 3 --reps 3 \
  --output tests/unit_test/build/oki_bandwidth_compare
```

In `report.html`, compare `bandwidth-with-assembly` and `bandwidth-without-assembly` for the same batch, checking `median_GB/s` and their ratio. Both groups maintain the same data volume and layout. The ratio reflects the bandwidth difference after changing the timing range and should not be interpreted as an end-to-end program speedup.

### Script Parameters and Output

| Parameter | Function and Default |
|---|---|
| `--binary PATH` | Benchmark path, defaults to `tests/unit_test/build/bench_oki` in repo. |
| `--smi PATH` | Specify `ppu-smi` path; recommended for other installation layouts. |
| `--sizes LIST` | Comma-separated command sizes, default `4K,8K,16K,32K,64K,128K`. |
| `--trials N` / `--reps N` | Independent process count / repetitions per process, both default 3. |
| `--pipeline-copies N` / `--pipeline-rounds N` | Copies per round / rounds for pipeline group, defaults 64 / 1024. |
| `--batch-copies N` | Copy count for single-batch comparison group, default 2048. |
| `--timeout N` | Single process timeout in seconds, default 120. |
| `--output DIR` | New output directory; defaults to timestamped directory. |
| `--resume` | Resume existing output directory with same config, binary, dependencies, and script. |
| `--dry-run` | Preview commands only, no device access or output file creation. |

Output includes `report.html`, `summary.csv`, `repetitions.csv`, `runs.csv`, `results.json`, `manifest.json`, `runner.py`, and raw logs in `logs/`.

Before each run, the script checks all PPUs for running processes—this check doesn't provide exclusive locking.
Results exceeding 400 GB/s also trigger an anomaly halt. This threshold is a validation constraint for the current script and should be checked for applicability when used with other link specifications.

Bandwidth per-run timing field is `event_ms` (milliseconds); aggregated `min_ms/median_ms/max_ms` also come from event timing. Latency output is in µs with `samples = rounds × reps` and `scope=all-QPs`.
Old and new host/clock64 timing results cannot be directly mixed as the same statistical baseline.

## ICN Data Completion Order

Each QP's submission order per round: all copy WQEs → ICN fence (`relax=false`) → CQ fence.
Each QP's remote fence value resides on the target device in space independent of the 256 MiB data window.
The ICN fence's data parameter receives the incrementing sequence number for this round. Normal execution and pre-assembly share this order. Data and CQ values are checked after each repetition; remote fence storage values are not interpreted by CQ sequence number.

When copies are present, `copies + 2` WQEs are reserved; for zero copies, only one CQ fence WQE is reserved.
With copies, automatic inflight budget is calculated accordingly: `max(1, QP_DEPTH / nextPow2(copies + 2) / 8)`.
Copy limit for modes with assembly is `QP_DEPTH / 8 - 2` (4094 for depth 32768); modes without assembly still cap at 2048. Bandwidth numerator only counts copy data volume; fence overhead is included in the event interval.
