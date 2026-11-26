# PCCL C++ host coverage experiment

The graph-only workflow (`PCCL host GCOV experiment`) was bootstrapped via a
branch-only push. It now runs only on manual dispatch; pushes for the M890P CX8
experiment do not also launch the CPU graph tests. There is no scheduled or PR
trigger. It does not alter daily/sanity builds, their gates, or performance
baselines. It does not collect Python or PPU kernel coverage, and does not enable GitHub Code Quality.

## Scope and execution

The dedicated library build uses the opt-in `HOST_GCOV=1 GCOV=0 DEBUG=0` mode.
Host C++ compilation and linking use `--coverage`, with `-O0` overriding the normal
host optimization. The default `host_archive` mode builds all host C++ objects and
links the actual graph-test dependencies from `libpccl_host_gcov.a`, discarding
unreachable sections with normal linker garbage collection. No unresolved-symbol
errors are ignored, and no fake kernels or changed assertions are introduced.
This archive is **not a complete PCCL runtime** and cannot validate communication
or PPU kernel execution. The `full` mode remains available for a complete runtime
build with all device dependencies normally optimized and uninstrumented.
The existing default flags and legacy `GCOV=1 DEBUG=1` behavior remain unchanged.
Both documented graph-search suites run: `graphSearchTestWithoutPXN.*` and
`graphSearchTestWithPXN.*` (with `--pxn`). These tests use topology XML fixtures;
no PPU device is passed to the CPU-only job. CPU execution with the selected SDK
must be verified experimentally; a build/test failure remains a failure.

The report includes `src/` host source, excluding CUDA files, collective device
code, public device headers and the bundled DOCA GPUNetIO vendor tree. Tests,
SDK headers, generated code outside `src/`, and other third-party packages are
not part of the denominator. Graph tests primarily exercise topology planning;
the resulting percentage is not comprehensive communication-path coverage.

## Outputs

Artifact: `pccl-host-coverage-<run_id>-<run_attempt>`:

- `html/index.html` and per-source annotated HTML;
- `coverage.xml` (Cobertura), `gcovr-summary.json`, `summary.json`, `summary.md`;
- `.gcno`/`.gcda` in `profiles.tar.gz`, build/test/gcovr logs and GTest XML.

No arbitrary percentage threshold is imposed. Zero executed tests, absent GCOV
data, zero executed product lines, or out-of-scope source are collection errors,
not successful 0% reports. Available coverage after a failed test is retained,
but the job remains failed. Termination can prevent counters from being written.
This first experiment uses serial processes; adding MPI tests later requires
explicit rank/process isolation and consistent-build aggregation.

## Local invocation in a Linux SDK environment

Install gcovr 8.4 and the existing PCCL/Google Test build prerequisites. Use a fresh
build directory and report directory; old profiles are never reused or deleted.

```sh
source /usr/local/PPU_SDK/envsetup.sh
python3 .github/workflows/scripts/pccl_host_coverage.py \
  --build-dir coverage/local-host/build --output coverage/local-host/report
```

The script records source revision and compiler/profiler/SDK metadata. The selected
SDK image uses the current PCCL wrapper libraries; the test link command overrides
its standalone legacy runtime names without changing test assertions.

Validation of helper fixtures or the reporting tool on macOS is not proof that
PCCL compiled or that product coverage was collected. Keep those receipts separate
from a real Linux SDK experiment. Publishing the branch, dispatching Actions, and
remote execution need their respective authorization; no push is implicit here.

## Validation record — 2026-11-19

The initial experiment branch was based on
`ffc3121da994a3f723028e4d76a734d9bc718b99`. Local helper contracts passed (14 tests),
as did the full `tests/xtest` suite at that revision (172 tests). Workflow syntax
was checked with actionlint and the existing custom runner label configured.
The macOS synthetic reporting fixture validates the reporting tools only; its
percentage is not PCCL product coverage.

Two CPU-only SDK-container attempts were made on the authorized Linux validation
host. No PPU devices were passed through:

| Attempt | Source snapshot SHA-256 | Dependency architectures | Result |
| --- | --- | --- | --- |
| Initial | `4215c91b78e35c62787b45412b9154dcc83cb1db030f52c14fc50a1723d324c3` | `ppu_10,ppu_15` | Docker stop / SIGTERM, exit 143, before library completion |
| Fresh retry | `7bf532fbf2cc255383a4916755381d36e5bc7b5fb720b63e4ccbc13815ecae95` | `ppu_15` | Docker stop / SIGTERM, exit 143, before library completion |

Docker recorded stops at 16:15:07 and 16:21:13 Asia/Shanghai. Neither container
reported OOM. These are interrupted attempts, **not passing builds**, completed
C++ test runs, or successful product coverage reports. The sender/reason for the
stop requests remains unconfirmed. The container's Docker exit status is
canonical; the bootstrap shell's EXIT-trap file recorded `0` during termination
and must not be used as a success signal.

The retry used a cached dependency layer derived from the same SDK image, but a
fresh, SHA-256-verified source archive and empty product build directory. It did
not reuse object files or counters from the first attempt. Diagnostics are kept
separately from the synthetic fixture and from any future successful report.
At the end of these host attempts no branch had been committed/pushed and no
GitHub Actions run had been dispatched. A complete instrumented build, both graph suites, and genuine nonzero PCCL host
coverage still require validation before this workflow can be considered ready.


### GitHub trial preparation

Only the four experiment files were moved onto the latest default-branch base,
`0da8662eb0e6d164ebd790d4a45b6007314a8b10`, for the GitHub trial. Prior unmerged
CI commits were not carried over. On that base the updated helper contracts
passed (15 tests), and the full available `tests/xtest` suite passed (105 tests).
Workflow syntax and the exact branch/path trigger restrictions were checked.
These are local tool/workflow tests, not completed PCCL C++ tests. The branch-only
push will perform the first GitHub build/test/report validation; consult its
Actions result before treating the experiment as ready.


### First GitHub trial results

- Run `44489627339`, commit `9fe0972`: dependency installation and helper tests
  passed, but reading Git HEAD failed on container workspace ownership before any
  build. Checkout is now pinned to `github.sha` and that SHA is passed explicitly.
- Run `44490388941`, commit `36eeaed`: the combined host/device debug build reached
  its 40-minute deadline while compiling device dependencies. It produced 120 host
  notes and 263 device notes, but no tests ran and no counters existed. This is a
  timeout, not a passing compilation or a valid product coverage percentage.

The next trial therefore uses host-only instrumentation while building **all**
device dependencies normally. Helper fixture metrics are also isolated from
`GITHUB_STEP_SUMMARY`; mocked percentages are never product coverage. No Issue or
PR was created for this isolated experiment. The host-only trial still requires
an actual complete build, both graph suites and a verified report.


### Native host graph trial

Run `44498202039`, commit `af0b4f0`, also timed out during the full device-dependency
build after 40 minutes. Host-only instrumentation was effective (119 host notes,
no device notes), but no tests ran and no counters existed. The default trial now
builds a native host archive to exercise the actual graph C++ code independently
of the out-of-scope PPU compiler workload. Select `library_kind=full` to retry the
complete runtime build separately. Native archive success must never be presented
as successful full runtime compilation, PPU correctness, or performance data.


The host coverage helper tests live in `tests/host_coverage/` and run explicitly
in this experiment. They are not xTest executor tests and do not change the
existing daily/sanity regression inventory or gates. Both inventories are
validated separately when publishing the experiment.


Run `44507363579`, commit `ff40563`, successfully compiled the native host archive
but failed to link the tests because their shared-library interposing globals
collided with production globals in the archive. Static host mode now uses the
production globals without linker error suppression, while preserving the
original fixture parameter defaults (`PCCL_NO_MEMSET=0`,
`PCCL_CHANNEL_SHUFFLE=0`) unless explicitly configured otherwise. No assertions
or Case filters were relaxed.


Run `44509124740`, commit `e05c684`, compiled the native archive and test executable
and passed all 30 graph tests (21 without PXN, 9 with PXN). Report collection
failed only on gcovr's default suspicious-hit heuristic: the NET-selection loop
condition had 5,052,574,822 executions. Matching GNU GCOV 13.3 JSON and raw GCDA
were audited: 4,129,114,662 loop-body traversals plus 923,460,160 exits exactly
match the condition count, with 18,944,829 function entries. This supports treating
it as a legitimate large counter, not a failed test or fabricated coverage.

The experiment now uses a finite suspicious-hit bound of `2^40` instead of the
32-bit heuristic. Counters are not rewritten; negative-hit, malformed-data,
missing-source and other GCOV parser errors remain fatal. Raw profiles and the
configured bound remain in the report for audit. The prior failed run remains
failed; a fresh successful Actions result is required for the updated collector.

## Successful native host report — 2026-11-19

GitHub Actions run `44512261349` validated implementation commit
`4db6626b41bdf3b6ad03bc7218dd2a09ecd3da40`. Native host archive compilation,
test-executable linking, both existing graph suites, report collection and artifact
upload all passed. The tests ran without PPU device mappings: 21 without PXN and
9 with PXN; all 30 passed, with no failures, errors, disabled or skipped tests.

| Host-source metric | Covered / instrumented | Percentage |
| --- | --- | --- |
| Lines | 1,996 / 36,272 | 5.50% |
| Functions | 185 / 2,765 | 6.69% |
| Branches | 1,810 / 75,525 | 2.40% |
| Graph-directory lines | 1,627 / 3,685 | 44.15% |

The report contains 154 host-source files and 120 `.gcno` / 109 `.gcda` files in
the raw profile archive. Source identity, scope, XML/JSON counts, GTest totals and
profile inventory were checked after downloading the artifact. The denominator
is the instrumented host build, not every line in the repository. The low overall
percentage reflects the deliberately graph-focused test scope: communication,
transport, scheduler and most runtime paths have not been exercised here.

Artifact `pccl-host-coverage-44512261349-1` includes annotated HTML, Cobertura XML,
JSON summary, raw profiles, compiler flags and build/test/collector logs. The
complete job took about 50 minutes, mostly SDK container initialization (about
46 minutes); the native build, graph tests and collection took about 2m33s.

**This is successful native host graph coverage, not a passing full-runtime build,
PPU kernel correctness result or performance baseline.** The earlier full-runtime
attempts remain recorded as timeouts. No Issue or PR was created and no default
branch, daily/sanity workflow or baseline data was modified.

## Two-node M890P CX8 experiment

`PCCL M890P CX8 host coverage experiment` is separate from the graph-only job
and normal CI. Only `codex/pccl-host-gcov-experiment` can start its jobs. It has
no schedule, PR trigger, or 810E or larger-node matrix.

- Build the complete core `libpccl.so` on a CPU-only runner, with the sanity
  SDK/image and optimized PPU kernels. Use `HOST_GCOV=1 GCOV=0 DEBUG=0` and
  atomic host profile counters. Only build the pinned MPI AllReduce benchmark;
  its code and optional plugin code are not part of the coverage report.
- Bind the library, compilation notes and benchmark to a checksummed artifact.
  Allocate two CX8 workers in the same rack only after a successful build. Use
  the existing two-node sanity Case: 16 ranks, 32 MiB float AllReduce sum, 10
  iterations. Require the benchmark's zero-mismatch correctness summary.
- Use xTest's rank-local device mapping and recovery. Each MPI rank verifies
  the loaded library path and writes its own counters and transport log.
  Check both workers for cross-host IB/RoCE channels, reject Socket data
  transport, and require profiles for all 16 ranks before merging.
- Merge counters from this build only with the matching GNU tools. Upload raw
  per-rank evidence, build/case logs and host-only HTML/Cobertura/JSON output.
  Failed tests, missing ranks or missing RDMA proof do not count as a passed
  coverage run. Instrumented timing is never used as a performance baseline.

The first hardware run is required to validate the SDK build and runtime on
`ICN64_890P_CX8`; local contract tests do not substitute for that result.
