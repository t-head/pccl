# PCCL C++ host coverage experiment

This workflow (`PCCL host GCOV experiment`) accepts manual dispatch and a narrowly
scoped push on `codex/pccl-host-gcov-experiment` when its workflow, helper or
contract tests change. The branch-only push bootstraps the experiment in GitHub
Actions without adding it to the default branch. It uses 8 CPU jobs and `ppu_15`
device dependencies when no dispatch inputs exist. There is no scheduled or PR
trigger. It does not alter daily/sanity builds, their gates, or performance
baselines. It does not collect Python or PPU kernel coverage, and does not enable GitHub Code Quality.

## Scope and execution

The dedicated library build uses the opt-in `HOST_GCOV=1 GCOV=0 DEBUG=0` mode.
Host C++ compilation and linking use `--coverage`, with `-O0` overriding the normal
host optimization. Device dependencies remain complete, normally optimized and
uninstrumented; no fake kernels or restricted operation/data-type builds are used.
The existing default flags and legacy `GCOV=1 DEBUG=1` behavior remain unchanged.
It builds device dependencies but does not measure them.
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
