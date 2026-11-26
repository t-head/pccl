# PCCL C++ host coverage on M890P CX8

`PCCL M890P CX8 host coverage` is an on-demand GitHub Actions workflow. Once
available on `main`, open **Actions → PCCL M890P CX8 host coverage → Run
workflow** and select `main` in `t-head/pccl`. It does not gate daily, sanity,
or merges. To validate before integration, this version also runs when the
workflow file changes on the isolated `codex/pccl-m890p-coverage-on-demand`
branch. Remove this branch-only push trigger before integration; `main`
remains manual-only.

## Scope and execution

- Use the same M890P `full` definitions and grouping as daily, restricted to
  **1 or 2 nodes** with the CX8 selector. Keep the Case commands, validation,
  card allocations and rack placement. This includes the existing two-rack
  Cases; there is no additional RDMA, Socket, or ICN/MNNVL acceptance check.
- At the current selection revision (`ec92aca`), the matrix is **7,783 Cases**:
  5,859 single-node and 1,924 two-node (9 cross-rack). The selection is generated
  from the checked-out definitions, not a fixed list. It excludes 810E and all
  Cases that require more than 2 nodes.
- The workflow builds one instrumented PCCL runtime on a CPU runner
  (`HOST_GCOV=1 GCOV=0 DEBUG=0`, GNU coverage instrumentation, atomic counters,
  `-O0`) and builds the normal daily benchmarks and native test executables
  against it. PPU kernels keep their release optimizations. The build and Case
  inputs are checksummed and staged once on NAS. The current 20-group matrix
  runs at most 4 groups in parallel, with up to 500 Cases per group.
- The existing xTest CSV executor runs and checks each group. Benchmark
  wrappers verify the loaded PCCL library and write GCOV data in independent
  slots per attempt, group, worker and local process rank. Sequential Cases
  accumulate in those slots instead of leaving separate counter trees per Case.
- A CPU reporting job validates the group results against the selection,
  verifies the counters against the same build's notes and GNU toolchain, and
  merges raw counters before rendering with gcovr 8.4. It does not average
  per-group coverage percentages.

The report covers PCCL host source under `src/`. Device code, tests, benchmark
code, external SDK headers, generated files outside `src/`, and the bundled
DOCA GPUNetIO vendor tree are not in the coverage denominator. Instrumented
run timings are **not** performance baseline data.

## Artifacts and incomplete runs

Open the run's **Summary → Artifacts** and download
`pccl-m890p-gcov-<run_id>-<run_attempt>-report`. Extract its ZIP before opening
`html/index.html`. It also contains `coverage.xml` (Cobertura),
`gcovr-summary.json`, and a `summary.json` that reconciles planned Cases,
statuses and counter availability by group.

If a Case fails or a group is missing, a report is uploaded when there are
valid counters from completed groups. It is marked **partial** and the
workflow does not pass. An incomplete group without reconciled Case results
cannot contribute counters. If no valid counters remain, only diagnostics
are uploaded, not a misleading 0% report. Check the per-group `-shard-…`
artifacts, the aggregate `-diagnostics` artifact and the job logs. Successful
runs clean their attempt-scoped NAS directory; incomplete runs retain it for
investigation. Use **Re-run all jobs**, not **Re-run failed jobs**, so the
runtime, selection and counters all come from the same attempt. Artifacts are
retained for 14 days.

## Prior hardware validation (single Case only)

The experimental workflow at commit `42885bb7c4fc213e8146248e07bacbd8396d4207`
passed [run 45778165684](https://github.com/t-head/pccl/actions/runs/45778165684)
on November 26, 2026. That was a **single** same-rack AllReduce (2 nodes × 8
cards), not a full-matrix run. The [report ZIP](https://github.com/t-head/pccl/actions/runs/45778165684/artifacts/15101166023)
contained host coverage for 157 files: 29.98% of lines, 40.92% of functions
and 10.67% of branches. The full coverage matrix still needs its own
end-to-end hardware validation.
