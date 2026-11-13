# Minimal CSV executor

The maintained Plan/Suite/Case definitions are exported by the trusted-side
`github_csv_export.py` tool. The public executor uses only the Python standard
library and does not install or import xtest/mmengine.

## Inputs

- `minimal-csv/export.json`: format and source/runner revision metadata.
- `minimal-csv/suites.csv`: shared resources, outer timeout and environment.
- `minimal-csv/standard.csv` and `minimal-csv/notsm.csv`: profile-specific Cases.
- `minimal-csv/envs/`: data-only shared environment files.

The composite Case key is `(plan, suite, name)`. Commands are captured at the
real Runner's `Popen(shell=True)` boundary. The executor does not parse or
reconstruct Runner arguments.

Validate an export without running Cases:

```bash
python3 -S -B tests/xtest/run_cases.py \
  --input tests/xtest/minimal-csv --profile standard --nodes 1 --validate-only
python3 -S -B tests/xtest/run_cases.py \
  --input tests/xtest/minimal-csv --profile notsm --nodes 1 --validate-only
```

Run a profile after the matching build has been staged:

```bash
python3 -S -B tests/xtest/run_cases.py \
  --input tests/xtest/minimal-csv --profile standard --nodes 1 \
  --test-root /workspace/build-output \
  --binary-dir /workspace/build-output/pccl-tests/build \
  --build-record /workspace/build-cache/output_standard.build.json \
  --pccl-revision 0123456789abcdef0123456789abcdef01234567 \
  --expected-sha256 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --hostfile /workspace/hosts.cfg \
  --output /workspace/results/standard
```

Use `--nodes 1` for single-node Cases and `--nodes 2` for the established
two-node Case; without a filter, every Case in the profile is selected.
CI preparation and result reconciliation both validate the complete exports
before selecting jobs. The supported layout is standard/one-node,
standard/two-node and notsm/one-node, with `VALIDATION_CARDS` cards per node.
Any exported Case outside that layout fails explicitly; it is never silently
dropped by a node filter.
`--hostfile` is required only when the selected profile contains multi-node
Cases. `test_root` may either contain `run_test.py` directly or contain a
`pccl-tests` checkout. Each Case gets a separate working directory with
explicit `run_test.py`, `run_test_config.py`, `build` and optional `hosts.cfg`
links. Private `python`/`python3` launchers for the active interpreter are added to
`PATH`, so CSV commands see the same Python environment as the executor
(including its benchmark dependencies), even when the system `python3` differs.

Only literal `{test_root}`, `{binary_dir}`, `{result_dir}`, `{path}`,
`{ld_library_path}`, `{hostfile}` and `{mpi_interface}` bindings are recognized.
The last four are bound only when needed. Use `--mpi-interface` to supply the
allocated worker's actual TCP interface when the command contains that binding. Environment files are parsed as data and are never
sourced. Commands alone are passed to `/bin/sh -c`.

Execution requires the build record that was validated with the restored
archive. The executor checks its PCCL and runner revisions, build profile and
archive SHA-256, then retains the identity as `build.json` beside the results.

## Result semantics

- `exit_code`: command success.
- `validation_summary`: command success plus a successful validation summary.
- `performance`: the above plus a valid performance table and `metrics.csv`.
  There is no baseline, bandwidth, latency or sample-stability gate.
- A single-node failure does not prevent later Cases from running.
- A multi-node failure stops later Cases because local process cleanup cannot
  prove remote MPI rank cleanup. Multi-node commands run only on the master.

`raw.log` is the unmodified diagnostic stream used for checks. Results and raw
logs must remain in the private results area; this executor intentionally
has no generic public-log redactor. Infrastructure remains responsible for
node allocation, MPI/RDMA setup and remote cleanup.

## Artifact-backed build distribution

`source-sanity.yml` treats GitHub
Actions artifacts as the authoritative build store. A per-profile staging job
validates a run-scoped NAS cache or downloads the artifact once, verifies its
source revisions, profile and SHA-256, and publishes it for all consumers. The
standard single-node and two-node jobs therefore share one download.

Each hardware consumer also validates the cache before allocating a worker.
After a failed run has been cleaned, “Re-run failed jobs” can therefore
download the retained build artifact and atomically republish the same
validated cache without rerunning the successful build or staging jobs.
Publication uses a persistent per-cache advisory lock outside the replaceable
directory. Revalidation, invalid-cache removal and publication hold the same
lock, so concurrent retry consumers cannot delete another publisher's valid
cache. Lock acquisition has a five-minute timeout. The NAS must support shared
`flock` locks; these helpers target the Linux CI environment.
Before publication, the cache root, profile parent and cache directories are
set to `0755` and artifact files to `0644`, allowing hardware workers with a
different NAS identity to traverse and read them. The run root and results
permissions are left unchanged so workers can still write their results.

Build-time NAS staging is removed after the required artifact upload. After
result reconciliation, a final `always()` cleanup removes only this run's
`staging/`, `cache/` and `shared-standard/` directories. `results/` and its raw
logs remain intact. Cache reuse is content-verified; file existence alone is
never sufficient.

## Experimental M890P daily execution

`daily-minimal-csv` contains 19 smoke Cases on one and two nodes.
`daily-full-csv` currently contains 8,899 standard Cases from 34 Suites, with
unsupported or out-of-scope selections recorded in `coverage.json`. The notsm
daily profile is intentionally empty.
The current resource policy executes only 1/2/4-node groups: 8,180 Cases in 20
groups. The 710 eight-node and 9 sixteen-node Cases remain in the export but are
listed as `EXCLUDED_BY_POLICY` in `execution-coverage.json`; they are not
reported as passed or skipped tests.

`group_cases.py` derives deterministic execution groups by profile, node count,
cards per node and shared environment, then splits large groups by
`--max-cases`. There is no hand-maintained shard list. Optional `--nodes` and
`--suite` arguments are debug filters, not shard definitions.

```bash
python3 -S -B tests/xtest/group_cases.py \
  --input tests/xtest/daily-minimal-csv \
  --output /tmp/daily-case-groups --max-cases 500
```

The group tree is hashed over every relative filename and file byte. A GitHub
artifact is the backup and one validated copy is published to the run-scoped
NAS cache with read-only files. Missing or invalid caches are restored from the
backup with an atomic, interprocess-locked publication; an invalid tree is
quarantined rather than deleted while readers may still use it. Every execution
group creates a private, revalidated, read-only copy before running any Case, so
concurrent recovery cannot change the commands or environment being executed.
Build and group downloads use bounded retries, artifacts are retained for seven
days, and no cache is shared across workflow runs.

`daily_reconcile.py` checks exact Case coverage, result records, raw logs,
performance metrics and the verified build identity for each group. Result CSV
and JSON records must have the complete schema and matching values. Performance
metrics are regenerated from the raw log and compared with every saved row.
Sizes and counts may be zero after rank division, times must be positive, and
bandwidth must be nonnegative. Daily results produce summaries and failure
notifications, not a PR required check.
Both preparation and final summary reject execution groups or unexpected result
directories that exceed the allowed node counts.

The internal `results` job loads the selected CSV Cases again. It rejects
missing, duplicate or unexpected Case keys, empty raw logs, missing or
inconsistent per-Case `result.json` records, and missing, empty or inconsistent
performance metrics. It also compares every retained build identity with the
build-job outputs, including requiring standard single-node and two-node
results to use the same archive SHA-256.

For pull requests, this internal job requires successful secret scanning,
commit checks, AI review, human approval and preparation. Approved runs still
reconcile and clean up after build or hardware failures. Cleanup runs after a
successful checkout even if reconciliation or artifact publication fails.

The final `PCCL Tests Gate` always runs on a GitHub-hosted runner with no token
permissions, checkout, repository-script execution or NAS access. It rejects
unsupported events and requires every build, hardware, result and cleanup job
to succeed. PR-only review skips are accepted only for main pushes and manual
dispatch; failed, cancelled or unexpectedly skipped required jobs fail the gate.
