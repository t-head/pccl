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
  --hostfile /workspace/hosts.cfg \
  --output /workspace/results/standard
```

Use `--nodes 1` for single-node Cases and `--nodes 2` for the established
two-node Case; without a filter, every Case in the profile is selected.
`--hostfile` is required only when the selected profile contains multi-node
Cases. `test_root` may either contain `run_test.py` directly or contain a
`pccl-tests` checkout. Each Case gets a separate working directory with
explicit `run_test.py`, `run_test_config.py`, optional `unit_test.py`, `build`
and optional `hosts.cfg` links. Private `python`/`python3` launchers for the active interpreter are added to
`PATH`, so CSV commands see the same Python environment as the executor
(including its benchmark dependencies), even when the system `python3` differs.

Only literal `{test_root}`, `{binary_dir}`, `{result_dir}`, `{path}`,
`{ld_library_path}`, `{hostfile}`, `{mpi_interface}` and `{python_env}` bindings
are recognized. The last five are bound only when needed. Use `--mpi-interface` to supply the
allocated worker's actual TCP interface when the command contains that binding. Environment files are parsed as data and are never
sourced. Commands alone are passed to `/bin/sh -c`.

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

Build-time NAS staging is removed after the required artifact upload. After
result reconciliation, a final `always()` cleanup removes only this run's
`staging/`, `cache/` and `shared-standard/` directories. `results/` and its raw
logs remain intact. Cache reuse is content-verified; file existence alone is
never sufficient.
