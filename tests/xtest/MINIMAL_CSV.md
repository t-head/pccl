# Experimental minimal CSV executor

This directory contains a local experiment, not a workflow or PR switchover.
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
explicit `run_test.py`, `run_test_config.py`, `build` and optional `hosts.cfg`
links. Private `python`/`python3` launchers for the active interpreter are added to
`PATH`, so CSV commands see the same Python environment as the executor
(including its benchmark dependencies), even when the system `python3` differs.

Only literal `{test_root}`, `{binary_dir}`, `{result_dir}`, `{path}`,
`{ld_library_path}`, `{hostfile}` and `{mpi_interface}` bindings are recognized.
The last four are bound only when needed. Use `--mpi-interface` to supply the
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
logs must remain in the private results area; this experiment intentionally
has no generic public-log redactor. Infrastructure remains responsible for
node allocation, MPI/RDMA setup and remote cleanup.

No hardware validation has been performed by this experiment.
