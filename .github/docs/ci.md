# CI checks and PPU tests

Maintained by the `ci` module. See the [workflow overview](../WORKFLOWS.md) for required PR gates.

CI scans for secrets and checks commit messages and DCO sign-offs. After AI review passes
and a reviewer with repository write access approves the PR, CI builds PCCL and runs the
PPU tests. `PCCL Tests Gate` reports the results of the required jobs.

| File | Purpose |
| --- | --- |
| [source-sanity.yml](../workflows/source-sanity.yml) | The sole PPU test pipeline and publisher of `PCCL Tests Gate`: PR prechecks and AI/human approval precede self-hosted builds, artifact-backed build distribution, one single-node job per profile, and one two-node job reusing the standard build. |
| [pccl-daily-full.yml](../workflows/pccl-daily-full.yml) | Experimental dual-platform M890P/810E daily smoke/full execution with automatic same-environment groups, artifact-backed build and Case-group distribution, and exact per-group reconciliation. Full execution selects 1/2/4/8-node groups and is not a PR gate. |
| [workflows/scripts/](../workflows/scripts/) | Build PCCL and the pinned public benchmark runner. |

The daily workflow generates its inputs from JSON test families. Its only dataset names are
`smoke` and `full`; the former `generated-smoke` and `generated-full` names do not exist.
`smoke` intentionally contains two single-node Cases. For multi-node debugging, select
`full` together with `nodes=2`, `nodes=4` or `nodes=8`, and optionally an exact Suite name.

`source-sanity.yml` runs on same-repository and fork pull requests, pushes to `main`, and manual dispatch. Every PR runs secret scanning, commit checks, AI review and human approval on GitHub-hosted runners first. Self-hosted builds and internal result processing start only after those gates succeed, so an unapproved fork PR receives no self-hosted allocation. The maintained Plan/Suite/Case definitions are the only test source. Their reviewed commands are exported to the versioned CSV files under `.github/xtest/minimal-csv`; GitHub does not install or import xtest/mmengine.

The final gate always runs on a GitHub-hosted runner without token permissions,
checkout, repository-script execution or NAS access. Internal result reconciliation
and cleanup are a separate `results` job; PRs must pass secret scanning, commit
checks, AI review, human approval and preparation before it can run. Approved
failed runs still clean up, while rejected PRs never execute internal result scripts.

Build jobs validate and publish the current Run's build archive to its run-scoped NAS cache before uploading the same files as a GitHub artifact backup. Test jobs probe that cache first and download the artifact only when the cache is missing or invalid. Successful runs remove transient build caches after testing and backup; failed or cancelled runs retain the formal cache for failed-job retries. Cache checks still enforce source revisions, build profile and archive SHA-256, while NAS permission or mount errors fail explicitly instead of being treated as cache misses.

## CSV Cases and commands

- `suites.csv` holds the composite Plan/Suite key, node and card resources, outer timeout and shared environment filename.
- `standard.csv` and `notsm.csv` hold one Case per row with `(plan, suite, name, check, cmd)`.
- `export.json` records the format, source and public-runner revisions and the exporting xtest version.
- `run_cases.py` executes the captured command through `/bin/sh -c` without reconstructing Runner arguments. Only documented literal runtime bindings are expanded.

Preparation and result reconciliation validate the full exports against the
scheduled resource layout: standard on one or two nodes and notsm on one node,
with `VALIDATION_CARDS` cards per node. Unsupported node or card requests fail
before execution rather than disappearing from the selected coverage.

Validate selections without running hardware:

```bash
python3 -S -B .github/xtest/run_cases.py   --input .github/xtest/minimal-csv --profile standard --nodes 1 --validate-only
python3 -S -B .github/xtest/run_cases.py   --input .github/xtest/minimal-csv --profile notsm --nodes 1 --validate-only
```

Result checks are `exit_code`, `validation_summary` and `performance`. Performance requires command success, a validation summary and a valid table, and emits `metrics.csv`; there is no baseline or sample-stability gate. Raw logs remain unmodified in the private results area.

## Build artifacts and NAS cache

standard and notsm are each built once and must be uploaded as GitHub Actions artifacts with source revisions, profile and archive SHA-256. A per-profile staging job reuses a valid run-scoped NAS cache or downloads the artifact once and validates it before publication. The standard single-node and two-node jobs therefore share one artifact download, while each single-node job extracts into its own local workspace.

The two-node master extracts to a NAS directory visible to both workers and removes that directory through an EXIT trap. After result reconciliation, the final cleanup removes only this run's `staging/`, `cache/` and `shared-standard/` directories. Results and raw logs are preserved. Cache hits and cleanup never rely on file existence alone: source revisions, profile, SHA-256 and run-scoped paths are verified. Build artifacts have an explicit retention period to control multi-GB storage use.

Retry publication takes a per-cache interprocess lock outside the directory being
replaced, with a five-minute wait limit. The NAS must honor shared advisory locks.
The cache root, profile parent and cache directories are searchable, and artifact
files are readable by different worker identities before publication (`0755`
directories, `0644` files), including the retry path. The writable run root and
results permissions are not narrowed by this repair.
