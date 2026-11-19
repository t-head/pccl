# GitHub workflows

CI and module review run independently. Pull requests targeting `main` must pass both
`PCCL Tests Gate` and `PCCL Review Gate`.

## CI checks and PPU tests

CI scans for secrets and checks commit messages and DCO sign-offs. After AI review passes
and a reviewer with repository write access approves the PR, CI builds PCCL and runs the
PPU tests. `PCCL Tests Gate` reports the results of the required jobs.

| File | Purpose |
| --- | --- |
| [source-sanity.yml](workflows/source-sanity.yml) | The sole PPU test pipeline and publisher of `PCCL Tests Gate`: PR prechecks and AI/human approval precede self-hosted builds, artifact-backed build distribution, one single-node job per profile, and one two-node job reusing the standard build. |
| [m890p-daily-full.yml](workflows/m890p-daily-full.yml) | Experimental M890P daily smoke/full execution with automatic same-environment groups, artifact-backed build and Case-group distribution, and exact per-group reconciliation. Full execution selects 1/2/4/8-node groups and is not a PR gate. |
| [workflows/scripts/](workflows/scripts/) | Build PCCL and the pinned public benchmark runner. |

The daily workflow generates its inputs from JSON test families. Its only dataset names are
`smoke` and `full`; the former `generated-smoke` and `generated-full` names do not exist.
`smoke` intentionally contains two single-node Cases. For multi-node debugging, select
`full` together with `nodes=2`, `nodes=4` or `nodes=8`, and optionally an exact Suite name.

`source-sanity.yml` runs on same-repository and fork pull requests, pushes to `main`, and manual dispatch. Every PR runs secret scanning, commit checks, AI review and human approval on GitHub-hosted runners first. Self-hosted builds and internal result processing start only after those gates succeed, so an unapproved fork PR receives no self-hosted allocation. The maintained Plan/Suite/Case definitions are the only test source. Their reviewed commands are exported to the versioned CSV files under `tests/xtest/minimal-csv`; GitHub does not install or import xtest/mmengine.

The final gate always runs on a GitHub-hosted runner without token permissions,
checkout, repository-script execution or NAS access. Internal result reconciliation
and cleanup are a separate `results` job; PRs must pass secret scanning, commit
checks, AI review, human approval and preparation before it can run. Approved
failed runs still clean up, while rejected PRs never execute internal result scripts.

Build jobs validate and publish the current Run's build archive to its run-scoped NAS cache before uploading the same files as a GitHub artifact backup. Test jobs probe that cache first and download the artifact only when the cache is missing or invalid. Successful runs remove transient build caches after testing and backup; failed or cancelled runs retain the formal cache for failed-job retries. Cache checks still enforce source revisions, build profile and archive SHA-256, while NAS permission or mount errors fail explicitly instead of being treated as cache misses.

### CSV Cases and commands

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
python3 -S -B tests/xtest/run_cases.py   --input tests/xtest/minimal-csv --profile standard --nodes 1 --validate-only
python3 -S -B tests/xtest/run_cases.py   --input tests/xtest/minimal-csv --profile notsm --nodes 1 --validate-only
```

Result checks are `exit_code`, `validation_summary` and `performance`. Performance requires command success, a validation summary and a valid table, and emits `metrics.csv`; there is no baseline or sample-stability gate. Raw logs remain unmodified in the private results area.

### Build artifacts and NAS cache

standard and notsm are each built once and must be uploaded as GitHub Actions artifacts with source revisions, profile and archive SHA-256. A per-profile staging job reuses a valid run-scoped NAS cache or downloads the artifact once and validates it before publication. The standard single-node and two-node jobs therefore share one artifact download, while each single-node job extracts into its own local workspace.

The two-node master extracts to a NAS directory visible to both workers and removes that directory through an EXIT trap. After result reconciliation, the final cleanup removes only this run's `staging/`, `cache/` and `shared-standard/` directories. Results and raw logs are preserved. Cache hits and cleanup never rely on file existence alone: source revisions, profile, SHA-256 and run-scoped paths are verified. Build artifacts have an explicit retention period to control multi-GB storage use.

Retry publication takes a per-cache interprocess lock outside the directory being
replaced, with a five-minute wait limit. The NAS must honor shared advisory locks.
The cache root, profile parent and cache directories are searchable, and artifact
files are readable by different worker identities before publication (`0755`
directories, `0644` files), including the retry path. The writable run root and
results permissions are not narrowed by this repair.

## Commit message previews

Multi-commit PRs targeting `main`, including forks, receive a squash body preview
when `PCCL Review Gate` passes for the current HEAD. Paste it into the **Squash and merge**
description and check the title separately.

Previews update or disappear as the PR and approvals change; they do not mean CI or
other merge checks have passed.

| File | Purpose |
| --- | --- |
| [commit-message.yml](workflows/commit-message.yml) | Run `PCCL Commit Message Preview` independently of CI. |
| [commit_message_squash.py](scripts/commit_message_squash.py) | Generate and publish the message and remove older previews. |
| [Squash message Skill](../.agents/skills/pccl-squash-message/SKILL.md) | Define how to summarize changes and preserve references and signing trailers. |
| [install_copilot.sh](scripts/install_copilot.sh) | Install Copilot CLI for message generation and module classification. |
| [copilot_cli.py](scripts/copilot_cli.py) | Extract the final Copilot response. |

## Reviewer assignment

When a PR is ready for review or updated, PCCL selects primary modules by ownership of
the main changes, rotates Reviewers (skipping the author), and invites their Maintainers.
The prompt requests one primary module for up to 500 added/deleted lines, or at most two
above 500; scripts do not enforce this limit. Other affected modules appear in the PR
comment for optional manual invitations, without required module approvals.

PCCL keeps eligible assignments and invites each person once across modules.

| File | Purpose |
| --- | --- |
| [reviewer-routing.yml](workflows/reviewer-routing.yml) | Coordinate module classification and review requests. |
| [Classification Skill](skills/pccl-review-assignment/SKILL.md) | Define how to identify the affected modules. |
| [modules.json](skills/pccl-review-assignment/modules.json) | Define module scopes and the Reviewer and Maintainer roster. |
| [review_automation.py](scripts/review_automation.py) | Classify changes, assign Reviewers and notify reviewers and Maintainers. |
| [reviewer_state.py](scripts/reviewer_state.py) | Preserve assignments and maintain the per-module rotation. |
| [reviewer_api.py](scripts/reviewer_api.py) | Access GitHub PRs, reviews and accounts. |

## Module approval

`PCCL Review Gate` requires valid approval of the current HEAD from each primary module's
Maintainer and any non-author Reviewer in its roster. If the default Reviewer is unavailable,
invite another from the same module. One approval counts for all roles and modules it covers.

- Maintainer authors are exempt for their own modules; Reviewer approval is waived only if no non-author candidate exists.
- The Lead Maintainer can approve both roles for all modules on another author's PR. Other CI and merge checks still apply.

The gate refreshes after invitation runs and when reviews change.

| File | Purpose |
| --- | --- |
| [maintainer-approval.yml](workflows/maintainer-approval.yml) | Run the module approval check and publish `PCCL Review Gate`. |
| [review-event.yml](workflows/review-event.yml) | Trigger an approval refresh when a review changes. |
| [maintainer_approval.py](scripts/maintainer_approval.py) | Evaluate same-module Reviewer and Maintainer approvals for the current commit. |
