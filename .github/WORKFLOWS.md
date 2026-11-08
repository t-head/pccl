# GitHub workflows

CI and module review run independently. Pull requests targeting `main` must pass both
`PCCL Tests Gate` and `PCCL Review Gate`.

## CI checks and PPU tests

CI scans for secrets and checks commit messages and DCO sign-offs. After AI review passes
and a reviewer with repository write access approves the PR, CI builds PCCL and runs the
PPU tests. `PCCL Tests Gate` reports the results of the required jobs.

| File | Purpose |
| --- | --- |
| [pr-test-ppu.yml](workflows/pr-test-ppu.yml) | Run only the PR Secret Scan and Commit Message / DCO prechecks; its legacy PPU jobs are retired. |
| [source-sanity.yml](workflows/source-sanity.yml) | The sole PPU test pipeline and publisher of `PCCL Tests Gate`: AI and human approval precede self-hosted builds, followed by two builds, one single-node job per profile, and one two-node job reusing the standard build. |
| [source-cluster-debug.yml](workflows/source-cluster-debug.yml) | Manual troubleshooting tool for the two-node MPI case; it rebuilds from source instead of reusing a build. |
| [workflows/scripts/](workflows/scripts/) | Build and test PCCL with the external test suite. |
| [casecmd.list](workflows/scripts/casecmd.list) | Define the PPU test matrix. |

`source-sanity.yml` runs on same-repository pull requests, pushes to `main`, and manual dispatch. It
never uses repository secrets, and it keeps the executable source of truth in
`pccl_test_suites`: the checked-in catalog records the reviewed revision, the exporter
hash, and the selection hash, so a change to the cases requires a new export. The reviewed
two-node selection currently contains only a standard execution; the notsm build covers
single-node sanity and is not a second two-node variant in this workflow. Each build also
publishes a build record containing the actual source revisions, profile and archive SHA-256.
The executor verifies that record against the archive and retains it in shard evidence; both
result gates check it again without requiring the original build host or archive to be present.
Each profile must resolve to one archive identity across all shards, and the two-node
standard execution must use the same archive as the single-node standard shards.
Performance command success and valid data are required, but sample stability is observational only
until a baseline exists. Raw `command.log` evidence remains on the private result store for
validation; `command-public.log` is the separately redacted diagnostic copy.

## Reading Cases and commands

The source catalog keeps the normalized execution model: one source Case can have several
execution profiles. Use the Case-first reader instead of joining `cases` and `executions`
manually:

```bash
python3 tests/xtest/catalog.py describe [--case CASE_ID_OR_NAME] [--json]
```

The three command representations have deliberately different roles:

| Representation | Meaning |
| --- | --- |
| `source_command` | Original source command, retained only for provenance and conversion checks. It is never an execution entry. |
| `executions[].spec.argv` | Execution template consumed by the GitHub runner. Its `{python}`, `{test_root}`, `{binary_dir}`, `{result_dir}` and `{mpi_options}` placeholders require runtime expansion. |
| `command-reproduction.txt` | The expanded, redacted command and relevant environment for one actual attempt. It is execution evidence, not a catalog input. |

The result gate also writes `case-summary.json`. It groups all execution results by source
Case and links each profile's `result.json` and, when present, `command-reproduction.txt`.
The uploaded `single-node-source-sanity-results` artifact includes those linked files under
the same relative paths. Result JSON is limited to the public result projection, and the
reproduction text is redacted again during publication instead of copying raw diagnostics.
The stable Case and execution IDs are not affected by the field-name cleanup.

## Commit message previews

Multi-commit PRs targeting `main` receive a proposed squash commit message.
Publishing a fresh preview places the message at the end of the conversation and removes
older previews. The preview supplies the message for GitHub's **Squash and merge** dialog.

For fork PRs, a Maintainer runs `PCCL Commit Message Preview` from `main` with `pr_number`.

| File | Purpose |
| --- | --- |
| [commit-message.yml](workflows/commit-message.yml) | Run `PCCL Commit Message Preview` independently of CI. |
| [commit_message_squash.py](scripts/commit_message_squash.py) | Generate and publish the message and remove older previews. |
| [Squash message Skill](../.agents/skills/pccl-squash-message/SKILL.md) | Define how to summarize changes and preserve references and signing trailers. |
| [install_copilot.sh](scripts/install_copilot.sh) | Install Copilot CLI for message generation and module classification. |
| [copilot_cli.py](scripts/copilot_cli.py) | Extract the final Copilot response. |

## Reviewer assignment

When a PR is ready for review or receives new changes, PCCL identifies the affected modules,
assigns Reviewers by rotation within each module and invites the corresponding Maintainers.
It keeps existing assignments and invites each person once across modules.

| File | Purpose |
| --- | --- |
| [reviewer-routing.yml](workflows/reviewer-routing.yml) | Coordinate module classification and review requests. |
| [Classification Skill](skills/pccl-review-assignment/SKILL.md) | Define how to identify the affected modules. |
| [modules.json](skills/pccl-review-assignment/modules.json) | Define module scopes and the Reviewer and Maintainer roster. |
| [review_automation.py](scripts/review_automation.py) | Classify changes, assign Reviewers and notify reviewers and Maintainers. |
| [reviewer_state.py](scripts/reviewer_state.py) | Preserve assignments and maintain the per-module rotation. |
| [reviewer_api.py](scripts/reviewer_api.py) | Access GitHub PRs, reviews and accounts. |

## Module approval

`PCCL Review Gate` checks that the Maintainers of the affected modules have approved the
current commit. Authors are exempt from this check for modules they maintain. The Lead
Maintainer can approve all modules on another author's PR.

The gate checks existing approvals after the invitation run finishes, even if an invitation
failed. It also refreshes whenever a review is submitted, edited or dismissed.

| File | Purpose |
| --- | --- |
| [maintainer-approval.yml](workflows/maintainer-approval.yml) | Run the module approval check and publish `PCCL Review Gate`. |
| [review-event.yml](workflows/review-event.yml) | Trigger an approval refresh when a review changes. |
| [maintainer_approval.py](scripts/maintainer_approval.py) | Evaluate module Maintainer approvals for the current commit. |
