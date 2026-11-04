# GitHub workflows

CI and module review run independently. Pull requests targeting `main` must pass both
`PCCL Tests Gate` and `PCCL Review Gate`.

## CI checks and PPU tests

CI scans for secrets and checks commit messages and DCO sign-offs. After AI review passes
and a reviewer with repository write access approves the PR, CI builds PCCL and runs the
PPU tests. `PCCL Tests Gate` reports the results of the required jobs.

| File | Purpose |
| --- | --- |
| [pr-test-ppu.yml](workflows/pr-test-ppu.yml) | Run `PCCL CI` and publish `PCCL Tests Gate`. |
| [workflows/scripts/](workflows/scripts/) | Build and test PCCL with the external test suite. |
| [casecmd.list](workflows/scripts/casecmd.list) | Define the PPU test matrix. |

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
