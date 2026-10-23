# Maintaining PCCL Reviewer Automation

This document is for maintainers of the review automation, not input to the
classification model. For the contributor review process, see
[CONTRIBUTING.md](../../CONTRIBUTING.md#code-review-process).

## Components

- The [classification Skill](../skills/pccl-review-assignment/SKILL.md) identifies
  affected modules from the complete PR diff. It does not select people or approve changes.
- The [review automation script](review_automation.py) classifies changes, selects people from the
  public [module roster](../skills/pccl-review-assignment/modules.json), sends invitations
  and checks Maintainer approvals.
- The [API client](reviewer_api.py) handles GitHub requests, pagination and account lookup.
- The [state module](reviewer_state.py) reads and writes classifications and rotations
  on `workflow/review`.
- The [routing workflow](../workflows/reviewer-routing.yml) runs classification,
  invitations and approval checks. The read-only
  [review trigger](../workflows/review-event.yml) signals review changes so a trusted
  `workflow_run` job can refresh approval for fork PRs as well as internal PRs.

## Assignments and Approvals

Each module selects one Reviewer in roster order, skipping the PR author. Modules
rotate independently, starting at the first eligible account when the previous
Reviewer is absent. Each account is invited only once, even when it fills multiple roles.

PR updates retain saved Reviewer choices, including accounts removed from the candidate
list. Only newly affected modules advance their rotations. Completed or manually removed
invitations are not requested again; pending invitations and current-commit approvals
are also respected. If no eligible Reviewer is available, update the roster or arrange
review manually.

Each module's Maintainer is requested unless they are the PR author. In that case,
that module's Maintainer approval is waived without requesting the Lead; Reviewer
assignment still applies. The `ci` module exempts its configured author ID from both
roles. Other modules still apply. The optional `author_review_exemption` is a single
numeric ID that must match the module's Maintainer; this dual-role exemption does not
automatically move to a replacement Maintainer.

`PCCL Maintainer Approval` checks each affected module's Maintainer, or the Lead
Maintainer for all modules. Maintainers need not wait for Reviewers; Reviewer approval
alone does not satisfy this status. Only humans' latest verdicts approving the current
commit count. Self-reviews, withdrawn approvals and old commits are excluded; comments
do not replace a verdict. Author exemptions still apply. The Human Review Gate controls
when PPU CI starts; the Maintainer status controls module approval for merging.

## Deployment and Roster Changes

Keep the workflow, classification Skill and roster on `main`. Target PRs to `main` or
`release/**`; manual runs start from `main` with the target PR number. Only trusted
workflow code executes; fork PR contents are read as data. Sync the lightweight
`review-event.yml` notifier to maintained release branches; until then, rerun routing
from `main` after reviews.

Before rollout, restrict writes to `workflow/review` to trusted routing automation and
authorized maintainers, and protect the workflow and roster. Make
`PCCL Maintainer Approval` required for `main` and `release/**`, keeping the native
approval count at one. Existing CI, CODEOWNERS and last-push rules still apply.
The workflow reports status; it does not merge PRs or grant permissions.

Roster accounts have a numeric GitHub ID and a readable login. IDs determine identity,
author exclusions and approval eligibility; current logins are resolved for invitations
and display. Account renames preserve assignments. Update personnel through a PR, then
rerun routing for affected open PRs. Personnel-only changes do not invalidate module
classifications.

## State and Recovery

The independent `workflow/review` branch holds all review state:

- `rotation.json` maps each module to its last assigned Reviewer's ID.
- `assignments/<PR>.json` contains `reviewers` (the saved module-to-ID mapping) and
  `classification` (the current validated result, PR number, head SHA and classification
  context).

The invitation job saves classification before selecting or contacting reviewers, so
invitation failures do not discard it. New Reviewer assignments and rotation changes are
committed together; updates and retries reuse existing choices. Invitation jobs queue
serially and conflicting writes fail without overwriting state. All target branches
share these records. Keep this branch separate from `main`; do not merge or rebase it.
Repair corrupt state rather than resetting it.

Approval events read the saved classification and verify its PR, commit, diff and
classification rules. Missing or stale results require rerunning routing.

Each batch of new invitees gets a new English comment mentioning their accounts and
modules; existing comments are never edited. The comment is posted before the review
request so a comment failure can be retried. Hidden account-ID markers prevent repeat
mentions if the invitation fails afterward. Comments do not determine approvals or
rotation. After a failure, fix access and rerun routing; saved choices remain reserved.
Git history provides the audit trail.
