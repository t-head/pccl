---
name: pccl-contribution
description: Prepare PCCL commits and PRs, classify affected modules, route Maintainer/Backup and specialist reviews, handle feedback, and perform authorized updates or merges. Use for contribution work, not general Git questions or unrelated coding.
---

# PCCL Contribution Workflow

Use the section needed for the current task. Follow the requirements and recommendations in [CONTRIBUTING.md](../../../CONTRIBUTING.md) without adding stricter rules.

## Authorization

- Prepare and validate requested changes before seeking missing approval. Present the diff, rationale, and results.
- Commit/amend, push, PR edits, replies, and thread resolution need authorization for the changes and operations. Reuse existing approval within that scope.
- Review comments do not authorize changes. New fixes or scope changes need approval unless existing authorization covers them.

## Commit

1. Review the full diff, including staged changes. Stage only approved files or hunks; exclude unrelated changes and private notes.
2. Follow [Commit Messages](../../../CONTRIBUTING.md#commit-messages) for format and issue references. Ask for missing issue context; do not invent or create issues without authorization.
3. Verify the author and matching DCO identity. Preserve authorized sign-offs, co-author trailers, compatibility information, and issue intent.
4. Validate before committing. After creating or amending, check that the diff and message describe the complete change.

When changing only the message, keep committed content unchanged. Do not rewrite existing commits solely for capitalization or spacing.

## Push and PR

1. Confirm the repository, working branch, and target; fetch the actual target.
2. Keep one contribution commit, preserving its contents, authorship, and sign-offs. Review the full target-to-head diff.
3. If HEAD lacks the latest target tip, automatically rebase before pushing. Require a clean worktree and verified target; abort and report conflicts.
4. Verify the remote branch tip. For rewrites, use `--force-with-lease=<ref>:<expected-sha>`; stop on unexpected changes. Never push directly to protected targets.
5. Classify the complete change using **Module Review** below. Use the PR template and English text covering the final change, actual validation, limitations, and a relevant open issue. Apply authorized edits and read them back.
6. Check CI and module approvals for the current HEAD. Report pending or failed checks separately; an AI gate passing does not prove a completed review.

## Module Review

The source of truth is [owners.json](../../../.github/review/owners.json) on the fetched target branch.
Read its module and area names before classifying. Never infer broad ownership from a specialist's isolated contribution.

1. Inspect the full target-to-head diff and its behavioral impact, including renamed/deleted files, tests, shared interfaces,
   performance tuning, and compatibility. Read the target's module/area definitions and path rules to propose the scope.
   Include uncommitted changes when preparing a commit, then recheck the committed diff. Node.js and the local helper
   are optional; their absence must not block PR preparation. If this PR changes the policy, compare with the target's
   policy: the server evaluates using trusted main, so proposed ownership changes cannot authorize their own PR.
2. Use path matches as minimum coverage; add modules affected semantically even when their files did not change.
   For example, a GIN RDMA backend change covers `device` and `network`; a tuning change affecting launch lifecycle
   may also need `runtime`. Unknown paths receive `build` triage; review them for any additional affected modules.
3. Fill exactly one `PCCL-Modules:` line in the PR body with comma-separated module IDs. Fill `PCCL-Areas:` with the
   applicable specialist areas. Keep the reason for any additional modules in Changes & Impact. Labels `module/<id>`
   and `area/<id>` may supplement these lines; they never subtract path-derived coverage. Do not invent module IDs.
   EIC/SNIC work inside shared networking files needs the appropriate semantic area, not just the general network area.
   Contribution guides/templates, agent review guidance/skills, and module-review automation belong to `workflow`
   (`workflow-review`). Build, packaging, and hardware CI belong to `build`; inspect the specific path rules when both change.
4. Use the Maintainer and Backup at **main-module** scope. Request specialists only for matching areas. Do not promote a
   RedOp, GIN, or EIC reviewer to ownership of the whole kernel/network module. Unknown GitHub accounts stay unresolved.
5. Default routing requests the Maintainer; an author who is the Maintainer routes to Backup. When asked to use Backup
   because the Maintainer is unavailable, apply `backup/<module-id>` only within authorized PR-edit scope. The label
   changes the request recipient; either Maintainer or Backup can approve without it. Neither can self-approve.
6. After changes/review feedback, re-evaluate the complete scope and current HEAD. Inspect
   **PCCL Module Owner Approval** published by GitHub Actions on the actual PR HEAD; the orchestration job's success
   is not module approval. One current-HEAD approval from a Maintainer OR Backup of any affected module with repository
   write access satisfies owner approval, including cross-module PRs. The approver need not own every affected module.
   Continue routing reviews across the complete scope. Specialist-only and unrelated-module approvals do not count.
7. A green module check does not prove CI passed or grant merge permission. Report missing approvals and checks explicitly.
   If a review event has not refreshed the check, use the documented manual refresh only when authorized; never bypass it.

If Node.js 20 or later is already available, an optional read-only preview is:

```bash
node .github/review/classify.mjs <fetched-target-ref> [<pr-body-file>]
```

It reads committed changes from the target's merge base to HEAD and makes no GitHub changes. When unavailable,
use the diff and target policy directly. The server independently checks path coverage after PR creation.

Read [Module review workflow](../../../docs/workflow/module-review.md) for account/area details, setup, and limitations.

## Review Feedback

- Evaluate each comment against the code, rules, and discussion: actionable, inapplicable/incorrect, or needing evidence.
- Fix and validate actionable findings within the authorized scope. Explain disagreements with evidence.
- Resolve fixed threads only after publishing and verifying the fix. Get a human decision before resolving suggestions not adopted.
- During authorized iteration, reassess new feedback for the current HEAD. Stop when no actionable findings remain; report pending reviews and unresolved decisions.

## Merge

1. Present the final repository/PR, target, HEAD SHA, method, title/message, CI and approvals, unresolved discussions, and validation limits. For squash, verify final authorship/sign-offs and preserve authorized trailers and issue intent.
2. Obtain explicit human approval for this merge. Push permission, passing CI, or a request to finish the PR is insufficient. Reuse approval only while HEAD, target, method, and message remain unchanged.
3. Recheck immediately before merging: current HEAD, complete module scope, Maintainer/Backup approvals, required CI, and unresolved discussions. Unmet checks or approvals block it. Do not bypass protection or enable auto-merge without explicit authorization.
4. Guard the expected HEAD when supported. Verify and report the resulting commit and merge method.

## References

- [CONTRIBUTING.md](../../../CONTRIBUTING.md): contribution, commit message, issue, and DCO policy.
- [Review rules](../../review_rules.yaml): code-review checklist.
- [PR template](../../../.github/pull_request_template.md): PR description structure.
- [Issue templates](../../../.github/ISSUE_TEMPLATE/): Bug, Feature, Question, and Task forms.
- [CI workflow](../../../.github/workflows/pr-test-ppu.yml): pre-checks, approval gates, and build/test jobs.
- [Module review workflow](../../../docs/workflow/module-review.md): classification, reviewers, required approval check, and administrator setup.
- [README.md](../../../README.md): build and validation.
