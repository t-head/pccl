---
name: pccl-contribution
description: Prepare PCCL issues, commits, and PRs, handle review feedback, and perform authorized updates or merges. Use for contribution work, not general Git questions or unrelated coding.
---

# PCCL Contribution Workflow

Use the section needed for the current task. Follow the requirements and recommendations in [CONTRIBUTING.md](../../../CONTRIBUTING.md) without adding stricter rules.

## Authorization

- Prepare and validate requested changes before seeking missing approval. Present the diff, rationale, and results.
- Issue creation/edits, commit/amend, push, PR edits, replies, and thread resolution need authorization for the changes and operations. Reuse existing approval within that scope.
- Review comments do not authorize changes. New fixes or scope changes need approval unless existing authorization covers them.

## Issue and PR Assignment

Assign new issues and PRs to their authors when you have permission.
Use a different assignee only when explicitly instructed.
If you lack permission, leave the item unassigned and report that a maintainer needs to assign it.

## Issue

Choose the template by the issue's purpose; do not default to `Task`.

| Template | When to use | Issue type |
|---|---|---|
| `BUG.yaml` | Incorrect behavior, failures, or performance regressions | `Bug` |
| `FEATURE.yaml` | A new capability or improvement | `Feature` |
| `TASK.yaml` | Scoped implementation work, testing, documentation, refactoring, or maintenance | `Task` |
| `QUESTION.yaml` | Usage, configuration, or interface questions | `Task` |

1. Read the matching [issue template](../../../.github/ISSUE_TEMPLATE/) and follow its field headings, prompts, and required fields. Keep optional fields optional; ask for missing required information instead of inventing it.
2. When using CLI/API tools, format the body from the template and explicitly set the issue type to the template's top-level `type` value. Copying the body alone does not apply template metadata.
3. After authorized creation, read back the title, body, and type. Correct omissions within the approved scope or report permission limits.

CLI example:

```bash
gh issue create --repo t-head/pccl \
  --title "<title>" --body-file "<body-file>" \
  --type "<selected-type>"
```

## Commit

1. Create a working branch from the confirmed target branch, or reuse this task's existing branch. Refer to the [branch naming recommendations](../../../CONTRIBUTING.md#getting-started) when naming a new branch.
2. Review the full diff, including staged changes. Stage only approved files or hunks; exclude unrelated changes and private notes.
3. Follow [Commit Messages](../../../CONTRIBUTING.md#commit-messages) for format and issue references. Ask for missing issue context; do not invent or create issues without authorization.
4. Verify the author and matching DCO identity. Preserve authorized sign-offs, co-author trailers, compatibility information, and issue intent.
5. Validate before committing. After creating or amending, check that the diff and message describe the complete change.

When changing only the message, keep committed content unchanged. Do not rewrite existing commits solely for capitalization or spacing.

## Push and PR

1. Confirm the repository, working branch, and target; fetch the actual target.
2. Allow follow-up commits during development and review without a count limit. Preserve authorship and authorized sign-offs. Review the full target-to-head diff.
3. Rebase when needed to resolve conflicts or integrate relevant target changes. Do not rewrite reviewed commits merely because the target advanced. Require a clean worktree before rebasing; abort and report unresolved conflicts.
4. Verify the remote branch tip. For rewrites, use `--force-with-lease=<ref>:<expected-sha>`; stop on unexpected changes. Never push directly to protected targets.
5. Use the PR template and English text covering the final change, actual validation, limitations, and a relevant open issue. Apply authorized edits and read them back.
6. Check CI and reviews for the current HEAD. Report pending or failed checks separately; an AI gate passing does not prove a completed review.

## Review Feedback

- Evaluate each comment against the code, rules, and discussion: actionable, inapplicable/incorrect, or needing evidence.
- Fix and validate actionable findings within the authorized scope. Explain disagreements with evidence.
- Resolve fixed threads only after publishing and verifying the fix. Get a human decision before resolving suggestions not adopted.
- During authorized iteration, reassess new feedback for the current HEAD. Stop when no actionable findings remain; report pending reviews and unresolved decisions.

## Merge

Use squash and merge. For a PR with two or more commits, prepare an AI draft with **Prepare squash commit** and review its complete message in the PR comment. For a single-commit PR, use its original message and preserve signing metadata; no AI draft is needed.

- Check the message against the complete PR diff, including compatibility impact and limitations. The bot comment is a suggestion, not an approval or a merge command.
- The preview carries its PR version and generation record. `/merge` reads the displayed message directly and verifies it against the current PR. Changed inputs require a fresh draft.
- Copilot reads the complete PR diff and message rules at the target SHA using read-only GitHub tools, then drafts the title and body. Code checks the PR version, preserves and deduplicates original issue references and signing identities, and applies the existing message validator. Human review confirms the summary's accuracy.
- Same-repository PR updates prepare the message automatically. For fork PRs or a fresh draft, run the workflow from `main` after it is merged. Large or incomplete diffs require manual preparation.
- Each new draft is posted at the end of the conversation and replaces old previews. Retrying the same publication does not post it twice. Single-commit PRs remove the preview; Actions logs remain available.
- After the merge workflow is on `main`, a human with repository Write, Maintain or Admin access can comment exactly `/merge` to confirm the displayed multi-commit message and request squash merging. Actions handles the command; no separate bot account or @mention is needed. PR or draft changes require a fresh message and command.
- The command checks current reviews and required checks without waiting or bypassing branch rules. On failure, complete the missing requirements and send `/merge` again. Single-commit PRs use their original message through the web button or the CLI command below. The web Squash and merge button remains available for every PR.

1. Present the final repository/PR, target, HEAD SHA, method, title/message, CI and approvals, unresolved discussions, and validation limits. For squash, verify final authorship/sign-offs and preserve authorized trailers and issue intent.
2. Obtain explicit human approval for this merge. Push permission, passing CI, or a request to finish the PR is insufficient. Reuse approval only while HEAD, target, method, and message remain unchanged.
3. Recheck immediately before merging; unmet checks or approvals block it. Do not bypass protection or enable auto-merge without explicit authorization.
4. Use the exact confirmed title and body with `gh pr merge --squash --match-head-commit <sha> --subject <title> --body-file <body-file>`. Do not append another sign-off or substitute GitHub’s concatenated default. Verify and report the resulting commit and merge method.

## References

- [CONTRIBUTING.md](../../../CONTRIBUTING.md): contribution, commit message, issue, and DCO policy.
- [Review rules](../../review_rules.yaml): code-review checklist.
- [PR template](../../../.github/pull_request_template.md): PR description structure.
- [Issue templates](../../../.github/ISSUE_TEMPLATE/): Bug, Feature, Question, and Task forms.
- [CI workflow](../../../.github/workflows/pr-test-ppu.yml): pre-checks, approval gates, and build/test jobs.
- [README.md](../../../README.md): build and validation.
