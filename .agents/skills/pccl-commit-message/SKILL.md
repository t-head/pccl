---
name: pccl-commit-message
description: Generate or update commit messages for the PCCL project following team conventions (Conventional Commits, optional GitHub issue references). Use when the user mentions commit, commit message, amend, 提交, 提交信息, GitHub pull requests, or asks for help writing or updating a commit message.
---

# PCCL Commit Message Generator

Generate or update Git commit messages for the PCCL GitHub workflow.

## Trigger

Activate when the user:
- Asks to create or update a commit message
- Mentions "commit", "提交", "commit message"
- Explicitly says "create commit" or "update commit"

## Determine Mode: Create vs Amend

- **Create**: There are staged or unstaged changes that need a new commit.
- **Amend**: The most recent commit message needs to be revised (typically before pushing or after amend).

Mode is determined interactively in Workflow Step 2. If user's intent is already clear (e.g., "帮我更新 commit message"), skip Step 2.

## Commit Message Format

```
<type>(<scope>): <subject>

<numbered list of key changes, each on its own line>

[optional: Closes #<issue_number> or Refs: #<issue_number>]
```

### Rules

1. **Header**: Use `<type>(<scope>): <subject>`; scope is optional. Types: `feat` (features), `fix` (bug fixes), `refactor` (restructuring without behavior changes), `docs` (documentation), `test` (tests), `chore` (build, dependencies, configuration), and `perf` (performance). Write the subject in English and imperative mood, starting lowercase with no trailing period; keep the title within 72 characters where possible. Describe the specific change; avoid standalone `update`, `fix bug`, or `wip` titles. PR titles follow the same format.
2. **Body**: Numbered list of key changes (English), explaining what changed and why. Each item starts with a verb in imperative mood. Separated from header by one blank line.
3. **GitHub Issues (optional)**: Reference a real, relevant GitHub issue with `Refs: #<number>`. Use `Closes #<number>` or `Fixes #<number>` only when the patch resolves the issue and automatic closure is intended; use `owner/repo#<number>` for another repository. No issue is required; do not invent one or ask for one just to fill the template.

### Example

```
fix(pccl): prevent race conditions in uint64 ld/st on 64-bit ga address

1. replace all normal volatile uint64 load/store with b32x2 intrinsic in kernel code
2. add volatile decorator in net.cc for head/tail(uint64) load/store
3. change ncclSteps type from int to uint64_t in prims_simple.h

Closes #123: glm/gpt training hang
```

## Workflow

### Step 1: Check for Unstaged Changes

Run `git status` to check for unstaged changes:
- If there are unstaged files, use `AskUserQuestion` to ask whether to include them in the commit
- If no unstaged changes exist, proceed to Step 2

### Step 2: Determine Commit Mode

Use `AskUserQuestion` to ask the user whether they want to **create** a new commit or **amend** the existing one:
- **Create**: New commit with staged/unstaged changes
- **Amend**: Modify the most recent commit message

If user's intent is already clear (e.g., "帮我更新 commit message"), skip this step.

### Step 3: Gather Diff

- **Create mode**: Run `git diff --cached` (staged). If empty, run `git diff` (unstaged).
- **Amend mode**: Run `git log -1 --format="%B"` to get existing message and `git diff HEAD~1` to see changes.

### Step 4: Extract Existing Metadata (Amend Mode Only)

From the existing commit message, extract and preserve:
- Relevant GitHub issue references, if present
- Existing `Co-authored-by` and `Signed-off-by` lines

### Step 5: Collect Missing Information via AskUserQuestion

Use `AskUserQuestion` tool to interactively gather required information:

**For Create Mode:**
1. **Files to include**: If unstaged files exist, ask which files to include
2. **GitHub Issue (optional)**: Use a known, relevant issue reference; otherwise omit it.

**For Amend Mode:**
1. Run `git diff HEAD~1` to get all changes since last commit
2. **Automatically generate new commit message** based on the full diff (do NOT ask user which fields to update)
3. **Present to user** for review with before/after comparison
4. **If user requests changes**: use `AskUserQuestion` to ask which specific field needs updating, then modify only that field

### Step 6: Generate Commit Message

Based on the diff analysis and collected metadata:
1. Write a concise header summarizing the change
2. List key changes as a numbered body
3. Append any relevant, verified GitHub issue reference
4. All commit messages should be in English.

### Step 7: User Review Loop

Present the generated commit message to the user. If in amend mode, show a before/after comparison:

```
=== BEFORE ===
<old commit message>

=== AFTER ===
<new commit message>
```
Ask the user to review using `AskUserQuestion`:

- If user approves: proceed to Step 8
- If user requests changes: use `AskUserQuestion` to ask which fields need updating (header, body, issue reference), then modify accordingly and present again. **Repeat until the user approves.**

**Repeat until the user approves.**

### Step 8: Apply Commit Message

- **Create mode**: Run `git commit -m "<message>"` (ensure changes are staged first)
- **Amend mode**: Run `git commit --amend -m "<message>"`

Important: Preserve existing `Co-authored-by` and `Signed-off-by` lines.

### Step 9: Code Review

After the commit is applied, perform an automated code review **before** pushing. Read the review rules from `.agents/review_rules.yaml` in the project root, then review the diff (`git diff HEAD~1`).

**Review process:**

1. Read `.agents/review_rules.yaml` for the full checklist and rules.
2. Run `git diff HEAD~1` to get the committed changes.
3. Apply the relevant rules against the diff:
   - Naming & Consistency
   - Error Handling
   - Memory Safety
   - Platform Compatibility
   - Thread Safety
   - Device Code
   - API & Interface
   - Code Style
4. Present findings to the user, categorized as:
   - **Critical**: Must fix before push (hard rejects per the rules)
   - **Warning**: Recommended to fix
   - **Info**: Minor suggestions or observations
5. If critical issues are found, help the user fix them (amend the commit) and re-review.
6. If no critical issues, proceed to the push decision.

### Step 10: Push Decision

After successful commit/amend and code review passes, ask the user:

"是否需要推送到远端分支用于 GitHub PR？"

- If yes: verify the actual remote and working branch, then push to that branch and use a PR for integration. Do not push directly to protected development or release branches. When updating an authorized amended commit already published on the PR branch, verify the remote tip and use an explicit `--force-with-lease`.
- If no: do nothing, inform the user they can push manually later

## Important Notes

- Always use `cat <<'EOF'` heredoc syntax when passing multi-line commit messages to git commands.
- If the user's working directory has both staged and unstaged changes, warn them and ask which files to include.
- The commit message body should be in English to match team conventions.
