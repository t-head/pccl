---
name: pccl-commit-message
description: Generate or update commit messages for the PCCL project following team conventions (pccl header format, aone tracking, test cases, reviewers). Use when the user mentions commit, commit message, amend, 提交, 提交信息, gerrit, push to gerrit, or asks for help writing or updating a commit message.
---

# PCCL Commit Message Generator

Generate or update Git commit messages for the PCCL Gerrit workflow.

## Trigger

Activate when the user:
- Asks to create or update a commit message
- Mentions "commit", "提交", "commit message", "gerrit"
- Explicitly says "create commit" or "update commit"

## Determine Mode: Create vs Amend

- **Create**: There are staged or unstaged changes that need a new commit.
- **Amend**: The most recent commit message needs to be revised (typically before pushing or after amend).

Mode is determined interactively in Workflow Step 2. If user's intent is already clear (e.g., "帮我更新 commit message"), skip Step 2.

## Commit Message Format

```
pccl: <concise summary of the patch in imperative mood>

<numbered list of key changes, each on its own line>

<to|fix> #<aone_id>: <work_item_description>
Test Case: <test_case_description>
Reviewed By: <reviewer_short_names>
```

### Rules

1. **Header**: Always `pccl: <summary>`. Summary is in English, imperative mood, lowercase start, no trailing period. Max 72 chars total.
2. **Body**: Numbered list of key changes (English). Each item starts with a verb in imperative mood. Separated from header by one blank line.
3. **Aone Tracking**: Use `Fix #<id>: <desc>` for bug fixes, `to #<id>: <desc>` for features/tasks. ID is the Aone work item number.
4. **Test Case**: Describe which test cases cover this change. If user doesn't specify, analyze the diff and suggest appropriate test cases.
5. **Reviewed By**: Short names of reviewers, comma-separated.
6. **Change-Id**: Do NOT generate or modify `Change-Id` lines — Gerrit's commit-msg hook handles this automatically.

### Example

```
pccl: fix race condition in uint64 ld/st on 64-bit ga address

1. replace all normal volatile uint64 load/store with b32x2 intrinsic in kernel code
2. add volatile decorator in net.cc for head/tail(uint64) load/store
3. change ncclSteps type from int to uint64_t in prims_simple.h

fix #53961472: glm/gpt training hang.
Test Case: LLM training.
Reviewed By: lss.
```

## Workflow

### Step 1: Check for Unstaged Changes

Run `git status` to check for unstaged changes:
- If there are unstaged files, use `AskUserQuestion` to ask whether to include them in the commit
- If no unstaged changes exist, proceed to Step 2

### Step 2: Determine Commit Mode

Use `AskUserQuestion` to ask the user whether they want to **create** a new commit or **amend** the existing one:
- **Create**: New commit with staged/unstaged changes
- **Amend**: Modify the most recent commit message (preserve Change-Id)

If user's intent is already clear (e.g., "帮我更新 commit message"), skip this step.

### Step 3: Gather Diff

- **Create mode**: Run `git diff --cached` (staged). If empty, run `git diff` (unstaged).
- **Amend mode**: Run `git log -1 --format="%B"` to get existing message and `git diff HEAD~1` to see changes.

### Step 4: Extract Existing Metadata (Amend Mode Only)

From the existing commit message, extract and preserve:
- Aone tracking line (`to/fix #<id>: ...`)
- Test Case line
- Reviewed By line
- Change-Id line (preserve as-is, do not modify)

### Step 5: Collect Missing Information via AskUserQuestion

Use `AskUserQuestion` tool to interactively gather required information:

**For Create Mode:**
1. **Files to include**: If unstaged files exist, ask which files to include
2. **Aone ID & description**: Ask for Aone work item ID and description
3. **Reviewed By**: Ask for reviewer short names
4. **Test Case**: Auto-analyze the diff to suggest test cases, then confirm with user

**For Amend Mode:**
1. Run `git diff HEAD~1` to get all changes since last commit
2. **Automatically generate new commit message** based on the full diff (do NOT ask user which fields to update)
3. **Present to user** for review with before/after comparison
4. **If user requests changes**: use `AskUserQuestion` to ask which specific field needs updating, then modify only that field

### Step 6: Generate Commit Message

Based on the diff analysis and collected metadata:
1. Write a concise header summarizing the change
2. List key changes as a numbered body
3. Append aone tracking, test case, and reviewer lines
4. All commit messages should be in English.

### Step 7: User Review Loop

Present the generated commit message to the user. If in amend mode, show a before/after comparison:

```
=== BEFORE ===
<old commit message>

=== AFTER ===
<new commit message>
``
Ask the user to review using `AskUserQuestion`:

- If user approves: proceed to Step 8
- If user requests changes: use `AskUserQuestion` to ask which fields need updating (header, body, aone, test case, reviewers), then modify accordingly and present again. **Repeat until the user approves.**

**Repeat until the user approves.**

### Step 8: Apply Commit Message

- **Create mode**: Run `git commit -m "<message>"` (ensure changes are staged first)
- **Amend mode**: Run `git commit --amend -m "<message>"`

Important: Preserve any existing `Change-Id` line at the end of the message when amending.

### Step 9: Code Review

After the commit is applied, perform an automated code review **before** pushing. Read the review rules from `PCCL_AI_CODE_REVIEW_RULES.md` in the project root, then review the diff (`git diff HEAD~1`).

**Review process:**

1. Read `PCCL_AI_CODE_REVIEW_RULES.md` for the full checklist and rules.
2. Run `git diff HEAD~1` to get the committed changes.
3. Apply the Section 16 "AI Review Checklist" against the diff:
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

Note: For changes in the `algo/` directory, apply relaxed rules per Section 15 of the review document.

### Step 10: Push Decision

After successful commit/amend and code review passes, ask the user:

"是否需要 push 到 Gerrit？默认命令: `git push origin HEAD:refs/for/master`"

- If yes: execute `git push origin HEAD:refs/for/master`
- If no: do nothing, inform the user they can push manually later

## Important Notes

- Always use `cat <<'EOF'` heredoc syntax when passing multi-line commit messages to git commands.
- Never modify or remove existing `Change-Id` lines — they are critical for Gerrit patch set tracking.
- If the user's working directory has both staged and unstaged changes, warn them and ask which files to include.
- The commit message body should be in English to match team conventions.
