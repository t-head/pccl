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
| [pccl-daily-full.yml](workflows/pccl-daily-full.yml) | Experimental dual-platform M890P/810E daily smoke/full execution with automatic same-environment groups, artifact-backed build and Case-group distribution, and exact per-group reconciliation. Full execution selects 1/2/4/8-node groups and is not a PR gate. |
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

## From review assignment to merge

When an open PR targeting `main` or `release/*` is ready for review, the bot's
**Review modules** comment lists the primary module, a secondary module if present,
other affected modules, and a link to the [full roster](skills/pccl-review-assignment/modules.json).
It mentions the assigned invitees as `@login, please review as **Reviewer**` or
`**Maintainer**` for their module, omitting author exemptions and manually cancelled invitations.
PCCL invites the primary module's Maintainer unless they are the PR author. If the author
is in that module's Reviewer list, the Reviewer role is exempt and no replacement Reviewer
is selected, even when other candidates exist. Otherwise, PCCL invites a rotating Reviewer.
For the secondary module, PCCL invites a rotating non-author Reviewer unless the author is
its sole Reviewer. If the secondary module has other candidates, author membership does
not waive the invitation. Automatic invitations work for forks too; external authors do
not need permission to invite people in the upstream repository.

### If the assignment needs adjusting

Use GitHub's **Reviewers** sidebar to request or remove a person's review. Only directly
invited people listed as the Lead, a Maintainer or a Reviewer in
[Module members](skills/pccl-review-assignment/modules.json) become required.
People outside that list can review, but their invitations and approvals do not affect
this gate. To make an unlisted reviewer required, add them to the module member list first.
Removing an invitation cancels that person's requirement. When replacing someone, remove
the original invitation as well as inviting the replacement; another same-module approval
does not automatically substitute. GitHub controls who can manage invitations, including
repository writers and organization members with triage access. Team requests and bots
are not individual approval requirements; invite the people whose approvals you need.

Submitting a review does not remove the requirement, even when GitHub no longer lists
a pending request. The invitation history retains it until an explicit removal.
If GitHub only offers **Re-request review** for someone who has already reviewed,
request again before removing that request. Confirm the cancellation in Gate **Details**;
cancelling an invitation does not dismiss an existing **Request changes** review or
bypass GitHub's own merge rules.

Manual invitations and cancellations carry forward across pushes and routing retries.
The bot does not restore a manually cancelled invitation; request that person again
in GitHub to restore the requirement. Obsolete automatic assignments are retired when
classification changes, without withdrawing manually managed invitations.

### What turns the review gate green?

At least one non-author module member must remain invited, and every required reviewer
must have a valid **Approve** for the current HEAD.
Comments are not approvals, and dismissed, changes-requested or old-commit verdicts do
not satisfy the gate. Gate **Details** identifies the people whose approvals are missing.
At least one module must have both a Reviewer and its Maintainer in the required set.
An author's own role in a classified affected module can satisfy that coverage without
an invitation or approval vote. Primary-module Reviewer membership qualifies even when
other candidates exist; secondary-module Reviewer membership qualifies only when the
author is its sole Reviewer. Membership in an unrelated module does not create an author
exemption for that module. One person's approval covers all roster roles they hold, but
does not waive another person's invitation.

The Lead's valid current-HEAD approval on another author's PR covers all invitees and
bypasses the complete-module requirement. Removing every invited module member keeps the gate
pending even with Lead approval. Removing one person is allowed, but the gate stays
pending if the remaining people and author exemptions no longer cover a complete module.

After an invitation change, push or review change, allow the gate to refresh. If an
invitation update was interrupted, its **Details** link identifies the run to retry.
For a merge to `main`, check both `PCCL Review Gate` and `PCCL Tests Gate`, along with
GitHub's other required checks. CI and hardware tests remain independent of review approval.

### Preparing the squash message

For a multi-commit PR targeting `main`, a successful review gate lets the preview
job generate a **Suggested squash commit body**. Match the commit printed in the
comment to the HEAD you plan to merge. Check its summary, issue references and
sign-offs, then paste the body into **Squash and merge** and review the title.
PR updates trigger cleanup without waiting for the review gate or generating a new body.
Cleanup reads the latest PR state and removes previews for older commits, or all previews
when the PR is closed, draft, single-commit or no longer targets `main`. Editing only the title
or description preserves the current-commit preview while the PR remains eligible.
Cleanup and generation share a per-PR queue; allow the workflow to finish and still check
the printed commit before copying.
Only a current-commit preview stays visible while the gate is pending or generation is retried.
A failed or cancelled gate run cannot generate a preview from an earlier green status,
and a red gate removes the remaining preview. Successful publication replaces any
current-commit preview only after the new comment is posted.
Source commit messages and the generated body are checked for known AI tool signatures
before publication. The generated body must also preserve every source `Signed-off-by`
and `Co-authored-by` trailer in first-occurrence order, removing exact duplicates.
Missing, changed or added trailers reject publication. A rejection keeps any current-commit
preview and asks for correction; it does not silently remove or invent human co-authors
or DCO sign-offs.

### Maintaining the workflow

[Routing](workflows/review-routing.yml) classifies the patch and passes the result to
the [invitation workflow](workflows/review-module.yml). Its completion refreshes the
[gate](workflows/review-gate.yml), including bot requests that do not trigger another
Actions run. Native [invitation events](workflows/review-command.yml) and
[review events](workflows/review-event.yml) also refresh the gate without rerunning
classification or sending invitations. Gate completion triggers the
[squash preview](workflows/review-squash-preview.yml). PR pushes, edits, draft conversions
and closures also trigger that workflow directly for cleanup only, without generating a body.

GitHub's invitation timeline is the durable record of required people and explicit
cancellations, not the pending-request list. The state branch stores only classification
and default rotation choices. The [classification prompt](skills/pccl-review-assignment/SKILL.md)
selects at most two modules in priority order, without a line-count threshold. Keep each
module's Maintainer separate from its Reviewer list.

When deploying this policy to `main`, let in-flight updates finish, then run
`gh workflow run review-routing.yml --ref main -f pr_number=NUMBER` for each open,
ready PR targeting `main` or `release/*`. Existing invitation history becomes authoritative.
After successfully publishing the concise module summary, routing removes its old
per-person assignment notices.
Check the invited people and wait for the new gate verdict before merging. The same
command refreshes an invalid saved classification.
