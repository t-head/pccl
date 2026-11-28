# Contribution and review workflows

Maintained by the `workflow` module. See the [workflow overview](../WORKFLOWS.md) for required PR gates.

## From review assignment to merge

When an open PR targeting `main` or `release/*` is ready for review, the bot's
**Review modules** comment lists the primary module, a secondary module if present,
other affected modules, and a link to the [full roster](../skills/pccl-review-assignment/modules.json).
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
[Module members](../skills/pccl-review-assignment/modules.json) become required.
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
in GitHub to restore the requirement. After reclassification, the bot withdraws obsolete
pending automatic requests, without withdrawing manually managed invitations. It does
not re-request obsolete automatic invitees just to withdraw them. Until explicitly
cancelled, their approval remains required; use the cancellation steps above when it
is no longer needed.
After a push, the bot re-requests current default or manually invited members who have
reviewed only older commits and have no pending request. It does not repeat pending
requests or re-request someone who has already reviewed the current HEAD.
These reminders leave manually managed invitations under manual control.

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

These requirements also apply to the Lead and to Lead-authored PRs; ordinary module
roles and author exemptions still count. Exceptional merges use GitHub's explicit
**Bypass rules and merge** action, if the merging user has the required bypass permission.
Being listed as Lead does not grant that permission, and bypassing a rule does not turn
this gate green.

After an invitation change, push or review change, allow the gate to refresh. If an
invitation update was interrupted, its **Details** link identifies the run to retry.
For a merge to `main`, check both `PCCL Review Gate` and `PCCL Tests Gate`, along with
GitHub's other required checks. CI and hardware tests remain independent of review approval.

### Preparing the squash message

For a multi-commit PR targeting `main`, a successful review gate lets the preview
job automatically generate a **Suggested squash commit body**. Match the commit printed
in the comment to the HEAD you plan to merge. Check its summary, issue references and
sign-offs, then paste the body into **Squash and merge** and review the title.
PR updates trigger cleanup without waiting for the review gate or generating a new body.
Cleanup reads the latest PR state and removes previews for older commits, or all previews
when the PR is closed, draft, single-commit or no longer targets `main`. Editing only the title
or description preserves the current-commit preview while the PR remains eligible.
Cleanup and generation share a per-PR queue; allow the workflow to finish and still check
the printed commit before copying.
On automatic refreshes, only a current-commit preview stays visible while the gate is pending.
A failed or cancelled gate run cannot generate a preview from an earlier green status,
and a red gate removes the remaining preview.

To generate a preview without waiting for the review gate, open **Actions → PCCL Commit
Message Preview → Run workflow**, select `main` and enter the PR number in `pr_number`.
GitHub controls who can run the workflow. Manual generation requires an open, non-draft,
multi-commit PR targeting `main`, but does not check the review gate. It neither approves
the PR nor changes merge permissions; later automatic refreshes still apply the cleanup
rules above.

Both generation paths check source commit messages and the generated body for known AI tool
signatures before publication. The generated body must also preserve every source
`Signed-off-by` and `Co-authored-by` trailer in first-occurrence order, removing exact
duplicates. Missing, changed or added trailers reject publication. A rejection keeps any
current-commit preview and asks for correction; it does not silently remove or invent
human co-authors or DCO sign-offs. Successful publication replaces any current-commit
preview only after the new comment is posted.

### Maintaining the workflow

[Routing](../workflows/review-routing.yml) classifies the patch and passes the result to
the [invitation workflow](../workflows/review-module.yml). Its completion refreshes the
[gate](../workflows/review-gate.yml), including bot requests that do not trigger another
Actions run. Native [invitation events](../workflows/review-command.yml) and
[review events](../workflows/review-event.yml) also refresh the gate without rerunning
classification or sending invitations. Gate completion triggers the
[squash preview](../workflows/review-squash-preview.yml). PR pushes, edits, draft conversions
and closures also trigger that workflow directly for cleanup only, without generating a body.

GitHub's invitation timeline is the durable record of required people and explicit
cancellations, not the pending-request list. The state branch stores only classification
and default rotation choices. The [classification prompt](../skills/pccl-review-assignment/SKILL.md)
selects at most two modules in priority order, without a line-count threshold. Keep each
module's Maintainer separate from its Reviewer list.

When deploying this policy to `main`, let in-flight updates finish, then run
`gh workflow run review-routing.yml --ref main -f pr_number=NUMBER` for each open,
ready PR targeting `main` or `release/*`. Existing invitation history becomes authoritative.
After successfully publishing the concise module summary, routing removes its old
per-person assignment notices.
Check the invited people and wait for the new gate verdict before merging. The same
command refreshes an invalid saved classification.
