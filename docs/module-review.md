# Module review routing

PR authors or their assistants select all affected modules in the PR description:

```markdown
## Affected Modules

- [ ] Env Check
- [ ] Dispatch Strategy
```

Select at least one option with `[x]`. Module names must match exactly. Checkboxes
inside HTML comments or fenced code examples do not count. These initial modules
are a pilot; extend the mapping and template before adopting this for other work.

The workflow validates selections on PR creation, description edits, new commits,
reopening, and transition from draft to ready. Env Check requests `@wensuyang`;
Dispatch Strategy requests `@honganlxh`. A PR may select both modules. Reviewers
are deduplicated, the PR author is skipped, and existing requests or submitted
approvals/change requests are not requested again. Drafts are validated but do
not send invitations until ready. Removing a selection does not withdraw an
existing review request. GitHub notifies reviewers when an invitation is sent.

The mapping is maintained in `.github/scripts/module-review.cjs`. Update the PR
template alongside it. AI assistants should determine selections from the actual
PR diff and ask when the affected module is unclear, rather than guessing to
pass validation.

`PR Modules` is a commit status on the PR head. Missing or unknown modules and
review-request API failures fail this status. To enforce it before merging, an
administrator must configure `PR Modules` as a required status check. This does
not require approval from the selected reviewers; approval policy is separate.

The workflow uses `pull_request_target` and checks out only its trusted workflow
revision. It never executes PR-head code. Repository/organization Actions policy
must allow this event. Installation must make the workflow available on the
branch GitHub uses for this event; a workflow present only in a proposed PR is
not automatically active. Test branch runs do not change production protections.
