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
| [m890p-daily-full.yml](workflows/m890p-daily-full.yml) | Experimental M890P daily smoke/full execution with automatic same-environment groups, artifact-backed build and Case-group distribution, and exact per-group reconciliation. Full execution selects 1/2/4/8-node groups and is not a PR gate. |
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

When an open PR targeting `main` or `release/*` is ready for review, start with the
bot's **Review modules** comment. It lists the primary module, an optional secondary
module, their required roles, and additional modules you may want someone to look at.
PCCL invites the primary module's Maintainer and a rotating Reviewer, skipping the
author's own role. For the secondary module, it invites only a rotating non-author
Reviewer, even when the author is on that module's Reviewer list. This works for forks
too. The [module roster](skills/pccl-review-assignment/modules.json) shows who can approve each role.

### If the assignment needs adjusting

A repository writer can add a missing module by commenting `/review-module add memory`.
That adds both roles. Append `reviewer` or `maintainer` to change just one. Use `rm`
to remove a requirement. For example, `/review-module rm memory reviewer` leaves any
Maintainer requirement in place. Changing an invitation in the GitHub UI only changes who is notified.

Both `add` and `rm` require repository write access or higher, including on the author's
own PR. Writers can remove automatic requirements too; the exemption applies to the
HEAD **when the command runs** and expires on the next commit. Manual additions carry
forward until removed. To revise a command, post a new comment. A `confused` reaction
points to a rejected or failed command; the Actions summary explains why. PCCL withdraws
obsolete bot invitations and respects manual invitations, cancellations and reviews
already submitted.

### What turns the review gate green?

The current HEAD needs approval for each remaining required role: both Maintainer and
Reviewer for the primary module, and only Reviewer for the secondary module by default.
Any non-author Reviewer in the same module can substitute for the default invitee.
Authors are exempt from their own roles, except that the secondary module always needs
a non-author Reviewer when its Reviewer role is required. Manual additions still require
the specified roles. One vote covers all roles that person holds. The Lead can approve
all remaining roles on another author's PR.

At least one module must retain both Reviewer and Maintainer requirements unless the
Lead approves the current HEAD on another author's PR. Removing a role is allowed, but
without a complete pair the gate waits for Lead approval or restoration of a complete
pair and its required approvals. An author exemption satisfies a role without counting
as an approval vote. Removing all requirements keeps the gate pending even with Lead approval.

After a push or a review change, allow the gate to refresh. If a module update was
interrupted, its **Details** link identifies the run to retry. For a merge to `main`,
check both `PCCL Review Gate` and `PCCL Tests Gate`, along with GitHub's other
required checks. The tests gate tracks CI and hardware tests independently of module approval.

### Preparing the squash message

For a multi-commit PR targeting `main`, a successful review gate lets the preview
job generate a **Suggested squash commit body**. Match the commit printed in the
comment to the HEAD you plan to merge. Check its summary, issue references and
sign-offs, then paste the body into **Squash and merge** and review the title.
An earlier preview stays visible while the gate is pending or generation is retried.
A red gate removes the old preview; a successful publication replaces it.

### Maintaining the workflow

[Routing](workflows/review-routing.yml) classifies the patch and passes the result to
the [invitation workflow](workflows/review-module.yml). A writer's
[module command](workflows/review-command.yml) reaches the same workflow after a
permission check. The [review event](workflows/review-event.yml) and routing results
trigger the [gate](workflows/review-gate.yml), whose completion triggers the
[squash preview](workflows/review-squash-preview.yml). The
[classification prompt](skills/pccl-review-assignment/SKILL.md) selects at most two
modules in priority order, regardless of the total changed line count. Keep each
module's Maintainer separate from its Reviewer list. If a secondary module has no
Reviewer other than the PR author, routing fails and reports the module; update its
Reviewer roster before retrying.

When deploying this policy to `main`, let in-flight updates finish, then run
`gh workflow run review-routing.yml --ref main -f pr_number=NUMBER` for each open,
ready PR targeting `main` or `release/*`. Wait for its new gate verdict before
merging. This replaces classifications and green statuses issued under the previous
policy. The same command refreshes an invalid saved classification; for a retired
module, remove its saved roles with `rm` first.
