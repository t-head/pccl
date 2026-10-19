# Module review workflow

[中文版](module-review.zh-CN.md)

This workflow routes PRs to the people responsible for the affected PCCL modules. It adds a required approval check
alongside the existing CI. Installation alone does not enforce anything: an administrator must complete the setup below.
The protected target is GitHub `main`; the policy does not cover `ppu1.7`.

## Contribution flow

Contributors do not need Node.js or a local classification command. Declare the affected modules in the PR;
the enabled GitHub Actions workflow checks path coverage and routes reviews automatically.

1. Prepare the change using the existing `pccl-contribution` skill and contribution rules, including issue references,
   one contribution commit, DCO, and relevant testing.
2. Review the full diff and choose the affected modules and specialist areas from the roster below and
   [owners.json](../../.github/review/owners.json). The `pccl-contribution` skill can help classify the change;
   contributors can also fill in the scope directly without using a skill or running a local tool.
3. Fill `PCCL-Modules:` and optionally `PCCL-Areas:` in the PR description. For example:

   ```text
   PCCL-Modules: device, network
   PCCL-Areas: device-gin, network-rdma
   ```

   This example covers a GIN RDMA backend change. Explain additional module impact in Changes & Impact.
4. The trusted workflow unions path matches, declarations, and `module/<id>` / `area/<id>` labels. It requests the
   module Maintainer and the matching specialist Reviewers, excluding the author and duplicate/pending requests.
5. **One current-HEAD approval from a Maintainer OR Backup of any affected module satisfies the module approval check.**
   A cross-module PR needs only one such approval; that person does not need to own every affected module.
   Review requests still cover all affected modules and matching specialist areas.
6. Complete the existing required CI and review rules. An authorized member of the merge team verifies current scope,
   approvals, HEAD, and CI before merging.

Removing a module label or editing the description cannot remove a module inferred from changed paths. Conversely,
path rules cannot detect every semantic dependency: the author, skill, and owners must add those modules explicitly.
Classification is a review responsibility, not a claim that filenames perfectly describe the architecture.

<details>
<summary>Optional local classification preview</summary>

If Node.js 20 or later is already available, this helper can preview path-based classification:

```bash
node .github/review/classify.mjs <fetched-target-ref> [<pr-body-file>]
```

`<fetched-target-ref>` is the verified, fetched target branch; `<pr-body-file>` is an optional PR description file.
The helper reads committed changes from the target's merge base to HEAD. It only prints a proposal; it does not edit
the PR, request reviewers, or detect every semantic dependency. Skipping this preview does not affect server-side
classification or approval requirements.

</details>

## Owners and specialist areas

[owners.json](../../.github/review/owners.json) is the authoritative roster and routing configuration.
Names below are GitHub logins, not internal accounts.

| Module ID | Scope | Maintainer | Backup |
| --- | --- | --- | --- |
| `runtime` | Runtime and task scheduling | `ctt284373` | `ck-milo` |
| `topology` | Topology discovery and path search | `xiangzhixyc` | `ck-milo` |
| `tuning` | Algorithm and protocol tuning | `ck-milo` | `honganlxh` |
| `network` | Network communication | `xiangzhixyc` | `snowdrift-zw` |
| `kernels` | Collective and P2P kernels | `ck-milo` | `xiangzhixyc` |
| `device` | Device API and GIN | `wensuyang` | `snowdrift-zw` |
| `profiler` | Profiler and Inspector | `levilee-lwl` | `whale494255` |
| `ras` | RAS and diagnostics | `ck-milo` | `QingxunLy` |
| `workflow` | Contribution, review, and merge workflow | `honganlxh` | `wensuyang` |
| `build` | Build, CI, and unclassified-path triage | `levilee-lwl` | `ck-milo` |

The 23 specialist areas and their Reviewer lists are under each module's `areas` in the configuration.
Reviewers advise within those areas; their approval satisfies the module gate only if they also hold a Maintainer or
Backup role in an affected module. This version does not require a separate specialist approval in addition
to the owner approval. A writer's outstanding changes-requested review still blocks the module check.

`zian.cza` remains an EIC candidate without a confirmed GitHub account. The workflow skips automatic requests for that
candidate until the mapping is filled. Other configured accounts are requested normally.

### Backup handling

The Backup has the same approval authority as the Maintainer for that module. Approval does not depend on which person
was requested. Default routing requests the Maintainer; if that person authored the PR, it requests the Backup.

If the Maintainer is unavailable, a collaborator can apply `backup/<module-id>`, for example `backup/device` or
`backup/network`, to request the Backup. The workflow does not remove existing human review requests. The label cannot
change who qualifies as an approver. To choose a different Backup, update the protected roster through a reviewed PR.
Both configured owners still need repository write access for their approvals to count.

Labels are optional: the PR body works without pre-creating module/area labels. Administrators or triagers can create
the labels they want to use, with the exact IDs from the policy.

### Routing boundaries

Path rules are ordered, and the first rule for each path wins. Renames use both old and new paths. Examples:

- `src/graph/tuning.cc` maps to `tuning`; other graph discovery/search files map to `topology`.
- GIN RDMA backend code maps to both `device` and `network`; Device API interfaces map to `device`.
- Extended kernels map to `kernels-4`; RedOp, symmetric, and CE/OKI/P2P DMA have narrower specialist areas.
- EIC/SNIC changes within shared networking files need `network-eic` / `network-snic` declarations as appropriate.
- `src/misc/drv_extension.cc` dispatch/resource tuning maps to `tuning`; add `runtime` or `kernels` when their behavior
  changes. Its physical location under `misc` does not determine the semantic scope.
- Shared host code has `runtime` coverage. Paths without a rule are additionally routed to `build` for triage and are
  listed in the check output. Review those paths and confirm any additional affected modules before approving.

Contribution guides, PR/issue templates, agent review guidance and skills, the ownership roster, module-review workflows,
and their scripts, tests, and documentation are routed to `workflow` / `workflow-review`.
Build scripts, packaging, and hardware CI (including `pr-test-ppu.yml` and its scripts) remain under `build`.
The specific contribution-workflow rules take precedence over the general `.github/` CI rule.
The evaluator uses the current trusted `main` configuration, not proposed policy changes from the PR. After a policy
change merges, the push event re-evaluates open PRs using the new policy. Request review from the affected module owners
for account changes, and verify repository access before rollout.

## Administrator setup

This version uses only GitHub Actions and its built-in **`GITHUB_TOKEN`** to request reviewers and publish
**PCCL Module Owner Approval**. No separate GitHub App, private key, PAT, or environment is required.

1. Merge the implementation through the existing approved contribution process. Protect `main` from unreviewed pushes
   and confirm that repository/organization Actions policies permit the evaluator's declared permissions:
   **Contents: read**, **Actions: read**, **Pull requests: write**, and **Checks: write**.
   The write permissions are scoped to the evaluator job; it has no contents-write permission and does not merge PRs.
2. Keep repository variable `PCCL_REVIEW_ENABLED` absent/false until ready for verification, then set it to `true`.
   The job reads this repository variable before running. GitHub supplies the token automatically; do not create a
   secret named `GITHUB_TOKEN` or provide a personal token.
3. Use a test PR and the manual **PCCL Module Review** run on `main` with its PR number. Confirm that
   **PCCL Module Owner Approval** appears on the actual PR HEAD with **GitHub Actions** as its source.
   Exercise the rollout cases below before making the check mandatory.
4. Add **PCCL Module Owner Approval** to the required status checks for `main`, selecting **GitHub Actions as the
   expected source**. Require this exact check, not the workflow's **Reconcile module approvals** job.
   The job can succeed after publishing a failed module check, because evaluating a PR is not approving it.
   Also require **Review policy tests** from GitHub Actions and retain existing CI/security and native review rules.
5. Keep required PR reviews, stale-review dismissal, last-push review, conversation resolution as applicable, and
   require branches to be up to date before merging. Preserve existing repository and organization protections.
6. Restrict actual updates/merges into `main` to the intended Maintainer/Backup merge team. With rulesets, use a
   **separate Restrict updates ruleset** and give that team a **For pull requests only** bypass of that restriction.
   Keep PR approval and required-check rules in a different ruleset where the team has no bypass. Combining all rules
   into the bypassed ruleset would also bypass the checks. Native branch protection's allowed push actors is another
   option where supported. Verify the account/plan's available controls in repository settings.
7. Validate with one authorized merge-team member and one specialist outside that team: both may review, but only the
   authorized team member should be able to merge a PR whose required approvals and CI have passed.

The expected source identifies GitHub Actions, not a particular workflow file. Reserve **PCCL Module Owner Approval**
for this evaluator; no other job, workflow, or status publisher should use that name. Review workflow and policy changes
carefully. Source selection does not isolate this check from other workflows using the same GitHub Actions identity.

Write access is repository-wide; it does not encode which module someone can merge. The module check controls whose
approval is needed, while the team/branch restriction controls who executes the merge. Include Backups in the merge
team if they should perform both actions. Existing collaborators with write access are not automatically removed or
restricted by these files. Configuring this setup requires administrator privileges.

This implementation does not add a `CODEOWNERS` file. Existing CODEOWNERS rules, if added separately, would remain an
independent path-based gate; they would not read `PCCL-Modules` or provide Maintainer/Backup priority.

## Events, checks, and limits

- `module-review.yml` runs trusted main code via `pull_request_target`, `workflow_run`, manual dispatch, and main pushes.
  It never checks out or executes a PR's code, downloads its artifacts, or consumes its caches.
- `module-review-events.yml` is an unprivileged signal for submitted/edited/dismissed reviews, including fork reviews.
  Its completion wakes the trusted evaluator, which reads live PRs, files, reviews, and permissions from GitHub.
  Missing fork/merge-ref association falls back to re-evaluating open target PRs. The relay's result is not approval.
- A periodic run, approximately every 15 minutes, recovers missed events and rechecks collaborator permissions.
  GitHub may delay scheduled runs. A manual refresh is available in Actions when a check is stale.
- Evaluations paginate files and reviews, bind approvals to the actual PR HEAD, reject self/bot approvals, and compare
  live state again before publishing. API failures, incomplete lists, and unsupported large PRs never produce success.
  PRs above GitHub's 3,000-file list limit must be split.
- Check runs attach to commits, not exclusively to a PR. Multiple open PRs sharing a HEAD SHA are rejected: use a
  distinct commit for each PR so unrelated module descriptions cannot share one successful result.
- Review requests are invitations, not evidence of approval. Failed invitations appear in the check output; maintainers
  can request those people manually. They never make a missing owner approval pass.
- Label/body updates made by other automation using `GITHUB_TOKEN` may not trigger another workflow run. Use a manual
  refresh or wait for periodic reconciliation. This flow does not rely on its own reviewer requests triggering events.
- Webhooks, API snapshots, and check updates are asynchronous. Editing labels/body, dismissing approval, or changing
  permissions can briefly leave an older result visible before reconciliation. If GitHub cannot accept a new check
  at all, the evaluator cannot withdraw an older result until service recovers. This is not an atomic merge service.
  Keep native review protections and verify the latest scope/approvals/check before merging; do not use this version
  for unattended auto-merge where an instantaneous module-policy guarantee is required.
- Merge queue (`merge_group`) is **not supported** in this version. Do not enable it with this required check until
  merge-group evaluation is implemented and tested.
- Existing `pccl checkin` and its Human Review Gate remain intact. Permission to start hardware CI is separate from
  module approval and from permission to merge. This workflow does not decide release/publishing permissions.

## Validation and rollout

Run local policy/API-mock tests with Node.js 20 or later:

```bash
node --test tests/review/*.test.mjs
```

Before enforcement, verify in a real test PR:

- Missing owner approval blocks; Maintainer approval passes; Backup approval also passes without an absence label.
- A specialist-only approval blocks; an author cannot approve their own module.
- A cross-module PR passes with one affected-module Maintainer or Backup approval; approvals from owners of unaffected
  modules do not count. Removing a module label cannot remove path-derived review scope.
- New commits, approval dismissal, draft conversion, and scope changes refresh the result correctly. Adding a module
  preserves an otherwise valid approval; removing the approver's only affected module makes that approval ineligible.
- Fork reviews trigger reconciliation; manual refresh works when the relay is delayed or unavailable.
- Editing the policy/check implementation inside a PR does not affect its own required evaluation.
- The exact required check appears on the PR HEAD from GitHub Actions; no other workflow/job uses that name.
  Missing approval still blocks when the orchestration job succeeds, and unrelated PRs cannot share a HEAD result.
- Both authorized merge-team and non-team writer accounts see the intended final merge restriction.

Local tests do not verify live GitHub token permissions, reviewer invitations, ruleset enforcement, or PPU CI.
Do not describe the system as enabled until the administrator setup and real-PR rollout checks have passed.

## GitHub references

- [Rules available for rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)
- [Repository roles](https://docs.github.com/en/organizations/managing-access-to-your-organizations-repositories/repository-roles-for-an-organization)
- [CODEOWNERS](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
- [Workflow events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)
- [GITHUB_TOKEN](https://docs.github.com/en/actions/concepts/security/github_token)
- [Workflow permissions](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#permissions)
