# PCCL review assignments

- rotation.json: last Reviewer ID per module.
- assignments/<PR>.json: assigned reviewer IDs and the current module classification.

Reviewer choices and rotation are committed together before new invitations. Retries reuse those choices.
Only trusted reviewer-routing automation and authorized maintainers may write this branch. Maintain the roster
in .github/skills/pccl-review-assignment/modules.json on main. Keep this branch
separate; do not merge or rebase it onto main.
