# GitHub workflows

[Checkin](workflows/pr-test-ppu.yml) runs contribution checks, the organization’s AI and human
review gates, and PPU builds and tests. `PCCL Tests Gate` summarizes their results.
The [test list](workflows/scripts/casecmd.list) defines the PPU matrix.

[Reviewer routing](workflows/reviewer-routing.yml) waits for Secret Scan and Commit Message & DCO
on the current PR revision before classifying modules and inviting reviewers. Copilot uses the
[module catalog and classification rules](skills/pccl-review-assignment/). The workflow rotates
reviewers per module and also invites its Maintainer. The `workflow/review` branch stores the
rotation and PR assignments so updates and retries keep the same people. Limit writes to this
branch to trusted automation and authorized maintainers.

When classification needs a human decision, the Lead Maintainer can run **Recommend PCCL reviewers**
from `main`, entering the PR number and comma-separated module IDs. The same prechecks and invitation
rules apply. Successful invitations publish `PCCL Reviewer Invitations` on the tested merge revision.
