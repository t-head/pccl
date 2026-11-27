"""Refresh module Reviewer and Maintainer approvals after routing or review changes."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import review_automation as routing
import reviewer_api
import reviewer_state

CHECK = "PCCL Review Gate"
REVIEW_TRIGGER = ".github/workflows/review-event.yml"
SOURCES = {".github/workflows/review-routing.yml": {"pull_request_target", "workflow_dispatch"},
           ".github/workflows/review-command.yml": {"pull_request_target"},
           REVIEW_TRIGGER: {"pull_request_review"}}


def event_pr_number(run, repository, sources=SOURCES):
    if (run["repository"]["full_name"] != repository or
            run["event"] not in sources.get(run["path"].split("@", 1)[0], set())):
        raise ValueError("Unexpected review workflow")
    match = re.fullmatch(r"PCCL review for PR #(.+)", run["display_title"], re.DOTALL)
    if not match:
        raise ValueError("Review workflow did not identify a PR")
    number = int(match[1])
    if number <= 0:
        raise ValueError("Invalid PR number")
    if run["pull_requests"] and not any(item["number"] == number for item in run["pull_requests"]):
        raise ValueError("Review workflow title conflicts with GitHub's PR association")
    return number  # Fork notifications can omit the PR list. Always read verdicts from the PR API.


def publish_status(api, sha, state, description):
    api.request("POST", f"/statuses/{sha}", {
        "context": CHECK, "state": state, "description": description,
        "target_url": f"https://github.com/{api.repository}/actions/runs/{os.environ['GITHUB_RUN_ID']}"})


def verify_review_trigger(api, sha):
    """A green gate relies on the PR workflow forwarding later review dismissals."""
    trusted_blob = subprocess.check_output(["git", "rev-parse", f"HEAD:{REVIEW_TRIGGER}"],
                                           cwd=routing.ROOT, text=True).strip()
    workflows = api.request("GET", f"/contents/.github/workflows?ref={sha}")
    if not any(file["path"] == REVIEW_TRIGGER and file["type"] == "file" and file["sha"] == trusted_blob
               for file in workflows):
        raise ValueError("The PR must include the deployed review notification workflow to refresh approvals")


def saved_review_state(api, number):
    return reviewer_state.read_pr_assignments(api, number)


def module_approval(api, pr, roster, skill, record, approvals):
    history = api.pages(f"/issues/{pr['number']}/timeline")
    required = routing.required_reviewers(pr, record, roster, history)
    if not required:
        reason = ("Waiting for module classification; rerun reviewer routing" if not record["classification"]
                  else "At least one invited module member other than the author must remain")
        return False, reason, {}
    required_summary = "\n\nRequired reviewers: " + ", ".join(f"@{api.login(user)}" for user in sorted(required))
    classification = record["classification"]
    if (not classification or classification.get("pr_number") != pr["number"] or
            classification.get("head_sha") != pr["head"]["sha"]):
        return False, "Waiting for module classification of the current commit; rerun reviewer routing" + required_summary, {}
    files = routing.fetch_pr_diff(api, pr)
    if classification.get("input_fingerprint") != routing.classification_fingerprint(pr, files, roster, skill):
        return False, "Module classification is stale. Rerun reviewer routing" + required_summary, {}
    requirements = routing.review_requirements(pr, record, roster, required)
    if not any({"Reviewer", "Maintainer"} <= roles for roles in requirements.values()):
        return (False, "Invite a same-module Reviewer and Maintainer; author exemptions count" +
                required_summary, {})
    missing = {"Reviewer": {}, "Maintainer": {}, "Invited reviewer": {}}
    for user in sorted(required - approvals):
        assigned = False
        for module, roles in requirements.items():
            entry = roster["modules"][module]
            for role, eligible in (("Reviewer", user in entry["reviewers"]),
                                   ("Maintainer", user == entry["maintainer"])):
                if role in roles and eligible:
                    missing[role].setdefault(user, []).append(module)
                    assigned = True
        if not assigned:
            missing["Invited reviewer"][user] = []
    pending, details = {}, []
    for role, users in missing.items():
        if users:
            pending[role] = [f"@{api.login(user)}" for user in users]
            accounts = [login + (f" ({', '.join(modules)})" if modules else "")
                        for login, modules in zip(pending[role], users.values())]
            details.append(f"{role} approval: " + ", ".join(accounts))
    if pending:
        reason = ("Awaiting " + "; ".join(details) +
                  ". Every invited module member must approve the current commit; remove an invitation to cancel its "
                  "requirement. A replacement does not waive another invitee")
        return False, reason + required_summary, pending
    return True, "All required reviewers approved the current commit; module coverage satisfied" + required_summary, {}


def refresh(api, number):
    pr = api.request("GET", f"/pulls/{number}")
    if (pr["state"] != "open" or pr["draft"] or pr["base"]["repo"]["full_name"] != api.repository or
            (pr["base"]["ref"] != routing.WORKFLOW_BRANCH and not pr["base"]["ref"].startswith("release/"))):
        return
    # Match PCCL Tests Gate: a status on the test merge makes GitHub require checks there instead.
    sha = pr["head"]["sha"]
    status = next((item for item in api.pages(f"/commits/{sha}/statuses") if item["context"] == CHECK), {})
    if (status.get("state"), status.get("description")) in {
            ("pending", routing.UPDATE_PENDING), ("error", routing.UPDATE_FAILED)}:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as output:
            output.write(f"PR #{number}: module update did not finish. Rerun the workflow linked from {CHECK}.\n")
        return
    publish_status(api, sha, "pending", "Checking required reviewers' approvals")
    try:
        roster, skill = routing.load_config()
        approved = routing.current_approvals(pr, api.pages(f"/pulls/{number}/reviews"))
        record = saved_review_state(api, number)
        ready, reason, pending = module_approval(api, pr, roster, skill, record, approved)
        if ready:
            verify_review_trigger(api, sha)
        description = ("Awaiting " + "; ".join(f"{role} approval: {', '.join(users)}"
                                              for role, users in pending.items()) if pending else reason.split("\n", 1)[0])
        if len(description) > 140:
            description = ("Awaiting " + "; ".join(f"{role} approval: {len(users)}"
                                                  for role, users in pending.items()) + "; see Details"
                           if pending else description[:127] + "; see Details")
        publish_status(api, sha, "success" if ready else "pending", description)
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as output:
            output.write(f"PR #{number} at `{pr['head']['sha']}`\n\n{reason}.\n")
    except Exception:
        publish_status(api, sha, "error", "Could not verify module approvals. See workflow logs")
        raise


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "refresh"
    if mode not in {"identify", "refresh"}:
        raise ValueError("Use identify or refresh")
    repository = os.environ["GITHUB_REPOSITORY"]
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    if os.environ["GITHUB_EVENT_NAME"] == "workflow_dispatch":
        if os.environ["GITHUB_REF"] != f"refs/heads/{routing.WORKFLOW_BRANCH}":
            raise ValueError("Approval refresh must run from the trusted default branch")
        number = int(event["inputs"]["pr_number"])
        if number <= 0:
            raise ValueError("Invalid PR number")
    else:
        number = event_pr_number(event["workflow_run"], repository)
    if mode == "identify":
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(f"pr_number={number}\n")
        return
    api = reviewer_api.GitHub(repository, os.environ["GITHUB_TOKEN"])
    refresh(api, number)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
