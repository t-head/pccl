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
           REVIEW_TRIGGER: {"pull_request_review"}}


def event_pr_number(run, repository, sources=SOURCES):
    if (run["repository"]["full_name"] != repository or
            run["event"] not in sources.get(run["path"].split("@", 1)[0], set())):
        raise ValueError("Unexpected review workflow")
    match = re.fullmatch(r"PCCL review for PR #([1-9][0-9]*)", run["display_title"])
    if not match:
        raise ValueError("Review workflow did not identify a PR")
    number = int(match[1])
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
    modules = routing.primary_modules(record, roster)
    requirements = routing.review_requirements(record, roster, pr["head"]["sha"], modules)
    if not requirements:
        return False, "At least one module must have a required role", {}
    if roster["lead_maintainer"] in approvals:
        return (True,
                "Lead Maintainer approved all module Reviewer and Maintainer requirements for the current commit", {})
    if not any({"Reviewer", "Maintainer"} <= roles for roles in requirements.values()):
        return False, "At least one module must require both Reviewer and Maintainer unless the Lead approves", {}
    classification = record["classification"]
    if (not classification or classification.get("pr_number") != pr["number"] or
            classification.get("head_sha") != pr["head"]["sha"]):
        return False, "Waiting for module classification of the current commit", {}
    files = routing.fetch_pr_diff(api, pr)
    if classification.get("input_fingerprint") != routing.classification_fingerprint(pr, files, roster, skill):
        return False, "Module classification is stale. Rerun reviewer routing", {}
    author = pr["user"]["id"]
    missing = {"Reviewer": {}, "Maintainer": {}}
    for module, roles in requirements.items():
        candidates = roster["modules"][module]["reviewers"]
        reviewer_exempt = author in candidates and module not in modules[1:]
        if "Reviewer" in roles and not reviewer_exempt and not approvals.intersection(candidates):
            reviewer = record["reviewers"].get(module)
            if reviewer not in candidates or reviewer == author:
                return False, f"Waiting for Reviewer assignment for {module}; rerun reviewer routing", {}
            missing["Reviewer"].setdefault(reviewer, []).append(module)
        owner = routing.module_maintainer(roster, module, author)
        if "Maintainer" in roles and owner is not None and owner not in approvals:
            missing["Maintainer"].setdefault(owner, []).append(module)
    pending, details = {}, []
    for role, users in missing.items():
        if users:
            pending[role] = [f"@{api.login(user)}" for user in users]
            accounts = [f"{login} ({', '.join(owned_modules)})"
                        for login, owned_modules in zip(pending[role], users.values())]
            details.append(f"{role} approval: " + ", ".join(accounts))
    if pending:
        reason = "Awaiting " + "; ".join(details)
        if "Reviewer" in pending:
            reason += (". Any non-author Reviewer in the same module's roster can approve the current commit "
                       "instead of the named default; other required approvals remain unchanged")
        reason += (". The Lead Maintainer can approve both roles for all modules on another author's PR "
                   "at the current commit")
        return False, reason, pending
    return True, "Reviewer and Maintainer requirements satisfied for the current commit", {}


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
    publish_status(api, sha, "pending", "Checking module Reviewer and Maintainer approvals")
    try:
        roster, skill = routing.load_config()
        approved = routing.current_approvals(pr, api.pages(f"/pulls/{number}/reviews"))
        record = saved_review_state(api, number)
        ready, reason, pending = module_approval(api, pr, roster, skill, record, approved)
        if ready:
            verify_review_trigger(api, sha)
        reviewer_hint = " (or a same-module Reviewer)"
        description = ("Awaiting " + "; ".join(f"{role} approval: {', '.join(users)}" +
                                              (reviewer_hint if role == "Reviewer" else "")
                                              for role, users in pending.items()) if pending else reason)
        if len(description) > 140:
            description = ("Awaiting " + "; ".join(f"{role} approval: {len(users)}" +
                                                  (reviewer_hint if role == "Reviewer" else "")
                                                  for role, users in pending.items()) + "; see Details")
        publish_status(api, sha, "success" if ready else "pending", description)
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as output:
            output.write(f"PR #{number} at `{pr['head']['sha']}`\n\n{reason}.\n")
    except Exception:
        publish_status(api, sha, "error", "Could not verify module approvals. See workflow logs")
        raise


def main():
    api = reviewer_api.GitHub(os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_TOKEN"])
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    if os.environ["GITHUB_EVENT_NAME"] == "workflow_dispatch":
        if os.environ["GITHUB_REF"] != f"refs/heads/{routing.WORKFLOW_BRANCH}":
            raise ValueError("Approval refresh must run from the trusted default branch")
        number = int(event["inputs"]["pr_number"])
        if number <= 0:
            raise ValueError("Invalid PR number")
    else:
        number = event_pr_number(event["workflow_run"], api.repository)
    refresh(api, number)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
