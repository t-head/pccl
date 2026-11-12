"""Refresh the module Maintainer check when routing finishes or a review changes."""

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
SOURCES = {".github/workflows/reviewer-routing.yml": {"pull_request_target", "workflow_dispatch"},
           REVIEW_TRIGGER: {"pull_request_review"}}


def event_pr_number(run, repository):
    if (run["repository"]["full_name"] != repository or
            run["event"] not in SOURCES.get(run["path"].split("@", 1)[0], set())):
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


def saved_classification(api, number):
    record = reviewer_state.read_state_file(api, reviewer_state.STATE_BRANCH,
                                            f"assignments/{number}.json", optional=True)
    return record.get("classification") if record else None


def module_approval(api, pr, roster, skill, classification, approvals):
    if (not classification or classification.get("pr_number") != pr["number"] or
            classification.get("head_sha") != pr["head"]["sha"]):
        return False, "Waiting for module classification of the current commit"
    files = routing.fetch_pr_diff(api, pr)
    if classification.get("input_fingerprint") != routing.classification_fingerprint(pr, files, roster, skill):
        return False, "Module classification is stale. Rerun reviewer routing"
    modules = routing.validate_classification(classification["proposal"], roster)["modules"]
    missing = {}
    for module in modules:
        owner = routing.module_maintainer(roster, module, pr["user"]["id"])
        if owner is not None and owner not in approvals:
            missing.setdefault(owner, []).append(module)
    if missing:
        maintainers = [f"@{api.login(owner)} ({', '.join(owned_modules)})"
                       for owner, owned_modules in missing.items()]
        return False, "Awaiting Maintainer approval: " + ", ".join(maintainers)
    return True, "Maintainer requirements satisfied for the current commit"


def refresh(api, number):
    pr = api.request("GET", f"/pulls/{number}")
    if (pr["state"] != "open" or pr["draft"] or pr["base"]["repo"]["full_name"] != api.repository or
            (pr["base"]["ref"] != routing.WORKFLOW_BRANCH and not pr["base"]["ref"].startswith("release/"))):
        return
    # Match PCCL Tests Gate: a status on the test merge makes GitHub require checks there instead.
    sha = pr["head"]["sha"]
    publish_status(api, sha, "pending", "Checking module Maintainer approvals")
    try:
        roster, skill = routing.load_config()
        approved = routing.current_approvals(pr, api.pages(f"/pulls/{number}/reviews"))
        classification = None
        if roster["lead_maintainer"] in approved:
            ready, reason = True, "Lead Maintainer approved the current commit"
        else:
            classification = saved_classification(api, number)
            ready, reason = module_approval(api, pr, roster, skill, classification, approved)
        if ready:
            verify_review_trigger(api, sha)
            fresh = api.request("GET", f"/pulls/{number}")
            if (routing.pr_snapshot_key(fresh) != routing.pr_snapshot_key(pr) or
                    (classification is not None and saved_classification(api, number) != classification) or
                    routing.current_approvals(fresh, api.pages(f"/pulls/{number}/reviews")) != approved):
                raise ValueError("PR or approvals changed during verification. Rerun the approval check")
        description = (reason if len(reason) <= 140 else
                       "Awaiting Maintainer approval; see Details for the full list")
        publish_status(api, sha, "success" if ready else "pending", description)
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as output:
            output.write(f"PR #{number} at `{pr['head']['sha']}`\n\n{reason}.\n")
    except Exception:
        publish_status(api, sha, "error", "Could not verify Maintainer approval. See workflow logs")
        raise


if __name__ == "__main__":
    try:
        api = reviewer_api.GitHub(os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_TOKEN"])
        run = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())["workflow_run"]
        refresh(api, event_pr_number(run, api.repository))
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
