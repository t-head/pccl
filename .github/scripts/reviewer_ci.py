# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
"""Wait for the contribution checks before spending a model call or inviting people."""

import time
from urllib.parse import urlencode

CHECKIN_PATH = ".github/workflows/pr-test-ppu.yml"
INVITATION_CHECK = "PCCL Reviewer Invitations"
PRECHECK_JOBS = {"Pre-check (Secret Scan) / Secret Scan (TruffleHog)", "Pre-check (Commit Message & DCO)"}


def current_checkin(api, pr):
    head = pr["head"]
    title = f"PCCL checkin for PR #{pr['number']} at {pr['merge_commit_sha']}"
    runs = api.pages(f"/actions/workflows/pr-test-ppu.yml/runs?event=pull_request&head_sha={head['sha']}",
                     key="workflow_runs")
    candidates = []
    fork_prs = None
    for run in runs:
        if (run["path"] != CHECKIN_PATH or run["event"] != "pull_request" or
                run["head_sha"] != head["sha"] or run["repository"]["full_name"] != api.repository or
                run["display_title"] != title):
            continue
        # A title cannot override GitHub's PR association. Fork runs may only identify a source branch.
        if run["pull_requests"]:
            if not any(item["number"] == pr["number"] for item in run["pull_requests"]):
                continue
        else:
            if (not run.get("head_repository") or not head.get("repo") or
                    run["head_repository"]["id"] != head["repo"]["id"] or run["head_branch"] != head["ref"]):
                continue
            if fork_prs is None:
                query = urlencode({"state": "all", "head": f"{head['repo']['owner']['login']}:{head['ref']}"})
                fork_prs = {item["number"] for item in api.pages(f"/pulls?{query}")
                            if item["head"]["sha"] == head["sha"]}
            if fork_prs != {pr["number"]}:
                raise ValueError("Cannot identify this fork run uniquely; use a distinct PR branch")
        candidates.append(run)
    return max(candidates, key=lambda item: (item["run_started_at"], item["id"]), default=None)


def wait_prechecks(api, pr):
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        fresh = api.request("GET", f"/pulls/{pr['number']}")
        if (fresh["state"] != "open" or fresh["draft"] or fresh["head"]["sha"] != pr["head"]["sha"] or
                fresh["base"]["ref"] != pr["base"]["ref"]):
            raise ValueError("PR changed while waiting for prechecks; rerun reviewer routing")
        if fresh["mergeable"] is False:
            raise ValueError("Resolve the merge conflict before requesting review")
        # GitHub initially returns null while computing a new test merge.
        run = current_checkin(api, fresh) if fresh["mergeable"] and fresh.get("merge_commit_sha") else None
        if run:
            jobs = api.pages(f"/actions/runs/{run['id']}/jobs?filter=latest", key="jobs")
            checks = {job["name"]: job for job in jobs if job["name"] in PRECHECK_JOBS}
            if any(job["conclusion"] not in {None, "success"} for job in checks.values()):
                raise ValueError("Prechecks did not pass; fix them before inviting reviewers")
            if checks.keys() == PRECHECK_JOBS and all(job["status"] == "completed" and job["conclusion"] == "success"
                                                     for job in checks.values()):
                return fresh["merge_commit_sha"]
            if run["status"] == "completed":
                raise ValueError("Checkin finished without successful prechecks")
        time.sleep(20)
    raise TimeoutError("No current prechecks; reopen or update the PR after a base change, otherwise rerun routing")
