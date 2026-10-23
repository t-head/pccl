# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.

import base64
import json
import re

import reviewer_api

STATE_BRANCH = "workflow/review"
STATE_README = """# PCCL review assignments

- rotation.json: last Reviewer ID per module.
- assignments/<PR>.json: reviewers and the current classification for that PR.

Classification is saved before invitations; new Reviewer choices and rotation
are committed together. Retries reuse saved choices. Only trusted reviewer-routing
automation and authorized maintainers may write this branch. Maintain the roster
in .github/skills/pccl-review-assignment/modules.json on main. Keep this branch
separate; do not merge or rebase it onto main.
"""


def read_state_file(api, revision, path, optional=False):
    try:
        file = api.request("GET", f"/contents/{path}?ref={revision}")
    except reviewer_api.GitHubError as error:
        if optional and error.status == 404:
            return None
        raise
    if file.get("encoding") != "base64":
        raise ValueError(f"Cannot read review state: {path}")
    record = json.loads(base64.b64decode(file["content"]))
    if not isinstance(record, dict):
        raise ValueError("Invalid review state; repair workflow/review before retrying")
    return record


def read_assignments(api, number):
    """Read both records at one revision; a missing branch starts the rotation."""
    empty = {"reviewers": {}, "classification": None}
    try:
        revision = api.request("GET", f"/git/ref/heads/{STATE_BRANCH}")["object"]["sha"]
    except reviewer_api.GitHubError as error:
        if error.status == 404:
            return empty, {}, None
        raise
    latest = read_state_file(api, revision, "rotation.json")
    record = read_state_file(api, revision, f"assignments/{number}.json", optional=True)
    if record is None:
        record = empty
    elif set(record) != {"reviewers", "classification"} or not isinstance(record["classification"], dict):
        raise ValueError("Invalid PR review state; repair workflow/review before retrying")
    for assignments in (latest, record["reviewers"]):
        if not isinstance(assignments, dict) or any(
                not re.fullmatch(r"[a-z][a-z0-9-]*", module) or not reviewer_api.valid_user_id(reviewer)
                for module, reviewer in assignments.items()):
            raise ValueError("Invalid Reviewer IDs; repair workflow/review before retrying")
    return record, latest, revision


def save_review_state(api, number, previous, record, latest, revision):
    if record == previous:
        return revision
    rotation = {**latest, **{module: reviewer for module, reviewer in record["reviewers"].items()
                            if module not in previous["reviewers"]}}
    files = {"rotation.json": rotation, f"assignments/{number}.json": record}
    tree = {"tree": [{"path": path, "mode": "100644", "type": "blob",
                       "content": json.dumps(data, indent=2, sort_keys=True) + "\n"}
                      for path, data in files.items()]}
    if revision:
        tree["base_tree"] = api.request("GET", f"/git/commits/{revision}")["tree"]["sha"]
    else:
        # An orphan branch contains only state, with no copy of main's code or workflows.
        tree["tree"].append({"path": "README.md", "mode": "100644", "type": "blob",
                             "content": STATE_README})
    tree_sha = api.request("POST", "/git/trees", tree)["sha"]
    commit = api.request("POST", "/git/commits", {
        "message": f"review: update PR #{number}", "tree": tree_sha,
        "parents": [revision] if revision else []})["sha"]
    if revision:
        # A concurrent edit rejects the update instead of overwriting another allocation.
        api.request("PATCH", f"/git/refs/heads/{STATE_BRANCH}", {"sha": commit, "force": False})
    else:
        api.request("POST", "/git/refs", {"ref": f"refs/heads/{STATE_BRANCH}", "sha": commit})
    return commit
