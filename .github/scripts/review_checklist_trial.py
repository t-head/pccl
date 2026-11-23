import copy
from datetime import datetime, timedelta, timezone
import json
import os
import re
import sys
import time

import maintainer_approval
import review_automation as routing
import reviewer_api
import reviewer_state

REPOSITORY = "t-head/pccl"
NUMBER = 195
BRANCH = "experiment/pr195-review-checklist"
CHECK = "PCCL Review Checklist Trial"
DURATION = 1800
INTERVAL = 15
MARKER = "<!-- pccl-review-checklist-trial:"
ROW = re.compile(r"^- \[([ xX])\] (.+) <!-- trial-role:([a-z][a-z0-9-]*):(Reviewer|Maintainer) -->$", re.M)
COMMENT_QUERY = """query($id: ID!) {
  node(id: $id) {
    ... on IssueComment {
      body lastEditedAt
      author { __typename login ... on Bot { databaseId } }
      editor { __typename login ... on User { databaseId } }
    }
  }
}"""


class TrialAPI(reviewer_api.GitHub):
    def __init__(self, token, sha):
        super().__init__(REPOSITORY, token)
        self.sha = sha
        self.result_id = None
        self.created = 0

    def request(self, method, path, data=None, *, repository=True):
        if method != "GET":
            if path == "/graphql" and not repository:
                allowed = method == "POST" and data.get("query") == COMMENT_QUERY
            elif path == f"/statuses/{self.sha}" and repository:
                allowed = method == "POST" and data.get("context") == CHECK
            elif path == f"/issues/{NUMBER}/comments" and repository:
                allowed = method == "POST" and self.created < 2 and data.get("body", "").startswith(MARKER)
                if allowed:
                    self.created += 1
            else:
                allowed = (repository and method == "PATCH" and self.result_id is not None and
                           path == f"/issues/comments/{self.result_id}" and
                           data.get("body", "").startswith(MARKER))
            if not allowed:
                raise ValueError(f"Write outside the isolated trial: {method} {path}")
        return super().request(method, path, data, repository=repository)


def authorize(api, actor):
    if actor.get("__typename") != "User" or not reviewer_api.valid_login(actor.get("login")):
        raise ValueError("Only human repository writers may change this checklist")
    permission = api.request("GET", f"/collaborators/{actor['login']}/permission")
    if (permission["permission"] not in {"write", "maintain", "admin"} or
            permission["user"]["id"] != actor.get("databaseId")):
        raise ValueError("Repository write access is required; this edit was not applied")


def parse_selection(body, template):
    normalized = lambda value: ROW.sub(lambda match: "- [ ] " + match[2] +
                                      f" <!-- trial-role:{match[3]}:{match[4]} -->", value.replace("\r\n", "\n")).strip()
    if normalized(body) != normalized(template):
        raise ValueError("Only toggle the existing checkboxes; other text changes were not applied")
    return {(match[3], match[4]) for match in ROW.finditer(body.replace("\r\n", "\n"))
            if match[1].lower() == "x"}


def apply_selection(record, before, after, sha, sequence):
    updated = copy.deepcopy(record)
    for module, role in before ^ after:
        command = {"comment_id": sequence, "required": (module, role) in after}
        if not command["required"]:
            command["excluded_head"] = sha
        updated.setdefault("manual_modules", {}).setdefault(module, {})[role] = command
    return updated


def row(api, module, role, selected, record, roster, author, secondary):
    entry = roster["modules"][module]
    if role == "Maintainer":
        person = entry["maintainer"]
        exempt = person == author
    else:
        exempt = author in entry["reviewers"] and module not in secondary
        person = author if exempt else record["reviewers"][module]
    mark = "x" if (module, role) in selected else " "
    label = f"{role} — `@{api.login(person)}`" + (" (author-exempt)" if exempt else "")
    return f"- [{mark}] {label} <!-- trial-role:{module}:{role} -->"


def checklist_body(api, pr, record, roster, modules, selected, expires, run_url):
    lines = [f"{MARKER}{os.environ['GITHUB_RUN_ID']}:checklist -->", "", "## Review requirements · live trial", "",
             f"PR #{NUMBER} · commit `{pr['head']['sha'][:12]}` · active until **{expires}**", "",
             "> Checking a role requires its approval; it does **not** mean approval was given.", "",
             "Toggle the boxes directly, or use **Edit** if GitHub does not expose clickable boxes.",
             f"The result comment below and **{CHECK}** refresh about every {INTERVAL} seconds.", ""]
    for index, module in enumerate(modules):
        lines += [f"### {'Primary' if index == 0 else 'Secondary'} module · `{module}`", ""]
        lines += [row(api, module, role, selected, record, roster, pr["user"]["id"], modules[1:])
                  for role in ("Reviewer", "Maintainer")]
        lines.append("")
    if len(modules) == 1:
        lines += ["Secondary module: none. The saved classification selected only one module.", ""]
    lines += ["<details>", "<summary>Other modules — optional manual requirements</summary>", "",
              "These are catalog choices, not an assertion that this PR changes every module.", ""]
    for module in roster["modules"]:
        if module in modules:
            continue
        lines += [f"**`{module}`**", ""]
        lines += [row(api, module, role, selected, record, roster, pr["user"]["id"], modules[1:])
                  for role in ("Reviewer", "Maintainer")]
        lines.append("")
    lines += ["</details>", "", "---", "",
              "Write access or higher is required, including for the PR author. Only edit checkboxes.",
              "The trial uses the branch's approval rules and real reviews on this commit, including author exemptions,",
              "same-module Reviewer substitutes and Lead approval. Removing every role keeps the trial pending.", "",
              "**Isolated trial:** no invitations, review requests, shared assignments or production Gate are changed.",
              "Account names are shown as code to avoid mentions. The initial scope comes from the saved classification;",
              "no new AI classification is claimed. Selections exist only for this run; a PR snapshot change stops it.",
              "Automatic exclusions apply to this HEAD; cross-commit persistence is not exercised by this short trial.", "",
              "Try unchecking only the primary Reviewer: the remaining Maintainer role is author-exempt on this PR.",
              "Then uncheck Maintainer too to see the empty-scope block; restore both boxes to return to the initial state.", "",
              f"[Trial run and logs]({run_url}) · This check is not a merge requirement.", ""]
    return "\n".join(lines)


def publish(api, state, description, run_url):
    api.request("POST", f"/statuses/{api.sha}", {
        "context": CHECK, "state": state, "description": description[:140], "target_url": run_url})


def result_body(state, reason, selected, actor, expires, checklist_url):
    roles = "; ".join(f"{module}: {', '.join(role for role in ('Reviewer', 'Maintainer') if (module, role) in selected)}"
                      for module in sorted({module for module, _ in selected})) or "none"
    return (f"{MARKER}{os.environ['GITHUB_RUN_ID']}:result -->\n\n"
            f"## {CHECK} · {state.upper()}\n\n"
            f"Applied requirements: `{roles}`\n\n"
            f"Last accepted edit: `{actor}`\n\n"
            f"```text\n{reason}\n```\n\n"
            f"Active until **{expires}**; refresh interval about {INTERVAL} seconds. "
            "No invitations or production Gate changes.\n\n"
            f"[Edit the trial checklist]({checklist_url})\n")


def stop(api, run_url):
    publish(api, "pending", "Trial inactive; further checkbox edits are not processed", run_url)
    result_id = int(os.environ["TRIAL_RESULT_ID"])
    comment = api.request("GET", f"/issues/comments/{result_id}")
    expected = f"{MARKER}{os.environ['GITHUB_RUN_ID']}:result -->"
    if (comment["user"]["id"] != routing.GITHUB_ACTIONS_BOT_ID or comment["user"]["type"] != "Bot" or
            not comment["body"].startswith(expected) or
            comment["issue_url"] != f"https://api.github.com/repos/{REPOSITORY}/issues/{NUMBER}"):
        raise ValueError("Unexpected trial result comment")
    api.result_id = result_id
    body = comment["body"].replace(f"## {CHECK} ·", f"## {CHECK} · INACTIVE — last result:", 1)
    body += "\n**Trial stopped.** Further checkbox edits are not processed. The last result above is historical.\n"
    api.request("PATCH", f"/issues/comments/{result_id}", {"body": body})


def run(api, run_url):
    actor = api.request("GET", f"/users/{os.environ['GITHUB_TRIGGERING_ACTOR']}", repository=False)
    authorize(api, {"__typename": actor["type"], "login": actor["login"], "databaseId": actor["id"]})
    pr = api.request("GET", f"/pulls/{NUMBER}")
    if (pr["head"]["sha"] != api.sha or pr["head"]["repo"]["full_name"] != REPOSITORY or
            pr["head"]["ref"] != "ci/squash-preview-check" or pr["base"]["repo"]["full_name"] != REPOSITORY or
            pr["base"]["ref"] != "main" or pr["state"] != "open" or pr["draft"]):
        raise ValueError("The trial requires the pinned open PR #195 snapshot")
    roster, skill = routing.load_config()
    record = reviewer_state.read_pr_assignments(api, NUMBER)
    source = record["classification"]
    if not source or source["head_sha"] != api.sha or source["pr_number"] != NUMBER:
        raise ValueError("Wait for a saved classification of the current PR commit")
    proposal = routing.validate_classification({
        **source["proposal"], "additional_modules": source["proposal"].get("additional_modules", [])}, roster)
    modules = proposal["modules"]
    files = routing.fetch_pr_diff(api, pr)
    record["classification"] = {**source, "proposal": proposal,
                                "input_fingerprint": routing.classification_fingerprint(pr, files, roster, skill)}
    record["reviewers"] = routing.select_reviewers(
        list(roster["modules"]), roster["modules"], record["reviewers"], {}, pr["user"]["id"], modules[1:])
    requirements = routing.review_requirements(record, roster, api.sha, modules)
    selected = {(module, role) for module, roles in requirements.items() for role in roles}
    expires = (datetime.now(timezone.utc) + timedelta(seconds=DURATION)).strftime("%Y-%m-%d %H:%M:%S UTC")
    template = checklist_body(api, pr, record, roster, modules, selected, expires, run_url)
    checklist = api.request("POST", f"/issues/{NUMBER}/comments", {"body": template})
    initial = result_body("pending", "Starting approval calculation", selected, "initial scope", expires,
                          checklist["html_url"])
    result = api.request("POST", f"/issues/{NUMBER}/comments", {"body": initial})
    api.result_id = result["id"]
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"result_id={api.result_id}\nhead={api.sha}\n")
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
        summary.write(f"[Trial checklist]({checklist['html_url']})\n\n[Live result]({result['html_url']})\n\n"
                      f"Expires at {expires}. No production Gate or invitations are changed.\n")
    print(json.dumps({"checklist": checklist["html_url"], "result": result["html_url"], "expires": expires}), flush=True)
    version, last_body, last_status = (None, template), initial, None
    last_actor, edit_error, sequence = "initial scope", None, 0
    deadline = time.monotonic() + DURATION
    while time.monotonic() < deadline:
        fresh = api.request("GET", f"/pulls/{NUMBER}")
        if routing.pr_snapshot_key(fresh) != routing.pr_snapshot_key(pr):
            print("PR snapshot changed; stopping without applying this checklist to another snapshot.", flush=True)
            break
        response = api.request("POST", "/graphql", {
            "query": COMMENT_QUERY, "variables": {"id": checklist["node_id"]}}, repository=False)
        if response.get("errors") or not (node := response.get("data", {}).get("node")):
            raise ValueError("Could not verify the checklist editor")
        author = node["author"] or {}
        if author.get("__typename") != "Bot" or author.get("databaseId") != routing.GITHUB_ACTIONS_BOT_ID:
            raise ValueError("The checklist is not owned by github-actions[bot]")
        current = (node["lastEditedAt"], node["body"])
        if current != version:
            version = current
            try:
                authorize(api, node["editor"] or {})
                after = parse_selection(node["body"], template)
                sequence += 1
                record = apply_selection(record, selected, after, api.sha, sequence)
                selected, last_actor, edit_error = after, "@" + node["editor"]["login"], None
            except ValueError as error:
                edit_error = str(error)
        approvals = routing.current_approvals(pr, api.pages(f"/pulls/{NUMBER}/reviews"))
        ready, reason, _ = maintainer_approval.module_approval(api, pr, roster, skill, record, approvals)
        if edit_error:
            ready, reason = False, f"Edit not applied: {edit_error}\n\nLast accepted requirements: {reason}"
        state = "success" if ready else "pending"
        body = result_body(state, reason, selected, last_actor, expires, checklist["html_url"])
        if body != last_body:
            api.request("PATCH", f"/issues/comments/{api.result_id}", {"body": body})
            last_body = body
            print(json.dumps({"state": state, "actor": last_actor, "requirements": sorted(selected)}), flush=True)
        status = (state, "Trial: " + reason.replace("\n", " ")[:130])
        if status != last_status:
            publish(api, *status, run_url)
            last_status = status
        time.sleep(min(INTERVAL, max(0, deadline - time.monotonic())))


def main():
    if (os.environ["GITHUB_REPOSITORY"] != REPOSITORY or
            os.environ["GITHUB_REF"] != f"refs/heads/{BRANCH}" or os.environ["GITHUB_EVENT_NAME"] != "push"):
        raise ValueError("This experiment only runs on its dedicated branch")
    sha = os.environ["TRIAL_HEAD"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("A pinned trial commit is required")
    api = TrialAPI(os.environ["GITHUB_TOKEN"], sha)
    run_url = f"https://github.com/{REPOSITORY}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    if sys.argv[1:] == ["stop"]:
        stop(api, run_url)
    elif sys.argv[1:]:
        raise ValueError("Use no arguments or stop")
    else:
        run(api, run_url)


if __name__ == "__main__":
    main()
