# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
import base64
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("routing", ROOT / ".github/scripts/request_reviewers.py")
routing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(routing)


class FakeGitHub:
    repository = "t-head/pccl"

    def __init__(self):
        self.pr = {"number": 19, "state": "open", "draft": False, "changed_files": 1,
                   "title": "Simplify reviewer routing", "body": "Read the complete diff.",
                   "head": {"sha": "a" * 40}, "user": {"login": "honganlxh"},
                   "base": {"ref": "main", "sha": "b" * 40,
                            "repo": {"full_name": self.repository}}, "requested_reviewers": []}
        self.files = [{"filename": ".github/workflows/reviewer-routing.yml", "status": "modified",
                       "additions": 1, "deletions": 1, "patch": "@@ -1 +1 @@\n-old\n+new"}]
        self.reviews, self.writes = [], []
        self.reads, self.change_after = 0, None
        self.state = {"version": 1, "last_reviewers": {}, "pull_requests": {}}
        self.state_version, self.conflict, self.fail_invitation = 1, None, False
        self.branch_exists, self.file_exists = True, True

    @property
    def invitations(self):
        return [write for write in self.writes if write[1].endswith("/requested_reviewers")]

    def request(self, method, path, data=None):
        if method == "GET" and path.startswith("/pulls/"):
            self.reads += 1
            if self.change_after == self.reads:
                self.pr["head"]["sha"] = "c" * 40
            return copy.deepcopy(self.pr)
        if method == "GET" and path.startswith("/contents/"):
            if not self.branch_exists or not self.file_exists:
                raise routing.GitHubError(method, path, 404)
            return {"sha": str(self.state_version), "encoding": "base64",
                    "content": base64.b64encode(json.dumps(self.state).encode()).decode()}
        if method == "GET" and path.startswith("/git/ref/"):
            if not self.branch_exists:
                raise routing.GitHubError(method, path, 404)
            return {"object": {"sha": "existing-state-commit"}}
        if method == "PUT":
            if self.conflict:
                conflict, self.conflict = self.conflict, None
                conflict(self)
            if data["sha"] != str(self.state_version):
                raise routing.GitHubError(method, path, 409)
            self.state = json.loads(base64.b64decode(data["content"]))
            self.state_version += 1
        if method == "POST" and path == "/git/trees":
            self.state = json.loads(data["tree"][0]["content"])
        if method == "POST" and path == "/git/refs":
            self.branch_exists, self.file_exists = True, True
        if method == "POST" and path.endswith("/requested_reviewers"):
            if self.fail_invitation:
                raise routing.GitHubError(method, path, 503)
            self.pr["requested_reviewers"].extend({"login": login} for login in data["reviewers"])
        self.writes.append((method, path, data))
        return {"sha": "created-object"}

    def pages(self, path):
        return copy.deepcopy(self.reviews if path.endswith("/reviews") else self.files)


class ReviewerRoutingTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"REVIEW_BASE_BRANCH": "main",
                                              "REVIEW_STATE_BRANCH": "automation/reviewer-rotation"})
        environment.start()
        self.addCleanup(environment.stop)
        self.api = FakeGitHub()
        self.roster, self.skill = routing.load_config()
        self.proposal = {"modules": ["workflow"],
                         "reason": "Changes reviewer selection."}

    def recommend(self):
        pr, files = routing.snapshot(self.api, 19)
        return routing.classify(pr, files, self.roster, self.skill, lambda _: json.dumps(self.proposal))

    def test_classification_precedes_invitation_and_routes_author_owned_module_to_lead(self):
        recommendation = self.recommend()
        self.assertEqual(self.api.writes, [])
        summary = routing.invite(self.api, 19, recommendation, self.roster, self.skill)
        self.assertEqual(self.api.invitations, [
            ("POST", "/pulls/19/requested_reviewers", {"reviewers": ["lishusen2026", "wensuyang"]})])
        self.assertEqual(self.api.state["pull_requests"]["19"]["modules"]["workflow"],
                         {"maintainer": "lishusen2026", "reviewer": "wensuyang"})
        self.assertIn("wensuyang", summary)
        self.assertNotIn("Module Owner Approval", summary)

    def test_existing_invitation_is_retained_without_another_request(self):
        self.api.pr["requested_reviewers"] = [{"login": "WenSuyang"}, {"login": "lishusen2026"}]
        routing.invite(self.api, 19, self.recommend(), self.roster, self.skill)
        self.assertEqual(self.api.invitations, [])

    def test_current_head_approval_is_not_invited_again(self):
        self.api.reviews = [{"id": index, "state": "APPROVED", "user": {"login": login},
                             "commit_id": self.api.pr["head"]["sha"]}
                            for index, login in enumerate(["wensuyang", "lishusen2026"])]
        routing.invite(self.api, 19, self.recommend(), self.roster, self.skill)
        self.assertEqual(self.api.invitations, [])

    def test_old_or_dismissed_approval_can_receive_an_invitation(self):
        for state, sha in [("APPROVED", "old"), ("DISMISSED", "a" * 40)]:
            with self.subTest(state=state):
                review = {"id": 2, "state": state, "user": {"login": "wensuyang"}, "commit_id": sha}
                assignments = {"workflow": {"reviewer": "wensuyang", "maintainer": "lishusen2026"}}
                self.assertIn("wensuyang", routing.pending_invitations(assignments, self.api.pr, [review]))

    def test_one_reviewer_and_maintainer_are_selected_for_a_module(self):
        self.api.pr["user"]["login"] = "external-contributor"
        proposal = {"modules": ["kernels"], "reason": "Changes RedOp."}
        _, assignments = routing.plan_assignments(self.roster, proposal, self.api.pr, self.api.state)
        self.assertEqual(assignments, {"kernels": {"reviewer": "xiangzhixyc", "maintainer": "honganlxh"}})

    def test_round_robin_advances_between_prs_in_roster_order(self):
        self.api.pr["user"]["login"] = "external-contributor"
        proposal = {"modules": ["build"], "reason": "Changes compilation."}
        state, selected = self.api.state, []
        for number in range(20, 24):
            self.api.pr["number"] = number
            state, assignments = routing.plan_assignments(self.roster, proposal, self.api.pr, state)
            selected.append(assignments["build"]["reviewer"])
        self.assertEqual(selected, ["xiangzhixyc", "ck-milo", "wensuyang", "xiangzhixyc"])

    def test_reruns_and_new_commits_reuse_the_pr_assignment(self):
        routing.invite(self.api, 19, self.recommend(), self.roster, self.skill)
        state, writes = copy.deepcopy(self.api.state), copy.deepcopy(self.api.writes)
        for head in ["a" * 40, "c" * 40]:
            self.api.pr["head"]["sha"] = head
            routing.invite(self.api, 19, self.recommend(), self.roster, self.skill)
        self.assertEqual(self.api.state, state)
        self.assertEqual(self.api.writes, writes)

    def test_adding_a_module_preserves_existing_assignments(self):
        self.api.pr["user"]["login"] = "external-contributor"
        state, original = routing.plan_assignments(
            self.roster, {"modules": ["build"]}, self.api.pr, self.api.state)
        updated, assignments = routing.plan_assignments(
            self.roster, {"modules": ["build", "enqueue"]}, self.api.pr, state)
        self.assertEqual(assignments["build"], original["build"])
        self.assertEqual(updated["last_reviewers"]["build"], state["last_reviewers"]["build"])
        self.assertEqual(assignments["enqueue"], {"reviewer": "ck-milo", "maintainer": "ctt284373"})

    def test_modules_prefer_distinct_reviewers_and_share_a_maintainer_request(self):
        self.api.pr["user"]["login"] = "external-contributor"
        proposal = {"modules": ["device-topology", "graph-search"]}
        _, assignments = routing.plan_assignments(self.roster, proposal, self.api.pr, self.api.state)
        self.assertEqual(assignments["device-topology"]["reviewer"], "xiangzhixyc")
        self.assertEqual(assignments["graph-search"]["reviewer"], "otoriemu000")
        self.assertEqual(routing.pending_invitations(assignments, self.api.pr, []),
                         ["honganlxh", "otoriemu000", "xiangzhixyc"])

    def test_a_shared_reviewer_and_maintainer_receive_one_request(self):
        self.api.pr["user"]["login"] = "external-contributor"
        _, assignments = routing.plan_assignments(
            self.roster, {"modules": ["ci"]}, self.api.pr, self.api.state)
        self.assertEqual(assignments["ci"], {"reviewer": "xuyinxin-xyx", "maintainer": "xuyinxin-xyx"})
        self.assertEqual(routing.pending_invitations(assignments, self.api.pr, []), ["xuyinxin-xyx"])

    def test_author_is_skipped_in_the_rotation(self):
        self.api.pr["user"]["login"] = "xiangzhixyc"
        _, assignments = routing.plan_assignments(
            self.roster, {"modules": ["build"]}, self.api.pr, self.api.state)
        self.assertEqual(assignments["build"]["reviewer"], "ck-milo")

    def test_unknown_accounts_are_never_selected_and_missing_roles_are_reported(self):
        self.api.pr["user"]["login"] = "QingxunLy"
        self.proposal = {"modules": ["ras"], "reason": "Changes hang monitoring."}
        summary = routing.invite(self.api, 19, self.recommend(), self.roster, self.skill)
        self.assertEqual(self.api.invitations[0][2], {"reviewers": ["ck-milo"]})
        self.assertIn("Manual assignment needed", summary)
        self.assertIn("舒磊", summary)
        self.assertIsNone(self.api.state["pull_requests"]["19"]["modules"]["ras"]["reviewer"])

    def test_lead_maintainer_cannot_be_assigned_to_their_own_pr(self):
        self.api.pr["user"]["login"] = "lishusen2026"
        _, assignments = routing.plan_assignments(
            self.roster, {"modules": ["other"]}, self.api.pr, self.api.state)
        self.assertEqual(assignments["other"], {"reviewer": "honganlxh", "maintainer": None})

    def test_concurrent_pr_updates_retry_without_losing_a_turn(self):
        self.api.pr["user"]["login"] = "external-contributor"
        proposal = {"modules": ["build"]}

        def another_pr(api):
            previous = dict(api.pr, number=18)
            api.state, _ = routing.plan_assignments(self.roster, proposal, previous, api.state)
            api.state_version += 1

        self.api.conflict = another_pr
        with patch.object(routing.time, "sleep"):
            assignments = routing.reserve_assignments(self.api, self.roster, proposal, self.api.pr)
        self.assertEqual(self.api.state["pull_requests"]["18"]["modules"]["build"]["reviewer"], "xiangzhixyc")
        self.assertEqual(assignments["build"]["reviewer"], "ck-milo")
        self.assertEqual(self.api.state["last_reviewers"]["build"], "ck-milo")

    def test_failed_invitation_preserves_the_assignment_for_retry(self):
        self.api.fail_invitation = True
        with self.assertRaises(routing.GitHubError):
            routing.invite(self.api, 19, self.recommend(), self.roster, self.skill)
        saved = copy.deepcopy(self.api.state)
        self.api.fail_invitation = False
        routing.invite(self.api, 19, self.recommend(), self.roster, self.skill)
        self.assertEqual(self.api.state, saved)
        self.assertEqual(len(self.api.invitations), 1)

    def test_state_is_initialized_on_a_dedicated_orphan_branch(self):
        self.api.branch_exists = False
        routing.invite(self.api, 19, self.recommend(), self.roster, self.skill)
        commits = [write for write in self.api.writes if write[1] == "/git/commits"]
        refs = [write for write in self.api.writes if write[1] == "/git/refs"]
        self.assertEqual(commits[0][2]["parents"], [])
        self.assertEqual(refs[0][2]["ref"], "refs/heads/automation/reviewer-rotation")

    def test_missing_or_invalid_state_is_not_silently_reset(self):
        self.api.file_exists = False
        with self.assertRaisesRegex(ValueError, "state file is missing"):
            routing.invite(self.api, 19, self.recommend(), self.roster, self.skill)
        self.api.file_exists = True
        self.api.state = {"version": 999}
        with self.assertRaisesRegex(ValueError, "Invalid rotation state"):
            routing.invite(self.api, 19, self.recommend(), self.roster, self.skill)
        self.assertEqual(self.api.writes, [])

    def test_state_cannot_be_written_to_the_source_branch(self):
        with patch.dict(os.environ, {"REVIEW_STATE_BRANCH": "main"}), \
                self.assertRaisesRegex(ValueError, "dedicated"):
            routing.invite(self.api, 19, self.recommend(), self.roster, self.skill)
        self.assertEqual(self.api.writes, [])

    def test_stored_accounts_removed_from_roster_are_not_invited(self):
        routing.invite(self.api, 19, self.recommend(), self.roster, self.skill)
        self.roster["people"]["suyang"]["github"] = "replacement"
        with self.assertRaisesRegex(ValueError, "Roster changed"):
            routing.invite(self.api, 19, self.recommend(), self.roster, self.skill)
        self.assertEqual(len(self.api.invitations), 1)

    def test_unknown_modules_extra_fields_and_model_accounts_are_rejected(self):
        invalid = [dict(self.proposal, modules=["unknown"]),
                   dict(self.proposal, areas=["device-gin"]),
                   dict(self.proposal, reviewers=["attacker"]),
                   dict(self.proposal, modules=[]), dict(self.proposal, modules=[{}])]
        for proposal in invalid:
            with self.subTest(proposal=proposal), self.assertRaises(ValueError):
                routing.validate(proposal, self.roster)

    def test_partial_and_binary_diffs_do_not_call_the_model(self):
        for file in [{**self.api.files[0], "additions": 2},
                     {**self.api.files[0], "patch": "", "additions": 0, "deletions": 0}]:
            with self.subTest(file=file):
                self.api.files = [file]
                with self.assertRaisesRegex(ValueError, "Incomplete or binary"):
                    self.recommend()
        self.api.pr["changed_files"] = 2
        with self.assertRaisesRegex(ValueError, "Incomplete PR file list"):
            self.recommend()

    def test_model_failure_or_unparseable_output_does_not_invite(self):
        def failed(_):
            raise RuntimeError("Unavailable")
        for ask in [failed, lambda _: "not JSON"]:
            with self.assertRaises((RuntimeError, ValueError)):
                routing.classify(self.api.pr, self.api.files, self.roster, self.skill, ask)
        self.assertEqual(self.api.writes, [])

    def test_changed_pr_and_configuration_invalidate_the_recommendation(self):
        original = self.recommend()
        for key in ["title", "body"]:
            with self.subTest(key=key):
                saved = self.api.pr[key]
                self.api.pr[key] = "edited"
                with self.assertRaisesRegex(ValueError, "changed"):
                    routing.invite(self.api, 19, original, self.roster, self.skill)
                self.api.pr[key] = saved
        changed = copy.deepcopy(self.roster)
        changed["people"]["suyang"]["github"] = "someone-else"
        for roster, skill in [(changed, self.skill), (self.roster, self.skill + "updated")]:
            with self.assertRaisesRegex(ValueError, "changed"):
                routing.invite(self.api, 19, original, roster, skill)
        self.assertEqual(self.api.writes, [])

    def test_push_during_invitation_aborts_without_writing(self):
        recommendation = self.recommend()
        self.api.change_after = self.api.reads + 2
        with self.assertRaisesRegex(ValueError, "changed before invitation"):
            routing.invite(self.api, 19, recommendation, self.roster, self.skill)
        self.assertEqual(self.api.invitations, [])

    def test_closed_or_draft_pr_does_not_invite(self):
        recommendation = self.recommend()
        for state, draft in [("closed", False), ("open", True)]:
            self.api.pr.update(state=state, draft=draft)
            self.assertIn("no invitation", routing.invite(self.api, 19, recommendation,
                                                         self.roster, self.skill))
        self.assertEqual(self.api.writes, [])

    def test_fork_head_is_supported_but_other_base_is_not(self):
        self.api.pr["head"]["repo"] = {"full_name": "external/pccl"}
        routing.invite(self.api, 19, self.recommend(), self.roster, self.skill)
        self.assertEqual(len(self.api.invitations), 1)
        self.api.pr["base"]["ref"] = "unrelated"
        with self.assertRaisesRegex(ValueError, "targeting"):
            self.recommend()

    def test_reusable_workflow_checks_the_configured_target(self):
        self.api.pr["base"]["ref"] = "integration"
        with patch.dict(os.environ, {"REVIEW_BASE_BRANCH": "integration"}):
            routing.invite(self.api, 19, self.recommend(), self.roster, self.skill)
        self.assertEqual(self.api.invitations[0][2], {"reviewers": ["lishusen2026", "wensuyang"]})

    def test_summary_uses_current_pr_and_escapes_model_markup(self):
        self.api.pr["number"] = 123
        proposal = dict(self.proposal, reason="Change <script>bad</script>\n[link](example)")
        summary = routing.classification_summary(self.api.pr, proposal)
        self.assertIn("PR #123", summary)
        self.assertIn(self.api.pr["head"]["sha"], summary)
        self.assertIn("&lt;script&gt;", summary)
        self.assertNotIn("<script>", summary)
        self.assertNotIn("PR #19", summary)

    def test_prompt_uses_skill_and_catalog_without_person_accounts(self):
        prompts = []
        routing.classify(self.api.pr, self.api.files, self.roster, self.skill,
                         lambda prompt: prompts.append(prompt) or json.dumps(self.proposal))
        self.assertIn(self.skill, prompts[0])
        self.assertIn(self.api.files[0]["patch"].replace("\n", "\\n"), prompts[0])
        self.assertNotIn("wensuyang", prompts[0])
        self.assertNotIn("honganlxh", prompts[0])

    def test_cli_handles_skill_frontmatter_and_structured_response(self):
        with tempfile.TemporaryDirectory() as directory:
            cli = Path(directory) / "copilot"
            cli.write_text(f"#!{sys.executable}\n" + '''import json, sys
assert any(arg.startswith('--prompt=---') for arg in sys.argv)
assert '--available-tools=' in sys.argv
assert '--no-custom-instructions' in sys.argv
assert '--disable-builtin-mcps' in sys.argv
print('{"type":"user.message","data":{"content":"invalid "echo""}}')
print(json.dumps({'type': 'assistant.message', 'data': {'content': '{"modules":["workflow"]}'}}))
''')
            cli.chmod(0o755)
            with patch.dict(os.environ, {"COPILOT_CLI": str(cli), "GITHUB_TOKEN": "fixture"}):
                self.assertEqual(json.loads(routing.ask_copilot('---\nname: example\n---')),
                                 {"modules": ["workflow"]})

    def test_catalog_timeout_retries_with_a_shared_five_minute_budget(self):
        timed_out = subprocess.CompletedProcess([], 1, "", "Model catalog request timed out after 30000ms")
        success = subprocess.CompletedProcess([], 0, json.dumps({
            "type": "assistant.message", "data": {"content": json.dumps(self.proposal)}}), "")
        with patch.dict(os.environ, {"COPILOT_CLI": "fixture", "GITHUB_TOKEN": "fixture"}), \
                patch.object(routing.subprocess, "run", side_effect=[timed_out, success]) as run, \
                patch.object(routing.time, "monotonic", side_effect=[0, 0, 30, 35]), \
                patch.object(routing.time, "sleep") as sleep, patch("sys.stderr"):
            self.assertEqual(json.loads(routing.ask_copilot("classify")), self.proposal)
            self.assertEqual([call.kwargs["timeout"] for call in run.call_args_list], [300, 265])
            sleep.assert_called_once_with(5)

    def test_other_cli_errors_are_not_retried_and_catalog_retries_are_bounded(self):
        for error, attempts in [("Permission denied", 1), ("Model catalog request timed out", 3)]:
            failed = subprocess.CompletedProcess([], 1, "", error)
            with self.subTest(error=error), \
                    patch.dict(os.environ, {"COPILOT_CLI": "fixture", "GITHUB_TOKEN": "fixture"}), \
                    patch.object(routing.subprocess, "run", return_value=failed) as run, \
                    patch.object(routing.time, "sleep"), patch("sys.stderr"), self.assertRaises(RuntimeError):
                routing.ask_copilot("classify")
            self.assertEqual(run.call_count, attempts)


if __name__ == "__main__":
    unittest.main()
