# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
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

    def request(self, method, path, data=None):
        if method == "GET":
            self.reads += 1
            if self.change_after == self.reads:
                self.pr["head"]["sha"] = "c" * 40
            return copy.deepcopy(self.pr)
        self.writes.append((method, path, data))
        return {"requested_reviewers": data["reviewers"]}

    def pages(self, path):
        return copy.deepcopy(self.reviews if path.endswith("/reviews") else self.files)


class ReviewerRoutingTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"REVIEW_BASE_BRANCH": "main"})
        environment.start()
        self.addCleanup(environment.stop)
        self.api = FakeGitHub()
        self.roster, self.skill = routing.load_config()
        self.proposal = {"modules": ["workflow"], "areas": ["workflow-review"],
                         "reason": "Changes reviewer selection."}

    def recommend(self):
        pr, files = routing.snapshot(self.api, 19)
        return routing.classify(pr, files, self.roster, self.skill, lambda _: json.dumps(self.proposal))

    def test_classification_precedes_invitation_and_routes_to_backup(self):
        recommendation = self.recommend()
        self.assertEqual(self.api.writes, [])
        summary = routing.invite(self.api, 19, recommendation, self.roster, self.skill)
        self.assertEqual(self.api.writes, [
            ("POST", "/pulls/19/requested_reviewers", {"reviewers": ["wensuyang"]})])
        self.assertIn("wensuyang", summary)
        self.assertNotIn("Module Owner Approval", summary)

    def test_existing_invitation_is_retained_without_another_request(self):
        self.api.pr["requested_reviewers"] = [{"login": "WenSuyang"}]
        routing.invite(self.api, 19, self.recommend(), self.roster, self.skill)
        self.assertEqual(self.api.writes, [])

    def test_current_head_approval_is_not_invited_again(self):
        self.api.reviews = [{"id": 1, "state": "APPROVED", "user": {"login": "wensuyang"},
                             "commit_id": self.api.pr["head"]["sha"]}]
        routing.invite(self.api, 19, self.recommend(), self.roster, self.skill)
        self.assertEqual(self.api.writes, [])

    def test_old_or_dismissed_approval_can_receive_an_invitation(self):
        for state, sha in [("APPROVED", "old"), ("DISMISSED", "a" * 40)]:
            with self.subTest(state=state):
                review = {"id": 2, "state": state, "user": {"login": "wensuyang"}, "commit_id": sha}
                self.assertEqual(routing.select_reviewers(self.roster, self.proposal,
                                                         self.api.pr, [review]), ["wensuyang"])

    def test_primary_and_specialists_are_selected_from_the_roster(self):
        self.api.pr["user"]["login"] = "external-contributor"
        proposal = {"modules": ["kernels"], "areas": ["kernels-2"], "reason": "Changes RedOp."}
        self.assertEqual(routing.select_reviewers(self.roster, proposal, self.api.pr, []),
                         ["ck-milo", "wensuyang"])

    def test_unknown_modules_areas_and_model_accounts_are_rejected(self):
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
        changed["workflow"]["backup"] = "someone-else"
        for roster, skill in [(changed, self.skill), (self.roster, self.skill + "updated")]:
            with self.assertRaisesRegex(ValueError, "changed"):
                routing.invite(self.api, 19, original, roster, skill)
        self.assertEqual(self.api.writes, [])

    def test_push_during_invitation_aborts_without_writing(self):
        recommendation = self.recommend()
        self.api.change_after = self.api.reads + 2
        with self.assertRaisesRegex(ValueError, "changed before invitation"):
            routing.invite(self.api, 19, recommendation, self.roster, self.skill)
        self.assertEqual(self.api.writes, [])

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
        self.assertEqual(len(self.api.writes), 1)
        self.api.pr["base"]["ref"] = "unrelated"
        with self.assertRaisesRegex(ValueError, "targeting"):
            self.recommend()

    def test_reusable_workflow_checks_the_configured_target(self):
        self.api.pr["base"]["ref"] = "integration"
        with patch.dict(os.environ, {"REVIEW_BASE_BRANCH": "integration"}):
            routing.invite(self.api, 19, self.recommend(), self.roster, self.skill)
        self.assertEqual(self.api.writes[0][2], {"reviewers": ["wensuyang"]})

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
