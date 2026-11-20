import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sanity_rounds
import wait_for_pass


def passed_summary():
    return {"requested_runs": 100, "attempted_runs": 100, "completed_runs": 100,
            "passed": 100, "failed": 0, "complete": True, "interrupted": False,
            "artifact_changed": False, "pccl_revision": "a" * 40, "pccl_tests_revision": "b" * 40,
            "failed_iterations": [], "target_hggc_failure_iterations": [],
            "hg_mem_map_invalid_argument_iterations": [], "timeout_iterations": [],
            "runs": [{"iteration": n, "status": "PASSED", "return_code": 0,
                      "fatal_signatures": [], "target_hggc_failure": False} for n in range(1, 101)],
            "artifacts": {name: {"sha256": "c" * 64, "sha256_after": "c" * 64}
                          for name in ("all_reduce_perf", "libpccl")}}


class FollowupTests(unittest.TestCase):
    def test_exact_100_passes_allow_followup(self):
        wait_for_pass.verify_summary(passed_summary(), "a" * 40, "b" * 40)

    def test_failed_partial_changed_or_wrong_version_deny_followup(self):
        for key, value in [("failed", 1), ("completed_runs", 99), ("complete", False),
                           ("artifact_changed", True), ("pccl_revision", "d" * 40),
                           ("failed_iterations", [3]), ("passed", True)]:
            with self.subTest(key=key):
                summary = passed_summary()
                summary[key] = value
                with self.assertRaises(ValueError):
                    wait_for_pass.verify_summary(summary, "a" * 40, "b" * 40)

    def test_aggregate_pass_cannot_hide_failed_case_or_changed_artifact(self):
        summary = passed_summary()
        summary["runs"][42]["return_code"] = 3
        with self.assertRaises(ValueError):
            wait_for_pass.verify_summary(summary, "a" * 40, "b" * 40)
        summary = passed_summary()
        summary["artifacts"]["libpccl"]["sha256_after"] = "d" * 64
        with self.assertRaises(ValueError):
            wait_for_pass.verify_summary(summary, "a" * 40, "b" * 40)

    def test_wait_then_allow_only_completed_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text(json.dumps(passed_summary()))
            states = iter([{"head_sha": "e" * 40, "run_attempt": 1, "status": "in_progress"},
                           {"head_sha": "e" * 40, "run_attempt": 1, "status": "completed", "conclusion": "success"}])
            sleeps = []
            with patch("builtins.print"):
                result = wait_for_pass.wait(1, "e" * 40, root / "summary.json", "a" * 40, "b" * 40,
                                           100, 1, root / "gate.json", get_state=lambda _: next(states),
                                           sleep=sleeps.append)
            self.assertTrue(result["should_run"])
            self.assertEqual(sleeps, [1])

    def test_failed_predecessor_never_reads_pass_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = {"head_sha": "e" * 40, "run_attempt": 1, "status": "completed", "conclusion": "failure"}
            result = wait_for_pass.wait(1, "e" * 40, root / "missing", "a" * 40, "b" * 40,
                                       10, 1, root / "gate.json", get_state=lambda _: state)
            self.assertFalse(result["should_run"])

    def test_round_summary_is_100_rounds_not_100_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            records = [{"round": n, "profiles": {p: [{"status": "PASSED"} for _ in range(count)]
                        for p, count in sanity_rounds.EXPECTED.items()}} for n in range(1, 101)]
            result = sanity_rounds.summarize(output, records, 100, {})
            self.assertTrue(result["complete"])
            self.assertEqual(result["expected_case_executions"], 2600)
            self.assertEqual(result["passed_cases"], 2600)
            records[0]["profiles"]["standard"][0]["status"] = "FAILED"
            result = sanity_rounds.summarize(output, records, 100, {})
            self.assertEqual(result["failed_rounds"], [1])
            self.assertEqual(result["failed_cases"], 1)
            self.assertTrue(result["complete"])

    def test_100_complete_rounds_continue_after_case_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for profile in sanity_rounds.EXPECTED:
                (root / f"output_{profile}.build.json").write_text(json.dumps({
                    "pccl_revision": "a" * 40, "artifact_sha256": "c" * 64}))
            calls = []

            def selection(path, profile, nodes):
                return {"profile": profile, "cases": [{"suite_spec": {"cards_per_node": 2}}
                        for _ in range(sanity_rounds.EXPECTED[profile])]}

            def run_all(inputs, source, binary, target, **kwargs):
                target.mkdir(parents=True)
                calls.append(inputs["profile"])
                rows = [{"status": "PASSED", "plan": "plan", "suite": "suite",
                         "name": f"case-{n}", "raw_log": ""} for n in range(len(inputs["cases"]))]
                if len(calls) == 5:
                    rows[0]["status"] = "FAILED"
                return rows

            executor = SimpleNamespace(load_inputs=selection, run_all=run_all,
                _verified_build_identity=lambda *args: {}, _write_summary=lambda *args: None)
            snapshot = {"return_code": 0, "text": "", "error": ""}
            with patch.object(sanity_rounds, "load_executor", return_value=executor), \
                    patch.object(sanity_rounds, "artifact_hashes", return_value={}), \
                    patch.object(sanity_rounds, "sha256", return_value="c" * 64), \
                    patch.object(sanity_rounds, "kernel_snapshot", return_value=snapshot), \
                    patch.dict(sanity_rounds.os.environ, {"PCCL_REVISION": "a" * 40}), \
                    patch("builtins.print"):
                code = sanity_rounds.run(root, root / "results", 100, "eth0")
            result = json.loads((root / "results/summary.json").read_text())
            self.assertEqual(code, 1)
            self.assertEqual(calls, ["standard", "notsm"] * 100)
            self.assertEqual(result["completed_rounds"], 100)
            self.assertEqual(result["executed_cases"], 2600)
            self.assertEqual(result["failed_cases"], 1)
            self.assertEqual(result["failed_rounds"], [3])


if __name__ == "__main__":
    unittest.main()
