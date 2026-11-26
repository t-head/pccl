"""Test the full M890P selection, isolated counters, and report reconciliation."""

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github/workflows/scripts"
sys.path.insert(0, str(SCRIPTS))
import pccl_m890p_coverage as coverage


class CoverageSelectionTest(unittest.TestCase):
    def test_builds_full_selection_without_changing_daily_commands_or_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, selected = root / "daily", root / "coverage"
            subprocess.run(["bash", ROOT / "tests/xtest/generate_m890p_daily.sh", "full", "a" * 40,
                            coverage.artifact_cache.PCCL_TESTS_REVISION, source],
                           stdout=subprocess.DEVNULL, check=True)
            with patch.dict(os.environ, {"GITHUB_OUTPUT": str(root / "outputs")}):
                coverage.prepare(source, selected)
            daily = {coverage.case_key(case): case for case in coverage.run_cases.load_inputs(
                source, "standard")["cases"] if case["suite_spec"]["nodes"] in (1, 2)}
            cases = {}
            groups = coverage.manifest(selected)
            for group in groups:
                rows = coverage.run_cases.load_inputs(selected / group["id"], "standard")["cases"]
                for case in rows:
                    key = coverage.case_key(case)
                    self.assertNotIn(key, cases)
                    cases[key] = case
                    self.assertEqual(case["suite_spec"]["nodes"], group["nodes"])
                    self.assertEqual(coverage.group_cases._rack_policy(case),
                                     (group["rack_policy"], group["rack_count"]))
            self.assertEqual(cases, daily)
            self.assertTrue(any(group["cross_rack"] for group in groups))
            self.assertEqual({group["nodes"] for group in groups}, {1, 2})
            matrix = json.loads(dict(line.split("=", 1) for line in
                                      (root / "outputs").read_text().splitlines())["groups"])["include"]
            self.assertEqual(sum(group["case_count"] for group in matrix), len(daily))


class CoverageEvidenceTest(unittest.TestCase):
    revision = "a" * 40
    digest = "b" * 64

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.runtime = self.root / "runtime"
        self.notes = self.runtime / "notes"
        (self.notes / "obj").mkdir(parents=True)
        (self.notes / "obj/init.gcno").write_bytes(b"notes")
        (self.runtime / "include").mkdir()
        self.selection = self.root / "selection"
        self.selection.mkdir()
        self.groups = []
        for index, nodes in enumerate((1, 2), 1):
            self.groups.append({"id": f"g{index:03d}", "case_count": 1, "nodes": nodes,
                                "platform": "m890p", "profile": "standard", "node_selector": coverage.SELECTOR,
                                "cross_rack": False, "rack_policy": "same", "rack_count": 1,
                                "case_keys": [["plan", "suite", f"case{index}"]]})
        coverage.write_json(self.selection / "groups.json", self.groups)
        self.key = "coverage-attempt"
        self.groups_sha = coverage.case_group_cache.tree_sha256(self.selection)
        self.env = {"PCCL_REVISION": self.revision, "RUNTIME_SHA": self.digest, "GROUPS_SHA": self.groups_sha}
        with patch.dict(os.environ, self.env):
            self.identity = coverage.build_identity()
        for tool in ("gcov-13", "gcov-tool-13"):
            (self.runtime / f"{tool}-version.txt").write_text("test version\n")

    def export(self, index):
        return self.root / f"{self.key}-shard-g{index:03d}"

    def evidence(self, index, status="PASSED", workers=None):
        group = self.groups[index - 1]
        directory = self.export(index)
        directory.mkdir()
        coverage.write_json(directory / "receipt.json", {"group_id": group["id"], "build": self.identity,
                                                          "selection_sha256": self.groups_sha})
        with (directory / "results.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["plan", "suite", "name", "status"])
            writer.writerow(group["case_keys"][0] + [status])
        for worker in range(workers if workers is not None else group["nodes"]):
            slot = directory / "profiles" / f"worker-{worker}" / "0" / "obj"
            slot.mkdir(parents=True)
            (slot / "init.gcda").write_bytes(b"counter")
        return directory

    def check_report(self, execution="success"):
        source = self.root / "source"
        source.mkdir()
        report = self.root / "report"
        metrics = {name: {"covered": 1, "total": 2} for name in ("lines", "functions", "branches")}

        def fake_merge(profiles, notes, output, logs):
            self.assertEqual(notes, self.notes)
            self.assertEqual(logs, report)
            if not profiles:
                raise ValueError("No valid GCOV profiles were collected")
            output.mkdir()

        with patch.dict(os.environ, self.env), patch.object(coverage, "SOURCE", source), \
                patch.object(coverage, "check_runtime"), patch.object(coverage, "version", return_value="test version"), \
                patch.object(coverage, "merge_profiles", side_effect=fake_merge) as merge, \
                patch.object(coverage, "collect", return_value=metrics):
            rc = coverage.report_coverage(self.runtime, self.selection, self.root, self.key, report, execution)
        return rc, coverage.read_json(report / "summary.json"), len(merge.call_args.args[0])

    def test_complete_requires_results_and_counters_from_all_workers(self):
        self.evidence(1)
        self.evidence(2)
        rc, summary, slots = self.check_report()
        self.assertEqual((rc, summary["status"], summary["counts"], slots), (0, "passed", {"PASSED": 2}, 3))

    def test_missing_worker_preserves_case_status_but_cannot_be_full(self):
        self.evidence(1)
        self.evidence(2, workers=1)
        rc, summary, slots = self.check_report()
        self.assertEqual((rc, summary["status"], summary["counts"], slots), (1, "partial", {"PASSED": 2}, 1))
        self.assertIn("Expected counters on 2 workers", summary["groups"][1]["error"])

    def test_failed_case_with_profiles_is_reported_but_not_a_pass(self):
        self.evidence(1)
        self.evidence(2, status="FAILED")
        rc, summary, slots = self.check_report(execution="failure")
        self.assertEqual((rc, summary["status"], summary["counts"], slots),
                         (1, "partial", {"PASSED": 1, "FAILED": 1}, 3))

    def test_missing_shard_is_not_treated_as_not_run_or_passed(self):
        self.evidence(1)
        rc, summary, slots = self.check_report(execution="failure")
        self.assertEqual((rc, summary["status"], summary["counts"], slots),
                         (1, "partial", {"PASSED": 1, "NO_RESULT": 1}, 1))

    def test_no_counters_does_not_create_a_report(self):
        rc, summary, slots = self.check_report()
        self.assertEqual((rc, summary["status"], summary["coverage"], slots), (1, "failed", None, 0))

    def test_stale_build_and_duplicate_case_results_are_rejected(self):
        archive = self.evidence(1)
        with self.assertRaisesRegex(ValueError, "does not match the selection and build"):
            coverage.result_counts(self.groups[0], archive, self.identity, "another selection")
        with (archive / "results.csv").open("a") as stream:
            stream.write("plan,suite,case1,PASSED\n")
        with self.assertRaisesRegex(ValueError, "results do not match"):
            coverage.result_counts(self.groups[0], archive, self.identity, self.groups_sha)

    def test_rejects_counter_without_matching_build_note_and_gcov_runtime_error(self):
        directory = self.evidence(1) / "profiles"
        slot = directory / "worker-0/0"
        (slot / "obj/foreign.gcda").write_bytes(b"unrelated")
        with self.assertRaisesRegex(ValueError, "does not match a build note"):
            coverage.group_profiles(directory, self.notes, 1)
        (slot / "obj/foreign.gcda").unlink()
        (slot / "gcov-errors.log").write_text("profile checksum mismatch")
        with self.assertRaisesRegex(ValueError, "GCOV runtime error"):
            coverage.group_profiles(directory, self.notes, 1)

    def test_merge_preserves_inputs_and_counter_inventory(self):
        profiles = []
        for index in range(2):
            directory = self.evidence(index + 1) / "profiles"
            profiles.extend(coverage.group_profiles(directory, self.notes, index + 1))
        report = self.root / "logs"
        report.mkdir()

        def fake_command(args, log, cwd, env, timeout):
            self.assertEqual(args[:2], ["gcov-tool-13", "merge"])
            shutil.copytree(args[2], args[5])
            shutil.copytree(args[3], args[5], dirs_exist_ok=True)
            log.write_text("gcov-tool merge\n")
            return 0

        build = self.root / "build"
        with patch.object(coverage, "command", side_effect=fake_command) as run:
            coverage.merge_profiles(profiles, self.notes, build, report)
        self.assertEqual((build / "obj/init.gcda").read_bytes(), b"counter")
        self.assertEqual((build / "obj/init.gcno").read_bytes(), b"notes")
        self.assertEqual(run.call_count, len(profiles) - 1)
        self.assertTrue(all((directory / "obj/init.gcda").is_file() for directory, _ in profiles))

    def test_rank_entrypoint_derives_run_scope_even_via_case_work_symlink(self):
        group = self.root / "groups/g001"
        bindir = group / "test-root/pccl-tests/build"
        bindir.mkdir(parents=True)
        (group / "test-root/coverage-runtime").symlink_to(self.runtime)
        (self.runtime / "benchmark").mkdir()
        (self.runtime / "lib").mkdir()
        library = self.runtime / "lib/libpccl.so.2"
        library.touch()
        (self.runtime / "lib/libpccl.so").symlink_to(library.name)
        test_bin = self.runtime / "benchmark/all_reduce_perf"
        test_bin.write_text('#!/bin/bash\nprintf "%s\\n" "$GCOV_PREFIX" "$GCOV_PREFIX_STRIP" "$1"\n')
        test_bin.chmod(0o755)
        tools = self.root / "tools"
        tools.mkdir()
        ldd = tools / "ldd"
        ldd.write_text(f"#!/bin/bash\necho 'libpccl.so.2 => {library}'\n")
        flock = tools / "flock"
        flock.write_text("#!/bin/bash\nexit 0\n")
        for tool in (ldd, flock):
            tool.chmod(0o755)
        (bindir / test_bin.name).symlink_to(SCRIPTS / "pccl_m890p_coverage_rank.sh")
        work = self.root / "work"
        work.mkdir()
        (work / "build").symlink_to(bindir)
        env = {**os.environ, "PATH": f"{tools}:{os.environ['PATH']}", "TMPDIR": str(self.root),
               "NODE_NAME": "test-worker", "OMPI_COMM_WORLD_LOCAL_RANK": "3"}
        result = subprocess.check_output(["bash", "./build/all_reduce_perf", "-t"], cwd=work, env=env, text=True)
        expected = group.resolve() / "export/profiles/test-worker/3"
        self.assertEqual(result.splitlines(), [str(expected), "4", "-t"])
        self.assertTrue((expected / ".loaded-all_reduce_perf").is_file())


if __name__ == "__main__":
    unittest.main()
