"""Contract tests for the two-node experiment; no devices or network required."""

import csv
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".github" / "workflows" / "scripts"))
import pccl_m890p_coverage as coverage


class M890PCoverageTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.ranks = root / "ranks"
        self.notes = root / "notes"
        (self.notes / "obj").mkdir(parents=True)
        (self.notes / "obj/init.gcno").write_bytes(b"note")
        for rank in range(16):
            path = self.ranks / f"rank-{rank}"
            (path / "obj").mkdir(parents=True)
            (path / "obj/init.gcda").write_bytes(b"counter")
            (path / "rank.tsv").write_text(f"{rank}\t{rank % 8}\tworker-{rank // 8}\n")

    def test_checked_in_case_retains_parameters_and_adds_correctness_validation(self):
        directory = ROOT / "tests/xtest/minimal-csv"
        inputs = coverage.select_case(directory)
        self.assertEqual(len(inputs["cases"]), 1)
        case = inputs["cases"][0]
        self.assertEqual((case["suite_spec"]["nodes"], case["suite_spec"]["cards_per_node"]), (2, 8))
        with (directory / "standard.csv").open(newline="") as stream:
            original = next(row for row in csv.DictReader(stream) if row["name"] == case["name"])
        self.assertEqual(case["cmd"], original["cmd"])
        self.assertEqual(case["check"], "validation_summary")

    def test_accepts_two_workers_with_complete_profiles_without_transport_logs(self):
        result = coverage.rank_profiles(self.ranks, self.notes)
        self.assertEqual(result["ranks_per_worker"], {"worker-0": 8, "worker-1": 8})
        self.assertEqual(result["ranks"], 16)
        self.assertEqual(result["counter_files_by_rank"], dict.fromkeys(range(16), 1))

    def test_rejects_a_missing_rank(self):
        shutil.rmtree(self.ranks / "rank-15")
        with self.assertRaisesRegex(ValueError, "missing=.*rank-15"):
            coverage.rank_profiles(self.ranks, self.notes)

    def test_rejects_gcov_runtime_errors(self):
        (self.ranks / "rank-4/gcov-errors.log").write_text("profile checksum mismatch")
        with self.assertRaisesRegex(ValueError, "GCOV runtime errors for rank 4"):
            coverage.rank_profiles(self.ranks, self.notes)

    def test_rejects_unmatched_counter(self):
        (self.ranks / "rank-4/obj/foreign.gcda").write_bytes(b"foreign")
        with self.assertRaisesRegex(ValueError, "does not match a build note"):
            coverage.rank_profiles(self.ranks, self.notes)

    def test_transport_logs_do_not_block_coverage(self):
        expected = coverage.rank_profiles(self.ranks, self.notes)
        for text in ("Using network IB\nMNNVL 1 cliqueId 7ffe cliqueSize 16\n",
                     "Channel 00/0 : 13[4000] -> 5[5000] [send] via NET/Socket/0\n"):
            with self.subTest(text=text):
                (self.ranks / "rank-13/pccl.log").write_text(text)
                self.assertEqual(coverage.rank_profiles(self.ranks, self.notes), expected)

    def test_report_restores_build_headers_and_excludes_absolute_sdk_paths(self):
        runtime = self.notes.parent
        source = runtime / "source"
        build = source / "build"
        report = runtime / "report"
        generated = Path("pccl_device/impl/core__funcs.h")
        header = runtime / "include" / generated
        header.parent.mkdir(parents=True)
        header.write_text("generated include from this build\n")

        def fake_merge(ranks, notes, output, logs):
            self.assertEqual((ranks, notes, output, logs), (self.ranks, self.notes, build, report))
            build.mkdir(parents=True)

        def fake_collect(src, output, logs, gcov):
            self.assertEqual((src, output, logs), (source, build, report))
            self.assertEqual(gcov, "gcov-13 --relative-only")
            self.assertEqual((output / "include" / generated).read_bytes(), header.read_bytes())
            return {"lines": {"covered": 1, "total": 2}}

        with patch.object(coverage, "merge_profiles", side_effect=fake_merge), \
                patch.object(coverage, "collect", side_effect=fake_collect):
            self.assertEqual(coverage.collect_rank_coverage(self.ranks, runtime, source, report),
                             {"lines": {"covered": 1, "total": 2}})

    def test_report_rejects_missing_build_headers(self):
        runtime = self.notes.parent
        with patch.object(coverage, "merge_profiles") as merge:
            with self.assertRaisesRegex(ValueError, "Missing build headers"):
                coverage.collect_rank_coverage(self.ranks, runtime, runtime / "source", runtime / "report")
            merge.assert_not_called()

    def test_rank_placement_must_be_eight_per_host(self):
        (self.ranks / "rank-15/rank.tsv").write_text("15\t7\tworker-0\n")
        with self.assertRaisesRegex(ValueError, "Unexpected MPI rank placement"):
            coverage.rank_profiles(self.ranks, self.notes)


if __name__ == "__main__":
    unittest.main()
