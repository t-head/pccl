# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
from pathlib import Path
import tempfile
import unittest

import daily_reconcile
import run_cases


PERFORMANCE_LOG = """# size count type redop root avg algbw busbw #wrong avg algbw busbw #wrong
# (B) (elements) (us) (GB/s) (GB/s) (us) (GB/s) (GB/s)
8 2 float sum -1 1.0 1.0 1.0 0 1.0 1.0 1.0 0
# Out of bounds values : 0 OK
"""


class ReconcileResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input = self.root / "input"
        (self.input / "envs").mkdir(parents=True)
        self.results = self.root / "results"
        self.results.mkdir()
        self.output = self.root / "output"
        self.pccl_revision = "3" * 40
        self.build_sha256 = "4" * 64
        (self.input / "envs" / "suite.env").write_text("PATH=/usr/bin\n")
        (self.input / "export.json").write_text(json.dumps({
            "format_version": 1, "source_revision": "1" * 40,
            "runner_revision": "2" * 40, "xtest_version": "0.12.0"}))
        self.write(self.input / "suites.csv", run_cases.SUITE_FIELDS, [{
            "plan": "plan", "suite": "suite", "nodes": "1", "cards_per_node": "8",
            "timeout": "10", "env": "suite.env"}])
        self.cases = [
            {"plan": "plan", "suite": "suite", "name": "exit_case", "check": "exit_code", "cmd": "true"},
            {"plan": "plan", "suite": "suite", "name": "perf_case", "check": "performance", "cmd": "true"},
        ]
        self.write(self.input / "standard.csv", run_cases.CASE_FIELDS, self.cases)
        (self.results / "build.json").write_text(json.dumps({
            "schema_version": 1, "pccl_revision": self.pccl_revision,
            "pccl_tests_revision": "2" * 40, "build_profile": "standard",
            "artifact_sha256": self.build_sha256}))
        for case in self.cases:
            self.write_result(case)
        self.write_results()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write(path, fields, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def write_result(self, case):
        directory = self.results / case["plan"] / case["suite"] / case["name"]
        directory.mkdir(parents=True)
        (directory / "raw.log").write_text(
            PERFORMANCE_LOG if case["check"] == "performance" else "completed\n")
        row = {**case, "status": "PASSED", "return_code": 0,
               "duration_seconds": 0.1, "error": "", "raw_log": "raw.log"}
        row.pop("cmd")
        (directory / "result.json").write_text(json.dumps(row))
        if case["check"] == "performance":
            self.write(directory / "metrics.csv", daily_reconcile.METRIC_FIELDS, [{
                "size": "8", "count": "2", "type": "float", "redop": "sum", "root": "-1",
                "placement": "out-of-place", "time_kind": "avg", "time": "1.0",
                "algbw": "1.0", "busbw": "1.0", "wrong": "0", "source_line": "3"},
               {"size": "8", "count": "2", "type": "float", "redop": "sum", "root": "-1",
                "placement": "in-place", "time_kind": "avg", "time": "1.0",
                "algbw": "1.0", "busbw": "1.0", "wrong": "0", "source_line": "3"}])

    def write_results(self):
        rows = []
        for case in self.cases:
            rows.append(json.loads((self.results / case["plan"] / case["suite"] / case["name"] /
                                    "result.json").read_text()))
        self.write(self.results / "results.csv", run_cases.RESULT_FIELDS, rows)
        (self.results / "summary.json").write_text(json.dumps({"total": len(rows), "counts": {"PASSED": len(rows)}}))

        self.rows = rows

    def reconcile(self):
        return daily_reconcile.reconcile(self.input, "standard", 1, "suite", self.results, self.output,
                                         self.pccl_revision, self.build_sha256)

    def test_valid_results_produce_a_report(self):
        report = self.reconcile()
        self.assertTrue(report["ok"])
        self.assertEqual(report["expected"], 2)
        self.assertEqual(report["actual"], 2)
        self.assertTrue((self.output / "reconciliation.json").is_file())

    def test_duplicate_missing_and_unexpected_results_are_rejected(self):
        self.rows[1] = self.rows[0]
        self.rows.append({**self.rows[0], "name": "unexpected"})
        self.write(self.results / "results.csv", run_cases.RESULT_FIELDS, self.rows)
        (self.results / "summary.json").write_text(json.dumps({"total": 3, "counts": {"PASSED": 3}}))
        report = self.reconcile()
        self.assertFalse(report["ok"])
        self.assertTrue(any("Duplicate result" in error for error in report["errors"]))
        self.assertTrue(any("Missing result" in error for error in report["errors"]))
        self.assertTrue(any("Unexpected result" in error for error in report["errors"]))

    def test_failed_result_and_invalid_metrics_are_rejected(self):
        case = self.cases[1]
        directory = self.results / case["plan"] / case["suite"] / case["name"]
        record = json.loads((directory / "result.json").read_text())
        record["status"] = "FAILED"
        record["return_code"] = 1
        record["error"] = "failed"
        (directory / "result.json").write_text(json.dumps(record))
        metrics = directory / "metrics.csv"
        metrics.write_text(metrics.read_text().replace("1.0", "0")
                           .replace("1.0", "0").replace("1.0", "0"))
        self.rows[1] = record
        self.write(self.results / "results.csv", run_cases.RESULT_FIELDS, self.rows)
        (self.results / "summary.json").write_text(json.dumps({"total": 2, "counts": {"PASSED": 1, "FAILED": 1}}))
        report = self.reconcile()
        self.assertFalse(report["ok"])
        self.assertTrue(any("FAILED" in error for error in report["errors"]))

    def test_nonpositive_performance_values_are_rejected(self):
        case = self.cases[1]
        metrics = self.results / case["plan"] / case["suite"] / case["name"] / "metrics.csv"
        metrics.write_text(metrics.read_text().replace("1.0", "0"))
        report = self.reconcile()
        self.assertFalse(report["ok"])
        self.assertTrue(any("non-positive" in error for error in report["errors"]))

    def test_negative_bandwidth_is_rejected(self):
        case = self.cases[1]
        metrics = self.results / case["plan"] / case["suite"] / case["name"] / "metrics.csv"
        metrics.write_text(metrics.read_text().replace(",1.0,1.0,0,3", ",-0.01,1.0,0,3"))
        report = self.reconcile()
        self.assertFalse(report["ok"])
        self.assertTrue(any("negative" in error for error in report["errors"]))

    def test_zero_bandwidth_from_small_measurement_rounding_is_accepted(self):
        case = self.cases[1]
        metrics = self.results / case["plan"] / case["suite"] / case["name"] / "metrics.csv"
        metrics.write_text(metrics.read_text().replace(",1.0,1.0,0,3", ",0.00,0.00,0,3"))
        log = self.results / case["plan"] / case["suite"] / case["name"] / "raw.log"
        log.write_text(log.read_text().replace(
            "1.0 1.0 1.0 0 1.0 1.0 1.0 0", "1.0 0.00 0.00 0 1.0 0.00 0.00 0"))
        report = self.reconcile()
        self.assertTrue(report["ok"])

    def test_zero_size_and_count_from_rank_division_are_accepted(self):
        case = self.cases[1]
        directory = self.results / case["plan"] / case["suite"] / case["name"]
        metrics = directory / "metrics.csv"
        metrics.write_text(metrics.read_text().replace("8,2,float", "0,0,float"))
        log = directory / "raw.log"
        log.write_text(log.read_text().replace("8 2 float", "0 0 float"))
        report = self.reconcile()
        self.assertTrue(report["ok"])

    def test_raw_log_and_every_metric_row_are_required(self):
        case = self.cases[1]
        directory = self.results / case["plan"] / case["suite"] / case["name"]
        (directory / "raw.log").write_text("completed\n")
        report = self.reconcile()
        self.assertFalse(report["ok"])
        self.assertTrue(any("raw log validation failed" in error for error in report["errors"]))

    def test_metrics_must_match_all_raw_log_rows(self):
        case = self.cases[1]
        metrics = self.results / case["plan"] / case["suite"] / case["name"] / "metrics.csv"
        metrics.write_text("\n".join(metrics.read_text().splitlines()[:-1]) + "\n")
        report = self.reconcile()
        self.assertFalse(report["ok"])
        self.assertTrue(any("do not match the raw log" in error for error in report["errors"]))

    def test_malformed_result_row_is_rejected(self):
        with (self.results / "results.csv").open("a") as stream:
            stream.write("plan,suite,exit_case,exit_code,PASSED,0\n")
        report = self.reconcile()
        self.assertFalse(report["ok"])
        self.assertTrue(any("Malformed CSV row" in error for error in report["errors"]))

    def test_result_record_requires_the_complete_schema(self):
        case = self.cases[1]
        path = self.results / case["plan"] / case["suite"] / case["name"] / "result.json"
        record = json.loads(path.read_text())
        del record["error"]
        path.write_text(json.dumps(record))
        report = self.reconcile()
        self.assertFalse(report["ok"])
        self.assertTrue(any("schema mismatch" in error for error in report["errors"]))

    def test_build_identity_mismatch_is_rejected(self):
        record = json.loads((self.results / "build.json").read_text())
        record["artifact_sha256"] = "5" * 64
        (self.results / "build.json").write_text(json.dumps(record))
        report = self.reconcile()
        self.assertFalse(report["ok"])
        self.assertTrue(any("Build identity" in error for error in report["errors"]))

    def test_timeout_null_return_code_preserves_the_timeout_diagnostic(self):
        case = self.cases[0]
        path = self.results / case["plan"] / case["suite"] / case["name"] / "result.json"
        record = json.loads(path.read_text())
        record.update(status="TIMEOUT", return_code=None, error="Case timed out after 10 seconds")
        path.write_text(json.dumps(record))
        self.write_results()
        (self.results / "summary.json").write_text(json.dumps({
            "total": 2, "counts": {"TIMEOUT": 1, "PASSED": 1}}))

        report = self.reconcile()

        self.assertFalse(report["ok"])
        self.assertEqual(report["errors"], [
            "plan/suite/exit_case: TIMEOUT: Case timed out after 10 seconds"])

    def test_non_object_result_json_is_reported_without_losing_the_report(self):
        case = self.cases[0]
        path = self.results / case["plan"] / case["suite"] / case["name"] / "result.json"
        path.write_text("null\n")

        report = self.reconcile()

        self.assertFalse(report["ok"])
        self.assertTrue(any("Expected a JSON object" in error for error in report["errors"]))
        self.assertTrue((self.output / "reconciliation.json").is_file())

    def test_non_object_summary_json_is_reported_without_losing_the_report(self):
        (self.results / "summary.json").write_text("[]\n")

        report = self.reconcile()

        self.assertFalse(report["ok"])
        self.assertTrue(any("Expected a JSON object" in error for error in report["errors"]))
        self.assertTrue((self.output / "reconciliation.json").is_file())

    def test_not_run_preserves_the_stop_reason_without_requiring_case_files(self):
        case = self.cases[0]
        path = self.results / case["plan"] / case["suite"] / case["name"] / "result.json"
        row = json.loads(path.read_text())
        row.update(status="NOT_RUN", return_code=None, error="Skipped after a remote worker failure")
        path.unlink()
        self.rows[0] = row
        self.write(self.results / "results.csv", run_cases.RESULT_FIELDS, self.rows)
        (self.results / "summary.json").write_text(json.dumps({
            "total": 2, "counts": {"NOT_RUN": 1, "PASSED": 1}}))

        report = self.reconcile()

        self.assertFalse(report["ok"])
        self.assertEqual(report["errors"], [
            "plan/suite/exit_case: NOT_RUN: Skipped after a remote worker failure"])


if __name__ == "__main__":
    unittest.main()
