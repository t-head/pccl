# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
from pathlib import Path
import tempfile
import unittest

import reconcile_results
import run_cases


class ReconcileResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input = self.root / "input"
        (self.input / "envs").mkdir(parents=True)
        self.results = self.root / "results"
        self.results.mkdir()
        self.output = self.root / "output"
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
        (directory / "raw.log").write_text("completed\n")
        row = {**case, "status": "PASSED", "return_code": 0,
               "duration_seconds": 0.1, "error": "", "raw_log": "raw.log"}
        row.pop("cmd")
        (directory / "result.json").write_text(json.dumps(row))
        if case["check"] == "performance":
            self.write(directory / "metrics.csv", reconcile_results.METRIC_FIELDS, [{
                "size": "8", "count": "2", "type": "float", "redop": "sum", "root": "-1",
                "placement": "out-of-place", "time_kind": "avg", "time": "1.0",
                "algbw": "1.0", "busbw": "1.0", "wrong": "0", "source_line": "10"}])

    def write_results(self):
        rows = []
        for case in self.cases:
            rows.append(json.loads((self.results / case["plan"] / case["suite"] / case["name"] /
                                    "result.json").read_text()))
        self.write(self.results / "results.csv", run_cases.RESULT_FIELDS, rows)
        (self.results / "summary.json").write_text(json.dumps({"total": len(rows), "counts": {"PASSED": len(rows)}}))

        self.rows = rows

    def reconcile(self):
        return reconcile_results.reconcile(self.input, "standard", 1, "suite", self.results, self.output)

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


if __name__ == "__main__":
    unittest.main()
