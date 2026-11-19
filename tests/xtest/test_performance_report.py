# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import daily_reconcile
import performance_report
import run_cases


HEADER = ("# size count type redop root event_time algbw busbw #wrong "
          "event_time algbw busbw #wrong\n")


def performance_log(skip_size=None):
    lines = [HEADER]
    for index in range(32):
        size = 8 * (2 ** index)
        if size == skip_size:
            continue
        lines.append(f"{size} 2 half sum -1 10.5 1.25 2.5 0 11.5 2.25 3.5 0\n")
    lines.append("# Out of bounds values : 0 OK\n")
    return "".join(lines)


def make_case(name, suite, check="exit_code"):
    return {
        "plan": "plan", "suite": suite, "name": name, "check": check,
        "cmd": "mpirun -np 2 all_reduce_perf -t 1 -g 1 -d half -b 8 -e 16G -f 2",
        "category": performance_report._category({"suite": suite, "check": check}),
        "nodes": 1, "cards_per_node": 8,
    }


class PerformanceReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.results = self.root / "results"
        self.case1 = make_case("test_op_all_reduce_perf_ranks_2_nnodes1",
                          "op_perf_shm_single_node_suite")
        self.case2 = make_case("test_op_local_register_all_reduce_float_perf_ranks_2_nnodes1",
                          "local_register_perf_shm_single_node_suite", "performance")
        self.case3 = make_case("test_op_all_reduce_perf_ranks_8_nnodes1",
                          "op_perf_shm_single_node_suite")
        self.cases = [self.case1, self.case2, self.case3]
        self.selection = self.root / "performance-cases.json"
        self.selection.write_text(json.dumps({
            "schema_version": 1, "profile": "standard", "cases": self.cases}))
        self.groups = self.root / "groups.json"
        self.groups.write_text(json.dumps([
            {"id": "g001", "case_keys": [list(performance_report._key(case))
                                         for case in (self.case1, self.case2)]},
        ]))
        self.coverage = self.root / "execution-coverage.json"
        self.coverage.write_text(json.dumps({
            "total_cases": 3, "selected_cases": 2, "excluded_cases": 1,
            "allowed_nodes": [1, 2, 4], "exclusions": []}))
        for case, status, log in ((self.case1, "PASSED", performance_log()),
                                  (self.case2, "FAILED", performance_log(skip_size=16))):
            directory = self.results / "g001" / Path(*performance_report._key(case))
            directory.mkdir(parents=True)
            (directory / "raw.log").write_text(log)
            row = {"plan": case["plan"], "suite": case["suite"], "name": case["name"],
                   "check": case["check"], "status": status, "return_code": "0",
                   "duration_seconds": 1.0, "error": "executor failed" if status == "FAILED" else "",
                   "raw_log": "raw.log"}
            (directory / "result.json").write_text(json.dumps(row))
        group = self.results / "g001"
        with (group / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=run_cases.RESULT_FIELDS, lineterminator="\n")
            writer.writeheader()
            for case, status in ((self.case1, "PASSED"), (self.case2, "FAILED")):
                writer.writerow({"plan": case["plan"], "suite": case["suite"], "name": case["name"],
                                 "check": case["check"], "status": status, "return_code": "0",
                                 "duration_seconds": 1.0,
                                 "error": "executor failed" if status == "FAILED" else "",
                                 "raw_log": "raw.log"})
        (group / "build.json").write_text(json.dumps({"archive_sha256": "abc123"}))
        self.reconciliation = self.root / "public-results"
        (self.reconciliation / "g001").mkdir(parents=True)
        (self.reconciliation / "g001" / "reconciliation.json").write_text(json.dumps({
            "ok": True, "errors": []}))

    def tearDown(self):
        self.temporary.cleanup()

    def collect(self):
        return performance_report.collect_report(
            self.selection, self.groups, self.coverage, self.results,
            {"run_id": "123", "dataset": "full"}, self.reconciliation)

    def prepare_actual_reconciliation(self):
        """Run the real reconciler with one ordinary and one healthy performance Case."""
        self.case2["cmd"] = "mpirun -np 2 all_reduce_perf -t 1 -g 1 -d half -b 8 -e 8 -f 2"
        self.selection.write_text(json.dumps({
            "schema_version": 1, "profile": "standard", "cases": self.cases}))
        key2 = performance_report._key(self.case2)
        directory2 = self.results / "g001" / Path(*key2)
        (directory2 / "raw.log").write_text(HEADER + (
            "8 2 half sum -1 10.5 1.25 2.5 0 11.5 2.25 3.5 0\n"
            "# Out of bounds values : 0 OK\n"))
        for case in (self.case1, self.case2):
            path = self.results / "g001" / Path(*performance_report._key(case)) / "result.json"
            record = json.loads(path.read_text())
            record.update(status="PASSED", error="")
            path.write_text(json.dumps(record))
        metrics = run_cases._check_result(self.case2, 0, directory2 / "raw.log")
        with (directory2 / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=run_cases.METRIC_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(metrics)
        group = self.results / "g001"
        rows = [json.loads((group / Path(*performance_report._key(case)) / "result.json").read_text())
                for case in (self.case1, self.case2)]
        with (group / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=run_cases.RESULT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        (group / "summary.json").write_text(json.dumps({
            "total": 2, "counts": {"PASSED": 2}}))
        pccl_revision = "3" * 40
        build_sha256 = "4" * 64
        (group / "build.json").write_text(json.dumps({
            "schema_version": 1, "pccl_revision": pccl_revision,
            "pccl_tests_revision": "2" * 40, "build_profile": "standard",
            "artifact_sha256": build_sha256}))
        input_directory = self.root / "actual-input"
        (input_directory / "envs").mkdir(parents=True)
        (input_directory / "envs" / "suite.env").write_text("PATH=/usr/bin\n")
        (input_directory / "export.json").write_text(json.dumps({
            "format_version": 1, "source_revision": "1" * 40,
            "runner_revision": "2" * 40, "xtest_version": "0.12.0"}))
        suites = [{"plan": case["plan"], "suite": case["suite"], "nodes": "1",
                   "cards_per_node": "8", "timeout": "10", "env": "suite.env"}
                  for case in (self.case1, self.case2)]
        with (input_directory / "suites.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=run_cases.SUITE_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(suites)
        with (input_directory / "standard.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=run_cases.CASE_FIELDS, lineterminator="\n")
            writer.writeheader()
            for case in (self.case1, self.case2):
                writer.writerow({field: case[field] for field in run_cases.CASE_FIELDS})
        output = self.root / "actual-reconciliation" / "g001"

        def reconcile():
            return daily_reconcile.reconcile(
                input_directory, "standard", 1, None, group, output,
                pccl_revision, build_sha256)

        return rows, output, reconcile

    def test_exit_code_raw_log_is_reported_without_changing_execution_gate(self):
        report = self.collect()
        rows = {row["name"]: row for row in report["coverage"]}
        first = rows[self.case1["name"]]
        second = rows[self.case2["name"]]
        third = rows[self.case3["name"]]
        self.assertEqual(first["execution_status"], "PASSED")
        self.assertEqual(first["data_status"], "VALID")
        self.assertEqual(first["valid_out_of_place"], 32)
        self.assertEqual(run_cases._check_result(
            self.case1, 0, self.results / "g001" / Path(*performance_report._key(self.case1)) / "raw.log"), [])

        self.assertEqual(second["execution_status"], "FAILED")
        self.assertEqual(second["data_status"], "FAILED")
        self.assertEqual(second["valid_out_of_place"], 0)
        self.assertEqual(second["missing_out_of_place"], 0)
        self.assertEqual(second["reason"], "executor failed")
        self.assertEqual(third["execution_status"], "EXCLUDED")
        self.assertEqual(third["data_status"], "NO_RESULT")

        self.assertEqual(report["summary"]["performance_cases"], 3)
        self.assertEqual(report["summary"]["metric_rows"], 64)
        self.assertEqual(report["summary"]["valid_out_of_place_detail"], 32)
        self.assertEqual(report["summary"]["execution"]["missing_selected_results"], 0)

    def test_expected_sweep_is_derived_from_the_case_command(self):
        sizes = performance_report.expected_sizes(self.case1["cmd"])
        self.assertEqual(len(sizes), 32)
        self.assertEqual(sizes[:4], [8, 16, 32, 64])
        self.assertEqual(sizes[-1], 16 * 1024 ** 3)
        self.assertEqual(performance_report.expected_sizes("true"), [])

    def test_report_preserves_duplicates_unsupported_rows_and_interleaved_logs(self):
        zero = "0 0 half null -1 7.88 0.00 0.00 0 7.58 0.00 0.00 0\n"
        body = performance_log().removeprefix(HEADER).splitlines()
        text = (HEADER + zero + zero + "\n".join(body[2:]) + "\n"
                + "262144 262144 half sum -1[na131g][interleaved-log]\n")
        text = text.replace("11.5 2.25 3.5 0", "11.5 2.25 3.5 N/A", 1)
        with self.assertRaises(ValueError):
            run_cases.parse_performance(text)
        log = self.root / "tolerant.log"
        log.write_text(text)
        parsed = performance_report._parse_case(
            self.case1, log, performance_report.expected_sizes(self.case1["cmd"]))
        self.assertEqual(parsed["data_status"], "VALID")
        self.assertEqual(parsed["valid_out_of_place"], 32)
        self.assertEqual(parsed["missing_out_of_place"], 0)
        self.assertEqual(len(parsed["rows"]), 64)
        self.assertEqual(sum(row["metric_status"] == "UNSUPPORTED" for row in parsed["rows"]), 1)
        self.assertEqual(sum(row["size"] == 0 for row in parsed["rows"]), 4)

    def test_duplicate_size_cannot_hide_an_expected_missing_measurement(self):
        text = HEADER + (
            "8 2 half sum -1 10 1 2 0 11 2 3 0\n"
            "8 2 half sum -1 10 1 2 0 11 2 3 0\n"
            "32 2 half sum -1 10 1 2 0 11 2 3 0\n")
        log = self.root / "missing-16.log"
        log.write_text(text)
        parsed = performance_report._parse_case(self.case1, log, [8, 16, 32])
        self.assertEqual(parsed["data_status"], "PARTIAL")
        self.assertEqual(parsed["missing_out_of_place"], 1)

    def test_reconciliation_failure_marks_results_unverified(self):
        key = performance_report._key(self.case1)
        (self.reconciliation / "g001" / "reconciliation.json").write_text(json.dumps({
            "ok": False, "errors": ["Result.json status mismatch for " + "/".join(key)]}))
        report = self.collect()
        row = next(row for row in report["coverage"] if row["name"] == self.case1["name"])
        self.assertEqual(row["execution_status"], "PASSED")
        self.assertEqual(row["data_status"], "UNVERIFIED")
        self.assertEqual(row["valid_out_of_place"], 0)
        self.assertEqual(report["summary"]["metric_rows"], 0)
        self.assertEqual(report["summary"]["performance_total"]["unverified"], 1)
        self.assertEqual(report["summary"]["performance_total"]["passed"], 0)

    def test_case_mismatch_preserves_other_passed_case_metrics(self):
        path = self.results / "g001" / "results.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[1].update(status="PASSED", error="")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=run_cases.RESULT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        key = performance_report._key(self.case1)
        (self.reconciliation / "g001" / "reconciliation.json").write_text(json.dumps({
            "ok": False, "errors": ["Result.json status mismatch for " + "/".join(key)]}))
        report = self.collect()
        first = next(row for row in report["coverage"] if row["name"] == self.case1["name"])
        second = next(row for row in report["coverage"] if row["name"] == self.case2["name"])
        self.assertEqual(first["data_status"], "UNVERIFIED")
        self.assertEqual(second["data_status"], "PARTIAL")
        self.assertEqual(report["summary"]["metric_rows"], 64)
        self.assertEqual(report["summary"]["performance_total"]["unverified"], 1)
        self.assertEqual(report["summary"]["performance_total"]["passed"], 1)

    def test_case_file_diagnostics_preserve_other_case_metrics(self):
        path = self.results / "g001" / "results.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[1].update(status="PASSED", error="")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=run_cases.RESULT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        key = performance_report._key(self.case1)
        for filename in ("result.json", "metrics.csv"):
            with self.subTest(filename=filename):
                diagnostic = self.results / "g001" / Path(*key) / filename
                (self.reconciliation / "g001" / "reconciliation.json").write_text(json.dumps({
                    "ok": False, "errors": [f"{diagnostic}: damaged"]}))
                report = self.collect()
                first = next(row for row in report["coverage"] if row["name"] == self.case1["name"])
                second = next(row for row in report["coverage"] if row["name"] == self.case2["name"])
                self.assertEqual(first["data_status"], "UNVERIFIED")
                self.assertEqual(second["data_status"], "PARTIAL")
                self.assertEqual(report["summary"]["metric_rows"], 64)

    def test_malformed_reconciliation_does_not_hide_another_group(self):
        groups = json.loads(self.groups.read_text())
        groups.append({"id": "g002", "case_keys": [list(performance_report._key(self.case3))]})
        self.groups.write_text(json.dumps(groups))
        self.coverage.write_text(json.dumps({
            "total_cases": 3, "selected_cases": 3, "excluded_cases": 0,
            "allowed_nodes": [1, 2, 4], "exclusions": []}))
        key = performance_report._key(self.case3)
        directory = self.results / "g002" / Path(*key)
        directory.mkdir(parents=True)
        (directory / "raw.log").write_text(performance_log())
        row = {"plan": self.case3["plan"], "suite": self.case3["suite"], "name": self.case3["name"],
               "check": self.case3["check"], "status": "PASSED", "return_code": "0",
               "duration_seconds": 1.0, "error": "", "raw_log": "raw.log"}
        (directory / "result.json").write_text(json.dumps(row))
        with (self.results / "g002" / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=run_cases.RESULT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerow(row)
        (self.results / "g002" / "build.json").write_text(json.dumps({"archive_sha256": "def456"}))
        (self.reconciliation / "g002").mkdir()
        (self.reconciliation / "g002" / "reconciliation.json").write_text(json.dumps({
            "ok": True, "errors": []}))
        for content in ("{", "[]"):
            with self.subTest(content=content):
                (self.reconciliation / "g001" / "reconciliation.json").write_text(content)
                report = self.collect()
                first = next(row for row in report["coverage"] if row["name"] == self.case1["name"])
                third = next(row for row in report["coverage"] if row["name"] == self.case3["name"])
                self.assertEqual(first["data_status"], "UNVERIFIED")
                self.assertEqual(third["data_status"], "VALID")
                self.assertEqual(third["group"], "g002")

    def test_duplicate_result_marks_only_the_duplicate_unverified(self):
        path = self.results / "g001" / "results.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[1].update(status="PASSED", error="")
        rows.append(dict(rows[0]))
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=run_cases.RESULT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        report = self.collect()
        first = next(row for row in report["coverage"] if row["name"] == self.case1["name"])
        second = next(row for row in report["coverage"] if row["name"] == self.case2["name"])
        self.assertEqual(first["data_status"], "UNVERIFIED")
        self.assertEqual(second["data_status"], "PARTIAL")
        self.assertEqual(report["summary"]["metric_rows"], 64)

    def test_result_from_an_unplanned_group_is_unverified(self):
        path = self.results / "g001" / "results.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        key = performance_report._key(self.case3)
        row = {"plan": self.case3["plan"], "suite": self.case3["suite"], "name": self.case3["name"],
               "check": self.case3["check"], "status": "PASSED", "return_code": "0",
               "duration_seconds": 1.0, "error": "", "raw_log": "raw.log"}
        rows.append(row)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=run_cases.RESULT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        directory = self.results / "g001" / Path(*key)
        directory.mkdir(parents=True)
        (directory / "raw.log").write_text(performance_log())
        (directory / "result.json").write_text(json.dumps(row))
        report = self.collect()
        first = next(row for row in report["coverage"] if row["name"] == self.case1["name"])
        third = next(row for row in report["coverage"] if row["name"] == self.case3["name"])
        self.assertEqual(first["data_status"], "VALID")
        self.assertEqual(third["execution_status"], "PASSED")
        self.assertEqual(third["group"], "g001")
        self.assertEqual(third["data_status"], "UNVERIFIED")
        self.assertEqual(third["valid_out_of_place"], 0)
        self.assertIn("does not match the planned group", third["reason"])

    def test_invalid_utf8_build_and_recovered_result_use_their_fallbacks(self):
        (self.results / "g001" / "build.json").write_bytes(b"\xff")
        (self.results / "g001" / "results.csv").unlink()
        result = self.results / "g001" / Path(*performance_report._key(self.case1)) / "result.json"
        result.write_bytes(b"\xff")
        report = self.collect()
        first = next(row for row in report["coverage"] if row["name"] == self.case1["name"])
        self.assertEqual(first["build_sha256"], "")
        self.assertEqual(first["execution_status"], "NO_RESULT")
        self.assertEqual(report["summary"]["execution"]["selected_result_count"], 1)
        self.assertEqual(report["summary"]["execution"]["missing_selected_results"], 1)

    def test_aggregate_summary_mismatch_does_not_invalidate_the_group(self):
        path = self.results / "g001" / "results.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[1].update(status="PASSED", error="")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=run_cases.RESULT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        (self.reconciliation / "g001" / "reconciliation.json").write_text(json.dumps({
            "ok": False, "errors": ["Summary total 1 does not match 2 result rows"]}))
        report = self.collect()
        first = next(row for row in report["coverage"] if row["name"] == self.case1["name"])
        second = next(row for row in report["coverage"] if row["name"] == self.case2["name"])
        self.assertEqual(first["data_status"], "VALID")
        self.assertEqual(second["data_status"], "PARTIAL")
        self.assertEqual(report["summary"]["performance_total"]["unverified"], 0)
        self.assertEqual(report["summary"]["metric_rows"], 128)

    def test_csv_error_recovers_retained_case_records(self):
        with mock.patch.object(performance_report.run_cases, "_read_csv",
                               side_effect=csv.Error("damaged quoted field")):
            report = self.collect()
        first = next(row for row in report["coverage"] if row["name"] == self.case1["name"])
        self.assertEqual(first["execution_status"], "PASSED")
        self.assertEqual(first["data_status"], "UNVERIFIED")
        self.assertEqual(report["summary"]["execution"]["selected_result_count"], 2)
        self.assertEqual(report["summary"]["execution"]["missing_selected_results"], 0)

    def test_recovery_includes_nonperformance_cases_in_execution_counts(self):
        case4 = {"plan": "plan", "suite": "function_suite", "name": "test_correctness"}
        groups = json.loads(self.groups.read_text())
        groups[0]["case_keys"].append(list(performance_report._key(case4)))
        self.groups.write_text(json.dumps(groups))
        self.coverage.write_text(json.dumps({
            "total_cases": 4, "selected_cases": 3, "excluded_cases": 1,
            "allowed_nodes": [1, 2, 4], "exclusions": []}))
        directory = self.results / "g001" / Path(*performance_report._key(case4))
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(json.dumps({
            "status": "PASSED", "check": "exit_code", "error": ""}))
        (self.results / "g001" / "results.csv").unlink()
        report = self.collect()
        execution = report["summary"]["execution"]
        self.assertEqual(execution["result_counts"], {"FAILED": 1, "PASSED": 2})
        self.assertEqual(execution["selected_result_count"], 3)
        self.assertEqual(execution["missing_selected_results"], 0)
        self.assertEqual(report["summary"]["performance_cases"], 3)

    def test_intact_csv_counts_correctness_not_run_without_result_json(self):
        case4 = {"plan": "plan", "suite": "function_suite", "name": "test_correctness"}
        groups = json.loads(self.groups.read_text())
        groups[0]["case_keys"].append(list(performance_report._key(case4)))
        self.groups.write_text(json.dumps(groups))
        self.coverage.write_text(json.dumps({
            "total_cases": 4, "selected_cases": 3, "excluded_cases": 1,
            "allowed_nodes": [1, 2, 4], "exclusions": []}))
        row = {"plan": case4["plan"], "suite": case4["suite"], "name": case4["name"],
               "check": "exit_code", "status": "NOT_RUN", "return_code": "",
               "duration_seconds": 0, "error": "Stopped after an earlier failure", "raw_log": ""}
        path = self.results / "g001" / "results.csv"
        with path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=run_cases.RESULT_FIELDS, lineterminator="\n")
            writer.writerow(row)

        report = self.collect()

        execution = report["summary"]["execution"]
        self.assertEqual(execution["result_counts"], {"FAILED": 1, "NOT_RUN": 1, "PASSED": 1})
        self.assertEqual(execution["selected_result_count"], 3)
        self.assertEqual(execution["missing_selected_results"], 0)
        self.assertEqual(report["summary"]["performance_cases"], 3)
        self.assertFalse((self.results / "g001" / Path(*performance_report._key(case4)) /
                          "result.json").exists())

    def test_failed_csv_status_with_a_record_mismatch_is_unverified(self):
        key = performance_report._key(self.case2)
        result = self.results / "g001" / Path(*key) / "result.json"
        record = json.loads(result.read_text())
        record["status"] = "PASSED"
        result.write_text(json.dumps(record))
        (self.reconciliation / "g001" / "reconciliation.json").write_text(json.dumps({
            "ok": False, "errors": ["Result.json status mismatch for " + "/".join(key)]}))
        report = self.collect()
        second = next(row for row in report["coverage"] if row["name"] == self.case2["name"])
        self.assertEqual(second["execution_status"], "FAILED")
        self.assertEqual(second["data_status"], "UNVERIFIED")
        self.assertEqual(second["valid_out_of_place"], 0)
        self.assertEqual(report["summary"]["performance_total"]["failed"], 0)
        self.assertEqual(report["summary"]["performance_total"]["unverified"], 1)
        self.assertEqual(report["summary"]["metric_rows"], 64)

    def test_wrong_group_result_remains_missing_from_the_planned_group(self):
        groups = json.loads(self.groups.read_text())
        groups.append({"id": "g002", "case_keys": [list(performance_report._key(self.case3))]})
        self.groups.write_text(json.dumps(groups))
        self.coverage.write_text(json.dumps({
            "total_cases": 3, "selected_cases": 3, "excluded_cases": 0,
            "allowed_nodes": [1, 2, 4], "exclusions": []}))
        key = performance_report._key(self.case3)
        row = {"plan": self.case3["plan"], "suite": self.case3["suite"], "name": self.case3["name"],
               "check": self.case3["check"], "status": "PASSED", "return_code": "0",
               "duration_seconds": 1.0, "error": "", "raw_log": "raw.log"}
        path = self.results / "g001" / "results.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows.append(row)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=run_cases.RESULT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        directory = self.results / "g001" / Path(*key)
        directory.mkdir(parents=True)
        (directory / "raw.log").write_text(performance_log())
        (directory / "result.json").write_text(json.dumps(row))
        report = self.collect()
        third = next(row for row in report["coverage"] if row["name"] == self.case3["name"])
        self.assertEqual(third["data_status"], "UNVERIFIED")
        self.assertEqual(report["summary"]["execution"]["selected_result_count"], 2)
        self.assertEqual(report["summary"]["execution"]["missing_selected_results"], 1)
        self.assertEqual(report["summary"]["execution"]["result_counts"],
                         {"FAILED": 1, "PASSED": 1})

    def test_planned_result_wins_over_wrong_group_in_either_traversal_order(self):
        groups = json.loads(self.groups.read_text())
        groups.append({"id": "g002", "case_keys": [list(performance_report._key(self.case3))]})
        self.coverage.write_text(json.dumps({
            "total_cases": 3, "selected_cases": 3, "excluded_cases": 0,
            "allowed_nodes": [1, 2, 4], "exclusions": []}))
        key = performance_report._key(self.case3)
        row = {"plan": self.case3["plan"], "suite": self.case3["suite"], "name": self.case3["name"],
               "check": self.case3["check"], "status": "PASSED", "return_code": "0",
               "duration_seconds": 1.0, "error": "", "raw_log": "raw.log"}
        with (self.results / "g001" / "results.csv").open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=run_cases.RESULT_FIELDS, lineterminator="\n")
            writer.writerow(row)
        directory = self.results / "g002" / Path(*key)
        directory.mkdir(parents=True)
        (directory / "raw.log").write_text(performance_log())
        (directory / "result.json").write_text(json.dumps(row))
        with (self.results / "g002" / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=run_cases.RESULT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerow(row)
        (self.results / "g002" / "build.json").write_text(json.dumps({"archive_sha256": "def456"}))
        (self.reconciliation / "g002").mkdir()
        (self.reconciliation / "g002" / "reconciliation.json").write_text(json.dumps({
            "ok": True, "errors": []}))

        executions = []
        for ordered in (groups, list(reversed(groups))):
            self.groups.write_text(json.dumps(ordered))
            report = self.collect()
            third = next(row for row in report["coverage"] if row["name"] == self.case3["name"])
            self.assertEqual(third["group"], "g002")
            self.assertEqual(third["execution_status"], "PASSED")
            self.assertEqual(third["data_status"], "UNVERIFIED")
            execution = report["summary"]["execution"]
            self.assertEqual(execution["selected_result_count"], 3)
            self.assertEqual(execution["missing_selected_results"], 0)
            self.assertEqual(execution["result_counts"], {"FAILED": 1, "PASSED": 2})
            executions.append(execution)
        self.assertEqual(executions[0], executions[1])

    def test_invalid_utf8_record_isolated_by_actual_reconciliation(self):
        _, output, reconcile = self.prepare_actual_reconciliation()
        baseline = self.collect()
        healthy = next(row for row in baseline["coverage"] if row["name"] == self.case2["name"])
        self.assertEqual(healthy["valid_out_of_place"], 1)
        path = self.results / "g001" / Path(*performance_report._key(self.case1)) / "result.json"
        path.write_bytes(b"\xff\xfe")

        reconciliation = reconcile()

        self.assertFalse(reconciliation["ok"])
        self.assertTrue((output / "reconciliation.json").is_file())
        self.assertTrue(any("result.json" in error and "utf-8" in error
                            for error in reconciliation["errors"]))
        self.reconciliation = output.parent
        report = self.collect()
        damaged = next(row for row in report["coverage"] if row["name"] == self.case1["name"])
        healthy = next(row for row in report["coverage"] if row["name"] == self.case2["name"])
        self.assertEqual(damaged["data_status"], "UNVERIFIED")
        self.assertEqual(healthy["data_status"], "VALID")
        self.assertEqual(healthy["valid_out_of_place"], 1)
        self.assertEqual(report["summary"]["metric_rows"], 2)

    def test_unexpected_row_isolated_by_actual_reconciliation(self):
        rows, output, reconcile = self.prepare_actual_reconciliation()
        unexpected = dict(rows[0])
        unexpected.update(suite="unexpected_suite", name="unexpected_case")
        rows.append(unexpected)
        group = self.results / "g001"
        with (group / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=run_cases.RESULT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        (group / "summary.json").write_text(json.dumps({
            "total": 3, "counts": {"PASSED": 3}}))

        reconciliation = reconcile()

        self.assertFalse(reconciliation["ok"])
        self.assertTrue(any("Unexpected result" in error for error in reconciliation["errors"]))
        self.reconciliation = output.parent
        report = self.collect()
        healthy = next(row for row in report["coverage"] if row["name"] == self.case2["name"])
        self.assertEqual(healthy["data_status"], "VALID")
        self.assertEqual(healthy["valid_out_of_place"], 1)
        self.assertEqual(report["summary"]["execution"]["result_counts"], {"PASSED": 2})
        self.assertEqual(report["summary"]["metric_rows"], 66)

    def test_report_skips_a_malformed_root_without_changing_the_strict_parser(self):
        text = HEADER + (
            "8 2 half sum -1 10 1 2 0 11 2 3 0\n"
            "16 2 half sum -1[WARN] 10 1 2 0 11 2 3 0\n"
            "32 2 half sum -1 10 1 2 0 11 2 3 0\n")
        strict = run_cases.parse_performance(text)
        self.assertEqual(len(strict), 6)
        self.assertIn("-1[WARN]", {row["root"] for row in strict})
        log = self.root / "interleaved-root.log"
        log.write_text(text)
        parsed = performance_report._parse_case(self.case1, log, [8, 16, 32])
        self.assertEqual(parsed["data_status"], "PARTIAL")
        self.assertEqual(parsed["valid_out_of_place"], 2)
        self.assertEqual(parsed["missing_out_of_place"], 1)
        self.assertEqual(len(parsed["rows"]), 6)

    def test_report_skips_a_numeric_token_with_interleaved_log_text(self):
        text = HEADER + (
            "8 2 half sum -1 10 1 2 0 11 2 3 0\n"
            "16 2 half sum -1 10 1 2 0 11[WARN] 2 3 0\n"
            "32 2 half sum -1 10 1 2 0 11 2 3 0\n")
        with self.assertRaises(ValueError):
            run_cases.parse_performance(text)
        log = self.root / "interleaved-number.log"
        log.write_text(text)
        parsed = performance_report._parse_case(self.case1, log, [8, 16, 32])
        self.assertEqual(parsed["data_status"], "PARTIAL")
        self.assertEqual(parsed["valid_out_of_place"], 2)
        self.assertEqual(parsed["missing_out_of_place"], 1)
        self.assertEqual(len(parsed["rows"]), 6)

    def test_nonpositive_time_and_negative_bandwidth_are_not_valid_points(self):
        log = self.results / "g001" / Path(*performance_report._key(self.case1)) / "raw.log"
        original = log.read_text()
        for old, new in (("10.5 1.25 2.5", "0 1.25 2.5"),
                         ("10.5 1.25 2.5", "10.5 -1.25 2.5")):
            with self.subTest(new=new):
                log.write_text(original.replace(old, new, 1))
                report = self.collect()
                row = next(row for row in report["coverage"] if row["name"] == self.case1["name"])
                self.assertEqual(row["execution_status"], "PASSED")
                self.assertEqual(row["data_status"], "INVALID")
                self.assertEqual(report["summary"]["metric_rows"], 0)

    def test_missing_or_corrupt_results_csv_recovers_unverified_case_records(self):
        path = self.results / "g001" / "results.csv"
        for mode in ("missing", "corrupt"):
            with self.subTest(mode=mode):
                if mode == "missing":
                    path.unlink()
                else:
                    path.write_text("not,a,valid,results,file\n")
                report = self.collect()
                first = next(row for row in report["coverage"] if row["name"] == self.case1["name"])
                second = next(row for row in report["coverage"] if row["name"] == self.case2["name"])
                self.assertEqual(first["execution_status"], "PASSED")
                self.assertEqual(first["data_status"], "UNVERIFIED")
                self.assertEqual(second["execution_status"], "FAILED")
                self.assertEqual(report["summary"]["metric_rows"], 0)

    def test_incomplete_recovered_result_is_rejected(self):
        (self.results / "g001" / "results.csv").unlink()
        result = self.results / "g001" / Path(*performance_report._key(self.case1)) / "result.json"
        result.write_text(json.dumps({"status": "PASSED", "check": "exit_code"}))
        report = self.collect()
        row = next(row for row in report["coverage"] if row["name"] == self.case1["name"])
        self.assertEqual(row["execution_status"], "NO_RESULT")
        self.assertEqual(row["data_status"], "NO_RESULT")

    def test_not_run_is_unfinished_not_failed(self):
        path = self.results / "g001" / "results.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["status"] = "NOT_RUN"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=run_cases.RESULT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        report = self.collect()
        row = next(row for row in report["coverage"] if row["name"] == self.case1["name"])
        self.assertEqual(row["execution_status"], "NOT_RUN")
        self.assertEqual(row["data_status"], "NOT_RUN")
        total = report["summary"]["performance_total"]
        self.assertEqual(total["unfinished"], 2)
        self.assertEqual(total["failed"], 1)

    def test_build_identity_failure_makes_the_group_unverified(self):
        (self.reconciliation / "g001" / "reconciliation.json").write_text(json.dumps({
            "ok": False, "errors": ["Build identity mismatch"]}))
        report = self.collect()
        first = next(row for row in report["coverage"] if row["name"] == self.case1["name"])
        self.assertEqual(first["data_status"], "UNVERIFIED")
        self.assertEqual(report["summary"]["metric_rows"], 0)
        self.assertEqual(report["summary"]["performance_total"]["unverified"], 1)

    def test_dtype_falls_back_to_the_observed_table_type(self):
        self.case1["cmd"] = "mpirun -np 16 all_reduce_perf -b 8 -e 16G -f 2"
        self.selection.write_text(json.dumps({
            "schema_version": 1, "profile": "standard", "cases": self.cases}))
        report = self.collect()
        row = next(row for row in report["coverage"] if row["name"] == self.case1["name"])
        self.assertEqual(row["dtype"], "half")
        self.assertTrue(all(metric["dtype"] == "half" for metric in report["metrics"]))

    def test_workflow_publishes_a_complete_report_with_a_pinned_writer(self):
        workflow = (Path(__file__).resolve().parents[2] /
                    ".github/workflows/m890p-daily-full.yml").read_text()
        self.assertIn("performance-cases.json", workflow)
        self.assertNotIn("GENERATE_NODES", workflow)
        self.assertIn("import xlsxwriter; assert xlsxwriter.__version__", workflow)
        self.assertIn("discover -s pccl/tests/xtest -p 'test_performance_report.py'", workflow)
        self.assertIn("performance_report.write_selection", workflow)
        self.assertIn("performance_report.py report", workflow)
        self.assertIn("--results-root /wl_nas/devops/pccl-artifact-cache-", workflow)
        self.assertIn("--reconciliation-root public-results", workflow)
        self.assertIn("name: m890p-daily-performance-report", workflow)
        self.assertIn("XlsxWriter==3.2.5", workflow)
        self.assertIn("PYTHONPATH=deps/xlsxwriter-3.2.5-py3-none-any.whl", workflow)
        self.assertNotIn("--target report-python", workflow)
        self.assertIn("4f4824234e1eaf9d95df9a8fe974585ff91d0f5e3d3f12ace5b71e443c1c6abd", workflow)
        self.assertIn("'performance-report': '${{ steps.performance-report.outcome }}'", workflow)
        self.assertIn("name: m890p-daily-full-python-deps", workflow)
        source = (Path(__file__).resolve().parent / "performance_report.py").read_text()
        self.assertIn("run_cases.parse_performance(", source)

    @unittest.skipUnless(importlib.util.find_spec("xlsxwriter"), "XlsxWriter is not installed")
    def test_workbook_is_a_valid_xlsx_artifact(self):
        output = self.root / "performance-report.xlsx"
        performance_report.write_workbook(self.collect(), output)
        self.assertGreater(output.stat().st_size, 5000)
        with zipfile.ZipFile(output) as archive:
            names = set(archive.namelist())
            self.assertIn("xl/workbook.xml", names)
            self.assertIn("xl/worksheets/sheet1.xml", names)
            workbook = archive.read("xl/workbook.xml").decode("utf-8")
            for name in ("报告概览", "单机2卡", "数据明细", "用例与覆盖", "Metadata"):
                self.assertIn(name, workbook)


if __name__ == "__main__":
    unittest.main()
