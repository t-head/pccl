# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from urllib.parse import urlunsplit
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

    def collect(self, platform=None):
        return performance_report.collect_report(
            self.selection, self.groups, self.coverage, self.results,
            {"run_id": "123", "dataset": "full"}, self.reconciliation, platform=platform)

    def add_810e_results(self):
        case = make_case("test_op_all_reduce_perf_ranks_2_nnodes1_810e",
                         "op_perf_810e_single_node_suite")
        case["plan"] = "pccl_810e_plan"
        excluded = {**case, "name": "test_op_all_reduce_perf_ranks_32_nnodes2_810e", "nodes": 2}
        correctness = {"plan": case["plan"], "suite": "correctness_810e_suite", "name": "function"}
        cases = self.cases + [case, excluded]
        self.selection.write_text(json.dumps({"schema_version": 1, "profile": "standard", "cases": cases}))
        groups = json.loads(self.groups.read_text())
        groups[0]["platform"] = "m890p"
        groups.append({"id": "g002", "platform": "oam810e_eic",
                       "case_keys": [list(performance_report._key(c)) for c in (case, correctness)]})
        self.groups.write_text(json.dumps(groups))
        self.coverage.write_text(json.dumps({
            "total_cases": 6, "selected_cases": 4, "excluded_cases": 2, "allowed_nodes": [1, 2],
            "exclusions": [{"platform": platform, "case_count": 1,
                            "case_keys": [list(performance_report._key(c))]}
                           for platform, c in (("m890p", self.case3), ("oam810e_eic", excluded))]}))
        rows = []
        for c, status in ((case, "PASSED"), (correctness, "FAILED")):
            directory = self.results / "g002" / Path(*performance_report._key(c))
            directory.mkdir(parents=True)
            row = {**c, "check": "exit_code", "status": status, "return_code": 0 if status == "PASSED" else 1,
                   "duration_seconds": 1, "error": "" if status == "PASSED" else "Command exited with status 1",
                   "raw_log": "raw.log"}
            row = {field: row[field] for field in run_cases.RESULT_FIELDS}
            (directory / "raw.log").write_text(performance_log())
            (directory / "result.json").write_text(json.dumps(row))
            rows.append(row)
        with (self.results / "g002/results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=run_cases.RESULT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        (self.results / "g002/build.json").write_text(json.dumps({"artifact_sha256": "6" * 64}))
        (self.reconciliation / "g002").mkdir()
        (self.reconciliation / "g002/reconciliation.json").write_text(json.dumps({
            "ok": False, "errors": ["/".join(performance_report._key(correctness)) + ": FAILED: command failed"]}))
        return case, excluded

    def test_platform_reports_have_independent_metrics_and_execution_totals(self):
        self.add_810e_results()
        for platform, expected_counts in (("m890p", {"FAILED": 1, "PASSED": 1}),
                                          ("oam810e_eic", {"FAILED": 1, "PASSED": 1})):
            with self.subTest(platform=platform):
                report = self.collect(platform)
                self.assertEqual(report["metadata"]["platform"], platform)
                self.assertEqual(report["metadata"]["platforms"], platform)
                self.assertTrue(all(row["platform"] == platform for row in report["coverage"] + report["metrics"]))
                self.assertEqual(report["summary"]["execution"], {
                    "total_cases": 3, "selected_cases": 2, "excluded_cases": 1,
                    "result_counts": expected_counts, "selected_result_count": 2, "missing_selected_results": 0})
                self.assertEqual(report["summary"]["performance_cases"], 3 if platform == "m890p" else 2)
                self.assertEqual(report["summary"]["metric_rows"], 64)
                self.assertEqual(report["summary"]["performance_total"]["valid"], 1)
        combined = self.collect()
        self.assertEqual(combined["summary"]["execution"]["total_cases"], 6)
        self.assertEqual(combined["summary"]["metric_rows"], 128)

    def test_platform_reports_preserve_excluded_and_missing_results(self):
        self.add_810e_results()
        (self.results / "g002/results.csv").unlink()
        for path in (self.results / "g002").rglob("result.json"):
            path.unlink()
        report = self.collect("oam810e_eic")
        self.assertEqual(report["summary"]["execution"]["result_counts"], {})
        self.assertEqual(report["summary"]["execution"]["missing_selected_results"], 2)
        self.assertEqual({row["execution_status"] for row in report["coverage"]}, {"EXCLUDED", "NO_RESULT"})
        self.assertEqual(report["summary"]["metric_rows"], 0)
        self.assertEqual(self.collect("m890p")["summary"]["metric_rows"], 64)

    def test_excluded_platform_without_groups_still_has_coverage(self):
        self.add_810e_results()
        groups = json.loads(self.groups.read_text())
        self.groups.write_text(json.dumps(groups[:1]))
        report = self.collect("oam810e_eic")
        self.assertEqual(report["metadata"]["platforms"], "oam810e_eic")
        self.assertEqual(report["summary"]["execution"]["selected_cases"], 0)
        self.assertEqual(report["summary"]["execution"]["excluded_cases"], 1)
        self.assertEqual(report["summary"]["performance_cases"], 2)
        self.assertEqual(report["summary"]["metric_rows"], 0)

    def test_unknown_report_platform_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported report platform"):
            self.collect("invalid")

    def test_report_step_runs_every_selected_platform_even_after_failure(self):
        workflow = (Path(__file__).resolve().parents[2] /
                    ".github/workflows/pccl-daily-full.yml").read_text()
        step = workflow.split("      - id: performance-report\n", 1)[1].split(
            "      - id: performance-upload\n", 1)[0]
        script = step.split("        run: |\n", 1)[1]
        script = "\n".join(line[10:] for line in script.splitlines())
        script = re.sub(r"\$\{\{.*?\}\}", "123", script)
        bin_directory = self.root / "mock-bin"
        bin_directory.mkdir()
        python = bin_directory / "python3"
        python.write_text('#!/bin/sh\n'
                          'while [ "$#" -gt 0 ]; do\n'
                          '  if [ "$1" = --platform ]; then shift; platform=$1; break; fi\n'
                          '  shift\n'
                          'done\n'
                          'printf "%s\\n" "$platform" >> "$CALLS"\n'
                          '[ "$platform" != "$FAIL_PLATFORM" ]\n')
        python.chmod(0o755)
        for selected, fail, expected in (("all", "m890p", ["m890p", "oam810e_eic"]),
                                        ("all", "", ["m890p", "oam810e_eic"]),
                                        ("m890p", "", ["m890p"]),
                                        ("oam810e_eic", "", ["oam810e_eic"])):
            with self.subTest(selected=selected, fail=fail):
                calls = self.root / "calls"
                calls.write_text("")
                result = subprocess.run(["/bin/sh", "-e", "-c", script],
                                        env=dict(os.environ, REPORT_PLATFORM=selected, FAIL_PLATFORM=fail,
                                                 CALLS=str(calls), PATH=str(bin_directory) + os.pathsep + os.environ['PATH']),
                                        capture_output=True, text=True, timeout=5)
                self.assertEqual(calls.read_text().splitlines(), expected)
                self.assertEqual(result.returncode, 1 if fail else 0, result.stderr)

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

    def test_mixed_platform_report_and_exclusions_keep_platform_identity(self):
        groups = json.loads(self.groups.read_text())
        groups[0]["platform"] = "oam810e_eic"
        groups.append({"id": "g002", "platform": "m890p", "case_keys": []})
        self.groups.write_text(json.dumps(groups))
        self.case3["plan"] = "pccl_810e_op_perf_single_node1_plan"
        self.selection.write_text(json.dumps({
            "schema_version": 1, "profile": "standard", "cases": self.cases}))
        report = self.collect()
        self.assertEqual(report["metadata"]["platforms"], "m890p,oam810e_eic")
        rows = {row["name"]: row for row in report["coverage"]}
        self.assertEqual(rows[self.case1["name"]]["platform"], "oam810e_eic")
        self.assertEqual(rows[self.case1["name"]]["topology"], "oam810e_eic:单机2卡")
        self.assertEqual(rows[self.case3["name"]]["platform"], "oam810e_eic")
        self.assertEqual(rows[self.case3["name"]]["topology"], "oam810e_eic:单机8卡")

    def test_810e_local_register_category_and_url_userinfo_redaction(self):
        case = make_case("local", "local_register_perf_810e_single_node_suite", "performance")
        self.assertEqual(case["category"], "local-register 性能")
        # Build synthetic userinfo separately to avoid a secret-scan false positive.
        url = urlunsplit(("https", "user:password@example.com", "/sdk.bin",
                          "signature=secret", "fragment"))
        self.assertEqual(performance_report.public_hpn_sdk_url(url), "https://example.com/sdk.bin")

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
                    ".github/workflows/pccl-daily-full.yml").read_text()
        self.assertIn("performance-cases.json", workflow)
        self.assertNotIn("GENERATE_NODES", workflow)
        self.assertIn("import xlsxwriter; assert xlsxwriter.__version__", workflow)
        self.assertIn("discover -s pccl/tests/xtest -p 'test_performance_report.py'", workflow)
        self.assertIn("performance_report.write_selection", workflow)
        self.assertIn("performance_report.py report", workflow)
        self.assertIn('--image "$VALIDATION_IMAGE"', workflow)
        self.assertIn('--hpn-sdk-url "$HPN_SDK_URL"', workflow)
        self.assertIn("--results-root /wl_nas/devops/pccl-artifact-cache-", workflow)
        self.assertIn("--reconciliation-root public-results", workflow)
        self.assertIn("name: pccl-daily-performance-report", workflow)
        self.assertIn('--platform "$platform"', workflow)
        self.assertIn('PCCL_${report_name}_daily_perf_', workflow)
        self.assertIn('path: public-results/PCCL_*_daily_perf_*.xlsx', workflow)
        self.assertIn('set -- m890p oam810e_eic', workflow)
        self.assertIn('--hpn-sdk-url "$HPN_SDK_URL" || status=1', workflow)
        self.assertIn("XlsxWriter==3.2.5", workflow)
        self.assertIn("PYTHONPATH=deps/xlsxwriter-3.2.5-py3-none-any.whl", workflow)
        self.assertNotIn("--target report-python", workflow)
        self.assertIn("4f4824234e1eaf9d95df9a8fe974585ff91d0f5e3d3f12ace5b71e443c1c6abd", workflow)
        self.assertIn("'performance-report': '${{ steps.performance-report.outcome }}'", workflow)
        self.assertIn("name: m890p-daily-full-python-deps", workflow)
        source = (Path(__file__).resolve().parent / "performance_report.py").read_text()
        self.assertIn("run_cases.parse_performance(", source)

    def test_report_cli_retains_image_and_hpn_sdk_url(self):
        base = ['performance_report.py', 'report', '--selection', 'selection', '--groups', 'groups',
                '--coverage', 'coverage', '--results-root', 'results', '--reconciliation-root',
                'reconciliation', '--output', 'report.xlsx', '--run-id', '123', '--dataset', 'smoke',
                '--source-revision', 'a' * 40, '--runner-revision', 'b' * 40,
                '--image', 'registry.example/ppu:test']
        for url in ('', 'https://zone-ci.oss-cn-hangzhou.aliyuncs.com/hpn-sdk/sdk.bin'):
            with self.subTest(url=url):
                arguments = base + (['--hpn-sdk-url', url] if url else [])
                with mock.patch.object(sys, 'argv', arguments):
                    with mock.patch.object(performance_report, 'collect_report',
                                           return_value={'summary': {}}) as collect:
                        with mock.patch.object(performance_report, 'write_workbook'):
                            performance_report.main()
                metadata = collect.call_args.args[4]
                self.assertEqual(metadata['image'], 'registry.example/ppu:test')
                self.assertEqual(metadata['hpn_sdk_url'], url or 'image-builtin')
                self.assertIsNone(collect.call_args.kwargs['platform'])

    def test_report_cli_redacts_hpn_sdk_url_credentials(self):
        url = 'https://zone-ci.oss-cn-hangzhou.aliyuncs.com/hpn-sdk/sdk.bin?Signature=secret'
        self.assertEqual(performance_report.public_hpn_sdk_url(url), url.split('?', 1)[0])
        self.assertEqual(performance_report.public_hpn_sdk_url(''), 'image-builtin')
        self.assertEqual(performance_report.public_hpn_sdk_url('https://[invalid'), 'invalid-url')

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

    @unittest.skipUnless(importlib.util.find_spec("xlsxwriter"), "XlsxWriter is not installed")
    def test_platform_cli_generates_two_isolated_valid_workbooks(self):
        case, excluded = self.add_810e_results()
        for platform, label in (("m890p", "M890P"), ("oam810e_eic", "810E")):
            with self.subTest(platform=platform):
                output = self.root / f"PCCL_{label}_daily_perf_123.xlsx"
                result = subprocess.run([
                    sys.executable, "-S", "-B", str(Path(performance_report.__file__)), "report",
                    "--selection", str(self.selection), "--groups", str(self.groups),
                    "--coverage", str(self.coverage), "--results-root", str(self.results),
                    "--reconciliation-root", str(self.reconciliation), "--output", str(output),
                    "--run-id", "123", "--dataset", "full", "--source-revision", "a" * 40,
                    "--runner-revision", "b" * 40, "--platform", platform,
                ], capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                summary = json.loads(result.stdout)
                self.assertEqual(summary["execution"]["total_cases"], 3)
                self.assertEqual(summary["metric_rows"], 64)
                with zipfile.ZipFile(output) as archive:
                    self.assertIsNone(archive.testzip())
                    self.assertIn("xl/workbook.xml", archive.namelist())
                    strings = archive.read("xl/sharedStrings.xml").decode("utf-8")
                    self.assertIn(f"PCCL {label} daily", strings)
                    expected = self.cases if platform == "m890p" else (case, excluded)
                    other = (case, excluded) if platform == "m890p" else self.cases
                    for row in expected:
                        self.assertIn(row["name"], strings)
                    for row in other:
                        self.assertNotIn('>' + row["name"] + '<', strings)


if __name__ == "__main__":
    unittest.main()
