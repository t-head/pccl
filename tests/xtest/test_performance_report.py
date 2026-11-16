# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

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

    def tearDown(self):
        self.temporary.cleanup()

    def collect(self):
        return performance_report.collect_report(
            self.selection, self.groups, self.coverage, self.results,
            {"run_id": "123", "dataset": "generated-full"})

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
        text = (HEADER + zero + zero + performance_log().removeprefix(HEADER)
                + "262144 262144 half sum -1[na131g][interleaved-log]\n")
        text = text.replace("11.5 2.25 3.5 0", "11.5 2.25 3.5 N/A", 1)
        with self.assertRaises(ValueError):
            run_cases.parse_performance(text)
        log = self.root / "tolerant.log"
        log.write_text(text)
        parsed = performance_report._parse_case(
            self.case1, log, performance_report.expected_sizes(self.case1["cmd"]))
        self.assertEqual(parsed["data_status"], "VALID")
        self.assertEqual(parsed["valid_out_of_place"], 34)
        self.assertEqual(parsed["missing_out_of_place"], 0)
        self.assertEqual(len(parsed["rows"]), 68)
        self.assertEqual(sum(row["metric_status"] == "UNSUPPORTED" for row in parsed["rows"]), 1)
        self.assertEqual(sum(row["size"] == 0 for row in parsed["rows"]), 4)

    def test_workflow_publishes_a_complete_report_with_a_pinned_writer(self):
        workflow = (Path(__file__).resolve().parents[2] /
                    ".github/workflows/m890p-daily-full.yml").read_text()
        self.assertIn("performance-cases.json", workflow)
        self.assertIn("performance_report.write_selection", workflow)
        self.assertIn("performance_report.py report", workflow)
        self.assertIn("--results-root /wl_nas/devops/pccl-artifact-cache-", workflow)
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
