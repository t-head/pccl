# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
from pathlib import Path
import tempfile
import unittest

import group_cases
import run_cases


class GroupCasesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input = self.root / "input"
        (self.input / "envs").mkdir(parents=True)
        (self.input / "envs" / "common.env").write_text("PATH=/usr/bin\n")
        (self.input / "envs" / "other.env").write_text("PATH=/bin\n")
        (self.input / "export.json").write_text(json.dumps({
            "format_version": 1, "source_revision": "1" * 40,
            "runner_revision": "2" * 40, "xtest_version": "0.12.0"}))
        self.suites = []
        self.cases = []

    def tearDown(self):
        self.temp.cleanup()

    def add(self, suite, name, nodes=1, env="common.env", plan="plan"):
        if not any(row["suite"] == suite for row in self.suites):
            self.suites.append({"plan": plan, "suite": suite, "nodes": str(nodes),
                                "cards_per_node": "8", "timeout": "300", "env": env})
        self.cases.append({"plan": plan, "suite": suite, "name": name,
                           "check": "exit_code", "cmd": "true"})

    def write(self):
        for filename, fields, rows in (("suites.csv", run_cases.SUITE_FIELDS, self.suites),
                                      ("standard.csv", run_cases.CASE_FIELDS, self.cases),
                                      ("notsm.csv", run_cases.CASE_FIELDS, [])):
            with (self.input / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)

    def test_same_environment_is_grouped_before_count_based_sharding(self):
        for suite, name in (("suite_a", "a"), ("suite_b", "b"), ("suite_a", "c"), ("suite_b", "d")):
            self.add(suite, name)
        self.write()
        groups = group_cases.build_groups(self.input, max_cases=3)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["nodes"], 1)
        self.assertEqual(len(groups[0]["cases"]), 3)
        self.assertEqual({(case["plan"], case["suite"]) for case in groups[0]["cases"]},
                         {("plan", "suite_a"), ("plan", "suite_b")})
        self.assertEqual([case["name"] for case in groups[0]["cases"]], ["a", "c", "b"])
        output = self.root / "groups"
        group_cases.write_groups(groups, output, self.input)
        with self.assertRaisesRegex(ValueError, "already exists"):
            group_cases.write_groups(groups, output, self.input)
        manifest = json.loads((output / "groups.json").read_text())
        self.assertEqual([entry["case_count"] for entry in manifest], [3, 1])
        for group in groups:
            selected = run_cases.load_inputs(output / group["id"], group["profile"], group["nodes"])
            self.assertEqual({(case["plan"], case["suite"], case["name"]) for case in selected["cases"]},
                             {(case["plan"], case["suite"], case["name"]) for case in group["cases"]})

    def test_different_environments_are_never_merged(self):
        self.add("suite_a", "a", env="common.env")
        self.add("suite_b", "b", env="other.env")
        self.write()
        groups = group_cases.build_groups(self.input, max_cases=500)
        self.assertEqual(len(groups), 2)
        self.assertEqual({group["env"] for group in groups}, {"common.env", "other.env"})

    def test_cross_rack_suites_are_isolated_and_carry_exact_allocation(self):
        self.add("op_perf_shm_2_rack_icn8_suite", "cross", nodes=2,
                 plan="pccl_shm_op_perf_2_rack_icn8_plan")
        self.add("ordinary_suite", "ordinary", nodes=2)
        self.write()
        groups = group_cases.build_groups(self.input, max_cases=500)
        self.assertEqual(len(groups), 2)
        cross = next(group for group in groups if group["cross_rack"])
        ordinary = next(group for group in groups if not group["cross_rack"])
        self.assertEqual((cross["rack_policy"], cross["rack_count"]), ("spread", 2))
        self.assertEqual((ordinary["rack_policy"], ordinary["rack_count"]), ("same", ""))
        self.assertEqual([case["name"] for case in cross["cases"]], ["cross"])
        self.assertEqual([case["name"] for case in ordinary["cases"]], ["ordinary"])
        output = self.root / "groups"
        group_cases.write_groups(groups, output, self.input)
        manifest = json.loads((output / "groups.json").read_text())
        self.assertEqual({(entry["rack_policy"], entry["rack_count"]) for entry in manifest},
                         {("same", ""), ("spread", 2)})

    def test_unknown_or_tampered_rack_allocations_are_rejected(self):
        self.add("op_perf_shm_2_rack_unknown_suite", "unknown", nodes=2)
        self.write()
        with self.assertRaisesRegex(ValueError, "no reviewed rack policy"):
            group_cases.build_groups(self.input, max_cases=500)
        for policy, count in (("invalid", ""), ("same", 2), ("spread", None),
                              ("spread", 1), ("spread", 3), ("spread", "2"), ("", 2)):
            with self.subTest(policy=policy, count=count), self.assertRaisesRegex(ValueError, "rack_"):
                group_cases._validate_rack_policy(policy, count, 2)

        self.suites = []
        self.cases = []
        self.add("ordinary", "ordinary", nodes=2)
        self.write()
        group = group_cases.build_groups(self.input, max_cases=500)[0]
        group["cross_rack"] = True
        group["rack_policy"] = "spread"
        group["rack_count"] = 2
        with self.assertRaisesRegex(ValueError, "rack metadata"):
            group_cases.write_groups([group], self.root / "tampered", self.input)

    def test_debug_filters_do_not_create_manual_shards(self):
        self.add("suite_a", "a")
        self.add("suite_a", "b")
        self.add("suite_b", "c", nodes=2)
        self.write()
        groups = group_cases.build_groups(self.input, max_cases=500, nodes=1, suite="suite_a")
        self.assertEqual(len(groups), 1)
        self.assertEqual([case["name"] for case in groups[0]["cases"]], ["a", "b"])
        with self.assertRaisesRegex(ValueError, "No Cases"):
            group_cases.build_groups(self.input, max_cases=500, nodes=1, suite="missing")
        with self.assertRaisesRegex(ValueError, "positive integer"):
            group_cases.build_groups(self.input, max_cases=0)

    def test_resource_policy_has_exact_selected_and_excluded_coverage(self):
        self.add("single", "a")
        self.add("single", "b")
        self.add("two", "c", nodes=2)
        self.add("four", "d", nodes=4)
        self.add("eight", "e", nodes=8)
        self.add("eight", "f", nodes=8)
        self.add("sixteen", "g", nodes=16)
        self.write()
        plan = group_cases.build_plan(self.input, max_cases=500, allowed_nodes=(1, 2, 4))
        self.assertEqual(sum(len(group["cases"]) for group in plan["groups"]), 4)
        self.assertEqual([(record["nodes"], record["case_count"])
                          for record in plan["exclusions"]], [(8, 2), (16, 1)])
        output = self.root / "groups"
        group_cases.write_plan(plan, output, self.input)
        coverage = json.loads((output / "execution-coverage.json").read_text())
        self.assertEqual(coverage["allowed_nodes"], [1, 2, 4])
        self.assertEqual(coverage["selected_cases"], 4)
        self.assertEqual(coverage["excluded_cases"], 3)
        self.assertEqual(coverage["total_cases"], 7)
        self.assertTrue(all(record["reason"] == group_cases.POLICY_EXCLUSION
                            for record in coverage["exclusions"]))
        with self.assertRaisesRegex(ValueError, "disabled by daily resource policy"):
            group_cases.build_plan(self.input, 500, nodes=8, allowed_nodes=(1, 2, 4))
        with self.assertRaisesRegex(ValueError, r"allowed_nodes.*61"):
            group_cases.build_plan(self.input, 500, allowed_nodes=(1, 2, 4, 8, 61))
        with self.assertRaisesRegex(ValueError, "Node filter must be one of"):
            group_cases.build_plan(self.input, 500, nodes=61)


if __name__ == "__main__":
    unittest.main()
