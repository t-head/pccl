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

    def add(self, suite, name, nodes=1, env="common.env"):
        if not any(row["suite"] == suite for row in self.suites):
            self.suites.append({"plan": "plan", "suite": suite, "nodes": str(nodes),
                                "cards_per_node": "8", "timeout": "300", "env": env})
        self.cases.append({"plan": "plan", "suite": suite, "name": name,
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


if __name__ == "__main__":
    unittest.main()
