# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import generate_cases
import group_cases
import run_cases


ROOT = Path(__file__).resolve().parent
DEFINITIONS = ROOT / "test_lists"
SUITES = DEFINITIONS / "suites.json"
SOURCE_REVISION = "1" * 40
RUNNER_REVISION = "2" * 40


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sort_rows(rows):
    return sorted(rows, key=lambda row: (row["plan"], row["suite"], row["name"]))


class PublicGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def bundle(self, family, profile="full", nodes=None):
        return generate_cases.generate_bundle(
            DEFINITIONS / f"{family}.json", SUITES, SOURCE_REVISION, RUNNER_REVISION, profile, nodes)

    def assert_matches_committed_export(self, family, expected_count):
        bundle = self.bundle(family)
        suite_names = {row["suite"] for row in bundle["rows"]}
        expected = sort_rows(row for row in read_csv(ROOT / "daily-full-csv/standard.csv")
                             if row["suite"] in suite_names)
        self.assertEqual(len(bundle["rows"]), expected_count)
        self.assertEqual(bundle["rows"], expected)
        expected_suites = []
        for row in read_csv(ROOT / "daily-full-csv/suites.csv"):
            if row["suite"] in suite_names:
                expected_suites.append({**row, "nodes": int(row["nodes"]),
                                        "cards_per_node": int(row["cards_per_node"]),
                                        "timeout": int(row["timeout"])})
        self.assertEqual(bundle["suites"], sorted(expected_suites,
                                                  key=lambda row: (row["plan"], row["suite"])))
        self.assertEqual(bundle["metadata"]["xtest_version"], "not-used")

    def test_op_perf_matches_all_54_committed_rows(self):
        self.assert_matches_committed_export("op_perf_shm", 54)

    def test_postsil_matches_all_3940_committed_rows_including_special_cases(self):
        self.assert_matches_committed_export("postsil_daily_shm", 3940)
        rows = self.bundle("postsil_daily_shm")["rows"]
        self.assertEqual(sum(row["name"].startswith("all_reduce_ranks_") for row in rows), 2)

    def test_profiles_filter_before_generating_combinations(self):
        definition = generate_cases.load_definition(DEFINITIONS / "postsil_daily_shm.json")
        self.assertEqual(len(generate_cases.generate_rows(definition, "smoke")), 1)
        self.assertEqual(len(generate_cases.generate_rows(definition, "all_reduce")), 2198)
        self.assertEqual(len(generate_cases.generate_rows(definition, "auto")), 30)
        self.assertEqual(len(generate_cases.generate_rows(definition, "fc")), 8)
        self.assertEqual(len(self.bundle("op_perf_shm", nodes=2)["rows"]), 9)
        with self.assertRaisesRegex(ValueError, "no node counts"):
            self.bundle("postsil_daily_shm", nodes=2)

    def test_temporary_output_is_deterministic_and_public_consumer_compatible(self):
        outputs = []
        for index in (1, 2):
            bundle = self.bundle("postsil_daily_shm", profile="smoke")
            output = self.root / f"generated-{index}"
            generate_cases.write_bundle(bundle, output, DEFINITIONS / "envs")
            outputs.append(output)
            selected = run_cases.load_inputs(output, "standard")
            self.assertEqual(len(selected["cases"]), 1)
            self.assertEqual(selected["cases"][0]["suite_spec"]["timeout"], 3600)
            self.assertEqual(selected["metadata"]["xtest_version"], "not-used")
            self.assertFalse(list(output.rglob("*.py")))
            self.assertEqual((output / "envs/op_perf.env").read_bytes(),
                             (DEFINITIONS / "envs/op_perf.env").read_bytes())
        for relative in ("standard.csv", "suites.csv", "notsm.csv", "export.json", "generation.json",
                         "envs/op_perf.env"):
            self.assertEqual((outputs[0] / relative).read_bytes(), (outputs[1] / relative).read_bytes())

    def test_generation_manifest_binds_definition_and_exact_case_keys(self):
        definition_path = DEFINITIONS / "op_perf_shm.json"
        bundle = self.bundle("op_perf_shm")
        manifest = bundle["generation"]
        keys = "".join(f"{row['plan']}/{row['suite']}/{row['name']}\n" for row in bundle["rows"])
        self.assertEqual(manifest["case_count"], 54)
        self.assertEqual(manifest["definition_sha256"], sha256(definition_path.read_bytes()).hexdigest())
        self.assertEqual(manifest["suite_definitions_sha256"], sha256(SUITES.read_bytes()).hexdigest())
        self.assertEqual(manifest["case_keys_sha256"], sha256(keys.encode()).hexdigest())

    def test_combined_generation_covers_both_families_without_csv_inputs(self):
        bundle = generate_cases.combine_bundles([
            self.bundle("op_perf_shm"), self.bundle("postsil_daily_shm")])
        self.assertEqual(bundle["generation"]["case_count"], 3994)
        self.assertEqual(len(bundle["generation"]["families"]), 2)
        self.assertEqual(len(bundle["rows"]), 3994)
        output = self.root / "combined"
        generate_cases.write_bundle(bundle, output, DEFINITIONS / "envs")
        self.assertEqual(len(run_cases.load_inputs(output, "standard")["cases"]), 3994)
        smoke = generate_cases.combine_bundles([
            self.bundle("op_perf_shm", profile="smoke"),
            self.bundle("postsil_daily_shm", profile="smoke")])
        self.assertEqual(len(smoke["rows"]), 2)

    def test_generated_bundle_uses_existing_grouping_and_resource_policy(self):
        bundle = generate_cases.combine_bundles([
            self.bundle("op_perf_shm"), self.bundle("postsil_daily_shm")])
        output = self.root / "group-input"
        generate_cases.write_bundle(bundle, output, DEFINITIONS / "envs")
        plan = group_cases.build_plan(output, 500, allowed_nodes=(1, 2, 4))
        self.assertEqual(sum(len(group["cases"]) for group in plan["groups"]), 3985)
        self.assertEqual(len(plan["groups"]), 10)
        self.assertEqual(sum(record["case_count"] for record in plan["exclusions"]), 9)
        self.assertEqual(plan["exclusions"][0]["nodes"], 8)
        smoke = generate_cases.combine_bundles([
            self.bundle("op_perf_shm", profile="smoke"),
            self.bundle("postsil_daily_shm", profile="smoke")])
        smoke_output = self.root / "group-smoke-input"
        generate_cases.write_bundle(smoke, smoke_output, DEFINITIONS / "envs")
        smoke_plan = group_cases.build_plan(smoke_output, 500, allowed_nodes=(1, 2, 4))
        self.assertEqual(len(smoke_plan["groups"]), 1)
        self.assertEqual(len(smoke_plan["groups"][0]["cases"]), 2)

    def test_runtime_does_not_read_the_legacy_case_csv(self):
        source = (ROOT / "generate_cases.py").read_text()
        self.assertNotIn("daily-full-csv", source)
        self.assertNotIn("daily-minimal-csv", source)

    def test_workflow_generates_explicit_datasets_before_grouping(self):
        workflow = (ROOT.parent.parent / ".github/workflows/m890p-daily-full.yml").read_text()
        self.assertIn('options: [smoke, full, generated-smoke, generated-full]', workflow)
        self.assertIn('tests/xtest/generate_cases.py', workflow)
        self.assertIn('--definition tests/xtest/test_lists/op_perf_shm.json', workflow)
        self.assertIn('--definition tests/xtest/test_lists/postsil_daily_shm.json', workflow)
        self.assertIn("--output generated-case-input", workflow)
        self.assertIn("case-groups/generation.json", workflow)
        self.assertLess(workflow.index('--output generated-case-input'),
                        workflow.index('group_cases.build_plan'))


if __name__ == "__main__":
    unittest.main()
