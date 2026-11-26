# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from hashlib import sha256
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

import generate_cases
import group_cases
import performance_report
import run_cases


ROOT = Path(__file__).resolve().parent
DEFINITIONS = ROOT / "test_lists"
SUITES = DEFINITIONS / "suites.json"
SOURCE_REVISION = "1" * 40
RUNNER_REVISION = "2" * 40
ALL_FAMILIES = (
    "op_perf_shm", "postsil_daily_shm", "pccl_fun_rdma_shm", "register_test_shm",
    "sanity_m890p_register", "sym_test_shm", "f8_test_shm", "device_api_shm",
    "local_register_perf_shm", "symetric_perf_shm", "copyengine_perf_shm",
    "op_perf_shm_2_rack", "op_perf_rdma_shm", "commabort_test_shm", "postsil_unit_test_shm",
)
def canonical_sha256(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


class PublicGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def bundle(self, family, profile="full", nodes=None):
        return generate_cases.generate_bundle(
            DEFINITIONS / f"{family}.json", SUITES, SOURCE_REVISION, RUNNER_REVISION, profile, nodes)

    def assert_matches_source_export(self, family, expected_count):
        bundle = self.bundle(family)
        manifest = json.loads((DEFINITIONS / "source-export.json").read_text())
        expected = next(record for record in manifest["families"] if record["family"] == family)
        self.assertEqual(len(bundle["rows"]), expected_count)
        self.assertEqual(expected["case_count"], expected_count)
        self.assertEqual(canonical_sha256(bundle["rows"]), expected["source_rows_sha256"])
        self.assertEqual(bundle["generation"]["definition_sha256"], expected["definition_sha256"])
        self.assertEqual(bundle["metadata"]["xtest_version"], "not-used")

    def test_op_perf_matches_all_54_committed_rows(self):
        self.assert_matches_source_export("op_perf_shm", 54)

    def test_postsil_matches_all_3940_committed_rows_including_special_cases(self):
        self.assert_matches_source_export("postsil_daily_shm", 3940)
        rows = self.bundle("postsil_daily_shm")["rows"]
        self.assertEqual(sum(row["name"].startswith("all_reduce_ranks_") for row in rows), 2)

    def test_native_unit_matches_all_1294_committed_rows(self):
        self.assert_matches_source_export("postsil_unit_test_shm", 1294)
        rows = self.bundle("postsil_unit_test_shm")["rows"]
        self.assertEqual(sum("build/collPcclUnitTest" in row["cmd"] for row in rows), 1252)
        self.assertEqual(sum("build/hggcGraphUnitTest" in row["cmd"] for row in rows), 42)
        self.assertTrue(all("unit_test.py" not in row["cmd"] for row in rows))
        self.assertTrue(all("--auto-detect" not in row["cmd"] for row in rows))
        self.assertTrue(all("--devices=" not in row["cmd"] for row in rows))
        self.assertEqual(sum("HGGC_VISIBLE_DEVICES=" in row["cmd"] for row in rows), 717)
        allowed = {
            "collPcclUnitTest": {"coll_type", "dtype", "num_gpus", "num_iterations",
                                  "max_num_elems", "min_num_elems", "num_elems"},
            "hggcGraphUnitTest": {"num_threads", "test_config_file", "num_gpus"},
        }
        for row in rows:
            binary = next(name for name in allowed if name in row["cmd"])
            for token in shlex.split(row["cmd"]):
                if token.startswith("--"):
                    self.assertIn(token[2:].split("=", 1)[0], allowed[binary])
        smoke = self.bundle("postsil_unit_test_shm", profile="smoke")["rows"]
        self.assertEqual(len(smoke), 2)
        self.assertEqual(sum("build/collPcclUnitTest" in row["cmd"] for row in smoke), 1)
        self.assertEqual(sum("build/hggcGraphUnitTest" in row["cmd"] for row in smoke), 1)

    def test_810e_native_unit_matches_dedicated_source_export(self):
        manifest = json.loads((DEFINITIONS / "source-export-810e.json").read_text())
        bundles = {family: self.bundle(family) for family in
                   ("postsil_unit_test_810e", "all_gather_unit_test_810e")}
        for family, bundle in bundles.items():
            expected = next(record for record in manifest["families"] if record["family"] == family)
            self.assertEqual(bundle["generation"]["definition_sha256"], expected["definition_sha256"])
            self.assertEqual(canonical_sha256(bundle["rows"]), expected["source_rows_sha256"])
        combined = generate_cases.combine_bundles(list(bundles.values()))
        self.assertEqual(len(combined["rows"]), 1300)
        self.assertEqual(canonical_sha256(combined["rows"]), manifest["case_rows_sha256"])
        self.assertEqual(canonical_sha256(combined["suites"]), manifest["suites_sha256"])
        self.assertEqual(sum(record["case_count"] for record in manifest["exclusions"]), 346)
        for row in combined["rows"]:
            self.assertNotIn("unit_test.py", row["cmd"])
            self.assertNotIn("--auto-detect", row["cmd"])
            self.assertNotIn("--auto_detect=", row["cmd"])
            self.assertNotIn("$PPU_HOME", row["cmd"])
            self.assertTrue(any(binary in row["cmd"] for binary in
                                ("collPcclUnitTest", "hggcGraphUnitTest")))
        smoke = self.bundle("postsil_unit_test_810e_smoke", profile="smoke")
        self.assertEqual(len(smoke["rows"]), 2)
        self.assertEqual(sum("build/collPcclUnitTest" in row["cmd"] for row in smoke["rows"]), 1)
        self.assertEqual(sum("build/hggcGraphUnitTest" in row["cmd"] for row in smoke["rows"]), 1)

    def test_810e_native_group_allocations_match_commands(self):
        bundles = [self.bundle("postsil_unit_test_810e"), self.bundle("all_gather_unit_test_810e")]
        combined = generate_cases.combine_bundles(bundles)
        output = self.root / "native-rank-groups"
        generate_cases.write_bundle(combined, output, DEFINITIONS / "envs")
        plan = group_cases.build_plan(output, 500)
        self.assertEqual(sum(len(group["cases"]) for group in plan["groups"]), 1300)
        allocations = {}
        for group in plan["groups"]:
            allocations[group["cards_per_node"]] = allocations.get(group["cards_per_node"], 0) + len(group["cases"])
        self.assertEqual(allocations, {1: 6, 2: 130, 4: 562, 8: 602})
        group_cases.write_plan(plan, self.root / "native-rank-group-output", output)

    def test_810e_dedicated_gin_definitions_are_deferred(self):
        self.assertFalse((DEFINITIONS / "device_api_gin_810e.json").exists())
        suites = json.loads(SUITES.read_text())["suites"]
        self.assertFalse(any("device_api_gin_810e" in row["suite"] for row in suites))

    def test_multi_node_rdma_matches_all_4090_committed_rows(self):
        self.assert_matches_source_export("pccl_fun_rdma_shm", 4090)

    def test_every_other_family_matches_its_committed_rows(self):
        expected_counts = {
            "register_test_shm": 354, "sanity_m890p_register": 9, "sym_test_shm": 127,
            "f8_test_shm": 20, "device_api_shm": 52, "local_register_perf_shm": 60,
            "symetric_perf_shm": 90, "copyengine_perf_shm": 33,
            "op_perf_shm_2_rack": 36, "op_perf_rdma_shm": 18, "commabort_test_shm": 16,
        }
        for family, expected_count in expected_counts.items():
            with self.subTest(family=family):
                self.assert_matches_source_export(family, expected_count)

    def test_all_families_match_the_complete_10193_case_export_and_execution_plan(self):
        combined = generate_cases.combine_bundles([self.bundle(family) for family in ALL_FAMILIES])
        manifest = json.loads((DEFINITIONS / "source-export.json").read_text())
        self.assertEqual(canonical_sha256(combined["rows"]), manifest["case_rows_sha256"])
        self.assertEqual(canonical_sha256(combined["suites"]), manifest["suites_sha256"])
        output = self.root / "all-generated"
        generate_cases.write_bundle(combined, output, DEFINITIONS / "envs")
        performance_selection = performance_report.write_selection(
            output, "standard", self.root / "performance-cases.json")
        self.assertEqual(len(performance_selection["cases"]), 168)
        two_node_selection = performance_report.write_selection(
            output, "standard", self.root / "performance-cases-2.json", nodes=2)
        self.assertEqual(len(two_node_selection["cases"]), 39)
        single_suite_selection = performance_report.write_selection(
            output, "standard", self.root / "performance-cases-suite.json",
            suite="op_perf_shm_single_node_suite")
        self.assertEqual(len(single_suite_selection["cases"]), 27)
        plan = group_cases.build_plan(output, 500, allowed_nodes=(1, 2, 4, 8))
        self.assertEqual(len(combined["rows"]), 10193)
        self.assertEqual(len(combined["suites"]), 35)
        self.assertEqual(sum(len(group["cases"]) for group in plan["groups"]), 10184)
        self.assertEqual(len(plan["groups"]), 28)
        self.assertEqual(sum(record["case_count"] for record in plan["exclusions"]), 9)

    def test_profiles_filter_before_generating_combinations(self):
        definition = generate_cases.load_definition(DEFINITIONS / "postsil_daily_shm.json")
        self.assertEqual(len(generate_cases.generate_rows(definition, "smoke")), 1)
        self.assertEqual(len(generate_cases.generate_rows(definition, "all_reduce")), 2198)
        self.assertEqual(len(generate_cases.generate_rows(definition, "auto")), 30)
        self.assertEqual(len(generate_cases.generate_rows(definition, "fc")), 8)
        self.assertEqual(len(self.bundle("op_perf_shm", nodes=2)["rows"]), 9)
        self.assertEqual(len(self.bundle("pccl_fun_rdma_shm", nodes=2)["rows"]), 1775)
        self.assertEqual(len(self.bundle("pccl_fun_rdma_shm", nodes=4)["rows"]), 1648)
        self.assertEqual(len(self.bundle("pccl_fun_rdma_shm", nodes=8)["rows"]), 667)
        self.assertEqual(len(self.bundle("pccl_fun_rdma_shm", profile="smoke")["rows"]), 1)
        self.assertEqual(len(self.bundle("pccl_fun_rdma_shm", profile="auto")["rows"]), 6)
        with self.assertRaisesRegex(ValueError, "no node counts"):
            self.bundle("postsil_daily_shm", nodes=2)
        definition = generate_cases.load_definition(DEFINITIONS / "postsil_daily_shm.json")
        definition["profiles"]["smoke"] = {"nodes": [1], "filters": {"nproc": [1]}}
        rows = generate_cases.generate_rows(definition, "smoke")
        self.assertTrue(rows)
        self.assertTrue(all("--nproc 1" in row["cmd"] and "--devices" not in row["cmd"]
                            for row in rows))
        definition["profiles"]["smoke"]["filters"]["nproc"] = [999]
        with self.assertRaisesRegex(ValueError, "selected no Cases"):
            generate_cases.generate_rows(definition, "smoke")

    def test_cli_filters_the_combined_selection_without_requiring_every_family_to_match(self):
        output = self.root / "combined-nodes-2"
        command = [
            sys.executable, "-S", "-B", str(ROOT / "generate_cases.py"),
            "--definition", str(DEFINITIONS / "op_perf_shm.json"),
            "--definition", str(DEFINITIONS / "postsil_daily_shm.json"),
            "--suites", str(SUITES), "--env-root", str(DEFINITIONS / "envs"),
            "--profile", "full", "--nodes", "2",
            "--source-revision", SOURCE_REVISION, "--runner-revision", RUNNER_REVISION,
            "--output", str(output),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        generation = json.loads(completed.stdout)
        self.assertEqual(generation["nodes"], 2)
        self.assertEqual(generation["case_count"], 9)
        self.assertEqual([record["family"] for record in generation["families"]], ["op_perf_shm"])
        selected = run_cases.load_inputs(output, "standard")
        self.assertEqual(len(selected["cases"]), 9)
        self.assertTrue(all(case["suite_spec"]["nodes"] == 2 for case in selected["cases"]))
        case_keys = "".join(f"{case['plan']}/{case['suite']}/{case['name']}\n"
                            for case in selected["cases"])
        expected_hash = sha256(case_keys.encode()).hexdigest()
        self.assertEqual(generation["case_keys_sha256"], expected_hash)
        self.assertEqual(generation["families"][0]["case_count"], 9)
        self.assertEqual(generation["families"][0]["case_keys_sha256"], expected_hash)
        self.assertEqual({case["suite_spec"]["env"] for case in selected["cases"]}, {"op_perf.env"})

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

    def test_invalid_staged_bundle_is_not_published(self):
        bundle = self.bundle("op_perf_shm")
        bundle["suites"][0]["nodes"] = "invalid"
        output = self.root / "invalid-output"
        with self.assertRaises(ValueError):
            generate_cases.write_bundle(bundle, output, DEFINITIONS / "envs")
        self.assertFalse(output.exists())

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
            self.bundle("postsil_daily_shm", profile="smoke"),
            self.bundle("postsil_unit_test_shm", profile="smoke")])
        self.assertEqual(len(smoke["rows"]), 4)

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

    def test_maintained_environment_digests_match_checked_in_files(self):
        manifest = json.loads((DEFINITIONS / "source-export.json").read_text())
        for name, expected_hash in manifest["maintained_environments"].items():
            with self.subTest(environment=name):
                environment = DEFINITIONS / "envs" / name
                self.assertEqual(sha256(environment.read_bytes()).hexdigest(), expected_hash)

    def test_source_export_binds_every_daily_family_to_the_real_source_rows(self):
        manifest = json.loads((DEFINITIONS / "source-export.json").read_text())
        combined = generate_cases.combine_bundles([self.bundle(family) for family in ALL_FAMILIES])
        self.assertEqual(manifest["source_revision"],
                         "80738685d8b2e2066223119594805e5255f4a8c8")
        self.assertEqual(manifest["case_count"], 10193)
        self.assertEqual(manifest["suite_count"], 35)
        self.assertEqual(manifest["additional_source_revisions"]["postsil_unit_test_shm"],
                         "cefc8624ec1a2b58bc54e76eedeefae90046d84f")
        self.assertEqual(manifest["case_rows_sha256"], canonical_sha256(combined["rows"]))
        self.assertEqual(manifest["suites_sha256"], canonical_sha256(combined["suites"]))
        self.assertFalse(manifest["csv_inputs_used"])
        self.assertFalse(manifest["csv_outputs_written"])
        records = {record["family"]: record for record in manifest["families"]}
        self.assertEqual(set(records), set(ALL_FAMILIES))
        for family in ALL_FAMILIES:
            with self.subTest(family=family):
                bundle = self.bundle(family)
                provenance = bundle["generation"]["provenance"]
                self.assertEqual(provenance["source_case_count"], len(bundle["rows"]))
                self.assertEqual(provenance["source_rows_sha256"],
                                 canonical_sha256(bundle["rows"]))
                self.assertEqual(provenance["validation"], "exact-real-runner-capture")
                self.assertEqual(records[family]["definition_sha256"],
                                 sha256((DEFINITIONS / f"{family}.json").read_bytes()).hexdigest())
        smoke = generate_cases.combine_bundles([
            self.bundle("op_perf_shm", profile="smoke"),
            self.bundle("postsil_daily_shm", profile="smoke"),
            self.bundle("postsil_unit_test_shm", profile="smoke")])
        smoke_output = self.root / "group-smoke-input"
        generate_cases.write_bundle(smoke, smoke_output, DEFINITIONS / "envs")
        smoke_plan = group_cases.build_plan(smoke_output, 500, allowed_nodes=(1, 2, 4))
        self.assertEqual(len(smoke_plan["groups"]), 1)
        self.assertEqual(len(smoke_plan["groups"][0]["cases"]), 4)

    def test_runtime_does_not_read_the_legacy_case_csv(self):
        source = (ROOT / "generate_cases.py").read_text()
        self.assertNotIn("daily-full-csv", source)
        self.assertNotIn("daily-minimal-csv", source)

    def test_workflow_generates_explicit_datasets_before_grouping(self):
        workflow = (ROOT.parent.parent / ".github/workflows/pccl-daily-full.yml").read_text()
        self.assertIn('options: [smoke, full]', workflow)
        self.assertIn('.github/xtest/generate_cases.py', workflow)
        self.assertIn('--definition .github/xtest/test_lists/op_perf_shm.json', workflow)
        self.assertIn('--definition .github/xtest/test_lists/postsil_daily_shm.json', workflow)
        for family in ALL_FAMILIES:
            self.assertIn(f'.github/xtest/test_lists/{family}.json', workflow)
        self.assertIn('if [[ "$DATASET" == full ]]', workflow)
        self.assertIn("--output generated-case-input", workflow)
        self.assertIn("case-groups/generation.json", workflow)
        self.assertNotIn("GENERATED_INPUT", workflow)
        self.assertNotIn("dataset.startswith('generated-')", workflow)
        self.assertNotIn("daily-full-csv", workflow)
        self.assertNotIn("daily-minimal-csv", workflow)
        self.assertLess(workflow.index('--output generated-case-input'),
                        workflow.index('group_cases.build_plan'))


if __name__ == "__main__":
    unittest.main()
