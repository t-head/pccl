# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import tempfile
import unittest

import generate_cases
import generate_pr_cases
import group_cases
import platform_cases
import run_cases


ROOT = Path(__file__).resolve().parent
DEFINITIONS = ROOT / "test_lists"
SHA = "a" * 40


class PlatformCasesTests(unittest.TestCase):
    def bundle(self, family, profile="full", platform="m890p"):
        return generate_cases.generate_bundle(
            DEFINITIONS / f"{family}.json", DEFINITIONS / "suites.json", SHA, SHA,
            profile, platform=platform)

    def test_default_platform_preserves_shared_family_rows(self):
        bundle = self.bundle("op_perf_shm")
        rows = generate_cases.generate_rows(generate_cases.load_definition(
            DEFINITIONS / "op_perf_shm.json"))
        self.assertEqual(bundle["rows"], rows)
        self.assertEqual(bundle["generation"]["platform"], "m890p")

    def test_810e_op_perf_and_eic_have_independent_identities(self):
        profile = platform_cases.load_platform("oam810e_eic")
        raw = self.bundle("op_perf_shm")
        converted = platform_cases.apply_platforms([raw], profile)
        self.assertEqual([len(bundle["rows"]) for bundle in converted], [36, 9])
        ordinary, eic = converted
        self.assertTrue(all(row["plan"].startswith("pccl_810e") for row in ordinary["rows"]))
        self.assertTrue(all("op_perf_eic" in row["suite"] for row in eic["rows"]))
        self.assertNotEqual({(row["plan"], row["suite"]) for row in ordinary["rows"]},
                            {(row["plan"], row["suite"]) for row in eic["rows"]})
        self.assertTrue(all("ranks_32_nnodes2" in row["name"] for row in eic["rows"]))
        self.assertTrue(all("-np 32 -npernode 16" in row["cmd"] for row in eic["rows"]))

    def test_810e_rdma_preserves_the_shared_rank_and_process_layout(self):
        bundle = self.bundle("pccl_fun_rdma_shm", platform="oam810e_eic")
        self.assertEqual(len(bundle["rows"]), 1775)
        self.assertTrue(all("--nnodes 2" in row["cmd"] for row in bundle["rows"]))
        sample = next(row for row in bundle["rows"] if "ranks_16" in row["name"])
        self.assertIn("--ranks 16", sample["cmd"])
        self.assertIn("--nprocs 8", sample["cmd"])

    def test_pr_bundle_has_standard_notsm_and_two_node_coverage(self):
        bundle = generate_pr_cases.build_pr_bundle(SHA, SHA)
        self.assertEqual(len(bundle["standard"]), 7)
        self.assertEqual(len(bundle["notsm"]), 1)
        self.assertEqual({row["nodes"] for row in bundle["suites"]}, {1, 2})
        self.assertTrue(any("op_perf_eic" in row["suite"] for row in bundle["standard"]))
        self.assertTrue(all(row["plan"].startswith(("pccl_810e", "sanity_810e"))
                            for row in bundle["standard"] + bundle["notsm"]))

    def test_groups_retain_platform_selector_and_results_identity(self):
        raw = [self.bundle("op_perf_shm", "smoke")]
        converted = platform_cases.apply_platforms(
            raw, platform_cases.load_platform("oam810e_eic"), [self.bundle("op_perf_shm")])
        combined = generate_cases.combine_bundles(raw + converted)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_cases.write_bundle(combined, root / "input", DEFINITIONS / "envs")
            plan = group_cases.build_plan(root / "input", 500, allowed_nodes=(1, 2))
            self.assertEqual({group["platform"] for group in plan["groups"]},
                             {"m890p", "oam810e_eic"})
            group_cases.write_plan(plan, root / "groups", root / "input")
            manifest = json.loads((root / "groups" / "groups.json").read_text())
            selectors = {row["platform"]: row["node_selector"] for row in manifest}
            self.assertEqual(selectors["oam810e_eic"], "board-type=OAM-810E,nic-type=eic")
            self.assertEqual(selectors["m890p"], "board-type=ZW-M890P,nic-type=cx8")

    def test_ib_environment_is_inherited_and_forwarded_to_mpi_ranks(self):
        direct = run_cases._with_ib_environment("mpirun --allow-run-as-root -np 2 hostname")
        self.assertIn("mpirun -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 ", direct)
        wrapped = run_cases._with_ib_environment(
            "python3 run_test.py --mpi-options='-x PATH --hostfile hosts'")
        self.assertIn("--mpi-options='-x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 ", wrapped)


if __name__ == "__main__":
    unittest.main()
