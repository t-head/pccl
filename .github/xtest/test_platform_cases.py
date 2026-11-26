# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
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
        self.assertEqual([len(bundle["rows"]) for bundle in converted], [27, 9])
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

    def test_810e_postsil_explicit_cases_preserve_rank_allocations(self):
        bundle = self.bundle("postsil_daily_shm", platform="oam810e_eic")
        for ranks in (4, 16):
            with self.subTest(ranks=ranks):
                row = next(row for row in bundle["rows"]
                           if row["name"] == f"all_reduce_ranks_{ranks}_nnodes1_postsil_daily_810e")
                arguments = shlex.split(row["cmd"])
                for option in ("-np", "-npernode"):
                    self.assertEqual(arguments[arguments.index(option) + 1], str(ranks))
                suite = next(suite for suite in bundle["suites"]
                             if (suite["plan"], suite["suite"]) == (row["plan"], row["suite"]))
                self.assertEqual(platform_cases.allocation_cards_per_node(
                    {**row, "suite_spec": suite}), ranks)

    def test_810e_excludes_all_fp8_cases(self):
        for family in ("local_register_perf_shm", "symetric_perf_shm"):
            bundle = self.bundle(family, platform="oam810e_eic")
            self.assertEqual(len(bundle["rows"]), 27)
            self.assertFalse(any("f8e4m3" in row["cmd"] or "f8e5m2" in row["cmd"]
                                 for row in bundle["rows"]))

    def test_810e_excludes_gin_cases_inherited_from_shared_device_api(self):
        bundle = self.bundle("device_api_shm", platform="oam810e_eic")
        self.assertEqual(len(bundle["rows"]), 28)
        self.assertFalse(any("GIN_TYPE=" in row["cmd"] for row in bundle["rows"]))
        m890p = self.bundle("device_api_shm")
        self.assertEqual(len(m890p["rows"]), 52)
        self.assertEqual(sum("GIN_TYPE=" in row["cmd"] for row in m890p["rows"]), 24)

    def test_pr_bundle_has_standard_notsm_and_two_node_coverage(self):
        bundle = generate_pr_cases.build_pr_bundle(SHA, SHA)
        self.assertEqual(len(bundle["standard"]), 68)
        self.assertEqual(len(bundle["notsm"]), 9)
        self.assertFalse(any("GIN_TYPE=" in row["cmd"]
                             for row in bundle["standard"] + bundle["notsm"]))
        self.assertEqual({row["nodes"] for row in bundle["suites"]}, {1, 2})
        self.assertEqual({row["cards_per_node"] for row in bundle["suites"]}, {2, 4})
        four_card_suites = {(row["plan"], row["suite"]) for row in bundle["suites"]
                            if row["cards_per_node"] == 4}
        self.assertEqual(four_card_suites,
                         {("pccl_810e_postsil_unit_test_4card_single_node1_plan",
                           "postsil_unit_test_810e_4card_single_node_suite")})
        graph_case = next(row for row in bundle["standard"]
                          if "GraphUnitTest" in row["cmd"]
                          and not row["name"].startswith("checkin_"))
        self.assertIn("HGGC_VISIBLE_DEVICES=0,1", graph_case["cmd"])
        self.assertNotIn("HGGC_VISIBLE_DEVICES=0,3", graph_case["cmd"])
        self.assertIn("devices_0_1", graph_case["name"])
        two_node = [row for row in bundle["standard"] if "nnodes2" in row["name"]]
        self.assertEqual(len(two_node), 2)
        self.assertTrue(all(row["check"] == "validation_summary" for row in two_node))
        self.assertTrue(any("op_perf_eic" in row["suite"] for row in bundle["standard"]))
        self.assertTrue(all(row["plan"].startswith(("pccl_810e", "sanity_810e"))
                            for row in bundle["standard"] + bundle["notsm"]))
        eic = next(row for row in bundle["standard"] if "op_perf_eic" in row["suite"])
        self.assertIn("ranks_4_nnodes2", eic["name"])
        self.assertIn("-np 4 -npernode 2", eic["cmd"])
        rdma = next(row for row in bundle["standard"] if "pccl_fun_rdma" in row["suite"])
        self.assertIn("ranks_4", rdma["name"])
        self.assertIn("process_2", rdma["name"])
        self.assertIn("--ranks 4", rdma["cmd"])
        self.assertIn("--nprocs 2", rdma["cmd"])
        single_op = [row for row in bundle["standard"]
                     if row["suite"] == "op_perf_810e_single_node_suite"]
        self.assertEqual(len(single_op), 5)
        self.assertTrue(all("ranks_2_nnodes1" in row["name"] for row in single_op))
        standard_register = [row for row in bundle["standard"]
                             if row["suite"] == "sanity_810e_register_suite"]
        self.assertEqual(len(standard_register), 9)
        self.assertEqual({row["name"] for row in standard_register},
                         {row["name"] for row in bundle["notsm"]})
        native = [row for row in bundle["standard"]
                  if "collPcclUnitTest" in row["cmd"] or "hggcGraphUnitTest" in row["cmd"]]
        self.assertEqual(len(native), 46)
        self.assertTrue(all("unit_test.py" not in row["cmd"] for row in native))
        two_card_suite = ("pccl_810e_postsil_unit_test_single_node1_plan",
                          "postsil_unit_test_810e_single_node_suite")
        four_card_suite = ("pccl_810e_postsil_unit_test_4card_single_node1_plan",
                           "postsil_unit_test_810e_4card_single_node_suite")
        self.assertEqual(Counter((row["plan"], row["suite"]) for row in native),
                         Counter({two_card_suite: 7, four_card_suite: 39}))
        checkin_coll = [row for row in native if "checkin_collPcclUnitTest" in row["name"]]
        self.assertEqual(len(checkin_coll), 37)
        self.assertEqual(sum("--num_gpus=2" in row["cmd"] for row in checkin_coll), 2)
        self.assertEqual(sum("--num_gpus=4" in row["cmd"] for row in checkin_coll), 35)
        self.assertTrue(all("--num_gpus=4" in row["cmd"]
                            for row in checkin_coll
                            if (row["plan"], row["suite"]) == four_card_suite))
        checkin_graph = [row for row in native if "checkin_hggcGraphUnitTest" in row["name"]]
        self.assertEqual(len(checkin_graph), 7)
        two_card_graph = [row for row in checkin_graph if "--num_threads=2" in row["cmd"]]
        self.assertEqual(len(two_card_graph), 3)
        self.assertTrue(all("HGGC_VISIBLE_DEVICES=0,1" in row["cmd"]
                            for row in two_card_graph))
        four_card_graph = [row for row in checkin_graph if "--num_threads=4" in row["cmd"]]
        self.assertEqual(len(four_card_graph), 4)
        self.assertTrue(all("HGGC_VISIBLE_DEVICES=0,1,2,3" in row["cmd"]
                            for row in four_card_graph))
        graph_search = [row for row in bundle["standard"] if "graphSearchTest" in row["cmd"]]
        self.assertEqual(len(graph_search), 3)
        self.assertTrue(all(row["plan"] == "pccl_810e_graph_search_test_single_node1_plan"
                            for row in graph_search))
        self.assertTrue(all("--root {test_root}/pccl/tests/graph_search_test/" in row["cmd"]
                            for row in graph_search))

    def test_graph_search_allocates_two_cards_on_both_platforms(self):
        generated = self.bundle("graph_search_test_shm", "full", platform="oam810e_eic")
        self.assertEqual({row["cards_per_node"] for row in generated["suites"]}, {2})
        for directory in (ROOT / "minimal-csv", ROOT / "minimal-csv-oam810e-eic"):
            with self.subTest(bundle=directory.name):
                cases = run_cases.load_inputs(directory, "standard", 1)["cases"]
                graph = [case for case in cases if "graphSearchTest" in case["cmd"]]
                self.assertEqual(len(graph), 3)
                self.assertEqual({case["suite_spec"]["cards_per_node"] for case in graph}, {2})
                self.assertEqual({platform_cases.allocation_cards_per_node(case)
                                  for case in graph}, {2})
        plan = group_cases.build_plan(ROOT / "minimal-csv-oam810e-eic", 500)
        allocated = [group["cards_per_node"] for group in plan["groups"]
                     for case in group["cases"] if "graphSearchTest" in case["cmd"]]
        self.assertEqual(len(allocated), 3)
        self.assertEqual(set(allocated), {2})

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

    def test_exclusions_retain_each_platform(self):
        raw = [self.bundle("op_perf_shm", "smoke")]
        converted = platform_cases.apply_platforms(
            raw, platform_cases.load_platform("oam810e_eic"), [self.bundle("op_perf_shm")])
        combined = generate_cases.combine_bundles(raw + converted)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_cases.write_bundle(combined, root / "input", DEFINITIONS / "envs")
            plan = group_cases.build_plan(root / "input", 500, allowed_nodes=(1,))
            self.assertTrue(plan["exclusions"])
            for record in plan["exclusions"]:
                self.assertTrue(all(platform_cases.case_platform(key) == record["platform"]
                                    for key in record["case_keys"]))

    def test_pr_bundle_round_trip_matches_committed_csv(self):
        bundle = generate_pr_cases.build_pr_bundle(SHA, SHA)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "input"
            generate_pr_cases.write_pr_bundle(bundle, output, DEFINITIONS / "envs")
            committed = ROOT / "minimal-csv-oam810e-eic"
            for name in ("standard.csv", "notsm.csv", "suites.csv"):
                self.assertEqual((output / name).read_bytes(), (committed / name).read_bytes())

    def test_pr_bundle_rejects_invalid_environment_names(self):
        bundle = generate_pr_cases.build_pr_bundle(SHA, SHA)
        bundle["environments"] = ["../outside.env"]
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Invalid environment file"):
                generate_pr_cases.write_pr_bundle(bundle, Path(temporary) / "input", DEFINITIONS / "envs")

    def test_ib_environment_is_inherited_and_forwarded_to_mpi_ranks(self):
        direct = run_cases._with_ib_environment("mpirun --allow-run-as-root -np 2 hostname")
        self.assertIn("mpirun -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 ", direct)
        wrapped = run_cases._with_ib_environment(
            "python3 run_test.py --mpi-options='-x PATH --hostfile hosts'")
        self.assertIn("--mpi-options='-x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 ", wrapped)
        unchanged = "python3 helper.py --mpirun-path mpirun_custom"
        self.assertEqual(run_cases._with_ib_environment(unchanged), unchanged)

    def test_810e_allocation_uses_actual_ranks_per_node(self):
        def case(cmd, nodes, name="case_ranks_8"):
            return {"plan": "pccl_810e_plan", "suite": "suite", "name": name, "cmd": cmd,
                    "suite_spec": {"nodes": nodes, "cards_per_node": 16}}

        self.assertEqual(platform_cases.allocation_cards_per_node(
            case("mpirun -np 8 hostname", 2)), 4)
        self.assertEqual(platform_cases.allocation_cards_per_node(
            case("mpirun -np 4 pcclCommAbortTest -t 2 -g 2", 1)), 16)
        self.assertEqual(platform_cases.allocation_cards_per_node(
            case("mpirun -np 8 pcclCommAbortTest -t 2 -g 2", 2)), 16)
        self.assertEqual(platform_cases.allocation_cards_per_node(
            case("mpirun -np 4 other_test -t 2 -g 2", 1)), 4)
        self.assertEqual(platform_cases.allocation_cards_per_node(
            case("run_test.py --ranks 8 --nnodes 2", 2)), 4)
        self.assertEqual(platform_cases.allocation_cards_per_node(
            case("build/hggcGraphUnitTest --num_threads=4 --num_gpus=2", 1)), 8)
        self.assertEqual(platform_cases.allocation_cards_per_node(
            case("HGGC_VISIBLE_DEVICES=0,3 build/hggcGraphUnitTest --num_threads=2", 1)), 4)
        self.assertEqual(platform_cases.allocation_cards_per_node(
            case("HGGC_VISIBLE_DEVICES=0,1,4,5 build/hggcGraphUnitTest --num_threads=4", 1)), 8)
        self.assertEqual(platform_cases.allocation_cards_per_node(
            case("run_test.py --ranks 2 --devices 0,3", 1)), 4)
        self.assertEqual(platform_cases.allocation_cards_per_node(
            case("run_test.py --ranks 4 --devices 0,1,4,5", 1)), 8)
        self.assertEqual(platform_cases.allocation_cards_per_node(
            case("HGGC_VISIBLE_DEVICES=0,1,4,5 build/collPcclUnitTest --num_gpus=4", 1)), 8)
        self.assertEqual(platform_cases.allocation_cards_per_node(
            case("run_test.py --ranks 8 --devices 0,1", 1)), 8)
        self.assertEqual(platform_cases.allocation_cards_per_node(
            case("build/collPcclUnitTest --num_gpus=4", 1)), 4)
        self.assertEqual(platform_cases.allocation_cards_per_node(
            case("all_reduce_perf -t 8 -g 1", 1)), 8)
        self.assertEqual(platform_cases.allocation_cards_per_node(
            case("run_test.py --nprocs 4 --nnodes 2", 2)), 4)
        self.assertEqual(platform_cases.allocation_cards_per_node(
            case("true", 2)), 4)
        graph = case("build/graphSearchTest --root {test_root}/pccl/tests/graph_search_test/ "
                     "--gtest_filter='graphSearchTestWithoutPXN.*'", 1,
                     name="checkin_graphSearchTest_without_pxn")
        graph["suite_spec"]["cards_per_node"] = 2
        self.assertEqual(platform_cases.allocation_cards_per_node(graph), 2)
        m890p = case("mpirun -np 2 hostname", 1)
        m890p["plan"] = "pccl_shm_plan"
        m890p["suite_spec"]["cards_per_node"] = 8
        self.assertEqual(platform_cases.allocation_cards_per_node(m890p), 8)
        with self.assertRaisesRegex(ValueError, "Cannot derive 810E allocation ranks"):
            platform_cases.allocation_cards_per_node(case("true", 1, name="unnamed"))

    def test_full_daily_generation_and_grouping_preserve_platform_layouts(self):
        workflow = (ROOT.parents[1] / '.github/workflows/pccl-daily-full.yml').read_text(encoding='utf-8')
        bundles = []
        families = {}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for variable, platform, expected in (("GENERATED_DEFINITIONS", "m890p", 10193),
                                                ("GENERATED_810E_DEFINITIONS", "oam810e_eic", 7666)):
                blocks = re.findall(r'^ *' + variable + r'\+?=\(\n(.*?)^ *\)',
                                    workflow, flags=re.MULTILINE | re.DOTALL)
                definitions = [Path(path).name for path in
                               re.findall(r'--definition (\S+)', '\n'.join(blocks))]
                self.assertTrue(definitions)
                command = [sys.executable, '-S', '-B', str(ROOT / 'generate_cases.py')]
                for definition in definitions:
                    command.extend(('--definition', str(DEFINITIONS / definition)))
                output = root / platform
                command.extend(('--suites', str(DEFINITIONS / 'suites.json'),
                                '--env-root', str(DEFINITIONS / 'envs'), '--profile', 'full',
                                '--platform', platform, '--source-revision', SHA,
                                '--runner-revision', SHA, '--output', str(output)))
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                bundle = generate_cases.load_bundle(output)
                self.assertEqual(len(bundle['rows']), expected)
                bundles.append(bundle)
                families[platform] = definitions
            self.assertNotIn('device_api_gin_810e.json', families['m890p'])
            self.assertNotIn('device_api_gin_810e.json', families['oam810e_eic'])
            self.assertNotIn('device_api_gin_810e', platform_cases.load_platform('oam810e_eic')['families'])
            self.assertFalse(any('GIN_TYPE=' in row['cmd'] for row in bundles[1]['rows']))
            combined = generate_cases.combine_bundles(bundles)
            self.assertEqual(len(combined['rows']), 17859)
            generate_cases.write_bundle(combined, root / 'input', DEFINITIONS / 'envs')
            plan = group_cases.build_plan(root / 'input', 500, allowed_nodes=(1, 2, 4, 8))
            totals = Counter()
            covered = Counter()
            for group in plan['groups']:
                platform = group['platform']
                cards = group['cards_per_node']
                totals[platform] += len(group['cases'])
                self.assertIn(cards, (1, 2, 4, 8, 16))
                self.assertLessEqual(len(group['cases']), 500)
                self.assertEqual(group['node_selector'], platform_cases.load_platform(platform)['node_selector'])
                if platform == 'm890p':
                    self.assertEqual(cards, 8)
                for case in group['cases']:
                    self.assertEqual(platform_cases.allocation_cards_per_node(case), cards)
                    # Enough local positions must be allocated without compacting any topology.
                    mapped, _, _ = run_cases._map_case_devices(case['cmd'], list(range(cards)), physical=False)
                    self.assertEqual(mapped, case['cmd'])
                    arguments = shlex.split(case['cmd'])
                    if any(Path(argument).name == 'pcclCommAbortTest' for argument in arguments):
                        per_node = 4 if platform == 'oam810e_eic' else 2
                        self.assertEqual(arguments[arguments.index('-np') + 1],
                                         str(group['nodes'] * per_node))
                        if '-npernode' in arguments:
                            self.assertEqual(arguments[arguments.index('-npernode') + 1], str(per_node))
                        self.assertEqual(cards, 16 if platform == 'oam810e_eic' else 8)
                        covered[(platform, 'commabort')] += 1
                    if 'op_perf_rdma' in case['suite']:
                        per_node = 16 if platform == 'oam810e_eic' else 8
                        self.assertEqual(arguments[arguments.index('-np') + 1],
                                         str(group['nodes'] * per_node))
                        self.assertEqual(arguments[arguments.index('-npernode') + 1], str(per_node))
                        self.assertEqual(cards, per_node)
                        covered[(platform, 'rdma')] += 1
                    if 'sym_test' in case['suite']:
                        covered[(platform, 'sym')] += 1
                        ranks = 16 if platform == 'oam810e_eic' else 8
                        if f'ranks_{ranks}' in case['name']:
                            self.assertEqual(arguments[arguments.index('--ranks') + 1], str(ranks))
                            self.assertEqual(arguments[arguments.index('--nproc') + 1], str(ranks))
                            self.assertEqual(cards, ranks)
                            covered[(platform, 'sym-eight-rank-source')] += 1
            self.assertEqual(totals, {'m890p': 10184, 'oam810e_eic': 7666})
            self.assertEqual(len(plan['groups']), 53)
            self.assertEqual(sum(record['case_count'] for record in plan['exclusions']), 9)
            for platform, rdma in (('m890p', 18), ('oam810e_eic', 6)):
                self.assertEqual(covered[(platform, 'commabort')], 16)
                self.assertEqual(covered[(platform, 'rdma')], rdma)
                self.assertEqual(covered[(platform, 'sym')], 127)
                self.assertGreater(covered[(platform, 'sym-eight-rank-source')], 0)
            group_cases.write_plan(plan, root / 'groups', root / 'input')
            manifest = json.loads((root / 'groups/groups.json').read_text(encoding='utf-8'))
            self.assertEqual(sum(row['case_count'] for row in manifest), 17850)
            self.assertEqual({row['platform'] for row in manifest}, {'m890p', 'oam810e_eic'})

    def test_smoke_cli_grouping_matches_documented_platform_counts(self):
        workflow = (ROOT.parents[1] / '.github/workflows/pccl-daily-full.yml').read_text()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundles = []
            for variable, platform in (("GENERATED_DEFINITIONS", "m890p"),
                                       ("GENERATED_810E_DEFINITIONS", "oam810e_eic")):
                block = re.search(r'^ *' + variable + r'=\(\n(.*?)^ *\)', workflow,
                                  flags=re.MULTILINE | re.DOTALL).group(1)
                command = [sys.executable, '-S', '-B', str(ROOT / 'generate_cases.py')]
                for definition in re.findall(r'--definition (\S+)', block):
                    command.extend(('--definition', str(DEFINITIONS / Path(definition).name)))
                output = root / platform
                command.extend(('--suites', str(DEFINITIONS / 'suites.json'),
                                '--env-root', str(DEFINITIONS / 'envs'), '--profile', 'smoke',
                                '--platform', platform, '--source-revision', SHA,
                                '--runner-revision', SHA, '--output', str(output)))
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                bundles.append(generate_cases.load_bundle(output))
            generate_cases.write_bundle(generate_cases.combine_bundles(bundles),
                                        root / 'input', DEFINITIONS / 'envs')
            plan = group_cases.build_plan(root / 'input', 500, allowed_nodes=(1, 2, 4, 8))
            self.assertEqual([(group['platform'], group['nodes'], group['cards_per_node'],
                               len(group['cases'])) for group in plan['groups']],
                             [('m890p', 1, 8, 4), ('oam810e_eic', 1, 2, 2),
                              ('oam810e_eic', 1, 4, 1), ('oam810e_eic', 2, 1, 1),
                              ('oam810e_eic', 2, 16, 1)])
            for name in ('MINIMAL_CSV.md', 'ON_DEMAND_TEST_LISTS.md'):
                text = (ROOT / name).read_text()
                self.assertIn('nine Cases in five groups', ' '.join(text.split()))
                self.assertNotIn('10,196', text)

    def test_quoted_device_spans_determine_the_allocation(self):
        for quote in ("'", '"'):
            for selection in (f"HGGC_VISIBLE_DEVICES={quote}0,3{quote}",
                              f"--devices={quote}0,3{quote}", f"--devices {quote}0,3{quote}"):
                with self.subTest(selection=selection):
                    case = {"plan": "pccl_810e_plan", "suite": "suite", "name": "ranks_2",
                            "cmd": "run_test.py --ranks 2 " + selection,
                            "suite_spec": {"nodes": 1, "cards_per_node": 16}}
                    self.assertEqual(platform_cases.allocation_cards_per_node(case), 4)

    def test_platform_conversion_rejects_missing_variants_and_name_substrings(self):
        raw = [self.bundle("op_perf_shm", "smoke")]
        with self.assertRaisesRegex(ValueError, "Missing variant bundle"):
            platform_cases.apply_platforms(
                raw, platform_cases.load_platform("oam810e_eic"), [])
        self.assertEqual(platform_cases._name_numbers("inp2_unp2_np16", {"np": {"16": 32}}),
                         "inp2_unp2_np32")


if __name__ == "__main__":
    unittest.main()
