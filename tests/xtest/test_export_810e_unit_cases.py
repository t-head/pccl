# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import tempfile
import unittest

import export_810e_unit_cases as exporter


class Export810ECaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def source(self, command):
        relative = "test/cases/example.py"
        path = self.root / "v2" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "metadata = dict(name='example_case')\n"
            "pipeline = [dict(type='runner', mpi_command=" + repr(command) + ")]\n")
        return relative

    def test_source_and_coll_options_support_both_syntaxes(self):
        relative = self.source(
            "python unit_test.py -t=collPcclUnitTest --auto_detect=1 --algo Ring "
            "--proto=Simple --devices 0,1 --data_size=32MB --coll_type=AllReduce")
        case = exporter.native_case(self.root, relative)
        self.assertEqual(case["name"], "example_case")
        self.assertEqual(case["cmd"],
                         "NCCL_ALGO=Ring NCCL_PROTO=Simple HGGC_VISIBLE_DEVICES=0,1 "
                         "build/collPcclUnitTest --coll_type=AllReduce")

    def test_graph_binary_and_config_paths_are_rewritten(self):
        relative = self.source(
            "python unit_test.py -t=cudaGraphUnitTest --devices=0,3 "
            "--test_config_file ./cuda_graph_test_configs/simple.json --num_threads=2")
        case = exporter.native_case(self.root, relative)
        self.assertIn("HGGC_VISIBLE_DEVICES=0,3 build/hggcGraphUnitTest", case["cmd"])
        self.assertIn("{test_root}/pccl/tests/unit_test/hggc_graph_test_configs/simple.json", case["cmd"])

        relative = self.source(
            "python unit_test.py -t=cudaGraphUnitTest "
            "--test_config_file ./feature_test_configs/allgather_unit_test.json")
        self.assertIn("feature_test_configs/allgather_unit_test.json",
                      exporter.native_case(self.root, relative)["cmd"])

    def test_public_binary_and_option_exclusions_are_explicit(self):
        for binary in ("p2pBandwidthLatencyTest", "alltoallvTest"):
            with self.subTest(binary=binary):
                relative = self.source(f"python unit_test.py -t={binary}")
                with self.assertRaisesRegex(ValueError, "Public build lacks"):
                    exporter.native_case(self.root, relative)
        relative = self.source("python unit_test.py -t=cudaGraphUnitTest --use_ext_kernel=1")
        with self.assertRaisesRegex(ValueError, "lacks --use_ext_kernel"):
            exporter.native_case(self.root, relative)

    def test_all_gather_wrapper_preserves_mpi_and_rewrites_paths(self):
        relative = self.source(
            "mpirun --allow-run-as-root -np 2 ./build/cudaGraphUnitTest "
            "-f ./feature_test_configs/allgather_unit_test.json")
        case = exporter.all_gather_case(self.root, relative)
        self.assertTrue(case["cmd"].startswith("mpirun --allow-run-as-root -np 2 build/hggcGraphUnitTest"))
        self.assertIn("{test_root}/pccl/tests/unit_test/feature_test_configs/allgather_unit_test.json",
                      case["cmd"])


if __name__ == "__main__":
    unittest.main()
