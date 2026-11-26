"""Contract tests for PCCL host instrumentation and coverage reports."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("host_coverage", ROOT / ".github/workflows/scripts/pccl_host_coverage.py")
coverage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coverage)


class HostCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        (self.source / "src/graph").mkdir(parents=True)
        (self.source / "src/graph/probe.cc").write_text("int probe() { return 1; }\n")

    def xml(self, name="src/graph/probe.cc", covered=2, lines=4):
        path = self.root / "coverage.xml"
        path.write_text(f'<coverage lines-valid="{lines}" lines-covered="{covered}" branches-valid="2" branches-covered="1"><packages><package><classes><class filename="{name}"/></classes></package></packages></coverage>')
        summary = self.root / "gcovr.json"
        summary.write_text(json.dumps({"function_covered": 1, "function_total": 2}))
        return path, summary

    def make_flags(self, *options):
        hgcc = self.root / "hgcc"
        hgcc.write_text("#!/bin/sh\necho 'HGGC version 2.2.0'\n")
        hgcc.chmod(0o700)
        makefile = self.root / "flags.mk"
        makefile.write_text(f"include {ROOT}/makefiles/common.mk\n"
                            "flags:\n\t@printf '%s\\n' 'CXX=$(CXXFLAGS)' 'HG=$(HGFLAGS)' "
                            "'LD=$(LDFLAGS)' 'HGLD=$(HGLDFLAGS)'\n")
        result = subprocess.check_output(["make", "-f", str(makefile), "flags", f"HGCC={hgcc}",
                                          "PPU_ARCHS=ppu_15", "HGTX=0", *options], text=True)
        return dict(line.split("=", 1) for line in result.splitlines()
                    if line.startswith(("CXX=", "HG=", "LD=", "HGLD=")))

    def test_host_gcov_keeps_device_dependencies_uninstrumented(self):
        flags = self.make_flags("GCOV=0", "DEBUG=0", "HOST_GCOV=1")
        self.assertIn("--coverage", flags["CXX"])
        self.assertEqual([flag for flag in flags["CXX"].split() if flag in ("-O0", "-O3")][-1], "-O0")
        self.assertIn("--coverage", flags["LD"])
        self.assertIn("-dopt 3", flags["HG"])
        self.assertNotIn("-G", flags["HG"].split())
        self.assertNotIn("--coverage", flags["HG"])
        self.assertNotIn("--coverage", flags["HGLD"])

    def test_default_and_legacy_gcov_build_flags_are_preserved(self):
        baseline = self.make_flags("GCOV=0", "DEBUG=0")
        self.assertEqual(baseline, self.make_flags("GCOV=0", "DEBUG=0", "HOST_GCOV=0"))
        self.assertNotIn("--coverage", " ".join(baseline.values()))
        legacy = self.make_flags("GCOV=1", "DEBUG=1")
        for field in ("CXX", "HG", "LD", "HGLD"):
            self.assertIn("--coverage", legacy[field])

    def test_large_counter_guard_is_finite_and_does_not_ignore_parse_errors(self):
        self.assertEqual(coverage.SUSPICIOUS_HITS_THRESHOLD, 1 << 40)
        helper = (ROOT / ".github/workflows/scripts/pccl_host_coverage.py").read_text()
        self.assertIn('"--gcov-suspicious-hits-threshold"', helper)
        self.assertNotIn('"--gcov-ignore-parse-errors"', helper)
        self.assertNotIn('"--gcov-ignore-errors"', helper)

    def test_summary_metrics_are_computed_from_counts(self):
        metrics = coverage.coverage_summary(*self.xml(), self.source)
        self.assertEqual(metrics["lines"], {"covered": 2, "total": 4, "percent": 50.0})
        self.assertEqual(metrics["branches"]["percent"], 50.0)
        self.assertEqual(metrics["functions"], {"covered": 1, "total": 2})

    def test_rejects_device_test_vendor_and_external_sources(self):
        paths = ["tests/probe.cc", "../src/probe.cc", "/src/probe.cc", "src/kernel.cu",
                 "src/kernel.cuh", "src/collectives/device/probe.cc", "src/include/pccl_device/probe.h",
                 "src/transport/net_ib/gdaki/doca-gpunetio/probe.cpp"]
        for name in paths:
            with self.subTest(name=name), self.assertRaises(ValueError):
                coverage.coverage_summary(*self.xml(name), self.source)
        self.assertTrue(coverage.host_source("src/collectives/device_helpers.cc"))

    def test_rejects_no_executed_product_lines_and_bad_counts(self):
        for covered, total in [(0, 4), (5, 4), (-1, 4)]:
            with self.subTest(covered=covered), self.assertRaises(ValueError):
                coverage.coverage_summary(*self.xml(covered=covered, lines=total), self.source)

    def test_rejects_source_symlink_outside_checkout(self):
        target = self.root / "outside.cc"
        target.write_text("int outside;\n")
        (self.source / "src/graph/escape.cc").symlink_to(target)
        with self.assertRaises(ValueError):
            coverage.coverage_summary(*self.xml("src/graph/escape.cc"), self.source)

    def test_missing_counters_are_not_a_successful_zero_percent_report(self):
        build = self.root / "build"
        build.mkdir()
        (build / "probe.gcno").write_bytes(b"fixture")
        output = self.root / "report"
        output.mkdir()
        with self.assertRaisesRegex(ValueError, "Missing GCOV data"):
            coverage.collect(self.source, build, output)
        self.assertFalse((output / "coverage.xml").exists())

    def test_collection_excludes_device_and_vendor_directories(self):
        build = self.root / "build"
        build.mkdir()
        for suffix in (".gcno", ".gcda"):
            (build / ("probe" + suffix)).write_bytes(b"fixture")
        output = self.root / "report"
        output.mkdir()
        with patch.object(coverage, "command", return_value=0) as run, \
                patch.object(coverage, "coverage_summary", return_value={}):
            coverage.collect(self.source, build, output)
        args = run.call_args.args[0]
        excludes = [args[index + 1] for index, value in enumerate(args) if value == "--exclude"]
        for directory in coverage.DEVICE_PATHS:
            expected = (self.source / directory / "probe.cc").as_posix()
            self.assertTrue(any(coverage.re.search(pattern, expected) for pattern in excludes))

    def test_command_failure_and_timeout_are_not_swallowed(self):
        env = os.environ.copy()
        self.assertEqual(coverage.command([sys.executable, "-c", "raise SystemExit(7)"],
                                         self.root / "failed.log", self.root, env, 5), 7)
        self.assertEqual(coverage.command([sys.executable, "-c", "import time; time.sleep(10)"],
                                         self.root / "timeout.log", self.root, env, 0.05), 124)


if __name__ == "__main__":
    unittest.main()
