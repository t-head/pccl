"""Contract tests for the C++ host coverage experiment (not Python coverage)."""

import argparse
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
        # Mocked coverage metrics must never appear in the real Actions job summary.
        summary_environment = patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": ""})
        summary_environment.start()
        self.addCleanup(summary_environment.stop)
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

    def test_native_archive_does_not_request_device_objects_or_ignore_link_errors(self):
        source_make = (ROOT / "src/Makefile").read_text()
        archive = source_make.split("$(LIBDIR)/libpccl_host_gcov.a:", 1)[1].split("binary :", 1)[0]
        self.assertIn("$(LIBOBJ)", archive)
        self.assertNotIn("$(DEVICELIB)", archive)
        helper = (ROOT / ".github/workflows/scripts/pccl_host_coverage.py").read_text()
        self.assertIn("-Wl,--gc-sections", helper)
        self.assertNotIn("--unresolved-symbols=ignore", helper)
        self.assertNotIn("--allow-shlib-undefined", helper)
        self.assertIn('f"PCCL_SO={library}"', helper)

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

    def test_gtest_cannot_succeed_with_zero_selected_tests(self):
        xml = self.root / "tests.xml"
        xml.write_text('<testsuites tests="0" failures="0"/>')
        with self.assertRaises(ValueError):
            coverage.test_summary([xml])

    def test_gtest_failure_and_skip_counts_are_preserved(self):
        xml = self.root / "tests.xml"
        xml.write_text('<testsuites tests="3" failures="1" errors="0" disabled="0"><testsuite><testcase><skipped/></testcase></testsuite></testsuites>')
        result = coverage.test_summary([xml])
        self.assertEqual(result["tests"], 3)
        self.assertEqual(result["failures"], 1)
        self.assertEqual(result["skipped"], 1)

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

    def test_existing_build_is_not_collected_after_rejection(self):
        build = self.root / "old-build"
        build.mkdir()
        (build / "old.gcda").write_bytes(b"fixture")
        args = argparse.Namespace(source=self.source, build_dir=build, output=self.root / "report",
                                  jobs=2, ppu_archs="ppu_15", library_kind="host_archive", source_revision=None, snapshot_sha256=None)
        with patch.object(coverage, "collect") as collect:
            self.assertNotEqual(coverage.experiment(args), 0)
            collect.assert_not_called()
        report = json.loads((args.output / "summary.json").read_text())
        self.assertEqual(report["report_status"], "unavailable")
        self.assertFalse((args.output / "profiles.tar.gz").exists())

    def test_command_failure_and_timeout_are_not_swallowed(self):
        env = os.environ.copy()
        self.assertEqual(coverage.command([sys.executable, "-c", "raise SystemExit(7)"],
                                         self.root / "failed.log", self.root, env, 5), 7)
        self.assertEqual(coverage.command([sys.executable, "-c", "import time; time.sleep(10)"],
                                         self.root / "timeout.log", self.root, env, 0.05), 124)

    def test_failed_product_test_keeps_report_but_fails_validation(self):
        sdk = self.root / "sdk"
        (sdk / "bin").mkdir(parents=True)
        (sdk / "bin/hgcc").write_text("fixture")
        build = self.root / "build"
        args = argparse.Namespace(source=self.source, build_dir=build, output=self.root / "report",
                                  jobs=2, ppu_archs="ppu_15", library_kind="host_archive", source_revision="a" * 40, snapshot_sha256="b" * 64)

        def tool(name):
            return str(sdk / "bin/hgcc") if name == "hgcc" else "/usr/bin/g++"

        def fake_command(command, log, cwd, env, timeout):
            if "src.host_gcov_archive" in command:
                self.assertIn("HOST_GCOV=1", command)
                self.assertIn("GCOV=0", command)
                self.assertIn("DEBUG=0", command)
                self.assertFalse(any(str(value).startswith(("TEST_DT=", "TEST_OP=")) for value in command))
                (build / "lib").mkdir()
                (build / "lib/libpccl_host_gcov.a").write_bytes(b"instrumented-library-fixture")
                (build / "lib/host_gcov_cxxflags.txt").write_text("-DPCCL_HGGC_VERSION=20200 -DHGGC_MAJOR=2 -DHGGC_MINOR=2")
            if any(str(value).startswith("--gtest_output=") for value in command):
                xml = next(str(value).split("xml:", 1)[1] for value in command
                           if str(value).startswith("--gtest_output="))
                failed = "without-pxn" in xml
                Path(xml).write_text(f'<testsuites tests="2" failures="{int(failed)}" errors="0" disabled="0"/>')
                return 7 if failed else 0
            return 0

        metrics = {"lines": {"covered": 2, "total": 4, "percent": 50.0},
                   "branches": {"covered": 1, "total": 2, "percent": 50.0}}
        with patch.dict(os.environ, {"PPU_SDK": str(sdk)}), \
                patch.object(coverage.shutil, "which", side_effect=tool), \
                patch.object(coverage, "version", side_effect=lambda p: "gcovr 8.4" if p == "gcovr" else "fixture"), \
                patch.object(coverage, "command", side_effect=fake_command), \
                patch.object(coverage, "collect", return_value=metrics):
            self.assertEqual(coverage.experiment(args), 7)
        report = json.loads((args.output / "summary.json").read_text())
        self.assertEqual(report["validation_status"], "failed")
        self.assertEqual(report["report_status"], "available")
        self.assertEqual(report["tests"]["failures"], 1)
        self.assertFalse(report["performance_baseline_eligible"])

    def test_collection_failure_marks_overall_validation_failed(self):
        sdk = self.root / "sdk"
        (sdk / "bin").mkdir(parents=True)
        (sdk / "bin/hgcc").write_text("fixture")
        build = self.root / "build"
        args = argparse.Namespace(source=self.source, build_dir=build, output=self.root / "report",
                                  jobs=2, ppu_archs="ppu_15", library_kind="host_archive", source_revision="a" * 40, snapshot_sha256="b" * 64)

        def tool(name):
            return str(sdk / "bin/hgcc") if name == "hgcc" else "/usr/bin/g++"

        def fake_command(command, log, cwd, env, timeout):
            if "src.host_gcov_archive" in command:
                self.assertIn("HOST_GCOV=1", command)
                self.assertIn("GCOV=0", command)
                self.assertIn("DEBUG=0", command)
                self.assertFalse(any(str(value).startswith(("TEST_DT=", "TEST_OP=")) for value in command))
                (build / "lib").mkdir()
                (build / "lib/libpccl_host_gcov.a").write_bytes(b"instrumented-library-fixture")
                (build / "lib/host_gcov_cxxflags.txt").write_text("-DPCCL_HGGC_VERSION=20200 -DHGGC_MAJOR=2 -DHGGC_MINOR=2")
            for value in command:
                if str(value).startswith("TESTCXXFLAGS="):
                    self.assertIn("-DPCCL_HGGC_VERSION=20200", value)
                if str(value).startswith("--gtest_output="):
                    Path(str(value).split("xml:", 1)[1]).write_text('<testsuites tests="2" failures="0"/>')
            return 0

        with patch.dict(os.environ, {"PPU_SDK": str(sdk)}), \
                patch.object(coverage.shutil, "which", side_effect=tool), \
                patch.object(coverage, "version", side_effect=lambda p: "gcovr 8.4" if p == "gcovr" else "fixture"), \
                patch.object(coverage, "command", side_effect=fake_command), \
                patch.object(coverage, "collect", side_effect=ValueError("fixture collection failure")):
            self.assertEqual(coverage.experiment(args), 1)
        report = json.loads((args.output / "summary.json").read_text())
        self.assertEqual(report["tests"]["failures"], 0)
        self.assertEqual(report["validation_status"], "failed")
        self.assertEqual(report["report_status"], "unavailable")

    def test_child_profile_overrides_do_not_leak_into_test_commands(self):
        text = (ROOT / ".github/workflows/scripts/pccl_host_coverage.py").read_text()
        self.assertIn('("GCOV_PREFIX", "GCOV_PREFIX_STRIP", "LLVM_PROFILE_FILE")', text)
        self.assertIn('env.pop(name, None)', text)

    def test_workflow_is_isolated_and_keeps_reports_after_failure(self):
        text = (ROOT / ".github/workflows/pccl-host-coverage.yml").read_text()
        self.assertIn("  workflow_dispatch:", text)
        for trigger in ("  pull_request:", "  schedule:"):
            self.assertNotIn(trigger, text)
        push = text.split("  push:\n", 1)[1].split("  workflow_dispatch:", 1)[0]
        self.assertIn("branches: [codex/pccl-host-gcov-experiment]", push)
        self.assertEqual(push.count("branches:"), 1)
        self.assertNotIn("*", push)
        for path in (".github/workflows/pccl-host-coverage.yml",
                     ".github/workflows/scripts/pccl_host_coverage.py", "tests/xtest/test_host_coverage.py",
                     "makefiles/common.mk", "src/Makefile"):
            self.assertIn("      - " + path, push)
        self.assertIn("if: always()", text)
        self.assertNotIn("code-quality: write", text)
        self.assertIn("gcovr==" + coverage.GCOVR_VERSION, text)

    def test_push_has_cpu_and_arch_defaults_without_dispatch_inputs(self):
        text = (ROOT / ".github/workflows/pccl-host-coverage.yml").read_text()
        self.assertIn("--cpus=${{ inputs.jobs || '8' }}", text)
        self.assertIn("BUILD_JOBS: ${{ inputs.jobs || '8' }}", text)
        self.assertIn("PPU_ARCHS: ${{ inputs.ppu_archs || 'ppu_15' }}", text)
        self.assertIn("cancel-in-progress: false", text)

    def test_workflow_pins_checkout_and_supplies_the_same_source_revision(self):
        text = (ROOT / ".github/workflows/pccl-host-coverage.yml").read_text()
        self.assertIn("ref: ${{ github.sha }}", text)
        self.assertIn('--source-revision "$GITHUB_SHA"', text)


if __name__ == "__main__":
    unittest.main()
