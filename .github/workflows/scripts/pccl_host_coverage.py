"""Experimental host-only PCCL GCOV build, graph tests, and report collection."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
from hashlib import sha256
import subprocess
import tarfile
from datetime import datetime, timezone
import xml.etree.ElementTree as ET


GCOVR_VERSION = "8.4"
TEST_MODES = (("without-pxn", "graphSearchTestWithoutPXN.*", []),
              ("with-pxn", "graphSearchTestWithPXN.*", ["--pxn"]))
DEVICE_PATHS = ("src/collectives/device/", "src/include/pccl_device/",
                "src/transport/net_ib/gdaki/doca-gpunetio/")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_summary(paths):
    result = {"tests": 0, "failures": 0, "errors": 0, "disabled": 0, "skipped": 0}
    for path in paths:
        root = ET.parse(path).getroot()
        tests = int(root.get("tests", "0"))
        if tests <= 0:
            raise ValueError(f"No tests executed: {path.name}")
        for key in result:
            if key != "skipped":
                result[key] += int(root.get(key, "0"))
        result["skipped"] += max(int(root.get("skipped", "0")), len(root.findall(".//testcase/skipped")))
    return result


def host_source(filename):
    path = Path(filename)
    return (not path.is_absolute() and ".." not in path.parts and
            bool(path.parts) and path.parts[0] == "src" and
            path.suffix not in (".cu", ".cuh") and
            not any(path.as_posix().startswith(p) for p in DEVICE_PATHS))


def coverage_summary(xml_path, json_path, source):
    root = ET.parse(xml_path).getroot()
    names = [item.get("filename", "") for item in root.findall(".//class")]
    if not names or any(not host_source(name) for name in names):
        raise ValueError("Coverage contains no host source or includes out-of-scope source")
    for name in names:
        path = (source / name).resolve()
        if not path.is_file() or not path.is_relative_to((source / "src").resolve()):
            raise ValueError(f"Coverage source does not match the checkout: {name}")
    lines = int(root.get("lines-valid", "0"))
    covered = int(root.get("lines-covered", "0"))
    if not 0 < covered <= lines:
        raise ValueError("No executed PCCL host lines; instrumentation must be verified")
    branches = int(root.get("branches-valid", "0"))
    branch_covered = int(root.get("branches-covered", "0"))
    if not 0 <= branch_covered <= branches:
        raise ValueError("Invalid branch counts")
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    return {"files": len(set(names)), "lines": {"covered": covered, "total": lines,
            "percent": round(100 * covered / lines, 2)},
            "branches": {"covered": branch_covered, "total": branches,
            "percent": round(100 * branch_covered / branches, 2) if branches else None},
            "functions": {"covered": raw.get("function_covered"), "total": raw.get("function_total")}}


def command(args, log, cwd, env, timeout):
    with log.open("w", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(map(str, args)) + "\n")
        stream.flush()
        process = subprocess.Popen(list(map(str, args)), cwd=cwd, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            return code if code >= 0 else 128 - code
        except subprocess.TimeoutExpired:
            # Stop this experiment's process group, not just make while its compilers keep running.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            stream.write("\nExperiment command timed out.\n")
            return 124


def collect(source, build, output, gcov_command="gcov"):
    profiles = sorted(p for p in build.rglob("*")
                      if p.suffix in (".gcno", ".gcda") and p.is_file() and not p.is_symlink())
    notes = sum(p.suffix == ".gcno" for p in profiles)
    counters = sum(p.suffix == ".gcda" for p in profiles)
    with tarfile.open(output / "profiles.tar.gz", "w:gz") as archive:
        for path in profiles:
            archive.add(path, arcname=path.relative_to(build), recursive=False)
    if not notes or not counters:
        raise ValueError(f"Missing GCOV data: {notes} .gcno files, {counters} .gcda files")
    (output / "html").mkdir()
    args = ["gcovr", "--config", os.devnull, "--root", str(source),
            "--filter", re.escape(str(source / "src")) + "/",
            "--gcov-executable", gcov_command]
    for path in DEVICE_PATHS:
        args += ["--exclude", re.escape(str(source / path).rstrip("/")) + "/"]
    args += ["--exclude", r".*\.(cu|cuh)$", "--html-details", str(output / "html/index.html"),
             "--cobertura", str(output / "coverage.xml"), "--json-summary", str(output / "gcovr-summary.json"),
             "--print-summary", str(build)]
    rc = command(args, output / "gcovr.log", source, os.environ.copy(), 180)
    if rc:
        raise ValueError(f"gcovr failed with exit code {rc}; see gcovr.log")
    metrics = coverage_summary(output / "coverage.xml", output / "gcovr-summary.json", source)
    metrics["profiles"] = {"notes": notes, "counters": counters}
    return metrics


def version(program):
    return subprocess.check_output([program, "--version"], text=True, timeout=10).splitlines()[0]


def experiment(args):
    source = args.source.resolve()
    build, output = args.build_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "pccl-host-cpp", "validation_status": "failed", "report_status": "unavailable",
              "performance_baseline_eligible": False, "gcovr_version_expected": GCOVR_VERSION,
              "build_config": {"GCOV": 1, "DEBUG": 1, "PPU_ARCHS": args.ppu_archs},
              "test_results": [], "errors": []}
    exit_code = 1
    owned_build = False
    try:
        if build.exists():
            raise ValueError("Build directory already exists; stale coverage must not be reused")
        sdk = Path(os.environ.get("PPU_SDK", "/usr/local/PPU_SDK")).resolve()
        hgcc = shutil.which("hgcc") or str(sdk / "bin/hgcc")
        if not Path(hgcc).is_file():
            raise ValueError("A Linux PPU SDK with hgcc is required; no product build was performed")
        cxx = shutil.which(os.environ.get("CXX", "g++"))
        if not cxx or not shutil.which("gcov") or not shutil.which("gcovr"):
            raise ValueError("g++, matching gcov, and gcovr must be installed")
        revision = args.source_revision or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("A full source commit is required")
        if args.snapshot_sha256 and not re.fullmatch(r"[0-9a-f]{64}", args.snapshot_sha256):
            raise ValueError("Invalid source snapshot SHA-256")
        report.update({"source_revision": revision, "snapshot_sha256": args.snapshot_sha256,
                       "sdk": str(sdk), "cxx_version": version(cxx), "gcov_version": version("gcov"),
                       "gcovr_version": version("gcovr"), "hgcc_version": version(hgcc)})
        if report["gcovr_version"] != "gcovr " + GCOVR_VERSION:
            raise ValueError("Use the pinned gcovr version " + GCOVR_VERSION)
        build.mkdir(parents=True)
        owned_build = True
        env = os.environ.copy()
        for name in ("GCOV_PREFIX", "GCOV_PREFIX_STRIP", "LLVM_PROFILE_FILE"):
            env.pop(name, None)
        env["LD_LIBRARY_PATH"] = ":".join([str(build / "lib"), str(sdk / "lib"), env.get("LD_LIBRARY_PATH", "")])
        build_args = ["make", f"-j{args.jobs}", "src.lib", "GCOV=1", "DEBUG=1", "NO_DLTO=1",
                      "HGGCRT_VERSION=13", f"PPU_ARCHS={args.ppu_archs}", f"BUILDDIR={build}",
                      f"CXX={cxx}", f"PPU_OPTION=-ccbin;{cxx}", f"HGCC={hgcc}"]
        rc = command(build_args, output / "build.log", source, env, 2400)
        report["build_exit_code"] = rc
        if rc:
            exit_code = rc
            raise ValueError(f"Instrumented library build failed with exit code {rc}")
        digest = sha256()
        with (build / "lib/libpccl.so").open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        report["library_sha256"] = digest.hexdigest()
        test_root = source / "tests/graph_search_test"
        gtest = test_root / "3rd_party/gtest/lib/libgtest.a"
        # Match the current library's SDK wrapper linkage; the standalone test defaults to legacy runtime names.
        ldflags = f"-pthread {build}/lib/libpccl.so -Wl,--disable-new-dtags -lrt -ldl -L{sdk}/lib -lhggc_wrapper -lhg_wrapper {gtest} --coverage"
        rc = command(["make", f"-j{args.jobs}", f"CXX={cxx}", f"PCCL_HOME={build}",
                      f"TESTBUILDDIR={build}/graph-search", f"TESTLDFLAGS={ldflags}"],
                     output / "test-build.log", test_root, env, 600)
        report["test_build_exit_code"] = rc
        if rc:
            exit_code = rc
            raise ValueError(f"Graph test build failed with exit code {rc}")
        exit_code = 0
        for mode, pattern, extra in TEST_MODES:
            xml = output / f"tests-{mode}.xml"
            rc = command([build / "graph-search/graphSearchTest", f"--root={test_root}/topo_and_reference_graph/",
                          f"--gtest_filter={pattern}", f"--gtest_output=xml:{xml}", *extra],
                         output / f"tests-{mode}.log", source, env, 300)
            report["test_results"].append({"mode": mode, "filter": pattern, "exit_code": rc})
            exit_code = exit_code or rc
        report["tests"] = test_summary([output / f"tests-{mode}.xml" for mode, _, _ in TEST_MODES])
        if report["tests"]["failures"] or report["tests"]["errors"] or report["tests"]["disabled"] or report["tests"]["skipped"]:
            exit_code = exit_code or 1
        report["validation_status"] = "passed" if exit_code == 0 else "failed"
    except (OSError, ValueError, subprocess.SubprocessError, ET.ParseError) as error:
        report["errors"].append(str(error))
        exit_code = exit_code or 1
    finally:
        try:
            if not owned_build:
                raise ValueError("No fresh instrumented build was started; existing profiles were not collected")
            report["coverage"] = collect(source, build, output)
            report["report_status"] = "available"
        except (OSError, ValueError, subprocess.SubprocessError, ET.ParseError) as error:
            report["errors"].append(str(error))
            exit_code = exit_code or 1
        if exit_code:
            report["validation_status"] = "failed"
        report["generated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "summary.json", report)
        text = "## PCCL C++ host GCOV experiment\n\n" + f"Validation: **{report['validation_status']}**; coverage report: **{report['report_status']}**.\n\n"
        if "coverage" in report:
            for name in ("lines", "branches"):
                metric = report["coverage"][name]
                text += f"- {name.title()}: {metric['covered']}/{metric['total']} ({metric['percent']}%)\n"
        text += "\nScope: host source only; this is not PPU kernel coverage or performance baseline data.\n"
        (output / "summary.md").write_text(text, encoding="utf-8")
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
                stream.write(text)
        print(json.dumps(report, sort_keys=True))
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--source-revision", help="Commit of the pinned checkout or a verified exported source snapshot")
    parser.add_argument("--snapshot-sha256", help="SHA-256 of the exported source archive")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--ppu-archs", choices=("ppu_10,ppu_15", "ppu_15"), default="ppu_10,ppu_15")
    return experiment(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
