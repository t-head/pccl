"""Collect and validate the PCCL host-only GCOV report."""

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tarfile
import xml.etree.ElementTree as ET


# Allow 64-bit counters from instrumented host loops, but keep a finite guard.
SUSPICIOUS_HITS_THRESHOLD = 1 << 40
DEVICE_PATHS = ("src/collectives/device/", "src/include/pccl_device/",
                "src/transport/net_ib/gdaki/doca-gpunetio/")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
            # Stop the collector process group so no subprocess keeps using the profiles.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            stream.write("\nCoverage command timed out.\n")
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
            "--gcov-executable", gcov_command,
            "--gcov-suspicious-hits-threshold", str(SUSPICIOUS_HITS_THRESHOLD)]
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
