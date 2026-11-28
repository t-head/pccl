#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select the M890P daily shards, run them against one build, and merge GCOV data."""

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from pccl_host_coverage import collect, command, version, write_json

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / "tests/xtest"))
import artifact_cache
import case_group_cache
import generate_cases
import group_cases
import run_cases

SELECTOR = "board-type=ZW-M890P,nic-type=cx8"
NODES = {1, 2}
MAX_CASES = 500
# Reuse the full definitions for these correctness suites without inheriting
# the dedicated benchmark suites in the Daily Full matrix.
COVERAGE_FAMILIES = (
    "postsil_daily_shm", "postsil_unit_test_shm", "pccl_fun_rdma_shm",
    "register_test_shm", "sanity_m890p_register", "sym_test_shm", "f8_test_shm",
    "device_api_shm", "commabort_test_shm",
)
GROUP_ID = re.compile(r"g[0-9]{3}")
MATRIX_FIELDS = ("id", "nodes", "cards_per_node", "node_selector", "rack_policy",
                 "rack_count", "cross_rack", "case_count")
STATUSES = {"PASSED", "FAILED", "CANCELLED", "TIMEOUT", "NOT_RUN"}
KEY_FIELDS = ("plan", "suite", "name")


def case_key(row):
    return tuple(row[field] for field in KEY_FIELDS)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def manifest(directory):
    groups = read_json(directory / "groups.json")
    if not 0 < len(groups) <= 256 or {group["nodes"] for group in groups} != NODES:
        raise ValueError("Expected a nonempty 1- and 2-node matrix within the Actions limit")
    ids, keys = set(), set()
    for group in groups:
        gid, nodes = group["id"], group["nodes"]
        if (not GROUP_ID.fullmatch(gid) or gid in ids or group["platform"] != "m890p"
                or group["node_selector"] != SELECTOR or group["profile"] != "standard"):
            raise ValueError(f"Unexpected M890P coverage group: {gid}")
        ids.add(gid)
        expected_racks = ("spread", 2) if group["cross_rack"] and nodes == 2 else ("same", 1)
        if ((group["cross_rack"] and nodes != 2) or
                (group["rack_policy"], group["rack_count"]) != expected_racks):
            raise ValueError(f"Unexpected rack policy in {gid}")
        case_keys = {tuple(key) for key in group["case_keys"]}
        if (len(case_keys) != len(group["case_keys"]) or not 0 < len(case_keys) <= MAX_CASES
                or len(case_keys) != group["case_count"] or keys & case_keys):
            raise ValueError(f"Empty, oversized or duplicate Case selection in {gid}")
        keys.update(case_keys)
    return groups


def prepare(source_revision, runner_revision, output):
    lists = SOURCE / "tests/xtest/test_lists"
    bundle = generate_cases.combine_bundles([
        generate_cases.generate_bundle(lists / f"{family}.json", lists / "suites.json",
                                       source_revision, runner_revision, profile="full", platform="m890p")
        for family in COVERAGE_FAMILIES
    ])
    with tempfile.TemporaryDirectory(prefix="m890p-coverage-cases-") as temp:
        inputs = Path(temp) / "input"
        generate_cases.write_bundle(bundle, inputs, lists / "envs")
        plan = group_cases.build_plan(inputs, MAX_CASES, allowed_nodes=NODES)
        group_cases.write_plan(plan, output, inputs)
        expected = {case_key(case) for case in run_cases.load_inputs(inputs, "standard")["cases"]
                    if case["suite_spec"]["nodes"] in NODES}

    groups = manifest(output)
    counts = Counter()
    selected = set()
    for group in groups:
        cases = run_cases.load_inputs(output / group["id"], group["profile"])["cases"]
        if [list(case_key(case)) for case in cases] != group["case_keys"]:
            raise ValueError(f"Case inputs and group manifest disagree: {group['id']}")
        counts[group["nodes"]] += len(cases)
        selected.update(map(case_key, cases))
    if selected != expected:
        raise ValueError("The grouped selection does not match the intended 1- and 2-node coverage Cases")
    outputs = {"groups_sha256": case_group_cache.tree_sha256(output),
               "groups": json.dumps({"include": [{field: group[field] for field in MATRIX_FIELDS}
                                                 for group in groups]}, separators=(",", ":"))}
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
        for key, value in outputs.items():
            print(f"{key}={value}", file=stream)
    print(f"Selected {len(selected)} Cases in {len(groups)} groups: "
          f"{counts[1]} single-node, {counts[2]} two-node.")


def build_identity():
    return {"schema_version": 1, "pccl_revision": os.environ["PCCL_REVISION"],
            "pccl_tests_revision": artifact_cache.PCCL_TESTS_REVISION,
            "build_profile": "standard", "artifact_sha256": os.environ["RUNTIME_SHA"]}


def check_runtime(runtime):
    if (runtime / "revisions.txt").read_text().splitlines() != [os.environ["PCCL_REVISION"],
                                                               artifact_cache.PCCL_TESTS_REVISION]:
        raise ValueError("The runtime does not match the selected source and pccl-tests revisions")
    for name, expected in artifact_cache.RUNNER_HASHES.items():
        if artifact_cache.file_sha256(runtime / "pccl-tests" / name) != expected:
            raise ValueError(f"Changed benchmark driver: {name}")


def stage_test_root(group_dir, runtime):
    test_root = group_dir / "test-root"
    runner = test_root / "pccl-tests"
    binary_dir = runner / "build"
    binary_dir.mkdir(parents=True)
    for name in ("lib", "include"):
        (test_root / name).symlink_to(runtime / name, target_is_directory=True)
    (test_root / "pccl").symlink_to(SOURCE, target_is_directory=True)
    (test_root / "coverage-runtime").symlink_to(runtime, target_is_directory=True)
    for name in artifact_cache.RUNNER_HASHES:
        (runner / name).symlink_to(runtime / "pccl-tests" / name)
    entrypoint = Path(__file__).with_name("pccl_m890p_coverage_rank.sh").resolve()
    for name in run_cases.DEVICE_WRAPPER_BINARIES:
        binary = runtime / "benchmark" / name
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f"Missing coverage benchmark: {name}")
        (binary_dir / name).symlink_to(entrypoint)
    return test_root, binary_dir


def run_group(shared, gid, hostfile, mpi_interface):
    runtime, selection = shared / "runtime", shared / "selection"
    check_runtime(runtime)
    case_group_cache.validate(selection, os.environ["GROUPS_SHA"])
    groups = {group["id"]: group for group in manifest(selection)}
    group = groups[gid]
    if group["nodes"] == 2:
        if not hostfile or len(hostfile.read_text().splitlines()) != group["nodes"]:
            raise ValueError("Two-node allocation does not match the manifest")
    else:
        hostfile = None
    if (shared / "runtime.sha256").read_text().strip() != os.environ["RUNTIME_SHA"]:
        raise ValueError("The staged runtime digest does not match the build")
    identity = build_identity()
    inputs = run_cases.load_inputs(selection / gid, group["profile"])
    if (inputs["metadata"]["source_revision"], inputs["metadata"]["runner_revision"]) != (
            identity["pccl_revision"], identity["pccl_tests_revision"]):
        raise ValueError("The execution selection does not match the runtime revisions")
    if ([list(case_key(case)) for case in inputs["cases"]] != group["case_keys"]
            or any(case["suite_spec"]["nodes"] != group["nodes"] for case in inputs["cases"])):
        raise ValueError(f"Execution inputs do not match the manifest: {gid}")
    group_dir = shared / "groups" / gid
    export = group_dir / "export"
    export.mkdir(parents=True, exist_ok=False)
    test_root, binary_dir = stage_test_root(group_dir, runtime)
    env = os.environ.copy()
    env.pop("PCCL_PATH", None)
    results = group_dir / "cases"
    rows = run_cases.run_all(inputs, test_root, binary_dir, results, hostfile, env, mpi_interface,
                             build_identity=identity, map_nvidia_devices=True)
    shutil.copyfile(results / "results.csv", export / "results.csv")
    # A timed-out or interrupted group must not look complete merely because it has counters.
    write_json(export / "receipt.json", {"group_id": gid, "build": identity,
                                         "selection_sha256": os.environ["GROUPS_SHA"]})
    return int(any(row["status"] != "PASSED" for row in rows))


def result_counts(group, download, identity, groups_sha):
    receipt = read_json(download / "receipt.json")
    if receipt != {"group_id": group["id"], "build": identity, "selection_sha256": groups_sha}:
        raise ValueError("Group evidence does not match the selection and build")
    with (download / "results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if (len(rows) != group["case_count"] or {case_key(row) for row in rows} !=
            {tuple(key) for key in group["case_keys"]} or any(row["status"] not in STATUSES for row in rows)):
        raise ValueError("Case results do not match the manifest")
    return Counter(row["status"] for row in rows)


def inventory(directory, notes):
    paths = {path.relative_to(directory) for path in directory.rglob("*.gcda")}
    for relative in paths:
        file = directory / relative
        if file.is_symlink() or not file.is_file() or not file.stat().st_size or not (
                notes / relative.with_suffix(".gcno")).is_file():
            raise ValueError(f"Counter does not match a build note: {directory.name}/{relative}")
    return paths


def group_profiles(directory, notes, expected_workers):
    nodes = sorted(path for path in directory.iterdir() if path.is_dir() and not path.is_symlink())
    if len(nodes) != expected_workers:
        raise ValueError(f"Expected counters on {expected_workers} workers, found {len(nodes)}")
    profiles = []
    for node in nodes:
        counters = set()
        for slot in sorted(node.iterdir()):
            if slot.is_symlink() or not slot.is_dir():
                raise ValueError(f"Invalid profile slot: {slot.name}")
            error = slot / "gcov-errors.log"
            if error.is_file() and error.stat().st_size:
                raise ValueError(f"GCOV runtime error on {node.name}, slot {slot.name}")
            paths = inventory(slot, notes)
            if paths:
                profiles.append((slot, paths))
                counters.update(paths)
        if Path("obj/init.gcda") not in counters:
            raise ValueError(f"PCCL init counters missing on {node.name}")
    return profiles


def merge_profiles(profiles, notes, build, report):
    """Merge only compatible raw counters; do not average report percentages."""
    if not profiles:
        raise ValueError("No valid GCOV profiles were collected")
    shutil.copytree(notes, build)
    expected = set()
    with tempfile.TemporaryDirectory(prefix="m890p-merge-") as temp:
        temp = Path(temp)
        accum = temp / "current"
        for index, (profile, paths) in enumerate(profiles):
            expected.update(paths)
            staged = temp / "staged"
            staged.mkdir()
            for relative in paths:
                path = staged / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(profile / relative, path)
            if index == 0:
                staged.rename(accum)
                continue
            output = temp / "merged"
            log = report / f"gcov-merge-{index:03d}.log"
            rc = command(["gcov-tool-13", "merge", accum, staged, "-o", output],
                         log, temp, os.environ.copy(), 60)
            if rc or len(log.read_text().splitlines()) != 1:
                raise ValueError(f"GCOV merge failed; see {log.name}")
            shutil.rmtree(accum)
            shutil.rmtree(staged)
            output.rename(accum)
        if inventory(accum, notes) != expected:
            raise ValueError("The GCOV merge did not preserve the counter file inventory")
        for relative in expected:
            shutil.copyfile(accum / relative, build / relative)


def report_coverage(runtime, selection, downloads, key, report, execution_result):
    report = report.resolve()  # gcovr executes with the compiled source directory as its cwd.
    report.mkdir(parents=True, exist_ok=False)
    summary = {"scope": "PCCL C++ host code", "status": "failed", "groups": [], "coverage": None,
               "performance_baseline_eligible": False}
    try:
        check_runtime(runtime)
        for tool in ("gcov-13", "gcov-tool-13"):
            if version(tool) != (runtime / f"{tool}-version.txt").read_text().strip():
                raise ValueError(f"Reporting {tool} differs from the build")
        groups_sha = os.environ["GROUPS_SHA"]
        case_group_cache.validate(selection, groups_sha)
        identity = build_identity()
        summary.update(build=identity, selection_sha256=groups_sha, execution_result=execution_result)
        totals, nodes = Counter(), Counter()
        profiles = []
        for group in manifest(selection):
            gid = group["id"]
            entry = {"id": gid, "nodes": group["nodes"], "rack_policy": group["rack_policy"],
                     "expected_cases": group["case_count"], "profiles": 0}
            summary["groups"].append(entry)
            nodes[group["nodes"]] += group["case_count"]
            download = downloads / f"{key}-shard-{gid}"
            counts = {"NO_RESULT": group["case_count"]}
            try:
                counts = result_counts(group, download, identity, groups_sha)
                collected = group_profiles(download / "profiles", runtime / "notes", group["nodes"])
                entry["profiles"] = len(collected)
                profiles.extend(collected)
            except (ValueError, OSError, KeyError) as error:
                entry["error"] = str(error)
            entry["counts"] = dict(counts)
            totals.update(counts)
        expected = sum(nodes.values())
        complete = (totals["PASSED"] == expected and execution_result == "success" and
                    all(entry["profiles"] for entry in summary["groups"]))
        summary.update(counts=dict(totals), expected_cases=expected, expected_by_nodes=dict(nodes),
                       groups_with_profiles=sum(bool(entry["profiles"]) for entry in summary["groups"]))
        build = SOURCE / "build"
        merge_profiles(profiles, runtime / "notes", build, report)
        shutil.copytree(runtime / "include", build / "include")
        summary["coverage"] = collect(SOURCE, build, report, "gcov-13 --relative-only")
        summary["status"] = "passed" if complete else "partial"
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        summary["error"] = str(error)
    finally:
        write_json(report / "summary.json", summary)
        lines = ["### PCCL M890P CX8 host coverage", "", f"- Result: **{summary['status']}**."]
        if "expected_cases" in summary:
            lines += [f"- Planned: {summary['expected_cases']} Cases across {len(summary['groups'])} groups.",
                      "- Results: " + ", ".join(f"{key} {value}" for key, value in sorted(summary["counts"].items())),
                      f"- Groups with GCOV profiles: {summary['groups_with_profiles']} / {len(summary['groups'])}."]
        if summary["status"] == "partial":
            lines.append("- **Partial report**: only valid counters available from this attempt were merged.")
        if summary["coverage"] is not None:
            for name in ("lines", "functions", "branches"):
                metric = summary["coverage"][name]
                lines.append(f"- {name.capitalize()}: {metric['covered']} / {metric['total']}")
        if "error" in summary:
            lines.append(f"- {summary['error']}")
        (report / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return int(summary["status"] != "passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--source-revision", required=True)
    prepare_parser.add_argument("--runner-revision", required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    sub.add_parser("binaries")
    run_parser = sub.add_parser("run-group")
    run_parser.add_argument("--shared", type=Path, required=True)
    run_parser.add_argument("--group", required=True)
    run_parser.add_argument("--hostfile", type=Path)
    run_parser.add_argument("--mpi-interface", required=True)
    report = sub.add_parser("report")
    for name in ("runtime", "selection", "downloads", "output"):
        report.add_argument(f"--{name}", type=Path, required=True)
    report.add_argument("--key", required=True)
    report.add_argument("--execution-result", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source_revision, args.runner_revision, args.output)
        return 0
    if args.command == "binaries":
        print("\n".join(run_cases.DEVICE_WRAPPER_BINARIES))
        return 0
    if args.command == "run-group":
        return run_group(args.shared.resolve(), args.group, args.hostfile, args.mpi_interface)
    return report_coverage(args.runtime.resolve(), args.selection, args.downloads, args.key,
                           args.output, args.execution_result)


if __name__ == "__main__":
    raise SystemExit(main())
