#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build deterministic execution groups from the exported CSV, never from hand-maintained shards."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile

import run_cases
import platform_cases


POLICY_EXCLUSION = "disabled by daily resource policy"
CROSS_RACK_SUITES = {
    ("pccl_shm_op_perf_2_rack_icn8_plan", "op_perf_shm_2_rack_icn8_suite"): 2,
    ("pccl_shm_op_perf_2_rack_icn16_plan", "op_perf_shm_2_rack_icn16_suite"): 4,
    ("pccl_shm_op_perf_2_rack_icn32_plan", "op_perf_shm_2_rack_icn32_suite"): 8,
    ("pccl_shm_op_perf_2_rack_icn64_plan", "op_perf_shm_2_rack_icn64_suite"): 16,
}


def _rack_policy(case):
    key = (case["plan"], case["suite"])
    expected_nodes = CROSS_RACK_SUITES.get(key)
    if expected_nodes is None:
        if "2_rack" in case["plan"] or "2_rack" in case["suite"]:
            raise ValueError(f"Cross-rack Suite has no reviewed rack allocation: {key}")
        return ("same", 1) if platform_cases.case_platform(case) == platform_cases.DEFAULT_PLATFORM else ("", 0)
    if case["suite_spec"]["nodes"] != expected_nodes:
        raise ValueError(f"Cross-rack Suite has an unexpected node count: {key}")
    return "spread", 2


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_plan(directory, max_cases, nodes=None, suite=None, allowed_nodes=None):
    if type(max_cases) is not int or max_cases <= 0:
        raise ValueError("max_cases must be a positive integer")
    if allowed_nodes is not None:
        allowed_nodes = frozenset(allowed_nodes)
        if (not allowed_nodes or any(type(value) is not int or value <= 0 for value in allowed_nodes)):
            raise ValueError("allowed_nodes must contain positive integers")
        if nodes is not None and nodes not in allowed_nodes:
            raise ValueError(f"Requested node count {nodes} is {POLICY_EXCLUSION}")
    grouped = {}
    excluded = {}
    metadata = None
    platforms = platform_cases.load_platforms()
    for profile in ("standard", "notsm"):
        try:
            selected = run_cases.load_inputs(directory, profile, nodes=nodes, suite=suite)
        except ValueError as error:
            if "no Cases" not in str(error):
                raise
            continue
        metadata = selected["metadata"]
        for case in selected["cases"]:
            spec = case["suite_spec"]
            platform = platform_cases.case_platform(case, platforms)
            rack_policy, rack_count = _rack_policy(case)
            if allowed_nodes is not None and spec["nodes"] not in allowed_nodes:
                excluded.setdefault(spec["nodes"], []).append(case)
                continue
            key = (platform, profile, spec["nodes"], spec["cards_per_node"], spec["env"],
                   rack_policy, rack_count)
            grouped.setdefault(key, []).append(case)
    if metadata is None:
        raise ValueError("No Cases match the requested selection")
    if not grouped:
        raise ValueError("No Cases match the requested selection and resource policy")
    groups = []
    for key in sorted(grouped):
        platform, profile, group_nodes, cards, env, rack_policy, rack_count = key
        cases = sorted(grouped[key], key=lambda case: (case["plan"], case["suite"], case["name"]))
        for start in range(0, len(cases), max_cases):
            chunk = cases[start:start + max_cases]
            groups.append({
                "id": f"g{len(groups) + 1:03d}",
                "profile": profile,
                "platform": platform,
                "node_selector": platforms[platform]["node_selector"],
                "nodes": group_nodes,
                "cards_per_node": cards,
                "env": env,
                "cross_rack": rack_policy == "spread",
                "rack_policy": rack_policy,
                "rack_count": rack_count,
                "cases": chunk,
                "metadata": metadata,
            })
    exclusions = []
    for node_count in sorted(excluded):
        cases = sorted(excluded[node_count], key=lambda case: (case["plan"], case["suite"], case["name"]))
        exclusions.append({
            "nodes": node_count, "case_count": len(cases), "reason": POLICY_EXCLUSION,
            "platform": platform_cases.case_platform(cases[0], platforms),
            "case_keys": [[case["plan"], case["suite"], case["name"]] for case in cases],
        })
    return {"groups": groups, "exclusions": exclusions,
            "allowed_nodes": None if allowed_nodes is None else sorted(allowed_nodes)}


def build_groups(directory, max_cases, nodes=None, suite=None, allowed_nodes=None):
    return build_plan(directory, max_cases, nodes=nodes, suite=suite,
                      allowed_nodes=allowed_nodes)["groups"]


def _coverage(plan):
    selected = sum(len(group["cases"]) for group in plan["groups"])
    excluded = sum(record["case_count"] for record in plan["exclusions"])
    return {
        "schema_version": 1,
        "allowed_nodes": plan["allowed_nodes"],
        "selected_groups": len(plan["groups"]),
        "selected_cases": selected,
        "excluded_cases": excluded,
        "total_cases": selected + excluded,
        "exclusions": plan["exclusions"],
    }


def write_groups(groups, output, input_directory, coverage=None):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Group output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        manifest = []
        for group in groups:
            allocations = {_rack_policy(case) for case in group["cases"]}
            if len(allocations) != 1:
                raise ValueError("An execution group mixes different rack allocations")
            rack_policy, rack_count = allocations.pop()
            if (group.get("rack_policy", "") != rack_policy
                    or group.get("rack_count", 0) != rack_count
                    or group.get("cross_rack", rack_policy == "spread") is not (rack_policy == "spread")):
                raise ValueError("Execution group rack metadata does not match its Cases")
            directory = temporary / group["id"]
            (directory / "envs").mkdir(parents=True)
            (directory / "export.json").write_text(
                json.dumps(group["metadata"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
            suite_rows = {}
            for case in group["cases"]:
                suite_rows[(case["plan"], case["suite"])] = case["suite_spec"]
            _write_csv(directory / "suites.csv", run_cases.SUITE_FIELDS,
                       [suite_rows[key] for key in sorted(suite_rows)])
            for profile in ("standard", "notsm"):
                rows = [case for case in group["cases"] if profile == group["profile"]]
                _write_csv(directory / f"{profile}.csv", run_cases.CASE_FIELDS,
                           [{key: case[key] for key in run_cases.CASE_FIELDS} for case in rows])
            shutil.copyfile(Path(input_directory) / "envs" / group["env"],
                            directory / "envs" / group["env"])
            manifest.append({
                "id": group["id"], "profile": group["profile"], "nodes": group["nodes"],
                "platform": group["platform"], "node_selector": group["node_selector"],
                "cards_per_node": group["cards_per_node"], "env": group["env"],
                "cross_rack": rack_policy == "spread",
                "rack_policy": rack_policy, "rack_count": rack_count,
                "case_count": len(group["cases"]),
                "plan_suites": sorted({(case["plan"], case["suite"]) for case in group["cases"]}),
                "case_keys": [[case["plan"], case["suite"], case["name"]] for case in group["cases"]],
            })
        (temporary / "groups.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if coverage is not None:
            (temporary / "execution-coverage.json").write_text(
                json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def write_plan(plan, output, input_directory):
    return write_groups(plan["groups"], output, input_directory, coverage=_coverage(plan))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=500)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--suite")
    parser.add_argument("--allowed-nodes", help="Comma-separated node counts allowed by resource policy")
    args = parser.parse_args()
    try:
        allowed = (None if args.allowed_nodes is None else
                   {int(value) for value in args.allowed_nodes.split(",")})
        plan = build_plan(args.input, args.max_cases, nodes=args.nodes,
                          suite=args.suite, allowed_nodes=allowed)
        write_plan(plan, args.output, args.input)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    coverage = _coverage(plan)
    coverage["exclusions"] = [
        {key: value for key, value in record.items() if key != "case_keys"}
        for record in coverage["exclusions"]
    ]
    print(json.dumps(coverage, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
