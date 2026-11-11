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


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_groups(directory, max_cases, nodes=None, suite=None):
    if type(max_cases) is not int or max_cases <= 0:
        raise ValueError("max_cases must be a positive integer")
    grouped = {}
    metadata = None
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
            key = (profile, spec["nodes"], spec["cards_per_node"], spec["env"])
            grouped.setdefault(key, []).append(case)
    if metadata is None:
        raise ValueError("No Cases match the requested selection")
    groups = []
    for key in sorted(grouped):
        profile, group_nodes, cards, env = key
        cases = sorted(grouped[key], key=lambda case: (case["plan"], case["suite"], case["name"]))
        for start in range(0, len(cases), max_cases):
            chunk = cases[start:start + max_cases]
            groups.append({
                "id": f"g{len(groups) + 1:03d}",
                "profile": profile,
                "nodes": group_nodes,
                "cards_per_node": cards,
                "env": env,
                "cases": chunk,
                "metadata": metadata,
            })
    return groups


def write_groups(groups, output, input_directory):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Group output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        manifest = []
        for group in groups:
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
                "cards_per_node": group["cards_per_node"], "env": group["env"],
                "case_count": len(group["cases"]),
                "plan_suites": sorted({(case["plan"], case["suite"]) for case in group["cases"]}),
                "case_keys": [[case["plan"], case["suite"], case["name"]] for case in group["cases"]],
            })
        (temporary / "groups.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=500)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--suite")
    args = parser.parse_args()
    try:
        groups = build_groups(args.input, args.max_cases, nodes=args.nodes, suite=args.suite)
        write_groups(groups, args.output, args.input)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps([{key: value for key, value in group.items() if key not in ("cases", "metadata")}
                      for group in groups], sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
