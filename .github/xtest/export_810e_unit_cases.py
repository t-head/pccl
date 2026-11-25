#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export the dedicated OAM-810E native unit Cases to public JSON families."""

import argparse
import ast
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
import shlex


POSTSIL_CSV = "csv/postsil_unit_test/pccl_810E_postsil_unit_test_plan.csv"
ALL_GATHER_CSV = "csv/all_gather_unit_test/pccl_810E_all_gather_unit_test_plan.csv"
POSTSIL_PLAN = "pccl_810e_postsil_unit_test_single_node1_plan"
POSTSIL_SUITE = "postsil_unit_test_810e_single_node_suite"
ALL_GATHER_PLAN = "pccl_810e_all_gather_unit_test_single_node1_plan"
ALL_GATHER_SUITE = "all_gather_unit_test_810e_single_node_suite"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def rows_sha256(rows):
    return sha256(canonical(rows)).hexdigest()


def source_case(root, relative):
    path = root / "v2" / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    name = command = None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        target = node.targets[0].id
        if target == "metadata" and isinstance(node.value, ast.Call):
            for keyword in node.value.keywords:
                if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                    name = keyword.value.value
        if target == "pipeline" and isinstance(node.value, ast.List) and node.value.elts:
            runner = node.value.elts[0]
            for keyword in runner.keywords:
                if keyword.arg == "mpi_command" and isinstance(keyword.value, ast.Constant):
                    command = keyword.value.value
    if not isinstance(name, str) or not SAFE_NAME.fullmatch(name) or not isinstance(command, str):
        raise ValueError(f"Unsupported source Case: {relative}")
    return name, command


def native_case(root, relative):
    name, source = source_case(root, relative)
    tokens = shlex.split(source)
    if not tokens or tokens[0] != "python" or tokens[1] != "unit_test.py":
        raise ValueError(f"Unsupported native wrapper: {relative}")
    try:
        binary_index, binary = next((index, token[3:]) for index, token in enumerate(tokens)
                                    if token.startswith("-t="))
    except StopIteration as error:
        raise ValueError(f"Missing native binary: {relative}") from error
    if binary in ("p2pBandwidthLatencyTest", "alltoallvTest"):
        raise ValueError(f"Public build lacks {binary}")
    if binary not in ("collPcclUnitTest", "cudaGraphUnitTest"):
        raise ValueError(f"Unsupported native binary {binary}: {relative}")
    if "--use_ext_kernel=1" in tokens:
        raise ValueError("Public hggcGraphUnitTest lacks --use_ext_kernel")
    environment = []
    options = []
    index = binary_index + 1
    while index < len(tokens):
        token = tokens[index]
        if token in ("--auto_detect=1", "--auto-detect", "--run_type=ppu_umd") \
                or token.startswith("--data_size="):
            index += 1
            continue
        if token.startswith(("--algo=", "--proto=", "--devices=")):
            option_name, value = token.split("=", 1)
            if option_name == "--algo":
                environment.append(f"NCCL_ALGO={value}")
            elif option_name == "--proto":
                environment.append(f"NCCL_PROTO={value}")
            else:
                environment.append(f"HGGC_VISIBLE_DEVICES={value}")
            index += 1
            continue
        if token.startswith("--test_config_file="):
            value = token.split("=", 1)[1]
            if "cuda_graph_test_configs/" in value:
                directory = "hggc_graph_test_configs"
            elif "feature_test_configs/" in value:
                directory = "feature_test_configs"
            else:
                raise ValueError(f"Unknown graph config path: {value}")
            options.extend(["--test_config_file",
                            f"{{test_root}}/pccl/tests/unit_test/{directory}/{Path(value).name}"])
            index += 1
            continue
        if token in ("--algo", "--proto", "--devices", "--test_config_file"):
            value = tokens[index + 1]
            if token == "--algo":
                environment.append(f"NCCL_ALGO={value}")
            elif token == "--proto":
                environment.append(f"NCCL_PROTO={value}")
            elif token == "--devices":
                environment.append(f"HGGC_VISIBLE_DEVICES={value}")
            else:
                if "cuda_graph_test_configs/" in value:
                    directory = "hggc_graph_test_configs"
                elif "feature_test_configs/" in value:
                    directory = "feature_test_configs"
                else:
                    raise ValueError(f"Unknown graph config path: {value}")
                options.extend([token, f"{{test_root}}/pccl/tests/unit_test/{directory}/{Path(value).name}"])
            index += 2
            continue
        options.append(token)
        index += 1
    public_binary = "hggcGraphUnitTest" if binary == "cudaGraphUnitTest" else binary
    command = " ".join(environment + [f"build/{public_binary}"] + options)
    return {"nodes": 1, "name": name, "check": "exit_code", "cmd": command}


def all_gather_case(root, relative):
    name, source = source_case(root, relative)
    expected = "mpirun --allow-run-as-root -np "
    if not source.startswith(expected) or "./build/cudaGraphUnitTest" not in source:
        raise ValueError(f"Unsupported all_gather wrapper: {relative}")
    command = source.replace("./build/cudaGraphUnitTest", "build/hggcGraphUnitTest")
    command = command.replace("./feature_test_configs/allgather_unit_test.json",
                              "{test_root}/pccl/tests/unit_test/feature_test_configs/allgather_unit_test.json")
    return {"nodes": 1, "name": name, "check": "exit_code", "cmd": command}


def definition(family, plan, suite, cases, source_revision, selection):
    rows = [{"plan": plan, "suite": suite, "name": case["name"],
             "check": case["check"], "cmd": case["cmd"]} for case in cases]
    rows.sort(key=lambda row: row["name"])
    provenance = {
        "source_revision": source_revision,
        "source_selection": selection,
        "source_case_count": len(rows),
        "source_rows_sha256": rows_sha256(rows),
        "source_suites": [f"{plan}/{suite}"],
        "validation": "dedicated-810e-source-command-adaptation",
    }
    return {"schema_version": 2, "family": family,
            "defaults": {"check": "exit_code", "case_name": "native_case"},
            "axes": {}, "layouts": {"native": [{"name": "native", "values": {}}]},
            "bindings": {"1": {"plan": plan, "suite": suite}}, "groups": [],
            "explicit_cases": cases, "profiles": {"full": {"include_explicit": True}},
            "provenance": provenance}, rows


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def export(root, output, source_revision):
    if not HEX40.fullmatch(source_revision):
        raise ValueError("Source revision must be a full lowercase Git SHA")
    postsil = []
    excluded = {}
    for row in csv.DictReader((root / POSTSIL_CSV).open(encoding="utf-8")):
        try:
            postsil.append(native_case(root, row["case path"]))
        except ValueError as error:
            reason = str(error)
            if not any(token in reason for token in ("Public build lacks", "lacks --use_ext_kernel")):
                raise
            excluded[reason] = excluded.get(reason, 0) + 1
    all_gather = [all_gather_case(root, row["case path"])
                  for row in csv.DictReader((root / ALL_GATHER_CSV).open(encoding="utf-8"))]
    coll = [case for case in postsil if "build/collPcclUnitTest" in case["cmd"]]
    graph = [case for case in postsil if "build/hggcGraphUnitTest" in case["cmd"]]
    if (len(coll), len(graph), len(all_gather)) != (1252, 45, 3):
        raise ValueError(f"Unexpected supported counts: coll={len(coll)}, graph={len(graph)}, all_gather={len(all_gather)}")
    expected_exclusions = {"Public build lacks p2pBandwidthLatencyTest": 308,
                           "Public build lacks alltoallvTest": 35,
                           "Public hggcGraphUnitTest lacks --use_ext_kernel": 3}
    if excluded != expected_exclusions:
        raise ValueError(f"Unexpected exclusions: {excluded}")
    coll_candidates = [case for case in coll if "--num_gpus=2" in case["cmd"]
                       and "HGGC_VISIBLE_DEVICES" not in case["cmd"]]

    def workload(case):
        match = re.search(r"--num_elems=([0-9]+)", case["cmd"])
        if match:
            return int(match.group(1))
        return int(re.search(r"--max_num_elems=([0-9]+)", case["cmd"]).group(1))

    coll_smoke = min(coll_candidates, key=workload)
    graph_smoke = next(case for case in graph if "--num_threads=2" in case["cmd"]
                       and "HGGC_VISIBLE_DEVICES=" in case["cmd"]
                       and len(case["cmd"].split("HGGC_VISIBLE_DEVICES=", 1)[1].split()[0].split(",")) == 2)
    documents = {}
    rows_by_family = {}
    for family, plan, suite, cases, selection in (
            ("postsil_unit_test_810e", POSTSIL_PLAN, POSTSIL_SUITE, postsil, POSTSIL_CSV),
            ("all_gather_unit_test_810e", ALL_GATHER_PLAN, ALL_GATHER_SUITE, all_gather, ALL_GATHER_CSV),
            ("postsil_unit_test_810e_smoke", POSTSIL_PLAN, POSTSIL_SUITE,
             [coll_smoke, graph_smoke], "two-card PR smoke")):
        document, rows = definition(family, plan, suite, cases, source_revision, selection)
        if family == "postsil_unit_test_810e_smoke":
            document["profiles"] = {"full": {"include_explicit": True},
                                    "pr": {"include_explicit": True}, "smoke": {"include_explicit": True}}
        documents[family] = document
        rows_by_family[family] = rows
        write_json(output / f"{family}.json", document)
    suites = [
        {"plan": POSTSIL_PLAN, "suite": POSTSIL_SUITE, "nodes": 1, "cards_per_node": 16,
         "timeout": 3600, "env": "op_perf.env"},
        {"plan": ALL_GATHER_PLAN, "suite": ALL_GATHER_SUITE, "nodes": 1, "cards_per_node": 16,
         "timeout": 3600, "env": "op_perf.env"},
    ]
    manifest = {
        "schema_version": 1, "source_revision": source_revision,
        "selection": "dedicated-810e-native-unit",
        "case_count": 1300, "suite_count": 2,
        "case_rows_sha256": rows_sha256(sorted(
            rows_by_family["postsil_unit_test_810e"] + rows_by_family["all_gather_unit_test_810e"],
            key=lambda row: (row["plan"], row["suite"], row["name"]))),
        "suites_sha256": rows_sha256(sorted(suites, key=lambda row: (row["plan"], row["suite"]))),
        "families": [
            {"family": family, "case_count": len(rows),
             "source_rows_sha256": rows_sha256(rows),
             "definition_sha256": sha256((output / f"{family}.json").read_bytes()).hexdigest()}
            for family, rows in rows_by_family.items()
        ],
        "exclusions": [{"case_count": count, "reason": reason}
                       for reason, count in sorted(excluded.items())],
    }
    write_json(output / "source-export-810e.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = export(args.source_root.resolve(), args.output.resolve(), args.source_revision)
    except (OSError, ValueError, KeyError, IndexError, TypeError, SyntaxError) as error:
        parser.error(str(error))
    print(json.dumps({"cases": manifest["case_count"], "suites": manifest["suite_count"],
                      "exclusions": manifest["exclusions"]}, sort_keys=True))


if __name__ == "__main__":
    main()
