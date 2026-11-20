#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the small, public 810E PR sanity bundle from shared families."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import tempfile

import generate_cases
import platform_cases
import run_cases


FAMILIES = ("sanity_m890p_register", "op_perf_shm", "postsil_daily_shm",
            "register_test_shm", "sym_test_shm", "pccl_fun_rdma_shm")
PR_CARDS_PER_NODE = 2


def _pr_platform():
    profile = platform_cases.load_platform("oam810e_eic")
    profile["cards_per_node"] = PR_CARDS_PER_NODE
    profile["families"]["op_perf_810e_eic"]["numeric"] = {
        "ranks": {"16": 4}, "np": {"16": 4}, "npernode": {"8": 2},
    }
    profile["families"]["pccl_fun_rdma_shm"]["numeric"] = {
        "ranks": {"2": 4}, "nprocs": {"1": 2},
    }
    return profile


def build_pr_bundle(source_revision, runner_revision):
    definitions = Path(__file__).resolve().parent / "test_lists"
    raw = [generate_cases.generate_bundle(
        definitions / f"{family}.json", definitions / "suites.json",
        source_revision, runner_revision, "smoke") for family in FAMILIES]
    variants = [generate_cases.generate_bundle(
        definitions / "op_perf_shm.json", definitions / "suites.json",
        source_revision, runner_revision, "full")]
    converted = platform_cases.apply_platforms(
        raw, _pr_platform(), variants)
    combined = generate_cases.combine_bundles(converted)
    notsm = generate_cases.generate_bundle(
        definitions / "sanity_m890p_register.json", definitions / "suites.json",
        source_revision, runner_revision, "smoke")
    notsm = platform_cases.apply_platform(notsm, _pr_platform())
    generation = {**combined["generation"], "profile": "pr",
                  "notsm_case_count": len(notsm["rows"])}
    return {"metadata": combined["metadata"], "generation": generation,
            "suites": combined["suites"], "standard": combined["rows"],
            "notsm": notsm["rows"], "environments": combined["environments"]}


def write_pr_bundle(bundle, output, env_root):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (temporary / "envs").mkdir()
        for name, value in (("export.json", bundle["metadata"]),
                            ("generation.json", bundle["generation"])):
            (temporary / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
        for name, fields, rows in (("suites.csv", run_cases.SUITE_FIELDS, bundle["suites"]),
                                  ("standard.csv", run_cases.CASE_FIELDS, bundle["standard"]),
                                  ("notsm.csv", run_cases.CASE_FIELDS, bundle["notsm"])):
            with (temporary / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        for name in bundle["environments"]:
            shutil.copyfile(Path(env_root) / name, temporary / "envs" / name)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--runner-revision", required=True)
    parser.add_argument("--env-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        bundle = build_pr_bundle(args.source_revision, args.runner_revision)
        write_pr_bundle(bundle, args.output, args.env_root)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(bundle["generation"], sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
