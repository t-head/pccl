#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reconcile private hardware results with the exact CSV selection and verified builds."""

import argparse
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import shutil

import artifact_cache
import run_cases


def _regular_file(path, description):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a regular file, not a symlink: {path}")
    return path


def _read_json(path, description):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {description}: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Invalid JSON constant in {description}: {value}")

    data = json.loads(_regular_file(path, description).read_text(encoding="utf-8"),
                      object_pairs_hook=unique, parse_constant=invalid)
    if not isinstance(data, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return data


def _key(row):
    return row["plan"], row["suite"], row["name"]


def _duration(record, row):
    value = record["duration_seconds"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid duration in per-Case result.json")
    try:
        json_duration = Decimal(str(value))
        csv_duration = Decimal(row["duration_seconds"])
    except InvalidOperation as error:
        raise ValueError("Invalid duration in results.csv") from error
    if not csv_duration.is_finite() or csv_duration < 0 or json_duration != csv_duration:
        raise ValueError("Per-Case and CSV durations differ")


def _validate_case(source, case, row):
    key = _key(case)
    label = "/".join(key)
    if row["check"] != case["check"]:
        raise ValueError(f"{label}: result check differs from the selected Case")
    expected = {"plan": case["plan"], "suite": case["suite"], "name": case["name"],
                "check": case["check"], "status": "PASSED", "return_code": 0,
                "error": "", "raw_log": "raw.log"}
    directory = source.joinpath(*key)
    record = _read_json(directory / "result.json", f"{label} per-Case record")
    if set(record) != set(run_cases.RESULT_FIELDS):
        raise ValueError(f"{label}: unexpected per-Case result fields")
    for field, value in expected.items():
        if record[field] != value:
            raise ValueError(f"{label}: invalid per-Case {field}: {record[field]!r}")
    for field in ("plan", "suite", "name", "check", "status", "error", "raw_log"):
        if row[field] != record[field]:
            raise ValueError(f"{label}: per-Case and CSV {field} values differ")
    if row["return_code"] != "0":
        raise ValueError(f"{label}: CSV return code is not zero")
    _duration(record, row)
    raw_log = _regular_file(directory / "raw.log", f"{label} raw log")
    if raw_log.stat().st_size == 0:
        raise ValueError(f"{label}: raw log is empty")
    metrics = run_cases._check_result(case, 0, raw_log)
    metric_path = directory / "metrics.csv"
    if case["check"] == "performance":
        metric_rows = run_cases._read_csv(
            _regular_file(metric_path, f"{label} performance metrics"), run_cases.METRIC_FIELDS)
        expected_metrics = [{field: str(value) for field, value in values.items()} for values in metrics]
        if not metric_rows or metric_rows != expected_metrics:
            raise ValueError(f"{label}: performance metrics do not match the raw log")
        return metric_path
    if metric_path.exists() or metric_path.is_symlink():
        raise ValueError(f"{label}: unexpected performance metrics")
    return None


def _validate_group(results, inputs, identity):
    name = inputs["name"]
    source = results / name
    expected_identity = {"schema_version": 1, "pccl_revision": identity["pccl_revision"],
                         "pccl_tests_revision": inputs["metadata"]["runner_revision"],
                         "build_profile": inputs["profile"], "artifact_sha256": identity["artifact_sha256"]}
    actual_identity = artifact_cache.read_record(
        _regular_file(source / "build.json", f"{name} build identity"))
    if actual_identity != expected_identity:
        raise ValueError(f"{name}: build identity differs from the verified {inputs['profile']} archive")
    cases = inputs["cases"]
    summary = _read_json(source / "summary.json", f"{name} summary")
    expected_summary = {"total": len(cases), "counts": {"PASSED": len(cases)}}
    if summary != expected_summary:
        raise ValueError(f"{name}: invalid summary; expected {expected_summary}, got {summary}")
    rows = run_cases._read_csv(
        _regular_file(source / "results.csv", f"{name} result rows"), run_cases.RESULT_FIELDS)
    keys = [_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name}: duplicate Case result keys")
    expected = {_key(case): case for case in cases}
    if set(keys) != set(expected):
        missing = sorted(set(expected) - set(keys))
        unexpected = sorted(set(keys) - set(expected))
        raise ValueError(f"{name}: Case coverage differs; missing={missing}, unexpected={unexpected}")
    metrics = []
    for row in rows:
        metric = _validate_case(source, expected[_key(row)], row)
        if metric is not None:
            metrics.append(metric)
    actual_metrics = set(source.glob("*/*/*/metrics.csv"))
    if set(metrics) != actual_metrics:
        raise ValueError(f"{name}: unexpected or misplaced performance metrics files")
    return {"source": source, "metrics": metrics, "passed": len(cases)}


def load_selections(directory, cards_per_node=8):
    """Reject any exported Case that the fixed CI resource layout cannot execute."""
    if type(cards_per_node) is not int or cards_per_node <= 0:
        raise ValueError("CI cards per node must be a positive integer")
    schedules = {"standard": ("standard", 1), "notsm": ("notsm", 1),
                 "cluster": ("standard", 2)}
    profiles = {profile: run_cases.load_inputs(directory, profile) for profile in ("standard", "notsm")}
    for profile, inputs in profiles.items():
        supported_nodes = {nodes for selected_profile, nodes in schedules.values()
                           if selected_profile == profile}
        for case in inputs["cases"]:
            suite = case["suite_spec"]
            if suite["nodes"] not in supported_nodes or suite["cards_per_node"] != cards_per_node:
                raise ValueError(
                    f"Unsupported CI resources for {'/'.join(_key(case))}: profile={profile}, "
                    f"nodes={suite['nodes']}, cards_per_node={suite['cards_per_node']}")
    selections = {}
    for name, (profile, nodes) in schedules.items():
        inputs = profiles[profile]
        cases = [case for case in inputs["cases"] if case["suite_spec"]["nodes"] == nodes]
        if not cases:
            raise ValueError(f"{name}: no Cases for the scheduled CI resources")
        selections[name] = {**inputs, "cases": cases, "name": name}
    return selections


def reconcile(directory, results, public, pccl_revision, standard_sha256, notsm_sha256,
              *, cards_per_node=8):
    """Validate every selected Case before publishing only non-sensitive summaries and metrics."""
    directory, results, public = Path(directory), Path(results), Path(public)
    if not artifact_cache.HEX40.fullmatch(pccl_revision):
        raise ValueError("Invalid PCCL revision")
    for value in (standard_sha256, notsm_sha256):
        if not artifact_cache.HEX64.fullmatch(value):
            raise ValueError("Invalid expected archive SHA-256")
    selections = load_selections(directory, cards_per_node)
    validated = {}
    errors = []
    for name, inputs in selections.items():
        archive_sha256 = notsm_sha256 if inputs["profile"] == "notsm" else standard_sha256
        identity = {"pccl_revision": pccl_revision, "artifact_sha256": archive_sha256}
        try:
            validated[name] = _validate_group(results, inputs, identity)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{name}: {error}")
    if errors:
        raise ValueError("\n".join(errors))
    if public.exists() or public.is_symlink():
        raise ValueError("Public results directory already exists")
    public.mkdir(parents=True)
    for name, group in validated.items():
        target = public / name
        target.mkdir()
        for filename in ("build.json", "summary.json", "results.csv"):
            shutil.copyfile(group["source"] / filename, target / filename)
        for source in group["metrics"]:
            destination = target / source.parent.relative_to(group["source"])
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination / "metrics.csv")
        print(f"{name}: {group['passed']}/{group['passed']} passed; build identity and Case records verified")
    return {name: group["passed"] for name, group in validated.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--pccl-revision", required=True)
    parser.add_argument("--standard-sha256", required=True)
    parser.add_argument("--notsm-sha256", required=True)
    parser.add_argument("--cards-per-node", type=int, default=8)
    args = parser.parse_args()
    try:
        reconcile(args.input, args.results, args.public, args.pccl_revision,
                  args.standard_sha256, args.notsm_sha256, cards_per_node=args.cards_per_node)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        for line in str(error).splitlines():
            print(f"::error::{line}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
