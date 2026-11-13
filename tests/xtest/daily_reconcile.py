#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate and summarize daily results without acting as a PR gate."""

import argparse
import csv
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import shutil
import sys

import artifact_cache
import run_cases


METRIC_FIELDS = run_cases.METRIC_FIELDS


def _read_json(path, errors):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"{path}: {error}")
        return {}


def _positive(value):
    try:
        number = Decimal(value)
    except (InvalidOperation, TypeError):
        return False
    return number.is_finite() and number > 0


def _nonnegative(value):
    try:
        number = Decimal(value)
    except (InvalidOperation, TypeError):
        return False
    return number.is_finite() and number >= 0


def _decimal(value):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError):
        return None
    return number if number.is_finite() else None


def _compare_record(record, row, key, errors):
    if set(record) != set(run_cases.RESULT_FIELDS):
        errors.append(f"Result.json schema mismatch for {'/'.join(key)}")
        return False
    valid = True
    for field in run_cases.RESULT_FIELDS:
        if field == "return_code":
            value = record[field]
            matches = ("" if value is None else str(value)) == row[field]
        elif field == "duration_seconds":
            left, right = _decimal(record[field]), _decimal(row[field])
            matches = left is not None and right is not None and left == right
        else:
            matches = record[field] == row[field]
        if not matches:
            errors.append(f"Result.json {field} mismatch for {'/'.join(key)}")
            valid = False
    return valid


def _check_metrics(directory, errors):
    path = directory / "metrics.csv"
    try:
        rows = run_cases._read_csv(path, METRIC_FIELDS)
        if not rows:
            raise ValueError("Performance metrics are empty")
        for row in rows:
            if not _positive(row["time"]):
                raise ValueError("Performance metrics contain a non-positive or invalid time")
            if any(not _nonnegative(row[key]) for key in ("size", "count")):
                raise ValueError("Performance metrics contain a negative or invalid size/count")
            if any(not _nonnegative(row[key]) for key in ("algbw", "busbw")):
                raise ValueError("Performance metrics contain a negative or invalid bandwidth")
            if row["wrong"] != "0" or not row["source_line"].isdigit() or int(row["source_line"]) <= 0:
                raise ValueError("Performance metrics contain invalid validation metadata")
    except (OSError, ValueError) as error:
        errors.append(f"{path}: {error}")


def reconcile(input_directory, profile, nodes, suite, results_directory, output_directory,
              pccl_revision, expected_sha256):
    selected = run_cases.load_inputs(input_directory, profile, nodes=nodes, suite=suite)
    expected = {(case["plan"], case["suite"], case["name"]): case for case in selected["cases"]}
    results = Path(results_directory)
    output = Path(output_directory)
    if output.exists():
        raise ValueError("Reconciliation output directory already exists")
    output.mkdir(parents=True)
    errors = []
    expected_build = {"schema_version": 1, "pccl_revision": pccl_revision,
                      "pccl_tests_revision": selected["metadata"]["runner_revision"],
                      "build_profile": profile, "artifact_sha256": expected_sha256}
    try:
        build_record = artifact_cache.read_record(results / "build.json")
        if build_record != expected_build:
            errors.append("Build identity differs from the selected profile, source or verified archive")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"{results / 'build.json'}: {error}")
    rows = []
    try:
        rows = run_cases._read_csv(results / "results.csv", run_cases.RESULT_FIELDS)
    except (OSError, ValueError) as error:
        errors.append(f"{results / 'results.csv'}: {error}")
    actual = {}
    counts = {}
    for row in rows:
        key = (row["plan"], row["suite"], row["name"])
        if not all(run_cases.SAFE_NAME.fullmatch(value) for value in key):
            errors.append(f"Unsafe result key: {key}")
            continue
        if key in actual:
            errors.append(f"Duplicate result: {'/'.join(key)}")
            continue
        actual[key] = row
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        if key not in expected:
            errors.append(f"Unexpected result: {'/'.join(key)}")
            continue
        case = expected[key]
        directory = results.joinpath(*key)
        record = _read_json(directory / "result.json", errors)
        if not _compare_record(record, row, key, errors):
            continue
        if row["check"] != case["check"]:
            errors.append(f"Result check mismatch for {'/'.join(key)}")
        if row["status"] != "PASSED":
            errors.append(f"{'/'.join(key)}: {row['status']}: {row['error']}")
            continue
        if row["return_code"] != "0":
            errors.append(f"Passed result has a nonzero return code: {'/'.join(key)}")
        log = directory / "raw.log"
        if not log.is_file() or log.stat().st_size == 0:
            errors.append(f"Missing or empty raw log: {'/'.join(key)}")
            continue
        try:
            expected_metrics = run_cases._check_result(case, 0, log)
        except ValueError as error:
            errors.append(f"{'/'.join(key)}: raw log validation failed: {error}")
            continue
        if case["check"] == "performance":
            try:
                actual_metrics = run_cases._read_csv(directory / "metrics.csv", METRIC_FIELDS)
            except (OSError, ValueError) as error:
                errors.append(f"{directory / 'metrics.csv'}: {error}")
                continue
            expected_metrics = [{field: str(value) for field, value in values.items()}
                                for values in expected_metrics]
            if actual_metrics != expected_metrics:
                errors.append(f"Performance metrics do not match the raw log: {'/'.join(key)}")
            _check_metrics(directory, errors)
    for key in sorted(set(expected) - set(actual)):
        errors.append(f"Missing result: {'/'.join(key)}")
    summary = _read_json(results / "summary.json", errors)
    if summary.get("total") != len(rows):
        errors.append(f"Summary total {summary.get('total')} does not match {len(rows)} result rows")
    if summary.get("counts") != counts:
        errors.append(f"Summary counts {summary.get('counts')} do not match {counts}")
    report = {
        "ok": not errors,
        "profile": profile,
        "nodes": nodes,
        "suite": suite,
        "source_revision": selected["metadata"]["source_revision"],
        "build_sha256": expected_sha256,
        "expected": len(expected),
        "actual": len(actual),
        "counts": counts,
        "errors": errors,
    }
    (output / "reconciliation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name in ("results.csv", "summary.json"):
        if (results / name).is_file():
            shutil.copyfile(results / name, output / name)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--profile", choices=("standard", "notsm"), required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--suite")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pccl-revision", required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    try:
        report = reconcile(args.input, args.profile, args.nodes, args.suite, args.results, args.output,
                           args.pccl_revision, args.expected_sha256)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
