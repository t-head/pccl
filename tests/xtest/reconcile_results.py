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

import run_cases


METRIC_FIELDS = ["size", "count", "type", "redop", "root", "placement", "time_kind",
                 "time", "algbw", "busbw", "wrong", "source_line"]


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


def _check_metrics(directory, errors):
    path = directory / "metrics.csv"
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != METRIC_FIELDS:
                raise ValueError(f"Unexpected metrics header: {reader.fieldnames}")
            rows = list(reader)
        if not rows:
            raise ValueError("Performance metrics are empty")
        for row in rows:
            if any(not _positive(row[key]) for key in ("size", "count", "time", "algbw", "busbw")):
                raise ValueError("Performance metrics contain a non-positive or invalid number")
            if row["wrong"] != "0" or not row["source_line"].isdigit() or int(row["source_line"]) <= 0:
                raise ValueError("Performance metrics contain invalid validation metadata")
    except (OSError, ValueError) as error:
        errors.append(f"{path}: {error}")


def reconcile(input_directory, profile, nodes, suite, results_directory, output_directory):
    selected = run_cases.load_inputs(input_directory, profile, nodes=nodes, suite=suite)
    expected = {(case["plan"], case["suite"], case["name"]): case for case in selected["cases"]}
    results = Path(results_directory)
    output = Path(output_directory)
    if output.exists():
        raise ValueError("Reconciliation output directory already exists")
    output.mkdir(parents=True)
    errors = []
    rows = []
    try:
        with (results / "results.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != run_cases.RESULT_FIELDS:
                raise ValueError(f"Unexpected results header: {reader.fieldnames}")
            rows = list(reader)
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
        for field in ("plan", "suite", "name", "check", "status", "return_code"):
            record_value = record.get(field)
            if field == "return_code" and record_value is not None:
                record_value = str(record_value)
            if record_value is None:
                record_value = ""
            if record_value != row[field]:
                errors.append(f"Result.json {field} mismatch for {'/'.join(key)}")
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
        if case["check"] == "performance":
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
    args = parser.parse_args()
    try:
        report = reconcile(args.input, args.profile, args.nodes, args.suite, args.results, args.output)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
