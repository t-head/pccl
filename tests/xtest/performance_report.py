#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a complete PCCL performance workbook from validated NAS results.

Execution status and performance-data status are deliberately independent.
In particular, an exit_code Case never becomes a performance-gated Case merely
because its raw log contains a benchmark table.
"""

import argparse
import csv
from collections import Counter
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import re
import shlex
from zoneinfo import ZoneInfo

import run_cases


SCHEMA_VERSION = 1
OP_PERF_SUITES = {
    "op_perf_shm_single_node_suite",
    "op_perf_shm_multi_nodes2_suite",
    "op_perf_shm_multi_nodes4_suite",
    "op_perf_shm_multi_nodes8_suite",
    "op_perf_shm_2_rack_icn8_suite",
    "op_perf_shm_2_rack_icn16_suite",
    "op_perf_shm_2_rack_icn32_suite",
    "op_perf_shm_2_rack_icn64_suite",
    "op_perf_rdma_shm_multi_nodes2_suite",
    "op_perf_rdma_shm_multi_nodes4_suite",
    "op_perf_rdma_shm_multi_nodes8_suite",
}
LOCAL_REGISTER_SUITES = {
    "local_register_perf_shm_single_node_suite",
    "local_register_perf_shm_multi_nodes2_suite",
}
PLACEMENTS = ("out-of-place", "in-place")


def _key(case):
    return case["plan"], case["suite"], case["name"]


def _category(case):
    if case["suite"] in OP_PERF_SUITES:
        return "普通 op_perf"
    if case["suite"] in LOCAL_REGISTER_SUITES:
        return "local-register 性能"
    if case["check"] == "performance":
        return "其他性能"
    return None


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def write_selection(input_directory, profile, output, nodes=None, suite=None):
    """Retain every reportable Case before resource-policy group exclusions."""
    selected = run_cases.load_inputs(input_directory, profile, nodes=nodes, suite=suite)
    cases = []
    for case in selected["cases"]:
        category = _category(case)
        if category is None:
            continue
        suite = case["suite_spec"]
        cases.append({
            "plan": case["plan"], "suite": case["suite"], "name": case["name"],
            "check": case["check"], "cmd": case["cmd"], "category": category,
            "nodes": suite["nodes"], "cards_per_node": suite["cards_per_node"],
        })
    cases.sort(key=lambda row: (_key(row), row["nodes"]))
    keys = [_key(row) for row in cases]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate performance-selection Case key")
    manifest = {"schema_version": SCHEMA_VERSION, "profile": profile, "cases": cases}
    _write_json(output, manifest)
    return manifest


def _load_selection(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if (set(manifest) != {"schema_version", "profile", "cases"}
            or manifest["schema_version"] != SCHEMA_VERSION
            or not isinstance(manifest["cases"], list)):
        raise ValueError("Unsupported performance-selection manifest")
    keys = []
    required = {"plan", "suite", "name", "check", "cmd", "category", "nodes", "cards_per_node"}
    for case in manifest["cases"]:
        if set(case) != required or _category(case) != case["category"]:
            raise ValueError("Invalid performance-selection Case")
        if type(case["nodes"]) is not int or case["nodes"] <= 0 \
                or type(case["cards_per_node"]) is not int or case["cards_per_node"] <= 0:
            raise ValueError("Invalid performance-selection resources")
        keys.append(_key(case))
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate performance-selection Case key")
    return manifest


def _size(value):
    match = re.fullmatch(r"([0-9]+)([KMG]?)B?", value, re.I)
    if not match:
        raise ValueError(f"Unsupported benchmark size: {value}")
    scale = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[match.group(2).upper()]
    return int(match.group(1)) * scale


def expected_sizes(command):
    """Derive the expected logarithmic sweep from the existing Case command."""
    tokens = shlex.split(command)
    options = {}
    for flag in ("-b", "-e", "-f"):
        try:
            index = tokens.index(flag)
        except ValueError:
            return []
        if index + 1 >= len(tokens):
            return []
        options[flag] = tokens[index + 1]
    try:
        begin, end, factor = _size(options["-b"]), _size(options["-e"]), int(options["-f"])
    except (ValueError, TypeError):
        return []
    if begin <= 0 or end < begin or factor <= 1:
        return []
    values = []
    current = begin
    while current <= end and len(values) < 64:
        values.append(current)
        current *= factor
    return values


def _ranks(case):
    match = re.search(r"(?:^|_)ranks_([0-9]+)(?:_|$)", case["name"])
    return int(match.group(1)) if match else case["nodes"] * case["cards_per_node"]


def _topology(case):
    ranks = _ranks(case)
    return f"单机{ranks}卡" if case["nodes"] == 1 else f"{case['nodes']}机{ranks}卡"


def _path_config(case):
    if "2_rack" in case["suite"]:
        return "跨机架 ICN 配置"
    if "rdma" in case["suite"]:
        return "RDMA 配置"
    return "常规配置"


def _operation_dtype(command):
    tokens = shlex.split(command)
    operation = ""
    dtype = ""
    for index, token in enumerate(tokens):
        name = Path(token).name
        if name.endswith("_perf"):
            operation = name
        if token == "--op" and index + 1 < len(tokens):
            operation = tokens[index + 1]
        if token.startswith("--op="):
            operation = token.split("=", 1)[1]
        if token in ("-d", "--data-type") and index + 1 < len(tokens):
            dtype = tokens[index + 1]
    return operation, dtype


def _build_sha(directory):
    path = directory / "build.json"
    if not path.is_file():
        return ""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ""
    if not isinstance(value, dict):
        return ""
    return (value.get("archive_sha256") or value.get("artifact_sha256")
            or value.get("build_sha256") or value.get("sha256", ""))


def _number(value):
    decimal = Decimal(value)
    return int(decimal) if decimal == decimal.to_integral_value() else float(decimal)


def _alignment(rows, expected):
    """Match sizes in order while preserving duplicates and zero placeholders."""
    if not expected:
        return 0, True
    index = 0
    missing = 0
    for row in rows:
        size = int(row["size"])
        if size == 0:
            if index < len(expected):
                index += 1
        elif index < len(expected) and size == expected[index]:
            index += 1
        elif index and size == expected[index - 1]:
            # Repeated measurement of the preceding expected size.
            continue
        elif index < len(expected) and size in expected[index + 1:]:
            # Missing expected sizes are explicit, but do not hide later rows.
            following = expected.index(size, index + 1)
            missing += following - index
            index = following + 1
        else:
            return missing + len(expected) - index, False
    missing += len(expected) - index
    return missing, missing == 0


def _parse_case(case, log_path, expected):
    """Use the executor's parser, then add non-gating completeness metadata."""
    text = log_path.read_bytes().decode("utf-8", errors="replace")
    parsed = run_cases.parse_performance(text, allow_unsupported=True, skip_malformed=True)
    rows = []
    for placement in PLACEMENTS:
        for source in (row for row in parsed if row["placement"] == placement):
            if Decimal(source["time"]) <= 0:
                raise ValueError("Performance time must be positive")
            if Decimal(source["algbw"]) < 0 or Decimal(source["busbw"]) < 0:
                raise ValueError("Performance bandwidth must be nonnegative")
            if int(source["size"]) < 0 or int(source["count"]) < 0:
                raise ValueError("Performance size and count must be nonnegative")
            wrong = source["wrong"]
            rows.append({"placement": placement, "size": int(source["size"]),
                         "type": source["type"],
                         "count": int(source["count"]), "time_kind": source["time_kind"],
                         "time": _number(source["time"]), "algbw": _number(source["algbw"]),
                         "busbw": _number(source["busbw"]),
                         "wrong": int(wrong) if wrong.isdigit() else wrong,
                         "source_line": int(source["source_line"]),
                         "metric_status": "UNSUPPORTED" if wrong == "N/A" else "VALID"})
    valid_out_of_place = sum(row["placement"] == "out-of-place"
                             and row["metric_status"] == "VALID" for row in rows)
    out_rows = [row for row in parsed if row["placement"] == "out-of-place"]
    missing_out_of_place, _ = _alignment(out_rows, expected)
    for placement in PLACEMENTS:
        placement_rows = [row for row in parsed if row["placement"] == placement]
        missing, _ = _alignment(placement_rows, expected)
        for _ in range(missing):
            rows.append({"placement": placement, "size": None, "count": None,
                         "type": None,
                         "time_kind": "", "time": None, "algbw": None, "busbw": None,
                         "wrong": None, "source_line": None,
                         "metric_status": "MISSING_LOG_ROW"})
    complete = all(_alignment([row for row in parsed if row["placement"] == placement],
                              expected)[1] for placement in PLACEMENTS)
    observed_types = sorted({row["type"] for row in parsed})
    return {"rows": rows, "data_status": "VALID" if complete else "PARTIAL",
            "valid_out_of_place": valid_out_of_place, "missing_out_of_place": missing_out_of_place,
            "reason": "", "types": observed_types}


def _verification(groups, reconciliation_root):
    """Consume existing reconciliation errors instead of trusting CSV PASSED alone."""
    global_unverified = set()
    unverified_cases = set()
    mismatched_cases = set()
    root = Path(reconciliation_root)
    for group in groups:
        path = root / group["id"] / "reconciliation.json"
        if not path.is_file():
            global_unverified.add(group["id"])
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            global_unverified.add(group["id"])
            continue
        if not isinstance(report, dict):
            global_unverified.add(group["id"])
            continue
        errors = report.get("errors", [])
        if not isinstance(errors, list) or (not report.get("ok", False) and not errors):
            global_unverified.add(group["id"])
            continue
        for error in errors:
            if not isinstance(error, str):
                global_unverified.add(group["id"])
                continue
            if error.startswith(("Summary total ", "Summary counts ", "Unexpected result: ")):
                continue
            for values in group["case_keys"]:
                key = tuple(values)
                name = "/".join(key)
                if error.startswith(name + ": ") or error.endswith(" for " + name) \
                        or error.endswith(": " + name) \
                        or any(f"/{name}/{filename}: " in error
                               for filename in ("result.json", "metrics.csv", "raw.log")):
                    unverified_cases.add(key)
                    if not re.match(rf"^{re.escape(name)}: (?:FAILED|TIMEOUT|CANCELLED|NOT_RUN):", error):
                        mismatched_cases.add(key)
                    break
            else:
                global_unverified.add(group["id"])
    return global_unverified, unverified_cases, mismatched_cases


def _recover_result(results_root, group_id, key):
    path = results_root / group_id / Path(*key) / "result.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict) or not isinstance(record.get("status"), str):
        return None
    if not isinstance(record.get("check"), str) or "error" not in record \
            or not isinstance(record["error"], str):
        return None
    return record


def collect_report(selection_path, groups_path, coverage_path, results_root, metadata,
                   reconciliation_root):
    manifest = _load_selection(selection_path)
    groups = json.loads(Path(groups_path).read_text(encoding="utf-8"))
    coverage = json.loads(Path(coverage_path).read_text(encoding="utf-8"))
    results_root = Path(results_root)
    selected = {}
    for group in groups:
        for values in group["case_keys"]:
            key = tuple(values)
            if key in selected and selected[key] != group["id"]:
                raise ValueError(f"Duplicate group assignment: {key}")
            selected[key] = group["id"]

    global_unverified, unverified_cases, mismatched_cases = _verification(groups, reconciliation_root)

    selection = {_key(case): dict(case) for case in manifest["cases"]}
    actual = {}
    planned_actual = {}
    seen = set()
    group_build = {}
    result_counts = Counter()
    for group in groups:
        group_id = group["id"]
        group_build[group_id] = _build_sha(results_root / group_id)
        path = results_root / group_id / "results.csv"
        try:
            rows = run_cases._read_csv(path, run_cases.RESULT_FIELDS) if path.is_file() else []
        except (OSError, ValueError, csv.Error):
            rows = []
        if not path.is_file() or not rows:
            global_unverified.add(group_id)
        for row in rows:
            key = _key(row)
            if key not in selected and key not in selection:
                continue
            if key in seen:
                unverified_cases.add(key)
                mismatched_cases.add(key)
            seen.add(key)
            if key not in actual:
                actual[key] = (group_id, row)
            if selected.get(key) == group_id and key not in planned_actual:
                planned_actual[key] = (group_id, row)
                actual[key] = (group_id, row)
                result_counts[row["status"]] += 1
        for key, planned_group in selected.items():
            if planned_group != group_id or key in planned_actual:
                continue
            record = _recover_result(results_root, group_id, key)
            if record is not None:
                if key in seen:
                    unverified_cases.add(key)
                    mismatched_cases.add(key)
                planned_actual[key] = (group_id, record)
                actual[key] = (group_id, record)
                result_counts[record["status"]] += 1

    ordered_cases = sorted(selection.values(),
                           key=lambda case: (selected.get(_key(case), "zzz"), _key(case)))
    coverage_rows = []
    metric_rows = []
    for index, case in enumerate(ordered_cases, 1):
        key = _key(case)
        case_id = f"C{index:03d}"
        planned_group = selected.get(key)
        actual_result = actual.get(key)
        actual_group = actual_result[0] if actual_result is not None else None
        group_id = actual_group if actual_result is not None else planned_group
        group_mismatch = actual_result is not None and actual_group != planned_group
        execution_status = actual_result[1]["status"] if actual_result else (
            "EXCLUDED" if group_id is None else "NO_RESULT")
        reason = actual_result[1]["error"] if actual_result else (
            "Excluded by the node resource policy" if group_id is None else "No retained result")
        operation, dtype = _operation_dtype(case["cmd"])
        base = {
            "case_id": case_id, "group": group_id or "", "category": case["category"],
            "nodes": case["nodes"], "ranks": _ranks(case), "topology": _topology(case),
            "path_config": _path_config(case), "operation": operation, "dtype": dtype,
            "execution_status": execution_status, "data_status": "NO_RESULT",
            "valid_out_of_place": 0, "missing_out_of_place": 0, "reason": reason,
            "name": case["name"], "plan": case["plan"], "suite": case["suite"],
            "check": case["check"], "cmd": case["cmd"], "raw_log_sha256": "",
            "build_sha256": group_build.get(group_id, ""),
        }
        case_metrics = []
        if actual_result is not None:
            directory = results_root / actual_result[0] / Path(*key)
            log_path = directory / "raw.log"
            if group_mismatch or key in mismatched_cases:
                base["data_status"] = "UNVERIFIED"
                base["reason"] = ("Result group does not match the planned group" if group_mismatch
                                  else "Reconciliation found inconsistent Case evidence")
            elif execution_status == "PASSED" and (group_id in global_unverified
                                                   or key in unverified_cases):
                base["data_status"] = "UNVERIFIED"
                base["reason"] = "Reconciliation failed or is unavailable"
            elif execution_status == "NOT_RUN":
                base["data_status"] = "NOT_RUN"
            elif execution_status != "PASSED":
                base["data_status"] = "FAILED"
            elif log_path.is_file() and log_path.stat().st_size:
                base["raw_log_sha256"] = sha256(log_path.read_bytes()).hexdigest()
                try:
                    parsed = _parse_case(case, log_path, expected_sizes(case["cmd"]))
                except ValueError as error:
                    message = str(error)
                    base["data_status"] = "UNAVAILABLE" if message.startswith("Missing performance table") \
                        else "INVALID"
                    base["reason"] = message
                else:
                    base.update({key_: value for key_, value in parsed.items()
                                 if key_ not in ("rows", "reason", "types")})
                    base["reason"] = parsed["reason"] or reason
                    if not dtype and parsed["types"]:
                        dtype = ",".join(parsed["types"])
                        base["dtype"] = dtype
                    for row in parsed["rows"]:
                        case_metrics.append({
                            "case_id": case_id, "group": group_id or "",
                            "category": case["category"], "topology": _topology(case),
                            "path_config": _path_config(case), "operation": operation,
                            "dtype": row.get("type") or dtype,
                            "execution_status": execution_status, **row,
                        })
            else:
                base["data_status"] = "NO_RESULT"
                base["reason"] = "Missing or empty raw.log"
        coverage_rows.append(base)
        metric_rows.extend(case_metrics)

    topology = {}
    for row in coverage_rows:
        summary = topology.setdefault(row["topology"], {
            "passed": 0, "valid": 0, "partial": 0, "failed": 0, "unfinished": 0,
            "unverified": 0,
            "valid_out_of_place": 0, "missing_out_of_place": 0,
        })
        summary["passed"] += row["execution_status"] == "PASSED" and row["data_status"] != "UNVERIFIED"
        summary["valid"] += row["data_status"] == "VALID"
        summary["partial"] += row["data_status"] == "PARTIAL"
        summary["failed"] += (row["execution_status"] not in (
            "PASSED", "NO_RESULT", "EXCLUDED", "NOT_RUN", "UNVERIFIED")
            and row["data_status"] != "UNVERIFIED")
        summary["unfinished"] += row["execution_status"] in ("NO_RESULT", "EXCLUDED", "NOT_RUN")
        summary["unverified"] += row["data_status"] == "UNVERIFIED"
        summary["valid_out_of_place"] += row["valid_out_of_place"]
        summary["missing_out_of_place"] += row["missing_out_of_place"]
    total = {field: sum(summary[field] for summary in topology.values())
             for field in ("passed", "valid", "partial", "failed", "unfinished",
                           "unverified",
                           "valid_out_of_place", "missing_out_of_place")}
    selected_result_count = len(planned_actual)
    summary = {
        "topology": topology, "performance_total": total,
        "performance_cases": len(coverage_rows), "metric_rows": len(metric_rows),
        "valid_out_of_place_detail": sum(row["placement"] == "out-of-place"
                                         and row["metric_status"] == "VALID"
                                         for row in metric_rows),
        "execution": {
            "total_cases": coverage.get("total_cases"),
            "selected_cases": coverage.get("selected_cases"),
            "excluded_cases": coverage.get("excluded_cases"),
            "result_counts": dict(sorted(result_counts.items())),
            "selected_result_count": selected_result_count,
            "missing_selected_results": max(0, len(selected) - selected_result_count),
        },
    }
    return {"schema_version": SCHEMA_VERSION, "metadata": dict(metadata),
            "coverage": coverage_rows, "metrics": metric_rows, "summary": summary}


def _cell(workbook, worksheet, row, column, value, cell_format=None):
    if value is None:
        worksheet.write_blank(row, column, None, cell_format)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        worksheet.write_number(row, column, value, cell_format)
    else:
        worksheet.write_string(row, column, str(value), cell_format)


def _row(workbook, worksheet, row, values, cell_format=None):
    for column, value in enumerate(values):
        _cell(workbook, worksheet, row, column, value, cell_format)


def _sheet_name(value, used):
    name = re.sub(r"[\\/*?:\[\]]", "_", value)[:31] or "Sheet"
    base = name
    suffix = 2
    while name in used:
        name = f"{base[:28]}_{suffix}"
        suffix += 1
    used.add(name)
    return name


def write_workbook(report, output):
    try:
        import xlsxwriter
    except ImportError as error:
        raise RuntimeError("XlsxWriter is required for Excel report generation") from error
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.xlsx")
    workbook = xlsxwriter.Workbook(temporary)
    workbook.set_properties({"title": "PCCL M890P daily performance report",
                             "author": "PCCL CI", "comments": "Generated from retained raw logs"})
    title = workbook.add_format({"font_name": "Microsoft YaHei", "font_size": 18, "bold": True,
                                 "font_color": "#17365D"})
    subtitle = workbook.add_format({"font_name": "Microsoft YaHei", "font_color": "#666666"})
    section = workbook.add_format({"font_name": "Microsoft YaHei", "bold": True,
                                   "bg_color": "#D9EAF7", "border": 1, "border_color": "#B4C6E7"})
    header = workbook.add_format({"font_name": "Microsoft YaHei", "bold": True,
                                  "font_color": "white", "bg_color": "#17365D", "border": 1,
                                  "text_wrap": True, "valign": "vcenter"})
    text = workbook.add_format({"font_name": "Microsoft YaHei", "border": 1,
                                "border_color": "#D9E1F2"})
    number = workbook.add_format({"font_name": "Microsoft YaHei", "border": 1,
                                  "border_color": "#D9E1F2", "num_format": "0.00"})
    warning = workbook.add_format({"font_name": "Microsoft YaHei", "border": 1,
                                   "bg_color": "#FFF2CC", "font_color": "#9C6500"})
    failed = workbook.add_format({"font_name": "Microsoft YaHei", "border": 1,
                                  "bg_color": "#FCE4D6", "font_color": "#C00000"})
    try:
        used_names = set()
        summary = report["summary"]
        metadata = report["metadata"]
        ws = workbook.add_worksheet(_sheet_name("报告概览", used_names))
        ws.hide_gridlines(2)
        ws.set_landscape()
        ws.fit_to_pages(1, 1)
        ws.set_margins(0.25, 0.25, 0.4, 0.4)
        ws.set_column("A:A", 26)
        ws.set_column("B:H", 16)
        ws.set_column("I:J", 18)
        ws.merge_range("A1:J1", "PCCL M890P daily 性能报告", title)
        ws.merge_range("A2:J2",
                       f"运行 {metadata.get('run_id', '')}；执行状态与性能数据完整性分开统计；未提供性能基线。",
                       subtitle)
        info = [("运行标识", metadata.get("run_id", "")), ("数据集", metadata.get("dataset", "")),
                ("源码版本", metadata.get("source_revision", "")),
                ("报告时间", metadata.get("generated_at", ""))]
        for index, (label, value) in enumerate(info, 3):
            ws.write(index, 0, label, section)
            ws.merge_range(index, 1, index, 5, value, text)
        start = 9
        _row(workbook, ws, start, ["性能子集覆盖"], section)
        columns = ["范围", "源Case通过", "完整曲线", "含缺行曲线", "执行失败", "未完成", "未验证",
                   "有效OO点数", "缺行数"]
        _row(workbook, ws, start + 1, columns, header)
        for offset, name in enumerate(sorted(summary["topology"], key=_topology_sort), start=2):
            values = summary["topology"][name]
            _row(workbook, ws, start + offset,
                 [name, values["passed"], values["valid"], values["partial"], values["failed"],
                  values["unfinished"], values["unverified"], values["valid_out_of_place"],
                  values["missing_out_of_place"]], text)
        total = summary["performance_total"]
        last = start + 2 + len(summary["topology"])
        _row(workbook, ws, last,
             ["性能合计", total["passed"], total["valid"], total["partial"], total["failed"],
              total["unfinished"], total["unverified"], total["valid_out_of_place"],
              total["missing_out_of_place"]], section)
        ws.write(last + 2, 0, "原始明细独立计数", section)
        ws.write(last + 2, 6, summary["valid_out_of_place_detail"], text)
        ws.write(last + 2, 7,
                 "一致" if summary["valid_out_of_place_detail"] == total["valid_out_of_place"] else "不一致",
                 text)
        execution = summary["execution"]
        ws.write(last + 4, 0, "全量执行状态（不是性能子集通过率）", section)
        ws.write(last + 5, 0, "全量计划 Case", text)
        ws.write(last + 5, 1, execution["total_cases"], text)
        ws.write(last + 6, 0, "资源策略选择 / 排除", text)
        ws.write(last + 6, 1, execution["selected_cases"], text)
        ws.write(last + 6, 2, execution["excluded_cases"], text)
        ws.write(last + 7, 0, "已留存执行结果", text)
        ws.write(last + 7, 1, execution["selected_result_count"], text)
        ws.write(last + 8, 0, "缺失执行结果", text)
        ws.write(last + 8, 1, execution["missing_selected_results"], text)
        ws.merge_range(last + 10, 0, last + 11, 7,
                       "exit_code Case 的性能数据仅用于报告，不改变执行门禁；缺表、缺行、失败和未执行均显式保留。",
                       warning)

        metric_columns = ["CaseID", "类别", "操作", "精度", "布局", "size(B)", "time(us)",
                          "algbw(GB/s)", "busbw(GB/s)", "#wrong", "指标状态", "源Case状态"]
        by_topology = {}
        for row in report["coverage"]:
            by_topology.setdefault(row["topology"], []).append(row)
        metrics_by_case = {}
        for row in report["metrics"]:
            metrics_by_case.setdefault(row["case_id"], []).append(row)
        for topology in sorted(by_topology, key=_topology_sort):
            ws = workbook.add_worksheet(_sheet_name(topology, used_names))
            ws.set_landscape()
            ws.fit_to_pages(1, 0)
            ws.set_margins(0.25, 0.25, 0.4, 0.4)
            ws.merge_range(0, 0, 0, len(metric_columns) - 1, f"PCCL M890P {topology}性能", title)
            ws.merge_range(1, 0, 1, len(metric_columns) - 1,
                           "当前值为原始日志中的 event_time；未提供同配置基线。", subtitle)
            _row(workbook, ws, 3, metric_columns, header)
            row_index = 4
            for case in by_topology[topology]:
                rows = metrics_by_case.get(case["case_id"]) or [{
                    "case_id": case["case_id"], "category": case["category"],
                    "operation": case["operation"], "dtype": case["dtype"], "placement": "",
                    "size": None, "time": None, "algbw": None, "busbw": None, "wrong": None,
                    "metric_status": case["data_status"], "execution_status": case["execution_status"],
                }]
                for metric in rows:
                    _row(workbook, ws, row_index,
                         [metric["case_id"], metric["category"], metric["operation"], metric["dtype"],
                          metric["placement"], metric["size"], metric["time"], metric["algbw"],
                          metric["busbw"], metric["wrong"], metric["metric_status"],
                          metric["execution_status"]], text)
                    row_index += 1
            ws.freeze_panes(4, 0)
            ws.autofilter(3, 0, max(3, row_index - 1), len(metric_columns) - 1)
            ws.set_column(0, 0, 10)
            ws.set_column(1, 2, 22)
            ws.set_column(3, 5, 14)
            ws.set_column(6, 8, 15, number)
            ws.set_column(9, 11, 18)

        detail_columns = ["CaseID", "组", "类别", "拓扑", "路径配置", "操作", "精度", "布局",
                          "size(B)", "count", "时间口径", "time(us)", "algbw(GB/s)", "busbw(GB/s)",
                          "#wrong", "指标状态", "日志行号", "源Case状态"]
        ws = workbook.add_worksheet(_sheet_name("数据明细", used_names))
        _row(workbook, ws, 0, detail_columns, header)
        for index, metric in enumerate(report["metrics"], 1):
            _row(workbook, ws, index,
                 [metric["case_id"], metric["group"], metric["category"], metric["topology"],
                  metric["path_config"], metric["operation"], metric["dtype"], metric["placement"],
                  metric["size"], metric["count"], metric["time_kind"], metric["time"],
                  metric["algbw"], metric["busbw"], metric["wrong"], metric["metric_status"],
                  metric["source_line"], metric["execution_status"]], text)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, max(0, len(report["metrics"])), len(detail_columns) - 1)
        ws.set_column(0, 1, 10)
        ws.set_column(2, 7, 22)
        ws.set_column(8, 17, 16)

        coverage_columns = ["CaseID", "类别", "组", "节点数", "测试Ranks", "路径配置", "操作", "精度",
                            "执行状态", "数据状态", "有效OO点数", "缺行数", "原因", "源Case名称",
                            "Plan", "Suite", "CSV检查类型", "命令模板", "原始日志SHA256", "构建SHA256"]
        ws = workbook.add_worksheet(_sheet_name("用例与覆盖", used_names))
        _row(workbook, ws, 0, coverage_columns, header)
        for index, case in enumerate(report["coverage"], 1):
            _row(workbook, ws, index,
                 [case["case_id"], case["category"], case["group"], case["nodes"], case["ranks"],
                  case["path_config"], case["operation"], case["dtype"], case["execution_status"],
                  case["data_status"], case["valid_out_of_place"], case["missing_out_of_place"],
                  case["reason"], case["name"], case["plan"], case["suite"], case["check"], case["cmd"],
                  case["raw_log_sha256"], case["build_sha256"]], text)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, len(report["coverage"]), len(coverage_columns) - 1)
        ws.set_column(0, 7, 18)
        ws.set_column(8, 12, 16)
        ws.set_column(13, 16, 42)
        ws.set_column(17, 17, 90)
        ws.set_column(18, 19, 66)

        ws = workbook.add_worksheet(_sheet_name("Metadata", used_names))
        _row(workbook, ws, 0, ["字段", "值", "说明"], header)
        metadata_rows = [(field, value, "运行输入") for field, value in metadata.items()] + [
            ("performance_cases", summary["performance_cases"], "包括未执行和资源排除的性能相关 Case"),
            ("metric_rows", summary["metric_rows"], "双布局原始行与显式缺行"),
            ("parser", "run_cases.parse_performance", "与执行器共用同一性能表解析实现"),
            ("gate_policy", "unchanged", "exit_code 不因为报告解析而改变门禁"),
            ("result_counts", json.dumps(summary["execution"]["result_counts"], sort_keys=True),
             "按全部执行组 results.csv 统计"),
        ]
        for index, (field, value, description) in enumerate(metadata_rows, 1):
            _row(workbook, ws, index, [field, value, description], text)
        ws.freeze_panes(1, 0)
        ws.set_column(0, 0, 28)
        ws.set_column(1, 1, 100)
        ws.set_column(2, 2, 66)
        workbook.close()
        temporary.replace(output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return output


def _topology_sort(name):
    match = re.fullmatch(r"(?:单机([0-9]+)卡|([0-9]+)机([0-9]+)卡)", name)
    if not match:
        return 99, 99
    if match.group(1):
        return 1, int(match.group(1))
    return int(match.group(2)), int(match.group(3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    selection = commands.add_parser("selection")
    selection.add_argument("--input", type=Path, required=True)
    selection.add_argument("--profile", default="standard")
    selection.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("report")
    report.add_argument("--selection", type=Path, required=True)
    report.add_argument("--groups", type=Path, required=True)
    report.add_argument("--coverage", type=Path, required=True)
    report.add_argument("--results-root", type=Path, required=True)
    report.add_argument("--reconciliation-root", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--run-id", required=True)
    report.add_argument("--run-url", default="")
    report.add_argument("--dataset", required=True)
    report.add_argument("--source-revision", required=True)
    report.add_argument("--runner-revision", required=True)
    args = parser.parse_args()
    if args.command == "selection":
        manifest = write_selection(args.input, args.profile, args.output)
        print(json.dumps({"performance_cases": len(manifest["cases"])}, sort_keys=True))
        return 0
    metadata = {
        "run_id": args.run_id, "run_url": args.run_url, "dataset": args.dataset,
        "source_revision": args.source_revision, "runner_revision": args.runner_revision,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
    }
    report_value = collect_report(args.selection, args.groups, args.coverage, args.results_root,
                                  metadata, args.reconciliation_root)
    write_workbook(report_value, args.output)
    print(json.dumps(report_value["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
