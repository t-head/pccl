#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the minimal CSV export with the Python standard library only.

The input is trusted test configuration. Commands are passed to /bin/sh -c;
this program deliberately does not reinterpret benchmark or Runner options.
Raw logs are private diagnostic output and are never generically redacted.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid

import artifact_cache


FORMAT_VERSION = 1
SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")


def _with_ib_environment(command):
    """Forward the common IB settings to local and remote MPI ranks."""
    exports = " ".join("-x " + name + "=1" for name in
                       ("NCCL_IB_ADAPTIVE_ROUTING", "NCCL_IB_OOO_RQ")
                       if name + "=1" not in command)
    if not exports:
        return command
    if re.search(r"(?:^|\s)mpirun(?=\s)", command):
        return re.sub(r"(^|\s)mpirun(?=\s)",
                      lambda match: match.group(1) + "mpirun " + exports, command, count=1)
    match = re.search(r"(--mpi-options?(?:=|\s+)(['\"]))", command)
    if match:
        return command[:match.end()] + exports + " " + command[match.end():]
    return command


def _allocated_devices(environment):
    """Return the finite physical PPU list injected by the device plugin."""
    value = environment.get("NVIDIA_VISIBLE_DEVICES", "").strip()
    if not value or value.lower() in ("all", "void", "none"):
        raise ValueError("810E execution requires a finite NVIDIA_VISIBLE_DEVICES allocation")
    devices = []
    for item in value.split(","):
        if not re.fullmatch(r"[0-9]+", item.strip()):
            raise ValueError("NVIDIA_VISIBLE_DEVICES must contain comma-separated PPU indices")
        device = int(item)
        if device in devices:
            raise ValueError("NVIDIA_VISIBLE_DEVICES contains duplicate PPU indices")
        devices.append(device)
    return devices


def _map_case_devices(command, allocated):
    """Map Case-local indices to the Pod's physical allocation and return the effective list."""
    explicit = []

    def remap(value):
        mapped = []
        for item in value.split(","):
            index = int(item)
            if index >= len(allocated):
                raise ValueError(f"Case device index {index} is outside the allocated PPU list")
            mapped.append(allocated[index])
        return ",".join(str(device) for device in mapped)

    def replace(match):
        mapped = remap(match.group(2))
        explicit.append(mapped)
        return match.group(1) + mapped

    command = re.sub(r"(HGGC_VISIBLE_DEVICES=)([0-9]+(?:,[0-9]+)*)", replace, command)
    command = re.sub(r"(--devices(?:=|\s+))([0-9]+(?:,[0-9]+)*)", replace, command)
    if len(set(explicit)) > 1:
        raise ValueError("Case contains conflicting HGGC_VISIBLE_DEVICES and --devices selections")
    allocated_value = ",".join(str(device) for device in allocated)
    return command, explicit[0] if explicit else allocated_value, allocated_value


def _with_device_environment(command, devices):
    """Forward the mapped PPU selection to local and remote MPI ranks."""
    if re.search(r"-x HGGC_VISIBLE_DEVICES=", command):
        return command
    export = "-x HGGC_VISIBLE_DEVICES=" + devices
    if re.search(r"(?:^|\s)mpirun(?=\s)", command):
        return re.sub(r"(^|\s)mpirun(?=\s)",
                      lambda match: match.group(1) + "mpirun " + export, command, count=1)
    match = re.search(r"(--mpi-options?(?:=|\s+)(['\"]))", command)
    if match:
        return command[:match.end()] + export + " " + command[match.end():]
    return command


VARIABLE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
PLACEHOLDER = re.compile(r"(?<!\$)\{([A-Za-z_][A-Za-z0-9_]*)\}")
SAFE_BINDING = re.compile(r"[A-Za-z0-9_./:+%@^,=-]+\Z")
SUMMARY = re.compile(r"^#\s*Out of bounds values\s*:\s*0\s+OK\s*$", re.M)
SUITE_FIELDS = ["plan", "suite", "nodes", "cards_per_node", "timeout", "env"]
CASE_FIELDS = ["plan", "suite", "name", "check", "cmd"]
CHECKS = {"exit_code", "validation_summary", "performance"}
RESULT_FIELDS = ["plan", "suite", "name", "check", "status", "return_code",
                 "duration_seconds", "error", "raw_log"]
METRIC_FIELDS = ["size", "count", "type", "redop", "root", "placement", "time_kind",
                 "time", "algbw", "busbw", "wrong", "source_line"]


def _read_csv(path, fields):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise ValueError(f"Unexpected CSV header in {path.name}: {reader.fieldnames}")
        rows = list(reader)
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed CSV row in {path.name}")
        if any("\0" in value or "\n" in value or "\r" in value for value in row.values()):
            raise ValueError(f"NUL or newline is not allowed in {path.name}")
    return rows


def _read_metadata(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(data) != {"format_version", "source_revision", "runner_revision", "xtest_version"}:
        raise ValueError("Unexpected export metadata fields")
    if data["format_version"] != FORMAT_VERSION:
        raise ValueError("Unsupported CSV export format")
    if not isinstance(data["source_revision"], str) or not HEX40.fullmatch(data["source_revision"]):
        raise ValueError("Invalid source revision")
    if not isinstance(data["runner_revision"], str) or not HEX40.fullmatch(data["runner_revision"]):
        raise ValueError("Invalid runner revision")
    if not isinstance(data["xtest_version"], str) or not data["xtest_version"]:
        raise ValueError("Invalid exporting xtest version")
    return data


def load_inputs(directory, profile, nodes=None, suite=None):
    """Read and fully validate one profile before allocating or running anything."""
    directory = Path(directory)
    if profile not in ("standard", "notsm"):
        raise ValueError("Profile must be standard or notsm")
    metadata = _read_metadata(directory / "export.json")
    suite_rows = _read_csv(directory / "suites.csv", SUITE_FIELDS)
    case_rows = _read_csv(directory / f"{profile}.csv", CASE_FIELDS)
    suites = {}
    for row in suite_rows:
        key = (row["plan"], row["suite"])
        if any(not SAFE_NAME.fullmatch(value) for value in key):
            raise ValueError("Plan and Suite names must be portable")
        if key in suites:
            raise ValueError("Duplicate Suite key")
        try:
            suite_nodes = int(row["nodes"])
            cards = int(row["cards_per_node"])
            timeout = int(row["timeout"])
        except ValueError as error:
            raise ValueError("Suite resources and timeout must be integers") from error
        if (str(suite_nodes) != row["nodes"] or str(cards) != row["cards_per_node"]
                or str(timeout) != row["timeout"] or suite_nodes <= 0 or cards <= 0 or timeout <= 0):
            raise ValueError("Suite resources and timeout must be positive integers")
        env_name = row["env"]
        if (not SAFE_NAME.fullmatch(env_name) or not env_name.endswith(".env")
                or not (directory / "envs" / env_name).is_file()):
            raise ValueError("Suite references a missing or unsafe environment file")
        suites[key] = {**row, "nodes": suite_nodes, "cards_per_node": cards, "timeout": timeout}
    cases = []
    keys = set()
    for row in case_rows:
        key = (row["plan"], row["suite"], row["name"])
        if not SAFE_NAME.fullmatch(row["name"]):
            raise ValueError("Case name must be portable")
        if key in keys:
            raise ValueError("Duplicate Case key")
        keys.add(key)
        suite_key = key[:2]
        if suite_key not in suites:
            raise ValueError("Case references a missing Suite")
        if row["check"] not in CHECKS:
            raise ValueError("Unknown Case check")
        if not row["cmd"].strip():
            raise ValueError("Case command must not be empty")
        cases.append({**row, "suite_spec": suites[suite_key]})
    if nodes is not None:
        if type(nodes) is not int or nodes <= 0:
            raise ValueError("Node filter must be a positive integer")
        cases = [case for case in cases if case["suite_spec"]["nodes"] == nodes]
    if suite is not None:
        if not SAFE_NAME.fullmatch(suite):
            raise ValueError("Suite filter must be a portable Suite name")
        cases = [case for case in cases if case["suite"] == suite]
    if not cases:
        raise ValueError("Profile contains no Cases for the requested node count and Suite")
    return {"metadata": metadata, "suites": suites, "cases": cases, "profile": profile,
            "directory": directory}


def _environment_file(path):
    rows = []
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            raise ValueError("Environment files are data, not shell scripts")
        key, separator, value = line.partition("=")
        if not separator or not ENV_NAME.fullmatch(key) or "\0" in value:
            raise ValueError(f"Invalid environment entry at line {number}")
        rows.append((key, value))
    if len({key for key, _ in rows}) != len(rows):
        raise ValueError("Duplicate environment entries")
    return rows


def _substitute(value, bindings):
    def replace(match):
        key = match.group(1)
        if key not in bindings:
            raise ValueError(f"Unsupported placeholder: {match.group(0)}")
        return bindings[key]
    return PLACEHOLDER.sub(replace, value)


def _command_binding(value):
    if not isinstance(value, str) or not value or not SAFE_BINDING.fullmatch(value):
        raise ValueError("Command path/environment bindings contain shell-unsafe characters")
    return value


def _expand_environment(rows, base, bindings):
    environment = dict(base)

    def replace(match):
        key = match.group(1) or match.group(2)
        if key not in environment:
            raise ValueError(f"Environment references an unknown variable: {key}")
        return environment[key]

    for key, value in rows:
        value = _substitute(value, bindings)
        if "`" in value or "$(" in value:
            raise ValueError("Shell evaluation is not allowed in environment files")
        value = VARIABLE.sub(replace, value)
        if "$" in value:
            raise ValueError("Environment contains an unsupported variable reference")
        environment[key] = value
    return environment


def _runner_root(test_root):
    nested = test_root / "pccl-tests"
    return nested if (nested / "run_test.py").is_file() else test_root


def _stage_interpreter(output):
    directory = output / "python-env" / "bin"
    directory.mkdir(parents=True)
    executable = Path(sys.executable).absolute()
    launcher = '#!/bin/sh\nexec ' + shlex.quote(str(executable)) + ' "$@"\n'
    for name in sorted({"python", "python3", "python3.11", executable.name}):
        path = directory / name
        path.write_text(launcher, encoding="utf-8")
        path.chmod(0o755)
    return directory


def _stage_work(work, runner_root, binary_dir, hostfile):
    work.mkdir(parents=True, exist_ok=False)
    for name in ("run_test.py", "run_test_config.py", "unit_test.py"):
        source = runner_root / name
        if source.is_file():
            (work / name).symlink_to(source)
    (work / "build").symlink_to(binary_dir, target_is_directory=True)
    if hostfile is not None:
        (work / "hosts.cfg").symlink_to(hostfile)


def _group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _stop_group(process, grace=0.5):
    """Stop the process group; detached Case processes are cleaned by cgroup recovery."""
    pgid = process.pid
    existed = _group_exists(pgid)
    group_error = None

    def send(sig):
        nonlocal group_error
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        except PermissionError as error:
            # macOS can reject a signal to a group containing an unreaped
            # zombie. Reap the direct child below before the final check.
            group_error = error

    if existed:
        send(signal.SIGTERM)
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
        if _group_exists(pgid):
            send(signal.SIGKILL)
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    process.wait(timeout=2)
    # Reaping the shell does not necessarily reap its background children.
    if _group_exists(pgid):
        send(signal.SIGKILL)
    if group_error is not None and _group_exists(pgid):
        raise OSError(f"Unable to clean process group {pgid}: {group_error}")
    return existed


def _number(value):
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("Performance table contains a non-numeric measurement") from error
    if not result.is_finite():
        raise ValueError("Performance table contains a non-finite measurement")
    return str(result)


def parse_performance(text, allow_unsupported=False, skip_malformed=False):
    """Parse dual-layout tables, with opt-in report tolerance only."""
    rows = []
    header = None
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        tokens = line.split()
        if line.startswith("#") and "size" in tokens and "count" in tokens:
            if (len(tokens) != 14 or tokens[1:5] != ["size", "count", "type", "redop"]
                    or tokens[5] != "root" or tokens[6] != tokens[10]
                    or tokens[7:10] != ["algbw", "busbw", "#wrong"]
                    or tokens[11:14] != ["algbw", "busbw", "#wrong"]):
                raise ValueError("Unsupported performance table header")
            header = tokens
            continue
        if not tokens or not re.fullmatch(r"[0-9]+", tokens[0]):
            continue
        if header is None:
            continue
        if len(tokens) != 13 or not re.fullmatch(r"[0-9]+", tokens[1]):
            if skip_malformed:
                continue
            raise ValueError("Truncated or malformed performance table row")
        if skip_malformed and not re.fullmatch(r"-?[0-9]+", tokens[4]):
            continue
        parsed_row = []
        malformed = False
        for offset, current in ((5, "out-of-place"), (9, "in-place")):
            wrong = tokens[offset + 3]
            if not re.fullmatch(r"[0-9]+", wrong) and not (allow_unsupported and wrong == "N/A"):
                if skip_malformed:
                    malformed = True
                    break
                raise ValueError("Performance #wrong value must be numeric")
            if wrong != "N/A" and int(wrong) != 0:
                raise ValueError("Performance data validation reported wrong values")
            try:
                values = (_number(tokens[offset]), _number(tokens[offset + 1]),
                          _number(tokens[offset + 2]))
            except ValueError:
                if skip_malformed:
                    malformed = True
                    break
                raise
            parsed_row.append({"size": tokens[0], "count": tokens[1], "type": tokens[2],
                               "redop": tokens[3], "root": tokens[4], "placement": current,
                               "time_kind": header[6], "time": values[0],
                               "algbw": values[1], "busbw": values[2],
                               "wrong": wrong, "source_line": line_number})
        if malformed:
            continue
        rows.extend(parsed_row)
    if header is None or not rows:
        raise ValueError("Missing performance table or measurements")
    return rows


def _check_result(case, return_code, log_path):
    if return_code != 0:
        raise ValueError(f"Command exited with status {return_code}")
    text = Path(log_path).read_bytes().decode("utf-8", errors="replace")
    if case["check"] in ("validation_summary", "performance") and not SUMMARY.search(text):
        raise ValueError("Missing successful validation summary")
    if case["check"] == "performance":
        return parse_performance(text)
    return []


def _write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _result_row(case, status, started, **values):
    row = {"plan": case["plan"], "suite": case["suite"], "name": case["name"],
           "check": case["check"], "status": status,
           "return_code": values.get("return_code"),
           "duration_seconds": round(time.monotonic() - started, 3),
           "error": values.get("error", ""),
           "raw_log": values.get("raw_log", "raw.log")}
    return row


def run_case(case, runtime, interrupted):
    suite = case["suite_spec"]
    directory = runtime["output"] / case["plan"] / case["suite"] / case["name"]
    work = directory / "work"
    directory.mkdir(parents=True, exist_ok=False)
    log_path = directory / "raw.log"
    started = time.monotonic()
    process = None
    return_code = None
    try:
        bindings = {"test_root": str(runtime["test_root"]), "binary_dir": str(runtime["binary_dir"]),
                    "result_dir": str(directory),
                    "python_env": str(runtime["output"] / "python-env")}
        environment = _expand_environment(
            _environment_file(runtime["input"] / "envs" / suite["env"]), runtime["base_environment"], bindings)
        environment["NCCL_IB_ADAPTIVE_ROUTING"] = "1"
        environment["NCCL_IB_OOO_RQ"] = "1"
        # Keep the build ahead of inherited installations while retaining the
        # exported SDK path order after it.
        if suite["nodes"] > 1:
            environment["PCCL_CASE_ID"] = runtime["recovery_id"]
        environment["PATH"] = (str(runtime["binary_dir"]) + os.pathsep
                               + str(runtime["python_bin"]) + os.pathsep + environment.get("PATH", ""))
        if suite["nodes"] > 1:
            environment["PATH"] = str(runtime["test_root"] / "pccl" / "tests" / "xtest" / "cluster-bin") \
                + os.pathsep + environment["PATH"]
        environment["LD_LIBRARY_PATH"] = str(runtime["test_root"] / "lib") + os.pathsep + environment.get(
            "LD_LIBRARY_PATH", "")
        environment["PYTHONUNBUFFERED"] = "1"
        bindings.update(path=_command_binding(environment.get("PATH", "")),
                        ld_library_path=_command_binding(environment.get("LD_LIBRARY_PATH", "")))
        if suite["nodes"] > 1:
            bindings["hostfile"] = _command_binding(str(runtime["hostfile"]))
        if "{mpi_interface}" in case["cmd"]:
            interface = runtime["mpi_interface"]
            if not isinstance(interface, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", interface):
                raise ValueError("This Case requires --mpi-interface with a valid interface name")
            bindings["mpi_interface"] = interface
        command_bindings = {key: _command_binding(value) for key, value in bindings.items()}
        command = _substitute(case["cmd"], command_bindings)
        if runtime["map_nvidia_devices"]:
            allocated = _allocated_devices(environment)
            command, effective, allocated_value = _map_case_devices(command, allocated)
            environment["HGGC_VISIBLE_DEVICES"] = allocated_value
            command = "HGGC_VISIBLE_DEVICES=" + allocated_value + " " + command
            command = _with_device_environment(command, effective)
        command = _with_ib_environment(command)
        _stage_work(work, runtime["runner_root"], runtime["binary_dir"], runtime["hostfile"])
        (directory / "command.txt").write_text(command + "\n", encoding="utf-8")
        popen_command = ["/bin/sh", "-c", command]
        if suite["nodes"] > 1:
            launcher = runtime["test_root"] / "pccl" / "tests" / "xtest" / "remote_recovery.py"
            popen_command = [sys.executable, str(launcher), "--case-id", runtime["recovery_id"],
                             f"--launch={command}"]
        with log_path.open("wb") as output:
            process = subprocess.Popen(popen_command, cwd=work, env=environment,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + suite["timeout"]
            while True:
                if interrupted[0] is not None:
                    raise RuntimeError(f"Cancelled by signal {interrupted[0]}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, suite["timeout"])
                try:
                    return_code = process.wait(timeout=min(remaining, 0.1))
                    break
                except subprocess.TimeoutExpired:
                    pass
        metrics = _check_result(case, return_code, log_path)
        if metrics:
            with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(metrics)
        row = _result_row(case, "PASSED", started, return_code=return_code)
    except subprocess.TimeoutExpired:
        row = _result_row(case, "TIMEOUT", started, return_code=return_code,
                          error=f"Case exceeded {suite['timeout']} seconds")
    except RuntimeError as error:
        row = _result_row(case, "CANCELLED" if interrupted[0] else "FAILED", started,
                          return_code=return_code, error=str(error))
    except (OSError, ValueError) as error:
        row = _result_row(case, "FAILED", started, return_code=return_code, error=str(error))
    finally:
        if process is not None:
            try:
                residual = _stop_group(process)
                if residual and row["status"] == "PASSED":
                    row = _result_row(case, "FAILED", started, return_code=return_code,
                                      error="Command left residual local processes")
            except OSError as error:
                if row["status"] == "PASSED":
                    row = _result_row(case, "FAILED", started, return_code=return_code,
                                      error=f"Process cleanup failed: {error}")
        row["raw_log"] = "raw.log" if log_path.is_file() else ""
        _write_json(directory / "result.json", row)
    return row


def _not_run(case, reason):
    return {"plan": case["plan"], "suite": case["suite"], "name": case["name"],
            "check": case["check"], "status": "NOT_RUN", "return_code": None,
            "duration_seconds": 0, "error": reason, "raw_log": ""}


def _remote_recovery(host, script, case_id):
    command = ["timeout", "-k", "2s", "10s", "cluster-exec", host,
               "python3", str(script), "--case-id", case_id]
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"safe": False, "errors": [str(error)]}
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    try:
        report = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        report = {"safe": False,
                  "errors": [f"Invalid remote recovery output: {result.stdout[-500:]}{result.stderr[-500:]}"]}
    if not isinstance(report, dict) or type(report.get("safe")) is not bool:
        report = {"safe": False,
                  "errors": [f"Invalid remote recovery response: {result.stdout[-500:]}{result.stderr[-500:]}"]}
    if result.returncode != 0 and "errors" not in report:
        report["errors"] = [f"Remote recovery exited with {result.returncode}: {result.stderr[-500:]}"]
    if result.returncode != 0:
        report["safe"] = False
    return report


def _recover_multinode(case, runtime, row):
    """Confirm remote Case processes are gone; never perform another MPI health probe."""
    hosts = []
    for line in Path(runtime["hostfile"]).read_text().splitlines():
        fields = line.split()
        if fields and not fields[0].startswith("#"):
            hosts.append(fields[0])
    script = runtime["test_root"] / "pccl" / "tests" / "xtest" / "remote_recovery.py"
    report = {"safe": False, "case_id": runtime["recovery_id"], "workers": {}}
    if not hosts or not script.is_file():
        report["error"] = "Missing allocated workers or remote recovery helper"
    else:
        with ThreadPoolExecutor(max_workers=len(hosts)) as executor:
            results = executor.map(
                lambda host: _remote_recovery(host, script, runtime["recovery_id"]), hosts)
            report["workers"] = dict(zip(hosts, results))
        report["safe"] = all(worker.get("safe") is True for worker in report["workers"].values())
    directory = runtime["output"] / case["plan"] / case["suite"] / case["name"]
    _write_json(directory / "recovery.json", report)
    for host, worker in report["workers"].items():
        residual = len(worker.get("remaining", worker.get("before", [])))
        print(f"Remote recovery {host}: safe={worker.get('safe')} candidates={residual}", flush=True)
    if not report["safe"]:
        print("Remote recovery report: " + json.dumps(report, sort_keys=True), flush=True)
    return report["safe"]


def _capture_dmesg(case, runtime):
    """Best-effort kernel diagnostics beside a failed Case's raw log."""
    suite = case["suite_spec"]
    directory = runtime["output"] / case["plan"] / case["suite"] / case["name"]
    if suite["nodes"] == 1:
        captures = [("dmesg.log", ["dmesg"])]
    else:
        hosts = [line.split()[0] for line in Path(runtime["hostfile"]).read_text().splitlines()
                 if line.split() and not line.split()[0].startswith("#")]
        captures = [(f"dmesg-worker-{index}.log", ["cluster-ssh", host, "dmesg"])
                    for index, host in enumerate(hosts)]
    for filename, command in captures:
        try:
            result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=15, check=False)
            content = result.stdout
            if result.returncode:
                content += f"\n[dmesg command exited {result.returncode}]\n".encode()
        except (OSError, subprocess.TimeoutExpired) as error:
            content = f"Unable to capture dmesg: {error}\n".encode()
        (directory / filename).write_bytes(content)


def _write_summary(output, rows):
    by_key = {(row["plan"], row["suite"], row["name"]): row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("Duplicate execution results")
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_json(output / "summary.json", {"total": len(rows), "counts": counts})


def run_all(inputs, test_root, binary_dir, output, hostfile=None, base_environment=None, mpi_interface=None,
            *, build_identity, map_nvidia_devices=False):
    test_root = Path(test_root).resolve()
    binary_dir = Path(binary_dir).resolve() if binary_dir else _runner_root(test_root) / "build"
    output = Path(output).resolve()
    hostfile = Path(hostfile).resolve() if hostfile else None
    if not test_root.is_dir() or not binary_dir.is_dir():
        raise ValueError("test_root and binary_dir must be existing directories")
    if output.exists():
        raise ValueError("Output directory already exists")
    multiple = any(case["suite_spec"]["nodes"] > 1 for case in inputs["cases"])
    if multiple and os.environ.get("NODE_RANK", "0") != "0":
        raise ValueError("Multi-node commands may run only on the master/controller")
    if any(case["suite_spec"]["nodes"] > 1 for case in inputs["cases"]):
        if hostfile is None or not hostfile.is_file():
            raise ValueError("Multi-node Cases require an allocated hostfile")
        for relative in ("cluster-bin/cluster-ssh", "cluster-bin/cluster-exec",
                         "cluster-bin/cluster_case_ssh.sh", "remote_recovery.py"):
            if not (test_root / "pccl" / "tests" / "xtest" / relative).is_file():
                raise ValueError(f"Multi-node recovery helper is missing: {relative}")
        if any("/run/cluster/bin/" in case["cmd"] for case in inputs["cases"]):
            raise ValueError("Absolute cluster-ssh paths bypass per-Case cgroup ownership")
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "build.json", build_identity)
    runtime = {"test_root": test_root, "binary_dir": binary_dir, "runner_root": _runner_root(test_root),
               "output": output, "hostfile": hostfile, "input": inputs["directory"],
               "python_bin": _stage_interpreter(output), "mpi_interface": mpi_interface,
               "base_environment": dict(os.environ if base_environment is None else base_environment),
               "map_nvidia_devices": map_nvidia_devices,
               "recovery_id": uuid.uuid4().hex}
    interrupted = [None]

    def cancel(signum, frame):
        interrupted[0] = signum

    previous = {sig: signal.signal(sig, cancel) for sig in (signal.SIGINT, signal.SIGTERM)}
    rows = []
    ordered = sorted(enumerate(inputs["cases"]),
                     key=lambda item: (item[1]["plan"], item[1]["suite"], item[0]))
    try:
        for _, case in ordered:
            if interrupted[0] is not None:
                rows.append(_not_run(case, "Cancelled before Case start"))
                continue
            row = run_case(case, runtime, interrupted)
            rows.append(row)
            print(f'[{len(rows)}/{len(inputs["cases"])}] '
                  f'{case["plan"]}/{case["suite"]}/{case["name"]}: {row["status"]}', flush=True)
            if row["status"] != "PASSED":
                recovered = True
                if case["suite_spec"]["nodes"] > 1:
                    recovered = _recover_multinode(case, runtime, row)
                _capture_dmesg(case, runtime)
                if case["suite_spec"]["nodes"] > 1 and (not recovered or row["status"] == "CANCELLED"):
                    reason = "Stopped after multi-node recovery failure; remote process cleanup is not confirmed"
                    rows.extend(_not_run(remaining, reason) for _, remaining in ordered[len(rows):])
                    break
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    expected = {(case["plan"], case["suite"], case["name"]) for case in inputs["cases"]}
    actual = {(row["plan"], row["suite"], row["name"]) for row in rows}
    if len(rows) != len(inputs["cases"]) or actual != expected:
        raise ValueError("Execution count does not match the selected Cases")
    _write_summary(output, rows)
    return rows


def _verified_build_identity(inputs, record_path, pccl_revision, expected_sha):
    """Bind execution results to the archive identity already validated by the workflow."""
    record = artifact_cache.read_record(record_path)
    expected = {"schema_version": 1, "pccl_revision": pccl_revision,
                "pccl_tests_revision": inputs["metadata"]["runner_revision"],
                "build_profile": inputs["profile"], "artifact_sha256": expected_sha}
    if record != expected:
        raise ValueError("Build record differs from the selected profile, source or verified archive")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--profile", choices=("standard", "notsm"), required=True)
    parser.add_argument("--nodes", type=int, help="Run only Cases whose Suite requests this node count")
    parser.add_argument("--suite", help="Execute only the exact Suite name")
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--binary-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hostfile", type=Path)
    parser.add_argument("--mpi-interface", help="Actual TCP interface for an exported {mpi_interface} binding")
    parser.add_argument("--build-record", type=Path, help="Verified output_PROFILE.build.json from the restored archive")
    parser.add_argument("--pccl-revision", help="PCCL revision verified in the build record")
    parser.add_argument("--expected-sha256", help="Archive SHA-256 verified in the build record")
    parser.add_argument("--map-nvidia-devices", action="store_true",
                        help="Map Case device selections to the Pod's NVIDIA_VISIBLE_DEVICES allocation")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        inputs = load_inputs(args.input, args.profile, args.nodes, suite=args.suite)
        if args.validate_only:
            if args.output is not None or args.test_root is not None:
                raise ValueError("--validate-only does not accept execution paths")
            print(json.dumps({"profile": args.profile, "cases": len(inputs["cases"]),
                              "suites": len({(c["plan"], c["suite"]) for c in inputs["cases"]})}, sort_keys=True))
            return 0
        if args.test_root is None or args.output is None:
            raise ValueError("Execution requires --test-root and --output")
        if args.build_record is None or args.pccl_revision is None or args.expected_sha256 is None:
            raise ValueError("Execution requires --build-record, --pccl-revision and --expected-sha256")
        build_identity = _verified_build_identity(
            inputs, args.build_record, args.pccl_revision, args.expected_sha256)
        rows = run_all(inputs, args.test_root, args.binary_dir, args.output, args.hostfile,
                       mpi_interface=args.mpi_interface, build_identity=build_identity,
                       map_nvidia_devices=args.map_nvidia_devices)
        return 0 if all(row["status"] == "PASSED" for row in rows) else 1
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
