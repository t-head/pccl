#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the minimal CSV export with the Python standard library only.

The input is trusted test configuration. Commands are passed to /bin/sh -c;
this program deliberately does not reinterpret benchmark or Runner options.
Raw logs are private diagnostic output and are never generically redacted.
"""

import argparse
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


FORMAT_VERSION = 1
SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
VARIABLE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
PLACEHOLDER = re.compile(r"(?<!\$)\{([A-Za-z_][A-Za-z0-9_]*)\}")
SAFE_BINDING = re.compile(r"[A-Za-z0-9_./:+%@^,=-]+\Z")
SUMMARY = re.compile(r"^#\s*Out of bounds values\s*:\s*0\s+OK\s*$", re.M)
SUITE_FIELDS = ["plan", "suite", "nodes", "cards_per_node", "timeout", "env"]
CASE_FIELDS = ["plan", "suite", "name", "check", "cmd"]
CHECKS = {"exit_code", "validation_summary", "performance"}
RESULT_FIELDS = ["plan", "suite", "name", "check", "status", "return_code",
                 "duration_seconds", "error", "raw_log"]


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


def load_inputs(directory, profile, nodes=None):
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
    if not cases:
        raise ValueError("Profile contains no Cases for the requested node count")
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
    directory = output / ".executor-bin"
    directory.mkdir()
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
    """Stop the local process group; remote MPI rank cleanup is infrastructure-owned."""
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


def parse_performance(text):
    """Parse observed dual-layout benchmark tables without a performance gate."""
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
            raise ValueError("Truncated or malformed performance table row")
        for offset, current in ((5, "out-of-place"), (9, "in-place")):
            wrong = tokens[offset + 3]
            if not re.fullmatch(r"[0-9]+", wrong):
                raise ValueError("Performance #wrong value must be numeric")
            if int(wrong) != 0:
                raise ValueError("Performance data validation reported wrong values")
            rows.append({"size": tokens[0], "count": tokens[1], "type": tokens[2],
                         "redop": tokens[3], "root": tokens[4], "placement": current,
                         "time_kind": header[6], "time": _number(tokens[offset]),
                         "algbw": _number(tokens[offset + 1]), "busbw": _number(tokens[offset + 2]),
                         "wrong": wrong, "source_line": line_number})
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
                    "result_dir": str(directory), "python_env": str(Path(sys.prefix).absolute())}
        environment = _expand_environment(
            _environment_file(runtime["input"] / "envs" / suite["env"]), runtime["base_environment"], bindings)
        # Keep the build ahead of inherited installations while retaining the
        # exported SDK path order after it.
        environment["PATH"] = (str(runtime["binary_dir"]) + os.pathsep
                               + str(runtime["python_bin"]) + os.pathsep + environment.get("PATH", ""))
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
        _stage_work(work, runtime["runner_root"], runtime["binary_dir"], runtime["hostfile"])
        (directory / "command.txt").write_text(command + "\n", encoding="utf-8")
        with log_path.open("wb") as output:
            process = subprocess.Popen(["/bin/sh", "-c", command], cwd=work, env=environment,
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
            fields = ["size", "count", "type", "redop", "root", "placement", "time_kind",
                      "time", "algbw", "busbw", "wrong", "source_line"]
            with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
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


def run_all(inputs, test_root, binary_dir, output, hostfile=None, base_environment=None, mpi_interface=None):
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
    output.mkdir(parents=True, exist_ok=False)
    runtime = {"test_root": test_root, "binary_dir": binary_dir, "runner_root": _runner_root(test_root),
               "output": output, "hostfile": hostfile, "input": inputs["directory"],
               "python_bin": _stage_interpreter(output), "mpi_interface": mpi_interface,
               "base_environment": dict(os.environ if base_environment is None else base_environment)}
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
            if row["status"] != "PASSED" and case["suite_spec"]["nodes"] > 1:
                reason = "Stopped after a multi-node failure; remote rank cleanup is not locally provable"
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--profile", choices=("standard", "notsm"), required=True)
    parser.add_argument("--nodes", type=int, help="Run only Cases whose Suite requests this node count")
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--binary-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hostfile", type=Path)
    parser.add_argument("--mpi-interface", help="Actual TCP interface for an exported {mpi_interface} binding")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        inputs = load_inputs(args.input, args.profile, args.nodes)
        if args.validate_only:
            if args.output is not None or args.test_root is not None:
                raise ValueError("--validate-only does not accept execution paths")
            print(json.dumps({"profile": args.profile, "cases": len(inputs["cases"]),
                              "suites": len({(c["plan"], c["suite"]) for c in inputs["cases"]})}, sort_keys=True))
            return 0
        if args.test_root is None or args.output is None:
            raise ValueError("Execution requires --test-root and --output")
        rows = run_all(inputs, args.test_root, args.binary_dir, args.output, args.hostfile,
                       mpi_interface=args.mpi_interface)
        return 0 if all(row["status"] == "PASSED" for row in rows) else 1
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
