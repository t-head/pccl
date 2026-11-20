#!/usr/bin/env python3
"""Run independent MPI launches against one build, preserving failure evidence."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


CASE_ENV = {
    "PCCL_GDR_CPU_FLUSH": "1",
    "PCCL_CPU_FLUSH_ASYNC": "1",
    "NCCL_IB_ADAPTIVE_ROUTING": "1",
    "NCCL_IB_OOO_RQ": "1",
    "NCCL_DEBUG": "INFO",
    "NCCL_DEBUG_SUBSYS": "ALLOC,PROFILE",
}
SIGNATURES = {
    "hg_mem_map_invalid_argument": r"hgMemMap[^\n]*invalid argument",
    "unhandled_hggc_error": r"unhandled hggc error",
    "test_failure": r"Test (?:PCCL )?failure",
    "qp_create_warning": r"create qp 0 failed",
    "qp_wr_warning": r"max send wr 384 is not power of 2",
}
FATAL_SIGNATURES = {"hg_mem_map_invalid_argument", "unhandled_hggc_error", "test_failure"}


def timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="milliseconds")


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mpi_command(binary: Path, interface: str, mpirun: str) -> list[str]:
    command = [mpirun, "--allow-run-as-root"]
    for name, value in CASE_ENV.items():
        command.extend(["-x", f"{name}={value}"])
    command.extend([
        "-np", "2", "-npernode", "2",
        "-mca", "btl_tcp_if_include", interface,
        "-mca", "btl", "^openib",
        "-mca", "btl_openib_warn_no_device_params_found", "0",
        str(binary), "-t", "1", "-g", "1", "-d", "half",
        "-b", "8", "-e", "16G", "-f", "2", "-n", "20", "-w", "5",
        "--elapsed_type", "1",
    ])
    return command


def scan_log(path: Path) -> dict[str, int]:
    patterns = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in SIGNATURES.items()}
    counts = dict.fromkeys(patterns, 0)
    with path.open(errors="replace") as stream:
        for line in stream:
            for name, pattern in patterns.items():
                counts[name] += len(pattern.findall(line))
    return counts


def kernel_snapshot() -> dict:
    """Read, never clear, the kernel ring buffer. Permissions may be unavailable."""
    try:
        result = subprocess.run(["dmesg"], capture_output=True, text=True, errors="replace", timeout=5)
        return {"time": timestamp(), "return_code": result.returncode,
                "text": result.stdout, "error": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"time": timestamp(), "return_code": None, "text": "", "error": str(error)}


def kernel_delta(before: dict, after: dict) -> tuple[str, str]:
    if before["return_code"] != 0 or after["return_code"] != 0:
        return "", "unavailable"
    old, new = before["text"].splitlines(keepends=True), after["text"].splitlines(keepends=True)
    if not old:
        return "".join(new), "initial_buffer_empty"
    if new[:len(old)] == old:
        return "".join(new[len(old):]), "appended"
    # Ring-buffer rollover: locate the retained suffix without clearing host logs.
    for start in range(len(new) - 1, -1, -1):
        if new[start] == old[-1]:
            overlap = min(start + 1, len(old))
            if new[start - overlap + 1:start + 1] == old[-overlap:]:
                return "".join(new[start + 1:]), "ring_buffer_rolled"
    return "".join(new), "no_overlap_full_after_not_an_exact_delta"


def stop_process_group(process: subprocess.Popen, grace: float) -> None:
    # Kill this launch's session only; never pkill other users' MPI/PPU jobs.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_once(command: list[str], directory: Path, timeout: float, grace: float) -> dict:
    started = timestamp()
    start = time.monotonic()
    before = kernel_snapshot()
    process = None
    reason = None
    code = None
    with (directory / "raw.log").open("wb") as output:
        try:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                reason = "TIMEOUT"
        except OSError as error:
            reason = "LAUNCH_ERROR"
            output.write((str(error) + "\n").encode())
        except KeyboardInterrupt:
            reason = "INTERRUPTED"
        finally:
            # A failed MPI launcher can exit before its remaining local ranks.
            if process is not None:
                stop_process_group(process, grace)
                code = process.returncode
    elapsed = time.monotonic() - start
    after = kernel_snapshot()
    counts = scan_log(directory / "raw.log")
    fatal = sorted(name for name in FATAL_SIGNATURES if counts[name])
    status = reason or ("FAILED" if code != 0 or fatal else "PASSED")
    delta, delta_mode = kernel_delta(before, after)
    (directory / "dmesg.delta.log").write_text(delta)
    save_json(directory / "dmesg.status.json", {
        "before": {key: value for key, value in before.items() if key != "text"},
        "after": {key: value for key, value in after.items() if key != "text"},
        "delta_mode": delta_mode,
    })
    if status != "PASSED":
        (directory / "dmesg.before.log").write_text(before["text"])
        (directory / "dmesg.after.log").write_text(after["text"])
    return {"status": status, "return_code": code, "started_at": started,
            "finished_at": timestamp(), "duration_seconds": round(elapsed, 3),
            "fatal_signatures": fatal, "signatures": counts,
            "target_hggc_failure": bool(counts["unhandled_hggc_error"]),
            "dmesg_delta_mode": delta_mode}


def write_summary(output: Path, records: list[dict], requested: int, metadata: dict,
                  interrupted: bool = False, artifact_changed: bool = False) -> dict:
    completed = [record for record in records if record["status"] != "INTERRUPTED"]
    failures = [record["iteration"] for record in completed if record["status"] != "PASSED"]
    passed = sum(record["status"] == "PASSED" for record in completed)
    summary = {
        "requested_runs": requested, "attempted_runs": len(records), "completed_runs": len(completed),
        "passed": passed, "failed": len(failures), "failed_iterations": failures,
        "first_failed_iteration": failures[0] if failures else None,
        "failure_rate_percent": round(100 * len(failures) / len(completed), 2) if completed else None,
        "target_hggc_failure_iterations": [r["iteration"] for r in completed if r["target_hggc_failure"]],
        "hg_mem_map_invalid_argument_iterations": [r["iteration"] for r in completed if r["signatures"]["hg_mem_map_invalid_argument"]],
        "timeout_iterations": [r["iteration"] for r in completed if r["status"] == "TIMEOUT"],
        "interrupted": interrupted, "artifact_changed": artifact_changed,
        "complete": len(completed) == requested and not interrupted,
        "pccl_revision": metadata["pccl_revision"], "pccl_tests_revision": metadata["pccl_tests_revision"],
        "artifacts": metadata["artifacts"], "updated_at": timestamp(), "runs": records,
    }
    save_json(output / "summary.json", summary)
    temporary = output / "summary.csv.tmp"
    with temporary.open("w", newline="") as stream:
        columns = ["iteration", "status", "return_code", "duration_seconds", "target_hggc_failure", "log"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(output / "summary.csv")
    return summary


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--timeout", type=positive_float, default=300)
    parser.add_argument("--kill-grace", type=positive_float, default=5)
    parser.add_argument("--mpi-interface", default="eth0")
    parser.add_argument("--mpirun", default="mpirun")
    args = parser.parse_args(argv)
    if args.runs <= 0:
        parser.error("--runs must be positive")
    binary = args.binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error("--binary must be an existing executable")
    launcher = shutil.which(args.mpirun)
    if launcher is None:
        parser.error("MPI launcher is not available")
    artifacts = {"all_reduce_perf": {"path": str(binary), "sha256": sha256(binary)}}
    if args.library:
        library = args.library.resolve()
        if not library.is_file():
            parser.error("--library must be an existing file")
        artifacts["libpccl"] = {"path": str(library), "sha256": sha256(library)}
    output = args.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("--output already exists; refusing to overwrite evidence")
    command = mpi_command(binary, args.mpi_interface, launcher)
    selected_environment = {key: value for key, value in os.environ.items()
                            if key.startswith(("PCCL_", "NCCL_", "OMPI_", "PPU_", "HG"))
                            or key in {"PATH", "LD_LIBRARY_PATH", "HWLOC_COMPONENTS"}}
    selected_environment.update(CASE_ENV)
    metadata = {
        "case": "test_op_all_reduce_perf_ranks_2_nnodes1",
        "plan": "pccl_810e_op_perf_single_node1_plan", "pr": 220,
        "pccl_revision": os.environ.get("PCCL_REVISION"),
        "pccl_tests_revision": os.environ.get("PCCL_TESTS_REVISION"),
        "started_at": timestamp(), "hostname": os.uname().nodename,
        "requested_runs": args.runs, "timeout_seconds": args.timeout,
        "mpi_interface": args.mpi_interface, "command": command,
        "environment": selected_environment, "artifacts": artifacts,
        "independent_processes": True,
    }
    save_json(output / "metadata.json", metadata)
    (output / "command.txt").write_text(shlex.join(command) + "\n")
    records = []
    interrupted = False
    changed = False
    previous = signal.getsignal(signal.SIGTERM)

    def on_termination(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_termination)
    write_summary(output, records, args.runs, metadata)
    try:
        for iteration in range(1, args.runs + 1):
            directory = output / f"run-{iteration:03d}"
            directory.mkdir()
            (directory / "command.txt").write_text(shlex.join(command) + "\n")
            record = run_once(command, directory, args.timeout, args.kill_grace)
            record.update(iteration=iteration, log=str(directory / "raw.log"))
            records.append(record)
            save_json(directory / "result.json", record)
            interrupted = record["status"] == "INTERRUPTED"
            write_summary(output, records, args.runs, metadata, interrupted)
            print(f"[{iteration:03d}/{args.runs}] {record['status']} "
                  f"rc={record['return_code']} {record['duration_seconds']:.3f}s", flush=True)
            if interrupted:
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        signal.signal(signal.SIGTERM, previous)
        for artifact in artifacts.values():
            try:
                current = sha256(Path(artifact["path"]))
            except OSError:
                current = None
            artifact["sha256_after"] = current
            changed = changed or current != artifact["sha256"]
        summary = write_summary(output, records, args.runs, metadata, interrupted, changed)
    print(f"Completed {summary['completed_runs']}/{args.runs}: "
          f"passed={summary['passed']}, failed={summary['failed']}. Evidence: {output}")
    return 0 if summary["complete"] and not summary["failed"] and not changed else 1


if __name__ == "__main__":
    sys.exit(main())
