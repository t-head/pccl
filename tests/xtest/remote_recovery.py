#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Find and clean processes started by a failed Case on one cluster worker."""

import argparse
import json
import os
from pathlib import Path
import signal
import time


INFRA_EXECUTABLES = {"ssh", "sshd"}
INFRA_CHILD_EXECUTABLES = {"bash", "sh", "cat", "hostname", "jq", "ps", "sleep", "timeout"}
INFRA_COMMANDS = {"/bin/sleep 0.2", "sleep 0.2"}


def _processes(proc_root):
    proc_root = Path(proc_root)
    uptime = float((proc_root / "uptime").read_text().split()[0])
    ticks = os.sysconf("SC_CLK_TCK")
    result = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            fields = stat.rsplit(") ", 1)[1].split()
            state = fields[0]
            start = int(fields[19]) / ticks
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace").strip()
            try:
                executable = os.path.basename(os.readlink(entry / "exe"))
            except OSError:
                executable = ""
            result[int(entry.name)] = {
                "pid": int(entry.name), "ppid": int(fields[1]), "pgid": int(fields[2]),
                "sid": int(fields[3]), "state": state, "age": max(0, uptime - start),
                "cmdline": cmdline, "executable": executable,
            }
        except (OSError, ValueError, IndexError):
            continue
    return result


def _protected_pids(processes):
    protected = set()
    pid = os.getpid()
    while pid > 1 and pid in processes and pid not in protected:
        protected.add(pid)
        pid = processes[pid]["ppid"]
    return protected


def scan_candidates(max_age, proc_root="/proc", protected_pids=None):
    processes = _processes(proc_root)
    protected = _protected_pids(processes) if protected_pids is None else set(protected_pids)
    candidates = []
    for process in processes.values():
        if process["age"] > max_age or process["state"].startswith(("Z", "X")):
            continue
        pid = process["pid"]
        if pid <= 1 or pid in protected or process["executable"] in INFRA_EXECUTABLES:
            continue
        command = process["cmdline"]
        if (not command or "/run/cluster/" in command or "/run/cluster-config/" in command
                or "remote_recovery.py" in command or command in INFRA_COMMANDS):
            continue
        parent = processes.get(process["ppid"], {})
        if ("/run/cluster/" in parent.get("cmdline", "")
                and process["executable"] in INFRA_CHILD_EXECUTABLES):
            continue
        candidates.append(process)
    return sorted(candidates, key=lambda process: process["pid"])


def _groups(candidates):
    return sorted({process["pgid"] for process in candidates if process["pgid"] > 1})


def recover(max_age, controller_pid=0, scanner=scan_candidates, sleeper=time.sleep, killer=os.killpg):
    def scan(age):
        if scanner is scan_candidates:
            protected = _protected_pids(_processes("/proc"))
            if controller_pid > 0:
                protected.add(controller_pid)
            return scanner(age, protected_pids=protected)
        return scanner(age)

    before = scan(max_age)
    report = {"safe": False, "before": before, "term_pgids": [], "kill_pgids": [],
              "remaining": [], "errors": []}
    if not before:
        report["safe"] = True
        return report
    protected = _protected_pids(_processes("/proc")) if scanner is scan_candidates else set()
    if controller_pid > 0:
        protected.add(controller_pid)
    protected_groups = {process["pgid"] for process in _processes("/proc").values()
                        if process["pid"] in protected} if scanner is scan_candidates else set()
    try:
        groups = _groups(before)
        if protected_groups.intersection(groups):
            raise ValueError("A candidate process belongs to the recovery session")
        for pgid in groups:
            try:
                killer(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        report["term_pgids"] = groups
    except (OSError, ValueError) as error:
        report["errors"].append(str(error))
        report["remaining"] = scan(max_age)
        return report
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        remaining = scan(max_age)
        if not remaining:
            report["safe"] = True
            return report
        sleeper(0.1)
    remaining = scan(max_age)
    try:
        groups = _groups(remaining)
        if protected_groups.intersection(groups):
            raise ValueError("A remaining process belongs to the recovery session")
        for pgid in groups:
            try:
                killer(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        report["kill_pgids"] = groups
    except (OSError, ValueError) as error:
        report["errors"].append(str(error))
    sleeper(0.2)
    report["remaining"] = scan(max_age)
    report["safe"] = not report["remaining"] and not report["errors"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-age", type=float, required=True)
    parser.add_argument("--controller-pid", type=int, default=0)
    args = parser.parse_args()
    if args.max_age <= 0 or args.max_age > 86400:
        parser.error("--max-age must be in (0, 86400]")
    if args.controller_pid < 0:
        parser.error("--controller-pid must be non-negative")
    report = recover(args.max_age, args.controller_pid)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["safe"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
