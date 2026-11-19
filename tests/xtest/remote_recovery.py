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


def _uptime(proc_root="/proc"):
    return float((Path(proc_root) / "uptime").read_text().split()[0])


def _processes(proc_root):
    proc_root = Path(proc_root)
    uptime = _uptime(proc_root)
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
                "sid": int(fields[3]), "state": state, "start": start,
                "age": max(0, uptime - start), "cmdline": cmdline, "executable": executable,
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


def _scan(processes, started_after, protected_pids):
    protected = _protected_pids(processes) if protected_pids is None else set(protected_pids)
    candidates = []
    for process in processes.values():
        if process["start"] < started_after or process["state"].startswith(("Z", "X")):
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


def scan_candidates(max_age, proc_root="/proc", protected_pids=None):
    processes = _processes(proc_root)
    return _scan(processes, _uptime(proc_root) - max_age, protected_pids)


def scan_candidates_since(started_after, proc_root="/proc", protected_pids=None):
    return _scan(_processes(proc_root), started_after, protected_pids)


def _groups(candidates):
    return sorted({process["pgid"] for process in candidates if process["pgid"] > 1})


def _identities(candidates):
    return {(process["pid"], process.get("start")) for process in candidates}


def _identity_records(identities):
    return [{"pid": pid, "start": start}
            for pid, start in sorted(identities, key=lambda value: (value[0], value[1] or -1))]


def recover(max_age, controller_pid=0, scanner=scan_candidates_since, sleeper=time.sleep,
            killer=os.killpg, discovery_grace=1.0):
    default_scanner = scanner is scan_candidates_since
    minimum_start = _uptime("/proc") - max_age if default_scanner else max_age

    def scan():
        if default_scanner:
            protected = _protected_pids(_processes("/proc"))
            if controller_pid > 0:
                protected.add(controller_pid)
            return scanner(minimum_start, protected_pids=protected)
        return scanner(minimum_start)

    before = scan()
    discovery_deadline = time.monotonic() + discovery_grace
    while not before and time.monotonic() < discovery_deadline:
        sleeper(0.1)
        before = scan()
    report = {"safe": False, "case_start": minimum_start, "before": before,
              "term_pgids": [], "kill_pgids": [], "term_identities": [], "kill_identities": [],
              "remaining": [], "errors": []}
    if not before:
        report["safe"] = True
        return report

    protected = _protected_pids(_processes("/proc")) if default_scanner else set()
    if controller_pid > 0:
        protected.add(controller_pid)
    protected_groups = {process["pgid"] for process in _processes("/proc").values()
                        if process["pid"] in protected} if default_scanner else set()
    term_identities = set()
    kill_identities = set()

    def signal_candidates(candidates, sig):
        groups = _groups(candidates)
        if protected_groups.intersection(groups):
            raise ValueError("A candidate process belongs to the recovery session or controller")
        for pgid in groups:
            try:
                killer(pgid, sig)
            except ProcessLookupError:
                pass
        return groups

    try:
        new_groups = [pgid for pgid in _groups(before) if pgid not in report["term_pgids"]]
        if new_groups:
            signal_candidates([process for process in before if process["pgid"] in new_groups],
                              signal.SIGTERM)
            report["term_pgids"].extend(new_groups)
            term_identities.update(_identities(before))
    except (OSError, ValueError) as error:
        report["errors"].append(str(error))
        report["remaining"] = scan()
        return report

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        remaining = scan()
        if not remaining:
            report["term_identities"] = _identity_records(term_identities)
            report["safe"] = True
            return report
        try:
            new_groups = [pgid for pgid in _groups(remaining) if pgid not in report["term_pgids"]]
            if new_groups:
                signal_candidates(
                    [process for process in remaining if process["pgid"] in new_groups], signal.SIGTERM)
                report["term_pgids"].extend(new_groups)
                term_identities.update(_identities(remaining))
        except (OSError, ValueError) as error:
            report["errors"].append(str(error))
            report["remaining"] = remaining
            return report
        sleeper(0.1)

    remaining = scan()
    try:
        kill_identities.update(_identities(remaining))
        signal_candidates(remaining, signal.SIGKILL)
        report["kill_pgids"] = _groups(remaining)
    except (OSError, ValueError) as error:
        report["errors"].append(str(error))
    sleeper(0.2)
    report["remaining"] = scan()
    report["term_identities"] = _identity_records(term_identities)
    report["kill_identities"] = _identity_records(kill_identities)
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
