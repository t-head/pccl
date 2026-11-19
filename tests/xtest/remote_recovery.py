#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Own and clean processes belonging to one execution group's cgroup."""

import argparse
import json
import os
from pathlib import Path
import re
import time


CASE_ID = re.compile(r"[a-f0-9]{32}\Z")


def _pids(path):
    return [int(value) for value in path.read_text().split()]


def _kill(path):
    (path / "cgroup.kill").write_text("1\n", encoding="ascii")


def launch(case_id, command, cgroup_root="/sys/fs/cgroup"):
    """Exec a controller-side Case after assigning this process to its cgroup.

    subprocess.Popen(start_new_session=True) establishes the process group before
    this helper runs, so the PID and process-group ID remain stable across exec.
    """
    if not CASE_ID.fullmatch(case_id):
        raise ValueError("Invalid recovery Case ID")
    directory = Path(cgroup_root) / f"pccl-recovery-{case_id}"
    directory.mkdir(mode=0o755, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Recovery cgroup is not a real directory")
    (directory / "cgroup.procs").write_text(f"{os.getpid()}\n", encoding="ascii")
    os.execv("/bin/sh", ["/bin/sh", "-c", command])


def recover(case_id, cgroup_root="/sys/fs/cgroup", sleeper=time.sleep):
    report = {"case_id": case_id, "before": [], "remaining": [], "safe": False, "errors": []}
    if not CASE_ID.fullmatch(case_id):
        report["errors"].append("Invalid recovery Case ID")
        return report
    directory = Path(cgroup_root) / f"pccl-recovery-{case_id}"
    try:
        if not directory.exists():
            report["safe"] = True
            return report
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Recovery cgroup is not a real directory")
        processes_path = directory / "cgroup.procs"
        report["before"] = _pids(processes_path)
        if report["before"]:
            _kill(directory)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                remaining = _pids(processes_path)
                if not remaining:
                    break
                sleeper(0.1)
        report["remaining"] = _pids(processes_path)
        if report["remaining"]:
            raise ValueError("Recovery cgroup still contains live processes")
        directory.rmdir()
        report["safe"] = True
    except (OSError, ValueError) as error:
        report["errors"].append(str(error))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--cgroup-root", default="/sys/fs/cgroup")
    parser.add_argument("--launch", metavar="COMMAND",
                        help="Assign this process to the Case cgroup and exec COMMAND via /bin/sh")
    args = parser.parse_args()
    if args.launch is not None:
        try:
            launch(args.case_id, args.launch, args.cgroup_root)
        except (OSError, ValueError) as error:
            parser.exit(1, f"Unable to launch Case in its recovery cgroup: {error}\n")
        raise AssertionError("os.execv returned unexpectedly")
    report = recover(args.case_id, args.cgroup_root)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["safe"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
