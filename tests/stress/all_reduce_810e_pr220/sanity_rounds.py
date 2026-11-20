#!/usr/bin/env python3
"""Repeat the fixed PR snapshot's entire 810E single-node sanity selection."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from stress import kernel_delta, kernel_snapshot, save_json, scan_log, sha256, timestamp


EXPECTED = {"standard": 17, "notsm": 9}


def load_executor(source):
    directory = source / "pccl" / "tests" / "xtest"
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("pr220_run_cases", directory / "run_cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artifact_hashes(root):
    paths = set()
    for profile in EXPECTED:
        built = root / f"output_{profile}"
        paths.update(path.resolve() for path in (built / "lib").glob("lib*.so*") if path.is_file())
        paths.update(path.resolve() for path in (built / "pccl-tests" / "build").iterdir()
                     if path.is_file() and path.suffix != ".o")
    return {str(path): sha256(path) for path in sorted(paths)}


def summarize(output, rounds, expected_rounds, artifacts, changed=False):
    cases = [case for row in rounds for group in row["profiles"].values() for case in group]
    completed = sum(all(len(row["profiles"].get(p, [])) == n for p, n in EXPECTED.items())
                    and all(case["status"] not in ("CANCELLED", "NOT_RUN")
                            for group in row["profiles"].values() for case in group) for row in rounds)
    failed = [case for case in cases if case["status"] != "PASSED"]
    result = {"requested_rounds": expected_rounds, "completed_rounds": completed,
              "cases_per_round": EXPECTED, "expected_case_executions": expected_rounds * sum(EXPECTED.values()),
              "executed_cases": len(cases), "passed_cases": len(cases) - len(failed),
              "failed_cases": len(failed), "failed_rounds": [row["round"] for row in rounds
              if any(case["status"] != "PASSED" for group in row["profiles"].values() for case in group)],
              "complete": completed == expected_rounds, "artifact_changed": changed,
              "artifact_sha256_before": artifacts, "updated_at": timestamp(), "rounds": rounds}
    save_json(output / "summary.json", result)
    return result


def run(root, output, rounds, interface):
    output.mkdir(parents=True, exist_ok=False)
    executor = load_executor(root / "output_standard")
    selections, identities = {}, {}
    for profile, expected in EXPECTED.items():
        source = root / f"output_{profile}"
        selection = executor.load_inputs(source / "pccl/tests/xtest/minimal-csv-oam810e-eic",
                                         profile, nodes=1)
        if len(selection["cases"]) != expected or any(case["suite_spec"]["cards_per_node"] != 2
                                                     for case in selection["cases"]):
            raise ValueError(f"Unexpected {profile} single-node selection")
        selections[profile] = selection
        identity = json.loads((root / f"output_{profile}.build.json").read_text())
        if sha256(root / f"output_{profile}.tar.gz") != identity["artifact_sha256"]:
            raise ValueError(f"Build archive changed: {profile}")
        identities[profile] = executor._verified_build_identity(selection,
            root / f"output_{profile}.build.json", os.environ["PCCL_REVISION"], identity["artifact_sha256"])
    artifacts = artifact_hashes(root)
    save_json(output / "metadata.json", {"started_at": timestamp(), "rounds": rounds,
              "profiles": identities, "cases_per_round": EXPECTED, "nodes": 1,
              "cards_per_node": 2, "mpi_interface": interface, "builds_per_profile": 1})
    records = []
    changed = False
    summarize(output, records, rounds, artifacts)
    try:
        for iteration in range(1, rounds + 1):
            record = {"round": iteration, "profiles": {}}
            records.append(record)
            for profile in EXPECTED:
                source = root / f"output_{profile}"
                target = output / f"round-{iteration:03d}" / profile
                before = kernel_snapshot()
                rows = executor.run_all(selections[profile], source, source / "pccl-tests/build", target,
                                        mpi_interface=interface, build_identity=identities[profile])
                after = kernel_snapshot()
                delta, mode = kernel_delta(before, after)
                (target / "dmesg.delta.log").write_text(delta)
                save_json(target / "dmesg.status.json", {"delta_mode": mode,
                          "before": {k: v for k, v in before.items() if k != "text"},
                          "after": {k: v for k, v in after.items() if k != "text"}})
                for row in rows:
                    case_dir = target / row["plan"] / row["suite"] / row["name"]
                    log = case_dir / row["raw_log"] if row["raw_log"] else None
                    if log is not None and log.is_file():
                        counts = scan_log(log)
                        if row["status"] == "PASSED" and any(counts[key] for key in (
                                "hg_mem_map_invalid_argument", "unhandled_hggc_error", "test_failure")):
                            row.update(status="FAILED", error="Fatal failure text in raw log despite zero exit")
                            executor._write_json(case_dir / "result.json", row)
                            (case_dir / "dmesg.log").write_text(after["text"])
                        save_json(case_dir / "failure-signatures.json", counts)
                executor._write_summary(target, rows)
                record["profiles"][profile] = rows
                summary = summarize(output, records, rounds, artifacts)
                print(f"[round {iteration:03d}/{rounds} {profile}] "
                      f"{sum(row['status'] == 'PASSED' for row in rows)}/{len(rows)} passed; "
                      f"cumulative failed cases={summary['failed_cases']}", flush=True)
                if any(row["status"] in ("CANCELLED", "NOT_RUN") for row in rows):
                    return 1
    finally:
        for path, digest in artifacts.items():
            try:
                changed = changed or sha256(Path(path)) != digest
            except OSError:
                changed = True
        summary = summarize(output, records, rounds, artifacts, changed)
    return 0 if summary["complete"] and not summary["failed_cases"] and not changed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--mpi-interface", default="eth0")
    args = parser.parse_args()
    if args.rounds <= 0:
        parser.error("rounds must be positive")
    return run(args.root.resolve(), args.output.resolve(), args.rounds, args.mpi_interface)


if __name__ == "__main__":
    raise SystemExit(main())
