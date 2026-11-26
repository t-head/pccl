"""Run the existing two-node M890P Case and collect host-only GCOV data."""

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
import re
import shutil
import shlex
import subprocess
import sys
import tarfile
import tempfile

from pccl_host_coverage import collect, command, version, write_json


SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / "tests" / "xtest"))
import artifact_cache
import run_cases


SUITE = "register_test_shm_multi_nodes2_suite"
CASE = ("test_op_all_reduce_perf_algo_Auto_proto_Auto_redop_sum_ranks_16_"
        "datatype_float_datasize_32MB_iters_10_threads_1_process_8_nnodes2_autodetect_rdma_gdr")
NODES = 2
CARDS = 8
RDMA = re.compile(r"\b(\d+)\[[\da-f]+\] -> (\d+)\[[\da-f]+\] \[(send|receive)\] via NET/IB/[^\s]+", re.I)
SOCKET = re.compile(r"\bvia NET/Socket/", re.I)


def select_case(directory):
    inputs = run_cases.load_inputs(directory, "standard", NODES, suite=SUITE)
    if len(inputs["cases"]) != 1:
        raise ValueError("The experiment must select exactly one Case")
    case = inputs["cases"][0]
    if case["name"] != CASE or case["suite_spec"]["cards_per_node"] != CARDS:
        raise ValueError("The existing two-node RDMA Case or its allocation changed")
    argv = shlex.split(case["cmd"])
    for flag, value in {"--op": "all_reduce_perf", "--nnodes": "2", "--ranks": "16",
                        "--nprocs": "8", "--threads": "1", "--iters": "10",
                        "--data-size": "32MB", "--data-type": "float", "--redop": "sum",
                        "--cross-node": "gdr"}.items():
        if argv.count(flag) != 1 or argv[argv.index(flag) + 1:argv.index(flag) + 2] != [value]:
            raise ValueError(f"Unexpected two-node Case argument: {flag}")
    # The coverage experiment needs correctness evidence, not just exit 0.
    case["check"] = "validation_summary"
    return inputs


def rank_profiles(ranks, notes):
    expected = {f"rank-{rank}" for rank in range(NODES * CARDS)}
    actual = {path.name for path in ranks.iterdir()}
    if expected != actual:
        raise ValueError(f"Incomplete rank directories: missing={sorted(expected - actual)}, "
                         f"unexpected={sorted(actual - expected)}")
    workers = defaultdict(list)
    host_by_rank = {}
    logs = {}
    counters = {}
    for rank in range(NODES * CARDS):
        path = ranks / f"rank-{rank}"
        rank_text, local, host = (path / "rank.tsv").read_text().strip().split("\t")
        if rank_text != str(rank) or not host:
            raise ValueError(f"Invalid rank receipt: {path.name}")
        workers[host].append(int(local))
        host_by_rank[rank] = host
        errors = path / "gcov-errors.log"
        if errors.is_file() and errors.stat().st_size:
            raise ValueError(f"GCOV runtime errors for rank {rank}; see {errors.name}")
        files = list(path.rglob("*.gcda"))
        if not files or not (path / "obj/init.gcda").is_file():
            raise ValueError(f"No PCCL init counters for rank {rank}")
        for item in files:
            relative = item.relative_to(path)
            if not item.stat().st_size or not (notes / relative.with_suffix(".gcno")).is_file():
                raise ValueError(f"Counter does not match a build note: rank {rank}: {relative}")
        counters[rank] = len(files)
        logs[rank] = (path / "pccl.log").read_text(errors="replace")
    if len(workers) != NODES or any(sorted(ids) != list(range(CARDS)) for ids in workers.values()):
        raise ValueError(f"Unexpected MPI rank placement: {dict(workers)}")
    rdma_workers = set()
    for rank, text in logs.items():
        if SOCKET.search(text):
            raise ValueError(f"Rank {rank} logged a Socket data channel")
        for source, dest, direction in RDMA.findall(text):
            src_host, dst_host = host_by_rank.get(int(source)), host_by_rank.get(int(dest))
            if src_host and dst_host and src_host != dst_host:
                owner = source if direction.lower() == "send" else dest
                if int(owner) == rank:
                    rdma_workers.add(host_by_rank[rank])
    if rdma_workers != workers.keys():
        raise ValueError(f"Missing cross-host IB/RoCE channels on: {sorted(workers.keys() - rdma_workers)}")
    return {"ranks": len(host_by_rank), "ranks_per_worker": dict(Counter(host_by_rank.values())),
            "counter_files_by_rank": counters, "cross_host_rdma": True}


def merge_profiles(ranks, notes, build, report):
    """Merge copies, leaving rank-local counters untouched for diagnostics."""
    if build.exists():
        raise ValueError(f"Refusing to reuse a coverage build directory: {build}")
    shutil.copytree(notes, build)
    with tempfile.TemporaryDirectory(prefix="m890p-merge-") as tmp:
        tmp = Path(tmp)
        accum = tmp / "current"
        accum.mkdir()
        for rank in range(NODES * CARDS):
            staged = tmp / "rank"
            staged.mkdir()
            for counter in (ranks / f"rank-{rank}").rglob("*.gcda"):
                path = staged / counter.relative_to(ranks / f"rank-{rank}")
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(counter, path)
            if rank == 0:
                accum.rmdir()
                staged.rename(accum)
                continue
            output = tmp / "merged"
            log = report / f"gcov-merge-rank-{rank}.log"
            rc = command(["gcov-tool-13", "merge", str(accum), str(staged), "-o", str(output)],
                         log, tmp, os.environ.copy(), 60)
            if rc or len(log.read_text().splitlines()) != 1:
                raise ValueError(f"GCOV merge failed for rank {rank}; see {log.name}")
            shutil.rmtree(accum)
            shutil.rmtree(staged)
            output.rename(accum)
        merged = {counter.relative_to(accum) for counter in accum.rglob("*.gcda")}
        expected = {counter.relative_to(rank) for rank in ranks.iterdir() for counter in rank.rglob("*.gcda")}
        if merged != expected:
            raise ValueError("The GCOV merge did not preserve the counter file inventory")
        for relative in sorted(merged):
            shutil.copyfile(accum / relative, build / relative)


def save_rank_evidence(ranks, report):
    with tarfile.open(report / "rank-evidence.tar.gz", "w:gz") as archive:
        for path in sorted(ranks.rglob("*")):
            if path.is_file() and path.name != "core":
                archive.add(path, arcname=path.relative_to(ranks), recursive=False)


def run(args):
    shared = args.shared.resolve()
    runtime = shared / "runtime"
    ranks = shared / "ranks"
    notes = runtime / "notes"
    report = shared / "report"
    build = SOURCE / "build"
    status = {"scope": "PCCL C++ host code", "topo": "ICN64_890P_CX8", "nodes": NODES,
              "cards_per_node": CARDS, "case": CASE, "status": "failed",
              "performance_baseline_eligible": False}
    try:
        ranks.mkdir()
        pccl_rev, tests_rev = (runtime / "revisions.txt").read_text().splitlines()
        identity = artifact_cache.create_record(shared / "runtime.tar.gz", pccl_rev, tests_rev, "standard")
        status["build"] = identity
        status["compiler"] = (runtime / "cxx-version.txt").read_text().strip()
        if version("gcov-13") != (runtime / "gcov-version.txt").read_text().strip():
            raise ValueError("Reporting gcov differs from the build's gcov")
        inputs = select_case(SOURCE / "tests" / "xtest" / "minimal-csv")
        if inputs["metadata"]["runner_revision"] != tests_rev:
            raise ValueError("The benchmark revision differs from the CSV export")
        (runtime / "pccl").symlink_to(SOURCE, target_is_directory=True)
        bin_dir = runtime / "rank-bin"
        bin_dir.mkdir()
        (bin_dir / "all_reduce_perf").symlink_to(Path(__file__).with_name("pccl_m890p_coverage_rank.sh"))
        environment = os.environ.copy()
        environment.pop("PCCL_PATH", None)
        # The pinned run_test.py forwards PCCL_* variables to every MPI rank.
        environment.update(PCCL_COVERAGE_RUNTIME=str(runtime), PCCL_DEBUG="INFO",
                           PCCL_DEBUG_SUBSYS="INIT,NET", PCCL_NET="IB", PCCL_IB_DISABLE="0",
                           PCCL_NET_GDR_LEVEL="SYS")
        results = shared / "case-results"
        try:
            rows = run_cases.run_all(inputs, runtime, bin_dir, results, args.hostfile,
                                     base_environment=environment, mpi_interface=args.mpi_interface,
                                     build_identity=identity, map_nvidia_devices=True)
        finally:
            # Keep diagnostics, not the working-directory links into the runtime or hostfile.
            if results.is_dir():
                shutil.copytree(results, report / "case",
                                ignore=shutil.ignore_patterns("work", "python-env", "dmesg.log"))
        if len(rows) != 1 or rows[0]["status"] != "PASSED":
            details = rows[0]["error"] if len(rows) == 1 else f"{len(rows)} results"
            raise ValueError(f"The single AllReduce Case did not pass validation: {details}")
        status["case_passed"] = True
        status["rank_evidence"] = rank_profiles(ranks, notes)
        merge_profiles(ranks, notes, build, report)
        status["coverage"] = collect(SOURCE, build, report, "gcov-13")
        status["status"] = "passed"
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        status["error"] = str(error)
        print(f"Coverage experiment failed: {error}", file=sys.stderr)
        return 1
    finally:
        if ranks.is_dir():
            save_rank_evidence(ranks, report)
        write_json(report / "summary.json", status)
        lines = ["### PCCL M890P CX8 host coverage", "",
                 f"- Result: **{status['status']}**; one AllReduce, 2 nodes × 8 cards.",
                 "- Host C++ coverage only; not a performance baseline."]
        if "error" in status:
            lines.append(f"- {status['error']}")
        elif "coverage" in status:
            for name in ("lines", "functions", "branches"):
                metric = status["coverage"][name]
                lines.append(f"- {name.capitalize()}: {metric['covered']} / {metric['total']}")
        (report / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-case")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--shared", type=Path, required=True)
    run_parser.add_argument("--hostfile", type=Path, required=True)
    run_parser.add_argument("--mpi-interface", required=True)
    args = parser.parse_args()
    if args.command == "check-case":
        select_case(SOURCE / "tests" / "xtest" / "minimal-csv")
        print(f"Selected exactly one existing Case: {CASE}")
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
