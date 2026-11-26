"""One-off offline recovery for the archived 16-rank run 45746419059."""

from collections import defaultdict
from datetime import datetime, timezone
from hashlib import file_digest, sha256
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile


SOURCE = Path("/workspace/source/pccl")
REVISION = "ccdfb190b321acdbba91bc1e8b4dbbcf6d79c379"
TESTS_REVISION = "394556512e87259e9ffa25225632c46b0a0bbcd5"
GCC = "(Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0"
DIGESTS = {
    "runtime.tar.gz": "c086931ec1a85e1930c613b78e7b702e55d7e513fb6693e92e1bc889552dd061",
    "rank-evidence.tar.gz": "3e5c54bcbd613b6fd9af63054305feb8821b04770340cf64fa3d69fdb45b9f70",
    "summary.json": "241f5617edac065b53f2abe6f999e8fa69efb11b61570dea6b42ded3a5abaadc",
}
# Canonical per-file counts from the independently recovered and audited report.
EXPECTED_COUNTS = "cb550081464570ab096934d94892b434205b8e1f5b47a215a5971e64fd1a8a57"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def extract_selected(path, destination, select):
    """Extract only necessary regular files; never unpack or run archived binaries."""
    with tarfile.open(path, "r:gz") as archive:
        selected = []
        names = set()
        for entry in archive:
            name = PurePosixPath(entry.name)
            if not select(name):
                continue
            require(entry.isfile() and not name.is_absolute() and ".." not in name.parts
                    and name not in names, "Unexpected selected archive member")
            names.add(name)
            selected.append(entry)
        archive.extractall(destination, members=selected, filter="data")


def notes_member(path):
    return path.parts[:2] == ("notes", "obj") and path.suffix == ".gcno"


def rank_member(path):
    return (len(path.parts) > 1 and path.parts[0].startswith("rank-")
            and (path.suffix == ".gcda" or path.name in {"rank.tsv", "gcov-errors.log"}))


def validate_ranks(ranks, notes):
    inventory = {path.relative_to(notes).with_suffix(".gcda") for path in notes.rglob("*.gcno")}
    require(len(inventory) == 122 and Path("obj/init.gcda") in inventory, "Incomplete build notes")
    require({path.name for path in ranks.iterdir()} == {f"rank-{i}" for i in range(16)},
            "Incomplete rank directory layout")
    workers = defaultdict(list)
    for i in range(16):
        rank = ranks / f"rank-{i}"
        global_rank, local_rank, host = (rank / "rank.tsv").read_text(encoding="utf-8").strip().split("\t")
        require(global_rank == str(i) and host, f"Invalid rank {i} receipt")
        workers[host].append(int(local_rank))
        errors = rank / "gcov-errors.log"
        require(not errors.exists() or errors.stat().st_size == 0, f"GCOV runtime error on rank {i}")
        counters = {path.relative_to(rank) for path in rank.rglob("*.gcda") if path.stat().st_size > 0}
        require(counters == inventory, f"Rank {i} counters do not match the build notes")
    require(len(workers) == 2 and all(sorted(ids) == list(range(8)) for ids in workers.values()),
            "Expected two workers with eight ranks each")


def replay(inputs, report):
    for name, expected in DIGESTS.items():
        with (inputs / name).open("rb") as stream:
            require(file_digest(stream, "sha256").hexdigest() == expected, f"Checksum mismatch: {name}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    require(revision == REVISION, "Incorrect original source revision")
    original = json.loads((inputs / "summary.json").read_text(encoding="utf-8"))
    build = original["build"]
    require((build["pccl_revision"], build["pccl_tests_revision"], build["artifact_sha256"])
            == (REVISION, TESTS_REVISION, DIGESTS["runtime.tar.gz"]), "Incorrect original build identity")
    require(original["case_passed"] is True and (original["topo"], original["nodes"], original["cards_per_node"])
            == ("ICN64_890P_CX8", 2, 8), "Original two-node Case was not verified")

    sys.path.insert(0, str(SOURCE / ".github/workflows/scripts"))
    from pccl_host_coverage import collect, version, write_json
    # Use only the offline merge, not the old launcher or the RDMA check.
    from pccl_m890p_coverage import merge_profiles

    gcov, gcov_tool, gcovr = (version(tool) for tool in ("gcov-13", "gcov-tool-13", "gcovr"))
    require(gcov == f"gcov {GCC}" and gcov_tool.split(" ", 1)[1] == GCC and gcovr == "gcovr 8.4",
            "Unexpected report tool version")
    with tempfile.TemporaryDirectory(prefix="pccl-coverage-replay-") as scratch:
        scratch = Path(scratch)
        extract_selected(inputs / "runtime.tar.gz", scratch, notes_member)
        ranks = scratch / "ranks"
        extract_selected(inputs / "rank-evidence.tar.gz", ranks, rank_member)
        notes = scratch / "notes"
        validate_ranks(ranks, notes)
        out = scratch / "report"
        out.mkdir()
        build_dir = SOURCE / "build"
        merge_profiles(ranks, notes, build_dir, out)
        # The original runtime did not ship the build-time copy of this public header.
        header = Path("pccl_device/impl/core__funcs.h")
        copy = build_dir / "include" / header
        copy.parent.mkdir(parents=True)
        shutil.copyfile(SOURCE / "src/include" / header, copy)
        metrics = collect(SOURCE, build_dir, out, "gcov-13 --relative-only")
        summary = json.loads((out / "gcovr-summary.json").read_text(encoding="utf-8"))
        files = sorted(summary["files"], key=lambda entry: entry["filename"])
        encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        require(sha256(encoded).hexdigest() == EXPECTED_COUNTS and metrics["files"] == 157
                and metrics["profiles"] == {"notes": 122, "counters": 122},
                "Coverage does not match the audited per-file counts")
        for kind, covered, total in (("lines", 11134, 37159), ("functions", 1153, 2820), ("branches", 8238, 77348)):
            require((metrics[kind]["covered"], metrics[kind]["total"]) == (covered, total),
                    f"Unexpected {kind} coverage total")
        require((out / "html/index.html").stat().st_size > 0, "Missing HTML entry point")
        report.mkdir()
        shutil.copytree(out / "html", report / "html")
        for filename in ("coverage.xml", "gcovr-summary.json"):
            shutil.copyfile(out / filename, report / filename)

    write_json(report / "recovery-provenance.json", {
        "report_status": "generated_offline",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "PCCL C++ host code",
        "workflow": {"repository": "t-head/pccl", "run_id": 45746419059, "run_attempt": 1,
                     "original_status": "failed", "report_gate_failure": "Extra IB/RoCE acceptance before merge",
                     "runtime_artifact_id": 15087017226, "diagnostics_artifact_id": 15087441668},
        "build": build,
        "case": {"name": original["case"], "passed": True, "topo": original["topo"],
                 "nodes": 2, "cards_per_node": 8},
        "coverage": metrics,
        "rank_evidence": {"ranks": 16, "ranks_per_worker": [8, 8], "counter_files": 1952},
        "input_sha256": DIGESTS,
        "audited_per_file_counts_sha256": EXPECTED_COUNTS,
        "report_tools": {"gcov": gcov, "gcov-tool": gcov_tool, "gcovr": gcovr},
        "validation": {"rebuilt_product": False, "activated_cluster": False, "reran_case": False},
        "performance_baseline_eligible": False,
    })
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    replay(Path(sys.argv[1]), Path(sys.argv[2]))
