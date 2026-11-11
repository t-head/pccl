#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate GitHub build artifacts and manage a run-scoped NAS cache."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time


HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
FIELDS = {"schema_version", "pccl_revision", "pccl_tests_revision", "build_profile", "artifact_sha256"}
PCCL_TESTS_REVISION = "394556512e87259e9ffa25225632c46b0a0bbcd5"
RUNNER_HASHES = {
    "run_test.py": "fdbdf7623703157b49b9605491fb1cb602fe50ec931ae6763c2645bef7c6b0fc",
    "run_test_config.py": "d58194c5c210ebaabe1ed42d7c2ced924c5dde22081285c22516e1ccdf611b1a",
}
PUBLICATION_TIMEOUT = 300


def file_sha256(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Build archive must be a regular file, not a symlink")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_record(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Build record must be a regular file, not a symlink")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate build record key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Invalid build record constant")

    record = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                        parse_constant=invalid)
    if not isinstance(record, dict) or set(record) != FIELDS:
        raise ValueError("Unexpected build record fields")
    return record


def create_record(archive, pccl_revision, pccl_tests_revision, profile):
    if profile not in ("standard", "notsm"):
        raise ValueError("Unsupported build profile")
    if not HEX40.fullmatch(pccl_revision) or not HEX40.fullmatch(pccl_tests_revision):
        raise ValueError("Invalid source revision")
    return {"schema_version": 1, "pccl_revision": pccl_revision,
            "pccl_tests_revision": pccl_tests_revision, "build_profile": profile,
            "artifact_sha256": file_sha256(archive)}


def write_record(path, record):
    if not isinstance(record, dict) or set(record) != FIELDS:
        raise ValueError("Unexpected build record fields")
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def paths(directory, profile):
    if profile not in ("standard", "notsm"):
        raise ValueError("Unsupported build profile")
    directory = Path(directory)
    return (directory / f"output_{profile}.tar.gz",
            directory / f"output_{profile}.build.json",
            directory / f"output_{profile}.tar.gz.sha256")


def validate(directory, profile, pccl_revision, pccl_tests_revision, expected_sha):
    if not HEX40.fullmatch(pccl_revision) or not HEX40.fullmatch(pccl_tests_revision):
        raise ValueError("Invalid source revision")
    if not HEX64.fullmatch(expected_sha):
        raise ValueError("Invalid expected archive SHA-256")
    archive, record_path, checksum_path = paths(directory, profile)
    record = read_record(record_path)
    if (type(record["schema_version"]) is not int or record["schema_version"] != 1
            or record["pccl_revision"] != pccl_revision
            or record["pccl_tests_revision"] != pccl_tests_revision
            or record["build_profile"] != profile or record["artifact_sha256"] != expected_sha):
        raise ValueError("Build record differs from the requested revision, profile or archive")
    if file_sha256(archive) != expected_sha:
        raise ValueError("Build archive SHA-256 mismatch")
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise ValueError("Build checksum must be a regular file, not a symlink")
    checksum = checksum_path.read_text(encoding="utf-8").split()
    if not checksum or checksum[0] != expected_sha:
        raise ValueError("Build checksum sidecar mismatch")
    return archive


@contextmanager
def _publication_lock(cache, timeout=PUBLICATION_TIMEOUT):
    # Keep the lock outside the replaceable directory and never unlink it:
    # replacing its inode would let waiting publishers acquire different locks.
    cache = Path(cache)
    lock_path = cache.parent / f".{cache.name}.publish.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o666)
    with os.fdopen(descriptor, "r+b") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError("Cache publication lock must be a regular file")
        os.fchmod(lock.fileno(), 0o666)
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting to publish cache {cache.name}")
                time.sleep(min(0.1, remaining))
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _shared_permissions(directory, profile):
    # mkdtemp uses 0700; hardware workers may have a different NAS identity.
    for path in paths(directory, profile):
        path.chmod(0o644)
    Path(directory).chmod(0o755)


def _shared_parent_permissions(cache, profile):
    """Make the cache/profile ancestors searchable without changing the run root."""
    cache = Path(cache)
    cache_root = cache.parent.parent
    if cache.parent.name != profile or cache_root.name != "cache":
        raise ValueError("Cache path must end in cache/PROFILE/SHA")
    for directory in (cache_root, cache.parent):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Cache parent directories must be real directories, not symlinks")
        directory.chmod(0o755)


def publish(download, cache, profile, pccl_revision, pccl_tests_revision, expected_sha):
    """Validate a downloaded artifact before atomically publishing one cache directory."""
    download, cache = Path(download), Path(cache)
    validate(download, profile, pccl_revision, pccl_tests_revision, expected_sha)
    cache.parent.mkdir(parents=True, exist_ok=True)
    _shared_parent_permissions(cache, profile)
    temporary = Path(tempfile.mkdtemp(prefix=f".{cache.name}.download-", dir=cache.parent))
    try:
        for path in paths(download, profile):
            shutil.copyfile(path, temporary / path.name)
        validate(temporary, profile, pccl_revision, pccl_tests_revision, expected_sha)
        _shared_permissions(temporary, profile)
        with _publication_lock(cache):
            if cache.exists():
                try:
                    validate(cache, profile, pccl_revision, pccl_tests_revision, expected_sha)
                except (OSError, ValueError, json.JSONDecodeError):
                    # Revalidate under the same lock that protects deletion and
                    # publication, so another publisher's valid cache survives.
                    shutil.rmtree(cache)
                else:
                    _shared_permissions(cache, profile)
                    return cache
            try:
                os.replace(temporary, cache)
            except OSError:
                # Preserve compatibility with an identical cache installed by
                # an older publisher, but never accept an unverified winner.
                validate(cache, profile, pccl_revision, pccl_tests_revision, expected_sha)
                _shared_permissions(cache, profile)
                return cache
            validate(cache, profile, pccl_revision, pccl_tests_revision, expected_sha)
            return cache
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def cleanup(root, run_id):
    """Remove only build artifacts owned by this run; preserve all test results."""
    if not run_id.isdigit():
        raise ValueError("Run ID must be numeric")
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Cleanup refuses a symlinked run root")
    if root.name != f"pccl-artifact-cache-{run_id}" or root.parent.name != "devops":
        raise ValueError("Cleanup root is not the expected run-scoped artifact directory")
    for name in ("staging", "cache", "shared-standard"):
        path = root / name
        if path.is_symlink():
            raise ValueError("Cleanup refuses a symlinked artifact directory")
        if path.exists():
            shutil.rmtree(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--archive", type=Path, required=True)
    create_parser.add_argument("--output", type=Path, required=True)
    create_parser.add_argument("--profile", choices=("standard", "notsm"), required=True)
    create_parser.add_argument("--pccl-revision", required=True)
    create_parser.add_argument("--pccl-tests-revision", required=True)
    for operation in ("validate", "publish"):
        command = subparsers.add_parser(operation)
        command.add_argument("--profile", choices=("standard", "notsm"), required=True)
        command.add_argument("--pccl-revision", required=True)
        command.add_argument("--pccl-tests-revision", required=True)
        command.add_argument("--expected-sha256", required=True)
        command.add_argument("--directory" if operation == "validate" else "--download", type=Path, required=True)
        if operation == "publish":
            command.add_argument("--cache", type=Path, required=True)
    cleanup_parser = subparsers.add_parser("cleanup")
    cleanup_parser.add_argument("--root", type=Path, required=True)
    cleanup_parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        if args.operation == "cleanup":
            cleanup(args.root, args.run_id)
            return 0
        if args.operation == "create":
            record = create_record(args.archive, args.pccl_revision, args.pccl_tests_revision, args.profile)
            write_record(args.output, record)
            print(json.dumps(record, sort_keys=True))
            return 0
        directory = args.directory if args.operation == "validate" else args.cache
        if args.operation == "publish":
            publish(args.download, args.cache, args.profile, args.pccl_revision,
                    args.pccl_tests_revision, args.expected_sha256)
        else:
            validate(directory, args.profile, args.pccl_revision, args.pccl_tests_revision,
                     args.expected_sha256)
        print(args.expected_sha256)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
