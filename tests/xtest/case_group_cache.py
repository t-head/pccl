#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate and atomically distribute automatic Case groups through a run-scoped NAS cache."""

import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time


HEX64 = re.compile(r"[0-9a-f]{64}\Z")
PUBLICATION_TIMEOUT = 300


def tree_sha256(directory):
    """Hash every relative filename and file byte, rejecting links and special files."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Case-group input must be a real directory")
    files = []
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("Case-group input cannot contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Case-group input contains a special file")
        files.append(path)
    if not files:
        raise ValueError("Case-group input is empty")
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(directory).as_posix()):
        name = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validate(directory, expected_sha256):
    if not HEX64.fullmatch(expected_sha256):
        raise ValueError("Invalid Case-group SHA-256")
    if tree_sha256(directory) != expected_sha256:
        raise ValueError("Case-group content SHA-256 mismatch")
    return Path(directory)


def _shared_permissions(directory):
    for path in [directory, *directory.rglob("*")]:
        if path.is_dir():
            path.chmod(0o777)
        else:
            path.chmod(path.stat().st_mode | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)


def publish(download, cache, expected_sha256):
    """Validate a downloaded tree before atomically replacing an invalid cache."""
    download, cache = Path(download), Path(cache)
    validate(download, expected_sha256)
    if cache.is_symlink():
        raise ValueError("Case-group cache cannot be a symlink")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.parent.chmod(0o777)
    lock_path = cache.parent / f".{cache.name}.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o666)
    with os.fdopen(descriptor, "r+b") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError("Case-group cache lock must be a regular file")
        os.fchmod(lock.fileno(), 0o666)
        deadline = time.monotonic() + PUBLICATION_TIMEOUT
        while True:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting to publish Case groups")
                time.sleep(min(0.1, remaining))
        temporary = Path(tempfile.mkdtemp(prefix=f".{cache.name}.download-", dir=cache.parent))
        staging = temporary / "tree"
        try:
            shutil.copytree(download, staging)
            validate(staging, expected_sha256)
            _shared_permissions(staging)
            if cache.exists():
                try:
                    validate(cache, expected_sha256)
                except (OSError, ValueError):
                    if cache.is_dir():
                        shutil.rmtree(cache)
                    else:
                        cache.unlink()
                else:
                    _shared_permissions(cache)
                    return cache
            os.replace(staging, cache)
            validate(cache, expected_sha256)
            return cache
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
            fcntl.flock(lock, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    fingerprint = commands.add_parser("fingerprint")
    fingerprint.add_argument("--directory", type=Path, required=True)
    for operation in ("validate", "publish"):
        command = commands.add_parser(operation)
        command.add_argument("--expected-sha256", required=True)
        command.add_argument("--directory" if operation == "validate" else "--download",
                             type=Path, required=True)
        if operation == "publish":
            command.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.operation == "fingerprint":
            print(tree_sha256(args.directory))
        elif args.operation == "validate":
            validate(args.directory, args.expected_sha256)
            print(args.expected_sha256)
        else:
            publish(args.download, args.cache, args.expected_sha256)
            print(args.expected_sha256)
        return 0
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
