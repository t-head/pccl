#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate optional content integrity for the download-artifact-retry action."""

import hashlib
import os
from pathlib import Path
import re
import sys


def _relative(value, label):
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a nonempty relative path without '..'")
    return path


def validate_options(download_path, sha256_file, expected_sha256):
    download = _relative(download_path, "Artifact download path")
    if bool(sha256_file) != bool(expected_sha256):
        raise ValueError("sha256-file and expected-sha256 must be provided together")
    if not expected_sha256:
        return download, None, None
    archive = _relative(sha256_file, "SHA-256 file")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected-sha256 must be a lowercase SHA-256 digest")
    return download, archive, expected_sha256


def validate_download(workspace, download_path, sha256_file, expected_sha256):
    download, archive, expected = validate_options(download_path, sha256_file, expected_sha256)
    if expected is None:
        return None
    workspace = Path(workspace).resolve()
    download = (workspace / download).resolve()
    download.relative_to(workspace)
    target = download / archive
    if target.is_symlink() or not target.is_file():
        raise ValueError("Downloaded SHA-256 target must be a regular file, not a symlink")
    target = target.resolve()
    target.relative_to(download)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"Downloaded archive SHA-256 mismatch: expected {expected}, got {actual}")
    print(f"Validated downloaded archive SHA-256: {actual}")
    return target


def main():
    try:
        options = validate_options(
            os.environ.get("DOWNLOAD_PATH", ""),
            os.environ.get("SHA256_FILE", ""),
            os.environ.get("EXPECTED_SHA256", ""))
        if os.environ.get("PREFLIGHT_ONLY") != "1" and options[2] is not None:
            validate_download(
                os.environ["GITHUB_WORKSPACE"],
                os.environ.get("DOWNLOAD_PATH", ""),
                os.environ.get("SHA256_FILE", ""),
                os.environ.get("EXPECTED_SHA256", ""))
    except (KeyError, OSError, ValueError) as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
