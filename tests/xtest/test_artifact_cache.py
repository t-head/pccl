# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import artifact_cache


PCCL = "a" * 40
RUNNER = "b" * 40


class ArtifactCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.download = self.root / "download"
        self.download.mkdir()
        self.archive = self.download / "output_standard.tar.gz"
        self.archive.write_bytes(b"small test archive\x00\xff")
        self.sha = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.record = {"schema_version": 1, "pccl_revision": PCCL, "pccl_tests_revision": RUNNER,
                       "build_profile": "standard", "artifact_sha256": self.sha}
        (self.download / "output_standard.build.json").write_text(json.dumps(self.record))
        (self.download / "output_standard.tar.gz.sha256").write_text(f"{self.sha}  /old/path/archive.tar.gz\n")

    def tearDown(self):
        self.temporary.cleanup()

    def validate(self, directory=None, expected=None, revision=PCCL, profile="standard"):
        return artifact_cache.validate(directory or self.download, profile, revision, RUNNER,
                                       expected or self.sha)

    def test_create_record_uses_the_same_identity_contract(self):
        output = self.root / "created.build.json"
        record = artifact_cache.create_record(self.archive, PCCL, RUNNER, "standard")
        artifact_cache.write_record(output, record)
        self.assertEqual(record["artifact_sha256"], self.sha)
        self.assertEqual(artifact_cache.read_record(output), record)

    def test_validate_and_publish_roundtrip(self):
        self.validate()
        cache = self.root / "cache" / "standard" / self.sha
        artifact_cache.publish(self.download, cache, "standard", PCCL, RUNNER, self.sha)
        self.validate(cache)
        # A second publisher may reuse an already complete identical cache.
        artifact_cache.publish(self.download, cache, "standard", PCCL, RUNNER, self.sha)
        self.validate(cache)

    def test_invalid_archive_record_and_profile_are_rejected(self):
        self.archive.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.validate()
        self.archive.write_bytes(b"different but declared")
        self.sha = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.record["artifact_sha256"] = self.sha
        (self.download / "output_standard.build.json").write_text(json.dumps(self.record))
        (self.download / "output_standard.tar.gz.sha256").write_text(f"{self.sha}  old\n")
        with self.assertRaisesRegex(ValueError, "revision"):
            self.validate(revision="c" * 40)
        for path in artifact_cache.paths(self.download, "standard"):
            (self.download / path.name.replace("standard", "notsm")).write_bytes(path.read_bytes())
        with self.assertRaisesRegex(ValueError, "profile"):
            artifact_cache.validate(self.download, "notsm", PCCL, RUNNER, self.sha)

    def test_symlinked_archive_is_never_executed(self):
        external = self.root / "external.tar.gz"
        self.archive.rename(external)
        self.archive.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.validate()

    def test_invalid_existing_cache_is_replaced_only_after_valid_download(self):
        cache = self.root / "cache" / "standard" / self.sha
        cache.mkdir(parents=True)
        (cache / "partial").write_text("incomplete")
        artifact_cache.publish(self.download, cache, "standard", PCCL, RUNNER, self.sha)
        self.validate(cache)
        self.assertFalse((cache / "partial").exists())

    def test_concurrent_identical_publisher_is_accepted_after_validation(self):
        cache = self.root / "cache" / "standard" / self.sha

        def competing_publish(source, destination):
            destination.mkdir()
            for path in artifact_cache.paths(self.download, "standard"):
                (destination / path.name).write_bytes(path.read_bytes())
            raise OSError("another publisher installed the cache first")

        with mock.patch("artifact_cache.os.replace", side_effect=competing_publish):
            artifact_cache.publish(self.download, cache, "standard", PCCL, RUNNER, self.sha)
        self.validate(cache)

    def test_shared_permissions_are_set_before_publication_even_with_restrictive_umask(self):
        cache = self.root / "cache" / "standard" / self.sha
        replace = os.replace

        def inspect_publication(source, destination):
            self.assertEqual(stat.S_IMODE(source.stat().st_mode), 0o755)
            for path in artifact_cache.paths(source, "standard"):
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)
            replace(source, destination)

        previous = os.umask(0o077)
        try:
            self.root.chmod(0o777)
            with mock.patch("artifact_cache.os.replace", side_effect=inspect_publication):
                artifact_cache.publish(self.download, cache, "standard", PCCL, RUNNER, self.sha)
        finally:
            os.umask(previous)
        self.assertEqual(stat.S_IMODE(self.root.stat().st_mode), 0o777)
        for directory in (self.root / 'cache', cache.parent, cache):
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o755)
        self.validate(cache)

    def test_reused_valid_cache_permissions_are_repaired(self):
        cache = self.root / "cache" / "standard" / self.sha
        artifact_cache.publish(self.download, cache, "standard", PCCL, RUNNER, self.sha)
        cache.chmod(0o700)
        cache.parent.chmod(0o700)
        self.root.joinpath('cache').chmod(0o700)
        for path in artifact_cache.paths(cache, "standard"):
            path.chmod(0o600)
        artifact_cache.publish(self.download, cache, "standard", PCCL, RUNNER, self.sha)
        for directory in (self.root / 'cache', cache.parent, cache):
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o755)
        for path in artifact_cache.paths(cache, "standard"):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)

    def test_publication_lock_has_a_bounded_wait_and_releases_after_errors(self):
        cache = self.root / "cache" / "standard" / self.sha
        cache.parent.mkdir(parents=True)
        with artifact_cache._publication_lock(cache):
            with self.assertRaisesRegex(TimeoutError, "Timed out"):
                with artifact_cache._publication_lock(cache, timeout=0.02):
                    self.fail("A second publisher acquired the held lock")
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with artifact_cache._publication_lock(cache):
                raise RuntimeError("injected failure")
        with artifact_cache._publication_lock(cache, timeout=0):
            self.assertTrue((cache.parent / f".{cache.name}.publish.lock").is_file())

    def test_symlinked_publication_lock_is_rejected(self):
        cache = self.root / "cache" / "standard" / self.sha
        cache.parent.mkdir(parents=True)
        external = self.root / "external-lock"
        external.write_text("keep")
        (cache.parent / f".{cache.name}.publish.lock").symlink_to(external)
        with self.assertRaises(OSError):
            artifact_cache.publish(self.download, cache, "standard", PCCL, RUNNER, self.sha)
        self.assertEqual(external.read_text(), "keep")
        self.assertFalse(cache.exists())

    def test_cache_must_use_the_scoped_profile_layout(self):
        cache = self.root / "unscoped" / self.sha
        with self.assertRaisesRegex(ValueError, "cache/PROFILE/SHA"):
            artifact_cache.publish(self.download, cache, "standard", PCCL, RUNNER, self.sha)

    def test_two_processes_replace_an_invalid_cache_once_under_the_same_lock(self):
        cache = self.root / "cache" / "standard" / self.sha
        cache.mkdir(parents=True)
        (cache / "partial").write_text("incomplete")
        script = '''
from contextlib import contextmanager
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
import artifact_cache
cache = Path(sys.argv[3])
original_lock = artifact_cache._publication_lock
@contextmanager
def observed_lock(path):
    Path(sys.argv[7]).touch()
    with original_lock(path):
        yield
artifact_cache._publication_lock = observed_lock
artifact_cache.publish(sys.argv[2], cache, "standard", sys.argv[4], sys.argv[5], sys.argv[6])
print(cache.stat().st_ino)
'''
        processes = []
        ready = [self.root / f"ready-{number}" for number in range(2)]
        try:
            with artifact_cache._publication_lock(cache):
                for marker in ready:
                    processes.append(subprocess.Popen(
                        [sys.executable, "-S", "-B", "-c", script,
                         str(Path(artifact_cache.__file__).parent), str(self.download), str(cache),
                         PCCL, RUNNER, self.sha, str(marker)],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
                deadline = time.monotonic() + 10
                while not all(marker.exists() for marker in ready):
                    if time.monotonic() >= deadline or any(process.poll() is not None for process in processes):
                        self.fail("Publishers did not reach the held interprocess lock")
                    time.sleep(0.01)
                self.assertTrue((cache / "partial").exists())
                self.assertTrue(all(process.poll() is None for process in processes))
            inodes = []
            for process in processes:
                output, error = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 0, error)
                inodes.append(int(output.strip()))
            self.assertEqual(inodes[0], inodes[1])
            self.assertEqual(cache.stat().st_ino, inodes[0])
            self.assertFalse((cache / "partial").exists())
            self.validate(cache)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=10)

    def test_cleanup_preserves_results_and_other_runs(self):
        root = self.root / "devops" / "pccl-artifact-cache-12345"
        for name in ("staging", "cache", "shared-standard", "results"):
            (root / name).mkdir(parents=True)
            (root / name / "keep-or-delete").write_text(name)
        other = self.root / "devops" / "pccl-artifact-cache-99999" / "cache"
        other.mkdir(parents=True)
        artifact_cache.cleanup(root, "12345")
        for name in ("staging", "cache", "shared-standard"):
            self.assertFalse((root / name).exists())
        self.assertTrue((root / "results" / "keep-or-delete").is_file())
        self.assertTrue(other.is_dir())
        with self.assertRaisesRegex(ValueError, "run-scoped"):
            artifact_cache.cleanup(root, "99999")


if __name__ == "__main__":
    unittest.main()
