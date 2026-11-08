# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

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
        cache = self.root / "cache" / self.sha
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
        cache = self.root / "cache" / self.sha
        cache.mkdir(parents=True)
        (cache / "partial").write_text("incomplete")
        artifact_cache.publish(self.download, cache, "standard", PCCL, RUNNER, self.sha)
        self.validate(cache)
        self.assertFalse((cache / "partial").exists())

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
