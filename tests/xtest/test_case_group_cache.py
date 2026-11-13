# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
from pathlib import Path
import stat
import tempfile
import unittest

import case_group_cache


def publish_worker(download, cache, digest, queue):
    try:
        case_group_cache.publish(Path(download), Path(cache), digest)
        queue.put(None)
    except Exception as error:
        queue.put(repr(error))


class CaseGroupCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.download = self.root / "download"
        (self.download / "g001" / "envs").mkdir(parents=True)
        (self.download / "groups.json").write_text('[{"id":"g001"}]\n')
        (self.download / "g001" / "standard.csv").write_text("plan,suite,name,check,cmd\n")
        (self.download / "g001" / "envs" / "op.env").write_text("VALUE=1\n")
        self.digest = case_group_cache.tree_sha256(self.download)
        self.cache = self.root / "cache" / "case-groups" / self.digest

    def tearDown(self):
        self.temp.cleanup()

    def test_tree_hash_covers_names_contents_additions_and_symlinks(self):
        self.assertEqual(self.digest, case_group_cache.tree_sha256(self.download))
        case_group_cache.validate(self.download, self.digest)
        (self.download / "g001" / "envs" / "op.env").write_text("VALUE=2\n")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            case_group_cache.validate(self.download, self.digest)
        (self.download / "g001" / "envs" / "op.env").write_text("VALUE=1\n")
        (self.download / "extra").write_text("unexpected")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            case_group_cache.validate(self.download, self.digest)
        (self.download / "extra").unlink()
        (self.download / "link").symlink_to(self.download / "groups.json")
        with self.assertRaisesRegex(ValueError, "symlinks"):
            case_group_cache.tree_sha256(self.download)

    def test_publish_replaces_invalid_cache_only_with_valid_download(self):
        case_group_cache.publish(self.download, self.cache, self.digest)
        case_group_cache.validate(self.cache, self.digest)
        self.assertFalse(self.cache.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH))
        (self.cache / "partial").write_text("invalid")
        case_group_cache.publish(self.download, self.cache, self.digest)
        case_group_cache.validate(self.cache, self.digest)
        self.assertFalse((self.cache / "partial").exists())
        quarantine = list(self.cache.parent.glob(f".{self.cache.name}.invalid-*"))
        self.assertEqual(len(quarantine), 1)
        self.assertEqual((quarantine[0] / "partial").read_text(), "invalid")
        original = (self.download / "groups.json").read_text()
        (self.download / "groups.json").write_text("corrupted")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            case_group_cache.publish(self.download, self.cache, self.digest)
        case_group_cache.validate(self.cache, self.digest)
        (self.download / "groups.json").write_text(original)

    def test_execution_stage_is_a_validated_read_only_private_copy(self):
        case_group_cache.publish(self.download, self.cache, self.digest)
        output = self.root / "worker" / "case-input"
        case_group_cache.stage(self.cache, output, self.digest)
        case_group_cache.validate(output, self.digest)
        self.assertFalse(output.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH))
        for path in output.rglob("*"):
            if path.is_file():
                self.assertFalse(path.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH))
        with self.assertRaisesRegex(ValueError, "already exists"):
            case_group_cache.stage(self.cache, output, self.digest)

    def test_concurrent_publishers_never_expose_partial_trees(self):
        context = multiprocessing.get_context("fork")
        for _ in range(5):
            queue = context.Queue()
            processes = [context.Process(target=publish_worker,
                                         args=(self.download, self.cache, self.digest, queue))
                         for _ in range(2)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(20)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual([queue.get(timeout=1) for _ in processes], [None, None])
            case_group_cache.validate(self.cache, self.digest)

    def test_symlinked_lock_is_rejected(self):
        self.cache.parent.mkdir(parents=True)
        external = self.root / "external"
        external.write_text("keep")
        (self.cache.parent / f".{self.cache.name}.lock").symlink_to(external)
        with self.assertRaises(OSError):
            case_group_cache.publish(self.download, self.cache, self.digest)
        self.assertEqual(external.read_text(), "keep")
        self.assertFalse(self.cache.exists())


if __name__ == "__main__":
    unittest.main()
