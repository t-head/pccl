# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / '.github/actions/download-artifact-retry/validate_sha256.py'
SPEC = importlib.util.spec_from_file_location('validate_download_sha256', SCRIPT)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class DownloadArtifactRetryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.download = self.root / 'download'
        self.download.mkdir()
        self.archive = self.download / 'output_standard.tar.gz'
        self.archive.write_bytes(b'artifact-content')
        self.sha256 = hashlib.sha256(self.archive.read_bytes()).hexdigest()

    def tearDown(self):
        self.temporary.cleanup()

    def test_valid_archive_and_empty_option_are_accepted(self):
        self.assertEqual(VALIDATOR.validate_download(
            self.root, 'download', 'output_standard.tar.gz', self.sha256), self.archive.resolve())
        self.assertIsNone(VALIDATOR.validate_download(self.root, 'download', '', ''))

    def test_hash_mismatch_is_retryable_failure(self):
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            VALIDATOR.validate_download(self.root, 'download', 'output_standard.tar.gz', '0' * 64)

    def test_invalid_options_fail_before_download(self):
        for options in (('download', 'output_standard.tar.gz', ''),
                        ('download', '', self.sha256),
                        ('download', 'output_standard.tar.gz', 'A' * 64),
                        ('', 'output_standard.tar.gz', self.sha256),
                        ('../download', 'output_standard.tar.gz', self.sha256),
                        ('download', '../output_standard.tar.gz', self.sha256)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                VALIDATOR.validate_options(*options)

    def test_symlinked_archive_is_rejected(self):
        self.archive.unlink()
        self.archive.symlink_to(self.root / 'elsewhere.tar.gz')
        with self.assertRaisesRegex(ValueError, 'regular file'):
            VALIDATOR.validate_download(self.root, 'download', 'output_standard.tar.gz', self.sha256)


if __name__ == '__main__':
    unittest.main()
