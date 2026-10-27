# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.

import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '.github/scripts'))
from check_commit_messages import check_range, validate_message

AUTHOR = ('Example Contributor', 'contributor@example.org')
MESSAGE = 'ci: improve contribution checks\n\nReject malformed messages.\n\nRefs #12\n\nSigned-off-by: Example Contributor <contributor@example.org>\n'


class MessageTests(unittest.TestCase):
    def test_valid_message(self):
        self.assertEqual(validate_message(MESSAGE, AUTHOR), [])

    def test_long_title_is_not_rejected(self):
        self.assertEqual(validate_message('Improve ' * 15 + '\n' + MESSAGE.split('\n', 1)[1], AUTHOR), [])

    def test_missing_or_mismatched_signoff(self):
        for message in [MESSAGE.split('Signed-off-by:')[0], MESSAGE.replace('Example Contributor <', 'Other Author <')]:
            with self.subTest(message=message):
                self.assertTrue(validate_message(message, AUTHOR))

    def test_invalid_message_fields(self):
        for message in [MESSAGE.replace('Refs #12', ''), MESSAGE.replace('\n\n', '\n', 1), '\n' + MESSAGE]:
            with self.subTest(message=message):
                self.assertTrue(validate_message(message, AUTHOR))

    def test_reference_formats(self):
        for reference in ['Refs t-head/pccl#12', 'Fixes #12', 'Closes #12', 'https://github.com/t-head/pccl/issues/12']:
            with self.subTest(reference=reference):
                self.assertEqual(validate_message(MESSAGE.replace('Refs #12', reference), AUTHOR), [])

    def test_range_accepts_many_commits_and_checks_every_author(self):
        original = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                def git(*args):
                    return subprocess.check_output(['git', *args], text=True, stderr=subprocess.DEVNULL).strip()
                git('init', '-q')
                git('config', 'user.name', AUTHOR[0])
                git('config', 'user.email', AUTHOR[1])
                git('config', 'commit.gpgsign', 'false')
                git('commit', '--allow-empty', '-qm', 'base')
                base = git('rev-parse', 'HEAD')
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertTrue(check_range(base, base))
                    for count in range(1, 13):
                        git('commit', '--allow-empty', '-qm', MESSAGE)
                        if count in (1, 5, 6, 12):
                            self.assertFalse(check_range(base, 'HEAD'))
                    git('commit', '--allow-empty', '-qm', MESSAGE.replace('Refs #12', ''))
                    git('commit', '--allow-empty', '-qm', MESSAGE)
                    self.assertTrue(check_range(base, 'HEAD'))
            finally:
                os.chdir(original)


if __name__ == '__main__':
    unittest.main()
