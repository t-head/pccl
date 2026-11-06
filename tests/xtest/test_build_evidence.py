# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import build_evidence


class BuildEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / 'output.tar.gz'
        self.archive.write_bytes(b'synthetic build archive')
        self.inputs = {'pccl_revision': 'a' * 40, 'pccl_tests_revision': 'b' * 40}

    def record(self):
        return build_evidence.create(self.archive, self.inputs['pccl_revision'],
                                     self.inputs['pccl_tests_revision'], 'standard')

    def test_create_load_and_cli_bind_the_actual_archive_digest(self):
        record = self.record()
        self.assertEqual(record['artifact_sha256'], hashlib.sha256(self.archive.read_bytes()).hexdigest())
        self.assertEqual(build_evidence.validate(record, self.inputs, 'standard', self.archive), record)
        output = self.root / 'build.json'
        build_evidence.write_json(output, record)
        self.assertEqual(build_evidence.load(output, self.inputs, 'standard', self.archive), record)
        cli_output = self.root / 'cli.json'
        result = subprocess.run([sys.executable, '-S', '-B', str(Path(build_evidence.__file__)), 'create',
                                 '--archive', str(self.archive), '--pccl-revision', self.inputs['pccl_revision'],
                                 '--pccl-tests-revision', self.inputs['pccl_tests_revision'],
                                 '--profile', 'standard', '--output', str(cli_output)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(cli_output.read_text()), record)
        with self.assertRaises(ValueError):
            build_evidence.write_json(output, record)

    def test_revision_profile_archive_and_schema_tampering_fail_closed(self):
        record = self.record()
        mutations = []
        for field, value in (('pccl_revision', 'c' * 40), ('pccl_tests_revision', 'd' * 40),
                             ('build_profile', 'notsm'), ('artifact_sha256', '0' * 64),
                             ('schema_version', True), ('schema_version', 2)):
            changed = copy.deepcopy(record)
            changed[field] = value
            mutations.append(changed)
        changed = copy.deepcopy(record)
        changed['unexpected'] = 'not allowed'
        mutations.append(changed)
        for changed in mutations:
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    build_evidence.validate(changed, self.inputs, 'standard', self.archive)
        with self.assertRaises(ValueError):
            build_evidence.validate(record, self.inputs, 'notsm', self.archive)
        self.archive.write_bytes(b'stale archive with a replaced checksum')
        with self.assertRaisesRegex(ValueError, 'archive'):
            build_evidence.validate(record, self.inputs, 'standard', self.archive)


if __name__ == '__main__':
    unittest.main()
