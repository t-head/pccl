# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import remote_recovery


CASE_ID = 'a' * 32
WRAPPER = Path(__file__).resolve().parent / 'cluster-bin' / 'cluster_case_ssh.sh'


class RemoteRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cgroup = self.root / f'pccl-recovery-{CASE_ID}'

    def tearDown(self):
        self.temporary.cleanup()

    def remove_cgroup(self, path):
        for child in path.iterdir():
            child.unlink()
        os.rmdir(path)

    def test_missing_cgroup_is_clean(self):
        report = remote_recovery.recover(CASE_ID, self.root)
        self.assertTrue(report['safe'])
        self.assertEqual(report['before'], [])

    def test_empty_cgroup_is_removed(self):
        self.cgroup.mkdir()
        (self.cgroup / 'cgroup.procs').write_text('')
        with mock.patch.object(Path, 'rmdir', side_effect=lambda: self.remove_cgroup(self.cgroup)):
            report = remote_recovery.recover(CASE_ID, self.root)
        self.assertTrue(report['safe'])
        self.assertFalse(self.cgroup.exists())

    def test_live_cgroup_is_killed_and_removed(self):
        self.cgroup.mkdir()
        (self.cgroup / 'cgroup.procs').write_text('123\n')
        with mock.patch.object(remote_recovery, '_pids', side_effect=[[123], [], []]) as pids:
            with mock.patch.object(remote_recovery, '_kill') as kill:
                with mock.patch.object(Path, 'rmdir', side_effect=lambda: self.remove_cgroup(self.cgroup)):
                    report = remote_recovery.recover(CASE_ID, self.root, sleeper=lambda _: None)
        self.assertTrue(report['safe'])
        self.assertEqual(report['before'], [123])
        self.assertEqual(report['remaining'], [])
        kill.assert_called_once_with(self.cgroup)
        self.assertEqual(pids.call_count, 3)
        self.assertFalse(self.cgroup.exists())

    def test_live_cgroup_that_cannot_be_emptied_is_unsafe(self):
        self.cgroup.mkdir()
        (self.cgroup / 'cgroup.procs').write_text('123\n')
        with mock.patch.object(remote_recovery, '_pids', side_effect=[[123], [123], [123]]):
            with mock.patch.object(remote_recovery, '_kill'):
                with mock.patch.object(remote_recovery.time, 'monotonic', side_effect=[0, 3]):
                    report = remote_recovery.recover(CASE_ID, self.root, sleeper=lambda _: None)
        self.assertFalse(report['safe'])
        self.assertEqual(report['remaining'], [123])
        self.assertTrue(self.cgroup.exists())

    def test_kill_failure_is_unsafe(self):
        self.cgroup.mkdir()
        (self.cgroup / 'cgroup.procs').write_text('123\n')
        with mock.patch.object(remote_recovery, '_kill', side_effect=OSError('read-only cgroup')):
            report = remote_recovery.recover(CASE_ID, self.root)
        self.assertFalse(report['safe'])
        self.assertIn('read-only cgroup', report['errors'][0])

    def test_invalid_case_id_is_rejected(self):
        report = remote_recovery.recover('invalid', self.root)
        self.assertFalse(report['safe'])
        self.assertIn('Invalid recovery Case ID', report['errors'])

    def test_wrapper_installs_remote_command_in_case_cgroup(self):
        real_ssh = self.root / 'real-ssh'
        calls = self.root / 'calls'
        real_ssh.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CALLS_FILE"\n')
        real_ssh.chmod(0o755)
        wrapper = self.root / 'cluster_case_ssh.sh'
        wrapper.write_text(WRAPPER.read_text().replace(
            'REAL_SSH=/run/cluster/bin/cluster-ssh', f'REAL_SSH={real_ssh}'))
        wrapper.chmod(0o755)
        environment = dict(os.environ, PCCL_CASE_ID=CASE_ID, CALLS_FILE=str(calls))
        result = subprocess.run([str(wrapper), '/x/cluster-ssh', 'worker.example', 'sleep', '1'],
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        command = calls.read_text()
        self.assertIn(f'pccl-recovery-{CASE_ID}', command)
        self.assertIn('cgroup.procs', command)
        self.assertIn("'sleep' '1'", command)
        environment['PCCL_CASE_ID'] = 'invalid'
        result = subprocess.run([str(wrapper), '/x/cluster-ssh', 'worker.example', 'sleep', '1'],
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 64)
        self.assertEqual(calls.read_text(), command)


if __name__ == '__main__':
    unittest.main()
