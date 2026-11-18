# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock

import remote_recovery


class RemoteRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.proc = Path(self.temporary.name)
        (self.proc / 'uptime').write_text('1000.0 0.0\n')

    def tearDown(self):
        self.temporary.cleanup()

    def process(self, pid, ppid, pgid, start, cmdline, executable='', state='S'):
        directory = self.proc / str(pid)
        directory.mkdir()
        fields = [str(pid), '(process)', state, str(ppid), str(pgid), str(pgid)] + ['0'] * 15 + [str(start)]
        (directory / 'stat').write_text(' '.join(fields) + '\n')
        (directory / 'cmdline').write_bytes(cmdline.encode() + b'\0')
        if executable:
            (directory / 'exe').symlink_to('/usr/bin/' + executable)

    def test_scan_only_returns_young_non_infrastructure_processes(self):
        self.process(100, 1, 100, 99500, 'sleep 300', 'sleep')
        self.process(101, 1, 101, 90000, 'sleep 300', 'sleep')
        self.process(102, 1, 102, 99500, '/bin/bash /run/cluster/master-session.sh', 'bash')
        self.process(103, 1, 103, 99500, '/usr/bin/ssh worker', 'ssh')
        self.process(104, 1, 104, 99500, 'sleep 300', 'sleep', state='Z')
        self.process(105, 1, 105, 99500, 'sleep 300', 'sleep')

        candidates = remote_recovery.scan_candidates(5, self.proc, protected_pids={105})

        self.assertEqual([process['pid'] for process in candidates], [100])

    def test_clean_worker_needs_no_signal(self):
        killer = mock.Mock()
        report = remote_recovery.recover(5, scanner=lambda _: [], killer=killer)
        self.assertTrue(report['safe'])
        killer.assert_not_called()

    def test_residual_process_group_is_terminated(self):
        candidate = {'pid': 100, 'ppid': 1, 'pgid': 100, 'cmdline': 'sleep 300'}
        killer = mock.Mock()
        reports = iter([[candidate], []])
        report = remote_recovery.recover(5, scanner=lambda _: next(reports), killer=killer)
        self.assertTrue(report['safe'])
        self.assertEqual(report['term_pgids'], [100])
        self.assertEqual(report['kill_pgids'], [])
        killer.assert_called_once_with(100, signal.SIGTERM)

    def test_stubborn_process_is_killed_and_remaining_process_fails(self):
        candidate = {'pid': 100, 'ppid': 1, 'pgid': 100, 'cmdline': 'sleep 300'}
        killer = mock.Mock()
        reports = iter([[candidate], [candidate], [candidate], []])
        with mock.patch.object(remote_recovery.time, 'monotonic', side_effect=[0, 0, 3]):
            report = remote_recovery.recover(
                5, scanner=lambda _: next(reports), sleeper=lambda _: None, killer=killer)
        self.assertTrue(report['safe'])
        self.assertEqual(report['kill_pgids'], [100])
        self.assertEqual(killer.call_args_list, [
            mock.call(100, signal.SIGTERM), mock.call(100, signal.SIGKILL)])
        reports = iter([[candidate], [candidate], [candidate], [candidate]])
        with mock.patch.object(remote_recovery.time, 'monotonic', side_effect=[0, 0, 3]):
            report = remote_recovery.recover(
                5, scanner=lambda _: next(reports), sleeper=lambda _: None, killer=killer)
        self.assertFalse(report['safe'])
        self.assertEqual(report['remaining'], [candidate])

    def test_main_emits_machine_readable_report(self):
        candidate = {'pid': 100, 'ppid': 1, 'pgid': 100, 'cmdline': 'sleep 300'}
        with mock.patch.object(remote_recovery, 'recover', return_value={
                'safe': True, 'before': [candidate], 'remaining': [], 'errors': []}):
            with mock.patch('sys.argv', ['remote_recovery.py', '--max-age', '5']):
                with mock.patch('builtins.print') as output:
                    remote_recovery.main()
        report = json.loads(output.call_args.args[0])
        self.assertTrue(report['safe'])
        self.assertEqual(report['before'], [candidate])


if __name__ == '__main__':
    unittest.main()
