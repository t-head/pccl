"""Exercise the real lsmod/grep pipeline with a fake read-only lsmod command."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('check_peermem_loaded.sh')


class LoadedModuleTests(unittest.TestCase):
    def execute(self, text, code=0, expected='fixture-node'):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shim = root / 'lsmod'
            shim.write_text('#!/bin/sh\ncat <<\'TEXT\'\n' + text + '\nTEXT\nexit ' + str(code) + '\n')
            shim.chmod(0o755)
            environment = dict(os.environ, PATH=str(root)+':'+os.environ['PATH'],
                               NODE_NAME='fixture-node', TARGET_HOSTNAME=expected)
            process = subprocess.run(['bash', str(SCRIPT), str(root/'result')], env=environment,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return process, json.loads((root/'result/check.json').read_text())

    def test_loaded_entry(self):
        process, record = self.execute('Module Size Used by\nalixpu_peermem 16384 0')
        self.assertEqual(process.returncode, 0)
        self.assertTrue(record['module_loaded'])
        self.assertEqual(record['grep_exit_code'], 0)

    def test_missing_entry_is_not_a_command_failure(self):
        process, record = self.execute('Module Size Used by\nalixpu 12566528 1')
        self.assertEqual(process.returncode, 0)
        self.assertFalse(record['module_loaded'])
        self.assertEqual(record['grep_exit_code'], 1)
        self.assertEqual(record['stdout'], '')

    def test_command_error_is_not_reported_as_unloaded(self):
        process, record = self.execute('', 2)
        self.assertEqual(process.returncode, 1)
        self.assertFalse(record['inspection_completed'])
        self.assertIsNone(record['module_loaded'])

    def test_wrong_host_is_rejected(self):
        process, record = self.execute('alixpu_peermem 16384 0', expected='different-node')
        self.assertEqual(process.returncode, 1)
        self.assertIsNone(record['module_loaded'])

    def test_literal_grep_match_is_not_confused_with_exact_module_name(self):
        process, record = self.execute('alixpu_peermem_aux 16384 0')
        self.assertEqual(process.returncode, 0)
        self.assertFalse(record['module_loaded'])
        self.assertEqual(record['grep_exit_code'], 0)


if __name__ == '__main__':
    unittest.main()
