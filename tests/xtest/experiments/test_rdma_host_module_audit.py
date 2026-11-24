"""Read-only boundary and host-path contracts for module inspection."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('module_audit', Path(__file__).with_name('rdma_host_module_audit.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class HostAuditTests(unittest.TestCase):
    def test_read_only_mount_is_required(self):
        with self.assertRaisesRegex(ValueError, 'read-only'):
            audit.audit(Path('/host'), 'kernel', '1 0 0:1 / /host rw - ext4 /dev/root rw')

    def test_nested_writable_mount_is_not_treated_as_readonly(self):
        mounts = '1 0 0:1 / /host ro - ext4 /dev/root rw\n2 1 0:2 / /host/usr rw - ext4 /dev/other rw'
        self.assertTrue(audit.readonly_mount('/host/lib/modules', mounts))
        self.assertFalse(audit.readonly_mount('/host/usr/src', mounts))
        self.assertFalse(audit.readonly_mount('/hostile/lib/modules', mounts))

    def test_mount_path_escaping(self):
        mounts = '1 0 0:1 / /host\\040root ro - ext4 /dev/root rw'
        self.assertTrue(audit.readonly_mount('/host root/lib/modules', mounts))

    def test_absolute_host_symlink_does_not_read_container_module(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'lib').mkdir()
            (root/'usr/lib/modules').mkdir(parents=True)
            (root/'lib/modules').symlink_to('/usr/lib/modules')
            self.assertEqual(audit.host_path(root, '/lib/modules'), root/'usr/lib/modules')

    def test_relative_host_symlink_is_resolved_inside_the_host(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'lib').symlink_to('usr/lib')
            self.assertEqual(audit.host_path(root, '/lib/modules/kernel'), root/'usr/lib/modules/kernel')

    def test_modinfo_preserves_dependency_version_and_parameter_requirements(self):
        output = 'name: alixpu_peermem\nversion: 2.3.1\nvermagic: kernel SMP\ndepends: alixpu,ib_core\nparm: peerdirect_support:int\n'
        with patch.object(audit.subprocess, 'run', return_value=SimpleNamespace(
                returncode=0, stdout=output, stderr='')) as run:
            info = audit.metadata(Path('/host/module.ko'))
        self.assertEqual(info['depends'], 'alixpu,ib_core')
        self.assertEqual(info['version'], '2.3.1')
        self.assertEqual(info['parameters'], ['peerdirect_support:int'])
        run.assert_called_once_with(['modinfo', '/host/module.ko'], capture_output=True, text=True, timeout=10)

    def test_current_kernel_module_inventory_uses_host_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            modules = root / 'lib/modules/test-kernel/extra'
            modules.mkdir(parents=True)
            (modules/'alixpu_peermem.ko').write_bytes(b'Fixture')
            original = Path.read_text
            def text(path, *args, **kwargs):
                if str(path) == '/proc/modules':
                    return 'alixpu 123 1 - Live 0x0\n'
                return original(path, *args, **kwargs)
            mountinfo = f'1 0 0:1 / {root} ro - ext4 /dev/root rw'
            with patch.object(Path, 'read_text', text), patch.object(audit, 'metadata', return_value={
                    'name': 'alixpu_peermem', 'vermagic': 'test-kernel SMP', 'version': '2.3.1'}):
                report = audit.audit(root, 'test-kernel', mountinfo)
            self.assertEqual(len(report['module_files']), 1)
            record = report['module_files'][0]
            self.assertTrue(record['for_running_kernel'])
            self.assertTrue(record['vermagic_matches_running_kernel'])
            self.assertIn('alixpu', report['loaded_modules'])
            self.assertNotIn('alixpu_peermem', report['loaded_modules'])

    def test_module_name_supports_compression_and_legacy_naming(self):
        for name in ('alixpu-peermem.ko', 'alixpu_peermem.ko.xz', 'alixpu_peer_mem.ko.zst'):
            self.assertTrue(audit.selected_module(audit.module_name(Path(name))))
        self.assertFalse(audit.selected_module('unrelated_module'))


if __name__ == '__main__':
    unittest.main()
