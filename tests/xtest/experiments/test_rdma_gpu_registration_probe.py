"""Exercise registration ownership and diagnostics without a GPU or RDMA."""
import ctypes
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('gpu_probe', Path(__file__).with_name('rdma_gpu_registration_probe.py'))
gpu_probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_probe)


class RegistrationProbeTests(unittest.TestCase):
    def test_failed_registration_keeps_errno_and_never_deregisters(self):
        def failed(*args):
            ctypes.set_errno(14)
            return None
        lib = SimpleNamespace(ibv_reg_mr=Mock(side_effect=failed), ibv_dereg_mr=Mock())
        result = gpu_probe.register(lib, 1, 0x100000, 4096, 0xf)
        self.assertFalse(result['success'])
        self.assertEqual(result['errno'], 14)
        lib.ibv_dereg_mr.assert_not_called()

    def test_successful_registration_is_deregistered(self):
        lib = SimpleNamespace(ibv_reg_mr=Mock(return_value=2), ibv_dereg_mr=Mock(return_value=0))
        result = gpu_probe.register(lib, 1, 0x100000, 4096, 0xf)
        self.assertTrue(result['success'])
        self.assertEqual(result['errno'], 0)
        lib.ibv_dereg_mr.assert_called_once_with(2)

    def test_zero_iova_and_relaxed_ordering_are_preserved(self):
        lib = SimpleNamespace(ibv_reg_mr_iova2=Mock(return_value=3), ibv_dereg_mr=Mock(return_value=0))
        gpu_probe.register(lib, 1, 0x100000, 4915200, 0x10000f, 0)
        lib.ibv_reg_mr_iova2.assert_called_once_with(1, 0x100000, 4915200, 0, 0x10000f)
        lib.ibv_dereg_mr.assert_called_once_with(3)

    @staticmethod
    def sdk(count):
        def device_count(pointer):
            pointer._obj.value = count
            return 0
        def malloc(pointer, size):
            pointer._obj.value = 0x100000
            return 0
        return SimpleNamespace(hggcGetDeviceCount=Mock(side_effect=device_count),
                               hggcSetDevice=Mock(return_value=0), hggcMalloc=Mock(side_effect=malloc),
                               hggcFree=Mock(return_value=0))

    def test_allocated_device_is_mapped_to_the_sdk_visibility_variable(self):
        environment = {'NVIDIA_VISIBLE_DEVICES': '3'}
        self.assertEqual(gpu_probe.configure_allocated_device(environment), '3')
        self.assertEqual(environment['HGGC_VISIBLE_DEVICES'], '3')
        for value in ('', 'all', '0,1', '-1', '١'):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                gpu_probe.configure_allocated_device({'NVIDIA_VISIBLE_DEVICES': value})

    def test_unexpected_gpu_visibility_skips_allocation(self):
        sdk = self.sdk(8)
        lib = SimpleNamespace(ibv_fork_init=Mock(return_value=0))
        with patch.dict(gpu_probe.os.environ, {'NVIDIA_VISIBLE_DEVICES': '3'}), \
                patch.object(gpu_probe, 'load_verbs', return_value=lib), \
                patch.object(gpu_probe.ctypes, 'CDLL', return_value=sdk), \
                patch.object(gpu_probe, 'loaded_libraries', return_value=[]):
            report = gpu_probe.probe()
        self.assertIn('Expected one allocated GPU', report['error'])
        sdk.hggcMalloc.assert_not_called()
        sdk.hggcFree.assert_not_called()

    def test_gpu_allocation_is_freed_after_registration_exception(self):
        sdk = self.sdk(1)
        def devices(pointer):
            pointer._obj.value = 1
            return (ctypes.c_void_p * 1)(1)
        lib = SimpleNamespace(ibv_fork_init=Mock(return_value=0),
                              ibv_get_device_list=Mock(side_effect=devices),
                              ibv_free_device_list=Mock(), ibv_get_device_name=Mock(return_value=b'mlx5_bond_0'),
                              ibv_open_device=Mock(return_value=2), ibv_close_device=Mock(return_value=0),
                              ibv_alloc_pd=Mock(return_value=3), ibv_dealloc_pd=Mock(return_value=0))
        with patch.dict(gpu_probe.os.environ, {'NVIDIA_VISIBLE_DEVICES': '3'}), \
                patch.object(gpu_probe, 'load_verbs', return_value=lib), \
                patch.object(gpu_probe.ctypes, 'CDLL', return_value=sdk), \
                patch.object(gpu_probe, 'register', side_effect=RuntimeError('Registration failed')), \
                patch.object(gpu_probe, 'loaded_libraries', return_value=[]):
            report = gpu_probe.probe()
        self.assertEqual(report['error'], 'Registration failed')
        sdk.hggcFree.assert_called_once()
        lib.ibv_dealloc_pd.assert_called_once_with(3)
        lib.ibv_close_device.assert_called_once_with(2)
        lib.ibv_free_device_list.assert_called_once()

    def test_loaded_pccl_invalidates_independence(self):
        with patch.dict(gpu_probe.os.environ, {'NVIDIA_VISIBLE_DEVICES': '3'}), \
                patch.object(gpu_probe, 'load_verbs', side_effect=OSError('Missing RDMA')), \
                patch.object(gpu_probe, 'loaded_libraries', return_value=['/lib/libpccl.so']):
            report = gpu_probe.probe()
        self.assertFalse(report['independent_of_pccl'])
        self.assertIn('Missing RDMA', report['error'])


if __name__ == '__main__':
    unittest.main()
