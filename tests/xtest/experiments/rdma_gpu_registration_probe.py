#!/usr/bin/env python3
"""Bounded GPU MR diagnosis through SDK + verbs only, with no PCCL library."""
import argparse
import ctypes
import json
import os
from pathlib import Path

from rdma_library_probe import loaded_libraries, load_verbs


def register(lib, pd, address, size, access, iova=None):
    ctypes.set_errno(0)
    if iova is None:
        mr = lib.ibv_reg_mr(pd, address, size, access)
    else:
        lib.ibv_reg_mr_iova2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                       ctypes.c_uint64, ctypes.c_uint]
        lib.ibv_reg_mr_iova2.restype = ctypes.c_void_p
        mr = lib.ibv_reg_mr_iova2(pd, address, size, iova, access)
    result = {'bytes': size, 'access': hex(access), 'iova': iova, 'success': bool(mr),
              'errno': 0 if mr else ctypes.get_errno()}
    if mr:
        result['deregistration_code'] = lib.ibv_dereg_mr(mr)
    return result


def probe():
    report = {'hostname': os.uname().nodename, 'node_rank': os.environ.get('NODE_RANK'),
              'visible_devices': os.environ.get('NVIDIA_VISIBLE_DEVICES'),
              'independent_of_pccl': None, 'devices': [], 'registrations': []}
    gpu = None
    allocated = False
    pointer = ctypes.c_void_p()
    try:
        lib = load_verbs()
        report['fork_init'] = lib.ibv_fork_init()
        gpu = ctypes.CDLL('libhggc_wrapper.so', use_errno=True)
        for name, args in {
                'hggcGetDeviceCount': [ctypes.POINTER(ctypes.c_int)],
                'hggcSetDevice': [ctypes.c_int],
                'hggcMalloc': [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
                'hggcFree': [ctypes.c_void_p]}.items():
            getattr(gpu, name).argtypes, getattr(gpu, name).restype = args, ctypes.c_int
        count = ctypes.c_int()
        code = gpu.hggcGetDeviceCount(ctypes.byref(count))
        report['gpu_device_count'] = count.value
        if code or count.value != 1:
            raise RuntimeError(f'Expected one allocated GPU: code={code}, count={count.value}')
        code = gpu.hggcSetDevice(0)
        if code:
            raise RuntimeError(f'hggcSetDevice failed: {code}')
        size = 4915200  # Same size as the failed PCCL proxy buffer registrations.
        code = gpu.hggcMalloc(ctypes.byref(pointer), size)
        report['gpu_allocation_code'] = code
        if code or not pointer.value:
            raise RuntimeError(f'hggcMalloc failed: {code}')
        allocated = True
        host = ctypes.create_string_buffer(4096)
        ndevices = ctypes.c_int()
        devices = lib.ibv_get_device_list(ctypes.byref(ndevices))
        if not devices:
            raise OSError(ctypes.get_errno(), 'ibv_get_device_list failed')
        try:
            for i in range(ndevices.value):
                name = lib.ibv_get_device_name(devices[i]).decode()
                report['devices'].append(name)
                context = lib.ibv_open_device(devices[i])
                if not context:
                    report['registrations'].append({'device': name, 'error': 'ibv_open_device failed',
                                                    'errno': ctypes.get_errno()})
                    continue
                try:
                    pd = lib.ibv_alloc_pd(context)
                    if not pd:
                        report['registrations'].append({'device': name, 'error': 'ibv_alloc_pd failed',
                                                        'errno': ctypes.get_errno()})
                        continue
                    try:
                        attempts = [('host', ctypes.addressof(host), len(host), 0xf, None),
                                    ('gpu_local_write', pointer.value, size, 1, None),
                                    ('gpu_strong_ordering', pointer.value, size, 0xf, None),
                                    ('gpu_relaxed_ordering', pointer.value, size, 0x10000f, pointer.value),
                                    ('gpu_zero_iova', pointer.value, size, 0x10000f, 0)]
                        for label, address, length, flags, iova in attempts:
                            result = register(lib, pd, address, length, flags, iova)
                            result.update(device=name, allocation=label)
                            report['registrations'].append(result)
                    finally:
                        lib.ibv_dealloc_pd(pd)
                finally:
                    lib.ibv_close_device(context)
        finally:
            lib.ibv_free_device_list(devices)
    except (OSError, AttributeError, RuntimeError) as error:
        report['error'] = str(error)
    finally:
        if gpu is not None and allocated:
            report['gpu_free_code'] = gpu.hggcFree(pointer)
    try:
        report['loaded_libraries'] = loaded_libraries()
        report['independent_of_pccl'] = not any('libpccl' in p for p in report['loaded_libraries'])
    except OSError as error:
        report['maps_error'] = str(error)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(probe(), indent=2) + '\n')
