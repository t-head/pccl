#!/usr/bin/env python3
"""Record the actual RDMA user libraries and a bounded host-memory registration.

No driver/SDK installation, remote connection or performance measurement.
"""
import ctypes
import json
import os
from pathlib import Path


def loaded_libraries():
    names = ('libibverbs', 'libmlx5', 'libhggc', 'libhg_', 'libpccl')
    return sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                   if any(name in line for name in names) and '/' in line})


def probe():
    report = {'hostname': os.uname().nodename,
              'ld_library_path': os.environ.get('LD_LIBRARY_PATH', ''),
              'path': os.environ.get('PATH', ''), 'devices': [], 'host_registration': None}
    try:
        try:
            lib = ctypes.CDLL('libibverbs.so', use_errno=True)
        except OSError:
            lib = ctypes.CDLL('libibverbs.so.1', use_errno=True)
        signatures = {
            'ibv_fork_init': ([], ctypes.c_int),
            'ibv_get_device_list': ([ctypes.POINTER(ctypes.c_int)], ctypes.POINTER(ctypes.c_void_p)),
            'ibv_free_device_list': ([ctypes.POINTER(ctypes.c_void_p)], None),
            'ibv_get_device_name': ([ctypes.c_void_p], ctypes.c_char_p),
            'ibv_open_device': ([ctypes.c_void_p], ctypes.c_void_p),
            'ibv_close_device': ([ctypes.c_void_p], ctypes.c_int),
            'ibv_alloc_pd': ([ctypes.c_void_p], ctypes.c_void_p),
            'ibv_dealloc_pd': ([ctypes.c_void_p], ctypes.c_int),
            'ibv_reg_mr': ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int], ctypes.c_void_p),
            'ibv_dereg_mr': ([ctypes.c_void_p], ctypes.c_int),
        }
        for name, (args, result) in signatures.items():
            getattr(lib, name).argtypes, getattr(lib, name).restype = args, result
        report['fork_init'] = lib.ibv_fork_init()
        count = ctypes.c_int()
        devices = lib.ibv_get_device_list(ctypes.byref(count))
        if not devices:
            raise OSError(ctypes.get_errno(), 'ibv_get_device_list failed')
        try:
            report['devices'] = [lib.ibv_get_device_name(devices[i]).decode() for i in range(count.value)]
            if count.value:
                context = lib.ibv_open_device(devices[0])
                if not context:
                    raise OSError(ctypes.get_errno(), 'ibv_open_device failed')
                try:
                    pd = lib.ibv_alloc_pd(context)
                    if not pd:
                        raise OSError(ctypes.get_errno(), 'ibv_alloc_pd failed')
                    try:
                        buffer = ctypes.create_string_buffer(4096)
                        ctypes.set_errno(0)
                        mr = lib.ibv_reg_mr(pd, ctypes.addressof(buffer), len(buffer), 1)
                        report['host_registration'] = {'bytes': len(buffer), 'success': bool(mr),
                                                       'errno': 0 if mr else ctypes.get_errno()}
                        if mr:
                            report['host_registration']['deregistration_code'] = lib.ibv_dereg_mr(mr)
                    finally:
                        lib.ibv_dealloc_pd(pd)
                finally:
                    lib.ibv_close_device(context)
        finally:
            lib.ibv_free_device_list(devices)
    except (OSError, AttributeError) as error:
        report['error'] = str(error)
    try:
        report['loaded_libraries'] = loaded_libraries()
    except OSError as error:
        report['maps_error'] = str(error)
    return report


if __name__ == '__main__':
    print(json.dumps(probe(), indent=2))
