#*************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#*************************************************************************

import os
import re
import shutil

PCCL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(PCCL_ROOT, 'build')
SRC_INCLUDE = os.path.join(BUILD_DIR, 'include')
DST_INCLUDE = os.path.join(BUILD_DIR, 'include_cuda')
REPLACEMENTS = [
    (r'__ppu_bfloat162', '__nv_bfloat162'),
    (r'__ppu_bfloat16', '__nv_bfloat16'),
    (r'__hg_cvt_fp8_to_halfraw', '__nv_cvt_fp8_to_halfraw'),
    (r'__hg_cvt_halfraw_to_fp8', '__nv_cvt_halfraw_to_fp8'),
    (r'__hg_fp8_storage_t', '__nv_fp8_storage_t'),
    (r'__hg_fp8_e4m3', '__nv_fp8_e4m3'),
    (r'__hg_fp8_e5m2', '__nv_fp8_e5m2'),
    (r'__HG_SATFINITE', '__NV_SATFINITE'),
    (r'__HG_E4M3', '__NV_E4M3'),
    (r'__HG_E5M2', '__NV_E5M2'),
    (r'(?<=[a-z])Hggc', 'Cuda'),
    (r'(?<=[a-z])Pccl', 'Nccl'),
    (r'hggc', 'cuda'),
    (r'HGGC', 'CUDA'),
    (r'Hggc', 'Cuda'),
    (r'pccl', 'nccl'),
    (r'PCCL', 'NCCL'),
    (r'Pccl', 'Nccl'),
    (r'\bHGmem', 'CUmem'),
    (r'\bHG_', 'CU_'),
]

POST_REPLACEMENTS = [
    (r'<cuda_v2/vector_types\.h>', '<vector_types.h>'),
]

LANE_SHIM = r"""
#if defined(__CUDACC__)
__device__ __forceinline__ int __ppu_read_laneid_shim() {
  int ret;
  asm("mov.u32 %0, %%laneid;" : "=r"(ret));
  return ret;
}
__device__ __forceinline__ unsigned int __ppu_read_lanemask_lt_shim() {
  unsigned int ret;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(ret));
  return ret;
}
#define __ppu_read_laneid() __ppu_read_laneid_shim()
#define __ppu_read_lanemask_lt() __ppu_read_lanemask_lt_shim()
#endif
"""

def apply_replacements(content):
    for pattern, repl in REPLACEMENTS:
        content = re.sub(pattern, repl, content)
    for pattern, repl in POST_REPLACEMENTS:
        content = re.sub(pattern, repl, content)
    return content


def inject_ptx_shims(filepath, content):
    if os.path.basename(filepath) == 'utility.h':
        guard = '#endif'
        idx = content.rfind(guard)
        if idx != -1:
            content = content[:idx] + LANE_SHIM + '\n' + content[idx:]
    return content


def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception:
        return

    content = inject_ptx_shims(filepath, content)
    content = apply_replacements(content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


# File renames: pccl-spelled filenames → nccl-spelled equivalents.
FILE_RENAMES = {
    'pccl.h': 'nccl.h',
}


def rename_files(root):
    """Rename known pccl-spelled files to their nccl equivalents."""
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname in FILE_RENAMES:
                src = os.path.join(dirpath, fname)
                dst = os.path.join(dirpath, FILE_RENAMES[fname])
                os.rename(src, dst)


def main():
    if os.path.exists(DST_INCLUDE):
        shutil.rmtree(DST_INCLUDE)
    shutil.copytree(SRC_INCLUDE, DST_INCLUDE)

    for dirpath, _, filenames in os.walk(DST_INCLUDE):
        for fname in filenames:
            if fname.endswith(('.h', '.cuh')):
                process_file(os.path.join(dirpath, fname))

    rename_files(DST_INCLUDE)


if __name__ == '__main__':
    main()
