#*************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#*************************************************************************

set -e

test_data_types="i8 u8 i32 u32 i64 u64 f16 f32 f64"
test_coll_ops="sendrecv all_reduce all_gather broadcast reduce reduce_scatter"

ARGPARSE_DESCRIPTION="pccl build script"      # this is optional
source $(dirname $0)/argparse.bash || exit 1
argparse "$@" <<EOF || exit 1
parser.add_argument('--build_type', default='rel_sym', type=str, help="Symbol control. rel: release version with some symbol. dbg: debug version. rel_no_sym: release version and strip symbol.")
parser.add_argument('--keep', default=False, action='store_true', help="Keep all intermediate files that are generated during internal compilation steps.")
parser.add_argument('--rebuild', default=False, action='store_true', help="Clean and rebuild.")
parser.add_argument('--verbose', default=False, action='store_true', help="Show more compile message.")
parser.add_argument('--trace', default=False, action='store_true', help="Enable TRACE logging.")
parser.add_argument('--test_dt', default='', type=str, help="Build kernels of specified data types. default: empty value, all data types.", choices=['', 'i8', 'u8', 'i32', 'u32', 'i64', 'u64', 'f16', 'f32', 'f64'])
parser.add_argument('--test_op', default='', type=str, help="Build specified kernels. default: empty value, all kernels.", choices=['', 'all_reduce', 'all_gather', 'broadcast', 'reduce', 'reduce_scatter', 'sendrecv'])
parser.add_argument('--use_dlto', default=False, action='store_true', help="Use dlto to do device code compiling.")
parser.add_argument('--kernel_loopback', default=False, action='store_true', help="Build pccl with kernel_loopback enabled for single PPU, only for profiling.")
parser.add_argument('--max_vregs', default=-1, type=int, help="Max vreg counts used on a warp level. Default: not set.")
parser.add_argument('--no_extra_warp', default=False, action='store_true', help="Build pccl Simple proto kernel without extra warp for sync.")
parser.add_argument('--ignore_fail', default=False, action='store_true', help="Don't stop at fail, and return 0 to CI.")
parser.add_argument('--cc', default=False, action='store_true', help="Build pccl for code coverage test.")
parser.add_argument('--warp_exec_mode', default='simd', type=str, help="Choose build mode: simt, simd or simd_simt_mixed.", choices=['simt', 'simd', 'simd_simt_mixed'])
parser.add_argument('--ppu_option', default="", type=str, help="For compiler optimize & bug fix.")
parser.add_argument('--use_sdk', default=True, action='store_true', help="Build pccl with external released ppusdk (default: enabled).")
parser.add_argument('--bypass_mode', default=0, type=int, help="Enable bypass operations, 0: no bypass, 1: load bypass, 2: store bypass, 3: both load/store bypass, 4: net bypass.", choices=[0, 1, 2, 3, 4])
parser.add_argument('--ppu_archs', default='ppu_10,ppu_15', type=str, help="Set ppu archs.")
parser.add_argument('--hggcrt_version', default=12, type=int, help="HGGC runtime version.")
parser.add_argument('--disable_tests', default=False, action='store_true', help="Disable tests.")
parser.add_argument('--no_tsm', default=False, action='store_true', help="Not using TSM to implement kernels.")
parser.add_argument('--no_hgtx', default=False, action='store_true', help="Disable hgtx when compiling.")
parser.add_argument('--no_ext_kernel', default=False, action='store_true', help="Not build ext kernel plugin.")
parser.add_argument('--cpu_num', default=0, type=int, help="Number of threads for make -j, default: 0 (auto).")
parser.add_argument('--hgcc_threads', default=1, type=int, help="Number of threads for hgcc parallel compilation, default: 1.")
parser.add_argument('--package_version', default='', type=str, help="Package version in the form a.b.c-xxxxxx. The a.b.c part must match version.mk. If not specified, version.mk version is used.")
parser.add_argument('--examples', default=False, action='store_true', help="After libpccl is built, also build all examples under docs/examples/ via docs/build_examples.sh. Requires PPU_SDK env (only --use_sdk toolchain is supported for examples).")
parser.add_argument('--doca_debug', default=False, action='store_true', help="Enable DOCA GPUNetIO debug mode: inserts doca_gpu_dev_verbs_wait() after every GIN RDMA operation for hang diagnosis.")
EOF

function build_dir
{
    build_dir=$1
    proj_type=$2
    if [[ $CPU_NUM -gt 0 ]]; then
        build_cmd_str="make -j${CPU_NUM}"
    else
        build_cmd_str="make -j"
    fi
    cd $build_dir
    if [[ $REBUILD ]]; then
        make clean || true
        rm -rf build
    fi
    if [[ $VERBOSE ]]; then
        build_cmd_str="${build_cmd_str} VERBOSE=1"
    fi
    if [[ $TRACE ]]; then
        build_cmd_str="${build_cmd_str} TRACE=1"
    fi
    if [[ $IGNORE_FAIL ]]; then
        build_cmd_str="${build_cmd_str} -i"
    fi
    if [[ $proj_type == "pccl_origin" ]]; then
        build_cmd_str="${build_cmd_str} default"
    fi
    if [[ $USE_DLTO ]]; then
        build_cmd_str="${build_cmd_str} NO_DLTO=0"
    fi
    if [[ $NO_TSM ]]; then
        build_cmd_str="${build_cmd_str} NO_TSM=1"
    fi

    if [[ $KERNEL_LOOPBACK ]]; then
        build_cmd_str="${build_cmd_str} KERNEL_LOOPBACK=1"
    fi

    if [[ $NO_EXTRA_WARP ]]; then
        build_cmd_str="${build_cmd_str} NO_EXTRA_WARP=1"
    fi

    if [[ $BUILD_TYPE == "rel_no_sym" ]]; then
        build_cmd_str="${build_cmd_str} REL_NO_SYM=1"
    elif [[ $BUILD_TYPE == "dbg" ]]; then
        build_cmd_str="${build_cmd_str} DEBUG=1"
    fi

    if [[ $BYPASS_MODE -ne 0 ]]; then
        if [[ $(( $BYPASS_MODE & 1 )) -ne 0 ]]; then
            echo "load operation bypas"
            build_cmd_str="${build_cmd_str} LOAD_BYPASS=1"
        fi
        if [[ $(( $BYPASS_MODE & 2 )) -ne 0 ]]; then
            echo "store operation bypas"
            build_cmd_str="${build_cmd_str} STORE_BYPASS=1"
        fi
        if [[ $(( $BYPASS_MODE & 4 )) -ne 0 ]]; then
            echo "network bypas"
            build_cmd_str="${build_cmd_str} NET_BYPASS=1"
        fi
    fi

    build_cmd_str="${build_cmd_str} HGGCRT_VERSION=${HGGCRT_VERSION} PPU_ARCHS=${PPU_ARCHS}"
    if [[ $WARP_EXEC_MODE == "simt" ]]; then
        build_cmd_str="${build_cmd_str} USE_SIMT=1"
    fi
    if [[ $USE_SDK ]]; then
        build_cmd_str="${build_cmd_str} USE_SDK=1"
    fi
    if [[ $NO_HOST_SPLITTER ]]; then
        export DEFAULT_ENABLE_SPLITTER=OFF
    fi
    if [[ $MAX_VREGS -ge 0 ]]; then
        build_cmd_str="${build_cmd_str} MAX_VREGS=${MAX_VREGS}"
    fi
    if [[ $KEEP ]]; then
        build_cmd_str="${build_cmd_str} KEEP=1"
    fi
    if [[ $TEST_DT != "" ]]; then
        build_cmd_str="${build_cmd_str} TEST_DT=${TEST_DT}"
    fi
    if [[ $TEST_OP != "" ]]; then
        build_cmd_str="${build_cmd_str} TEST_OP=${TEST_OP}"
    fi
    if [[ $CC ]]; then
        build_cmd_str="${build_cmd_str} CC=1"
    fi
    if [[ -n $PPU_OPTION ]]; then
        echo "using extra ppu option = ${PPU_OPTION}"
        # Replace space with ; which will be interpreted to cmake string list
        ppu_option_=$(sed "s/ /;/g" <<< ${PPU_OPTION})
        build_cmd_str="${build_cmd_str} PPU_OPTION=${ppu_option_}"
    fi
    if [[ $DISABLE_TESTS ]]; then
        build_cmd_str="${build_cmd_str} DISABLE_TESTS=1"
    fi
    if [[ ${NO_HGTX}  ]]; then
        build_cmd_str="${build_cmd_str} HGTX=0"
    fi
    if [[ ${NO_EXT_KERNEL}  ]]; then
        build_cmd_str="${build_cmd_str} NO_EXT_KERNEL=1"
    fi
    if [[ $DOCA_DEBUG ]]; then
        build_cmd_str="${build_cmd_str} DOCA_DEBUG=1"
    fi
    if [[ -n "$PACKAGE_VERSION" ]]; then
        build_cmd_str="${build_cmd_str} PACKAGE_VERSION=${PACKAGE_VERSION}"
    fi
    if [[ $HGCC_THREADS -gt 0 ]]; then
        build_cmd_str="${build_cmd_str} HGCC_THREADS=${HGCC_THREADS}"
    fi

    $build_cmd_str
}

if [[ -z "${PPU_SOURCE_HOME}" ]]; then
    PPU_SOURCE_HOME=$(dirname $(pwd))
    PCCL_ROOT=$(pwd)
else
    PCCL_ROOT=${PPU_SOURCE_HOME}/pccl
fi

# build pccl
echo "HGGCRT_VERSION=${HGGCRT_VERSION} PPU_ARCHS=${PPU_ARCHS}"
build_dir ${PCCL_ROOT} pccl_origin

if [[ $EXAMPLES ]]; then
    EXAMPLES_SCRIPT="${PCCL_ROOT}/docs/build_examples.sh"
    if [[ ! -x "${EXAMPLES_SCRIPT}" ]]; then
        echo "ERROR: ${EXAMPLES_SCRIPT} not found or not executable" >&2
        exit 1
    fi
    echo "=== Building examples under docs/examples/ ==="

    examples_args=(--ppu_archs "${PPU_ARCHS}" --hggcrt_version "${HGGCRT_VERSION}")
    if command -v mpicxx >/dev/null 2>&1; then
        examples_args+=(--use_mpi)
    else
        echo "WARNING: no MPI install found. Building non-MPI examples only." >&2
    fi
    [[ $REBUILD ]]     && examples_args+=(--rebuild)
    [[ $VERBOSE ]]     && examples_args+=(--verbose)
    [[ $IGNORE_FAIL ]] && examples_args+=(--ignore_fail)
    "${EXAMPLES_SCRIPT}" "${examples_args[@]}"
fi

echo "Begin to generate cuda header"
python3 ${PCCL_ROOT}/scripts/generate_cuda_header.py
