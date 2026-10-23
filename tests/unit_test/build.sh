#!/bin/bash
#*************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#*************************************************************************

set -e

build_type="rel_sym"
keep=0
no_local_rw=0
use_mpi=0
rebuild=0
dry_run=0
verbose=0
ppu_option=""
warp_exec_mode="simd"
enable_ib=0
enable_numa=0
ppu_archs="ppu_10,ppu_15"
cpu_num=0

function usage
{
    script_name=$0
    echo "Usage:"
    echo "  $script_name [--use_mpi] [--rebuild] [--cpu_num <int>] [--dry_run]"
    echo "               [--build_type <str>] [--keep] [--verbose]"
    echo "               [--enable_ib] [--enable_numa] [--no_local_rw]"
    echo "               [--warp_exec_mode <str>] [--ppu_option <str>]"
    echo "               [--ppu_archs <str>]"
    echo ""
    echo "  Parameters:"
    echo "    use_mpi:         Specify if we want to build multi-process unit_test version."
    echo "    rebuild:         Make clean/remove build directory."
    echo "    cpu_num:         Specify the number of parallel jobs for make -j (default: use all CPUs)."
    echo "    dry_run:         Show what commands would be executed without actually running them."
    echo "    build_type:      Symbol control. rel_sym: release with some symbols (default)."
    echo "                     dbg: debug version."
    echo "    keep:            Retain all intermediate files generated during compilation."
    echo "    verbose:         Print detailed build output, including all executed commands."
    echo "    enable_ib:       Enable IB test build."
    echo "    enable_numa:     Enable support of numa.h on host (now only support in docker)."
    echo "    no_local_rw:     Build test with no local hbm r/w. By default, the option is off."
    echo "    warp_exec_mode:  Specify the warp execution mode (simd/simt, default: simd)."
    echo "    ppu_option:      For compiler optimize & bug fix."
    echo "    ppu_archs:       Specify the ppu archs, any combination of ppu_10,ppu_15."
}

function build_test
{
    if [[ $rebuild -eq 1 ]]; then
        make clean
    fi
    if [ $cpu_num -gt 0 ]; then
        build_cmd_str="make -j${cpu_num}"
    else
        build_cmd_str="make -j"
    fi
    if [[ $dry_run -eq 1 ]]; then
        build_cmd_str="${build_cmd_str} --dry-run"
    fi
    build_cmd_str="${build_cmd_str} PPU_ARCHS=${ppu_archs}"
    if [[ $warp_exec_mode == "simt" ]]; then
        build_cmd_str="${build_cmd_str} USE_SIMT=1"
    fi
    if [[ $use_mpi -eq 1 ]]; then
        build_cmd_str="${build_cmd_str} MPI=1"
    fi
    if [[ $verbose -eq 1 ]]; then
        build_cmd_str="${build_cmd_str} VERBOSE=1"
    fi
    if [[ $build_type == "dbg" ]]; then
        build_cmd_str="${build_cmd_str} DEBUG=1"
    fi
    if [[ $keep -eq 1 ]]; then
        build_cmd_str="${build_cmd_str} KEEP=1"
    fi
    if [ $enable_ib -eq 1 ]; then
        build_cmd_str+=" IB_TEST=1"
    fi
    if [ $no_local_rw -eq 1 ]; then
        build_cmd_str+=" NO_LOCAL_RW=1"
    fi
    if [ $enable_numa -eq 1 ]; then
        build_cmd_str+=" NUMA=1"
    fi
    if [ "$ppu_option" != "" ]; then
        build_cmd_str="${build_cmd_str} PPU_OPTION=${ppu_option}"
    fi

    # Build command executing
    $build_cmd_str
}

while [[ $# -ge 1 ]]
do
    key="$1"
    case $key in
        --build_type)
            build_type="$2"
            shift
            ;;
        --keep)
            keep=1
            ;;
        --no_local_rw)
            no_local_rw=1
            ;;
        --use_mpi)
            use_mpi=1
            ;;
        --rebuild)
            rebuild=1
            ;;
        --dry_run)
            dry_run=1
            ;;
        --enable_ib)
            enable_ib=1
            ;;
        --enable_numa)
            enable_numa=1
            ;;
        --verbose)
            verbose=1
            ;;
        --ppu_option)
            ppu_option="$2"
            shift
            ;;
        --warp_exec_mode)
            warp_exec_mode=$2
            shift
            ;;
        --ppu_archs)
            ppu_archs="$2"
            shift
            ;;
        --cpu_num)
            cpu_num="$2"
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $key"
            usage
            exit 1
            ;;
    esac
    shift
done

build_test
