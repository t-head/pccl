#!/usr/bin/env bash
#*************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#*************************************************************************

# Build all (or a single) example under docs/examples/.
#
# Companion to build.sh. build.sh builds libpccl; this script builds the
# example binaries that link against build/lib/libpccl.so. Mirrors build.sh's
# argparse style and forwards a curated subset of make variables that
# makefiles/{common,examples}.mk understand.
#
# Common usage (from the pccl repo root):
#   ./docs/build_examples.sh                       # build every example, mix 1.0+1.5
#   ./docs/build_examples.sh --use_mpi --rebuild   # MPI build (binary also runs single-process)
#   ./docs/build_examples.sh --example alltoall_gin
#   ./docs/build_examples.sh --example 6.2         # group.index shorthand
#
# Or invoked transparently by build.sh --examples (which forwards PCCL's
# PPU_ARCHS / HGGCRT_VERSION / REBUILD / VERBOSE / IGNORE_FAIL).
set -e

ARGPARSE_DESCRIPTION="pccl examples build script"
# argparse.bash lives at the pccl repo root; this script lives in docs/.
source "$(dirname "$0")/../argparse.bash" || exit 1
argparse "$@" <<EOF || exit 1
parser.add_argument('--rebuild', default=False, action='store_true', help="make clean each example before rebuilding")
parser.add_argument('--use_mpi', default=False, action='store_true', help="Build with MPI=1. The resulting binary still runs as a single process when invoked directly (OpenMPI singleton); mpirun -np N works as usual.")
parser.add_argument('--ppu_archs', default='ppu_10,ppu_15', type=str, help="Comma-separated hgcc target archs for fat binaries. Default 'ppu_10,ppu_15' mixes 1.0 and 1.5.")
parser.add_argument('--hggcrt_version', default=12, type=int, help="HGGC runtime version (12 or 13). Default 12 (aligned with build.sh).", choices=[12, 13])
parser.add_argument('--mpi_home', default='', type=str, help="MPI install prefix (used with --use_mpi)")
parser.add_argument('--debug', default=False, action='store_true', help="Build with -O0 -g")
parser.add_argument('--verbose', default=False, action='store_true', help="make VERBOSE=1")
parser.add_argument('--keep', default=False, action='store_true', help="Keep intermediate PTX/ISA files (KEEP=1) for instruction analysis")
parser.add_argument('--jobs', default=8, type=int, help="Parallel make jobs (-j)")
parser.add_argument('--ignore_fail', default=False, action='store_true', help="Continue past failures, summarize at the end (exit non-zero if any failed)")
parser.add_argument('--example', default='', type=str, help="Build a single example. Accepts a substring of the example name (e.g. 'alltoall_gin'), or 'group.index' shorthand (e.g. '6.2' for 06_device_api/02_alltoall_gin).")
parser.add_argument('--list', default=False, action='store_true', help="List discovered examples and exit")
EOF

PCCL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXAMPLES_DIR="$PCCL_ROOT/docs/examples"

# build.sh's --use_sdk path (PPU_SDK env) is the only supported toolchain
# here; the source-tree (PPU_SOURCE_HOME) path is not exercised by the
# examples flow and the SDK ships hgcc + headers + libs as one bundle.
if [[ -z "$PPU_SDK" || ! -x "$PPU_SDK/bin/hgcc" ]]; then
  echo "ERROR: PPU_SDK env var must point to an SDK with bin/hgcc (got '${PPU_SDK:-<unset>}')." >&2
  exit 1
fi

# Sanity: libpccl must exist; otherwise examples will fail at link time.
if [[ ! -f "$PCCL_ROOT/build/lib/libpccl.so" && ! -f "$PCCL_ROOT/build/lib/libpccl.so.2" ]]; then
  echo "WARNING: $PCCL_ROOT/build/lib/libpccl.so not found. Run ./build.sh first." >&2
fi

# PCCL_HOME points to the build output directory (include/ + lib/).
export PCCL_HOME="$PCCL_ROOT/build"

if [[ $USE_MPI ]]; then
  if [[ -n "$MPI_HOME" && -x "$MPI_HOME/bin/mpicxx" ]]; then
    : # explicit --mpi_home is valid, use it as-is
  elif [[ -n "$MPI_HOME" ]]; then
    # user set --mpi_home but it has no mpicxx, try system fallback
    sys_mpicxx="$(command -v mpicxx 2>/dev/null || true)"
    if [[ -n "$sys_mpicxx" ]]; then
      sys_root="$(cd "$(dirname "$sys_mpicxx")/.." && pwd)"
      echo "INFO: --mpi_home '$MPI_HOME' has no mpicxx; falling back to system MPI at $sys_root" >&2
      MPI_HOME="$sys_root"
    else
      echo "ERROR: --use_mpi but neither $MPI_HOME/bin/mpicxx nor a system mpicxx on PATH was found. Install MPI or pass --mpi_home <prefix>." >&2
      exit 1
    fi
  else
    # no --mpi_home set, use system MPI
    sys_mpicxx="$(command -v mpicxx 2>/dev/null || true)"
    if [[ -n "$sys_mpicxx" ]]; then
      sys_root="$(cd "$(dirname "$sys_mpicxx")/.." && pwd)"
      echo "INFO: Using system MPI at $sys_root" >&2
      MPI_HOME="$sys_root"
    else
      echo "ERROR: --use_mpi but no mpicxx found on PATH. Install MPI or pass --mpi_home <prefix>." >&2
      exit 1
    fi
  fi
fi

# Discover leaf examples: any docs/examples/**/Makefile whose directory has a
# main.cc or main.cu. Excludes docs/examples/Makefile and category Makefiles.
# When --use_mpi is not set, skip example dirs whose Makefile hard-codes
# `MPI = 1` (e.g. */*_mpi/) because their compile rule unconditionally uses
# $(MPICXX) and would fail with "mpicxx: command not found".
mapfile -t LEAF_DIRS < <(
  find "$EXAMPLES_DIR" -mindepth 2 -name Makefile -not -path '*/common/*' -printf '%h\n' \
    | sort -u \
    | while read -r d; do
        if [[ -f "$d/main.cc" || -f "$d/main.cu" ]]; then
          if [[ -z $USE_MPI ]] && grep -Eq '^[[:space:]]*MPI[[:space:]]*=[[:space:]]*1' "$d/Makefile"; then
            continue
          fi
          echo "$d"
        fi
      done
)

# --example resolver. Accepts:
#   - substring of leaf path or directory basename (e.g. "alltoall_gin")
#   - "group.index" shorthand (e.g. "6.2" => 06_device_api/02_alltoall_gin)
# Multiple matches => print and exit so the user can disambiguate.
resolve_example() {
  local query="$1"
  local matches=()

  # group.index shorthand: digits.digits
  if [[ "$query" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
    local grp idx
    grp=$(printf '%02d' "${BASH_REMATCH[1]}")
    idx=$(printf '%02d' "${BASH_REMATCH[2]}")
    for d in "${LEAF_DIRS[@]}"; do
      local rel=${d#$EXAMPLES_DIR/}
      if [[ "$rel" == ${grp}_*/${idx}_* ]]; then
        matches+=("$d")
      fi
    done
  else
    for d in "${LEAF_DIRS[@]}"; do
      local rel=${d#$EXAMPLES_DIR/}
      local base=${d##*/}
      if [[ "$rel" == *"$query"* || "$base" == *"$query"* ]]; then
        matches+=("$d")
      fi
    done
  fi

  if [[ ${#matches[@]} -eq 0 ]]; then
    echo "ERROR: --example '$query' matched no example. Use --list to see all." >&2
    exit 1
  fi
  if [[ ${#matches[@]} -gt 1 ]]; then
    echo "ERROR: --example '$query' is ambiguous, matched ${#matches[@]} examples:" >&2
    for m in "${matches[@]}"; do echo "  ${m#$EXAMPLES_DIR/}" >&2; done
    exit 1
  fi
  LEAF_DIRS=("${matches[0]}")
}

if [[ $LIST ]]; then
  echo "Discovered examples (${#LEAF_DIRS[@]}):"
  for d in "${LEAF_DIRS[@]}"; do echo "  ${d#$EXAMPLES_DIR/}"; done
  exit 0
fi

if [[ -n $EXAMPLE ]]; then
  resolve_example "$EXAMPLE"
fi

# make variables forwarded to every example's make invocation. USE_SDK=1 is
# always set: the source-tree toolchain path is not supported here.
# Do NOT pass CXXFLAGS on the command line — common.mk uses
# `CXXFLAGS := ... $(CXXFLAGS)` and a command-line override would discard
# common.mk's whole flag pipeline. examples.mk's `CXXFLAGS += -DMPI_SUPPORT`
# is triggered by MPI=1 and applies on top of common.mk's CXXFLAGS, which
# is what we want.
# Compute FILE_PREFIX_MAP (same logic as top-level Makefile:28-30)
CXX="${CXX:-g++}"
if $CXX -ffile-prefix-map=/tmp=. -x c /dev/null -c -o /dev/null 2>/dev/null; then
  FILE_PREFIX_MAP="-ffile-prefix-map"
else
  FILE_PREFIX_MAP="-fdebug-prefix-map"
fi
rm -f /dev/null.o 2>/dev/null  # cleanup if -o /dev/null didn't work as expected

MAKE_VARS=(
  "USE_SDK=1"
  "PPU_SDK=$PPU_SDK"
  "HGGCRT_VERSION=$HGGCRT_VERSION"
  "PPU_ARCHS=$PPU_ARCHS"
  "PCCL_SRC_ROOT=$PCCL_ROOT"
  "FILE_PREFIX_MAP=$FILE_PREFIX_MAP"
)
[[ $USE_MPI ]]  && MAKE_VARS+=("MPI=1" "MPI_HOME=$MPI_HOME")
[[ $DEBUG ]]    && MAKE_VARS+=("DEBUG=1")
[[ $VERBOSE ]]  && MAKE_VARS+=("VERBOSE=1")
[[ $KEEP ]]     && MAKE_VARS+=("KEEP=1")

echo "================================================================"
echo " pccl examples build"
echo "----------------------------------------------------------------"
echo "  pccl root    : $PCCL_ROOT"
echo "  rebuild      : ${REBUILD:-0}"
echo "  use_mpi      : ${USE_MPI:-0}    MPI_HOME=$MPI_HOME"
echo "  ppu_sdk      : $PPU_SDK"
echo "  ppu_archs    : $PPU_ARCHS"
echo "  hggcrt_ver   : $HGGCRT_VERSION"
echo "  debug        : ${DEBUG:-0}"
echo "  jobs         : -j$JOBS"
echo "  examples     : ${#LEAF_DIRS[@]} target(s)"
echo "================================================================"

fail_count=0
fail_dirs=()
ok_dirs=()

for d in "${LEAF_DIRS[@]}"; do
  rel=${d#$EXAMPLES_DIR/}
  echo
  echo "==> $rel"
  pushd "$d" > /dev/null

  if [[ $REBUILD ]]; then
    make "${MAKE_VARS[@]}" clean 2>/dev/null || true
  fi

  set +e
  make -j"$JOBS" "${MAKE_VARS[@]}"
  rc=$?
  set -e

  popd > /dev/null

  if [[ $rc -ne 0 ]]; then
    fail_count=$((fail_count+1))
    fail_dirs+=("$rel")
    if [[ ! $IGNORE_FAIL ]]; then
      echo
      echo "FAILED: $rel  (use --ignore_fail to continue past failures)"
      exit $rc
    fi
  else
    ok_dirs+=("$rel")
  fi
done

echo
echo "================================================================"
if [[ $fail_count -eq 0 ]]; then
  echo " OK: ${#ok_dirs[@]} / ${#LEAF_DIRS[@]} examples built"
  exit 0
else
  echo " FAIL: $fail_count / ${#LEAF_DIRS[@]} examples failed:"
  for f in "${fail_dirs[@]}"; do echo "   - $f"; done
  exit 1
fi
