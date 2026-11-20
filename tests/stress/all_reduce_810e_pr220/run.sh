#!/usr/bin/env bash
# Build one fixed PR snapshot, then launch independent MPI runs.
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --help ]]; then
    cat <<'EOF'
Usage: bash run.sh

Run inside a Linux PPU SDK container with two allocated 810E devices.
Environment:
  PCCL_REVISION       Fixed PR #220 snapshot (default: 085dcd1fec4e...)
  PCCL_TESTS_REVISION Fixed benchmark snapshot (default: 394556512e87...)
  PCCL_SOURCE        Optional clean Git checkout at PCCL_REVISION
  PCCL_TESTS_SOURCE  Optional clean Git checkout at PCCL_TESTS_REVISION
  PPU_SDK           SDK directory (default: /usr/local/PPU_SDK)
  HPN_SDK_URL       Optional installer URL accepted by PR #220's installer
  MPI_INTERFACE     MPI control interface (default: eth0, as in the incident)
  RUNS              Independent process launches (default: 100)
  RUN_TIMEOUT       Seconds per run (default: 300)
  WORK_ROOT         New directory for sources, build logs and results

Builds PCCL once and pccl-tests once. Never rebuilds or retries a failed run.
Exit codes: 0 all passed; 1 failed/incomplete stress; 2 configuration error.
EOF
    exit 0
fi
[[ $# -eq 0 ]] || { echo 'Use --help for configuration.' >&2; exit 2; }
[[ $(uname -s) == Linux ]] || { echo 'Run this on Linux with the PPU SDK and two allocated 810E devices.' >&2; exit 2; }

export PCCL_REVISION=${PCCL_REVISION:-085dcd1fec4e4a96495cf1aa351f72c9f49ed8ba}
export PCCL_TESTS_REVISION=${PCCL_TESTS_REVISION:-394556512e87259e9ffa25225632c46b0a0bbcd5}
export MPI_INTERFACE=${MPI_INTERFACE:-eth0}
RUNS=${RUNS:-100}
RUN_TIMEOUT=${RUN_TIMEOUT:-300}
[[ $PCCL_REVISION =~ ^[0-9a-f]{40}$ && $PCCL_TESTS_REVISION =~ ^[0-9a-f]{40}$ ]] || {
    echo 'Source revisions must be full 40-character commit hashes.' >&2; exit 2;
}
python3 - "$RUNS" "$RUN_TIMEOUT" <<'PY'
import math
import sys
try:
    assert int(sys.argv[1]) > 0
    assert math.isfinite(float(sys.argv[2])) and float(sys.argv[2]) > 0
except (ValueError, AssertionError):
    raise SystemExit('RUNS must be positive; RUN_TIMEOUT must be positive and finite.')
PY
[[ -d /sys/class/net/$MPI_INTERFACE ]] || { echo "Missing MPI interface: $MPI_INTERFACE" >&2; exit 2; }

WORK_ROOT=${WORK_ROOT:-"$PWD/all-reduce-stress-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)-$$"}
mkdir -p -- "$(dirname -- "$WORK_ROOT")"
mkdir -- "$WORK_ROOT" || { echo 'WORK_ROOT must be a new directory; existing evidence is never overwritten.' >&2; exit 2; }
WORK_ROOT=$(cd -- "$WORK_ROOT" && pwd)
export WORK_ROOT
phase=checkout
finish() {
    local rc=$?
    printf '{"phase":"%s","exit_code":%s}\n' "$phase" "$rc" > "$WORK_ROOT/status.json"
}
trap finish EXIT
echo "Evidence directory: $WORK_ROOT"

checkout() {
    local repo=$1 revision=$2 provided=$3 target=$4
    if [[ -n $provided ]]; then
        target=$(cd -- "$provided" && pwd)
    else
        git init -q "$target"
        git -C "$target" remote add origin "https://github.com/t-head/$repo.git"
        git -C "$target" fetch --depth 1 origin "$revision"
        git -C "$target" checkout -q --detach FETCH_HEAD
    fi
    local actual
    actual=$(git -c safe.directory="$target" -C "$target" rev-parse HEAD)
    [[ $actual == "$revision" ]] || { echo "$repo revision mismatch: $actual != $revision" >&2; return 2; }
    git -c safe.directory="$target" -C "$target" diff --quiet HEAD -- || {
        echo "$repo contains tracked edits. Use a clean checkout." >&2; return 2;
    }
    # Untracked files may also affect a build; do not reuse pre-existing builds.
    [[ -z $(git -c safe.directory="$target" -C "$target" ls-files --others --exclude-standard) && ! -e $target/build ]] || {
        echo "$repo has untracked files or existing build output. Use a fresh checkout." >&2; return 2;
    }
}
checkout pccl "$PCCL_REVISION" "${PCCL_SOURCE:-}" "$WORK_ROOT/pccl" 2>&1 | tee "$WORK_ROOT/pccl-checkout.log"
# Functions run in a subshell in a pipeline; resolve the source explicitly.
PCCL_SOURCE=$(cd -- "${PCCL_SOURCE:-$WORK_ROOT/pccl}" && pwd)
checkout pccl-tests "$PCCL_TESTS_REVISION" "${PCCL_TESTS_SOURCE:-}" "$WORK_ROOT/pccl-tests" 2>&1 | tee "$WORK_ROOT/pccl-tests-checkout.log"
PCCL_TESTS_SOURCE=$(cd -- "${PCCL_TESTS_SOURCE:-$WORK_ROOT/pccl-tests}" && pwd)
export PCCL_SOURCE PCCL_TESTS_SOURCE

phase=sdk_setup
PPU_SDK=${PPU_SDK:-/usr/local/PPU_SDK}
[[ -f $PPU_SDK/envsetup.sh ]] || { echo "Missing $PPU_SDK/envsetup.sh" >&2; exit 2; }
# The vendor environment is not guaranteed to support nounset.
# shellcheck disable=SC1091
source "$PPU_SDK/envsetup.sh"
bash "$PCCL_SOURCE/.github/workflows/scripts/install_hpn_sdk.sh" 2>&1 | tee "$WORK_ROOT/hpn-install.log"
command -v mpirun >/dev/null || { echo 'mpirun is missing; install Open MPI in the SDK image.' >&2; exit 2; }

phase=pccl_build
printf 'cd %q && bash build.sh --use_sdk\n' "$PCCL_SOURCE" > "$WORK_ROOT/build.command.txt"
(cd -- "$PCCL_SOURCE" && bash build.sh --use_sdk) 2>&1 | tee "$WORK_ROOT/pccl-build.log"
export PCCL_HOME="$PCCL_SOURCE/build"
phase=benchmark_build
printf 'cd %q && bash build.sh --use_mpi\n' "$PCCL_TESTS_SOURCE" >> "$WORK_ROOT/build.command.txt"
(cd -- "$PCCL_TESTS_SOURCE" && bash build.sh --use_mpi) 2>&1 | tee "$WORK_ROOT/pccl-tests-build.log"

export LD_LIBRARY_PATH="$PCCL_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
BINARY="$PCCL_TESTS_SOURCE/build/all_reduce_perf"
LIBRARY="$PCCL_HOME/lib/libpccl.so"
[[ -x $BINARY && -f $LIBRARY ]] || { echo 'Expected all_reduce_perf or libpccl.so was not built.' >&2; exit 2; }
phase=runtime_check
ldd "$BINARY" | tee "$WORK_ROOT/ldd.log"
if grep -q 'not found' "$WORK_ROOT/ldd.log"; then
    echo 'The benchmark has unresolved shared libraries.' >&2; exit 2
fi
python3 - "$WORK_ROOT/ldd.log" "$LIBRARY" <<'PY'
from pathlib import Path
import re
import sys
text = Path(sys.argv[1]).read_text()
match = re.search(r'libpccl\.so\S*\s+=>\s+(\S+)', text)
if not match or Path(match[1]).resolve() != Path(sys.argv[2]).resolve():
    raise SystemExit('all_reduce_perf does not resolve to the freshly built libpccl.so.')
PY
{
    uname -a
    mpirun --version
    command -v hgcc >/dev/null && hgcc --version
    ulimit -a
    command -v ip >/dev/null && ip -4 -o addr show dev "$MPI_INTERFACE"
    command -v ibv_devinfo >/dev/null && ibv_devinfo
    command -v rdma >/dev/null && rdma link show
    true
} > "$WORK_ROOT/environment.log" 2>&1 || true

phase=stress
python3 "$SCRIPT_DIR/stress.py" --binary "$BINARY" --library "$LIBRARY" \
    --mpi-interface "$MPI_INTERFACE" --runs "$RUNS" --timeout "$RUN_TIMEOUT" \
    --output "$WORK_ROOT/results"
phase=complete
