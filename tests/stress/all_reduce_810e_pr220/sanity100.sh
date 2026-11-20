#!/usr/bin/env bash
# Build each PR sanity profile once, then reuse it for all 100 complete rounds.
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${WORK_ROOT:?}" "${PCCL_REVISION:?}" "${PCCL_TESTS_REVISION:?}"
mkdir -- "$WORK_ROOT"
phase=setup
finish() {
    local rc=$?
    printf '{"phase":"%s","exit_code":%s}\n' "$phase" "$rc" > "$WORK_ROOT/status.json"
}
trap finish EXIT
# shellcheck disable=SC1091
source /usr/local/PPU_SDK/envsetup.sh
for repo in pccl pccl-tests; do
    expected=$PCCL_REVISION
    [[ $repo != pccl-tests ]] || expected=$PCCL_TESTS_REVISION
    actual=$(git -c safe.directory="/workspace/source/$repo" -C "/workspace/source/$repo" rev-parse HEAD)
    [[ $actual == "$expected" ]] || { echo "Unexpected $repo revision: $actual" >&2; exit 2; }
done
bash /workspace/source/pccl/.github/workflows/scripts/install_hpn_sdk.sh 2>&1 | tee "$WORK_ROOT/hpn-install.log"
mkdir "$WORK_ROOT/notsm-source"
cp -a /workspace/source/pccl /workspace/source/pccl-tests "$WORK_ROOT/notsm-source/"

phase=standard_build
export source_home=/workspace/source
export build_output="$WORK_ROOT/output_standard"
bash /workspace/source/pccl/.github/workflows/scripts/build.sh 2>&1 | tee "$WORK_ROOT/standard-build.log"
phase=notsm_build
export source_home="$WORK_ROOT/notsm-source"
export build_output="$WORK_ROOT/output_notsm"
bash "$source_home/pccl/.github/workflows/scripts/notsm_build.sh" 2>&1 | tee "$WORK_ROOT/notsm-build.log"

phase=artifact_identity
python3 - "$WORK_ROOT" <<'PY'
import os
from pathlib import Path
import sys
import tarfile
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'output_standard/pccl/tests/xtest'))
import artifact_cache
for profile in ('standard', 'notsm'):
    archive = root / f'output_{profile}.tar.gz'
    with tarfile.open(archive, 'w:gz') as target:
        target.add(root / f'output_{profile}', arcname=f'output_{profile}')
    record = artifact_cache.create_record(archive, os.environ['PCCL_REVISION'],
                                         os.environ['PCCL_TESTS_REVISION'], profile)
    artifact_cache.write_record(root / f'output_{profile}.build.json', record)
    for name, expected in artifact_cache.RUNNER_HASHES.items():
        actual = artifact_cache.file_sha256(root / f'output_{profile}/pccl-tests' / name)
        if actual != expected:
            raise SystemExit(f'Unexpected benchmark runner content: {profile}/{name}')
PY
phase=python_environment
python3 -m venv "$WORK_ROOT/python-env"
"$WORK_ROOT/python-env/bin/python" -m pip install --disable-pip-version-check --no-index \
    --find-links /workspace/source/deps -r "$WORK_ROOT/output_standard/pccl/tests/xtest/requirements.txt"
"$WORK_ROOT/python-env/bin/python" -m pip check
unset PCCL_PATH
export HWLOC_COMPONENTS=-gl
export OMPI_MCA_orte_base_help_aggregate=0
export OMPI_MCA_orte_keep_fqdn_hostnames=1
export OMPI_MCA_oob_tcp_if_include="${MPI_INTERFACE:-eth0}"
export OMPI_MCA_btl_tcp_if_include="${MPI_INTERFACE:-eth0}"
export NCCL_SOCKET_IFNAME="${MPI_INTERFACE:-eth0}"
phase=sanity_rounds
"$WORK_ROOT/python-env/bin/python" -B "$SCRIPT_DIR/sanity_rounds.py" --root "$WORK_ROOT" \
    --output "$WORK_ROOT/results" --rounds 100 --mpi-interface "${MPI_INTERFACE:-eth0}"
phase=complete
