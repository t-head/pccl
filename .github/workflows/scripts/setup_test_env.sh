#!/bin/bash
set -e

# Common test environment setup for GitHub CI run_*.sh scripts.
# This script is sourced by each test runner; it assumes TEST_ROOT points to
# the build output directory (e.g. /workspace/source/output_hggcrt13).

python3 -m pip install filelock==3.20.0 || true
export NCCL_DEBUG=INFO
source /usr/local/PPU_SDK/envsetup.sh
export LD_LIBRARY_PATH=${TEST_ROOT}/lib:${LD_LIBRARY_PATH}
export PATH=${TEST_ROOT}/pccl-tests/build:${PATH}

echo "TEST_ROOT=${TEST_ROOT}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "PATH=${PATH}"
