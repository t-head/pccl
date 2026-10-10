#!/bin/bash
set -e

# Build script for pccl GitHub CI (ubuntu24_hggcrt13).
# The pccl compile command is delegated to pccl/.github/workflows/scripts/pccl_build.sh.
# pccl-tests is also built so that the run_*.sh test scripts can exercise the freshly
# built pccl libraries; its compile commands are inlined below (bash build.sh --use_mpi)
# because the cross-repo pccl-tests checkout (baseline branch) does not carry .github
# CI wrapper scripts.

# Point apt at internal-friendly mirrors (legacy sources.list + deb822 .sources).
# Some images ship with an internal-only T-head apt proxy that times out in the
# GitHub runner network; detect and replace it with a public mirror.
if grep -Rq "art.eng.t-head.cn" /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null; then
  echo "Replacing art.eng.t-head.cn apt sources with mirrors.aliyun.com"
  UBUNTU_CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME")
  rm -f /etc/apt/sources.list.d/*.sources
  printf '%s\n' \
    "deb http://mirrors.aliyun.com/ubuntu/ ${UBUNTU_CODENAME} main restricted universe multiverse" \
    "deb http://mirrors.aliyun.com/ubuntu/ ${UBUNTU_CODENAME}-updates main restricted universe multiverse" \
    "deb http://mirrors.aliyun.com/ubuntu/ ${UBUNTU_CODENAME}-backports main restricted universe multiverse" \
    "deb http://mirrors.aliyun.com/ubuntu/ ${UBUNTU_CODENAME}-security main restricted universe multiverse" \
    > /etc/apt/sources.list
else
  if [ -f /etc/apt/sources.list ]; then
    sed -i 's|archive.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list
    sed -i 's|security.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list
  fi
  for f in /etc/apt/sources.list.d/*.sources; do
    [ -f "$f" ] || continue
    sed -i 's|archive.ubuntu.com|mirrors.aliyun.com|g' "$f"
    sed -i 's|security.ubuntu.com|mirrors.aliyun.com|g' "$f"
  done
fi
apt update
apt install -y libboost-program-options-dev

# Validate required environment variables.
if [ -z "${source_home}" ]; then
    echo "Error: source_home environment variable is not set"
    exit 1
fi
if [ -z "${build_output}" ]; then
    echo "Error: build_output environment variable is not set"
    exit 1
fi

echo $PWD
echo "pwd has:"
ls
echo "copy test runner scripts to build output"
SCRIPT_DIR="${source_home}/pccl/.github/workflows/scripts"
PCCL_TESTS_DIR="${source_home}/pccl-tests"
mkdir -p ${build_output}
cp ${SCRIPT_DIR}/pr-ut-runner.py ${build_output}
cp ${SCRIPT_DIR}/casecmd.list ${build_output}
cp ${SCRIPT_DIR}/setup_test_env.sh ${build_output}
cp ${SCRIPT_DIR}/run_*.sh ${build_output}
cp ${PCCL_TESTS_DIR}/run_test.py ${build_output}
cp ${PCCL_TESTS_DIR}/run_test_config.py ${build_output}

echo "######################### $(date +%Y%m%d%H%M%S) Setup PPU SDK ##########################"
source /usr/local/PPU_SDK/envsetup.sh
which hgcc || true

echo "######################### $(date +%Y%m%d%H%M%S) Build pccl ##########################"
bash ${source_home}/pccl/.github/workflows/scripts/pccl_build.sh

echo "######################### $(date +%Y%m%d%H%M%S) Collect pccl artifacts ##########################"
mkdir -p ${build_output}/lib
find ${source_home}/pccl/build/lib -name "lib*.so*" ! -name "*.dbginfo" | xargs -i cp -arv {} ${build_output}/lib/
mkdir -p ${build_output}/include
cp -arv ${source_home}/pccl/build/include/* ${build_output}/include/
export PCCL_HOME=${source_home}/pccl/build

echo "######################### $(date +%Y%m%d%H%M%S) Build pccl-tests ##########################"
cd ${source_home}/pccl-tests
bash build.sh --use_mpi

echo "######################### $(date +%Y%m%d%H%M%S) Collect final output ##########################"
mv ${source_home}/pccl ${build_output}
mv ${source_home}/pccl-tests ${build_output}

echo "######################### $(date +%Y%m%d%H%M%S) Build finished ##########################"
