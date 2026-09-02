#!/bin/bash
set -e
pip3 install --upgrade meson ninja
# 检查必要的环境变量
if [ -z "${source_home}" ]; then
    echo "Error: source_home environment variable is not set"
    exit 1
fi
if [ -z "${build_output}" ]; then
    echo "Error: build_output environment variable is not set"
    exit 1
fi
echo "11.162.25.191 art-pub.eng.t-head.cn" >> /etc/hosts
echo $PWD
echo "pwd has:"
ls

echo "######################### $(date +%Y%m%d%H%M%S) Download Prebuilt PPU_SDK ##########################"
wget https://art-pub.eng.t-head.cn/artifactory/apackage/daily/ppu/latest/PPU_SDK/ppu_1.0.0_Ubuntu2404_v13.run
ls
bash ppu_1.0.0_Ubuntu2404_v13.run --prefix=${source_home} --silent

echo "######################### $(date +%Y%m%d%H%M%S) Assemble comm-internal-testbox ##########################"
cp -r ${source_home}/pccl-tests/* ${source_home}/comm-internal-testbox/tests/pccl-tests
cp -r ${source_home}/comm-toolbox/* ${source_home}/comm-internal-testbox/comm-toolbox

echo "######################### $(date +%Y%m%d%H%M%S) Build pccl ##########################"
source ${source_home}/PPU_SDK/envsetup.sh
cd ${source_home}/pccl
bash build.sh --use_sdk

echo "######################### $(date +%Y%m%d%H%M%S) Collect pccl artifacts ##########################"
mkdir -p ${build_output}/lib
find ${source_home}/pccl/build/lib -name "lib*.so*" ! -name "*.dbginfo" | xargs -i cp -arv {} ${build_output}/lib/
cp -arv ${source_home}/pccl/build/include/* ${build_output}/include/
export PCCL_HOME=${source_home}/pccl/build

echo "######################### $(date +%Y%m%d%H%M%S) Build pccl tests (graph search) ##########################"
cd ${source_home}/pccl/tests
bash build.sh

echo "######################### $(date +%Y%m%d%H%M%S) Build comm-internal-testbox ##########################"
cd ${source_home}/comm-internal-testbox
bash build.sh --common_args "--use_mpi" --exclude sailbandwidth,composite_test
bash build.sh --include composite_test

echo "######################### $(date +%Y%m%d%H%M%S) Collect final output ##########################"
mv ${source_home}/comm-internal-testbox ${build_output}
mv ${source_home}/PPU_SDK ${build_output}
mv ${source_home}/pccl ${build_output}

echo "######################### $(date +%Y%m%d%H%M%S) Build finished ##########################"
