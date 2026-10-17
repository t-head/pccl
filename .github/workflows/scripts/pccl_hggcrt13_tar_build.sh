#!/bin/bash
set -e
cd ${source_home}/pccl
bash build.sh --use_sdk --hggcrt_version 13 --cpu_num ${cpu} --rebuild --package_version ${package_version}
