#!/bin/bash
set -e
cd ${source_home}/pccl
bash build.sh --use_sdk --cpu_num ${cpu} --rebuild --package_version ${package_version}
