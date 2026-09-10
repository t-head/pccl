#!/bin/bash
set -e

bash build.sh --use_sdk --hggcrt_version 13 --cpu_num ${cpu} --build_type=rel_no_sym --rebuild --package_version ${package_version}
