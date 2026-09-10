#!/bin/bash
set -e

bash build.sh --use_sdk --cpu_num ${cpu} --build_type=rel_no_sym --rebuild --package_version ${package_version}
