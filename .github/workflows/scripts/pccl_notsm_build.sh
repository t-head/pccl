#!/bin/bash
set -e
cd ${source_home}/pccl
bash build.sh --use_sdk --no_tsm --enable_tests
# See pccl_build.sh: a test build swaps `binary` for `tests`, so a second test-free pass is needed
# to produce pcclras/pcclparam while keeping the test artifacts.
bash build.sh --use_sdk --no_tsm