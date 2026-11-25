#!/bin/bash
set -e
cd ${source_home}/pccl
bash build.sh --use_sdk --enable_tests
# src/Makefile replaces `binary` with `tests` when DISABLE_TESTS=0, so the test build alone never
# produces pcclras/pcclparam. Re-run without --enable_tests: it is incremental and leaves the test
# artifacts built above in place.
bash build.sh --use_sdk