#!/bin/bash
set -ex

source "$(dirname "$0")/setup_test_env.sh"

echo "######################### $(date +%Y%m%d%H%M%S) customAllreduce test ##########################"
echo $PWD
echo "pwd has:"
ls

mpirun --allow-run-as-root -mca orte_base_help_aggregate 0 --mca btl ^openib -np 2 all_reduce_perf -b 33K -e 33K -f 2 -d half
mpirun --allow-run-as-root -mca orte_base_help_aggregate 0 --mca btl ^openib -np 2 all_reduce_perf -b 301K -e 301K -f 2 -d half

echo "######################### $(date +%Y%m%d%H%M%S) customAllreduce test finish ##########################"
