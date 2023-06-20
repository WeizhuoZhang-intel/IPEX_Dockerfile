#!/bin/bash
# set -x

kmp=${1:-''}
log_dir=${2:-log_dir}
rm -rf ${log_dir}
mkdir -p $log_dir

# collec result
function run_benchmark {
    # generate cmd
    python generate_run_script.py ${kmp}
    # run benchmark
    for item in `ls | grep run_ | grep .sh`; do
        bash ${item}
    done
}

run_benchmark
