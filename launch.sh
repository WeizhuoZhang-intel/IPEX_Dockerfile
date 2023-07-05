#!/bin/bash
# set -x

ds=${1:-disable}
log_dir=${2:-log_dir}
rm -rf ${log_dir}
mkdir -p $log_dir

# collec result
function run_benchmark {
    # generate cmd
    if [[ $ds == "enable" ]]; then
        echo "Enable DeepSpeed script generation"
        python generate_run_script.py -d
    else
        python generate_run_script.py
    fi
    # run benchmark
    for item in `ls | grep run_ | grep .sh`; do
        bash ${item}
    done
}

run_benchmark
