#!/bin/bash
# set -x

kmp=${1:-''}
log_dir=${2:-log_dir}
ds=${3:-disable}
rm -rf ${log_dir}
mkdir -p $log_dir

# collec result
function run_benchmark {
    # generate cmd
    python generate_run_script.py ${kmp}
    if [[ $ds == "enable" ]]; then
        echo "Enable DeepSpeed script generation"
        python generate_run_script.py ${kmp} --deepspeed
    fi
    # run benchmark
    for item in `ls | grep run_ | grep .sh`; do
        bash ${item}
    done
}

run_benchmark
