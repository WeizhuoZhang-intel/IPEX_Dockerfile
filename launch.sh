#!/bin/bash
# set -x

yml_file=${yml_file:-''}
rm -rf log_dir
mkdir -p log_dir

# collec result
function run_benchmark {
    # generate cmd
    python generate_run_script.py ${kmp} --yml_file yml_files/${yml_file}
    # run benchmark
    for item in `ls | grep run_ | grep .sh`; do
        bash ${item}
    done
}

run_benchmark
