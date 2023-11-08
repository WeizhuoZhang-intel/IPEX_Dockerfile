# Generate runnable scripts for benchmarking
import argparse
import yaml
parser = argparse.ArgumentParser("Generation script", add_help=False)
parser.add_argument("-k","--extra_kmp",action="store_true",default=False,help="llm extra kmp configuration")
parser.add_argument("-d","--deepspeed",action="store_true",default=False,help="only for deepspeed")
parser.add_argument("--nightly",action="store_true",default=False,help="only for nightly regular track")
parser.add_argument("--weekly",action="store_true",default=False,help="only for weekly regular track")
parser.add_argument("--debug",action="store_true",default=False,help="only for weekly regular track")
parser.add_argument("--publicds",action="store_true",default=False,help="only for nightly regular track public deepspeed")
args = parser.parse_args()

fetch_device_info = '''
sockets_num=$(lscpu |grep 'Socket(s):' |sed 's/[^0-9]//g')
cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')
phsical_cores_num=$(echo |awk -v sockets_num=${sockets_num} -v cores_per_socket=${cores_per_socket} '{
    print sockets_num * cores_per_socket;
}')
numa_nodes_num=$(numactl -H |grep 'node [0-9]* cpus: [0-9].*' |wc -l)
threads_per_core=$(lscpu |grep 'Thread(s) per core:' |sed 's/[^0-9]//g')
cores_per_node=$(numactl -H |grep "node 0 cpus:" |sed 's/.*://' |awk -v tpc=$threads_per_core '{print int(NF / tpc)}')
'''

deepspeed_ccl_func = '''
function deepspeed_core_config() {
    sockets_num=$(lscpu |grep 'Socket(s):' |sed 's/[^0-9]//g')
    cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')
    numa_nodes_num=$(numactl -H |grep 'node [0-9]* cpus: [0-9].*' |wc -l)
    threads_per_core=$(lscpu |grep 'Thread(s) per core:' |sed 's/[^0-9]//g')
    cores_per_node=$(numactl -H |grep "node 0 cpus:" |sed 's/.*://' |awk -v tpc=$threads_per_core '{print int(NF / tpc)}')
    local_rank=$1
    OOB_TOTAL_CORES_USE=$((${cores_per_node}*${local_rank}))
    hbm_index=0
    cores_per_instance=${cores_per_node}
    numa_nodes_use_=1,$local_rank

    device_array=($(numactl -H |grep "node [0-9]* cpus:" |sed "s/.*node//;s/cpus://" |sed -n "${numa_nodes_use_}p" |\\
                    awk -v cpn=${cores_per_node} '{for(i=1;i<=cpn+1;i++) {printf(" %s ",$i)} printf("\\n");}' |grep '[0-9]' |\\
                    awk -v cpi=${cores_per_instance} -v cpn=${cores_per_node} -v cores=${OOB_TOTAL_CORES_USE} -v hi=${hbm_index} '{
                if(cores == "") { if(cpi > cpn) {cores = cpi}else {cores = NF} }
                for( i=2; i<=cores; i++ ) {
                    if($i != "") {
                        if((i-1) % cpi == 0) {
                            print $i";"$1+hi
                        }else {
                            printf $i","
                        }
                    }
                }
            }' |sed "s/,$//"))

    echo ${device_array}
    export LLM_DEEPSPEED_COMM_CORES=0
    deepspeed_cores_list=($(echo ${device_array[@]} |sed 's/ /\\n/g' |awk -F ';' '{print $1}' |awk -F ',' -v cores=$LLM_DEEPSPEED_COMM_CORES 'BEGIN{
                    busy = "";
                    idle = "";
                }{
                    for (i=1;i<=NF;i++) {
                        if(i==1) {
                            idle = idle","$i;
                            if(cores==0) {
                                busy = busy","$i;
                            }
                        }else {
                            if(i>cores) {
                                busy = busy","$i;
                            }
                        }
                    }
                }END{
                    printf("%s\\n%s", idle, busy);
                }' |sed 's/^,//'))

    echo $deepspeed_cores_list
    # return $deepspeed_cores_list
}
'''

collect_result = '''
function collect_perf_logs_llm() {
    # latency
    sleep 5s
    latency=($(grep -i 'inference latency:' $log_dir/$1 |sed -e 's/.*atency://;s/[^0-9.]//g;s/\.$//' |awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num > 0) {
                printf("%d  %.6f", num, sum / num);
            }else {
                printf("0  0");
            }
        }
    '))
    first_latency=($(grep -i 'First token average latency:' $log_dir/$1 |sed -e 's/.*atency://;s/[^0-9.]//g;s/\.$//' |awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num > 0) {
                printf("%.6f", sum / num);
            }else {
                printf("0");
            }
        }
    '))
    avg_latency=($(grep -i 'Average 2... latency:' $log_dir/$1 |sed -e 's/.*atency://;s/[^0-9.]//g;s/\.$//' |awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num > 0) {
                printf("%.6f", sum / num);
            }else {
                printf("0");
            }
        }
    '))
    p90_latency=($(grep -i 'P90 2... latency:' $log_dir/$1 |sed -e 's/.*atency://;s/[^0-9.]//g;s/\.$//' |awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num > 0) {
                printf("%.6f", sum / num);
            }else {
                printf("0");
            }
        }
    '))
    p99_latency=($(grep -i 'P99 2... latency:' $log_dir/$1 |sed -e 's/.*atency://;s/[^0-9.]//g;s/\.$//' |awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num > 0) {
                printf("%.6f", sum / num);
            }else {
                printf("0");
            }
        }
    '))
 
    peak_memory=$(grep '^Total' ${log_dir}/mem-usage-$1 |sed 's/[^0-9. ]//g' |awk 'BEGIN{peak=0}{if($NF > peak){peak = $NF}}END{print peak / 1024}') || peak_memory=0
    printf $1 |tee -a ${log_dir}/summary.log
    printf ", ${latency[1]},${first_latency},${avg_latency},${p90_latency},${p99_latency},${peak_memory} \\n" |tee -a ${log_dir}/summary.log
}
'''

def generate_commands(yml_file,mode,extra_kmp):
    data = yaml.load(open(yml_file, 'r'),Loader=yaml.FullLoader)
    generated_file = "run_"+mode+".sh"
    with open(generated_file, "w") as runfile:
        lines = []
        lines.append("#!/bin/bash")
        lines.append("set -x")
        lines.append("# Env config")
        lines.append("export WORKDIR=/root/workspace")
        lines.append("export HF_HOME=/root/.cache/huggingface")
        lines.append("# source $HOME/oneCCL_install/env/setvars.sh")
        lines.append(f"export LD_PRELOAD={data['envconfig']['LD_PRELOAD']}")
        lines.append(f"export KMP_BLOCKTIME={data['envconfig']['KMP_BLOCKTIME']}")
        # lines.append(f"export KMP_AFFINITY={data['envconfig']['KMP_AFFINITY']}")
        lines.append("export TRANSFORMERS_OFFLINE=0")
        lines.append("pip install --upgrade huggingface_hub")
        lines.append("huggingface-cli login --token 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")
        # lines.append("export DNNL_VERBOSE=1")
        # if extra_kmp:
        lines.append(f"export KMP_TPAUSE={data['envconfig']['LLM_EXTRA_KMP']['KMP_TPAUSE']}")
        lines.append(f"export KMP_SETTINGS={data['envconfig']['LLM_EXTRA_KMP']['KMP_SETTINGS']}")
        lines.append(f"export KMP_FORJOIN_BARRIER_PATTERN={data['envconfig']['LLM_EXTRA_KMP']['KMP_FORJOIN_BARRIER_PATTERN']}")
        lines.append(f"export KMP_PLAIN_BARRIER_PATTERN={data['envconfig']['LLM_EXTRA_KMP']['KMP_PLAIN_BARRIER_PATTERN']}")
        lines.append(f"export KMP_REDUCTION_BARRIER_PATTERN={data['envconfig']['LLM_EXTRA_KMP']['KMP_REDUCTION_BARRIER_PATTERN']}")
        lines.append("log_dir=${1:-log_dir}")
        lines.append("mkdir -p $log_dir")
        lines.append("mkdir -p /root/workspace/qmodel")
        lines.append("# device info")
        lines.append(fetch_device_info)
        lines.append(collect_result)    
        lines.append(deepspeed_ccl_func)
        lines.append("")
        if mode.endswith('default'):
            lines.append("# Run Workload")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:

                            lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}_greedy_False_NUMA_1_BF16.log 2>&1 || true &")
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu \
                                            --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token}  --num-iter 50 --dtype {dtype} --ipex --jit --token-latency \
                                            2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}_greedy_False_NUMA_1_BF16.log")
                            lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}_greedy_False_NUMA_1_BF16.log")

        if mode.endswith('compile'):
            lines.append("# Run Workload")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:

                            lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}_greedy_False_NUMA_1_compile.log 2>&1 || true &")
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu \
                                            --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token}  --num-iter 50 --dtype {dtype} --ipex --token-latency --compile \
                                            2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}_greedy_False_NUMA_1_compile.log")
                            lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}_greedy_False_NUMA_1_compile.log")

        if mode.endswith('bssweep'):
            lines.append("# Run Workload")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for beam in data['modelargs'][mode]['greedy']:
                                for bs in data['modelargs'][mode]['batchsize']:
                                    lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_1_bf16.log 2>&1 || true &")
                                    if beam == True:
                                        lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu \
                                                    --benchmark -m {model_id} --input-tokens {input_token} --greedy --batch-size {bs} --max-new-tokens {output_token}  --num-iter 50 --dtype {dtype} --ipex --jit --token-latency \
                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_1_bf16.log")
                                    else:
                                        lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu \
                                                    --benchmark -m {model_id} --input-tokens {input_token} --batch-size {bs} --max-new-tokens {output_token}  --num-iter 50 --dtype {dtype} --ipex --jit --token-latency \
                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_1_bf16.log")
                                    lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_1_bf16.log")

        if mode.endswith('int8'):
            lines.append("# Run Workload")
            for model_id in data['modelargs'][mode]['modelid']:
                if 't5' not in model_id:
                    lines.append(f"rm -rf {data['modelargs'][mode]['outputdir']}")
                    lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}")
                    lines.append("mkdir saved_results")
                    if model_id == "EleutherAI/gpt-neox-20b":
                        lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --lambada --output-dir {data['modelargs'][mode]['outputdir']} --jit --int8 -m {model_id}")
                    elif model_id == "tiiuae/falcon-40b":
                        lines.append(f"python run_falcon_int8.py --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']} --jit --int8-bf16-mixed -m {model_id} --config-file /root/workspace/IPEX_Dockerfile/model_config/tiiuae_falcon-40b_config.json")
                    elif model_id != "THUDM/chatglm2-6b" and model_id != "baichuan-inc/Baichuan-13B-Chat":
                        lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']} --jit --int8-bf16-mixed -m {model_id}")
                for input_token in data['modelargs'][mode]['inputtokens']:
                    for output_token in data['modelargs'][mode]['maxnewtokens']:
                        weighttype = "BF16"
                        if model_id == "EleutherAI/gpt-neox-20b":
                            lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_woq-int8_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log 2>&1 || true &")
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} \
                                        --input-tokens {input_token} --max-new-tokens {output_token} --jit --int8 -m {model_id} --benchmark --token-latency --num-iter 50 \
                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_woq-int8_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")
                            lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_woq-int8_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")
                        # elif model_id == "tiiuae/falcon-40b":
                        #     lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_woq-int8_{input_token}-{output_token}_greedy_True_NUMA_1_BF16.log 2>&1 || true &")
                        #     lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu \
                        #                 --input-tokens {input_token} --max-new-tokens {output_token} --jit --int8-bf16-mixed -m {model_id} --lambada --benchmark --token-latency --num-iter 50 --ipex-weight-only-quantization --config-file=model_config/tiiuae_falcon-40b_config.json\
                        #                 2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_woq-int8_{input_token}-{output_token}_greedy_True_NUMA_1_BF16.log")
                        #     lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_woq-int8_{input_token}-{output_token}_greedy_True_NUMA_1_BF16.log")
                        elif 't5' in model_id:
                            lines.append(f"rm -rf {data['modelargs'][mode]['outputdir']}")
                            lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}")
                            lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']} --jit --int8-bf16-mixed -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token}")

                            lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_woq-int8_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log 2>&1 || true &")
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} \
                                        --input-tokens {input_token} --max-new-tokens {output_token} --jit --int8-bf16-mixed -m {model_id} --benchmark --token-latency --num-iter 50 \
                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_woq-int8_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")
                            lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_woq-int8_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")                            
                        elif model_id != "THUDM/chatglm2-6b" and model_id != "baichuan-inc/Baichuan-13B-Chat":
                            lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_woq-int8_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log 2>&1 || true &")
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} \
                                        --input-tokens {input_token} --max-new-tokens {output_token} --jit --int8-bf16-mixed -m {model_id} --benchmark --token-latency --num-iter 50 \
                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_woq-int8_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")
                            lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_woq-int8_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")
                        elif model_id == "THUDM/chatglm2-6b" or model_id == "baichuan-inc/Baichuan-13B-Chat":
                            lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_woq-int8_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log 2>&1 || true &")
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu \
                                        --input-tokens {input_token} --max-new-tokens {output_token} --jit --int8-bf16-mixed -m {model_id} --benchmark --token-latency --ipex-weight-only-quantization --num-iter 50 \
                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_woq-int8_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")
                            lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_woq-int8_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")
                        else:
                            lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_woq-int8_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log 2>&1 || true &")
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} \
                                        --input-tokens {input_token} --max-new-tokens {output_token} --jit --int8-bf16-mixed -m {model_id} --benchmark --token-latency --num-iter 50 \
                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_woq-int8_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")
                            lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_woq-int8_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")                            

        if mode.endswith('int4'):
            lines.append("# Run Workload")
            for model_id in data['modelargs'][mode]['modelid']:
                if 't5' not in model_id:
                    lines.append(f"rm -rf {data['modelargs'][mode]['outputdir']}")
                    lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}")
                    lines.append("mkdir saved_results")
                    if model_id == "EleutherAI/gpt-neox-20b":
                        lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --lambada --output-dir {data['modelargs'][mode]['outputdir']} --jit --int8 --weight-dtype INT4 --lowp-mode INT8 -m {model_id}")
                    elif model_id == "tiiuae/falcon-40b":
                        lines.append(f"python run_falcon_int8.py --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']} --jit --int8-bf16-mixed -m {model_id} --weight-dtype INT4 --lowp-mode INT8 --config-file /root/workspace/IPEX_Dockerfile/model_config/tiiuae_falcon-40b_config.json")
                    elif model_id != "THUDM/chatglm2-6b" and model_id != "baichuan-inc/Baichuan-13B-Chat":
                        lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --lambada --output-dir {data['modelargs'][mode]['outputdir']} --jit --int8-bf16-mixed --weight-dtype INT4 --lowp-mode INT8 -m {model_id}")
                for input_token in data['modelargs'][mode]['inputtokens']:
                    for output_token in data['modelargs'][mode]['maxnewtokens']:
                        weighttype = "BF16"
                        if model_id == "EleutherAI/gpt-neox-20b":
                            lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_woq-int4_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log 2>&1 || true &")
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} \
                                        --input-tokens {input_token} --max-new-tokens {output_token} --jit --weight-dtype INT4 --lowp-mode INT8 --int8 -m {model_id} --lambada --benchmark --token-latency --num-iter 50 \
                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_woq-int4_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")
                            lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_woq-int4_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")
                        # elif model_id == "tiiuae/falcon-40b":
                        #     lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_woq-int4_{input_token}-{output_token}_greedy_True_NUMA_1_BF16.log 2>&1 || true &")
                        #     lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu \
                        #                 --input-tokens {input_token} --max-new-tokens {output_token} --jit --int8-bf16-mixed -m {model_id} --lambada --benchmark --token-latency --num-iter 50 --ipex-weight-only-quantization --config-file=model_config/tiiuae_falcon-40b_config.json\
                        #                 2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_woq-int4_{input_token}-{output_token}_greedy_True_NUMA_1_BF16.log")
                        #     lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_woq-int4_{input_token}-{output_token}_greedy_True_NUMA_1_BF16.log")
                        elif 't5' in model_id:
                            lines.append(f"rm -rf {data['modelargs'][mode]['outputdir']}")
                            lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}")
                            lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']} --jit --int8-bf16-mixed --weight-dtype INT4 --lowp-mode INT8 -m {model_id} --input-tokens {input_token}")
                            lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_woq-int4_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log 2>&1 || true &")
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} \
                                        --input-tokens {input_token} --max-new-tokens {output_token} --jit --weight-dtype INT4 --lowp-mode INT8 --int8-bf16-mixed -m {model_id} --benchmark --token-latency --num-iter 50 \
                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_woq-int4_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")
                            lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_woq-int4_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")                            
                        elif model_id != "THUDM/chatglm2-6b" and model_id != "baichuan-inc/Baichuan-13B-Chat":
                            lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_woq-int4_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log 2>&1 || true &")
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} \
                                        --input-tokens {input_token} --max-new-tokens {output_token} --jit --weight-dtype INT4 --lowp-mode INT8 --int8-bf16-mixed -m {model_id} --benchmark --token-latency --num-iter 50 \
                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_woq-int4_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")
                            lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_woq-int4_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")
                        elif model_id == "THUDM/chatglm2-6b" or model_id == "baichuan-inc/Baichuan-13B-Chat":
                            lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_woq-int4_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log 2>&1 || true &")
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu \
                                        --input-tokens {input_token} --max-new-tokens {output_token} --jit --weight-dtype INT4 --lowp-mode INT8 --int8-bf16-mixed -m {model_id} --benchmark --token-latency --ipex-weight-only-quantization --num-iter 50 \
                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_woq-int4_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")
                            lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_woq-int4_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")
                        else:
                            lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_woq-int4_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log 2>&1 || true &")
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} \
                                        --input-tokens {input_token} --max-new-tokens {output_token} --jit --int8-bf16-mixed --weight-dtype INT4 --lowp-mode INT8 -m {model_id} --benchmark --token-latency --num-iter 50 \
                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_woq-int4_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")
                            lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_woq-int4_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log") 
                    
        if mode.endswith('int8static'):
            lines.append("# Run Workload")
            for model_id in data['modelargs'][mode]['modelid']:
                lines.append(f"rm -rf {data['modelargs'][mode]['outputdir']}")
                lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}")
                lines.append("mkdir saved_results")
                if model_id == "EleutherAI/gpt-neox-20b":
                    lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-smooth-quant --lambada --output-dir {data['modelargs'][mode]['outputdir']} --jit --int8 -m {model_id}")
                elif 't5' in model_id:
                    lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-smooth-quant --lambada --output-dir {data['modelargs'][mode]['outputdir']} --jit --int8-bf16-mixed -m {model_id}")
                elif model_id == "tiiuae/falcon-40b":
                    lines.append(f"python run_falcon_int8.py --ipex-smooth-quant --lambada --output-dir {data['modelargs'][mode]['outputdir']} --jit --int8-bf16-mixed -m {model_id} --config-file /root/workspace/IPEX_Dockerfile/model_config/tiiuae_falcon-40b_config.json")
                else:
                    lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-smooth-quant --lambada --output-dir {data['modelargs'][mode]['outputdir']} --jit --int8-bf16-mixed -m {model_id}")
                for input_token in data['modelargs'][mode]['inputtokens']:
                    for output_token in data['modelargs'][mode]['maxnewtokens']:
                        
                        if model_id == "EleutherAI/gpt-neox-20b":
                            lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_static-int8_{input_token}-{output_token}_greedy_False_NUMA_1_BF16.log 2>&1 || true &")
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} \
                                        --input-tokens {input_token} --max-new-tokens {output_token} --jit --int8 -m {model_id} --benchmark --token-latency --num-iter 50 \
                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_static-int8_{input_token}-{output_token}_greedy_False_NUMA_1_BF16.log")
                            lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_static-int8_{input_token}-{output_token}_greedy_False_NUMA_1_BF16.log")
                        # elif model_id == "tiiuae/falcon-40b":
                        #     lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_woq-int8_{input_token}-{output_token}_greedy_True_NUMA_1_BF16.log 2>&1 || true &")
                        #     lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu \
                        #                 --input-tokens {input_token} --max-new-tokens {output_token} --jit --int8-bf16-mixed -m {model_id} --lambada --benchmark --token-latency --num-iter 50 --ipex-weight-only-quantization --config-file=model_config/tiiuae_falcon-40b_config.json\
                        #                 2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_woq-int8_{input_token}-{output_token}_greedy_True_NUMA_1_BF16.log")
                        #     lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_woq-int8_{input_token}-{output_token}_greedy_True_NUMA_1_BF16.log")
                        elif 't5' in mode:
                            lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_static-int8_{input_token}-{output_token}_greedy_False_NUMA_1_BF16.log 2>&1 || true &")
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} \
                                        --input-tokens {input_token} --max-new-tokens {output_token} --jit --int8-bf16-mixed -m {model_id} --benchmark --token-latency --num-iter 50 \
                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_static-int8_{input_token}-{output_token}_greedy_False_NUMA_1_BF16.log")
                            lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_static-int8_{input_token}-{output_token}_greedy_False_NUMA_1_BF16.log")    
                        elif model_id == "THUDM/chatglm2-6b" or model_id == "baichuan-inc/Baichuan-13B-Chat":
                            lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_woq-int4_{input_token}-{output_token}_greedy_False_NUMA_1_BF16.log 2>&1 || true &")
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu \
                                        --input-tokens {input_token} --max-new-tokens {output_token} --jit --int8-bf16-mixed -m {model_id} --benchmark --token-latency --ipex-weight-only-quantization --num-iter 50 \
                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_woq-int4_{input_token}-{output_token}_greedy_False_NUMA_1_BF16.log")
                            lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_woq-int4_{input_token}-{output_token}_greedy_False_NUMA_1_BF16.log")                        
                        else:
                            lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_static-int8_{input_token}-{output_token}_greedy_False_NUMA_1_BF16.log 2>&1 || true &")
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} \
                                        --input-tokens {input_token} --max-new-tokens {output_token} --jit --int8-bf16-mixed -m {model_id} --benchmark --token-latency --num-iter 50 \
                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_static-int8_{input_token}-{output_token}_greedy_False_NUMA_1_BF16.log")
                            lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_static-int8_{input_token}-{output_token}_greedy_False_NUMA_1_BF16.log")

        if mode.endswith('2s'):
            lines.append("# Run Workload")

            # lines.append("export CCL_WORKER_COUNT=1")
            # lines.append("export CCL_PROCESS_LAUNCHER=none")
            # lines.append("export CCL_ATL_TRANSPORT=ofi")
            # lines.append("export CCL_ATL_SHM=1")

            # lines.append(f"export OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']}")
            # lines.append("export CCL_WORKER_AFFINITY=0,56")

            # lines.append("export DS_SHM_ALLREDUCE=1")
            lines.append("unset KMP_AFFINITY")

            for model_id in data['modelargs'][mode]['modelid']:
                lines.append(f"rm -rf {data['modelargs'][mode]['shardpath']}")
                lines.append(f"mkdir -p {data['modelargs'][mode]['shardpath']}")

                if 'bloom' not in model_id:
                    lines.append(f"python create_shard_model.py -m {model_id}  --save-path {data['modelargs'][mode]['shardpath']}")
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for numa in data['modelargs'][mode]['localrank']:
                                lines.append(f"export local_rank={numa}")
                                lines.append("deepspeed_core_config ${local_rank}")
                                lines.append("export CCL_WORKER_AFFINITY=${deepspeed_cores_list}")
                                lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")


                                lines.append(f"nohup bash /root/workspace/get_mem.sh  >> $log_dir/mem-usage-llm_deepspeed_{model_id.replace('/','-')}_ds-{dtype}_{input_token}-{output_token}_greedy_False_NUMA_{numa}_BF16.log 2>&1 || true &")
                                if 'bloom' in model_id:
                                    lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {numa} --bind_core_list $core_list {data['modelargs'][mode]['scriptname']} --device {data['modelargs'][mode]['device'][0]} --benchmark -m {model_id} --dtype {dtype} --input-tokens {input_token} \
                                            --max-new-tokens {output_token} --ipex --jit --token-latency --num-iter 50 2>&1 | tee -a $log_dir/llm_deepspeed_{model_id.replace('/','-')}_ds-{dtype}_{input_token}-{output_token}_greedy_False_NUMA_{numa}_BF16.log") 
                                else:
                                    lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {numa} --bind_core_list $core_list {data['modelargs'][mode]['scriptname']} --device {data['modelargs'][mode]['device'][0]} --benchmark -m {data['modelargs'][mode]['shardpath']} --dtype {dtype} --input-tokens {input_token} \
                                            --max-new-tokens {output_token} --ipex --jit --token-latency --num-iter 50 2>&1 | tee -a $log_dir/llm_deepspeed_{model_id.replace('/','-')}_ds-{dtype}_{input_token}-{output_token}_greedy_False_NUMA_{numa}_BF16.log") 
                                lines.append(f"collect_perf_logs_llm llm_deepspeed_{model_id.replace('/','-')}_ds-{dtype}_{input_token}-{output_token}_greedy_False_NUMA_{numa}_BF16.log")
        if mode.endswith('2s1'):
            lines.append("# Run Workload")

            # lines.append("export CCL_WORKER_COUNT=1")
            # lines.append("export CCL_PROCESS_LAUNCHER=none")
            # lines.append("export CCL_ATL_TRANSPORT=ofi")
            # lines.append("export CCL_ATL_SHM=1")

            # lines.append(f"export OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']}")
            # lines.append("export CCL_WORKER_AFFINITY=0,56")

            # lines.append("export DS_SHM_ALLREDUCE=1")
            lines.append("unset KMP_AFFINITY")

            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for numa in data['modelargs'][mode]['localrank']:
                                lines.append(f"export local_rank={numa}")
                                lines.append("deepspeed_core_config ${local_rank}")
                                lines.append("export CCL_WORKER_AFFINITY=${deepspeed_cores_list}")
                                lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")
                      
                                lines.append(f"nohup bash /root/workspace/get_mem.sh  >> $log_dir/mem-usage-llm_deepspeed_{model_id.replace('/','-')}_ds-{dtype}_{input_token}-{output_token}_greedy_True_NUMA_{numa}_BF16.log 2>&1 || true &")
                                lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {numa} --bind_core_list $core_list {data['modelargs'][mode]['scriptname']} --device {data['modelargs'][mode]['device'][0]} --benchmark -m {model_id} --dtype {dtype} --input-tokens {input_token} \
                                            --max-new-tokens {output_token} --ipex --jit --token-latency --num-iter 50 2>&1 | tee -a $log_dir/llm_deepspeed_{model_id.replace('/','-')}_ds-{dtype}_{input_token}-{output_token}_greedy_True_NUMA_{numa}_BF16.log") 
                                lines.append(f"collect_perf_logs_llm llm_deepspeed_{model_id.replace('/','-')}_ds-{dtype}_{input_token}-{output_token}_greedy_True_NUMA_{numa}_BF16.log")

        if mode.endswith('2swoq'):
            lines.append("# Run Workload")

            # lines.append("export CCL_WORKER_COUNT=1")
            # lines.append("export CCL_PROCESS_LAUNCHER=none")
            # lines.append("export CCL_ATL_TRANSPORT=ofi")
            # lines.append("export CCL_ATL_SHM=1")

            # lines.append(f"export OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']}")
            # lines.append("export CCL_WORKER_AFFINITY=0,56")

            # lines.append("export DS_SHM_ALLREDUCE=1")
            lines.append("unset KMP_AFFINITY")

            for model_id in data['modelargs'][mode]['modelid']:
                lines.append(f"rm -rf {data['modelargs'][mode]['shardpath']}")
                lines.append(f"mkdir -p {data['modelargs'][mode]['shardpath']}")
                if 'bloom' not in model_id:
                    lines.append(f"python create_shard_model.py -m {model_id}  --save-path {data['modelargs'][mode]['shardpath']}")
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for numa in data['modelargs'][mode]['localrank']:
                                lines.append(f"export local_rank={numa}")
                                lines.append("deepspeed_core_config ${local_rank}")
                                lines.append("export CCL_WORKER_AFFINITY=${deepspeed_cores_list}")
                                lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")
  

                                if model_id == "EleutherAI/gpt-neox-20b":
                                    lines.append(f"nohup bash /root/workspace/get_mem.sh  >> $log_dir/mem-usage-llm_deepspeed_{model_id.replace('/','-')}_ds-woq-int8_{input_token}-{output_token}_greedy_False_NUMA_{numa}_BF16.log 2>&1 || true &")
                                    lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {numa} --bind_core_list $core_list {data['modelargs'][mode]['scriptname']} --device {data['modelargs'][mode]['device'][0]} --benchmark -m {data['modelargs'][mode]['shardpath']} --dtype float32 --input-tokens {input_token} \
                                                --max-new-tokens {output_token} --ipex --jit --ipex-weight-only-quantization --token-latency --num-iter 50 2>&1 | tee -a $log_dir/llm_deepspeed_{model_id.replace('/','-')}_ds-woq-int8_{input_token}-{output_token}_greedy_False_NUMA_{numa}_BF16.log") 
                                    lines.append(f"collect_perf_logs_llm llm_deepspeed_{model_id.replace('/','-')}_ds-woq-int8_{input_token}-{output_token}_greedy_False_NUMA_{numa}_BF16.log")
                                elif 'bloom' in model_id:                                
                                    lines.append(f"nohup bash /root/workspace/get_mem.sh  >> $log_dir/mem-usage-llm_deepspeed_{model_id.replace('/','-')}_ds-woq-int8_{input_token}-{output_token}_greedy_False_NUMA_{numa}_BF16.log 2>&1 || true &")
                                    lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {numa} --bind_core_list $core_list {data['modelargs'][mode]['scriptname']} --device {data['modelargs'][mode]['device'][0]} --benchmark -m {model_id} --int8-bf16-mixed --input-tokens {input_token} \
                                                --max-new-tokens {output_token} --ipex --jit --ipex-weight-only-quantization --token-latency --num-iter 50 2>&1 | tee -a $log_dir/llm_deepspeed_{model_id.replace('/','-')}_ds-woq-int8_{input_token}-{output_token}_greedy_False_NUMA_{numa}_BF16.log") 
                                    lines.append(f"collect_perf_logs_llm llm_deepspeed_{model_id.replace('/','-')}_ds-woq-int8_{input_token}-{output_token}_greedy_False_NUMA_{numa}_BF16.log")                                
                                else:
                                    lines.append(f"nohup bash /root/workspace/get_mem.sh  >> $log_dir/mem-usage-llm_deepspeed_{model_id.replace('/','-')}_ds-woq-int8_{input_token}-{output_token}_greedy_False_NUMA_{numa}_BF16.log 2>&1 || true &")
                                    lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {numa} --bind_core_list $core_list {data['modelargs'][mode]['scriptname']} --device {data['modelargs'][mode]['device'][0]} --benchmark -m {data['modelargs'][mode]['shardpath']} --int8-bf16-mixed --input-tokens {input_token} \
                                                --max-new-tokens {output_token} --ipex --jit --ipex-weight-only-quantization --token-latency --num-iter 50 2>&1 | tee -a $log_dir/llm_deepspeed_{model_id.replace('/','-')}_ds-woq-int8_{input_token}-{output_token}_greedy_False_NUMA_{numa}_BF16.log") 
                                    lines.append(f"collect_perf_logs_llm llm_deepspeed_{model_id.replace('/','-')}_ds-woq-int8_{input_token}-{output_token}_greedy_False_NUMA_{numa}_BF16.log")

        if mode.endswith('2swoq4'):
            lines.append("# Run Workload")

            lines.append("export CCL_WORKER_COUNT=1")
            lines.append("export CCL_PROCESS_LAUNCHER=none")
            lines.append("export CCL_ATL_TRANSPORT=ofi")
            lines.append("export CCL_ATL_SHM=1")

            lines.append(f"export OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']}")
            # lines.append("export CCL_WORKER_AFFINITY=0,56")

            lines.append("export DS_SHM_ALLREDUCE=1")
            lines.append("unset KMP_AFFINITY")

            for model_id in data['modelargs'][mode]['modelid']:
                # lines.append(f"rm -rf {data['modelargs'][mode]['shardpath']}")
                # lines.append(f"mkdir -p {data['modelargs'][mode]['shardpath']}")
                # if 'bloom' not in model_id:
                #     lines.append(f"python create_shard_model.py -m {model_id}  --save-path {data['modelargs'][mode]['shardpath']}")
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for numa in data['modelargs'][mode]['localrank']:
                                lines.append(f"export local_rank={numa}")
                                lines.append("deepspeed_core_config ${local_rank}")
                                lines.append("export CCL_WORKER_AFFINITY=${deepspeed_cores_list}")
                                lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")


                                if model_id == "EleutherAI/gpt-neox-20b":
                                    lines.append(f"nohup bash /root/workspace/get_mem.sh  >> $log_dir/mem-usage-llm_deepspeed_{model_id.replace('/','-')}_ds-woq-int4_{input_token}-{output_token}_greedy_False_NUMA_{numa}_BF16.log 2>&1 || true &")
                                    lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {numa} --bind_core_list $core_list {data['modelargs'][mode]['scriptname']} --device {data['modelargs'][mode]['device'][0]} --benchmark -m {data['modelargs'][mode]['shardpath']} --dtype float32 --input-tokens {input_token} \
                                                --max-new-tokens {output_token} --ipex --jit --ipex-weight-only-quantization --weight-dtype INT4 --lowp-mode INT8 --token-latency --num-iter 50 2>&1 | tee -a $log_dir/llm_deepspeed_{model_id.replace('/','-')}_ds-woq-int4_{input_token}-{output_token}_greedy_False_NUMA_{numa}_BF16.log") 
                                    lines.append(f"collect_perf_logs_llm llm_deepspeed_{model_id.replace('/','-')}_ds-woq-int4_{input_token}-{output_token}_greedy_False_NUMA_{numa}_BF16.log")
                                elif 'bloom' in model_id:                                
                                    lines.append(f"nohup bash /root/workspace/get_mem.sh  >> $log_dir/mem-usage-llm_deepspeed_{model_id.replace('/','-')}_ds-woq-int4_{input_token}-{output_token}_greedy_False_NUMA_{numa}_BF16.log 2>&1 || true &")
                                    lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {numa} --bind_core_list $core_list {data['modelargs'][mode]['scriptname']} --device {data['modelargs'][mode]['device'][0]} --benchmark -m {model_id} --int8-bf16-mixed --input-tokens {input_token} \
                                                --max-new-tokens {output_token} --ipex --jit --ipex-weight-only-quantization --weight-dtype INT4 --lowp-mode INT8 --token-latency --num-iter 50 2>&1 | tee -a $log_dir/llm_deepspeed_{model_id.replace('/','-')}_ds-woq-int4_{input_token}-{output_token}_greedy_False_NUMA_{numa}_BF16.log") 
                                    lines.append(f"collect_perf_logs_llm llm_deepspeed_{model_id.replace('/','-')}_ds-woq-int4_{input_token}-{output_token}_greedy_False_NUMA_{numa}_BF16.log")                                
                                else:
                                    lines.append(f"nohup bash /root/workspace/get_mem.sh  >> $log_dir/mem-usage-llm_deepspeed_{model_id.replace('/','-')}_ds-woq-int4_{input_token}-{output_token}_greedy_False_NUMA_{numa}_BF16.log 2>&1 || true &")
                                    lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {numa} --bind_core_list $core_list {data['modelargs'][mode]['scriptname']} --device {data['modelargs'][mode]['device'][0]} --benchmark -m {data['modelargs'][mode]['shardpath']} --int8-bf16-mixed --input-tokens {input_token} \
                                                --max-new-tokens {output_token} --ipex --jit --ipex-weight-only-quantization --weight-dtype INT4 --lowp-mode INT8 --token-latency --num-iter 50 2>&1 | tee -a $log_dir/llm_deepspeed_{model_id.replace('/','-')}_ds-woq-int4_{input_token}-{output_token}_greedy_False_NUMA_{numa}_BF16.log") 
                                    lines.append(f"collect_perf_logs_llm llm_deepspeed_{model_id.replace('/','-')}_ds-woq-int4_{input_token}-{output_token}_greedy_False_NUMA_{numa}_BF16.log")

        if mode.endswith('oob'):
            lines.append("# Run Workload")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            weighttype = "BF16"
                            if dtype == "float32":
                                weighttype = "FP32"

                            lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log 2>&1 || true &")
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu \
                                         --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token}  --num-iter 50 --dtype {dtype} --prompt 'how can you do that?' --ipex \
                                            2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")
                            lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")

        if mode.endswith('profile'):
            lines.append("# DS Env config")
            lines.append(f"export OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']}")
            lines.append("unset KMP_AFFINITY")
            lines.append("# Run workload")
            lines.append("export CCL_WORKER_COUNT=1")
            lines.append("export CCL_PROCESS_LAUNCHER=none")
            lines.append("export CCL_ATL_TRANSPORT=ofi")
            lines.append("export CCL_ATL_SHM=1")
            lines.append("export FI_PROVIDER=tcp")
            lines.append("export DS_SHM_ALLREDUCE=1")
            lines.append("export TRANSFORMERS_OFFLINE=0")
            lines.append("pip install --upgrade huggingface_hub")
            lines.append("huggingface-cli login --token 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")
            
            for rank in data['modelargs'][mode]['localrank']:
                    for model_id in data['modelargs'][mode]['modelid']:
                        for dtype in data['modelargs'][mode]['dtype']:
                            lines.append(f"rm -rf {data['modelargs'][mode]['shardpath']}")
                            lines.append(f"mkdir -p {data['modelargs'][mode]['shardpath']}")
                            lines.append(f"python create_shard_model.py -m {model_id}  --save-path {data['modelargs'][mode]['shardpath']}")

                            for input_token in data['modelargs'][mode]['inputtokens']:
                                for beam in data['modelargs'][mode]['greedy']:
                                    for maxtoken in data['modelargs'][mode]['maxnewtokens']:

                                        lines.append(f"export local_rank={rank}")
                                        lines.append("deepspeed_core_config ${local_rank}")
                                        lines.append("export CCL_WORKER_AFFINITY=${deepspeed_cores_list}")
                                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")

                                        lines.append(f"nohup bash /root/workspace/get_mem.sh  >> $log_dir/mem-usage-llm_deepspeed_{model_id.replace('/','-')}_{dtype}_{input_token}-{maxtoken}_greedy_{beam}_NUMA_{rank}_BF16.log 2>&1 || true &")

                                        lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list {data['modelargs'][mode]['scriptname']} --benchmark --device {data['modelargs'][mode]['device'][0]} -m {data['modelargs'][mode]['shardpath']} --dtype {dtype} --input-tokens {input_token} \
                                        --max-new-tokens {maxtoken} --ipex --jit --profile --token-latency --num-iter {data['launcher']['iternum']}  2>&1 | tee -a $log_dir/llm_deepspeed_{model_id.replace('/','-')}_{dtype}_{input_token}-{maxtoken}_greedy_{beam}_NUMA_{rank}_BF16.log") 
                                        lines.append(f"collect_perf_logs_llm llm_deepspeed_{model_id.replace('/','-')}_{dtype}_{input_token}-{maxtoken}_greedy_{beam}_NUMA_{rank}_BF16.log")


        if mode.endswith('instance'):
            lines.append("# Run Workload")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            weighttype = "BF16"
                            if dtype == "float32":
                                weighttype = "FP32"

                            lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log 2>&1 || true &")
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C 0-55 python {data['modelargs'][mode]['scriptname']} --device cpu \
                                         --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token}  --num-iter 50 --dtype {dtype} --ipex --jit --token-latency \
                                            2>&1 | tee -a $log_dir/llm_1_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log & \
                                            OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 1 -C 56-111 python {data['modelargs'][mode]['scriptname']} --device cpu \
                                         --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token}  --num-iter 50 --dtype {dtype} --ipex --jit --token-latency \
                                            2>&1 | tee -a $log_dir/llm_2_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")
                            lines.append("wait")
                            lines.append(f"collect_perf_logs_llm llm_1_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")
                            lines.append(f"collect_perf_logs_llm llm_2_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")
                            
        if mode.endswith('instance_woq'):
            lines.append("# Run Workload")
            for model_id in data['modelargs'][mode]['modelid']:
                lines.append(f"rm -rf {data['modelargs'][mode]['outputdir']}")
                lines.append(f"mkdir {data['modelargs'][mode]['outputdir']}")
                lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --lambada --output-dir {data['modelargs'][mode]['outputdir']} --jit --int8-bf16-mixed --lowp-mode 'BF16' -m {model_id}")
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            weighttype = "BF16"
                            if dtype == "float32":
                                weighttype = "FP32"

                            # lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log 2>&1 || true &")
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C 0-55 python {data['modelargs'][mode]['scriptname']} --device cpu --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}\
                                         --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token}  --num-iter 50 --jit --int8-bf16-mixed --token-latency \
                                            2>&1 | tee -a $log_dir/llm_1_default_{model_id.replace('/','-')}_woq{dtype}_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log & \
                                            OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 1 -C 56-111 python {data['modelargs'][mode]['scriptname']} --device cpu --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}\
                                         --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token}  --num-iter 50 --jit --int8-bf16-mixed --token-latency \
                                            2>&1 | tee -a $log_dir/llm_2_default_{model_id.replace('/','-')}_woq{dtype}_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")
                            lines.append("wait")
                            lines.append(f"collect_perf_logs_llm llm_1_default_{model_id.replace('/','-')}_woq{dtype}_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")
                            lines.append(f"collect_perf_logs_llm llm_2_default_{model_id.replace('/','-')}_woq{dtype}_{input_token}-{output_token}_greedy_False_NUMA_1_{weighttype}.log")

        # if mode == "default":
        #     lines.append("# Run workload")
        #     for model_id in data['modelargs'][mode]['modelid']:
        #         for dtype in data['modelargs'][mode]['dtype']:
        #             for input_token in data['modelargs'][mode]['inputtokens']:
        #                     for beam in data['modelargs'][mode]['greedy']:                               
        #                         lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}.log 2>&1 || true &")
        #                         if beam == True:
        #                             lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device {data['modelargs'][mode]['device'][0]} -m {model_id} --input-tokens {input_token} --greedy --dtype {dtype} --ipex --jit --token-latency --profile 2>&1 | tee -a $log_dir/llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}.log")
        #                         else:
        #                             lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device {data['modelargs'][mode]['device'][0]} -m {model_id} --input-tokens {input_token} --dtype {dtype} --ipex --jit --token-latency --profile 2>&1 | tee -a $log_dir/llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}.log")
        #                         lines.append(f"collect_perf_logs_llm llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}.log")
        # if mode == "tpp":
        #     lines.append("# Run workload")
        #     for model_id in data['modelargs'][mode]['modelid']:
        #         for dtype in data['modelargs'][mode]['dtype']:
        #             for input_token in data['modelargs'][mode]['inputtokens']:
        #                 for beam in data['modelargs'][mode]['greedy']:
        #                     lines.append(f"nohup bash /root/workspace/get_mem.sh  >> $log_dir/mem-usage-llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}.log 2>&1 || true &")
        #                     if beam == True:
        #                         lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python \
        #                         {data['modelargs'][mode]['scriptname']} --device {data['modelargs'][mode]['device'][0]} -m {model_id} --input-tokens {input_token} --greedy --dtype {dtype} \
        #                         --ipex-tpp --ipex --jit --token-latency 2>&1 | tee -a $log_dir/llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}.log")
        #                     else:
        #                         lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device {data['modelargs'][mode]['device'][0]} -m {model_id} --input-tokens {input_token} --dtype {dtype} --ipex-tpp --ipex --jit --token-latency 2>&1 | tee -a $log_dir/llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}.log")
        #                     lines.append(f"collect_perf_logs_llm llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}.log")



        # if mode.endswith('int8'):
        #     for model_id in data['modelargs'][mode]['modelid']:
        #         lines.append("# GPT-J quantization")
        #         lines.append(f"mkdir {data['modelargs'][mode]['outputdir']}")
        #         lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --lambada --output-dir {data['modelargs'][mode]['outputdir']} --jit --int8-bf16-mixed --lowp-mode 'BF16' -m {model_id}")
        #         for input_token in data['modelargs'][mode]['inputtokens']:
        #             for output in data['modelargs'][mode]['maxnewtokens']:
        #                 for beam in data['modelargs'][mode]['greedy']:
        #                     lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output} --jit --int8-bf16-mixed -m {model_id} --lambada --benchmark --token-latency \
        #                      2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_int8_{input_token}-{output}_greedy_{beam}.log")

        #                     lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_int8_{input_token}-{output}_greedy_{beam}.log") 
        # if mode.endswith('bf16'):
        #     for model_id in data['modelargs'][mode]['modelid']:
        #         # lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --lambada --output-dir {data['modelargs'][mode]['outputdir']} --jit --int8-bf16-mixed --lowp-mode 'BF16' -m {model_id}")
        #         for input_token in data['modelargs'][mode]['inputtokens']:
        #             for output in data['modelargs'][mode]['maxnewtokens']:
        #                 for beam in data['modelargs'][mode]['greedy']:
        #                     lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -N {data['launcher']['numactlN']} -m {data['launcher']['numactlM']} python {data['modelargs'][mode]['scriptname']} --device cpu --input-tokens {input_token} --max-new-tokens {output} --jit -m {model_id} --dtype bfloat16 --lambada --benchmark --token-latency \
        #                      2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_bf16_{input_token}-{output}_greedy_{beam}.log")
        
        # if mode == 'deepspeed':
        #     lines.append("# DS Env config")
        #     lines.append(f"export OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']}")
        #     lines.append("unset KMP_AFFINITY")
        #     lines.append("# Run workload")
        #     for model_id in data['modelargs'][mode]['modelid']:
        #         for dtype in data['modelargs'][mode]['dtype']:
        #             for input_token in data['modelargs'][mode]['inputtokens']:
        #                 for beam in data['modelargs'][mode]['greedy']:
        #                     lines.append(f"nohup bash /root/workspace/get_mem.sh  >> $log_dir/mem-usage-llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}.log 2>&1 || true &")
        #                     if beam == True:
        #                         lines.append(f"deepspeed --enable_each_rank_log=/root/workspace/IPEX_Dockerfile/log_dir/ --bind_cores_to_rank {data['modelargs'][mode]['scriptname']} --benchmark --device {data['modelargs'][mode]['device'][0]} -m {model_id} --greedy --dtype {dtype} --input-tokens {input_token} \
        #                         --ipex --jit --token-latency --profile 2>&1 | tee -a $log_dir/llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}.log") 
        #                     else:
        #                         lines.append(f"deepspeed --enable_each_rank_log=/root/workspace/IPEX_Dockerfile/log_dir/ --bind_cores_to_rank {data['modelargs'][mode]['scriptname']} --benchmark --device {data['modelargs'][mode]['device'][0]} -m {model_id} --dtype {dtype} --input-tokens {input_token} \
        #                         --ipex --jit --token-latency --profile 2>&1 | tee -a $log_dir/llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}.log") 
        #                     lines.append(f"collect_perf_logs_llm llm_{mode}_{model_id.replace('/','-')}_{dtype}_{input_token}_greedy_{beam}.log")     


        lines.append(f"sleep 5s")
        lines.append("")
        runfile.writelines([line + "\n" for line in lines])
    return generated_file


if __name__ == '__main__':
    #for mode in 'default','gptj_int8','llama_int8','deepspeed':
    yml_file = 'bench_preci.yml'
    if args.deepspeed:
        yml_file = 'bench_ds_preci.yml'
    if args.nightly:
        yml_file = 'bench_nightly.yml'
    if args.weekly:
        yml_file = 'bench_weekly.yml'
    if args.debug:
        yml_file = 'bench_debug.yml'
    if args.publicds:
        yml_file = 'bench_publicds_nightly.yml'
    data = yaml.load(open(yml_file, 'r'),Loader=yaml.FullLoader) 
    for mode in data['modelargs'].keys():
        generate_commands(yml_file, mode, args.extra_kmp)
