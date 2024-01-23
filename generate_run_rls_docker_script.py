# Generate runnable scripts for benchmarking
import argparse
import yaml
parser = argparse.ArgumentParser("Generation script", add_help=False)
parser.add_argument("-k","--extra_kmp",action="store_true",default=False,help="llm extra kmp configuration")
parser.add_argument("-d","--deepspeed",action="store_true",default=False,help="only for deepspeed")
parser.add_argument("--nightly",action="store_true",default=False,help="only for nightly regular track")
parser.add_argument("--weekly",action="store_true",default=False,help="only for weekly regular track")
parser.add_argument("--emr_weekly",action="store_true",default=False,help="only for weekly regular track")
parser.add_argument("--emr_nightly",action="store_true",default=False,help="only for weekly regular track")
parser.add_argument("--hbm_weekly",action="store_true",default=False,help="only for weekly regular track")
parser.add_argument("--hbm_weekly_m",action="store_true",default=False,help="only for weekly regular track")
parser.add_argument("--hbm_nightly",action="store_true",default=False,help="only for weekly regular track")
parser.add_argument("--gnr_weekly",action="store_true",default=False,help="only for weekly regular track")
parser.add_argument("--debug",action="store_true",default=False,help="only for debug regular track")
parser.add_argument("--rls",action="store_true",default=False,help="only for rls track")
parser.add_argument("--acc",action="store_true",default=False,help="only for rls track")
parser.add_argument("--rlsemr",action="store_true",default=False,help="only for rls track")
parser.add_argument("--acc_cluster",action="store_true",default=False,help="only for rls track")
parser.add_argument("--gptq",action="store_true",default=False,help="only for gptq track")
parser.add_argument("--cpudeviceweekly",action="store_true",default=False,help="only for gptq track")
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

timeprocess = '''

function get_file_info() {
    local file="$1"
    local size=$(stat --printf="%s" "$file")
    local modified=$(stat --printf="%Y" "$file")
    echo "$size $modified"
}

'''
startprocess = '''
function start_process() {
    local command="$1"
    local log_file="$2"
    
    $command | tee -a "$log_file" &
    echo $!
}

'''
monitorprocess = '''

function monitor_file() {
    local pid="$1"
    local log_file="$2"
    declare -A file_info

    file_info["last"]=$(get_file_info "$log_file")

    while true; do

        sleep 60

        local current_info=$(get_file_info "$log_file")

        if [ "${file_info["last"]}" == "$current_info" ]; then
            kill $pid
            break
        fi

        file_info["last"]=$current_info
    done
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

collect_acc_result = '''
function collect_acc_logs_llm() {
    # latency
    sleep 5s
    ppl_value=$(tail -n 5 $log_dir/$1 | grep 'lambada_openai.*ppl' | awk -F'|' '{print $5}' | tr -d '[:space:]')
    acc_value=$(tail -n 5 $log_dir/$1 | awk -F'|' '/acc/ {gsub(/^[ \\t]+|[ \\t]+$/, "", $5); print $5}')

    printf $1 |tee -a ${log_dir}/accsummary.log
    printf ", ${ppl_value},${acc_value} \\n" |tee -a ${log_dir}/accsummary.log
}
'''
collect_accnorm_result = '''
function collect_accnorm_logs_llm() {
    # latency
    sleep 5s

    acc_value=$(tail -n 5 $log_dir/$1 | awk -F'|' '/acc / {gsub(/^[ \\t]+|[ \\t]+$/, "", $5); print $5}')
    acc_norm_value=$(tail -n 5 $log_dir/$1 | awk -F'|' '/acc_norm/ {gsub(/^[ \\t]+|[ \\t]+$/, "", $5); print $5}')

    printf $1 |tee -a ${log_dir}/accsummary.log
    printf ", ${acc_norm},${acc_value} \\n" |tee -a ${log_dir}/accsummary.log
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
        lines.append("export WORKDIR=/root/workspace/llm")
        lines.append("export HF_HOME=/root/.cache/huggingface")
        lines.append("export TRANSFORMERS_OFFLINE=0")
        lines.append("pip install --upgrade huggingface_hub")
        lines.append("huggingface-cli login --token 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")
        lines.append("log_dir=/root/workspace/log")

        lines.append("# device info")
        lines.append(fetch_device_info)
        lines.append(collect_result)   
        lines.append(collect_acc_result)   
        lines.append(collect_accnorm_result)  
        lines.append(deepspeed_ccl_func)
        lines.append(timeprocess)
        lines.append(startprocess)
        lines.append(monitorprocess)
        lines.append("")
        # lines.append("cp prompt.json ./single_instance") 

        if mode.endswith('bf16'):
            lines.append("# Run Workload")  
            lines.append("export WORK_DIR=./")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for beam in data['modelargs'][mode]['greedy']:
                                for bs in data['modelargs'][mode]['batchsize']:
                                    for rank in data['modelargs'][mode]['localrank']:
                                        lines.append(f"export local_rank={rank}")
                                        lines.append("deepspeed_core_config ${local_rank}")
                                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                            
                                        if beam == True:   
                                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                        --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype {dtype} --batch-size {bs} --ipex --token-latency --profile   \
                                                            2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                        else:   
                                            if 'mpt' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype {dtype} --batch-size {bs} --ipex --token-latency --profile  --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")    
                                            else:

                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype {dtype} --batch-size {bs} --ipex --token-latency --profile \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")


        if mode.endswith('bf16compiler'):
            lines.append("# Run Workload")  
            lines.append("export WORK_DIR=./")
            lines.append("export TORCHINDUCTOR_FREEZING=1")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for beam in data['modelargs'][mode]['greedy']:
                                for bs in data['modelargs'][mode]['batchsize']:
                                    for rank in data['modelargs'][mode]['localrank']:
                                        lines.append(f"export local_rank={rank}")
                                        lines.append("deepspeed_core_config ${local_rank}")
                                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_compile_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                            
                                        if beam == True:   
                                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python single_instance/run_generation.py   \
                                                        --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype {dtype} --batch-size {bs} --ipex --token-latency --profile --torch-compile    \
                                                            2>&1 | tee -a $log_dir/llm_compile_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                        else:   
                                            if 'falcon' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python single_instance/run_generation.py \
                                                            --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype {dtype} --batch-size {bs} --ipex --token-latency --profile --torch-compile  --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                2>&1 | tee -a $log_dir/llm_compile_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                            elif 'mpt' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python single_instance/run_generation.py \
                                                            --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype {dtype} --batch-size {bs} --ipex --token-latency --profile --torch-compile  --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                2>&1 | tee -a $log_dir/llm_compile_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                   
                                            
                                            else:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python single_instance/run_generation.py \
                                                            --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype {dtype} --batch-size {bs} --ipex --token-latency --profile --torch-compile   \
                                                                2>&1 | tee -a $log_dir/llm_compile_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_compile_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")



        if mode.endswith('bf16pt'):
            lines.append("# Run Workload")  
            lines.append("export WORK_DIR=./")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for beam in data['modelargs'][mode]['greedy']:
                                for bs in data['modelargs'][mode]['batchsize']:
                                    for rank in data['modelargs'][mode]['localrank']:
                                        lines.append(f"export local_rank={rank}")
                                        lines.append("deepspeed_core_config ${local_rank}")
                                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_pt-{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                            
                                        if beam == True:   
                                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                        --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype {dtype} --batch-size {bs} --token-latency --profile  \
                                                            2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_pt-{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                        else:   
                                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                        --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype {dtype} --batch-size {bs} --token-latency --profile  \
                                                            2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_pt-{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_pt-{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")



        if mode.endswith('static8'):
            lines.append("# Run Workload")  
            lines.append("export WORK_DIR=./")
            for model_id in data['modelargs'][mode]['modelid']:
                lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/{model_id}")
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for beam in data['modelargs'][mode]['greedy']:
                                for bs in data['modelargs'][mode]['batchsize']:
                                    for rank in data['modelargs'][mode]['localrank']:
                                        lines.append(f"export local_rank={rank}")
                                        lines.append("deepspeed_core_config ${local_rank}")
                                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                        
                                        if 'fp32' in dtype:
                                            if beam == True:  
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-smooth-quant --qconfig-summary-file {data['modelargs'][mode]['configpath']}/{model_id}/best_configure.json --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:  
                                                if 'falcon' in model_id:
                                                    lines.append(f"python run.py  \
                                                                --benchmark -m {model_id} --ipex-smooth-quant --qconfig-summary-file {data['modelargs'][mode]['configpath']}/{model_id}/best_configure.json --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency --profile  --config-file utils/model_config/tiiuae_falcon-40b_config.json   \
                                                                    2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                                else: 
                                                    lines.append(f"python run.py  \
                                                                --benchmark -m {model_id} --ipex-smooth-quant --qconfig-summary-file {data['modelargs'][mode]['configpath']}/{model_id}/best_configure.json --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency --profile     \
                                                                    2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")       
                                        elif 'bf16' in dtype:
                                            if beam == True:   
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-smooth-quant --qconfig-summary-file {data['modelargs'][mode]['configpath']}/{model_id}/best_configure.json --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --batch-size {bs} --quant-with-amp --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-smooth-quant --qconfig-summary-file {data['modelargs'][mode]['configpath']}/{model_id}/best_configure.json --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --quant-with-amp --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                           
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")



        if mode.endswith('autotune'):
            lines.append("# Run Workload")  
            lines.append("export WORK_DIR=./")
            for model_id in data['modelargs'][mode]['modelid']:
                lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/{model_id}")
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for beam in data['modelargs'][mode]['greedy']:
                                for bs in data['modelargs'][mode]['batchsize']:
                                    for rank in data['modelargs'][mode]['localrank']:
                                        lines.append(f"export local_rank={rank}")
                                        lines.append("deepspeed_core_config ${local_rank}")
                                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                        
                                        if 'fp32' in dtype:
                                            if beam == True:  
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --alpha auto --ipex-smooth-quant  --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:  
                                                if 'falcon' in model_id:
                                                    lines.append(f"OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  \
                                                                --benchmark -m {model_id} --ipex-smooth-quant --alpha auto --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency --profile  --config-file utils/model_config/tiiuae_falcon-40b_config.json   \
                                                                    2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                                else: 
                                                    lines.append(f"python run.py  \
                                                                --benchmark -m {model_id} --ipex-smooth-quant --alpha auto --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency --profile     \
                                                                    2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")       
                                        elif 'bf16' in dtype:
                                            if beam == True:   
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-smooth-quant --alpha auto --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --batch-size {bs} --quant-with-amp --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-smooth-quant --alpha auto --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --quant-with-amp --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                           
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")



        if mode.endswith('static81'):
            lines.append("# Run Workload")  
            lines.append("export WORK_DIR=./")
            for model_id in data['modelargs'][mode]['modelid']:
                lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}")
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for beam in data['modelargs'][mode]['greedy']:
                                for bs in data['modelargs'][mode]['batchsize']:
                                    for rank in data['modelargs'][mode]['localrank']:
                                        lines.append(f"export local_rank={rank}")
                                        lines.append("deepspeed_core_config ${local_rank}")
                                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                        
                                        if 'fp32' in dtype:
                                            if beam == True:  
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-smooth-quant --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency --profile     \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-smooth-quant --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --token-latency --profile     \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")       
                                          
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")


        if mode.endswith('static8quant'):
            lines.append("# Run Workload")  
            lines.append("export WORK_DIR=./")
            for model_id in data['modelargs'][mode]['modelid']:
                for rank in data['modelargs'][mode]['localrank']:
                    lines.append(f"export local_rank={rank}")
                    lines.append("deepspeed_core_config ${local_rank}")
                    lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")
                    
                
                    for dtype in data['modelargs'][mode]['dtype']:
                        if 'fp32' in dtype:

                            lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/{model_id}")
                            lines.append(f"python single_instance/run_quantization.py --ipex-smooth-quant --alpha {data['modelargs'][mode]['alpha']} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id}  -m {model_id} ")
                        elif 'bf16' in dtype:
                            lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/{model_id}")
                            lines.append(f"python single_instance/run_quantization.py --ipex-smooth-quant --alpha {data['modelargs'][mode]['alpha']} --output-dir {data['modelargs'][mode]['outputdir']}/bf16/{model_id} --quant-with-amp -m {model_id} ")                            
            
                        for bs in data['modelargs'][mode]['batchsize']:
                            if 'fp32' in dtype:
                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['outputdir']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --batch-size {bs} --ipex  --tasks lambada_openai \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}-{bs}_{data['launcher']['hw']}.log")
                            # elif 'bs1' in dtype:
                            #     lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}  -m {model_id} --dtype int8 --batch-size {bs} --ipex  --tasks lambada_openai \
                            #             2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}-{bs}_{data['launcher']['hw']}.log") 
                            else:
                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['outputdir']}/bf16/{model_id}/best_model.pt  -m {model_id} --dtype int8 --batch-size {bs} --quant-with-amp --ipex  --tasks lambada_openai \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}-{bs}_{data['launcher']['hw']}.log")                                         
            
                    lines.append(f"collect_acc_logs_llm llm_accuracy_{model_id.replace('/','-')}_{dtype}-{bs}_{data['launcher']['hw']}.log")





        if mode.endswith('woq8'):
            lines.append("# Run Workload")  
            lines.append("export WORK_DIR=./")
            for model_id in data['modelargs'][mode]['modelid']:
                lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/{model_id}")
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for beam in data['modelargs'][mode]['greedy']:
                                for bs in data['modelargs'][mode]['batchsize']:
                                    for rank in data['modelargs'][mode]['localrank']:
                                        lines.append(f"export local_rank={rank}")
                                        lines.append("deepspeed_core_config ${local_rank}")
                                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                            
                                        if beam == True: 
                                            if 'falcon' in model_id:  
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --weight-dtype INT8 --quant-with-amp --batch-size {bs} --token-latency --profile  --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            elif 'neox' in model_id or 'dolly' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --weight-dtype INT8  --batch-size {bs} --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                
                                            elif 'mpt' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --weight-dtype INT8 --quant-with-amp --batch-size {bs} --token-latency --profile    --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                
                                            else:    
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --weight-dtype INT8 --quant-with-amp --batch-size {bs} --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                        else:   
                                            if 'fp32' in dtype:

                                                if 'falcon' in model_id:
                                                    lines.append(f"python run.py  \
                                                                --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --weight-dtype INT8 --batch-size {bs} --token-latency --profile  --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                    2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                elif 'neox' in model_id or 'dolly' in model_id:
                                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                                --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --weight-dtype INT8  --batch-size {bs} --token-latency --profile   \
                                                                    2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                  
                                                elif 'mpt' in model_id:
                                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                                --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --weight-dtype INT8 --batch-size {bs} --token-latency --profile  --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                   
                                                
                                                else:
                                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                                --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --weight-dtype INT8 --batch-size {bs} --token-latency --profile   \
                                                                    2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  

                                            else:

                                                if 'falcon' in model_id:
                                                    lines.append(f"python run.py  \
                                                                --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --weight-dtype INT8 --quant-with-amp --batch-size {bs} --token-latency --profile  --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                    2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                elif 'neox' in model_id or 'dolly' in model_id:
                                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                                --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --weight-dtype INT8  --batch-size {bs} --token-latency --profile    \
                                                                    2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                  
                                                elif 'mpt' in model_id:
                                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                                --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --weight-dtype INT8 --quant-with-amp --batch-size {bs} --token-latency --profile  --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                   
                                                
                                                else:
                                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                                --benchmark -m {model_id} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --weight-dtype INT8 --quant-with-amp --batch-size {bs} --token-latency --profile \
                                                                    2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")

        if mode.endswith('woq4'):
            lines.append("# Run Workload")  
            lines.append("export WORK_DIR=./")
            for model_id in data['modelargs'][mode]['modelid']:
                lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}")
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for beam in data['modelargs'][mode]['greedy']:
                                for bs in data['modelargs'][mode]['batchsize']:
                                    for rank in data['modelargs'][mode]['localrank']:
                                        lines.append(f"export local_rank={rank}")
                                        lines.append("deepspeed_core_config ${local_rank}")
                                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                            
                                        if beam == True:   
                                            if 'falcon' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --gptq --weight-dtype INT4 --quant-with-amp --batch-size {bs} --token-latency --profile  --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            elif 'neox' in model_id or 'dolly' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --gptq --weight-dtype INT4 --batch-size {bs} --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                
                                            elif 'mpt' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --gptq --weight-dtype INT4 --quant-with-amp --batch-size {bs} --token-latency --profile  --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                            else:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --gptq --weight-dtype INT4 --quant-with-amp --batch-size {bs} --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                        else:   
                                            if 'falcon' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --gptq --weight-dtype INT4 --quant-with-amp --batch-size {bs} --token-latency --profile  --config-file utils/model_config/tiiuae_falcon-40b_config.json \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")    
                                            elif 'neox' in model_id or 'dolly' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --gptq --weight-dtype INT4 --batch-size {bs} --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                 
                                            elif 'mpt' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --gptq --weight-dtype INT4 --quant-with-amp --batch-size {bs} --token-latency --profile  --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                 
                                            
                                            else:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python run.py  \
                                                            --benchmark -m {model_id} --ipex-weight-only-quantization --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --gptq --weight-dtype INT4 --quant-with-amp --batch-size {bs} --token-latency --profile    \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                           
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")


        if mode.endswith('bf16ds'):
            lines.append("# Run Workload") 
            # lines.append("cp prompt.json ./distributed") 
            lines.append("export WORK_DIR=./")
            lines.append("unset KMP_AFFINITY")
            for model_id in data['modelargs'][mode]['modelid']:
                lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/{model_id}")
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for beam in data['modelargs'][mode]['greedy']:
                                for bs in data['modelargs'][mode]['batchsize']:
                                    for rank in data['modelargs'][mode]['localrank']:
                                        lines.append(f"export local_rank={rank}")
                                        lines.append("deepspeed_core_config ${local_rank}")
                                        lines.append("export CCL_WORKER_AFFINITY=${deepspeed_cores_list}")
                                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                        if data['modelargs'][mode]['shard'] == True:
                                            if beam == True:   
                                                lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                            --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile --autotp  \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:  
                                                if 'bloom' in model_id:
                                                    lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp    \
                                                                    2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                                else:

                                                    lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py \
                                                            --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile --autotp  \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                        else:

                                            if beam == True:   
                                                lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                            --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp     \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                if 'mpt' in model_id:
                                                    lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp    --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")   
                                                elif 'bloom' in model_id:
                                                    lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp    \
                                                                    2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                                                                    
                                                else:

                                                    lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp     \
                                                                    2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")

        if mode.endswith('bf16dsp'):
            lines.append("# Run Workload") 
            # lines.append("cp prompt.json ./distributed") 
            lines.append("export WORK_DIR=./")
            lines.append("unset KMP_AFFINITY")
            for model_id in data['modelargs'][mode]['modelid']:
                lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/{model_id}")
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for beam in data['modelargs'][mode]['greedy']:
                                for bs in data['modelargs'][mode]['batchsize']:
                                    for rank in data['modelargs'][mode]['localrank']:
                                        lines.append(f"export local_rank={rank}")
                                        lines.append("deepspeed_core_config ${local_rank}")
                                        lines.append("export CCL_WORKER_AFFINITY=${deepspeed_cores_list}")
                                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                        if data['modelargs'][mode]['shard'] == True:
                                            if beam == True:   
                                                lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                            --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile --autotp  \
                                                                2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py \
                                                            --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile --autotp  \
                                                                2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                        else:

                                            if beam == True:   
                                                lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                            --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp --shard-model    \
                                                                2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                if 'mpt' in model_id:
                                                    lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp --shard-model  --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  

                                                else:

                                                    lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp --shard-model    \
                                                                    2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")


        if mode.endswith('bf16dstime'):
            lines.append("# Run Workload") 
            # lines.append("cp prompt.json ./distributed") 
            lines.append("export WORK_DIR=./")
            lines.append("unset KMP_AFFINITY")
            for model_id in data['modelargs'][mode]['modelid']:
                lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}")
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for beam in data['modelargs'][mode]['greedy']:
                                for bs in data['modelargs'][mode]['batchsize']:
                                    for rank in data['modelargs'][mode]['localrank']:
                                        lines.append(f"export local_rank={rank}")
                                        lines.append("deepspeed_core_config ${local_rank}")
                                        lines.append("export CCL_WORKER_AFFINITY=${deepspeed_cores_list}")
                                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")

                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                        if data['modelargs'][mode]['shard'] == True:
                                            if beam == True:   
                                                lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                            --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile --autotp  \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py \
                                                            --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile --autotp  \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                        else:

                                            if beam == True:   
                                                lines.append(f"deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                            --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp --shard-model    \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                            --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp --shard-model    \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                          
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")


        if mode.endswith('woqds'):
            lines.append("# Run Workload") 
            # lines.append("cp prompt.json ./distributed") 
            lines.append("export WORK_DIR=./")
            lines.append("unset KMP_AFFINITY")
            for model_id in data['modelargs'][mode]['modelid']:
                # lines.append(f"rm -rf {data['modelargs'][mode]['outputdir']}")
                lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/{model_id}")
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for beam in data['modelargs'][mode]['greedy']:
                                for bs in data['modelargs'][mode]['batchsize']:
                                    for rank in data['modelargs'][mode]['localrank']:
                                        lines.append(f"export local_rank={rank}")
                                        lines.append("deepspeed_core_config ${local_rank}")
                                        lines.append("export CCL_WORKER_AFFINITY=${deepspeed_cores_list}")
                                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                        if data['modelargs'][mode]['shard'] == True:
                                            if beam == True:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                        
                                                    else:

                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                else:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                        
                                                    else:

                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                          
                                                    
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                      
                                                    
                                                    else:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                else:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py \
                                                                    --benchmark -m {model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                          
                                                    else:

                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                        else:

                                            if beam == True:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                        
                                                
                                                    else:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                else:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                        
                                                    
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                        
                                                    
                                                    else:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                          
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                         
                                                    else:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")      
                                                else:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")     
                                                    elif 'bloom' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                           
                                                    
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                            
                                                    
                                                    
                                                    else:                                                      
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --autotp     \
                                                                        2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")



        if mode.endswith('woqdsp'):
            lines.append("# Run Workload") 
            # lines.append("cp prompt.json ./distributed") 
            lines.append("export WORK_DIR=./")
            lines.append("unset KMP_AFFINITY")
            for model_id in data['modelargs'][mode]['modelid']:
                lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/{model_id}")
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for beam in data['modelargs'][mode]['greedy']:
                                for bs in data['modelargs'][mode]['batchsize']:
                                    for rank in data['modelargs'][mode]['localrank']:
                                        lines.append(f"export local_rank={rank}")
                                        lines.append("deepspeed_core_config ${local_rank}")
                                        lines.append("export CCL_WORKER_AFFINITY=${deepspeed_cores_list}")
                                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                        if data['modelargs'][mode]['shard'] == True:
                                            if beam == True:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    else:

                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                else:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    else:

                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                        
                                                    
                                                    else:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                else:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                          
                                                    
                                                    else:

                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                        else:

                                            if beam == True:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --autotp --shard-model    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    else:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --autotp --shard-model    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                else:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --autotp --shard-model    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    else:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --autotp --shard-model    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --autotp --shard-model    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --autotp --shard-model    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                          
                                                    
                                                    else:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --autotp --shard-model    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")      
                                                else:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --autotp --shard-model    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")     
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --autotp --shard-model    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                     
                                                    else:                                                      
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --autotp --shard-model    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_mode-p_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")


        if mode.endswith('bf16dsm'):
            lines.append("# Run Workload") 
            # lines.append("cp prompt.json ./distributed") 
            lines.append("export WORK_DIR=./")
            lines.append("unset KMP_AFFINITY")
            for model_id in data['modelargs'][mode]['modelid']:
                lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/{model_id}")
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for beam in data['modelargs'][mode]['greedy']:
                                for bs in data['modelargs'][mode]['batchsize']:
                                    for rank in data['modelargs'][mode]['localrank']:
                                        lines.append(f"export local_rank={rank}")
                                        lines.append("deepspeed_core_config ${local_rank}")
                                        lines.append("export CCL_WORKER_AFFINITY=${deepspeed_cores_list}")
                                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                        if data['modelargs'][mode]['shard'] == True:
                                            if beam == True:   
                                                lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                            --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile --autotp  \
                                                                2>&1 | tee -a $log_dir/llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py \
                                                            --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile --autotp  \
                                                                2>&1 | tee -a $log_dir/llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                        else:

                                            if beam == True:   
                                                lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                            --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp    \
                                                                2>&1 | tee -a $log_dir/llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            elif 'mpt' in model_id:
                                                lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                            --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp    \
                                                                2>&1 | tee -a $log_dir/llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                  
                                            
                                            else:   
                                                lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                            --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --dtype bfloat16 --batch-size {bs} --ipex --token-latency --profile  --autotp    \
                                                                2>&1 | tee -a $log_dir/llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")




        if mode.endswith('woqdsm'):
            lines.append("# Run Workload") 
            # lines.append("cp prompt.json ./distributed") 
            lines.append("export WORK_DIR=./")
            lines.append("unset KMP_AFFINITY")
            for model_id in data['modelargs'][mode]['modelid']:
                lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/{model_id}")
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for beam in data['modelargs'][mode]['greedy']:
                                for bs in data['modelargs'][mode]['batchsize']:
                                    for rank in data['modelargs'][mode]['localrank']:
                                        lines.append(f"export local_rank={rank}")
                                        lines.append("deepspeed_core_config ${local_rank}")
                                        lines.append("export CCL_WORKER_AFFINITY=${deepspeed_cores_list}")
                                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")

                                        
                                        lines.append(f"nohup bash /root/workspace/get_mem.sh >> $log_dir/mem-usage-llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log 2>&1 || true &")
                                        if data['modelargs'][mode]['shard'] == True:
                                            if beam == True:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    else:

                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                else:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    else:

                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile   --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    else:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                else:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                                    else:

                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py \
                                                                    --benchmark -m {data['modelargs'][mode]['outputdir']} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp  --autotp  \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")  
                                        else:

                                            if beam == True:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    else:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                else:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                                    else:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --greedy --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:   
                                                if 'int4' in dtype:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log") 
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                          
                                                    
                                                    else:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization  --batch-size {bs} --weight-dtype INT4 --ipex --token-latency --profile  --quant-with-amp --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")      
                                                else:
                                                    if 'neox' in model_id or 'dolly' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")     
                                                    elif 'mpt' in model_id:
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                            
                                                    
                                                    else:                                                      
                                                        lines.append(f"timeout 30m deepspeed --bind_cores_to_rank --num_accelerators {rank} --bind_core_list $core_list run.py  \
                                                                    --benchmark -m {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --ipex-weight-only-quantization --weight-dtype INT8 --batch-size {bs} --ipex --token-latency --profile  --quant-with-amp --autotp    \
                                                                        2>&1 | tee -a $log_dir/llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                            
                                
                                        lines.append(f"collect_perf_logs_llm llm_mode-m_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")




        if mode.endswith('gptq'):
            lines.append("# Run Workload")  
            lines.append("export WORK_DIR=./")
            # lines.append("unset KMP_BLOCKTIME KMP_TPAUSE KMP_SETTINGS KMP_AFFINITY KMP_FORJOIN_BARRIER_PATTERN KMP_PLAIN_BARRIER_PATTERN KMP_REDUCTION_BARRIER_PATTERN")
            for model_id in data['modelargs'][mode]['modelid']:
                lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/{model_id}")
                lines.append("ls utils")
                lines.append("pwd")
                lines.append(f"python utils/run_gptq.py --model {model_id} --output-dir {data['modelargs'][mode]['outputdir']}/{model_id}")
                # lines.append("wait")
                if 'falcon' in model_id: 
                    lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --quant-with-amp -m {model_id} --low-precision-checkpoint {data['modelargs'][mode]['outputdir']}/{model_id}/gptq_checkpoint_g128.pt --config-file utils/model_config/tiiuae_falcon-40b_config.json")
                elif 'neox' in model_id or 'dolly' in model_id:
                    lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id}  -m {model_id} --low-precision-checkpoint {data['modelargs'][mode]['outputdir']}/{model_id}/gptq_checkpoint_g128.pt")                    
                elif 'mpt' in model_id:
                    lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --quant-with-amp -m {model_id} --low-precision-checkpoint {data['modelargs'][mode]['outputdir']}/{model_id}/gptq_checkpoint_g128.pt --config-file=utils/model_config/mosaicml_mpt-7b_config.json")                    
                
                else:
                    lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --quant-with-amp -m {model_id} --low-precision-checkpoint {data['modelargs'][mode]['outputdir']}/{model_id}/gptq_checkpoint_g128.pt")

        if mode.endswith('gptqacc'):
            lines.append("# Run Workload")  
            lines.append("export WORK_DIR=./")
            # lines.append("source /tools/env_activate.sh")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for rank in data['modelargs'][mode]['localrank']:
                        lines.append(f"export local_rank={rank}")
                        lines.append("deepspeed_core_config ${local_rank}")
                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")                    
                        lines.append("ls utils")
                        lines.append("pwd")
                        # lines.append(f"python utils/run_gptq.py --model {model_id} --output-dir {data['modelargs'][mode]['outputdir']}")
                        # lines.append("wait")
                        if 'codegen' in model_id:
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python single_instance/run_accuracy.py -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt --batch-size 1 \
                                            --dtype int8   --quant-with-amp --tasks hellaswag \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                        
                        elif 'neox' in model_id or 'dolly' in model_id:
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python single_instance/run_accuracy.py -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt --batch-size 1 \
                                            --dtype int8   --tasks lambada_openai \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        elif 'mpt' in model_id:
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python single_instance/run_accuracy.py -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt --batch-size 1 \
                                            --dtype int8   --quant-with-amp --tasks hellaswag --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        else:
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python single_instance/run_accuracy.py -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt --batch-size 1 \
                                            --dtype int8   --quant-with-amp --tasks lambada_openai \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")

                        if 'codegen' in model_id in model_id or 'mpt' in model_id:
                            lines.append(f"collect_accnorm_logs_llm llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        else:
                            lines.append(f"collect_acc_logs_llm llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")


        if mode.endswith('gptqperf'):
            lines.append("# Run Workload")  
            lines.append("export WORK_DIR=./")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for input_token in data['modelargs'][mode]['inputtokens']:
                        for output_token in data['modelargs'][mode]['maxnewtokens']:
                            for beam in data['modelargs'][mode]['greedy']:
                                for bs in data['modelargs'][mode]['batchsize']:
                                    for rank in data['modelargs'][mode]['localrank']:
                                        lines.append(f"export local_rank={rank}")
                                        lines.append("deepspeed_core_config ${local_rank}")
                                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")
                                        # lines.append(f"python utils/run_gptq.py --model {model_id} --output-dir {data['modelargs'][mode]['outputdir']}")
                                        # lines.append("wait")
                                        if beam == True:
                                            if 'neox' in model_id or 'dolly' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python {data['modelargs'][mode]['scriptname']} \
                                                            -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt --benchmark --greedy  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --ipex-weight-only-quantization --weight-dtype INT4 --token-latency --profile  \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            else:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python {data['modelargs'][mode]['scriptname']} \
                                                            -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt --benchmark --greedy --quant-with-amp --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --ipex-weight-only-quantization --weight-dtype INT4 --token-latency --profile  \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                        else:
                                            if 'neox' in model_id or 'dolly' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python {data['modelargs'][mode]['scriptname']} \
                                                            -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt --benchmark  --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --ipex-weight-only-quantization --weight-dtype INT4 --token-latency --profile  \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                            elif 'mpt' in model_id:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python {data['modelargs'][mode]['scriptname']} --config-file=utils/model_config/mosaicml_mpt-7b_config.json \
                                                            -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt --benchmark --quant-with-amp --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --ipex-weight-only-quantization --weight-dtype INT4 --token-latency --profile  \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")                                                
                                            
                                            else:
                                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m {data['launcher']['numactlM']} -C $core_list python {data['modelargs'][mode]['scriptname']} \
                                                            -m {model_id} --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt --benchmark --quant-with-amp --input-tokens {input_token} --max-new-tokens {output_token} --num-iter {data['launcher']['iternum']} --batch-size {bs} --ipex-weight-only-quantization --weight-dtype INT4 --token-latency --profile  \
                                                                2>&1 | tee -a $log_dir/llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")
                                        
                                        lines.append(f"collect_perf_logs_llm llm_default_{model_id.replace('/','-')}_{dtype}_{input_token}-{output_token}-{bs}_greedy_{beam}_NUMA_{rank}_{data['launcher']['hw']}.log")


        if mode.endswith('defaultacc'):
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for rank in data['modelargs'][mode]['localrank']:
                        lines.append(f"export local_rank={rank}")
                        lines.append("deepspeed_core_config ${local_rank}")
                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")
                        # if 'codegen' in model_id:
                        #     lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py  -m {model_id} --dtype {dtype} --ipex  --tasks hellaswag --batch-size 1\
                        #                 2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        # elif 'mpt' in model_id:
                        #     lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py  -m {model_id} --dtype {dtype} --ipex  --tasks lambada_openai --batch-size 1 --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                        #                 2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        # else:
                        #     lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py  -m {model_id} --dtype {dtype} --ipex  --tasks lambada_openai --batch-size 1 \
                        #                 2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                            
                        if 'codegen' in model_id:
                            lines.append(f"python single_instance/run_accuracy.py  -m {model_id} --dtype {dtype} --ipex  --tasks hellaswag --batch-size 1\
                                        2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        elif 'mpt' in model_id:
                            lines.append(f"python single_instance/run_accuracy.py  -m {model_id} --dtype {dtype} --ipex  --tasks hellaswag --batch-size 1 --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                        2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                                                         
                        else:
                            lines.append(f"python single_instance/run_accuracy.py  -m {model_id} --dtype {dtype} --ipex  --tasks lambada_openai --batch-size 1 \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        if 'codegen' in model_id or 'mpt' in model_id:
                            lines.append(f"collect_accnorm_logs_llm llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        else:
                            lines.append(f"collect_acc_logs_llm llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")

        if mode.endswith('customacc'):
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for rank in data['modelargs'][mode]['localrank']:
                        for task in data['modelargs'][mode]['tasks']:
                            lines.append(f"export local_rank={rank}")
                            lines.append("deepspeed_core_config ${local_rank}")
                            lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")
                            if 'codegen' in model_id:
                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py  -m {model_id} --dtype {dtype} --ipex  --tasks hellaswag --batch-size 1\
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                            elif 'mpt' in model_id:
                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py  -m {model_id} --dtype {dtype} --ipex  --tasks {task} --batch-size 1 --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{task}_{data['launcher']['hw']}.log")                                
                            
                            else:
                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py  -m {model_id} --dtype {dtype} --ipex  --tasks {task} --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{task}_{data['launcher']['hw']}.log")
                            if 'codegen' in model_id or 'mpt' in model_id:
                                lines.append(f"collect_accnorm_logs_llm llm_accuracy_{model_id.replace('/','-')}_{dtype}_{task}_{data['launcher']['hw']}.log")
                            else:
                                lines.append(f"collect_acc_logs_llm llm_accuracy_{model_id.replace('/','-')}_{dtype}_{task}_{data['launcher']['hw']}.log")


        if mode.endswith('defaultaccnocore'):
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    for rank in data['modelargs'][mode]['localrank']:
                        lines.append(f"export local_rank={rank}")
                        lines.append("deepspeed_core_config ${local_rank}")
                        lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")
                        if 'codegen' in model_id:
                            lines.append(f"python single_instance/run_accuracy.py  -m {model_id} --dtype {dtype} --ipex  --tasks hellaswag --batch-size 56\
                                        2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        else:
                            lines.append(f"python single_instance/run_accuracy.py  -m {model_id} --dtype {dtype} --ipex  --tasks lambada_openai --batch-size 56\
                                        2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        if 'codegen' in model_id or 'mpt' in model_id:
                            lines.append(f"collect_accnorm_logs_llm llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        else:
                            lines.append(f"collect_acc_logs_llm llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")

        if mode.endswith('static8acc'):
            for model_id in data['modelargs'][mode]['modelid']:
                for rank in data['modelargs'][mode]['localrank']:
                    lines.append(f"export local_rank={rank}")
                    lines.append("deepspeed_core_config ${local_rank}")
                    lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")
                    for dtype in data['modelargs'][mode]['dtype']:
                        for bs in data['modelargs'][mode]['batchsize']:
                            if 'fp32' in dtype:
                                lines.append(f"python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --batch-size {bs} --ipex --tasks lambada_openai \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                            # elif 'bs1' in dtype:
                            #     lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}  -m {model_id} --dtype int8 --batch-size {bs} --ipex  --tasks lambada_openai \
                            #             2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}-{bs}_{data['launcher']['hw']}.log") 
                            else:
                                lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --batch-size {bs} --quant-with-amp --ipex --tasks lambada_openai \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                            lines.append(f"collect_acc_logs_llm llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")

        if mode.endswith('woq8acc'):
            for model_id in data['modelargs'][mode]['modelid']:
                for rank in data['modelargs'][mode]['localrank']:
                    lines.append(f"export local_rank={rank}")
                    lines.append("deepspeed_core_config ${local_rank}")
                    lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")
                    for dtype in data['modelargs'][mode]['dtype']:
                        for bs in data['modelargs'][mode]['batchsize']:
                            if 'fp32' in dtype:
                                if 'codegen' in model_id:                             
                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --ipex --tasks hellaswag --batch-size {bs} \
                                                2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                                elif 'neox' in model_id or 'dolly' in model_id:
                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --ipex  --tasks lambada_openai --batch-size {bs} \
                                                2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                                elif 'falcon' in model_id:
                                    lines.append(f"python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --ipex  --tasks lambada_openai --config-file utils/model_config/tiiuae_falcon-40b_config.json --batch-size {bs} \
                                                2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                                elif 'mpt' in model_id:
                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --ipex  --tasks hellaswag --batch-size {bs} --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                                2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                                
                                # elif 'dolly' in model_id:
                                #     lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --quant-with-amp --ipex  --tasks hellaswag --batch-size {bs} \
                                #                 2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                               
                                else:
                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --ipex --tasks lambada_openai --batch-size {bs} \
                                                2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")

                            else:
                                if 'codegen' in model_id:                             
                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --ipex --quant-with-amp --tasks hellaswag --batch-size {bs} \
                                                2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                                elif 'neox' in model_id or 'dolly' in model_id:
                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --ipex  --tasks lambada_openai --batch-size {bs} \
                                                2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                                elif 'falcon' in model_id:
                                    lines.append(f"python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --quant-with-amp --ipex  --tasks lambada_openai --config-file utils/model_config/tiiuae_falcon-40b_config.json --batch-size {bs} \
                                                2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                                elif 'mpt' in model_id:
                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --quant-with-amp --ipex  --tasks hellaswag --batch-size {bs} --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                                2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                                
                                # elif 'dolly' in model_id:
                                #     lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --quant-with-amp --ipex  --tasks hellaswag --batch-size {bs} \
                                #                 2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                               
                                else:
                                    lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}/{model_id}/best_model.pt  -m {model_id} --dtype int8 --ipex --quant-with-amp --tasks lambada_openai --batch-size {bs} \
                                                2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                            
                            if 'codegen' in model_id in model_id or 'mpt' in model_id:
                                lines.append(f"collect_accnorm_logs_llm llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                            else:
                                lines.append(f"collect_acc_logs_llm llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                            



        if mode.endswith('woq8acc1'):
            for model_id in data['modelargs'][mode]['modelid']:
                for rank in data['modelargs'][mode]['localrank']:
                    lines.append(f"export local_rank={rank}")
                    lines.append("deepspeed_core_config ${local_rank}")
                    lines.append("export core_list=0-$(($cores_per_node*$local_rank-1))")
                    for dtype in data['modelargs'][mode]['dtype']:
                        if 'codegen' in model_id:                             
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}  -m {model_id} --dtype int8 --quant-with-amp --ipex  --tasks hellaswag --batch-size 56 \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        elif 'neox' in model_id or 'dolly' in model_id:
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}  -m {model_id} --dtype int8 --ipex  --tasks lambada_openai --batch-size 56 \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        elif 'falcon' in model_id:
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}  -m {model_id} --dtype int8 --quant-with-amp --ipex  --tasks lambada_openai --config-file utils/model_config/tiiuae_falcon-40b_config.json --batch-size 56 \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        else:
                            lines.append(f"OMP_NUM_THREADS={data['launcher']['OMP_NUM_THREADS']} numactl -m 0 -C $core_list python single_instance/run_accuracy.py --quantized-model-path {data['modelargs'][mode]['quantizedmodelpath']}  -m {model_id} --dtype int8 --quant-with-amp --ipex  --tasks lambada_openai --batch-size 56 \
                                        2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        if 'codegen' in model_id or 'mpt' in model_id:
                            lines.append(f"collect_accnorm_logs_llm llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        else:
                            lines.append(f"collect_acc_logs_llm llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")


        if mode.endswith('bf16dsacc'):
            # lines.append("cd ./distributed")
            lines.append("unset KMP_AFFINITY")
            for model_id in data['modelargs'][mode]['modelid']:
                if 'falcon' in model_id: 
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --ipex  --tasks lambada_openai  --config-file utils/model_config/tiiuae_falcon-40b_config.json --batch-size 1 \
                                    2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_ds-bfloat16_{data['launcher']['hw']}.log")                
                elif 'codegen' in model_id:
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --ipex  --tasks hellaswag  --batch-size 1 \
                                    2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_ds-bfloat16_{data['launcher']['hw']}.log")                    
                elif 'gpt-j' in model_id:
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --ipex  --tasks lambada_openai  --batch-size 1 \
                                    2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_ds-bfloat16_{data['launcher']['hw']}.log")                    
                elif 'mpt' in model_id:
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --ipex  --tasks hellaswag  --batch-size 1 --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                    2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_ds-bfloat16_{data['launcher']['hw']}.log")                    
                # elif 'dolly' in model_id:
                #     lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --ipex  --tasks hellaswag  --batch-size 1 \
                #                     2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_ds-bfloat16_{data['launcher']['hw']}.log")                  
                else:
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --ipex  --tasks lambada_openai  --batch-size 1 \
                                    2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_ds-bfloat16_{data['launcher']['hw']}.log")
                
                if 'mpt' in model_id or 'codegen' in model_id:
                    lines.append(f"collect_accnorm_logs_llm llm_accuracy_{model_id.replace('/','-')}_ds-bfloat16_{data['launcher']['hw']}.log")
                else:
                    lines.append(f"collect_acc_logs_llm llm_accuracy_{model_id.replace('/','-')}_ds-bfloat16_{data['launcher']['hw']}.log")

        if mode.endswith('bf16dsaccnoipex'):
            # lines.append("cd ./distributed")
            lines.append("unset KMP_AFFINITY")
            for model_id in data['modelargs'][mode]['modelid']:
                if 'falcon' in model_id: 
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --tasks lambada_openai  --config-file utils/model_config/tiiuae_falcon-40b_config.json --batch-size 1 \
                                    2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_ds-bfloat16_{data['launcher']['hw']}.log")                
                elif 'codegen' in model_id:
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --tasks hellaswag  --batch-size 1 \
                                    2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_ds-bfloat16_{data['launcher']['hw']}.log")                    
                elif 'gpt-j' in model_id:
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --tasks lambada_openai  --batch-size 1 \
                                    2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_ds-bfloat16_{data['launcher']['hw']}.log")                    
                elif 'mpt' in model_id:
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --tasks hellaswag  --batch-size 1 --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                    2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_ds-bfloat16_{data['launcher']['hw']}.log")                    
                # elif 'dolly' in model_id:
                #     lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --ipex  --tasks hellaswag  --batch-size 1 \
                #                     2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_ds-bfloat16_{data['launcher']['hw']}.log")                  
                else:
                    lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank distributed/run_accuracy_with_deepspeed.py  --model {model_id} --dtype bfloat16 --tasks lambada_openai  --batch-size 1 \
                                    2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_ds-bfloat16_{data['launcher']['hw']}.log")
                
                if 'mpt' in model_id or 'codegen' in model_id:
                    lines.append(f"collect_accnorm_logs_llm llm_accuracy_{model_id.replace('/','-')}_ds-bfloat16_{data['launcher']['hw']}.log")
                else:
                    lines.append(f"collect_acc_logs_llm llm_accuracy_{model_id.replace('/','-')}_ds-bfloat16_{data['launcher']['hw']}.log")

        if mode.endswith('woqdsacc'):

            lines.append("unset KMP_AFFINITY")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    if 'int8' in dtype:
                        if 'neox' in model_id or 'dolly' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --dtype float32 --ipex  --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        elif 'falcon' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --quant-with-amp --ipex  --tasks lambada_openai  --ipex-weight-only-quantization --config-file utils/model_config/tiiuae_falcon-40b_config.json --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        elif 'codegen' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --quant-with-amp --ipex  --tasks hellaswag  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        elif 'mpt' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --quant-with-amp --ipex  --tasks hellaswag  --ipex-weight-only-quantization --batch-size 1 --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        # elif 'dolly' in model_id:
                        #     lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --quant-with-amp --ipex  --tasks hellaswag  --ipex-weight-only-quantization --batch-size 1 \
                        #                     2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                           
                        else:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --quant-with-amp --ipex  --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        if 'codegen' in model_id or 'mpt' in model_id:
                            lines.append(f"collect_accnorm_logs_llm llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        else:
                            lines.append(f"collect_acc_logs_llm llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")

                    elif 'int4' in dtype:
                        if 'neox' in model_id or 'dolly' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --dtype float32 --ipex  --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        elif 'falcon' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --quant-with-amp --ipex  --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 --config-file utils/model_config/tiiuae_falcon-40b_config.json\
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        elif 'codegen' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --quant-with-amp --ipex  --tasks hellaswag  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        elif 'mpt' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --quant-with-amp --ipex  --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        # elif 'dolly' in model_id:
                        #     lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --quant-with-amp --ipex  --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 \
                        #                     2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                           
                        else:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --quant-with-amp --ipex  --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        if 'codegen' in model_id or 'mpt' in model_id:
                            lines.append(f"collect_accnorm_logs_llm llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        else:
                            lines.append(f"collect_acc_logs_llm llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")


        if mode.endswith('woqdsaccnoipex'):

            lines.append("unset KMP_AFFINITY")
            for model_id in data['modelargs'][mode]['modelid']:
                for dtype in data['modelargs'][mode]['dtype']:
                    if 'int8' in dtype:
                        if 'neox' in model_id or 'dolly' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --dtype float32 --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        elif 'falcon' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --quant-with-amp --tasks lambada_openai  --ipex-weight-only-quantization --config-file utils/model_config/tiiuae_falcon-40b_config.json --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        elif 'codegen' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --quant-with-amp --tasks hellaswag  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        elif 'mpt' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --quant-with-amp --tasks hellaswag  --ipex-weight-only-quantization --batch-size 1 --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        # elif 'dolly' in model_id:
                        #     lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --quant-with-amp --tasks hellaswag  --ipex-weight-only-quantization --batch-size 1 \
                        #                     2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                           
                        else:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --quant-with-amp --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        if 'codegen' in model_id or 'mpt' in model_id:
                            lines.append(f"collect_accnorm_logs_llm llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        else:
                            lines.append(f"collect_acc_logs_llm llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")

                    elif 'int4' in dtype:
                        if 'neox' in model_id or 'dolly' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --dtype float32 --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        elif 'falcon' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --quant-with-amp --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 --config-file utils/model_config/tiiuae_falcon-40b_config.json\
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        elif 'codegen' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --quant-with-amp --tasks hellaswag  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        elif 'mpt' in model_id:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --quant-with-amp --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 --config-file=utils/model_config/mosaicml_mpt-7b_config.json\
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                            
                        # elif 'dolly' in model_id:
                        #     lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --quant-with-amp --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 \
                        #                     2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")                           
                        else:
                            lines.append(f"deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank ./distributed/run_accuracy_with_deepspeed.py --model {model_id} --weight-dtype INT4 --lowp-mode INT8 --quant-with-amp --tasks lambada_openai  --ipex-weight-only-quantization --batch-size 1 \
                                            2>&1 | tee -a $log_dir/llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        if 'codegen' in model_id or 'mpt' in model_id:
                            lines.append(f"collect_accnorm_logs_llm llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")
                        else:
                            lines.append(f"collect_acc_logs_llm llm_accuracy_{model_id.replace('/','-')}_{dtype}_{data['launcher']['hw']}.log")


        
        if mode.endswith('shard'):
            lines.append("# Run Workload")  
            lines.append("export WORK_DIR=./")
            
            for model_id in data['modelargs'][mode]['modelid']:
                lines.append(f"mkdir -p {data['modelargs'][mode]['outputdir']}/{model_id}")
                lines.append(f"python utils/create_shard_model.py -m {model_id} --save-path {data['modelargs'][mode]['outputdir']}/{model_id}")
                
        if mode.endswith('gptqquant'):
            lines.append("# Run Workload")  
            lines.append("export WORK_DIR=./")
            
            for model_id in data['modelargs'][mode]['modelid']:
                if 'falcon' in model_id: 
                    lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --quant-with-amp --lowp-mode BF16 -m {model_id} --low-precision-checkpoint {data['modelargs'][mode]['configpath']}/{model_id}/gptq_checkpoint_g128.pt --config-file utils/model_config/tiiuae_falcon-40b_config.json")
                elif 'neox' in model_id or 'dolly' in model_id:
                    lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id}  -m {model_id} --lowp-mode BF16 --low-precision-checkpoint {data['modelargs'][mode]['configpath']}/{model_id}/gptq_checkpoint_g128.pt")                    
                elif 'mpt' in model_id:
                    lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --quant-with-amp --lowp-mode BF16 -m {model_id} --low-precision-checkpoint {data['modelargs'][mode]['configpath']}/{model_id}/gptq_checkpoint_g128.pt --config-file=utils/model_config/mosaicml_mpt-7b_config.json")                    
                
                else:
                    lines.append(f"python {data['modelargs'][mode]['scriptname']} --ipex-weight-only-quantization --output-dir {data['modelargs'][mode]['outputdir']}/{model_id} --quant-with-amp --lowp-mode BF16 -m {model_id} --low-precision-checkpoint {data['modelargs'][mode]['configpath']}/{model_id}/gptq_checkpoint_g128.pt")



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
        yml_file = 'bench_nightly_docker.yml'
    if args.weekly:
        yml_file = 'bench_weekly_docker.yml'
    if args.emr_weekly:
        yml_file = 'bench_emr_weekly_docker.yml'
    if args.emr_nightly:
        yml_file = 'bench_emr_nightly_docker.yml'
    if args.hbm_weekly:
        yml_file = 'bench_hbm_weekly_docker.yml'
    if args.hbm_weekly_m:
        yml_file = 'bench_hbm_weekly_m_docker.yml'
    if args.hbm_nightly:
        yml_file = 'bench_hbm_nightly_docker.yml'
    if args.gnr_weekly:
        yml_file = 'bench_gnr_weekly_docker.yml'
    if args.debug:
        yml_file = 'bench_debug_docker.yml'
    if args.rls:
        yml_file = 'bench_rls_docker.yml'
    if args.acc:      
        yml_file = 'bench_acc_docker.yml'
    if args.acc_cluster:    
        yml_file = 'bench_acc_cluster.yml'
    if args.rlsemr:
        yml_file = 'bench_rls_docker_emr.yml'
    if args.gptq:
        yml_file = 'bench_gptq_docker.yml'
    if args.cpudeviceweekly:
        yml_file = 'bench_cpu_device_weekly.yml'    
    if args.publicds:
        yml_file = 'bench_publicds_nightly.yml'
    data = yaml.load(open(yml_file, 'r'),Loader=yaml.FullLoader) 
    for mode in data['modelargs'].keys():
        generate_commands(yml_file, mode, args.extra_kmp)
